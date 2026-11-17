# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Group Sequence Policy Optimization on complete rollout sequences.

The differentiable sequence ratio is exp(mean_response(log pi - log pi_old)).
Unlike rollout mismatch correction, this ratio belongs to the actor objective
and must retain its gradient. The callback receives one complete optimizer batch
with raw scalar advantages, even when Weaver forwards are split into chunks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from northrl.weaver.custom_losses import CustomLossCallback, _resolve_required_tensor


def attach_gspo_trajectory_metadata(
    datums: list[dict[str, Any]], sources: list[tuple[int, int]]
) -> None:
    """Attach GSPO parent identities and contiguous emitted segment ordinals in place."""
    if len(datums) != len(sources):
        raise ValueError("Complete-trajectory loss requires a parent source for every Datum")
    sizes: dict[int, int] = {}
    for parent, _ in sources:
        sizes[parent] = sizes.get(parent, 0) + 1
    offsets: dict[int, int] = {}
    for datum, (parent, _) in zip(datums, sources, strict=True):
        offset = offsets.get(parent, 0)
        datum.setdefault("metadata", {})["northrl_gspo"] = {
            "trajectory_id": str(parent),
            "segment_index": offset,
            "segment_count": sizes[parent],
        }
        offsets[parent] = offset + 1


def _trajectory_groups(data: Sequence[Any]) -> list[list[int]]:
    """Validate parent ownership before coupling segments in the objective."""
    ownership = [(getattr(datum, "metadata", None) or {}).get("northrl_gspo") for datum in data]
    if all(item is None for item in ownership):
        # Direct factory callers may provide ordinary, complete single-sequence rows.
        return [[index] for index in range(len(data))]
    groups: dict[str, list[int]] = {}
    expected_counts: dict[str, int] = {}
    segment_indices: dict[str, set[int]] = {}
    for index, item in enumerate(ownership):
        if not isinstance(item, Mapping):
            raise ValueError("GSPO requires complete trajectory metadata on every datum.")
        parent = item.get("trajectory_id")
        segment = item.get("segment_index")
        count = item.get("segment_count")
        if not isinstance(segment, int) or isinstance(segment, bool):
            raise ValueError("Invalid GSPO segment_index in trajectory metadata.")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError("Invalid GSPO segment_count in trajectory metadata.")
        if not isinstance(parent, str) or not parent or not 0 <= segment < count:
            raise ValueError("Invalid GSPO trajectory metadata.")
        if parent in expected_counts and expected_counts[parent] != count:
            raise ValueError("Inconsistent GSPO segment_count for a trajectory.")
        expected_counts[parent] = count
        seen = segment_indices.setdefault(parent, set())
        if segment in seen:
            raise ValueError("Duplicate GSPO segment_index in a trajectory.")
        seen.add(segment)
        groups.setdefault(parent, []).append(index)
    for parent, indices in groups.items():
        if len(indices) != expected_counts[parent]:
            raise ValueError(
                "GSPO requires all segments of each trajectory in one optimizer batch."
            )
    return list(groups.values())


def _segment_statistics(
    datum: Any, logprobs: torch.Tensor
) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    """Return a differentiable log-ratio sum, response count and scalar advantage."""
    if logprobs.ndim != 1:
        raise ValueError("GSPO expects one-dimensional target logprobs.")
    # Accumulate long response log ratios in at least fp32; retain fp64 callers.
    dtype = torch.float64 if logprobs.dtype == torch.float64 else torch.float32
    values = logprobs.to(dtype=dtype)
    mask = _resolve_required_tensor(
        datum, ("loss_mask",), dtype=dtype, device=values.device, shape=values.shape
    )
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("GSPO loss_mask must contain only 0 or 1.")
    active = mask.bool()
    count = int(active.sum().item())
    if not count:
        # The SDK needs zero gradients, not None, even for entirely masked rows.
        return values[:0].sum(), 0, None
    old = _resolve_required_tensor(
        datum,
        ("sampling_logprobs", "logprobs"),
        dtype=dtype,
        device=values.device,
        shape=values.shape,
    )[active]
    advantages = _resolve_required_tensor(
        datum, ("advantages",), dtype=dtype, device=values.device, shape=values.shape
    )[active]
    current = values[active]
    if not all(torch.isfinite(tensor).all() for tensor in (current, old, advantages)):
        raise ValueError("GSPO requires finite active logprobs and advantages.")
    if not torch.all(advantages == advantages[0]):
        raise ValueError("GSPO requires a constant scalar advantage across each trajectory.")
    return (current - old).sum(), count, advantages[0]


def build_gspo_loss(params: Mapping[str, Any]) -> CustomLossCallback:
    """Build the clipped GSPO objective; all clip widths must be explicit."""
    supported = {"clip_ratio_low", "clip_ratio_high", "entropy_coeff"}
    if unknown := set(params) - supported:
        raise ValueError(f"Unsupported GSPO loss parameters: {sorted(unknown)}")
    if float(params.get("entropy_coeff", 0.0)) != 0.0:
        raise ValueError("GSPO custom loss requires entropy_coeff=0.")
    try:
        low = float(params["clip_ratio_low"])
        high = float(params["clip_ratio_high"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("GSPO requires explicit clip_ratio_low and clip_ratio_high.") from exc
    if not math.isfinite(low) or not 0 <= low < 1:
        raise ValueError("GSPO clip_ratio_low must be finite and in [0, 1).")
    if not math.isfinite(high) or high < 0:
        raise ValueError("GSPO clip_ratio_high must be finite and nonnegative.")

    def callback(
        data: Sequence[Any], logprob_tensors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not data or len(data) != len(logprob_tensors):
            raise ValueError("GSPO requires one logprob tensor per datum in a nonempty batch.")

        sequence_losses = []
        ratios = []
        clipped = []
        empty_losses = []
        total_tokens = 0
        for indices in _trajectory_groups(data):
            sums = []
            response_count = 0
            advantage = None
            for index in indices:
                log_ratio_sum, count, segment_advantage = _segment_statistics(
                    data[index], logprob_tensors[index]
                )
                sums.append(log_ratio_sum)
                response_count += count
                if segment_advantage is not None:
                    if advantage is not None and segment_advantage != advantage:
                        raise ValueError(
                            "GSPO requires the same scalar advantage in every trajectory segment."
                        )
                    advantage = segment_advantage
            log_ratio_sum = torch.stack(sums).sum()
            if response_count == 0:
                empty_losses.append(log_ratio_sum)
                continue
            assert advantage is not None
            ratio = torch.exp(log_ratio_sum / response_count)
            if not torch.isfinite(ratio):
                raise ValueError(
                    "GSPO sequence importance ratio overflowed; check policy logprobs."
                )
            unclipped_loss = -ratio * advantage
            clipped_loss = -ratio.clamp(1.0 - low, 1.0 + high) * advantage
            sequence_losses.append(torch.maximum(unclipped_loss, clipped_loss))
            ratios.append(ratio.detach())
            clipped.append((clipped_loss > unclipped_loss).detach())
            total_tokens += response_count

        # Zero-advantage parents remain in the mean. Entirely empty parents have
        # no sequence ratio and are excluded; every emitted row still gets a gradient.
        if sequence_losses:
            loss = torch.stack(sequence_losses).mean()
            if empty_losses:
                loss = loss + torch.stack(empty_losses).sum()
        else:
            loss = torch.stack(empty_losses).sum()
        metrics: dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "loss:sum": float(loss.detach().cpu()),
            "tokens": total_tokens,
            "num_samples": len(data),
            "gspo/active_trajectories": len(sequence_losses),
            "gspo/empty_trajectories": len(empty_losses),
            "metric_weights": {
                "loss": len(sequence_losses),
                "gspo/ratio_mean": len(sequence_losses),
                "gspo/clip_fraction": len(sequence_losses),
            },
        }
        if ratios:
            sequence_ratios = torch.stack(ratios)
            metrics.update(
                {
                    "gspo/ratio_mean": float(sequence_ratios.mean().cpu()),
                    "gspo/ratio_min": float(sequence_ratios.min().cpu()),
                    "gspo/ratio_max": float(sequence_ratios.max().cpu()),
                    "gspo/clip_fraction": float(torch.stack(clipped).float().mean().cpu()),
                }
            )
        return loss, metrics

    return callback
