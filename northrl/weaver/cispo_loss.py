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

"""CISPO: detached clipped importance weights multiplying token log-probabilities."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from .custom_losses import CustomLossCallback, _resolve_required_tensor


def build_cispo_loss(params: Mapping[str, Any]) -> CustomLossCallback:
    """Build a token-mean objective evaluated once over the full optimizer batch.

    L = -mean_tokens(stop_gradient(clip(exp(logp - old_logp))) * A * logp).
    Clipping weights must not stop the logp gradient, even outside the bounds.
    """
    if float(params.get("entropy_coeff", 0.0)) != 0.0:
        raise ValueError("CISPO requires entropy_coeff=0")
    if params.get("loss_agg_mode", "token-mean") not in {"token-mean", "dapo"}:
        raise ValueError("CISPO requires token-mean loss aggregation")
    bounds = []
    for key in ("clip_ratio_low", "clip_ratio_high"):
        raw = params.get(key)
        if raw is None:
            raw = params.get("clip_ratio")
        if raw is None or isinstance(raw, bool):
            raise ValueError(f"CISPO requires an explicit finite non-negative {key}")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"CISPO requires a finite non-negative {key}")
        bounds.append(value)
    low, high = bounds[0], bounds[1]

    def callback(
        data: Sequence[Any], logprob_tensors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not data or len(data) != len(logprob_tensors):
            raise ValueError("CISPO requires matching non-empty data and logprob tensors")

        losses = []
        ratios = []
        differences = []
        total_tokens = 0
        for datum, logprobs in zip(data, logprob_tensors, strict=True):
            if logprobs.ndim != 1 or not logprobs.is_floating_point():
                raise ValueError("CISPO expects one floating-point logprob vector per datum")
            # exp(20) overflows fp16. Keep double inputs for numerical checks.
            current = logprobs.to(
                torch.float64 if logprobs.dtype == torch.float64 else torch.float32
            )
            old = _resolve_required_tensor(
                datum,
                ("logprobs",),
                dtype=current.dtype,
                device=current.device,
                shape=current.shape,
            )
            advantages = _resolve_required_tensor(
                datum,
                ("advantages",),
                dtype=current.dtype,
                device=current.device,
                shape=current.shape,
            )
            mask = _resolve_required_tensor(
                datum, ("loss_mask",), dtype=torch.bool, device=current.device, shape=current.shape
            )
            # Select before arithmetic: masked prompt/padding values may be non-finite.
            selected = current[mask]
            old_selected = old[mask].detach()
            adv_selected = advantages[mask].detach()
            if not all(
                torch.isfinite(tensor).all() for tensor in (selected, old_selected, adv_selected)
            ):
                raise ValueError("CISPO requires finite logprobs and advantages on active tokens")
            log_ratio = (selected.detach() - old_selected).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            weights = ratio.clamp(min=1.0 - low, max=1.0 + high)
            losses.append(-(weights * adv_selected * selected).sum())
            ratios.append(ratio)
            differences.append(log_ratio)
            total_tokens += selected.numel()

        # The holder calls us once, including when remote FB requests are split.
        loss = torch.stack(losses).sum() / max(total_tokens, 1)
        metrics: dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "loss:sum": float(loss.detach().cpu()),
            "tokens": total_tokens,
            "num_samples": len(data),
            "entropy_coeff": 0.0,
            "metric_weights": {
                key: total_tokens for key in ("loss", "ppo_clip/clipfrac", "ppo_clip/ppo_kl")
            },
        }
        if total_tokens:
            all_ratios = torch.cat(ratios)
            metrics.update(
                {
                    "ratio:mean": float(all_ratios.mean().cpu()),
                    "ratio:min": float(all_ratios.min().cpu()),
                    "ratio:max": float(all_ratios.max().cpu()),
                    "ppo_clip/clipfrac": float(
                        ((all_ratios < 1.0 - low) | (all_ratios > 1.0 + high)).float().mean().cpu()
                    ),
                    "ppo_clip/ppo_kl": float(-torch.cat(differences).mean().cpu()),
                }
            )
        return loss, metrics

    return callback
