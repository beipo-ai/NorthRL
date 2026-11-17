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

from __future__ import annotations

import math
from array import array
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any

import numpy as np

from northrl.utils.prompt_mean import prompt_group_key, prompt_group_token_counts

from ..northrl_types import Trajectory  # pylint: disable=relative-beyond-top-level
from .trajectory_alignment import (  # pylint: disable=relative-beyond-top-level
    align_logprobs_to_target,
    align_loss_mask_to_target,
    align_response_sampling_mask_to_target,
    drop_leading_placeholder_for_targets,
)
from .trajectory_sequences import (  # pylint: disable=relative-beyond-top-level
    iter_training_sequences,
)


def _mask_target_values(
    values: list[float] | array, loss_mask: list[int] | array
) -> list[float] | array:
    """Zero out target positions that do not contribute to the loss."""
    if (
        isinstance(values, array)
        and values.typecode == "d"
        and isinstance(loss_mask, array)
        and loss_mask.typecode == "b"
    ):
        count = min(len(values), len(loss_mask))
        mask = np.asarray(loss_mask)[:count]
        packed = array("d", [0.0]) * count
        # A masked NaN/Inf must become +0, not the result of multiplying by zero.
        np.multiply(np.asarray(values)[:count], mask, out=np.asarray(packed), where=mask != 0)
        return packed
    return [float(value) * float(mask) if mask else 0.0 for value, mask in zip(values, loss_mask)]


def _build_aligned_masked_arrays(
    values: list[float] | array, loss_mask: list[int] | array
) -> tuple[list[float] | array, list[float] | array]:
    """Build arrays AFTER the alignment helpers have returned built-in floats/ints.

    A verified 0*1* mask needs neither per-position type conversion nor a
    per-position mask branch. Keep multiplication by 1.0 rather than copying
    values so special floating-point values retain their historical behavior.
    Short, non-prefix and non-binary masks use the unchanged general path.
    """
    if (
        isinstance(values, array)
        and values.typecode == "d"
        and isinstance(loss_mask, array)
        and loss_mask.typecode == "b"
    ):
        return _mask_target_values(values, loss_mask), loss_mask
    if len(values) == len(loss_mask) and len(values) >= 1024:
        try:
            boundary = loss_mask.index(1)
        except ValueError:
            boundary = len(loss_mask)
        response_length = len(loss_mask) - boundary
        if loss_mask == [0] * boundary + [1] * response_length:
            return (
                [0.0] * boundary + [value * 1.0 for value in values[boundary:]],
                [0.0] * boundary + [1.0] * response_length,
            )
    return _mask_target_values(values, loss_mask), [float(mask) for mask in loss_mask]


def _expand_response_advantages(
    values: Sequence[float], loss_mask: list[int] | array
) -> list[float] | array:
    """Place response-only values at targets whose aligned mask is exactly one.

    Preserve short-input zero padding and ignore unused trailing values. Compact
    built-in buffers can scatter directly into an owned double buffer; custom
    sequences retain the historical indexing and float-conversion behavior.
    """
    if (
        type(values) is array  # pylint: disable=unidiomatic-typecheck
        and values.typecode in "fd"
        and type(loss_mask) is array  # pylint: disable=unidiomatic-typecheck
        and loss_mask.typecode == "b"
    ):
        packed = array("d", [0.0]) * len(loss_mask)
        positions = np.flatnonzero(np.asarray(loss_mask) == 1)
        count = min(len(positions), len(values))
        # Copy rather than multiply: keep signed zero and float64 NaN payloads.
        # Widening float32 signaling NaNs follows float(array('f')[i]), which
        # quiets the NaN without surfacing NumPy's floating-point warning.
        with np.errstate(invalid="ignore"):
            np.asarray(packed)[positions[:count]] = np.asarray(values)[:count]
        return packed

    expanded = []
    response_idx = 0
    for mask in loss_mask:
        if mask == 1 and response_idx < len(values):
            expanded.append(float(values[response_idx]))
            response_idx += 1
        else:
            expanded.append(0.0)
    return expanded


def _has_nonzero_advantages(values: list[float] | array) -> bool:
    """Scan compact numeric buffers without recreating Python token objects."""
    if type(values) is array and values.typecode == "d":  # pylint: disable=unidiomatic-typecheck
        return bool(np.count_nonzero(np.asarray(values)))
    return any(value != 0.0 for value in values)


def _align_mis_fields_to_target(
    trajectory: Trajectory | dict[str, Any],
) -> tuple[list[float], list[int]] | None:
    """Validate and align optional MIS weights and masks to target tokens."""
    has_weights = "correction_weights" in trajectory
    has_mask = "correction_mask" in trajectory
    if has_weights != has_mask:
        raise ValueError("correction_weights and correction_mask must be provided together.")
    if not has_weights:
        return None

    aligned = {}
    for name in ("correction_weights", "correction_mask"):
        values = trajectory[name]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a token-aligned sequence.")
        target_values = drop_leading_placeholder_for_targets(
            name=name,
            tokens=trajectory["tokens"],
            token_aligned_values=values,
        )
        if name == "correction_weights":
            if any(
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for value in values
            ):
                raise ValueError("correction_weights must contain finite nonnegative numbers.")
        elif any(not isinstance(value, Real) or value not in (0, 1) for value in values):
            raise ValueError("correction_mask must contain only 0 or 1.")
        if values and values[0] != 0:
            raise ValueError(f"{name}[0] must be 0; the first token has no target logprob.")
        aligned[name] = target_values

    return (
        [float(value) for value in aligned["correction_weights"]],
        [int(value) for value in aligned["correction_mask"]],
    )


def convert_trajectories_to_datums(
    trajectories: list[Trajectory],
    *,
    metadata_builder: Callable[[Trajectory | dict[str, Any], int], dict[str, Any]] | None = None,
    drop_zero_advantage_datums: bool = True,
    prompt_mean: bool = False,
    datum_sources: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """
    Convert complete trajectories to Datum dictionaries for importance_sampling loss.

    The historical function and parameter names are retained for compatibility.

    Handles both scalar-advantage and per-token-advantage trajectories
    automatically:
    - If ``traj["token_advantages"]`` is present and not None, those response-only
      advantages are used; an empty sequence expands to zero advantages.
    - Otherwise the scalar ``traj["advantage"]`` is broadcast to every response token
      (standard GRPO path).

    When ``traj["ref_logprobs"]`` is present, it must follow the
    same trajectory-level convention as ``old_log_probs``: length == len(tokens),
    with position 0 as the no-context placeholder. It is aligned to target
    positions and written into ``loss_fn_inputs["ref_logprobs"]`` so the
    weaver-trainer KL term can pick it up.

    Optional token-aligned ``correction_weights`` and ``correction_mask`` must
    appear together. After alignment using the original trajectory loss mask,
    they scale Datum advantages and filter its loss mask respectively. They
    remain client-side fields and never change the denominator, reference
    logprobs, or sampling candidate sets sent to Weaver.

    Sequences whose final loss mask is all zero are omitted after MIS alignment,
    independently of the advantage values. No Datum or source entry is appended
    for them; the original trajectories remain available for group statistics.

    The returned dicts always contain a ``loss_mask`` key so the weaver backend can
    use it.  The tinker backend strips it before sending to the server (the tinker
    server does not accept extra fields in loss_fn_inputs).

    Args:
        trajectories: Complete trajectories, with optional nested training sequences.
        The IS denominator is read from old_log_probs. For inference-sourced
        denominators, this is the rollout behavior-policy logprob. For
        recomputed denominators, RemoteApiTrainer overwrites old_log_probs.
        Sequence logprob fields (old_log_probs/ref_logprobs) and loss_mask
        are token-aligned with tokens. Datum logprob fields and loss_mask are
        target-aligned with target_tokens=tokens[1:], the positions where the
        model can compute per-token logprobs/losses for Weaver.
        Sequence sampling_mask is response-aligned and is expanded here when
        the datum needs a target-aligned sampling_mask.
        metadata_builder: Optional callback used to attach metadata while each
            trajectory is converted to a datum.
        drop_zero_advantage_datums: Whether to drop a datum whose active token
            advantages are all zero. GRPO disables this to retain the same
            active response tokens as Slime; other algorithms retain the legacy
            behavior. All-zero loss masks are always omitted.
        datum_sources: Optional output list mapping each appended Datum to its
            (parent index, segment index), before optimizer padding.
        prompt_mean: Attach client-side denominators from complete groups and
            original masks, before MIS rejection or Datum filtering.
    """
    group_tokens = prompt_group_token_counts(trajectories) if prompt_mean else {}
    indexed_sequences = (
        (parent_index, segment_index, sequence)
        for parent_index, parent in enumerate(trajectories)
        for segment_index, sequence in enumerate(iter_training_sequences([parent]))
    )
    datums_data = []
    for sample_idx, (parent_index, segment_index, traj) in enumerate(indexed_sequences):
        tokens = traj["tokens"]
        # Match forward_logprob: no prediction targets means no Weaver Datum
        # and, for R2, no recorded routing entry to replay during training.
        if len(tokens) <= 1:
            continue
        loss_mask = traj["loss_mask"]
        old_log_probs = traj.get("old_log_probs")
        if old_log_probs is None:
            raise KeyError("Training sequence missing required old_log_probs denominator.")

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        loss_fn_inputs = {"target_tokens": target_tokens}

        target_old_logprobs = align_logprobs_to_target(
            tokens=tokens,
            logprobs=old_log_probs,
            name="old_log_probs",
            compact=isinstance(tokens, array),
        )
        target_loss_mask = align_loss_mask_to_target(
            tokens=tokens,
            loss_mask=loss_mask,
            compact=isinstance(tokens, array),
        )
        # Keep non-response positions neutral in the IS ratio. The loss also
        # applies target_loss_mask, but zeroing here makes debug tensors explicit.
        masked_logprobs, float_loss_mask = _build_aligned_masked_arrays(
            target_old_logprobs, target_loss_mask
        )
        loss_fn_inputs["logprobs"] = masked_logprobs
        loss_fn_inputs["loss_mask"] = float_loss_mask

        # Optional reference-policy logprobs for the KL term. Sequence-level
        # ref_logprobs follow the same token-aligned contract as old_log_probs:
        # length == len(tokens), with position 0 as the no-context placeholder.
        ref_lp_full = traj.get("ref_logprobs")
        if ref_lp_full:
            target_ref_lp = align_logprobs_to_target(
                tokens=tokens,
                logprobs=ref_lp_full,
                name="ref_logprobs",
                compact=isinstance(tokens, array),
            )
            loss_fn_inputs["ref_logprobs"] = _mask_target_values(target_ref_lp, target_loss_mask)

        # --- decide per-token advantages ---
        # An empty response-only sequence expands to all-zero advantages instead
        # of falling through to the scalar `advantage` path.
        token_advantages: list[float] | array
        per_token_adv = traj.get("token_advantages")
        if per_token_adv is not None:
            token_advantages = _expand_response_advantages(per_token_adv, target_loss_mask)
        else:
            # Standard path: broadcast scalar advantage with mask
            advantage = traj["advantage"]
            # Alignment has converted every mask to a Python int. Multiply once
            # per distinct value instead of launching a scalar Tensor operation
            # per token. Keep multiplication before float conversion to preserve
            # the advantage dtype's rounding and signed-zero behavior.
            if isinstance(target_loss_mask, array):
                masks, positions = np.unique(np.asarray(target_loss_mask), return_inverse=True)
                # Keep scalar multiplication in the original advantage dtype.
                values = np.array([float(advantage * int(m)) for m in masks], dtype=np.float64)
                token_advantages = array("d")
                token_advantages.frombytes(memoryview(values[positions]).cast("B"))
            else:
                masked_advantages = {m: float(advantage * m) for m in set(target_loss_mask)}
                token_advantages = [masked_advantages[m] for m in target_loss_mask]

        mis_fields = _align_mis_fields_to_target(traj)
        if drop_zero_advantage_datums and not _has_nonzero_advantages(token_advantages):
            continue

        loss_fn_inputs["advantages"] = token_advantages

        sampling_mask = traj.get("sampling_mask")
        if sampling_mask is not None and not isinstance(sampling_mask, dict):
            loss_fn_inputs["sampling_mask"] = align_response_sampling_mask_to_target(
                tokens=tokens,
                loss_mask=loss_mask,
                sampling_mask=sampling_mask,
            )

        if mis_fields is not None:
            mis_weights, mis_mask = mis_fields
            loss_fn_inputs["advantages"] = [
                advantage * weight for advantage, weight in zip(token_advantages, mis_weights)
            ]
            loss_fn_inputs["loss_mask"] = [
                float(mask * correction) for mask, correction in zip(target_loss_mask, mis_mask)
            ]
            if drop_zero_advantage_datums and not _has_nonzero_advantages(
                loss_fn_inputs["advantages"]
            ):
                continue

        # Test the final mask after MIS rejection, before creating the Datum and
        # its source entry. Zero advantages alone do not make a sample empty.
        if not any(loss_fn_inputs["loss_mask"]):
            continue

        datum_data = {
            "input_tokens": input_tokens,
            "loss_fn_inputs": loss_fn_inputs,
        }
        if prompt_mean:
            # Local normalization data; Weaver receives only inputs and metadata.
            datum_data["_prompt_mean_denominator"] = len(group_tokens) * max(
                group_tokens[prompt_group_key(traj)], 1
            )
        if metadata_builder is not None:
            datum_data["metadata"] = metadata_builder(traj, sample_idx)
        if isinstance(sampling_mask, dict):
            from northrl.utils.sampling_mask_refs import align_sampling_mask_ref

            datum_data.setdefault("metadata", {}).update(
                align_sampling_mask_ref(tokens, loss_mask, sampling_mask)
            )
        distribution = traj.get("sampler_distribution_data")
        if distribution is not None:
            from northrl.utils.score_centering import align_sampler_distribution

            sc_fields, sc_metadata = align_sampler_distribution(tokens, loss_mask, distribution)
            datum_data["loss_fn_inputs"].update(sc_fields)
            datum_data.setdefault("metadata", {}).update(sc_metadata)
        if isinstance(tokens, array):
            # Keep E2B batches in packed numeric buffers. Python token objects
            # and backend tensors are only built for the submitted microbatch.
            # Doubles retain Python-float precision through normalization.
            for key in ("logprobs", "advantages", "ref_logprobs"):
                if key in loss_fn_inputs and not isinstance(loss_fn_inputs[key], array):
                    loss_fn_inputs[key] = array("d", loss_fn_inputs[key])
            if not isinstance(loss_fn_inputs["loss_mask"], array):
                loss_fn_inputs["loss_mask"] = array("b", map(int, loss_fn_inputs["loss_mask"]))
        datums_data.append(datum_data)
        if datum_sources is not None:
            datum_sources.append((parent_index, segment_index))

    return datums_data


def convert_trajectories_to_datums_cross_entropy(
    trajectories: list[Trajectory],
    *,
    datum_sources: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """
    Convert trajectories to serializable datum dictionaries for cross-entropy loss (SFT).

    Uses a minimal protocol: only ``target_tokens`` and ``loss_mask`` are required —
    no ``logprobs`` or ``advantages``.
    """
    indexed_sequences = (
        (parent_index, segment_index, sequence)
        for parent_index, parent in enumerate(trajectories)
        for segment_index, sequence in enumerate(iter_training_sequences([parent]))
    )
    datums_data = []
    for parent_index, segment_index, traj in indexed_sequences:
        if isinstance(traj, Trajectory) and traj.is_managed_sample_ref:
            sample_ref = traj.extra_fields["sample_ref"]
            datums_data.append(
                {
                    "kind": "sample_ref",
                    "datum_id": traj.extra_fields["datum_id"],
                    "dataset": sample_ref["dataset"],
                    "version": sample_ref["version"],
                    "sample_idx": sample_ref["sample_idx"],
                    # Internal NorthRL observability carrier;
                    # WeaverServiceHolder does not forward it into the SDK
                    # Datum.
                    "content_visibility": traj.extra_fields.get("content_visibility"),
                    # Keep caller-supplied auxiliary loss inputs attached to
                    # the datum while Weaver materializes the token-owned SFT
                    # fields from the SampleRef.
                    "loss_fn_inputs": dict(traj.extra_fields.get("loss_fn_inputs", {})),
                }
            )
            if datum_sources is not None:
                datum_sources.append((parent_index, segment_index))
            continue

        tokens = traj["tokens"]
        loss_mask = traj["loss_mask"]
        assert tokens is not None and loss_mask is not None

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        target_loss_mask = align_loss_mask_to_target(
            tokens=tokens,
            loss_mask=loss_mask,
        )

        datum_data = {
            "input_tokens": input_tokens,
            "loss_fn_inputs": {
                "target_tokens": target_tokens,
                "loss_mask": [int(m) for m in target_loss_mask],
            },
        }
        datum_id = traj.get("datum_id")
        if datum_id is not None:
            datum_data["datum_id"] = datum_id
        datums_data.append(datum_data)
        if datum_sources is not None:
            datum_sources.append((parent_index, segment_index))

    return datums_data
