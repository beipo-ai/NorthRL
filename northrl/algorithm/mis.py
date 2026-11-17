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

"""Compute frozen rollout-to-trainer importance weights and rejection masks."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.config_schema import MISConfig

_LOG_RATIO_BOUND = 20.0
_LEVELS = {"token", "sequence", "geometric"}


@dataclass(slots=True)
class MISResult:
    """Token-aligned weights, independent rejection masks, and batch statistics."""

    correction_weights: list[list[float]]
    correction_mask: list[list[int]]
    metrics: dict[str, float]


def _positive_finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _validate_config(config: MISConfig) -> tuple[float, float, float, float, float | None]:
    if config.tis_mode not in {"none", "truncate", "clip", "mask"}:
        raise ValueError(f"Unknown tis_mode: {config.tis_mode!r}.")
    if config.tis_level not in _LEVELS or config.rs_level not in _LEVELS:
        raise ValueError("tis_level and rs_level must be token, sequence, or geometric.")
    upper = _positive_finite(config.tis_upper_bound, "tis_upper_bound")
    lower = (
        _positive_finite(config.tis_lower_bound, "tis_lower_bound")
        if config.tis_lower_bound is not None
        else 1.0 / upper
    )
    rs_lower = (
        _positive_finite(config.rs_lower_bound, "rs_lower_bound")
        if config.rs_lower_bound is not None
        else lower
    )
    rs_upper = (
        _positive_finite(config.rs_upper_bound, "rs_upper_bound")
        if config.rs_upper_bound is not None
        else upper
    )
    if config.enabled and config.tis_mode in {"clip", "mask"} and lower >= upper:
        raise ValueError("TIS lower bound must be smaller than its upper bound.")
    if config.enabled and config.use_rs and rs_lower >= rs_upper:
        raise ValueError("RS lower bound must be smaller than its upper bound.")
    if (
        config.enabled
        and config.tis_batch_normalize
        and config.tis_level == "geometric"
        and config.tis_mode != "none"
    ):
        raise ValueError("tis_batch_normalize does not support geometric TIS weights.")
    veto = (
        _positive_finite(config.rs_veto_threshold, "rs_veto_threshold")
        if config.rs_veto_threshold is not None
        else None
    )
    return lower, upper, rs_lower, rs_upper, veto


def _log_ratio(values: list[float], valid: list[int], level: str) -> list[float]:
    if level == "token":
        return values
    # Aggregate before bounding: large positive and negative token differences
    # may cancel, which would be lost if individual differences were clipped.
    divisor = max(len(valid), 1) if level == "geometric" else 1
    try:
        total = math.fsum(values[index] / divisor for index in valid)
    except OverflowError:
        # The aggregate can exceed float range while its bounded exponential
        # remains defined. Scaling also preserves the sign of cancellation.
        scale = max(abs(values[index]) for index in valid)
        total = math.fsum(values[index] / scale / divisor for index in valid) * scale
    return [total] * len(values)


def _ratio(log_ratio: float) -> float:
    return math.exp(max(-_LOG_RATIO_BOUND, min(_LOG_RATIO_BOUND, log_ratio)))


def _weight_metrics(name: str, values: list[float]) -> dict[str, float]:
    return {
        f"{name}_mean": math.fsum(value / len(values) for value in values),
        f"{name}_min": min(values, default=0.0),
        f"{name}_max": max(values, default=0.0),
    }


def compute_mis_weights(
    config: MISConfig,
    *,
    rollout_log_probs: Sequence[Sequence[float]],
    recomputed_log_probs: Sequence[Sequence[float]],
    loss_masks: Sequence[Sequence[int]],
) -> MISResult:
    """Compare frozen recomputed logprobs with rollout logprobs.

    Inputs and outputs align with the full trajectory tokens, including a
    masked first-token placeholder. Only original valid positions are read;
    masked prompt/tool logprobs may contain placeholders. Input objects are
    never modified, and masked output positions always contain weight/mask 0.

    TIS changes weights while TIS mask mode, RS and veto change an independent
    mask. Rejected positions retain their TIS weights. Optional normalization
    uses original masks over this entire input batch, including rejected
    positions, and runs after bounding without clipping again. Sequence-mode
    normalization averages per-sequence means, including zero-mask rows as 0.
    ``none`` uses unit weights and supports RS without importance weighting.

    Weight statistics cover original valid positions, including rejected ones.
    "Raw" weights include the log-ratio safety bound but precede TIS handling.
    Sequence rejection counts sequences with no remaining valid tokens.
    """
    lower, upper, rs_lower, rs_upper, veto = _validate_config(config)
    if not len(rollout_log_probs) == len(recomputed_log_probs) == len(loss_masks):
        raise ValueError("Logprob and loss_mask batches must have the same number of sequences.")

    weights_batch: list[list[float]] = []
    masks_batch: list[list[int]] = []
    valid_indices: list[list[int]] = []
    raw_weights: list[float] = []
    clipped_tokens = 0
    vetoed_sequences = 0

    for sequence_index, (rollout, recomputed, mask) in enumerate(
        zip(rollout_log_probs, recomputed_log_probs, loss_masks, strict=True)
    ):
        if not len(rollout) == len(recomputed) == len(mask):
            raise ValueError(
                f"Sequence {sequence_index}: logprob and loss_mask lengths must match."
            )
        if any(value not in (0, 1) for value in mask):
            raise ValueError(f"Sequence {sequence_index}: loss_mask must contain only 0 or 1.")
        if len(mask) > 0 and mask[0] != 0:
            raise ValueError(
                f"Sequence {sequence_index}: loss_mask[0] must be the masked placeholder."
            )
        valid = [index for index, value in enumerate(mask) if value == 1]
        valid_indices.append(valid)
        differences = [0.0] * len(mask)
        for index in valid:
            try:
                old = float(recomputed[index])
                sampled = float(rollout[index])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Sequence {sequence_index}, token {index}: valid logprobs must be finite."
                ) from exc
            if not math.isfinite(old) or not math.isfinite(sampled):
                raise ValueError(
                    f"Sequence {sequence_index}, token {index}: valid logprobs must be finite."
                )
            differences[index] = old - sampled
            if not math.isfinite(differences[index]):
                raise ValueError(
                    f"Sequence {sequence_index}, token {index}: logprob difference overflowed."
                )

        weights = [0.0] * len(mask)
        modified_mask = [int(value) for value in mask]
        if not config.enabled:
            for index in valid:
                weights[index] = 1.0
            raw_weights.extend(1.0 for _ in valid)
        else:
            tis_log_ratios = _log_ratio(differences, valid, config.tis_level)
            for index in valid:
                raw = _ratio(tis_log_ratios[index])
                raw_weights.append(raw)
                if config.tis_mode == "none":
                    weights[index] = 1.0
                elif config.tis_mode == "truncate":
                    weights[index] = min(raw, upper)
                    clipped_tokens += int(raw > upper)
                elif config.tis_mode == "clip":
                    weights[index] = max(lower, min(raw, upper))
                    clipped_tokens += int(raw < lower or raw > upper)
                else:
                    weights[index] = raw
                    modified_mask[index] = int(lower <= raw <= upper)

            if config.use_rs:
                rs_log_ratios = _log_ratio(differences, valid, config.rs_level)
                for index in valid:
                    modified_mask[index] *= int(
                        rs_lower <= _ratio(rs_log_ratios[index]) <= rs_upper
                    )
                if veto is not None and any(differences[index] < math.log(veto) for index in valid):
                    modified_mask = [0] * len(mask)
                    vetoed_sequences += 1

        weights_batch.append(weights)
        masks_batch.append(modified_mask)

    original_tokens = sum(len(valid) for valid in valid_indices)
    original_sequences = sum(bool(valid) for valid in valid_indices)
    normalization = 1.0
    if config.enabled and config.tis_batch_normalize:
        if config.tis_level == "sequence" and config.tis_mode != "none":
            sequence_means = [
                math.fsum(weights[index] / len(valid) for index in valid)
                for weights, valid in zip(weights_batch, valid_indices, strict=True)
            ]
            mean = math.fsum(value / len(sequence_means) for value in sequence_means)
        else:
            mean = math.fsum(
                weights[index] / original_tokens
                for weights, valid in zip(weights_batch, valid_indices, strict=True)
                for index in valid
            )
        if mean > 1e-8:
            normalization = mean
            weights_batch = [[weight / mean for weight in row] for row in weights_batch]

    kept_tokens = sum(sum(row) for row in masks_batch)
    kept_sequences = sum(any(row) for row in masks_batch)
    final_weights = [
        weights[index]
        for weights, valid in zip(weights_batch, valid_indices, strict=True)
        for index in valid
    ]
    metrics = {
        "original_tokens": float(original_tokens),
        "kept_tokens": float(kept_tokens),
        "original_sequences": float(original_sequences),
        "kept_sequences": float(kept_sequences),
        "token_rejection_fraction": (original_tokens - kept_tokens) / max(original_tokens, 1),
        "sequence_rejection_fraction": (original_sequences - kept_sequences)
        / max(original_sequences, 1),
        "veto_fraction": vetoed_sequences / max(original_sequences, 1),
        "clipped_fraction": clipped_tokens / max(original_tokens, 1),
        "batch_norm_factor": normalization,
        **_weight_metrics("raw_weight", raw_weights),
        **_weight_metrics("final_weight", final_weights),
    }
    return MISResult(weights_batch, masks_batch, metrics)


def _diagnostic_exp(value: float, *, subtract_one: bool = False) -> float:
    """Retain the true exponential; an unrepresentable diagnostic is omitted later."""
    try:
        return math.expm1(value) if subtract_one else math.exp(value)
    except OverflowError:
        return math.inf


def _diagnostic_mean(values: Sequence[float]) -> float:
    # Dividing before summing avoids overflow when the finite mean is much
    # smaller than the sum. Non-finite intermediate metrics are omitted later.
    try:
        return math.fsum(value / len(values) for value in values)
    except (OverflowError, ValueError):
        return math.inf


def _mismatch_sequence_metrics(
    old_logprobs: list[float], rollout_logprobs: list[float], log_ratios: list[float]
) -> dict[str, float]:
    training_log_ppl = -_diagnostic_mean(old_logprobs)
    rollout_log_ppl = -_diagnostic_mean(rollout_logprobs)
    log_ppl_diff = -_diagnostic_mean(log_ratios)
    sequence_log_ratio = _log_ratio(log_ratios, list(range(len(log_ratios))), "sequence")[0]
    return {
        "training_log_ppl": training_log_ppl,
        "training_ppl": _diagnostic_exp(training_log_ppl),
        "rollout_log_ppl": rollout_log_ppl,
        "rollout_ppl": _diagnostic_exp(rollout_log_ppl),
        "kl": log_ppl_diff,
        "k3_kl": _diagnostic_mean(
            [_diagnostic_exp(value, subtract_one=True) - value for value in log_ratios]
        ),
        "log_ppl_diff": log_ppl_diff,
        "log_ppl_abs_diff": abs(log_ppl_diff),
        "ppl_ratio": _diagnostic_exp(log_ppl_diff),
        "chi2_token": _diagnostic_mean([_ratio(value) ** 2 for value in log_ratios]) - 1,
        "chi2_seq": _ratio(sequence_log_ratio) ** 2 - 1,
    }


def compute_mismatch_metrics(
    *,
    rollout_log_probs: Sequence[Sequence[float]],
    recomputed_log_probs: Sequence[Sequence[float]],
    loss_masks: Sequence[Sequence[int]],
    loss_agg_mode: str = "token-mean",
    group_ids: Sequence[Hashable] | None = None,
) -> dict[str, float]:
    """Report rollout/training mismatch without modifying correction or loss inputs.

    Inputs use the full-token alignment and original masks of
    ``compute_mis_weights``. Only original valid positions are read. Metrics
    are computed per sequence, then averaged over the complete input batch:
    ``token-mean`` (or its ``dapo`` alias) weights sequences by valid-token
    count; ``sample-mean`` gives each nonempty sequence equal weight.
    ``prompt-mean`` requires one group key per sequence (for example,
    ``(model_tag, group_id)``), weights sequences by valid tokens within each
    group, and gives nonempty groups equal weight. All modes retain the
    per-sequence metric definitions; in particular, ``chi2_seq`` still depends
    on sequence boundaries. Empty sequences and groups do not contribute to
    diagnostic denominators, even if the training loss counts empty groups.

    Invalid effective inputs raise ValueError so the caller can report an
    unavailable diagnostic without changing training behavior. Exponential
    overflow only omits the affected aggregate metric and increments
    ``nonfinite_metric_count``. PPL and K3 are never silently clipped; chi2
    retains Slime's explicit log-ratio safety bound of +/-20.
    """
    if loss_agg_mode not in {"token-mean", "sample-mean", "prompt-mean", "dapo"}:
        raise ValueError(f"Unsupported mismatch loss_agg_mode: {loss_agg_mode!r}.")
    if not len(rollout_log_probs) == len(recomputed_log_probs) == len(loss_masks):
        raise ValueError("Logprob and loss_mask batches must have the same number of sequences.")

    group_token_counts: dict[Hashable, int] = {}
    if loss_agg_mode == "prompt-mean":
        if group_ids is None:
            raise ValueError("prompt-mean mismatch metrics require group_ids for every sequence.")
        if len(group_ids) != len(loss_masks):
            raise ValueError("group_ids must have the same number of sequences as loss_masks.")
        try:
            group_token_counts = dict.fromkeys(group_ids, 0)
        except TypeError as exc:
            raise ValueError("group_ids must contain hashable group keys.") from exc

    sequence_metrics: list[dict[str, float]] = []
    token_counts: list[int] = []
    sequence_group_ids: list[Hashable] = []
    for sequence_index, (rollout, recomputed, mask) in enumerate(
        zip(rollout_log_probs, recomputed_log_probs, loss_masks, strict=True)
    ):
        if not len(rollout) == len(recomputed) == len(mask):
            raise ValueError(
                f"Sequence {sequence_index}: logprob and loss_mask lengths must match."
            )
        if any(value not in (0, 1) for value in mask):
            raise ValueError(f"Sequence {sequence_index}: loss_mask must contain only 0 or 1.")
        if len(mask) > 0 and mask[0] != 0:
            raise ValueError(
                f"Sequence {sequence_index}: loss_mask[0] must be the masked placeholder."
            )
        old_values: list[float] = []
        rollout_values: list[float] = []
        log_ratios: list[float] = []
        for token_index, active in enumerate(mask):
            if not active:
                continue
            try:
                old = float(recomputed[token_index])
                sampled = float(rollout[token_index])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Sequence {sequence_index}, token {token_index}: valid logprobs must be finite."
                ) from exc
            if not math.isfinite(old) or not math.isfinite(sampled):
                raise ValueError(
                    f"Sequence {sequence_index}, token {token_index}: valid logprobs must be finite."
                )
            log_ratio = old - sampled
            if not math.isfinite(log_ratio):
                raise ValueError(
                    f"Sequence {sequence_index}, token {token_index}: logprob difference overflowed."
                )
            old_values.append(old)
            rollout_values.append(sampled)
            log_ratios.append(log_ratio)
        if log_ratios:
            token_counts.append(len(log_ratios))
            sequence_metrics.append(
                _mismatch_sequence_metrics(old_values, rollout_values, log_ratios)
            )
            if loss_agg_mode == "prompt-mean":
                assert group_ids is not None
                group_id = group_ids[sequence_index]
                sequence_group_ids.append(group_id)
                group_token_counts[group_id] += len(log_ratios)

    metrics = {
        "mismatch_tokens": float(sum(token_counts)),
        "mismatch_sequences": float(len(token_counts)),
        "nonfinite_metric_count": 0.0,
    }
    if loss_agg_mode == "prompt-mean":
        metrics["mismatch_groups"] = float(sum(count > 0 for count in group_token_counts.values()))
    if not sequence_metrics:
        return metrics

    if loss_agg_mode == "prompt-mean":
        aggregation_weights = [
            count / group_token_counts[group_id] / metrics["mismatch_groups"]
            for count, group_id in zip(token_counts, sequence_group_ids, strict=True)
        ]
    else:
        aggregation_counts = (
            token_counts if loss_agg_mode != "sample-mean" else [1] * len(token_counts)
        )
        denominator = sum(aggregation_counts)
        aggregation_weights = [count / denominator for count in aggregation_counts]
    for name in sequence_metrics[0]:
        values = [row[name] for row in sequence_metrics]
        if not all(math.isfinite(value) for value in values):
            metrics["nonfinite_metric_count"] += 1
            continue
        try:
            value = math.fsum(
                value * weight for value, weight in zip(values, aggregation_weights, strict=True)
            )
        except (OverflowError, ValueError):
            value = math.inf
        if math.isfinite(value):
            metrics[name] = value
        else:
            metrics["nonfinite_metric_count"] += 1
    return metrics
