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

"""Redistribute token advantages once across all trajectories of a training step.

The operator consumes already estimated advantages and does not depend on an
estimator, optimizer batching, loss aggregation, or sequence segmentation. The
reported sums are unweighted over response tokens, not policy gradients. Tokens
with unknown diagnostics retain their original advantages. Sign classification
uses each token's original advantage, so token-valued estimators need not broadcast
a scalar.
"""

import logging
from array import array
from collections import deque
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np

from northrl.northrl_types import Trajectory
from northrl.utils.trajectory_sequences import iter_trajectory_sequences

_UNKNOWN, _CLEAN, _ERROR = 0, 1, 2
_STABLE_REASONS = frozenset(
    {
        "invalid_arguments_json",
        "missing_tool_name",
        "invalid_tool_call_structure",
        "arguments_not_object",
        "duplicate_parameter",
    }
)
_PREFIX = "algorithm/advantage_postprocess/"
_NORMAL_FINISH_REASONS = frozenset({"stop", "tool_calls", "end_turn", "tool_use"})
_EXTERNAL_STOP_REASONS = frozenset({"length", "max_tokens", "stop_sequence"})


def clear_postprocessing_cache(trajectories: list[Trajectory]) -> None:
    """Discard saved bases and shaped outputs before a fresh estimation.

    Call even when postprocessing is now disabled. Token estimates without a
    saved base are untouched. Replaying ``process`` on an existing estimate
    must not call this function, so it can reuse the original base values.
    """
    for trajectory in trajectories:
        for sequence in iter_trajectory_sequences(trajectory):
            fields = sequence.extra_fields if isinstance(sequence, Trajectory) else sequence
            if "base_token_advantages" in fields:
                fields.pop("base_token_advantages")
                fields.pop("token_advantages", None)


@dataclass
class _Sequence:
    record: Trajectory | dict[str, Any]
    base: np.ndarray
    states: np.ndarray
    errors: int
    clean: int
    scalar: float | None


def _classify(annotation: Mapping[str, Any]) -> int:
    """Use explicit error evidence; only completed checks can establish clean turns."""
    if (
        annotation.get("version") != 1
        or annotation.get("source") != "northgate"
        or annotation.get("truncated") is not False
        or annotation.get("finish_reason") in _EXTERNAL_STOP_REASONS
    ):
        return _UNKNOWN
    diagnostics = annotation.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return _UNKNOWN
    reasons = diagnostics.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        return _UNKNOWN
    malformed = diagnostics.get("malformed")
    invalid = diagnostics.get("invalid")
    if (
        diagnostics.get("checked") is True
        and malformed is False
        and not reasons
        and invalid in (None, 0)
    ):
        return _CLEAN
    if malformed is not True and not (
        isinstance(invalid, Real)
        and not isinstance(invalid, bool)
        and isfinite(float(invalid))
        and float(invalid) > 0
    ):
        return _UNKNOWN
    if _STABLE_REASONS.intersection(reasons):
        return _ERROR
    if "undeclared_tool" in reasons and annotation.get("tools_declared") is True:
        return _ERROR
    if (
        "incomplete_tool_call" in reasons
        and annotation.get("finish_reason") in _NORMAL_FINISH_REASONS
    ):
        return _ERROR
    return _UNKNOWN


def _prepare(trajectory: Trajectory, sequence: Trajectory | dict[str, Any]) -> _Sequence:
    tokens, loss_mask = sequence.get("tokens"), sequence.get("loss_mask")
    if tokens is None or loss_mask is None:
        raise ValueError("Advantage postprocessing requires tokens and loss_mask")
    mask = np.asarray(loss_mask)
    if mask.ndim != 1 or len(mask) != len(tokens) or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Advantage postprocessing requires a token-aligned binary loss_mask")
    if len(mask) and mask[0] != 0:
        raise ValueError("loss_mask must exclude token zero, which has no prediction context")
    valid = mask.astype(bool)
    response_count = int(np.count_nonzero(valid))
    original = sequence.get("base_token_advantages")
    if original is None:
        original = sequence.get("token_advantages")
    scalar = None
    if original is None:
        scalar = trajectory.get("advantage", sequence.get("advantage"))
        if scalar is None and response_count:
            raise ValueError("Advantage postprocessing requires an estimated advantage")
        scalar = float(scalar) if response_count else 0.0
        original = np.full(response_count, scalar, dtype=np.float64)
    base = np.asarray(original, dtype=np.float64)
    if base.ndim != 1 or len(base) != response_count or not np.isfinite(base).all():
        raise ValueError("token_advantages must be finite and aligned with loss_mask[1:] responses")
    # Read the original buffer until commit; transformations use separate outputs.
    # Packed inputs therefore need no additional float64 snapshot.
    spans = []
    annotations = sequence.get("turn_annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("turn_annotations must be a list")
    seen_turns: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("Each turn annotation must be a mapping")
        turn_id = annotation.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id or turn_id in seen_turns:
            raise ValueError("Turn annotations require distinct nonempty turn_id values")
        seen_turns.add(turn_id)
        span = annotation.get("response_span")
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(not isinstance(value, Integral) or isinstance(value, bool) for value in span)
        ):
            raise ValueError("response_span must contain two integer token positions")
        start, end = span
        if not 1 <= start < end <= len(tokens):
            raise ValueError(
                "Turn response_span must be nonempty, in bounds, and exclude token zero"
            )
        if not valid[start:end].all():
            raise ValueError("Turn annotation response_span must contain only response tokens")
        spans.append((start, end, _classify(annotation)))
    # Sorted, nonoverlapping response spans let us count masked gaps once. Keep
    # states response-only, rather than allocating states and coverage for all
    # repeated prompt/context tokens in every segment.
    states = np.zeros(response_count, dtype=np.uint8)
    previous_end = response_offset = errors = clean = 0
    for start, end, state in sorted(spans):
        if start < previous_end:
            raise ValueError("Turn annotation response_span values must not overlap")
        response_offset += int(np.count_nonzero(valid[previous_end:start]))
        count = end - start
        states[response_offset : response_offset + count] = state
        response_offset += count
        previous_end = end
        errors += count if state == _ERROR else 0
        clean += count if state == _CLEAN else 0
    return _Sequence(sequence, base, states, errors, clean, scalar)


def _store_values(values: np.ndarray, *, compact: bool) -> list[float] | array:
    """Keep compact trajectories free of per-token Python float objects."""
    if not compact:
        return values.tolist()
    packed = array("d")
    packed.frombytes(memoryview(np.ascontiguousarray(values)).cast("B"))
    return packed


def _advantage_abs_sum(values: np.ndarray, *, positive: bool) -> float:
    selected = values[values > 0] if positive else values[values < 0]
    return abs(float(selected.sum(dtype=np.float64)))


# Reference: MiMo 2.6 Pro RL.
@dataclass(frozen=True)
class ToolErrorRebalancePostprocessor:
    """Redistribute scalar or token advantages over the original complete step."""

    negative_multiplier: float = 2.0
    positive_scale_max: float = 2.0
    negative_scale_min: float = 0.5

    def process(
        self,
        trajectories: list[Trajectory],
        metrics: MutableMapping[str, float],
        *,
        logger: logging.Logger | None = None,
    ) -> list[Trajectory]:
        """Redistribute once across the complete step, before MIS or batch splitting.

        Response-only arrays follow the original ``loss_mask[1:] == 1`` order.
        Saved base advantages make replay idempotent. Call
        ``clear_postprocessing_cache`` before a new estimation. Validate the
        whole step before mutating sequences.
        """
        del logger
        kappa = float(self.negative_multiplier)
        alpha_max = float(self.positive_scale_max)
        beta_min = float(self.negative_scale_min)
        if (
            not all(isfinite(value) for value in (kappa, alpha_max, beta_min))
            or kappa <= 1
            or alpha_max < 1
            or not 0 < beta_min <= 1
        ):
            raise ValueError(
                "Require finite negative_multiplier > 1, positive_scale_max >= 1, "
                "and 0 < negative_scale_min <= 1"
            )

        sequences: deque[_Sequence] = deque()
        counts = dict.fromkeys(
            (
                "response_tokens",
                "error_tokens",
                "clean_tokens",
                "zero_advantage_tokens",
                "zero_advantage_error_tokens",
            ),
            0,
        )
        pool_counts = [0, 0, 0, 0]
        pool_advantage_abs_sums: list[list[float]] = [[], [], [], []]
        positive_before_rows, negative_before_rows = [], []
        for trajectory in trajectories:
            if trajectory.status != "success" or trajectory.get("group_skipped", False):
                continue
            for record in iter_trajectory_sequences(trajectory):
                seq = _prepare(trajectory, record)
                sequences.append(seq)
                counts["response_tokens"] += len(seq.base)
                counts["error_tokens"] += seq.errors
                counts["clean_tokens"] += seq.clean
                if seq.scalar is not None:
                    zeros = len(seq.base) if seq.scalar == 0 else 0
                    zero_errors = seq.errors if seq.scalar == 0 else 0
                    positive_sum = float(seq.base.sum(dtype=np.float64)) if seq.scalar > 0 else 0.0
                    negative_abs_sum = (
                        abs(float(seq.base.sum(dtype=np.float64))) if seq.scalar < 0 else 0.0
                    )
                else:
                    zero_mask = seq.base == 0
                    zeros = int(np.count_nonzero(zero_mask))
                    zero_errors = int(np.count_nonzero(zero_mask & (seq.states == _ERROR)))
                    positive_sum = _advantage_abs_sum(seq.base, positive=True)
                    negative_abs_sum = _advantage_abs_sum(seq.base, positive=False)
                counts["zero_advantage_tokens"] += zeros
                counts["zero_advantage_error_tokens"] += zero_errors
                positive_before_rows.append(positive_sum)
                negative_before_rows.append(negative_abs_sum)
                # Keep the original per-sequence NumPy reduction order, followed
                # by math.fsum across sequences, without retaining selected arrays.
                for index, (state, positive) in enumerate(
                    ((_ERROR, True), (_CLEAN, True), (_ERROR, False), (_CLEAN, False))
                ):
                    if seq.scalar is not None:
                        if not (seq.scalar > 0 if positive else seq.scalar < 0):
                            pool_advantage_abs_sums[index].append(0.0)
                            continue
                        selected = seq.base[seq.states == state]
                    else:
                        selected = seq.base[
                            (seq.states == state) & ((seq.base > 0) if positive else (seq.base < 0))
                        ]
                    pool_counts[index] += len(selected)
                    pool_advantage_abs_sums[index].append(
                        abs(float(selected.sum(dtype=np.float64)))
                    )
                    del selected
        pos_errors, pos_clean, neg_errors, neg_clean = pool_counts
        pos_error_sum, pos_clean_sum, neg_error_abs_sum, neg_clean_abs_sum = map(
            fsum, pool_advantage_abs_sums
        )
        added = (kappa - 1) * neg_error_abs_sum
        alpha = 1 + pos_error_sum / pos_clean_sum if pos_clean_sum else 1.0
        beta = 1 - added / neg_clean_abs_sum if neg_clean_abs_sum else 1.0
        alpha_clamped = alpha > alpha_max
        beta_clamped = beta < beta_min
        alpha, beta = min(alpha, alpha_max), max(beta, beta_min)

        staged: list[
            tuple[Trajectory | dict[str, Any], list[float] | array, list[float] | array]
        ] = []
        positive_after_rows, negative_after_rows = [], []
        while sequences:
            seq = sequences.popleft()
            output = seq.base.copy()
            if seq.scalar is not None:
                if seq.scalar > 0:
                    output[seq.states == _ERROR] = 0
                    output[seq.states == _CLEAN] *= alpha
                elif seq.scalar < 0:
                    output[seq.states == _ERROR] *= kappa
                    output[seq.states == _CLEAN] *= beta
            else:
                positive, negative = seq.base > 0, seq.base < 0
                output[(seq.states == _ERROR) & positive] = 0
                output[(seq.states == _ERROR) & negative] *= kappa
                output[(seq.states == _CLEAN) & positive] *= alpha
                output[(seq.states == _CLEAN) & negative] *= beta
            if not np.isfinite(output).all():
                raise ValueError("Advantage postprocessing produced nonfinite advantages")
            positive_after_rows.append(_advantage_abs_sum(output, positive=True))
            negative_after_rows.append(_advantage_abs_sum(output, positive=False))
            compact = isinstance(seq.record.get("tokens"), array)
            staged.append(
                (
                    seq.record,
                    _store_values(seq.base, compact=compact),
                    _store_values(output, compact=compact),
                )
            )
            # Retain only independent final buffers until commit.
            del seq, output
        positive_before, negative_before = fsum(positive_before_rows), fsum(negative_before_rows)
        positive_after, negative_after = fsum(positive_after_rows), fsum(negative_after_rows)
        stats = {
            **counts,
            "diagnostic_coverage": (counts["error_tokens"] + counts["clean_tokens"])
            / max(counts["response_tokens"], 1),
            "positive_error_tokens": pos_errors,
            "positive_clean_tokens": pos_clean,
            "negative_error_tokens": neg_errors,
            "negative_clean_tokens": neg_clean,
            "positive_error_advantage_sum": pos_error_sum,
            "positive_clean_advantage_sum": pos_clean_sum,
            "negative_error_advantage_abs_sum": neg_error_abs_sum,
            "negative_clean_advantage_abs_sum": neg_clean_abs_sum,
            "positive_scale": alpha,
            "negative_scale": beta,
            "positive_scale_clamped": alpha_clamped,
            "negative_scale_clamped": beta_clamped,
            "positive_no_clean_tokens": pos_error_sum > 0 and not pos_clean_sum,
            "negative_no_clean_tokens": added > 0 and not neg_clean_abs_sum,
            "positive_advantage_sum_before": positive_before,
            "positive_advantage_sum_after": positive_after,
            "negative_advantage_abs_sum_before": negative_before,
            "negative_advantage_abs_sum_after": negative_after,
        }
        for record, base, output in staged:
            record["base_token_advantages"] = base
            record["token_advantages"] = output
        if metrics is not None:
            metrics.update({_PREFIX + key: float(value) for key, value in stats.items()})
        return trajectories
