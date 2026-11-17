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

"""Metrics for one consumed batch, with explicit parent/sequence identities.

Runtime observations belong to a parent trajectory's final execution, except
sandbox_retries and duration_seconds, which cover its entire execution lifetime.
Missing observations are omitted, never synthesized from parent values on segments.
"""

from __future__ import annotations

import logging
import math
from array import array
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from northrl.algorithm.groups import _identity_value
from northrl.northrl_types import Trajectory
from northrl.remote_rollout.metrics import TOOL_FORMAT_REASONS
from northrl.utils.trajectory_sequences import iter_trajectory_sequences
from northrl.utils.verifier_metrics import log_verifier_metrics

logger = logging.getLogger(__name__)

RUNTIME_DISTRIBUTIONS = (
    "model_requests",
    "tool_calls",
    "tool_timeouts",
    "model_request_timeouts",
    "sandbox_retries",
    "duration_seconds",
)
SEGMENT_DISTRIBUTIONS = ("tool_calls", "tool_timeouts", "model_request_timeouts")
RUNTIME_TOTALS = {
    "sandbox_retries": "batch/sandbox_retry_count",
    "tool_calls": "tool/calls_emitted_count",
    "tool_results": "tool/results_observed_count",
    "tool_errors": "tool/result_error_count",
    "tool_unobserved": "tool/unobserved_call_count",
    "tool_unassigned": "tool/unassigned_result_count",
    "model_requests": "model_request/count",
    "model_requests_completed": "model_request/completed_count",
    "request_usage_count": "model_request/token_usage_observed_count",
    "tool_timeouts": "timeout/tool_count",
    "model_request_timeouts": "timeout/model_request_count",
}
TOOL_ERRORS = (
    "command_not_found",
    "dependency_missing",
    "network",
    "permission",
    "resource",
    "file_not_found",
    "edit",
    "test_or_compile",
    "other",
)
TOOL_PARSE = {
    "calls_detected": "calls_detected",
    "calls_invalid": "calls_invalid",
    "format_error_turn": "format_error_turn",
    "format_unknown_turn": "format_unknown_turn",
    "call_counts_unknown_turn": "call_counts_unknown_turn",
    **{
        f"format_reason_{reason}_turn": f"format_reason_{reason}_turn"
        for reason in TOOL_FORMAT_REASONS
    },
    "arguments_json_error": "arguments_json_error",
    "arguments_type_error": "arguments_type_error",
}


def finite_number(value: Any) -> float | None:
    """Accept only finite real observations, including scalar tensor values."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            return None
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _nonnegative(value: Any) -> float | None:
    value = finite_number(value)
    return value if value is not None and value >= 0 else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def group_key(trajectory: Trajectory) -> tuple[Any, Any] | None:
    """Return an explicit model-scoped group; ungrouped data has no fake group."""
    if "group_id" not in trajectory:
        return None
    return (_identity_value(trajectory.get("model_tag")), _identity_value(trajectory["group_id"]))


def _distribution(
    metrics: dict[str, Any], prefix: str, values: Sequence[float], *, mean: bool = True
) -> None:
    if not values:
        return
    metrics[f"{prefix}_min"] = min(values)
    metrics[f"{prefix}_max"] = max(values)
    if mean:
        metrics[f"{prefix}_mean"] = float(np.mean(values))


def _observations(records: list[Mapping[str, Any]], key: str) -> list[float]:
    return [number for record in records if (number := _nonnegative(record.get(key))) is not None]


def _count_total(metrics: dict[str, Any], name: str, values: Sequence[float]) -> None:
    if values:
        total = sum(values)
        metrics[name] = int(total) if float(total).is_integer() else total


def _rollout_timeout(trajectory: Trajectory, runtime: Mapping[str, Any]) -> bool | None:
    for record, key in ((runtime, "rollout_timed_out"), (trajectory, "e2b_timed_out")):
        value = record.get(key)
        if isinstance(value, bool) or value in (0, 1):
            return bool(value)
    return None


def _log_request_metrics(runtime: list[Mapping[str, Any]], metrics: dict[str, Any]) -> None:
    for name in ("input_tokens", "output_tokens", "total_tokens", "duration_seconds"):
        observations = []
        for record in runtime:
            summary = _mapping(_mapping(record.get("request_stats")).get(name))
            count = _nonnegative(summary.get("count"))
            total = _nonnegative(summary.get("sum"))
            minimum = _nonnegative(summary.get("min"))
            maximum = _nonnegative(summary.get("max"))
            if count and total is not None and minimum is not None and maximum is not None:
                observations.append((count, total, minimum, maximum))
        if observations:
            count = sum(item[0] for item in observations)
            metrics[f"model_request/{name}_mean"] = sum(item[1] for item in observations) / count
            metrics[f"model_request/{name}_min"] = min(item[2] for item in observations)
            metrics[f"model_request/{name}_max"] = max(item[3] for item in observations)


def _execution_span_seconds(parents: list[Trajectory]) -> float | None:
    """Measure a nonempty parent cohort only when every wall-clock interval is valid."""
    if not parents:
        return None
    rows = [_mapping(parent.get("runtime_metrics")) for parent in parents]
    starts = _observations(rows, "started_at")
    ends = _observations(rows, "finished_at")
    if len(starts) != len(rows) or len(ends) != len(rows):
        return None
    if any(end < start for start, end in zip(starts, ends, strict=True)):
        return None
    return max(ends) - min(starts)


def _group_duration(parents: list[Trajectory]) -> float | None:
    """Measure a complete sampling group's span, including staggered starts."""
    if any(parent.get("expected_rollouts", len(parents)) != len(parents) for parent in parents):
        return None
    if any("run_id" in parent for parent in parents):
        run_ids = [parent.get("run_id") for parent in parents]
        if any(not isinstance(slot, int) or isinstance(slot, bool) for slot in run_ids):
            return None
        if set(run_ids) != set(range(len(parents))):
            return None
    return _execution_span_seconds(parents)


def _timing_summary(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    runtime = [_mapping(parent.get("runtime_metrics")) for parent in trajectories]
    for field, prefix in (
        ("duration_seconds", "trajectory"),
        ("verifier_duration_seconds", "verifier"),
    ):
        rows = runtime
        if prefix == "verifier":
            rows = [row for row in runtime if row.get("verifier_skipped_length") is not True]
        values = _observations(rows, field)
        metrics[f"{prefix}/duration_seconds_observed_count"] = len(values)
        _distribution(metrics, f"{prefix}/duration_seconds", values)
    metrics["verifier/skipped_length_count"] = sum(
        row.get("verifier_skipped_length") is True for row in runtime
    )
    groups: dict[tuple[Any, Any], list[Trajectory]] = defaultdict(list)
    for parent in trajectories:
        if (
            (key := group_key(parent)) is not None
            and key[1] not in (None, "")
            and not (isinstance(key[1], Real) and not math.isfinite(key[1]))
        ):
            groups[key].append(parent)
    durations = [
        duration
        for parents in groups.values()
        if (duration := _group_duration(parents)) is not None
    ]
    metrics["group/duration_seconds_observed_count"] = len(durations)
    _distribution(metrics, "group/duration_seconds", durations)


def log_timing_metrics(
    trajectories: list[Trajectory], metrics: dict[str, Any], *, prefix: str = ""
) -> None:
    """Publish batch means, their observed counts, and reward-function cohorts.

    Groups are model-scoped sampling groups; trajectories are parent executions.
    Verifier observations cover one complete reward evaluation, including its
    internal retries. Length short-circuits have no verifier duration sample.
    """
    summary: dict[str, Any] = {}
    _timing_summary(trajectories, summary)
    cohorts: dict[str, list[Trajectory]] = defaultdict(list)
    for parent in trajectories:
        name = parent.get("reward_fn")
        if isinstance(name, str) and name:
            cohorts[name].append(parent)
    for name, parents in cohorts.items():
        cohort: dict[str, Any] = {}
        _timing_summary(parents, cohort)
        summary.update({f"reward_fn/{name}/{key}": value for key, value in cohort.items()})
    metrics.update({f"{prefix}{key}": value for key, value in summary.items()})
    logger.info("[timing-metrics] %s", {f"{prefix}{key}": value for key, value in summary.items()})


def log_step_metrics(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    """Aggregate the unfiltered input batch without changing the supplied records."""
    log_verifier_metrics(trajectories, metrics)
    log_timing_metrics(trajectories, metrics)
    _log_reward_function_rewards(trajectories, metrics)
    length_count = sum(parent.get("finish_reason") == "length" for parent in trajectories)
    length_fraction = length_count / len(trajectories) if trajectories else 0.0
    metrics["trajectory/finish_reason_length_count"] = length_count
    metrics["trajectory/finish_reason_length_fraction"] = length_fraction
    logger.info(
        "[finish-reason] training batch: length=%d/%d (%.2f%%)",
        length_count,
        len(trajectories),
        length_fraction * 100,
    )
    sequences = [list(iter_trajectory_sequences(parent)) for parent in trajectories]
    # Managed SampleRef tokens/masks are server-private; they contribute zero local tokens.
    tokens = [[response_token_count(seq["loss_mask"] or ()) for seq in seqs] for seqs in sequences]
    counts = [len(seqs) for seqs in sequences]
    parent_tokens = [sum(values) for values in tokens]
    runtime = [_mapping(parent.get("runtime_metrics")) for parent in trajectories]
    segment_runtime = [_mapping(seq.get("segment_metrics")) for seqs in sequences for seq in seqs]
    metrics.update(
        {
            "batch/trajectory_count": len(trajectories),
            "batch/segment_count": sum(counts),
            "batch/tokens_total": sum(
                len(seq["tokens"] or ()) for seqs in sequences for seq in seqs
            ),
            "batch/response_tokens_total": sum(parent_tokens),
            "batch/failed_trajectory_count": sum(
                parent.status == "failed" for parent in trajectories
            ),
            "batch/skipped_trajectory_count": sum(
                parent.status == "skipped" for parent in trajectories
            ),
            "batch/empty_trajectory_count": sum(count == 0 for count in counts),
        }
    )
    _distribution(
        metrics, "segment/response_tokens", [value for values in tokens for value in values]
    )
    _distribution(metrics, "trajectory/response_tokens", parent_tokens)
    _distribution(metrics, "trajectory/segments", counts)

    groups: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for index, parent in enumerate(trajectories):
        if (key := group_key(parent)) is not None:
            groups[key].append(index)
    if groups:
        metrics["batch/group_count"] = len(groups)
        metrics["batch/skipped_group_count"] = sum(
            any(trajectories[index].get("group_skipped", False) for index in indices)
            for indices in groups.values()
        )
        for name, counts_per_parent in (("response_tokens", parent_tokens), ("segments", counts)):
            _distribution(
                metrics,
                f"group/{name}",
                [sum(counts_per_parent[i] for i in ids) for ids in groups.values()],
                mean=False,
            )
        _distribution(
            metrics, "group/trajectories", [len(ids) for ids in groups.values()], mean=False
        )

    for name in RUNTIME_DISTRIBUTIONS:
        _distribution(metrics, f"trajectory/{name}", _observations(runtime, name))
        if name == "duration_seconds":
            # Group timing is computed once above, with complete-group checks.
            continue
        group_values = []
        for indices in groups.values():
            rows = [runtime[index] for index in indices]
            values = _observations(rows, name)
            if len(values) == len(rows):
                group_values.append(sum(values))
        _distribution(metrics, f"group/{name}", group_values, mean=False)

    for name in SEGMENT_DISTRIBUTIONS:
        _distribution(metrics, f"segment/{name}", _observations(segment_runtime, name))
    for name, destination in RUNTIME_TOTALS.items():
        _count_total(metrics, destination, _observations(runtime, name))
    for name, suffix in (
        ("tool_timeouts", "tool_timeout"),
        ("model_request_timeouts", "model_request_timeout"),
    ):
        parent_values = _observations(runtime, name)
        if parent_values:
            metrics[f"timeout/trajectory_with_{suffix}_count"] = sum(
                value > 0 for value in parent_values
            )
        segment_values = _observations(segment_runtime, name)
        if segment_values:
            metrics[f"timeout/segment_with_{suffix}_count"] = sum(
                value > 0 for value in segment_values
            )
    timed_out = [
        _rollout_timeout(parent, row) for parent, row in zip(trajectories, runtime, strict=True)
    ]
    if any(value is not None for value in timed_out):
        metrics["timeout/rollout_count"] = sum(value is True for value in timed_out)
        metrics["timeout/segment_from_timed_out_rollout_count"] = sum(
            count for count, timeout in zip(counts, timed_out, strict=True) if timeout is True
        )

    for category in TOOL_ERRORS:
        values = _observations(
            [_mapping(row.get("tool_error_counts")) for row in runtime], category
        )
        _count_total(metrics, f"tool/error_{category}_count", values)
    for name, destination in TOOL_PARSE.items():
        rows = [_mapping(row.get("tool_parse_counts")) for row in runtime]
        parse_values: list[float] = []
        for row in rows:
            value = row.get(name, row.get(name + "s"))
            if (number := _nonnegative(value)) is not None:
                parse_values.append(number)
        _count_total(metrics, f"tool/{destination}_count", parse_values)
    if any(runtime) or any(segment_runtime):
        logger.info(
            "[metrics-observation] trajectories=%d segments=%d "
            "parent_observed=%s segment_observed=%s",
            len(trajectories),
            sum(counts),
            {name: len(_observations(runtime, name)) for name in RUNTIME_DISTRIBUTIONS},
            {name: len(_observations(segment_runtime, name)) for name in SEGMENT_DISTRIBUTIONS},
        )
    _log_request_metrics(runtime, metrics)
    _log_outcomes(trajectories, sequences, metrics)


def _log_reward_function_rewards(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    """Count each parent's finite final reward once within its reward function."""
    rewards_by_function: dict[str, list[float]] = defaultdict(list)
    for parent in trajectories:
        name = parent.get("reward_fn")
        if not isinstance(name, str) or not name:
            continue
        rewards = rewards_by_function[name]
        if (reward := finite_number(parent.reward)) is not None:
            rewards.append(reward)
    for name, rewards in rewards_by_function.items():
        prefix = f"reward_fn/{name}/reward"
        metrics[f"{prefix}_observed_count"] = len(rewards)
        _distribution(metrics, prefix, rewards)
        if rewards:
            metrics[f"{prefix}_std"] = float(np.std(rewards))


def _log_outcomes(
    trajectories: list[Trajectory], sequences: list[list[Any]], metrics: dict[str, Any]
) -> None:
    statuses = [parent.get("e2b_status", parent.get("execution_status")) for parent in trajectories]
    if any(status is not None for status in statuses):
        metrics["outcome/env_error_count"] = sum(status == "env_error" for status in statuses)
        metrics["outcome/task_failed_count"] = sum(status == "task_failed" for status in statuses)
    reasons = [
        parent.get("e2b_termination_reason", parent.get("termination_reason"))
        for parent in trajectories
    ]
    if any(reason is not None for reason in reasons):
        metrics["outcome/context_limit_reached_count"] = sum(
            reason == "context_length_exceeded" for reason in reasons
        )
    truncated_segments = [
        bool(seq.get("is_truncated", False)) for seqs in sequences for seq in seqs
    ]
    if any("is_truncated" in seq for seqs in sequences for seq in seqs):
        metrics["segment/truncated_count"] = sum(truncated_segments)
        if truncated_segments and all("is_truncated" in seq for seqs in sequences for seq in seqs):
            metrics["segment/truncated_percentage"] = (
                100 * sum(truncated_segments) / len(truncated_segments)
            )
        metrics["outcome/truncated_trajectory_count"] = sum(
            any(bool(seq.get("is_truncated", False)) for seq in seqs) for seqs in sequences
        )
    resolved = []
    for parent in trajectories:
        score = _mapping(parent.get("score"))
        raw_value = score.get("swebench_resolved", score.get("resolved"))
        value = float(raw_value) if isinstance(raw_value, bool) else finite_number(raw_value)
        if value is not None and value in (0, 1):
            resolved.append(value)
    if resolved:
        metrics["outcome/resolved_count"] = sum(value == 1 for value in resolved)
        metrics["outcome/evaluated_trajectory_count"] = len(resolved)
        metrics["outcome/resolved_percentage"] = 100 * sum(resolved) / len(resolved)
    # Additional task scores are explicitly registered; runtime score keys must
    # not silently become unrelated, per-trajectory averaged W&B fields.
    task_scores: dict[str, list[float]] = defaultdict(list)
    for parent in trajectories:
        score = _mapping(parent.get("score"))
        score_names = set(parent.get("metric_score_names", ())) | ({"correct"} & set(score))
        for name in score_names:
            values = task_scores[str(name)]
            raw_value = score.get(name)
            value = finite_number(int(raw_value) if isinstance(raw_value, bool) else raw_value)
            if value is not None:
                values.append(value)
    for name, values in task_scores.items():
        if values:
            metrics[f"outcome/score_{name}_mean"] = float(np.mean(values))
        metrics[f"outcome/score_{name}_observed_count"] = len(values)


def response_token_count(mask: Any) -> int:
    """Count positive mask entries without changing special-type behavior."""
    # Subclasses may override iteration; retain their legacy path.
    # pylint: disable=unidiomatic-typecheck
    if type(mask) is array and mask.typecode in "bBhHiIlLqQfd":
        return int(np.count_nonzero(np.asarray(mask) > 0))
    return sum(value > 0 for value in mask)


def _effective_masks_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    # Subclasses may override iteration; retain their legacy path.
    # pylint: disable=unidiomatic-typecheck
    if all(type(mask) is array and mask.typecode in "bBhHiIlLqQfd" for mask in (left, right)):
        return bool(np.array_equal(np.asarray(left) > 0, np.asarray(right) > 0))
    return not any((a > 0) != (b > 0) for a, b in zip(left, right, strict=True))


def datum_response_token_count(datum: Mapping[str, Any]) -> int:
    """Count the final effective mask; managed masks remain server-private."""
    if datum.get("kind") == "sample_ref":
        return 0
    return response_token_count(datum["loss_fn_inputs"]["loss_mask"])


def log_trainable_metrics(
    trajectories: list[Trajectory],
    datums_data: Sequence[Mapping[str, Any]],
    metrics: dict[str, Any],
    *,
    datum_sources: Sequence[tuple[int, int]],
) -> list[int]:
    """Count original parents/sequences from final masks, before optimizer padding.

    ``datum_sources[i]`` identifies Datum i by (parent index, segment index) in
    the supplied prepared trajectory list. A flat parent uses segment index 0.
    The mapping is captured by conversion; token equality is not an identity.
    Duplicate identities are ignored only when their effective masks agree.
    Return per-Datum counts, including duplicates, for this training call to
    reuse after splitting. Masks must be final before calling this function.
    """
    if len(datums_data) != len(datum_sources):
        raise ValueError("Every Datum requires its original parent/segment identity")
    parent_sequences = [list(iter_trajectory_sequences(parent)) for parent in trajectories]
    masks: dict[tuple[int, int], Sequence[Any]] = {}
    counts: dict[tuple[int, int], int] = {}
    datum_counts: list[int] = []
    # Managed SampleRef datums are trainable by construction, but their masks
    # stay server-private, so they add parents/segments and zero local tokens.
    managed_sources: set[tuple[int, int]] = set()
    for datum, source in zip(datums_data, datum_sources, strict=True):
        parent_index, segment_index = source
        if not (0 <= parent_index < len(trajectories)) or not (
            0 <= segment_index < len(parent_sequences[parent_index])
        ):
            raise ValueError(f"Unknown Datum source {source}")
        parent = trajectories[parent_index]
        if parent.status != "success" or parent.get("group_skipped", False):
            raise ValueError("A trainable Datum cannot originate from a failed/skipped parent")
        if datum.get("kind") == "sample_ref":
            managed_sources.add(source)
            datum_counts.append(0)
            continue
        mask = datum["loss_fn_inputs"]["loss_mask"]
        if source in masks:
            previous = masks[source]
            if len(previous) != len(mask) or not _effective_masks_equal(previous, mask):
                raise ValueError("Duplicate Datum source has inconsistent effective masks")
        else:
            # Retain the existing mask only for duplicate validation instead of
            # allocating one full boolean tuple for every training sequence.
            masks[source] = mask
            counts[source] = datum_response_token_count(datum)
        datum_counts.append(counts[source])
    active_sources = {source for source, count in counts.items() if count > 0} | managed_sources
    active_parents = {source[0] for source in active_sources}
    metrics.update(
        {
            "batch/trainable_trajectory_count": len(active_parents),
            "batch/trainable_segment_count": len(active_sources),
            "batch/trainable_response_tokens_total": sum(counts.values()),
        }
    )
    if any(group_key(parent) is not None for parent in trajectories):
        metrics["batch/trainable_group_count"] = len(
            {key for index in active_parents if (key := group_key(trajectories[index])) is not None}
        )
    else:
        metrics.pop("batch/trainable_group_count", None)
    return datum_counts
