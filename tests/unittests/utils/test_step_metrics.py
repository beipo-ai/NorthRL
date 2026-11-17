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

"""Batch ownership, incomplete observations, and final-mask counting contracts."""

import math

import pytest

from northrl.northrl_types import Trajectory
from northrl.remote_rollout.metrics import TOOL_FORMAT_REASONS, RolloutMetrics
from northrl.utils.logging_utils import log_grpo_metrics, log_logprob_metrics, log_rollout_metrics
from northrl.utils.metric_catalog import METRIC_SPECS, normalize_step_metrics
from northrl.utils.step_metrics import log_step_metrics, log_timing_metrics, log_trainable_metrics
from northrl.utils.trajectory_sequences import iter_training_sequences, iter_trajectory_sequences


def _segment(count, **extra):
    return {"tokens": list(range(count + 1)), "loss_mask": [0] + [1] * count, **extra}


def _parent(group, counts, runtime=None, **extra):
    fields = {"group_id": group, **extra}
    if runtime is not None:
        fields["runtime_metrics"] = runtime
    return Trajectory(segments=[_segment(count) for count in counts], reward=0, extra_fields=fields)


def test_northgate_diagnostics_reach_step_metrics_once_per_parent():
    observed = RolloutMetrics()
    observed.record_model_result(
        {
            "tool_call_diagnostics": {
                "checked": True,
                "malformed": True,
                "detected": 2,
                "invalid": 1,
                "reasons": list(TOOL_FORMAT_REASONS),
            }
        }
    )
    unknown = RolloutMetrics()
    unknown.record_model_result(
        {
            "tool_call_diagnostics": {
                "checked": True,
                "malformed": True,
                "detected": None,
                "invalid": None,
                "reasons": ["incomplete_tool_call"],
            }
        }
    )
    metrics: dict[str, float] = {}
    log_rollout_metrics(
        [
            _parent("a", [2, 3], observed.runtime_snapshot()),
            _parent("a", [], unknown.runtime_snapshot()),
            _parent("b", [1]),  # Old/custom artifacts without diagnostics stay absent.
        ],
        metrics,
    )
    assert metrics["tool/calls_detected_count"] == 2
    assert metrics["tool/calls_invalid_count"] == 1
    assert metrics["tool/format_error_turn_count"] == 2
    assert metrics["tool/format_unknown_turn_count"] == 0
    assert metrics["tool/call_counts_unknown_turn_count"] == 1
    for reason in TOOL_FORMAT_REASONS:
        name = f"tool/format_reason_{reason}_turn_count"
        assert metrics[name] == (2 if reason == "incomplete_tool_call" else 1)
        assert name in METRIC_SPECS
    assert "tool/call_counts_unknown_turn_count" in METRIC_SPECS
    assert normalize_step_metrics(metrics)["tool/calls_invalid_count"] == 1


def test_one_batch_parent_segment_group_counts_include_failed_empty_and_skipped():
    parents = [_parent("a", [2, 4]), _parent("a", [3]), _parent("b", [8]), _parent("b", [])]
    parents[1]["group_skipped"] = True
    parents[3].status = "failed"
    parents[3].reward = None
    metrics = {}
    log_rollout_metrics(parents, metrics)
    assert metrics["batch/trajectory_count"] == 4
    assert metrics["batch/segment_count"] == 4
    assert metrics["batch/group_count"] == 2
    assert metrics["batch/skipped_group_count"] == 1
    assert metrics["batch/empty_trajectory_count"] == 1
    assert metrics["batch/failed_trajectory_count"] == 1
    assert metrics["batch/tokens_total"] == 21
    assert metrics["batch/response_tokens_total"] == 17
    assert metrics["segment/response_tokens_mean"] == 17 / 4
    assert metrics["trajectory/response_tokens_mean"] == 17 / 4
    assert metrics["trajectory/response_tokens_min"] == 0
    assert metrics["trajectory/response_tokens_max"] == 8
    assert metrics["group/response_tokens_min"] == 8
    assert metrics["group/response_tokens_max"] == 9
    assert not any(key.startswith(("rollout/", "coverage/")) for key in metrics)
    assert not any(key.startswith("group/") and key.endswith("_mean") for key in metrics)


def test_batch_tokens_include_context_and_masked_sequences_and_reset_each_step():
    parents = [
        Trajectory(tokens=[1, 2, 3, 4, 5, 6], loss_mask=[0, 0, 1, 0, 1, 0], reward=1),
        Trajectory(
            segments=[
                {"tokens": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 0]},
                {"tokens": [1, 2, 3], "loss_mask": [0, 0, 0]},
            ],
            status="failed",
        ),
    ]
    metrics: dict[str, float] = {}

    log_rollout_metrics(parents, metrics)

    assert metrics["batch/tokens_total"] == 13
    assert metrics["batch/response_tokens_total"] == 3

    log_rollout_metrics([parents[0]], metrics)
    assert metrics["batch/tokens_total"] == 6
    assert metrics["batch/response_tokens_total"] == 2


def test_managed_sample_refs_contribute_no_local_batch_tokens():
    managed = Trajectory(
        extra_fields={
            "sample_ref": {"dataset": "test", "version": "v1", "sample_idx": 0},
            "input_token_count": 7,
            "datum_id": "managed-0",
        }
    )
    metrics: dict[str, float] = {}

    log_rollout_metrics([managed, _parent("a", [2])], metrics)

    assert metrics["batch/tokens_total"] == 3
    assert metrics["batch/response_tokens_total"] == 2


def test_empty_flat_result_is_zero_segments_and_ungrouped_has_no_group_metrics():
    parent = Trajectory(tokens=[], loss_mask=[], reward=None, status="failed")
    assert list(iter_trajectory_sequences(parent)) == []
    metrics = {}
    log_rollout_metrics([parent], metrics)
    assert metrics["batch/empty_trajectory_count"] == 1
    assert metrics["batch/segment_count"] == 0
    assert metrics["batch/tokens_total"] == 0
    assert metrics["trajectory/response_tokens_mean"] == 0
    assert "segment/response_tokens_mean" not in metrics
    assert "batch/group_count" not in metrics
    assert "batch/sandbox_retry_count" not in metrics
    assert "timeout/rollout_count" not in metrics


def test_length_finish_reason_counts_parents_instead_of_segments(caplog):
    parents = [
        _parent("a", [2, 4], finish_reason="length"),
        _parent("a", [], finish_reason="length"),
        _parent("b", [8], finish_reason="stop"),
        _parent("b", [1]),
    ]
    parents[1].status = "failed"
    parents[1].reward = None
    # A segment's finish reason cannot override the parent observation.
    parents[2].segments[0]["finish_reason"] = "length"
    metrics = {}
    with caplog.at_level("INFO", logger="northrl.utils.step_metrics"):
        log_rollout_metrics(parents, metrics)
    assert metrics["trajectory/finish_reason_length_count"] == 2
    assert metrics["trajectory/finish_reason_length_fraction"] == 0.5
    assert "training batch: length=2/4 (50.00%)" in caplog.text

    # Counters describe each consumed batch, rather than accumulating process state.
    log_rollout_metrics([_parent("next", [1], finish_reason="stop")], metrics)
    assert metrics["trajectory/finish_reason_length_count"] == 0
    assert metrics["trajectory/finish_reason_length_fraction"] == 0


def test_empty_batch_has_zero_length_count_and_fraction():
    metrics = {}
    log_rollout_metrics([], metrics)
    assert metrics["batch/tokens_total"] == 0
    assert metrics["trajectory/finish_reason_length_count"] == 0
    assert metrics["trajectory/finish_reason_length_fraction"] == 0
    assert "timing/batch_rollout_span_seconds" not in metrics


def test_reward_function_rewards_count_final_parent_rewards_once():
    parents = [
        _parent("a", [2, 4], reward_fn="code_e2b"),
        _parent("a", [3], reward_fn="code_e2b", group_skipped=True),
        _parent("b", [], reward_fn="code_e2b"),
        _parent("c", [1], reward_fn="math"),
    ]
    for parent, reward in zip(parents, [-2, 0, 4, 8], strict=True):
        parent.reward = reward
    parents[2].status = "failed"
    # Segment-local scores and names must not duplicate or replace parent results.
    parents[0].segments[0].update(reward=100, reward_fn="segment_only")
    parents[0].segments[1].update(reward=-100, reward_fn="segment_only")
    metrics = {}

    log_rollout_metrics(parents, metrics)

    assert metrics["reward_fn/code_e2b/reward_observed_count"] == 3
    assert metrics["reward_fn/code_e2b/reward_mean"] == pytest.approx(2 / 3)
    assert metrics["reward_fn/code_e2b/reward_min"] == -2
    assert metrics["reward_fn/code_e2b/reward_max"] == 4
    assert metrics["reward_fn/code_e2b/reward_std"] == pytest.approx(math.sqrt(56 / 9))
    assert metrics["reward_fn/math/reward_observed_count"] == 1
    for statistic in ("mean", "min", "max"):
        assert metrics[f"reward_fn/math/reward_{statistic}"] == 8
    assert metrics["reward_fn/math/reward_std"] == 0
    assert not any(key.startswith("reward_fn/segment_only/") for key in metrics)
    # The existing all-functions reward population remains unchanged.
    assert metrics["trajectory/reward_observed_count"] == 4
    assert metrics["trajectory/reward_mean"] == 2.5


@pytest.mark.parametrize(
    "invalid_reward", [None, float("nan"), float("inf"), -float("inf"), True, False, "3"]
)
def test_reward_function_rewards_ignore_invalid_observations_but_keep_zero(invalid_reward):
    parents = [
        _parent("a", [1], reward_fn="mixed"),
        _parent("a", [2], reward_fn="mixed"),
        _parent("b", [], reward_fn="missing"),
    ]
    parents[0].reward = invalid_reward
    parents[2].reward = invalid_reward
    metrics = {}

    log_step_metrics(parents, metrics)

    assert metrics["reward_fn/mixed/reward_observed_count"] == 1
    for statistic in ("mean", "min", "max", "std"):
        assert metrics[f"reward_fn/mixed/reward_{statistic}"] == 0
        assert f"reward_fn/missing/reward_{statistic}" not in metrics
    assert metrics["reward_fn/missing/reward_observed_count"] == 0


def test_reward_function_rewards_are_batch_local_and_require_a_nonempty_string_name():
    first = _parent("a", [1], reward_fn="math")
    first.reward = 7
    first_metrics = {}
    log_step_metrics([first], first_metrics)
    assert first_metrics["reward_fn/math/reward_mean"] == 7

    second = _parent("b", [1], reward_fn="math")
    invalid_names = [None, "", False, 42, ["math"]]
    next_parents = [second] + [_parent("invalid", [1], reward_fn=name) for name in invalid_names]
    next_metrics = {}
    log_step_metrics(next_parents, next_metrics)
    reward_metrics = {
        key: value
        for key, value in next_metrics.items()
        if key.startswith("reward_fn/") and "/reward_" in key
    }
    assert reward_metrics == {
        "reward_fn/math/reward_observed_count": 1,
        "reward_fn/math/reward_mean": 0,
        "reward_fn/math/reward_min": 0,
        "reward_fn/math/reward_max": 0,
        "reward_fn/math/reward_std": 0,
    }
    empty_metrics = {}
    log_step_metrics([], empty_metrics)
    assert not any(key.startswith("reward_fn/") for key in empty_metrics)


def test_group_timing_survives_retirement_of_consumed_batch_span():
    parents = [
        _parent(
            "a",
            [2, 3],
            {"started_at": 100, "finished_at": 110},
            expected_rollouts=2,
            run_id=0,
            reward_fn="code_e2b",
        ),
        _parent(
            "a",
            [1],
            {"started_at": 105, "finished_at": 120},
            expected_rollouts=2,
            run_id=1,
            reward_fn="code_e2b",
        ),
        _parent(
            "b",
            [],
            {"started_at": 200, "finished_at": 207},
            expected_rollouts=3,
            run_id=5,
            group_skipped=True,
            reward_fn="ifeval",
        ),
    ]
    parents[-1].status = "failed"
    parents[-1].reward = None
    # Segment observations and later filtering do not define the consumed batch.
    parents[0].segments[0]["runtime_metrics"] = {"started_at": 0, "finished_at": 999}
    metrics = {}

    log_rollout_metrics(parents, metrics)

    assert "timing/batch_rollout_span_seconds" not in metrics
    assert metrics["batch/trajectory_count"] == 3
    assert metrics["group/duration_seconds_observed_count"] == 1
    assert metrics["group/duration_seconds_mean"] == 20
    assert not any(key.startswith("reward_fn/") and "batch_rollout_span" in key for key in metrics)
    log_trainable_metrics(
        parents, [_datum([1]), _datum([1])], metrics, datum_sources=[(0, 0), (0, 0)]
    )
    assert metrics["batch/trainable_trajectory_count"] == 1
    assert "timing/batch_rollout_span_seconds" not in metrics


@pytest.mark.parametrize(
    "invalid_runtime",
    [
        None,
        {},
        {"finished_at": 120},
        {"started_at": 105},
        {"started_at": float("nan"), "finished_at": 120},
        {"started_at": 105, "finished_at": float("inf")},
        {"started_at": -1, "finished_at": 120},
        {"started_at": 105, "finished_at": -1},
        {"started_at": 120, "finished_at": 105},
        {"started_at": True, "finished_at": 120},
        {"started_at": 105, "finished_at": False},
        {"started_at": "105", "finished_at": 120},
        {"started_at": 105, "finished_at": "120"},
    ],
)
def test_group_timing_omits_invalid_groups_without_discarding_valid_groups(invalid_runtime):
    parents = [
        _parent("valid", [1], {"started_at": 100, "finished_at": 110}),
        _parent("incomplete", [], invalid_runtime, group_skipped=True),
    ]
    metrics = {}

    log_rollout_metrics(parents, metrics)

    assert "timing/batch_rollout_span_seconds" not in metrics
    assert metrics["group/duration_seconds_observed_count"] == 1
    assert metrics["group/duration_seconds_mean"] == 10


def test_validation_timing_does_not_publish_consumed_training_batch_span():
    metrics = {}

    log_timing_metrics(
        [_parent("a", [1], {"started_at": 100, "finished_at": 110})],
        metrics,
        prefix="val/",
    )

    assert metrics["val/group/duration_seconds_mean"] == 10
    assert not any("batch_rollout_span" in key for key in metrics)


def test_runtime_only_final_execution_but_retries_and_lifetime_are_parent_owned():
    parent = _parent(
        "a",
        [2, 3],
        {
            "tool_calls": 7,
            "tool_timeouts": 2,
            "model_request_timeouts": 1,
            "sandbox_retries": 3,
            "duration_seconds": 120,
            "started_at": 1000,
            "finished_at": 1120,
            "rollout_timed_out": True,
        },
    )
    parent["score"] = {"tool_calls_emitted": 99, "generated_tokens": 99999}
    parent.segments[0]["segment_metrics"] = {"tool_calls": 3, "tool_timeouts": 1}
    # This segment intentionally has no event association. Do not copy parent=2.
    sequences = list(iter_training_sequences([parent]))
    assert all("runtime_metrics" not in segment for segment in sequences)
    assert "segment_metrics" not in sequences[1]
    metrics = {}
    log_rollout_metrics([parent], metrics)
    assert metrics["batch/sandbox_retry_count"] == 3
    assert metrics["trajectory/sandbox_retries_mean"] == 3
    assert metrics["trajectory/duration_seconds_mean"] == 120
    assert metrics["tool/calls_emitted_count"] == 7
    assert metrics["segment/tool_calls_mean"] == 3
    assert metrics["timeout/tool_count"] == 2
    assert metrics["timeout/trajectory_with_tool_timeout_count"] == 1
    assert metrics["timeout/segment_with_tool_timeout_count"] == 1
    assert metrics["timeout/segment_from_timed_out_rollout_count"] == 2
    assert metrics["timeout/model_request_count"] == 1
    assert "timeout/segment_with_model_request_timeout_count" not in metrics
    assert not any(key.startswith("segment/duration") for key in metrics)
    next_metrics = {}
    log_rollout_metrics([_parent("next", [1], {"sandbox_retries": 0})], next_metrics)
    assert next_metrics["batch/sandbox_retry_count"] == 0
    assert "tool/calls_emitted_count" not in next_metrics


def test_incomplete_group_runtime_is_excluded_and_group_duration_is_span():
    parents = [
        _parent(
            "a",
            [1],
            {"tool_calls": 2, "started_at": 100, "finished_at": 110, "duration_seconds": 10},
        ),
        _parent("a", [1], {"started_at": 105, "finished_at": 120, "duration_seconds": 15}),
        _parent(
            "b",
            [1],
            {"tool_calls": 9, "started_at": 200, "finished_at": 207, "duration_seconds": 7},
        ),
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)
    assert metrics["trajectory/tool_calls_mean"] == 5.5
    assert metrics["group/tool_calls_min"] == metrics["group/tool_calls_max"] == 9
    assert metrics["group/duration_seconds_min"] == 7
    assert metrics["group/duration_seconds_max"] == 20


def test_model_request_mean_is_pooled_by_requests_not_parent_means():
    parents = [
        _parent(
            "a",
            [1],
            {
                "model_requests": 1,
                "request_usage_count": 1,
                "request_stats": {
                    "input_tokens": {"count": 1, "sum": 100, "min": 100, "max": 100},
                    "duration_seconds": {"count": 1, "sum": 2, "min": 2, "max": 2},
                },
            },
        ),
        _parent(
            "a",
            [1],
            {
                "model_requests": 3,
                "request_usage_count": 2,
                "request_stats": {
                    "input_tokens": {"count": 2, "sum": 50, "min": 20, "max": 30},
                    "duration_seconds": {"count": 3, "sum": 18, "min": 4, "max": 8},
                },
            },
        ),
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)
    assert metrics["model_request/count"] == 4
    assert metrics["model_request/token_usage_observed_count"] == 3
    assert metrics["model_request/input_tokens_mean"] == 50
    assert metrics["model_request/input_tokens_min"] == 20
    assert metrics["model_request/input_tokens_max"] == 100
    assert metrics["model_request/duration_seconds_mean"] == 5
    assert "model_request/output_tokens_mean" not in metrics


def test_batch_timing_uses_complete_group_spans_and_counts_parent_trajectories_once():
    parents = [
        _parent(
            "a",
            [2, 3],
            {
                "started_at": 100,
                "finished_at": 110,
                "duration_seconds": 10,
                "verifier_duration_seconds": 2,
            },
            expected_rollouts=2,
            reward_fn="code_e2b",
        ),
        _parent(
            "a",
            [1],
            {
                "started_at": 105,
                "finished_at": 120,
                "duration_seconds": 15,
                "verifier_duration_seconds": 4,
            },
            expected_rollouts=2,
            reward_fn="code_e2b",
        ),
        _parent(
            "b",
            [1, 2],
            {
                "started_at": 200,
                "finished_at": 207,
                "duration_seconds": 7,
                "verifier_duration_seconds": 1,
            },
            expected_rollouts=1,
            reward_fn="ifeval",
        ),
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["batch/segment_count"] == 5
    assert metrics["group/duration_seconds_observed_count"] == 2
    assert metrics["group/duration_seconds_mean"] == 13.5
    assert metrics["group/duration_seconds_min"] == 7
    assert metrics["group/duration_seconds_max"] == 20
    assert metrics["trajectory/duration_seconds_observed_count"] == 3
    assert metrics["trajectory/duration_seconds_mean"] == pytest.approx(32 / 3)
    assert metrics["trajectory/duration_seconds_min"] == 7
    assert metrics["trajectory/duration_seconds_max"] == 15
    assert metrics["verifier/duration_seconds_observed_count"] == 3
    assert metrics["verifier/duration_seconds_mean"] == pytest.approx(7 / 3)
    assert metrics["verifier/duration_seconds_min"] == 1
    assert metrics["verifier/duration_seconds_max"] == 4
    assert metrics["reward_fn/code_e2b/group/duration_seconds_mean"] == 20
    assert metrics["reward_fn/code_e2b/trajectory/duration_seconds_observed_count"] == 2
    assert metrics["reward_fn/code_e2b/trajectory/duration_seconds_mean"] == 12.5
    assert metrics["reward_fn/ifeval/group/duration_seconds_mean"] == 7
    assert not any("task/duration_seconds_" in key for key in metrics)


@pytest.mark.parametrize(
    "invalid_runtime",
    [
        {"finished_at": 120},
        {"started_at": float("nan"), "finished_at": 120},
        {"started_at": 105, "finished_at": float("inf")},
        {"started_at": 120, "finished_at": 105},
        {"started_at": -1, "finished_at": 120},
    ],
)
def test_group_timing_excludes_entire_group_if_a_member_has_invalid_timestamps(invalid_runtime):
    parents = [
        _parent("bad", [1], {"started_at": 100, "finished_at": 110}, expected_rollouts=2),
        _parent("bad", [1], invalid_runtime, expected_rollouts=2),
        _parent("valid", [1], {"started_at": 200, "finished_at": 207}, expected_rollouts=1),
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["group/duration_seconds_observed_count"] == 1
    assert metrics["group/duration_seconds_mean"] == 7
    assert metrics["group/duration_seconds_min"] == metrics["group/duration_seconds_max"] == 7


def test_incomplete_expected_group_excludes_span_but_keeps_observed_trajectory_duration():
    parents = [
        _parent(
            "incomplete",
            [1],
            {"started_at": 100, "finished_at": 110, "duration_seconds": 10},
            expected_rollouts=2,
            reward_fn="code_e2b",
        )
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["group/duration_seconds_observed_count"] == 0
    assert "group/duration_seconds_mean" not in metrics
    assert "group/duration_seconds_min" not in metrics
    assert "group/duration_seconds_max" not in metrics
    assert metrics["trajectory/duration_seconds_observed_count"] == 1
    assert metrics["trajectory/duration_seconds_mean"] == 10
    assert metrics["reward_fn/code_e2b/group/duration_seconds_observed_count"] == 0
    assert "reward_fn/code_e2b/group/duration_seconds_mean" not in metrics
    assert "reward_fn/code_e2b/group/duration_seconds_min" not in metrics
    assert "reward_fn/code_e2b/group/duration_seconds_max" not in metrics


def test_group_timing_keeps_same_group_id_for_different_models_separate():
    parents = [
        _parent(
            "same",
            [1],
            {"started_at": start, "finished_at": end},
            model_tag=model,
            expected_rollouts=2,
        )
        for model, start, end in [
            ("actor", 100, 110),
            ("actor", 105, 120),
            ("other", 200, 203),
            ("other", 201, 209),
        ]
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["group/duration_seconds_observed_count"] == 2
    assert metrics["group/duration_seconds_mean"] == 14.5
    assert metrics["group/duration_seconds_min"] == 9
    assert metrics["group/duration_seconds_max"] == 20


@pytest.mark.parametrize(
    "group_fields",
    [
        {},
        {"group_id": None},
        {"group_id": ""},
        {"group_id": float("nan")},
        {"group_id": float("inf")},
    ],
)
def test_timing_ignores_missing_or_empty_group_identity(group_fields):
    parent = Trajectory(
        tokens=[1, 2],
        loss_mask=[0, 1],
        reward=0,
        extra_fields={
            **group_fields,
            "runtime_metrics": {
                "started_at": 100,
                "finished_at": 110,
                "duration_seconds": 10,
            },
        },
    )
    metrics = {}
    log_rollout_metrics([parent], metrics)

    assert metrics["group/duration_seconds_observed_count"] == 0
    assert "group/duration_seconds_mean" not in metrics
    assert "group/duration_seconds_min" not in metrics
    assert "group/duration_seconds_max" not in metrics
    assert metrics["trajectory/duration_seconds_observed_count"] == 1
    assert metrics["trajectory/duration_seconds_mean"] == 10


@pytest.mark.parametrize(
    "slot_fields",
    [
        [{"run_id": 0}, {"run_id": 0}],
        [{"run_id": 0}, {"run_id": 2}],
        [{"run_id": 0}, {}],
        [{}, {"run_id": 1}],
        [{"run_id": False}, {"run_id": 1}],
        [{"run_id": 0}, {"run_id": True}],
        [{"run_id": -1}, {"run_id": 0}],
        [{"run_id": "0"}, {"run_id": 1}],
    ],
    ids=[
        "duplicate",
        "out-of-range",
        "missing-second",
        "missing-first",
        "false",
        "true",
        "negative",
        "string",
    ],
)
def test_timing_invalid_run_slots_exclude_group_span_but_preserve_parent_means(slot_fields):
    parents = [
        _parent(
            "same",
            [1],
            {
                "started_at": start,
                "finished_at": end,
                "duration_seconds": end - start,
                "verifier_duration_seconds": verifier,
            },
            expected_rollouts=2,
            reward_fn="code_e2b",
            **slots,
        )
        for slots, start, end, verifier in zip(
            slot_fields, [100, 105], [110, 120], [2, 4], strict=True
        )
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["group/duration_seconds_observed_count"] == 0
    assert "group/duration_seconds_mean" not in metrics
    assert "group/duration_seconds_min" not in metrics
    assert "group/duration_seconds_max" not in metrics
    assert metrics["reward_fn/code_e2b/group/duration_seconds_observed_count"] == 0
    assert "reward_fn/code_e2b/group/duration_seconds_mean" not in metrics
    assert "reward_fn/code_e2b/group/duration_seconds_min" not in metrics
    assert "reward_fn/code_e2b/group/duration_seconds_max" not in metrics
    assert metrics["trajectory/duration_seconds_observed_count"] == 2
    assert metrics["trajectory/duration_seconds_mean"] == 12.5
    assert metrics["verifier/duration_seconds_observed_count"] == 2
    assert metrics["verifier/duration_seconds_mean"] == 3


def test_timing_reverse_run_slot_order_keeps_complete_group_span():
    parents = [
        _parent(
            "same",
            [1],
            {"started_at": start, "finished_at": end},
            expected_rollouts=2,
            run_id=run_id,
        )
        for run_id, start, end in [(1, 100, 110), (0, 105, 120)]
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["group/duration_seconds_observed_count"] == 1
    assert metrics["group/duration_seconds_mean"] == 20


@pytest.mark.parametrize("invalid", [None, -1, float("nan"), float("inf"), True, "slow"])
def test_timing_means_exclude_invalid_observations_and_keep_real_zero_durations(invalid):
    parents = [
        _parent(
            str(index),
            [1],
            {"duration_seconds": value, "verifier_duration_seconds": value},
        )
        for index, value in enumerate([0, 2, invalid])
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    for prefix in ("trajectory", "verifier"):
        assert metrics[f"{prefix}/duration_seconds_observed_count"] == 2
        assert metrics[f"{prefix}/duration_seconds_mean"] == 1
        assert metrics[f"{prefix}/duration_seconds_min"] == 0
        assert metrics[f"{prefix}/duration_seconds_max"] == 2
    assert metrics["group/duration_seconds_observed_count"] == 0


def test_length_short_circuit_does_not_reduce_verifier_mean():
    parents = [
        _parent(
            "called",
            [1],
            {"duration_seconds": 10, "verifier_duration_seconds": 4},
            reward_fn="ifeval",
        ),
        _parent(
            "length",
            [1, 2],
            {
                "duration_seconds": 15,
                "verifier_skipped_length": True,
                "verifier_duration_seconds": 0,
            },
            finish_reason="length",
            reward_fn="ifeval",
        ),
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["trajectory/duration_seconds_observed_count"] == 2
    assert metrics["trajectory/duration_seconds_mean"] == 12.5
    assert metrics["verifier/duration_seconds_observed_count"] == 1
    assert metrics["verifier/duration_seconds_mean"] == 4
    assert metrics["verifier/skipped_length_count"] == 1
    assert metrics["reward_fn/ifeval/verifier/duration_seconds_observed_count"] == 1
    assert metrics["reward_fn/ifeval/verifier/duration_seconds_mean"] == 4
    assert metrics["reward_fn/ifeval/verifier/skipped_length_count"] == 1


def test_reward_function_timing_is_pooled_by_trajectory_count_and_batches_are_independent():
    parents = [
        _parent(
            str(index),
            [1],
            {"duration_seconds": duration, "verifier_duration_seconds": duration},
            reward_fn=name,
        )
        for index, (name, duration) in enumerate(
            [("ifeval", 10), ("code_e2b", 2), ("code_e2b", 4), ("code_e2b", 6)]
        )
    ]
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["verifier/duration_seconds_mean"] == 5.5
    assert metrics["trajectory/duration_seconds_mean"] == 5.5
    assert metrics["verifier/duration_seconds_observed_count"] == 4
    assert metrics["reward_fn/ifeval/verifier/duration_seconds_mean"] == 10
    assert metrics["reward_fn/ifeval/verifier/duration_seconds_observed_count"] == 1
    assert metrics["reward_fn/code_e2b/verifier/duration_seconds_mean"] == 4
    assert metrics["reward_fn/code_e2b/verifier/duration_seconds_observed_count"] == 3

    next_metrics = {}
    log_rollout_metrics([_parent("next", [1])], next_metrics)
    assert next_metrics["trajectory/duration_seconds_observed_count"] == 0
    assert next_metrics["verifier/duration_seconds_observed_count"] == 0
    assert next_metrics["group/duration_seconds_observed_count"] == 0
    assert next_metrics["verifier/skipped_length_count"] == 0
    assert "verifier/duration_seconds_mean" not in next_metrics
    assert "trajectory/duration_seconds_mean" not in next_metrics
    assert not any(key.startswith("reward_fn/") for key in next_metrics)


def _datum(mask):
    return {"loss_fn_inputs": {"loss_mask": mask}}


def test_trainable_final_masks_identity_mapping_and_duplicate_padding():
    parents = [_parent("a", [3, 3]), _parent("a", [2]), _parent("b", [2])]
    datums = [
        _datum([1, 0, 1]),
        _datum([0, 0, 0]),
        _datum([0, 0]),
        _datum([1, 1]),
        _datum([1, 0, 1]),
    ]
    metrics = {}
    counts = log_trainable_metrics(
        parents, datums, metrics, datum_sources=[(0, 0), (0, 1), (1, 0), (2, 0), (0, 0)]
    )
    assert counts == [2, 0, 0, 2, 2]
    assert metrics == {
        "batch/trainable_trajectory_count": 2,
        "batch/trainable_segment_count": 2,
        "batch/trainable_response_tokens_total": 4,
        "batch/trainable_group_count": 2,
    }
    parents[2]["group_skipped"] = True
    with pytest.raises(ValueError, match="failed/skipped"):
        log_trainable_metrics(parents, [_datum([1, 1])], {}, datum_sources=[(2, 0)])
    with pytest.raises(ValueError, match="identity"):
        log_trainable_metrics(parents, datums, {}, datum_sources=[])
    with pytest.raises(ValueError, match="inconsistent"):
        log_trainable_metrics(
            parents, [_datum([1, 0]), _datum([1, 1])], {}, datum_sources=[(0, 0), (0, 0)]
        )


@pytest.mark.parametrize("second_mask", [[0, 1], [1, 0, 0]])
def test_duplicate_sources_require_same_effective_positions_not_just_count(second_mask):
    with pytest.raises(ValueError, match="inconsistent effective masks"):
        log_trainable_metrics(
            [_parent("a", [2])],
            [_datum([1, 0]), _datum(second_mask)],
            {},
            datum_sources=[(0, 0), (0, 0)],
        )


def test_trainable_counts_are_refreshed_after_mutable_masks_change():
    parents = [_parent("a", [2])]
    datums = [_datum([1, 0]), _datum([2, -1])]
    sources = [(0, 0), (0, 0)]
    metrics = {}
    assert log_trainable_metrics(parents, datums, metrics, datum_sources=sources) == [1, 1]
    assert metrics["batch/trainable_response_tokens_total"] == 1

    for datum in datums:
        datum["loss_fn_inputs"]["loss_mask"][:] = [1, 1]
    assert log_trainable_metrics(parents, datums, metrics, datum_sources=sources) == [2, 2]
    assert metrics["batch/trainable_response_tokens_total"] == 2


def test_managed_datum_counts_do_not_require_server_private_masks():
    metrics = {}
    counts = log_trainable_metrics(
        [_parent("a", [2]), _parent("b", [2])],
        [{"kind": "sample_ref"}, _datum([1, 0])],
        metrics,
        datum_sources=[(0, 0), (1, 0)],
    )
    assert counts == [0, 1]
    assert metrics["batch/trainable_segment_count"] == 2
    assert metrics["batch/trainable_response_tokens_total"] == 1


def test_sampling_logprob_token_weighted_but_ppl_segment_weighted():
    parent = _parent("a", [1, 3])
    parent.segments[0]["old_log_probs"] = [0, -1]
    parent.segments[1]["old_log_probs"] = [0, -2, -2, -2]
    metrics = {}
    log_logprob_metrics([parent], metrics)
    assert metrics["sampling/logprob_mean"] == -1.75
    assert metrics["sampling/ppl_mean"] == pytest.approx((math.e + math.exp(2)) / 2)


def test_actual_rl_converter_mapping_survives_skips_and_mis_masks():
    from northrl.utils.finetune_service_utils import convert_trajectories_to_datums

    parents = [_parent("a", [2, 2]), _parent("b", [2]), _parent("c", [2])]
    for parent in parents:
        parent["advantage"] = 1
        for segment in parent.segments:
            segment["old_log_probs"] = [0, -1, -1]
    parents[1]["group_skipped"] = True
    parents[2]["advantage"] = 0
    parents[0].segments[0]["correction_weights"] = [0, 1, 1]
    parents[0].segments[0]["correction_mask"] = [0, 1, 0]
    parents[0].segments[1]["correction_weights"] = [0, 1, 1]
    parents[0].segments[1]["correction_mask"] = [0, 0, 0]
    sources = []
    datums = convert_trajectories_to_datums(parents, datum_sources=sources)
    assert sources == [(0, 0)]
    metrics = {}
    log_trainable_metrics(parents, datums, metrics, datum_sources=sources)
    assert metrics["batch/trainable_trajectory_count"] == 1
    assert metrics["batch/trainable_segment_count"] == 1
    assert metrics["batch/trainable_response_tokens_total"] == 1
    assert metrics["batch/trainable_group_count"] == 1


def test_sft_identical_token_rows_are_distinct_sources_and_zero_mask_is_not_trainable():
    from northrl.utils.finetune_service_utils import convert_trajectories_to_datums_cross_entropy

    parents = [
        Trajectory(tokens=[1, 2, 3], loss_mask=mask) for mask in ([0, 1, 1], [0, 1, 1], [0, 0, 0])
    ]
    sources = []
    datums = convert_trajectories_to_datums_cross_entropy(parents, datum_sources=sources)
    assert sources == [(0, 0), (1, 0), (2, 0)]
    metrics = {"batch/trainable_group_count": 0}
    log_trainable_metrics(parents, datums, metrics, datum_sources=sources)
    assert metrics == {
        "batch/trainable_trajectory_count": 2,
        "batch/trainable_segment_count": 2,
        "batch/trainable_response_tokens_total": 4,
    }


def test_task_score_boolean_is_valid_and_nonfinite_registered_score_has_zero_observations():
    parent = _parent(
        "a",
        [1],
        score={"correct": True, "custom": float("nan"), "tool_calls_emitted": 50},
        metric_score_names=["custom"],
    )
    metrics = {}
    log_rollout_metrics([parent], metrics)
    assert metrics["outcome/score_correct_mean"] == 1
    assert metrics["outcome/score_correct_observed_count"] == 1
    assert metrics["outcome/score_custom_observed_count"] == 0
    assert "outcome/score_custom_mean" not in metrics
    assert "outcome/score_tool_calls_emitted_mean" not in metrics


def test_preparation_updates_skipped_groups_without_rewriting_input_counts():
    parents = [_parent("a", [2]), _parent("a", [3]), _parent("b", [4])]
    metrics = {}
    log_rollout_metrics(parents, metrics)
    assert metrics["batch/skipped_group_count"] == 0
    parents[0]["group_skipped"] = parents[1]["group_skipped"] = True
    log_grpo_metrics(parents, metrics)
    assert metrics["batch/skipped_group_count"] == 1
    assert metrics["batch/trajectory_count"] == 3
    assert metrics["batch/segment_count"] == 3
    assert metrics["group/valid_group_count"] == 1


def test_step_verifier_failures_are_parent_owned_and_preserve_runtime_metrics():
    parents = [
        _parent(
            "code",
            [1, 2, 3],
            {"sandbox_retries": 7, "duration_seconds": 10, "verifier_duration_seconds": 10},
            reward_fn="code_e2b",
            score={
                "reason": "timeout_after_retry",
                "metrics": {"timeout_count": 4, "retry_count": 2},
            },
        ),
        _parent(
            "code",
            [1],
            {"sandbox_retries": 0, "duration_seconds": 2, "verifier_duration_seconds": 2},
            reward_fn="code_e2b",
            score={
                "correct": False,
                "reason": "wrong_answer",
                "metrics": {"timeout_count": 1, "retry_count": 1},
            },
        ),
        _parent(
            "stem",
            [1],
            reward_fn="reasoning360",
            score={"reason": "reasoning360/stem/judge_error"},
        ),
        _parent(
            "length",
            [1],
            {"verifier_skipped_length": True},
            reward_fn="ifeval",
            finish_reason="length",
            score={"reason": "finish_reason_length", "correct": False},
        ),
    ]
    parents[0].segments[0]["score"] = {"reason": "judge_error"}
    metrics = {}
    log_rollout_metrics(parents, metrics)

    assert metrics["batch/trajectory_count"] == 4
    assert metrics["batch/segment_count"] == 6
    assert metrics["verifier/result_count"] == 4
    assert metrics["verifier/service_error_count"] == 2
    assert metrics["verifier/judge_error_count"] == 1
    assert metrics["verifier/timeout_after_retry_count"] == 1
    assert metrics["verifier/timeout_count"] == 5
    assert metrics["verifier/retry_count"] == 3
    assert metrics["verifier/retried_trajectory_count"] == 2
    assert metrics["verifier/retry_completed_count"] == 1
    assert metrics["reward_fn/code_e2b/verifier/service_error_count"] == 1
    assert metrics["reward_fn/reasoning360/verifier/judge_error_count"] == 1
    assert metrics["reward_fn/ifeval/verifier/service_error_count"] == 0
    assert metrics["verifier/duration_seconds_observed_count"] == 2
    assert metrics["verifier/duration_seconds_mean"] == 6
    assert metrics["reward_fn/code_e2b/verifier/duration_seconds_observed_count"] == 2
    assert metrics["reward_fn/code_e2b/verifier/duration_seconds_mean"] == 6
    assert metrics["verifier/skipped_length_count"] == 1
    assert metrics["trajectory/duration_seconds_mean"] == 6
    assert metrics["batch/sandbox_retry_count"] == 7

    next_metrics = {}
    log_rollout_metrics([_parent("next", [1], score={"correct": True})], next_metrics)
    assert next_metrics["verifier/result_count"] == 1
    assert next_metrics["verifier/service_error_count"] == 0
    assert next_metrics["verifier/retry_count"] == 0
    assert not any(key.startswith("reward_fn/") for key in next_metrics)
