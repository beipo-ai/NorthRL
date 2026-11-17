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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from northrl.activity_tracker import ActivityTracker, ActivityTrackerProxy
from northrl.northrl_types import Trajectory
from northrl.validator import Validator


def _trajectory(reward: float) -> Trajectory:
    return Trajectory(
        tokens=[1],
        loss_mask=[0],
        reward=reward,
        is_val=True,
        extra_fields={"score": {"reward": reward}, "data_source": "test"},
    )


def test_validator_keeps_trajectory_arriving_after_metrics_snapshot():
    validator = Validator(
        OmegaConf.create(
            {
                "validate": {
                    "aggregation": "mean",
                    "data": {"validation_repeat_n": 1},
                }
            }
        )
    )
    current = _trajectory(1.0)
    late = _trajectory(0.0)
    validator.put_trajectory(current)
    validator.set_activity_tracker(
        SimpleNamespace(
            experiment_logger_post=lambda **_: validator.put_trajectory(late),
        )
    )

    metrics = validator.compute_and_log_metrics()

    assert metrics["val/segment_count"] == metrics["val/trajectory_count"] == 1
    assert validator._trajectories == [late]  # pylint: disable=protected-access


def test_validator_counts_complete_trajectories_independently_of_segments():
    validator = Validator(OmegaConf.create({"validate": {"aggregation": "mean"}}))
    validator.set_activity_tracker(SimpleNamespace(experiment_logger_post=lambda **_: None))
    for reward, count in [(1.0, 3), (0.0, 1)]:
        validator.put_trajectory(
            Trajectory(
                **(
                    {"tokens": [1, 2], "loss_mask": [0, 1]}
                    if count == 1
                    else {
                        "segments": [{"tokens": [1, 2], "loss_mask": [0, 1]} for _ in range(count)]
                    }
                ),
                reward=reward,
                is_val=True,
                extra_fields={"score": {"correct": reward}},
            )
        )
    metrics = validator.compute_and_log_metrics()
    assert metrics["val/correct_mean"] == 0.5
    assert metrics["val/trajectory_count"] == 2
    assert metrics["val/segment_count"] == 4
    assert "val/sequence_count" not in metrics
    assert "val/num_samples" not in metrics


def test_validation_length_counts_use_parent_and_dataset_denominators(caplog):
    validator = Validator(OmegaConf.create({"validate": {"aggregation": "mean"}}))
    post_metrics = MagicMock()
    validator.set_activity_tracker(SimpleNamespace(experiment_logger_post=post_metrics))
    records = [
        {"dataset_name": "code", "data_source": "ignored", "finish_reason": "length"},
        {"dataset_name": "code", "finish_reason": "stop"},
        {"dataset_name": "code"},
        {"data_source": "ifeval", "finish_reason": "length"},
        {"finish_reason": None},
    ]
    for index, fields in enumerate(records):
        validator.put_trajectory(
            Trajectory(
                segments=[
                    {"tokens": [1, 2], "loss_mask": [0, 1], "finish_reason": "length"}
                    for _ in range(3 if index == 0 else 1)
                ],
                reward=None,
                is_val=True,
                extra_fields=fields,
            )
        )
    with caplog.at_level("INFO", logger="northrl.validator"):
        metrics = validator.compute_and_log_metrics()

    assert metrics["val/trajectory_count"] == 5
    assert metrics["val/segment_count"] == 7
    assert metrics["val/finish_reason_length_count"] == 2
    assert metrics["val/finish_reason_length_fraction"] == 0.4
    assert metrics["val/code/trajectory_count"] == 3
    assert metrics["val/code/finish_reason_length_count"] == 1
    assert metrics["val/code/finish_reason_length_fraction"] == pytest.approx(1 / 3)
    assert metrics["val/ifeval/finish_reason_length_count"] == 1
    assert metrics["val/ifeval/finish_reason_length_fraction"] == 1
    assert metrics["val/all/finish_reason_length_count"] == 0
    assert metrics["val/all/finish_reason_length_fraction"] == 0
    assert not any("ignored" in key for key in metrics)
    assert "validation: length=2/5 (40.00%)" in caplog.text
    assert "validation dataset=code: length=1/3 (33.33%)" in caplog.text
    post_metrics.assert_called_once_with(backend="wandb", data=metrics, commit=True)

    validator.put_trajectory(_trajectory(1.0))
    next_metrics = validator.compute_and_log_metrics()
    assert next_metrics["val/finish_reason_length_count"] == 0
    assert next_metrics["val/finish_reason_length_fraction"] == 0
    assert "val/code/finish_reason_length_count" not in next_metrics


def test_validation_timing_uses_parent_observations_with_val_prefix_and_independent_snapshots():
    validator = Validator(OmegaConf.create({"validate": {"aggregation": "mean"}}))
    post_metrics = MagicMock()
    validator.set_activity_tracker(SimpleNamespace(experiment_logger_post=post_metrics))
    records = [
        (
            "code",
            "code_e2b",
            1,
            2,
            {
                "started_at": 100,
                "finished_at": 110,
                "duration_seconds": 10,
                "verifier_duration_seconds": 2,
            },
        ),
        (
            "code",
            "code_e2b",
            0,
            2,
            {
                "started_at": 105,
                "finished_at": 120,
                "duration_seconds": 15,
                "verifier_duration_seconds": 4,
            },
        ),
        (
            "length",
            "ifeval",
            0,
            1,
            {
                "started_at": 200,
                "finished_at": 205,
                "duration_seconds": 5,
                "verifier_skipped_length": True,
            },
        ),
    ]
    for index, (group, reward_fn, reward, expected_rollouts, runtime) in enumerate(records):
        validator.put_trajectory(
            Trajectory(
                segments=[
                    {"tokens": [1, 2], "loss_mask": [0, 1]} for _ in range(2 if index == 0 else 1)
                ],
                reward=reward,
                is_val=True,
                extra_fields={
                    "group_id": group,
                    "expected_rollouts": expected_rollouts,
                    "reward_fn": reward_fn,
                    "dataset_name": "dataset",
                    "finish_reason": "length" if group == "length" else "stop",
                    "runtime_metrics": runtime,
                    "score": {
                        "correct": bool(reward),
                        "duration_seconds": 999,
                        "verifier_duration_seconds": 888,
                    },
                },
            )
        )
    validator.put_trajectory(
        Trajectory(segments=[], reward=None, is_val=True, extra_fields={"dataset_name": "dataset"})
    )

    metrics = validator.compute_and_log_metrics()

    assert metrics["val/trajectory_count"] == 4
    assert metrics["val/segment_count"] == 4
    assert metrics["val/group/duration_seconds_observed_count"] == 2
    assert metrics["val/group/duration_seconds_mean"] == 12.5
    assert metrics["val/trajectory/duration_seconds_observed_count"] == 3
    assert metrics["val/trajectory/duration_seconds_mean"] == 10
    assert metrics["val/verifier/duration_seconds_observed_count"] == 2
    assert metrics["val/verifier/duration_seconds_mean"] == 3
    assert metrics["val/verifier/skipped_length_count"] == 1
    assert metrics["val/reward_fn/code_e2b/group/duration_seconds_mean"] == 20
    assert metrics["val/reward_fn/code_e2b/verifier/duration_seconds_mean"] == 3
    assert metrics["val/reward_fn/ifeval/verifier/duration_seconds_observed_count"] == 0
    assert metrics["val/reward_fn/ifeval/verifier/skipped_length_count"] == 1
    assert "val/reward_fn/ifeval/verifier/duration_seconds_mean" not in metrics
    assert metrics["val/correct_mean"] == pytest.approx(1 / 3)
    assert metrics["val/reward_mean"] == pytest.approx(1 / 3)
    assert "val/duration_seconds_mean" not in metrics
    assert "val/verifier_duration_seconds_mean" not in metrics
    assert "val/dataset/duration_seconds_mean" not in metrics
    assert "val/dataset/verifier_duration_seconds_mean" not in metrics
    assert all(key.startswith("val/") for key in metrics)
    post_metrics.assert_called_once_with(backend="wandb", data=metrics, commit=True)

    validator.put_trajectory(_trajectory(1))
    next_metrics = validator.compute_and_log_metrics()
    assert next_metrics["val/group/duration_seconds_observed_count"] == 0
    assert next_metrics["val/trajectory/duration_seconds_observed_count"] == 0
    assert next_metrics["val/verifier/duration_seconds_observed_count"] == 0
    assert next_metrics["val/verifier/skipped_length_count"] == 0
    assert "val/trajectory/duration_seconds_mean" not in next_metrics
    assert "val/verifier/duration_seconds_mean" not in next_metrics
    assert not any(key.startswith("val/reward_fn/") for key in next_metrics)


@pytest.mark.parametrize("step", [0, 7])
def test_validator_commits_metrics_at_current_training_step(step):
    tracker = object.__new__(ActivityTracker)
    tracker._current_training_step = step  # pylint: disable=protected-access
    tracker.experiment_logger = MagicMock()
    proxy = ActivityTrackerProxy(tracker)
    validator = Validator(OmegaConf.create({"validate": {"aggregation": "mean"}}))
    validator.set_activity_tracker(proxy)
    validator.put_trajectory(_trajectory(1.0))

    metrics = validator.compute_and_log_metrics()

    tracker.experiment_logger.log.assert_called_once_with(data=metrics, step=step, commit=True)


def test_activity_tracker_preserves_partial_posts_and_forwards_completed_round_flush():
    tracker = object.__new__(ActivityTracker)
    tracker.experiment_logger = MagicMock()
    proxy = ActivityTrackerProxy(tracker)

    proxy.experiment_logger_post(backend="wandb", data={"training/loss": 0.5}, step=3)
    tracker.experiment_logger.log.assert_called_once_with(data={"training/loss": 0.5}, step=3)
    tracker.experiment_logger.flush.assert_not_called()

    proxy.experiment_logger_flush(3)
    tracker.experiment_logger.flush.assert_called_once_with(3)


def test_validation_uses_finite_task_scores_and_shared_dataset_priority():
    validator = Validator(OmegaConf.create({"validate": {"aggregation": "mean"}}))
    validator.set_activity_tracker(SimpleNamespace(experiment_logger_post=lambda **_: None))
    for fields in [
        {
            "dataset_name": "preferred",
            "data_source": "ignored",
            "score": {"correct": 1, "custom": 4, "model_turns": 99},
            "metric_score_names": ["custom"],
        },
        {
            "dataset_name": "preferred",
            "score": {"correct": float("nan"), "custom": float("inf")},
            "metric_score_names": ["custom"],
        },
        {"data_source": "fallback", "score": {"correct": False}},
        {"score": {"correct": 1}},
    ]:
        validator.put_trajectory(
            Trajectory(tokens=[], loss_mask=[], reward=None, extra_fields=fields)
        )
    metrics = validator.compute_and_log_metrics()
    assert metrics["val/correct_mean"] == pytest.approx(2 / 3)
    assert metrics["val/correct_observed_count"] == 3
    assert metrics["val/preferred/correct_mean"] == 1
    assert metrics["val/preferred/correct_observed_count"] == 1
    assert metrics["val/custom_mean"] == 4
    assert metrics["val/custom_observed_count"] == 1
    assert metrics["val/fallback/correct_mean"] == 0
    assert metrics["val/all/correct_mean"] == 1
    assert metrics["val/segment_count"] == 0
    assert not any("model_turns" in key or "ignored" in key for key in metrics)


def test_validation_pass_at_k_keeps_deduplication_missing_slots_and_omits_avg_alias():
    validator = Validator(
        OmegaConf.create(
            {"validate": {"aggregation": "pass_at_k", "data": {"validation_repeat_n": 2}}}
        )
    )
    validator.set_activity_tracker(SimpleNamespace(experiment_logger_post=lambda **_: None))
    for group_id, run_id, correct in [("a", 0, True), ("a", 0, False), ("b", 0, False)]:
        validator.put_trajectory(
            Trajectory(
                segments=[],
                reward=None,
                extra_fields={
                    "dataset_name": "data",
                    "data_source": "old",
                    "group_id": group_id,
                    "run_id": run_id,
                    "score": {"correct": correct},
                },
            )
        )
    metrics = validator.compute_and_log_metrics()
    assert metrics["val/data/pass@1"] == pytest.approx(0.25)
    assert metrics["val/data/pass@2"] == pytest.approx(0.5)
    assert metrics["val/duplicate_run_count"] == 1
    assert metrics["val/data/complete_group_ratio"] == 0
    assert metrics["val/data/correct_observed_count"] == 3
    assert not any("avg@" in key or "val/old/" in key for key in metrics)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), None, "invalid"])
def test_validation_nonfinite_correct_is_not_a_pass(invalid):
    validator = Validator(
        OmegaConf.create(
            {"validate": {"aggregation": "pass_at_k", "data": {"validation_repeat_n": 1}}}
        )
    )
    validator.set_activity_tracker(SimpleNamespace(experiment_logger_post=lambda **_: None))
    validator.put_trajectory(
        Trajectory(segments=[], reward=None, extra_fields={"score": {"correct": invalid}})
    )
    metrics = validator.compute_and_log_metrics()
    assert metrics["val/correct_observed_count"] == 0
    assert "val/correct_mean" not in metrics
    assert metrics["val/all/pass@1"] == 0


def test_validation_publishes_verifier_failure_counters_without_changing_reward_metrics():
    validator = Validator(OmegaConf.create({"validate": {"aggregation": "mean"}}))
    post_metrics = MagicMock()
    validator.set_activity_tracker(SimpleNamespace(experiment_logger_post=post_metrics))
    records = [
        (0, "reasoning360", {"reason": "reasoning360/stem/judge_error", "correct": False}),
        (
            1,
            "code_e2b",
            {
                "reason": "ok",
                "correct": True,
                "metrics": {"timeout_count": 3, "retry_count": 2},
            },
        ),
        (0, "code_e2b", {"reason": "no_code", "correct": False}),
    ]
    for reward, reward_fn, score in records:
        validator.put_trajectory(
            Trajectory(
                segments=[
                    {
                        "tokens": [1, 2],
                        "loss_mask": [0, 1],
                        "score": {"reason": "timeout_after_retry"},
                    }
                    for _ in range(2)
                ],
                reward=reward,
                is_val=True,
                extra_fields={
                    "score": score,
                    "reward_fn": reward_fn,
                    "runtime_metrics": {"verifier_duration_seconds": 4},
                },
            )
        )

    metrics = validator.compute_and_log_metrics()

    assert metrics["val/trajectory_count"] == 3
    assert metrics["val/segment_count"] == 6
    assert metrics["val/verifier/result_count"] == 3
    assert metrics["val/verifier/service_error_count"] == 1
    assert metrics["val/verifier/judge_error_count"] == 1
    assert metrics["val/verifier/timeout_after_retry_count"] == 0
    assert metrics["val/verifier/no_code_count"] == 1
    assert metrics["val/verifier/timeout_count"] == 3
    assert metrics["val/verifier/retry_count"] == 2
    assert metrics["val/verifier/retried_trajectory_count"] == 1
    assert metrics["val/verifier/retry_completed_count"] == 1
    assert metrics["val/reward_fn/code_e2b/verifier/result_count"] == 2
    assert metrics["val/reward_fn/code_e2b/verifier/no_code_count"] == 1
    assert metrics["val/reward_fn/reasoning360/verifier/service_error_count"] == 1
    assert metrics["val/verifier/duration_seconds_observed_count"] == 3
    assert metrics["val/verifier/duration_seconds_mean"] == 4
    assert metrics["val/reward_fn/code_e2b/verifier/duration_seconds_observed_count"] == 2
    assert metrics["val/reward_fn/code_e2b/verifier/duration_seconds_mean"] == 4
    assert metrics["val/reward_fn/reasoning360/verifier/duration_seconds_observed_count"] == 1
    assert metrics["val/reward_fn/reasoning360/verifier/duration_seconds_mean"] == 4
    assert metrics["val/reward_mean"] == pytest.approx(1 / 3)
    assert metrics["val/correct_mean"] == pytest.approx(1 / 3)
    post_metrics.assert_called_once_with(backend="wandb", data=metrics, commit=True)

    validator.put_trajectory(_trajectory(1))
    next_metrics = validator.compute_and_log_metrics()
    assert next_metrics["val/verifier/result_count"] == 1
    assert next_metrics["val/verifier/service_error_count"] == 0
    assert next_metrics["val/verifier/retry_count"] == 0
    assert not any(key.startswith("val/reward_fn/") for key in next_metrics)
