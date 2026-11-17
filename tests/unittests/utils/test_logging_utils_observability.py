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

"""Tests for observability metrics in logging_utils."""

import pytest
import torch

from northrl.northrl_types import Trajectory
from northrl.utils.logging_utils import (
    log_advantage_distribution_metrics,
    log_grpo_metrics,
    log_logprob_metrics,
    log_rollout_metrics,
)


def _make_traj(reward=0.0, advantage=None, old_log_probs=None, loss_mask=None, tokens=None):
    """Helper to build a minimal Trajectory for testing."""
    if tokens is None:
        tokens = [1, 2, 3, 4, 5]
    if loss_mask is None:
        loss_mask = [0, 0, 1, 1, 1]
    traj = Trajectory(tokens=tokens, loss_mask=loss_mask, reward=reward, is_val=False)
    if advantage is not None:
        traj["advantage"] = advantage
    if old_log_probs is not None:
        traj["old_log_probs"] = old_log_probs
    return traj


# ---------- log_rollout_metrics ----------


class TestLogRolloutMetrics:
    def test_counts_e2b_timeout_and_truncation_by_rollout(self):
        trajectories = []
        for rollout_id, timed_out, segment_flags in [
            ("rollout-a", True, [False, False]),
            ("rollout-b", False, [True]),
            ("rollout-c", False, [False, True]),
        ]:
            segments = [{"tokens": [1, 2], "loss_mask": [0, 1]} for _ in segment_flags]
            for segment, is_truncated in zip(segments, segment_flags, strict=True):
                segment["is_truncated"] = is_truncated
            trajectories.append(
                Trajectory(
                    segments=segments,
                    reward=0.0,
                    extra_fields={
                        "e2b_rollout_id": rollout_id,
                        "e2b_timed_out": timed_out,
                        "score": {"e2b_timed_out": float(timed_out)},
                    },
                )
            )
        metrics = {}
        log_rollout_metrics(trajectories, metrics)
        assert metrics["batch/segment_count"] == 5
        assert metrics["batch/trajectory_count"] == 3
        assert metrics["timeout/rollout_count"] == 1
        assert metrics["outcome/truncated_trajectory_count"] == 2
        assert metrics["segment/truncated_count"] == 2

    def test_e2b_event_counts_are_logged_as_zero(self):
        trajectory = _make_traj(reward=1.0)
        trajectory["e2b_rollout_id"] = "rollout-a"
        trajectory["e2b_timed_out"] = False
        trajectory["is_truncated"] = False

        metrics: dict = {}
        log_rollout_metrics([trajectory], metrics)

        assert metrics["timeout/rollout_count"] == 0
        assert metrics["outcome/truncated_trajectory_count"] == 0
        assert metrics["segment/truncated_count"] == 0


# ---------- log_advantage_distribution_metrics ----------


class TestLogAdvantageDistributionMetrics:
    def test_basic_stats(self):
        trajs = [
            _make_traj(advantage=1.0),
            _make_traj(advantage=-0.5),
            _make_traj(advantage=0.0),
            _make_traj(advantage=2.0),
        ]
        metrics: dict = {}
        log_advantage_distribution_metrics(trajs, metrics)

        assert metrics["trajectory/advantage_mean"] == pytest.approx(0.625)
        assert metrics["trajectory/advantage_min"] == pytest.approx(-0.5)
        assert metrics["trajectory/advantage_max"] == pytest.approx(2.0)
        assert metrics["trajectory/advantage_positive_count"] == 2
        assert metrics["trajectory/advantage_negative_count"] == 1
        assert metrics["trajectory/advantage_zero_count"] == 1
        assert metrics["trajectory/advantage_zero_percentage"] == pytest.approx(25.0)
        assert metrics["trajectory/advantage_abs_mean"] == pytest.approx(0.875)

    def test_all_zero_advantages_collapse_scenario(self):
        """When all advantages are 0, the zero percentage should be 100 — signals mode collapse."""
        trajs = [_make_traj(advantage=0.0) for _ in range(8)]
        metrics: dict = {}
        log_advantage_distribution_metrics(trajs, metrics)

        assert metrics["trajectory/advantage_zero_count"] == 8
        assert metrics["trajectory/advantage_zero_percentage"] == pytest.approx(100.0)
        assert metrics["trajectory/advantage_std"] == pytest.approx(0.0)

    def test_empty_trajectories(self):
        metrics: dict = {}
        log_advantage_distribution_metrics([], metrics)
        assert "trajectory/advantage_mean" not in metrics

    def test_missing_advantage_field_is_not_reported_as_zero(self):
        trajs = [Trajectory(tokens=[1, 2], loss_mask=[0, 1], reward=1.0, is_val=False)]
        metrics: dict = {}
        log_advantage_distribution_metrics(trajs, metrics)
        assert "trajectory/advantage_mean" not in metrics
        assert "trajectory/advantage_zero_count" not in metrics


# ---------- log_logprob_metrics ----------


class TestLogLogprobMetrics:
    def test_basic_stats(self):
        trajs = [
            _make_traj(
                old_log_probs=[-1.0, -2.0, -0.5, -1.5, -3.0],
                loss_mask=[0, 0, 1, 1, 1],
            ),
        ]
        metrics: dict = {}
        log_logprob_metrics(trajs, metrics)

        # response-only logprobs: [-0.5, -1.5, -3.0]
        assert metrics["sampling/logprob_mean"] == pytest.approx((-0.5 - 1.5 - 3.0) / 3)
        assert metrics["sampling/logprob_min"] == pytest.approx(-3.0)
        assert metrics["sampling/logprob_max"] == pytest.approx(-0.5)

    def test_no_old_log_probs(self):
        trajs = [_make_traj()]  # no old_log_probs field
        metrics: dict = {}
        log_logprob_metrics(trajs, metrics)
        assert "sampling/logprob_mean" not in metrics

    def test_empty_trajectories(self):
        metrics: dict = {}
        log_logprob_metrics([], metrics)
        assert "sampling/logprob_mean" not in metrics

    def test_multiple_trajectories(self):
        trajs = [
            _make_traj(
                old_log_probs=[-1.0, -2.0, -1.0],
                loss_mask=[0, 1, 1],
                tokens=[1, 2, 3],
            ),
            _make_traj(
                old_log_probs=[-3.0, -4.0, -2.0],
                loss_mask=[0, 1, 1],
                tokens=[4, 5, 6],
            ),
        ]
        metrics: dict = {}
        log_logprob_metrics(trajs, metrics)

        # response old_log_probs: [-2.0, -1.0] + [-4.0, -2.0]
        assert metrics["sampling/logprob_mean"] == pytest.approx((-2.0 - 1.0 - 4.0 - 2.0) / 4)
        assert metrics["sampling/logprob_min"] == pytest.approx(-4.0)


def _identified_trajectories():
    magnitude = 0.5 / (0.5**0.5 + 1e-6)
    return [
        Trajectory(
            **(
                {"tokens": [1, 2], "loss_mask": [0, 1]}
                if count == 1
                else {"segments": [{"tokens": [1, 2], "loss_mask": [0, 1]} for _ in range(count)]}
            ),
            reward=reward,
            extra_fields={
                "group_id": "group",
                "run_id": run_id,
                "rollout_id": rollout_id,
                "expected_rollouts": 2,
                "model_tag": "actor",
                "score": {"correct": reward},
                "metric_score_names": ["correct"],
                "advantage": magnitude if reward else -magnitude,
            },
        )
        for rollout_id, run_id, reward, count in [("a", 0, 1.0, 3), ("b", 1, 0.0, 1)]
    ]


def test_rewards_scores_and_advantages_count_complete_trajectories_once():
    trajectories = _identified_trajectories()
    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    log_grpo_metrics(trajectories, metrics)
    log_advantage_distribution_metrics(trajectories, metrics)
    assert metrics["batch/segment_count"] == 4
    assert "rollout/sequence_count" not in metrics
    assert trajectories[1].segments is None
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["trajectory/reward_mean"] == 0.5
    assert metrics["outcome/score_correct_mean"] == 0.5
    assert metrics["group/reward_std_min"] == pytest.approx(0.5**0.5)
    assert metrics["trajectory/advantage_positive_count"] == 1
    assert metrics["trajectory/advantage_negative_count"] == 1
    assert metrics["trajectory/advantage_mean"] == pytest.approx(0.0)
    assert metrics["trajectory/advantage_token_weighted_mean"] == pytest.approx(
        0.25 / (0.5**0.5 + 1e-6)
    )


def test_zero_segment_trajectory_still_contributes_final_reward_and_score():
    trajectories = _identified_trajectories()
    trajectories[0]["score"] = {"correct": 0.8}
    trajectories[1].segments = []
    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    assert metrics["batch/segment_count"] == 3
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["trajectory/reward_mean"] == 0.5
    assert metrics["outcome/score_correct_mean"] == 0.4


def test_all_zero_segment_trajectories_still_log_known_rewards():
    trajectories = _identified_trajectories()
    for trajectory in trajectories:
        trajectory.segments = []
    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    assert metrics["batch/segment_count"] == 0
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["trajectory/reward_mean"] == 0.5


@pytest.mark.parametrize("invalid_reward", [None, float("nan"), float("inf"), True, "1"])
def test_discarded_groups_exclude_invalid_outcome_rewards(invalid_reward):
    trajectories = _identified_trajectories()
    for trajectory in trajectories:
        trajectory["group_skipped"] = True
    trajectories[1].reward = invalid_reward
    trajectories[1].status = "failed"
    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["trajectory/reward_observed_count"] == 1
    assert metrics["trajectory/reward_mean"] == 1.0
    assert metrics["batch/failed_trajectory_count"] == 1
    assert metrics["batch/skipped_group_count"] == 1


def test_successful_nonfinite_reward_is_rejected_by_metrics():
    trajectory = _identified_trajectories()[0]
    trajectory.reward = float("nan")
    with pytest.raises(ValueError, match="finite"):
        log_rollout_metrics([trajectory], {})


def test_pool_discarded_nonfinite_group_still_produces_safe_rollout_metrics():
    from omegaconf import OmegaConf

    from northrl.trajectory_pool import GroupedTrajectoryStore

    store = GroupedTrajectoryStore(OmegaConf.create({"group_size": 2}), model_tag="actor")
    trajectories = _identified_trajectories()
    trajectories[1].reward = float("nan")
    for trajectory in trajectories:
        store.put_trajectory(trajectory)
    completed = store.take_groups(num_groups=1)
    assert len(completed) == 2
    assert all(trajectory.get("group_skipped") for trajectory in completed)
    metrics = {}
    log_rollout_metrics(completed, metrics)
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["batch/skipped_group_count"] == 1
    assert metrics["trajectory/reward_observed_count"] == 1
    assert metrics["trajectory/reward_mean"] == 1.0


def test_grpo_metrics_count_single_trajectory_group_with_zero_std():
    trajectory = _identified_trajectories()[0]
    trajectory["advantage"] = 0.0
    metrics = {}
    log_grpo_metrics([trajectory], metrics)
    assert metrics["group/valid_group_count"] == 1
    assert metrics["group/reward_std_min"] == 0.0


def test_logprob_metrics_read_mixed_parents_without_counting_skipped_groups():
    ordinary = _make_traj(tokens=[1, 2], loss_mask=[0, 1], old_log_probs=[0.0, -1.0])
    multiple = Trajectory(
        segments=[
            {"tokens": [3, 4], "loss_mask": [0, 1], "old_log_probs": [0.0, -2.0]},
            {"tokens": [5, 6], "loss_mask": [0, 1], "old_log_probs": [0.0, -3.0]},
        ],
        reward=1.0,
    )
    skipped = _make_traj(tokens=[7, 8], loss_mask=[0, 1], old_log_probs=[0.0, -100.0])
    skipped["group_skipped"] = True
    metrics = {}
    log_logprob_metrics([ordinary, multiple, skipped], metrics)
    assert metrics["sampling/logprob_mean"] == -2.0
    assert ordinary.segments is None
    assert "group_id" not in multiple.segments[0]


def test_swe_outcomes_and_tokens_use_complete_trajectories_including_zero_segments():
    trajectories = [
        Trajectory(
            segments=[
                {"tokens": [1, 2], "loss_mask": [0, 1], "is_truncated": index == 2}
                for index in range(3)
            ],
            reward=1.0,
            extra_fields={
                "e2b_rollout_id": "solved",
                "e2b_status": "success",
                "e2b_termination_reason": "context_length_exceeded",
                "score": {"resolved": 1.0, "e2b_task_failed": 0.0, "e2b_env_error": 0.0},
            },
        ),
        Trajectory(
            tokens=[3, 4],
            loss_mask=[0, 1],
            reward=0.0,
            extra_fields={
                "e2b_rollout_id": "task-failed",
                "e2b_status": "task_failed",
                "score": {"resolved": 0.0, "e2b_task_failed": 1.0, "e2b_env_error": 0.0},
            },
        ),
        Trajectory(
            segments=[],
            reward=0.0,
            extra_fields={
                "e2b_rollout_id": "zero-segments",
                "e2b_status": "success",
                "e2b_termination_reason": "context_length_exceeded",
                "score": {"resolved": 0.0, "e2b_task_failed": 0.0, "e2b_env_error": 0.0},
            },
        ),
        Trajectory(
            segments=[],
            reward=None,
            status="failed",
            extra_fields={
                "e2b_rollout_id": "infra-failed",
                "e2b_status": "env_error",
                "e2b_timed_out": True,
                "score": {"e2b_task_failed": 0.0, "e2b_env_error": 1.0},
            },
        ),
    ]
    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    assert metrics["batch/trajectory_count"] == 4
    assert metrics["batch/segment_count"] == 4
    assert "rollout/sequence_count" not in metrics
    assert metrics["batch/response_tokens_total"] == 4
    assert metrics["trajectory/reward_observed_count"] == 3
    assert metrics["trajectory/reward_mean"] == pytest.approx(1 / 3)
    assert metrics["outcome/resolved_percentage"] == pytest.approx(100 / 3)
    assert metrics["batch/failed_trajectory_count"] == 1
    assert metrics["outcome/task_failed_count"] == 1
    assert metrics["outcome/env_error_count"] == 1
    assert "rollout/e2b_task_failed_ratio" not in metrics
    assert "rollout/e2b_env_error_ratio" not in metrics
    assert metrics["outcome/context_limit_reached_count"] == 2
    assert metrics["outcome/truncated_trajectory_count"] == 1
    assert metrics["segment/truncated_count"] == 1


def test_multi_segment_singleton_uses_main_grpo_zero_std():
    trajectory = _identified_trajectories()[0]
    assert len(trajectory.segments) == 3
    metrics = {}
    log_grpo_metrics([trajectory], metrics)
    assert metrics["group/valid_group_count"] == 1
    assert metrics["group/singleton_group_count"] == 1
    assert metrics["group/reward_std_min"] == 0.0


def test_swe_task_failure_is_a_grpo_outcome_and_skipped_group_has_no_advantages():
    trajectories = _identified_trajectories()
    trajectories[1]["e2b_status"] = "task_failed"
    trajectories[1].segments = []
    metrics = {}
    log_grpo_metrics(trajectories, metrics)
    log_advantage_distribution_metrics(trajectories, metrics)
    # A known zero reward participates even without a training sequence.
    assert metrics["group/reward_std_min"] == pytest.approx(0.5**0.5)
    assert metrics["group/singleton_group_count"] == 0
    assert metrics["trajectory/advantage_mean"] == pytest.approx(0.0)
    assert metrics["trajectory/advantage_token_weighted_mean"] == trajectories[0]["advantage"]

    for trajectory in trajectories:
        trajectory["group_skipped"] = True
    skipped_metrics = {}
    log_rollout_metrics(trajectories, skipped_metrics)
    log_grpo_metrics(trajectories, skipped_metrics)
    log_advantage_distribution_metrics(trajectories, skipped_metrics)
    assert skipped_metrics["trajectory/reward_observed_count"] == 2
    assert skipped_metrics["batch/skipped_group_count"] == 1
    assert skipped_metrics["group/singleton_group_count"] == 0
    assert "trajectory/advantage_mean" not in skipped_metrics


@pytest.mark.parametrize(
    ("identities", "expected_groups"),
    [
        pytest.param(
            [{"model_tag": "actor", "group_id": torch.tensor(7)} for _ in range(2)],
            1,
            id="equal-scalar-tensors",
        ),
        pytest.param(
            [{"model_tag": model, "group_id": "g"} for model in ("actor", "other")],
            2,
            id="model-scoped-groups",
        ),
        pytest.param([{}, {}], 2, id="missing-group-is-singleton"),
        pytest.param([{"group_id": None}, {"group_id": None}], 1, id="explicit-none-group"),
        pytest.param([{"group_id": ""}, {"group_id": ""}], 1, id="explicit-empty-group"),
    ],
)
def test_group_metrics_match_grpo_identity_rules(identities, expected_groups):
    from northrl.algorithm.advantage.grpo import compute_grpo_advantage_for_trajectories

    trajectories = [
        Trajectory(segments=[], reward=float(index), extra_fields=dict(identity))
        for index, identity in enumerate(identities)
    ]
    compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)
    metrics = {}
    log_grpo_metrics(trajectories, metrics)
    assert metrics["group/valid_group_count"] == expected_groups
    assert metrics["group/singleton_group_count"] == (2 if expected_groups == 2 else 0)
    assert metrics["group/reward_std_min"] == pytest.approx(
        0.0 if expected_groups == 2 else 0.5**0.5
    )
    assert all(float(trajectory["advantage"]) == 0 for trajectory in trajectories) == (
        expected_groups == 2
    )

    for trajectory in trajectories:
        trajectory["group_skipped"] = True
    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    if any("group_id" in trajectory for trajectory in trajectories):
        assert metrics["batch/skipped_group_count"] == expected_groups
    else:
        assert "batch/skipped_group_count" not in metrics
