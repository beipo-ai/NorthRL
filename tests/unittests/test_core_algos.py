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

"""GRPO operates on complete trajectory outcomes, independently of segmentation."""

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from northrl.algorithm.advantage.grpo import (
    compute_grpo_advantage_for_trajectories,
    compute_grpo_outcome_advantage,
)
from northrl.northrl_types import Trajectory
from northrl.utils.trajectory_sequences import iter_training_sequences


def _trajectory(reward, run_id=0, segment_count=1, **extra) -> Trajectory:
    fields = (
        {"tokens": [1, 2], "loss_mask": [0, 1]}
        if segment_count == 1
        else {"segments": [{"tokens": [1, 2], "loss_mask": [0, 1]} for _ in range(segment_count)]}
    )
    return Trajectory(
        **fields,
        reward=reward,
        extra_fields={"group_id": "group", "run_id": run_id, **extra},
    )


def _compute_advantages(rewards, *, use_run_ids=True):
    trajectories = [_trajectory(reward, run_id) for run_id, reward in enumerate(rewards)]
    compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=use_run_ids)
    return torch.stack([torch.as_tensor(trajectory["advantage"]) for trajectory in trajectories])


def _compute_tensor_advantages(
    rewards: Sequence[float], run_ids: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[Any, torch.Tensor]]:
    reward_tensor = torch.tensor(rewards).unsqueeze(-1)
    advantages, _, group_stds = compute_grpo_outcome_advantage(
        reward_tensor,
        torch.ones_like(reward_tensor),
        torch.zeros(len(rewards), dtype=torch.long),
        run_ids=run_ids,
    )
    return advantages.squeeze(-1), group_stds


@pytest.mark.parametrize(
    "run_ids",
    [None, torch.arange(16), torch.tensor([0] * 8 + [1] * 8), torch.zeros(16, dtype=torch.long)],
)
def test_tensor_grpo_identical_decimal_rewards_have_zero_advantage(run_ids):
    advantages, _ = _compute_tensor_advantages([0.1] * 16, run_ids)
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_identical_decimal_rewards_have_exactly_zero_advantage():
    advantages = _compute_advantages([0.1] * 16)
    assert torch.equal(advantages, torch.zeros_like(advantages))


@pytest.mark.parametrize("rewards", [[0] * 16, [1] * 16, [0, 1] * 8])
def test_integer_rewards_are_normalized_as_floats(rewards):
    advantages = _compute_advantages(rewards)
    assert torch.isfinite(advantages).all()
    if len(set(rewards)) == 1:
        assert torch.equal(advantages, torch.zeros_like(advantages))


def test_identical_decimal_rewards_without_run_ids_have_zero_advantage():
    advantages = _compute_advantages([0.1] * 512, use_run_ids=False)
    assert torch.equal(advantages, torch.zeros_like(advantages))


@pytest.mark.parametrize("use_run_ids", [False, True])
def test_std_scaling_keeps_the_legacy_population_or_sample_choice(use_run_ids):
    rewards = [0.1, 1.0, 0.1, 1.0]
    advantages = _compute_advantages(rewards, use_run_ids=use_run_ids)
    dtype = torch.float32 if use_run_ids else torch.float64
    rewards_tensor = torch.tensor(rewards, dtype=dtype)
    std, mean = torch.std_mean(rewards_tensor, correction=int(use_run_ids))
    expected = ((rewards_tensor - mean) / (std + 1e-6)).to(advantages.dtype)
    torch.testing.assert_close(advantages, expected, rtol=0, atol=0)


def test_tensor_grpo_mixed_rewards_keep_sample_std_normalization():
    rewards = [0.1, 1.0, 0.1, 1.0]
    advantages, stds = _compute_tensor_advantages(rewards)
    tensor = torch.tensor(rewards)
    std, mean = torch.std_mean(tensor)
    torch.testing.assert_close(advantages, (tensor - mean) / (std + 1e-6), rtol=0, atol=0)
    torch.testing.assert_close(stds[0], std, rtol=0, atol=0)


@pytest.mark.parametrize("segment_counts", [(1, 1), (3, 1), (1, 4), (3, 5), (3, 0)])
@pytest.mark.parametrize("use_run_ids", [False, True])
def test_complete_trajectory_rewards_ignore_segment_counts(segment_counts, use_run_ids):
    trajectories = [
        _trajectory(1.0, 0, segment_counts[0]),
        _trajectory(0.0, 1, segment_counts[1]),
    ]
    result = compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=use_run_ids)
    std = 0.5**0.5 if use_run_ids else 0.5
    expected = 0.5 / (std + 1e-6)
    assert result is trajectories
    assert [float(trajectory["advantage"]) for trajectory in result] == pytest.approx(
        [expected, -expected]
    )
    sequences = list(iter_training_sequences(result))
    assert [float(segment["advantage"]) for segment in sequences] == pytest.approx(
        [expected] * segment_counts[0] + [-expected] * segment_counts[1]
    )
    assert all("reward" not in sequence for sequence in sequences if isinstance(sequence, dict))


@pytest.mark.parametrize("segment_counts", [(1, 1), (3, 1), (1, 4), (3, 5)])
def test_tensor_grpo_run_rewards_ignore_number_of_training_segments(segment_counts):
    winning, losing = segment_counts
    advantages, stds = _compute_tensor_advantages(
        [1.0] * winning + [0.0] * losing, torch.tensor([0] * winning + [1] * losing)
    )
    expected = 0.5 / (0.5**0.5 + 1e-6)
    assert float(stds[0]) == pytest.approx(0.5**0.5)
    assert advantages.tolist() == pytest.approx([expected] * winning + [-expected] * losing)


@pytest.mark.parametrize("segment_count", [0, 1, 4])
@pytest.mark.parametrize("use_run_ids", [False, True])
def test_single_complete_trajectory_always_has_zero_advantage(segment_count, use_run_ids):
    trajectory = _trajectory(2.0, segment_count=segment_count)
    compute_grpo_advantage_for_trajectories([trajectory], use_run_ids=use_run_ids)
    assert float(trajectory["advantage"]) == 0.0


@pytest.mark.parametrize("segment_count", [1, 4])
def test_tensor_single_trajectory_has_zero_advantage(segment_count):
    advantages, stds = _compute_tensor_advantages(
        [2.0] * segment_count, torch.zeros(segment_count, dtype=torch.long)
    )
    assert torch.equal(advantages, torch.zeros_like(advantages))
    assert stds[0] == 0.0


def test_tensor_conflicting_final_rewards_for_the_same_run_are_rejected():
    with pytest.raises(ValueError, match="final reward"):
        _compute_tensor_advantages([0.0, 1.0], torch.tensor([0, 0]))


def test_complete_objects_are_not_merged_by_repeated_legacy_run_ids():
    trajectories = [_trajectory(1.0, 0, 3, rollout_id="a"), _trajectory(0.0, 0, 1, rollout_id="b")]
    compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)
    expected = 0.5 / (0.5**0.5 + 1e-6)
    assert [float(trajectory["advantage"]) for trajectory in trajectories] == pytest.approx(
        [expected, -expected]
    )


def test_duplicate_complete_trajectory_ids_are_rejected():
    trajectories = [_trajectory(1.0, 0, rollout_id="same"), _trajectory(0.0, 1, rollout_id="same")]
    with pytest.raises(ValueError, match="Duplicate complete trajectory"):
        compute_grpo_advantage_for_trajectories(trajectories)


def test_known_group_includes_a_successful_zero_segment_trajectory():
    trajectories = [
        _trajectory(1.0, 0, 3, expected_rollouts=2, rollout_id="a"),
        _trajectory(0.0, 1, 0, expected_rollouts=2, rollout_id="b"),
    ]
    compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)
    expected = 0.5 / (0.5**0.5 + 1e-6)
    assert [float(trajectory["advantage"]) for trajectory in trajectories] == pytest.approx(
        [expected, -expected]
    )


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"expected_rollouts": 3}, "Inconsistent expected_rollouts"),
        ({"run_id": 0}, "Incomplete trajectory group"),
        ({"run_id": 2}, "run_id must identify"),
        ({"group_skipped": True}, "Inconsistent group_skipped"),
    ],
)
def test_known_group_contract_is_validated(changes, error):
    trajectories = [
        _trajectory(1.0, 0, expected_rollouts=2),
        _trajectory(0.0, 1, expected_rollouts=2),
    ]
    trajectories[1].extra_fields.update(changes)
    with pytest.raises(ValueError, match=error):
        compute_grpo_advantage_for_trajectories(trajectories)


def test_partial_known_group_cannot_supply_a_baseline():
    with pytest.raises(ValueError, match="Incomplete trajectory group"):
        compute_grpo_advantage_for_trajectories([_trajectory(1.0, expected_rollouts=2)])


def test_failed_execution_discards_the_whole_group_before_training():
    successful = _trajectory(1.0, 0, expected_rollouts=2)
    failed = _trajectory(None, 1, 0, expected_rollouts=2)
    failed.status = "failed"
    trajectories = [successful, failed]
    compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)
    assert all(trajectory.get("group_skipped") for trajectory in trajectories)
    assert list(iter_training_sequences(trajectories)) == []
    assert all("advantage" not in trajectory for trajectory in trajectories)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True, "1", None])
def test_invalid_successful_final_rewards_are_rejected(reward):
    with pytest.raises(ValueError, match="finite"):
        compute_grpo_advantage_for_trajectories([_trajectory(reward)])
