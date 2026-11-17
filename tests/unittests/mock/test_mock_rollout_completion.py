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

"""Mock replay preserves the complete-trajectory boundary."""

import copy
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from northrl.mock import MockRolloutWorker
from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import TrajectoryPool
from northrl.utils.async_utils import run_coro
from northrl.utils.trajectory_sequences import iter_trajectory_sequences


def _trajectory(group, run, count=1, *, reward=None, status="success"):
    identity = {
        "group_id": group,
        "run_id": run,
        "model_tag": "legacy",
        "rollout_id": f"saved-{group}-{run}",
        "expected_rollouts": 2,
    }
    sequences = [
        {"tokens": [1, 2 + index], "loss_mask": [0, 1], **identity, "old_log_probs": [0.0, -0.25]}
        for index in range(count)
    ]
    fields = (
        {"tokens": sequences[0]["tokens"], "loss_mask": sequences[0]["loss_mask"]}
        if count == 1
        else {"segments": sequences}
    )
    if count == 1:
        identity["old_log_probs"] = sequences[0]["old_log_probs"]
    return Trajectory(
        **fields,
        reward=float(run) if reward is None else reward,
        status=status,
        extra_fields=identity,
    )


def _setup(config, trajectories, *, groups_per_batch=1, repeat=2):
    worker = MockRolloutWorker(config)
    worker._mock_delay = 0
    worker._loaded_trajectories = trajectories
    pool = TrajectoryPool(
        OmegaConf.create(
            {
                "key_list": ["group_id"],
                "group_size": repeat,
                "groups_per_batch": groups_per_batch,
                "batch_size": groups_per_batch * repeat,
                "need_sampling": False,
            }
        )
    )
    pool.set_module_references(None, None)
    pool.put_trajectory = Mock(wraps=pool.put_trajectory)
    worker._trajectory_pool = pool
    return worker, pool


@pytest.mark.parametrize("entrypoint", ["execute_rollout", "rollout"])
def test_file_replay_keeps_complete_rollouts_and_ignores_outer_task_identity(
    rollout_worker_config, entrypoint
):
    saved = [
        _trajectory("a", 0, 2, reward=0.25),
        _trajectory("b", 1, 3),
        _trajectory("a", 1, 0, reward=0.75),
        _trajectory("b", 0, 1),
    ]
    original = copy.deepcopy(saved)
    worker, pool = _setup(rollout_worker_config, saved, groups_per_batch=2)
    task = {
        "group_id": "outer",
        "run_id": 0,
        "rollout_id": "outer-rollout",
        "expected_rollouts": 2,
        "prompt": "trigger",
    }
    assert run_coro(getattr(worker, entrypoint)(task)) is None
    trajectories = pool.get_trajectories("actor")
    assert (
        len(trajectories)
        == len(saved)
        == worker._processed_count
        == pool.put_trajectory.call_count
        == 4
    )
    assert len({trajectory["group_id"] for trajectory in trajectories}) == 2
    assert len({trajectory["rollout_id"] for trajectory in trajectories}) == 4
    assert [
        sum(1 for _ in iter_trajectory_sequences(trajectory)) for trajectory in trajectories
    ] == [2, 0, 1, 3]
    assert [trajectory.reward for trajectory in trajectories] == [0.25, 0.75, 0.0, 1.0]
    assert saved == original
    for trajectory in trajectories:
        assert trajectory["group_id"] not in {"outer", "a", "b"}
        assert trajectory["rollout_id"] != "outer-rollout"
        assert trajectory["expected_rollouts"] == 2
        assert trajectory.status == "success" and not trajectory["group_skipped"]
        for segment in iter_trajectory_sequences(trajectory):
            assert segment["old_log_probs"] == [0.0, -0.25]
            assert not isinstance(segment, dict) or "reward" not in segment
            for field in ("group_id", "run_id", "rollout_id", "expected_rollouts", "model_tag"):
                assert segment[field] == trajectory[field]
    assert run_coro(worker.execute_rollout({**task, "rollout_id": "another-outer"})) is None
    assert pool.put_trajectory.call_count == 4
    assert pool.is_empty()
    assert pool.get_trajectories("actor") is None


@pytest.mark.parametrize(
    "malformation,message",
    [
        ("missing_group", "requires group_id"),
        ("missing_run", "requires group_id"),
        ("incomplete_group", "all expected rollout slots"),
        ("duplicate_run", "duplicate trajectories"),
        ("mismatched_repeat", "Saved expected_rollouts"),
    ],
)
def test_file_replay_rejects_ambiguous_groups_before_publishing(
    rollout_worker_config, malformation, message
):
    saved = [_trajectory("a", 0), _trajectory("a", 1)]
    if malformation == "missing_group":
        del saved[0].extra_fields["group_id"]
    elif malformation == "missing_run":
        del saved[0].extra_fields["run_id"]
    elif malformation == "incomplete_group":
        saved.pop()
    elif malformation == "duplicate_run":
        saved.append(copy.deepcopy(saved[0]))
    else:
        saved[0]["expected_rollouts"] = 3
    worker, pool = _setup(rollout_worker_config, saved)
    with pytest.raises(ValueError, match=message):
        run_coro(worker.execute_rollout({"expected_rollouts": 2}))
    assert pool.is_empty()
    pool.put_trajectory.assert_not_called()
    assert not worker._all_loaded_into_pool


def test_file_replay_does_not_guess_repeat_count_from_segments(rollout_worker_config):
    worker, pool = _setup(rollout_worker_config, [_trajectory("a", 0, 3), _trajectory("a", 1, 0)])
    with pytest.raises(ValueError, match="expected_rollouts"):
        run_coro(worker.execute_rollout({"prompt": "trigger"}))
    pool.put_trajectory.assert_not_called()


def test_generated_mock_rollout_keeps_normal_execution_wrapper(rollout_worker_config):
    worker, pool = _setup(rollout_worker_config, [], repeat=1)
    task = {
        "prompt": "generated",
        "group_id": "actual-group",
        "run_id": 0,
        "rollout_id": "actual-rollout",
        "expected_rollouts": 1,
    }
    assert run_coro(worker.execute_rollout(task)) == "success"
    [trajectory] = pool.get_trajectories("actor")
    assert trajectory["rollout_id"] == "actual-rollout"
    assert trajectory.segments is None and trajectory.status == "success"
    assert len(trajectory.tokens) == 15
    assert pool.put_trajectory.call_count == 1


def test_file_replay_raises_on_failed_insertion_without_fabricating_a_result(rollout_worker_config):
    worker, pool = _setup(rollout_worker_config, [_trajectory("a", 0), _trajectory("a", 1)])
    pool.put_trajectory = Mock(return_value="fail")
    with pytest.raises(RuntimeError, match="Failed to replay"):
        run_coro(worker.execute_rollout({"expected_rollouts": 2}))
    assert pool.put_trajectory.call_count == 1
    assert worker._processed_count == 0
    assert pool.is_empty()


@pytest.mark.parametrize(
    "status,reward", [("failed", None), ("skipped", 0.0), ("success", float("nan"))]
)
def test_file_replay_retains_terminal_outcomes_for_whole_group_skip(
    rollout_worker_config, status, reward
):
    failed = _trajectory("a", 1, 0, status=status)
    failed.reward = reward
    worker, pool = _setup(rollout_worker_config, [_trajectory("a", 0, 2), failed])
    run_coro(worker.execute_rollout({"expected_rollouts": 2}))
    trajectories = pool.get_trajectories("actor")
    assert len(trajectories) == 2
    assert all(trajectory["group_skipped"] for trajectory in trajectories)
    assert trajectories[1].status == status
