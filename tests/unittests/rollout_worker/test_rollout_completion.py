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

"""A worker publishes one whole trajectory after the execution finishes."""

import asyncio
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from northrl.algorithm.advantage.grpo import compute_grpo_advantage_for_trajectories
from northrl.data_loader import BaseDataLoader
from northrl.mock import MockRolloutWorker
from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.trajectory_pool import TrajectoryPool
from northrl.utils.trajectory_sequences import iter_training_sequences


def _trajectory(reward=1.0, count=1):
    if count == 1:
        return Trajectory(tokens=[1, 2], loss_mask=[0, 1], reward=reward)
    return Trajectory(
        segments=[{"tokens": [1, 2], "loss_mask": [0, 1]} for _ in range(count)],
        reward=reward,
    )


def _worker(config):
    worker = MockRolloutWorker(config)
    worker._trajectory_pool = MagicMock()
    worker._trajectory_pool.put_trajectory.return_value = "success"
    return worker


def test_worker_publishes_all_segments_together_after_return(rollout_worker_config):
    worker = _worker(rollout_worker_config)
    task = BaseDataLoader.repeat_item({"prompt": "test"}, 1)[0]
    trajectory = _trajectory(count=3)

    async def rollout(_task):
        result = worker._put_trajectory(trajectory)
        worker._trajectory_pool.put_trajectory.assert_not_called()
        return result

    worker.rollout = rollout
    assert asyncio.run(worker.execute_rollout(task)) == "success"
    worker._trajectory_pool.put_trajectory.assert_called_once_with(trajectory)
    assert trajectory["rollout_id"] == task["rollout_id"]
    assert len(trajectory.segments) == 3
    assert trajectory.reward == 1.0
    assert all("reward" not in segment for segment in trajectory.segments)
    assert worker._active_task is worker._pending_trajectory is None


def test_failure_after_staging_never_publishes_a_successful_partial_result(rollout_worker_config):
    worker = _worker(rollout_worker_config)

    async def rollout(_task):
        worker._put_trajectory(_trajectory(count=3))
        raise RuntimeError("rollout failed after staging")

    worker.rollout = rollout
    with pytest.raises(RuntimeError, match="after staging"):
        asyncio.run(worker.execute_rollout({"prompt": "test"}))
    worker._trajectory_pool.put_trajectory.assert_called_once()
    failed = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert failed.status == "failed"
    assert failed.segments == []
    assert failed.reward is None
    assert worker._active_task is worker._pending_trajectory is None


def test_worker_rejects_multiple_trajectory_submissions_for_one_rollout(rollout_worker_config):
    worker = _worker(rollout_worker_config)

    async def rollout(_task):
        worker._put_trajectory(_trajectory())
        return worker._put_trajectory(_trajectory())

    worker.rollout = rollout
    with pytest.raises(RuntimeError, match="one complete Trajectory"):
        asyncio.run(worker.execute_rollout({"prompt": "test"}))
    worker._trajectory_pool.put_trajectory.assert_called_once()
    assert worker._trajectory_pool.put_trajectory.call_args.args[0].status == "failed"


def test_worker_rejects_a_bare_segment(rollout_worker_config):
    worker = _worker(rollout_worker_config)
    with pytest.raises(TypeError, match="complete Trajectory"):
        worker._put_trajectory({"tokens": [1, 2], "loss_mask": [0, 1]})
    worker._trajectory_pool.put_trajectory.assert_not_called()


@pytest.mark.parametrize("scored", [False, True])
def test_zero_segment_trajectory_has_an_explicit_outcome(rollout_worker_config, scored):
    worker = _worker(rollout_worker_config)

    async def rollout(_task):
        if scored:
            return worker._put_trajectory(_trajectory(reward=0.5, count=0))
        return None

    worker.rollout = rollout
    asyncio.run(worker.execute_rollout({"prompt": "test"}))
    outcome = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert outcome.status == ("success" if scored else "skipped")
    assert outcome.reward == (0.5 if scored else None)
    assert outcome.segments == []


def test_validation_receives_one_complete_trajectory(rollout_worker_config):
    worker = _worker(rollout_worker_config)
    worker._validator = MagicMock()
    worker._validator.put_trajectory.return_value = "success"
    trajectory = _trajectory(count=3)
    trajectory.is_val = True

    async def rollout(_task):
        return worker._put_trajectory(trajectory)

    worker.rollout = rollout
    assert asyncio.run(worker.execute_rollout({"is_val": True})) == "success"
    worker._trajectory_pool.put_trajectory.assert_not_called()
    worker._validator.put_trajectory.assert_called_once_with(trajectory)


def test_fully_async_complete_containers_reach_grpo_before_segment_expansion(rollout_worker_config):
    worker = _worker(rollout_worker_config)
    pool = TrajectoryPool(
        OmegaConf.create(
            {
                "key_list": ["group_id"],
                "group_size": 2,
                "groups_per_batch": 1,
                "batch_size": 2,
                "need_sampling": False,
            }
        )
    )
    pool.set_module_references(dataloader=MagicMock(), weight_sync_controller=MagicMock())
    pool.set_activity_tracker(MagicMock())
    worker._trajectory_pool = pool
    first, second = BaseDataLoader.repeat_item({"prompt": "test"}, 2)

    async def rollout(task):
        return worker._put_trajectory(
            _trajectory(reward=1.0, count=3)
            if task["run_id"] == 0
            else _trajectory(reward=0.0, count=0)
        )

    worker.rollout = rollout
    asyncio.run(worker.execute_rollout(first))
    assert pool.get_trajectories() is None
    asyncio.run(worker.execute_rollout(second))
    trainer = RemoteApiGrpoTrainer.__new__(RemoteApiGrpoTrainer)
    trainer._trajectory_pool = pool
    trainer._identifier = "actor"
    trajectories = trainer._get_trajectories()
    assert len(trajectories) == 2
    assert [len(trajectory.segments) for trajectory in trajectories] == [3, 0]
    compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)
    expected = 0.5 / (2**-0.5 + 1e-6)
    assert float(trajectories[0]["advantage"]) == pytest.approx(expected)
    assert float(trajectories[1]["advantage"]) == pytest.approx(-expected)
    segments = list(iter_training_sequences(trajectories))
    assert len(segments) == 3
    assert all(float(segment["advantage"]) == pytest.approx(expected) for segment in segments)
    assert pool.get_trajectories() is None


def test_single_turn_worker_preserves_direct_trajectory_fields(rollout_worker_config):
    worker = _worker(rollout_worker_config)
    trajectory = Trajectory(tokens=[1, 2], loss_mask=[0, 1], reward=0.5)

    async def rollout(_task):
        return worker._put_trajectory(trajectory)

    worker.rollout = rollout
    asyncio.run(worker.execute_rollout({"prompt": "single"}))
    worker._trajectory_pool.put_trajectory.assert_called_once_with(trajectory)
    assert trajectory.segments is None
    assert trajectory.tokens == [1, 2]
    assert trajectory.loss_mask == [0, 1]


def test_execution_runtime_is_frozen_before_pool_submission(rollout_worker_config):
    worker = _worker(rollout_worker_config)
    trajectory = _trajectory()

    async def rollout(_task):
        return worker._put_trajectory(trajectory)

    worker.rollout = rollout
    asyncio.run(worker.execute_rollout({"prompt": "test"}))
    runtime = trajectory["runtime_metrics"]
    assert runtime["finished_at"] >= runtime["started_at"] > 0
    assert runtime["duration_seconds"] >= 0
    assert "sandbox_retries" not in runtime
    assert "tool_calls" not in runtime


def test_failed_execution_preserves_known_request_timeout(rollout_worker_config):
    from types import SimpleNamespace

    worker = _worker(rollout_worker_config)

    async def generate(**kwargs):
        raise TimeoutError("request timed out")

    original = SimpleNamespace(generate=generate)
    worker._inference_client = original

    async def rollout(_task):
        await worker._inference_client.generate(messages=[])

    worker.rollout = rollout
    with pytest.raises(TimeoutError):
        asyncio.run(worker.execute_rollout({"prompt": "test"}))
    failed = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert failed["runtime_metrics"]["model_request_timeouts"] == 1
    assert failed["runtime_metrics"]["duration_seconds"] >= 0
    assert failed.segments == []
    assert worker._inference_client is original


def test_unused_custom_client_does_not_invent_zero_request_metrics(rollout_worker_config):
    worker = _worker(rollout_worker_config)
    worker._inference_client = MagicMock()
    trajectory = _trajectory()

    async def rollout(_task):
        return worker._put_trajectory(trajectory)

    worker.rollout = rollout
    asyncio.run(worker.execute_rollout({"prompt": "test"}))
    assert "model_requests" not in trajectory["runtime_metrics"]
    assert "model_request_timeouts" not in trajectory["runtime_metrics"]
