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

"""Fully async batching across controller, pool, trainer, and rollout admission."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from northrl.controller import NorthRLController
from northrl.mock import MockRolloutWorker
from northrl.northrl_types import Trajectory
from northrl.rollout_worker import base_rollout_worker as worker_module
from northrl.trainer import base_trainer as trainer_module
from northrl.trainer.base_trainer import BaseTrainer
from northrl.trajectory_pool import TrajectoryPool


def _root_config(algorithm="grpo", mode="fully-async"):
    return OmegaConf.create(
        {
            "algorithm": {"type": algorithm},
            "data": {"train": {"batch_size": 2, "rollout_repeat_n": 2}},
            "training": {"total_steps": 1, "weight_sync": {"mode": mode}},
            "runtime": {"debug": {"enable_data_dump": False}},
        }
    )


def _pool_config(config):
    controller = object.__new__(NorthRLController)
    controller._config = config
    return controller._derive_trajectory_pool_config()


@pytest.mark.parametrize("algorithm", ["grpo", "opd", "sft"])
@pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async", "no-sync"])
def test_controller_requires_valid_groups_only_for_grouped_fully_async(algorithm, mode):
    config = _pool_config(_root_config(algorithm, mode))

    assert config.require_trainable_groups is (
        algorithm in ("grpo", "opd") and mode == "fully-async"
    )
    assert config.groups_per_batch == 2
    assert config.group_size == 2


class _RecordingTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.batches = []

    async def train(self, trajectories):
        self.batches.append(trajectories)
        return {"training/optimizer_step_count": 1}, None, None


def _trainer_with_pool(*, source_finished=False):
    config = _root_config()
    trainer = _RecordingTrainer(config)
    tracker = MagicMock()
    tracker.track.side_effect = lambda *_args: nullcontext()
    tracker.is_rollout_worker_quiescent.return_value = source_finished
    sync = MagicMock()
    sync.is_waiting_for_validation.return_value = False
    loader = SimpleNamespace(
        can_return_item=lambda: not source_finished,
        is_finished=lambda: source_finished,
    )
    pool = TrajectoryPool(_pool_config(config))
    pool.set_activity_tracker(tracker)
    pool.set_module_references(loader, sync)
    trainer.set_activity_tracker(tracker)
    trainer.set_module_name("trainer")
    trainer.set_module_references(pool, sync)
    return trainer, pool, tracker, sync


def _put_group(pool, group_id, *, failed=False):
    for run_id in range(2):
        status = "failed" if failed else "success"
        trajectory = Trajectory(
            tokens=[1, 2],
            loss_mask=[0, 1],
            reward=None if status == "failed" else float(run_id),
            status=status,
            extra_fields={
                "model_tag": "actor",
                "group_id": group_id,
                "run_id": run_id,
                "rollout_id": f"{group_id}-{run_id}",
                "expected_rollouts": 2,
                "runtime_metrics": {"duration_seconds": 1.5},
            },
        )
        assert pool.put_trajectory(trajectory) == "success"


def test_rejected_groups_wait_for_replacements_and_publish_counters_once(monkeypatch):
    trainer, pool, tracker, sync = _trainer_with_pool()
    _put_group(pool, "rejected", failed=True)
    waits = []

    async def complete_replacements(delay):
        waits.append(delay)
        assert trainer.get_train_step() == 0
        assert trainer.batches == []
        tracker.experiment_logger_post.assert_not_called()
        _put_group(pool, "valid-1")
        _put_group(pool, "valid-2")

    monkeypatch.setattr(trainer_module.asyncio, "sleep", complete_replacements)
    asyncio.run(trainer._main_loop())

    assert len(waits) == 1
    assert trainer.get_train_step() == 1
    assert len(trainer.batches) == 1
    assert {trajectory["group_id"] for trajectory in trainer.batches[0]} == {
        "valid-1",
        "valid-2",
    }
    tracker.experiment_logger_post.assert_called_once()
    post = tracker.experiment_logger_post.call_args.kwargs
    assert post["step"] == 1
    assert post["data"]["pool/discarded_groups_total"] == 1
    assert post["data"]["pool/skipped_groups_total"] == 1
    assert post["data"]["pool/failed_trajectories_total"] == 2
    assert post["data"]["pool/discarded_rollout_seconds_total"] == 3
    tracker.set_training_step.assert_called_once_with(1)
    tracker.experiment_logger_flush.assert_called_once_with(1)
    sync.trajectory_pool_notify_batch_ready.assert_called_once_with("actor")
    sync.train_worker_notify_weight_update.assert_called_once()


@pytest.mark.parametrize("include_tail", [False, True])
def test_exhausted_rejected_groups_and_tail_do_not_advance_training(monkeypatch, include_tail):
    trainer, pool, tracker, sync = _trainer_with_pool(source_finished=True)
    _put_group(pool, "rejected", failed=True)
    if include_tail:
        _put_group(pool, "tail")

    async def stop_idle_trainer(_delay):
        trainer._stop_event.set()

    monkeypatch.setattr(trainer_module.asyncio, "sleep", stop_idle_trainer)
    asyncio.run(trainer._main_loop())

    assert pool.is_empty()
    assert trainer.batches == []
    assert trainer.get_train_step() == 0
    assert pool.get_discarded_metrics()["pool/tail_groups_total"] == int(include_tail)
    tracker.experiment_logger_post.assert_not_called()
    tracker.set_training_step.assert_not_called()
    sync.trajectory_pool_notify_batch_ready.assert_not_called()
    sync.train_worker_notify_weight_update.assert_not_called()


@pytest.mark.parametrize(
    "mode,watermark,ready_groups,should_wait",
    [
        ("fully-async", 1, 1, False),
        ("fully-async", 1, 3, False),
        ("fully-async", 1, 4, True),
        ("fully-async", 8, 4, False),
        ("fully-async", 8, 8, True),
        ("sync", 1, 1, True),
        ("batch-async", 1, 1, True),
        ("no-sync", 1, 1, True),
    ],
)
@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_fully_async_admission_watermark_allows_one_full_training_batch(
    monkeypatch, rollout_worker_config, mode, watermark, ready_groups, should_wait, algorithm
):
    rollout_worker_config.training.weight_sync.mode = mode
    rollout_worker_config.algorithm.type = algorithm
    rollout_worker_config.rollout.max_ready_groups = watermark
    rollout_worker_config.data.train.batch_size = 4
    worker = MockRolloutWorker(rollout_worker_config)
    pool = MagicMock()
    pool.ready_group_count.return_value = ready_groups
    worker._trajectory_pool = pool
    waits = []

    async def consume_ready_batch(delay):
        waits.append(delay)
        assert worker._pool_capacity_waiting is True
        pool.ready_group_count.return_value = 0

    monkeypatch.setattr(worker_module.asyncio, "sleep", consume_ready_batch)
    asyncio.run(worker._wait_for_trajectory_pool_capacity())

    assert bool(waits) is should_wait
    assert worker._pool_capacity_waiting is False
