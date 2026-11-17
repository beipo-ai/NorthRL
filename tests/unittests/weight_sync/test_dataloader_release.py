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

"""Ordered dataloaders control when workers can start the next rollout batch."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from northrl.controller import NorthRLController
from northrl.mock import MockDataLoader
from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import TrajectoryPool
from northrl.weight_sync.weight_sync_controller import WeightSyncController


class _RolloutActivity:
    """Drive the real pool readiness check without starting inference workers."""

    def __init__(self):
        self.quiescent = False

    def is_rollout_worker_quiescent(self):
        return self.quiescent


def _make_pipeline(*, mode="sync", staleness_threshold=0, validate_freq=0, backend="mock"):
    controller = WeightSyncController(
        OmegaConf.create(
            {
                "training": {
                    "weight_sync": {"mode": mode, "staleness_threshold": staleness_threshold}
                },
                "backend": {"type": backend},
                "algorithm": {"type": "grpo"},
                "rollout": {"type": "default"},
                "validate": {"schedule": {"every_n_steps": validate_freq}},
            }
        )
    )
    loader = MockDataLoader(OmegaConf.create({"batch_size": 2, "keep_batch_order": True}))
    pool = TrajectoryPool(
        OmegaConf.create(
            {
                "batch_size": 2,
                "groups_per_batch": 2,
                "group_size": 1,
                "key_list": ["group_id"],
            }
        )
    )
    activity = _RolloutActivity()
    pool.set_activity_tracker(activity)
    pool.set_module_references(loader, controller)
    controller.set_module_references(loader, pool)
    return controller, loader, pool, activity


def _trajectory(item_id):
    return Trajectory(
        tokens=[item_id],
        loss_mask=[1],
        reward=1.0,
        extra_fields={
            "model_tag": "actor",
            "group_id": f"group-{item_id}",
            "run_id": 0,
            "rollout_id": f"rollout-{item_id}",
            "expected_rollouts": 1,
        },
    )


def _complete(pool, trajectory):
    assert pool.put_trajectory(trajectory) == "success"


def _finish_first_batch(loader, pool, activity):
    first = loader.get_next_item()
    assert first is not None and first["id"] == 0
    first_result = _trajectory(first["id"])
    _complete(pool, first_result)
    activity.quiescent = True
    assert pool.get_trajectories("actor") is None  # The second task is still unclaimed.

    activity.quiescent = False
    second = loader.get_next_item()
    assert second is not None and second["id"] == 1
    assert loader.get_next_item() is None  # Claiming every task cannot fetch another batch.
    assert pool.get_trajectories("actor") is None  # One worker is still finishing its task.

    second_result = _trajectory(second["id"])
    _complete(pool, second_result)
    # Every scheduled outcome is published, so the trainer can claim the full
    # group batch before workers report idle. The real callback gates the next batch.
    assert not activity.quiescent
    assert pool.get_trajectories("actor") == [first_result, second_result]
    activity.quiescent = True


@pytest.mark.parametrize("mode", ["sync", "batch-async"])
def test_before_training_validation_releases_only_first_batch(mode, monkeypatch):
    weight_sync, loader, pool, activity = _make_pipeline(mode=mode)
    activity.quiescent = True
    claimed_tasks = []

    def resume_worker_and_claim_first_batch():
        # Model the rollout thread running immediately after the worker switches
        # back to training, before the controller continues its handoff.
        activity.quiescent = False
        claimed_tasks.extend(loader.get_next_item() for _ in range(2))
        assert [task["id"] for task in claimed_tasks] == [0, 1]
        assert loader.get_next_item() is None
        assert pool.is_empty()

    controller = NorthRLController.__new__(NorthRLController)
    controller.validator = SimpleNamespace(
        is_complete=lambda: True, compute_and_log_metrics=lambda: None
    )
    controller.rollout_workers = [SimpleNamespace(end_validate=resume_worker_and_claim_first_batch)]
    controller.weight_sync_controller = weight_sync
    monkeypatch.setattr(controller, "_check_module_exceptions", lambda: True)

    controller._end_validate("actor")

    # Completing the handoff must not release batch 2 while batch 1 is running.
    assert loader.get_next_item() is None
    results = []
    for task in claimed_tasks:
        trajectory = Trajectory(
            tokens=[task["id"]],
            loss_mask=[1],
            reward=1.0,
            extra_fields={
                "model_tag": "actor",
                **{
                    key: task[key]
                    for key in ("group_id", "run_id", "rollout_id", "expected_rollouts")
                },
            },
        )
        _complete(pool, trajectory)
        results.append(trajectory)
    activity.quiescent = True
    assert pool.get_progress_info()["ready_groups"] == 2
    assert pool.get_trajectories("actor") == results
    assert pool.is_empty()
    assert loader.get_next_item() is None


@pytest.mark.parametrize(
    ("mode", "release_after"),
    [
        ("sync", "training"),
        ("batch-async", "training"),
        ("batch-async", "skip"),
        ("sync", "validation"),
    ],
)
def test_next_batch_waits_for_training_or_validation(mode, release_after):
    controller, loader, pool, activity = _make_pipeline(
        mode=mode, validate_freq=1 if release_after == "validation" else 0
    )
    _finish_first_batch(loader, pool, activity)
    assert loader.get_next_item() is None

    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        weights_updated=release_after != "skip",
    )
    if release_after == "validation":
        assert controller.is_waiting_for_validation()
        assert loader.get_next_item() is None
        controller.end_validate("actor")

    assert not controller.is_waiting_for_validation()
    for expected_id in (2, 3):
        item = loader.get_next_item()
        assert item is not None and item["id"] == expected_id
        _complete(pool, _trajectory(item["id"]))
    assert loader.get_next_item() is None


@pytest.mark.parametrize("release_after", ["training", "skip", "validation"])
def test_final_round_does_not_fetch_another_batch(release_after):
    controller, loader, pool, activity = _make_pipeline(
        mode="batch-async", validate_freq=1 if release_after == "validation" else 0
    )
    _finish_first_batch(loader, pool, activity)
    assert loader.get_next_item() is None

    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        weights_updated=release_after != "skip",
        continue_training=False,
    )
    assert loader.get_next_item() is None
    if release_after == "validation":
        assert controller.is_waiting_for_validation()
        controller.end_validate("actor")
    assert not controller.is_waiting_for_validation()
    assert loader.get_next_item() is None


def test_batch_async_releases_before_training_without_releasing_twice():
    controller, loader, pool, activity = _make_pipeline(mode="batch-async", staleness_threshold=2)
    _finish_first_batch(loader, pool, activity)

    # The next batch is available while training the first batch. Consume every
    # task so an accidental second release would visibly expose task 4.
    for expected_id in (2, 3):
        item = loader.get_next_item()
        assert item is not None and item["id"] == expected_id
    assert loader.get_next_item() is None
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None
    )
    assert loader.get_next_item() is None


def test_pool_accepts_completed_trajectories_while_weights_are_pending_or_syncing():
    controller, loader, pool, activity = _make_pipeline(backend="weaver")
    _finish_first_batch(loader, pool, activity)
    assert controller._rollout_services["actor"].state == "need_sync"
    assert loader.get_next_item() is None

    # Direct submissions test the pool's contract independently of worker
    # scheduling. An ordered batch has already published every scheduled outcome;
    # this test does not pretend extra results arise in that normal path.
    _complete(pool, _trajectory(90))
    sync_entered = threading.Event()
    finish_sync = threading.Event()

    async def update_sampling_client(sampling_path, weight_version):
        assert (sampling_path, weight_version) == ("/samplers/1", "export-1")
        sync_entered.set()
        assert await asyncio.to_thread(finish_sync.wait, 5), "sync was not released"

    controller.set_weaver_service_holder(
        SimpleNamespace(update_sampling_client=update_sampling_client)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        sync = executor.submit(
            controller.train_worker_notify_weight_update,
            "trainer",
            "actor",
            sampling_path="/samplers/1",
            weight_version="export-1",
        )
        try:
            assert sync_entered.wait(timeout=5)
            assert controller._rollout_services["actor"].state == "syncing"
            assert loader.get_next_item() is None
            _complete(pool, _trajectory(91))
            assert pool.get_progress_info()["finished_trajectories"] == 2
        finally:
            finish_sync.set()
        sync.result(timeout=5)

    item = loader.get_next_item()
    assert item is not None and item["id"] == 2
    assert pool.get_progress_info()["finished_trajectories"] == 2


def test_sync_zero_segment_group_is_consumed_and_skipped_round_releases_data():
    controller, loader, pool, activity = _make_pipeline()
    for item_id in (0, 1):
        item = loader.get_next_item()
        assert item["id"] == item_id
        trajectory = _trajectory(item_id)
        trajectory.segments = []
        trajectory.status = "skipped"
        _complete(pool, trajectory)
    assert loader.get_next_item() is None
    activity.quiescent = True
    assert not pool.is_empty()
    trajectories = pool.get_trajectories("actor")
    assert len(trajectories) == 2
    assert all(
        trajectory["group_skipped"] and not trajectory.segments for trajectory in trajectories
    )
    assert pool.is_empty()
    assert pool.get_trajectories("actor") is None
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        weights_updated=False,
    )
    assert loader.get_next_item()["id"] == 2
