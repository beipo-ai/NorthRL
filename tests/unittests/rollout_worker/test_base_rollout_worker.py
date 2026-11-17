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

"""
Tests for BaseRolloutWorker
"""

import asyncio
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from northrl.activity_tracker import ActivityTracker, ActivityTrackerProxy
from northrl.controller import NorthRLController
from northrl.exceptions import WeightVersionMismatchError
from northrl.mock import MockRolloutWorker
from northrl.northrl_types import Trajectory
from northrl.utils.async_utils import submit_rollout_coro


class _ActivityTracker:
    def __init__(self) -> None:
        self.active = False

    @contextmanager
    def track(self, module, work):
        del module, work
        self.active = True
        try:
            yield
        finally:
            self.active = False


def test_base_rollout_worker_initialization(rollout_worker_config):
    """Test BaseRolloutWorker initialization through MockRolloutWorker"""
    worker = MockRolloutWorker(rollout_worker_config)

    assert worker._config is not None
    assert worker._inference_client is not None
    assert worker._stop_event is not None


def test_rollout_worker_has_module_name(rollout_worker_config):
    """Test that rollout worker has module name functionality"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Set module name
    worker.set_module_name("test_rollout_worker")
    assert worker.get_module_name() == "test_rollout_worker"


def test_rollout_worker_set_module_references(rollout_worker_config):
    """Test setting module references"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Create mock references
    class MockTrajectoryPool:
        pass

    class MockDataLoader:
        pass

    class MockWeightSyncController:
        pass

    class MockValidator:
        pass

    trajectory_pool = MockTrajectoryPool()
    dataloader = MockDataLoader()
    weight_sync_controller = MockWeightSyncController()
    validator = MockValidator()

    # Set references
    worker.set_module_references(
        trajectory_pool=trajectory_pool,
        dataloader=dataloader,
        weight_sync_controller=weight_sync_controller,
        validate_dataloader=dataloader,
        validator=validator,
    )

    assert worker._trajectory_pool is trajectory_pool
    assert worker._dataloader is dataloader
    assert worker._weight_sync_controller is weight_sync_controller
    assert worker._validator is validator


def test_rollout_worker_stop_event(rollout_worker_config):
    """Test stop event functionality"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Initially not set
    assert not worker._stop_event.is_set()

    # Set the stop event
    worker._stop_event.set()
    assert worker._stop_event.is_set()

    # Clear it
    worker._stop_event.clear()
    assert not worker._stop_event.is_set()


def test_rollout_worker_registers_activity_before_claiming_task(rollout_worker_config):
    worker = MockRolloutWorker(rollout_worker_config)
    tracker = _ActivityTracker()

    class _DataLoader:
        def get_next_item(self):
            assert tracker.active
            worker.request_stop()
            return None

    worker._activity_tracker = tracker
    worker._dataloader = _DataLoader()

    asyncio.run(worker._main_loop())

    assert not tracker.active


def test_rollout_worker_does_not_claim_another_task_after_stop(rollout_worker_config):
    worker = MockRolloutWorker(rollout_worker_config)
    tracker = _ActivityTracker()
    remaining_tasks = [{"task_id": 1}, {"task_id": 2}]
    claimed_tasks = []

    class _DataLoader:
        def get_next_item(self):
            task = remaining_tasks.pop(0) if remaining_tasks else None
            if task is not None:
                claimed_tasks.append(task)
            return task

    async def _rollout(task):
        worker.request_stop()
        return "success"

    worker._activity_tracker = tracker
    worker._dataloader = _DataLoader()
    worker.rollout = _rollout

    asyncio.run(worker._main_loop())

    assert claimed_tasks == [{"task_id": 1}]
    assert remaining_tasks == [{"task_id": 2}]


def test_rollout_version_mismatch_fails_worker_and_controller_health(rollout_worker_config):
    worker = MockRolloutWorker(rollout_worker_config)
    worker.set_module_name("rollout_worker_0")
    tracker = ActivityTracker(rollout_worker_config)
    worker.set_activity_tracker(ActivityTrackerProxy(tracker))
    remaining_tasks = [{"task_id": 1}, {"task_id": 2}]
    attempted_tasks = []
    mismatch = WeightVersionMismatchError("expected='version-1', observed='version-2'")

    class _DataLoader:
        def get_next_item(self):
            return remaining_tasks.pop(0) if remaining_tasks else None

    async def _rollout(task):
        attempted_tasks.append(task)
        # Keep a regression from retrying forever if the special catch returns.
        if len(attempted_tasks) > 1:
            worker.request_stop()
        raise mismatch

    worker._dataloader = _DataLoader()
    worker._trajectory_pool = MagicMock()
    worker._weight_sync_controller = MagicMock()
    worker.rollout = _rollout
    controller = object.__new__(NorthRLController)
    controller._config = rollout_worker_config
    controller.activity_tracker = tracker
    assert controller._check_module_exceptions()

    worker.run()
    future = worker._task_future
    try:
        with pytest.raises(WeightVersionMismatchError) as raised:
            future.result(timeout=5)
        assert raised.value is mismatch
        assert [task["task_id"] for task in attempted_tasks] == [1]
        assert remaining_tasks == [{"task_id": 2}]
        assert tracker.is_quiescent()
        assert tracker.get_error_health_status()["error_level_count"] == 1
        assert not controller._check_module_exceptions()
    finally:
        worker.stop()


def test_rollout_worker_routes_trajectory_by_task_marker(rollout_worker_config):
    worker = MockRolloutWorker(rollout_worker_config)
    worker._trajectory_pool = MagicMock()
    worker._trajectory_pool.put_trajectory.return_value = "training"
    worker._validator = MagicMock()
    worker._validator.put_trajectory.return_value = "validation"

    training_trajectory = Trajectory(tokens=[1], loss_mask=[0], reward=0.0, is_val=False)
    validation_trajectory = Trajectory(tokens=[1], loss_mask=[0], reward=0.0, is_val=True)

    worker.begin_validate()
    assert worker._inference_client._is_validation is True
    assert worker._put_trajectory(training_trajectory) == "training"

    worker.end_validate()
    assert worker._inference_client._is_validation is False
    assert worker._put_trajectory(validation_trajectory) == "validation"

    worker._trajectory_pool.put_trajectory.assert_called_once_with(training_trajectory)
    worker._validator.put_trajectory.assert_called_once_with(validation_trajectory)


def _start_pending_rollout(worker):
    started = threading.Event()
    finished = threading.Event()
    release_callbacks = []

    async def rollout():
        loop = asyncio.get_running_loop()
        release = asyncio.Event()
        release_callbacks.append(lambda: loop.call_soon_threadsafe(release.set))
        started.set()
        try:
            await asyncio.wait_for(release.wait(), timeout=5)
        finally:
            finished.set()

    worker._task_future = submit_rollout_coro(rollout())
    assert started.wait(timeout=5)
    return worker._task_future, finished, release_callbacks[0]


def test_worker_wait_stopped_refreshes_on_caller_thread_and_drains(rollout_worker_config):
    worker = MockRolloutWorker(rollout_worker_config)
    future, finished, release = _start_pending_rollout(worker)
    callback_threads = []

    def refresh():
        callback_threads.append(threading.get_ident())
        if len(callback_threads) >= 2:
            release()

    worker.request_stop()
    try:
        worker.wait_stopped(timeout=5, on_wait=refresh, wait_interval=0.005)
    finally:
        release()

    assert len(callback_threads) >= 2
    assert set(callback_threads) == {threading.get_ident()}
    assert finished.is_set()
    assert future.done() and not future.cancelled()
    assert worker._task_future is None


def test_worker_wait_stopped_keeps_total_timeout_while_refreshing(rollout_worker_config):
    worker = MockRolloutWorker(rollout_worker_config)
    future, finished, release = _start_pending_rollout(worker)
    refreshes = []

    def refresh():
        refreshes.append(time.monotonic())
        # Time spent displaying progress must count toward the original budget.
        time.sleep(0.02)

    worker.request_stop()
    started_at = time.monotonic()
    try:
        worker.wait_stopped(timeout=0.035, on_wait=refresh, wait_interval=0.005)
    finally:
        release()

    assert time.monotonic() - started_at < 1.0
    assert refreshes
    assert future.cancelled()
    assert finished.wait(timeout=5)
    assert worker._task_future is None


@pytest.mark.parametrize("with_progress", [False, True])
def test_worker_wait_stopped_treats_task_timeout_as_failure(
    rollout_worker_config, caplog, with_progress
):
    worker = MockRolloutWorker(rollout_worker_config)
    future = Future()
    future.set_exception(TimeoutError("rollout request timed out"))
    worker._task_future = future
    refresh = lambda: pytest.fail("A completed task must not be polled again")

    worker.wait_stopped(on_wait=refresh if with_progress else None, wait_interval=0.005)

    assert worker._task_future is None
    assert not future.cancelled()
    assert "stopped after task failure: rollout request timed out" in caplog.text
    assert "Timed out waiting for rollout worker" not in caplog.text
