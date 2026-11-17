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

"""Task readiness must wake idle workers without repeated empty claims."""

import asyncio
import threading
from collections import deque
from contextlib import asynccontextmanager, contextmanager

import pytest
from omegaconf import OmegaConf

from northrl.data_loader.data_loader import BaseDataLoader, SequentialDataLoader
from northrl.rollout_worker import base_rollout_worker as worker_module
from northrl.rollout_worker.base_rollout_worker import BaseRolloutWorker


class _Loader(SequentialDataLoader):
    """Real sequential-buffer behavior backed by in-memory batches."""

    def __init__(self, *, keep_batch_order=True):
        super().__init__(OmegaConf.create({"batch_size": 8, "keep_batch_order": keep_batch_order}))
        self._data_index = 1  # The initial ordered round has already been claimed.
        self.batches = deque()
        self.claims = 0
        self.before_claim = None
        self.after_empty = None
        self.empty_seen = threading.Event()
        self.subscribed = threading.Event()
        self.unsubscribed = threading.Event()
        self.callbacks = set()
        self.callbacks_lock = threading.Lock()

    def _fetch_batch_data(self):
        return self.batches.popleft() if self.batches else []

    def _reset_iterator(self):
        pass

    def is_finished(self):
        return False

    def release(self, *task_ids):
        self.batches.append([{"task_id": task_id} for task_id in task_ids])
        self.release_next_batch()

    def get_next_item(self):
        self.claims += 1
        callback, self.before_claim = self.before_claim, None
        if callback is not None:
            callback()
        item = super().get_next_item()
        if item is None:
            self.empty_seen.set()
            callback, self.after_empty = self.after_empty, None
            if callback is not None:
                callback()
        return item

    def subscribe_ready(self, callback):
        unsubscribe = super().subscribe_ready(callback)
        with self.callbacks_lock:
            self.callbacks.add(callback)
            self.unsubscribed.clear()
            self.subscribed.set()

        def tracked_unsubscribe():
            unsubscribe()
            with self.callbacks_lock:
                self.callbacks.discard(callback)
                if not self.callbacks:
                    self.unsubscribed.set()

        return tracked_unsubscribe

    def subscriber_count(self):
        with self.callbacks_lock:
            return len(self.callbacks)


class _Tracker:
    def __init__(self):
        self.active = False

    @contextmanager
    def track(self, *_args):
        self.active = True
        try:
            yield
        finally:
            self.active = False


class _Worker(BaseRolloutWorker):
    def __init__(self, config, loader, validation=None):
        super().__init__(config)
        self._activity_tracker = _Tracker()
        self.set_module_references(object(), loader, object(), validation or _Loader(), object())
        self.processed = []
        self.on_processed = lambda _task: None

    async def rollout(self, task):
        raise AssertionError("Notification tests must never execute a real rollout")

    async def execute_rollout(self, task):
        self.processed.append(task["task_id"])
        self.on_processed(task)
        await asyncio.sleep(0)  # Allow multiple awakened workers to share the batch.
        return "success"


@pytest.fixture
def observe_idle_wait(monkeypatch):
    """Observe a real unset Event.wait without replacing its wait semantics."""
    waits = asyncio.Queue()

    class ObservedEvent(asyncio.Event):
        async def wait(self):
            if not self.is_set():
                waits.put_nowait(self)
            return await super().wait()

    class AsyncioProxy:
        Event = ObservedEvent

        def __getattr__(self, name):
            return getattr(asyncio, name)

    monkeypatch.setattr(worker_module, "asyncio", AsyncioProxy())
    return waits


@asynccontextmanager
async def _running(*workers):
    tasks = [asyncio.create_task(worker._main_loop()) for worker in workers]
    try:
        yield tasks
    finally:
        for worker in workers:
            worker.request_stop()
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    raise result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def _next(queue, timeout=0.5):
    return await asyncio.wait_for(queue.get(), timeout)


def _cross_thread(call):
    """Complete a cross-thread release at a precise point in a synchronous claim."""
    errors = []

    def run():
        try:
            call()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive(), "Readiness callback deadlocked the releasing thread"
    if errors:
        raise errors[0]


@pytest.mark.parametrize("release_at", ["before_claim", "after_empty"])
def test_cross_thread_release_is_not_lost_around_empty_claim(rollout_worker_config, release_at):
    async def scenario():
        loader = _Loader()
        worker = _Worker(rollout_worker_config, loader)
        processed = asyncio.Queue()
        worker.on_processed = processed.put_nowait
        setattr(loader, release_at, lambda: _cross_thread(lambda: loader.release("ready")))
        async with _running(worker):
            assert (await _next(processed))["task_id"] == "ready"
        assert worker.processed == ["ready"]

    asyncio.run(scenario())


def test_release_wakes_a_worker_already_waiting(rollout_worker_config, observe_idle_wait):
    async def scenario():
        loader = _Loader()
        worker = _Worker(rollout_worker_config, loader)
        processed = asyncio.Queue()
        worker.on_processed = processed.put_nowait
        async with _running(worker):
            await _next(observe_idle_wait)
            await asyncio.to_thread(loader.release, "released")
            assert (await _next(processed))["task_id"] == "released"

    asyncio.run(scenario())


def test_fully_async_advances_batches_without_release_or_notifications(
    rollout_worker_config, observe_idle_wait, monkeypatch
):
    async def scenario():
        loader = _Loader(keep_batch_order=False)
        loader.batches.extend(
            [[{"task_id": task_id} for task_id in batch] for batch in ([0, 1], [2], [3, 4])]
        )

        def unexpected_release_or_notification():
            pytest.fail("Fully async batches must advance without release or notification")

        monkeypatch.setattr(loader, "release_next_batch", unexpected_release_or_notification)
        monkeypatch.setattr(loader, "_notify_ready", unexpected_release_or_notification)
        worker = _Worker(rollout_worker_config, loader)
        async with _running(worker) as tasks:
            # The first real idle wait must occur only after every batch was consumed.
            await _next(observe_idle_wait)
            assert worker.processed == [0, 1, 2, 3, 4]
            assert not loader.batches
            assert loader.empty_seen.is_set()
            assert not worker._activity_tracker.active
            worker.request_stop()
            await asyncio.wait_for(asyncio.shield(tasks[0]), 0.5)
        assert loader.subscriber_count() == 0

    asyncio.run(scenario())


def test_notifying_loader_does_not_poll_and_stop_wakes_waiter(
    rollout_worker_config, observe_idle_wait
):
    async def scenario():
        loader = _Loader()
        worker = _Worker(rollout_worker_config, loader)
        async with _running(worker) as tasks:
            await _next(observe_idle_wait)
            # Event.wait is suspended and no producer has signalled it: idle
            # workers must already have relinquished activity ownership.
            assert not worker._activity_tracker.active
            claims = loader.claims
            # Cover both the former 0.1-second poll and the legacy 1-second fallback.
            await asyncio.sleep(1.1)
            assert loader.claims == claims
            assert observe_idle_wait.empty()
            worker.request_stop()
            await asyncio.wait_for(asyncio.shield(tasks[0]), 0.5)
        assert loader.subscriber_count() == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["release", "stop"])
def test_notification_during_capacity_wait(rollout_worker_config, monkeypatch, action):
    async def scenario():
        loader = _Loader()
        worker = _Worker(rollout_worker_config, loader)
        entered, resume = asyncio.Event(), asyncio.Event()
        processed = asyncio.Queue()
        worker.on_processed = processed.put_nowait

        async def wait_for_capacity():
            entered.set()
            await resume.wait()

        monkeypatch.setattr(worker, "_wait_for_trajectory_pool_capacity", wait_for_capacity)
        async with _running(worker) as tasks:
            try:
                await asyncio.wait_for(entered.wait(), 0.5)
                assert not worker._activity_tracker.active
                assert loader.claims == 0
                if action == "release":
                    loader.release("after-capacity")
                else:
                    worker.request_stop()
            finally:
                resume.set()
            if action == "release":
                assert (await _next(processed))["task_id"] == "after-capacity"
                assert worker.processed == ["after-capacity"]
            else:
                await asyncio.wait_for(asyncio.shield(tasks[0]), 0.5)
                assert loader.claims == 0
                assert worker.processed == []

    asyncio.run(scenario())


@pytest.mark.parametrize("start_in_validation", [False, True])
def test_validation_switch_wakes_idle_worker(
    rollout_worker_config, observe_idle_wait, start_in_validation
):
    async def scenario():
        train, validation = _Loader(), _Loader()
        worker = _Worker(rollout_worker_config, train, validation)
        processed = asyncio.Queue()
        worker.on_processed = processed.put_nowait
        if start_in_validation:
            worker.begin_validate()
        target = train if start_in_validation else validation
        target.release("other-mode")  # Notification precedes subscription and cannot be relied on.
        async with _running(worker):
            await _next(observe_idle_wait)
            if start_in_validation:
                worker.end_validate()
            else:
                worker.begin_validate()
            assert (await _next(processed))["task_id"] == "other-mode"
        assert train.subscriber_count() == validation.subscriber_count() == 0

    asyncio.run(scenario())


def test_reset_wakes_worker_and_rechecks_reset_loader(rollout_worker_config, observe_idle_wait):
    async def scenario():
        loader = _Loader()
        worker = _Worker(rollout_worker_config, loader)
        processed = asyncio.Queue()
        worker.on_processed = processed.put_nowait
        async with _running(worker):
            await _next(observe_idle_wait)
            loader.batches.append([{"task_id": "after-reset"}])
            await asyncio.to_thread(loader.reset)
            assert (await _next(processed))["task_id"] == "after-reset"

    asyncio.run(scenario())


def test_legacy_loader_without_notifications_still_rechecks(
    rollout_worker_config, observe_idle_wait
):
    class LegacyLoader(_Loader):
        subscribe_ready = BaseDataLoader.subscribe_ready

    async def scenario():
        loader = LegacyLoader()
        worker = _Worker(rollout_worker_config, loader)
        processed = asyncio.Queue()
        worker.on_processed = processed.put_nowait
        async with _running(worker):
            await _next(observe_idle_wait)
            loader.release("legacy")
            assert (await _next(processed, timeout=2))["task_id"] == "legacy"

    asyncio.run(scenario())


def test_multiple_workers_claim_a_released_batch_once(rollout_worker_config, observe_idle_wait):
    async def scenario():
        loader = _Loader()
        workers = [_Worker(rollout_worker_config, loader) for _ in range(4)]
        processed = asyncio.Queue()
        for worker in workers:
            worker.on_processed = processed.put_nowait
        async with _running(*workers):
            for _ in workers:
                await _next(observe_idle_wait)
            loader.release(*range(8))
            received = [(await _next(processed))["task_id"] for _ in range(8)]
            assert sorted(received) == list(range(8))
        assert sum(len(worker.processed) for worker in workers) == 8
        assert loader.subscriber_count() == 0

    asyncio.run(scenario())


def test_run_cancel_restart_unregisters_both_loader_subscriptions(rollout_worker_config):
    train, validation = _Loader(), _Loader()
    worker = _Worker(rollout_worker_config, train, validation)
    received = threading.Event()
    worker.on_processed = lambda _task: received.set()
    worker.run()
    try:
        assert train.empty_seen.wait(timeout=2)
        assert train.subscriber_count() == validation.subscriber_count() == 1
        worker._task_future.cancel()
        assert train.unsubscribed.wait(timeout=2)
        assert validation.unsubscribed.wait(timeout=2)
        assert train.subscriber_count() == validation.subscriber_count() == 0

        train.empty_seen.clear()
        worker.run()
        assert train.empty_seen.wait(timeout=2)
        assert train.subscriber_count() == validation.subscriber_count() == 1
        train.release("restarted")
        assert received.wait(timeout=2)
        worker.request_stop()
        worker._task_future.result(timeout=2)
        assert train.subscriber_count() == validation.subscriber_count() == 0
        assert worker.processed == ["restarted"]
    finally:
        worker.stop()
