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

import asyncio
import threading
import time

import pytest

from northrl.remote_rollout.wire import (
    DownloadQueue,
    get_download_queue,
    run_background_io,
    run_blocking,
)


async def _turns():
    # Give both callers and newly created operation tasks a scheduling turn.
    for _ in range(4):
        await asyncio.sleep(0)


async def _cancel_all(tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    ("queue_factory", "normal_capacity", "control_capacity"),
    [(DownloadQueue, 63, 1), (get_download_queue, 120, 8)],
    ids=["default", "shared"],
)
def test_capacity_queues_more_without_calling_factories(
    queue_factory, normal_capacity, control_capacity
):
    async def scenario():
        queue = queue_factory()
        release = asyncio.Event()
        started = []
        factories_called = []

        async def operation(label):
            started.append(label)
            await release.wait()
            return label

        def submit(label, priority):
            def factory():
                factories_called.append(label)
                return operation(label)

            return asyncio.create_task(queue.run(factory, priority=priority))

        background_labels = [f"background-{index}" for index in range(normal_capacity + 2)]
        control_labels = [f"control-{index}" for index in range(control_capacity + 2)]
        tasks = [submit(label, "background") for label in background_labels]
        try:
            await _turns()
            assert started == background_labels[:normal_capacity]
            tasks.extend(submit(label, "control") for label in control_labels)
            await _turns()
            assert queue.active_count == normal_capacity + control_capacity
            assert queue.pending_count == 4
            assert (
                started == background_labels[:normal_capacity] + control_labels[:control_capacity]
            )
            assert factories_called == started
            release.set()
            assert await asyncio.gather(*tasks) == background_labels + control_labels
            assert queue.active_count == queue.pending_count == 0
        finally:
            release.set()
            await _cancel_all(tasks)

    asyncio.run(scenario())


def test_all_priority_queues_are_unbounded():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        release = asyncio.Event()
        called = []

        async def operation(label):
            called.append(label)
            await release.wait()
            return label

        def submit(label, priority="critical"):
            return asyncio.create_task(queue.run(lambda: operation(label), priority=priority))

        tasks = [submit("active", "background"), submit("control-active", "control")]
        try:
            await _turns()
            tasks.extend(
                submit(f"{priority}-{index}", priority)
                for priority in ("critical", "background", "control")
                for index in range(70)
            )
            await _turns()
            assert queue.active_count == 2
            assert queue.pending_count == 210
            assert called == ["active", "control-active"]
            assert all(not task.done() for task in tasks)
            release.set()
            results = await asyncio.gather(*tasks)
            assert len(results) == len(called) == 212
            assert all(isinstance(result, str) for result in results)
        finally:
            release.set()
            await _cancel_all(tasks)
        assert queue.active_count == queue.pending_count == 0

    asyncio.run(scenario())


def test_queued_critical_starts_before_earlier_background():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        release = asyncio.Event()
        critical_release = asyncio.Event()
        critical_started = asyncio.Event()
        order = []

        async def blocker():
            await release.wait()

        async def background():
            order.append("background")

        async def critical():
            order.append("critical")
            critical_started.set()
            await critical_release.wait()

        tasks = [asyncio.create_task(queue.run(blocker))]
        try:
            await _turns()
            tasks.append(asyncio.create_task(queue.run(background, priority="background")))
            tasks.append(asyncio.create_task(queue.run(critical)))
            await _turns()
            assert queue.pending_count == 2
            release.set()
            await asyncio.wait_for(critical_started.wait(), 1)
            assert order == ["critical"]
            critical_release.set()
            await asyncio.gather(*tasks)
            assert order == ["critical", "background"]
        finally:
            release.set()
            critical_release.set()
            await _cancel_all(tasks)

    asyncio.run(scenario())


def test_cancel_queued_never_calls_factory():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        release = asyncio.Event()
        called = []

        async def operation():
            called.append(True)
            return "ok"

        blocker = asyncio.create_task(queue.run(release.wait))
        await _turns()
        queued = asyncio.create_task(queue.run(operation))
        await _turns()
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert queue.pending_count == 0
        assert called == []
        replacement = asyncio.create_task(queue.run(operation))
        await _turns()
        assert queue.pending_count == 1
        release.set()
        assert await replacement == "ok"
        await blocker

    asyncio.run(scenario())


def test_active_cancellation_drains_cleanup_even_when_cancelled_again():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        entered = asyncio.Event()
        cleaning = asyncio.Event()
        finish_cleanup = asyncio.Event()
        next_called = asyncio.Event()

        async def operation():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()

        async def next_operation():
            next_called.set()

        first = asyncio.create_task(queue.run(operation))
        await entered.wait()
        first.cancel()
        await cleaning.wait()
        second = asyncio.create_task(queue.run(next_operation))
        await _turns()
        first.cancel()
        await _turns()
        assert not first.done()
        assert queue.active_count == 1
        assert not next_called.is_set()
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert next_called.is_set()
        assert queue.active_count == queue.pending_count == 0

    asyncio.run(scenario())


def test_worker_cancelled_before_first_execution_releases_slot(monkeypatch):
    async def scenario():
        queue = DownloadQueue(max_active=2)
        loop = asyncio.get_running_loop()
        create_task = loop.create_task
        called = []

        async def operation():
            called.append(True)
            return "ok"

        def cancel_before_start(coroutine, **kwargs):
            task = create_task(coroutine, **kwargs)
            if kwargs.get("name") == "northrl-download":
                task.cancel()
            return task

        with monkeypatch.context() as patch:
            patch.setattr(loop, "create_task", cancel_before_start)
            with pytest.raises(asyncio.CancelledError):
                await queue.run(operation)
        assert called == []
        assert queue.active_count == queue.pending_count == 0
        assert await queue.run(operation) == "ok"

    asyncio.run(scenario())


def test_previously_caught_cancellation_does_not_discard_new_queued_work():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        release = asyncio.Event()

        async def operation():
            return "ok"

        async def owner():
            task = asyncio.current_task()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.sleep(0)
            assert task.cancelling() == 1
            return await queue.run(operation)

        blocker = asyncio.create_task(queue.run(release.wait))
        await _turns()
        pending = asyncio.create_task(owner())
        await _turns()
        assert queue.pending_count == 1
        release.set()
        assert await pending == "ok"
        await blocker

    asyncio.run(scenario())


def test_cancellation_after_grant_before_caller_resumes_releases_slot(monkeypatch):
    async def scenario():
        queue = DownloadQueue(max_active=2)
        release = asyncio.Event()
        called = []

        async def operation():
            called.append(True)
            return "ok"

        blocker = asyncio.create_task(queue.run(release.wait))
        await _turns()
        pending = asyncio.create_task(queue.run(operation))
        await _turns()
        assert queue.pending_count == 1
        start_ready = queue._start_ready

        def grant_then_cancel():
            start_ready()
            if queue.active_count == 1 and queue.pending_count == 0:
                pending.cancel()

        with monkeypatch.context() as patch:
            patch.setattr(queue, "_start_ready", grant_then_cancel)
            release.set()
            await blocker
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert called == []
        assert queue.active_count == queue.pending_count == 0
        assert await queue.run(operation) == "ok"

    asyncio.run(scenario())


def test_cancelling_caller_does_not_recancel_worker_cleanup():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        entered = asyncio.Event()
        cleaning = asyncio.Event()
        finish_cleanup = asyncio.Event()
        workers = []

        async def operation():
            workers.append(asyncio.current_task())
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()

        caller = asyncio.create_task(queue.run(operation))
        await entered.wait()
        workers[0].cancel()
        await cleaning.wait()
        caller.cancel()
        await _turns()
        assert not caller.done()
        assert not workers[0].done()
        assert workers[0].cancelling() == 1
        assert queue.active_count == 1
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert queue.active_count == 0

    asyncio.run(scenario())


def test_deadline_covers_queue_wait():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        release = asyncio.Event()
        called = []

        async def operation():
            called.append(True)

        blocker = asyncio.create_task(queue.run(release.wait))
        await _turns()
        with pytest.raises(TimeoutError):
            await queue.run(operation, deadline=time.monotonic() + 0.01)
        assert called == []
        assert queue.pending_count == 0
        release.set()
        await blocker
        await queue.run(operation)
        assert called == [True]

    asyncio.run(scenario())


def test_expired_deadline_does_not_call_factory():
    async def scenario():
        queue = DownloadQueue()
        called = []

        async def operation():
            called.append(True)

        with pytest.raises(TimeoutError):
            await queue.run(operation, deadline=time.monotonic() - 1)
        assert called == []
        assert queue.active_count == queue.pending_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout_kind", ["external", "deadline"])
def test_timeout_waits_for_operation_cleanup(timeout_kind):
    async def scenario():
        queue = DownloadQueue(max_active=2)
        cleaning = asyncio.Event()
        finish_cleanup = asyncio.Event()

        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()

        call = (
            asyncio.wait_for(queue.run(operation), 0.01)
            if timeout_kind == "external"
            else queue.run(operation, deadline=time.monotonic() + 0.01)
        )
        task = asyncio.create_task(call)
        await asyncio.wait_for(cleaning.wait(), 1)
        assert not task.done()
        assert queue.active_count == 1
        finish_cleanup.set()
        with pytest.raises(TimeoutError):
            await task
        assert queue.active_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("synchronous", [False, True])
def test_operation_exception_releases_slot(synchronous):
    async def scenario():
        queue = DownloadQueue(max_active=2)

        async def failure():
            raise ValueError("operation failed")

        def factory():
            if synchronous:
                raise ValueError("operation failed")
            return failure()

        async def success():
            return "ok"

        with pytest.raises(ValueError, match="operation failed"):
            await queue.run(factory)
        assert queue.active_count == 0
        assert await queue.run(success) == "ok"

    asyncio.run(scenario())


def test_shared_queue_is_per_loop():
    async def scenario():
        queue = get_download_queue()
        assert get_download_queue() is queue

        async def operation():
            assert queue.active_count == 1
            return "ok"

        assert await get_download_queue().run(operation) == "ok"
        return queue

    assert asyncio.run(scenario()) is not asyncio.run(scenario())


@pytest.mark.parametrize(
    ("runner", "capacity"),
    [(run_blocking, 2), (run_background_io, 4)],
    ids=["result", "background"],
)
def test_blocking_limit_holds_permit_until_cancelled_thread_really_finishes(runner, capacity):
    async def scenario():
        loop = asyncio.get_running_loop()
        releases = [threading.Event() for _ in range(capacity + 1)]
        started = [asyncio.Event() for _ in range(capacity + 1)]
        active = 0
        maximum = 0
        lock = threading.Lock()

        def operation(index, *, result):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            loop.call_soon_threadsafe(started[index].set)
            try:
                assert releases[index].wait(5), "test did not release blocking operation"
                return result
            finally:
                with lock:
                    active -= 1

        tasks = [
            asyncio.create_task(runner(operation, index, result=index))
            for index in range(capacity + 1)
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in started[:capacity])), 1
            )
            assert not started[capacity].is_set()
            tasks[0].cancel()
            await _turns()
            tasks[0].cancel()
            await _turns()
            assert not tasks[0].done()
            assert not started[capacity].is_set()
            releases[0].set()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            await asyncio.wait_for(started[capacity].wait(), 1)
            for release in releases[1:]:
                release.set()
            assert await asyncio.gather(*tasks[1:]) == list(range(1, capacity + 1))
            assert maximum == capacity
        finally:
            for release in releases:
                release.set()
            await _cancel_all(tasks)

    asyncio.run(scenario())


@pytest.mark.parametrize("outside_deadline", [False, True])
def test_download_slot_includes_blocking_postprocessing_during_cancellation(
    outside_deadline,
):
    async def scenario():
        loop = asyncio.get_running_loop()
        queue = DownloadQueue(max_active=2)
        thread_started = asyncio.Event()
        thread_release = threading.Event()
        second_started = asyncio.Event()

        def postprocess():
            loop.call_soon_threadsafe(thread_started.set)
            assert thread_release.wait(5), "test did not release postprocessing"

        async def operation():
            if not outside_deadline:
                await run_blocking(postprocess)

        async def second_operation():
            second_started.set()

        first = asyncio.create_task(
            queue.run(
                operation,
                postprocess=((lambda _: run_blocking(postprocess)) if outside_deadline else None),
            )
        )
        second = None
        try:
            await asyncio.wait_for(thread_started.wait(), 1)
            first.cancel()
            second = asyncio.create_task(queue.run(second_operation))
            await _turns()
            assert not first.done()
            assert queue.active_count == 1
            assert not second_started.is_set()
            thread_release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            await second
            assert second_started.is_set()
        finally:
            thread_release.set()
            await _cancel_all([first, second] if second is not None else [first])

    asyncio.run(scenario())
