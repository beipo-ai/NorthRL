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

"""Concurrency and lifecycle guarantees for sampling's synchronous work."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from northrl.weaver.sampling_preparation import SamplingPreparationWorker


async def _entered(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 3), "preparation callback did not start"


def test_slow_preparation_leaves_sdk_loop_responsive():
    async def exercise() -> None:
        worker = SamplingPreparationWorker()
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        sdk_thread = threading.get_ident()
        callback_threads = []

        def slow_render(message: str, *, suffix: str) -> str:
            callback_threads.append(threading.get_ident())
            started.set()
            try:
                assert release.wait(3), "slow render was not released"
                return message + suffix
            finally:
                completed.set()

        task = asyncio.create_task(worker.run(slow_render, "prompt", suffix="!"))
        try:
            await _entered(started)
            probe = await asyncio.wait_for(asyncio.sleep(0, result="responsive"), 1)
            assert probe == "responsive"
            assert not completed.is_set(), "probe ran only after blocking work completed"
            assert len(callback_threads) == 1
            assert callback_threads[0] != sdk_thread
        finally:
            release.set()
        assert await task == "prompt!"
        await worker.close()

    asyncio.run(exercise())


def test_concurrent_tokenizer_operations_share_one_slot():
    class Tokenizer:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        def access(self, text: str) -> str:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                # Encoding and decoding may temporarily modify tokenizer state.
                time.sleep(0.005)
                assert self.active == 1, "shared tokenizer was used concurrently"
                return text.upper()
            finally:
                self.active -= 1

    async def exercise() -> None:
        worker = SamplingPreparationWorker()
        tokenizer = Tokenizer()
        operations = [f"{'encode' if index % 2 else 'decode'}-{index}" for index in range(30)]
        results = await asyncio.gather(
            *(worker.run(tokenizer.access, operation) for operation in operations)
        )
        assert results == [operation.upper() for operation in operations]
        assert tokenizer.max_active == 1
        assert tokenizer.active == 0
        await worker.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("callback_fails", [False, True])
def test_repeated_cancellation_drains_active_callback_before_admitting_next(callback_fails):
    async def exercise() -> None:
        worker = SamplingPreparationWorker()
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        next_started = threading.Event()
        error = ValueError("invalid prompt")

        def blocked_encode() -> list[int]:
            started.set()
            try:
                assert release.wait(3), "cancelled encoding was not released"
                if callback_fails:
                    raise error
                return [1, 2]
            finally:
                completed.set()

        def next_encode() -> list[int]:
            assert completed.is_set(), "next tokenizer call overlapped cancelled encoding"
            next_started.set()
            return [3, 4]

        active = asyncio.create_task(worker.run(blocked_encode))
        await _entered(started)
        waiting = asyncio.create_task(worker.run(next_encode))
        try:
            for _ in range(3):
                active.cancel()
                await asyncio.sleep(0)
                assert not active.done(), "cancelled request detached its background callback"
                assert not next_started.is_set(), "cancellation prematurely released the slot"
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await active
        if callback_fails:
            assert raised.value.__cause__ is error
        assert completed.is_set(), "cancellation returned before callback stopped"
        assert await waiting == [3, 4]
        await worker.close()

    asyncio.run(exercise())


def test_cancelled_queue_does_not_create_background_callbacks():
    async def exercise() -> None:
        worker = SamplingPreparationWorker()
        started = threading.Event()
        release = threading.Event()
        calls = []

        def blocking_first() -> None:
            calls.append("first")
            started.set()
            assert release.wait(3), "first callback was not released"

        active = asyncio.create_task(worker.run(blocking_first))
        await _entered(started)
        queued = [asyncio.create_task(worker.run(calls.append, index)) for index in range(40)]
        await asyncio.sleep(0)
        for task in queued:
            task.cancel()
        try:
            results = await asyncio.gather(*queued, return_exceptions=True)
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert calls == ["first"]
        finally:
            release.set()
        await active
        assert calls == ["first"], "cancelled queued requests ran later in the executor"
        assert await worker.run(lambda: "next") == "next"
        await worker.close()

    asyncio.run(exercise())


def test_close_rejects_new_and_queued_work_and_drains_before_cancelling():
    async def exercise() -> None:
        worker = SamplingPreparationWorker()
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        calls = []

        def blocking_load() -> object:
            started.set()
            try:
                assert release.wait(3), "tokenizer load was not released"
                return object()
            finally:
                completed.set()

        active = asyncio.create_task(worker.run(blocking_load))
        await _entered(started)
        queued = asyncio.create_task(worker.run(calls.append, "queued"))
        await asyncio.sleep(0)
        closing = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        try:
            with pytest.raises(RuntimeError, match="closed"):
                await worker.run(calls.append, "new")
            for _ in range(3):
                closing.cancel()
                await asyncio.sleep(0)
                assert not closing.done(), "cancelled close abandoned the active callback"
            assert not completed.is_set()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="closed"):
            await active
        with pytest.raises(RuntimeError, match="closed"):
            await queued
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert completed.is_set()
        assert calls == [], "closing holder admitted queued or new work"
        await worker.close()
        with pytest.raises(RuntimeError, match="closed"):
            await worker.run(calls.append, "after-close")

    asyncio.run(exercise())


def test_callback_error_preserves_identity_and_releases_slot():
    async def exercise() -> None:
        worker = SamplingPreparationWorker()
        error = ValueError("invalid tool argument")

        def fail() -> None:
            raise error

        with pytest.raises(ValueError) as raised:
            await worker.run(fail)
        assert raised.value is error
        assert await worker.run(lambda: [1, 2, 3]) == [1, 2, 3]
        await worker.close()

    asyncio.run(exercise())


def test_independent_holders_can_prepare_concurrently():
    async def exercise() -> None:
        workers = [SamplingPreparationWorker(), SamplingPreparationWorker()]
        rendezvous = threading.Barrier(2)

        def prepare(value: int) -> int:
            rendezvous.wait(timeout=3)
            return value

        assert await asyncio.gather(
            *(worker.run(prepare, index) for index, worker in enumerate(workers))
        ) == [0, 1]
        await asyncio.gather(*(worker.close() for worker in workers))

    asyncio.run(exercise())
