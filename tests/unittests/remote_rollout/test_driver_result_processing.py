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

"""Local processing must not discard complete downloads or wait behind trace writes."""

import asyncio
import threading
import time
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northrl.remote_rollout import e2b_executor as executor
from northrl.remote_rollout.contracts import RemoteRolloutResult, RemoteSegment, RolloutInput
from northrl.remote_rollout.wire import DownloadQueue, dump_artifact, encode_result, run_blocking


def _payload():
    return dump_artifact(
        encode_result(
            RemoteRolloutResult(
                rollout_id="result",
                reward=1.0,
                segments=[
                    RemoteSegment(
                        tokens=[1, 2],
                        loss_mask=[0, 1],
                        log_probs=[0.0, -0.1],
                        old_log_probs=[0.0, -0.2],
                    )
                ],
            )
        )
    )


async def _started(event):
    async with asyncio.timeout(5):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_trace_processing_cannot_occupy_result_processing_slots():
    async def scenario():
        release = threading.Event()
        started = [threading.Event() for _ in range(4)]

        def write_trace(payload, index):
            started[index].set()
            assert release.wait(5)
            return payload

        sandbox = SimpleNamespace(files=SimpleNamespace(read=AsyncMock(return_value=_payload())))
        traces = [
            asyncio.create_task(
                executor._download_file(
                    sandbox,
                    executor._TRACE_PATH,
                    decode=lambda payload, index=index: write_trace(payload, index),
                )
            )
            for index in range(4)
        ]
        try:
            await asyncio.gather(*(_started(event) for event in started))
            result = await asyncio.wait_for(
                executor._read_result(sandbox, RolloutInput(rollout_id="result", task={})), 2
            )
            assert result.reward == 1.0
            assert isinstance(result.segments[0].tokens, array)
            assert not any(task.done() for task in traces)
        finally:
            release.set()
            await asyncio.gather(*traces)

    asyncio.run(scenario())


def test_complete_result_survives_remote_deadline_while_waiting_for_processing():
    async def scenario():
        release = threading.Event()
        started = [threading.Event(), threading.Event()]
        downloaded = asyncio.Event()

        def block(index):
            started[index].set()
            assert release.wait(5)

        async def read(*args, **kwargs):
            downloaded.set()
            return _payload()

        sandbox = SimpleNamespace(files=SimpleNamespace(read=AsyncMock(side_effect=read)))
        blockers = [asyncio.create_task(run_blocking(block, index)) for index in range(2)]
        result_task = None
        try:
            await asyncio.gather(*(_started(event) for event in started))
            budget = executor._DOWNLOAD_BUDGET.set((1.0, time.monotonic() + 0.1))
            try:
                result_task = asyncio.create_task(
                    executor._read_result(sandbox, RolloutInput(rollout_id="result", task={}))
                )
            finally:
                executor._DOWNLOAD_BUDGET.reset(budget)
            await asyncio.wait_for(downloaded.wait(), 1)
            await asyncio.sleep(0.15)
            assert not result_task.done()
            release.set()
            result = await asyncio.wait_for(result_task, 2)
            assert result.status == "success"
            assert result.reward == 1.0
            assert sandbox.files.read.await_count == 1
        finally:
            release.set()
            await asyncio.gather(*blockers)
            if result_task is not None:
                await asyncio.gather(result_task, return_exceptions=True)

    asyncio.run(scenario())


def test_invalid_download_still_fails_validation():
    async def scenario():
        sandbox = SimpleNamespace(files=SimpleNamespace(read=AsyncMock(return_value=_payload())))
        with pytest.raises(ValueError, match="ID does not match"):
            await executor._read_result(sandbox, RolloutInput(rollout_id="other", task={}))
        sandbox.files.read.return_value = b"not gzip"
        with pytest.raises(ValueError, match="valid gzip"):
            await executor._read_result(sandbox, RolloutInput(rollout_id="result", task={}))

    asyncio.run(scenario())


def test_slow_processing_bounds_downloaded_buffers_and_keeps_control_slot():
    async def scenario():
        queue = DownloadQueue(max_active=3)
        asyncio.get_running_loop()._northrl_download_queue = queue
        release = threading.Event()
        started = [threading.Event(), threading.Event()]

        def block(index):
            started[index].set()
            assert release.wait(5)

        sandbox = SimpleNamespace(files=SimpleNamespace(read=AsyncMock(return_value=_payload())))
        blockers = [asyncio.create_task(run_blocking(block, index)) for index in range(2)]
        results = []
        try:
            await asyncio.gather(*(_started(event) for event in started))
            results = [
                asyncio.create_task(
                    executor._read_result(sandbox, RolloutInput(rollout_id="result", task={}))
                )
                for _ in range(20)
            ]
            async with asyncio.timeout(2):
                while sandbox.files.read.await_count < 2 or queue.pending_count < 18:
                    await asyncio.sleep(0)
            assert sandbox.files.read.await_count == 2
            assert queue.active_count == 2
            control = SimpleNamespace(files=SimpleNamespace(read=AsyncMock(return_value="ok")))
            assert (
                await asyncio.wait_for(
                    executor._download_file(control, "/control", priority="control"), 1
                )
                == "ok"
            )
            assert not any(task.done() for task in results)
            release.set()
            assert all(result.reward == 1.0 for result in await asyncio.gather(*results))
            assert sandbox.files.read.await_count == 20
            assert queue.active_count == queue.pending_count == 0
        finally:
            release.set()
            await asyncio.gather(*blockers, *results, return_exceptions=True)

    asyncio.run(scenario())
