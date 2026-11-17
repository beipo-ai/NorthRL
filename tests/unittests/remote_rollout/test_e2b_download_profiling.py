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

"""Download diagnostics distinguish admission, remote reads, and local processing."""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.wire import DownloadQueue, run_background_io
from northrl.utils import profiling


@pytest.fixture(autouse=True)
def profile_events(monkeypatch):
    events = Mock(wraps=e2b_executor.profile_event)
    monkeypatch.setattr(e2b_executor, "profile_event", events)
    return events


def _sandbox(read):
    return SimpleNamespace(sandbox_id="sandbox-test", files=SimpleNamespace(read=read))


def _profile(caplog):
    records = [
        record for record in caplog.records if "stage=driver.file-download " in record.message
    ]
    assert len(records) == 1
    event = e2b_executor.profile_event.call_args
    return records[0], {"stage": event.args[0], "status": event.args[1], **event.kwargs}


def test_result_success_is_visible_at_info_with_bytes_and_processing(caplog):
    caplog.set_level(logging.INFO, logger=profiling.__name__)
    read = AsyncMock(return_value=b"compressed-result")

    def process(raw):
        time.sleep(0.01)
        return {"payload": raw.decode()}

    result = asyncio.run(
        e2b_executor._download_file(
            _sandbox(read),
            e2b_executor._RESULT_PATH,
            decode=process,
            file_format="bytes",
        )
    )
    assert result == {"payload": "compressed-result"}
    read.assert_awaited_once_with(e2b_executor._RESULT_PATH, user="root", format="bytes")
    record, profile = _profile(caplog)
    assert record.levelno == logging.INFO
    assert profile["sandbox_id"] == "sandbox-test"
    assert profile["priority"] == "critical"
    assert profile["status"] == profile["phase"] == "completed"
    assert profile["bytes"] == len(b"compressed-result")
    assert profile["processing_seconds"] >= 0.01
    assert profile["duration_seconds"] >= profile["processing_seconds"]
    assert profile["queue_active_at_submit"] == profile["queue_pending_at_submit"] == 0


def test_zero_timestamps_are_valid_and_sandbox_id_is_formatted(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=profiling.__name__)
    timestamps = iter([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(e2b_executor, "time", SimpleNamespace(monotonic=lambda: next(timestamps)))
    sandbox = _sandbox(AsyncMock(return_value=b"result"))
    sandbox.sandbox_id = SimpleNamespace(id="sandbox-test")
    assert (
        asyncio.run(
            e2b_executor._download_file(
                sandbox,
                e2b_executor._RESULT_PATH,
                decode=lambda raw: raw,
                blocking_decode=False,
            )
        )
        == b"result"
    )
    _, profile = _profile(caplog)
    assert profile["sandbox_id"] == str(sandbox.sandbox_id)
    assert profile["queue_wait_seconds"] == profile["read_seconds"] == 0
    assert profile["processing_queue_wait_seconds"] == 0
    assert profile["processing_seconds"] == profile["duration_seconds"] == 1


def test_fast_background_read_does_not_build_hidden_debug_event(caplog, profile_events):
    caplog.set_level(logging.INFO, logger=profiling.__name__)
    formatted = []

    class UnformattedSandboxId:
        def __str__(self):
            formatted.append(True)
            return "sandbox-test"

    sandbox = _sandbox(AsyncMock(return_value=b"trace"))
    sandbox.sandbox_id = UnformattedSandboxId()
    assert (
        asyncio.run(
            e2b_executor._download_file(
                sandbox,
                "/trace",
                priority="background",
            )
        )
        == b"trace"
    )
    profile_events.assert_not_called()
    assert not formatted


@pytest.mark.parametrize("read_fails", [False, True])
def test_profile_failure_does_not_change_download_outcome(monkeypatch, caplog, read_fails):
    caplog.set_level(logging.INFO, logger=profiling.__name__)
    diagnostic = Mock(side_effect=ValueError("diagnostics unavailable"))
    monkeypatch.setattr(e2b_executor, "profile_event", diagnostic)
    original_error = TimeoutError("original network timeout")
    read = (
        AsyncMock(side_effect=original_error) if read_fails else AsyncMock(return_value=b"result")
    )
    download = e2b_executor._download_file(_sandbox(read), e2b_executor._RESULT_PATH)
    if read_fails:
        with pytest.raises(TimeoutError) as raised:
            asyncio.run(download)
        assert raised.value is original_error
    else:
        assert asyncio.run(download) == b"result"
    diagnostic.assert_called_once()


def test_queue_timeout_records_wait_without_starting_read(caplog):
    caplog.set_level(logging.INFO, logger=profiling.__name__)

    async def scenario():
        queue = DownloadQueue(max_active=2)
        asyncio.get_running_loop()._northrl_download_queue = queue
        started, release = asyncio.Event(), asyncio.Event()

        async def occupy():
            started.set()
            await release.wait()

        blocker = asyncio.create_task(queue.run(occupy))
        read = AsyncMock()
        await started.wait()
        try:
            with pytest.raises(TimeoutError):
                await e2b_executor._download_file(
                    _sandbox(read), "/queued", deadline=time.monotonic() + 0.03
                )
            read.assert_not_awaited()
            assert queue.active_count == 1
            assert queue.pending_count == 0
        finally:
            release.set()
            await blocker

    asyncio.run(scenario())
    record, profile = _profile(caplog)
    assert record.levelno == logging.WARNING
    assert profile["phase"] == "queue"
    assert profile["queue_wait_seconds"] >= 0.02
    assert profile["read_seconds"] == profile["processing_seconds"] == 0
    assert profile["bytes"] is None
    assert profile["queue_active_at_submit"] == 1
    assert profile["queue_pending_at_submit"] == 0
    assert profile["error_type"] == "TimeoutError"


def test_read_timeout_records_elapsed_network_wait(caplog):
    caplog.set_level(logging.INFO, logger=profiling.__name__)

    async def read(*_args, **_kwargs):
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        asyncio.run(e2b_executor._download_file(_sandbox(read), "/slow-read", io_seconds=0.03))
    _, profile = _profile(caplog)
    assert profile["phase"] == "read"
    assert profile["read_seconds"] >= 0.02
    assert profile["processing_queue_wait_seconds"] == profile["processing_seconds"] == 0
    assert profile["duration_seconds"] >= profile["read_seconds"]


@pytest.mark.parametrize("timeout", [False, True])
def test_processing_pool_wait_is_separate_from_read_and_callback(caplog, timeout):
    caplog.set_level(logging.DEBUG, logger=profiling.__name__)
    release = threading.Event()
    processed = []

    async def scenario():
        loop = asyncio.get_running_loop()
        started = [asyncio.Event() for _ in range(4)]
        read_finished = asyncio.Event()

        def occupy(index):
            loop.call_soon_threadsafe(started[index].set)
            assert release.wait(2)

        async def read(*_args, **_kwargs):
            read_finished.set()
            return b"payload"

        def process(raw):
            processed.append(raw)
            return raw.decode()

        workers = [asyncio.create_task(run_background_io(occupy, index)) for index in range(4)]
        download = None
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
            download = asyncio.create_task(
                e2b_executor._download_file(
                    _sandbox(read),
                    "/processing",
                    decode=process,
                    deadline=time.monotonic() + 0.05 if timeout else None,
                )
            )
            await asyncio.wait_for(read_finished.wait(), 1)
            await asyncio.sleep(0.025)
            assert not processed
            if timeout:
                with pytest.raises(TimeoutError):
                    await download
                assert not processed
            else:
                release.set()
                assert await download == "payload"
        finally:
            release.set()
            await asyncio.gather(*workers)
            if download is not None and not download.done():
                download.cancel()
                await asyncio.gather(download, return_exceptions=True)

    asyncio.run(scenario())
    _, profile = _profile(caplog)
    assert profile["processing_queue_wait_seconds"] >= 0.02
    assert profile["bytes"] == 7
    assert profile["phase"] == ("processing_queue" if timeout else "completed")
    if timeout:
        assert profile["processing_seconds"] == 0
    else:
        assert profile["processing_seconds"] > 0


def test_processing_failure_keeps_callback_time_and_received_bytes(caplog):
    caplog.set_level(logging.INFO, logger=profiling.__name__)

    def process(_raw):
        time.sleep(0.01)
        raise ValueError("invalid gzip")

    with pytest.raises(ValueError, match="invalid gzip"):
        asyncio.run(
            e2b_executor._download_file(
                _sandbox(AsyncMock(return_value=b"invalid")), "/invalid", decode=process
            )
        )
    _, profile = _profile(caplog)
    assert profile["phase"] == "processing"
    assert profile["bytes"] == 7
    assert profile["processing_seconds"] >= 0.01
    assert profile["error_type"] == "ValueError"


@pytest.mark.parametrize("priority", ["background", "control"])
def test_expected_missing_file_stays_debug(caplog, priority):
    caplog.set_level(logging.DEBUG, logger=profiling.__name__)
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            e2b_executor._download_file(
                _sandbox(AsyncMock(side_effect=FileNotFoundError("not ready"))),
                "/not-ready",
                priority=priority,
            )
        )
    record, profile = _profile(caplog)
    assert record.levelno == logging.DEBUG
    assert profile["phase"] == "read"
    assert profile["error_type"] == "FileNotFoundError"


def test_cancellation_drains_processing_before_logging_and_releasing_slot(caplog):
    caplog.set_level(logging.DEBUG, logger=profiling.__name__)
    release = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        queue = DownloadQueue(max_active=2)
        loop._northrl_download_queue = queue
        started = asyncio.Event()

        def process(raw):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(2)
            return raw

        task = asyncio.create_task(
            e2b_executor._download_file(
                _sandbox(AsyncMock(return_value=b"payload")), "/drain", decode=process
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.sleep(0.025)
            assert not task.done()
            assert queue.active_count == 1
            assert not any(
                "stage=driver.file-download " in record.message for record in caplog.records
            )
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.active_count == 0

    asyncio.run(scenario())
    record, profile = _profile(caplog)
    assert record.levelno == logging.DEBUG
    assert profile["phase"] == "processing"
    assert profile["processing_seconds"] >= 0.02
    assert profile["error_type"] == "CancelledError"
