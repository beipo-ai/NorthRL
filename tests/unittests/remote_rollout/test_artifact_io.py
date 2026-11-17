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

"""Compressed artifact boundaries and asynchronous writer lifecycle regressions."""

import asyncio
import gzip
import json
import threading
from contextvars import ContextVar

import pytest

from northrl.remote_rollout import wire
from northrl.remote_rollout.contracts import AgentResult, RemoteRolloutTrace, RemoteSegment
from northrl.remote_rollout.wire import (
    CoalescingArtifactWriter,
    CompactAgentStageResult,
    CompactRolloutResult,
    dump_artifact,
    encode_trace,
    load_agent_stage_gzip,
    load_result_gzip,
    load_trace_gzip,
    run_artifact_io,
    write_artifact,
)


def _trace():
    return encode_trace(
        RemoteRolloutTrace(
            rollout_id="gzip-rollout",
            metadata={"task_id": "task", "unicode": "测试"},
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


def _result(revision=0):
    return CompactRolloutResult(schema_version=2, rollout_id="gzip-rollout", reward=float(revision))


def _stage():
    return CompactAgentStageResult(
        schema_version=2,
        rollout_id="gzip-rollout",
        attempt_id="attempt",
        workspace="/workspace",
        agent_result=AgentResult(),
        trace=_trace(),
    )


@pytest.mark.parametrize(
    "factory,loader",
    [
        (_trace, load_trace_gzip),
        (_result, load_result_gzip),
        (_stage, load_agent_stage_gzip),
    ],
)
def test_artifacts_round_trip_gzip_bytes_and_sdk_bytearray(factory, loader):
    model = factory()
    payload = dump_artifact(model)
    assert payload[:2] == b"\x1f\x8b"
    assert payload[4:8] == b"\0" * 4
    assert payload == dump_artifact(model)
    assert gzip.decompress(payload) == model.model_dump_json().encode()
    assert loader(payload) == model
    assert loader(bytearray(payload)) == model


@pytest.mark.parametrize("loader", [load_trace_gzip, load_result_gzip, load_agent_stage_gzip])
@pytest.mark.parametrize("payload", [b'{"schema_version":2}', b"not gzip", b"\x1f\x8b", "text"])
def test_artifact_loaders_reject_plain_and_bad_gzip(loader, payload):
    with pytest.raises(ValueError):
        loader(payload)


def test_artifact_loader_rejects_bad_crc_and_legacy_json():
    payload = bytearray(dump_artifact(_trace()))
    payload[-8] ^= 1
    with pytest.raises(ValueError, match="valid gzip"):
        load_trace_gzip(payload)
    legacy = _trace().model_dump(mode="json")
    legacy.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version"):
        load_trace_gzip(gzip.compress(json.dumps(legacy).encode()))


def test_atomic_write_preserves_previous_file_on_publication_failure(tmp_path, monkeypatch):
    path = tmp_path / "artifacts" / "result.json.gz"
    write_artifact(path, _result(1))
    before = path.read_bytes()
    replace = wire.os.replace

    def reject_replace(source, destination):
        assert destination == path
        assert path.read_bytes() == before
        assert load_result_gzip(source.read_bytes()).reward == 2
        raise OSError("publication failed")

    monkeypatch.setattr(wire.os, "replace", reject_replace)
    with pytest.raises(OSError, match="publication failed"):
        write_artifact(path, _result(2))
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]
    monkeypatch.setattr(wire.os, "replace", replace)
    write_artifact(path, _result(3))
    assert load_result_gzip(path.read_bytes()).reward == 3
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("parent_link", [True, False])
def test_atomic_write_refuses_symlinks(tmp_path, parent_link):
    target = tmp_path / "target"
    if parent_link:
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)
        path = link / "result.json.gz"
    else:
        target.write_bytes(b"untouched")
        path = tmp_path / "result.json.gz"
        path.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        write_artifact(path, _result())
    if not parent_link:
        assert target.read_bytes() == b"untouched"


async def _wait_for_thread_event(event):
    async with asyncio.timeout(3):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_coalescing_writer_keeps_one_latest_snapshot_and_does_not_block_loop(tmp_path):
    started = threading.Event()
    release = threading.Event()
    writes = []
    path = tmp_path / "result.json.gz"

    def slow_write(destination, snapshot):
        writes.append(snapshot.reward)
        if len(writes) == 1:
            started.set()
            assert release.wait(3)
        write_artifact(destination, snapshot)

    async def scenario():
        writer = CoalescingArtifactWriter(path, write=slow_write)
        try:
            writer.submit(_result(1))
            await _wait_for_thread_event(started)
            for revision in range(2, 10):
                writer.submit(_result(revision))
            # Reaching this await while publication is blocked proves off-loop work.
            await asyncio.sleep(0.01)
            assert writes == [1]
        finally:
            release.set()
            await writer.aclose()
        assert writes == [1, 9]
        assert not writer._thread.is_alive()
        assert load_result_gzip(path.read_bytes()).reward == 9
        with pytest.raises(RuntimeError, match="closed"):
            writer.submit(_result(10))

    asyncio.run(scenario())


def test_cancelled_close_drains_latest_snapshot_and_joins_worker(tmp_path):
    started = threading.Event()
    release = threading.Event()
    path = tmp_path / "result.json.gz"

    def slow_write(destination, snapshot):
        started.set()
        assert release.wait(3)
        write_artifact(destination, snapshot)

    async def scenario():
        writer = CoalescingArtifactWriter(path, write=slow_write)
        writer.submit(_result(1))
        await _wait_for_thread_event(started)
        writer.submit(_result(2))
        close = asyncio.create_task(writer.aclose())
        await asyncio.sleep(0)
        close.cancel()
        await asyncio.sleep(0.01)
        assert not close.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await close
        assert not writer._thread.is_alive()
        assert load_result_gzip(path.read_bytes()).reward == 2

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_writer_early_close_and_write_failure_leave_no_live_worker(tmp_path):
    async def scenario():
        empty = CoalescingArtifactWriter(tmp_path / "empty.json.gz")
        await empty.aclose()
        assert empty._thread is None

        def fail(*_args):
            raise OSError("disk failure")

        writer = CoalescingArtifactWriter(tmp_path / "bad.json.gz", write=fail)
        writer.submit(_result())
        with pytest.raises(OSError, match="disk failure"):
            await writer.aclose()
        assert not writer._thread.is_alive()

    asyncio.run(scenario())


def test_background_io_preserves_context_and_keyword_arguments():
    rollout_id = ContextVar("rollout_id", default="unset")

    def work(value, *, suffix):
        return rollout_id.get(), value + suffix

    async def scenario():
        rollout_id.set("active-rollout")
        assert await run_artifact_io(work, "a", suffix="b") == ("active-rollout", "ab")

    asyncio.run(scenario())


@pytest.mark.parametrize("worker_error", [False, True])
@pytest.mark.parametrize("cancel_all", [False, True])
def test_background_io_drains_before_propagating_cancellation(worker_error, cancel_all):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        assert release.wait(3)
        finished.set()
        if worker_error:
            raise OSError("worker failed after cancellation")
        return 42

    async def scenario():
        task = asyncio.create_task(run_artifact_io(work))
        await _wait_for_thread_event(started)
        if cancel_all:
            # Match asyncio runner shutdown: cancel every Task, including any
            # internal wrapper. The real thread must remain owned until it exits.
            for pending in asyncio.all_tasks():
                if pending is not asyncio.current_task():
                    pending.cancel()
        else:
            task.cancel()
        try:
            await asyncio.sleep(0.01)
            assert not task.done()
            assert not finished.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    try:
        asyncio.run(scenario())
    finally:
        release.set()
