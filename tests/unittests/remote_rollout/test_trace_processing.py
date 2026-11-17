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

"""Diagnostic snapshot validation must not recreate token objects in the client."""

import asyncio
import gzip
import json
import os
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.trace_processing import TraceSnapshotWriter, trace_snapshot_writer
from northrl.remote_rollout.wire import CompactRolloutTrace, dump_artifact, load_trace_gzip


def _payload(token=2, rollout_id="r"):
    return dump_artifact(
        CompactRolloutTrace.model_validate(
            {
                "schema_version": 2,
                "rollout_id": rollout_id,
                "segments": [
                    {
                        "base_segment": None,
                        "prefix_len": 0,
                        "suffix_tokens": [1, token],
                        "response_spans": [[1, 2]],
                        "response_log_probs": [-0.1],
                        "old_response_log_probs": {"ref": "response_log_probs"},
                    }
                ],
            }
        )
    )


def test_snapshots_validate_in_child_share_pool_and_join_on_last_close(tmp_path, monkeypatch):
    payload = _payload()
    parent_decode = Mock(side_effect=AssertionError("parent must not decode a periodic trace"))
    monkeypatch.setattr(e2b_executor, "load_trace_gzip", parent_decode)
    path = tmp_path / "trace.json.gz"
    sandbox = SimpleNamespace(
        files=SimpleNamespace(read=AsyncMock(return_value=bytearray(payload)))
    )

    async def scenario():
        async with trace_snapshot_writer(path, "r") as writer:
            state = asyncio.get_running_loop()._northrl_trace_snapshot_pool
            async with trace_snapshot_writer(tmp_path / "other.gz", "other"):
                assert asyncio.get_running_loop()._northrl_trace_snapshot_pool is state
                assert state.users == 2
                result = await e2b_executor._download_trace(
                    sandbox,
                    path,
                    log_context="test",
                    rollout_id="r",
                    quiet=True,
                    priority="background",
                    snapshot_writer=writer,
                )
                assert result is None
                child_pid = await asyncio.to_thread(state.executor.submit(os.getpid).result)
                assert child_pid != os.getpid()
                processes = list(state.executor._processes.values())
            assert state.users == 1
            assert any(process.is_alive() for process in processes)
        assert not hasattr(asyncio.get_running_loop(), "_northrl_trace_snapshot_pool")
        assert all(not process.is_alive() for process in processes)

    asyncio.run(scenario())
    parent_decode.assert_not_called()
    assert path.read_bytes() == payload
    assert load_trace_gzip(path.read_bytes()).rollout_id == "r"


@pytest.mark.parametrize("invalid", ["gzip", "token", "identity", "version"])
def test_bad_snapshot_keeps_last_valid_file_and_cache(tmp_path, invalid):
    path = tmp_path / "trace.gz"
    valid = _payload()
    if invalid == "gzip":
        bad = b"invalid gzip"
    else:
        data = json.loads(gzip.decompress(valid))
        if invalid == "token":
            data["segments"][0]["suffix_tokens"][-1] = -1
        elif invalid == "identity":
            data["rollout_id"] = "foreign"
        else:
            data["schema_version"] = 1
        bad = gzip.compress(json.dumps(data).encode())

    async def scenario():
        async with trace_snapshot_writer(path, "r") as writer:
            assert (await asyncio.to_thread(writer.save, valid))[2] == "saved"
            with pytest.raises(ValueError, match="Invalid partial trace snapshot") as error:
                await asyncio.to_thread(writer.save, bad)
            assert all(isinstance(arg, str) and len(arg) < 200 for arg in error.value.args)
            assert path.read_bytes() == valid
            assert (await asyncio.to_thread(writer.save, valid))[2] == "unchanged"
            changed = _payload(3)
            assert (await asyncio.to_thread(writer.save, changed))[2] == "saved"
            assert path.read_bytes() == changed

    asyncio.run(scenario())


def test_snapshot_cache_skips_identical_bytes_and_retries_failed_writes(tmp_path):
    completed = Future()
    completed.set_result((1.0, 2.0))
    failed = Future()
    failed.set_exception(OSError("disk unavailable"))
    executor = Mock()
    executor.submit.side_effect = [failed, completed, completed]
    writer = TraceSnapshotWriter(executor, tmp_path / "trace.gz", "r")
    payload = _payload()
    with pytest.raises(OSError, match="disk unavailable"):
        writer.save(payload)
    assert writer.save(payload) == (1.0, 2.0, "saved")
    assert writer.save(bytearray(payload)) == (0.0, 0.0, "unchanged")
    assert executor.submit.call_count == 2
    assert writer.save(_payload(3))[2] == "saved"
    assert executor.submit.call_count == 3


def test_periodic_cancellation_drains_snapshot_before_pool_closes(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    events = []
    path = tmp_path / "trace.gz"
    sandbox = SimpleNamespace(files=SimpleNamespace(read=AsyncMock(return_value=_payload())))

    def save(self, payload):
        started.set()
        assert release.wait(5)
        path.write_bytes(payload)
        events.append("saved")
        return 0.0, 0.0, "saved"

    monkeypatch.setattr(TraceSnapshotWriter, "save", save)
    monkeypatch.setattr(e2b_executor, "_TRACE_SYNC_INTERVAL_SECONDS", 0)

    async def scenario():
        task = asyncio.create_task(
            e2b_executor._sync_trace_periodically(sandbox, path, log_context="test", rollout_id="r")
        )
        try:
            async with asyncio.timeout(5):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert not path.exists()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        events.append("closed")
        assert not hasattr(asyncio.get_running_loop(), "_northrl_trace_snapshot_pool")

    asyncio.run(scenario())
    assert events == ["saved", "closed"]


def test_pool_closes_when_no_snapshot_is_submitted(tmp_path):
    async def scenario():
        async with trace_snapshot_writer(tmp_path / "unused.gz", "r"):
            pass
        assert not hasattr(asyncio.get_running_loop(), "_northrl_trace_snapshot_pool")

    asyncio.run(scenario())


def test_optional_pool_start_failure_does_not_fail_agent_outcome(tmp_path, monkeypatch, capsys):
    @asynccontextmanager
    async def unavailable(*args):
        raise OSError("cannot create child pool")
        yield  # pragma: no cover

    monkeypatch.setattr(e2b_executor, "trace_snapshot_writer", unavailable)
    asyncio.run(
        e2b_executor._sync_trace_periodically(
            None, tmp_path / "unused.gz", log_context="test", rollout_id="r"
        )
    )
    assert "cannot create child pool" in capsys.readouterr().err
