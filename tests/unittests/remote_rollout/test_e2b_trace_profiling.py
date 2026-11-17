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

"""Trace diagnostics distinguish decoding from local filesystem work."""

import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.contracts import RemoteRolloutTrace, RemoteSegment
from northrl.remote_rollout.wire import dump_artifact, encode_trace
from northrl.utils import profiling


@pytest.mark.parametrize("failure", [None, "decode", "local_save", "profile"])
def test_trace_profile_separates_decode_and_local_save(monkeypatch, tmp_path, caplog, failure):
    caplog.set_level(logging.INFO, logger=profiling.__name__)
    trace = encode_trace(
        RemoteRolloutTrace(
            rollout_id="rollout-1",
            segments=[
                RemoteSegment(
                    tokens=[1, 2],
                    loss_mask=[0, 1],
                    log_probs=[0.0, -0.1],
                    old_log_probs=[0.0, -0.1],
                )
            ],
        )
    )
    payload = dump_artifact(trace)
    sandbox = SimpleNamespace(
        sandbox_id="sandbox-1", files=SimpleNamespace(read=AsyncMock(return_value=payload))
    )
    destination = tmp_path / "trace.json.gz"
    clock = [0.0]
    load = e2b_executor.load_trace_gzip
    save = e2b_executor._atomic_write_bytes
    profiles = Mock(
        side_effect=RuntimeError("diagnostics unavailable") if failure == "profile" else None
    )

    def timed_load(data):
        clock[0] += 2.0
        if failure == "decode":
            raise ValueError("invalid trace")
        return load(data)

    def timed_save(path, data):
        clock[0] += 3.0
        if failure == "local_save":
            raise OSError("local filesystem unavailable")
        return save(path, data)

    monkeypatch.setattr(
        e2b_executor,
        "time",
        SimpleNamespace(**{**vars(time), "perf_counter": lambda: clock[0]}),
    )
    monkeypatch.setattr(e2b_executor, "load_trace_gzip", timed_load)
    monkeypatch.setattr(e2b_executor, "_atomic_write_bytes", timed_save)
    monkeypatch.setattr(e2b_executor, "profile_event", profiles)

    result = asyncio.run(
        e2b_executor._download_trace(
            sandbox,
            destination,
            log_context="test",
            rollout_id="rollout-1",
            priority="background",
        )
    )

    events = [call for call in profiles.call_args_list if call.args[0] == "driver.trace-processing"]
    assert len(events) == 1
    final = events[0]
    assert final.kwargs["sandbox_id"] == "sandbox-1"
    assert final.kwargs["rollout_id"] == "rollout-1"
    assert "priority" not in final.kwargs
    assert "duration_seconds" not in final.kwargs
    assert "error_type" not in final.kwargs
    assert final.kwargs["decode_seconds"] == 2.0
    if failure == "decode":
        assert result is None
        assert final.args[1] == "failed"
        assert final.kwargs["level"] == logging.WARNING
        assert final.kwargs["local_save_seconds"] == 0.0
        assert final.kwargs["local_save_status"] == "not_started"
        assert not destination.exists()
    else:
        assert result == trace
        assert final.args[1] == "completed"
        assert final.kwargs["level"] == logging.INFO
        assert final.kwargs["local_save_seconds"] == 3.0
        if failure == "local_save":
            assert final.kwargs["local_save_status"] == "failed"
            assert not destination.exists()
        else:
            assert final.kwargs["local_save_status"] == "saved"
            assert destination.read_bytes() == payload
