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

"""Full packed results must survive IPC, cancellation, and worker failure."""

import asyncio
import gzip
import json
import logging
import os
import threading
import time
from array import array
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.contracts import RolloutInput
from northrl.remote_rollout.result_processing import (
    get_result_processor,
    result_processing,
)
from northrl.remote_rollout.wire import (
    CompactRolloutResult,
    decode_result,
    dump_artifact,
    run_blocking,
)
from northrl.utils import profiling


def _result(status="success", revision=0):
    segments = []
    if status != "env_error":
        for index in range(3):
            segments.append(
                {
                    "base_segment": index - 1 if index else None,
                    "prefix_len": 2 if index else 0,
                    "suffix_tokens": [11 + revision, 13, 17, 19],
                    "response_spans": [[2, 4]],
                    "response_log_probs": [-0.125, -0.25],
                    "old_response_log_probs": (
                        {"ref": "response_log_probs"} if index else [-0.5, -0.75]
                    ),
                    "segment_metrics": {"tool_results": index},
                }
            )
    return CompactRolloutResult(
        schema_version=2,
        rollout_id=f"r-{revision}",
        status=status,
        reward=0.5 if status == "success" else 0.0,
        segments=segments,
        error_type=None if status == "success" else "TestFailure",
        error_message=None if status == "success" else "错误 Ω",
        termination_reason="context_length_exceeded",
        length_details={"context_tokens": 6},
        message_format="openai",
        message_system="system Ω",
        tools=[{"name": "tool", "description": "测试"}],
        messages=[{"role": "assistant", "content": "result Ω"}],
        metrics={"revision": float(revision), "tool_calls": 2.0},
    )


@pytest.mark.parametrize("status", ["success", "task_failed", "env_error"])
def test_real_ipc_preserves_full_result_and_packed_buffer_bytes(status, monkeypatch):
    models = [_result(status, revision) for revision in range(8)]
    expected = [decode_result(model, packed=True) for model in models]
    parent_parse = Mock(side_effect=AssertionError("parent must not parse result token lists"))
    monkeypatch.setattr(e2b_executor, "load_result_gzip", parent_parse)

    async def scenario():
        async with result_processing("process"):
            processor = get_result_processor()
            for model, reference in zip(models, expected, strict=True):
                sandbox = SimpleNamespace(
                    files=SimpleNamespace(
                        read=AsyncMock(return_value=bytearray(dump_artifact(model)))
                    )
                )
                actual = await e2b_executor._read_result(
                    sandbox, RolloutInput(rollout_id=model.rollout_id, task={})
                )
                assert actual == reference
                for segment, original in zip(actual.segments, reference.segments, strict=True):
                    for name in ["tokens", "loss_mask", "log_probs", "old_log_probs"]:
                        value = getattr(segment, name)
                        assert isinstance(value, array)
                        assert value.typecode == getattr(original, name).typecode
                        assert value.tobytes() == getattr(original, name).tobytes()
            child_pid = await asyncio.to_thread(processor._pool().submit(os.getpid).result)
            assert child_pid != os.getpid()
            children = list(processor._pool()._processes.values())
        assert all(not child.is_alive() for child in children)
        assert get_result_processor() is None
        assert not hasattr(asyncio.get_running_loop(), "_northrl_result_processor")

    asyncio.run(scenario())
    parent_parse.assert_not_called()


@pytest.mark.parametrize("invalid", ["gzip", "identity", "token", "schema", "graph"])
def test_invalid_result_fails_closed_without_returning_decoded_error_inputs(invalid):
    model = _result()
    data = json.loads(model.model_dump_json())
    if invalid == "gzip":
        payload = b"not gzip"
    else:
        if invalid == "identity":
            data["rollout_id"] = "foreign"
        if invalid == "token":
            data["segments"][0]["suffix_tokens"][0] = -1
        if invalid == "schema":
            data["schema_version"] = 1
        if invalid == "graph":
            data["segments"][1]["base_segment"] = 2
        payload = gzip.compress(json.dumps(data).encode())

    async def scenario():
        async with result_processing("process"):
            with pytest.raises(ValueError, match="Invalid rollout result") as error:
                await asyncio.to_thread(get_result_processor().decode, payload, model.rollout_id)
            assert all(isinstance(arg, str) and len(arg) < 200 for arg in error.value.args)
            # A data rejection does not poison the resident worker.
            actual = await asyncio.to_thread(
                get_result_processor().decode, dump_artifact(model), model.rollout_id
            )
            assert actual == decode_result(model, packed=True)

    asyncio.run(scenario())


def test_concurrent_rollout_owners_share_one_worker_and_keep_result_identities():
    async def scenario():
        ready = asyncio.Event()
        owners = []

        async def rollout(revision):
            async with result_processing("process"):
                owners.append(get_result_processor())
                if len(owners) == 8:
                    ready.set()
                await ready.wait()
                model = _result(revision=revision)
                sandbox = SimpleNamespace(
                    files=SimpleNamespace(read=AsyncMock(return_value=dump_artifact(model)))
                )
                actual = await e2b_executor._read_result(
                    sandbox, RolloutInput(rollout_id=model.rollout_id, task={})
                )
                assert actual == decode_result(model, packed=True)
                return actual.rollout_id

        assert await asyncio.gather(*(rollout(i) for i in range(8))) == [f"r-{i}" for i in range(8)]
        assert all(owner is owners[0] for owner in owners)
        assert owners[0].users == 0
        assert owners[0]._executor is None
        assert not hasattr(asyncio.get_running_loop(), "_northrl_result_processor")

    asyncio.run(scenario())


def test_pool_is_lazy_shared_and_thread_scope_does_not_leak_process_mode():
    async def scenario():
        async with result_processing("thread"):
            assert get_result_processor() is None
            assert not hasattr(asyncio.get_running_loop(), "_northrl_result_processor")
        async with result_processing("process"):
            owner = get_result_processor()
            assert owner._executor is None
            async with result_processing("process"):
                assert get_result_processor() is owner
                assert owner.users == 2
            async with result_processing("thread"):
                assert get_result_processor() is None
            assert get_result_processor() is owner
            assert owner.users == 1
        assert not hasattr(asyncio.get_running_loop(), "_northrl_result_processor")

    asyncio.run(scenario())


@pytest.mark.parametrize("diagnostics", [False, True])
def test_child_failure_is_reported_and_next_recovery_gets_a_fresh_worker(diagnostics, monkeypatch):
    monkeypatch.setenv("NORTHRL_PROFILE_RESULTS", "1" if diagnostics else "0")

    async def scenario():
        async with result_processing("process"):
            processor = get_result_processor()
            pool = processor._pool()
            old_pid = await asyncio.to_thread(pool.submit(os.getpid).result)
            with pytest.raises(BrokenProcessPool):
                await asyncio.to_thread(pool.submit(os._exit, 7).result)
            model = _result()
            with pytest.raises(BrokenProcessPool):
                await asyncio.to_thread(processor.decode, dump_artifact(model), model.rollout_id)
            actual = await asyncio.to_thread(
                processor.decode, dump_artifact(model), model.rollout_id
            )
            assert actual == decode_result(model, packed=True)
            new_pid = await asyncio.to_thread(processor._pool().submit(os.getpid).result)
            assert new_pid != old_pid

    asyncio.run(scenario())


def test_downloaded_result_retains_slot_and_survives_expired_remote_budget():
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    downloaded = asyncio.Event()
    model = _result()

    def block(index):
        started[index].set()
        assert release.wait(5)

    async def read(*args, **kwargs):
        downloaded.set()
        return dump_artifact(model)

    async def scenario():
        async with result_processing("process"):
            blockers = [asyncio.create_task(run_blocking(block, i)) for i in range(2)]
            task = None
            try:
                async with asyncio.timeout(5):
                    while not all(event.is_set() for event in started):
                        await asyncio.sleep(0.001)
                token = e2b_executor._DOWNLOAD_BUDGET.set((1.0, time.monotonic() + 0.05))
                try:
                    task = asyncio.create_task(
                        e2b_executor._read_result(
                            SimpleNamespace(files=SimpleNamespace(read=read)),
                            RolloutInput(rollout_id=model.rollout_id, task={}),
                        )
                    )
                finally:
                    e2b_executor._DOWNLOAD_BUDGET.reset(token)
                await asyncio.wait_for(downloaded.wait(), 2)
                await asyncio.sleep(0.1)
                assert not task.done()
                assert e2b_executor.get_download_queue().active_count == 1
                release.set()
                assert (await asyncio.wait_for(task, 10)) == decode_result(model, packed=True)
            finally:
                release.set()
                await asyncio.gather(*blockers)
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_cancellation_drains_decode_before_owner_closes(monkeypatch):
    started, release = threading.Event(), threading.Event()
    events = []

    def decode(self, payload, rollout_id):
        started.set()
        assert release.wait(5)
        events.append("decoded")
        return decode_result(_result(), packed=True)

    from northrl.remote_rollout.result_processing import ResultProcessor

    monkeypatch.setattr(ResultProcessor, "decode", decode)

    async def scenario():
        async with result_processing("process"):
            sandbox = SimpleNamespace(files=SimpleNamespace(read=AsyncMock(return_value=b"raw")))
            task = asyncio.create_task(
                e2b_executor._read_result(sandbox, RolloutInput(rollout_id="r-0", task={}))
            )
            try:
                async with asyncio.timeout(5):
                    while not started.is_set():
                        await asyncio.sleep(0.001)
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done()
                assert e2b_executor.get_download_queue().active_count == 1
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        events.append("closed")

    asyncio.run(scenario())
    assert events == ["decoded", "closed"]


def test_public_rollout_keeps_processor_through_retries_and_restores_context(monkeypatch):
    async def run(*args, **kwargs):
        assert get_result_processor() is not None
        assert kwargs["template"] == "test"
        return decode_result(_result(), packed=True)

    monkeypatch.setattr(e2b_executor, "_run_rollout_with_retries", run)

    async def scenario():
        result = await e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="r-0", task={}),
            OmegaConf.create({"rollout": {"execution": {"result_processing": "process"}}}),
            gateway={},
            template="test",
            timeout_seconds=1,
        )
        assert result.rollout_id == "r-0"
        assert get_result_processor() is None
        assert not hasattr(asyncio.get_running_loop(), "_northrl_result_processor")

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["thread", "process"])
@pytest.mark.parametrize("enabled", [False, True])
def test_result_diagnostics_report_decode_counts_and_timings(mode, enabled, monkeypatch):
    monkeypatch.setenv("NORTHRL_PROFILE_RESULTS", "1" if enabled else "0")
    events = Mock(wraps=profiling.profile_event)
    monkeypatch.setattr(profiling, "profile_event", events)
    model = _result()
    payload = dump_artifact(model)
    reference = decode_result(model, packed=True)

    async def scenario():
        async with result_processing(mode):
            for _ in range(2):
                actual = await e2b_executor._read_result(
                    SimpleNamespace(files=SimpleNamespace(read=AsyncMock(return_value=payload))),
                    RolloutInput(rollout_id=model.rollout_id, task={}),
                )
                assert actual == reference

    asyncio.run(scenario())
    records = [
        call
        for call in events.call_args_list
        if call.args[0].startswith("driver.result-") and call.kwargs.get("level") == logging.INFO
    ]
    if not enabled:
        assert records == []
        return
    decoded = [call.kwargs for call in records if call.args[0] == "driver.result-decode"]
    assert len(decoded) == 2
    for fields in decoded:
        assert fields["compressed_bytes"] == len(payload)
        assert fields["expanded_token_positions"] == sum(len(s.tokens) for s in reference.segments)
        assert fields["segments"] == 3
        assert fields["mode"] == mode
        assert fields["level"] == logging.INFO
        prefix = "child_" if mode == "process" else ""
        for stage in ("parse", "restore"):
            assert fields[f"{prefix}{stage}_wall_seconds"] >= 0
            assert fields[f"{prefix}{stage}_cpu_seconds"] >= 0
        if mode == "process":
            assert fields["child_pid"] != os.getpid()
            assert fields["parent_round_trip_wall_seconds"] >= (
                fields["child_parse_wall_seconds"] + fields["child_restore_wall_seconds"]
            )
        assert not any("ipc" in name for name in fields)
        assert all(isinstance(value, (int, float, str)) for value in fields.values())
