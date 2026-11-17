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

"""Tests for the Driver-side E2B rollout executor."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from northrl.remote_rollout import e2b_executor, evaluator_logs, wire
from northrl.remote_rollout.contracts import (
    AgentResult,
    AgentStageResult,
    RemoteRolloutResult,
    RemoteRolloutTrace,
    RemoteSegment,
    RolloutInput,
)
from northrl.remote_rollout.wire import (
    CompactAgentStageResult,
    DownloadQueue,
    decode_trace,
    dump_artifact,
    encode_agent_stage,
    encode_result,
    encode_trace,
    get_download_queue,
    load_trace_gzip,
    run_blocking,
)

_SWEREBENCH_ENTRYPOINT = (
    Path(__file__).resolve().parents[3]
    / "northrl/rollout_entrypoints/swerebench_claude_code_rollout.py"
)


def _gateway():
    return {
        "base_url": "https://northgate.example",
        "model": "weaver-shared-sampler",
        "sampling_session_id": "shared-sampler",
        "weight_version": "actor-step-7",
    }


class _FakeFiles:
    def __init__(
        self,
        result_json: str | bytes,
        trace_json: str | bytes | None,
        result_error: Exception | None,
        trace_error: Exception | None,
        state_json: str | None = None,
    ) -> None:
        self._result_json = result_json
        self._state_json = state_json
        self._evaluation_state_json = state_json
        self._trace_json = trace_json
        self._result_error = result_error
        self._trace_error = trace_error
        self.writes: list[tuple[str, str | bytes]] = []
        self.reads: list[str] = []
        self.stage_json: str | bytes | None = None
        self.events: list[tuple[str, str]] = []

    async def write(self, path: str, data: str | bytes, *, user: str) -> None:
        assert user == "root"
        self.writes.append((path, data))
        self.events.append(("write", path))

    async def read(self, path: str, *, user, format="text", request_timeout=None):
        assert user == "root"
        if path in {
            e2b_executor._RESULT_PATH,
            e2b_executor._TRACE_PATH,
            e2b_executor._AGENT_STAGE_PATH,
        }:
            assert format == "bytes"
        self.reads.append(path)
        self.events.append(("read", path))
        if path == e2b_executor._REQUESTS_PATH:
            raise FileNotFoundError(path)
        if path == e2b_executor._EVALUATION_TERMINAL_PATH:
            raise FileNotFoundError(path)
        if path.startswith(e2b_executor._EVALUATOR_LOG_ROOT + "/"):
            raise FileNotFoundError(path)
        if path == e2b_executor._AGENT_STAGE_PATH:
            if self.stage_json is None:
                raise FileNotFoundError(path)
            return self.stage_json
        if path == e2b_executor._STATE_PATH:
            if self._state_json is None:
                raise FileNotFoundError(path)
            try:
                state = json.loads(self._state_json)
            except ValueError:
                return self._state_json
            if isinstance(state, dict):
                payload = json.loads(dict(self.writes)[e2b_executor._INPUT_PATH])
                state.setdefault("attempt_id", payload["attempt_id"])
            return json.dumps(state)
        if path == e2b_executor._TRACE_PATH:
            if self._trace_error is not None:
                raise self._trace_error
            if self._trace_json is None:
                raise FileNotFoundError(path)
            return self._trace_json
        if self._result_error is not None:
            raise self._result_error
        return self._result_json


class _FakeCommands:
    def __init__(
        self, error: BaseException | None, entrypoint_error: BaseException | None = None
    ) -> None:
        self._error = error
        self._entrypoint_error = entrypoint_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, command: str, **kwargs: Any) -> None:
        assert kwargs.get("user") == "root"
        self.calls.append((command, kwargs))
        self.files.events.append(("command", command))
        if "--phase evaluate" in command and self.files._evaluation_state_json is not None:
            self.files._state_json = self.files._evaluation_state_json
        if self._error is not None:
            raise self._error
        if (
            f"--phase {_FakeAsyncSandbox.error_phase}" in command
            and self._entrypoint_error is not None
        ):
            raise self._entrypoint_error
        if "--phase agent" in command:
            payload = json.loads(dict(self.files.writes)[e2b_executor._INPUT_PATH])
            stage = AgentStageResult(
                rollout_id=payload["input_data"]["rollout_id"],
                attempt_id=payload["attempt_id"],
                workspace="/task-workspace",
                agent_result=AgentResult(),
                trace=_trace(payload["input_data"]["rollout_id"]),
            )
            self.files.stage_json = dump_artifact(encode_agent_stage(stage))
            self.files._state_json = json.dumps(
                {
                    "rollout_id": stage.rollout_id,
                    "attempt_id": stage.attempt_id,
                    "phase": "agent_completed",
                }
            )


class _FakeSandbox:
    def __init__(
        self,
        result_json: str | bytes,
        trace_json: str | bytes | None,
        command_error: Exception | None,
        kill_error: Exception | None,
        result_error: Exception | None = None,
        trace_error: Exception | None = None,
        state_json: str | None = None,
        entrypoint_error: BaseException | None = None,
    ) -> None:
        self.files = _FakeFiles(result_json, trace_json, result_error, trace_error, state_json)
        self.commands = _FakeCommands(command_error, entrypoint_error)
        self.commands.files = self.files
        self.sandbox_id = "test-sandbox-id"
        self._kill_error = kill_error
        self.killed = False

    async def kill(self) -> None:
        self.killed = True
        if self._kill_error is not None:
            raise self._kill_error


class _FakeAsyncSandbox:
    result_json = ""
    command_error: BaseException | None = None
    entrypoint_error: BaseException | None = None
    error_phase = "agent"
    state_json: str | None = None
    trace_json: str | bytes | None = None
    kill_error: Exception | None = None
    result_error: Exception | None = None
    trace_error: Exception | None = None
    create_calls: list[dict[str, Any]] = []
    instances: list[_FakeSandbox] = []

    @classmethod
    async def create(cls, **kwargs: Any) -> _FakeSandbox:
        cls.create_calls.append(kwargs)
        sandbox = _FakeSandbox(
            cls.result_json,
            cls.trace_json,
            cls.command_error,
            cls.kill_error,
            cls.result_error,
            cls.trace_error,
            cls.state_json,
            cls.entrypoint_error,
        )
        cls.instances.append(sandbox)
        return sandbox


def _result(rollout_id: str) -> RemoteRolloutResult:
    return RemoteRolloutResult(
        rollout_id=rollout_id,
        segments=[
            RemoteSegment(
                tokens=[1, 2],
                loss_mask=[0, 1],
                log_probs=[0.0, -0.1],
                old_log_probs=[0.0, -0.2],
            )
        ],
        reward=1.0,
        metrics={"tests_passed": 1.0},
    )


def _trace(rollout_id: str) -> RemoteRolloutTrace:
    return RemoteRolloutTrace(
        rollout_id=rollout_id,
        segments=_result(rollout_id).segments,
        message_format="anthropic",
        tools=[
            {
                "name": "Read",
                "description": "Read a workspace file.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        messages=[
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": [{"type": "text", "text": "working"}]},
        ],
    )


def _prefix_trace(rollout_id: str) -> RemoteRolloutTrace:
    """Two independent training segments with a shared, masked prompt prefix."""
    prompt = list(range(128))
    segments = [
        RemoteSegment(
            tokens=[*prompt, 500],
            loss_mask=[*[0] * len(prompt), 1],
            log_probs=[*[0.0] * len(prompt), -0.1],
            old_log_probs=[*[0.0] * len(prompt), -0.1],
        ),
        RemoteSegment(
            tokens=[*prompt, 501, 502],
            loss_mask=[*[0] * len(prompt), 1, 1],
            log_probs=[*[0.0] * len(prompt), -0.2, -0.3],
            old_log_probs=[*[0.0] * len(prompt), -0.4, -0.5],
        ),
    ]
    return _trace(rollout_id).model_copy(
        update={
            "segments": segments,
            "metadata": {"task_id": "task-1", "group_id": "group-1"},
            "message_system": "system instructions",
            "metrics": {"tool_calls_emitted": 3.0},
        }
    )


def _config():
    return OmegaConf.create(
        {
            "models": {
                "actor": {
                    "name": "${oc.env:ACTOR_MODEL,actor-default}",
                    "tokenizer": "driver-tokenizer-path",
                }
            },
            "rollout": {
                "num_workers": 2,
                "max_concurrent_requests": 4,
                "sampling": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 20,
                    "min_p": 0.0,
                    "max_tokens": 512,
                },
                "parsers": {
                    "type": "builtin",
                    "reasoning": "think_tag",
                    "tool": "qwen25",
                },
                "type": "e2b",
                "execution": {
                    "entrypoint": {
                        "target": str(_SWEREBENCH_ENTRYPOINT),
                    },
                },
            },
            "backend": {
                "type": "weaver",
                "base_url": "https://weaver.example",
                "api_key": "config-key",
            },
            "training": {"weight_sync": {"freeze_inflight": True}},
            "algorithm": {"sampling_logprob_source": "inference"},
            "data": {
                "max_prompt_length": 1024,
                "max_sequence_length": 4096,
            },
            "runtime": {
                "logger": {
                    "feishu": {"url": "${oc.env:UNRELATED_SECRET,}"},
                }
            },
        }
    )


@pytest.fixture(autouse=True)
def _fake_e2b(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(_result("rollout-1")))
    _FakeAsyncSandbox.trace_json = None
    _FakeAsyncSandbox.command_error = None
    _FakeAsyncSandbox.entrypoint_error = None
    _FakeAsyncSandbox.error_phase = "agent"
    _FakeAsyncSandbox.state_json = None
    _FakeAsyncSandbox.kill_error = None
    _FakeAsyncSandbox.result_error = None
    _FakeAsyncSandbox.trace_error = None
    _FakeAsyncSandbox.create_calls.clear()
    _FakeAsyncSandbox.instances.clear()
    e2b_module = ModuleType("e2b")
    e2b_module.AsyncSandbox = _FakeAsyncSandbox
    monkeypatch.setitem(sys.modules, "e2b", e2b_module)
    monkeypatch.setenv("NORTHGATE_API_KEY", "northgate-key")
    monkeypatch.setenv("WEAVER_API_KEY", "host-weaver-secret")
    monkeypatch.setenv("WEAVER_BASE_URL", "https://host-weaver.example")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("NORTHRL_E2B_EVALUATOR_ENV", raising=False)
    monkeypatch.delenv("NORTHRL_E2B_AGENT_ENV", raising=False)


@pytest.fixture
def bounded_io(monkeypatch):
    """Exercise real retry loops with short deadlines in permanent-failure tests."""
    original_retry = e2b_executor._retry_io
    original_recover = e2b_executor._recover_trace
    original_observe = e2b_executor._wait_for_agent_completion

    def short_deadline(kwargs):
        kwargs = dict(kwargs)
        if kwargs.get("deadline") is not None:
            kwargs["deadline"] = min(kwargs["deadline"], e2b_executor.time.monotonic() + 0.04)
        kwargs["io_seconds"] = min(kwargs.get("io_seconds", 60), 0.01)
        return kwargs

    async def retry(*args, **kwargs):
        return await original_retry(*args, **short_deadline(kwargs))

    async def recover(*args, **kwargs):
        return await original_recover(*args, **short_deadline(kwargs))

    async def observe(*args, **kwargs):
        return await original_observe(*args, **short_deadline(kwargs))

    monkeypatch.setattr(e2b_executor, "_retry_io", retry)
    monkeypatch.setattr(e2b_executor, "_recover_trace", recover)
    monkeypatch.setattr(e2b_executor, "_wait_for_agent_completion", observe)
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_INITIAL_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_MAX_DELAY_SECONDS", 0.002)


@pytest.mark.parametrize("error", [ConnectionError("disconnected"), TimeoutError("slow read")])
@pytest.mark.usefixtures("bounded_io")
def test_transient_result_download_recovers_without_rerunning_agent(monkeypatch, error):
    original_read = _FakeFiles.read
    result_reads = 0

    async def read(self, path, *, user, format="text", request_timeout=None):
        nonlocal result_reads
        if path == e2b_executor._RESULT_PATH:
            result_reads += 1
            if result_reads <= 2:
                raise error
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    result = _run()

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert result.metrics["sandbox_retries"] == 0.0
    assert result_reads == 3
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.usefixtures("bounded_io")
def test_transient_evaluation_upload_recovers_in_original_sandbox(monkeypatch):
    original_write = _FakeFiles.write
    uploads = 0

    async def write(self, path, data, *, user):
        nonlocal uploads
        if path == e2b_executor._EVALUATION_ARCHIVE_PATH:
            assert self.stage_json is not None
            uploads += 1
            if uploads <= 2:
                raise ConnectionError("temporary upload failure")
        return await original_write(self, path, data, user=user)

    monkeypatch.setattr(_FakeFiles, "write", write)
    result = _run()

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert result.metrics["sandbox_retries"] == 0.0
    assert uploads == 3
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.usefixtures("bounded_io")
def test_disconnected_agent_observes_completion_without_dispatching_again(monkeypatch):
    original_run = _FakeCommands.run
    original_read = _FakeFiles.read
    state_reads = 0

    async def run(self, command, **kwargs):
        if "--phase evaluate" in command:
            assert state_reads >= 2
        await original_run(self, command, **kwargs)
        if "--phase agent" in command:
            raise ConnectionError("lost command stream after dispatch")

    async def read(self, path, *, user, format="text", request_timeout=None):
        nonlocal state_reads
        if path == e2b_executor._STATE_PATH:
            state_reads += 1
            self._state_json = _state("agent" if state_reads == 1 else "agent_completed")
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeCommands, "run", run)
    monkeypatch.setattr(_FakeFiles, "read", read)
    result = _run()

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.usefixtures("bounded_io")
def test_disconnected_evaluator_waits_for_its_original_result(monkeypatch):
    original_run = _FakeCommands.run
    original_read = _FakeFiles.read
    result_reads = 0

    async def run(self, command, **kwargs):
        await original_run(self, command, **kwargs)
        if "--phase evaluate" in command:
            raise ConnectionError("lost evaluator stream")

    async def read(self, path, *, user, format="text", request_timeout=None):
        nonlocal result_reads
        if path == e2b_executor._RESULT_PATH:
            result_reads += 1
            if result_reads == 1:
                raise FileNotFoundError("evaluator is still writing result")
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeCommands, "run", run)
    monkeypatch.setattr(_FakeFiles, "read", read)
    result = _run()

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert result_reads == 2
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.parametrize(
    "observation_error",
    [FileNotFoundError("result pending"), ConnectionError("result transport disconnected")],
)
def test_pending_evaluator_result_gets_independent_recovery_budget(monkeypatch, observation_error):
    original_run = _FakeCommands.run
    original_read = _FakeFiles.read
    original_retry = e2b_executor._retry_io
    observation_deadline = None
    recovery_reads = 0

    async def retry(call, **kwargs):
        nonlocal observation_deadline
        if kwargs["operation"] == "evaluation result observation":
            observation_deadline = kwargs["deadline"]
        return await original_retry(call, **kwargs)

    async def run(self, command, **kwargs):
        await original_run(self, command, **kwargs)
        if "--phase evaluate" in command:
            raise ConnectionError("evaluator stream disconnected")

    async def read(self, path, *, user, format="text", request_timeout=None):
        nonlocal recovery_reads
        if path == e2b_executor._RESULT_PATH:
            assert observation_deadline is not None
            if e2b_executor.time.monotonic() < observation_deadline:
                raise observation_error
            recovery_reads += 1
            if recovery_reads == 1:
                raise FileNotFoundError("runtime is still committing result")
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(e2b_executor, "_retry_io", retry)
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_INITIAL_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_MAX_DELAY_SECONDS", 0.02)
    monkeypatch.setattr(_FakeCommands, "run", run)
    monkeypatch.setattr(_FakeFiles, "read", read)
    config = _config()
    config.rollout.execution.evaluation_timeout_seconds = 1
    monkeypatch.setattr(e2b_executor.ExecutionBudgets, "recovery_seconds", 1)
    result = _run(config)

    recovered = _without_driver_runtime(result)
    recovered.metrics.pop("e2b_timed_out", None)
    assert recovered == _result("rollout-1")
    assert recovery_reads == 2
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.parametrize("status", ["success", "task_failed"])
def test_hung_trace_download_preserves_already_validated_result(monkeypatch, tmp_path, status):
    original_read = _FakeFiles.read
    expected = _result("rollout-1") if status == "success" else _failed("task_failed")
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(expected))
    trace_reads = 0
    trace_cancelled = 0

    async def read(self, path, *, user, format="text", request_timeout=None):
        nonlocal trace_reads, trace_cancelled
        if path == e2b_executor._TRACE_PATH:
            trace_reads += 1
            try:
                await asyncio.Event().wait()
            finally:
                trace_cancelled += 1
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    config = _config()
    monkeypatch.setattr(e2b_executor.ExecutionBudgets, "io_seconds", 1)
    monkeypatch.setattr(e2b_executor.ExecutionBudgets, "recovery_seconds", 1)
    result = _run(config, trace_output_path=tmp_path / "trace.json")

    assert _without_driver_runtime(result) == expected
    assert trace_reads == trace_cancelled == 1
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed


@pytest.mark.parametrize(
    ("status", "priority"),
    [("success", "background"), ("env_error", "background"), ("task_failed", "critical")],
)
def test_only_training_trace_recovery_uses_critical_priority(
    monkeypatch, tmp_path, status, priority
):
    original_download = e2b_executor._download_trace
    observed_priorities = []
    expected = _result("rollout-1") if status == "success" else _failed(status)
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(expected))
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    if status != "success":
        _FakeAsyncSandbox.entrypoint_error = RuntimeError("agent failed")

    async def download(*args, **kwargs):
        observed_priorities.append(kwargs.get("priority", "critical"))
        return await original_download(*args, **kwargs)

    monkeypatch.setattr(e2b_executor, "_download_trace", download)
    result = _run(trace_output_path=tmp_path / "trace.json.gz")

    assert observed_priorities == [priority]
    assert result.status == status
    if status == "task_failed":
        assert _dense_segments(result.segments) == _trace("rollout-1").segments
    else:
        assert _without_driver_runtime(result) == expected
    assert _FakeAsyncSandbox.instances[0].killed


def test_io_retries_share_one_deadline_without_resetting_it(monkeypatch):
    clock = [0.0]
    calls = []
    sleeps = []
    timeouts = []
    original_wait_for = asyncio.wait_for

    async def call():
        calls.append(clock[0])
        clock[0] += 2.0
        raise ConnectionError("still unavailable")

    async def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    async def wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(e2b_executor, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(
        e2b_executor,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": sleep, "wait_for": wait_for}),
    )
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_INITIAL_DELAY_SECONDS", 1.0)
    with pytest.raises(ConnectionError, match="still unavailable"):
        asyncio.run(
            e2b_executor._retry_io(
                call,
                deadline=5.0,
                io_seconds=4.0,
                operation="test download",
                log_context="test",
            )
        )
    assert calls == [0.0, 3.0]
    assert sleeps == [1.0]
    assert timeouts == [4.0, 2.0]
    assert clock[0] == 5.0


def test_sandbox_bootstrap_is_valid_posix_shell() -> None:
    completed = subprocess.run(
        ["sh", "-n", str(e2b_executor._BOOTSTRAP_SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_rollout_in_e2b_transfers_json_and_closes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACTOR_MODEL", "resolved-actor")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-send")
    input_data = RolloutInput(
        rollout_id="rollout-1",
        task={"instruction": "修复它"},
    )
    config = _config()

    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            input_data,
            config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
        )
    )

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert _FakeAsyncSandbox.create_calls == [
        {
            "template": "swebench-template",
            "timeout": 6930,
        }
    ]

    sandbox = _FakeAsyncSandbox.instances[0]
    writes = dict(sandbox.files.writes)
    source_archive = writes["/tmp/northrl-source.tar.gz"]
    raw_input = writes["/tmp/northrl-rollout-input.json"]
    assert isinstance(source_archive, bytes)
    assert isinstance(raw_input, str)
    with tarfile.open(fileobj=BytesIO(source_archive), mode="r:gz") as archive:
        source_names = archive.getnames()
    assert (
        "northrl-src/northrl/rollout_entrypoints/swerebench_claude_code_rollout.py" in source_names
    )
    assert e2b_executor._BOOTSTRAP_SCRIPT_ARCHIVE_PATH.as_posix() in source_names
    assert "northrl-src/entrypoint.py" in source_names
    assert all(
        name.startswith("northrl-src/northrl/") or name == "northrl-src/entrypoint.py"
        for name in source_names
    )
    assert all(
        name.endswith(".py") or name == e2b_executor._BOOTSTRAP_SCRIPT_ARCHIVE_PATH.as_posix()
        for name in source_names
    )
    assert [path for path in sandbox.files.reads if path != e2b_executor._METRICS_PATH] == [
        e2b_executor._STATE_PATH,
        e2b_executor._RESULT_PATH,
    ]
    assert len(sandbox.commands.calls) == 5
    assert sandbox.commands.calls[0] == (
        "tar --no-same-owner -xzf /tmp/northrl-source.tar.gz -C /opt",
        {
            "user": "root",
            "timeout": pytest.approx(600, abs=5),
            "on_stdout": e2b_executor._stream_e2b_stdout,
            "on_stderr": e2b_executor._stream_e2b_stderr,
        },
    )
    assert sandbox.commands.calls[1] == (
        e2b_executor._BOOTSTRAP_COMMAND,
        {
            "user": "root",
            "timeout": pytest.approx(600, abs=5),
            "request_timeout": 0,
            "on_stdout": e2b_executor._stream_e2b_stdout,
            "on_stderr": e2b_executor._stream_e2b_stderr,
        },
    )
    command, command_kwargs = sandbox.commands.calls[2]
    assert command.startswith("/opt/northrl-runtime/bin/python -I -c ")
    assert "--entrypoint /opt/northrl-src/entrypoint.py" in command
    assert "--input /tmp/northrl-rollout-input.json" in command
    assert "--output /opt/northrl-result/result.json.gz" in command
    assert "--trace-output /opt/northrl-result/trace.json.gz" in command
    assert command_kwargs == {
        "user": "root",
        "timeout": pytest.approx(4800, abs=5),
        "request_timeout": 0,
        "envs": {
            "PYTHONPATH": "/opt/northrl-src",
            "NORTHGATE_API_KEY": "northgate-key",
            "PYTHONSAFEPATH": "1",
        },
        "cwd": "/opt/northrl-src",
        "on_stdout": e2b_executor._stream_e2b_stdout,
        "on_stderr": e2b_executor._stream_e2b_stderr,
    }
    assert sandbox.killed

    payload = json.loads(raw_input)
    assert payload["input_data"] == input_data.model_dump(mode="json")
    assert payload["gateway"] == _gateway()
    assert "model_path" not in payload
    assert payload["config"] == {
        "models": {
            "actor": {
                "name": "resolved-actor",
            }
        },
        "rollout": {
            "num_workers": 2,
            "max_concurrent_requests": 4,
            "keep_sampling_mask": False,
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.0,
                "max_tokens": 512,
            },
            "execution": {
                "token_prefix_reuse": False,
                "preparation_timeout_seconds": 600,
                "agent_timeout_seconds": 3600,
                "agent_finalize_timeout_seconds": 600,
                "evaluation_timeout_seconds": 600,
            },
        },
        "algorithm": {"sampling_logprob_source": "inference"},
        "data": {
            "max_prompt_length": 1024,
            "max_sequence_length": 4096,
        },
    }
    assert config.backend.api_key == "config-key"
    assert config.training.weight_sync.freeze_inflight
    assert "config-key" not in raw_input
    assert "do-not-send" not in raw_input
    assert "host-weaver-secret" not in raw_input
    assert "host-weaver.example" not in raw_input


@pytest.mark.skipif(
    not hasattr(socket, "TCP_USER_TIMEOUT") or not Path("/proc/self/fd").is_dir(),
    reason="TCP_USER_TIMEOUT socket configuration requires Linux",
)
def test_set_e2b_tcp_user_timeout_on_established_connection() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        with socket.socket() as client:
            client.connect(("127.0.0.1", port))
            connection, _ = server.accept()
            with connection:
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 30_000)
                sandbox = SimpleNamespace(envd_api_url=f"http://127.0.0.1:{port}")

                e2b_executor._set_e2b_tcp_user_timeout(sandbox)

                assert (
                    client.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)
                    == e2b_executor._E2B_TCP_USER_TIMEOUT_MS
                )


@pytest.mark.skipif(
    not hasattr(socket, "TCP_USER_TIMEOUT") or not Path("/proc/self/fd").is_dir(),
    reason="file descriptor leak check requires Linux TCP_USER_TIMEOUT support",
)
def test_set_e2b_tcp_user_timeout_does_not_leak_non_socket_fds() -> None:
    with tempfile.TemporaryFile() as ordinary_file:
        assert ordinary_file.fileno() >= 0
        open_fds_before = len(os.listdir("/proc/self/fd"))

        e2b_executor._set_e2b_tcp_user_timeout(SimpleNamespace(envd_api_url="http://127.0.0.1:1"))

        assert len(os.listdir("/proc/self/fd")) == open_fds_before


@pytest.mark.parametrize("outcome", ["success", "cancel", "timeout"])
def test_slow_socket_setup_keeps_rollout_loop_responsive_and_cancellable(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    setup_threads = []

    def slow_setup(sandbox):
        setup_threads.append(threading.get_ident())
        entered.set()
        try:
            assert release.wait(5), "test did not release socket setup"
        finally:
            finished.set()

    monkeypatch.setattr(e2b_executor, "_set_e2b_tcp_user_timeout", slow_setup)
    if outcome == "timeout":
        clock = [0.0]
        monkeypatch.setattr(
            e2b_executor,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000.0 + clock[0]),
        )
        original_run = _FakeCommands.run

        async def extract_near_deadline(self, command, **kwargs):
            await original_run(self, command, **kwargs)
            # Only a fraction of the preparation budget remains for socket setup.
            clock[0] = 599.8

        monkeypatch.setattr(_FakeCommands, "run", extract_near_deadline)

    async def scenario():
        loop_thread = threading.get_ident()
        task = asyncio.create_task(
            e2b_executor._run_rollout_in_e2b_once(
                RolloutInput(rollout_id="rollout-1", task={}),
                _config(),
                gateway=_gateway(),
                template="swebench-template",
                timeout_seconds=3600,
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2), "socket setup did not start"
            assert not task.done()

            async def unrelated_network_work():
                await asyncio.sleep(0)
                return threading.get_ident()

            assert await asyncio.wait_for(unrelated_network_work(), 1) == loop_thread
            assert len(setup_threads) == 1
            assert setup_threads[0] != loop_thread
            assert not finished.is_set()
            if outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert _FakeAsyncSandbox.instances[0].killed
                assert not finished.is_set()
            elif outcome == "timeout":
                result = await asyncio.wait_for(task, 2)
                assert result.status == "env_error"
                assert result.error_type == "TimeoutError"
                assert _FakeAsyncSandbox.instances[0].killed
                assert not finished.is_set()
                assert len(_FakeAsyncSandbox.instances[0].commands.calls) == 1
            else:
                release.set()
                assert _without_driver_runtime(await task) == _result("rollout-1")
        finally:
            release.set()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert await asyncio.to_thread(finished.wait, 2)

    asyncio.run(scenario())
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed


@pytest.mark.parametrize("processing", ["thread", "process"])
def test_run_rollout_in_e2b_downloads_final_trace_on_success(tmp_path: Path, processing) -> None:
    expected_trace = _trace("rollout-1")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(expected_trace))
    trace_output = tmp_path / "rollout-trace.json"
    config = _config()
    config.rollout.execution.result_processing = processing

    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
            trace_output_path=trace_output,
        )
    )

    assert _without_driver_runtime(result) == _result("rollout-1")
    sandbox = _FakeAsyncSandbox.instances[-1]
    assert [path for path in sandbox.files.reads if path != e2b_executor._METRICS_PATH] == [
        e2b_executor._STATE_PATH,
        "/opt/northrl-result/result.json.gz",
        "/opt/northrl-result/trace.json.gz",
        e2b_executor._REQUESTS_PATH,
        f"{e2b_executor._EVALUATOR_LOG_ROOT}/status.json",
    ]
    assert decode_trace(load_trace_gzip(trace_output.read_bytes())) == expected_trace


@pytest.mark.parametrize("remaining_seconds", [900, 30])
@pytest.mark.parametrize("recovering", [False, True])
def test_result_download_extends_driver_and_sdk_timeouts_within_deadline(
    monkeypatch, remaining_seconds, recovering
):
    expected = _result("rollout-1")
    driver_timeouts = []
    sdk_timeouts = []
    original_wait_for = asyncio.wait_for

    async def read(path, *, user, format, request_timeout=None):
        assert path == e2b_executor._RESULT_PATH
        assert user == "root"
        assert format == "bytes"
        sdk_timeouts.append(request_timeout)
        return dump_artifact(encode_result(expected))

    async def wait_for(awaitable, timeout):
        if getattr(awaitable, "cr_code", None) is read.__code__:
            driver_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(
        e2b_executor,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "wait_for": wait_for}),
    )
    sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
    input_data = RolloutInput(rollout_id="rollout-1", task={})

    async def run():
        deadline = e2b_executor.time.monotonic() + remaining_seconds
        if recovering:
            return await e2b_executor._command_failure_result(
                sandbox,
                input_data,
                ConnectionError("evaluator disconnected"),
                trace=None,
                log_context="test",
                deadline=deadline,
                io_seconds=60,
            )
        return await e2b_executor._retry_io(
            lambda: e2b_executor._read_result(sandbox, input_data),
            deadline=deadline,
            io_seconds=60,
            log_context="test",
            operation="result download",
            queued_download=True,
        )

    assert _without_driver_runtime(asyncio.run(run())) == expected
    assert driver_timeouts == sdk_timeouts
    assert len(driver_timeouts) == 1
    limit = min(120, remaining_seconds)
    assert limit - 1 < driver_timeouts[0] <= limit


def test_result_download_reconstructs_dense_training_data_and_outcome():
    trace = _prefix_trace("rollout-1")
    expected = RemoteRolloutResult(
        rollout_id=trace.rollout_id,
        segments=trace.segments,
        message_format=trace.message_format,
        message_system=trace.message_system,
        tools=trace.tools,
        messages=trace.messages,
        metrics={**trace.metrics, "tests_passed": 1.0},
        reward=0.75,
        termination_reason="context_length_exceeded",
        length_details={"prompt_tokens": 128},
    )
    sandbox = _FakeSandbox(dump_artifact(encode_result(expected)), None, None, None)

    actual = asyncio.run(
        e2b_executor._read_result(sandbox, RolloutInput(rollout_id="rollout-1", task={}))
    )

    assert _without_driver_runtime(actual) == expected
    assert all(isinstance(segment, RemoteSegment) for segment in actual.segments)


def test_agent_completion_verification_reads_only_small_commit_record():
    read = AsyncMock(return_value=_state("agent_completed", attempt_id="attempt-1"))
    sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
    actual = asyncio.run(
        e2b_executor._verify_agent_completion(
            sandbox,
            RolloutInput(rollout_id="rollout-1", task={}),
            "attempt-1",
            deadline=e2b_executor.time.monotonic() + 1,
            io_seconds=1,
            log_context="test",
        )
    )
    assert actual is None
    read.assert_awaited_once_with(e2b_executor._STATE_PATH, user="root")


def test_run_rollout_in_e2b_uses_absolute_script_path() -> None:
    config = _config()
    config.rollout.execution.entrypoint = {
        "target": str(_SWEREBENCH_ENTRYPOINT),
    }
    assert e2b_executor._resolve_entrypoint(config).script_path == _SWEREBENCH_ENTRYPOINT

    asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
        )
    )

    command = _FakeAsyncSandbox.instances[0].commands.calls[2][0]
    assert command.startswith("/opt/northrl-runtime/bin/python -I -c ")
    assert "--entrypoint /opt/northrl-src/entrypoint.py" in command
    assert "--entrypoint-kind" not in command
    assert "--entrypoint-target" not in command
    assert "--input /tmp/northrl-rollout-input.json" in command
    assert "--output /opt/northrl-result/result.json.gz" in command
    assert "--trace-output /opt/northrl-result/trace.json.gz" in command


@pytest.mark.parametrize("timeout_type", ["httpx", "httpcore", "builtin", "wrapped"])
@pytest.mark.usefixtures("bounded_io")
def test_completion_record_read_timeout_warns_and_evaluates_after_bounded_retry(
    monkeypatch, caplog, timeout_type
) -> None:
    import httpcore
    import httpx

    error = {
        "httpx": httpx.ReadTimeout("completion record read timed out"),
        "httpcore": httpcore.ReadTimeout("completion record read timed out"),
        "builtin": TimeoutError("completion record read timed out"),
        "wrapped": RuntimeError("completion record read failed"),
    }[timeout_type]
    if timeout_type == "wrapped":
        error.__cause__ = httpx.ReadTimeout("completion record read timed out")
    original_read = _FakeFiles.read

    async def read(self, path, *, user, format="text", request_timeout=None):
        if path == e2b_executor._STATE_PATH:
            raise error
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    events = []
    with caplog.at_level(logging.WARNING):
        result = _run(event_callback=events.append)

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._EVALUATION_ARCHIVE_PATH in dict(sandbox.files.writes)
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed
    finished = [event for event in events if event["kind"] == "attempt_finished"]
    assert len(finished) == 1
    assert finished[0]["status"] == "success"
    warning = next(
        record for record in caplog.records if "agent completion record" in record.getMessage()
    )
    assert warning.levelno == logging.WARNING
    assert "continuing to evaluation with sandbox-local validation" in warning.getMessage()
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_completion_record_permission_error_still_prevents_evaluation(monkeypatch) -> None:
    error = PermissionError("denied")
    original_read = _FakeFiles.read

    async def read(self, path, *, user, format="text", request_timeout=None):
        if path == e2b_executor._STATE_PATH:
            raise error
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    config = _config()
    config.rollout.execution.env_error_retries = 0
    result = _run(config)

    assert result.status == "env_error"
    assert result.error_type == type(error).__name__
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._EVALUATION_ARCHIVE_PATH not in dict(sandbox.files.writes)
    assert not any("--phase evaluate" in command for command, _ in sandbox.commands.calls)
    assert sandbox.killed


@pytest.mark.parametrize("error_type", ["connection", "httpx", "httpcore"])
@pytest.mark.usefixtures("bounded_io")
def test_completion_record_transient_read_errors_retry_without_rerunning_agent(
    monkeypatch, error_type
):
    import httpcore
    import httpx

    error = {
        "connection": ConnectionError("connection reset"),
        "httpx": httpx.ReadError("connection reset"),
        "httpcore": httpcore.ReadError("connection reset"),
    }[error_type]
    original_read = _FakeFiles.read
    completion_reads = 0

    async def read(self, path, **kwargs):
        nonlocal completion_reads
        if path == e2b_executor._STATE_PATH:
            completion_reads += 1
            if completion_reads < 3:
                raise error
        return await original_read(self, path, **kwargs)

    monkeypatch.setattr(_FakeFiles, "read", read)
    result = _run()

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert completion_reads == 3
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._AGENT_STAGE_PATH not in sandbox.files.reads
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.parametrize("deadline_seconds,io_seconds", [(600, 0.02), (0.02, 60)])
def test_completion_record_verification_respects_the_shorter_budget(
    caplog, deadline_seconds, io_seconds
):
    async def stalled_read(*_args, **_kwargs):
        await asyncio.Event().wait()

    read = AsyncMock(side_effect=stalled_read)
    sandbox = SimpleNamespace(files=SimpleNamespace(read=read))

    async def scenario():
        await asyncio.wait_for(
            e2b_executor._verify_agent_completion(
                sandbox,
                RolloutInput(rollout_id="rollout-1", task={}),
                "attempt-1",
                deadline=e2b_executor.time.monotonic() + deadline_seconds,
                io_seconds=io_seconds,
                log_context="test",
            ),
            timeout=1,
        )

    asyncio.run(scenario())
    assert any(
        "continuing to evaluation with sandbox-local validation" in m for m in caplog.messages
    )
    read.assert_awaited_once_with(e2b_executor._STATE_PATH, user="root")


@pytest.mark.usefixtures("bounded_io")
def test_completion_record_corruption_after_transport_retry_is_not_ignored():
    read = AsyncMock(side_effect=[TimeoutError("temporary timeout"), "not JSON"])
    sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(
            e2b_executor._verify_agent_completion(
                sandbox,
                RolloutInput(rollout_id="rollout-1", task={}),
                "attempt-1",
                deadline=e2b_executor.time.monotonic() + 1,
                io_seconds=1,
                log_context="test",
            )
        )
    assert read.await_count == 2


def test_completion_record_read_cancellation_still_closes_sandbox(monkeypatch, caplog) -> None:
    original_read = _FakeFiles.read

    async def read(self, path, *, user, format="text", request_timeout=None):
        if path == e2b_executor._STATE_PATH:
            raise asyncio.CancelledError()
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    with pytest.raises(asyncio.CancelledError):
        _run()

    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._EVALUATION_ARCHIVE_PATH not in dict(sandbox.files.writes)
    assert sandbox.killed
    assert not any(
        "agent completion record read timed out" in message for message in caplog.messages
    )


def test_completion_record_verification_deadline_allows_local_validation(
    monkeypatch, caplog
) -> None:
    original_verify = e2b_executor._verify_agent_completion
    original_read = _FakeFiles.read

    async def read(self, path, *, user, format="text", request_timeout=None):
        if path == e2b_executor._STATE_PATH:
            await asyncio.Event().wait()
        return await original_read(self, path, user=user, format=format)

    async def verify(*args, **kwargs):
        kwargs["deadline"] = min(kwargs["deadline"], e2b_executor.time.monotonic() + 0.01)
        return await original_verify(*args, **kwargs)

    monkeypatch.setattr(_FakeFiles, "read", read)
    monkeypatch.setattr(e2b_executor, "_verify_agent_completion", verify)
    config = _config()
    config.rollout.execution.env_error_retries = 0
    result = _run(config)

    assert _without_driver_runtime(result) == _result("rollout-1")
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._EVALUATION_ARCHIVE_PATH in dict(sandbox.files.writes)
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed
    assert (
        sum("agent completion record read timed out" in message for message in caplog.messages) == 1
    )


@pytest.mark.parametrize("prepacked", [False, True])
@pytest.mark.parametrize(
    ("task_fields", "metadata"),
    [
        ({"task_id": "wtforms__wtforms-614"}, {}),
        ({}, {"task_id": "wtforms__wtforms-614"}),
        ({"task_id": "wrong-legacy-id"}, {"task_id": "wtforms__wtforms-614"}),
    ],
)
def test_evaluation_assets_upload_only_after_agent_completion(
    tmp_path: Path, task_fields: dict[str, Any], metadata: dict[str, Any], prepacked: bool
) -> None:
    task_files_root = tmp_path / "tasks"
    task_dir = task_files_root / "wtforms__wtforms-614"
    (task_dir / "tests").mkdir(parents=True)
    expected_assets = {
        "run-tests.sh": b"#!/bin/bash\necho PASSED\n",
        "tests/config.json": b'{"instance_id":"wtforms__wtforms-614"}\n',
        "log_parsers.py": b"def parse_log(value): return value\n",
        "swe_constants.py": b"MAP = {}\n",
    }
    for relative_path, content in expected_assets.items():
        (task_dir / relative_path).write_bytes(content)

    config = _config()
    config.rollout.execution.entrypoint = {
        "target": str(_SWEREBENCH_ENTRYPOINT),
        "task_files_root": str(task_files_root),
    }
    artifact_options = {}
    if prepacked:
        artifact_options["prepared_artifacts"] = e2b_executor.prepare_e2b_artifacts(
            config, ["wtforms__wtforms-614"]
        )

    asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(
                rollout_id="rollout-1",
                task={"instruction": "fix it", **task_fields},
                metadata=metadata,
            ),
            config,
            gateway=_gateway(),
            template="opaque-template-id",
            timeout_seconds=3600,
            **artifact_options,
        )
    )

    sandbox = _FakeAsyncSandbox.instances[0]
    command = sandbox.commands.calls[2][0]
    assert command.startswith("/opt/northrl-runtime/bin/python -I -c ")
    assert "--entrypoint /opt/northrl-src/entrypoint.py" in command
    assert "--input /tmp/northrl-rollout-input.json" in command
    assert "--output /opt/northrl-result/result.json.gz" in command
    assert "--trace-output /opt/northrl-result/trace.json.gz" in command
    source_archive = dict(sandbox.files.writes)[e2b_executor._SOURCE_ARCHIVE_PATH]
    assert isinstance(source_archive, bytes)
    with tarfile.open(fileobj=BytesIO(source_archive), mode="r:gz") as archive:
        assert (
            "northrl-src/northrl/rollout_entrypoints/swerebench_claude_code_rollout.py"
            in archive.getnames()
        )
        assert not any("northrl-assets" in name for name in archive.getnames())
        assert not any(name.endswith("/run-tests.sh") for name in archive.getnames())
    evaluation_archive = dict(sandbox.files.writes)[e2b_executor._EVALUATION_ARCHIVE_PATH]
    with tarfile.open(fileobj=BytesIO(evaluation_archive), mode="r:gz") as archive:
        for relative_path, expected_content in expected_assets.items():
            member = archive.extractfile(relative_path)
            assert member is not None
            content = member.read()
            if relative_path == "run-tests.sh":
                restored = tmp_path / "unpacked-run-tests.sh"
                restored.write_bytes(content)
                evaluator_logs._restore_script(restored)
                content = restored.read_bytes()
            assert content == expected_content
        assert all(member.mode == (0o700 if member.isdir() else 0o600) for member in archive)
    events = sandbox.files.events
    agent_index = next(i for i, event in enumerate(events) if "--phase agent" in event[1])
    stage_index = events.index(("read", e2b_executor._STATE_PATH))
    upload_index = events.index(("write", e2b_executor._EVALUATION_ARCHIVE_PATH))
    evaluation_index = next(i for i, event in enumerate(events) if "--phase evaluate" in event[1])
    assert agent_index < stage_index < upload_index < evaluation_index
    assert len(_FakeAsyncSandbox.instances) == 1
    evaluation_environment = sandbox.commands.calls[-1][1]["envs"]
    assert "WEAVER_API_KEY" not in evaluation_environment
    assert "WEAVER_BASE_URL" not in evaluation_environment


@pytest.mark.parametrize("task_id", [None, "", "  ", 1, "../escape", "x" * 256])
def test_invalid_metadata_task_id_does_not_fall_back_to_task(tmp_path: Path, task_id: Any) -> None:
    task_files_root = tmp_path / "tasks"
    (task_files_root / "legacy-id").mkdir(parents=True)
    config = _config()
    config.rollout.execution.env_error_retries = 0
    config.rollout.execution.entrypoint.task_files_root = str(task_files_root)

    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(
                rollout_id="rollout-1",
                task={"instruction": "fix it", "task_id": "legacy-id"},
                metadata={"task_id": task_id},
            ),
            config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
        )
    )

    assert result.status == "env_error"
    assert result.error_type == "ValueError"
    assert "metadata.task_id" in result.error_message
    assert not _FakeAsyncSandbox.create_calls


def test_build_source_archive_rejects_task_asset_symlink(tmp_path: Path) -> None:
    task_files_root = tmp_path / "tasks"
    task_dir = task_files_root / "wtforms__wtforms-614"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "run-tests.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_dir / "tests/config.json").write_text(
        '{"instance_id":"wtforms__wtforms-614"}',
        encoding="utf-8",
    )
    external_parser = tmp_path / "external-parser.py"
    external_parser.write_text("PARSERS = {}\n", encoding="utf-8")
    (task_dir / "log_parsers.py").symlink_to(external_parser)
    (task_dir / "swe_constants.py").write_text("MAP = {}\n", encoding="utf-8")
    entrypoint = e2b_executor._EntrypointSpec(
        script_path=_SWEREBENCH_ENTRYPOINT,
        task_files_root=task_files_root,
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        e2b_executor._build_evaluation_archive(
            entrypoint,
            task_id="wtforms__wtforms-614",
        )


def test_evaluation_archive_preserves_task_specific_structure(tmp_path: Path) -> None:
    task_dir = tmp_path / "coding-task-42"
    (task_dir / "private-tests").mkdir(parents=True)
    (task_dir / "private-tests/expected.txt").write_text("task-specific answer")
    (task_dir / "test.sh").write_text("#!/bin/sh\n")
    (task_dir / "test.sh").chmod(0o777)
    entrypoint = e2b_executor._EntrypointSpec(
        script_path=tmp_path / "my_task_rollout.py", task_files_root=tmp_path
    )
    data = e2b_executor._build_evaluation_archive(entrypoint, task_id="coding-task-42")
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
        assert archive.getnames() == ["private-tests", "private-tests/expected.txt", "test.sh"]
        assert archive.extractfile("private-tests/expected.txt").read() == b"task-specific answer"
        assert archive.getmember("test.sh").mode == 0o700


@pytest.mark.parametrize("task_id", [None, "../escape", "owner__repo/escape", "x" * 256])
def test_build_source_archive_rejects_invalid_task_id(
    tmp_path: Path,
    task_id: str | None,
) -> None:
    task_files_root = tmp_path / "tasks"
    task_files_root.mkdir()
    entrypoint = e2b_executor._EntrypointSpec(
        script_path=_SWEREBENCH_ENTRYPOINT,
        task_files_root=task_files_root,
    )

    with pytest.raises(ValueError, match="metadata.task_id"):
        e2b_executor._build_evaluation_archive(entrypoint, task_id=task_id)


def test_run_rollout_in_e2b_uploads_configured_script_entrypoint(tmp_path: Path) -> None:
    script = tmp_path / "openhands rollout.py"
    script_source = "print('external entrypoint')\n"
    script.write_text(script_source, encoding="utf-8")
    config = _config()
    config.rollout.execution.entrypoint = {
        "target": str(script),
    }

    asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
        )
    )

    sandbox = _FakeAsyncSandbox.instances[0]
    source_archive = dict(sandbox.files.writes)[e2b_executor._SOURCE_ARCHIVE_PATH]
    assert isinstance(source_archive, bytes)
    with tarfile.open(fileobj=BytesIO(source_archive), mode="r:gz") as archive:
        member = archive.extractfile("northrl-src/entrypoint.py")
        assert member is not None
        assert member.read().decode("utf-8") == script_source
    command = sandbox.commands.calls[2][0]
    assert command.startswith("/opt/northrl-runtime/bin/python -I -c ")
    assert "/opt/northrl-src/entrypoint.py" in command
    assert "--input /tmp/northrl-rollout-input.json" in command
    assert "--output /opt/northrl-result/result.json.gz" in command
    assert "--trace-output /opt/northrl-result/trace.json.gz" in command


def _dense_segments(segments):
    """Compare sequence values independently of the driver's packed storage."""
    return [
        segment.model_copy(
            update={
                name: list(getattr(segment, name))
                for name in ("tokens", "loss_mask", "log_probs", "old_log_probs")
            }
        )
        for segment in segments
    ]


def _without_driver_runtime(result: RemoteRolloutResult) -> RemoteRolloutResult:
    """Compare task outcomes separately from wallclock metadata tested explicitly."""
    return result.model_copy(
        update={
            "segments": _dense_segments(result.segments),
            "metrics": {
                key: value
                for key, value in result.metrics.items()
                if key
                not in {
                    "sandbox_retries",
                    "rollout_started_at",
                    "rollout_finished_at",
                    "rollout_duration_seconds",
                }
            },
        }
    )


def _run(
    config=None, *, trace_output_path=None, event_callback=None, **kwargs
) -> RemoteRolloutResult:
    return asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            _config() if config is None else config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
            trace_output_path=trace_output_path,
            event_callback=event_callback,
            **kwargs,
        )
    )


def test_default_artifact_preparation_scans_task_directories_and_keeps_assets_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "northrl"
    task_files_root = package / "evaluation"
    expected_assets = {
        "task-1": {"nested/check.py": b"EXPECTED = 'first'\n", "test.sh": b"echo first\n"},
        "task_2": {"answer.py": b"EXPECTED = 'second'\n"},
    }
    for task_id, assets in expected_assets.items():
        for relative_path, content in assets.items():
            path = task_files_root / task_id / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    (task_files_root / "README.md").write_text("task root documentation", encoding="utf-8")
    (task_files_root / "latest").symlink_to(task_files_root / "task-1", target_is_directory=True)
    hidden_directory = task_files_root / ".metadata"
    hidden_directory.mkdir()
    (hidden_directory / "private.py").write_text("SECRET = 'metadata'\n", encoding="utf-8")
    script = package / "hook.py"
    script.write_text("def run(): pass\n", encoding="utf-8")
    (package / "public.py").write_text("PUBLIC = True\n", encoding="utf-8")
    config = _config()
    config.rollout.execution.entrypoint = {
        "target": str(script),
        "task_files_root": str(task_files_root),
    }
    monkeypatch.setattr(e2b_executor, "_PACKAGE_ROOT", package)

    artifacts = e2b_executor.prepare_e2b_artifacts(config)

    assert set(artifacts.evaluation_archives) == set(expected_assets)
    for task_id, assets in expected_assets.items():
        with tarfile.open(
            fileobj=BytesIO(artifacts.evaluation_archives[task_id]), mode="r:gz"
        ) as archive:
            contents = {
                member.name: archive.extractfile(member).read()
                for member in archive
                if member.isfile()
            }
        assert contents == assets
    with tarfile.open(fileobj=BytesIO(artifacts.source_archive), mode="r:gz") as archive:
        assert "northrl-src/northrl/public.py" in archive.getnames()
        assert "northrl-src/entrypoint.py" in archive.getnames()
        assert not any("/evaluation/" in name for name in archive.getnames())


def test_artifact_preparation_logs_bounded_progress_and_final_totals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config()
    config.rollout.execution.entrypoint.task_files_root = str(tmp_path)
    elapsed = [0.0]

    def build_evaluation_archive(entrypoint, *, task_id):
        elapsed[0] += 0.25
        return b"archive"

    monkeypatch.setattr(e2b_executor, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    monkeypatch.setattr(e2b_executor, "_build_source_archive", Mock(return_value=b"source"))
    monkeypatch.setattr(e2b_executor, "_build_evaluation_archive", build_evaluation_archive)

    with caplog.at_level(logging.INFO, logger=e2b_executor.__name__):
        e2b_executor.prepare_e2b_artifacts(config, [f"task-{index}" for index in range(100)])

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 6  # Start, four time-bounded updates, completion.
    assert all(f"task_files_root={tmp_path}" in message for message in messages)
    assert "tasks=100" in messages[0]
    for completed, message in zip((20, 40, 60, 80), messages[1:-1]):
        assert f"tasks={completed}/100" in message
        assert f"archive_bytes={6 + completed * 7}" in message
    assert "Prepared E2B artifacts:" in messages[-1]
    assert "tasks=100 source_bytes=6 evaluation_bytes=700 elapsed=25.0s" in messages[-1]


def test_prepared_artifacts_are_reused_across_tasks_and_environment_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_files_root = tmp_path / "tasks"
    for task_id in ("task-1", "task-2"):
        task_dir = task_files_root / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "expected.txt").write_text(task_id, encoding="utf-8")
    chat_template = tmp_path / "chat.jinja"
    chat_template.write_text("original chat template", encoding="utf-8")
    config = _config()
    config.rollout.execution.entrypoint.task_files_root = str(task_files_root)
    config.rollout.execution.chat_template = str(chat_template)
    source_builder = Mock(wraps=e2b_executor._build_source_archive)
    evaluation_builder = Mock(wraps=e2b_executor._build_evaluation_archive)
    monkeypatch.setattr(e2b_executor, "_build_source_archive", source_builder)
    monkeypatch.setattr(e2b_executor, "_build_evaluation_archive", evaluation_builder)

    artifacts = e2b_executor.prepare_e2b_artifacts(config, ["task-1", "task-1", "task-2"])

    source_builder.assert_called_once()
    assert evaluation_builder.call_count == 2
    assert set(artifacts.evaluation_archives) == {"task-1", "task-2"}
    with pytest.raises(TypeError):
        artifacts.evaluation_archives["task-1"] = b"changed"
    source_builder.side_effect = AssertionError("source archive rebuilt during rollout")
    evaluation_builder.side_effect = AssertionError("evaluation archive rebuilt during rollout")
    chat_template.write_text("changed after preparation", encoding="utf-8")
    for task_id in ("task-1", "task-2"):
        (task_files_root / task_id / "expected.txt").unlink()

    original_write = _FakeFiles.write
    failed_upload = False

    async def write(self, path, data, *, user):
        nonlocal failed_upload
        if path == e2b_executor._SOURCE_ARCHIVE_PATH and not failed_upload:
            failed_upload = True
            raise ConnectionError("first sandbox upload failed")
        await original_write(self, path, data, user=user)

    monkeypatch.setattr(_FakeFiles, "write", write)

    async def run_tasks():
        for task_id in ("task-1", "task-2"):
            result = await e2b_executor.run_rollout_in_e2b(
                RolloutInput(rollout_id="rollout-1", task={}, metadata={"task_id": task_id}),
                config,
                gateway=_gateway(),
                template="swebench-template",
                timeout_seconds=3600,
                prepared_artifacts=artifacts,
            )
            assert result.status == "success"
            assert result.metrics["sandbox_retries"] == 0.0

    asyncio.run(run_tasks())

    assert len(_FakeAsyncSandbox.instances) == 2
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)
    for sandbox, task_id in zip(_FakeAsyncSandbox.instances, ("task-1", "task-2")):
        writes = dict(sandbox.files.writes)
        assert writes[e2b_executor._SOURCE_ARCHIVE_PATH] is artifacts.source_archive
        assert (
            writes[e2b_executor._EVALUATION_ARCHIVE_PATH] is artifacts.evaluation_archives[task_id]
        )
        assert not any("chat-template" in path for path in writes)
        with tarfile.open(
            fileobj=BytesIO(writes[e2b_executor._EVALUATION_ARCHIVE_PATH]), mode="r:gz"
        ) as archive:
            assert archive.extractfile("expected.txt").read().decode("utf-8") == task_id
    source_builder.assert_called_once()
    assert evaluation_builder.call_count == 2


def test_standalone_rollout_prepares_once_off_event_loop_and_reuses_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver_thread = threading.get_ident()
    preparation_threads = []
    original_source_builder = e2b_executor._build_source_archive

    def build_source(entrypoint):
        preparation_threads.append(threading.get_ident())
        return original_source_builder(entrypoint)

    source_builder = Mock(side_effect=build_source)
    evaluation_builder = Mock(wraps=e2b_executor._build_evaluation_archive)
    monkeypatch.setattr(e2b_executor, "_build_source_archive", source_builder)
    monkeypatch.setattr(e2b_executor, "_build_evaluation_archive", evaluation_builder)
    original_write = _FakeFiles.write
    failed_upload = False

    async def write(self, path, data, *, user):
        nonlocal failed_upload
        if path == e2b_executor._SOURCE_ARCHIVE_PATH and not failed_upload:
            failed_upload = True
            raise ConnectionError("first sandbox upload failed")
        await original_write(self, path, data, user=user)

    monkeypatch.setattr(_FakeFiles, "write", write)

    result = _run()

    assert result.status == "success"
    assert result.metrics["sandbox_retries"] == 0.0
    assert len(_FakeAsyncSandbox.instances) == 1
    source_builder.assert_called_once()
    evaluation_builder.assert_called_once()
    assert preparation_threads[0] != driver_thread


def _failed(status="env_error", *, rollout_id="rollout-1") -> RemoteRolloutResult:
    return RemoteRolloutResult(
        rollout_id=rollout_id,
        status=status,
        segments=[],
        reward=0.0,
        error_type="RuntimeError",
        error_message="failed",
    )


def _state(phase: str, **kwargs) -> str:
    return json.dumps({"rollout_id": "rollout-1", "phase": phase, **kwargs})


def test_missing_entrypoint_is_environment_error_before_sandbox_create() -> None:
    config = _config()
    del config.rollout.execution.entrypoint
    result = _run(config)
    assert result.status == "env_error"
    assert result.reward == 0.0
    assert result.error_type == "ValueError"
    assert "entrypoint is required" in result.error_message
    assert result.segments == []
    assert not _FakeAsyncSandbox.create_calls


@pytest.mark.parametrize("mounts", [[], {"relative/path": "shared-data"}])
def test_invalid_volume_mounts_fail_before_preparation_or_attempts(monkeypatch, mounts):
    config = _config()
    config.rollout.execution.volume_mounts = mounts
    config.rollout.execution.env_error_retries = 3
    prepare = Mock()
    run_once = AsyncMock()
    monkeypatch.setattr(e2b_executor, "prepare_e2b_artifacts", prepare)
    monkeypatch.setattr(e2b_executor, "_run_rollout_in_e2b_once", run_once)
    events = []

    result = _run(config, event_callback=events.append)

    assert result.status == "env_error"
    assert result.error_type == "ValueError"
    assert "volume_mounts" in result.error_message
    assert result.reward == 0.0
    assert result.segments == []
    assert result.metrics["sandbox_retries"] == 0.0
    prepare.assert_not_called()
    run_once.assert_not_awaited()
    assert not _FakeAsyncSandbox.create_calls
    assert len(events) == 1
    assert events[0]["kind"] == "rollout_finished"
    assert events[0]["attempts"] == 0


@pytest.mark.parametrize("policy", [None, {"deny_domains": ["https://github.com"]}])
def test_invalid_network_policy_fails_before_preparation_or_attempts(monkeypatch, policy):
    config = _config()
    config.rollout.execution.network_policy = policy
    config.rollout.execution.env_error_retries = 3
    prepare = Mock()
    run_once = AsyncMock()
    monkeypatch.setattr(e2b_executor, "prepare_e2b_artifacts", prepare)
    monkeypatch.setattr(e2b_executor, "_run_rollout_in_e2b_once", run_once)
    events = []

    result = _run(config, event_callback=events.append)

    assert result.status == "env_error"
    assert result.error_type == "ValueError"
    assert "network_policy" in result.error_message
    assert result.metrics["sandbox_retries"] == 0.0
    prepare.assert_not_called()
    run_once.assert_not_awaited()
    assert not _FakeAsyncSandbox.create_calls
    assert events[0]["attempts"] == 0


@pytest.mark.parametrize(
    "task_id, expected",
    [
        (None, {"deny_domains": ["github.com"]}),
        ("ordinary", {"deny_domains": ["github.com"]}),
        ("special", {"deny_domains": ["gitlab.com"]}),
        ("open", {"deny_domains": []}),
        ("isolated", {"mode": "local_only", "deny_domains": []}),
    ],
)
def test_only_resolved_host_network_policy_reaches_sandbox(task_id, expected):
    config = _config()
    config.rollout.execution.network_policy = {
        "deny_domains": ["*.GitHub.COM."],
        "task_overrides": {
            "special": {"deny_domains": ["gitlab.com"]},
            "open": {"deny_domains": []},
            "isolated": {"mode": "local_only"},
            "private-other-task": {"deny_domains": ["private.example"]},
        },
    }
    input_data = RolloutInput(
        rollout_id="rollout-1",
        task={"task_id": "open", "network_policy": {"deny_domains": []}},
        metadata={} if task_id is None else {"task_id": task_id},
    )
    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            input_data,
            config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
        )
    )
    assert result.status == "success"
    raw_input = dict(_FakeAsyncSandbox.instances[0].files.writes)[e2b_executor._INPUT_PATH]
    payload = json.loads(raw_input)
    assert payload["config"]["rollout"]["execution"]["network_policy"] == expected
    assert "private-other-task" not in raw_input
    assert "private.example" not in raw_input
    assert "task_overrides" not in payload["config"]["rollout"]["execution"]["network_policy"]


def test_gateway_rollout_does_not_read_or_upload_local_codec(tmp_path: Path) -> None:
    config = _config()
    config.rollout.execution.chat_template = str(tmp_path / "not-present.jinja")
    config.rollout.execution.pop("tokenizer", None)
    del config.rollout.parsers

    assert _run(config).status == "success"
    writes = dict(_FakeAsyncSandbox.instances[0].files.writes)
    payload = json.loads(writes[e2b_executor._INPUT_PATH])
    assert payload["gateway"] == _gateway()
    assert set(payload["config"]["rollout"]["execution"]) == {
        "token_prefix_reuse",
        "preparation_timeout_seconds",
        "agent_timeout_seconds",
        "agent_finalize_timeout_seconds",
        "evaluation_timeout_seconds",
    }
    assert "parsers" not in payload["config"]["rollout"]
    assert "tokenizer" not in payload["config"]["models"]["actor"]
    assert str(tmp_path) not in writes[e2b_executor._INPUT_PATH]
    assert not any("chat-template" in path for path in writes)


@pytest.mark.parametrize("enabled", [False, True])
def test_token_prefix_reuse_is_uploaded_to_sandbox(enabled):
    config = _config()
    config.rollout.execution.token_prefix_reuse = enabled

    assert _run(config).status == "success"

    writes = dict(_FakeAsyncSandbox.instances[0].files.writes)
    payload = json.loads(writes[e2b_executor._INPUT_PATH])
    assert payload["config"]["rollout"]["execution"]["token_prefix_reuse"] is enabled


def test_missing_northgate_key_fails_without_forwarding_weaver_credentials(monkeypatch) -> None:
    monkeypatch.delenv("NORTHGATE_API_KEY")
    config = _config()
    config.rollout.execution.env_error_retries = 0

    result = _run(config)

    assert result.status == "env_error"
    assert "NORTHGATE_API_KEY is required" in result.error_message
    assert not _FakeAsyncSandbox.create_calls


def test_gateway_handoff_preserves_sampling_requirements_and_omits_extra_credentials() -> None:
    config = _config()
    config.rollout.keep_sampling_mask = True
    config.algorithm.update(
        {
            "mis": {"enabled": True},
            "router_replay": {"enabled": True, "mode": "R3"},
            "loss": {"mode": "score_centering", "params": {"topk": 64}},
        }
    )
    config.validate = {"sampling": {"temperature": 0, "max_tokens": 128}}
    gateway = {**_gateway(), "api_key": "must-not-be-in-json"}
    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            config,
            gateway=gateway,
            template="swebench-template",
            timeout_seconds=3600,
        )
    )
    assert result.status == "success"
    raw = dict(_FakeAsyncSandbox.instances[0].files.writes)[e2b_executor._INPUT_PATH]
    payload = json.loads(raw)
    assert payload["gateway"] == _gateway()
    assert "must-not-be-in-json" not in raw
    assert payload["config"]["rollout"]["keep_sampling_mask"]
    assert payload["config"]["algorithm"] == OmegaConf.to_container(config.algorithm)
    assert payload["config"]["validate"]["sampling"] == {"temperature": 0, "max_tokens": 128}


@pytest.mark.parametrize("field", ["base_url", "model", "sampling_session_id", "weight_version"])
def test_incomplete_gateway_fails_before_sandbox_create(field) -> None:
    gateway = _gateway()
    del gateway[field]
    config = _config()
    config.rollout.execution.env_error_retries = 0
    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            config,
            gateway=gateway,
            template="swebench-template",
            timeout_seconds=3600,
        )
    )
    assert result.status == "env_error"
    assert f"gateway.{field}" in result.error_message
    assert not _FakeAsyncSandbox.create_calls


@pytest.mark.parametrize(
    ("entrypoint", "exception", "match"),
    [
        ({"target": "agent.py"}, ValueError, "target.*absolute host path"),
        ({"target": "~/agent.py"}, ValueError, "target.*absolute host path"),
        (
            {"target": str(_SWEREBENCH_ENTRYPOINT), "task_files_root": "evaluation"},
            ValueError,
            "task_files_root.*absolute host path",
        ),
        (
            {"target": str(_SWEREBENCH_ENTRYPOINT), "task_files_root": "~/evaluation"},
            ValueError,
            "task_files_root.*absolute host path",
        ),
        ({"target": "/my_task.rollout"}, ValueError, "target.*\\.py file"),
        ({}, ValueError, "target.*non-empty"),
        ({"target": " "}, ValueError, "target.*non-empty"),
        ({"target": "/missing.py"}, FileNotFoundError, "not found"),
    ],
)
def test_invalid_entrypoint_is_environment_error_before_sandbox_create(
    entrypoint: dict[str, str],
    exception: type[Exception],
    match: str,
) -> None:
    config = _config()
    config.rollout.execution.entrypoint = entrypoint
    result = _run(config)
    assert result.status == "env_error"
    assert result.error_type == exception.__name__
    assert re.search(match, result.error_message)
    assert not _FakeAsyncSandbox.create_calls


def test_wrong_result_id_is_environment_error_and_never_returns_foreign_tokens() -> None:
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(_result("wrong-rollout")))
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("wrong-rollout")))
    result = _run()
    assert result.status == "env_error"
    assert "does not match rollout input" in result.error_message
    assert result.segments == []
    assert len(_FakeAsyncSandbox.instances) == 1
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


def test_extract_failure_retries_three_times_even_when_cleanup_fails() -> None:
    _FakeAsyncSandbox.command_error = RuntimeError("extract failed")
    _FakeAsyncSandbox.kill_error = RuntimeError("kill failed")
    result = _run()
    assert result.status == "env_error"
    assert result.error_message == "extract failed"
    assert result.segments == []
    assert len(_FakeAsyncSandbox.instances) == 4
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)
    assert all(
        all(path == e2b_executor._METRICS_PATH for path in sandbox.files.reads)
        for sandbox in _FakeAsyncSandbox.instances
    )


def test_creation_failure_retries_three_times_and_returns_environment_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(side_effect=ConnectionError("E2B is unavailable"))
    monkeypatch.setattr(_FakeAsyncSandbox, "create", create)
    result = _run()
    assert result.status == "env_error"
    assert result.error_type == "ConnectionError"
    assert result.error_message == "E2B is unavailable"
    assert create.await_count == 4


@pytest.mark.parametrize("second_setup_seconds", [0.0, 260.0])
def test_preparation_budget_is_shared_across_bootstrap_retries(monkeypatch, second_setup_seconds):
    clock = [0.0]
    bootstrap_calls = 0
    original_run = _FakeCommands.run
    monkeypatch.setattr(
        e2b_executor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000.0 + clock[0]),
    )
    monkeypatch.setattr(wire, "time", e2b_executor.time)

    async def run(self, command, **kwargs):
        nonlocal bootstrap_calls
        await original_run(self, command, **kwargs)
        if command == e2b_executor._BOOTSTRAP_COMMAND:
            bootstrap_calls += 1
            if bootstrap_calls == 1:
                clock[0] += 350.0
                raise RuntimeError("first bootstrap failed after consuming preparation time")
            clock[0] += second_setup_seconds
            if second_setup_seconds:
                raise RuntimeError("second bootstrap exhausted the remaining preparation time")

    monkeypatch.setattr(_FakeCommands, "run", run)
    result = _run()

    assert bootstrap_calls == 2
    assert len(_FakeAsyncSandbox.instances) == 2
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)
    first, second = _FakeAsyncSandbox.instances
    assert first.commands.calls[0][1]["timeout"] == 600
    assert second.commands.calls[0][1]["timeout"] == 250
    assert second.commands.calls[1][1]["timeout"] == 250
    if second_setup_seconds:
        assert result.status == "env_error"
        assert not any("--phase agent" in command for command, _ in second.commands.calls)
        assert result.metrics["sandbox_retries"] == 1.0
    else:
        assert result.status == "success"
        assert second.commands.calls[2][1]["timeout"] == 4450


def _creation_retry_clock(monkeypatch: pytest.MonkeyPatch):
    clock = [0.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(
        e2b_executor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000.0 + clock[0]),
    )
    monkeypatch.setattr(wire, "time", e2b_executor.time)
    monkeypatch.setattr(e2b_executor.random, "uniform", lambda low, high: high)
    monkeypatch.setattr(e2b_executor.asyncio, "sleep", sleep)
    return clock, sleeps


def test_creation_500_backoff_charges_rpc_and_sleep_time(monkeypatch) -> None:
    class SandboxException(RuntimeError):
        pass

    class PlacementError(SandboxException):
        pass

    clock, sleeps = _creation_retry_clock(monkeypatch)
    sandbox = object()
    outcomes = iter([PlacementError("500: Failed to place sandbox")] * 2 + [sandbox])

    async def create(**kwargs):
        clock[0] += 2.0
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    create_mock = AsyncMock(side_effect=create)
    budget = e2b_executor._SandboxCreationRetryBudget(600.0)
    result = asyncio.run(
        e2b_executor._create_sandbox_with_backoff(
            create_mock,
            template="template-1",
            sandbox_timeout=3660,
            rollout_id="rollout-1",
            retry_budget=budget,
        )
    )
    assert result is sandbox
    assert [call.kwargs for call in create_mock.await_args_list] == [
        {"template": "template-1", "timeout": 3660}
    ] * 3
    assert sleeps == [1.0, 2.0]
    assert budget.remaining_seconds == pytest.approx(591.0)


@pytest.mark.parametrize(
    ("exception_name", "message"),
    [
        ("SandboxException", "429: rate limited"),
        ("SandboxException", "500 without status delimiter"),
        ("RuntimeError", "500: Failed to place sandbox"),
    ],
)
def test_creation_backoff_only_retries_sdk_http_500(monkeypatch, exception_name, message) -> None:
    _, sleeps = _creation_retry_clock(monkeypatch)
    error = type(exception_name, (RuntimeError,), {})(message)
    create = AsyncMock(side_effect=error)

    with pytest.raises(type(error)) as raised:
        asyncio.run(
            e2b_executor._create_sandbox_with_backoff(
                create,
                template="template-1",
                sandbox_timeout=3660,
                rollout_id="rollout-1",
                retry_budget=e2b_executor._SandboxCreationRetryBudget(600.0),
            )
        )
    assert raised.value is error
    assert create.await_count == 1
    assert sleeps == []


def test_creation_budget_exhaustion_stops_outer_environment_retries(monkeypatch) -> None:
    class SandboxException(RuntimeError):
        pass

    clock, sleeps = _creation_retry_clock(monkeypatch)
    error = SandboxException("500: Failed to place sandbox")
    create = AsyncMock(side_effect=error)
    monkeypatch.setattr(_FakeAsyncSandbox, "create", create)
    events = []

    result = _run(event_callback=events.append)

    assert result.status == "env_error"
    assert result.error_type == "SandboxException"
    assert result.error_message == str(error)
    assert result.metrics["sandbox_retries"] == 0.0
    assert sum(event["kind"] == "attempt_started" for event in events) == 1
    assert create.await_count > 4
    assert clock[0] == pytest.approx(600.0)
    assert max(sleeps) == 60.0
    assert sleeps[-1] < 60.0


def test_creation_budget_is_shared_after_bootstrap_environment_retry(monkeypatch) -> None:
    class SandboxException(RuntimeError):
        pass

    clock, sleeps = _creation_retry_clock(monkeypatch)
    error = SandboxException("500: Failed to place sandbox")
    sandbox = _FakeSandbox("", None, RuntimeError("bootstrap failed"), None)
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            clock[0] += 100.0
        if calls == 2:
            return sandbox
        raise error

    monkeypatch.setattr(_FakeAsyncSandbox, "create", AsyncMock(side_effect=create))
    events = []
    result = _run(event_callback=events.append)

    assert sandbox.killed
    assert result.status == "env_error"
    assert result.error_type == "SandboxException"
    assert result.error_message == str(error)
    assert result.metrics["sandbox_retries"] == 1.0
    assert sum(event["kind"] == "attempt_started" for event in events) == 2
    assert clock[0] == pytest.approx(600.0)
    assert sum(sleeps) == pytest.approx(400.0)


@pytest.mark.parametrize("stage", ["create", "sleep"])
def test_creation_backoff_propagates_cancellation(monkeypatch, stage) -> None:
    class SandboxException(RuntimeError):
        pass

    _creation_retry_clock(monkeypatch)
    cancellation = asyncio.CancelledError("cancelled creation")
    error = cancellation if stage == "create" else SandboxException("500: unavailable")
    create = AsyncMock(side_effect=error)
    if stage == "sleep":
        monkeypatch.setattr(e2b_executor.asyncio, "sleep", AsyncMock(side_effect=cancellation))

    with pytest.raises(asyncio.CancelledError, match="cancelled creation"):
        asyncio.run(
            e2b_executor._create_sandbox_with_backoff(
                create,
                template="template-1",
                sandbox_timeout=3660,
                rollout_id="rollout-1",
                retry_budget=e2b_executor._SandboxCreationRetryBudget(600.0),
            )
        )
    assert create.await_count == 1


def test_hanging_sandbox_creation_is_bounded_by_remaining_budget() -> None:
    cancelled = []

    async def create(**kwargs):
        try:
            await asyncio.sleep(3600)
        finally:
            cancelled.append(True)

    budget = e2b_executor._SandboxCreationRetryBudget(0.01)
    with pytest.raises(e2b_executor._SandboxCreationRetryExhausted) as raised:
        asyncio.run(
            e2b_executor._create_sandbox_with_backoff(
                create,
                template="template-1",
                sandbox_timeout=3660,
                rollout_id="rollout-1",
                retry_budget=budget,
            )
        )
    assert isinstance(raised.value.original_error, TimeoutError)
    assert cancelled == [True]
    assert budget.remaining_seconds <= 0.0


@pytest.mark.parametrize("error", [RuntimeError("upload failed"), TimeoutError("upload timeout")])
@pytest.mark.usefixtures("bounded_io")
def test_upload_failure_is_environment_error_and_always_closes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(_FakeFiles, "write", AsyncMock(side_effect=error))
    result = _run()
    assert result.status == "env_error"
    assert result.error_type == type(error).__name__
    assert len(_FakeAsyncSandbox.instances) == 4
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


def test_environment_retries_use_fresh_sandboxes_and_stop_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _FakeSandbox(
        dump_artifact(encode_result(_failed())), None, RuntimeError("bootstrap failed"), None
    )
    second = _FakeSandbox(dump_artifact(encode_result(_result("rollout-1"))), None, None, None)
    create = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(_FakeAsyncSandbox, "create", create)
    result = _run()
    assert _without_driver_runtime(result) == _result("rollout-1")
    assert create.await_count == 2
    assert first.killed and second.killed


@pytest.mark.parametrize("host_configured", [False, True])
def test_only_host_volume_mounts_reach_each_replacement_sandbox(monkeypatch, host_configured):
    config = _config()
    mounts = {"/mnt/data": "shared-data", "/opt/tools": "shared-tools"}
    if host_configured:
        config.rollout.execution.volume_mounts = mounts
    input_data = RolloutInput(
        rollout_id="rollout-1",
        task={"volume_mounts": {"/mnt/data": "task-overwrite", "/task-only": "task-volume"}},
    )
    original_run = _FakeCommands.run
    bootstrap_calls = 0

    async def run(self, command, **kwargs):
        nonlocal bootstrap_calls
        await original_run(self, command, **kwargs)
        if command == e2b_executor._BOOTSTRAP_COMMAND:
            bootstrap_calls += 1
            if bootstrap_calls == 1:
                raise RuntimeError("first sandbox bootstrap failed")

    monkeypatch.setattr(_FakeCommands, "run", run)
    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            input_data,
            config,
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
        )
    )

    assert result.status == "success"
    assert result.metrics["sandbox_retries"] == 1.0
    assert bootstrap_calls == 2
    assert len(_FakeAsyncSandbox.create_calls) == 2
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)
    for call in _FakeAsyncSandbox.create_calls:
        if host_configured:
            assert call["volume_mounts"] == mounts
        else:
            assert "volume_mounts" not in call


@pytest.mark.parametrize("status", ["success", "task_failed"])
def test_only_environment_results_are_retried(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    expected = _result("rollout-1") if status == "success" else _failed(status)
    run_once = AsyncMock(return_value=expected)
    monkeypatch.setattr(e2b_executor, "_run_rollout_in_e2b_once", run_once)
    result = _run()
    assert _without_driver_runtime(result) == expected
    assert run_once.await_count == 1


@pytest.mark.parametrize("retries", [0, 1, 3])
def test_environment_retry_count_and_exhausted_outcome(
    monkeypatch: pytest.MonkeyPatch,
    retries: int,
) -> None:
    config = _config()
    config.rollout.execution.env_error_retries = retries
    failures = [_failed().model_copy(update={"error_message": str(i)}) for i in range(retries + 1)]
    run_once = AsyncMock(side_effect=failures)
    monkeypatch.setattr(e2b_executor, "_run_rollout_in_e2b_once", run_once)
    assert _without_driver_runtime(_run(config)) == failures[-1]
    assert run_once.await_count == retries + 1


@pytest.mark.parametrize("retries", [None, -1, True, 1.5, "3"])
def test_invalid_environment_retry_config_returns_environment_error(retries) -> None:
    config = _config()
    config.rollout.execution.env_error_retries = retries
    result = _run(config)
    assert result.status == "env_error"
    assert "env_error_retries must be a non-negative integer" in result.error_message
    assert not _FakeAsyncSandbox.create_calls


def test_cleanup_failure_keeps_successful_result_without_retry() -> None:
    _FakeAsyncSandbox.kill_error = RuntimeError("cleanup failed")
    assert _without_driver_runtime(_run()) == _result("rollout-1")
    assert len(_FakeAsyncSandbox.instances) == 1


@pytest.mark.parametrize("stage", ["extract", "command"])
def test_cancellation_closes_sandbox_without_retrying(stage: str) -> None:
    cancellation = asyncio.CancelledError("cancelled by caller")
    if stage == "extract":
        _FakeAsyncSandbox.command_error = cancellation
    else:
        _FakeAsyncSandbox.entrypoint_error = cancellation
    _FakeAsyncSandbox.kill_error = RuntimeError("cleanup failed")
    with pytest.raises(asyncio.CancelledError, match="cancelled by caller"):
        _run()
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed


def test_run_entrypoint_failure_prints_full_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    root_cause = ValueError("remote root cause")
    command_error = RuntimeError("command failed\nremote traceback detail")
    command_error.__cause__ = root_cause
    sandbox = _FakeSandbox(
        dump_artifact(encode_result(_result("rollout-1"))),
        None,
        command_error,
        None,
    )

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            e2b_executor._run_entrypoint_with_trace_sync(
                sandbox,
                entrypoint_command="python entrypoint.py",
                timeout_seconds=60,
                sandbox_env={"PYTHONPATH": "/untrusted", "PYTHONSAFEPATH": "0"},
                trace_destination=None,
                log_context="rollout='rollout-1' sandbox='sandbox-1'",
                rollout_id="rollout-1",
            )
        )

    assert raised.value is command_error
    assert sandbox.commands.calls[0][1]["envs"]["PYTHONPATH"] == e2b_executor._SOURCE_ROOT
    assert sandbox.commands.calls[0][1]["envs"]["PYTHONSAFEPATH"] == "1"
    assert sandbox.commands.calls[0][1]["cwd"] == e2b_executor._SOURCE_ROOT
    stderr = capsys.readouterr().err
    assert "entrypoint failed; full traceback follows" in stderr
    assert "Traceback (most recent call last):" in stderr
    assert "ValueError: remote root cause" in stderr
    assert "The above exception was the direct cause" in stderr
    assert "RuntimeError: command failed\nremote traceback detail" in stderr


@pytest.mark.parametrize("phase", ["environment", "agent"])
@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_command_failures_use_phase_and_preserve_real_partial_trace(
    tmp_path: Path,
    phase: str,
    timed_out: bool,
) -> None:
    expected_trace = _trace("rollout-1")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(expected_trace))
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.entrypoint_error = (
        TimeoutError("rollout timed out") if timed_out else RuntimeError("rollout crashed")
    )
    _FakeAsyncSandbox.state_json = _state(
        phase, **({"status": "task_failed"} if phase == "agent" else {})
    )
    trace_output = tmp_path / "rollout-trace.json"
    result = _run(trace_output_path=trace_output)
    assert result.status == ("env_error" if phase == "environment" else "task_failed")
    assert _dense_segments(result.segments) == (expected_trace.segments if phase == "agent" else [])
    assert result.message_format == (expected_trace.message_format if phase == "agent" else None)
    assert result.tools == (expected_trace.tools if phase == "agent" else None)
    assert result.messages == (expected_trace.messages if phase == "agent" else [])
    assert result.reward == 0.0
    assert result.metrics.get("e2b_timed_out", 0.0) == float(timed_out)
    assert len(_FakeAsyncSandbox.create_calls) == 1
    assert decode_trace(load_trace_gzip(trace_output.read_bytes())) == expected_trace
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_command_failure_without_phase_is_environment_error(timed_out: bool) -> None:
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.entrypoint_error = (
        TimeoutError("runtime startup timed out") if timed_out else RuntimeError("script failed")
    )
    result = _run()
    assert result.status == "env_error"
    assert result.segments == []
    assert result.metrics.get("e2b_timed_out", 0.0) == float(timed_out)
    assert len(_FakeAsyncSandbox.instances) == 1
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


@pytest.mark.parametrize("status", ["env_error", "task_failed"])
@pytest.mark.usefixtures("bounded_io")
def test_matching_final_failure_result_takes_priority_over_phase(status: str) -> None:
    expected = _failed(status)
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(expected))
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("script raised after writing result")
    _FakeAsyncSandbox.state_json = _state("agent" if status == "env_error" else "environment")
    assert _without_driver_runtime(_run()) == expected
    assert len(_FakeAsyncSandbox.instances) == 1
    assert all(e2b_executor._STATE_PATH not in s.files.reads for s in _FakeAsyncSandbox.instances)


def test_failure_result_keeps_matching_trace_when_its_trajectories_are_empty() -> None:
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(_failed("task_failed")))
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("agent failed")
    result = _run()
    assert result.status == "task_failed"
    assert _dense_segments(result.segments) == _trace("rollout-1").segments
    assert result.tools == _trace("rollout-1").tools
    assert len(_FakeAsyncSandbox.instances) == 1


def test_command_failure_without_deadline_restores_dense_trace() -> None:
    trace = _prefix_trace("rollout-1")
    expected = _failed("task_failed")
    read = AsyncMock(return_value=dump_artifact(encode_result(expected)))
    result = asyncio.run(
        e2b_executor._command_failure_result(
            SimpleNamespace(files=SimpleNamespace(read=read)),
            RolloutInput(rollout_id="rollout-1", task={}),
            RuntimeError("agent failed"),
            trace=encode_trace(trace),
            log_context="test",
        )
    )

    assert result.status == "task_failed"
    assert result.reward == expected.reward
    assert result.error_type == expected.error_type
    assert _dense_segments(result.segments) == trace.segments
    assert result.messages == trace.messages
    assert result.tools == trace.tools
    assert result.metrics == {**trace.metrics, **expected.metrics}
    read.assert_awaited_once_with(
        e2b_executor._RESULT_PATH, user="root", format="bytes", request_timeout=120
    )


def test_phase_marker_preserves_reported_error_details() -> None:
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("command exited 1")
    _FakeAsyncSandbox.state_json = _state(
        "environment",
        status="env_error",
        error_type="FileNotFoundError",
        error_message="Claude executable is missing",
    )
    result = _run()
    assert result.status == "env_error"
    assert result.error_type == "FileNotFoundError"
    assert result.error_message == "Claude executable is missing"


@pytest.mark.parametrize(
    "state_json",
    [
        "not json",
        "[]",
        _state("invalid"),
        _state("invalid", status="task_failed"),
        _state("agent", status="unknown"),
        json.dumps({"phase": "agent"}),
        _state("environment", rollout_id="foreign-rollout"),
        _state("agent", attempt_id="previous-attempt"),
        _state("agent", attempt_id=None),
    ],
)
def test_invalid_phase_marker_never_reruns_agent_or_returns_training_segments(
    state_json: str,
) -> None:
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("generic command failed")
    _FakeAsyncSandbox.state_json = state_json
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    result = _run()
    assert result.status == "env_error"
    assert result.segments == []
    assert len(_FakeAsyncSandbox.instances) == 1
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


@pytest.mark.usefixtures("bounded_io")
def test_foreign_result_and_trace_do_not_contaminate_task_failure(tmp_path: Path) -> None:
    trace_output = tmp_path / "trace.json"
    _FakeAsyncSandbox.result_json = dump_artifact(
        encode_result(_failed("env_error", rollout_id="foreign"))
    )
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("foreign")))
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("command failed")
    _FakeAsyncSandbox.state_json = _state("agent", status="task_failed")
    result = _run(trace_output_path=trace_output)
    assert result.status == "task_failed"
    assert result.segments == []
    assert not trace_output.exists()
    assert len(_FakeAsyncSandbox.instances) == 1


@pytest.mark.usefixtures("bounded_io")
def test_final_result_download_failure_remains_environment_error() -> None:
    _FakeAsyncSandbox.result_error = TimeoutError("result download timed out")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    _FakeAsyncSandbox.state_json = _state("agent")
    result = _run()
    assert result.status == "env_error"
    assert result.segments == []
    assert result.metrics["e2b_timed_out"] == 1.0
    assert len(_FakeAsyncSandbox.instances) == 1


@pytest.mark.parametrize("remote_hangs", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_current_attempt_local_trace_survives_remote_download_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote_hangs: bool,
) -> None:
    trace_output = tmp_path / "trace.json"
    expected_trace = _prefix_trace("rollout-1")
    original_read = _FakeFiles.read
    trace_reads = 0

    async def read(self, path: str, *, user, format="text", request_timeout=None):
        nonlocal trace_reads
        if path == e2b_executor._TRACE_PATH:
            trace_reads += 1
            if remote_hangs:
                await asyncio.Event().wait()
        return await original_read(self, path, user=user, format=format)

    async def run_and_checkpoint(*args, **kwargs):
        trace_output.write_bytes(dump_artifact(encode_trace(expected_trace)))
        raise TimeoutError("agent timed out")

    monkeypatch.setattr(_FakeFiles, "read", read)
    monkeypatch.setattr(e2b_executor, "_run_entrypoint_with_trace_sync", run_and_checkpoint)
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.trace_error = TimeoutError("trace download timed out")
    _FakeAsyncSandbox.state_json = _state("agent", status="task_failed")
    result = _run(trace_output_path=trace_output)
    assert result.status == "task_failed"
    assert _dense_segments(result.segments) == expected_trace.segments
    assert result.tools == expected_trace.tools
    assert result.messages == expected_trace.messages
    assert result.message_system == expected_trace.message_system
    assert result.metrics["tool_calls_emitted"] == 3.0
    assert len(_FakeAsyncSandbox.instances) == 1
    assert trace_reads == 1
    assert _FakeAsyncSandbox.instances[0].killed


@pytest.mark.parametrize("remote_hangs", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_failure_recovery_retries_trace_without_repeating_rollout(
    monkeypatch: pytest.MonkeyPatch,
    remote_hangs: bool,
) -> None:
    expected_trace = _trace("rollout-1")
    original_read = _FakeFiles.read
    trace_reads = 0

    async def read(self, path: str, *, user, format="text", request_timeout=None):
        nonlocal trace_reads
        if path == e2b_executor._TRACE_PATH:
            trace_reads += 1
            if trace_reads == 1:
                if remote_hangs:
                    await asyncio.Event().wait()
                raise FileNotFoundError(path)
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(expected_trace))
    _FakeAsyncSandbox.entrypoint_error = TimeoutError("agent timed out")
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.state_json = _state("agent", status="task_failed")
    result = _run()
    assert result.status == "task_failed"
    assert _dense_segments(result.segments) == expected_trace.segments
    assert result.messages == expected_trace.messages
    assert trace_reads == 2
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed


@pytest.mark.parametrize("remote_hangs", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_exhausted_trace_downloads_still_classify_agent_timeout(
    monkeypatch: pytest.MonkeyPatch,
    remote_hangs: bool,
) -> None:
    original_read = _FakeFiles.read
    trace_reads = 0

    async def read(self, path: str, *, user, format="text", request_timeout=None):
        nonlocal trace_reads
        if path == e2b_executor._TRACE_PATH:
            trace_reads += 1
            if remote_hangs:
                await asyncio.Event().wait()
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    _FakeAsyncSandbox.entrypoint_error = TimeoutError("agent timed out")
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.state_json = _state("agent", status="task_failed")
    result = _run()
    assert result.status == "task_failed"
    assert result.segments == []
    assert result.metrics["e2b_timed_out"] == 1.0
    assert trace_reads >= 1
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed


def test_cancellation_during_trace_recovery_closes_sandbox_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = _FakeFiles.read
    trace_reads = 0
    _FakeAsyncSandbox.entrypoint_error = TimeoutError("agent timed out")
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(_failed("task_failed")))

    async def scenario() -> None:
        trace_started = asyncio.Event()

        async def read(self, path: str, *, user, format="text", request_timeout=None):
            nonlocal trace_reads
            if path == e2b_executor._TRACE_PATH:
                trace_reads += 1
                trace_started.set()
                await asyncio.Event().wait()
            return await original_read(self, path, user=user, format=format)

        monkeypatch.setattr(_FakeFiles, "read", read)
        rollout_task = asyncio.create_task(
            e2b_executor.run_rollout_in_e2b(
                RolloutInput(rollout_id="rollout-1", task={}),
                _config(),
                gateway=_gateway(),
                template="swebench-template",
                timeout_seconds=3600,
            )
        )
        await asyncio.wait_for(trace_started.wait(), 1.0)
        rollout_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rollout_task

    asyncio.run(scenario())
    assert trace_reads == 1
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed
    assert e2b_executor._STATE_PATH not in _FakeAsyncSandbox.instances[0].files.reads


@pytest.mark.usefixtures("bounded_io")
def test_same_id_trace_from_previous_attempt_is_cleared_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_output = tmp_path / "trace.json"
    trace_output.write_bytes(dump_artifact(encode_trace(_trace("rollout-1"))))
    calls = 0
    original_run = _FakeCommands.run

    async def fail_in_each_phase(self, command, **kwargs):
        nonlocal calls
        if command == e2b_executor._BOOTSTRAP_COMMAND and calls == 0:
            assert not trace_output.exists()
            trace_output.write_bytes(dump_artifact(encode_trace(_trace("rollout-1"))))
            calls += 1
            raise RuntimeError("bootstrap failed before agent dispatch")
        await original_run(self, command, **kwargs)
        if "--phase agent" in command:
            assert not trace_output.exists()
            self.files._state_json = _state("agent", status="task_failed")
            calls += 1
            raise RuntimeError("agent failed")

    monkeypatch.setattr(_FakeCommands, "run", fail_in_each_phase)
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    result = _run(trace_output_path=trace_output)
    assert result.status == "task_failed"
    assert result.segments == []
    assert len(_FakeAsyncSandbox.instances) == 2
    assert not trace_output.exists()


@pytest.mark.parametrize("missing_type", ["builtin", "sdk"])
def test_background_trace_poll_skips_missing_file_then_downloads(tmp_path, caplog, missing_type):
    class FileNotFoundException(Exception):
        pass

    FileNotFoundException.__module__ = "e2b.exceptions"
    missing = (
        FileNotFoundError("no trace") if missing_type == "builtin" else FileNotFoundException()
    )
    payload = dump_artifact(encode_trace(_trace("rollout-1")))
    read = AsyncMock(side_effect=[missing, payload])
    sandbox = SimpleNamespace(sandbox_id="sandbox-1", files=SimpleNamespace(read=read))
    destination = tmp_path / "trace.json.gz"

    async def run():
        kwargs = dict(log_context="test", rollout_id="rollout-1", quiet=True, priority="background")
        assert await e2b_executor._download_trace(sandbox, destination, **kwargs) is None
        assert not destination.exists()
        assert any("status=failed" in message for message in caplog.messages)
        assert not any("status=completed" in message for message in caplog.messages)
        assert not any("stage=driver.trace-processing " in message for message in caplog.messages)
        trace = await e2b_executor._download_trace(sandbox, destination, **kwargs)
        assert trace.rollout_id == "rollout-1"

    with caplog.at_level(logging.DEBUG, logger="northrl.utils.profiling"):
        asyncio.run(run())

    assert destination.read_bytes() == payload
    profiles = [
        record for record in caplog.records if "stage=driver.file-download " in record.message
    ]
    assert len(profiles) == 2
    assert all(record.levelno == logging.DEBUG for record in profiles)
    assert any("status=completed" in record.message for record in profiles)
    assert sum("stage=driver.trace-processing " in message for message in caplog.messages) == 1
    assert read.await_count == 2


@pytest.mark.parametrize(
    "error, options, propagates, profile_visible",
    [
        (FileNotFoundError("no trace"), {"priority": "critical"}, False, True),
        (FileNotFoundError("no trace"), {"quiet": False}, False, False),
        (FileNotFoundError("no trace"), {"raise_read_errors": True}, True, False),
        (TimeoutError("read timed out"), {}, False, True),
        (ValueError("invalid trace"), {}, False, True),
        (asyncio.CancelledError(), {}, True, False),
    ],
)
def test_trace_poll_reports_errors_without_duplicate_profiles(
    tmp_path, caplog, capsys, error, options, propagates, profile_visible
):
    read = AsyncMock(side_effect=error)
    sandbox = SimpleNamespace(sandbox_id="sandbox-1", files=SimpleNamespace(read=read))
    kwargs = dict(log_context="test", rollout_id="rollout-1", quiet=True, priority="background")
    kwargs.update(options)

    async def run():
        download = e2b_executor._download_trace(sandbox, tmp_path / "trace.json.gz", **kwargs)
        if propagates:
            with pytest.raises(type(error)):
                await download
        else:
            assert await download is None

    with caplog.at_level(logging.INFO, logger="northrl.utils.profiling"):
        asyncio.run(run())

    failures = [
        record for record in caplog.records if "stage=driver.file-download " in record.message
    ]
    assert len(failures) == int(profile_visible)
    if profile_visible:
        assert failures[0].levelno == logging.WARNING
        assert "status=failed" in failures[0].message
        assert f"error_type='{type(error).__name__}'" in failures[0].message
    assert not any("stage=driver.trace-processing " in message for message in caplog.messages)
    if not kwargs["quiet"]:
        assert "partial trace unavailable" in capsys.readouterr().err


def test_foreign_locally_synced_trace_is_rejected(tmp_path: Path) -> None:
    trace_output = tmp_path / "trace.json"
    trace_output.write_bytes(dump_artifact(encode_trace(_trace("foreign"))))
    assert (
        e2b_executor._load_locally_synced_trace(
            trace_output,
            log_context="test",
            rollout_id="rollout-1",
        )
        is None
    )


@pytest.mark.parametrize("version", [None, 1, 3])
def test_trace_download_and_local_recovery_reject_non_v2_files(tmp_path, version):
    payload = _trace("rollout-1").model_dump(mode="json")
    if version is not None:
        payload["schema_version"] = version
    payload = gzip.compress(json.dumps(payload).encode())
    sandbox = _FakeSandbox(dump_artifact(encode_result(_result("rollout-1"))), payload, None, None)
    downloaded = tmp_path / "downloaded.json"

    assert (
        asyncio.run(
            e2b_executor._download_trace(
                sandbox,
                downloaded,
                log_context="test",
                rollout_id="rollout-1",
            )
        )
        is None
    )
    assert not downloaded.exists()
    downloaded.write_bytes(payload)
    assert (
        e2b_executor._load_locally_synced_trace(
            downloaded, log_context="test", rollout_id="rollout-1"
        )
        is None
    )


def test_environment_failure_keeps_trace_metrics_without_decoding(monkeypatch):
    compact = encode_trace(_prefix_trace("rollout-1"))
    decode = Mock(side_effect=AssertionError("environment failures must not expand training data"))
    monkeypatch.setattr(e2b_executor, "decode_trace", decode)

    result = e2b_executor._failure_result(
        RolloutInput(rollout_id="rollout-1", task={}),
        TimeoutError("result download timed out"),
        status="env_error",
        trace=compact,
    )

    assert result.segments == []
    assert result.messages == []
    assert result.reward == 0.0
    assert result.metrics["tool_calls_emitted"] == 3.0
    assert result.metrics["e2b_timed_out"] == 1.0
    assert e2b_executor._with_failure_trace(result, compact) == result
    decode.assert_not_called()


def test_periodic_trace_sync_writes_before_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_trace = _prefix_trace("rollout-1")
    sandbox = _FakeSandbox(
        dump_artifact(encode_result(_result("rollout-1"))),
        dump_artifact(encode_trace(expected_trace)),
        None,
        None,
    )
    trace_output = tmp_path / "periodic-trace.json"
    monkeypatch.setattr(e2b_executor, "_TRACE_SYNC_INTERVAL_SECONDS", 0)
    decode = Mock(side_effect=AssertionError("periodic trace sync must not expand training arrays"))
    monkeypatch.setattr(e2b_executor, "decode_trace", decode)

    async def scenario() -> None:
        task = asyncio.create_task(
            e2b_executor._sync_trace_periodically(
                sandbox,
                trace_output,
                log_context="test",
                rollout_id="rollout-1",
            )
        )
        while not trace_output.exists():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    payload = trace_output.read_bytes()
    text = gzip.decompress(payload).decode()
    saved_trace = load_trace_gzip(payload)
    assert saved_trace == encode_trace(expected_trace)
    assert text.count("\n") == 0
    assert json.loads(text)["schema_version"] == 2
    assert all("tokens" not in segment for segment in json.loads(text)["segments"])
    assert all("loss_mask" not in segment for segment in json.loads(text)["segments"])
    assert (
        e2b_executor._load_locally_synced_trace(
            trace_output, log_context="test", rollout_id="rollout-1"
        )
        == saved_trace
    )
    decode.assert_not_called()
    assert [path for path in sandbox.files.reads if path != e2b_executor._METRICS_PATH] == [
        e2b_executor._TRACE_PATH
    ]


@pytest.mark.parametrize("exit_code", [1, 126, 127, 137])
def test_python_launch_failure_without_state_does_not_rerun_dispatched_agent(
    exit_code: int,
) -> None:
    error = RuntimeError("configured Python is unavailable")
    error.exit_code = exit_code
    _FakeAsyncSandbox.entrypoint_error = error
    _FakeAsyncSandbox.result_error = FileNotFoundError("no result")
    result = _run()
    assert result.status == "env_error"
    assert result.error_message == "configured Python is unavailable"
    assert result.segments == []
    assert len(_FakeAsyncSandbox.instances) == 1
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


@pytest.mark.usefixtures("bounded_io")
def test_exit_127_after_agent_phase_is_still_task_failure() -> None:
    error = RuntimeError("agent removed command")
    error.exit_code = 127
    _FakeAsyncSandbox.entrypoint_error = error
    _FakeAsyncSandbox.result_error = FileNotFoundError("no result")
    _FakeAsyncSandbox.state_json = _state("agent", status="task_failed")
    result = _run()
    assert result.status == "task_failed"
    assert len(_FakeAsyncSandbox.instances) == 1


@pytest.mark.usefixtures("bounded_io")
def test_disconnected_command_preserves_trusted_task_failure_and_error() -> None:
    _FakeAsyncSandbox.entrypoint_error = ConnectionError("lost command stream")
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.state_json = _state(
        "agent",
        status="task_failed",
        error_type="AgentExecutionError",
        error_message="trusted runtime reports agent failure",
    )
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))

    result = _run()

    assert result.status == "task_failed"
    assert result.error_type == "AgentExecutionError"
    assert result.error_message == "trusted runtime reports agent failure"
    assert result.reward == 0.0
    assert _dense_segments(result.segments) == _trace("rollout-1").segments
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert not any("--phase evaluate" in command for command, _ in sandbox.commands.calls)
    assert sandbox.killed


@pytest.mark.usefixtures("bounded_io")
def test_command_api_connection_failure_is_environment_error_even_during_agent() -> None:
    _FakeAsyncSandbox.entrypoint_error = ConnectionError("E2B connection lost")
    _FakeAsyncSandbox.result_error = FileNotFoundError("no result")
    _FakeAsyncSandbox.state_json = _state("agent")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    result = _run()
    assert result.status == "env_error"
    assert result.error_message == "E2B connection lost"
    assert result.segments == []
    assert len(_FakeAsyncSandbox.instances) == 1


def test_completed_result_survives_command_connection_failure() -> None:
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = ConnectionError("E2B connection lost after completion")
    assert _without_driver_runtime(_run()) == _result("rollout-1")
    assert len(_FakeAsyncSandbox.instances) == 1


def test_returned_task_failure_recovers_trace_without_local_destination() -> None:
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(_failed("task_failed")))
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    result = _run()
    assert result.status == "task_failed"
    assert _dense_segments(result.segments) == _trace("rollout-1").segments
    assert result.tools == _trace("rollout-1").tools
    assert len(_FakeAsyncSandbox.instances) == 1


def test_invalid_retry_setting_clears_preexisting_trace(tmp_path: Path) -> None:
    trace_output = tmp_path / "trace.json"
    trace_output.write_bytes(dump_artifact(encode_trace(_trace("rollout-1"))))
    config = _config()
    config.rollout.execution.env_error_retries = "bad"
    result = _run(config, trace_output_path=trace_output)
    assert result.status == "env_error"
    assert not trace_output.exists()


def test_real_transport_exception_types_are_distinguished_from_timeouts() -> None:
    import httpx
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    assert e2b_executor._is_command_transport_error(httpx.ConnectError("connection refused"))
    assert e2b_executor._is_command_transport_error(ConnectError(Code.UNAVAILABLE, "disconnected"))
    assert e2b_executor._is_command_transport_error(httpx.ReadTimeout("read timed out"))
    assert not e2b_executor._is_command_transport_error(TimeoutError("attempt budget expired"))
    deadline = ConnectError(Code.DEADLINE_EXCEEDED, "command deadline")
    deadline.__cause__ = TimeoutError("deadline expired")
    assert not e2b_executor._is_command_transport_error(deadline)


@pytest.mark.parametrize("command_fails", [False, True])
def test_caller_cancellation_during_trace_sync_cleanup_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command_fails: bool,
) -> None:
    original_run = _FakeCommands.run

    async def run_then_cancel(self, command: str, **kwargs: Any) -> None:
        await original_run(self, command, **kwargs)
        if command in (e2b_executor._EXTRACT_COMMAND, e2b_executor._BOOTSTRAP_COMMAND):
            return
        current_task = asyncio.current_task()
        assert current_task is not None
        asyncio.get_running_loop().call_soon(current_task.cancel)
        if command_fails:
            raise RuntimeError("command failed as caller cancelled")

    monkeypatch.setattr(_FakeCommands, "run", run_then_cancel)

    async def scenario() -> None:
        rollout_task = asyncio.create_task(
            e2b_executor.run_rollout_in_e2b(
                RolloutInput(rollout_id="rollout-1", task={}),
                _config(),
                gateway=_gateway(),
                template="swebench-template",
                timeout_seconds=3600,
                trace_output_path=tmp_path / "trace.json",
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await rollout_task
        assert rollout_task.cancelled()
        assert rollout_task.cancelling() == 1

    asyncio.run(scenario())
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sandbox.killed
    # Only best-effort diagnostics run during cleanup; no task result is recovered.
    assert [path for path in sandbox.files.reads if path != e2b_executor._METRICS_PATH] == [
        e2b_executor._REQUESTS_PATH
    ]


@pytest.fixture
def live_metrics_files(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read = _FakeFiles.read

    async def read(self, path: str, *, user, format="text", request_timeout=None):
        assert user == "root"
        if path == e2b_executor._METRICS_PATH:
            self.reads.append(path)
            self.events.append(("read", path))
            return json.dumps(
                {
                    "rollout_id": "rollout-1",
                    "heartbeat": 1,
                    "phase": "agent",
                    "metrics": {"trace_calls": 2},
                }
            )
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)


def test_live_metrics_without_debug_report_final_snapshot_and_lifecycle(live_metrics_files) -> None:
    events = []
    assert _without_driver_runtime(_run(event_callback=events.append)) == _result("rollout-1")
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "attempt_started"
    assert kinds[-3:] == ["sandbox_closed", "attempt_finished", "rollout_finished"]
    assert events[-3]["confirmed"] is True
    assert all(event["rollout_id"] == "rollout-1" and event["attempt"] == 1 for event in events)
    assert events[-1]["attempts"] == 1
    assert events[-2]["duration_seconds"] >= 0
    assert events[-2]["timed_out"] is False
    snapshot = next(event for event in events if event["kind"] == "snapshot")
    assert snapshot["metrics"] == {"trace_calls": 2.0}
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._METRICS_PATH in sandbox.files.reads
    assert e2b_executor._TRACE_PATH not in sandbox.files.reads
    payload = json.loads(
        next(data for path, data in sandbox.files.writes if path == e2b_executor._INPUT_PATH)
    )
    assert "live_metrics_enabled" not in payload["config"]["rollout"]
    assert "live_metrics_interval_seconds" not in payload["config"]["rollout"]


def test_live_metrics_poll_during_bootstrap_before_result(
    monkeypatch: pytest.MonkeyPatch, live_metrics_files
) -> None:
    events = []
    original_run = _FakeCommands.run

    async def run(self, command: str, **kwargs: Any) -> None:
        await original_run(self, command, **kwargs)
        if command == e2b_executor._BOOTSTRAP_COMMAND:
            for _ in range(100):
                if any(event["kind"] == "snapshot" for event in events):
                    break
                await asyncio.sleep(0)
            assert any(event["kind"] == "snapshot" for event in events)
            assert not any(event.get("phase") == "result-download" for event in events)

    monkeypatch.setattr(_FakeCommands, "run", run)
    assert _run(event_callback=events.append).status == "success"


def test_metrics_callback_failure_does_not_change_rollout(live_metrics_files) -> None:
    def callback(event) -> None:
        raise RuntimeError("collector failed")

    assert _without_driver_runtime(_run(event_callback=callback)) == _result("rollout-1")
    assert _FakeAsyncSandbox.instances[0].killed


def test_metrics_cover_failed_attempts_and_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, live_metrics_files
) -> None:
    first = _FakeSandbox(
        dump_artifact(encode_result(_failed())),
        None,
        RuntimeError("bootstrap failed"),
        RuntimeError("kill failed"),
    )
    second = _FakeSandbox(dump_artifact(encode_result(_result("rollout-1"))), None, None, None)
    monkeypatch.setattr(_FakeAsyncSandbox, "create", AsyncMock(side_effect=[first, second]))
    events = []
    assert _run(event_callback=events.append).status == "success"
    finished = [event for event in events if event["kind"] == "attempt_finished"]
    assert [(event["attempt"], event["status"]) for event in finished] == [
        (1, "env_error"),
        (2, "success"),
    ]
    assert [event["confirmed"] for event in events if event["kind"] == "sandbox_closed"] == [
        False,
        True,
    ]
    assert events[-1]["attempts"] == 2


def test_live_metrics_cancellation_reports_outcome_and_closes_sandbox(live_metrics_files) -> None:
    _FakeAsyncSandbox.entrypoint_error = asyncio.CancelledError("caller cancelled")
    events = []
    with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
        _run(event_callback=events.append)
    assert events[-3]["kind"] == "sandbox_closed"
    assert events[-3]["confirmed"] is True
    assert events[-2]["status"] == "cancelled"
    assert events[-1]["status"] == "cancelled"
    assert len(_FakeAsyncSandbox.instances) == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"rollout_id": "different-run"},
        {"heartbeat": True},
        {"heartbeat": -1},
        {"phase": ""},
        {"metrics": {"count": float("nan")}},
        {"metrics": {"count": float("inf")}},
        {"metrics": {"count": "bad"}},
        {"metrics": {"count": True}},
    ],
)
def test_invalid_live_metrics_are_observation_errors(updates) -> None:
    snapshot = {"rollout_id": "rollout-1", "heartbeat": 1, "phase": "agent", "metrics": {}}
    snapshot.update(updates)
    files = SimpleNamespace(read=AsyncMock(return_value=json.dumps(snapshot)))
    events = []
    asyncio.run(
        e2b_executor._read_live_metrics(
            SimpleNamespace(files=files),
            rollout_id="rollout-1",
            event_callback=events.append,
            timeout_seconds=0.1,
        )
    )
    assert [event["kind"] for event in events] == ["poll_error"]


def test_live_metrics_read_timeout_is_bounded() -> None:
    cancelled = []

    async def read(path: str, *, user, format="text", request_timeout=None):
        assert user == "root"
        try:
            await asyncio.sleep(60)
            return "{}"
        finally:
            cancelled.append(True)

    events = []
    asyncio.run(
        e2b_executor._read_live_metrics(
            SimpleNamespace(files=SimpleNamespace(read=read)),
            rollout_id="rollout-1",
            event_callback=events.append,
            timeout_seconds=0.001,
        )
    )
    assert cancelled == [True]
    assert events == [{"kind": "poll_error", "error_type": "TimeoutError"}]


def test_live_metrics_poll_every_five_minutes_with_bounded_reads(monkeypatch) -> None:
    read = AsyncMock()
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError])
    monkeypatch.setattr(e2b_executor, "_read_live_metrics", read)
    monkeypatch.setattr(e2b_executor.asyncio, "sleep", sleep)
    sandbox = object()
    events = []
    callback = events.append
    state = e2b_executor._LiveMetricsState()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            e2b_executor._poll_live_metrics(
                sandbox,
                rollout_id="rollout-1",
                event_callback=callback,
                state=state,
            )
        )

    assert [call.args for call in sleep.await_args_list] == [(300.0,), (300.0,)]
    assert read.await_count == 2
    for call in read.await_args_list:
        assert call.args == (sandbox,)
        assert call.kwargs == {
            "rollout_id": "rollout-1",
            "event_callback": callback,
            "timeout_seconds": 5.0,
            "state": state,
        }


def test_timeout_failure_preserves_trace_metrics() -> None:
    trace = _trace("rollout-1").model_copy(update={"metrics": {"trace_calls": 3.0}})
    result = e2b_executor._failure_result(
        RolloutInput(rollout_id="rollout-1", task={}),
        TimeoutError("timed out"),
        status="task_failed",
        trace=encode_trace(trace),
    )
    assert result.metrics["trace_calls"] == 3.0
    assert result.metrics["e2b_timed_out"] == 1.0


@pytest.mark.parametrize("command_fails", [False, True])
def test_caller_cancellation_during_live_metrics_cleanup_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, live_metrics_files, command_fails: bool
) -> None:
    original_run = _FakeCommands.run
    events = []

    async def run_then_cancel(self, command: str, **kwargs: Any) -> None:
        await original_run(self, command, **kwargs)
        if command in (e2b_executor._EXTRACT_COMMAND, e2b_executor._BOOTSTRAP_COMMAND):
            return
        current_task = asyncio.current_task()
        assert current_task is not None
        asyncio.get_running_loop().call_soon(current_task.cancel)
        if command_fails:
            raise RuntimeError("command failed while cancelled")

    monkeypatch.setattr(_FakeCommands, "run", run_then_cancel)
    with pytest.raises(asyncio.CancelledError):
        _run(event_callback=events.append)
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed
    assert events[-1]["status"] == "cancelled"
    assert any(event["kind"] == "sandbox_closed" and event["confirmed"] for event in events)


def test_final_segment_event_uses_returned_trajectories_after_retry(monkeypatch):
    result = _result("rollout-1")
    result = result.model_copy(update={"segments": result.segments * 3})
    execute = AsyncMock(
        side_effect=[
            RemoteRolloutResult(rollout_id="rollout-1", status="env_error", reward=0.0),
            result,
        ]
    )
    monkeypatch.setattr(e2b_executor, "_run_rollout_in_e2b_once", execute)
    events = []
    assert _without_driver_runtime(_run(event_callback=events.append)) == result
    final_events = [event for event in events if event["kind"] == "rollout_finished"]
    assert len(final_events) == 1
    assert final_events[0]["trajectory_segments"] == 3
    assert final_events[0]["attempts"] == 2
    assert not any(event["kind"] == "snapshot" for event in events)


def test_final_tool_metrics_survive_missed_polling_and_failed_retry(monkeypatch):
    failed = _failed().model_copy(
        update={
            "metrics": {
                "tool_results_observed": 1.0,
                "tool_result_errors": 1.0,
                "tool_error_command_not_found": 1.0,
                "e2b_timed_out": 1.0,
            }
        }
    )
    result = _result("rollout-1").model_copy(
        update={"metrics": {"tool_results_observed": 2.0, "tool_result_errors": 0.0}}
    )
    monkeypatch.setattr(
        e2b_executor, "_run_rollout_in_e2b_once", AsyncMock(side_effect=[failed, result])
    )
    events = []
    assert _without_driver_runtime(_run(event_callback=events.append)) == result
    snapshots = [event for event in events if event["kind"] == "snapshot"]
    assert [event["attempt"] for event in snapshots] == [1, 2]
    assert snapshots[0]["metrics"] == {
        "tool_results_observed": 1.0,
        "tool_result_errors": 1.0,
        "tool_error_command_not_found": 1.0,
    }
    assert snapshots[1]["metrics"] == result.metrics
    for snapshot in snapshots:
        finished = next(
            event
            for event in events
            if event["kind"] == "attempt_finished" and event["attempt"] == snapshot["attempt"]
        )
        assert events.index(snapshot) < events.index(finished)
        assert snapshot["rollout_id"] == "rollout-1"
        assert "heartbeat" not in snapshot
    assert events[-1]["status"] == "success"


@pytest.mark.parametrize("prepacked", [False, True])
@pytest.mark.parametrize(
    "invalid_stage",
    ["missing", "json", "object", "attempt", "rollout", "phase", "failure", "missing_attempt"],
)
def test_invalid_completion_record_prevents_evaluation_upload(
    monkeypatch, invalid_stage, prepacked
) -> None:
    original_read = _FakeFiles.read

    async def read(self, path, *, user, format="text", request_timeout=None):
        data = await original_read(self, path, user=user, format=format)
        if path != e2b_executor._STATE_PATH:
            return data
        if invalid_stage == "missing":
            raise FileNotFoundError(path)
        if invalid_stage == "json":
            return "not JSON"
        if invalid_stage == "object":
            return "[]"
        record = json.loads(data)
        if invalid_stage == "attempt":
            record["attempt_id"] = "previous-attempt"
        elif invalid_stage == "rollout":
            record["rollout_id"] = "different-rollout"
        elif invalid_stage == "phase":
            record["phase"] = "agent"
        elif invalid_stage == "failure":
            record["status"] = "env_error"
        elif invalid_stage == "missing_attempt":
            record.pop("attempt_id")
        return json.dumps(record)

    monkeypatch.setattr(_FakeFiles, "read", read)
    config = _config()
    config.rollout.execution.env_error_retries = 0
    artifact_options = {}
    if prepacked:
        artifact_options["prepared_artifacts"] = e2b_executor.prepare_e2b_artifacts(config)
    result = _run(config, **artifact_options)
    assert result.status == "env_error"
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._EVALUATION_ARCHIVE_PATH not in dict(sandbox.files.writes)
    assert not any("--phase evaluate" in command for command, _ in sandbox.commands.calls)
    assert sandbox.killed


@pytest.mark.parametrize("exit_code", [1, 124])
@pytest.mark.parametrize("has_trace", [False, True])
def test_committed_agent_exit_status_does_not_gate_evaluation(
    monkeypatch, exit_code, has_trace
) -> None:
    original_run = _FakeCommands.run
    expected = RemoteRolloutResult(
        rollout_id="rollout-1",
        reward=1.0,
        segments=_result("rollout-1").segments if has_trace else [],
        metrics={"tests_passed": 1.0},
    )
    _FakeAsyncSandbox.result_json = ""

    async def run(self, command, **kwargs):
        await original_run(self, command, **kwargs)
        if "--phase agent" in command:
            record = json.loads(gzip.decompress(self.files.stage_json))
            record["agent_result"]["exit_code"] = exit_code
            if not has_trace:
                record["trace"] = None
            self.files.stage_json = dump_artifact(CompactAgentStageResult.model_validate(record))
        elif "--phase evaluate" in command:
            self.files._result_json = dump_artifact(encode_result(expected))

    monkeypatch.setattr(_FakeCommands, "run", run)
    result = _run()

    assert _without_driver_runtime(result) == expected
    assert result.status == "success"
    assert result.reward == 1.0
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    events = sandbox.files.events
    stage_index = events.index(("read", e2b_executor._STATE_PATH))
    upload_index = events.index(("write", e2b_executor._EVALUATION_ARCHIVE_PATH))
    evaluation_index = next(i for i, event in enumerate(events) if "--phase evaluate" in event[1])
    assert stage_index < upload_index < evaluation_index
    assert sandbox.killed


@pytest.mark.parametrize("trusted_agent_phase", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_uncommitted_agent_never_uploads_assets_or_accepts_unscored_success(
    trusted_agent_phase: bool,
) -> None:
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("agent command failed")
    _FakeAsyncSandbox.state_json = (
        _state("agent", status="task_failed") if trusted_agent_phase else None
    )
    # Even a valid-looking success result cannot replace the required agent stage.
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(_result("rollout-1")))
    result = _run()
    assert result.status == ("task_failed" if trusted_agent_phase else "env_error")
    assert result.reward == 0.0
    assert "swebench_resolved" not in result.metrics
    assert len(_FakeAsyncSandbox.instances) == 1
    for sandbox in _FakeAsyncSandbox.instances:
        assert e2b_executor._EVALUATION_ARCHIVE_PATH not in dict(sandbox.files.writes)
        assert not any("--phase evaluate" in command for command, _ in sandbox.commands.calls)
        assert sandbox.killed


@pytest.mark.usefixtures("bounded_io")
def test_evaluation_upload_failure_does_not_run_evaluator(monkeypatch) -> None:
    original_write = _FakeFiles.write

    async def write(self, path, data, *, user):
        assert user == "root"
        if path == e2b_executor._EVALUATION_ARCHIVE_PATH:
            assert self.stage_json is not None
            raise ConnectionError("evaluation upload failed")
        await original_write(self, path, data, user=user)

    monkeypatch.setattr(_FakeFiles, "write", write)
    config = _config()
    config.rollout.execution.env_error_retries = 0
    result = _run(config)
    assert result.status == "env_error"
    sandbox = _FakeAsyncSandbox.instances[0]
    assert not any("--phase evaluate" in command for command, _ in sandbox.commands.calls)
    assert sandbox.killed


def test_evaluator_crash_is_environment_failure_without_training_trace() -> None:
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("broken evaluator")
    _FakeAsyncSandbox.result_error = FileNotFoundError("no score")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    config = _config()
    config.rollout.execution.env_error_retries = 0
    result = _run(config)
    assert result.status == "env_error"
    assert result.segments == []
    assert result.reward == 0.0
    assert _FakeAsyncSandbox.instances[0].killed


def test_source_archive_excludes_private_python_assets_inside_package(
    monkeypatch, tmp_path
) -> None:
    package = tmp_path / "northrl"
    evaluation_root = package / "evaluation"
    task_dir = evaluation_root / "task-1"
    task_dir.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (task_dir / "secret_answers.py").write_text("ANSWER = 42")
    monkeypatch.setattr(e2b_executor, "_PACKAGE_ROOT", package)
    entrypoint = e2b_executor._EntrypointSpec(
        script_path=package / "__init__.py", task_files_root=evaluation_root
    )
    data = e2b_executor._build_source_archive(entrypoint)
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
        assert "northrl-src/northrl/__init__.py" in archive.getnames()
        assert not any("evaluation" in name for name in archive.getnames())
    private_script = task_dir / "secret_answers.py"
    script_entrypoint = e2b_executor._EntrypointSpec(
        script_path=private_script,
        task_files_root=evaluation_root,
    )
    with pytest.raises(ValueError, match="must not be inside"):
        e2b_executor._build_source_archive(script_entrypoint)


def test_agent_and_evaluator_have_independent_stage_timeouts(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        e2b_executor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000.0 + clock[0]),
    )
    monkeypatch.setattr(wire, "time", e2b_executor.time)
    original_run = _FakeCommands.run

    async def run(self, command, **kwargs):
        await original_run(self, command, **kwargs)
        clock[0] += 100.0

    monkeypatch.setattr(_FakeCommands, "run", run)
    assert _without_driver_runtime(_run()) == _result("rollout-1")
    timeouts = [kwargs["timeout"] for _, kwargs in _FakeAsyncSandbox.instances[0].commands.calls]
    assert timeouts == [600, 500, 4600, 600, 605]
    evaluation_argv = shlex.split(_FakeAsyncSandbox.instances[0].commands.calls[-1][0])
    assert float(evaluation_argv[4]) == 600
    assert float(evaluation_argv[evaluation_argv.index("--timeout-seconds") + 1]) == 600


def test_completed_agent_outer_deadline_does_not_consume_evaluation_budget(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        e2b_executor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000.0 + clock[0]),
    )
    monkeypatch.setattr(wire, "time", e2b_executor.time)
    original_run = _FakeCommands.run

    async def run(self, command, **kwargs):
        await original_run(self, command, **kwargs)
        if "--phase agent" in command:
            clock[0] = 4801.0

    monkeypatch.setattr(_FakeCommands, "run", run)
    config = _config()
    config.rollout.execution.env_error_retries = 0
    assert _run(config).status == "success"
    sandbox = _FakeAsyncSandbox.instances[0]
    assert e2b_executor._EVALUATION_ARCHIVE_PATH in dict(sandbox.files.writes)
    assert sandbox.commands.calls[-1][1]["timeout"] == 605
    assert sandbox.killed


@pytest.mark.usefixtures("bounded_io")
def test_hung_failure_recovery_is_bounded_and_still_kills_sandbox(monkeypatch) -> None:
    async def hung_read(self, path, *, user, format="text", request_timeout=None):
        await asyncio.Event().wait()

    monkeypatch.setattr(_FakeFiles, "read", hung_read)
    monkeypatch.setattr(e2b_executor, "_LIVE_METRICS_READ_TIMEOUT_SECONDS", 0.01)
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("agent command failed")
    config = _config()
    config.rollout.execution.env_error_retries = 0
    assert _run(config).status == "env_error"
    assert _FakeAsyncSandbox.instances[0].killed


def test_hung_kill_is_bounded_and_reported_without_losing_result(monkeypatch) -> None:
    async def hung_kill(self):
        self.killed = True
        await asyncio.Event().wait()

    monkeypatch.setattr(_FakeSandbox, "kill", hung_kill)
    monkeypatch.setattr(e2b_executor, "_SANDBOX_KILL_TIMEOUT_SECONDS", 0.01)
    events = []
    assert _without_driver_runtime(_run(event_callback=events.append)) == _result("rollout-1")
    closed = next(event for event in events if event["kind"] == "sandbox_closed")
    assert closed["confirmed"] is False


def test_cancellation_during_evaluation_closes_sandbox_without_retry() -> None:
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = asyncio.CancelledError("caller cancelled evaluation")
    with pytest.raises(asyncio.CancelledError, match="caller cancelled evaluation"):
        _run()
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed


def test_explicit_workspace_is_passed_to_platform_not_task_cli() -> None:
    config = _config()
    config.rollout.execution.entrypoint.workspace = "/workspace/my task"
    assert _without_driver_runtime(_run(config)) == _result("rollout-1")
    commands = _FakeAsyncSandbox.instances[0].commands.calls
    assert "--workspace '/workspace/my task'" in commands[2][0]
    assert "--workspace '/workspace/my task'" in commands[-1][0]
    assert "northrl.remote_rollout.e2b_runtime" in commands[2][0]


@pytest.mark.parametrize("segment_count", [0, 2])
def test_context_exhaustion_emits_final_reason_without_retry(monkeypatch, segment_count):
    details = {"prompt_tokens": 4096, "requested_max_tokens": 512, "max_sequence_length": 4096}
    result = RemoteRolloutResult(
        rollout_id="rollout-1",
        segments=_result("rollout-1").segments * segment_count,
        reward=1.0,
        termination_reason="context_length_exceeded",
        length_details=details,
    )
    execute = AsyncMock(return_value=result)
    monkeypatch.setattr(e2b_executor, "_run_rollout_in_e2b_once", execute)
    events = []
    assert _without_driver_runtime(_run(event_callback=events.append)) == result
    execute.assert_awaited_once()
    final_events = [event for event in events if event["kind"] == "rollout_finished"]
    assert len(final_events) == 1
    assert final_events[0]["termination_reason"] == "context_length_exceeded"
    assert final_events[0]["length_details"] == details
    assert final_events[0]["trajectory_segments"] == segment_count


def test_recovered_final_result_preserves_context_reason_and_reward():
    expected = _result("rollout-1").model_copy(
        update={
            "termination_reason": "context_length_exceeded",
            "length_details": {"prompt_tokens": 4096},
        }
    )
    _FakeAsyncSandbox.result_json = dump_artifact(encode_result(expected))
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = RuntimeError("command failed after result was written")
    result = _run()
    assert _without_driver_runtime(result) == expected
    assert result.reward == 1.0
    assert len(_FakeAsyncSandbox.instances) == 1


@pytest.mark.parametrize(
    "error", [RuntimeError("evaluator failed"), ConnectionError("disconnected")]
)
@pytest.mark.usefixtures("bounded_io")
def test_phase_recovery_preserves_context_reason_with_separate_evaluation_failure(error):
    details = {"prompt_tokens": 4096, "requested_max_tokens": 512, "max_sequence_length": 4096}
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = error
    _FakeAsyncSandbox.state_json = _state(
        "evaluator",
        termination_reason="context_length_exceeded",
        length_details=details,
    )
    config = _config()
    config.rollout.execution.env_error_retries = 0
    result = _run(config)
    assert result.status == "env_error"
    assert result.termination_reason == "context_length_exceeded"
    assert result.length_details == details


def test_run_rollout_in_e2b_rejects_legacy_result():
    legacy = _result("rollout-1").model_dump()
    legacy["trajectories"] = legacy.pop("segments")
    _FakeAsyncSandbox.result_json = gzip.compress(json.dumps(legacy).encode())

    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            _config(),
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
        )
    )

    assert result.status == "env_error"
    assert result.error_type == "ValidationError"
    assert result.segments == []
    assert result.reward == 0.0


@pytest.mark.parametrize("legacy_value", [None, 0, 1])
def test_direct_executor_rejects_legacy_timeout_retries_before_sandbox_create(legacy_value):
    config = _config()
    config.rollout.execution.timeout_retries = legacy_value

    result = _run(config)

    assert result.status == "env_error"
    assert result.error_type == "ValueError"
    assert "timeout_retries is no longer supported" in result.error_message
    assert "env_error_retries" in result.error_message
    assert _FakeAsyncSandbox.instances == []


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("transport", ["rpc", "http"])
@pytest.mark.usefixtures("bounded_io")
def test_transport_timeout_does_not_rerun_agent_or_emit_training_segments(wrapped, transport):
    import httpx
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    if transport == "rpc":
        error = ConnectError(Code.UNAVAILABLE, "sandbox connection timed out")
        error.__cause__ = TimeoutError("transport timeout; command budget not exhausted")
    else:
        error = httpx.ReadTimeout("sandbox connection timed out")
    if wrapped:
        wrapper = RuntimeError("command stream failed")
        wrapper.__cause__ = error
        error = wrapper
    _FakeAsyncSandbox.entrypoint_error = error
    _FakeAsyncSandbox.result_error = FileNotFoundError("agent has not completed")
    _FakeAsyncSandbox.state_json = _state("agent")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))

    result = _run()
    assert result.status == "env_error"
    assert result.segments == []
    assert len(_FakeAsyncSandbox.instances) == 1
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


@pytest.mark.parametrize("has_timeout_cause", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_explicit_rpc_command_deadline_preserves_task_failure(has_timeout_cause):
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    error = ConnectError(Code.DEADLINE_EXCEEDED, "command budget expired")
    if has_timeout_cause:
        error.__cause__ = TimeoutError("command budget expired")
    _FakeAsyncSandbox.entrypoint_error = error
    _FakeAsyncSandbox.result_error = FileNotFoundError("no final result")
    _FakeAsyncSandbox.state_json = _state("agent", status="task_failed")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))

    result = _run()
    assert result.status == "task_failed"
    assert result.reward == 0.0
    assert _dense_segments(result.segments) == _trace("rollout-1").segments
    assert len(_FakeAsyncSandbox.instances) == 1
    assert _FakeAsyncSandbox.instances[0].killed


@pytest.mark.parametrize("sdk_error", [False, True])
def test_missing_metrics_are_pending_only_before_first_snapshot(sdk_error):
    class FileNotFoundException(Exception):
        pass

    FileNotFoundException.__module__ = "e2b.exceptions"
    missing = FileNotFoundException("missing") if sdk_error else FileNotFoundError("missing")
    snapshot = json.dumps(
        {"rollout_id": "rollout-1", "heartbeat": 1, "phase": "agent", "metrics": {}}
    )
    sandbox = SimpleNamespace(
        files=SimpleNamespace(read=AsyncMock(side_effect=[missing, missing, snapshot, missing]))
    )
    state = e2b_executor._LiveMetricsState()
    events = []

    async def scenario():
        for _ in range(4):
            await e2b_executor._read_live_metrics(
                sandbox,
                rollout_id="rollout-1",
                event_callback=events.append,
                timeout_seconds=1,
                state=state,
            )

    asyncio.run(scenario())
    assert [event["kind"] for event in events] == ["snapshot", "poll_error"]
    assert events[-1]["error_type"] == type(missing).__name__
    assert state.received_snapshot


@pytest.mark.parametrize("failure", ["connection", "sandbox_missing", "invalid_json"])
def test_first_metrics_read_reports_real_observation_failures(failure):
    class SandboxNotFoundException(Exception):
        pass

    SandboxNotFoundException.__module__ = "e2b.exceptions"
    value = {
        "connection": ConnectionError("offline"),
        "sandbox_missing": SandboxNotFoundException("sandbox gone"),
        "invalid_json": "{bad",
    }[failure]
    read = (
        AsyncMock(side_effect=value)
        if isinstance(value, Exception)
        else AsyncMock(return_value=value)
    )
    state = e2b_executor._LiveMetricsState()
    events = []
    asyncio.run(
        e2b_executor._read_live_metrics(
            SimpleNamespace(files=SimpleNamespace(read=read)),
            rollout_id="rollout-1",
            event_callback=events.append,
            timeout_seconds=1,
            state=state,
        )
    )
    assert [event["kind"] for event in events] == ["poll_error"]
    assert not state.received_snapshot


def test_final_metrics_read_remembers_first_snapshot(monkeypatch):
    original_read = _FakeFiles.read
    original_run = _FakeCommands.run
    metrics_reads = 0

    async def read(self, path, *, user, format="text", request_timeout=None):
        assert user == "root"
        nonlocal metrics_reads
        if path != e2b_executor._METRICS_PATH:
            return await original_read(self, path, user=user, format=format)
        metrics_reads += 1
        if metrics_reads > 1:
            raise FileNotFoundError(path)
        return json.dumps(
            {"rollout_id": "rollout-1", "heartbeat": 1, "phase": "agent", "metrics": {}}
        )

    async def run(self, command, **kwargs):
        await original_run(self, command, **kwargs)
        if command == e2b_executor._BOOTSTRAP_COMMAND:
            for _ in range(100):
                if metrics_reads:
                    break
                await asyncio.sleep(0)
            assert metrics_reads == 1

    monkeypatch.setattr(_FakeFiles, "read", read)
    monkeypatch.setattr(_FakeCommands, "run", run)
    events = []
    assert _run(event_callback=events.append).status == "success"
    assert metrics_reads == 2
    assert sum(event["kind"] == "snapshot" for event in events) == 1
    assert sum(event["kind"] == "poll_error" for event in events) == 1


@pytest.mark.parametrize("fails", [False, True])
def test_request_journals_survive_cleanup_and_retries(monkeypatch, tmp_path, fails):
    original_read = _FakeFiles.read
    journal_attempts = []

    async def read(self, path, *, user, format="text", request_timeout=None):
        assert user == "root"
        if path == e2b_executor._REQUESTS_PATH:
            payload = json.loads(dict(self.writes)[e2b_executor._INPUT_PATH])
            attempt_id = payload["attempt_id"]
            journal_attempts.append(attempt_id)
            return (
                json.dumps(
                    {
                        "rollout_id": "rollout-1",
                        "attempt_id": attempt_id,
                        "request_seq": 1,
                        "status": "error" if fails else "success",
                        "duration_seconds": 12.5,
                    }
                )
                + "\n"
            )
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    if fails:
        _FakeAsyncSandbox.entrypoint_error = RuntimeError("agent crashed")
    config = _config()
    config.rollout.execution.env_error_retries = 1
    result = _run(config, trace_output_path=tmp_path / "trace.json")
    assert result.status == ("env_error" if fails else "success")
    expected_attempts = 1
    paths = list(tmp_path.glob("trace.requests.*.jsonl"))
    assert len(paths) == len(journal_attempts) == expected_attempts
    assert {json.loads(path.read_text())["attempt_id"] for path in paths} == set(journal_attempts)
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


def test_request_journal_rejects_foreign_attempt_and_keeps_complete_lines(tmp_path):
    destination = tmp_path / "trace.json"
    record = {"rollout_id": "r", "attempt_id": "a", "duration_seconds": 1}
    sandbox = SimpleNamespace(files=SimpleNamespace(read=AsyncMock()))
    sandbox.files.read.return_value = json.dumps(record) + '\n{"partial":'
    asyncio.run(
        e2b_executor._download_request_journal(
            sandbox, destination, rollout_id="r", attempt_id="a", log_context="test"
        )
    )
    assert json.loads((tmp_path / "trace.requests.a.jsonl").read_text()) == record
    asyncio.run(
        e2b_executor._download_request_journal(
            sandbox, destination, rollout_id="r", attempt_id="b", log_context="test"
        )
    )
    assert not (tmp_path / "trace.requests.b.jsonl").exists()


def test_request_journal_download_timeout_does_not_prevent_cleanup(monkeypatch, tmp_path):
    original_read = _FakeFiles.read

    async def read(self, path, *, user, format="text", request_timeout=None):
        assert user == "root"
        if path == e2b_executor._REQUESTS_PATH:
            await asyncio.Event().wait()
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    monkeypatch.setattr(e2b_executor, "_LIVE_METRICS_READ_TIMEOUT_SECONDS", 0.01)
    assert _run(trace_output_path=tmp_path / "trace.json").status == "success"
    assert _FakeAsyncSandbox.instances[0].killed


def test_evaluation_archive_wraps_only_root_grader_and_preserves_host_assets(monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    task = assets / "task-1"
    (task / "nested").mkdir(parents=True)
    source = b'printf "out\\n"; printf "err\\n" >&2; exit 7\n'
    (task / "run-tests.sh").write_bytes(source)
    (task / "nested/run-tests.sh").write_bytes(source)
    (task / "data.txt").write_text("unchanged")
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"input_data": {"rollout_id": "r"}, "attempt_id": "a"}))
    monkeypatch.setattr(e2b_executor, "_BOOTSTRAP_INTERPRETER", sys.executable)
    monkeypatch.setattr(e2b_executor, "_SOURCE_ROOT", str(Path(__file__).resolve().parents[3]))
    monkeypatch.setattr(e2b_executor, "_INPUT_PATH", str(input_path))
    monkeypatch.setattr(e2b_executor, "_EVALUATOR_LOG_ROOT", str(tmp_path / "logs"))
    entrypoint = e2b_executor._EntrypointSpec(_SWEREBENCH_ENTRYPOINT, assets)
    archive_bytes = e2b_executor._build_evaluation_archive(entrypoint, task_id="task-1")
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        wrapped = archive.extractfile("run-tests.sh").read()
        assert wrapped != source
        assert archive.extractfile("nested/run-tests.sh").read() == source
        assert archive.extractfile("data.txt").read() == b"unchanged"
        assert archive.getmember("run-tests.sh").mode == 0o600
    assert (task / "run-tests.sh").read_bytes() == source
    uploaded = tmp_path / "uploaded-run-tests.sh"
    uploaded.write_bytes(wrapped)
    result = subprocess.run(
        ["/bin/bash", str(uploaded)], capture_output=True, check=False, timeout=5
    )
    assert result.returncode == 7
    assert result.stdout == b"out\n" and result.stderr == b"err\n"
    assert (tmp_path / "logs/stdout.log").read_bytes() == result.stdout
    assert (tmp_path / "logs/stderr.log").read_bytes() == result.stderr
    assert uploaded.read_bytes() == source


@pytest.mark.parametrize("outcome", ["success", "task_failed", "env_error", "timeout", "cancelled"])
@pytest.mark.usefixtures("bounded_io")
def test_evaluator_logs_download_before_kill_for_every_attempt(monkeypatch, tmp_path, outcome):
    original_read = _FakeFiles.read
    original_kill = _FakeSandbox.kill
    attempts = []

    async def read(self, path, *, user, format="text", request_timeout=None):
        if not path.startswith(e2b_executor._EVALUATOR_LOG_ROOT + "/"):
            return await original_read(self, path, user=user, format=format)
        assert user == "root"
        payload = json.loads(dict(self.writes)[e2b_executor._INPUT_PATH])
        if path.endswith("status.json"):
            assert format == "text"
            attempts.append(payload["attempt_id"])
            return json.dumps(
                {
                    "rollout_id": "rollout-1",
                    "attempt_id": payload["attempt_id"],
                    "finished": outcome not in ("timeout", "cancelled"),
                    "returncode": 9 if outcome == "env_error" else 0,
                }
            )
        assert format == "stream"

        async def chunks():
            yield b"full output\n\x00\xff"

        return chunks()

    async def kill(self):
        payload = json.loads(dict(self.files.writes)[e2b_executor._INPUT_PATH])
        directory = tmp_path / f"trace.evaluation.{payload['attempt_id']}"
        assert (directory / "stdout.log").read_bytes() == b"full output\n\x00\xff"
        assert (directory / "stderr.log").read_bytes() == b"full output\n\x00\xff"
        assert json.loads((directory / "status.json").read_text())["downloads"]["stdout"][
            "complete"
        ]
        await original_kill(self)

    monkeypatch.setattr(_FakeFiles, "read", read)
    monkeypatch.setattr(_FakeSandbox, "kill", kill)
    if outcome == "task_failed":
        _FakeAsyncSandbox.result_json = dump_artifact(
            encode_result(
                _result("rollout-1").model_copy(update={"status": "task_failed", "reward": 0.0})
            )
        )
    elif outcome in ("env_error", "timeout", "cancelled"):
        _FakeAsyncSandbox.error_phase = "evaluate"
        _FakeAsyncSandbox.entrypoint_error = {
            "env_error": RuntimeError("evaluator crashed"),
            "timeout": TimeoutError("evaluation deadline"),
            "cancelled": asyncio.CancelledError("caller cancelled"),
        }[outcome]
        _FakeAsyncSandbox.result_error = FileNotFoundError("no result")
    config = _config()
    config.rollout.execution.env_error_retries = 1
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            _run(config, trace_output_path=tmp_path / "trace.json")
    else:
        result = _run(config, trace_output_path=tmp_path / "trace.json")
        assert result.status == ("env_error" if outcome == "timeout" else outcome)
    count = 1
    assert len(set(attempts)) == len(attempts) == count
    assert len(list(tmp_path.glob("trace.evaluation.*"))) == count
    assert all(sandbox.killed for sandbox in _FakeAsyncSandbox.instances)


@pytest.mark.parametrize("failure", ["connection", "timeout", "truncated"])
def test_partial_evaluator_downloads_are_retained_and_marked(monkeypatch, tmp_path, failure):
    closed = []
    monkeypatch.setattr(e2b_executor, "_EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS", 1.0)

    async def read(path, *, user, format="text", request_timeout=None):
        if format == "text":
            return json.dumps(
                {
                    "rollout_id": "r",
                    "attempt_id": "a",
                    "finished": True,
                    "bytes": {"stdout": 100, "stderr": 7},
                }
            )

        async def chunks():
            try:
                yield b"partial" if path.endswith("stdout.log") else b"all err"
                if path.endswith("stdout.log"):
                    if failure == "connection":
                        raise ConnectionError("disconnected")
                    if failure == "timeout":
                        await asyncio.Event().wait()
            finally:
                closed.append(path)

        return chunks()

    sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
    asyncio.run(
        e2b_executor._download_evaluator_logs(
            sandbox, tmp_path / "trace.json", rollout_id="r", attempt_id="a", log_context="test"
        )
    )
    directory = tmp_path / "trace.evaluation.a"
    assert (directory / "stdout.log").read_bytes() == b"partial"
    assert (directory / "stderr.log").read_bytes() == b"all err"
    status = json.loads((directory / "status.json").read_text())
    assert status["downloads"]["stdout"]["complete"] is False
    assert status["downloads"]["stdout"]["error"]
    assert status["downloads"]["stderr"]["complete"] is True
    assert len(closed) == 2


@pytest.mark.parametrize(
    "failure", ["missing", "wrong_attempt", "wrong_rollout", "invalid_json", "timeout"]
)
def test_unavailable_evaluator_logs_do_not_create_misleading_artifacts(
    monkeypatch, tmp_path, failure
):
    monkeypatch.setattr(e2b_executor, "_EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS", 0.01)

    async def read(_path, *, user, format="text", request_timeout=None):
        if failure == "missing":
            raise FileNotFoundError()
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "invalid_json":
            return "not JSON"
        return json.dumps(
            {
                "rollout_id": "foreign" if failure == "wrong_rollout" else "r",
                "attempt_id": "foreign" if failure == "wrong_attempt" else "a",
            }
        )

    sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
    asyncio.run(
        e2b_executor._download_evaluator_logs(
            sandbox, tmp_path / "trace.json", rollout_id="r", attempt_id="a", log_context="test"
        )
    )
    assert not list(tmp_path.iterdir())


def test_evaluator_download_timeout_does_not_change_result_or_skip_kill(monkeypatch, tmp_path):
    original_read = _FakeFiles.read

    async def read(self, path, *, user, format="text", request_timeout=None):
        if path.startswith(e2b_executor._EVALUATOR_LOG_ROOT + "/"):
            await asyncio.Event().wait()
        return await original_read(self, path, user=user, format=format)

    monkeypatch.setattr(_FakeFiles, "read", read)
    monkeypatch.setattr(e2b_executor, "_EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS", 0.01)
    assert _run(trace_output_path=tmp_path / "trace.json").status == "success"
    assert _FakeAsyncSandbox.instances[0].killed


def test_debug_log_level_reaches_sandbox(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    _run()
    agent_call = next(
        kwargs
        for command, kwargs in _FakeAsyncSandbox.instances[0].commands.calls
        if "--phase agent" in command
    )
    assert agent_call["envs"]["LOG_LEVEL"] == "DEBUG"
    evaluation_call = next(
        kwargs
        for command, kwargs in _FakeAsyncSandbox.instances[0].commands.calls
        if "--phase evaluate" in command
    )
    assert evaluation_call["envs"]["LOG_LEVEL"] == "DEBUG"
    assert "WEAVER_API_KEY" not in evaluation_call["envs"]


def test_evaluator_environment_is_only_forwarded_to_evaluation(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "evaluation-only-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "unselected-secret")
    monkeypatch.delenv("OPTIONAL_JUDGE_SETTING", raising=False)
    monkeypatch.setenv("NORTHRL_E2B_EVALUATOR_ENV", "TEST_JUDGE_KEY, OPTIONAL_JUDGE_SETTING")
    config = _config()
    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            config,
            gateway=_gateway(),
            template="test-template",
            timeout_seconds=3600,
        )
    )
    assert result.status == "success"
    sandbox = _FakeAsyncSandbox.instances[0]
    agent_env = next(
        kwargs["envs"] for command, kwargs in sandbox.commands.calls if "--phase agent" in command
    )
    evaluation_env = next(
        kwargs["envs"]
        for command, kwargs in sandbox.commands.calls
        if "--phase evaluate" in command
    )
    assert "TEST_JUDGE_KEY" not in agent_env
    assert evaluation_env["TEST_JUDGE_KEY"] == "evaluation-only-secret"
    assert "OPTIONAL_JUDGE_SETTING" not in evaluation_env
    assert "UNRELATED_SECRET" not in evaluation_env
    assert "WEAVER_API_KEY" not in evaluation_env
    assert "NORTHRL_E2B_EVALUATOR_ENV" not in evaluation_env
    payload = dict(sandbox.files.writes)[e2b_executor._INPUT_PATH]
    assert "evaluation-only-secret" not in payload
    assert "unselected-secret" not in payload
    assert "TEST_JUDGE_KEY" not in payload
    assert "NORTHRL_E2B_EVALUATOR_ENV" not in payload


def test_agent_environment_is_only_forwarded_to_agent_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SERPER_API_KEY", "agent-only-search-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "unselected-secret")
    monkeypatch.delenv("OPTIONAL_SEARCH_SETTING", raising=False)
    monkeypatch.setenv("NORTHRL_E2B_AGENT_ENV", "SERPER_API_KEY, OPTIONAL_SEARCH_SETTING")
    _FakeAsyncSandbox.trace_json = dump_artifact(encode_trace(_trace("rollout-1")))
    trace_path = tmp_path / "trace.json"
    result = _run(trace_output_path=trace_path)
    assert result.status == "success"
    sandbox = _FakeAsyncSandbox.instances[0]
    agent_calls = []
    for command, kwargs in sandbox.commands.calls:
        environment = kwargs.get("envs", {})
        if "--phase agent" in command:
            assert environment["SERPER_API_KEY"] == "agent-only-search-secret"
            assert environment["NORTHGATE_API_KEY"] == "northgate-key"
            agent_calls.append(command)
        else:
            # Includes source extraction, bootstrap, evaluation and diagnostics.
            assert "SERPER_API_KEY" not in environment
        assert "OPTIONAL_SEARCH_SETTING" not in environment
        assert "UNRELATED_SECRET" not in environment
        assert "NORTHRL_E2B_AGENT_ENV" not in environment
        assert "agent-only-search-secret" not in command
    assert len(agent_calls) == 1
    for _, payload in sandbox.files.writes:
        if isinstance(payload, str):
            assert "agent-only-search-secret" not in payload
            assert "unselected-secret" not in payload
    assert "agent-only-search-secret" not in repr(result)
    assert trace_path.exists()
    assert "agent-only-search-secret" not in repr(load_trace_gzip(trace_path.read_bytes()))
    output = capsys.readouterr()
    assert "agent-only-search-secret" not in output.out + output.err
    assert "SERPER_API_KEY" not in repr(_FakeAsyncSandbox.create_calls)


@pytest.mark.parametrize(
    "value",
    [
        "bad-name",
        "A,A",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        "PYTHONHOME",
        "PATH",
        "LOG_LEVEL",
        "NORTHGATE_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "A,,B",
        "A,",
    ],
)
def test_invalid_agent_environment_fails_before_sandbox_create(monkeypatch, value):
    monkeypatch.setenv("NORTHRL_E2B_AGENT_ENV", value)
    result = _run()
    assert result.status == "env_error"
    assert result.error_message is not None
    assert "NORTHRL_E2B_AGENT_ENV" in result.error_message
    assert _FakeAsyncSandbox.create_calls == []


@pytest.mark.parametrize("value", [None, "", "  "])
def test_unset_agent_environment_forwards_no_host_variables(monkeypatch, value):
    monkeypatch.setenv("SERPER_API_KEY", "unselected-secret")
    if value is not None:
        monkeypatch.setenv("NORTHRL_E2B_AGENT_ENV", value)
    assert e2b_executor._resolve_agent_environment() == {}


@pytest.mark.parametrize(
    "value", ["bad-name", "A,A", "PYTHONPATH", "PYTHONSAFEPATH", "LOG_LEVEL", "A,,B", "A,"]
)
def test_invalid_evaluator_environment_fails_before_sandbox_create(monkeypatch, value):
    monkeypatch.setenv("NORTHRL_E2B_EVALUATOR_ENV", value)
    config = _config()
    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            config,
            gateway=_gateway(),
            template="test-template",
            timeout_seconds=3600,
        )
    )
    assert result.status == "env_error"
    assert "NORTHRL_E2B_EVALUATOR_ENV" in result.error_message
    assert _FakeAsyncSandbox.create_calls == []


@pytest.mark.parametrize("value", [None, "", "  "])
def test_unset_evaluator_environment_forwards_no_host_variables(monkeypatch, value):
    monkeypatch.setenv("TEST_JUDGE_KEY", "unselected-secret")
    if value is not None:
        monkeypatch.setenv("NORTHRL_E2B_EVALUATOR_ENV", value)
    assert e2b_executor._resolve_evaluator_environment() == {}


@pytest.mark.parametrize("has_result", [False, True])
@pytest.mark.usefixtures("bounded_io")
def test_supervisor_timeout_is_terminal_and_preserves_any_completed_result(has_result):
    error = RuntimeError("supervisor exited with status 124")
    error.exit_code = 124
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = error
    _FakeAsyncSandbox.state_json = _state("evaluator")
    if not has_result:
        _FakeAsyncSandbox.result_error = FileNotFoundError("no evaluation result")

    result = _run()

    assert result.metrics["e2b_timed_out"] == 1.0
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sandbox.killed
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.files.reads.count(e2b_executor._RESULT_PATH) == 1
    if has_result:
        assert result.status == "success"
        assert result.reward == _result("rollout-1").reward
        assert _dense_segments(result.segments) == _result("rollout-1").segments
    else:
        assert result.status == "env_error"
        assert result.segments == []


def _evaluation_terminal(files, **updates):
    payload = json.loads(dict(files.writes)[e2b_executor._INPUT_PATH])
    return json.dumps(
        {
            "rollout_id": payload["input_data"]["rollout_id"],
            "attempt_id": payload["attempt_id"],
            "phase": "evaluation_finished",
            "exit_code": 124,
            **updates,
        }
    )


@pytest.mark.parametrize("command_error", ["sdk_timeout", "disconnected"])
@pytest.mark.parametrize("exit_code", [0, 124])
def test_terminal_evaluator_without_result_does_not_spend_recovery_budget(
    monkeypatch, command_error, exit_code
):
    class TimeoutException(Exception):
        pass

    TimeoutException.__module__ = "e2b.exceptions"
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = (
        TimeoutException("lost SDK command outcome")
        if command_error == "sdk_timeout"
        else ConnectionError("lost evaluator command stream")
    )
    _FakeAsyncSandbox.state_json = _state("evaluator")
    _FakeAsyncSandbox.result_error = FileNotFoundError("no evaluation result")
    original_read = _FakeFiles.read
    terminal_reads = 0

    async def read(self, path, **kwargs):
        nonlocal terminal_reads
        if path == e2b_executor._EVALUATION_TERMINAL_PATH:
            terminal_reads += 1
            return _evaluation_terminal(self, exit_code=exit_code)
        return await original_read(self, path, **kwargs)

    monkeypatch.setattr(_FakeFiles, "read", read)

    async def scenario():
        # Keep the real 900-second recovery budget; a regression must hit this
        # watchdog instead of passing because a fixture shortened that budget.
        assert e2b_executor.ExecutionBudgets.recovery_seconds == 900
        return await asyncio.wait_for(
            e2b_executor.run_rollout_in_e2b(
                RolloutInput(rollout_id="rollout-1", task={}),
                _config(),
                gateway=_gateway(),
                template="swebench-template",
                timeout_seconds=3600,
            ),
            timeout=5,
        )

    result = asyncio.run(scenario())
    assert result.status == "env_error"
    assert result.segments == []
    if command_error == "sdk_timeout" or exit_code == 124:
        assert result.metrics["e2b_timed_out"] == 1.0
    assert terminal_reads == 1
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sandbox.files.reads.count(e2b_executor._RESULT_PATH) == (
        2 if command_error == "sdk_timeout" else 3
    )
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.parametrize(
    "last_read_error", [None, TimeoutError("slow result"), ConnectionError("reset")]
)
@pytest.mark.usefixtures("bounded_io")
def test_terminal_evaluator_preserves_late_result_and_retries_real_download(
    monkeypatch, last_read_error
):
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = ConnectionError("lost evaluator stream")
    original_read = _FakeFiles.read
    sleep = AsyncMock(wraps=asyncio.sleep)
    result_reads = 0
    result_timeouts = []

    async def read(self, path, **kwargs):
        nonlocal result_reads
        if path == e2b_executor._EVALUATION_TERMINAL_PATH:
            return _evaluation_terminal(self)
        if path == e2b_executor._RESULT_PATH:
            result_reads += 1
            result_timeouts.append(kwargs["request_timeout"])
            if result_reads == 1:
                raise FileNotFoundError("publication raced the first read")
            if result_reads == 2 and last_read_error is not None:
                raise last_read_error
        return await original_read(self, path, **kwargs)

    monkeypatch.setattr(_FakeFiles, "read", read)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    # Only result recovery delays matter here, not periodic metrics polling.
    monkeypatch.setattr(e2b_executor, "_poll_live_metrics", AsyncMock())
    result = _run()

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert result_reads == (2 if last_read_error is None else 3)
    if last_read_error is None:
        sleep.assert_not_awaited()
    else:
        sleep.assert_awaited_once()
    assert all(timeout > 0 for timeout in result_timeouts)
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.parametrize(
    "terminal_record",
    [
        FileNotFoundError("evaluator still running"),
        TimeoutError("control read timed out"),
        ConnectionError("control connection reset"),
        "invalid json",
        "[]",
        {"rollout_id": "another-rollout"},
        {"attempt_id": "previous-attempt"},
        {"phase": "evaluator"},
        {"exit_code": True},
        {"exit_code": "124"},
        {"exit_code": -1},
    ],
)
@pytest.mark.usefixtures("bounded_io")
def test_unconfirmed_evaluator_keeps_waiting_for_original_result(monkeypatch, terminal_record):
    _FakeAsyncSandbox.error_phase = "evaluate"
    _FakeAsyncSandbox.entrypoint_error = ConnectionError("lost evaluator stream")
    # The runtime can report a failure before it finishes writing its result.
    # This ordinary phase marker must not be mistaken for supervisor completion.
    _FakeAsyncSandbox.state_json = _state("evaluator", status="env_error")
    original_read = _FakeFiles.read
    result_reads = 0
    terminal_reads = 0

    async def read(self, path, **kwargs):
        nonlocal result_reads, terminal_reads
        if path == e2b_executor._EVALUATION_TERMINAL_PATH:
            terminal_reads += 1
            if isinstance(terminal_record, Exception):
                raise terminal_record
            if isinstance(terminal_record, dict):
                return _evaluation_terminal(self, **terminal_record)
            return terminal_record
        if path == e2b_executor._RESULT_PATH:
            result_reads += 1
            if result_reads <= 2:
                raise FileNotFoundError("evaluator has not committed result")
        return await original_read(self, path, **kwargs)

    monkeypatch.setattr(_FakeFiles, "read", read)
    result = _run()

    assert _without_driver_runtime(result) == _result("rollout-1")
    assert result_reads == 3
    assert terminal_reads == 2
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase evaluate" in command for command, _ in sandbox.commands.calls) == 1
    assert sandbox.killed


@pytest.mark.parametrize("artifact", ["result", "trace"])
@pytest.mark.parametrize("compressed", [False, True])
def test_artifacts_require_gzip_and_accept_sdk_bytearrays(tmp_path, artifact, compressed):
    models = {
        "result": encode_result(_result("rollout-1")),
        "trace": encode_trace(_prefix_trace("rollout-1")),
    }
    payload = bytearray(dump_artifact(models[artifact]))
    if not compressed:
        payload = bytearray(gzip.decompress(payload))
    read = AsyncMock(return_value=payload)
    sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
    destination = tmp_path / "trace.json.gz"

    async def scenario():
        if artifact == "result":
            return await e2b_executor._read_result(
                sandbox, RolloutInput(rollout_id="rollout-1", task={})
            )
        return await e2b_executor._download_trace(
            sandbox,
            destination,
            log_context="test",
            rollout_id="rollout-1",
            raise_read_errors=True,
        )

    if compressed:
        result = asyncio.run(scenario())
        assert result.rollout_id == "rollout-1"
        if artifact == "trace":
            assert destination.read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="gzip"):
            asyncio.run(scenario())
        assert not destination.exists()
    assert read.await_args.kwargs["format"] == "bytes"


def test_queued_download_wait_does_not_spend_network_timeout():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        asyncio.get_running_loop()._northrl_download_queue = queue
        started = asyncio.Event()
        release = asyncio.Event()

        async def occupy():
            started.set()
            await release.wait()

        blocker = asyncio.create_task(queue.run(occupy))
        await started.wait()
        read = AsyncMock(return_value="downloaded")
        sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
        deadline = e2b_executor.time.monotonic() + 1
        task = asyncio.create_task(
            e2b_executor._retry_io(
                lambda: e2b_executor._download_file(sandbox, "/file"),
                deadline=deadline,
                io_seconds=0.01,
                log_context="test",
                operation="download",
                queued_download=True,
            )
        )
        try:
            await asyncio.sleep(0.05)
            assert not task.done()
            read.assert_not_awaited()
            assert queue.pending_count == 1
        finally:
            release.set()
        assert await task == "downloaded"
        await blocker
        read.assert_awaited_once()

    asyncio.run(scenario())


def test_queued_download_still_obeys_overall_deadline():
    async def scenario():
        queue = DownloadQueue(max_active=2)
        asyncio.get_running_loop()._northrl_download_queue = queue
        started = asyncio.Event()
        release = asyncio.Event()

        async def occupy():
            started.set()
            await release.wait()

        blocker = asyncio.create_task(queue.run(occupy))
        await started.wait()
        read = AsyncMock()
        sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
        try:
            with pytest.raises(TimeoutError):
                await e2b_executor._retry_io(
                    lambda: e2b_executor._download_file(sandbox, "/file"),
                    deadline=e2b_executor.time.monotonic() + 0.03,
                    io_seconds=1,
                    log_context="test",
                    operation="download",
                    queued_download=True,
                )
            read.assert_not_awaited()
            assert queue.pending_count == 0
        finally:
            release.set()
            await blocker

    asyncio.run(scenario())


def test_result_processing_keeps_download_slot_without_network_timeout(monkeypatch):
    release = threading.Event()
    original_decode = e2b_executor.decode_result

    async def scenario():
        loop = asyncio.get_running_loop()
        queue = DownloadQueue(max_active=2)
        loop._northrl_download_queue = queue
        started = asyncio.Event()

        def slow_decode(result, **kwargs):
            assert threading.current_thread() is not threading.main_thread()
            loop.call_soon_threadsafe(started.set)
            assert release.wait(2)
            return original_decode(result, **kwargs)

        monkeypatch.setattr(e2b_executor, "decode_result", slow_decode)
        read = AsyncMock(return_value=dump_artifact(encode_result(_result("rollout-1"))))
        sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
        task = asyncio.create_task(
            e2b_executor._retry_io(
                lambda: e2b_executor._read_result(
                    sandbox, RolloutInput(rollout_id="rollout-1", task={})
                ),
                deadline=e2b_executor.time.monotonic() + 2,
                io_seconds=0.01,
                log_context="test",
                operation="result",
                queued_download=True,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.sleep(0.05)
            assert not task.done()
            assert queue.active_count == 1
        finally:
            release.set()
        assert _without_driver_runtime(await task) == _result("rollout-1")
        assert queue.active_count == 0

    asyncio.run(scenario())


def test_cancelled_trace_write_finishes_before_next_attempt_can_publish(monkeypatch, tmp_path):
    release = threading.Event()
    original_write = e2b_executor._atomic_write_bytes
    destination = tmp_path / "trace.json.gz"
    old_payload = dump_artifact(encode_trace(_trace("rollout-1")))
    new_payload = dump_artifact(encode_trace(_prefix_trace("rollout-1")))

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def slow_write(path, payload):
            if payload == old_payload:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(2)
            original_write(path, payload)

        monkeypatch.setattr(e2b_executor, "_atomic_write_bytes", slow_write)
        read = AsyncMock(side_effect=[old_payload, new_payload])
        sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
        kwargs = dict(log_context="test", rollout_id="rollout-1", priority="background")
        task = asyncio.create_task(e2b_executor._download_trace(sandbox, destination, **kwargs))
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert get_download_queue().active_count == 1
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert destination.read_bytes() == old_payload
        destination.unlink()  # Same cleanup that begins the next sandbox attempt.
        await e2b_executor._download_trace(sandbox, destination, **kwargs)
        assert destination.read_bytes() == new_payload
        assert list(tmp_path.iterdir()) == [destination]

    asyncio.run(scenario())


@pytest.mark.parametrize("control", ["heartbeat", "state"])
def test_control_reads_remain_available_when_normal_downloads_and_cpu_are_saturated(control):
    release = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        queue = DownloadQueue(max_active=2)
        loop._northrl_download_queue = queue
        started = [asyncio.Event(), asyncio.Event()]
        network_started, network_release = asyncio.Event(), asyncio.Event()

        async def occupy_network():
            network_started.set()
            await network_release.wait()

        network = asyncio.create_task(queue.run(occupy_network))

        def occupy(index):
            loop.call_soon_threadsafe(started[index].set)
            assert release.wait(2)

        workers = [asyncio.create_task(run_blocking(occupy, index)) for index in range(2)]
        events = []
        read = AsyncMock(
            return_value=json.dumps(
                {
                    "rollout_id": "rollout-1",
                    "attempt_id": "attempt-1",
                    "heartbeat": 7,
                    "phase": "agent_completed",
                    "metrics": {"calls": 2},
                }
            )
        )
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
            await network_started.wait()
            sandbox = SimpleNamespace(files=SimpleNamespace(read=read))
            if control == "heartbeat":
                await e2b_executor._read_live_metrics(
                    sandbox,
                    rollout_id="rollout-1",
                    event_callback=events.append,
                    timeout_seconds=0.05,
                )
                assert events == [
                    {
                        "kind": "snapshot",
                        "heartbeat": 7,
                        "phase": "agent_completed",
                        "metrics": {"calls": 2.0},
                    }
                ]
            else:
                await e2b_executor._wait_for_agent_completion(
                    sandbox,
                    RolloutInput(rollout_id="rollout-1", task={}),
                    "attempt-1",
                    deadline=e2b_executor.time.monotonic() + 0.05,
                    io_seconds=0.01,
                    log_context="test",
                )
                read.assert_awaited_once_with(e2b_executor._STATE_PATH, user="root")
        finally:
            release.set()
            network_release.set()
            await asyncio.gather(network, *workers)

    asyncio.run(scenario())


def test_gzip_trace_companions_keep_existing_filename_convention(tmp_path):
    record = {"rollout_id": "rollout-1", "attempt_id": "attempt-1", "request_id": "request-1"}
    read = AsyncMock(return_value=json.dumps(record) + "\n")
    asyncio.run(
        e2b_executor._download_request_journal(
            SimpleNamespace(files=SimpleNamespace(read=read)),
            tmp_path / "e2b_rollout_abc.json.gz",
            rollout_id="rollout-1",
            attempt_id="attempt-1",
            log_context="test",
        )
    )
    target = tmp_path / "e2b_rollout_abc.requests.attempt-1.jsonl"
    assert json.loads(target.read_text()) == record
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("local_destination", [False, True])
def test_trace_recovery_does_not_enqueue_local_reads_after_its_deadline(
    monkeypatch, tmp_path, local_destination
):
    release = threading.Event()
    local_load = Mock(side_effect=AssertionError("expired local read must not start"))
    monkeypatch.setattr(e2b_executor, "_load_locally_synced_trace", local_load)
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_INITIAL_DELAY_SECONDS", 0.001)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = [asyncio.Event() for _ in range(4)]

        def occupy(index):
            loop.call_soon_threadsafe(started[index].set)
            assert release.wait(2)

        workers = [
            asyncio.create_task(e2b_executor.run_background_io(occupy, index)) for index in range(4)
        ]
        sandbox = SimpleNamespace(
            files=SimpleNamespace(read=AsyncMock(side_effect=FileNotFoundError()))
        )
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
            result = await asyncio.wait_for(
                e2b_executor._recover_trace(
                    sandbox,
                    tmp_path / "trace.json.gz" if local_destination else None,
                    log_context="test",
                    rollout_id="rollout-1",
                    deadline=e2b_executor.time.monotonic() + 0.03,
                    io_seconds=0.01,
                ),
                0.2,
            )
            assert result is None
        finally:
            release.set()
            await asyncio.gather(*workers)
        local_load.assert_not_called()

    asyncio.run(scenario())


def test_evaluator_stream_cancel_during_open_closes_owned_file(monkeypatch, tmp_path):
    release = threading.Event()
    handles = []
    original_open = Path.open

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def slow_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if path.suffix == ".log":
                handles.append(handle)
                if path.name == "stdout.log":
                    loop.call_soon_threadsafe(started.set)
                    assert release.wait(2)
            return handle

        monkeypatch.setattr(Path, "open", slow_open)

        async def read(path, *, user, format="text", request_timeout=None):
            if format == "text":
                return json.dumps({"rollout_id": "r", "attempt_id": "a", "finished": True})

            async def chunks():
                yield b"output"

            return chunks()

        task = asyncio.create_task(
            e2b_executor._download_evaluator_logs(
                SimpleNamespace(files=SimpleNamespace(read=read)),
                tmp_path / "trace.json.gz",
                rollout_id="r",
                attempt_id="a",
                log_context="test",
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert handles and all(handle.closed for handle in handles)
        assert get_download_queue().active_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("disconnected_extractor", [False, True])
def test_evaluation_preparation_does_not_spend_evaluator_budget(
    monkeypatch, caplog, disconnected_extractor
):
    from northrl.utils import profiling

    clock = [0.0]
    monkeypatch.setattr(
        e2b_executor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000 + clock[0]),
    )
    monkeypatch.setattr(wire, "time", e2b_executor.time)
    monkeypatch.setattr(profiling, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    original_write, original_run, original_read = (
        _FakeFiles.write,
        _FakeCommands.run,
        _FakeFiles.read,
    )

    async def write(self, path, data, **kwargs):
        if path == e2b_executor._EVALUATION_ARCHIVE_PATH:
            clock[0] += 200
        return await original_write(self, path, data, **kwargs)

    async def run(self, command, **kwargs):
        result = await original_run(self, command, **kwargs)
        if command == e2b_executor._EXTRACT_EVALUATION_COMMAND:
            clock[0] += 100
            if disconnected_extractor:
                raise ConnectionError("lost extraction acknowledgement")
        elif "--phase evaluate" in command:
            clock[0] += 7
        return result

    async def read(self, path, **kwargs):
        if path == e2b_executor._EVALUATION_READY_PATH:
            clock[0] += 50
            return "ready"
        return await original_read(self, path, **kwargs)

    monkeypatch.setattr(_FakeFiles, "write", write)
    monkeypatch.setattr(_FakeCommands, "run", run)
    monkeypatch.setattr(_FakeFiles, "read", read)
    caplog.set_level(logging.DEBUG, logger="northrl.utils.profiling")
    assert _without_driver_runtime(_run()) == _result("rollout-1")
    sandbox = _FakeAsyncSandbox.instances[0]
    extraction_calls = [
        kwargs
        for command, kwargs in sandbox.commands.calls
        if command == e2b_executor._EXTRACT_EVALUATION_COMMAND
    ]
    assert len(extraction_calls) == 1
    assert extraction_calls[0]["timeout"] == 400
    command, kwargs = sandbox.commands.calls[-1]
    assert kwargs["timeout"] == 605
    argv = shlex.split(command)
    assert float(argv[argv.index("--timeout-seconds") + 1]) == 600
    expected = {
        "driver.evaluation-upload": 200,
        "driver.evaluation-extract": 150 if disconnected_extractor else 100,
        "driver.evaluation-preparation": 350 if disconnected_extractor else 300,
        "driver.evaluator-command": 7,
    }
    for stage, seconds in expected.items():
        assert any(
            f"stage={stage} status=completed" in record.message
            and f"duration_seconds={seconds:.3f}" in record.message
            for record in caplog.records
        )


@pytest.mark.parametrize("hung_phase", ["upload", "extract"])
def test_evaluation_preparation_timeout_never_launches_evaluator_or_restarts_agent(
    monkeypatch, hung_phase, caplog
):
    monkeypatch.setattr(e2b_executor.ExecutionBudgets, "evaluation_preparation_seconds", 0.02)
    original_write, original_run = _FakeFiles.write, _FakeCommands.run

    async def write(self, path, data, **kwargs):
        if hung_phase == "upload" and path == e2b_executor._EVALUATION_ARCHIVE_PATH:
            await asyncio.Event().wait()
        return await original_write(self, path, data, **kwargs)

    async def run(self, command, **kwargs):
        if hung_phase == "extract" and command == e2b_executor._EXTRACT_EVALUATION_COMMAND:
            await asyncio.Event().wait()
        return await original_run(self, command, **kwargs)

    monkeypatch.setattr(_FakeFiles, "write", write)
    monkeypatch.setattr(_FakeCommands, "run", run)
    result = _run()
    assert result.status == "env_error"
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert not any("--phase evaluate" in command for command, _ in sandbox.commands.calls)
    assert sandbox.killed
    assert "stage=driver.evaluation-preparation status=failed" in caplog.text


def test_background_cleanup_confirms_after_rollout_returns(monkeypatch):
    from northrl.remote_rollout import sandbox_cleanup

    monkeypatch.setattr(sandbox_cleanup, "_CLEANUP_RETRY_DELAY_SECONDS", 0)
    # Check the production integration while SDK status semantics are covered separately.
    release = asyncio.Event()

    async def probe(sandbox, sandbox_id):
        assert sandbox is _FakeAsyncSandbox.instances[0]
        assert sandbox_id == "test-sandbox-id"
        await release.wait()
        return True

    monkeypatch.setattr(sandbox_cleanup, "_confirm_absent", probe)
    _FakeAsyncSandbox.kill_error = TimeoutError("lost delete response")
    events = []

    async def run():
        result = await e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="rollout-1", task={}),
            _config(),
            gateway=_gateway(),
            template="swebench-template",
            timeout_seconds=3600,
            event_callback=events.append,
        )
        assert _without_driver_runtime(result) == _result("rollout-1")
        assert events[-1]["kind"] == "rollout_finished"
        assert [event["confirmed"] for event in events if event["kind"] == "sandbox_closed"] == [
            False
        ]
        pending = list(sandbox_cleanup._pending_cleanups)
        assert len(pending) == 1
        release.set()
        await asyncio.gather(*pending)
        closed = [event for event in events if event["kind"] == "sandbox_closed"]
        assert [event["confirmed"] for event in closed] == [False, True]
        assert all(
            event["sandbox_id"] == "test-sandbox-id" and event["attempt"] == 1 for event in closed
        )

    asyncio.run(run())


def test_cancellation_during_initial_kill_still_schedules_confirmation(monkeypatch):
    scheduled = Mock()
    monkeypatch.setattr(e2b_executor, "schedule_sandbox_cleanup", scheduled)

    async def kill(self):
        raise asyncio.CancelledError("cancelled during cleanup")

    monkeypatch.setattr(_FakeSandbox, "kill", kill)
    events = []
    with pytest.raises(asyncio.CancelledError, match="cancelled during cleanup"):
        _run(event_callback=events.append)
    scheduled.assert_called_once()
    assert scheduled.call_args.kwargs["sandbox_id"] == "test-sandbox-id"
    assert [event["confirmed"] for event in events if event["kind"] == "sandbox_closed"] == [False]


@pytest.mark.parametrize("failed_phase", ["upload", "extract"])
def test_evaluation_preparation_retries_share_one_deadline(monkeypatch, failed_phase):
    clock = [0.0]
    monkeypatch.setattr(
        e2b_executor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000 + clock[0]),
    )
    monkeypatch.setattr(wire, "time", e2b_executor.time)
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_INITIAL_DELAY_SECONDS", 0.001)
    original_write, original_run = _FakeFiles.write, _FakeCommands.run
    attempts = []

    def fail():
        attempts.append(clock[0])
        clock[0] += 300
        if failed_phase == "upload":
            raise TimeoutError("upload failed")
        error = RuntimeError("extractor exited")
        error.exit_code = 1
        raise error

    async def write(self, path, data, **kwargs):
        if failed_phase == "upload" and path == e2b_executor._EVALUATION_ARCHIVE_PATH:
            fail()
        return await original_write(self, path, data, **kwargs)

    async def run(self, command, **kwargs):
        if failed_phase == "extract" and command == e2b_executor._EXTRACT_EVALUATION_COMMAND:
            fail()
        return await original_run(self, command, **kwargs)

    monkeypatch.setattr(_FakeFiles, "write", write)
    monkeypatch.setattr(_FakeCommands, "run", run)
    assert _run().status == "env_error"
    assert attempts == [0, 300]
    assert len(_FakeAsyncSandbox.instances) == 1
    sandbox = _FakeAsyncSandbox.instances[0]
    assert sum("--phase agent" in command for command, _ in sandbox.commands.calls) == 1
    assert not any("--phase evaluate" in command for command, _ in sandbox.commands.calls)
    assert sandbox.killed
