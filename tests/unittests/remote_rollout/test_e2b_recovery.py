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

"""Recovery boundaries independent of the sandbox executor's fake transport."""

import asyncio
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.contracts import RemoteRolloutResult, RolloutInput
from northrl.remote_rollout.wire import dump_artifact, encode_result


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_transfer_retries_only_recoverable_http_statuses(status):
    error = httpx.HTTPStatusError(
        "file transfer failed",
        request=httpx.Request("GET", "https://sandbox.example/file"),
        response=httpx.Response(status),
    )
    assert e2b_executor._is_retryable_io_error(error) is (status == 429 or status >= 500)


@pytest.mark.parametrize("missing_type", ["builtin", "sdk"])
def test_missing_trace_gets_one_publication_retry(monkeypatch, missing_type):
    class FileNotFoundException(Exception):
        pass

    FileNotFoundException.__module__ = "e2b.exceptions"
    missing = (
        FileNotFoundError("no trace") if missing_type == "builtin" else FileNotFoundException()
    )
    reads = AsyncMock(side_effect=missing)
    sandbox = SimpleNamespace(files=SimpleNamespace(read=reads))
    monkeypatch.setattr(e2b_executor, "_IO_RETRY_INITIAL_DELAY_SECONDS", 0)

    async def run():
        return await e2b_executor._recover_trace(
            sandbox,
            None,
            log_context="test",
            rollout_id="rollout-1",
            deadline=e2b_executor.time.monotonic() + 1,
            io_seconds=1,
        )

    assert asyncio.run(run()) is None
    assert reads.await_count == 2


def test_trace_read_timeout_uses_local_copy_without_spending_recovery_budget(monkeypatch, tmp_path):
    cancelled = []
    cached = object()

    async def hung_read(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    sandbox = SimpleNamespace(files=SimpleNamespace(read=hung_read))
    monkeypatch.setattr(e2b_executor, "_load_locally_synced_trace", lambda *args, **kwargs: cached)

    async def run():
        return await asyncio.wait_for(
            e2b_executor._recover_trace(
                sandbox,
                tmp_path / "trace.json.gz",
                log_context="test",
                rollout_id="rollout-1",
                deadline=e2b_executor.time.monotonic() + 900,
                io_seconds=0.01,
            ),
            timeout=1,
        )

    assert asyncio.run(run()) is cached
    assert cancelled == [True]


def test_rpc_deadline_without_timeout_cause_is_recognized():
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    error = ConnectError(Code.DEADLINE_EXCEEDED, "command deadline expired")
    assert error.__cause__ is None
    assert e2b_executor._is_timeout_exception(error)
    assert not e2b_executor._is_command_transport_error(error)


@pytest.mark.parametrize("exit_code", [0, 17])
def test_evaluation_supervisor_preserves_arguments_environment_output_and_exit_code(
    tmp_path, monkeypatch, exit_code
):
    monkeypatch.setattr(e2b_executor, "_BOOTSTRAP_INTERPRETER", sys.executable)
    monkeypatch.setenv("NORTHRL_SUPERVISOR_TEST_VALUE", "inherited")
    argument = "spaces 'single' \"double\" $HOME `literal`\nnewline"
    script = (
        "import os, sys; "
        "print(os.environ['NORTHRL_SUPERVISOR_TEST_VALUE']); "
        "print(repr(sys.argv[1])); "
        "print('stderr remains visible', file=sys.stderr); "
        f"sys.exit({exit_code})"
    )
    command = e2b_executor._supervise_evaluation_command(
        shlex.join([sys.executable, "-c", script, argument]),
        2.0,
        rollout_id=argument,
        attempt_id="attempt 'with' spaces",
        terminal_path=str(tmp_path / "terminal.json"),
    )
    completed = subprocess.run(
        shlex.split(command), cwd=tmp_path, capture_output=True, text=True, timeout=5
    )
    assert completed.returncode == exit_code
    assert completed.stdout == "inherited\n" + repr(argument) + "\n"
    assert completed.stderr == "stderr remains visible\n"
    terminal_path = tmp_path / "terminal.json"
    assert json.loads(terminal_path.read_text()) == {
        "rollout_id": argument,
        "attempt_id": "attempt 'with' spaces",
        "phase": "evaluation_finished",
        "exit_code": exit_code,
    }
    assert stat.S_IMODE(terminal_path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".evaluation-terminal.*"))


def _assert_process_stopped(pid):
    deadline = time.monotonic() + 1
    while True:
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        except FileNotFoundError:
            return
        state = next(line for line in lines if line.startswith("State:")).split()[1]
        if state == "Z":
            return
        assert time.monotonic() < deadline, f"process {pid} remained alive with state {state}"
        time.sleep(0.005)


@pytest.mark.parametrize("mode", ["soft_exit", "hard_timeout", "supervisor_cancel"])
def test_evaluation_supervisor_stops_real_process_group(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(e2b_executor, "_BOOTSTRAP_INTERPRETER", sys.executable)
    child_script = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('child ready', flush=True); "
        "open('child.pid', 'w').write(str(os.getpid())); time.sleep(30)"
    )
    parent_script = (
        "import os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "open('parent.pid', 'w').write(str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}])\n"
        "while not os.path.exists('child.pid') or not open('child.pid').read():\n"
        "    time.sleep(0.005)\n"
        "print('parent ready', flush=True)\n"
        + ("sys.exit(17)\n" if mode == "soft_exit" else "time.sleep(30)\n")
    )
    budget = 0.5 if mode == "hard_timeout" else 5.0
    command = e2b_executor._supervise_evaluation_command(
        shlex.join([sys.executable, "-c", parent_script]),
        budget,
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        terminal_path=str(tmp_path / "terminal.json"),
    )
    supervisor = subprocess.Popen(
        shlex.split(command),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if mode != "soft_exit":
            ready_deadline = time.monotonic() + 2
            while not (tmp_path / "child.pid").exists() or not (tmp_path / "child.pid").read_text():
                assert time.monotonic() < ready_deadline
                time.sleep(0.005)
            assert supervisor.poll() is None
            assert not (tmp_path / "terminal.json").exists()
            if mode == "supervisor_cancel":
                supervisor.terminate()
        output, errors = supervisor.communicate(timeout=3)
        assert errors == ""
        assert "child ready" in output
        assert (
            supervisor.returncode
            == {
                "soft_exit": 17,
                "hard_timeout": 124,
                "supervisor_cancel": 128 + signal.SIGTERM,
            }[mode]
        )
        for name in ("parent.pid", "child.pid"):
            _assert_process_stopped(int((tmp_path / name).read_text()))
        assert json.loads((tmp_path / "terminal.json").read_text()) == {
            "rollout_id": "rollout-1",
            "attempt_id": "attempt-1",
            "phase": "evaluation_finished",
            "exit_code": supervisor.returncode,
        }
    finally:
        for name in ("parent.pid", "child.pid"):
            path = tmp_path / name
            if path.exists() and path.read_text():
                try:
                    os.kill(int(path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.communicate(timeout=3)


def test_evaluation_supervisor_terminal_record_is_atomically_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(e2b_executor, "_BOOTSTRAP_INTERPRETER", sys.executable)
    terminal_path = tmp_path / "terminal.json"
    previous = {"attempt_id": "previous"}
    terminal_path.write_text(json.dumps(previous))
    terminal_path.chmod(0o644)
    # A large identity makes a partial direct write observable if publication
    # ever stops using an atomic replacement.
    rollout_id = "rollout-" + "x" * 60_000
    command = e2b_executor._supervise_evaluation_command(
        shlex.join([sys.executable, "-c", "import time; time.sleep(0.1)"]),
        2.0,
        rollout_id=rollout_id,
        attempt_id="new-attempt",
        terminal_path=str(terminal_path),
    )
    expected = {
        "rollout_id": rollout_id,
        "attempt_id": "new-attempt",
        "phase": "evaluation_finished",
        "exit_code": 0,
    }
    supervisor = subprocess.Popen(
        shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 5
        while supervisor.poll() is None:
            assert json.loads(terminal_path.read_text()) in (previous, expected)
            assert time.monotonic() < deadline
            time.sleep(0.001)
        _, errors = supervisor.communicate(timeout=1)
        assert supervisor.returncode == 0
        assert errors == b""
        assert json.loads(terminal_path.read_text()) == expected
        assert stat.S_IMODE(terminal_path.stat().st_mode) == 0o600
        assert not list(tmp_path.glob(".evaluation-terminal.*"))
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.communicate(timeout=3)


def test_evaluation_supervisor_does_not_publish_when_cleanup_cannot_reap_process(tmp_path):
    terminal_path = tmp_path / "terminal.json"
    # An uninterruptible process cannot be created reliably in a unit test.
    # Model that subprocess boundary inside a separate supervisor interpreter.
    setup = """import subprocess
import sys

class UnreapableProcess:
    pid = 2147483647

    def wait(self, timeout):
        raise subprocess.TimeoutExpired('evaluator', timeout)

subprocess.Popen = lambda *args, **kwargs: UnreapableProcess()
"""
    arguments = [
        "supervisor",
        "0.01",
        "0.01",
        str(terminal_path),
        "rollout-1",
        "attempt-1",
        "evaluator",
    ]
    script = (
        setup
        + f"sys.argv = {arguments!r}\n"
        + f"exec({e2b_executor._EVALUATION_SUPERVISOR_CODE!r})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=5
    )
    assert completed.returncode == 124
    assert completed.stderr == ""
    assert not terminal_path.exists()
    assert not list(tmp_path.glob(".evaluation-terminal.*"))


def test_evaluation_supervisor_record_write_failure_preserves_evaluator_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(e2b_executor, "_BOOTSTRAP_INTERPRETER", sys.executable)
    terminal_path = tmp_path / "terminal.json"
    # Replacement cannot overwrite a directory, including when tests run as
    # root. This fails after temporary-file creation and exercises its cleanup.
    terminal_path.mkdir()
    command = e2b_executor._supervise_evaluation_command(
        shlex.join([sys.executable, "-c", "import sys; sys.exit(17)"]),
        2.0,
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        terminal_path=str(terminal_path),
    )
    completed = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=5)
    assert completed.returncode == 17
    assert "Evaluation terminal record unavailable" in completed.stderr
    assert terminal_path.is_dir()
    assert not list(tmp_path.glob(".evaluation-terminal.*"))


@pytest.mark.parametrize("has_result", [False, True])
def test_confirmed_evaluation_timeout_preserves_result_or_reports_unscored_failure(has_result):
    input_data = RolloutInput(rollout_id="rollout-1", task={})
    expected_result = RemoteRolloutResult(rollout_id=input_data.rollout_id, reward=1.0)
    reads = []

    async def read(path, **_kwargs):
        reads.append(path)
        if path == e2b_executor._RESULT_PATH:
            if has_result:
                return dump_artifact(encode_result(expected_result))
            raise FileNotFoundError(path)
        return json.dumps(
            {"rollout_id": input_data.rollout_id, "attempt_id": "attempt-1", "phase": "evaluator"}
        )

    error = e2b_executor._EvaluationCommandTimeout("evaluation budget expired")
    assert error.exit_code == 124
    assert e2b_executor._is_timeout_exception(error)
    assert not e2b_executor._is_command_transport_error(error)
    result = asyncio.run(
        e2b_executor._command_failure_result(
            SimpleNamespace(files=SimpleNamespace(read=read)),
            input_data,
            error,
            trace=None,
            log_context="test",
            default_status="env_error",
            attempt_id="attempt-1",
            deadline=time.monotonic() + 10,
            io_seconds=1,
        )
    )
    assert reads.count(e2b_executor._RESULT_PATH) == 1
    assert result.metrics["e2b_timed_out"] == 1.0
    if has_result:
        assert result.status == "success"
        assert result.reward == 1.0
        assert e2b_executor._STATE_PATH not in reads
    else:
        assert result.status == "env_error"
        assert not result.segments
