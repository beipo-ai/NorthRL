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

"""Exercise stage recovery and scoring boundaries without contacting E2B."""

import asyncio
import logging
import ssl
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import e2b
import httpx
import pytest
from e2b import CommandResult

from northrl.rewards import e2b_checker, evaluate_reward, verinstruct
from northrl.rewards._runtime_context import reward_runtime_context


@pytest.fixture(autouse=True)
def no_real_e2b(monkeypatch):
    create = AsyncMock(side_effect=AssertionError("Unexpected real sandbox creation"))
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    sleep = AsyncMock()
    monkeypatch.setattr(e2b_checker.asyncio, "sleep", sleep)
    monkeypatch.setattr(e2b_checker.random, "uniform", Mock(return_value=1.5))
    return sleep


def _sandbox(stdout="ok"):
    return SimpleNamespace(
        files=SimpleNamespace(write=AsyncMock()),
        commands=SimpleNamespace(
            run=AsyncMock(
                return_value=CommandResult(stdout=stdout, stderr="", exit_code=0, error=None)
            )
        ),
        kill=AsyncMock(),
    )


def _run(metrics=None):
    return asyncio.run(
        e2b_checker.run_e2b_checker("print('ok')", "/tmp/check.py", "test", metrics=metrics)
    )


@pytest.mark.parametrize("verifier", ["if_verify", "verinstruct"])
def test_both_verifiers_recover_after_two_placement_failures(monkeypatch, no_real_e2b, verifier):
    stdout = "__NORTHRL_IF_VERIFY_VERDICT__ 1\n" if verifier == "if_verify" else "True\n"
    sandbox = _sandbox(stdout)
    create = AsyncMock(
        side_effect=[e2b.SandboxException("500: Failed to place sandbox") for _ in range(2)]
        + [sandbox]
    )
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    soft = AsyncMock(return_value={"reward": 1.0, "correct": True, "reason": "pass"})
    monkeypatch.setattr(verinstruct, "recast_reward", soft)
    task = {
        "reward_fn": verifier,
        "prompt": "Reply YES",
        "metadata": {
            "checker": "def evaluate(response): return response == 'YES'",
            "instruction": "Reply YES",
            "hard_functions": [
                "def check_following(instruction,response): return response == 'YES'"
            ],
        },
    }
    score = asyncio.run(evaluate_reward(task, {"response": "YES"}))
    assert score["reward"] == 1
    assert score["metrics"] == {"create_retry_count": 2.0, "retry_count": 2.0}
    assert create.await_count == 3
    assert all(c == create.await_args_list[0] for c in create.await_args_list)
    assert no_real_e2b.await_count == 2
    sandbox.files.write.assert_awaited_once()
    sandbox.commands.run.assert_awaited_once()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)
    assert soft.await_count == int(verifier == "verinstruct")


def test_three_placement_failures_raise_without_fabricating_a_verdict(monkeypatch, no_real_e2b):
    failures = [e2b.SandboxException("500: Failed to place sandbox") for _ in range(3)]
    create = AsyncMock(side_effect=failures)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with pytest.raises(e2b.SandboxException) as caught:
        _run()
    assert caught.value is failures[-1]
    assert create.await_count == 3
    assert no_real_e2b.await_count == 2


@pytest.mark.parametrize(
    "error",
    [
        e2b.SandboxException("429: busy"),
        e2b.SandboxException("503: unavailable"),
        httpx.ReadError("connection reset"),
    ],
)
def test_transient_creation_errors_share_the_placement_budget(monkeypatch, no_real_e2b, error):
    sandbox = _sandbox()
    create = AsyncMock(side_effect=[error, e2b.SandboxException("500: busy"), sandbox])
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    assert _run().stdout == "ok"
    assert create.await_count == 3
    assert no_real_e2b.await_count == 2


@pytest.mark.parametrize(
    "error",
    [
        e2b.AuthenticationException("invalid key"),
        e2b.TemplateException("invalid template"),
        e2b.SandboxException("403: denied"),
        ssl.SSLCertVerificationError("invalid certificate"),
        ValueError("invalid configuration"),
    ],
)
def test_permanent_creation_errors_never_retry(monkeypatch, no_real_e2b, error):
    create = AsyncMock(side_effect=error)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with pytest.raises(type(error)) as caught:
        _run()
    assert caught.value is error
    create.assert_awaited_once()
    no_real_e2b.assert_not_awaited()


def test_creation_budget_is_not_reset_after_existing_timeout_retry(monkeypatch, no_real_e2b):
    sandbox = _sandbox()
    sandbox.commands.run.side_effect = e2b.TimeoutException("command timeout")
    last = e2b.SandboxException("500: last placement failure")
    create = AsyncMock(
        side_effect=[
            e2b.SandboxException("503: first placement failure"),
            sandbox,
            e2b.SandboxException("500: second placement failure"),
            last,
        ]
    )
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with pytest.raises(e2b.SandboxException) as caught:
        _run()
    assert caught.value is last
    assert create.await_count == 4
    assert no_real_e2b.await_count == 3
    sandbox.kill.assert_awaited_once()


def test_cancellation_during_creation_backoff_never_creates_another_sandbox(
    monkeypatch, no_real_e2b
):
    create = AsyncMock(side_effect=e2b.SandboxException("500: unavailable"))
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    no_real_e2b.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _run()
    create.assert_awaited_once()


def test_worker_stop_prevents_creation_recovery(monkeypatch, no_real_e2b):
    stop = Event()

    async def fail(**_kwargs):
        stop.set()
        raise e2b.SandboxException("500: unavailable")

    create = AsyncMock(side_effect=fail)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _run()
    create.assert_awaited_once()
    no_real_e2b.assert_not_awaited()


def test_creation_warning_is_short_and_redacted(monkeypatch, no_real_e2b, caplog):
    caplog.set_level(logging.WARNING, logger=e2b_checker.__name__)
    monkeypatch.setenv("DIAGNOSTIC_API_KEY", "sk-private-key")
    sandbox = _sandbox()
    create = AsyncMock(
        side_effect=[
            e2b.SandboxException("500: sk-private-key https://u:secret@host.test/path?key=secret"),
            sandbox,
        ]
    )
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    assert _run().stdout == "ok"
    assert len(caplog.records) == 1
    assert len(caplog.text) < 700
    assert "sk-private-key" not in caplog.text
    assert "secret" not in caplog.text


@pytest.mark.parametrize("stage", ["upload", "run"])
def test_upload_and_run_recover_after_two_transient_failures(monkeypatch, no_real_e2b, stage):
    sandboxes = [_sandbox() for _ in range(3)]
    error = e2b.SandboxException("503: other stage unavailable")
    for sandbox in sandboxes[:2]:
        operation = sandbox.files.write if stage == "upload" else sandbox.commands.run
        operation.side_effect = error
    create = AsyncMock(side_effect=sandboxes)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    metrics = {}
    assert _run(metrics).stdout == "ok"
    assert metrics == {"retry_count": 2.0, f"{stage}_retry_count": 2.0}
    assert create.await_count == 3
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once_with(request_timeout=10)
        sandbox.files.write.assert_awaited_once_with(
            "/tmp/check.py", "print('ok')", request_timeout=10
        )
    assert no_real_e2b.await_count == 2


def test_seven_mixed_attempts_keep_independent_budgets_and_identical_requests(monkeypatch):
    sandboxes = [_sandbox() for _ in range(5)]
    for sandbox in sandboxes[:2]:
        sandbox.files.write.side_effect = e2b.SandboxException("503: upload unavailable")
    sandboxes[2].commands.run.side_effect = TimeoutError("command timeout")
    sandboxes[3].commands.run.side_effect = httpx.ReadError("command connection reset")
    create = AsyncMock(
        side_effect=[e2b.SandboxException("500: Failed to place sandbox") for _ in range(2)]
        + sandboxes
    )
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    metrics = {}
    assert _run(metrics).stdout == "ok"
    assert create.await_count == 7
    assert all(call == create.await_args_list[0] for call in create.await_args_list)
    assert metrics == {
        "retry_count": 6.0,
        "create_retry_count": 2.0,
        "upload_retry_count": 2.0,
        "run_retry_count": 2.0,
        "timeout_count": 1.0,
    }
    for sandbox in sandboxes:
        sandbox.files.write.assert_awaited_once_with(
            "/tmp/check.py", "print('ok')", request_timeout=10
        )
        sandbox.kill.assert_awaited_once_with(request_timeout=10)
    for sandbox in sandboxes[:2]:
        sandbox.commands.run.assert_not_awaited()
    for sandbox in sandboxes[2:]:
        assert sandbox.commands.run.await_args.args == ("python3 -I /tmp/check.py",)


@pytest.mark.parametrize("stage", ["upload", "run"])
def test_non_timeout_stage_exhaustion_raises(monkeypatch, stage, no_real_e2b):
    sandboxes = [_sandbox() for _ in range(3)]
    failure = httpx.ReadError("connection reset")
    for sandbox in sandboxes:
        operation = sandbox.files.write if stage == "upload" else sandbox.commands.run
        operation.side_effect = failure
    create = AsyncMock(side_effect=sandboxes)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with pytest.raises(httpx.ReadError) as caught:
        _run()
    assert caught.value is failure
    assert create.await_count == 3
    assert no_real_e2b.await_count == 2


def test_mixed_run_failures_never_become_a_timeout_zero(monkeypatch):
    sandboxes = [_sandbox() for _ in range(3)]
    failures = [
        httpx.ReadError("connection reset"),
        TimeoutError(),
        e2b.TimeoutException("timeout"),
    ]
    for sandbox, failure in zip(sandboxes, failures):
        sandbox.commands.run.side_effect = failure
    create = AsyncMock(side_effect=sandboxes)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    metrics = {}
    with pytest.raises(e2b.TimeoutException) as caught:
        _run(metrics)
    assert caught.value is failures[-1]
    assert metrics == {"retry_count": 2.0, "run_retry_count": 2.0, "timeout_count": 2.0}
    assert create.await_count == 3


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
def test_local_stage_timeout_recovers_with_a_fresh_attempt_deadline(monkeypatch, stage):
    real_timeout = asyncio.timeout
    timers = []

    def short_timeout(seconds):
        timer = real_timeout(0.2 if seconds == 60 else 0.01)
        if seconds == 60:
            timers.append(timer)
        return timer

    async def stall(*_args, **_kwargs):
        await asyncio.Event().wait()

    sandboxes = [_sandbox(), _sandbox()]
    create = AsyncMock(side_effect=[stall, sandboxes[1]])
    if stage == "create":
        calls = 0

        async def create_once(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                await stall()
            return sandboxes[1]

        create.side_effect = create_once
    else:
        create.side_effect = sandboxes
        operation = sandboxes[0].files.write if stage == "upload" else sandboxes[0].commands.run
        operation.side_effect = stall
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    monkeypatch.setattr(e2b_checker.asyncio, "timeout", short_timeout)
    metrics = {}
    assert _run(metrics).stdout == "ok"
    assert metrics == {"timeout_count": 1.0, "retry_count": 1.0, f"{stage}_retry_count": 1.0}
    assert len(timers) == 2 and timers[0] is not timers[1]
    assert timers[1].when() > timers[0].when()
    assert not any(timer.expired() for timer in timers)
    assert create.await_count == 2


def test_simultaneous_stage_and_attempt_deadline_never_retries(monkeypatch, no_real_e2b):
    real_timeout_at = asyncio.timeout_at
    deadline = None

    def same_deadline(_seconds):
        nonlocal deadline
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + 0.01
        return real_timeout_at(deadline)

    async def stall(**_kwargs):
        await asyncio.Event().wait()

    create = AsyncMock(side_effect=stall)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    monkeypatch.setattr(e2b_checker.asyncio, "timeout", same_deadline)
    with pytest.raises(e2b_checker.E2BCheckerError, match="attempt deadline expired"):
        _run()
    create.assert_awaited_once()
    no_real_e2b.assert_not_awaited()


def test_slow_cleanup_cannot_reclassify_a_stage_failure_as_total_expiry(monkeypatch):
    real_timeout = asyncio.timeout

    def short_attempt(seconds):
        return real_timeout(0.01 if seconds == 60 else seconds)

    async def slow_kill(**_kwargs):
        future = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_later(0.02, future.set_result, None)
        await future

    first, second = _sandbox(), _sandbox()
    first.files.write.side_effect = httpx.ReadError("upload reset")
    first.kill.side_effect = slow_kill
    create = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    monkeypatch.setattr(e2b_checker.asyncio, "timeout", short_attempt)
    assert _run().stdout == "ok"
    assert create.await_count == 2


def test_internal_command_cancellation_retries_with_the_run_budget(monkeypatch):
    sandboxes = [_sandbox() for _ in range(3)]
    for sandbox in sandboxes[:2]:
        sandbox.commands.run.side_effect = asyncio.CancelledError()
    create = AsyncMock(side_effect=sandboxes)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    metrics = {}
    assert _run(metrics).stdout == "ok"
    assert metrics == {"retry_count": 2.0, "run_retry_count": 2.0}
    assert create.await_count == 3


def test_internal_command_cancellation_exhausts_without_reward_zero(monkeypatch):
    sandbox = _sandbox()
    sandbox.commands.run.side_effect = asyncio.CancelledError()
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with pytest.raises(ConnectionError, match="owned checker command task cancelled"):
        _run()
    assert create.await_count == 3


def test_worker_stop_during_command_cancellation_never_retries(monkeypatch, no_real_e2b):
    stop = Event()

    async def stop_command(*_args, **_kwargs):
        stop.set()
        raise asyncio.CancelledError()

    sandbox = _sandbox()
    sandbox.commands.run.side_effect = stop_command
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _run()
    create.assert_awaited_once()
    no_real_e2b.assert_not_awaited()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


def test_worker_stop_during_deadline_cleanup_takes_priority(monkeypatch, no_real_e2b):
    stop = Event()
    real_timeout = asyncio.timeout
    original_cleanup = e2b_checker._cleanup_attempt

    async def stall(**_kwargs):
        await asyncio.Event().wait()

    async def cleanup_then_stop(*args):
        await original_cleanup(*args)
        stop.set()

    monkeypatch.setattr(e2b_checker, "_cleanup_attempt", cleanup_then_stop)
    monkeypatch.setattr(
        e2b_checker.asyncio,
        "timeout",
        lambda seconds: real_timeout(0.01 if seconds == 60 else seconds),
    )
    create = AsyncMock(side_effect=stall)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _run()
    create.assert_awaited_once()
    no_real_e2b.assert_not_awaited()


def test_external_cancellation_during_cleanup_is_not_swallowed(monkeypatch, no_real_e2b):
    async def scenario():
        started = asyncio.Event()

        async def stall_kill(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        sandbox = _sandbox()
        sandbox.kill.side_effect = stall_kill
        create = AsyncMock(return_value=sandbox)
        monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
        task = asyncio.create_task(e2b_checker.run_e2b_checker("pass", "/tmp/check.py", "test"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        create.assert_awaited_once()
        sandbox.kill.assert_awaited_once_with(request_timeout=10)
        no_real_e2b.assert_not_awaited()

    asyncio.run(scenario())


def test_unresolved_owned_command_blocks_retry_but_still_kills_sandbox(monkeypatch, no_real_e2b):
    real_timeout = asyncio.timeout

    def short_timeout(seconds):
        return real_timeout(0.01 if seconds in (3, 10) else seconds)

    async def scenario():
        release = asyncio.Event()
        tasks = []

        async def resist_cancel(*_args, **_kwargs):
            tasks.append(asyncio.current_task())
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return CommandResult(stdout="late", stderr="", exit_code=0, error=None)

        sandbox = _sandbox()
        sandbox.commands.run.side_effect = resist_cancel
        create = AsyncMock(return_value=sandbox)
        monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
        monkeypatch.setattr(e2b_checker.asyncio, "timeout", short_timeout)
        try:
            with pytest.raises(
                e2b_checker.E2BCheckerError, match="owned command task did not stop"
            ):
                await e2b_checker.run_e2b_checker("pass", "/tmp/check.py", "test")
            create.assert_awaited_once()
            sandbox.kill.assert_awaited_once_with(request_timeout=10)
            no_real_e2b.assert_not_awaited()
        finally:
            release.set()
            await asyncio.gather(*tasks)

    asyncio.run(scenario())


def test_cleanup_failure_does_not_block_an_eligible_retry(monkeypatch, caplog):
    first, second = _sandbox(), _sandbox()
    first.commands.run.side_effect = httpx.ReadError("run connection reset")
    first.kill.side_effect = RuntimeError("kill unavailable")
    create = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    assert _run().stdout == "ok"
    assert create.await_count == 2
    first.kill.assert_awaited_once_with(request_timeout=10)
    assert "preserve_original_error" in caplog.text


def test_missing_sandbox_is_already_cleaned_up(monkeypatch, caplog):
    sandbox = _sandbox()
    sandbox.kill.side_effect = e2b.SandboxNotFoundException("already gone")
    monkeypatch.setattr(e2b.AsyncSandbox, "create", AsyncMock(return_value=sandbox))
    assert _run().stdout == "ok"
    sandbox.kill.assert_awaited_once_with(request_timeout=10)
    assert "cleanup unconfirmed" not in caplog.text


@pytest.mark.parametrize("exit_code", [126, 127])
def test_returned_missing_python_exit_is_a_template_error(monkeypatch, exit_code):
    sandbox = _sandbox()
    sandbox.commands.run.return_value = CommandResult(
        stdout="", stderr="missing", exit_code=exit_code, error=None
    )
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    with pytest.raises(RuntimeError, match="executable python3"):
        _run()
    create.assert_awaited_once()


def test_command_exit_logs_bounded_redacted_tails_without_retry(monkeypatch, caplog):
    monkeypatch.setenv("CHECKER_API_KEY", "sk-private-checker-key")
    sandbox = _sandbox()
    sandbox.commands.run.side_effect = e2b.CommandExitException(
        stdout="A" * 100_000 + "sk-private-checker-key",
        stderr="B" * 100_000 + "sk-private-checker-key https://u:secret@host/path?key=secret",
        exit_code=1,
        error=None,
    )
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    result = _run()
    assert result.exit_code == 1
    create.assert_awaited_once()
    assert len(caplog.records) == 1 and len(caplog.text) < 2500
    assert "sk-private-checker-key" not in caplog.text and "secret" not in caplog.text
    assert "command_exit_code" in caplog.text
