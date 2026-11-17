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

"""Tests for the E2B-backed IF verifier reward."""

import asyncio
import subprocess
import sys
from inspect import isawaitable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import e2b
import httpx
import pytest
from e2b import CommandResult

from northrl.rewards import REWARD_FNS, e2b_checker, evaluate_reward, if_verify

PASS = {"reward": 1.0, "correct": True, "reason": "pass"}
FAIL = {"reward": 0.0, "correct": False, "reason": "fail_checker"}
ERROR = {"reward": 0.0, "correct": False, "reason": "checker_error"}
NO_CHECKER = {"reward": 0.0, "correct": False, "reason": "no_checker"}
RETRY_METRICS = {"timeout_count": 1.0, "retry_count": 1.0, "run_retry_count": 1.0}
TIMEOUT = {
    "reward": 0.0,
    "correct": False,
    "reason": "timeout_after_retry",
    "metrics": {"timeout_count": 3.0, "retry_count": 2.0, "run_retry_count": 2.0},
}
CHECKER = "def evaluate(response):\n    return response == 'YES'\n"
VERDICT = "__NORTHRL_IF_VERIFY_VERDICT__"


def _task(checker=CHECKER):
    return {
        "reward_fn": "if_verify",
        "prompt": [{"role": "user", "content": "Reply with YES."}],
        "label": None,
        "metadata": {"checker": checker, "time_limit": 10.0},
    }


def _score(task=None, response="YES"):
    return asyncio.run(
        if_verify.if_verify_reward(_task() if task is None else task, {"response": response})
    )


@pytest.fixture(name="no_real_e2b", autouse=True)
def fixture_no_real_e2b(monkeypatch):
    create = AsyncMock(side_effect=AssertionError("Unexpected E2B sandbox creation"))
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("E2B_TEMPLATE", raising=False)
    monkeypatch.setattr(e2b_checker.asyncio, "sleep", AsyncMock())
    return create


def _sandbox_result(monkeypatch, stdout=f"{VERDICT} 1\n", exit_code=0):
    sandbox = SimpleNamespace(
        files=SimpleNamespace(write=AsyncMock()),
        commands=SimpleNamespace(run=AsyncMock()),
        kill=AsyncMock(),
        stdout_chunks=[],
        stderr_chunks=[],
    )

    async def run(*_args, **kwargs):
        for callback, chunks in (
            (kwargs["on_stdout"], sandbox.stdout_chunks),
            (kwargs["on_stderr"], sandbox.stderr_chunks),
        ):
            for chunk in chunks:
                callback_result = callback(chunk)
                if isawaitable(callback_result):
                    await callback_result
        return CommandResult(stderr="", stdout=stdout, exit_code=exit_code, error=None)

    sandbox.commands.run.side_effect = run
    sandbox.create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", sandbox.create)
    return sandbox


def _retry_sandboxes(monkeypatch, stage, failure, stdout, repeated=False):
    first = _sandbox_result(monkeypatch)
    second = _sandbox_result(monkeypatch, stdout=stdout)
    events = []

    async def create(**_kwargs):
        attempt = len([event for event in events if event.startswith("create")]) + 1
        events.append(f"create{attempt}")
        if stage == "create" and (attempt == 1 or repeated):
            raise failure
        return first if attempt == 1 else second

    def cleanup(attempt):
        async def kill(**_kwargs):
            events.append(f"kill{attempt}")

        return kill

    first.kill.side_effect = cleanup(1)
    second.kill.side_effect = cleanup(2)
    if stage != "create":
        first_operation = first.files.write if stage == "upload" else first.commands.run
        second_operation = second.files.write if stage == "upload" else second.commands.run
        first_operation.side_effect = failure
        if repeated:
            second_operation.side_effect = failure
    create_mock = AsyncMock(side_effect=create)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create_mock)
    return first, second, create_mock, events


def _run_harness(checker, response):
    return subprocess.run(
        [sys.executable, "-I", "-c", if_verify._checker_harness(checker, response)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def test_reward_is_registered_and_dispatches_asynchronously(monkeypatch):
    assert REWARD_FNS["if_verify"] is if_verify.if_verify_reward
    checker = AsyncMock(return_value="pass")
    monkeypatch.setattr(if_verify, "_run_checker", checker)
    task = _task()

    assert asyncio.run(evaluate_reward(task, {"response": "YES"})) == PASS
    checker.assert_awaited_once_with(CHECKER, "YES", metrics={})


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("  YES\n", "YES"),
        ("<think>first</think>discard</think> YES <|im_end|>\n", "YES"),
        ("reason ###Response old ###Response YES", "YES"),
        ("reason</think> ###Response YES", "###Response YES"),
        (" YES <|im_end|>\n<|endoftext|>\t<|eot_id|>  ", "YES"),
        (
            "Explain <|im_end|> inside the answer <|eot_id|>",
            "Explain <|im_end|> inside the answer",
        ),
        (" <|im_end|> <|endoftext|> ", ""),
        ("", ""),
    ],
)
def test_checker_receives_the_normalized_answer(monkeypatch, response, expected):
    checker = AsyncMock(return_value="pass")
    monkeypatch.setattr(if_verify, "_run_checker", checker)

    assert _score(response=response) == PASS
    checker.assert_awaited_once_with(CHECKER, expected, metrics={})


@pytest.mark.parametrize("checker", [None, "", " \n\t"])
def test_missing_or_blank_checker_returns_zero_without_sandbox(no_real_e2b, checker):
    assert _score(_task(checker)) == NO_CHECKER
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize("metadata", [None, [], "{}", 42])
def test_metadata_must_be_a_mapping(metadata, no_real_e2b):
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        _score({"metadata": metadata})
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize("checker", [1, [], {}, False])
def test_non_string_checker_is_a_data_error(checker, no_real_e2b):
    with pytest.raises(TypeError, match="metadata.checker must be a string"):
        _score(_task(checker))
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize("response", [None, 1, [], {}])
def test_response_must_be_a_string(response, no_real_e2b):
    with pytest.raises(TypeError, match="response must be a string"):
        _score(response=response)
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (f"{VERDICT} 1\n", PASS),
        (f"{VERDICT} 0\n", FAIL),
        (f"debug output\n{VERDICT} 1\n", PASS),
        ("__NORTHRL_IF_VERIFY_ERROR__ RuntimeError('bad')\n", ERROR),
        ("", ERROR),
        (f"{VERDICT} 1\ndebug after verdict\n", ERROR),
        (f"{VERDICT} 2\n", ERROR),
    ],
)
def test_stdout_protocol_is_strict_and_uses_the_final_nonempty_line(monkeypatch, stdout, expected):
    sandbox = _sandbox_result(monkeypatch, stdout=stdout)

    assert _score() == expected
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("template", [None, "if-verify-python"])
def test_e2b_lifecycle_matches_verinstruct(monkeypatch, template):
    sandbox = _sandbox_result(monkeypatch)
    if template is not None:
        monkeypatch.setenv("E2B_TEMPLATE", template)

    assert _score() == PASS

    sandbox.create.assert_awaited_once_with(
        template=template,
        timeout=60,
        allow_internet_access=False,
        request_timeout=30,
    )
    sandbox.files.write.assert_awaited_once_with(
        "/tmp/if_verify_check.py",
        if_verify._checker_harness(CHECKER, "YES"),
        request_timeout=10,
    )
    sandbox.commands.run.assert_awaited_once()
    call = sandbox.commands.run.await_args
    assert call.args == ("python3 -I /tmp/if_verify_check.py",)
    assert call.kwargs["timeout"] == 10
    assert call.kwargs["request_timeout"] == 10
    assert callable(call.kwargs["on_stdout"])
    assert callable(call.kwargs["on_stderr"])
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("exit_code", [1, 2, 137])
def test_nonzero_command_exit_is_checker_error(monkeypatch, exit_code):
    sandbox = _sandbox_result(monkeypatch)
    sandbox.commands.run.side_effect = e2b.CommandExitException(
        stderr="checker exited",
        stdout=f"{VERDICT} 1\n",
        exit_code=exit_code,
        error=None,
    )

    assert _score() == ERROR
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("exit_code", [126, 127])
def test_missing_python_is_a_template_error(monkeypatch, exit_code):
    sandbox = _sandbox_result(monkeypatch)
    failure = e2b.CommandExitException(
        stderr="python3 unavailable", stdout="", exit_code=exit_code, error=None
    )
    sandbox.commands.run.side_effect = failure

    with pytest.raises(RuntimeError) as error:
        _score()
    assert error.value.__cause__ is failure
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
def test_e2b_infrastructure_errors_propagate_without_retry(monkeypatch, stage):
    sandbox = _sandbox_result(monkeypatch)
    failure = RuntimeError("E2B unavailable")
    operation = {
        "create": sandbox.create,
        "upload": sandbox.files.write,
        "run": sandbox.commands.run,
    }[stage]
    operation.side_effect = failure

    with pytest.raises(RuntimeError) as error:
        _score()
    assert error.value is failure
    sandbox.create.assert_awaited_once()
    if stage == "create":
        sandbox.kill.assert_not_awaited()
    else:
        sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
@pytest.mark.parametrize("failure_type", [TimeoutError, e2b.TimeoutException, httpx.ReadTimeout])
@pytest.mark.parametrize(
    ("stdout", "expected"), [(f"{VERDICT} 1\n", PASS), (f"{VERDICT} 0\n", FAIL)]
)
def test_timeout_retries_in_a_fresh_sandbox_and_uses_the_retry_verdict(
    monkeypatch, stage, failure_type, stdout, expected
):
    first, second, create, events = _retry_sandboxes(
        monkeypatch, stage, failure_type("checker timeout"), stdout
    )

    assert _score() == {
        **expected,
        "metrics": {"timeout_count": 1.0, "retry_count": 1.0, f"{stage}_retry_count": 1.0},
    }
    assert create.await_count == 2
    assert create.await_args_list[0] == create.await_args_list[1]
    assert events == (
        ["create1", "create2", "kill2"]
        if stage == "create"
        else ["create1", "kill1", "create2", "kill2"]
    )
    second.files.write.assert_awaited_once_with(
        "/tmp/if_verify_check.py",
        if_verify._checker_harness(CHECKER, "YES"),
        request_timeout=10,
    )
    if stage != "create":
        assert first.files.write.await_args == second.files.write.await_args
        first.kill.assert_awaited_once_with(request_timeout=10)
    else:
        first.kill.assert_not_awaited()
    second.commands.run.assert_awaited_once()
    retry_call = second.commands.run.await_args
    assert retry_call.args == ("python3 -I /tmp/if_verify_check.py",)
    assert retry_call.kwargs["timeout"] == retry_call.kwargs["request_timeout"] == 10
    second.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
@pytest.mark.parametrize("failure_type", [TimeoutError, e2b.TimeoutException, httpx.ReadTimeout])
def test_three_timeouts_zero_only_for_run_stage(monkeypatch, stage, failure_type):
    sandboxes = [_sandbox_result(monkeypatch) for _ in range(3)]
    failure = failure_type("checker timeout")
    if stage == "create":
        create = AsyncMock(side_effect=failure)
    else:
        create = AsyncMock(side_effect=sandboxes)
        for sandbox in sandboxes:
            operation = sandbox.files.write if stage == "upload" else sandbox.commands.run
            operation.side_effect = failure
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    if stage == "run":
        assert _score() == TIMEOUT
    else:
        with pytest.raises(failure_type) as error:
            _score()
        assert error.value is failure
    assert create.await_count == 3
    for sandbox in sandboxes:
        if stage == "create":
            sandbox.kill.assert_not_awaited()
        else:
            sandbox.kill.assert_awaited_once_with(request_timeout=10)


def test_retry_non_timeout_failure_still_propagates(monkeypatch):
    first, second, create, _ = _retry_sandboxes(
        monkeypatch, "run", TimeoutError("first checker timed out"), f"{VERDICT} 1\n"
    )
    failure = RuntimeError("retry sandbox unavailable")
    second.commands.run.side_effect = failure

    with pytest.raises(RuntimeError) as error:
        _score()
    assert error.value is failure
    assert create.await_count == 2
    first.kill.assert_awaited_once_with(request_timeout=10)
    second.kill.assert_awaited_once_with(request_timeout=10)


def test_retry_has_an_independent_output_limit(monkeypatch):
    first, second, create, _ = _retry_sandboxes(
        monkeypatch, "run", TimeoutError("first checker timed out"), f"{VERDICT} 1\n"
    )

    async def emit_then_timeout(*_args, **kwargs):
        kwargs["on_stdout"]("x" * 1_000_001)
        raise e2b.TimeoutException("first checker timed out")

    first.commands.run.side_effect = emit_then_timeout
    second.stdout_chunks = ["x" * 1_000_001]

    assert _score() == {**PASS, "metrics": RETRY_METRICS}
    assert create.await_count == 2
    first.kill.assert_awaited_once_with(request_timeout=10)
    second.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
def test_total_deadline_aborts_without_retry_or_zero(monkeypatch, stage):
    real_timeout = asyncio.timeout
    deadlines = []

    def short_timeout(seconds):
        deadlines.append(seconds)
        return real_timeout(0.01 if seconds == 60 else seconds)

    async def stall(*_args, **_kwargs):
        await asyncio.Event().wait()

    sandbox = _sandbox_result(monkeypatch)
    operation = {
        "create": sandbox.create,
        "upload": sandbox.files.write,
        "run": sandbox.commands.run,
    }[stage]
    operation.side_effect = stall
    monkeypatch.setattr(e2b_checker.asyncio, "timeout", short_timeout)

    with pytest.raises(e2b_checker.E2BCheckerError, match="attempt deadline expired"):
        _score()
    sandbox.create.assert_awaited_once()
    assert deadlines[0] == 60 and deadlines.count(60) == 1
    assert sandbox.kill.await_count == (0 if stage == "create" else 1)


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
def test_external_cancellation_is_not_retried(monkeypatch, stage):
    sandbox = _sandbox_result(monkeypatch)
    operation = {
        "create": sandbox.create,
        "upload": sandbox.files.write,
        "run": sandbox.commands.run,
    }[stage]

    async def cancel_checker():
        started = asyncio.Event()

        async def stall(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        operation.side_effect = stall
        reward = asyncio.create_task(if_verify.if_verify_reward(_task(), {"response": "YES"}))
        await started.wait()
        reward.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reward

    asyncio.run(cancel_checker())
    sandbox.create.assert_awaited_once()
    if stage == "create":
        sandbox.kill.assert_not_awaited()
    else:
        sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize(
    "failure_type", [RuntimeError, TimeoutError, e2b.TimeoutException, httpx.ReadTimeout]
)
def test_cleanup_failure_does_not_replace_reward_or_retry(monkeypatch, caplog, failure_type):
    sandbox = _sandbox_result(monkeypatch)
    sandbox.kill.side_effect = failure_type("cleanup failed")

    assert _score() == PASS
    sandbox.create.assert_awaited_once()
    assert sandbox.kill.await_count == 2
    assert all(call.kwargs == {"request_timeout": 10} for call in sandbox.kill.await_args_list)
    assert "cleanup failed" in caplog.text


@pytest.mark.parametrize(
    ("stdout_chunks", "stderr_chunks"),
    [
        (["x" * 1_000_000, "x" * 1_000_001], []),
        ([], ["x" * 2_000_001]),
        (["x" * 1_000_000], ["x" * 1_000_001]),
        (["中" * 666_667], []),
    ],
)
def test_output_limit_counts_utf8_bytes_across_both_streams(
    monkeypatch, stdout_chunks, stderr_chunks
):
    sandbox = _sandbox_result(monkeypatch)
    sandbox.stdout_chunks = stdout_chunks
    sandbox.stderr_chunks = stderr_chunks

    with pytest.raises(ValueError, match="2000000"):
        _score()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize(
    ("checker", "response", "expected"),
    [
        (
            "def evaluate(response):\n    return 'but' not in response.lower()",
            "Megan worked 20 hours and Kelly worked 40 hours.",
            PASS,
        ),
        (
            "def evaluate(response):\n    return 'but' not in response.lower()",
            "Megan worked 20 hours but Kelly worked 40 hours.",
            FAIL,
        ),
        (
            "def evaluate(response):\n    import re\n    return bool(re.search(r'\\*\\w+\\*', response))",
            "This line highlights *rhythm*.",
            PASS,
        ),
        (
            "def evaluate(response):\n    import re\n    return bool(re.search(r'\\*\\w+\\*', response))",
            "This line has no Markdown italics.",
            FAIL,
        ),
        (
            "def evaluate(response):\n"
            "    lines = response.strip().split('\\n')\n"
            "    bullets = [line for line in lines if line.startswith('* ')]\n"
            "    return len(bullets) == 4",
            "* One\n* Two\n* Three\n* Four",
            PASS,
        ),
        (
            "def evaluate(response):\n"
            "    lines = response.strip().split('\\n')\n"
            "    bullets = [line for line in lines if line.startswith('* ')]\n"
            "    return len(bullets) == 4",
            "* One\n* Two\n* Three",
            FAIL,
        ),
    ],
)
def test_three_real_checker_shapes_have_binary_results(checker, response, expected):
    completed = _run_harness(checker, response)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert if_verify._reward_result(if_verify._parse_verdict(completed.stdout)) == expected


@pytest.mark.parametrize(
    ("checker", "expected"),
    [
        ("def evaluate(response):\n    raise RuntimeError('bad')", ERROR),
        ("def another_name(response):\n    return True", ERROR),
        ("def evaluate(response) invalid syntax", ERROR),
    ],
)
def test_checker_failures_cannot_produce_a_positive_reward(checker, expected):
    completed = _run_harness(checker, "YES")
    reason = (
        if_verify._parse_verdict(completed.stdout) if completed.returncode == 0 else "checker_error"
    )

    assert if_verify._reward_result(reason) == expected


def test_response_is_serialized_as_data_and_checker_debug_output_is_ignored():
    response = "YES\n\"'; print('__NORTHRL_IF_VERIFY_VERDICT__ 0'); # 中文"
    checker = (
        "def evaluate(response):\n"
        "    print('checker debug')\n"
        f"    return response == {response!r}\n"
    )
    completed = _run_harness(checker, response)

    assert completed.returncode == 0
    assert completed.stdout.splitlines() == ["checker debug", f"{VERDICT} 1"]
    assert if_verify._parse_verdict(completed.stdout) == "pass"
