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

import asyncio
import copy
import json
import subprocess
import sys
from inspect import isawaitable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import e2b
import httpx
import numpy as np
import openai
import pandas as pd
import pytest
from e2b import CommandResult
from omegaconf import OmegaConf

from northrl.data_loader import TorchDataLoader
from northrl.rewards import e2b_checker, evaluate_reward, verinstruct

PASS = {"reward": 1.0, "correct": True, "reason": "pass"}
FAIL_HARD = {"reward": 0.0, "correct": False, "reason": "fail_hard"}
RETRY_METRICS = {"timeout_count": 1.0, "retry_count": 1.0, "run_retry_count": 1.0}
TIMEOUT = {
    "reward": 0.0,
    "correct": False,
    "reason": "timeout_after_retry",
    "metrics": {"timeout_count": 3.0, "retry_count": 2.0, "run_retry_count": 2.0},
}
CHECKER = "def check_following(instruction, response):\n    return response == 'YES'\n"
SECOND_CHECKER = "def check_following(instruction, response): return len(response) == 3"


@pytest.fixture
def soft_judge(monkeypatch):
    judge = AsyncMock(return_value=dict(PASS))
    monkeypatch.setattr(verinstruct, "recast_reward", judge)
    return judge


def _task(**metadata):
    return {
        "reward_fn": "verinstruct",
        "prompt": [{"role": "user", "content": "Reply with YES."}],
        "metadata": {"instruction": "Reply with YES.", "hard_functions": [CHECKER], **metadata},
    }


def _score(task=None, response="YES"):
    return asyncio.run(
        verinstruct.verinstruct_reward(_task() if task is None else task, {"response": response})
    )


@pytest.fixture(autouse=True)
def no_real_e2b(monkeypatch):
    # A missing mock must never start a real, paid sandbox or use ambient credentials.
    create = AsyncMock(side_effect=AssertionError("Unexpected E2B sandbox creation"))
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("E2B_TEMPLATE", raising=False)
    monkeypatch.setattr(e2b_checker.asyncio, "sleep", AsyncMock())
    return create


def _sandbox_result(monkeypatch, stdout="True\n", exit_code=0):
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
                result = callback(chunk)
                if isawaitable(result):
                    await result
        return CommandResult(stderr="", stdout=stdout, exit_code=exit_code, error=None)

    sandbox.commands.run.side_effect = run
    sandbox.create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", sandbox.create)
    return sandbox


def _retry_sandboxes(monkeypatch, stage, failure, stdout="True\n", repeated=False):
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


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("  YES\n", "YES"),
        ("<think>first</think>discard</think> YES <|im_end|>\n", "YES"),
        ("reason ###Response old ###Response YES", "YES"),
        ("reason</think> ###Response YES", "###Response YES"),
        (" YES <|im_end|>\n<|endoftext|>\t<|eot_id|>  ", "YES"),
        ("Explain <|im_end|> inside the answer <|eot_id|>", "Explain <|im_end|> inside the answer"),
        (" <|im_end|> <|endoftext|> ", ""),
        ("", ""),
    ],
)
def test_both_stages_receive_the_same_normalized_answer(
    monkeypatch, soft_judge, response, expected
):
    sandbox = _sandbox_result(monkeypatch)
    task = _task()
    result = {"response": response, "finish_reason": "stop", "prompt_str": "rendered"}
    original_task, original_result = copy.deepcopy(task), copy.deepcopy(result)
    original_harness = verinstruct._hard_harness
    harness_calls = []

    def capture_harness(instruction, answer, functions):
        harness_calls.append((instruction, answer, functions))
        return original_harness(instruction, answer, functions)

    monkeypatch.setattr(verinstruct, "_hard_harness", capture_harness)

    assert asyncio.run(verinstruct.verinstruct_reward(task, result)) == PASS
    assert harness_calls == [("Reply with YES.", expected, [CHECKER])]
    soft_judge.assert_awaited_once_with(task, {**result, "response": expected})
    assert task == original_task
    assert result == original_result
    sandbox.create.assert_awaited_once()


@pytest.mark.parametrize("metadata", [None, {}, {"hard_functions": None}, {"hard_functions": []}])
def test_no_hard_constraints_need_no_sandbox_configuration(no_real_e2b, soft_judge, metadata):
    task = {"prompt": "Reply with YES.", "metadata": metadata}

    assert _score(task) == PASS
    soft_judge.assert_awaited_once_with(task, {"response": "YES"})
    no_real_e2b.assert_not_awaited()


def test_missing_metadata_skips_hard_checks(no_real_e2b, soft_judge):

    assert _score({"prompt": "Reply with YES."}) == PASS
    soft_judge.assert_awaited_once()
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize("metadata", [[], "{}", 42])
def test_invalid_metadata_fails_before_judging(soft_judge, metadata):
    with pytest.raises(TypeError):
        _score({"prompt": "Reply with YES.", "metadata": metadata})
    soft_judge.assert_not_awaited()


@pytest.mark.parametrize("functions", ["source", {"source": CHECKER}, [None], [123], (CHECKER,)])
def test_hard_functions_must_be_a_list_of_source_strings(soft_judge, functions):
    with pytest.raises(TypeError):
        _score(_task(hard_functions=functions))
    soft_judge.assert_not_awaited()


def test_numpy_hard_functions_are_supported(monkeypatch, soft_judge):
    _sandbox_result(monkeypatch)
    original_harness = verinstruct._hard_harness
    received = []

    def capture(instruction, response, functions):
        received.append(functions)
        return original_harness(instruction, response, functions)

    monkeypatch.setattr(verinstruct, "_hard_harness", capture)

    assert _score(_task(hard_functions=np.array([CHECKER, SECOND_CHECKER], dtype=object))) == PASS
    assert received == [[CHECKER, SECOND_CHECKER]]
    assert isinstance(received[0], list)
    soft_judge.assert_awaited_once()


@pytest.mark.parametrize(
    ("prompt", "metadata", "expected"),
    [
        ("prompt", {"instruction": "instruction", "prompt_text": "text"}, "instruction"),
        ("prompt", {"instruction": "", "prompt_text": "text"}, "text"),
        ("prompt", {}, "prompt"),
        ([{"role": "user", "content": "chat instruction"}], {}, ""),
        ("prompt", {"instruction": 123}, "123"),
    ],
)
def test_hard_instruction_matches_bp_precedence(
    monkeypatch, soft_judge, prompt, metadata, expected
):
    _sandbox_result(monkeypatch)
    harness = []

    def capture(instruction, response, functions):
        harness.append((instruction, response, functions))
        return "print(True)"

    monkeypatch.setattr(verinstruct, "_hard_harness", capture)
    task = {"prompt": prompt, "metadata": {"hard_functions": [CHECKER], **metadata}}

    assert _score(task) == PASS
    assert harness == [(expected, "YES", [CHECKER])]
    assert soft_judge.await_args.args[0] is task


@pytest.mark.parametrize("template", [None, "verinstruct-python"])
def test_e2b_lifecycle_has_execution_limits_and_network_disabled(monkeypatch, soft_judge, template):
    sandbox = _sandbox_result(monkeypatch)
    if template is not None:
        monkeypatch.setenv("E2B_TEMPLATE", template)

    assert _score() == PASS

    sandbox.create.assert_awaited_once_with(
        template=template, timeout=60, allow_internet_access=False, request_timeout=30
    )
    sandbox.files.write.assert_awaited_once_with(
        "/tmp/verinstruct_check.py",
        verinstruct._hard_harness("Reply with YES.", "YES", [CHECKER]),
        request_timeout=10,
    )
    sandbox.commands.run.assert_awaited_once()
    call = sandbox.commands.run.await_args
    assert call.args == ("python3 -I /tmp/verinstruct_check.py",)
    assert call.kwargs["timeout"] == 10
    assert call.kwargs["request_timeout"] == 10
    assert callable(call.kwargs["on_stdout"])
    assert callable(call.kwargs["on_stderr"])
    sandbox.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_awaited_once()


@pytest.mark.parametrize("stdout", ["False\n", "debug output\nTrue\n", "true", "", "True False"])
def test_hard_failure_short_circuits_soft_judge(monkeypatch, soft_judge, stdout):
    sandbox = _sandbox_result(monkeypatch, stdout=stdout)
    monkeypatch.delenv("LLM_JUDGE_URL", raising=False)
    monkeypatch.delenv("LLM_JUDGE_MODEL", raising=False)

    assert _score() == FAIL_HARD
    soft_judge.assert_not_awaited()
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("exit_code", [1, 2, 137])
def test_nonzero_command_exit_is_a_hard_failure(monkeypatch, soft_judge, exit_code):
    sandbox = _sandbox_result(monkeypatch)
    sandbox.commands.run.side_effect = e2b.CommandExitException(
        stderr="checker exited", stdout="True\n", exit_code=exit_code, error=None
    )

    assert _score() == FAIL_HARD
    soft_judge.assert_not_awaited()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("exit_code", [126, 127])
def test_cannot_launch_python_is_a_template_error(monkeypatch, soft_judge, exit_code):
    sandbox = _sandbox_result(monkeypatch)
    failure = e2b.CommandExitException(
        stderr="python3: not found or executable", stdout="", exit_code=exit_code, error=None
    )
    sandbox.commands.run.side_effect = failure

    with pytest.raises(RuntimeError) as error:
        _score()
    assert error.value.__cause__ is failure
    soft_judge.assert_not_awaited()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


def test_nonzero_exit_cannot_pass_even_with_true_stdout(monkeypatch, soft_judge):
    sandbox = _sandbox_result(monkeypatch, stdout="True\n", exit_code=1)

    assert _score() == FAIL_HARD
    soft_judge.assert_not_awaited()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize(
    ("soft_result", "expected"),
    [
        (PASS, PASS),
        (
            {**PASS, "metrics": {"soft_judge_score": 0.75}},
            {**PASS, "metrics": {"soft_judge_score": 0.75}},
        ),
        (
            {"reward": 0.0, "correct": False, "reason": "fail_llm"},
            {"reward": 0.0, "correct": False, "reason": "fail_soft"},
        ),
        (
            {"reward": 0.0, "correct": False, "reason": "no_verdict"},
            {"reward": 0.0, "correct": False, "reason": "no_verdict"},
        ),
    ],
)
def test_soft_stage_preserves_verdict_and_maps_failure_reason(soft_judge, soft_result, expected):
    soft_judge.return_value = dict(soft_result)

    assert _score({"prompt": "Reply with YES."}) == expected


def test_soft_judge_failures_propagate(soft_judge):
    failure = RuntimeError("judge unavailable")
    soft_judge.side_effect = failure

    with pytest.raises(RuntimeError) as error:
        _score({"prompt": "Reply with YES."})
    assert error.value is failure


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
def test_e2b_non_timeout_errors_propagate_without_retry(monkeypatch, soft_judge, stage):
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
    operation.assert_awaited_once()
    sandbox.create.assert_awaited_once()
    soft_judge.assert_not_awaited()
    if stage == "create":
        sandbox.kill.assert_not_awaited()
    else:
        sandbox.kill.assert_awaited_once_with(request_timeout=10)


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
@pytest.mark.parametrize("failure_type", [TimeoutError, e2b.TimeoutException, httpx.ReadTimeout])
@pytest.mark.parametrize(
    ("soft_result", "expected"),
    [
        (PASS, PASS),
        (
            {"reward": 0.0, "correct": False, "reason": "fail_llm"},
            {"reward": 0.0, "correct": False, "reason": "fail_soft"},
        ),
    ],
)
def test_timeout_retries_in_a_fresh_sandbox_and_still_uses_the_soft_judge(
    monkeypatch, soft_judge, stage, failure_type, soft_result, expected
):
    first, second, create, events = _retry_sandboxes(
        monkeypatch, stage, failure_type("checker timeout")
    )
    soft_judge.return_value = dict(soft_result)

    assert _score() == {
        **expected,
        "metrics": {
            **expected.get("metrics", {}),
            "timeout_count": 1.0,
            "retry_count": 1.0,
            f"{stage}_retry_count": 1.0,
        },
    }
    assert soft_judge.return_value == soft_result
    assert create.await_count == 2
    assert create.await_args_list[0] == create.await_args_list[1]
    assert events == (
        ["create1", "create2", "kill2"]
        if stage == "create"
        else ["create1", "kill1", "create2", "kill2"]
    )
    second.files.write.assert_awaited_once_with(
        "/tmp/verinstruct_check.py",
        verinstruct._hard_harness("Reply with YES.", "YES", [CHECKER]),
        request_timeout=10,
    )
    if stage != "create":
        assert first.files.write.await_args == second.files.write.await_args
        first.kill.assert_awaited_once_with(request_timeout=10)
    else:
        first.kill.assert_not_awaited()
    second.commands.run.assert_awaited_once()
    retry_call = second.commands.run.await_args
    assert retry_call.args == ("python3 -I /tmp/verinstruct_check.py",)
    assert retry_call.kwargs["timeout"] == retry_call.kwargs["request_timeout"] == 10
    second.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_awaited_once_with(_task(), {"response": "YES"})


def test_retry_hard_failure_still_short_circuits_the_soft_judge(monkeypatch, soft_judge):
    _, second, create, _ = _retry_sandboxes(
        monkeypatch, "run", e2b.TimeoutException("checker timeout"), stdout="False\n"
    )

    assert _score() == {**FAIL_HARD, "metrics": RETRY_METRICS}
    assert create.await_count == 2
    second.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_not_awaited()


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
@pytest.mark.parametrize("failure_type", [TimeoutError, e2b.TimeoutException, httpx.ReadTimeout])
def test_three_timeouts_zero_only_for_run_and_skip_soft_judge(
    monkeypatch, soft_judge, stage, failure_type
):
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
    soft_judge.assert_not_awaited()


def test_retry_non_timeout_failure_still_propagates_and_skips_soft_judge(monkeypatch, soft_judge):
    first, second, create, _ = _retry_sandboxes(
        monkeypatch, "run", TimeoutError("first checker timed out")
    )
    failure = RuntimeError("retry sandbox unavailable")
    second.commands.run.side_effect = failure

    with pytest.raises(RuntimeError) as error:
        _score()
    assert error.value is failure
    assert create.await_count == 2
    first.kill.assert_awaited_once_with(request_timeout=10)
    second.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_not_awaited()


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
def test_cancellation_propagates_and_kills_created_sandbox(monkeypatch, soft_judge, stage):
    sandbox = _sandbox_result(monkeypatch)
    operation = {
        "create": sandbox.create,
        "upload": sandbox.files.write,
        "run": sandbox.commands.run,
    }[stage]

    async def check_cancellation():
        started = asyncio.Event()

        async def stall(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        operation.side_effect = stall
        reward = asyncio.create_task(verinstruct.verinstruct_reward(_task(), {"response": "YES"}))
        await started.wait()
        reward.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reward

    asyncio.run(check_cancellation())
    sandbox.create.assert_awaited_once()
    if stage == "create":
        sandbox.kill.assert_not_awaited()
    else:
        sandbox.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_not_awaited()


@pytest.mark.parametrize("stage", ["create", "upload", "run"])
def test_total_timeout_aborts_without_retry_or_zero(monkeypatch, soft_judge, stage):
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
    if stage == "create":
        sandbox.kill.assert_not_awaited()
    else:
        sandbox.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["pass", "hard_failure", "error"])
def test_cleanup_failure_does_not_replace_the_result_or_primary_error(
    monkeypatch, soft_judge, caplog, outcome
):
    sandbox = _sandbox_result(monkeypatch, stdout="False" if outcome == "hard_failure" else "True")
    sandbox.kill.side_effect = RuntimeError("cleanup failed")
    if outcome == "error":
        failure = RuntimeError("execution failed")
        sandbox.commands.run.side_effect = failure
        with pytest.raises(type(failure)) as error:
            _score()
        assert error.value is failure
        soft_judge.assert_not_awaited()
    else:
        assert _score() == (PASS if outcome == "pass" else FAIL_HARD)
    assert sandbox.kill.await_count == (1 if outcome == "error" else 2)
    assert "cleanup failed" in caplog.text


def test_cleanup_has_a_separate_bounded_timeout(monkeypatch, soft_judge, caplog):
    real_timeout = asyncio.timeout
    deadlines = []

    def short_timeout(seconds):
        deadlines.append(seconds)
        return real_timeout(0.01 if seconds == 10 else seconds)

    async def stall(**_kwargs):
        await asyncio.Event().wait()

    sandbox = _sandbox_result(monkeypatch)
    sandbox.kill.side_effect = stall
    monkeypatch.setattr(e2b_checker.asyncio, "timeout", short_timeout)

    assert _score() == PASS
    assert deadlines == [60, 30, 10, 10, 10, 10]
    sandbox.create.assert_awaited_once()
    assert sandbox.kill.await_count == 2
    soft_judge.assert_awaited_once()
    assert caplog.records


@pytest.mark.parametrize("failure_type", [TimeoutError, e2b.TimeoutException, httpx.ReadTimeout])
def test_cleanup_timeout_does_not_retry_or_skip_soft_judge(
    monkeypatch, soft_judge, caplog, failure_type
):
    sandbox = _sandbox_result(monkeypatch)
    sandbox.kill.side_effect = failure_type("cleanup timed out")

    assert _score() == PASS
    sandbox.create.assert_awaited_once()
    assert sandbox.kill.await_count == 2
    soft_judge.assert_awaited_once()
    assert "cleanup timed out" in caplog.text


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
    monkeypatch, soft_judge, stdout_chunks, stderr_chunks
):
    sandbox = _sandbox_result(monkeypatch)
    sandbox.stdout_chunks = stdout_chunks
    sandbox.stderr_chunks = stderr_chunks

    with pytest.raises(ValueError, match="2000000"):
        _score()
    sandbox.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_not_awaited()


def test_output_at_exact_byte_limit_is_allowed(monkeypatch, soft_judge):
    sandbox = _sandbox_result(monkeypatch)
    sandbox.stdout_chunks = ["True\n"]
    sandbox.stderr_chunks = ["x" * 1_999_995]

    assert _score() == PASS
    sandbox.kill.assert_awaited_once_with(request_timeout=10)
    soft_judge.assert_awaited_once()


@pytest.mark.parametrize(
    ("functions", "stdout"),
    [
        ([CHECKER], "True\n"),
        (["def check_following(instruction, response): return False"], "False\n"),
        (["def check_following(instruction, response): raise RuntimeError('bad')"], "False\n"),
        (["def another_name(instruction, response): return True"], "False\n"),
        (["invalid python code !"], "False\n"),
        (
            [
                "def check_following(instruction, response): return False",
                "print('must not run')\ndef check_following(instruction, response): return True",
            ],
            "False\n",
        ),
        (
            [
                "hidden = 1\ndef check_following(instruction, response): return True",
                "def check_following(instruction, response): return hidden == 1",
            ],
            "False\n",
        ),
    ],
)
def test_hard_harness_checks_authored_sources_in_separate_namespaces(functions, stdout):
    # These fixtures are authored here, never loaded from an external dataset.
    harness = verinstruct._hard_harness("Reply with YES.", "YES", functions)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", harness],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    assert completed.stdout == stdout
    assert completed.stderr == ""


def test_hard_harness_quotes_instruction_and_response_as_data():
    instruction = "包含中文、引号 '\" 与换行\nprint('not executable')"
    response = "YES\n\"'; print('not executable'); #"
    checker = (
        "def check_following(instruction, response):\n"
        f"    return instruction == {instruction!r} and response == {response!r}\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", verinstruct._hard_harness(instruction, response, [checker])],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    assert completed.stdout == "True\n"
    assert completed.stderr == ""


@pytest.mark.parametrize("file_format", ["jsonl", "parquet"])
def test_mixed_dataset_rows_dispatch_metadata_to_verinstruct(
    monkeypatch, soft_judge, tmp_path, file_format
):
    records = [
        {**_task(hard_functions=[CHECKER, SECOND_CHECKER]), "label": None},
        {"reward_fn": "verinstruct", "prompt": [{"role": "user", "content": "Reply with YES."}]},
        {
            "reward_fn": "verinstruct",
            "prompt": [{"role": "user", "content": "Reply with YES."}],
            "metadata": None,
        },
    ]
    data_path = tmp_path / f"verinstruct.{file_format}"
    if file_format == "jsonl":
        data_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    else:
        pd.DataFrame(records).to_parquet(data_path, index=False)
    config = OmegaConf.create(
        {
            "files": [str(data_path)],
            "batch_size": 1,
            "rollout_repeat_n": 1,
            "prompt_key": "prompt",
            "filter_prompts": False,
            "shuffle": False,
            "drop_last": False,
            "max_prompt_length": 1024,
            "tokenizer": None,
            "seed": 42,
            "keep_batch_order": False,
        }
    )
    sandbox = _sandbox_result(monkeypatch)
    loader = TorchDataLoader(config)

    for index in range(len(records)):
        item = loader.get_next_item()
        assert item["reward_fn"] == "verinstruct"
        assert item.get("label") is None
        if index == 0:
            assert item["metadata"]["instruction"] == "Reply with YES."
            assert list(item["metadata"]["hard_functions"]) == [CHECKER, SECOND_CHECKER]
        else:
            assert item.get("metadata") is None
        assert (
            asyncio.run(
                evaluate_reward(
                    item,
                    {"response": "<think>reason</think> YES <|im_end|>"},
                    default_reward_fn="recast",
                )
            )
            == PASS
        )
        assert soft_judge.await_args.args == (item, {"response": "YES"})
    assert soft_judge.await_count == 3
    sandbox.create.assert_awaited_once()
    assert loader.get_next_item() is None


def test_soft_stage_uses_real_openai_sdk_with_normalized_answer(monkeypatch):
    requests = []
    clients = []
    settings = []
    original_http_client = openai.DefaultAsyncHttpxClient
    monkeypatch.setenv("LLM_JUDGE_URL", "http://judge.test/v1")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "if-verifier-example")
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-only-key")
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "9.5")

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "if-verifier-example",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Missing one condition. [[0]]"},
                    }
                ],
            },
        )

    def client(**kwargs):
        settings.append(kwargs)
        result = original_http_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", client)
    messages = [{"role": "user", "content": "Reply with YES."}]
    task = {
        "reward_fn": "verinstruct",
        "prompt": np.array(messages, dtype=object),
        "label": "LABEL MUST NOT APPEAR",
        "metadata": {"hard_functions": []},
    }

    score = asyncio.run(
        evaluate_reward(
            task,
            {"response": "<think>private reasoning</think> YES <|im_end|>", "prompt_str": "IGNORE"},
        )
    )

    assert score == {"reward": 0.0, "correct": False, "reason": "fail_soft"}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://judge.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-only-key"
    assert request.extensions["timeout"] == {"connect": 9.5, "read": 9.5, "write": 9.5, "pool": 9.5}
    assert json.loads(request.content) == {
        "model": "if-verifier-example",
        "messages": [
            {
                "role": "user",
                "content": (
                    "请判断回复是否符合指令的全部要求。\n"
                    f"【指令】{messages}\n【回复】YES\n"
                    "若完全符合,请输出 [[1]];否则输出 [[0]]。"
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    assert settings == [{"trust_env": False}]
    assert all(client.is_closed for client in clients)
