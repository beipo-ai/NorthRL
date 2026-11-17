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

"""Exercise prepared-label judging without executing answers or contacting E2B."""

import asyncio
import copy
import hashlib
import json
import logging
import shlex
import ssl
import subprocess
import sys
import threading
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import e2b
import httpx
import pytest
from e2b import CommandResult

from northrl.rewards import REWARD_FNS, code_e2b, evaluate_reward
from northrl.rewards._runtime_context import reward_runtime_context, reward_stop_requested

CODE = "print(input())"
CASES = [{"stdin": "a\n", "expected_stdout": "a\n"}, {"stdin": "b\n", "expected_stdout": "b\n"}]
PLACEMENT_ERROR_MESSAGE = "500: Failed to place sandbox"


def _task(cases=None, **settings):
    return {
        "reward_fn": "code_e2b",
        "label": {
            "language": "python",
            "input_mode": "stdio",
            "cases": copy.deepcopy(CASES if cases is None else cases),
            **settings,
        },
    }


def _result(scores=(1.0, 1.0)):
    tests = [{"passed": score == 1.0, "score": score} for score in scores]
    return {
        "accepted": all(test["passed"] for test in tests),
        "score": sum(scores) / len(scores),
        "tests": tests,
    }


def _score(task=None, response=CODE, *, deadline=None):
    async def run():
        async with asyncio.timeout(deadline):
            return await code_e2b.code_e2b_reward(
                _task() if task is None else task, {"response": response}
            )

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def no_real_e2b(monkeypatch):
    create = AsyncMock(side_effect=AssertionError("Unexpected E2B sandbox creation"))
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    for variable in (
        "E2B_TEMPLATE",
        "E2B_DOMAIN",
        "E2B_COMMAND_TIMEOUT",
        "E2B_REQUEST_TIMEOUT",
        "E2B_UPLOAD_REQUEST_TIMEOUT",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("E2B_API_KEY", "test-api-key")
    monkeypatch.setenv("E2B_TEMPLATE", "test-code-judge")
    return create


@pytest.fixture(autouse=True)
def mock_tcp_user_timeout(monkeypatch):
    report = SimpleNamespace(
        status="configured",
        host="rpc.example.test",
        port=443,
        timeout_ms=3600000,
        matched_sockets=2,
        configured_sockets=1,
        failed_sockets=1,
        previous_timeouts_ms=(60000,),
        current_timeouts_ms=(3600000,),
    )
    helper = Mock(return_value=report)
    monkeypatch.setattr(code_e2b, "set_e2b_tcp_user_timeout", helper, raising=False)
    return helper


def _sandbox(monkeypatch, result=None, *, stdout=None, exit_code=0):
    result = _result() if result is None else result
    sandbox = SimpleNamespace(
        sandbox_id="test-sandbox",
        files=SimpleNamespace(write=AsyncMock()),
        warmup_run=AsyncMock(
            return_value=CommandResult(stderr="", stdout="", exit_code=0, error=None)
        ),
        warmup_handle=SimpleNamespace(pid=123, wait=AsyncMock(), disconnect=AsyncMock()),
        judge_run=AsyncMock(
            return_value=CommandResult(
                stderr="",
                stdout=json.dumps(result) if stdout is None else stdout,
                exit_code=exit_code,
                error=None,
            )
        ),
        get_host=lambda port: f"{port}-test-sandbox.e2b.example.test",
        is_running=AsyncMock(return_value=True),
        kill=AsyncMock(),
    )

    async def run(command, **kwargs):
        operation = sandbox.warmup_run if command == "true" else sandbox.judge_run
        result = await operation(command, **kwargs)
        if command == "true" and kwargs.get("background"):
            sandbox.warmup_handle.wait.return_value = result
            return sandbox.warmup_handle
        return result

    sandbox.commands = SimpleNamespace(run=AsyncMock(side_effect=run))
    sandbox.create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", sandbox.create)
    return sandbox


def _use_sandbox_sequence(monkeypatch, sandboxes):
    created = []

    async def create(**_kwargs):
        assert len(created) < len(sandboxes), "Unexpected additional sandbox attempt"
        if created and not isinstance(created[-1], BaseException):
            created[-1].kill.assert_awaited_once()
        sandbox = sandboxes[len(created)]
        if isinstance(sandbox, BaseException):
            created.append(sandbox)
            raise sandbox
        sandbox.sandbox_id = f"test-sandbox-{len(created) + 1}"
        created.append(sandbox)
        return sandbox

    create_mock = AsyncMock(side_effect=create)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create_mock)
    return create_mock


@pytest.fixture
def warmup_backoff(monkeypatch):
    delay = Mock(return_value=1.5)
    sleep = AsyncMock()
    monkeypatch.setattr(code_e2b.random, "uniform", delay)
    monkeypatch.setattr(code_e2b.asyncio, "sleep", sleep)
    return SimpleNamespace(delay=delay, sleep=sleep)


def test_reward_registry_dispatches_async_prepared_label(monkeypatch):
    assert REWARD_FNS["code_e2b"] is code_e2b.code_e2b_reward
    assert "leetcode" not in REWARD_FNS
    sandbox = _sandbox(monkeypatch)

    score = asyncio.run(
        evaluate_reward(_task(), {"response": CODE}, default_reward_fn="math_answer_only")
    )

    assert score["reward"] == 1.0
    assert score["correct"] is True
    assert score["status"] == "ok"
    assert score["reason"] == "accepted"
    assert score["metrics"] == {"passed_cases": 2.0, "total_cases": 2.0, "completed_batches": 1.0}
    assert score["sandbox_result"] == _result()
    sandbox.create.assert_awaited_once()


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("  print(1)\n", "print(1)"),
        ("before\n```python\nfirst\n```\nthen\n```py\nsecond\n```", "second"),
        ("<think>```python\nwrong\n```</think>\n```python\nright\n```", "right"),
        ("<think>reasoning</think>plain source", "plain source"),
        ("```python\nprint(1)\n```\n```python\n\n```", "print(1)"),
        ("```python\nprint(1)\n```\n```py\n \t\n```", "print(1)"),
        (
            "```python\nfirst\n```\n```py\n\n```\n```python\nlast\n```\n```py\n \n```",
            "last",
        ),
    ],
)
def test_extracts_final_code_after_thinking(response, expected):
    assert code_e2b._extract_code(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        " \n\t",
        "<think>reasoning</think> \n",
        "```python\n\n```",
        "<think>```python\nprint('thinking only')\n```</think>\n",
        "Explanation\n```python\n \t\n```\n```py\n\n```\n",
    ],
)
def test_empty_code_returns_zero_without_sandbox_and_logs_original_response(
    response, no_real_e2b, caplog, mock_tcp_user_timeout, monkeypatch
):
    monkeypatch.delenv("E2B_TEMPLATE")
    task = {**_task(), "task_id": 17, "group_id": "empty-group", "sample_id": "empty-sample"}
    result = {"response": response, "finish_reason": "stop"}

    score = asyncio.run(evaluate_reward(task, result))

    assert score["status"] == "ok"
    assert score["reward"] == 0.0
    assert score["correct"] is False
    assert score["reason"] == "no_code"
    no_real_e2b.assert_not_awaited()
    mock_tcp_user_timeout.assert_not_called()
    records = [record for record in caplog.records if record.name == code_e2b.__name__]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].args[-1] == response
    assert "sample_id='empty-sample'" in records[0].getMessage()
    assert "finish_reason='stop'" in records[0].getMessage()
    assert "response=" + repr(response) in records[0].getMessage()
    assert result == {"response": response, "finish_reason": "stop"}


def test_trailing_empty_fence_uploads_previous_nonempty_source(monkeypatch):
    sandbox = _sandbox(monkeypatch)
    response = (
        "<think>Reasoning.</think>\n```py\nprint('earlier')\n```\n"
        f"```python\n{CODE}\n```\n```python\n \t\n```"
    )

    score = _score(response=response)

    assert score["reward"] == 1.0
    request = json.loads(sandbox.files.write.await_args.args[1])
    assert request["code"] == CODE
    assert request["cases"] == CASES
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize("response", [None, 1, [], {}])
def test_non_string_code_is_input_error(response, no_real_e2b):
    with pytest.raises(TypeError):
        _score(response=response)
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize("label", [None, "{}", [], 42])
@pytest.mark.parametrize("response", [CODE, ""], ids=["with_source", "empty"])
def test_label_must_be_prepared_mapping(label, response, no_real_e2b):
    with pytest.raises(TypeError, match="label"):
        _score({"label": label}, response=response)
    no_real_e2b.assert_not_awaited()


def test_label_preserves_function_checker_and_timeouts_without_mutation():
    task = _task(
        cases=[{"input": [1], "output": 1}],
        input_mode="function_call",
        fn_name="answer",
        compile_timeout=3,
        run_timeout=4,
        checker_timeout=5,
        generated_checker="checker_source",
        custom_option={"exact": True},
    )
    original = copy.deepcopy(task)

    request = code_e2b._build_request(task, CODE)

    assert request == {**original["label"], "code": CODE}
    assert task == original


def test_template_inserts_python_literal_only_once():
    source = 'def answer():\n    return "x\\ny"'
    task = _task(code_template="exec(__MODEL_CODE__)\n# __MODEL_CODE__")
    original = copy.deepcopy(task)

    request = code_e2b._build_request(task, source)

    assert request["code"] == f"exec({source!r})\n# __MODEL_CODE__"
    assert "code_template" not in request
    assert task == original


@pytest.mark.parametrize("template", [42, "exec('missing placeholder')"])
@pytest.mark.parametrize("response", [CODE, ""], ids=["with_source", "empty"])
def test_invalid_template_fails_before_create(template, response, no_real_e2b):
    with pytest.raises(ValueError, match="code_template"):
        _score(_task(code_template=template), response=response)
    no_real_e2b.assert_not_awaited()


def _prelude_task(dataset, *, mode="function_call", fn_name="solve"):
    task = _task(input_mode=mode, fn_name=fn_name)
    task["metadata"] = (
        {"data_source": dataset}
        if dataset.startswith("livecodebench_")
        else {"source": "deepcoder", "provenance": {"dataset": dataset}}
    )
    return task


def _execute_prelude_request(request, *, inputs=(), stdin="", restricted_builtins=False):
    # A child process contains module registration and the LCB recursion limit;
    # these must never change pytest's own globals. Reproduce the native
    # dispatcher precedence to catch collisions such as itertools.count.
    if request["input_mode"] == "function_call":
        script = f"""import builtins, json, sys, types
module = types.ModuleType('candidate')
module.__dict__['_checker_private'] = 'not candidate state'
if {restricted_builtins!r}:
    module.__dict__['__builtins__'] = {{**vars(builtins), 'open': None}}
exec(compile({request['code']!r}, '<prepared>', 'exec'), module.__dict__)
entry = getattr(module, {request['fn_name']!r}, None)
entry_kind = 'function'
if entry is None:
    entry = getattr(module.Solution(), {request['fn_name']!r})
    entry_kind = 'method'
value = entry(*{inputs!r})
print(json.dumps({{'value': value, 'entry_kind': entry_kind,
                  'outer_counter': 'Counter' in module.__dict__,
                  'outer_bootstrap': '_code_e2b_prepare' in module.__dict__}}))
"""
    else:
        script = request["code"]
    return subprocess.run(
        [sys.executable, "-c", script],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize(
    "dataset,source,inputs,expected,entry_kind,fn_name",
    [
        (
            "primeintellect",
            "def solve(values: List[int]):\n"
            "    return [all(name in globals() for name in "
            "('List','Dict','Set','Tuple','Union','Optional')), "
            "'Counter' in globals(), 'np' in globals()]",
            ([1, 2],),
            [True, False, False],
            "function",
            "solve",
        ),
        (
            "taco",
            "def solve(values: List[int]):\n"
            "    return [int(np.sum(values)), time.__name__, 'pd' in globals()]",
            ([1, 2, 3],),
            [6, "time", False],
            "function",
            "solve",
        ),
        *[
            (
                dataset,
                "class Solution:\n"
                "    def count(self, values: List[int]):\n"
                "        return [sum(Counter(values).values()), "
                "'np' in globals(), sys.getrecursionlimit()]",
                ([1, 1, 2],),
                [3, False, 50000],
                "method",
                "count",
            )
            for dataset in ("lcbv5", "livecodebench_v5", "livecodebench_v6")
        ],
        (
            "lcbv5",
            "def count(values: List[int]):\n    return len(values)",
            ([1, 2, 3],),
            3,
            "function",
            "count",
        ),
        (
            "taco",
            "from math import gcd",
            (6, 9),
            3,
            "function",
            "gcd",
        ),
    ],
)
def test_python_prelude_executes_with_only_publisher_bindings(
    dataset, source, inputs, expected, entry_kind, fn_name
):
    request = code_e2b._build_request(_prelude_task(dataset, fn_name=fn_name), source)
    result = _execute_prelude_request(request, inputs=inputs)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["value"] == expected
    assert output["entry_kind"] == entry_kind
    assert output["outer_counter"] is False
    assert output["outer_bootstrap"] is False


@pytest.mark.parametrize(
    "dataset,source,stdin,expected",
    [
        ("primeintellect", "print('List' in globals())", "", "False\n"),
        ("taco", "print(int(np.sum(list(map(int, input().split())))))", "1 2 3\n", "6\n"),
        (
            "taco",
            "a, b = map(int, stdin.readline().split())\nstdout.write(str(a + b) + '\\n')",
            "2 3\n",
            "5\n",
        ),
        ("taco", "stdout.buffer.write(stdin.buffer.read())", "hello\n", "hello\n"),
        (
            "taco",
            "from __future__ import annotations\n"
            "assert stdin is sys.stdin and stdout is sys.stdout\n"
            "stdout.write(str(int(np.sum(list(map(int, stdin.read().split()))))))",
            "1 2 3\n",
            "6",
        ),
        (
            "taco",
            "from io import StringIO\n"
            "stdin = StringIO('candidate')\nstdout = StringIO()\n"
            "stdout.write(stdin.read())\nprint(stdout.getvalue())",
            "unused\n",
            "candidate\n",
        ),
        ("lcbv5", "print(Counter(input().split())['a'])", "a b a\n", "2\n"),
        ("livecodebench_v6", "print(Counter(input().split())['a'])", "a b a\n", "2\n"),
    ],
)
def test_python_prelude_stdio_matches_source_contract(dataset, source, stdin, expected):
    task = _prelude_task(dataset, mode="stdio")
    request = code_e2b._build_request(task, source)
    result = _execute_prelude_request(request, stdin=stdin)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_taco_stdio_aliases_bind_to_each_execution_streams():
    request = code_e2b._build_request(
        _prelude_task("taco", mode="stdio"),
        "assert stdin is sys.stdin and stdout is sys.stdout\nstdout.write(stdin.read())",
    )
    # A caller can replace streams between evaluations. Bind inside each
    # candidate execution, never to a driver stream or a previous test case.
    script = f"""import contextlib, io, sys
for text in ('first\\n', 'second\\n'):
    sys.stdin = io.StringIO(text)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec({request['code']!r}, {{'__name__': 'candidate'}})
    assert captured.getvalue() == text
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dataset", ["taco", "primeintellect"])
def test_python_prelude_does_not_add_stdio_aliases_to_call_based_contracts(dataset):
    source = "def solve():\n    return ['stdin' in globals(), 'stdout' in globals()]"
    request = code_e2b._build_request(_prelude_task(dataset), source)
    result = _execute_prelude_request(request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["value"] == [False, False]


@pytest.mark.parametrize(
    "dataset,source,missing",
    [
        ("primeintellect", "def solve():\n    return Counter([1])", "Counter"),
        ("taco", "def solve():\n    return pd.Series([1])", "pd"),
        ("lcbv5", "def solve():\n    return np.sum([1])", "np"),
    ],
)
def test_python_prelude_never_silently_supplies_other_publishers_libraries(
    dataset, source, missing
):
    request = code_e2b._build_request(_prelude_task(dataset), source)
    result = _execute_prelude_request(request)
    assert result.returncode != 0
    assert "NameError" in result.stderr
    assert f"name '{missing}' is not defined" in result.stderr


@pytest.mark.parametrize(
    "metadata,language,mode",
    [
        (None, "python", "stdio"),
        ([], "python", "function_call"),
        ("lcbv5", "python", "function_call"),
        ({"source": "deepcoder", "provenance": None}, "python", "function_call"),
        ({"source": "deepcoder", "provenance": {"dataset": "other"}}, "python", "stdio"),
        ({"source": "deepcoder", "provenance": {"dataset": "primeintellect"}}, "python", "stdio"),
        ({"source": "codeforces", "provenance": {"dataset": "lcbv5"}}, "python", "stdio"),
        ({"source": "codeforces"}, "python", "file"),
        ({"source": "codeforces"}, "cpp", "file"),
        ({"source": "deepcoder", "provenance": {"dataset": "taco"}}, "python", "file"),
        ({"source": "deepcoder", "provenance": {"dataset": "taco"}}, "cpp", "stdio"),
        ({"source": "leetcode"}, "python", "stdio"),
        ({"data_source": "livecodebench_v6"}, "cpp", "stdio"),
        ({"data_source": "livecodebench_v6"}, "python", "file"),
        ({"provenance": {"source_file": "/data/lcbv5.jsonl"}}, "python", "stdio"),
    ],
)
def test_python_prelude_leaves_unmatched_contracts_unchanged(metadata, language, mode):
    task = _task(language=language, input_mode=mode)
    task["metadata"] = metadata
    assert code_e2b._build_request(task, CODE)["code"] == CODE


@pytest.mark.parametrize("dataset", ["primeintellect", "taco", "lcbv5", "livecodebench_v6"])
@pytest.mark.parametrize("mode", ["stdio", "function_call"])
def test_python_prelude_preserves_task_headers_and_train_validation_parity(dataset, mode):
    task = _prelude_task(dataset, mode=mode)
    task["label"].update(compile_timeout=3, run_timeout=4, generated_checker="checker-source")
    original = copy.deepcopy(task)
    training = code_e2b._build_request({**task, "is_val": False}, CODE)
    validation = code_e2b._build_request({**task, "is_val": True}, CODE)
    assert training == validation
    assert {k: v for k, v in training.items() if k != "code"} == original["label"]
    assert task == original
    assert training == code_e2b._build_request(task, CODE)


@pytest.mark.parametrize("dataset", ["primeintellect", "taco", "lcbv5", "livecodebench_v6"])
@pytest.mark.parametrize("mode", ["stdio", "function_call"])
def test_python_prelude_explicit_template_stays_authoritative(dataset, mode):
    task = _prelude_task(dataset, mode=mode)
    task["label"]["code_template"] = "exec(__MODEL_CODE__)\n# __MODEL_CODE__"
    assert code_e2b._build_request(task, CODE)["code"] == f"exec({CODE!r})\n# __MODEL_CODE__"
    task["label"]["code_template"] = "missing marker"
    with pytest.raises(ValueError, match="code_template"):
        code_e2b._build_request(task, CODE)


def test_python_prelude_never_turns_empty_or_truncated_answers_into_scored_source(no_real_e2b):
    task = _prelude_task("lcbv5")
    assert _score(task, response="```python\n \n```")["reason"] == "no_code"
    truncated = asyncio.run(evaluate_reward(task, {"response": CODE, "finish_reason": "length"}))
    assert truncated["reason"] == "finish_reason_length"
    no_real_e2b.assert_not_awaited()


def test_python_prelude_preserves_registered_module_annotations_and_pickle():
    source = """from __future__ import annotations
from dataclasses import dataclass
from typing import get_type_hints
import importlib, pickle
@dataclass
class Payload:
    values: List[int]
def solve(values: List[int]):
    value = pickle.loads(pickle.dumps(Payload(values)))
    return [value.values, Payload.__annotations__['values'],
            get_type_hints(Payload)['values'] == List[int],
            importlib.import_module(__name__).solve is solve,
            '_checker_private' in globals()]
"""
    request = code_e2b._build_request(_prelude_task("primeintellect"), source)
    result = _execute_prelude_request(request, inputs=([1, 2],))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["value"] == [[1, 2], "List[int]", True, True, False]


def test_python_prelude_preserves_main_file_and_candidate_overrides():
    source = """from __future__ import annotations
import __main__
Counter = lambda value: {'answer': value}
if __name__ == '__main__':
    assert __main__.Counter is Counter
    print(Counter(input())['answer'])
"""
    request = code_e2b._build_request(_prelude_task("lcbv5", mode="stdio"), source)
    result = _execute_prelude_request(request, stdin="hello\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "hello\n"


def test_python_prelude_preserves_builtin_context_and_does_not_export_bootstrap():
    task = _prelude_task("primeintellect", fn_name="_code_e2b_prepare")
    request = code_e2b._build_request(task, "def _code_e2b_prepare():\n    return open is None")
    result = _execute_prelude_request(request, restricted_builtins=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["value"] is True


@pytest.mark.parametrize("class_body", ["pass", "def solve(self): return 999"])
def test_python_prelude_preserves_explicit_function_precedence_over_solution(class_body):
    source = f"class Solution:\n    {class_body}\ndef solve():\n    return 7"
    request = code_e2b._build_request(_prelude_task("lcbv5"), source)
    result = _execute_prelude_request(request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["value"] == 7
    assert json.loads(result.stdout)["entry_kind"] == "function"


def test_case_count_batches_preserve_every_case_and_serialized_bytes():
    cases = [{"stdin": str(index), "expected_stdout": str(index)} for index in range(205)]
    request = code_e2b._build_request(_task(cases), CODE)

    batches = list(code_e2b._iter_request_batches(request))

    assert [len(batch["cases"]) for batch, _, _ in batches] == [100, 100, 5]
    assert [case for batch, _, _ in batches for case in batch["cases"]] == cases
    for batch, payload, payload_bytes in batches:
        assert json.loads(payload) == batch
        assert len(payload.encode("utf-8")) == payload_bytes


def test_byte_batches_measure_utf8_and_allow_one_oversized_case(monkeypatch):
    monkeypatch.setattr(code_e2b, "_BATCH_BYTES", 100)
    cases = [{"stdin": "中" * 30, "expected_stdout": "x"}, {"stdin": "y", "expected_stdout": "z"}]
    request = code_e2b._build_request(_task(cases), CODE)

    batches = list(code_e2b._iter_request_batches(request))

    assert [batch["cases"] for batch, _, _ in batches] == [[cases[0]], [cases[1]]]
    assert batches[0][2] > code_e2b._BATCH_BYTES
    assert len(batches[0][1].encode("utf-8")) == batches[0][2]


def test_packing_releases_event_loop_and_uploads_utf8_bytes(monkeypatch):
    cases = [{"stdin": "中文\n", "expected_stdout": "中文\n"}, *CASES[1:]]
    task = _task(cases)
    sandbox = _sandbox(monkeypatch)
    original_batches = code_e2b._iter_request_batches
    expected_batch, expected_payload, expected_bytes = next(
        original_batches(code_e2b._build_request(task, CODE))
    )
    release_packing = threading.Event()

    async def run():
        loop = asyncio.get_running_loop()
        packing_started = asyncio.Event()

        def blocking_batches(request):
            loop.call_soon_threadsafe(packing_started.set)
            # A bounded wait also makes a synchronous implementation fail instead of hang.
            assert release_packing.wait(5), "Packing blocked the reward event loop"
            yield from original_batches(request)

        monkeypatch.setattr(code_e2b, "_iter_request_batches", blocking_batches)
        reward = asyncio.create_task(code_e2b.code_e2b_reward(task, {"response": CODE}))
        try:
            await asyncio.wait_for(packing_started.wait(), timeout=5)
            # This coroutine must resume while packing is still waiting in another thread.
            sandbox.create.assert_not_awaited()
            release_packing.set()
            return await reward
        finally:
            release_packing.set()
            if not reward.done():
                reward.cancel()
            await asyncio.gather(reward, return_exceptions=True)

    assert asyncio.run(run())["reward"] == 1.0
    payload = sandbox.files.write.await_args.args[1]
    assert isinstance(payload, bytes)
    assert payload == expected_payload.encode("utf-8")
    assert len(payload) == expected_bytes
    assert json.loads(payload) == expected_batch


@pytest.mark.parametrize("stage", ["transport", "parse"])
@pytest.mark.parametrize("cancel", [False, True], ids=["concurrent_upload", "cancellation"])
def test_slow_postprocessing_allows_other_uploads_and_cancellation(monkeypatch, stage, cancel):
    first = _sandbox(monkeypatch)
    second = _sandbox(monkeypatch)
    create = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    release = threading.Event()
    function_name = "set_e2b_tcp_user_timeout" if stage == "transport" else "_parse_result"
    original = getattr(code_e2b, function_name)

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        uploaded = asyncio.Event()
        blocked_once = False

        def slow_operation(*args):
            nonlocal blocked_once
            if not blocked_once:
                blocked_once = True
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5), "Postprocessing blocked the rollout event loop"
            return original(*args)

        async def other_upload(*_args, **_kwargs):
            uploaded.set()

        monkeypatch.setattr(code_e2b, function_name, slow_operation)
        second.files.write.side_effect = other_upload
        reward = asyncio.create_task(code_e2b.code_e2b_reward(_task(), {"response": CODE}))
        other_reward = None
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not reward.done()
            if cancel:
                reward.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(reward, timeout=1)
                # Cancellation must clean up without waiting for the blocking worker.
                assert not release.is_set()
                first.kill.assert_awaited_once()
                create.assert_awaited_once()
                if stage == "transport":
                    first.judge_run.assert_not_awaited()
            else:
                other_reward = asyncio.create_task(
                    code_e2b.code_e2b_reward(_task(), {"response": CODE})
                )
                await asyncio.wait_for(uploaded.wait(), timeout=1)
                assert not reward.done()
                release.set()
                assert (await reward)["reward"] == 1.0
                assert (await other_reward)["reward"] == 1.0
        finally:
            release.set()
            pending = [task for task in (reward, other_reward) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(run())
    first.kill.assert_awaited_once()


def test_next_batch_is_packed_only_after_previous_sandbox_finishes(monkeypatch):
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 1)
    sandbox = _sandbox(monkeypatch, _result((1.0,)))
    original_batches = code_e2b._iter_request_batches
    operations = []
    verdict = sandbox.judge_run.return_value

    def tracked_batches(request):
        for index, prepared in enumerate(original_batches(request)):
            assert sandbox.kill.await_count == index
            operations.append("pack")
            yield prepared

    async def judge(*_args, **_kwargs):
        operations.append("judge")
        return verdict

    async def cleanup(**_kwargs):
        operations.append("cleanup")

    monkeypatch.setattr(code_e2b, "_iter_request_batches", tracked_batches)
    sandbox.judge_run.side_effect = judge
    sandbox.kill.side_effect = cleanup

    score = _score()

    assert score["reward"] == 1.0
    assert operations == ["pack", "judge", "cleanup", "pack", "judge", "cleanup"]
    assert sandbox.create.await_count == 2


@pytest.mark.parametrize(
    ("unserializable_value", "error_type"),
    [(object(), TypeError), (float("nan"), ValueError)],
    ids=["unsupported_object", "non_finite_number"],
)
def test_packing_error_propagates_before_sandbox_creation(
    unserializable_value, error_type, no_real_e2b
):
    task = _task(cases=[{"stdin": unserializable_value}, *CASES])

    with pytest.raises(error_type):
        _score(task)

    no_real_e2b.assert_not_awaited()


def test_cancellation_during_packing_skips_sandbox_and_later_batches(monkeypatch, no_real_e2b):
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 1)
    original_batches = code_e2b._iter_request_batches
    release_packing = threading.Event()
    packed_batches = []

    async def run():
        loop = asyncio.get_running_loop()
        packing_started = asyncio.Event()

        def blocking_batches(request):
            for index, prepared in enumerate(original_batches(request)):
                packed_batches.append(index)
                if index == 0:
                    loop.call_soon_threadsafe(packing_started.set)
                    assert release_packing.wait(5), "Packing blocked reward cancellation"
                yield prepared

        monkeypatch.setattr(code_e2b, "_iter_request_batches", blocking_batches)
        reward = asyncio.create_task(code_e2b.code_e2b_reward(_task(), {"response": CODE}))
        try:
            await asyncio.wait_for(packing_started.wait(), timeout=5)
            reward.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reward
        finally:
            # Cancellation cannot interrupt an active worker; let its current batch finish.
            release_packing.set()
            await asyncio.gather(reward, return_exceptions=True)

    asyncio.run(run())

    assert packed_batches == [0]
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize("cases", [None, [], {}, "bad"])
@pytest.mark.parametrize("response", [CODE, ""], ids=["with_source", "empty"])
def test_invalid_case_collection_never_creates_sandbox(cases, response, no_real_e2b):
    task = _task()
    task["label"]["cases"] = cases
    with pytest.raises((TypeError, ValueError)):
        _score(task, response=response)
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize(
    "judge_request",
    [
        {"code": "x" * 500, "cases": [{}]},
        {"code": "x", "cases": [{"stdin": "x" * 500}]},
    ],
)
def test_hard_request_limit_rejects_source_or_indivisible_case(monkeypatch, judge_request):
    monkeypatch.setattr(code_e2b, "_MAX_REQUEST_BYTES", 200)
    with pytest.raises(ValueError):
        list(code_e2b._iter_request_batches(judge_request))


@pytest.mark.parametrize(
    "scores,expected_reward,expected_reason",
    [
        ((1.0, 1.0), 1.0, "accepted"),
        ((1.0, 0.0), 0.0, "failed_tests"),
        ((0.0, 0.0), 0.0, "failed_tests"),
        ((0.25, 0.75), 0.0, "failed_tests"),
    ],
)
def test_parser_returns_binary_reward_and_retains_diagnostic_scores(
    scores, expected_reward, expected_reason
):
    raw = _result(scores)
    parsed = code_e2b._parse_result(json.dumps(raw), len(scores))
    assert parsed["reward"] == expected_reward
    assert parsed["correct"] is (expected_reason == "accepted")
    assert parsed["reason"] == expected_reason
    assert parsed["sandbox_result"] == raw


@pytest.mark.parametrize(
    "case_scores,expected_reward",
    [
        ((1.0, 1.0), 1.0),
        ((1.0, 0.0), 0.0),
        ((0.0, 0.0), 0.0),
        ((0.25, 0.75), 0.0),
    ],
)
def test_single_batch_reward_requires_every_case_to_pass(monkeypatch, case_scores, expected_reward):
    raw = _result(case_scores)
    sandbox = _sandbox(monkeypatch, raw)

    verdict = _score()

    assert verdict["reward"] == expected_reward
    assert verdict["correct"] is bool(expected_reward)
    assert verdict["reason"] == ("accepted" if expected_reward else "failed_tests")
    assert verdict["sandbox_result"] == raw
    assert verdict["metrics"]["total_cases"] == len(case_scores)
    sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize(
    "result",
    [
        [],
        {"error": "judge unavailable"},
        {"accepted": 1, "score": 1, "tests": [{"passed": True}]},
        {"accepted": True, "score": True, "tests": [{"passed": True}]},
        {"accepted": False, "score": -0.1, "tests": [{"passed": False}]},
        {"accepted": False, "score": 1.1, "tests": [{"passed": False}]},
        {"accepted": False, "score": float("nan"), "tests": [{"passed": False}]},
        {"accepted": False, "score": float("inf"), "tests": [{"passed": False}]},
        {"accepted": True, "score": 1, "tests": []},
        {"accepted": True, "score": 1, "tests": [{"passed": True}, {"passed": True}]},
        {"accepted": False, "score": 0, "tests": ["invalid"]},
        {"accepted": True, "score": 1, "tests": [{"passed": 1}]},
        {"accepted": False, "score": 0, "tests": [{"passed": False, "score": True}]},
        {"accepted": True, "score": 1, "tests": [{"passed": False}]},
        {"accepted": False, "score": 1, "tests": [{"passed": True}]},
        {"accepted": True, "score": 0.5, "tests": [{"passed": True, "score": 0.5}]},
        {"accepted": False, "score": 0.5, "tests": [{"passed": False, "score": 0.25}]},
    ],
)
def test_protocol_rejects_missing_inconsistent_or_invalid_results(result):
    with pytest.raises(ValueError):
        code_e2b._parse_result(json.dumps(result), 1)


@pytest.mark.parametrize("stdout", ["", "debug\n{}", "{}\nextra", "not json"])
def test_protocol_rejects_non_json_output(stdout):
    with pytest.raises(ValueError):
        code_e2b._parse_result(stdout, 1)


def test_hard_result_limit_counts_utf8_bytes(monkeypatch):
    monkeypatch.setattr(code_e2b, "_MAX_RESULT_BYTES", 30)
    with pytest.raises(ValueError):
        code_e2b._parse_result(json.dumps({"message": "中" * 10}, ensure_ascii=False), 1)


@pytest.mark.parametrize(
    "case_scores,expected_reward",
    [
        ([1.0] * 101, 1.0),
        ([0.0] + [1.0] * 100, 0.0),
        ([1.0] * 100 + [0.0], 0.0),
        ([0.25] + [1.0] * 100, 0.0),
    ],
    ids=["all_pass", "first_batch_failure", "last_batch_failure", "partial_case_score"],
)
def test_sequential_batches_require_all_pass_and_weight_only_diagnostic_scores(
    monkeypatch, case_scores, expected_reward
):
    cases = [{"stdin": str(index), "expected_stdout": str(index)} for index in range(101)]
    first = _sandbox(monkeypatch, _result(case_scores[:100]))
    second = _sandbox(monkeypatch, _result(case_scores[100:]))
    events = []

    async def create(**_kwargs):
        sandbox = first if not events else second
        if sandbox is second:
            first.kill.assert_awaited_once()
        events.append("create")
        return sandbox

    create_mock = AsyncMock(side_effect=create)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create_mock)

    score = _score(_task(cases))

    assert score["reward"] == expected_reward
    assert score["correct"] is bool(expected_reward)
    assert score["reason"] == ("accepted" if expected_reward else "failed_tests")
    assert score["metrics"] == {
        "passed_cases": float(sum(case_score == 1 for case_score in case_scores)),
        "total_cases": 101.0,
        "completed_batches": 2.0,
    }
    assert score["sandbox_result"]["score"] == pytest.approx(sum(case_scores) / 101)
    assert score["sandbox_result"]["tests"] == _result(case_scores)["tests"]
    assert create_mock.await_count == 2
    for sandbox, expected_cases in ((first, cases[:100]), (second, cases[100:])):
        assert json.loads(sandbox.files.write.await_args.args[1])["cases"] == expected_cases
        sandbox.kill.assert_awaited_once()


def test_e2b_lifecycle_uploads_only_json_and_runs_fixed_judge(monkeypatch):
    sandbox = _sandbox(monkeypatch)
    source = "print('$(touch /tmp/unsafe); `echo unsafe`')"
    monkeypatch.setenv("E2B_DOMAIN", "e2b.example.test")
    monkeypatch.setenv("E2B_TEMPLATE", "custom-evaluator")
    monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "300")
    monkeypatch.setenv("E2B_REQUEST_TIMEOUT", "25")

    assert _score(response=source)["correct"] is True

    sandbox.create.assert_awaited_once_with(
        template="custom-evaluator",
        api_key="test-api-key",
        timeout=1200,
        request_timeout=25.0,
        allow_internet_access=False,
        domain="e2b.example.test",
    )
    write = sandbox.files.write.await_args
    assert write.args[0] == "/tmp/code-verifier-request.json"
    assert json.loads(write.args[1])["code"] == source
    assert write.kwargs == {"user": "root", "request_timeout": 300.0}
    run = sandbox.commands.run.await_args
    assert run.args == (
        "/root/miniconda3/bin/python /opt/e2b-sandboxfusion/judge.py --stdio"
        " < /tmp/code-verifier-request.json",
    )
    assert run.kwargs["user"] == "root"
    assert run.kwargs["timeout"] == 300.0
    assert run.kwargs["request_timeout"] == 0
    warmup = sandbox.commands.run.await_args_list[0]
    assert warmup.args == ("true",)
    assert warmup.kwargs["timeout"] == 25
    assert warmup.kwargs["request_timeout"] == 25
    assert warmup.kwargs["background"] is True
    sandbox.warmup_handle.wait.assert_awaited_once()
    sandbox.warmup_handle.disconnect.assert_awaited_once()
    assert callable(run.kwargs["on_stdout"])
    assert callable(run.kwargs["on_stderr"])
    sandbox.kill.assert_awaited_once_with(request_timeout=10)


def test_command_warmup_precedes_socket_tuning_and_judge(monkeypatch, mock_tcp_user_timeout):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setenv("E2B_REQUEST_TIMEOUT", "4")
    operations = []
    verdict = sandbox.judge_run.return_value
    report = mock_tcp_user_timeout.return_value

    async def warmup(*_args, **_kwargs):
        operations.append("warmup")
        return CommandResult(stderr="", stdout="", exit_code=0, error=None)

    def tune(actual_sandbox):
        assert actual_sandbox is sandbox
        sandbox.judge_run.assert_not_awaited()
        operations.append("transport")
        return report

    async def judge(*_args, **_kwargs):
        mock_tcp_user_timeout.assert_called_once_with(sandbox)
        operations.append("judge")
        return verdict

    sandbox.warmup_run.side_effect = warmup
    mock_tcp_user_timeout.side_effect = tune
    sandbox.judge_run.side_effect = judge

    assert _score()["correct"] is True

    assert operations == ["warmup", "transport", "judge"]
    sandbox.warmup_run.assert_awaited_once_with(
        "true", background=True, user="root", timeout=4.0, request_timeout=4.0
    )
    assert sandbox.judge_run.await_args.kwargs["request_timeout"] == 0


@pytest.mark.parametrize("failures", [1, 2, 4])
@pytest.mark.parametrize("case_scores", [(1.0, 1.0), (1.0, 0.5)], ids=["pass", "partial"])
@pytest.mark.parametrize(
    "upload_setting,expected_timeouts",
    [(None, [300, 450, 600, 750, 900]), ("600", [600, 900, 1200, 1500, 1800])],
)
def test_upload_retry_reuses_request_and_preserves_real_verdict(
    monkeypatch, caplog, warmup_backoff, failures, case_scores, upload_setting, expected_timeouts
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    if upload_setting is not None:
        monkeypatch.setenv("E2B_UPLOAD_REQUEST_TIMEOUT", upload_setting)
    sandboxes = [_sandbox(monkeypatch) for _ in range(failures)]
    sandboxes.append(_sandbox(monkeypatch, _result(case_scores)))
    for sandbox in sandboxes[:-1]:
        sandbox.files.write.side_effect = httpx.ReadTimeout("timed out test-api-key e2b_secret")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)
    task = _task(
        code_template="model = __MODEL_CODE__",
        generated_checker="private-checker",
        compile_timeout=17,
        run_timeout=4,
    )
    task.update(task_id=123, group_id="group", sample_id="sample")
    original_task = copy.deepcopy(task)

    async def backoff(delay):
        assert delay == 1.5
        # Cleanup must complete before waiting, not merely before the next create.
        sandboxes[create_mock.await_count - 1].kill.assert_awaited_once()

    warmup_backoff.sleep.side_effect = backoff

    score = _score(task)

    assert score["reward"] == float(all(value == 1 for value in case_scores))
    assert score["reason"] == ("accepted" if score["correct"] else "failed_tests")
    assert score["sandbox_result"] == _result(case_scores)
    assert score["metrics"] == {
        "passed_cases": float(sum(value == 1 for value in case_scores)),
        "total_cases": 2.0,
        "completed_batches": 1.0,
        "upload_failure_count": float(failures),
        "upload_retry_count": float(failures),
    }
    assert "timed_out" not in score
    assert task == original_task
    assert create_mock.await_count == failures + 1
    payloads = [sandbox.files.write.await_args.args[1] for sandbox in sandboxes]
    assert all(payload == payloads[0] for payload in payloads)
    for sandbox, upload_timeout in zip(sandboxes, expected_timeouts):
        sandbox.files.write.assert_awaited_once_with(
            code_e2b._REQUEST_PATH, payloads[0], user="root", request_timeout=upload_timeout
        )
        sandbox.kill.assert_awaited_once()
    for sandbox in sandboxes[:-1]:
        sandbox.commands.run.assert_not_awaited()
        sandbox.warmup_handle.disconnect.assert_not_awaited()
    sandboxes[-1].judge_run.assert_awaited_once()
    assert warmup_backoff.delay.call_count == failures
    assert (
        warmup_backoff.delay.call_args_list
        == [call(5.0, 10.0), call(10.0, 20.0), call(20.0, 40.0), call(30.0, 60.0)][:failures]
    )
    assert warmup_backoff.sleep.await_count == failures
    retry_events = [
        record.code_e2b_retry for record in caplog.records if hasattr(record, "code_e2b_retry")
    ]
    assert [event["event"] for event in retry_events] == ["upload_retry_start"] * failures + [
        "upload_recovered"
    ]
    failed = _batch_events(caplog, "failure")
    assert [event["upload_timeout_s"] for event in failed] == expected_timeouts[:failures]
    assert all(event["stage"] == "upload" and event["retryable_upload"] for event in failed)
    assert all(event["run_elapsed_s"] is None for event in failed)
    assert not any(event["retryable_timeout"] or event["retryable_warmup"] for event in failed)
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert "test-api-key" not in caplog.text
    assert "e2b_secret" not in caplog.text
    assert "private-checker" not in caplog.text


@pytest.mark.parametrize(
    "error_factory",
    [
        TimeoutError,
        e2b.TimeoutException,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        ConnectionResetError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    ],
)
def test_five_upload_failures_raise_instead_of_zero_scoring(
    monkeypatch, caplog, warmup_backoff, error_factory
):
    sandboxes = [_sandbox(monkeypatch) for _ in range(5)]
    for sandbox in sandboxes:
        sandbox.files.write.side_effect = error_factory("upload failure")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "upload"
    assert error.value.retryable_upload is True
    assert error.value.retryable_warmup is False
    assert error.value.is_timeout is False
    assert create_mock.await_count == 5
    assert warmup_backoff.sleep.await_count == 4
    for sandbox in sandboxes:
        sandbox.commands.run.assert_not_awaited()
        sandbox.kill.assert_awaited_once()
    exhausted = [
        record
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] == "upload_exhausted"
    ]
    assert len(exhausted) == 1
    assert exhausted[0].levelno == logging.ERROR
    assert exhausted[0].code_e2b_retry["upload_failure_count"] == 5
    assert exhausted[0].code_e2b_retry["upload_retry_count"] == 4


def test_local_upload_deadline_retries_when_sdk_ignores_request_timeout(
    monkeypatch, caplog, warmup_backoff
):
    monkeypatch.setenv("E2B_UPLOAD_REQUEST_TIMEOUT", "0.01")
    first, retry = _sandbox(monkeypatch), _sandbox(monkeypatch)
    create = _use_sandbox_sequence(monkeypatch, [first, retry])
    cancelled = []

    async def hanging_upload(*_args, **kwargs):
        assert kwargs["request_timeout"] == 0.01
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    first.files.write.side_effect = hanging_upload
    score = _score(deadline=2)

    assert score["reward"] == 1.0
    assert score["metrics"]["upload_failure_count"] == 1
    assert score["metrics"]["upload_retry_count"] == 1
    assert cancelled == [True]
    assert create.await_count == 2
    first.kill.assert_awaited_once()
    first.commands.run.assert_not_awaited()
    retry.judge_run.assert_awaited_once()
    failure = _batch_events(caplog, "failure")[0]
    assert failure["stage"] == "upload"
    assert failure["error_type"] == "TimeoutError"
    assert failure["upload_timeout_s"] == 0.01
    assert failure["retryable_upload"] is True
    assert failure["attempt_deadline_expired"] is False
    assert failure["retryable_timeout"] is False
    assert retry.files.write.await_args.kwargs["request_timeout"] == 0.015


@pytest.mark.parametrize("stage", ["upload", "backoff"])
def test_caller_cancellation_during_upload_recovery_is_not_retried(
    monkeypatch, warmup_backoff, stage
):
    sandbox = _sandbox(monkeypatch)

    async def run():
        started = asyncio.Event()

        async def wait_for_cancellation(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        if stage == "upload":
            sandbox.files.write.side_effect = wait_for_cancellation
        else:
            sandbox.files.write.side_effect = httpx.ReadTimeout("upload timeout")
            warmup_backoff.sleep.side_effect = wait_for_cancellation
        task = asyncio.create_task(code_e2b.code_e2b_reward(_task(), {"response": CODE}))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    asyncio.run(run())

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.commands.run.assert_not_awaited()
    assert warmup_backoff.sleep.await_count == int(stage == "backoff")


@pytest.mark.parametrize("stage", ["upload", "backoff"])
def test_worker_stop_during_upload_recovery_prevents_next_sandbox(
    monkeypatch, warmup_backoff, stage
):
    sandbox = _sandbox(monkeypatch)
    stop = Event()

    async def upload(*_args, **_kwargs):
        if stage == "upload":
            stop.set()
        raise httpx.ReadTimeout("upload timeout")

    async def backoff(_delay):
        stop.set()

    sandbox.files.write.side_effect = upload
    warmup_backoff.sleep.side_effect = backoff
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.commands.run.assert_not_awaited()
    assert warmup_backoff.sleep.await_count == int(stage == "backoff")


@pytest.mark.parametrize(
    "error_factory",
    [
        e2b.AuthenticationException,
        e2b.InvalidArgumentException,
        e2b.NotEnoughSpaceException,
        e2b.FileNotFoundException,
        e2b.TemplateException,
        e2b.SandboxException,
        httpx.LocalProtocolError,
        httpx.UnsupportedProtocol,
        RuntimeError,
        ValueError,
        TypeError,
    ],
)
def test_nontransient_upload_failures_do_not_retry(monkeypatch, warmup_backoff, error_factory):
    sandbox = _sandbox(monkeypatch)
    sandbox.files.write.side_effect = error_factory("nontransient upload failure")

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "upload"
    assert error.value.retryable_upload is False
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.commands.run.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()


def test_upload_certificate_error_is_not_retried(monkeypatch, warmup_backoff):
    sandbox = _sandbox(monkeypatch)
    error = httpx.ConnectError("certificate verification failed")
    error.__cause__ = ssl.SSLCertVerificationError("untrusted certificate")
    sandbox.files.write.side_effect = error

    with pytest.raises(code_e2b.CodeE2BError) as raised:
        _score()

    assert raised.value.retryable_upload is False
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("stage", ["create", "upload", "warmup_start", "warmup_end", "run"])
def test_whole_batch_deadline_never_retries_or_zero_scores(
    monkeypatch, caplog, warmup_backoff, stage
):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setattr(
        code_e2b, "_timeouts", lambda _batch, **_kwargs: (60.0, 60.0, 300.0, 0.01, 60)
    )

    async def hanging_operation(*_args, **_kwargs):
        await asyncio.Event().wait()

    operation = {
        "create": sandbox.create,
        "upload": sandbox.files.write,
        "warmup_start": sandbox.warmup_run,
        "warmup_end": sandbox.warmup_handle.wait,
        "run": sandbox.judge_run,
    }[stage]
    operation.side_effect = hanging_operation

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score(deadline=2)

    assert error.value.stage == ("warmup" if stage.startswith("warmup") else stage)
    assert error.value.error_type == "TimeoutError"
    assert error.value.attempt_deadline_expired is True
    assert error.value.retryable_create is False
    assert error.value.retryable_upload is False
    assert error.value.retryable_warmup is False
    assert error.value.is_timeout is False
    assert "attempt deadline expired" in str(error.value)
    sandbox.create.assert_awaited_once()
    if stage == "create":
        sandbox.kill.assert_not_awaited()
    else:
        sandbox.kill.assert_awaited_once()
    if stage in ("create", "upload"):
        sandbox.commands.run.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()
    assert not any(hasattr(record, "code_e2b_retry") for record in caplog.records)
    failure = _batch_events(caplog, "failure")[0]
    assert failure["attempt_deadline_expired"] is True
    assert not any(
        failure[key]
        for key in ("retryable_create", "retryable_upload", "retryable_warmup", "retryable_timeout")
    )


@pytest.mark.parametrize("stage", ["create", "upload", "warmup", "run"])
def test_outer_retry_prioritizes_expired_deadline_even_with_retryable_flags(
    monkeypatch, warmup_backoff, stage
):
    error = code_e2b.CodeE2BError(
        stage,
        TimeoutError("whole attempt expired"),
        attempt_deadline_expired=True,
        retryable_create=True,
        retryable_upload=True,
        retryable_warmup=True,
        is_timeout=True,
    )
    run_batch = AsyncMock(side_effect=error)
    monkeypatch.setattr(code_e2b, "_run_batch", run_batch)

    with pytest.raises(code_e2b.CodeE2BError) as raised:
        _score()

    assert raised.value is error
    run_batch.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize(
    "events",
    [
        ["upload", "warmup", "upload", "run", "upload", "warmup", "upload", "pass"],
        ["run", "upload", "warmup", "upload", "upload", "warmup", "upload", "run"],
        ["upload", "warmup", "upload", "run", "upload", "warmup", "upload", "upload"],
        ["warmup", "upload", "upload", "run", "warmup", "upload", "upload", "warmup"],
    ],
)
def test_upload_warmup_and_run_budgets_are_independent_and_never_reset(
    monkeypatch, warmup_backoff, events
):
    if events[-1] == "run":
        events = [*events, "run"]
    sandboxes = [_sandbox(monkeypatch) for _ in events]
    for sandbox, event in zip(sandboxes, events):
        if event == "upload":
            sandbox.files.write.side_effect = httpx.ReadTimeout("upload timeout")
        elif event == "warmup":
            sandbox.warmup_run.side_effect = e2b.TimeoutException("warmup timeout")
        elif event == "run":
            sandbox.judge_run.side_effect = e2b.TimeoutException("judge timeout")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    if events[-1] in ("upload", "warmup"):
        with pytest.raises(code_e2b.CodeE2BError) as error:
            _score()
        assert error.value.stage == events[-1]
    else:
        score = _score()
        assert score["reward"] == float(events[-1] == "pass")
        assert score["reason"] == ("accepted" if events[-1] == "pass" else "timeout_after_retry")
        assert score["metrics"]["upload_failure_count"] == 4.0
        assert score["metrics"]["upload_retry_count"] == 4.0
        assert score["metrics"]["warmup_failure_count"] == 2.0
        assert score["metrics"]["warmup_retry_count"] == 2.0
        assert score["metrics"]["timeout_count"] == events.count("run")
        assert score["metrics"]["retry_count"] == min(events.count("run"), 2)
    assert create_mock.await_count == len(events)
    assert warmup_backoff.sleep.await_count == 6
    payloads = [sandbox.files.write.await_args.args[1] for sandbox in sandboxes]
    assert all(payload == payloads[0] for payload in payloads)
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()


def test_upload_retries_only_current_batch_with_fresh_per_batch_budget(monkeypatch, warmup_backoff):
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 1)
    cases = [{"stdin": str(index), "expected_stdout": str(index)} for index in range(3)]
    events = ["pass", "upload", "upload", "pass", "upload", "upload", "pass"]
    sandboxes = [_sandbox(monkeypatch, _result((1.0,))) for _ in events]
    for sandbox, event in zip(sandboxes, events):
        if event == "upload":
            sandbox.files.write.side_effect = httpx.ReadTimeout("upload timeout")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    score = _score(_task(cases))

    assert score["reward"] == 1.0
    assert score["metrics"] == {
        "passed_cases": 3.0,
        "total_cases": 3.0,
        "completed_batches": 3.0,
        "upload_failure_count": 4.0,
        "upload_retry_count": 4.0,
    }
    uploaded = [
        json.loads(sandbox.files.write.await_args.args[1])["cases"] for sandbox in sandboxes
    ]
    assert uploaded == [cases[:1]] + [cases[1:2]] * 3 + [cases[2:]] * 3
    assert [sandbox.files.write.await_args.kwargs["request_timeout"] for sandbox in sandboxes] == [
        300,
        300,
        450,
        600,
        300,
        450,
        600,
    ]
    assert create_mock.await_count == 7
    assert sum(sandbox.judge_run.await_count for sandbox in sandboxes) == 3


def test_upload_budget_is_local_to_each_answer(monkeypatch, warmup_backoff):
    events = ["upload", "upload", "pass"] * 2
    sandboxes = [_sandbox(monkeypatch) for _ in events]
    for sandbox, event in zip(sandboxes, events):
        if event == "upload":
            sandbox.files.write.side_effect = httpx.ReadTimeout("upload timeout")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    for _ in range(2):
        score = _score()
        assert score["reward"] == 1.0
        assert score["metrics"]["upload_retry_count"] == 2.0
    assert create_mock.await_count == 6
    assert [sandbox.files.write.await_args.kwargs["request_timeout"] for sandbox in sandboxes] == [
        300,
        450,
        600,
        300,
        450,
        600,
    ]


@pytest.mark.parametrize("stage", ["upload", "backoff", "cleanup"])
def test_upload_cancellation_does_not_create_another_sandbox(monkeypatch, warmup_backoff, stage):
    sandbox = _sandbox(monkeypatch)
    sandbox.files.write.side_effect = (
        asyncio.CancelledError() if stage == "upload" else httpx.ReadTimeout("upload timeout")
    )
    if stage == "backoff":
        warmup_backoff.sleep.side_effect = asyncio.CancelledError()
    elif stage == "cleanup":
        sandbox.kill.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.commands.run.assert_not_awaited()
    assert warmup_backoff.sleep.await_count == int(stage == "backoff")


def test_cleanup_error_during_upload_recovery_does_not_replace_verdict(
    monkeypatch, caplog, warmup_backoff
):
    first, retry = _sandbox(monkeypatch), _sandbox(monkeypatch)
    first.files.write.side_effect = httpx.ReadTimeout("upload timeout")
    first.kill.side_effect = RuntimeError("cleanup failed test-api-key e2b_secret")
    create_mock = _use_sandbox_sequence(monkeypatch, [first, retry])

    score = _score()

    assert score["reward"] == 1.0
    assert score["metrics"]["upload_retry_count"] == 1.0
    assert create_mock.await_count == 2
    assert _batch_events(caplog, "cleanup_failure")[0]["operation"] == "kill"
    assert "test-api-key" not in caplog.text
    assert "e2b_secret" not in caplog.text


def test_nontransient_failure_after_upload_retry_propagates(monkeypatch, warmup_backoff):
    first, retry = _sandbox(monkeypatch), _sandbox(monkeypatch)
    first.files.write.side_effect = httpx.ReadTimeout("upload timeout")
    retry.files.write.side_effect = e2b.AuthenticationException("invalid credentials")
    create_mock = _use_sandbox_sequence(monkeypatch, [first, retry])

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "upload"
    assert error.value.retryable_upload is False
    assert create_mock.await_count == 2
    assert warmup_backoff.sleep.await_count == 1
    for sandbox in (first, retry):
        sandbox.kill.assert_awaited_once()
        sandbox.commands.run.assert_not_awaited()


@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("case_scores", [(1.0, 1.0), (1.0, 0.0)], ids=["pass", "fail"])
@pytest.mark.parametrize("failure_phase", ["start", "end"])
def test_warmup_retry_reuses_request_and_preserves_real_verdict(
    monkeypatch, caplog, warmup_backoff, failures, case_scores, failure_phase
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandboxes = [_sandbox(monkeypatch) for _ in range(failures)]
    sandboxes.append(_sandbox(monkeypatch, _result(case_scores)))
    for sandbox in sandboxes[:-1]:
        operation = sandbox.warmup_run if failure_phase == "start" else sandbox.warmup_handle.wait
        operation.side_effect = e2b.TimeoutException("warmup timed out test-api-key e2b_secret")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)
    task = _task(
        code_template="model = __MODEL_CODE__",
        generated_checker="private-checker",
        compile_timeout=17,
        run_timeout=4,
    )
    task.update(task_id=123, group_id="group", sample_id="sample")
    original_task = copy.deepcopy(task)

    score = _score(task)

    assert score["reward"] == float(all(value == 1 for value in case_scores))
    assert score["reason"] == ("accepted" if score["correct"] else "failed_tests")
    assert score["sandbox_result"] == _result(case_scores)
    assert score["metrics"] == {
        "passed_cases": float(sum(value == 1 for value in case_scores)),
        "total_cases": 2.0,
        "completed_batches": 1.0,
        "warmup_failure_count": float(failures),
        "warmup_retry_count": float(failures),
    }
    assert "timed_out" not in score
    assert task == original_task
    assert create_mock.await_count == failures + 1
    payloads = [sandbox.files.write.await_args.args[1] for sandbox in sandboxes]
    assert all(payload == payloads[0] for payload in payloads)
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()
        sandbox.warmup_run.assert_awaited_once()
    for sandbox in sandboxes[:-1]:
        sandbox.judge_run.assert_not_awaited()
    sandboxes[-1].judge_run.assert_awaited_once()
    assert warmup_backoff.delay.call_count == failures
    warmup_backoff.delay.assert_called_with(1.0, 3.0)
    assert warmup_backoff.sleep.await_count == failures
    warmup_backoff.sleep.assert_awaited_with(1.5)
    retry_events = [
        record.code_e2b_retry for record in caplog.records if hasattr(record, "code_e2b_retry")
    ]
    assert [event["event"] for event in retry_events] == ["warmup_retry_start"] * failures + [
        "warmup_recovered"
    ]
    failed = _batch_events(caplog, "failure")
    assert [event["warmup_phase"] for event in failed] == [
        "awaiting_start" if failure_phase == "start" else "awaiting_end"
    ] * failures
    assert all(event["run_elapsed_s"] is None for event in failed)
    assert all(event["retryable_warmup"] for event in failed)
    assert not any(event["retryable_timeout"] for event in failed)
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert "test-api-key" not in caplog.text
    assert "e2b_secret" not in caplog.text
    assert "private-checker" not in caplog.text


@pytest.mark.parametrize(
    "error_factory",
    [
        TimeoutError,
        e2b.TimeoutException,
        ConnectionError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
    ],
)
def test_three_warmup_failures_raise_instead_of_zero_scoring(
    monkeypatch, caplog, warmup_backoff, error_factory
):
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    for sandbox in sandboxes:
        sandbox.warmup_run.side_effect = error_factory("warmup failure")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "warmup"
    assert error.value.retryable_warmup is True
    assert error.value.is_timeout is False
    assert create_mock.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    for sandbox in sandboxes:
        sandbox.judge_run.assert_not_awaited()
        sandbox.kill.assert_awaited_once()
    exhausted = [
        record
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] == "warmup_exhausted"
    ]
    assert len(exhausted) == 1
    assert exhausted[0].levelno == logging.ERROR
    assert exhausted[0].code_e2b_retry["warmup_failure_count"] == 3
    assert exhausted[0].code_e2b_retry["warmup_retry_count"] == 2


@pytest.mark.parametrize("stage", ["start", "end"])
@pytest.mark.parametrize(
    "error_factory", [e2b.AuthenticationException, e2b.InvalidArgumentException, RuntimeError]
)
def test_nontransient_warmup_failures_do_not_retry(
    monkeypatch, warmup_backoff, stage, error_factory
):
    sandbox = _sandbox(monkeypatch)
    operation = sandbox.warmup_run if stage == "start" else sandbox.warmup_handle.wait
    operation.side_effect = error_factory("nontransient failure")

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "warmup"
    assert error.value.retryable_warmup is False
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.judge_run.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()


def test_warmup_transport_classifier_handles_current_sdk_stream_errors():
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from pyqwest import ReadError, StreamError, WriteError

    for cause in (
        ConnectionResetError("reset"),
        ReadError("read"),
        StreamError("reset", 8),
        WriteError("write"),
    ):
        assert code_e2b._is_warmup_transport_error(cause)
        wrapped = ConnectError(Code.UNKNOWN, "wrapped stream error")
        wrapped.__cause__ = cause
        assert code_e2b._is_warmup_transport_error(wrapped)
    wrapped.__cause__ = ValueError("invalid response body")
    assert not code_e2b._is_warmup_transport_error(wrapped)
    wrapped.__cause__ = asyncio.CancelledError()
    assert not code_e2b._is_warmup_transport_error(wrapped)
    for code in (Code.UNAUTHENTICATED, Code.PERMISSION_DENIED, Code.INVALID_ARGUMENT):
        wrapped = ConnectError(code, "not a transient failure")
        wrapped.__cause__ = ConnectionError("must not override explicit error status")
        assert not code_e2b._is_warmup_transport_error(wrapped)


@pytest.mark.parametrize(
    "events",
    [
        ["warmup", "run", "warmup", "pass"],
        ["run", "warmup", "warmup", "run"],
        ["warmup", "run", "warmup", "warmup"],
    ],
)
def test_warmup_and_run_budgets_are_independent_and_never_reset(
    monkeypatch, caplog, warmup_backoff, events
):
    if events[-1] == "run":
        events = [*events, "run"]
    sandboxes = [_sandbox(monkeypatch) for _ in events]
    for sandbox, event in zip(sandboxes, events):
        if event == "warmup":
            sandbox.warmup_run.side_effect = e2b.TimeoutException("warmup timeout")
        elif event == "run":
            sandbox.judge_run.side_effect = e2b.TimeoutException("judge timeout")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    if events[-1] == "warmup":
        with pytest.raises(code_e2b.CodeE2BError) as error:
            _score()
        assert error.value.stage == "warmup"
    else:
        score = _score()
        assert score["reward"] == float(events[-1] == "pass")
        assert score["reason"] == ("accepted" if events[-1] == "pass" else "timeout_after_retry")
        assert score["metrics"]["timeout_count"] == events.count("run")
        assert score["metrics"]["retry_count"] == min(events.count("run"), 2)
        assert score["metrics"]["warmup_failure_count"] == 2.0
        assert score["metrics"]["warmup_retry_count"] == 2.0
    assert create_mock.await_count == len(events)
    assert warmup_backoff.sleep.await_count == 2
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()
    timeout_events = [
        record.code_e2b_retry
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] in ("retry_start", "timeout_exhausted")
    ]
    assert [event["timeout_count"] for event in timeout_events] == list(
        range(1, events.count("run") + 1)
    )


def test_warmup_retries_only_current_batch_with_fresh_per_batch_budget(monkeypatch, warmup_backoff):
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 1)
    cases = [{"stdin": str(index), "expected_stdout": str(index)} for index in range(3)]
    events = ["pass", "warmup", "warmup", "pass", "warmup", "warmup", "pass"]
    sandboxes = [_sandbox(monkeypatch, _result((1.0,))) for _ in events]
    for sandbox, event in zip(sandboxes, events):
        if event == "warmup":
            sandbox.warmup_run.side_effect = TimeoutError("warmup timeout")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    score = _score(_task(cases))

    assert score["reward"] == 1.0
    assert score["metrics"] == {
        "passed_cases": 3.0,
        "total_cases": 3.0,
        "completed_batches": 3.0,
        "warmup_failure_count": 4.0,
        "warmup_retry_count": 4.0,
    }
    uploaded = [
        json.loads(sandbox.files.write.await_args.args[1])["cases"] for sandbox in sandboxes
    ]
    assert uploaded == [
        cases[:1],
        cases[1:2],
        cases[1:2],
        cases[1:2],
        cases[2:],
        cases[2:],
        cases[2:],
    ]
    assert create_mock.await_count == 7


def test_warmup_budget_is_local_to_each_answer(monkeypatch, warmup_backoff):
    events = ["warmup", "warmup", "pass"] * 2
    sandboxes = [_sandbox(monkeypatch) for _ in events]
    for sandbox, event in zip(sandboxes, events):
        if event == "warmup":
            sandbox.warmup_run.side_effect = TimeoutError("warmup timeout")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)
    task = _task()

    for _ in range(2):
        score = _score(task)
        assert score["reward"] == 1.0
        assert score["metrics"]["warmup_retry_count"] == 2.0
    assert create_mock.await_count == 6


def test_cancellation_during_warmup_backoff_does_not_create_another_sandbox(
    monkeypatch, warmup_backoff
):
    sandbox = _sandbox(monkeypatch)
    sandbox.warmup_run.side_effect = TimeoutError("warmup timeout")
    warmup_backoff.sleep.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.judge_run.assert_not_awaited()


@pytest.mark.parametrize("phase", ["wait", "disconnect"])
def test_warmup_handle_cancellation_still_kills_sandbox(monkeypatch, phase):
    sandbox = _sandbox(monkeypatch)
    if phase == "wait":
        caller = None

        async def capture_caller(*_args, **_kwargs):
            nonlocal caller
            caller = asyncio.current_task()

        async def cancel_caller():
            caller.cancel()
            await asyncio.Event().wait()

        sandbox.files.write.side_effect = capture_caller
        sandbox.warmup_handle.wait.side_effect = cancel_caller
    else:
        sandbox.warmup_handle.disconnect.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    if phase == "wait":
        sandbox.judge_run.assert_not_awaited()


def test_warmup_start_and_wait_share_one_deadline(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setenv("E2B_REQUEST_TIMEOUT", "0.05")
    completed = sandbox.warmup_run.return_value

    async def delayed_start(*_args, **_kwargs):
        await asyncio.sleep(0.03)
        return completed

    async def delayed_end():
        await asyncio.sleep(0.03)
        return completed

    sandbox.warmup_run.side_effect = delayed_start
    sandbox.warmup_handle.wait.side_effect = delayed_end
    batch = code_e2b._build_request(_task(), CODE)

    with pytest.raises(code_e2b.CodeE2BError) as error:
        asyncio.run(code_e2b._run_batch(batch, json.dumps(batch)))

    assert error.value.retryable_warmup is True
    assert _batch_events(caplog, "failure")[0]["warmup_phase"] == "awaiting_end"
    sandbox.warmup_handle.disconnect.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.judge_run.assert_not_awaited()


def test_optional_command_transport_imports_are_not_required(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_connectrpc(name, *args, **kwargs):
        if name.startswith("connectrpc"):
            raise ImportError("older SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_connectrpc)
    assert code_e2b._is_warmup_transport_error(ConnectionError("reset"))
    assert not code_e2b._is_warmup_transport_error(RuntimeError("unclassified"))


def test_warmup_disconnect_failure_preserves_verdict_and_kills_sandbox(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    sandbox.warmup_handle.disconnect.side_effect = RuntimeError("disconnect failed test-api-key")

    assert _score()["reward"] == 1.0

    sandbox.kill.assert_awaited_once()
    failures = _batch_events(caplog, "cleanup_failure")
    assert failures[0]["operation"] == "warmup_disconnect"
    assert "test-api-key" not in caplog.text


def test_real_sdk_command_handle_wait_and_disconnect_are_compatible(monkeypatch, caplog):
    from e2b.sandbox_async.commands.command_handle import AsyncCommandHandle
    from protobuf import Oneof

    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)

    async def end_events():
        yield SimpleNamespace(
            event=SimpleNamespace(
                event=Oneof(
                    "end",
                    SimpleNamespace(
                        exit_code=0,
                        error="",
                        has_field=lambda _name: False,
                    ),
                )
            )
        )

    async def run(command, **kwargs):
        if command == "true":
            assert kwargs["background"] is True
            return AsyncCommandHandle(pid=456, handle_kill=AsyncMock(), events=end_events())
        return await sandbox.judge_run(command, **kwargs)

    sandbox.commands.run.side_effect = run

    assert _score()["reward"] == 1.0

    assert _batch_events(caplog, "warmup_started")[0]["warmup_pid"] == 456
    assert _batch_events(caplog, "warmup_finished")[0]["exit_code"] == 0
    sandbox.kill.assert_awaited_once()


def test_judge_local_deadline_bounds_stream_without_sdk_request_timeout(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "0.01")

    async def hanging_judge(*_args, **kwargs):
        assert kwargs["timeout"] == 0.01
        assert kwargs["request_timeout"] == 0
        await asyncio.Event().wait()

    sandbox.judge_run.side_effect = hanging_judge

    async def run():
        async with asyncio.timeout(2):
            batch = code_e2b._build_request(_task(), CODE)
            return await code_e2b._run_batch(batch, json.dumps(batch))

    with pytest.raises(code_e2b.CodeE2BError) as error:
        asyncio.run(run())

    assert error.value.stage == "run"
    assert error.value.is_timeout is True
    failure = _batch_events(caplog, "failure")[0]
    assert failure["error_type"] == "TimeoutError"
    assert failure["run_elapsed_s"] >= failure["command_timeout_s"]
    assert failure["command_time_remaining_s"] == 0
    sandbox.is_running.assert_awaited_once()
    sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize("timeout_kind", ["local_deadline", "sdk_timeout"])
@pytest.mark.parametrize("case_scores", [(1.0, 1.0), (1.0, 0.0)], ids=["pass", "fail"])
def test_run_timeout_retries_in_new_sandbox_and_uses_real_verdict(
    monkeypatch, caplog, timeout_kind, case_scores
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    first = _sandbox(monkeypatch)
    retry = _sandbox(monkeypatch, _result(case_scores))
    create_mock = _use_sandbox_sequence(monkeypatch, [first, retry])
    if timeout_kind == "local_deadline":
        monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "0.01")

        async def hanging_judge(*_args, **_kwargs):
            await asyncio.Event().wait()

        first.judge_run.side_effect = hanging_judge
    else:
        first.judge_run.side_effect = e2b.TimeoutException("command timed out test-api-key")
    task = _task()
    task.update(task_id="timeout-task", group_id="retry-group", sample_id="sample-a")
    task["metadata"] = {"private": "must-not-log"}
    original_task = copy.deepcopy(task)

    score = _score(task, deadline=2)

    correct = all(value == 1.0 for value in case_scores)
    assert score["status"] == "ok"
    assert score["reward"] == float(correct)
    assert score["correct"] is correct
    assert score["reason"] == ("accepted" if correct else "failed_tests")
    assert score["timed_out"] is False
    assert score["sandbox_result"] == _result(case_scores)
    assert score["metrics"] == {
        "passed_cases": float(sum(value == 1.0 for value in case_scores)),
        "total_cases": 2.0,
        "completed_batches": 1.0,
        "timeout_count": 1.0,
        "retry_count": 1.0,
        "timeout_batches": 0.0,
        "evaluated_cases": 2.0,
        "unevaluated_cases": 0.0,
    }
    assert task == original_task
    assert create_mock.await_count == 2
    assert first is not retry
    assert first.files.write.await_args.args == retry.files.write.await_args.args
    for sandbox in (first, retry):
        sandbox.warmup_run.assert_awaited_once()
        sandbox.judge_run.assert_awaited_once()
        sandbox.kill.assert_awaited_once()
    first.is_running.assert_awaited_once()
    retry.is_running.assert_not_awaited()
    failures = _batch_events(caplog, "failure")
    assert len(failures) == 1
    assert failures[0]["attempt"] == 1
    assert failures[0]["retryable_timeout"] is True
    timeout_records = [
        record
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] != "evaluation_complete"
    ]
    assert len(timeout_records) == 1
    assert timeout_records[0].code_e2b_retry["event"] == "retry_start"
    assert all(record.levelno == logging.WARNING for record in timeout_records)
    summaries = [
        record
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] == "evaluation_complete"
    ]
    assert len(summaries) == 1
    assert summaries[0].levelno == logging.INFO
    assert summaries[0].code_e2b_retry == {
        "event": "evaluation_complete",
        "task_id": "timeout-task",
        "group_id": "retry-group",
        "sample_id": "sample-a",
        "finish_reason": None,
        "reward": score["reward"],
        "reason": score["reason"],
        "timed_out": False,
        **score["metrics"],
    }
    assert "test-api-key" not in caplog.text
    assert "must-not-log" not in caplog.text


@pytest.mark.parametrize("timeout_kind", ["local_deadline", "sdk_timeout"])
def test_three_run_timeouts_return_zero_without_inventing_case_results(
    monkeypatch, caplog, timeout_kind
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)
    if timeout_kind == "local_deadline":
        monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "0.01")

        async def hanging_judge(*_args, **_kwargs):
            await asyncio.Event().wait()

        for sandbox in sandboxes:
            sandbox.judge_run.side_effect = hanging_judge
    else:
        for sandbox in sandboxes:
            sandbox.judge_run.side_effect = e2b.TimeoutException("judge deadline reached")

    score = _score(deadline=2)

    assert score == {
        "status": "ok",
        "reward": 0.0,
        "correct": False,
        "reason": "timeout_after_retry",
        "timed_out": True,
        "sandbox_result": {"accepted": False, "score": None, "tests": []},
        "metrics": {
            "passed_cases": 0.0,
            "total_cases": 2.0,
            "completed_batches": 0.0,
            "timeout_count": 3.0,
            "retry_count": 2.0,
            "timeout_batches": 1.0,
            "evaluated_cases": 0.0,
            "unevaluated_cases": 2.0,
        },
    }
    assert create_mock.await_count == 3
    assert sandboxes[0].files.write.await_args.args == sandboxes[1].files.write.await_args.args
    for sandbox in sandboxes:
        sandbox.judge_run.assert_awaited_once()
        sandbox.is_running.assert_awaited_once()
        sandbox.kill.assert_awaited_once()
    failures = _batch_events(caplog, "failure")
    assert [failure["attempt"] for failure in failures] == [1, 2, 3]
    assert [failure["sandbox_id"] for failure in failures] == [
        "test-sandbox-1",
        "test-sandbox-2",
        "test-sandbox-3",
    ]
    timeout_records = [
        record
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] != "evaluation_complete"
    ]
    assert [record.code_e2b_retry["event"] for record in timeout_records] == [
        "retry_start",
        "retry_start",
        "timeout_exhausted",
    ]
    assert not any(record.levelno >= logging.ERROR for record in timeout_records)
    summaries = [
        record
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] == "evaluation_complete"
    ]
    assert len(summaries) == 1
    assert summaries[0].levelno == logging.INFO
    assert summaries[0].code_e2b_retry == {
        "event": "evaluation_complete",
        "task_id": None,
        "group_id": None,
        "sample_id": None,
        "finish_reason": None,
        "reward": 0.0,
        "reason": "timeout_after_retry",
        "timed_out": True,
        **score["metrics"],
    }


def test_multibatch_timeout_retries_only_affected_cases_and_continues_remaining_batches(
    monkeypatch,
):
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 2)
    cases = [{"stdin": str(index), "expected_stdout": str(index)} for index in range(5)]
    sandboxes = [
        _sandbox(monkeypatch),
        _sandbox(monkeypatch, _result((1.0, 1.0))),
        _sandbox(monkeypatch),
        _sandbox(monkeypatch),
        _sandbox(monkeypatch),
        _sandbox(monkeypatch, _result((1.0,))),
    ]
    for index in (0, 2, 3, 4):
        sandboxes[index].judge_run.side_effect = e2b.TimeoutException("command timed out")
    create_mock = _use_sandbox_sequence(monkeypatch, sandboxes)

    score = _score(_task(cases))

    assert score["reward"] == 0.0
    assert score["reason"] == "timeout_after_retry"
    assert score["timed_out"] is True
    assert score["sandbox_result"] == {
        "accepted": False,
        "score": None,
        "tests": _result((1.0, 1.0, 1.0))["tests"],
    }
    assert score["metrics"] == {
        "passed_cases": 3.0,
        "total_cases": 5.0,
        "completed_batches": 2.0,
        "timeout_count": 4.0,
        "retry_count": 3.0,
        "timeout_batches": 1.0,
        "evaluated_cases": 3.0,
        "unevaluated_cases": 2.0,
    }
    assert create_mock.await_count == 6
    uploaded_cases = [
        json.loads(sandbox.files.write.await_args.args[1])["cases"] for sandbox in sandboxes
    ]
    assert uploaded_cases == [cases[:2], cases[:2], cases[2:4], cases[2:4], cases[2:4], cases[4:]]
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (e2b.SandboxException(PLACEMENT_ERROR_MESSAGE), True),
        (e2b.SandboxException(f"  {PLACEMENT_ERROR_MESSAGE}\n"), True),
        (e2b.SandboxException("500: unrelated server error"), True),
        (e2b.SandboxException("503: Failed to place sandbox"), True),
        (e2b.SandboxException(f"prefix {PLACEMENT_ERROR_MESSAGE}"), False),
        (e2b.SandboxException(f"{PLACEMENT_ERROR_MESSAGE}: other error"), True),
        (e2b.AuthenticationException(PLACEMENT_ERROR_MESSAGE), False),
        (e2b.TemplateException(PLACEMENT_ERROR_MESSAGE), False),
        (RuntimeError(PLACEMENT_ERROR_MESSAGE), False),
    ],
)
def test_sdk_service_status_requires_explicit_sdk_type_and_status_prefix(error, expected):
    assert (code_e2b._e2b_retry_reason(error, "create") is not None) is expected


@pytest.mark.parametrize("placement_error", [False, True], ids=["timeout", "placement"])
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("case_scores", [(1.0, 1.0), (1.0, 0.0)], ids=["pass", "fail"])
def test_create_retry_has_fresh_deadlines_and_preserves_request_and_verdict(
    monkeypatch, caplog, warmup_backoff, failures, case_scores, placement_error
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "90")
    monkeypatch.setenv("E2B_REQUEST_TIMEOUT", "7")
    sandbox = _sandbox(monkeypatch, _result(case_scores))
    errors = [
        (
            e2b.SandboxException(PLACEMENT_ERROR_MESSAGE)
            if placement_error
            else httpx.ReadTimeout("create failure test-api-key e2b_secret")
        )
        for _ in range(failures)
    ]
    create_mock = _use_sandbox_sequence(monkeypatch, [*errors, sandbox])
    original_timeout = asyncio.timeout
    deadlines = []

    def record_timeout(delay):
        timer = original_timeout(delay)
        if delay == 1200:  # The complete attempt receives at least twenty minutes.
            deadlines.append(timer)
        return timer

    monkeypatch.setattr(code_e2b.asyncio, "timeout", record_timeout)
    task = _task(code_template="model = __MODEL_CODE__", generated_checker="private-checker")
    task.update(task_id=123, group_id="group", sample_id="sample")
    original_task = copy.deepcopy(task)

    score = _score(task)

    assert score["reward"] == float(all(value == 1 for value in case_scores))
    assert score["sandbox_result"] == _result(case_scores)
    assert score["reason"] == ("accepted" if score["correct"] else "failed_tests")
    assert score["metrics"] == {
        "passed_cases": float(sum(value == 1 for value in case_scores)),
        "total_cases": 2.0,
        "completed_batches": 1.0,
        "create_failure_count": float(failures),
        "create_retry_count": float(failures),
    }
    assert "timed_out" not in score
    assert task == original_task
    assert create_mock.await_count == failures + 1
    options = [call.kwargs for call in create_mock.await_args_list]
    assert all(option == options[0] and option["timeout"] == 1200 for option in options)
    assert len(deadlines) == failures + 1
    assert len({id(timer) for timer in deadlines}) == failures + 1
    assert not any(timer.expired() for timer in deadlines)
    payload = json.loads(sandbox.files.write.await_args.args[1])
    assert payload == code_e2b._build_request(original_task, CODE)
    sandbox.files.write.assert_awaited_once()
    sandbox.judge_run.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    assert warmup_backoff.sleep.await_count == failures
    assert warmup_backoff.delay.call_count == failures
    warmup_backoff.delay.assert_called_with(1.0, 3.0)
    events = [
        record.code_e2b_retry["event"]
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
    ]
    assert events == ["create_retry_start"] * failures + ["create_recovered"]
    assert all(event["retryable_create"] for event in _batch_events(caplog, "failure"))
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert "test-api-key" not in caplog.text
    assert "e2b_secret" not in caplog.text
    assert "private-checker" not in caplog.text


@pytest.mark.parametrize(
    "failure_kinds",
    [("placement", "placement"), ("placement", "timeout"), ("timeout", "placement")],
)
@pytest.mark.parametrize("recover", [True, False], ids=["recovered", "exhausted"])
def test_placement_and_timeout_share_existing_create_retry_budget(
    monkeypatch, caplog, warmup_backoff, failure_kinds, recover
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    errors = [
        (
            e2b.SandboxException(PLACEMENT_ERROR_MESSAGE)
            if kind == "placement"
            else httpx.ReadTimeout("creation timed out")
        )
        for kind in failure_kinds
    ]
    final = sandbox if recover else e2b.SandboxException(PLACEMENT_ERROR_MESSAGE)
    create_mock = _use_sandbox_sequence(monkeypatch, [*errors, final])

    if recover:
        task = _task(code_template="model = __MODEL_CODE__", generated_checker="private-checker")
        original_task = copy.deepcopy(task)
        score = _score(task)
        assert score["reward"] == 1.0
        assert score["reason"] == "accepted"
        assert score["metrics"]["create_failure_count"] == 2.0
        assert score["metrics"]["create_retry_count"] == 2.0
        assert "timeout_count" not in score["metrics"]
        assert task == original_task
        assert json.loads(sandbox.files.write.await_args.args[1]) == code_e2b._build_request(
            original_task, CODE
        )
        sandbox.files.write.assert_awaited_once()
        sandbox.judge_run.assert_awaited_once()
        sandbox.kill.assert_awaited_once()
    else:
        with pytest.raises(code_e2b.CodeE2BError) as raised:
            _score()
        assert raised.value.stage == "create"
        assert raised.value.retryable_create is True
        assert raised.value.is_timeout is False
        assert raised.value.attempt_deadline_expired is False
        sandbox.files.write.assert_not_awaited()
        sandbox.judge_run.assert_not_awaited()
        sandbox.kill.assert_not_awaited()

    assert create_mock.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    assert warmup_backoff.delay.call_args_list == [call(1.0, 3.0)] * 2
    retry_events = [
        record.code_e2b_retry for record in caplog.records if hasattr(record, "code_e2b_retry")
    ]
    assert [event["event"] for event in retry_events] == [
        "create_retry_start",
        "create_retry_start",
        "create_recovered" if recover else "create_exhausted",
    ]
    assert retry_events[-1]["create_failure_count"] == (2 if recover else 3)
    assert retry_events[-1]["create_retry_count"] == 2


@pytest.mark.parametrize("stage", ["upload", "warmup", "run"])
def test_sdk_service_error_retries_upload_warmup_and_shared_run(monkeypatch, warmup_backoff, stage):
    task = _shared_task(monkeypatch) if stage == "run" else _task()
    sandbox, retry = _sandbox(monkeypatch), _sandbox(monkeypatch)
    operation = {
        "upload": sandbox.files.write,
        "warmup": sandbox.warmup_run,
        "run": sandbox.judge_run,
    }[stage]
    operation.side_effect = e2b.SandboxException(PLACEMENT_ERROR_MESSAGE)
    create = _use_sandbox_sequence(monkeypatch, [sandbox, retry])
    assert _score(task)["reward"] == 1
    assert create.await_count == 2
    sandbox.kill.assert_awaited_once()
    retry.kill.assert_awaited_once()
    warmup_backoff.sleep.assert_awaited_once()


def test_placement_error_after_whole_attempt_expiry_is_not_retried(
    monkeypatch, caplog, warmup_backoff
):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setattr(
        code_e2b, "_timeouts", lambda _batch, **_kwargs: (60.0, 60.0, 300.0, 0.01, 60)
    )

    async def create_after_deadline(**_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A late SDK error must not override the expired outer deadline.
            raise e2b.SandboxException(PLACEMENT_ERROR_MESSAGE) from None

    sandbox.create.side_effect = create_after_deadline

    with pytest.raises(code_e2b.CodeE2BError) as raised:
        _score(deadline=2)

    assert raised.value.stage == "create"
    assert raised.value.error_type == "SandboxException"
    assert raised.value.attempt_deadline_expired is True
    assert raised.value.retryable_create is False
    sandbox.create.assert_awaited_once()
    sandbox.files.write.assert_not_awaited()
    sandbox.kill.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()
    assert not any(hasattr(record, "code_e2b_retry") for record in caplog.records)


@pytest.mark.parametrize(
    "error_type",
    [
        TimeoutError,
        e2b.TimeoutException,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        ConnectionResetError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    ],
)
def test_three_create_failures_raise_without_zero_scoring(
    monkeypatch, caplog, warmup_backoff, error_type
):
    sandbox = _sandbox(monkeypatch)
    sandbox.create.side_effect = error_type("setup timed out")

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "create"
    assert error.value.retryable_create is True
    assert error.value.attempt_deadline_expired is False
    assert error.value.is_timeout is False
    assert error.value.retryable_upload is False
    assert error.value.retryable_warmup is False
    assert sandbox.create.await_count == 3
    sandbox.judge_run.assert_not_awaited()
    sandbox.files.write.assert_not_awaited()
    sandbox.kill.assert_not_awaited()
    assert warmup_backoff.sleep.await_count == 2
    exhausted = [
        record
        for record in caplog.records
        if hasattr(record, "code_e2b_retry")
        and record.code_e2b_retry["event"] == "create_exhausted"
    ]
    assert len(exhausted) == 1
    assert exhausted[0].levelno == logging.ERROR
    assert exhausted[0].code_e2b_retry["create_failure_count"] == 3
    assert exhausted[0].code_e2b_retry["create_retry_count"] == 2


@pytest.mark.parametrize(
    "error_type",
    [
        e2b.AuthenticationException,
        e2b.InvalidArgumentException,
        e2b.TemplateException,
        e2b.SandboxException,
        httpx.LocalProtocolError,
        httpx.UnsupportedProtocol,
        ssl.SSLCertVerificationError,
        RuntimeError,
        ValueError,
        TypeError,
    ],
)
def test_nontransient_create_failures_do_not_retry(monkeypatch, warmup_backoff, error_type):
    sandbox = _sandbox(monkeypatch)
    sandbox.create.side_effect = error_type("nontransient create failure")

    with pytest.raises(code_e2b.CodeE2BError) as raised:
        _score()

    assert raised.value.stage == "create"
    assert raised.value.retryable_create is False
    sandbox.create.assert_awaited_once()
    sandbox.files.write.assert_not_awaited()
    sandbox.kill.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()


def test_create_certificate_error_is_not_retried(monkeypatch, warmup_backoff):
    sandbox = _sandbox(monkeypatch)
    error = httpx.ConnectError("certificate verification failed")
    error.__cause__ = ssl.SSLCertVerificationError("untrusted certificate")
    sandbox.create.side_effect = error

    with pytest.raises(code_e2b.CodeE2BError) as raised:
        _score()

    assert raised.value.retryable_create is False
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("placement_error", [False, True], ids=["timeout", "placement"])
@pytest.mark.parametrize("cancel_stage", ["create", "backoff"])
def test_create_cancellation_never_creates_another_sandbox(
    monkeypatch, warmup_backoff, cancel_stage, placement_error
):
    sandbox = _sandbox(monkeypatch)
    sandbox.create.side_effect = (
        asyncio.CancelledError()
        if cancel_stage == "create"
        else (
            e2b.SandboxException(PLACEMENT_ERROR_MESSAGE)
            if placement_error
            else httpx.ReadTimeout("create failed")
        )
    )
    if cancel_stage == "backoff":
        warmup_backoff.sleep.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_not_awaited()
    sandbox.files.write.assert_not_awaited()
    assert warmup_backoff.sleep.await_count == int(cancel_stage == "backoff")


@pytest.mark.parametrize("ending", ["pass", "run", "create", "upload", "warmup"])
def test_create_upload_warmup_run_budgets_are_independent_and_bounded(
    monkeypatch, warmup_backoff, ending
):
    events = [
        "create",
        "upload",
        "warmup",
        "upload",
        "run",
        "create",
        "upload",
        "warmup",
        "upload",
        ending,
    ]
    if ending == "run":
        events.append("run")
    attempts = []
    sandboxes = []
    for event in events:
        if event == "create":
            attempts.append(httpx.ReadTimeout("creation timed out"))
            continue
        sandbox = _sandbox(monkeypatch)
        if event == "upload":
            sandbox.files.write.side_effect = httpx.ReadTimeout("upload timed out")
        elif event == "warmup":
            sandbox.warmup_run.side_effect = e2b.TimeoutException("warmup timed out")
        elif event == "run":
            sandbox.judge_run.side_effect = e2b.TimeoutException("judge timed out")
        attempts.append(sandbox)
        sandboxes.append(sandbox)
    create_mock = _use_sandbox_sequence(monkeypatch, attempts)

    if ending in ("create", "upload", "warmup"):
        with pytest.raises(code_e2b.CodeE2BError) as raised:
            _score()
        assert raised.value.stage == ending
    else:
        score = _score()
        assert score["reward"] == float(ending == "pass")
        assert score["reason"] == ("accepted" if ending == "pass" else "timeout_after_retry")
        assert score["metrics"]["create_failure_count"] == 2.0
        assert score["metrics"]["create_retry_count"] == 2.0
        assert score["metrics"]["upload_retry_count"] == 4.0
        assert score["metrics"]["warmup_retry_count"] == 2.0
        assert score["metrics"]["timeout_count"] == events.count("run")
    assert create_mock.await_count == len(events)
    assert warmup_backoff.sleep.await_count == 8
    payloads = [sandbox.files.write.await_args.args[1] for sandbox in sandboxes]
    assert all(payload == payloads[0] for payload in payloads)
    # Create, warmup, and judge retries must neither grow nor reset upload time.
    assert [sandbox.files.write.await_args.kwargs["request_timeout"] for sandbox in sandboxes] == [
        300,
        450,
        450,
        600,
        600,
        750,
        750,
        900,
        900,
    ][: len(sandboxes)]
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()


def test_create_retry_budget_is_per_batch_and_does_not_rerun_completed_batches(
    monkeypatch, warmup_backoff
):
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 1)
    cases = [{"stdin": str(index), "expected_stdout": str(index)} for index in range(3)]
    sandboxes = [_sandbox(monkeypatch, _result((1.0,))) for _ in cases]
    create_mock = _use_sandbox_sequence(
        monkeypatch,
        [
            sandboxes[0],
            httpx.ReadTimeout("create failed"),
            httpx.ReadTimeout("create failed"),
            sandboxes[1],
            httpx.ReadTimeout("create failed"),
            httpx.ReadTimeout("create failed"),
            sandboxes[2],
        ],
    )

    score = _score(_task(cases))

    assert score["reward"] == 1.0
    assert score["metrics"]["create_failure_count"] == 4.0
    assert score["metrics"]["create_retry_count"] == 4.0
    assert create_mock.await_count == 7
    uploaded = [
        json.loads(sandbox.files.write.await_args.args[1])["cases"] for sandbox in sandboxes
    ]
    assert uploaded == [[case] for case in cases]
    for sandbox in sandboxes:
        sandbox.judge_run.assert_awaited_once()


def test_create_retry_budget_is_local_to_each_answer(monkeypatch, warmup_backoff):
    sandboxes = [_sandbox(monkeypatch) for _ in range(2)]
    attempts = []
    for sandbox in sandboxes:
        attempts.extend(
            [httpx.ReadTimeout("create failed"), httpx.ReadTimeout("create failed"), sandbox]
        )
    create_mock = _use_sandbox_sequence(monkeypatch, attempts)

    for _ in sandboxes:
        score = _score()
        assert score["reward"] == 1.0
        assert score["metrics"]["create_retry_count"] == 2.0
    assert create_mock.await_count == 6


def test_non_timeout_retry_failure_propagates_instead_of_zero_scoring(monkeypatch):
    first = _sandbox(monkeypatch)
    retry = _sandbox(monkeypatch, stdout="not json")
    first.judge_run.side_effect = e2b.TimeoutException("command timed out")
    create_mock = _use_sandbox_sequence(monkeypatch, [first, retry])

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "parse"
    assert error.value.is_timeout is False
    assert create_mock.await_count == 2
    first.kill.assert_awaited_once()
    retry.kill.assert_awaited_once()


@pytest.mark.parametrize("cancel_stage", ["retry_judge", "first_probe"])
def test_cancellation_during_timeout_handling_propagates_without_zero_scoring(
    monkeypatch, cancel_stage
):
    first = _sandbox(monkeypatch)
    retry = _sandbox(monkeypatch)
    first.judge_run.side_effect = e2b.TimeoutException("command timed out")
    create_mock = _use_sandbox_sequence(monkeypatch, [first, retry])
    if cancel_stage == "retry_judge":
        retry.judge_run.side_effect = asyncio.CancelledError()
    else:

        async def cancel_probe(**_kwargs):
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        first.is_running.side_effect = cancel_probe

    with pytest.raises(asyncio.CancelledError):
        _score()

    assert create_mock.await_count == (2 if cancel_stage == "retry_judge" else 1)
    first.kill.assert_awaited_once()
    if cancel_stage == "retry_judge":
        retry.kill.assert_awaited_once()
    else:
        retry.judge_run.assert_not_awaited()
        retry.kill.assert_not_awaited()


def test_command_warmup_local_deadline_probes_then_cleans_up(
    monkeypatch, caplog, mock_tcp_user_timeout
):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setenv("E2B_REQUEST_TIMEOUT", "0.01")

    async def hanging_warmup(*_args, **kwargs):
        assert kwargs["timeout"] == 0.01
        assert kwargs["request_timeout"] == 0.01
        await asyncio.Event().wait()

    async def probe(**_kwargs):
        sandbox.kill.assert_not_awaited()
        return True

    sandbox.warmup_run.side_effect = hanging_warmup
    sandbox.is_running.side_effect = probe

    async def run():
        async with asyncio.timeout(2):
            batch = code_e2b._build_request(_task(), CODE)
            return await code_e2b._run_batch(batch, json.dumps(batch))

    with pytest.raises(code_e2b.CodeE2BError) as error:
        asyncio.run(run())

    assert error.value.stage == "warmup"
    assert error.value.retryable_warmup is True
    failure = _batch_events(caplog, "failure")[0]
    assert failure["error_type"] == "TimeoutError"
    assert failure["warmup_timeout_s"] == 0.01
    assert failure["run_elapsed_s"] is None
    assert failure["command_time_remaining_s"] is None
    mock_tcp_user_timeout.assert_not_called()
    sandbox.judge_run.assert_not_awaited()
    sandbox.is_running.assert_awaited_once()
    sandbox.kill.assert_awaited_once()


def test_cancelled_command_warmup_cleans_up_without_tuning_or_probe(
    monkeypatch, caplog, mock_tcp_user_timeout
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    caller = None

    async def capture_caller(*_args, **_kwargs):
        nonlocal caller
        caller = asyncio.current_task()

    async def cancel_caller(*_args, **_kwargs):
        caller.cancel()
        await asyncio.Event().wait()

    sandbox.files.write.side_effect = capture_caller
    sandbox.warmup_run.side_effect = cancel_caller

    with pytest.raises(asyncio.CancelledError):
        _score()

    cancelled = _batch_events(caplog, "cancelled")[0]
    assert cancelled["stage"] == "warmup"
    assert cancelled["run_elapsed_s"] is None
    mock_tcp_user_timeout.assert_not_called()
    sandbox.judge_run.assert_not_awaited()
    sandbox.is_running.assert_not_awaited()
    sandbox.kill.assert_awaited_once()


def test_nonzero_command_warmup_never_starts_judging(monkeypatch, mock_tcp_user_timeout):
    sandbox = _sandbox(monkeypatch)
    sandbox.warmup_run.return_value = CommandResult(stderr="", stdout="", exit_code=1, error=None)

    with pytest.raises(code_e2b.CodeE2BError, match="warmup exited with code 1") as error:
        _score()

    assert error.value.stage == "warmup"
    mock_tcp_user_timeout.assert_not_called()
    sandbox.judge_run.assert_not_awaited()
    sandbox.kill.assert_awaited_once()


def test_unavailable_socket_tuning_retains_reward_and_reports_status(
    monkeypatch, caplog, mock_tcp_user_timeout
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    _sandbox(monkeypatch)
    mock_tcp_user_timeout.return_value = SimpleNamespace(
        status="unsupported",
        host=None,
        port=None,
        timeout_ms=3600000,
        matched_sockets=0,
        configured_sockets=0,
        failed_sockets=0,
        previous_timeouts_ms=(),
        current_timeouts_ms=(),
    )

    assert _score()["reward"] == 1.0

    completed = _batch_events(caplog, "batch_complete")[0]
    assert completed["tcp_timeout_status"] == "unsupported"
    assert completed["tcp_configured_sockets"] == 0


def test_configured_template_and_estimated_timeout_include_generated_checker(monkeypatch):
    monkeypatch.setenv("E2B_TEMPLATE", "custom-code-judge")
    sandbox = _sandbox(monkeypatch)
    task = _task(compile_timeout=10, run_timeout=2, checker_timeout=5, generated_checker="checker")
    command, rpc, upload, lifetime, estimated = code_e2b._timeouts(task["label"])

    assert (command, rpc, upload, lifetime, estimated) == (72, 60, 300, 1200, 72)
    _score(task)
    assert sandbox.create.await_args.kwargs["template"] == "custom-code-judge"


@pytest.mark.parametrize(
    "rpc_setting,upload_setting,expected_upload",
    [(60, None, 300), (600, None, 600), (60, "900", 900)],
)
def test_upload_timeout_is_independent_and_included_in_attempt_lifetime(
    monkeypatch, rpc_setting, upload_setting, expected_upload
):
    monkeypatch.setenv("E2B_REQUEST_TIMEOUT", str(rpc_setting))
    if upload_setting is not None:
        monkeypatch.setenv("E2B_UPLOAD_REQUEST_TIMEOUT", upload_setting)
    sandbox = _sandbox(monkeypatch)

    assert _score()["reward"] == 1.0

    assert sandbox.files.write.await_args.kwargs["request_timeout"] == expected_upload
    assert sandbox.create.await_args.kwargs["request_timeout"] == rpc_setting
    warmup_budget = min(rpc_setting, 30)
    assert sandbox.warmup_run.await_args.kwargs["timeout"] == warmup_budget
    assert sandbox.judge_run.await_args.kwargs["timeout"] == 60
    assert sandbox.create.await_args.kwargs["timeout"] == max(
        1200, 60 + 2 * rpc_setting + expected_upload + warmup_budget + 30
    )


@pytest.mark.parametrize(
    "command_timeout,expected_lifetime",
    [(60, 1200), (720, 1200), (720.1, 1201), (900, 1380)],
)
def test_attempt_budget_has_twenty_minute_floor_without_clipping_longer_commands(
    command_timeout, expected_lifetime
):
    command, rpc, upload, lifetime, _estimated = code_e2b._timeouts(
        _task()["label"], command_timeout=command_timeout
    )

    assert lifetime == expected_lifetime
    assert command == command_timeout
    assert rpc == 60
    assert upload == 300


def test_large_default_judge_budget_reserves_upload_within_one_hour(monkeypatch):
    monkeypatch.setenv("E2B_UPLOAD_REQUEST_TIMEOUT", "600")
    task = _task(CASES * 1000)
    command, rpc, upload, lifetime, estimated = code_e2b._timeouts(task["label"])

    assert estimated > 3600
    assert lifetime == 3600
    assert rpc == 60
    assert upload == 600
    assert command == 3600 - (2 * 60 + 600 + 30 + 30)


@pytest.mark.parametrize(
    "configured_command,expected_command,expected_lifetimes",
    [(None, 2820, [3600, 3900, 4200, 4500, 4800]), (4000, 4000, [4780, 5080, 5380, 5680, 5980])],
)
def test_upload_retries_extend_lifetime_without_shortening_judge_budget(
    monkeypatch, configured_command, expected_command, expected_lifetimes
):
    monkeypatch.setenv("E2B_UPLOAD_REQUEST_TIMEOUT", "600")
    task = _task(CASES * 1000)
    budgets = [
        code_e2b._timeouts(
            task["label"], command_timeout=configured_command, upload_retry_count=retry_count
        )
        for retry_count in range(5)
    ]

    assert [command for command, *_rest in budgets] == [expected_command] * 5
    assert [rpc for _command, rpc, *_rest in budgets] == [60] * 5
    assert [upload for _command, _rpc, upload, *_rest in budgets] == [600, 900, 1200, 1500, 1800]
    assert [lifetime for *_prefix, lifetime, _estimated in budgets] == expected_lifetimes
    assert (
        code_e2b._timeouts(task["label"], command_timeout=configured_command, upload_retry_count=20)
        == budgets[-1]
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("E2B_COMMAND_TIMEOUT", "0"),
        ("E2B_COMMAND_TIMEOUT", "nan"),
        ("E2B_COMMAND_TIMEOUT", "86400"),
        ("E2B_REQUEST_TIMEOUT", "-1"),
        ("E2B_UPLOAD_REQUEST_TIMEOUT", "0"),
        ("E2B_UPLOAD_REQUEST_TIMEOUT", "-1"),
        ("E2B_UPLOAD_REQUEST_TIMEOUT", "nan"),
        ("E2B_UPLOAD_REQUEST_TIMEOUT", "inf"),
        ("E2B_UPLOAD_REQUEST_TIMEOUT", "86400"),
    ],
)
def test_invalid_timeout_setting_fails_before_create(monkeypatch, no_real_e2b, setting, value):
    monkeypatch.setenv(setting, value)
    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()
    assert error.value.stage == "config"
    no_real_e2b.assert_not_awaited()


def test_missing_api_key_is_config_failure(no_real_e2b, monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()
    assert error.value.stage == "config"
    no_real_e2b.assert_not_awaited()


@pytest.mark.parametrize("template", [None, "", " \t\n"])
@pytest.mark.parametrize("shared", [False, True], ids=["uploaded-cases", "shared-cases"])
def test_missing_or_blank_template_fails_before_create(
    no_real_e2b, monkeypatch, warmup_backoff, template, shared
):
    if template is None:
        monkeypatch.delenv("E2B_TEMPLATE")
    else:
        monkeypatch.setenv("E2B_TEMPLATE", template)
    task = _shared_task(monkeypatch) if shared else _task()

    with pytest.raises(code_e2b.CodeE2BError, match="E2B_TEMPLATE must name") as error:
        _score(task)

    assert error.value.stage == "config"
    no_real_e2b.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("stage", ["create", "upload", "warmup", "run", "parse"])
def test_infrastructure_failures_raise_with_stage_and_always_cleanup(
    monkeypatch, caplog, capsys, stage
):
    sandbox = _sandbox(monkeypatch, stdout="not json" if stage == "parse" else None)
    if stage != "parse":
        operation = {
            "create": sandbox.create,
            "upload": sandbox.files.write,
            "warmup": sandbox.warmup_run,
            "run": sandbox.judge_run,
        }[stage]
        operation.side_effect = RuntimeError("service failed test-api-key e2b_secret")

    task = {
        **_task(),
        "task_id": 17,
        "group_id": "group-a",
        "sample_id": "sample-a-test-api-key",
    }
    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score(task)

    assert error.value.stage == stage
    message = str(error.value)
    assert "task_id=17" in message
    assert "group_id='group-a'" in message
    assert "sample_id='sample-a-[REDACTED]'" in message
    assert "batch_index=1" in message
    assert "attempt=1" in message
    assert "payload_bytes=" in message
    assert "test-api-key" not in message
    assert "e2b_secret" not in message
    assert capsys.readouterr().err == ""
    assert _batch_events(caplog, "failure")[0]["stage"] == stage
    if stage == "create":
        assert "sandbox_id=" not in message
        sandbox.kill.assert_not_awaited()
    else:
        assert "sandbox_id='test-sandbox'" in message
        sandbox.kill.assert_awaited_once()


def test_nonzero_judge_exit_never_produces_zero_reward(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch, exit_code=1)
    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()
    assert error.value.stage == "run"
    sandbox.kill.assert_awaited_once()
    run_complete = next(
        fields for fields in _batch_events(caplog, "stage_complete") if fields["stage"] == "run"
    )
    assert _batch_events(caplog, "failure")[0]["run_elapsed_s"] == run_complete["run_elapsed_s"]


def test_command_exit_exception_is_infrastructure_failure(monkeypatch):
    sandbox = _sandbox(monkeypatch)
    sandbox.judge_run.side_effect = e2b.CommandExitException(
        stderr="judge unavailable",
        stdout="",
        exit_code=127,
        error=None,
    )
    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()
    assert error.value.stage == "run"
    sandbox.kill.assert_awaited_once()


def test_stream_output_limit_counts_utf8_bytes_across_both_streams(monkeypatch):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setattr(code_e2b, "_MAX_RESULT_BYTES", 5)

    async def run(*_args, **kwargs):
        kwargs["on_stdout"]("中")
        kwargs["on_stderr"]("中")
        raise AssertionError("Excess output must fail before the judge returns")

    sandbox.judge_run.side_effect = run
    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()
    assert error.value.stage == "run"
    assert "output" in str(error.value)
    sandbox.kill.assert_awaited_once()


def test_cancellation_propagates_and_kills_created_sandbox(monkeypatch):
    sandbox = _sandbox(monkeypatch)
    sandbox.judge_run.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _score()
    sandbox.kill.assert_awaited_once()


def test_cleanup_error_keeps_score_and_redacts_credentials(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    sandbox.kill.side_effect = RuntimeError("cleanup failed test-api-key e2b_secret")
    score = _score()
    assert score["reward"] == 1.0
    assert "kill" in caplog.text.lower()
    assert "test-api-key" not in caplog.text
    assert "e2b_secret" not in caplog.text


def _batch_events(caplog, event):
    return [
        record.code_e2b_batch
        for record in caplog.records
        if hasattr(record, "code_e2b_batch") and record.code_e2b_batch["event"] == event
    ]


def test_batch_diagnostics_correlate_tasks_budgets_and_stage_timings(
    monkeypatch, caplog, mock_tcp_user_timeout
):
    caplog.set_level(logging.DEBUG, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "300")
    monkeypatch.setenv("E2B_REQUEST_TIMEOUT", "25")
    task = {**_task(), "task_id": 17, "group_id": "group-a", "sample_id": "sample-a"}

    score = asyncio.run(code_e2b.code_e2b_reward(task, {"response": CODE, "finish_reason": "stop"}))

    assert score["reward"] == 1.0
    completed = _batch_events(caplog, "batch_complete")
    assert len(completed) == 1
    fields = completed[0]
    assert fields["task_id"] == 17
    assert fields["group_id"] == "group-a"
    assert fields["sample_id"] == "sample-a"
    assert fields["finish_reason"] == "stop"
    assert fields["batch_index"] == 1
    assert fields["case_count"] == 2
    assert fields["payload_bytes"] == len(sandbox.files.write.await_args.args[1])
    assert fields["command_timeout_s"] == 300
    assert fields["command_request_timeout_s"] == 0
    assert fields["warmup_timeout_s"] == 25
    assert fields["rpc_timeout_s"] == 25
    assert fields["upload_timeout_s"] == 300
    assert fields["lifetime_timeout_s"] == 1200
    assert fields["estimated_timeout_s"] == 60
    assert fields["sandbox_id"] == "test-sandbox"
    assert fields["sandbox_host"] == "49983-test-sandbox.e2b.example.test"
    assert fields["sandbox_port"] == 49983
    report = mock_tcp_user_timeout.return_value
    assert fields["tcp_user_timeout_ms"] == report.timeout_ms
    assert fields["tcp_matched_sockets"] == report.matched_sockets
    assert fields["tcp_configured_sockets"] == report.configured_sockets
    assert fields["tcp_failed_sockets"] == report.failed_sockets
    assert fields["tcp_timeout_status"] == report.status
    assert fields["tcp_previous_timeouts_ms"] == report.previous_timeouts_ms
    assert fields["tcp_current_timeouts_ms"] == report.current_timeouts_ms
    assert fields["total_elapsed_s"] >= fields["run_elapsed_s"] >= 0
    run_complete = next(
        fields for fields in _batch_events(caplog, "stage_complete") if fields["stage"] == "run"
    )
    assert fields["run_elapsed_s"] == run_complete["run_elapsed_s"]
    assert {fields["stage"] for fields in _batch_events(caplog, "stage_complete")} >= {
        "create",
        "upload",
        "warmup",
        "transport",
        "run",
        "parse",
    }
    # Retained log records describe their own stage rather than a mutable final state.
    assert _batch_events(caplog, "stage_start")[0]["stage"] == "sdk_import"
    sandbox.is_running.assert_not_awaited()


@pytest.mark.parametrize("running", [True, False])
def test_run_failure_probes_before_cleanup_and_logs_early_failure(monkeypatch, caplog, running):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "300")
    sandbox.judge_run.side_effect = RuntimeError("connection closed early")

    async def probe(**kwargs):
        sandbox.kill.assert_not_awaited()
        assert 0 < kwargs["request_timeout"] <= 3
        return running

    sandbox.is_running.side_effect = probe
    with pytest.raises(code_e2b.CodeE2BError, match="connection closed early") as error:
        _score()

    assert error.value.stage == "run"
    failure = _batch_events(caplog, "failure")[0]
    assert failure["error_type"] == "RuntimeError"
    assert failure["sandbox_id"] == "test-sandbox"
    assert 0 <= failure["run_elapsed_s"] < failure["command_timeout_s"]
    assert 0 < failure["command_time_remaining_s"] <= 300
    state = _batch_events(caplog, "connectivity_probe")[0]
    assert state["connectivity"] == ("running" if running else "not_running")
    assert state["run_elapsed_s"] == failure["run_elapsed_s"]
    sandbox.is_running.assert_awaited_once()
    sandbox.kill.assert_awaited_once()


def test_probe_error_is_unknown_redacted_and_preserves_original_failure(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    sandbox.judge_run.side_effect = RuntimeError("original run failure test-api-key")
    sandbox.is_running.side_effect = RuntimeError("probe failed test-api-key e2b_secret")

    with pytest.raises(code_e2b.CodeE2BError, match="original run failure"):
        _score()

    state = _batch_events(caplog, "connectivity_probe")[0]
    assert state["connectivity"] == "unknown"
    assert "probe failed" in state["probe_error"]
    assert "test-api-key" not in caplog.text
    assert "e2b_secret" not in caplog.text
    sandbox.kill.assert_awaited_once()


def test_hanging_connectivity_probe_is_bounded_and_keeps_run_error(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    sandbox.judge_run.side_effect = RuntimeError("original run failure")
    monkeypatch.setattr(code_e2b, "_SANDBOX_PROBE_TIMEOUT_S", 0.01)

    async def hanging_probe(**_kwargs):
        await asyncio.Event().wait()

    sandbox.is_running.side_effect = hanging_probe

    async def run():
        async with asyncio.timeout(2):
            return await code_e2b.code_e2b_reward(_task(), {"response": CODE})

    with pytest.raises(code_e2b.CodeE2BError, match="original run failure"):
        asyncio.run(run())

    assert _batch_events(caplog, "connectivity_probe")[0]["connectivity"] == "unknown"
    sandbox.kill.assert_awaited_once()


def test_missing_probe_method_preserves_run_failure(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    del sandbox.is_running
    del sandbox.get_host
    sandbox.judge_run.side_effect = RuntimeError("run failure")

    with pytest.raises(code_e2b.CodeE2BError, match="run failure"):
        _score()

    assert _batch_events(caplog, "connectivity_probe")[0]["connectivity"] == "unknown"
    sandbox.kill.assert_awaited_once()


def test_diagnostic_host_excludes_url_credentials_and_query(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    sandbox.get_host = lambda _port: (
        "https://user:test-api-key@envd.example.test:49983/path?token=e2b_secret"
    )
    sandbox.envd_api_url = "https://user:unknown-password@rpc.example.test:443/?token=unknown-token"

    _score()

    assert _batch_events(caplog, "batch_complete")[0]["sandbox_host"] == "envd.example.test"
    assert _batch_events(caplog, "batch_complete")[0]["envd_host"] == "rpc.example.test"
    assert _batch_events(caplog, "batch_complete")[0]["envd_port"] == 443
    assert "test-api-key" not in caplog.text
    assert "e2b_secret" not in caplog.text
    assert "?token=" not in caplog.text
    assert "unknown-password" not in caplog.text
    assert "unknown-token" not in caplog.text


def test_sdk_error_urls_do_not_expose_unconfigured_credentials(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    sandbox.judge_run.side_effect = RuntimeError(
        "disconnected https://user:unknown-password@rpc.example.test/run?token=unknown-token#secret"
    )

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    for text in (str(error.value), caplog.text):
        assert "unknown-password" not in text
        assert "unknown-token" not in text
        assert "#secret" not in text
        assert "rpc.example.test" in text


def test_cancellation_logs_context_without_probing_or_masking_cancel(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    sandbox.judge_run.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _score()

    cancelled = _batch_events(caplog, "cancelled")[0]
    assert cancelled["stage"] == "run"
    assert cancelled["sandbox_id"] == "test-sandbox"
    sandbox.is_running.assert_not_awaited()
    sandbox.kill.assert_awaited_once()


def test_multibatch_diagnostics_identify_each_case_subset(
    monkeypatch, caplog, mock_tcp_user_timeout
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 1)
    sandbox = _sandbox(monkeypatch, _result((1.0,)))

    _score()

    completed = _batch_events(caplog, "batch_complete")
    assert [(fields["batch_index"], fields["case_count"]) for fields in completed] == [
        (1, 1),
        (2, 1),
    ]
    sandbox.is_running.assert_not_awaited()
    assert sandbox.warmup_run.await_count == 2
    assert mock_tcp_user_timeout.call_count == 2
    assert sandbox.kill.await_count == 2


@pytest.mark.parametrize("phase", ["start", "wait"])
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("scores", [(1.0, 1.0), (0.0, 0.0)])
def test_owned_warmup_cancellation_recovers_with_identical_payload(
    monkeypatch, caplog, warmup_backoff, phase, failures, scores
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandboxes = [_sandbox(monkeypatch, _result(scores)) for _ in range(failures + 1)]
    for sandbox in sandboxes[:-1]:
        operation = sandbox.warmup_run if phase == "start" else sandbox.warmup_handle.wait
        operation.side_effect = asyncio.CancelledError()
    create = _use_sandbox_sequence(monkeypatch, sandboxes)
    task = _task()
    original = copy.deepcopy(task)

    score = _score(task)

    assert task == original
    assert score["reward"] == scores[0]
    assert score["metrics"]["warmup_retry_count"] == failures
    assert score["metrics"]["warmup_failure_count"] == failures
    assert create.await_count == failures + 1
    assert warmup_backoff.sleep.await_count == failures
    assert warmup_backoff.delay.call_args_list == [((1.0, 3.0),)] * failures
    payloads = [sandbox.files.write.await_args.args[1] for sandbox in sandboxes]
    assert all(payload == payloads[0] for payload in payloads)
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()
    for sandbox in sandboxes[:-1]:
        sandbox.judge_run.assert_not_awaited()
        if phase == "wait":
            sandbox.warmup_handle.disconnect.assert_awaited_once()
    sandboxes[-1].judge_run.assert_awaited_once()
    cancellations = _batch_events(caplog, "warmup_cancelled")
    assert len(cancellations) == failures
    assert all(fields["caller_cancelling"] == 0 for fields in cancellations)
    assert all(fields["cancellation_kind"] == "internal_warmup" for fields in cancellations)
    assert all(fields["owned_warmup_cancelled"] for fields in cancellations)
    assert not any(fields["warmup_abort_requested"] for fields in cancellations)


def test_owned_warmup_cancellation_exhausts_existing_budget_without_zero_score(
    monkeypatch, caplog, warmup_backoff
):
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    for sandbox in sandboxes:
        sandbox.warmup_handle.wait.side_effect = asyncio.CancelledError()
    create = _use_sandbox_sequence(monkeypatch, sandboxes)

    with pytest.raises(code_e2b.CodeE2BError) as error:
        _score()

    assert error.value.stage == "warmup"
    assert error.value.error_type == "_UnexpectedWarmupCancellation"
    assert error.value.retryable_warmup is True
    assert not error.value.is_timeout
    assert create.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()
        sandbox.judge_run.assert_not_awaited()
    exhausted = next(
        record.code_e2b_retry
        for record in caplog.records
        if getattr(record, "code_e2b_retry", {}).get("event") == "warmup_exhausted"
    )
    assert exhausted["warmup_failure_count"] == 3
    assert exhausted["warmup_retry_count"] == 2


def test_internal_cancellation_and_warmup_timeout_share_failure_budget(monkeypatch, warmup_backoff):
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    sandboxes[0].warmup_handle.wait.side_effect = asyncio.CancelledError()
    sandboxes[1].warmup_run.side_effect = TimeoutError("warmup timeout")
    sandboxes[2].warmup_handle.wait.side_effect = asyncio.CancelledError()
    create = _use_sandbox_sequence(monkeypatch, sandboxes)

    with pytest.raises(code_e2b.CodeE2BError):
        _score()

    assert create.await_count == 3
    assert warmup_backoff.sleep.await_count == 2


def test_internal_warmup_cancel_does_not_consume_other_stage_budgets(monkeypatch, warmup_backoff):
    # Setup stages have independent recovery budgets; judging keeps its one retry.
    events = ["create", "create", "upload", "upload", "warmup", "run", "warmup", "pass"]
    sandboxes = []
    for event in events:
        if event == "create":
            sandboxes.append(TimeoutError("create timeout"))
            continue
        sandbox = _sandbox(monkeypatch)
        if event == "upload":
            sandbox.files.write.side_effect = TimeoutError("upload timeout")
        elif event == "warmup":
            sandbox.warmup_handle.wait.side_effect = asyncio.CancelledError()
        elif event == "run":
            sandbox.judge_run.side_effect = e2b.TimeoutException("judge timeout")
        sandboxes.append(sandbox)
    create = _use_sandbox_sequence(monkeypatch, sandboxes)

    score = _score()

    assert score["reward"] == 1.0
    assert score["metrics"]["create_retry_count"] == 2
    assert score["metrics"]["upload_retry_count"] == 2
    assert score["metrics"]["warmup_retry_count"] == 2
    assert score["metrics"]["timeout_count"] == 1
    assert score["metrics"]["timeout_batches"] == 0
    assert create.await_count == 8


def test_graceful_worker_stop_is_not_mistaken_for_internal_cancellation(
    monkeypatch, caplog, warmup_backoff
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    stop = Event()

    async def stop_during_wait():
        stop.set()
        raise asyncio.CancelledError()

    sandbox.warmup_handle.wait.side_effect = stop_during_wait
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.judge_run.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()
    fields = _batch_events(caplog, "warmup_cancelled")[0]
    assert fields["caller_cancelling"] == 0
    assert fields["worker_stop_requested"] is True
    assert fields["cancellation_kind"] == "propagated"
    assert not reward_stop_requested()  # The context must also reset on errors.


@pytest.mark.parametrize("stage", ["create", "upload", "warmup", "run"])
def test_worker_stop_prevents_retry_after_any_stage_failure(monkeypatch, warmup_backoff, stage):
    sandbox = _sandbox(monkeypatch)
    stop = Event()

    async def stopped(*_args, **_kwargs):
        stop.set()
        raise TimeoutError("stage timed out during graceful stop")

    operation = {
        "create": sandbox.create,
        "upload": sandbox.files.write,
        "warmup": sandbox.warmup_handle.wait,
        "run": sandbox.judge_run,
    }[stage]
    operation.side_effect = stopped

    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()
    if stage != "create":
        sandbox.kill.assert_awaited_once()


def test_worker_stop_during_backoff_prevents_new_sandbox(monkeypatch, warmup_backoff):
    first = _sandbox(monkeypatch)
    retry = _sandbox(monkeypatch)
    first.warmup_handle.wait.side_effect = asyncio.CancelledError()
    create = _use_sandbox_sequence(monkeypatch, [first, retry])
    stop = Event()

    async def stop_in_backoff(_delay):
        stop.set()

    warmup_backoff.sleep.side_effect = stop_in_backoff
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _score()

    assert create.await_count == 1
    first.kill.assert_awaited_once()
    retry.judge_run.assert_not_awaited()


def test_worker_already_stopping_creates_no_sandbox(no_real_e2b):
    stop = Event()
    stop.set()
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _score()
    no_real_e2b.assert_not_awaited()


def test_attempt_expiry_prevents_internal_cancel_reclassification(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    sandbox.warmup_handle.wait.side_effect = asyncio.CancelledError()
    state = code_e2b._WarmupState()
    batch = code_e2b._build_request(_task(), CODE)
    diagnostics = code_e2b._BatchDiagnostics(batch, json.dumps(batch), None)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await code_e2b._await_warmup(
                sandbox, state, diagnostics, 30, SimpleNamespace(expired=lambda: True)
            )
        await code_e2b._disconnect_warmup(state, diagnostics)

    asyncio.run(run())
    fields = _batch_events(caplog, "warmup_cancelled")[0]
    assert fields["caller_cancelling"] == 0
    assert fields["attempt_deadline_expired"] is True
    assert fields["cancellation_kind"] == "propagated"


def test_unresolved_warmup_task_blocks_retry_and_releases_late_handle(monkeypatch, caplog):
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setattr(code_e2b, "_COMMAND_WARMUP_TIMEOUT_S", 0.01)
    monkeypatch.setattr(code_e2b, "_WARMUP_CLEANUP_TIMEOUT_S", 0.01)
    late_state = {}

    async def run():
        release = asyncio.Event()

        async def late_start(command, **_kwargs):
            assert command == "true"
            late_state["task"] = asyncio.current_task()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Simulate an SDK transport that delays cancellation completion.
                await release.wait()
                return sandbox.warmup_handle

        sandbox.commands.run.side_effect = late_start
        try:
            with pytest.raises(code_e2b.CodeE2BError) as error:
                await code_e2b.code_e2b_reward(_task(), {"response": CODE})
            assert error.value.warmup_cleanup_incomplete is True
            sandbox.create.assert_awaited_once()
            sandbox.kill.assert_awaited_once()
        finally:
            release.set()
            await asyncio.gather(late_state["task"], return_exceptions=True)

    asyncio.run(run())
    sandbox.warmup_handle.wait.assert_not_awaited()
    sandbox.warmup_handle.disconnect.assert_awaited_once()
    assert _batch_events(caplog, "warmup_cleanup_incomplete")


def test_retry_loop_rejects_incomplete_cleanup_even_with_retryable_flag(monkeypatch):
    error = code_e2b.CodeE2BError(
        "warmup",
        code_e2b._UnexpectedWarmupCancellation("cancelled"),
        retryable_warmup=True,
        warmup_cleanup_incomplete=True,
    )
    run = AsyncMock(side_effect=error)
    monkeypatch.setattr(code_e2b, "_run_batch", run)
    with pytest.raises(code_e2b.CodeE2BError) as caught:
        _score()
    assert caught.value is error
    run.assert_awaited_once()


def test_internal_cancellation_retries_only_failed_batch(monkeypatch, warmup_backoff):
    monkeypatch.setattr(code_e2b, "_BATCH_CASES", 1)
    sandboxes = [_sandbox(monkeypatch, _result((1.0,))) for _ in range(3)]
    sandboxes[1].warmup_handle.wait.side_effect = asyncio.CancelledError()
    create = _use_sandbox_sequence(monkeypatch, sandboxes)

    score = _score()

    assert score["reward"] == 1.0
    assert score["metrics"]["completed_batches"] == 2
    assert score["metrics"]["warmup_retry_count"] == 1
    assert create.await_count == 3
    uploaded = [json.loads(sandbox.files.write.await_args.args[1]) for sandbox in sandboxes]
    assert uploaded[0]["cases"] == CASES[:1]
    assert uploaded[1]["cases"] == uploaded[2]["cases"] == CASES[1:]
    assert [sandbox.judge_run.await_count for sandbox in sandboxes] == [1, 0, 1]


def test_internal_cancellation_budget_is_separate_for_each_answer(monkeypatch, warmup_backoff):
    sandboxes = [_sandbox(monkeypatch) for _ in range(6)]
    for index in (0, 1, 3, 4):
        sandboxes[index].warmup_handle.wait.side_effect = asyncio.CancelledError()
    create = _use_sandbox_sequence(monkeypatch, sandboxes)
    task = {**_task(), "task_id": 123, "group_id": "same-prompt"}

    for _ in range(2):
        score = _score(task)
        assert score["reward"] == 1.0
        assert score["metrics"]["warmup_retry_count"] == 2
    assert create.await_count == 6


def test_caller_cancel_and_child_cancel_race_never_retries(monkeypatch, caplog, warmup_backoff):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    sandbox = _sandbox(monkeypatch)
    caller = None

    async def capture_caller(*_args, **_kwargs):
        nonlocal caller
        caller = asyncio.current_task()

    async def simultaneous_cancel():
        caller.cancel()
        raise asyncio.CancelledError()

    sandbox.files.write.side_effect = capture_caller
    sandbox.warmup_handle.wait.side_effect = simultaneous_cancel

    with pytest.raises(asyncio.CancelledError):
        _score()

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.judge_run.assert_not_awaited()
    warmup_backoff.sleep.assert_not_awaited()
    fields = _batch_events(caplog, "warmup_cancelled")[0]
    assert fields["caller_cancelling"] == 1
    assert fields["cancellation_kind"] == "propagated"


def _shared_task(monkeypatch, cases=None):
    task = _task(cases)
    digest = "a" * 64
    reader = "b" * 64
    task["metadata"] = {
        "cases_ref": {
            "protocol_version": 1,
            "manifest_path": f"{code_e2b._VEPFS_ROOT}/manifests/{digest}.json",
            "manifest_sha256": digest,
            "case_count": len(task["label"]["cases"]),
        }
    }
    monkeypatch.setenv("E2B_VEPFS_VOLUME", "test-cases-v1")
    monkeypatch.setenv("E2B_VEPFS_READER_PATH", f"{code_e2b._VEPFS_ROOT}/readers/{reader}.py")
    monkeypatch.setenv("E2B_VEPFS_READER_SHA256", reader)
    return task


def test_shared_entire_task_uses_one_sandbox_without_uploading_cases(monkeypatch):
    cases = [{"stdin": str(i), "expected_stdout": str(i)} for i in range(205)]
    task = _shared_task(monkeypatch, cases)
    before = copy.deepcopy(task)
    sandbox = _sandbox(monkeypatch, _result([1.0] * len(cases)))
    monkeypatch.setattr(
        code_e2b, "_iter_request_batches_async", Mock(side_effect=AssertionError("legacy batching"))
    )
    score = _score(task)
    assert score["reward"] == 1
    assert len(score["sandbox_result"]["tests"]) == 205
    assert score["metrics"]["completed_batches"] == 1
    sandbox.create.assert_awaited_once()
    assert sandbox.create.call_args.kwargs["volume_mounts"] == {"/mnt/data": "test-cases-v1"}
    sandbox.files.write.assert_not_awaited()
    assert "hashlib.sha256" in sandbox.judge_run.call_args.args[0]
    assert task == before
    sandbox.kill.assert_awaited_once()


def test_shared_large_source_uploads_only_descriptor(monkeypatch):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    response = "#" + "x" * 70000 + "\nprint(input())"
    score = _score(task, response)
    assert score["reward"] == 1
    payload = json.loads(sandbox.files.write.call_args.args[1])
    assert payload["code"] == response
    assert payload["reference"] == task["metadata"]["cases_ref"]
    assert "cases" not in payload
    assert "expected_stdout" not in sandbox.files.write.call_args.args[1].decode()


@pytest.mark.parametrize("change", ["count", "hash", "path", "version", "config"])
def test_shared_invalid_reference_never_falls_back_to_upload(monkeypatch, no_real_e2b, change):
    task = _shared_task(monkeypatch)
    ref = task["metadata"]["cases_ref"]
    if change == "count":
        ref["case_count"] += 1
    if change == "hash":
        ref["manifest_sha256"] = "bad"
    if change == "path":
        ref["manifest_path"] = "/etc/passwd"
    if change == "version":
        ref["protocol_version"] = True
    if change == "config":
        monkeypatch.delenv("E2B_VEPFS_VOLUME")
    with pytest.raises(ValueError):
        _score(task)
    no_real_e2b.assert_not_awaited()


def test_shared_header_and_code_are_bound_without_serializing_cases(monkeypatch):
    task = _shared_task(monkeypatch)
    task["label"].update(run_timeout=3.0, code_template="exec(__MODEL_CODE__)")
    request = code_e2b._build_request(task, "print(1)")
    shared = code_e2b._prepare_vepfs_request(task, request)
    descriptor = json.loads(shared.payload)
    expected_header = {
        k: v for k, v in task["label"].items() if k not in {"cases", "code_template"}
    }
    assert (
        descriptor["request_header_sha256"]
        == hashlib.sha256(code_e2b._canonical_json_bytes(expected_header)).hexdigest()
    )
    assert descriptor["code"] == "exec('print(1)')"
    assert "cases" not in descriptor


@pytest.mark.parametrize("padding", [0, 70000], ids=["inline", "uploaded-descriptor"])
@pytest.mark.parametrize("dataset", ["livecodebench_v6", "taco"])
def test_python_prelude_shared_transport_keeps_reference_and_header(monkeypatch, padding, dataset):
    task = _shared_task(monkeypatch)
    task["metadata"].update(_prelude_task(dataset)["metadata"])
    source = "#" + "x" * padding + "\nprint(input())"
    before = copy.deepcopy(task)
    request = code_e2b._build_request(task, source)
    shared = code_e2b._prepare_vepfs_request(task, request)
    descriptor = json.loads(shared.payload)
    expected_header = {k: v for k, v in task["label"].items() if k != "cases"}
    assert (
        descriptor["request_header_sha256"]
        == hashlib.sha256(code_e2b._canonical_json_bytes(expected_header)).hexdigest()
    )
    assert descriptor["code"] == request["code"] != source
    assert descriptor["reference"] == before["metadata"]["cases_ref"]
    assert "cases" not in descriptor
    command, needs_upload = shared.command(code_e2b._REQUEST_PATH)
    assert needs_upload is bool(padding)
    sandbox = _sandbox(monkeypatch)
    assert _score(task, response=source)["reward"] == 1
    assert sandbox.judge_run.call_args.args[0] == command
    if needs_upload:
        assert sandbox.files.write.await_args.args[1] == shared.payload
    else:
        sandbox.files.write.assert_not_awaited()
    assert task == before


def test_shared_timeout_estimate_does_not_cap_whole_task_at_one_hour(monkeypatch):
    task = _shared_task(monkeypatch, CASES * 500)
    request = code_e2b._build_request(task, CODE)
    shared = code_e2b._timeouts(request, whole_task=True)
    legacy = code_e2b._timeouts(request)
    assert shared[0] == shared[4] > 3600
    assert legacy[0] < 3600


def test_shared_transport_failure_retries_then_raises_instead_of_scoring_zero(
    monkeypatch, warmup_backoff
):
    task = _shared_task(monkeypatch)
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    for sandbox in sandboxes:
        sandbox.judge_run.side_effect = httpx.ReadError("disconnected")
    create = _use_sandbox_sequence(monkeypatch, sandboxes)
    with pytest.raises(code_e2b.CodeE2BError):
        _score(task)
    assert create.await_count == 3
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()
    assert warmup_backoff.sleep.await_count == 2


@pytest.mark.parametrize("timeout_kind", ["local_deadline", "sdk_timeout", "http_timeout"])
def test_shared_three_run_timeouts_return_zero_without_inventing_case_results(
    monkeypatch, warmup_backoff, caplog, timeout_kind
):
    task = _shared_task(monkeypatch)
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    create = _use_sandbox_sequence(monkeypatch, sandboxes)
    if timeout_kind == "local_deadline":
        monkeypatch.setenv("E2B_COMMAND_TIMEOUT", "0.01")

        async def hanging_judge(*_args, **kwargs):
            assert kwargs["timeout"] == 0.01
            assert kwargs["request_timeout"] == 0
            await asyncio.Event().wait()

        for sandbox in sandboxes:
            sandbox.judge_run.side_effect = hanging_judge
    else:
        timeout_type = e2b.TimeoutException if timeout_kind == "sdk_timeout" else httpx.ReadTimeout
        for sandbox in sandboxes:
            sandbox.judge_run.side_effect = timeout_type("judge command timed out")

    score = _score(task, deadline=2)

    assert score == {
        "status": "ok",
        "reward": 0.0,
        "correct": False,
        "reason": "timeout_after_retry",
        "timed_out": True,
        "sandbox_result": {"accepted": False, "score": None, "tests": []},
        "metrics": {
            "passed_cases": 0.0,
            "total_cases": 2.0,
            "completed_batches": 0.0,
            "timeout_count": 3.0,
            "retry_count": 2.0,
            "timeout_batches": 1.0,
            "evaluated_cases": 0.0,
            "unevaluated_cases": 2.0,
            "run_failure_count": 3.0,
            "run_retry_count": 2.0,
        },
    }
    assert create.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    for sandbox in sandboxes:
        sandbox.judge_run.assert_awaited_once()
        sandbox.is_running.assert_awaited_once()
        sandbox.kill.assert_awaited_once()
    failures = _batch_events(caplog, "failure")
    assert [failure["attempt"] for failure in failures] == [1, 2, 3]
    assert all(not failure["attempt_deadline_expired"] for failure in failures)
    if timeout_kind == "local_deadline":
        assert all(failure["command_time_remaining_s"] == 0 for failure in failures)


def test_shared_per_case_tle_is_a_verdict_not_a_command_retry(monkeypatch, warmup_backoff):
    task = _shared_task(monkeypatch)
    response = _result([0.0, 1.0])
    response["tests"][0]["exec_info"] = {"run_result": {"status": "TimeLimitExceeded"}}
    sandbox = _sandbox(monkeypatch, response)
    score = _score(task)
    assert score["reward"] == 0
    assert len(score["sandbox_result"]["tests"]) == 2
    sandbox.create.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("failure_stage", ["warmup", "run"])
def test_shared_kill_failure_does_not_block_retry_of_original_failure(
    monkeypatch, warmup_backoff, caplog, failure_stage
):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    if failure_stage == "warmup":
        sandbox.warmup_run.side_effect = httpx.ReadError("warmup disconnected")
    elif failure_stage == "run":
        sandbox.judge_run.side_effect = e2b.TimeoutException("whole-task deadline")
    sandbox.kill.side_effect = httpx.ReadError("kill disconnected test-api-key")
    retry = _sandbox(monkeypatch)
    create = _use_sandbox_sequence(monkeypatch, [sandbox, retry])
    assert _score(task)["reward"] == 1
    assert create.await_count == 2
    sandbox.kill.assert_awaited_once()
    retry.kill.assert_awaited_once()
    warmup_backoff.sleep.assert_awaited_once()
    assert _batch_events(caplog, "cleanup_unconfirmed")[0]["action"] == "preserve_original_error"
    assert "test-api-key" not in caplog.text


@pytest.mark.parametrize("cancel_stage", ["run", "warmup_cleanup", "kill"])
def test_shared_cleanup_failure_never_masks_external_cancellation(
    monkeypatch, warmup_backoff, cancel_stage
):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    sandbox.kill.side_effect = httpx.ReadError("kill disconnected")
    if cancel_stage == "run":
        sandbox.judge_run.side_effect = asyncio.CancelledError("caller stopped")
    elif cancel_stage == "warmup_cleanup":
        monkeypatch.setattr(
            code_e2b,
            "_disconnect_warmup",
            AsyncMock(side_effect=asyncio.CancelledError("caller stopped during cleanup")),
        )
    else:
        sandbox.kill.side_effect = asyncio.CancelledError("caller stopped during kill")

    with pytest.raises(asyncio.CancelledError):
        _score(task)

    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


def test_shared_retry_guard_allows_incomplete_sandbox_cleanup_but_stays_bounded(
    monkeypatch, warmup_backoff
):
    task = _shared_task(monkeypatch)
    error = code_e2b.CodeE2BError(
        "run",
        e2b.TimeoutException("whole-task deadline"),
        is_timeout=True,
        sandbox_cleanup_incomplete=True,
    )
    run = AsyncMock(side_effect=error)
    monkeypatch.setattr(code_e2b, "_run_batch", run)
    score = _score(task)
    assert score["reward"] == 0.0
    assert score["reason"] == "timeout_after_retry"
    assert score["timed_out"] is True
    assert score["sandbox_result"] == {"accepted": False, "score": None, "tests": []}
    assert score["metrics"]["timeout_count"] == 3
    assert score["metrics"]["run_failure_count"] == 3
    assert score["metrics"]["run_retry_count"] == 2
    assert run.await_count == 3
    assert warmup_backoff.sleep.await_count == 2


def test_shared_cleanup_failure_preserves_verdict_inside_callers_handled_exception(monkeypatch):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    sandbox.kill.side_effect = httpx.ReadError("kill disconnected")
    try:
        raise ValueError("previous operation failed")
    except ValueError:
        assert _score(task)["reward"] == 1
    assert sandbox.kill.await_count == 2
    sandbox.judge_run.assert_awaited_once()


@pytest.mark.parametrize("scores", [(1.0, 1.0), (0.0, 0.0), (1.0, 0.5)])
@pytest.mark.parametrize("cleanup", ["recovers", "exhausted", "absent", "times_out"])
def test_valid_shared_verdict_survives_at_most_one_extra_cleanup(
    monkeypatch, warmup_backoff, caplog, scores, cleanup
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    task = _shared_task(monkeypatch)
    original = copy.deepcopy(task)
    sandbox = _sandbox(monkeypatch, _result(scores))
    if cleanup == "recovers":
        sandbox.kill.side_effect = [httpx.ReadError("kill failed test-api-key"), None]
    elif cleanup == "absent":
        sandbox.kill.side_effect = e2b.SandboxNotFoundException("already absent")
    elif cleanup == "times_out":
        monkeypatch.setattr(code_e2b, "_SANDBOX_CLEANUP_TIMEOUT_S", 0.01)

        async def hang(**_kwargs):
            await asyncio.Event().wait()

        sandbox.kill.side_effect = hang
    else:
        sandbox.kill.side_effect = httpx.ReadError("kill failed test-api-key")
    score = _score(task, deadline=2)
    assert task == original
    assert score["sandbox_result"] == _result(scores)
    assert score["reward"] == float(all(value == 1 for value in scores))
    sandbox.create.assert_awaited_once()
    sandbox.judge_run.assert_awaited_once()
    assert sandbox.kill.await_count == (1 if cleanup == "absent" else 2)
    warmup_backoff.sleep.assert_not_awaited()
    assert all(
        c.kwargs["request_timeout"] <= code_e2b._SANDBOX_CLEANUP_TIMEOUT_S
        for c in sandbox.kill.await_args_list
    )
    unconfirmed = _batch_events(caplog, "cleanup_unconfirmed")
    assert bool(unconfirmed) == (cleanup in {"exhausted", "times_out"})
    if unconfirmed:
        assert unconfirmed[0]["cleanup_confirmed"] is False
        assert unconfirmed[0]["action"] == "return_verdict"
        assert unconfirmed[0]["cleanup_attempts"] == 2
    assert "test-api-key" not in caplog.text


@pytest.mark.parametrize("cancel", ["task", "worker_stop"])
def test_cancellation_before_second_cleanup_never_retries_or_returns_verdict(monkeypatch, cancel):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    stop = Event()

    async def fail_cleanup(**_kwargs):
        if cancel == "task":
            asyncio.current_task().cancel()
        else:
            stop.set()
        raise httpx.ReadError("kill failed while stopping")

    sandbox.kill.side_effect = fail_cleanup
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _score(task)
    sandbox.create.assert_awaited_once()
    sandbox.judge_run.assert_awaited_once()
    sandbox.kill.assert_awaited_once()


def test_cancellation_during_second_cleanup_propagates(monkeypatch):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    sandbox.kill.side_effect = [httpx.ReadError("first kill failed"), asyncio.CancelledError()]
    with pytest.raises(asyncio.CancelledError):
        _score(task)
    assert sandbox.kill.await_count == 2
    sandbox.judge_run.assert_awaited_once()


@pytest.mark.parametrize("failure_stage", ["upload", "warmup", "run"])
def test_shared_third_attempt_recovers_despite_unconfirmed_failed_sandbox_cleanup(
    monkeypatch, warmup_backoff, failure_stage
):
    task = _shared_task(monkeypatch)
    response = CODE if failure_stage != "upload" else "#" + "x" * 70000 + "\n" + CODE
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    for sandbox in sandboxes[:2]:
        operation = {
            "upload": sandbox.files.write,
            "warmup": sandbox.warmup_run,
            "run": sandbox.judge_run,
        }[failure_stage]
        operation.side_effect = e2b.SandboxException("503: temporary service error")
        sandbox.kill.side_effect = httpx.ReadError("cleanup unconfirmed")
    create = _use_sandbox_sequence(monkeypatch, sandboxes)
    original = copy.deepcopy(task)
    score = _score(task, response)
    assert task == original
    assert score["reward"] == 1
    assert create.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    assert score["metrics"][failure_stage + "_retry_count"] == 2
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()
    if failure_stage == "run":
        assert score["metrics"]["retry_count"] == 2
        commands = [sandbox.judge_run.await_args.args[0] for sandbox in sandboxes]
        assert commands == [commands[0]] * 3
    assert all(
        c.kwargs["timeout"] == create.await_args_list[0].kwargs["timeout"]
        for c in create.await_args_list
    )
    if failure_stage == "upload":
        assert [
            sandbox.files.write.await_args.kwargs["request_timeout"] for sandbox in sandboxes
        ] == [300, 450, 600]


def test_shared_run_timeouts_and_transport_failures_share_budget_and_report_real_retries(
    monkeypatch, warmup_backoff
):
    task = _shared_task(monkeypatch)
    first, second, last = [_sandbox(monkeypatch) for _ in range(3)]
    first.judge_run.side_effect = e2b.SandboxException("503: unavailable")
    second.judge_run.side_effect = e2b.TimeoutException("judge timeout")
    create = _use_sandbox_sequence(monkeypatch, [first, second, last])
    score = _score(task)
    assert create.await_count == 3
    assert score["metrics"]["run_failure_count"] == 2
    assert score["metrics"]["run_retry_count"] == 2
    assert score["metrics"]["timeout_count"] == 1
    assert score["metrics"]["retry_count"] == 2


@pytest.mark.parametrize("terminal_timeout", [False, True])
@pytest.mark.parametrize("non_timeout_kind", ["transport", "command_exit"])
def test_shared_mixed_run_failures_only_score_zero_when_final_failure_is_timeout(
    monkeypatch, warmup_backoff, terminal_timeout, non_timeout_kind
):
    task = _shared_task(monkeypatch)
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    non_timeout = (
        httpx.ReadError("command stream disconnected")
        if non_timeout_kind == "transport"
        else e2b.CommandExitException(exit_code=1, stdout="", stderr="evaluator failed", error=None)
    )
    errors = [e2b.TimeoutException("judge command timed out") for _ in range(2)]
    errors.insert(0 if terminal_timeout else 2, non_timeout)
    for sandbox, error in zip(sandboxes, errors):
        sandbox.judge_run.side_effect = error
    create = _use_sandbox_sequence(monkeypatch, sandboxes)

    if terminal_timeout:
        score = _score(task)
        assert score["status"] == "ok"
        assert score["reward"] == 0.0
        assert score["correct"] is False
        assert score["reason"] == "timeout_after_retry"
        assert score["timed_out"] is True
        assert score["sandbox_result"] == {"accepted": False, "score": None, "tests": []}
        assert score["metrics"] == {
            "passed_cases": 0.0,
            "total_cases": 2.0,
            "completed_batches": 0.0,
            "timeout_count": 2.0,
            "retry_count": 2.0,
            "timeout_batches": 1.0,
            "evaluated_cases": 0.0,
            "unevaluated_cases": 2.0,
            "run_failure_count": 3.0,
            "run_retry_count": 2.0,
        }
    else:
        with pytest.raises(code_e2b.CodeE2BError) as caught:
            _score(task)
        assert caught.value.error_type == type(non_timeout).__name__
        assert caught.value.retryable_run
        assert not caught.value.is_timeout
    assert create.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize("guard", ["attempt_deadline_expired", "warmup_cleanup_incomplete"])
def test_shared_cleanup_relaxation_never_bypasses_other_retry_guards(
    monkeypatch, warmup_backoff, guard
):
    task = _shared_task(monkeypatch)
    error = code_e2b.CodeE2BError(
        "run", TimeoutError(), is_timeout=True, sandbox_cleanup_incomplete=True, **{guard: True}
    )
    run = AsyncMock(side_effect=error)
    monkeypatch.setattr(code_e2b, "_run_batch", run)
    with pytest.raises(code_e2b.CodeE2BError) as caught:
        _score(task)
    assert caught.value is error
    run.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


def test_invalid_shared_verdict_with_failed_cleanup_preserves_protocol_error(
    monkeypatch, warmup_backoff
):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch, stdout="{truncated-json")
    sandbox.kill.side_effect = httpx.ReadError("kill unconfirmed")
    with pytest.raises(code_e2b.CodeE2BError) as caught:
        _score(task)
    assert caught.value.stage == "parse"
    assert caught.value.sandbox_cleanup_incomplete
    sandbox.create.assert_awaited_once()
    sandbox.judge_run.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("stage", ["warmup", "run"])
def test_actual_shared_attempt_expiry_with_cleanup_failure_never_retries(
    monkeypatch, warmup_backoff, stage
):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    monkeypatch.setattr(code_e2b, "_timeouts", lambda *_args, **_kwargs: (60, 60, 300, 0.02, 60))

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    operation = sandbox.warmup_run if stage == "warmup" else sandbox.judge_run
    operation.side_effect = hang
    sandbox.kill.side_effect = httpx.ReadError("kill unconfirmed")
    with pytest.raises(code_e2b.CodeE2BError) as caught:
        _score(task, deadline=2)
    assert caught.value.stage == stage
    assert caught.value.attempt_deadline_expired
    assert caught.value.sandbox_cleanup_incomplete
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("stage", ["create", "upload", "warmup", "run"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504, 599])
@pytest.mark.parametrize("structured", [False, True])
def test_retryable_http_statuses_are_explicit_and_stage_bounded(stage, status, structured):
    error = (
        httpx.HTTPStatusError(
            "request failed",
            request=httpx.Request("POST", "https://e2b.test"),
            response=httpx.Response(status),
        )
        if structured
        else e2b.SandboxException(f"{status}: service response")
    )
    assert (code_e2b._e2b_retry_reason(error, stage) is not None) == (
        status in (408, 429) or status >= 500
    )
    assert code_e2b._e2b_retry_reason(error, "parse") is None


@pytest.mark.parametrize("stage", ["create", "upload", "warmup", "run"])
def test_rate_limit_and_missing_sandbox_are_classified_by_resource_and_stage(stage):
    # SDK 2.50 also exposes the structured HTTP 429 status on this exception.
    assert code_e2b._e2b_retry_reason(e2b.RateLimitException("limit"), stage) in {
        "rate_limit",
        "http_429",
    }
    assert (
        code_e2b._e2b_retry_reason(e2b.SandboxNotFoundException("gone"), stage) is not None
    ) == (stage != "create")
    for error in (
        e2b.FileNotFoundException("503: file missing"),
        e2b.TemplateException("503: incompatible"),
    ):
        assert code_e2b._e2b_retry_reason(error, stage) is None


def test_rpc_service_codes_retry_without_cause_but_permanent_codes_never_retry():
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    allowed = {
        Code.UNAVAILABLE,
        Code.INTERNAL,
        Code.DEADLINE_EXCEEDED,
        Code.RESOURCE_EXHAUSTED,
        Code.ABORTED,
        Code.UNKNOWN,
    }
    for code in Code:
        error = ConnectError(code, "service error")
        for stage in ("create", "upload", "warmup", "run"):
            assert (code_e2b._e2b_retry_reason(error, stage) is not None) == (code in allowed)
        if code not in allowed:
            error.__cause__ = httpx.ReadError("network cause cannot override RPC status")
            assert code_e2b._e2b_retry_reason(error, "run") is None


@pytest.mark.parametrize("stage", ["create", "upload", "warmup", "run"])
def test_positive_http_status_never_overrides_certificate_or_protocol_error(stage):
    for cause in (ssl.SSLCertVerificationError("invalid certificate"), ValueError("invalid data")):
        error = e2b.SandboxException("503: overloaded")
        error.__cause__ = cause
        assert code_e2b._e2b_retry_reason(error, stage) is None
    assert code_e2b._e2b_retry_reason(RuntimeError("503: unrelated text"), stage) is None


def test_handled_caller_exception_cannot_change_retry_classification():
    try:
        raise ValueError("old operation")
    except ValueError:
        try:
            raise httpx.ReadError("new connection failure")
        except httpx.ReadError as error:
            assert code_e2b._e2b_retry_reason(error, "upload") == "transport"
    try:
        raise httpx.ReadError("old connection failure")
    except httpx.ReadError:
        try:
            raise RuntimeError("new unrelated failure")
        except RuntimeError as error:
            assert code_e2b._e2b_retry_reason(error, "run") is None


@pytest.mark.parametrize("code_size", [20, 70000])
def test_bootstrap_delivers_binary_stdin_without_executing_candidate(tmp_path, code_size):
    reader = tmp_path / "reader.py"
    reader.write_text("import json,sys; print(json.dumps(json.load(sys.stdin.buffer)))")
    payload = json.dumps({"code": "x" * code_size, "unicode": "测试"}).encode()
    shared = code_e2b._VepfsRequest(
        "volume", str(reader), hashlib.sha256(reader.read_bytes()).hexdigest(), payload
    )
    request = tmp_path / "request.json"
    command, upload = shared.command(str(request))
    assert upload == (code_size == 70000)
    if upload:
        request.write_bytes(payload)
    argv = shlex.split(command)
    argv[0] = sys.executable
    result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout) == json.loads(payload)


@pytest.mark.parametrize("tamper", ["hash", "symlink"])
def test_bootstrap_rejects_tampered_reader_before_execution(tmp_path, tamper):
    marker = tmp_path / "executed"
    source = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    reader = tmp_path / "reader.py"
    reader.write_text(source)
    digest = hashlib.sha256(reader.read_bytes()).hexdigest()
    if tamper == "hash":
        reader.write_text(source + "\n# changed")
    else:
        real = tmp_path / "real.py"
        reader.rename(real)
        reader.symlink_to(real)
    shared = code_e2b._VepfsRequest("volume", str(reader), digest, b"{}")
    command, upload = shared.command(str(tmp_path / "request.json"))
    assert not upload
    argv = shlex.split(command)
    argv[0] = sys.executable
    result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == code_e2b._VEPFS_BOOTSTRAP_CONFIG_EXIT
    assert not marker.exists()


@pytest.mark.parametrize("returned", [False, True])
def test_permanent_command_error_logs_bounded_tails_without_retry(
    monkeypatch, warmup_backoff, caplog, returned
):
    caplog.set_level(logging.INFO, logger=code_e2b.__name__)
    task = _shared_task(monkeypatch)
    first = _sandbox(monkeypatch)
    secret = "sk-diagnostic-secret"
    monkeypatch.setenv("DIAGNOSTIC_API_KEY", secret)
    message = f"unexpected reader failure {secret} https://u:password@host.test/path?token=secret"
    stdout = json.dumps({"error": {"type": "ValueError", "message": message}})
    stderr = "DEBUG: preview\n" * 3000 + "末尾错误: " + message
    result = CommandResult(exit_code=1, stdout=stdout, stderr=stderr, error=message)
    if returned:
        first.judge_run.return_value = result
    else:
        first.judge_run.side_effect = e2b.CommandExitException(**vars(result))
    with pytest.raises(code_e2b.CodeE2BError) as caught:
        _score(task)
    assert not caught.value.retryable_run
    first.create.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()
    failures = _batch_events(caplog, "failure")
    assert len(failures) == 1
    fields = failures[0]
    assert fields["command_exit_code"] == 1
    assert "judge_error_type" not in fields
    assert "judge_error_message" not in fields
    assert fields["stdout_chars"] == len(stdout)
    assert fields["stderr_chars"] == len(stderr)
    assert len(fields["stderr_tail"].encode("utf-8")) <= 1024
    assert len(fields["stdout_tail"].encode("utf-8")) <= 512
    assert "末尾错误" in fields["stderr_tail"]
    assert "DEBUG: preview" not in fields["error"]
    for sensitive in [secret, "password", "token=secret", "/path?"]:
        assert sensitive not in caplog.text


def test_diagnostic_handles_surrogates_and_bounds_multibyte_output():
    error = e2b.CommandExitException(
        exit_code=1,
        stdout="输出" * 1000 + "\ud800",
        stderr="错误" * 1000 + "\ud800",
        error=None,
    )
    details = code_e2b._command_failure_details(error)
    assert len(details["stdout_tail"].encode("utf-8")) <= 512
    assert len(details["stderr_tail"].encode("utf-8")) <= 1024
    assert details["stdout_tail"].endswith("?")
    assert details["stderr_tail"].endswith("?")


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_diagnostic_redacts_plain_and_json_escaped_credentials(monkeypatch, ensure_ascii):
    secret = 'dummy-quote"backslash\\newline\n中文-password'
    monkeypatch.setenv("DIAGNOSTIC_PASSWORD", secret)
    error = e2b.CommandExitException(
        exit_code=1,
        stdout=json.dumps({"error": {"message": secret}}, ensure_ascii=ensure_ascii),
        stderr=secret,
        error=None,
    )
    details = code_e2b._command_failure_details(error)
    assert json.loads(details["stdout_tail"])["error"]["message"] == "[REDACTED]"
    assert details["stderr_tail"] == "[REDACTED]"


def test_error_redaction_removes_longer_credentials_before_their_prefix(monkeypatch):
    monkeypatch.setenv("DIAGNOSTIC_SHORT_TOKEN", "dummy-shared-prefix")
    monkeypatch.setenv("DIAGNOSTIC_LONG_TOKEN", "dummy-shared-prefix-private-suffix")
    assert code_e2b._error_message("dummy-shared-prefix-private-suffix") == "[REDACTED]"


@pytest.mark.parametrize("exit_code", [1, 134, 137, 139])
@pytest.mark.parametrize("returned", [False, True])
def test_shared_command_exit_recovers_with_identical_request(
    monkeypatch, warmup_backoff, exit_code, returned
):
    task = _shared_task(monkeypatch)
    first, last = [_sandbox(monkeypatch) for _ in range(2)]
    result = CommandResult(stderr="evaluator failed", stdout="", exit_code=exit_code, error=None)
    if returned:
        first.judge_run.return_value = result
    else:
        first.judge_run.side_effect = e2b.CommandExitException(**vars(result))
    create = _use_sandbox_sequence(monkeypatch, [first, last])
    before = copy.deepcopy(task)
    score = _score(task)
    assert score["reward"] == 1
    assert task == before
    assert create.await_count == 2
    assert create.await_args_list[0] == create.await_args_list[1]
    assert first.judge_run.await_args.args == last.judge_run.await_args.args
    for key in ("user", "timeout", "request_timeout"):
        assert first.judge_run.await_args.kwargs[key] == last.judge_run.await_args.kwargs[key]
    assert score["metrics"]["run_failure_count"] == 1
    assert score["metrics"]["run_retry_count"] == 1
    warmup_backoff.sleep.assert_awaited_once_with(1.5)


def test_shared_stale_file_handle_and_timeout_share_existing_run_budget(
    monkeypatch, warmup_backoff, caplog
):
    task = _shared_task(monkeypatch)
    first, second, last = [_sandbox(monkeypatch) for _ in range(3)]
    first.judge_run.side_effect = e2b.CommandExitException(
        exit_code=1,
        stdout="",
        stderr='Traceback (most recent call last):\n  File "<string>", line 1\n'
        "OSError: [Errno 116] Stale file handle\n",
        error=None,
    )
    first.kill.side_effect = httpx.ReadError("cleanup unconfirmed")
    second.judge_run.side_effect = e2b.TimeoutException("command timed out")
    last.judge_run.return_value.stdout = json.dumps(_result([1.0, 0.5]))
    create = _use_sandbox_sequence(monkeypatch, [first, second, last])
    score = _score(task)
    assert create.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    assert score["reward"] == 0
    assert score["sandbox_result"]["score"] == 0.75
    assert score["metrics"]["run_failure_count"] == 2
    assert score["metrics"]["run_retry_count"] == 2
    assert score["metrics"]["timeout_count"] == 1
    failure = _batch_events(caplog, "failure")[0]
    assert failure["retry_reason"] == "command_exit"
    assert "Stale file handle" in failure["stderr_tail"]


def test_shared_three_command_exits_raise_instead_of_scoring_zero(monkeypatch, warmup_backoff):
    task = _shared_task(monkeypatch)
    sandboxes = [_sandbox(monkeypatch) for _ in range(3)]
    for sandbox in sandboxes:
        sandbox.judge_run.side_effect = e2b.CommandExitException(
            exit_code=1, stdout="", stderr="unexpected evaluator failure", error=None
        )
    create = _use_sandbox_sequence(monkeypatch, sandboxes)
    with pytest.raises(code_e2b.CodeE2BError) as caught:
        _score(task)
    assert caught.value.retryable_run
    assert create.await_count == 3
    assert warmup_backoff.sleep.await_count == 2
    for sandbox in sandboxes:
        sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize("exit_code", [0, 2, 78, 126, 127, 143, True])
def test_command_exit_does_not_override_permanent_exit_codes(exit_code):
    error = e2b.CommandExitException(
        exit_code=exit_code, stdout="", stderr="503: unrelated output", error=None
    )
    error.__cause__ = httpx.ReadError("a network cause cannot override the exit code")
    assert code_e2b._e2b_retry_reason(error, "run") is None


@pytest.mark.parametrize(
    "error_type",
    [
        "ValueError",
        "TypeError",
        "AssertionError",
        "FileNotFoundError",
        "PermissionError",
        "JSONDecodeError",
    ],
)
def test_shared_structured_permanent_judge_error_does_not_retry(
    monkeypatch, warmup_backoff, error_type
):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    sandbox.judge_run.side_effect = e2b.CommandExitException(
        exit_code=1,
        stdout=json.dumps(
            {
                "accepted": False,
                "tests": [],
                "error": {"type": error_type, "message": "invalid request"},
            }
        ),
        stderr="evaluator rejected request",
        error=None,
    )
    with pytest.raises(code_e2b.CodeE2BError):
        _score(task)
    sandbox.create.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("guard", ["attempt_deadline_expired", "warmup_cleanup_incomplete"])
def test_shared_command_exit_recovery_never_bypasses_retry_guards(
    monkeypatch, warmup_backoff, guard
):
    task = _shared_task(monkeypatch)
    error = code_e2b.CodeE2BError(
        "run", RuntimeError("evaluator exited"), retryable_run=True, **{guard: True}
    )
    run = AsyncMock(side_effect=error)
    monkeypatch.setattr(code_e2b, "_run_batch", run)
    with pytest.raises(code_e2b.CodeE2BError) as caught:
        _score(task)
    assert caught.value is error
    run.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


def test_command_exit_recovery_obeys_worker_stop(monkeypatch, warmup_backoff):
    task = _shared_task(monkeypatch)
    sandbox = _sandbox(monkeypatch)
    stop = Event()

    async def fail(*_args, **_kwargs):
        stop.set()
        raise e2b.CommandExitException(exit_code=1, stdout="", stderr="failed", error=None)

    sandbox.judge_run.side_effect = fail
    with reward_runtime_context(stop), pytest.raises(asyncio.CancelledError):
        _score(task)
    sandbox.create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    warmup_backoff.sleep.assert_not_awaited()


@pytest.mark.parametrize("failure", ["missing", "stale"])
def test_bootstrap_distinguishes_missing_reader_from_stale_handle(tmp_path, failure):
    reader = tmp_path / "reader.py"
    reader.write_text("print('reader ran')")
    digest = hashlib.sha256(reader.read_bytes()).hexdigest()
    if failure == "missing":
        reader.unlink()
    shared = code_e2b._VepfsRequest("volume", str(reader), digest, b"{}")
    command, _ = shared.command(str(tmp_path / "request.json"))
    argv = shlex.split(command)
    argv[0] = sys.executable
    if failure == "stale":
        argv[2] = (
            "import os\ndef fail(*a,**k): raise OSError(116,'Stale file handle')\nos.open=fail\n"
            + argv[2]
        )
    result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == (78 if failure == "missing" else 1)
    assert "reader ran" not in result.stdout
    assert ("FileNotFoundError" if failure == "missing" else "Stale file handle") in result.stderr


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ('{"error":{"type":"ValueError","message":"invalid input"}}', "ValueError"),
        ('{"error":{"type":"RuntimeError"}}', "RuntimeError"),
        ('{"error":{"type":true}}', None),
        ('{"error":{"type":[]}}', None),
        ('{"error":{}}', None),
        ('{"error":"ValueError"}', None),
        ('{"accepted":false}', None),
        ("[1,2]", None),
        ("null", None),
        ("not JSON", None),
        ("", None),
        ("[" * 2000 + "]" * 2000, None),
        (" " * (64 * 1024 + 1), None),
        (None, None),
    ],
)
def test_retry_error_type_parser_is_bounded_and_tolerates_invalid_output(stdout, expected):
    assert code_e2b._judge_error_type(stdout) == expected


def test_unstructured_stderr_data_error_keeps_existing_exit_fallback_policy():
    # Exit 1 without the structured envelope is not proof of a transient
    # infrastructure fault. Preserve the policy under review, not a heuristic
    # that interprets arbitrary checker stderr as a trusted error envelope.
    error = e2b.CommandExitException(
        exit_code=1, stdout="", stderr="ValueError: malformed test case", error=None
    )
    assert code_e2b._e2b_retry_reason(error, "run") == "command_exit"
