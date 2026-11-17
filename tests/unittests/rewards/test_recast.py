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
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import openai
import pytest

from northrl.rewards import evaluate_reward, llm_judge_retry, recast


@pytest.fixture(autouse=True)
def judge_backoff(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(llm_judge_retry, "sleep", sleep)
    monkeypatch.setattr(llm_judge_retry.random, "uniform", lambda low, high: 1.5)
    return sleep


@pytest.fixture(autouse=True)
def judge_environment(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_URL", "http://judge.test/v1")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "instruction-judge")
    monkeypatch.delenv("LLM_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_JUDGE_TIMEOUT_S", raising=False)


def _completion(content, reasoning_content=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _connection_error():
    request = httpx.Request("POST", "http://judge.test/v1/chat/completions")
    return openai.APIConnectionError(message="judge unavailable", request=request)


def _install_judge(monkeypatch, outcomes):
    create = AsyncMock(side_effect=outcomes)
    client = MagicMock()
    client.chat.completions.create = create
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    constructor = MagicMock(return_value=client)
    http_constructor = MagicMock()
    monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", http_constructor)
    return SimpleNamespace(
        create=create,
        client=client,
        constructor=constructor,
        http_constructor=http_constructor,
    )


def _score():
    return asyncio.run(
        recast.recast_reward({"prompt": "Give three suggestions."}, {"response": "A"})
    )


@pytest.mark.parametrize(
    ("content", "expected_reward", "expected_correct", "expected_reason"),
    [
        ("All requirements met. [[1]]", 1.0, True, "pass"),
        ("One requirement is missing. [[0]]", 0.0, False, "fail_llm"),
        ("Initially [[1]], but the final verdict is [[ 0 ]].", 0.0, False, "fail_llm"),
        ("Initially [[0]], but the final verdict is [[\n1\t]].", 1.0, True, "pass"),
        ("Looks good, but there is no verdict token.", 0.0, False, "no_verdict"),
        ("[[2]]", 0.0, False, "no_verdict"),
        ("", 0.0, False, "no_verdict"),
        (None, 0.0, False, "no_verdict"),
    ],
)
def test_verdict_uses_last_binary_token_without_retry(
    monkeypatch, caplog, content, expected_reward, expected_correct, expected_reason
):
    judge = _install_judge(monkeypatch, [_completion(content)])

    assert _score() == {
        "reward": expected_reward,
        "correct": expected_correct,
        "reason": expected_reason,
    }
    assert judge.create.await_count == 1
    judge.client.__aexit__.assert_awaited_once()
    warnings = [
        record
        for record in caplog.records
        if record.name == recast.__name__ and record.levelno == logging.WARNING
    ]
    assert len(warnings) == int(expected_reason == "no_verdict")
    if warnings:
        assert "no parseable [[0]]/[[1]] verdict" in warnings[0].getMessage()


def test_no_verdict_warning_has_task_context_and_bounded_judge_preview(monkeypatch, caplog):
    text = "No binary verdict. " + "x" * 800 + "TAIL_MUST_NOT_BE_LOGGED"
    judge = _install_judge(monkeypatch, [_completion(text)])
    task = {
        "prompt": "Reply with YES.",
        "reward_fn": "recast",
        "task_id": 123,
        "group_id": "group-123",
        "sample_id": "sample-123",
    }

    result = asyncio.run(recast.recast_reward(task, {"response": "YES"}))

    assert result == {"reward": 0.0, "correct": False, "reason": "no_verdict"}
    assert judge.create.await_count == 1
    assert "task_id=123" in caplog.text
    assert "group_id='group-123'" in caplog.text
    assert "sample_id='sample-123'" in caplog.text
    assert "judge_model='instruction-judge'" in caplog.text
    assert "TAIL_MUST_NOT_BE_LOGGED" not in caplog.text


def test_verinstruct_soft_no_verdict_emits_recast_warning_without_retry(monkeypatch, caplog):
    judge = _install_judge(monkeypatch, [_completion("The response looks correct.")])

    result = asyncio.run(
        evaluate_reward(
            {"reward_fn": "verinstruct", "prompt": "Reply with YES.", "metadata": {}},
            {"response": "YES"},
        )
    )

    assert result == {"reward": 0.0, "correct": False, "reason": "no_verdict"}
    assert judge.create.await_count == 1
    assert "no parseable [[0]]/[[1]] verdict" in caplog.text
    assert "reward_fn='verinstruct'" in caplog.text


@pytest.mark.parametrize("content", [None, ""])
def test_empty_content_uses_reasoning_content(monkeypatch, content):
    judge = _install_judge(monkeypatch, [_completion(content, "All requirements met. [[1]]")])

    assert _score()["reward"] == 1.0
    assert judge.create.await_count == 1


def test_nonempty_content_takes_precedence_over_reasoning_content(monkeypatch):
    _install_judge(monkeypatch, [_completion("[[0]]", "[[1]]")])

    assert _score()["reason"] == "fail_llm"


@pytest.mark.parametrize("array_prompt", [False, True])
def test_full_prompt_and_response_are_judged_without_label_or_rendered_prompt(
    monkeypatch, array_prompt
):
    judge = _install_judge(monkeypatch, [_completion("[[1]]")])
    messages = [
        {"role": "system", "content": "Use Chinese."},
        {"role": "user", "content": "Give three suggestions."},
    ]
    prompt = SimpleNamespace(tolist=lambda: messages) if array_prompt else messages
    response = "<think>Some reasoning</think>建议一、建议二、建议三。<|im_end|>"
    task = {
        "prompt": prompt,
        "label": "LABEL MUST NOT BE INCLUDED",
        "metadata": {"prompt": "METADATA MUST NOT REPLACE THE INSTRUCTION"},
    }

    assert (
        asyncio.run(
            recast.recast_reward(
                task,
                {"response": response, "prompt_str": "RENDERED PROMPT MUST NOT BE USED"},
            )
        )["reward"]
        == 1.0
    )
    assert judge.create.await_args.kwargs["messages"] == [
        {
            "role": "user",
            "content": (
                "请判断回复是否符合指令的全部要求。\n"
                f"【指令】{messages}\n【回复】{response}\n"
                "若完全符合,请输出 [[1]];否则输出 [[0]]。"
            ),
        }
    ]


def test_short_name_dispatches_recast_without_label(monkeypatch):
    _install_judge(monkeypatch, [_completion("[[1]]")])

    result = asyncio.run(
        evaluate_reward(
            {"reward_fn": "recast", "prompt": "Give three suggestions."},
            {"response": "A, B, C"},
            default_reward_fn="math_with_format",
        )
    )

    assert result == {"reward": 1.0, "correct": True, "reason": "pass"}


def test_judge_request_uses_bounded_defaults_and_bypasses_proxy(monkeypatch):
    judge = _install_judge(monkeypatch, [_completion("[[1]]")])

    _score()

    settings = judge.constructor.call_args.kwargs
    assert settings["api_key"] == "_"
    assert settings["timeout"] == 180.0
    assert settings["max_retries"] == 0
    assert settings["http_client"] is judge.http_constructor.return_value
    judge.http_constructor.assert_called_once_with(trust_env=False)
    request = judge.create.await_args.kwargs
    assert request["model"] == "instruction-judge"
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 8192


def test_judge_uses_configured_key_and_timeout(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "7.5")
    judge = _install_judge(monkeypatch, [_completion("[[1]]")])

    _score()

    settings = judge.constructor.call_args.kwargs
    assert settings["api_key"] == "test-key"
    assert settings["timeout"] == 7.5


def test_retry_rotates_endpoints_and_closes_each_client(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_URL", " http://judge-1.test/v1 \n\n http://judge-2.test/v1\n")
    monkeypatch.setattr(recast.random, "randrange", lambda count: count - 1)
    judge = _install_judge(monkeypatch, [_connection_error(), _completion("[[1]]")])

    assert _score()["reward"] == 1.0
    assert [call.kwargs["base_url"] for call in judge.constructor.call_args_list] == [
        "http://judge-2.test/v1",
        "http://judge-1.test/v1",
    ]
    assert judge.create.await_count == 2
    assert judge.client.__aenter__.await_count == 2
    assert judge.client.__aexit__.await_count == 2


@pytest.mark.parametrize(
    "malformed",
    [SimpleNamespace(choices=[]), None, _completion(123)],
)
def test_malformed_api_response_is_retried(monkeypatch, malformed):
    judge = _install_judge(monkeypatch, [malformed, _completion("[[1]]")])

    assert _score()["reward"] == 1.0
    assert judge.create.await_count == 2


@pytest.mark.parametrize("base_timeout", [7.5, 180.0])
def test_exhausted_requests_raise_with_original_cause(
    monkeypatch, caplog, judge_backoff, base_timeout
):
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", str(base_timeout))
    failure = openai.APITimeoutError(request=httpx.Request("POST", "http://judge.test/v1"))
    judge = _install_judge(monkeypatch, [_connection_error() for _ in range(4)] + [failure])

    with pytest.raises(RuntimeError, match="Recast judge failed after 5 attempts") as error:
        _score()

    assert error.value.__cause__ is failure
    assert judge.create.await_count == 5
    assert judge.client.__aexit__.await_count == 5
    assert judge_backoff.await_args_list == [call(1.5), call(3.0), call(6.0), call(12.0)]
    assert [args.kwargs["timeout"] for args in judge.constructor.call_args_list] == [
        base_timeout,
        base_timeout,
        base_timeout,
        base_timeout,
        300.0,
    ]
    assert "attempt=5/5" in caplog.text and "timeout_s=300.0" in caplog.text


@pytest.mark.parametrize("failures", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("verdict", ["[[0]]", "[[1]]"])
def test_bounded_backoff_reuses_original_request(monkeypatch, judge_backoff, failures, verdict):
    judge = _install_judge(
        monkeypatch, [_connection_error() for _ in range(failures)] + [_completion(verdict)]
    )

    async def check_closed_before_wait(delay):
        assert judge.client.__aexit__.await_count == judge.create.await_count

    judge_backoff.side_effect = check_closed_before_wait
    assert _score()["reward"] == float(verdict == "[[1]]")
    assert judge.create.await_count == failures + 1
    assert judge_backoff.await_args_list == [call(1.5), call(3.0), call(6.0), call(12.0)][:failures]
    assert all(args == judge.create.await_args_list[0] for args in judge.create.await_args_list)
    assert all(settings.kwargs["max_retries"] == 0 for settings in judge.constructor.call_args_list)
    assert [args.kwargs["timeout"] for args in judge.constructor.call_args_list] == (
        [180.0] * min(failures + 1, 4) + ([300.0] if failures == 4 else [])
    )


def test_third_attempt_rotates_endpoint_and_logs_task_context(monkeypatch, caplog, judge_backoff):
    monkeypatch.setenv("LLM_JUDGE_URL", "http://judge-1.test/v1\nhttp://judge-2.test/v1")
    monkeypatch.setattr(recast.random, "randrange", lambda count: 0)
    judge = _install_judge(
        monkeypatch, [_connection_error(), _connection_error(), _completion("[[1]]")]
    )
    task = {
        "prompt": "Reply YES.",
        "reward_fn": "recast",
        "task_id": 123,
        "group_id": "group-123",
        "sample_id": "sample-123",
    }

    assert asyncio.run(recast.recast_reward(task, {"response": "YES"}))["reward"] == 1.0
    assert [args.kwargs["base_url"] for args in judge.constructor.call_args_list] == [
        "http://judge-1.test/v1",
        "http://judge-2.test/v1",
        "http://judge-1.test/v1",
    ]
    assert "attempt=1/5" in caplog.text and "attempt=2/5" in caplog.text
    assert "sample-123" in caplog.text and "elapsed_s=" in caplog.text
    assert judge_backoff.await_count == 2


@pytest.mark.parametrize("success", [True, False])
def test_verinstruct_soft_inherits_shared_retry_budget(monkeypatch, judge_backoff, success):
    final = _completion("[[1]]") if success else _connection_error()
    judge = _install_judge(monkeypatch, [_connection_error() for _ in range(4)] + [final])
    coro = evaluate_reward(
        {"reward_fn": "verinstruct", "prompt": "Reply YES.", "metadata": {}},
        {"response": "YES"},
    )
    if success:
        assert asyncio.run(coro)["reward"] == 1.0
    else:
        with pytest.raises(RuntimeError, match="failed after 5 attempts"):
            asyncio.run(coro)
    assert judge.create.await_count == 5
    assert judge_backoff.await_args_list == [call(1.5), call(3.0), call(6.0), call(12.0)]
    assert [args.kwargs["timeout"] for args in judge.constructor.call_args_list] == [
        180.0,
        180.0,
        180.0,
        180.0,
        300.0,
    ]


def test_authentication_error_does_not_retry(monkeypatch, judge_backoff):
    response = httpx.Response(401, request=httpx.Request("POST", "http://judge.test/v1"))
    failure = openai.AuthenticationError("invalid key", response=response, body=None)
    judge = _install_judge(monkeypatch, [failure])

    with pytest.raises(RuntimeError, match="failed after 1 attempts") as error:
        _score()
    assert error.value.__cause__ is failure
    assert judge.create.await_count == 1
    judge_backoff.assert_not_awaited()


def test_cancellation_during_backoff_propagates_after_client_closed(monkeypatch, judge_backoff):
    judge = _install_judge(monkeypatch, [_connection_error(), _completion("[[1]]")])
    judge_backoff.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _score()
    assert judge.create.await_count == 1
    judge.client.__aexit__.assert_awaited_once()


def test_retry_budget_is_local_to_each_answer(monkeypatch, judge_backoff):
    judge = _install_judge(
        monkeypatch,
        ([_connection_error() for _ in range(4)] + [_completion("[[1]]")]) * 2,
    )
    for _ in range(2):
        assert _score()["reward"] == 1.0
    assert judge.create.await_count == 10
    assert judge_backoff.await_args_list == [call(1.5), call(3.0), call(6.0), call(12.0)] * 2
    assert [args.kwargs["timeout"] for args in judge.constructor.call_args_list] == [
        180.0,
        180.0,
        180.0,
        180.0,
        300.0,
    ] * 2


@pytest.mark.parametrize("previous_failures", [0, 4])
def test_cancellation_is_not_retried_or_converted_to_reward(
    monkeypatch, judge_backoff, previous_failures
):
    judge = _install_judge(
        monkeypatch,
        [_connection_error() for _ in range(previous_failures)] + [asyncio.CancelledError()],
    )

    with pytest.raises(asyncio.CancelledError):
        _score()

    assert judge.create.await_count == previous_failures + 1
    assert judge.client.__aexit__.await_count == previous_failures + 1
    assert judge_backoff.await_count == previous_failures


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_JUDGE_URL", None),
        ("LLM_JUDGE_URL", " \n\t"),
        ("LLM_JUDGE_MODEL", None),
        ("LLM_JUDGE_MODEL", " "),
        ("LLM_JUDGE_TIMEOUT_S", "invalid"),
        ("LLM_JUDGE_TIMEOUT_S", "0"),
        ("LLM_JUDGE_TIMEOUT_S", "-1"),
        ("LLM_JUDGE_TIMEOUT_S", "nan"),
        ("LLM_JUDGE_TIMEOUT_S", "inf"),
    ],
)
def test_invalid_configuration_fails_before_creating_clients(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    judge = _install_judge(monkeypatch, [])

    with pytest.raises(ValueError):
        _score()

    judge.constructor.assert_not_called()
    judge.http_constructor.assert_not_called()
