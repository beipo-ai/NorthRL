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
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import openai
import pytest

from northrl.rewards import llm_judge_retry, math
from northrl.utils.async_utils import flush_sdk_logs, run_coro

JUDGE_CASES = [
    pytest.param(
        "math_llm",
        "<think>reasoning</think>The answer is \\boxed{42}",
        id="math-llm",
    ),
    pytest.param(
        "math_boxed_llm",
        "The answer is \\boxed{42}",
        id="math-boxed-llm",
    ),
]


@pytest.fixture(autouse=True)
def judge_backoff(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(llm_judge_retry, "sleep", sleep)
    monkeypatch.setattr(llm_judge_retry.random, "uniform", lambda low, high: 1.5)
    return sleep


@pytest.fixture(autouse=True)
def judge_environment(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_URL", "http://judge.test/v1")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "test-judge")
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-key")


def _completion(judge: bool = True):
    content = json.dumps(
        {
            "final_student_answer": "42",
            "judge": judge,
        }
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _raw_completion(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _reasoning_completion(content: str):
    message = SimpleNamespace(content="", reasoning_content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _connection_error():
    request = httpx.Request(
        "POST",
        "http://judge.test/v1/chat/completions",
    )
    return openai.APIConnectionError(message="judge unavailable", request=request)


def _timeout_error():
    request = httpx.Request("POST", "http://judge.test/v1/chat/completions")
    return openai.APITimeoutError(request=request)


def _authentication_error():
    request = httpx.Request("POST", "http://judge.test/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError(message="invalid key", response=response, body=None)


def _install_client(monkeypatch, side_effect):
    create = AsyncMock(side_effect=side_effect)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(math.openai, "AsyncOpenAI", constructor)
    return create, constructor


def _score(judge_mode: str, model_response: str) -> float:
    return run_coro(
        math.compute_score(
            prompt_str="",
            solution_str=model_response,
            ground_truth="42",
            stage="train",
            raw_prompt=[{"role": "user", "content": "What is 40 + 2?"}],
            judge_mode=judge_mode,
        )
    )


def _force_new_rule_miss(monkeypatch):
    monkeypatch.setattr(math, "grade_answer_mathd", MagicMock(return_value=False))
    monkeypatch.setattr(math, "grade_answer_sympy", MagicMock(return_value=False))


@pytest.mark.parametrize(
    ("model_response", "expected_reward"),
    [
        ("<think>unfinished reasoning \\boxed{42}", 1.0),
        ("<think>reasoning</think>The answer is \\boxed{41}", 0.0),
        ("The answer is 42", 0.0),
    ],
)
def test_answer_only_reward_is_binary_and_ignores_think_tags(
    model_response,
    expected_reward,
):
    assert _score("math_answer_only", model_response) == expected_reward


@pytest.mark.parametrize(("judge_mode", "model_response"), JUDGE_CASES)
@pytest.mark.parametrize(("verdict", "expected_reward"), [(True, 1.0), (False, 0.1)])
def test_valid_judge_verdict_is_scored(
    monkeypatch,
    judge_mode,
    model_response,
    verdict,
    expected_reward,
):
    create, _ = _install_client(monkeypatch, [_completion(verdict)])

    assert _score(judge_mode, model_response) == expected_reward
    assert create.await_count == 1


@pytest.mark.parametrize(("judge_mode", "model_response"), JUDGE_CASES)
def test_judge_recovers_on_second_attempt(monkeypatch, judge_mode, model_response):
    create, _ = _install_client(
        monkeypatch,
        [_connection_error(), _completion()],
    )

    assert _score(judge_mode, model_response) == 1.0
    assert create.await_count == 2


def test_judge_retry_rotates_to_next_endpoint(monkeypatch):
    first_create = AsyncMock(side_effect=[_connection_error()])
    second_create = AsyncMock(side_effect=[_completion()])
    clients = [
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=first_create),
            )
        ),
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=second_create),
            )
        ),
    ]
    monkeypatch.setenv(
        "LLM_JUDGE_URL",
        "http://judge-1.test/v1\nhttp://judge-2.test/v1",
    )
    monkeypatch.setattr(
        math.openai,
        "AsyncOpenAI",
        MagicMock(side_effect=clients),
    )
    monkeypatch.setattr(math.random, "randrange", MagicMock(return_value=0))

    assert _score("math_boxed_llm", "The answer is \\boxed{42}") == 1.0
    assert first_create.await_count == 1
    assert second_create.await_count == 1


@pytest.mark.parametrize(("judge_mode", "model_response"), JUDGE_CASES)
def test_persistent_judge_outage_propagates(monkeypatch, judge_mode, model_response):
    create, _ = _install_client(
        monkeypatch,
        [_connection_error() for _ in range(5)],
    )

    with pytest.raises(math.LLMJudgeError, match="failed after 5 attempts") as exc_info:
        _score(judge_mode, model_response)

    assert isinstance(exc_info.value.__cause__, openai.APIConnectionError)
    assert create.await_count == 5


@pytest.mark.parametrize(("judge_mode", "model_response"), JUDGE_CASES)
def test_invalid_json_is_retried(monkeypatch, judge_mode, model_response):
    create, _ = _install_client(
        monkeypatch,
        [_raw_completion("{not-json"), _completion()],
    )

    assert _score(judge_mode, model_response) == 1.0
    assert create.await_count == 2


@pytest.mark.parametrize(("judge_mode", "model_response"), JUDGE_CASES)
@pytest.mark.parametrize(
    "invalid_content",
    [
        "{not-json",
        json.dumps({"final_student_answer": "42", "judge": "false"}),
        json.dumps({"final_student_answer": "42"}),
        json.dumps({"judge": False}),
        json.dumps({"final_student_answer": None, "judge": False}),
    ],
)
def test_persistent_invalid_response_propagates(
    monkeypatch,
    judge_mode,
    model_response,
    invalid_content,
):
    create, _ = _install_client(
        monkeypatch,
        [_raw_completion(invalid_content) for _ in range(5)],
    )

    with pytest.raises(math.LLMJudgeError, match="failed after 5 attempts"):
        _score(judge_mode, model_response)

    assert create.await_count == 5


def test_hybrid_judge_initialization_failure_propagates(monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_URL")
    monkeypatch.setattr(
        math.RewardMathFn,
        "__call__",
        MagicMock(return_value=(0.1, False)),
    )

    with pytest.raises(math.LLMJudgeError, match="Failed to initialize"):
        run_coro(
            math.compute_score(
                prompt_str="",
                solution_str="<think>reasoning</think>42",
                ground_truth="42",
                stage="train",
                raw_prompt=[{"role": "user", "content": "What is 40 + 2?"}],
                judge_mode="math_rule_llm",
            )
        )


def test_judge_client_has_bounded_timeout_and_no_sdk_retries(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "7.5")
    _, constructor = _install_client(monkeypatch, [])

    math.MathBoxedLLMJudgeFn(math.RewardConfig())

    assert constructor.call_args.kwargs["timeout"] == 7.5
    assert constructor.call_args.kwargs["max_retries"] == 0


def test_slime_compatible_llm_judge_uses_http_protocol(monkeypatch):
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": "<thinking>ok</thinking>\\boxed{Correct}"}}]
    }
    post = MagicMock(return_value=response)
    monkeypatch.setattr(math.requests, "post", post)
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "90")

    assert _score("math_llmjudge", "Reasoning without a box") == 1.0

    response.raise_for_status.assert_called_once_with()
    post.assert_called_once()
    url = post.call_args.args[0]
    request = post.call_args.kwargs
    assert url == "http://judge.test/v1/chat/completions"
    assert request["timeout"] == 90.0
    assert request["json"]["model"] == "test-judge"
    assert request["json"]["temperature"] == 0.0
    assert request["json"]["max_tokens"] == 2048
    assert "chat_template_kwargs" not in request["json"]
    judge_prompt = request["json"]["messages"][0]["content"]
    assert "Model Solution: Reasoning without a box" in judge_prompt
    assert "Golden Answer: 42" in judge_prompt


def test_slime_compatible_llm_judge_extracts_last_boxed_answer(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": r"\boxed{Incorrect}"}}]}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(math.requests, "post", post)

    assert _score("math_llmjudge", r"first \boxed{41}, final \boxed{42}") == 0.0
    student_answer = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "Model Solution: 42" in student_answer
    assert "Model Solution: first" not in student_answer


def test_slime_compatible_llm_judge_failure_returns_zero(monkeypatch):
    monkeypatch.setattr(math.requests, "post", MagicMock(side_effect=RuntimeError("down")))

    assert _score("math_llmjudge", r"\boxed{42}") == 0.0


@pytest.mark.parametrize(
    ("content", "expected_reward"),
    [
        ("The answer is [[1]]", 1.0),
        ("The answer is [[0]]", 0.0),
        ("initial [[0]], corrected [[1]]", 1.0),
        ("initial [[1]], corrected [[0]]", 0.0),
    ],
)
def test_new_math_llm_uses_last_binary_verdict(
    monkeypatch,
    content,
    expected_reward,
):
    _force_new_rule_miss(monkeypatch)
    create, _ = _install_client(monkeypatch, [_raw_completion(content)])

    assert _score("math_rule_llm_answer_only", "Natural-language answer: 42") == expected_reward
    assert create.await_count == 1


def test_new_math_llm_uses_bp_prompt_full_task_and_request_settings(monkeypatch):
    _force_new_rule_miss(monkeypatch)
    create, _ = _install_client(monkeypatch, [_raw_completion("[[1]]")])
    raw_prompt = [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"}]

    score = run_coro(
        math.compute_score(
            prompt_str="rendered prompt",
            solution_str="<think>hidden</think>natural final answer",
            ground_truth="42",
            stage="train",
            raw_prompt=raw_prompt,
            judge_mode="math_rule_llm_answer_only",
        )
    )

    assert score == 1.0
    request = create.call_args.kwargs
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 8192
    judge_prompt = request["messages"][0]["content"]
    assert f"【题目】{raw_prompt}" in judge_prompt
    assert "【参考答案】42" in judge_prompt
    assert "【学生解答】natural final answer" in judge_prompt
    assert "输出 [[1]];否则输出 [[0]]" in judge_prompt


def test_new_math_llm_retries_no_verdict_then_uses_reasoning_content(monkeypatch):
    _force_new_rule_miss(monkeypatch)
    create, _ = _install_client(
        monkeypatch,
        [_raw_completion("cannot decide"), _reasoning_completion("final [[1]]")],
    )

    assert _score("math_rule_llm_answer_only", "answer without boxed") == 1.0
    assert create.await_count == 2


def test_new_math_explicit_negative_is_not_logged_as_infrastructure_failure(
    monkeypatch,
    caplog,
):
    _force_new_rule_miss(monkeypatch)
    _install_client(monkeypatch, [_raw_completion("[[0]]")])

    with caplog.at_level(logging.WARNING):
        assert _score("math_rule_llm_answer_only", "wrong answer") == 0.0

    assert "assigning reward 0" not in caplog.text


def test_new_math_unparseable_llm_response_warns_and_returns_zero(
    monkeypatch,
    caplog,
):
    _force_new_rule_miss(monkeypatch)
    create, _ = _install_client(
        monkeypatch,
        [_raw_completion("cannot decide") for _ in range(5)],
    )

    with caplog.at_level(logging.WARNING):
        assert _score("math_rule_llm_answer_only", "wrong answer") == 0.0

    assert create.await_count == 5
    assert "math_rule_llm_answer_only LLM fallback failed; assigning reward 0" in caplog.text
    assert "reason=no_verdict" in caplog.text


def test_new_math_missing_endpoint_warns_and_returns_zero(monkeypatch, caplog):
    _force_new_rule_miss(monkeypatch)
    monkeypatch.delenv("LLM_JUDGE_URL")

    with caplog.at_level(logging.WARNING):
        assert _score("math_rule_llm_answer_only", "wrong answer") == 0.0

    assert "reason=no_endpoint" in caplog.text


def test_new_math_persistent_llm_failure_warns_and_returns_zero(monkeypatch, caplog):
    _force_new_rule_miss(monkeypatch)
    create, _ = _install_client(monkeypatch, [_connection_error() for _ in range(5)])

    with caplog.at_level(logging.WARNING):
        assert _score("math_rule_llm_answer_only", "wrong answer") == 0.0

    assert create.await_count == 5
    assert "reason=request_error" in caplog.text


@pytest.mark.parametrize(
    ("error_factory", "reason"),
    [
        (_timeout_error, "timeout"),
        (_authentication_error, "authentication_error"),
    ],
)
def test_new_math_classifies_llm_infrastructure_failures(
    monkeypatch,
    caplog,
    error_factory,
    reason,
):
    _force_new_rule_miss(monkeypatch)
    create, _ = _install_client(monkeypatch, [error_factory() for _ in range(5)])

    with caplog.at_level(logging.WARNING):
        assert _score("math_rule_llm_answer_only", "wrong answer") == 0.0

    assert create.await_count == (1 if reason == "authentication_error" else 5)
    assert f"reason={reason}" in caplog.text


@pytest.mark.parametrize(("judge_mode", "model_response"), JUDGE_CASES)
@pytest.mark.parametrize("failures", [0, 1, 2, 3, 4])
def test_math_bounded_backoff_preserves_request(
    monkeypatch, judge_backoff, judge_mode, model_response, failures
):
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "180")
    create, constructor = _install_client(
        monkeypatch, [_connection_error() for _ in range(failures)] + [_completion()]
    )
    assert _score(judge_mode, model_response) == 1.0
    assert create.await_count == failures + 1
    assert judge_backoff.await_args_list == [call(1.5), call(3.0), call(6.0), call(12.0)][:failures]
    assert [args.kwargs["timeout"] for args in create.await_args_list] == (
        [180.0] * min(failures + 1, 4) + ([300.0] if failures == 4 else [])
    )
    requests = [
        {key: value for key, value in args.kwargs.items() if key != "timeout"}
        for args in create.await_args_list
    ]
    assert all(request == requests[0] for request in requests)
    assert constructor.call_args.kwargs["timeout"] == 180.0
    assert constructor.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize(("judge_mode", "model_response"), JUDGE_CASES)
def test_final_math_request_timeout_preserves_exhausted_failure_policy(
    monkeypatch, caplog, judge_backoff, judge_mode, model_response
):
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "180")
    failure = _timeout_error()
    create, _ = _install_client(monkeypatch, [_connection_error() for _ in range(4)] + [failure])

    with pytest.raises(math.LLMJudgeError, match="failed after 5 attempts") as raised:
        _score(judge_mode, model_response)

    assert raised.value.__cause__ is failure
    assert create.await_count == 5
    assert [args.kwargs["timeout"] for args in create.await_args_list] == [
        180.0,
        180.0,
        180.0,
        180.0,
        300.0,
    ]
    assert judge_backoff.await_count == 4
    flush_sdk_logs()
    assert "attempt=5/5" in caplog.text and "timeout_s=300.0" in caplog.text


def test_concurrent_math_final_attempt_does_not_mutate_shared_client_timeout(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "180")
    recorded_timeouts = {"slow": [], "fast": []}

    async def run():
        final_started = asyncio.Event()
        release_final = asyncio.Event()

        async def completion(**request):
            name = request["messages"][0]["content"]
            recorded_timeouts[name].append(request["timeout"])
            assert client.timeout == 180.0
            if name == "slow":
                if len(recorded_timeouts[name]) < 5:
                    raise _connection_error()
                final_started.set()
                await release_final.wait()
            return _completion()

        create, constructor = _install_client(monkeypatch, completion)
        client = constructor.return_value
        client.timeout = 180.0
        judge = math.MathBoxedLLMJudgeFn(math.RewardConfig())

        async def fast_request():
            await final_started.wait()
            try:
                return await judge._call_judge("fast")
            finally:
                release_final.set()

        results = await asyncio.wait_for(
            asyncio.gather(judge._call_judge("slow"), fast_request()), timeout=2
        )
        assert all(result["judge"] is True for result in results)
        assert client.timeout == 180.0
        constructor.assert_called_once()
        assert create.await_count == 6

    asyncio.run(run())
    assert recorded_timeouts == {"slow": [180.0, 180.0, 180.0, 180.0, 300.0], "fast": [180.0]}


def test_math_timeout_overrides_reach_real_sdk_without_nested_retries(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", "180")
    constructor = math.openai.AsyncOpenAI
    requests = []
    clients = []

    async def respond(request):
        requests.append(request)
        if len(requests) <= 4:
            return httpx.Response(503, json={"error": {"message": "temporary test failure"}})
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-judge",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"judge": True, "final_student_answer": "42"}),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def build_client(**kwargs):
        assert kwargs["max_retries"] == 0
        client = constructor(
            **kwargs,
            http_client=openai.DefaultAsyncHttpxClient(
                transport=httpx.MockTransport(respond), trust_env=False
            ),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(math.openai, "AsyncOpenAI", build_client)

    async def run():
        judge = math.MathBoxedLLMJudgeFn(math.RewardConfig())
        try:
            assert (await judge._call_judge("original question"))["judge"] is True
            assert clients[0].timeout == 180.0
        finally:
            for client in clients:
                await client.close()

    asyncio.run(run())
    assert len(clients) == 1
    assert len(requests) == 5
    assert [request.extensions["timeout"]["read"] for request in requests] == [
        180.0,
        180.0,
        180.0,
        180.0,
        300.0,
    ]
    assert all(request.content == requests[0].content for request in requests)
    assert clients[0].is_closed()


@pytest.mark.parametrize("verdict", ["[[0]]", "[[1]]"])
def test_answer_only_fallback_recovers_on_third_attempt(monkeypatch, judge_backoff, verdict):
    _force_new_rule_miss(monkeypatch)
    create, _ = _install_client(
        monkeypatch, [_timeout_error(), _timeout_error(), _raw_completion(verdict)]
    )
    assert _score("math_rule_llm_answer_only", "answer without boxed") == float(verdict == "[[1]]")
    assert create.await_count == 3
    assert judge_backoff.await_args_list == [call(1.5), call(3.0)]


@pytest.mark.parametrize(
    ("stage", "previous_failures"),
    [("request", 0), ("request", 4), ("backoff", 0), ("backoff", 3)],
)
def test_math_cancellation_is_not_retried_or_scored(
    monkeypatch, judge_backoff, stage, previous_failures
):
    final = asyncio.CancelledError() if stage == "request" else _connection_error()
    outcomes = [_connection_error() for _ in range(previous_failures)] + [final]
    create, _ = _install_client(monkeypatch, outcomes)
    if stage == "backoff":
        judge_backoff.side_effect = [None] * previous_failures + [asyncio.CancelledError()]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            math.compute_score(
                prompt_str="",
                solution_str="answer without boxed",
                ground_truth="42",
                stage="train",
                raw_prompt=[{"role": "user", "content": "What is 40 + 2?"}],
                judge_mode="math_rule_llm_answer_only",
            )
        )
    assert create.await_count == previous_failures + 1
    assert judge_backoff.await_count == previous_failures + int(stage == "backoff")


def test_new_math_calls_sympy_synchronously_without_soft_timeout(monkeypatch):
    monkeypatch.setattr(math, "grade_answer_mathd", MagicMock(return_value=False))
    sympy_check = MagicMock(return_value=True)
    monkeypatch.setattr(math, "grade_answer_sympy", sympy_check)
    create, _ = _install_client(monkeypatch, [_raw_completion("[[1]]")])

    assert not hasattr(math, "MATH_SYMPY_TIMEOUT_S")
    assert _score("math_rule_llm_answer_only", r"\boxed{41}") == 1.0

    sympy_check.assert_called_once_with("41", "42")
    assert create.await_count == 0


def test_new_math_rule_exception_falls_back_to_llm(monkeypatch):
    monkeypatch.setattr(math, "grade_answer_mathd", MagicMock(side_effect=ValueError("bad")))
    create, _ = _install_client(monkeypatch, [_raw_completion("[[1]]")])

    assert _score("math_rule_llm_answer_only", r"\boxed{42}") == 1.0
    assert create.await_count == 1
