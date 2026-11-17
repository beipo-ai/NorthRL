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

from northrl.rewards import REWARD_FNS, evaluate_reward, llm_judge_retry
from northrl.rewards import reasoning360 as reasoning360_module
from northrl.rewards.reasoning360 import (
    _STEM_JUDGE_PROMPT,
    _extract_stem_answer,
    _score_arc_grid,
    _score_graph,
    _score_ordering,
    _score_zebra,
    reasoning360_reward,
)


@pytest.fixture(autouse=True)
def judge_backoff(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(llm_judge_retry, "sleep", sleep)
    monkeypatch.setattr(llm_judge_retry.random, "uniform", lambda low, high: 1.5)
    return sleep


@pytest.fixture(autouse=True)
def judge_environment(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_URL", "http://judge.test/v1")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "stem-judge")
    monkeypatch.delenv("LLM_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_JUDGE_TIMEOUT_S", raising=False)


def _task(data_source, label, **metadata):
    return {
        "reward_fn": "reasoning360",
        "label": label,
        "metadata": {"data_source": data_source, **metadata},
    }


def _score(task, response):
    return asyncio.run(reasoning360_reward(task, {"response": response}))


# Handwritten test cases with explicit answers, independent of debug recipes.
_SAMPLE_TASKS = [
    {
        **_task(source, label, question=question),
        "sample_id": f"test-reasoning360-{source}",
        "prompt": [{"role": "user", "content": question}],
    }
    for source, label, question in [
        (
            "logic__graph_logical_dataset",
            "true",
            "A implies B. A is true. Is B true? Answer true or false in <answer> tags.",
        ),
        (
            "logic__zebra_puzzle_dataset",
            {"header": ["Position", "Pet"], "rows": [["1", "cat"], ["2", "dog"]]},
            "House 1 has a cat and house 2 has a dog. Return a table with header "
            "[Position, Pet] and its rows in <answer> tags.",
        ),
        (
            "logic__ordering_puzzle_dataset",
            ["A", "B", "C"],
            "A comes before B and B before C. List their complete order in <answer> tags.",
        ),
        (
            "simulation__barc",
            [[1, 0], [0, 1]],
            "Copy this grid unchanged: [[1, 0], [0, 1]]. Return it in <answer> tags.",
        ),
        (
            "simulation__arcagi1",
            [[2, 2], [2, 2]],
            "Replace every 1 with 2 in [[1, 1], [1, 1]]. Return the grid in <answer> tags.",
        ),
        (
            "simulation__arcagi2",
            [[3, 0], [0, 3]],
            "Replace every 1 with 3 in [[1, 0], [0, 1]]. Return the grid in <answer> tags.",
        ),
        (
            "stem_web",
            "20N",
            "What force accelerates a 10 kg mass at 2 m/s^2? Give the answer in newtons.",
        ),
    ]
]


def _completion(content=None, reasoning_content=None):
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


def test_registered_reward_dispatches_asynchronously():
    assert REWARD_FNS["reasoning360"] is reasoning360_reward
    result = asyncio.run(
        evaluate_reward(
            _task("logic__graph_logical_dataset", "jedlab"),
            {"response": "<answer>JEDLAB</answer>"},
        )
    )
    assert result["reward"] == 1.0


@pytest.mark.parametrize(
    ("data_source", "expected_verifier"),
    [
        ("logic__graph_logical_dataset", "graph"),
        ("logic__zebra_puzzle_dataset", "zebra"),
        ("logic__ordering_puzzle_dataset", "ordering"),
        ("simulation__barc", "arc_grid"),
        ("simulation__arcagi1", "arc_grid"),
        ("simulation__arcagi2", "arc_grid"),
        ("stem_web", "stem_web"),
    ],
)
def test_all_seven_sources_route_to_the_expected_five_verifiers(
    monkeypatch, data_source, expected_verifier
):
    calls = []

    def local(name):
        def scorer(_response, _label):
            calls.append(name)
            return 1.0

        return scorer

    async def stem(_question, _student, _reference):
        calls.append("stem_web")
        return 1.0, False

    monkeypatch.setattr(reasoning360_module, "_score_graph", local("graph"))
    monkeypatch.setattr(reasoning360_module, "_score_zebra", local("zebra"))
    monkeypatch.setattr(reasoning360_module, "_score_ordering", local("ordering"))
    monkeypatch.setattr(reasoning360_module, "_score_arc_grid", local("arc_grid"))
    monkeypatch.setattr(reasoning360_module, "_judge_stem", stem)
    metadata = {"question": "Question"} if data_source == "stem_web" else {}
    response = r"<answer>\boxed{answer}</answer>" if data_source == "stem_web" else "x"

    assert _score(_task(data_source, "answer", **metadata), response)["reward"] == 1.0
    assert calls == [expected_verifier]


@pytest.mark.parametrize(
    "data_source",
    ["simulation__arcagi", "simulation__arcagi3", "logic__graph_unknown", "stem_web_extra"],
)
def test_similar_but_unsupported_sources_fail_fast(data_source):
    with pytest.raises(ValueError, match="Unsupported Reasoning360 data_source"):
        _score(_task(data_source, "answer"), "<answer>answer</answer>")


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [(1.0, 1.0), (0.999, 1.0), (0.998, 0.0)],
)
def test_raw_score_uses_strict_binary_threshold(monkeypatch, raw_score, expected):
    monkeypatch.setattr(reasoning360_module, "_score_graph", lambda _response, _label: raw_score)
    result = _score(_task("logic__graph_logical_dataset", "answer"), "response")
    assert result["reward"] == expected
    assert result["metrics"]["reasoning360_raw_score"] == raw_score


@pytest.mark.parametrize("raw_score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_raw_score_fails_fast(monkeypatch, raw_score):
    monkeypatch.setattr(reasoning360_module, "_score_graph", lambda _response, _label: raw_score)
    with pytest.raises(ValueError, match="invalid score"):
        _score(_task("logic__graph_logical_dataset", "answer"), "response")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<answer>wrong</answer> then <answer>JEDLAB</answer>", 1.0),
        ("<answer>jed lab</answer>", 0.0),
        ("<answer>jedlab1</answer>", 0.0),
        ("<answer>wrong</answer>", 0.0),
        ("jedlab", 0.0),
    ],
)
def test_graph_matches_bp_last_tag_and_letters_only(response, expected):
    assert _score_graph(response, "jedlab") == expected


def test_zebra_preserves_bp_target_region_accuracy_and_final_binarization():
    label = {
        "header": ["Position", "Food", "Pet"],
        "rows": [["1", "kale", "bird"], ["2", "papaya", "cat"]],
    }
    correct = (
        '<answer>{"header": ["ignored"], "rows": '
        '[["1", "kale", "bird"], ["2", "papaya", "cat"], ["extra"]]}</answer>'
    )
    partial = (
        "<answer>{'header': [], 'rows': "
        "[['1', 'kale', 'bird'], ['2', 'papaya', 'dog']]}</answer>"
    )
    assert _score_zebra(correct, label) == 1.0
    assert _score_zebra(partial, label) == pytest.approx(5 / 6)
    assert _score(_task("logic__zebra_puzzle_dataset", label), partial)["reward"] == 0.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<answer>['robin', 'hawk', 'dove']</answer>", 1.0),
        ("<answer>[Robin, Hawk, Dove]</answer>", 1.0),
        ("<answer>['hawk', 'robin', 'dove']</answer>", 0.0),
        ("<answer>['robin', 'hawk']</answer>", 0.0),
        ("['robin', 'hawk', 'dove']", 0.0),
    ],
)
def test_ordering_requires_the_complete_order(response, expected):
    assert _score_ordering(response, ["robin", "hawk", "dove"]) == expected


@pytest.mark.parametrize(
    "data_source", ["simulation__barc", "simulation__arcagi1", "simulation__arcagi2"]
)
def test_all_three_grid_sources_accept_correct_and_reject_wrong_grids(data_source):
    label = [[0, 0, 4], [0, 8, 6], [5, 3, 6]]
    correct = "reasoning <answer>[[0, 0, 4],\n[0, 8, 6],\n[5, 3, 6]]</answer>"
    wrong = "<answer>[[0, 0, 4], [0, 8, 6], [5, 3, 7]]</answer>"
    assert _score(_task(data_source, label), correct)["reward"] == 1.0
    assert _score(_task(data_source, label), wrong)["reward"] == 0.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"<answer>\boxed{100\mathrm{KJ}}</answer>", r"100\mathrm{KJ}"),
        (r"<answer>plain answer</answer>", "plain answer"),
        (r"draft \boxed{wrong}; final \boxed{100KJ}", "100KJ"),
        (r"draft \fbox{100KJ}", "100KJ"),
        ("plain answer", None),
    ],
)
def test_stem_answer_extraction_matches_bp(response, expected):
    assert _extract_stem_answer(response) == expected


def test_stem_judge_uses_bp_prompt_and_northrl_async_request(monkeypatch):
    judge = _install_judge(monkeypatch, [_completion("Final Decision: Yes")])
    task = _task("stem_web", "100KJ", question="How do you convert bar/m^3 to KJ?")

    result = _score(task, r"<answer>\boxed{100KJ}</answer>")

    assert result["reward"] == 1.0
    request = judge.create.await_args.kwargs
    expected_prompt = _STEM_JUDGE_PROMPT.format(
        question=task["metadata"]["question"], reference="100KJ", student="100KJ"
    )
    assert request == {
        "model": "stem-judge",
        "messages": [{"role": "user", "content": expected_prompt}],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    settings = judge.constructor.call_args.kwargs
    assert settings["base_url"] == "http://judge.test/v1"
    assert settings["api_key"] == "_"
    assert settings["timeout"] == 180.0
    assert settings["max_retries"] == 0
    judge.http_constructor.assert_called_once_with(trust_env=False)
    judge.client.__aexit__.assert_awaited_once()


@pytest.mark.parametrize(
    ("content", "reasoning_content", "expected"),
    [
        ("Final Decision: Yes", None, 1.0),
        ("Final Decision: No", None, 0.0),
        ("No valid verdict", None, 0.0),
        (None, "Final Decision: Yes", 1.0),
    ],
)
def test_stem_verdict_and_reasoning_content(
    monkeypatch, caplog, content, reasoning_content, expected
):
    judge = _install_judge(monkeypatch, [_completion(content, reasoning_content)])
    result = _score(
        _task("stem_web", "100KJ", question="Question"),
        r"\boxed{100KJ}",
    )
    assert result["reward"] == expected
    assert judge.create.await_count == 1
    warnings = [
        record
        for record in caplog.records
        if record.name == reasoning360_module.__name__ and record.levelno == logging.WARNING
    ]
    assert len(warnings) == int(content == "No valid verdict")


@pytest.mark.parametrize(
    ("content", "warns"),
    [
        ("The student's answer appears correct.", True),
        ("Final Decision: YES", True),
        ("final decision: yes", True),
        ("Final Decision: Yesterday", True),
        ("Final Decision: Nope", True),
        ("Final Decision: Yes\nFinal Decision: No", True),
        ("Reasoning first. Final Decision: Yes.", False),
        ("Reasoning first. Final Decision: No.", False),
        ("Final Decision: Yes\nConfirmed: Final Decision: Yes", False),
    ],
)
def test_stem_verdict_warnings_do_not_change_substring_score_or_retry(
    monkeypatch, caplog, content, warns
):
    judge = _install_judge(monkeypatch, [_completion(content)])

    result = _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")

    # Even malformed Yes prefixes and conflicting verdicts retain the old score.
    expected = float("Final Decision: Yes" in content)
    assert result == {
        "reward": expected,
        "correct": bool(expected),
        "reason": "reasoning360/stem_web/" + ("pass" if expected else "fail"),
        "metrics": {"reasoning360_raw_score": expected},
    }
    assert judge.create.await_count == 1
    warnings = [
        record
        for record in caplog.records
        if record.name == reasoning360_module.__name__ and record.levelno == logging.WARNING
    ]
    assert len(warnings) == int(warns)
    if warnings:
        assert "no unambiguous Final Decision: Yes/No verdict" in warnings[0].getMessage()


def test_stem_no_verdict_warning_bounds_judge_response_preview(monkeypatch, caplog):
    text = "No verdict. " + "x" * 800 + "TAIL_MUST_NOT_BE_LOGGED"
    judge = _install_judge(monkeypatch, [_completion(text)])

    result = _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")

    assert result["reward"] == 0.0
    assert result["reason"] == "reasoning360/stem_web/fail"
    assert judge.create.await_count == 1
    assert "judge_model='stem-judge'" in caplog.text
    assert "TAIL_MUST_NOT_BE_LOGGED" not in caplog.text


def test_stem_without_answer_marker_does_not_call_judge(monkeypatch):
    judge = _install_judge(monkeypatch, [])
    result = _score(_task("stem_web", "100KJ", question="Question"), "plain answer")
    assert result["reward"] == 0.0
    judge.constructor.assert_not_called()


def test_stem_request_retries_rotated_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_URL", "http://judge-1.test/v1\nhttp://judge-2.test/v1")
    monkeypatch.setattr(reasoning360_module.random, "randrange", lambda count: count - 1)
    judge = _install_judge(monkeypatch, [_connection_error(), _completion("Final Decision: Yes")])

    result = _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")

    assert result["reward"] == 1.0
    assert [call.kwargs["base_url"] for call in judge.constructor.call_args_list] == [
        "http://judge-2.test/v1",
        "http://judge-1.test/v1",
    ]


@pytest.mark.parametrize("missing", ["LLM_JUDGE_URL", "LLM_JUDGE_MODEL"])
def test_stem_configuration_failure_warns_and_returns_zero(monkeypatch, caplog, missing):
    monkeypatch.delenv(missing)
    result = _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")
    assert result["reward"] == 0.0
    assert result["reason"].endswith("/judge_error")
    assert "assigning reward 0" in caplog.text


@pytest.mark.parametrize("base_timeout", [7.5, 180.0])
def test_stem_exhausted_requests_warn_and_return_zero(
    monkeypatch, caplog, judge_backoff, base_timeout
):
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_S", str(base_timeout))
    judge = _install_judge(monkeypatch, [_connection_error() for _ in range(5)])
    result = _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")
    assert result["reward"] == 0.0
    assert result["reason"].endswith("/judge_error")
    assert judge.create.await_count == 5
    assert "assigning reward 0" in caplog.text
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
@pytest.mark.parametrize("verdict", ["Final Decision: Yes", "Final Decision: No"])
def test_stem_bounded_backoff_preserves_request_and_score(
    monkeypatch, judge_backoff, failures, verdict
):
    judge = _install_judge(
        monkeypatch, [_connection_error() for _ in range(failures)] + [_completion(verdict)]
    )
    result = _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")

    assert result["reward"] == float(verdict.endswith("Yes"))
    assert judge.create.await_count == failures + 1
    assert judge_backoff.await_args_list == [call(1.5), call(3.0), call(6.0), call(12.0)][:failures]
    assert all(args == judge.create.await_args_list[0] for args in judge.create.await_args_list)
    assert all(args.kwargs["max_retries"] == 0 for args in judge.constructor.call_args_list)
    assert [args.kwargs["timeout"] for args in judge.constructor.call_args_list] == (
        [180.0] * min(failures + 1, 4) + ([300.0] if failures == 4 else [])
    )


def test_stem_authentication_error_keeps_zero_reward_policy(monkeypatch, judge_backoff):
    response = httpx.Response(401, request=httpx.Request("POST", "http://judge.test/v1"))
    failure = openai.AuthenticationError("invalid key", response=response, body=None)
    judge = _install_judge(monkeypatch, [failure])

    result = _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")
    assert result["reward"] == 0.0
    assert result["reason"].endswith("/judge_error")
    assert judge.create.await_count == 1
    judge_backoff.assert_not_awaited()


@pytest.mark.parametrize(
    ("stage", "previous_failures"),
    [("request", 0), ("request", 4), ("backoff", 0), ("backoff", 3)],
)
def test_stem_cancellation_is_not_converted_to_zero(
    monkeypatch, judge_backoff, stage, previous_failures
):
    final = asyncio.CancelledError() if stage == "request" else _connection_error()
    outcomes = [_connection_error() for _ in range(previous_failures)] + [final]
    judge = _install_judge(monkeypatch, outcomes)
    if stage == "backoff":
        judge_backoff.side_effect = [None] * previous_failures + [asyncio.CancelledError()]

    with pytest.raises(asyncio.CancelledError):
        _score(_task("stem_web", "answer", question="Question"), r"\boxed{answer}")
    assert judge.create.await_count == previous_failures + 1
    assert judge.client.__aexit__.await_count == previous_failures + 1
    assert judge_backoff.await_count == previous_failures + int(stage == "backoff")


@pytest.mark.parametrize(
    "task",
    [
        {"label": "answer", "metadata": {}},
        {"label": "answer", "metadata": {"data_source": ""}},
        {"label": None, "metadata": {"data_source": "logic__graph_logical_dataset"}},
        {"label": "answer", "metadata": {"data_source": "stem_web"}},
    ],
)
def test_invalid_data_contract_fails_fast(task):
    with pytest.raises((KeyError, TypeError, ValueError)):
        _score(task, "<answer>answer</answer>")


def test_samples_cover_all_seven_sources():
    assert len(_SAMPLE_TASKS) == 7
    assert len({sample["sample_id"] for sample in _SAMPLE_TASKS}) == 7
    assert all(isinstance(sample["prompt"], list) and sample["prompt"] for sample in _SAMPLE_TASKS)
    sources = {sample["metadata"]["data_source"] for sample in _SAMPLE_TASKS}
    assert sources == {
        "logic__graph_logical_dataset",
        "logic__zebra_puzzle_dataset",
        "logic__ordering_puzzle_dataset",
        "simulation__barc",
        "simulation__arcagi1",
        "simulation__arcagi2",
        "stem_web",
    }


@pytest.mark.parametrize(
    "sample",
    _SAMPLE_TASKS[:6],
    ids=[sample["metadata"]["data_source"] for sample in _SAMPLE_TASKS[:6]],
)
def test_local_samples_accept_labels_and_reject_controlled_wrong_answers(sample):
    source = sample["metadata"]["data_source"]
    label = sample["label"]
    if source == "logic__graph_logical_dataset":
        correct = f"<answer>{label}</answer>"
        wrong = "<answer>false</answer>"
    elif source == "logic__ordering_puzzle_dataset":
        correct = f"<answer>{label!r}</answer>"
        wrong = f"<answer>{list(reversed(label))!r}</answer>"
    elif source == "logic__zebra_puzzle_dataset":
        correct = f"<answer>{json.dumps(label)}</answer>"
        wrong_label = {**label, "rows": [["1", "dog"], ["2", "cat"]]}
        wrong = f"<answer>{json.dumps(wrong_label)}</answer>"
    else:
        correct = f"<answer>{label!r}</answer>"
        wrong_grid = [row[:] for row in label]
        wrong_grid[0][0] += 1
        wrong = f"<answer>{wrong_grid!r}</answer>"

    assert _score(sample, correct)["reward"] == 1.0
    assert _score(sample, wrong)["reward"] == 0.0


def test_stem_sample_uses_its_question_label_and_answer(monkeypatch):
    sample = _SAMPLE_TASKS[6]
    assert sample["metadata"]["data_source"] == "stem_web"
    observed = {}

    async def judge(question, student, reference):
        observed.update(question=question, student=student, reference=reference)
        return 1.0, False

    monkeypatch.setattr(reasoning360_module, "_judge_stem", judge)

    result = _score(sample, r"<answer>\boxed{20N}</answer>")

    assert result["reward"] == 1.0
    assert observed == {
        "question": sample["metadata"]["question"],
        "student": "20N",
        "reference": sample["label"],
    }
