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
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.rewards import REWARD_FNS, evaluate_reward
from northrl.rewards.math import MathRewardDataError, math_reward


@pytest.fixture
def compute_score(monkeypatch):
    from northrl.rewards import math

    score = AsyncMock(return_value=0.75)
    monkeypatch.setattr(math, "compute_score", score)
    return score


@pytest.mark.parametrize(
    "judge_mode",
    [
        "math_with_format",
        "math_answer_only",
        "math_llmjudge",
        "math_llm",
        "math_rule_llm",
        "math_rule_llm_answer_only",
        "math_boxed_llm",
        "tinker_aime_scalar_match",
    ],
)
def test_existing_modes_receive_sample_and_generation_fields(compute_score, judge_mode):
    prompt = [{"role": "user", "content": "What is 40 + 2?"}]
    task = {"label": "42", "prompt": prompt, "is_val": True}
    result = {"response": r"\boxed{42}", "prompt_str": "rendered prompt"}

    assert asyncio.run(math_reward(task, result, judge_mode=judge_mode)) == 0.75

    compute_score.assert_awaited_once_with(
        prompt_str="rendered prompt",
        solution_str=r"\boxed{42}",
        ground_truth="42",
        stage="val",
        raw_prompt=prompt,
        judge_mode=judge_mode,
    )
    assert compute_score.call_args.kwargs["raw_prompt"] is prompt


@pytest.mark.parametrize(
    "reward_fn",
    [
        "math_with_format",
        "math_answer_only",
        "math_llmjudge",
        "math_llm",
        "math_rule_llm",
        "math_boxed_llm",
        "tinker_aime_scalar_match",
    ],
)
def test_builtin_names_route_to_the_matching_judge(compute_score, reward_fn):
    task = {"reward_fn": reward_fn, "label": "42"}

    assert asyncio.run(evaluate_reward(task, {"response": "answer"})) == {"reward": 0.75}

    compute_score.assert_awaited_once()
    assert compute_score.call_args.kwargs["judge_mode"] == reward_fn
    assert not {
        "math",
        "rule",
        "slime_compatible",
        "llm_judge",
        "rule+llm_judge",
        "boxed_llm_judge",
        "aime",
    }.intersection(REWARD_FNS)


@pytest.mark.parametrize(
    ("answer_fields", "expected"),
    [
        ({"label": "42", "ground_truth": "wrong"}, "42"),
        ({"label": 0, "ground_truth": "wrong"}, 0),
        ({"label": "", "reward_model": {"ground_truth": "wrong"}}, ""),
        ({"label": None, "ground_truth": "42"}, "42"),
        ({"ground_truth": "42", "reward_model": {"ground_truth": "wrong"}}, "42"),
        ({"ground_truth": "", "reward_model": {"ground_truth": "42"}}, "42"),
        ({"reward_model": {"ground_truth": "42"}}, "42"),
        ({"label": None, "ground_truth": None, "reward_model": None}, ""),
    ],
)
def test_label_precedence_and_legacy_answers(compute_score, answer_fields, expected):
    asyncio.run(math_reward(answer_fields, {"response": "answer"}))

    assert compute_score.call_args.kwargs["ground_truth"] == expected
    assert compute_score.call_args.kwargs["stage"] == "train"
    assert compute_score.call_args.kwargs["prompt_str"] == ""
    assert compute_score.call_args.kwargs["judge_mode"] == "math_with_format"


def test_array_prompt_is_converted_for_existing_llm_judges(compute_score):
    messages = [{"role": "user", "content": "question"}]
    prompt = Mock(tolist=Mock(return_value=messages))

    asyncio.run(math_reward({"label": "42", "prompt": prompt}, {"response": "answer"}))

    assert compute_score.call_args.kwargs["raw_prompt"] is messages


@pytest.mark.parametrize(("random_value", "expected"), [(0.25, 1.0), (0.75, 0.0)])
def test_mock_data_keeps_binary_random_reward(monkeypatch, compute_score, random_value, expected):
    monkeypatch.setattr("northrl.rewards.math.random.random", lambda: random_value)

    assert asyncio.run(math_reward({"ground_truth": "whatever"}, {})) == expected
    compute_score.assert_not_awaited()


@pytest.mark.parametrize(
    ("response", "expected_by_mode"),
    [
        (r"The answer is \boxed{42}", (1.0, 1.0, 1.0)),
        (r"The answer is \boxed{41}", (0.1, 0.0, 0.0)),
        (r"<think>unfinished reasoning \boxed{42}", (0.0, 1.0, 1.0)),
        ("The answer is 42", (0.0, 0.0, 1.0)),
    ],
)
def test_existing_rule_scores_are_preserved(response, expected_by_mode):
    task = {"label": "42", "prompt": [{"role": "user", "content": "What is 40 + 2?"}]}
    modes = ("math_with_format", "math_answer_only", "tinker_aime_scalar_match")

    scores = tuple(
        asyncio.run(math_reward(task, {"response": response}, judge_mode=mode)) for mode in modes
    )

    assert scores == expected_by_mode


def test_judge_failure_is_not_replaced_by_a_score(compute_score):
    failure = RuntimeError("judge unavailable")
    compute_score.side_effect = failure

    with pytest.raises(RuntimeError) as error:
        asyncio.run(math_reward({"label": "42"}, {"response": "answer"}))

    assert error.value is failure


def test_new_math_reward_name_is_registered(compute_score):
    task = {
        "reward_fn": "math_rule_llm_answer_only",
        "prompt": [{"role": "user", "content": "What is 40 + 2?"}],
        "label": "42",
    }

    assert asyncio.run(evaluate_reward(task, {"response": r"\boxed{42}"})) == {"reward": 0.75}
    assert compute_score.call_args.kwargs["judge_mode"] == "math_rule_llm_answer_only"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (42, "42"),
        (4.5, "4.5"),
        ([42, "forty-two"], ["42", "forty-two"]),
    ],
)
def test_new_math_reward_normalizes_supported_labels_at_the_boundary(
    compute_score,
    label,
    expected,
):
    task = {
        "prompt": [{"role": "user", "content": "What is the answer?"}],
        "label": label,
    }

    asyncio.run(
        math_reward(
            task,
            {"response": r"\boxed{42}"},
            judge_mode="math_rule_llm_answer_only",
        )
    )

    assert compute_score.call_args.kwargs["ground_truth"] == expected


@pytest.mark.parametrize(
    "response",
    [
        r"First attempt: \boxed{41}. Final answer: \boxed{42}",
        r"<think>reasoning \boxed{41}</think>Final answer: \boxed{42}",
        r"###Response old \boxed{41}</think>new \boxed{42}",
        r"</think>old \boxed{41}###Response final \boxed{42}",
    ],
)
def test_new_math_reward_uses_last_boxed_in_the_final_answer_region(response):
    task = {
        "prompt": [{"role": "user", "content": "What is 40 + 2?"}],
        "label": r"\boxed{42}",
    }

    score = asyncio.run(
        math_reward(task, {"response": response}, judge_mode="math_rule_llm_answer_only")
    )

    assert score == 1.0


@pytest.mark.parametrize(
    "task",
    [
        {"label": "42"},
        {"prompt": [], "label": "42"},
        {"prompt": [{"role": "user", "content": ""}], "label": "42"},
        {"prompt": [{"role": "assistant", "content": "question"}], "label": "42"},
        {"prompt": [{"role": "user", "content": "question"}]},
        {"prompt": [{"role": "user", "content": "question"}], "label": None},
        {"prompt": [{"role": "user", "content": "question"}], "label": " "},
        {"prompt": [{"role": "user", "content": "question"}], "label": " Whatever "},
        {"prompt": [{"role": "user", "content": "question"}], "label": {}},
    ],
)
def test_new_math_reward_rejects_invalid_standard_data(task):
    with pytest.raises(MathRewardDataError):
        asyncio.run(
            math_reward(
                task,
                {"response": r"\boxed{42}"},
                judge_mode="math_rule_llm_answer_only",
            )
        )


def test_debug_math_samples_are_valid_and_locally_judgeable(monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_URL", raising=False)
    samples_path = Path(__file__).parents[3] / "recipe/debug_rollout/samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    math_samples = [
        sample for sample in samples if sample.get("reward_fn") == "math_rule_llm_answer_only"
    ]

    assert len(math_samples) == 4
    for sample in math_samples:
        result = {"response": rf"The answer is \boxed{{{sample['label']}}}"}
        assert asyncio.run(evaluate_reward(sample, result)) == {"reward": 1.0}
