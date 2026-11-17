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

import pytest

from northrl.rewards import REWARD_FNS, evaluate_reward
from northrl.rewards.nemotron_mcqa import rule_nemotron_mcqa_reward

PASS = {"reward": 1.0, "correct": True, "reason": "pass"}


def _score(response, label="D"):
    return rule_nemotron_mcqa_reward({"label": label}, {"response": response})


@pytest.mark.parametrize(
    "response",
    [
        "Answer: D",
        "answer: d",
        "AnSwEr: (D).",
        "<answer>D</answer>",
        r"Answer: \boxed{D}",
        r"Answer: \\boxed{D}",
        r"<answer>\boxed{D}</answer>",
        r"Answer: \boxed{{D}}",
        r"Answer: \boxed{D}trailing text",
    ],
)
def test_supported_answer_formats_pass(response):
    assert _score(response) == PASS


def test_last_answer_marker_wins():
    assert _score("Answer: A\nAfter checking again, Answer: D") == PASS


def test_last_complete_answer_tag_wins():
    assert _score("<answer>A</answer>\nAfter checking again, <answer>D</answer>") == PASS


def test_answer_tag_has_bp_compatible_priority_over_answer_marker():
    assert _score("<answer>D</answer>\nAnswer: A") == PASS


def test_first_boxed_answer_in_selected_segment_wins():
    assert _score(r"Answer: \boxed{D}, not \boxed{A}") == PASS


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (r"The final choice is \boxed{D}", "missing_answer"),
        ("Answer:", "missing_answer"),
        ("<answer></answer>", "missing_answer"),
        ("<answer>D", "missing_answer"),
        ("Answer: A", "wrong_choice"),
        ("Answer: D because it is correct", "wrong_choice"),
        (r"Answer: \boxed{D", "wrong_choice"),
    ],
)
def test_unsupported_or_wrong_answers_return_zero(response, reason):
    assert _score(response) == {
        "reward": 0.0,
        "correct": False,
        "reason": reason,
    }


def test_label_comparison_is_case_insensitive():
    assert _score("Answer: D", label="d") == PASS


@pytest.mark.parametrize(
    ("task", "error"),
    [
        ({}, KeyError),
        ({"label": None}, TypeError),
        ({"label": 4}, TypeError),
        ({"label": ""}, ValueError),
        ({"label": "   "}, ValueError),
    ],
)
def test_invalid_label_fails_fast(task, error):
    with pytest.raises(error):
        rule_nemotron_mcqa_reward(task, {"response": "Answer: D"})


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({}, KeyError),
        ({"response": None}, TypeError),
        ({"response": ["Answer: D"]}, TypeError),
    ],
)
def test_invalid_response_fails_fast(result, error):
    with pytest.raises(error):
        rule_nemotron_mcqa_reward({"label": "D"}, result)


def test_reward_is_registered_and_normalized_by_dispatcher():
    assert REWARD_FNS["rule_nemotron_mcqa"] is rule_nemotron_mcqa_reward
    task = {"reward_fn": "rule_nemotron_mcqa", "label": "D"}

    assert asyncio.run(evaluate_reward(task, {"response": r"Answer: \boxed{D}"})) == PASS


@pytest.mark.parametrize(
    ("sample", "wrong_choice"),
    [
        pytest.param(
            {
                "prompt": [{"role": "user", "content": "What is 2 + 2? A. 1 B. 2 C. 3 D. 4"}],
                "label": "D",
                "reward_fn": "rule_nemotron_mcqa",
                "metadata": {"rm_type": "nemotron_mcqa"},
            },
            "A",
            id="addition",
        ),
        pytest.param(
            {
                "prompt": [{"role": "user", "content": "What is 3 * 3? A. 6 B. 9 C. 12 D. 15"}],
                "label": "B",
                "reward_fn": "rule_nemotron_mcqa",
                "metadata": {"rm_type": "nemotron_mcqa"},
            },
            "D",
            id="multiplication",
        ),
    ],
)
def test_mcqa_samples_follow_the_standard_contract_and_score_locally(sample, wrong_choice):
    assert isinstance(sample["prompt"], list)
    assert isinstance(sample["label"], str) and sample["label"]
    assert sample["metadata"]["rm_type"] == "nemotron_mcqa"
    assert wrong_choice in "ABCD" and wrong_choice != sample["label"]

    result = {"response": rf"Answer: \boxed{{{sample['label']}}}"}
    assert asyncio.run(evaluate_reward(sample, result)) == PASS
    assert asyncio.run(evaluate_reward(sample, {"response": f"Answer: {wrong_choice}"})) == {
        "reward": 0.0,
        "correct": False,
        "reason": "wrong_choice",
    }
