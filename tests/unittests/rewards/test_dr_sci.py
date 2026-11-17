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

import pytest

from northrl.rewards import REWARD_FNS, evaluate_reward
from northrl.rewards.dr_sci import rule_dr_sci_reward

PASS = {"reward": 1.0, "correct": True, "reason": "pass"}


def _score(response, label="B"):
    return rule_dr_sci_reward({"label": label}, {"response": response})


@pytest.mark.parametrize(
    ("response", "label"),
    [
        ("<answer>B</answer>", "B"),
        ("<answer>  B  </answer>", "B"),
        (r"<answer>\boxed{B}</answer>", "B"),
        (r"<answer>The final answer is: $\boxed{0.134}$</answer>", "0.134"),
        (r"<answer>\frac{1}{2}</answer>", "0.5"),
        (r"<answer>0.5</answer>", r"\boxed{\frac{1}{2}}"),
    ],
)
def test_supported_equivalent_answers_pass(response, label):
    assert _score(response, label) == PASS


def test_first_complete_answer_tag_wins():
    assert _score("<answer>B</answer>\n<answer>A</answer>") == PASS
    assert _score("<answer>A</answer>\n<answer>B</answer>") == {
        "reward": 0.0,
        "correct": False,
        "reason": "wrong_answer",
    }


@pytest.mark.parametrize(
    "response",
    [
        r"The final answer is \boxed{B}",
        r"Answer: \boxed{B}",
        "<answer></answer>",
        "<answer>   </answer>",
        "<answer>B",
        "B</answer>",
        "<ANSWER>B</ANSWER>",
        "</answer><answer>B",
    ],
)
def test_missing_or_malformed_answer_tag_returns_zero(response):
    assert _score(response) == {
        "reward": 0.0,
        "correct": False,
        "reason": "missing_answer",
    }


def test_wrong_answer_returns_zero():
    assert _score("<answer>A</answer>") == {
        "reward": 0.0,
        "correct": False,
        "reason": "wrong_answer",
    }


@pytest.mark.parametrize(
    ("task", "error"),
    [
        ({}, KeyError),
        ({"label": None}, TypeError),
        ({"label": 0.134}, TypeError),
        ({"label": ""}, ValueError),
        ({"label": "   "}, ValueError),
    ],
)
def test_invalid_label_fails_fast(task, error):
    with pytest.raises(error):
        rule_dr_sci_reward(task, {"response": "<answer>B</answer>"})


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({}, KeyError),
        ({"response": None}, TypeError),
        ({"response": ["<answer>B</answer>"]}, TypeError),
    ],
)
def test_invalid_response_fails_fast(result, error):
    with pytest.raises(error):
        rule_dr_sci_reward({"label": "B"}, result)


def test_reward_is_registered_and_normalized_by_dispatcher():
    assert REWARD_FNS["rule_dr_sci"] is rule_dr_sci_reward
    task = {"reward_fn": "rule_dr_sci", "label": "0.134"}

    assert asyncio.run(evaluate_reward(task, {"response": "<answer>0.134</answer>"})) == PASS


def test_debug_samples_follow_the_standard_contract_and_score_locally():
    samples_path = Path(__file__).parents[3] / "recipe/debug_rollout/samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    dr_sci_samples = [sample for sample in samples if sample.get("reward_fn") == "rule_dr_sci"]

    assert len(dr_sci_samples) == 2
    assert {sample["label"] for sample in dr_sci_samples} == {"B", "0.134"}
    for sample in dr_sci_samples:
        assert isinstance(sample["prompt"], list)
        assert isinstance(sample["label"], str) and sample["label"]
        assert sample["metadata"]["rm_type"] == "dr_sci"
        result = {
            "response": rf"<answer>The final answer is: $\boxed{{{sample['label']}}}$</answer>"
        }
        assert asyncio.run(evaluate_reward(sample, result)) == PASS
