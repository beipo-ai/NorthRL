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

"""Score DR-SCI answers with its answer-tag and math-equivalence contract."""

from typing import Any

from .evaluation_utils.deepscaler_utlis import grade_answer_verl

_ANSWER_TAG_START = "<answer>"
_ANSWER_TAG_END = "</answer>"


def _extract_answer_tag(response: str) -> str | None:
    """Return the first correctly closed, non-empty answer-tag body."""
    start_index = response.find(_ANSWER_TAG_START)
    if start_index < 0:
        return None
    start_index += len(_ANSWER_TAG_START)

    end_index = response.find(_ANSWER_TAG_END, start_index)
    if end_index < 0:
        return None

    answer = response[start_index:end_index].strip()
    return answer or None


def rule_dr_sci_reward(
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Compare the first tagged final answer with the sample's reference label."""
    label = task["label"]
    if not isinstance(label, str):
        raise TypeError("DR-SCI label must be a string")
    if not label.strip():
        raise ValueError("DR-SCI label must not be empty")

    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("DR-SCI response must be a string")

    answer = _extract_answer_tag(response)
    if answer is None:
        return {
            "reward": 0.0,
            "correct": False,
            "reason": "missing_answer",
        }

    correct = grade_answer_verl(f"\\boxed{{{answer}}}", label)
    return {
        "reward": float(correct),
        "correct": correct,
        "reason": "pass" if correct else "wrong_answer",
    }
