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

"""Score Nemotron multiple-choice answers with its required answer markers."""

from typing import Any

_ANSWER_TAG_START = "<answer>"
_ANSWER_TAG_END = "</answer>"
_ANSWER_MARKER = "answer:"
_EDGE_PUNCTUATION = " \t\n\r.。;；:：'\"()[]{}" + "\x60"


def _find_matching_brace(text: str, open_brace_index: int) -> int:
    depth = 0
    for index in range(open_brace_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _extract_boxed_body(text: str) -> str | None:
    marker_index = text.find("\\boxed")
    if marker_index < 0:
        marker_index = text.find(r"\\boxed")
    if marker_index < 0:
        return None

    open_brace_index = text.find("{", marker_index)
    if open_brace_index < 0:
        return None
    close_brace_index = _find_matching_brace(text, open_brace_index)
    if close_brace_index < 0:
        return None
    return text[open_brace_index + 1 : close_brace_index].strip()


def _extract_answer_tag(text: str) -> str | None:
    start_index = text.rfind(_ANSWER_TAG_START)
    end_index = text.rfind(_ANSWER_TAG_END)
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        return None
    return text[start_index + len(_ANSWER_TAG_START) : end_index].strip()


def _extract_answer_segment(text: str) -> str:
    tagged_answer = _extract_answer_tag(text)
    if tagged_answer is not None:
        return tagged_answer

    marker_index = text.lower().rfind(_ANSWER_MARKER)
    if marker_index < 0:
        return ""
    return text[marker_index + len(_ANSWER_MARKER) :].strip()


def _normalize_prediction(response: str) -> str:
    answer = _extract_answer_segment(response)
    boxed_body = _extract_boxed_body(answer)
    if boxed_body is not None:
        answer = boxed_body
    return answer.strip().upper().replace("\\", "").strip(_EDGE_PUNCTUATION).lower()


def rule_nemotron_mcqa_reward(
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Compare the response's declared final choice with the sample label."""
    label = task["label"]
    if not isinstance(label, str):
        raise TypeError("Nemotron MCQA label must be a string")
    if not label.strip():
        raise ValueError("Nemotron MCQA label must not be empty")

    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("Nemotron MCQA response must be a string")

    prediction = _normalize_prediction(response)
    correct = bool(prediction) and prediction == label.lower()
    if correct:
        reason = "pass"
    elif not prediction:
        reason = "missing_answer"
    else:
        reason = "wrong_choice"
    return {
        "reward": float(correct),
        "correct": correct,
        "reason": reason,
    }
