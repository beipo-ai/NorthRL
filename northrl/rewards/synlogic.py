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

"""Route and normalize the 35 local rule verifiers used by SynLogic."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Any

from ._synlogic.base import SynLogicData, SynLogicDependencyError
from ._synlogic.registry import get_verifier

logger = logging.getLogger(__name__)

_BOXED_MARKER = "\\boxed"
_ANSWER_START = "<answer>"
_ANSWER_END = "</answer>"


def _last_boxed_answer(response: str) -> str:
    marker_index = response.rfind(_BOXED_MARKER)
    if marker_index < 0:
        return ""
    open_brace = response.find("{", marker_index + len(_BOXED_MARKER))
    if open_brace < 0:
        return ""
    depth = 0
    for index in range(open_brace, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[open_brace + 1 : index].strip()
    return ""


def _extract_answer(response: str) -> str:
    """Match bp's boxed-first, answer-tag-second, full-response policy."""
    if _BOXED_MARKER in response:
        return _last_boxed_answer(response)
    start = response.find(_ANSWER_START)
    if start >= 0:
        start += len(_ANSWER_START)
        end = response.find(_ANSWER_END, start)
        if end >= 0:
            return response[start:end].strip()
    return response


def _raw_correct(raw_score: Any) -> bool:
    if raw_score is True:
        return True
    if raw_score is False:
        return False
    if not isinstance(raw_score, Real):
        raise TypeError(f"SynLogic verifier returned non-numeric score: {raw_score!r}")
    score = float(raw_score)
    if not isfinite(score):
        raise ValueError(f"SynLogic verifier returned non-finite score: {raw_score!r}")
    return score >= 0.999


def _verify_once(
    verifier: Any, game_data: SynLogicData, answer: str, data_source: str
) -> tuple[Any, bool]:
    """Run one representation, converting unexpected verifier failures to zero."""
    try:
        return verifier.verify(game_data, answer), False
    except SynLogicDependencyError:
        raise
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.warning(
            "SynLogic verifier %r failed; assigning reward 0: %s: %s",
            data_source,
            type(error).__name__,
            error,
        )
        return 0.0, True


def synlogic_reward(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Apply one task-specific SynLogic rule and return strict binary reward."""
    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("SynLogic response must be a string")

    metadata = task["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("SynLogic metadata must be a mapping")
    data_source = metadata.get("data_source")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("SynLogic metadata.data_source must be a non-empty string")
    data_source = data_source.strip()
    game_data_str = metadata.get("game_data_str")
    if not isinstance(game_data_str, str) or not game_data_str.strip():
        raise ValueError("SynLogic metadata.game_data_str must be a non-empty string")

    verifier = get_verifier(data_source)
    game_data = SynLogicData.from_json_str(game_data_str)
    extracted = _extract_answer(response)

    raw_score, verifier_error = _verify_once(verifier, game_data, extracted, data_source)
    if not _raw_correct(raw_score):
        raw_score, second_error = _verify_once(verifier, game_data, response, data_source)
        verifier_error = verifier_error or second_error
    correct = _raw_correct(raw_score)
    if correct:
        reason_suffix = "pass"
    else:
        reason_suffix = "verifier_error" if verifier_error else "fail"

    score = float(raw_score) if isinstance(raw_score, Real) and isfinite(raw_score) else 0.0
    return {
        "reward": float(correct),
        "correct": correct,
        "reason": f"synlogic/{data_source}/{reason_suffix}",
        "metrics": {"synlogic_raw_score": score},
    }
