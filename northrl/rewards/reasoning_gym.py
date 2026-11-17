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

"""Score Reasoning Gym tasks with its rule scorers and bp-compatible fixes."""

import logging
import re
from collections.abc import Callable, Mapping
from functools import cache
from importlib import metadata as importlib_metadata
from math import isfinite, isqrt
from numbers import Real
from typing import Any

logger = logging.getLogger(__name__)

_EXPECTED_REASONING_GYM_VERSION = "0.1.25"
_BOXED_MARKER = "\\boxed"
_ANSWER_TAG_START = "<answer>"
_ANSWER_TAG_END = "</answer>"
_RUSH_HOUR_MOVE = re.compile(r"([A-Z])([+-])(\d+)")


class ReasoningGymDependencyError(RuntimeError):
    """The pinned local Reasoning Gym rule runtime is unavailable."""


@cache
def _load_score_factory() -> Callable[[str], Callable[..., Any]]:
    """Load the pinned scorer factory only when this reward is selected."""
    try:
        installed_version = importlib_metadata.version("reasoning_gym")
        # pylint: disable-next=import-outside-toplevel
        from reasoning_gym import get_score_answer_fn  # type: ignore[import-untyped]
    except (ImportError, importlib_metadata.PackageNotFoundError) as error:
        raise ReasoningGymDependencyError(
            "rule_reasoning_gym requires reasoning-gym==0.1.25"
        ) from error

    if installed_version != _EXPECTED_REASONING_GYM_VERSION:
        raise ReasoningGymDependencyError(
            "rule_reasoning_gym requires reasoning-gym=="
            f"{_EXPECTED_REASONING_GYM_VERSION}, found {installed_version}"
        )
    return get_score_answer_fn


def _last_boxed_answer(response: str) -> str:
    """Return the body of the final balanced LaTeX ``\\boxed{...}``."""
    marker_index = response.rfind(_BOXED_MARKER)
    if marker_index < 0:
        return ""

    open_brace_index = response.find("{", marker_index + len(_BOXED_MARKER))
    if open_brace_index < 0:
        return ""

    depth = 0
    for index in range(open_brace_index, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[open_brace_index + 1 : index]
    return ""


def _extract_answer(response: str) -> str:
    """Apply bp-rl-judge's boxed, answer-tag, then full-response priority."""
    if _BOXED_MARKER in response:
        return _last_boxed_answer(response)
    if _ANSWER_TAG_START in response and _ANSWER_TAG_END in response:
        start = response.find(_ANSWER_TAG_START) + len(_ANSWER_TAG_START)
        end = response.find(_ANSWER_TAG_END, start)
        return response[start:end].strip()
    return response


def _arc_strict(answer: str, entry: Mapping[str, Any]) -> float:
    """Compare ARC output as an exact integer grid, avoiding the library tuple/list bug."""
    try:
        grid = [
            [int(cell) for cell in line.split()]
            for line in answer.strip().splitlines()
            if line.strip()
        ]
        return 1.0 if grid == entry["metadata"]["output"] else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


# The intentionally compact state transition mirrors bp-rl-judge's audited fix.
# pylint: disable=too-many-locals,too-many-return-statements,too-many-branches
def _rush_hour_sim(answer: str, entry: Mapping[str, Any]) -> float:
    """Simulate Rush Hour moves by vehicle label, avoiding the library index bug."""
    try:
        board_config = entry["metadata"]["board_config"]
        if not isinstance(board_config, str) or not board_config:
            return 0.0
        board_size = isqrt(len(board_config))
        if board_size * board_size != len(board_config):
            return 0.0

        cars: dict[str, list[tuple[int, int]]] = {}
        walls: set[tuple[int, int]] = set()
        for index, marker in enumerate(board_config):
            row, column = divmod(index, board_size)
            if marker == "o":
                continue
            if marker == "x":
                walls.add((row, column))
                continue
            cars.setdefault(marker, []).append((row, column))

        state: dict[str, tuple[tuple[int, int], int, str]] = {
            label: (
                min(positions),
                len(positions),
                "H" if len({row for row, _ in positions}) == 1 else "V",
            )
            for label, positions in cars.items()
        }

        def occupied_cells(label: str) -> list[tuple[int, int]]:
            (row, column), length, orientation = state[label]
            if orientation == "H":
                return [(row, column + offset) for offset in range(length)]
            return [(row + offset, column) for offset in range(length)]

        for label, sign, count_text in _RUSH_HOUR_MOVE.findall(answer):
            if label not in state:
                return 0.0
            direction = 1 if sign == "+" else -1
            for _ in range(int(count_text)):
                (row, column), length, orientation = state[label]
                occupied = {
                    cell: other_label
                    for other_label in state
                    for cell in occupied_cells(other_label)
                }
                if orientation == "H":
                    probe = (row, column + length) if direction == 1 else (row, column - 1)
                    new_position = (row, column + direction)
                else:
                    probe = (row + length, column) if direction == 1 else (row - 1, column)
                    new_position = (row + direction, column)

                if not (0 <= probe[0] < board_size and 0 <= probe[1] < board_size):
                    return 0.0
                if probe in walls or (probe in occupied and occupied[probe] != label):
                    return 0.0
                state[label] = (new_position, length, orientation)

        (_, target_column), target_length, target_orientation = state["A"]
        solved = target_orientation == "H" and target_column + target_length == board_size
        return 1.0 if solved else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


# pylint: enable=too-many-locals,too-many-return-statements,too-many-branches


def _evaluate_raw_score(
    answer: str,
    entry: Mapping[str, Any],
    source_dataset: str,
) -> tuple[Any, bool]:
    """Return ``(raw_score, scorer_error)`` for one answer."""
    if source_dataset in {"arc_agi", "rearc"}:
        return _arc_strict(answer, entry), False
    if source_dataset == "rush_hour":
        return _rush_hour_sim(answer, entry), False

    try:
        score_function = _load_score_factory()(source_dataset)
        return score_function(answer, entry), False
    except ReasoningGymDependencyError:
        raise
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Reasoning Gym scorer %r failed; assigning reward 0: %s: %s",
            source_dataset,
            type(error).__name__,
            error,
        )
        return 0.0, True


def _normalize_raw_score(raw_score: Any, source_dataset: str) -> float:
    if not isinstance(raw_score, Real):
        logger.warning(
            "Reasoning Gym scorer %r returned a non-numeric score %r; assigning reward 0",
            source_dataset,
            raw_score,
        )
        return 0.0
    score = float(raw_score)
    if not isfinite(score):
        logger.warning(
            "Reasoning Gym scorer %r returned a non-finite score %r; assigning reward 0",
            source_dataset,
            raw_score,
        )
        return 0.0
    return score


def rule_reasoning_gym_reward(
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Apply the selected Reasoning Gym rule and bp's final score policy."""
    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("Reasoning Gym response must be a string")

    entry = task["metadata"]
    if not isinstance(entry, Mapping):
        raise TypeError("Reasoning Gym metadata must be a mapping")
    task_metadata = entry.get("metadata")
    if not isinstance(task_metadata, Mapping):
        raise TypeError("Reasoning Gym metadata.metadata must be a mapping")
    source_dataset = task_metadata.get("source_dataset")
    if not isinstance(source_dataset, str) or not source_dataset.strip():
        raise ValueError("Reasoning Gym metadata.metadata.source_dataset must be non-empty")
    source_dataset = source_dataset.strip()

    answer = _extract_answer(response)
    raw_score, scorer_error = _evaluate_raw_score(answer, entry, source_dataset)
    score = _normalize_raw_score(raw_score, source_dataset)
    correct = score >= 0.999
    reward = score if source_dataset == "boxnet" else float(correct)
    if scorer_error:
        outcome = "scorer_error"
    else:
        outcome = "pass" if correct else "fail"
    return {
        "reward": reward,
        "correct": correct,
        "reason": f"reasoning_gym/{source_dataset}/{outcome}",
        "metrics": {"reasoning_gym_raw_score": score},
    }
