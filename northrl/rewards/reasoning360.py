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

"""Score the five verifier families used by the Reasoning360 dataset."""

import ast
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Mapping
from typing import Any

from .llm_judge_retry import MAX_JUDGE_ATTEMPTS as _MAX_JUDGE_ATTEMPTS
from .llm_judge_retry import judge_timeout_for_attempt, wait_for_judge_retry

logger = logging.getLogger(__name__)

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>")
_GRID_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_GRAPH_ANSWER_PATTERN = re.compile(r"[A-Za-z]+")
_STEM_VERDICT_PATTERN = re.compile(r"Final Decision: (Yes|No)\b")
_DEFAULT_JUDGE_TIMEOUT_S = 180.0
_JUDGE_MAX_TOKENS = 8192

_SOURCE_TO_VERIFIER = {
    "logic__graph_logical_dataset": "graph",
    "logic__zebra_puzzle_dataset": "zebra",
    "logic__ordering_puzzle_dataset": "ordering",
    "simulation__barc": "arc_grid",
    "simulation__arcagi1": "arc_grid",
    "simulation__arcagi2": "arc_grid",
    "stem_web": "stem_web",
}

_STEM_JUDGE_PROMPT = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {reference}\n\n"
    "### Student Answer: {student}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the "
    "ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent "
    "to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's "
    'answer is incorrect, output "Final Decision: No". Assistant:'
)


def _last_answer_tag(response: str, *, multiline: bool = False) -> str | None:
    pattern = _GRID_ANSWER_PATTERN if multiline else _ANSWER_PATTERN
    matches = list(pattern.finditer(response))
    return matches[-1].group(1).strip() if matches else None


def _score_graph(response: str, label: Any) -> float:
    if not isinstance(label, str):
        raise TypeError("Reasoning360 graph label must be a string")
    answer = _last_answer_tag(response)
    if answer is None or _GRAPH_ANSWER_PATTERN.fullmatch(answer) is None:
        return 0.0
    return float(answer.casefold() == label.casefold())


def _validate_zebra_label(label: Any) -> tuple[list[Any], list[list[Any]]]:
    if not isinstance(label, Mapping):
        raise TypeError("Reasoning360 zebra label must be a mapping")
    header = label.get("header")
    rows = label.get("rows")
    if not isinstance(header, list) or not header:
        raise ValueError("Reasoning360 zebra label.header must be a non-empty list")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reasoning360 zebra label.rows must be a non-empty list")
    if not all(isinstance(row, list) and len(row) == len(header) for row in rows):
        raise ValueError("Reasoning360 zebra label.rows must be rectangular")
    return header, rows


def _parse_mapping(text: str) -> Mapping[str, Any] | None:
    for parser in (ast.literal_eval, json.loads):
        try:
            value = parser(text)
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _score_zebra(response: str, label: Any) -> float:
    header, expected_rows = _validate_zebra_label(label)
    answer = _last_answer_tag(response)
    if answer is None:
        return 0.0
    prediction = _parse_mapping(answer)
    if prediction is None:
        return 0.0
    try:
        predicted_rows = prediction["rows"]
        correct = sum(
            predicted_rows[row][column] == expected_rows[row][column]
            for row in range(len(expected_rows))
            for column in range(len(header))
        )
    except (KeyError, IndexError, TypeError):
        return 0.0
    return correct / (len(expected_rows) * len(header))


def _parse_ordering(text: str) -> list[str] | None:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        if not text.startswith("[") or not text.endswith("]"):
            return None
        value = [part.strip().strip("'\"") for part in text[1:-1].split(",")]
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        return None
    normalized = [item.strip().casefold() for item in value]
    return normalized if all(normalized) else None


def _score_ordering(response: str, label: Any) -> float:
    if not isinstance(label, list) or not label or not all(isinstance(item, str) for item in label):
        raise TypeError("Reasoning360 ordering label must be a non-empty list of strings")
    answer = _last_answer_tag(response)
    prediction = _parse_ordering(answer) if answer is not None else None
    expected = [item.strip().casefold() for item in label]
    return float(prediction == expected)


def _balanced_grid_literal(text: str) -> str | None:
    start = text.find("[[")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _valid_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], list)
        and bool(value[0])
        and all(isinstance(row, list) and len(row) == len(value[0]) for row in value)
        and all(
            isinstance(cell, int) and not isinstance(cell, bool) for row in value for cell in row
        )
    )


def _score_arc_grid(response: str, label: Any) -> float:
    if not _valid_grid(label):
        raise TypeError("Reasoning360 ARC/BARC label must be a rectangular integer grid")
    answer = _last_answer_tag(response, multiline=True)
    if answer is None:
        return 0.0
    literal = _balanced_grid_literal(answer.replace("...", "-1"))
    if literal is None:
        return 0.0
    try:
        prediction = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return 0.0
    return float(_valid_grid(prediction) and prediction == label)


def _last_boxed_answer(text: str) -> str | None:
    marker_index = text.rfind("\\boxed")
    if marker_index < 0:
        marker_index = text.rfind("\\fbox")
    if marker_index < 0:
        return None
    open_brace = text.find("{", marker_index)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index].strip()
    return None


def _extract_stem_answer(response: str) -> str | None:
    if "<answer>" in response and "</answer>" in response:
        tagged = response.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        return _last_boxed_answer(tagged) or tagged
    return _last_boxed_answer(response)


def _judge_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if not content:
        content = getattr(message, "reasoning_content", None)
    if not content:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, Mapping):
            content = extra.get("reasoning_content")
    if not isinstance(content, str):
        raise ValueError("Reasoning360 judge response has no text content")
    return content


def _judge_configuration() -> tuple[list[str], str, str, float]:
    urls = [url.strip() for url in os.environ.get("LLM_JUDGE_URL", "").splitlines() if url.strip()]
    if not urls:
        raise ValueError("LLM_JUDGE_URL must contain at least one judge base URL")
    model = os.environ.get("LLM_JUDGE_MODEL", "").strip()
    if not model:
        raise ValueError("LLM_JUDGE_MODEL must name the judge model")
    try:
        timeout = float(os.environ.get("LLM_JUDGE_TIMEOUT_S", str(_DEFAULT_JUDGE_TIMEOUT_S)))
    except ValueError as error:
        raise ValueError("LLM_JUDGE_TIMEOUT_S must be a number") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("LLM_JUDGE_TIMEOUT_S must be a positive finite number")
    return urls, model, os.environ.get("LLM_JUDGE_API_KEY", "_"), timeout


async def _judge_stem(question: str, student: str, reference: str) -> tuple[float, bool]:
    """Return ``(score, infrastructure_error)`` using bp's prompt and verdict."""
    import openai  # pylint: disable=import-outside-toplevel

    try:
        urls, model, api_key, timeout = _judge_configuration()
        offset = random.randrange(len(urls))
        prompt = _STEM_JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            student=student,
        )
        for attempt in range(_MAX_JUDGE_ATTEMPTS):
            attempt_timeout = judge_timeout_for_attempt(timeout, attempt + 1)
            attempt_started = time.monotonic()
            try:
                async with openai.AsyncOpenAI(
                    base_url=urls[(offset + attempt) % len(urls)],
                    api_key=api_key,
                    timeout=attempt_timeout,
                    max_retries=0,
                    http_client=openai.DefaultAsyncHttpxClient(trust_env=False),
                ) as client:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=_JUDGE_MAX_TOKENS,
                    )
                text = _judge_message_text(completion.choices[0].message)
                # Diagnose missing/contradictory complete verdict tokens without
                # changing bp's original substring-based score or retry behavior.
                if len(set(_STEM_VERDICT_PATTERN.findall(text))) != 1:
                    logger.warning(
                        "Reasoning360 STEM judge returned no unambiguous Final Decision: Yes/No "
                        "verdict; retaining existing scoring behavior; judge_model=%r; "
                        "judge_response_preview=%r",
                        model,
                        text[:512],
                    )
                return float("Final Decision: Yes" in text), False
            except Exception as error:  # pylint: disable=broad-exception-caught
                if not await wait_for_judge_retry(
                    error,
                    failed_attempt=attempt + 1,
                    judge_name="Reasoning360 STEM",
                    logger=logger,
                    elapsed_s=time.monotonic() - attempt_started,
                    timeout_s=attempt_timeout,
                    context={"judge_model": model},
                ):
                    raise
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Reasoning360 STEM judge failed; assigning reward 0: %s: %s",
            type(error).__name__,
            error,
        )
        return 0.0, True
    raise AssertionError("unreachable")


async def reasoning360_reward(
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Route one Reasoning360 row to its audited local or LLM verifier."""
    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("Reasoning360 response must be a string")
    metadata = task["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("Reasoning360 metadata must be a mapping")
    data_source = metadata.get("data_source")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("Reasoning360 metadata.data_source must be non-empty")
    data_source = data_source.strip()
    try:
        verifier = _SOURCE_TO_VERIFIER[data_source]
    except KeyError as error:
        raise ValueError(f"Unsupported Reasoning360 data_source: {data_source!r}") from error
    if "label" not in task or task["label"] is None:
        raise ValueError("Reasoning360 sample requires a non-null top-level label")
    label = task["label"]

    judge_error = False
    if verifier == "graph":
        raw_score = _score_graph(response, label)
    elif verifier == "zebra":
        raw_score = _score_zebra(response, label)
    elif verifier == "ordering":
        raw_score = _score_ordering(response, label)
    elif verifier == "arc_grid":
        raw_score = _score_arc_grid(response, label)
    else:
        if not isinstance(label, str) or not label.strip():
            raise TypeError("Reasoning360 stem_web label must be a non-empty string")
        question = metadata.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Reasoning360 stem_web metadata.question must be non-empty")
        answer = _extract_stem_answer(response)
        if answer is None:
            raw_score = 0.0
        else:
            raw_score, judge_error = await _judge_stem(question, answer, label)

    if not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score):
        raise ValueError(f"Reasoning360 verifier returned an invalid score: {raw_score!r}")
    correct = raw_score >= 0.999
    if judge_error:
        outcome = "judge_error"
    else:
        outcome = "pass" if correct else "fail"
    return {
        "reward": float(correct),
        "correct": correct,
        "reason": f"reasoning360/{data_source}/{outcome}",
        "metrics": {"reasoning360_raw_score": float(raw_score)},
    }
