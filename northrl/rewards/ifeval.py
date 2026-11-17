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

"""Score instruction-following responses with vendored IFEvalG rules."""

import logging
from collections.abc import Mapping
from functools import cache
from threading import Lock
from typing import Any

LOGGER = logging.getLogger(__name__)
_INSTRUCTION_REGISTRY_LOCK = Lock()

_EOS_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")
_PREPARE_NLTK_DATA = (
    "Set NLTK_DATA to a persistent directory and run "
    '`python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab`.'
)


class IFEvalDependencyError(RuntimeError):
    """The local IFEval rule runtime is not installed or prepared."""


def _strip_trailing_eos(text: str) -> str:
    """Strip whitespace and repeated trailing stop tokens without changing the body."""
    end = len(text)
    while end and text[end - 1].isspace():
        end -= 1

    while end:
        for token in _EOS_TOKENS:
            start = end - len(token)
            if start >= 0 and text.startswith(token, start, end):
                end = start
                while end and text[end - 1].isspace():
                    end -= 1
                break
        else:
            break
    return text[:end].lstrip()


def _answer_text(response: str) -> str:
    """Select the final-answer region using the bp-rl-judge cleanup contract."""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    elif "###Response" in response:
        response = response.rsplit("###Response", 1)[-1]
    return _strip_trailing_eos(response)


@cache
def _instruction_registry() -> Mapping[str, type]:
    """Load the rule registry once and fail clearly when local dependencies are absent."""
    # functools.cache allows concurrent cache misses to execute this function.
    # langdetect publishes its global factory before loading the language
    # profiles, so every first scorer must wait for initialization to finish.
    with _INSTRUCTION_REGISTRY_LOCK:
        try:
            import nltk  # type: ignore[import-untyped]
            from langdetect import DetectorFactory, detector_factory  # type: ignore[import-untyped]

            from .ifevalg.instructions_registry import INSTRUCTION_DICT
        except ImportError as error:
            raise IFEvalDependencyError(
                "IFEval requires nltk==3.9.4 and langdetect==1.0.9"
            ) from error

        DetectorFactory.seed = 0
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("tokenizers/punkt_tab/english")
            nltk.data.load("nltk:tokenizers/punkt/english.pickle")
        except LookupError as error:
            raise IFEvalDependencyError(
                f"IFEval NLTK data is unavailable. {_PREPARE_NLTK_DATA}"
            ) from error
        detector_factory.init_factory()
        return INSTRUCTION_DICT


def _result(*, total: int, failed: int, unknown: int, reason: str) -> dict[str, Any]:
    correct = total > 0 and failed == 0 and unknown == 0
    return {
        "reward": float(correct),
        "correct": correct,
        "reason": reason,
        "metrics": {
            "ifeval_constraints_total": float(total),
            "ifeval_constraints_failed": float(failed),
            "ifeval_constraints_unknown": float(unknown),
        },
    }


def ifeval_reward(
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return one only when every structured IFEval constraint passes."""
    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("IFEval response must be a string")

    metadata = task["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("IFEval metadata must be a mapping")

    instructions = metadata.get("instructions")
    if instructions is None or instructions == []:
        return _result(total=0, failed=0, unknown=0, reason="no_instructions")
    if not isinstance(instructions, list):
        raise TypeError("IFEval metadata.instructions must be a list")

    registry = _instruction_registry()
    from .ifevalg.instructions import language_check_context

    answer = _answer_text(response)
    prompt_text = metadata.get("prompt_text")
    failed_count = 0
    unknown_count = 0

    for index, specification in enumerate(instructions):
        if not isinstance(specification, Mapping):
            raise TypeError(f"IFEval instruction at index {index} must be a mapping")

        instruction_id = specification.get("id")
        if not isinstance(instruction_id, str) or not instruction_id.strip():
            raise ValueError(f"IFEval instruction at index {index} requires a non-empty id")

        checker_class = registry.get(instruction_id)
        if checker_class is None:
            unknown_count += 1
            LOGGER.warning("Unknown IFEval verifier id %r", instruction_id)
            continue

        raw_kwargs = specification.get("kwargs")
        if raw_kwargs is None:
            raw_kwargs = {}
        if not isinstance(raw_kwargs, Mapping):
            failed_count += 1
            LOGGER.warning("IFEval verifier %r kwargs must be a mapping", instruction_id)
            continue

        try:
            checker = checker_class(instruction_id)
            argument_keys = checker.get_instruction_args_keys() or []
            missing_keys = [
                key
                for key in argument_keys
                if key != "prompt" and (key not in raw_kwargs or raw_kwargs[key] is None)
            ]
            if missing_keys:
                failed_count += 1
                LOGGER.warning(
                    "IFEval verifier %r is missing required kwargs: %s",
                    instruction_id,
                    ", ".join(missing_keys),
                )
                continue

            checker_kwargs = {key: value for key, value in raw_kwargs.items() if value is not None}
            if "prompt" in argument_keys:
                if not isinstance(prompt_text, str):
                    failed_count += 1
                    LOGGER.warning(
                        "IFEval verifier %r requires metadata.prompt_text", instruction_id
                    )
                    continue
                checker_kwargs["prompt"] = prompt_text

            checker.configure(checker_kwargs)
            with language_check_context(
                {
                    "instruction_id": instruction_id,
                    "task_id": task.get("task_id"),
                    "group_id": task.get("group_id"),
                    "finish_reason": result.get("finish_reason"),
                    "response": response,
                    "answer": answer,
                }
            ):
                if not bool(checker.check_following(answer)):
                    failed_count += 1
        except (IndexError, KeyError) as error:
            failed_count += 1
            LOGGER.warning(
                "IFEval verifier %r rejected malformed response data with %s: %s",
                instruction_id,
                type(error).__name__,
                error,
            )
        except (ImportError, LookupError) as error:
            raise IFEvalDependencyError(
                f"IFEval verifier {instruction_id!r} is missing a local dependency. "
                f"{_PREPARE_NLTK_DATA}"
            ) from error
        except Exception as error:  # pylint: disable=broad-exception-caught
            failed_count += 1
            LOGGER.warning(
                "IFEval verifier %r raised %s: %s",
                instruction_id,
                type(error).__name__,
                error,
            )

    if unknown_count:
        reason = "unknown_verifier"
    elif failed_count:
        reason = "fail_ifeval"
    else:
        reason = "pass"
    return _result(
        total=len(instructions),
        failed=failed_count,
        unknown=unknown_count,
        reason=reason,
    )
