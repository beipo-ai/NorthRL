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

"""
Math reward entry point and scoring implementations.

Adapt one dataset sample and its generated response to rule-based, LLM-based,
or Tinker AIME scoring.
"""

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar, cast

import openai
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

from .evaluation_utils import tinker_aime_scalar_match
from .evaluation_utils.deepscaler_utlis import (
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    grade_answer_verl,
    last_boxed_only_string,
)
from .evaluation_utils.judge_prompt import scale_verify_promptv5
from .evaluation_utils.llm_judge import call_oai_rm_llm
from .evaluation_utils.reward_config import AsyncRewardFn, RewardConfig, RewardFn
from .llm_judge_retry import MAX_JUDGE_ATTEMPTS as LLM_JUDGE_MAX_ATTEMPTS
from .llm_judge_retry import judge_timeout_for_attempt, wait_for_judge_retry

THINKING_START = "<think>"
THINKING_END = "</think>"

logger = logging.getLogger(__name__)

DEFAULT_LLM_JUDGE_TIMEOUT_S = 30.0
MATH_BINARY_VERDICT_RE = re.compile(r"\[\[\s*([01])\s*\]\]")
MATH_ANSWER_ONLY_JUDGE_MAX_TOKENS = 8192
MATH_ANSWER_ONLY_JUDGE_PROMPT = (
    "你是严格的数学阅卷老师。判断【学生解答】给出的最终答案是否与【参考答案】"
    "数学等价(形式/表述不同但等价也算正确;只看最终答案是否正确,不看过程)。\n"
    "【题目】{prompt}\n【参考答案】{label}\n【学生解答】{response}\n"
    "若最终答案正确(与参考答案等价)输出 [[1]];否则输出 [[0]]。"
)
SLIME_MATH_LLM_JUDGE_PROMPT = """\
# System Role: Deterministic Mathematical Autograder
You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the Model Solution is mathematically equivalent to the Golden Answer. You must NOT grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)
Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.
* **Algebraic Equivalence:** e.g., 'n(n+1)/2' is equivalent to 'n^2/2 + n/2'. You must verify the algebra.
* **Numerical Equivalence:** e.g., '1/2' is equivalent to '0.5'; 'sqrt(2)/2' is equivalent to '1/sqrt(2)'.
* **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {{1, 2}} is equivalent to {{2, 1}}).
* **Partial Credit:** No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
* **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.

# 3. Output Protocol (Strict Compliance Required)
You must execute the task using a two-part structure. Failure to follow this structure will result in task failure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags.
Make your thinking concise. This section details your reasoning process and must follow these steps sequentially:
1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. Detail the steps taken to verify mathematical equivalence (e.g., simplification, algebraic manipulation). You must actively try to prove they are the same before concluding they are different.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").

**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY** the final grade.
* If Correct: \\boxed{{Correct}}
* If Incorrect: \\boxed{{Incorrect}}
**CRITICAL CONSTRAINT: Do not add any text, explanations, or formatting outside the <thinking> tags or the final \\boxed{{}} output.**

# 4. Input Data
Here is the problem, model solution, and golden answer to grade:
Problem: {QUESTION}
Model Solution: {STUDENT_ANSWER}
Golden Answer: {REFERENCE_ANSWER}
"""

_ParsedJudgeResponse = TypeVar("_ParsedJudgeResponse")


class LLMJudgeError(RuntimeError):
    """Raised when the LLM judge cannot produce a trustworthy verdict."""


class MathRewardDataError(ValueError):
    """Raised when a math sample does not satisfy the required data contract."""


class MathBinaryVerdictError(ValueError):
    """Raised when an answer-only judge response has no binary verdict token."""


class _LLMJudgeResponse(TypedDict):
    judge: bool
    final_student_answer: str


def _parse_llm_judge_response(content: Any) -> _LLMJudgeResponse:
    if not isinstance(content, str):
        raise ValueError("LLM judge response content must be a string")

    content = content.strip()
    if "```json" in content:
        content = content.split("```json", maxsplit=1)[1].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("LLM judge response must be a JSON object")
    judge = result.get("judge")
    if not isinstance(judge, bool):
        raise ValueError("LLM judge response field 'judge' must be a boolean")
    final_student_answer = result.get("final_student_answer")
    if not isinstance(final_student_answer, str):
        raise ValueError("LLM judge response field 'final_student_answer' must be a string")
    return {
        "judge": judge,
        "final_student_answer": final_student_answer,
    }


def _is_incomplete_judge_answer(final_student_answer: str) -> bool:
    return (
        final_student_answer == ""
        or "Incomplete" in final_student_answer
        or "None" in final_student_answer
    )


def _parse_math_binary_verdict(content: Any) -> bool:
    if not isinstance(content, str):
        raise MathBinaryVerdictError("LLM judge response content must be a string")
    matches = list(MATH_BINARY_VERDICT_RE.finditer(content))
    if not matches:
        raise MathBinaryVerdictError("LLM judge response contains no [[0]] or [[1]] verdict")
    return matches[-1].group(1) == "1"


def _boxed_content(content: str) -> str | None:
    boxed = last_boxed_only_string(content)
    if boxed is None:
        return None
    left_brace = boxed.find("{")
    if left_brace < 0 or not boxed.endswith("}"):
        return None
    return boxed[left_brace + 1 : -1].strip()


def _slime_math_student_answer(response: str) -> str:
    return _boxed_content(response) or response


def _math_answer_region(response: str) -> str:
    """Return the region after the final answer marker, or the full response."""
    think_pos = response.rfind(THINKING_END)
    response_pos = response.rfind("###Response")
    if think_pos < 0 and response_pos < 0:
        search_start = 0
    elif think_pos >= response_pos:
        search_start = think_pos + len(THINKING_END)
    else:
        search_start = response_pos + len("###Response")
    return response[search_start:]


def _validate_math_rule_llm_task(
    task: dict[str, Any], raw_prompt: Any
) -> tuple[list[dict[str, str]], str | list[str]]:
    if not isinstance(raw_prompt, list) or not raw_prompt:
        raise MathRewardDataError("math sample requires a non-empty top-level prompt list")
    for index, message in enumerate(raw_prompt):
        if not isinstance(message, dict):
            raise MathRewardDataError(f"math prompt message {index} must be a dictionary")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise MathRewardDataError(
                f"math prompt message {index} requires a non-empty string role"
            )
        if not isinstance(content, str) or not content.strip():
            raise MathRewardDataError(
                f"math prompt message {index} requires non-empty string content"
            )
    if raw_prompt[-1]["role"] != "user":
        raise MathRewardDataError("the final math prompt message must have role='user'")

    if "label" not in task or task["label"] is None:
        raise MathRewardDataError("math sample requires a non-null top-level label")
    label = task["label"]
    labels = label if isinstance(label, list) else [label]
    if not labels:
        raise MathRewardDataError("math sample label list must not be empty")
    for value in labels:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise MathRewardDataError("math sample label has an unsupported type")
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise MathRewardDataError("math sample label must not be empty")
            if normalized.casefold() == "whatever":
                raise MathRewardDataError("placeholder label 'whatever' is not supported")
        else:
            try:
                finite = math.isfinite(float(value))
            except (OverflowError, ValueError) as exc:
                raise MathRewardDataError("math sample label must be finite") from exc
            if not finite:
                raise MathRewardDataError("math sample label must be finite")
    if isinstance(label, list):
        return raw_prompt, [str(value) for value in label]
    return raw_prompt, str(label)


def _normalized_math_ground_truths(
    ground_truth: str | list[str],
) -> list[str]:
    ground_truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    processed_ground_truths: list[str] = []
    for truth in ground_truths:
        text = str(truth).strip()
        if "\\boxed" in text:
            boxed_truth = extract_answer(text)
            if boxed_truth is None:
                raise MathRewardDataError("math sample label contains an invalid boxed answer")
            text = boxed_truth
        processed_ground_truths.append(text)
    return processed_ground_truths


def _judge_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if not content:
        content = getattr(message, "reasoning_content", None)
    if not content:
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            content = model_extra.get("reasoning_content")
    if not isinstance(content, str):
        raise ValueError("LLM judge response has no string content or reasoning_content")
    return content


class _BaseMathLLMJudgeFn(AsyncRewardFn):  # pylint: disable=abstract-method
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        urls_env = os.environ.get("LLM_JUDGE_URL")
        if urls_env is None:
            raise ValueError("LLM_JUDGE_URL environment variable is not set")
        urls = [url.strip() for url in urls_env.splitlines() if url.strip()]
        if not urls:
            raise ValueError("LLM_JUDGE_URL environment variable contains no usable URL")

        timeout_value = os.environ.get("LLM_JUDGE_TIMEOUT_S", str(DEFAULT_LLM_JUDGE_TIMEOUT_S))
        try:
            timeout_s = float(timeout_value)
        except ValueError as exc:
            raise ValueError("LLM_JUDGE_TIMEOUT_S must be a number") from exc
        if timeout_s <= 0:
            raise ValueError("LLM_JUDGE_TIMEOUT_S must be greater than zero")

        self._judge_timeout_s = timeout_s
        api_key = os.environ.get("LLM_JUDGE_API_KEY", "_")
        self.clients = [
            openai.AsyncOpenAI(
                base_url=url,
                api_key=api_key,
                timeout=timeout_s,
                max_retries=0,
            )
            for url in urls
        ]
        self._client_offset = random.randrange(len(self.clients))

        model_name = os.environ.get("LLM_JUDGE_MODEL")
        if model_name is None:
            raise ValueError("LLM_JUDGE_MODEL environment variable is not set")
        self.model_name: str = model_name

    async def _call_judge_text(
        self,
        client: openai.AsyncOpenAI,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float | None,
        timeout_s: float,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "timeout": timeout_s,
        }
        if temperature is not None:
            request["temperature"] = temperature
        response = await client.chat.completions.create(**request)
        return _judge_message_text(response.choices[0].message)

    async def _call_judge_with_parser(
        self,
        prompt: str,
        parser: Callable[[Any], _ParsedJudgeResponse],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> _ParsedJudgeResponse:
        for attempt in range(1, LLM_JUDGE_MAX_ATTEMPTS + 1):
            attempt_timeout = judge_timeout_for_attempt(self._judge_timeout_s, attempt)
            attempt_started = time.monotonic()
            try:
                client_index = (self._client_offset + attempt - 1) % len(self.clients)
                client = self.clients[client_index]
                content = await self._call_judge_text(
                    client,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=attempt_timeout,
                )
                return parser(content)
            except Exception as exc:
                if not await wait_for_judge_retry(
                    exc,
                    failed_attempt=attempt,
                    judge_name="Math",
                    logger=logger,
                    elapsed_s=time.monotonic() - attempt_started,
                    timeout_s=attempt_timeout,
                    context={"judge_model": self.model_name, "judge_mode": type(self).__name__},
                ):
                    raise LLMJudgeError(f"LLM judge failed after {attempt} attempts") from exc

        raise AssertionError("unreachable")

    async def _call_judge(self, prompt: str) -> _LLMJudgeResponse:
        return await self._call_judge_with_parser(
            prompt,
            _parse_llm_judge_response,
            max_tokens=1024,
        )


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):

        # Extract solution based on thinking format usage.
        # Logic:
        # 1. If THINKING_START exists → model is using thinking format
        #    - If THINKING_END also exists → extract solution after </think>
        #    - If THINKING_END missing → truncated thinking, return format_error
        # 2. If THINKING_START doesn't exist → model doesn't use thinking format
        #    - Use full response as solution
        combined_text = prompt + model_response
        if THINKING_START in combined_text:
            # Model is using thinking format
            if THINKING_END in combined_text:
                # Complete thinking block - extract solution after </think>
                model_solution = model_response.split(THINKING_END)[1]
            else:
                # Thinking started but not finished - truncated
                logger.debug(
                    f"Reward = {self.config.format_error_reward}, due to thinking truncation"
                )
                return self.config.format_error_reward, False
        else:
            # Model doesn't use thinking format - use full response
            logger.debug("No thinking tags found, using full response as solution")
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        logger.debug(f"model_answer: {model_answer}")
        if model_answer is None:
            logger.debug(f"Reward = {self.config.format_error_reward}, due to model_answer is None")
            return self.config.format_error_reward, False

        # Convert single answer to list for uniform processing
        if isinstance(ground_truth, (str, float, int)):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        # print('ground_truths', ground_truths)
        # print('processed_ground_truths', processed_ground_truths)

        if not processed_ground_truths:
            logger.debug(
                f"Reward = {self.config.unk_error_reward}, due to processed_ground_truths is empty"
            )
            return self.config.unk_error_reward, False

        # Check against all possible correct answers
        for gt in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, gt) or grade_answer_sympy(
                model_answer, gt
            )
            if is_correct:
                logger.debug(f"Reward = {self.config.correct_reward}, due to correct answer")
                return self.config.correct_reward, True

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for gt in processed_ground_truths:
                try:
                    orm_response = call_oai_rm_llm(
                        prompt=prompt,
                        model_output=model_answer,
                        ground_truth=gt,
                        temperature=0.0,
                    )

                    if orm_response and "[[YES]]" in orm_response:
                        return self.config.correct_reward, True
                    else:
                        return self.config.format_reward, False
                except Exception as e:
                    print("Error calling LLM ORM:", e)
                    continue

        logger.debug(f"Reward = {self.config.format_reward}, due to no correct answer")
        return self.config.format_reward, False


class SlimeCompatibleMathRewardFn(RewardFn):
    """Binary boxed-answer reward compatible with Slime's ``rm_type=math``."""

    def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        del prompt, raw_prompt
        is_correct = grade_answer_verl(model_response, ground_truth)
        return float(is_correct), is_correct


class SlimeCompatibleMathLLMJudgeFn(AsyncRewardFn):
    """LLM judge compatible with Slime's ``rm_type=math_llmjudge``."""

    async def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        del prompt
        if raw_prompt is None or not raw_prompt:
            return 0.0, False

        question = raw_prompt[-1].get("content")
        if not isinstance(question, str):
            return 0.0, False

        judge_prompt = SLIME_MATH_LLM_JUDGE_PROMPT.format(
            QUESTION=question,
            STUDENT_ANSWER=_slime_math_student_answer(str(model_response)),
            REFERENCE_ANSWER=str(ground_truth),
        )

        try:
            urls_env = os.environ.get("LLM_JUDGE_URL")
            model_name = os.environ.get("LLM_JUDGE_MODEL")
            if not urls_env or not model_name:
                return 0.0, False

            timeout_value = os.environ.get("LLM_JUDGE_TIMEOUT_S", str(DEFAULT_LLM_JUDGE_TIMEOUT_S))
            timeout_s = float(timeout_value)
            if timeout_s <= 0:
                return 0.0, False

            base_url = next((url.strip() for url in urls_env.splitlines() if url.strip()), "")
            if not base_url:
                return 0.0, False
            url = base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": judge_prompt}],
                "temperature": 0.0,
                "max_tokens": 2048,
            }
            response = await asyncio.to_thread(
                requests.post,
                url,
                json=payload,
                headers={"Authorization": f"Bearer {os.environ.get('LLM_JUDGE_API_KEY', '_')}"},
                timeout=timeout_s,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            verdict = _boxed_content(content)
            is_correct = isinstance(verdict, str) and verdict.strip().lower() == "correct"
            return (1.0, True) if is_correct else (0.0, False)
        except Exception as exc:  # Slime-compatible behavior: judge failures score zero.
            logger.warning("Slime-compatible math LLM judge failed: %s", exc)
            return 0.0, False


class MathJudgeV5Fn(_BaseMathLLMJudgeFn):

    async def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):

        if THINKING_START in prompt + model_response and THINKING_END in prompt + model_response:
            model_solution = model_response.split(THINKING_END)[1]
        else:
            return self.config.format_error_reward, False

        if raw_prompt is None:
            return self.config.format_error_reward, False

        question = raw_prompt[-1]["content"]
        answer = model_solution
        # print('【question】', question)
        # print('【answer】', answer)
        prompt_j = scale_verify_promptv5.replace("【problem】", question)
        # Convert ground_truth to string for prompt
        gt_str = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
        prompt_j = prompt_j.replace("【ground_truth】", gt_str)
        prompt_j = prompt_j.replace("【answer】", answer)
        result = await self._call_judge(prompt_j)
        final_student_answer = result["final_student_answer"]
        if _is_incomplete_judge_answer(final_student_answer):
            return self.config.format_reward, False
        if result["judge"]:
            return self.config.llm_correct_reward, True
        return self.config.format_reward, False


class MathBoxedLLMJudgeFn(_BaseMathLLMJudgeFn):

    async def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        model_solution = model_response

        if raw_prompt is None:
            return self.config.format_error_reward, False

        question = raw_prompt[-1]["content"]
        answer = extract_answer(model_solution)
        if answer is None:
            return self.config.format_reward, False
        prompt_j = scale_verify_promptv5.replace("【problem】", question)
        gt_str = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
        prompt_j = prompt_j.replace("【ground_truth】", gt_str)
        prompt_j = prompt_j.replace("【answer】", answer)
        result = await self._call_judge(prompt_j)
        final_student_answer = result["final_student_answer"]
        if _is_incomplete_judge_answer(final_student_answer):
            return self.config.format_reward, False
        if result["judge"]:
            return self.config.llm_correct_reward, True
        return self.config.format_reward, False


class MathAnswerOnlyLLMJudgeFn(_BaseMathLLMJudgeFn):
    """bp-compatible binary LLM fallback for answer-only math scoring."""

    async def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        del prompt
        if raw_prompt is None:
            raise MathRewardDataError("math LLM fallback requires the top-level prompt")
        judge_prompt = MATH_ANSWER_ONLY_JUDGE_PROMPT.format(
            prompt=str(raw_prompt),
            label=str(ground_truth),
            response=model_response,
        )
        verdict = await self._call_judge_with_parser(
            judge_prompt,
            _parse_math_binary_verdict,
            max_tokens=MATH_ANSWER_ONLY_JUDGE_MAX_TOKENS,
            temperature=0.0,
        )
        return (1.0, True) if verdict else (0.0, False)


def _math_llm_failure_reason(exc: Exception) -> str:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, MathBinaryVerdictError):
            return "no_verdict"
        if isinstance(current, (TimeoutError, openai.APITimeoutError)):
            return "timeout"
        if isinstance(current, openai.AuthenticationError):
            return "authentication_error"
        current = current.__cause__
    return "request_error"


class MathRuleLLMAnswerOnlyFn(AsyncRewardFn):
    """Binary math reward: local boxed-answer rules, then an LLM fallback."""

    async def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        answer_region = _math_answer_region(model_response)
        ground_truths = _normalized_math_ground_truths(ground_truth)
        try:
            model_answer = extract_answer(answer_region)
        except Exception as exc:  # extraction failure is recoverable via the LLM
            logger.debug(
                "%s boxed-answer extraction failed with %s; falling back to LLM",
                "math_rule_llm_answer_only",
                type(exc).__name__,
            )
            model_answer = None

        if model_answer is not None:
            for truth in ground_truths:
                try:
                    if grade_answer_mathd(model_answer, truth):
                        return 1.0, True
                    sympy_correct = grade_answer_sympy(model_answer, truth)
                    if sympy_correct:
                        return 1.0, True
                except Exception as exc:  # rule failure is recoverable via the LLM
                    logger.debug(
                        "%s local rule check failed with %s; falling back to LLM",
                        "math_rule_llm_answer_only",
                        type(exc).__name__,
                    )
                    break

        try:
            llm_fn = MathAnswerOnlyLLMJudgeFn(self.config)
        except MathRewardDataError:
            raise
        except Exception as exc:
            reason = (
                "no_endpoint"
                if isinstance(exc, ValueError) and "LLM_JUDGE_URL" in str(exc)
                else "initialization_error"
            )
            logger.warning(
                "%s LLM fallback failed; assigning reward 0: reason=%s error_type=%s",
                "math_rule_llm_answer_only",
                reason,
                type(exc).__name__,
            )
            return 0.0, False

        try:
            return await llm_fn(
                prompt=prompt,
                model_response=answer_region,
                ground_truth=ground_truth,
                raw_prompt=raw_prompt,
            )
        except MathRewardDataError:
            raise
        except Exception as exc:
            logger.warning(
                "%s LLM fallback failed; assigning reward 0: reason=%s error_type=%s",
                "math_rule_llm_answer_only",
                _math_llm_failure_reason(exc),
                type(exc).__name__,
            )
            return 0.0, False


class HybridMathJudgeFnV5(AsyncRewardFn):

    async def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        # print('prompt', prompt)
        # print('model_response', model_response)

        rule_fn = RewardMathFn(self.config)
        reward_score, is_correct = rule_fn(
            prompt=prompt,
            model_response=model_response,
            ground_truth=ground_truth,
            raw_prompt=raw_prompt,
        )
        if reward_score == 1:
            return reward_score, is_correct
        try:
            llm_fn = MathJudgeV5Fn(self.config)
        except (ValueError, openai.OpenAIError) as exc:
            raise LLMJudgeError("Failed to initialize LLM judge") from exc
        reward_score, is_correct = await llm_fn(
            prompt=prompt,
            model_response=model_response,
            ground_truth=ground_truth,
            raw_prompt=raw_prompt,
        )
        return reward_score, is_correct


class TinkerAIMEScalarMatchFn(RewardFn):
    """tinker-style AIME scalar match (boxed -> integer) as a RewardFn.

    See ``evaluation_utils/tinker_aime_scalar_match.py`` for the extraction/match
    logic and its source (tinker). Returns ``correct_reward`` on an exact integer
    match, else ``incorrect_reward``.
    """

    def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        del prompt, raw_prompt  # tinker scores the response against the gold integer only
        if tinker_aime_scalar_match.is_correct(model_response, ground_truth):
            return self.config.correct_reward, True
        return self.config.incorrect_reward, False


async def compute_score(  # pylint: disable=redefined-outer-name
    prompt_str: str,
    solution_str: str,
    ground_truth: str | list[str],
    stage: str,
    raw_prompt: list[dict[str, Any]] | None = None,
    judge_mode: str | None = None,
):
    reward_config = RewardConfig()
    if stage == "val":
        reward_config.use_math_orm = False

    reward_fn: RewardFn | AsyncRewardFn
    if judge_mode is None or judge_mode == "math_with_format":
        reward_fn = RewardMathFn(reward_config)
        raw_prompt = None
    elif judge_mode == "math_answer_only":
        reward_fn = SlimeCompatibleMathRewardFn(reward_config)
        raw_prompt = None
    elif judge_mode == "math_llmjudge":
        reward_fn = SlimeCompatibleMathLLMJudgeFn(reward_config)
    elif judge_mode == "math_llm":
        reward_fn = MathJudgeV5Fn(reward_config)
    elif judge_mode == "math_rule_llm":
        reward_fn = HybridMathJudgeFnV5(reward_config)
    elif judge_mode == "math_boxed_llm":
        reward_fn = MathBoxedLLMJudgeFn(reward_config)
    elif judge_mode == "math_rule_llm_answer_only":
        reward_fn = MathRuleLLMAnswerOnlyFn(reward_config)
    elif judge_mode == "tinker_aime_scalar_match":
        reward_fn = TinkerAIMEScalarMatchFn(reward_config)
        raw_prompt = None
    else:
        raise ValueError(f"Invalid judge mode: {judge_mode}")

    try:
        if isinstance(reward_fn, AsyncRewardFn):
            reward_score, is_correct = await reward_fn(
                prompt=prompt_str,
                model_response=solution_str,
                ground_truth=ground_truth,
                raw_prompt=raw_prompt,
            )
        else:
            sync_reward_fn = cast(RewardFn, reward_fn)
            reward_score, is_correct = sync_reward_fn(
                prompt=prompt_str,
                model_response=solution_str,
                ground_truth=ground_truth,
                raw_prompt=raw_prompt,
            )
        logger.debug(f"Reward: {reward_score}, is_correct: {is_correct}")
        return reward_score
    except (LLMJudgeError, MathRewardDataError):
        raise
    except Exception as e:
        print("Error computing score:", e)
        return reward_config.unk_error_reward


async def math_reward(
    task: dict[str, Any], result: dict[str, Any], *, judge_mode: str = "math_with_format"
) -> float:
    """Score with the existing math implementation, accepting new and legacy data."""
    raw_prompt = task.get("prompt")
    if raw_prompt is not None and hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()

    # Legacy math modes intentionally accept several historical label shapes.
    # The new answer-only mode validates and normalizes its stricter contract
    # before the shared scorer is called.
    ground_truth: Any
    if judge_mode == "math_rule_llm_answer_only":
        raw_prompt, ground_truth = _validate_math_rule_llm_task(task, raw_prompt)
    else:
        ground_truth = task.get("label")
        if ground_truth is None:
            ground_truth = task.get("ground_truth", "")
            if not ground_truth and "reward_model" in task:
                ground_truth = (task.get("reward_model") or {}).get("ground_truth", "")

        if ground_truth == "whatever":
            return 1.0 if random.random() < 0.5 else 0.0

    return await compute_score(
        prompt_str=result.get("prompt_str", ""),
        solution_str=result["response"],
        ground_truth=ground_truth,
        stage="val" if task.get("is_val", False) else "train",
        raw_prompt=raw_prompt,
        judge_mode=judge_mode,
    )


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig())

    problem = "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."
    model_response = "<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}"
    example_ground_truth = ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24"]
    output = reward(
        prompt=problem, model_response=model_response, ground_truth=example_ground_truth
    )
    print(output)
