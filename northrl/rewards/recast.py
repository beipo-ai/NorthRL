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

"""Judge instruction following with bp-rl-judge's RECAST prompt and verdicts."""

import logging
import os
import random
import re
import time
from math import isfinite
from typing import Any

from .llm_judge_retry import MAX_JUDGE_ATTEMPTS, judge_timeout_for_attempt, wait_for_judge_retry

logger = logging.getLogger(__name__)

JUDGE_PROMPT = (
    "请判断回复是否符合指令的全部要求。\n"
    "【指令】{prompt}\n【回复】{response}\n"
    "若完全符合,请输出 [[1]];否则输出 [[0]]。"
)
VERDICT_PATTERN = re.compile(r"\[\[\s*([01])\s*\]\]")


async def recast_reward(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Score the full response as 0/1; propagate persistent judge request failures."""
    prompt = task["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("Recast response must be a string")
    judge_prompt = JUDGE_PROMPT.format(prompt=str(prompt), response=response)

    urls = [url.strip() for url in os.environ.get("LLM_JUDGE_URL", "").splitlines() if url.strip()]
    if not urls:
        raise ValueError("LLM_JUDGE_URL must contain at least one judge base URL")
    model = os.environ.get("LLM_JUDGE_MODEL", "").strip()
    if not model:
        raise ValueError("LLM_JUDGE_MODEL must name the judge model")
    timeout = float(os.environ.get("LLM_JUDGE_TIMEOUT_S", "180"))
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("LLM_JUDGE_TIMEOUT_S must be a positive finite number")
    api_key = os.environ.get("LLM_JUDGE_API_KEY", "_")

    import openai

    offset = random.randrange(len(urls))
    for attempt in range(MAX_JUDGE_ATTEMPTS):
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
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    max_tokens=8192,
                )
            message = completion.choices[0].message
            text = message.content or getattr(message, "reasoning_content", None) or ""
            verdicts = VERDICT_PATTERN.findall(text)
        except Exception as exc:
            if not await wait_for_judge_retry(
                exc,
                failed_attempt=attempt + 1,
                judge_name="Recast",
                logger=logger,
                elapsed_s=time.monotonic() - attempt_started,
                timeout_s=attempt_timeout,
                context={
                    "reward_fn": task.get("reward_fn"),
                    "task_id": task.get("task_id"),
                    "group_id": task.get("group_id"),
                    "sample_id": task.get("sample_id"),
                    "judge_model": model,
                },
            ):
                raise RuntimeError(f"Recast judge failed after {attempt + 1} attempts") from exc
            continue

        if not verdicts:
            logger.warning(
                "Recast judge returned no parseable [[0]]/[[1]] verdict; assigning reward 0; "
                "reward_fn=%r; task_id=%r; group_id=%r; sample_id=%r; judge_model=%r; "
                "judge_response_preview=%r",
                task.get("reward_fn"),
                task.get("task_id"),
                task.get("group_id"),
                task.get("sample_id"),
                model,
                text[:512],
            )
            return {"reward": 0.0, "correct": False, "reason": "no_verdict"}
        correct = verdicts[-1] == "1"
        return {
            "reward": float(correct),
            "correct": correct,
            "reason": "pass" if correct else "fail_llm",
        }

    raise AssertionError("unreachable")
