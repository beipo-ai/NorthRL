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

"""Call the reward function selected by one sample's ``reward_fn``.

Short names map directly to ordinary Python callables. A sample's name takes
precedence over the configured default. Each function receives ``(task, result)``
and returns a finite scalar or a dictionary with ``reward`` and optional
``correct``, ``metrics`` and ``reason`` fields.

Built-in functions cover mathematical, scientific, multiple-choice, and
sandboxed executable answers, along with LLM instruction-following judgment.
"""

import asyncio
import inspect
from collections.abc import Callable, Mapping
from functools import partial
from math import isfinite
from numbers import Real
from typing import Any

from .code_e2b import code_e2b_reward
from .dr_sci import rule_dr_sci_reward
from .if_verify import if_verify_reward
from .ifeval import ifeval_reward
from .math import math_reward
from .nemotron_mcqa import rule_nemotron_mcqa_reward
from .reasoning360 import reasoning360_reward
from .reasoning_gym import rule_reasoning_gym_reward
from .recast import recast_reward
from .synlogic import synlogic_reward
from .verinstruct import verinstruct_reward

REWARD_FNS: dict[str, Callable] = {
    "math_with_format": math_reward,
    "math_answer_only": partial(math_reward, judge_mode="math_answer_only"),
    "math_llmjudge": partial(math_reward, judge_mode="math_llmjudge"),
    "math_llm": partial(math_reward, judge_mode="math_llm"),
    "math_rule_llm": partial(math_reward, judge_mode="math_rule_llm"),
    "math_rule_llm_answer_only": partial(
        math_reward,
        judge_mode="math_rule_llm_answer_only",
    ),
    "math_boxed_llm": partial(math_reward, judge_mode="math_boxed_llm"),
    "tinker_aime_scalar_match": partial(math_reward, judge_mode="tinker_aime_scalar_match"),
    "rule_nemotron_mcqa": rule_nemotron_mcqa_reward,
    "rule_dr_sci": rule_dr_sci_reward,
    "rule_reasoning_gym": rule_reasoning_gym_reward,
    "reasoning360": reasoning360_reward,
    "if_verify": if_verify_reward,
    "ifeval": ifeval_reward,
    "code_e2b": code_e2b_reward,
    "synlogic": synlogic_reward,
    "recast": recast_reward,
    "verinstruct": verinstruct_reward,
}


def _finite_scalar(value: Any, field: str) -> float:
    if not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"Reward {field} must be a finite number, got {value!r}")
    return float(value)


def _normalize_result(value: Any) -> dict[str, Any]:
    result = dict(value) if isinstance(value, Mapping) else {"reward": value}
    if "reward" not in result:
        raise ValueError("Reward result dict must contain 'reward'")
    result["reward"] = _finite_scalar(result["reward"], "value")
    if "correct" in result and not isinstance(result["correct"], bool):
        raise ValueError("Reward result 'correct' must be a boolean")
    if "metrics" in result:
        metrics = result["metrics"]
        if not isinstance(metrics, Mapping) or not all(isinstance(key, str) for key in metrics):
            raise ValueError("Reward result 'metrics' must map strings to finite numbers")
        result["metrics"] = {
            key: _finite_scalar(number, f"metric {key!r}") for key, number in metrics.items()
        }
    if "reason" in result and not isinstance(result["reason"], str):
        raise ValueError("Reward result 'reason' must be a string")
    return result


async def evaluate_reward(
    task: dict[str, Any],
    result: dict[str, Any],
    functions: Mapping[str, Callable] | None = None,
    *,
    default_reward_fn: str | None = None,
) -> dict[str, Any]:
    """Evaluate one sample, preferring its reward name over the configured default.

    Length-limited completions receive zero without invoking a verifier.
    Synchronous functions run in a worker thread with the caller's runtime
    context. Async functions and awaitable results run on the caller's loop.
    Cancellation stops waiting but cannot interrupt a running synchronous scorer.
    Unknown names and judge failures propagate to the caller.
    """
    if result.get("finish_reason") == "length":
        return {"reward": 0.0, "correct": False, "reason": "finish_reason_length"}

    name = task.get("reward_fn")
    if name is None:
        name = default_reward_fn
    if name is None:
        raise KeyError("reward_fn")
    function = (REWARD_FNS if functions is None else functions)[name]
    if inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
        getattr(function, "__call__", None)
    ):
        value = function(task, result)
    else:
        value = await asyncio.to_thread(function, task, result)
    if inspect.isawaitable(value):
        value = await value
    return _normalize_result(value)
