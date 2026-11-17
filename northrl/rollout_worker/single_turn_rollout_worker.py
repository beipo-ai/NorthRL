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

"""Single-turn generation, reward evaluation, and trajectory construction."""

import logging
import time
import uuid
from math import isfinite
from numbers import Real
from typing import Any

from omegaconf import DictConfig

from ..northrl_types import Trajectory
from ..rewards import evaluate_reward
from ..rewards._runtime_context import reward_runtime_context
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class SingleTurnRolloutWorker(BaseRolloutWorker):
    """Generate one response, evaluate its reward, and emit one trajectory."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._reward_fn = config.rollout.get("reward_fn")
        self._max_prompt_length = config.data.max_prompt_length

    async def rollout(self, task: dict[str, Any]) -> str | None:
        """Process a chat task and route its trajectory to training or validation."""
        started_at = time.time()
        started_monotonic = time.monotonic()
        result = await self._generate_and_evaluate(task)
        if result is None:
            return None
        return self._build_trajectory(
            task, result, started_at=started_at, started_monotonic=started_monotonic
        )

    async def _generate_and_evaluate(self, task: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a response with a token-length guard and evaluate its reward."""
        if task.get("prompt") is None:
            logger.warning(f"Task missing 'prompt' field: {task}")
            return None

        assert self._inference_client is not None, "Inference client not initialized"

        # Get prompt - handle both tensor and list formats
        prompt = task["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()

        # Use the sampling service's tokenizer and chat template for the length
        # guard.  Counting Python characters can disagree substantially with
        # the model's actual context length.
        prompt_tokens = await self._inference_client.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True
        )
        if not isinstance(prompt_tokens, list):
            raise TypeError("Tokenized chat template must return a list of token IDs")

        prompt_len = len(prompt_tokens)
        if prompt_len > self._max_prompt_length:
            logger.warning(
                "Prompt too long (%d tokens > %d), skipping",
                prompt_len,
                self._max_prompt_length,
            )
            return None

        # Keep the rendered string for reward APIs and trajectory diagnostics.
        prompt_str = await self._inference_client.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )

        # Call LLM - get OpenAI-style payload
        llm_result: dict[str, Any] = await self._inference_client.generate(messages=prompt)

        if "choices" not in llm_result or not llm_result["choices"]:
            raise ValueError("LLM result missing 'choices' field or is empty")

        first_choice = llm_result["choices"][0]

        message = first_choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("LLM result missing response content")
        response = message["content"]

        finish_reason = first_choice.get("finish_reason")
        if finish_reason is None:
            raise ValueError("LLM result missing finish_reason")

        if response is None:
            if finish_reason != "length":
                logger.warning("LLM returned no response")
                return None
            # Keep length-limited generations as zero-reward trajectories, even
            # when the model exhausted its budget before producing an answer.
            response = ""

        # Evaluate this response before returning the trajectory fields.
        result = {
            **llm_result,
            "response": response,
            "finish_reason": finish_reason,
            "prompt_str": prompt_str,
        }
        verifier_started = None if finish_reason == "length" else time.monotonic()
        with reward_runtime_context(self._stop_event):
            reward_result = await evaluate_reward(task, result, default_reward_fn=self._reward_fn)
        runtime_metrics = dict(result.get("runtime_metrics") or {})
        if verifier_started is None:
            runtime_metrics.pop("verifier_duration_seconds", None)
            runtime_metrics["verifier_skipped_length"] = True
        else:
            runtime_metrics.pop("verifier_skipped_length", None)
            runtime_metrics["verifier_duration_seconds"] = max(
                0.0, time.monotonic() - verifier_started
            )
        result["runtime_metrics"] = runtime_metrics
        reward = reward_result["reward"]
        logger.debug("Reward: %s", reward)
        result["reward"] = reward
        result["score"] = {
            "reward_score": reward,
            "format_correct": reward > 0.0,
            "correct": reward >= 1.0,
            **{key: value for key, value in reward_result.items() if key != "reward"},
        }
        return result

    def _build_trajectory(
        self,
        task: dict[str, Any],
        rollout_result: dict[str, Any],
        *,
        started_at: float,
        started_monotonic: float,
    ) -> str | None:
        """Preserve training fields and task metadata from the generated result."""
        ground_truth = task.get("label")
        if ground_truth is None:
            ground_truth = task.get("ground_truth") or (task.get("reward_model") or {}).get(
                "ground_truth", ""
            )
        # In the new schema ModelTag is hardcoded to "actor" (single trainable
        # model; identifier as a user-facing field has been removed).
        identifier = "actor"

        # Validate the generation result
        if "northrl_train" not in rollout_result:
            logger.warning("Rollout result missing 'northrl_train' field, skipping")
            return None

        train_fields = rollout_result["northrl_train"]

        # Get prompt and response tokens
        prompt_tokens = train_fields.get("prompt_tokens", [])
        response_tokens = train_fields.get("response_tokens", [])
        response_logprobs = train_fields.get("response_logprobs", [])
        response_old_logprobs = train_fields.get("response_old_logprobs", response_logprobs)

        # Create tokens, loss_mask, and the canonical IS-denominator logprobs.
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
        old_log_probs = [0.0] * len(prompt_tokens) + response_old_logprobs

        # Invalidate the reward if the sequence used for training is truncated.
        tokens, loss_mask, old_log_probs, is_truncated = self._check_and_truncate(
            tokens, loss_mask, old_log_probs
        )
        sampling_mask = rollout_result.get("sampling_mask", None)
        moe_topk_indices_ref = rollout_result.get("moe_topk_indices_ref", None)
        if sampling_mask is not None:
            from northrl.utils.sampling_mask_refs import trim_sampling_mask

            sampling_mask = trim_sampling_mask(sampling_mask, sum(bool(v) for v in loss_mask))

        if is_truncated and rollout_result.get("reward"):
            logger.warning("Reward reset to 0.0 due to sequence truncation")
            rollout_result["reward"] = 0.0

        self._save_rollout_trace(
            "single_turn_rollout",
            task,
            {
                "turn_idx": 0,
                "rollout_result": rollout_result,
                "response": rollout_result.get("response"),
                "prompt_str": rollout_result.get("prompt_str"),
                "reward": rollout_result.get("reward", 0.0),
                "score": rollout_result.get("score", {}),
                "tokens": tokens,
                "loss_mask": loss_mask,
                "old_log_probs": old_log_probs,
                "sampling_mask": sampling_mask,
                "prompt_token_count": len(prompt_tokens),
                "response_token_count": sum(loss_mask),
                "finish_reason": rollout_result.get("finish_reason", "stop"),
                "is_truncated": is_truncated,
            },
        )

        # Create Trajectory dataclass
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=rollout_result.get("reward", 0.0),
            is_val=task.get("is_val", False),
            extra_fields={
                # Metadata fields
                "ground_truth": ground_truth,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
                "temperature": self._config.rollout.sampling.temperature,
                "finish_reason": rollout_result.get("finish_reason", "stop"),
                "model_tag": identifier,  # identifier serves as model_tag for trajectory routing
                "sampling_mask": sampling_mask,
                # router-replay R3: opaque ref to the server-offloaded MoE
                # routing-index shard (the indices never cross the wire).
                "moe_topk_indices_ref": moe_topk_indices_ref,
                # Canonical IS denominator (0.0 for prompt, actual logprobs for response).
                "old_log_probs": old_log_probs,
                "is_truncated": is_truncated,
            },
        )

        # Add extra task fields (exclude prompt and ground_truth)
        excluded_keys = {"prompt", "ground_truth", "is_val", "reward_model"}
        for k, v in task.items():
            if k not in excluded_keys:
                trajectory[k] = v

        # Add any additional rollout_result fields (exclude LLM-returned fields)
        llm_return_keys = {
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "northrl_train",
            "tool_calls",
            "response",
            "finish_reason",
            "prompt_tokens",
            "response_tokens",
            "response_logprobs",
            "response_old_logprobs",
            "loss_mask",
            "reward",
        }
        for k, v in rollout_result.items():
            if k not in llm_return_keys and k not in trajectory:
                trajectory[k] = v

        # Validation with one rollout per sample bypasses the loader's group
        # stamping. Give each such sample its own complete sampling group.
        if trajectory.is_val and (
            trajectory.get("group_id") in (None, "")
            or (
                isinstance(trajectory.get("group_id"), Real)
                and not isfinite(trajectory["group_id"])
            )
        ):
            trajectory["group_id"] = uuid.uuid4().hex
            trajectory["expected_rollouts"] = 1
            trajectory["run_id"] = 0

        # Store the function actually selected by evaluate_reward, including a
        # configured fallback when the sample's field is absent or null.
        reward_fn = task.get("reward_fn")
        trajectory["reward_fn"] = self._reward_fn if reward_fn is None else reward_fn

        runtime_metrics = {
            **(trajectory.get("runtime_metrics") or {}),
            **rollout_result["runtime_metrics"],
        }
        if rollout_result.get("finish_reason") == "length":
            runtime_metrics.pop("verifier_duration_seconds", None)
        else:
            runtime_metrics.pop("verifier_skipped_length", None)
        runtime_metrics.update(
            started_at=started_at,
            finished_at=time.time(),
            duration_seconds=max(0.0, time.monotonic() - started_monotonic),
        )
        trajectory["runtime_metrics"] = runtime_metrics

        # Put trajectory into pool
        return self._put_trajectory(trajectory)
