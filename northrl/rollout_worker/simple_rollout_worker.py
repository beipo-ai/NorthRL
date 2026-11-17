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
Derived Rollout Worker for NorthRL framework
"""

import logging
import re
from typing import Any

from ..northrl_types import Trajectory
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class SimpleRolloutWorker(BaseRolloutWorker):
    """
    User-defined RolloutWorker logic.
    """

    async def rollout(self, task: dict[str, Any]) -> str | None:
        if "prompt" not in task:
            logger.error(f"Task missing 'prompt' field: {task}")
            return None

        assert self._inference_client is not None, "Inference client not initialized"
        completion_result = await self._inference_client.completion(task["prompt"])
        return self._build_trajectory_from_completion(task, completion_result)

    def _build_trajectory_from_completion(
        self, task: dict[str, Any], completion_result: dict[str, Any]
    ) -> str | None:
        # Remote clients use the canonical northrl_train envelope; mock/legacy
        # clients may still return these fields directly.
        train_fields = completion_result.get("northrl_train", completion_result)
        prompt_tokens = train_fields.get("prompt_tokens", [])
        response_tokens = train_fields.get("response_tokens", [])
        response_logprobs = train_fields.get("response_logprobs", [])
        response_old_logprobs = train_fields.get("response_old_logprobs", response_logprobs)
        response_sampling_masks = completion_result.get("sampling_mask")
        moe_topk_indices_ref = completion_result.get("moe_topk_indices_ref")

        # Create tokens and loss_mask. old_log_probs is the canonical IS denominator
        # (post-sampling-mask logprob from inference; falls back to raw response logprobs
        # when the backend does not return a separate post-sampling field).
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
        old_log_probs = [0.0] * len(prompt_tokens) + response_old_logprobs

        # Truncate before evaluation so reward reflects the actual training data
        tokens, loss_mask, old_log_probs, is_truncated = self._check_and_truncate(
            tokens, loss_mask, old_log_probs
        )
        if response_sampling_masks is not None:
            from northrl.utils.sampling_mask_refs import trim_sampling_mask

            response_sampling_masks = trim_sampling_mask(
                response_sampling_masks, sum(bool(v) for v in loss_mask)
            )

        # Extract answer from response using <answer></answer> tags
        choices = completion_result.get("choices") or [{}]
        response = completion_result.get("response", choices[0].get("text", ""))
        finish_reason = completion_result.get(
            "finish_reason", choices[0].get("finish_reason", "stop")
        )
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else ""

        # Get ground truth from task
        ground_truth = task.get("ground_truth", "")
        if "reward_model" in task:
            ground_truth = task["reward_model"].get("ground_truth", ground_truth)

        # Calculate reward: 0 if truncated, otherwise 1.0 if answer matches ground_truth
        if is_truncated:
            logger.warning("Reward reset to 0.0 due to sequence truncation")
            reward = 0.0
        else:
            reward = 1.0 if extracted_answer == ground_truth else 0.0

        # ModelTag is hardcoded to "actor" in the new schema (single trainable model).
        identifier = "actor"

        self._save_rollout_trace(
            "simple_rollout",
            task,
            {
                "prompt": task["prompt"],
                "completion_result": completion_result,
                "response": response,
                "extracted_answer": extracted_answer,
                "ground_truth": ground_truth,
                "reward": reward,
                "tokens": tokens,
                "loss_mask": loss_mask,
                "old_log_probs": old_log_probs,
                "sampling_mask": response_sampling_masks,
                "prompt_token_count": len(prompt_tokens),
                "response_token_count": sum(loss_mask),
                "finish_reason": finish_reason,
                "is_truncated": is_truncated,
            },
        )

        # Create Trajectory dataclass
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                # Metadata fields
                "ground_truth": ground_truth,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
                "temperature": self._config.rollout.sampling.temperature,
                "finish_reason": finish_reason,
                "model_tag": identifier,
                "old_log_probs": old_log_probs,
                "is_truncated": is_truncated,
                "response": response,
                "extracted_answer": extracted_answer,
                "sampling_mask": response_sampling_masks,
                # router-replay R3: opaque ref to the server-offloaded MoE
                # routing-index shard (the indices never cross the wire).
                "moe_topk_indices_ref": moe_topk_indices_ref,
            },
        )

        distribution = completion_result.get("sampler_distribution_data")
        if distribution is not None:
            from ..utils.score_centering import trim_sampler_distribution

            trajectory.extra_fields["sampler_distribution_data"] = trim_sampler_distribution(
                distribution, sum(loss_mask)
            )

        # Add extra task fields (exclude prompt and ground_truth)
        excluded_keys = {"prompt", "ground_truth", "is_val", "reward_model"}
        for k, v in task.items():
            if k not in excluded_keys:
                trajectory[k] = v

        # Put the trajectory into the trajectory pool
        return self._put_trajectory(trajectory)
