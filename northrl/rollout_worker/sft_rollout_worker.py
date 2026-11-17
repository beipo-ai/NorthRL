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

"""Pass local-token or opaque managed SFT data through as trajectories."""

from collections.abc import Mapping
from typing import Any, cast

from ..northrl_types import Trajectory
from .base_rollout_worker import BaseRolloutWorker


class SFTRolloutWorker(BaseRolloutWorker):
    """
    Rollout worker for SFT that converts loader data to Trajectories.

    No LLM inference is performed. Local items carry token-aligned
    ``input_ids``/``loss_mask``. Managed items carry an opaque ``sample_ref``
    without managed token IDs, plus optional datum-attached loss inputs.

    Config:
        - need_llm_inference: Must be ``false`` (no inference service needed).
        - identifier: Model tag used for trajectory routing (default: "default").
    """

    async def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Convert a single data-loader item into a single-segment Trajectory.

        Args:
            task: Dict with exactly one of token fields or ``sample_ref``.

        Returns:
            Result of ``_put_trajectory``: 'success' or 'fail'.
        """
        input_ids_value = task.get("input_ids")
        loss_mask_value = task.get("loss_mask")
        sample_ref = task.get("sample_ref")
        has_token_field = input_ids_value is not None or loss_mask_value is not None
        has_sample_ref = sample_ref is not None
        if has_token_field == has_sample_ref:
            raise ValueError(
                "SFTRolloutWorker requires exactly one of tokenized input_ids/loss_mask "
                "or sample_ref"
            )

        input_ids: list[int] | None = None
        loss_mask: list[int] | None = None
        input_token_count = task.get("input_token_count")
        attached_loss_inputs = task.get("loss_fn_inputs", {})
        if not isinstance(attached_loss_inputs, Mapping):
            raise TypeError("SFT loss_fn_inputs must be a mapping when provided")
        if has_sample_ref:
            if not isinstance(input_token_count, int) or isinstance(input_token_count, bool):
                raise ValueError("Managed SFT sample_ref requires integer input_token_count")
            if input_token_count <= 0:
                raise ValueError("Managed SFT input_token_count must be positive")
            datum_id = task.get("datum_id")
            if not isinstance(datum_id, str) or not datum_id:
                raise ValueError("Managed SFT sample_ref requires datum_id")
        else:
            if input_ids_value is None or loss_mask_value is None:
                raise ValueError(
                    "SFTRolloutWorker requires input_ids and loss_mask; "
                    f"got keys={list(task.keys())}"
                )
            loss_mask = list(loss_mask_value)
            input_ids = list(input_ids_value)
            if len(input_ids) != len(loss_mask):
                raise ValueError(
                    "SFTRolloutWorker requires token-aligned input_ids and loss_mask: "
                    f"{len(input_ids)} != {len(loss_mask)}"
                )
            actual_count = len(input_ids) - 1
            if input_token_count is not None and input_token_count != actual_count:
                raise ValueError(
                    "Local SFT input_token_count does not match len(input_ids)-1: "
                    f"{input_token_count} != {actual_count}"
                )
            input_token_count = actual_count

        # ModelTag is hardcoded to "actor" in the new schema (single trainable model).
        identifier = "actor"

        trajectory = Trajectory(
            # Managed SFT is the internal token-free exception to Trajectory's
            # historical list annotations; runtime XOR validation lives in
            # Trajectory.__post_init__.
            tokens=cast(list[int], input_ids),
            loss_mask=cast(list[int], loss_mask),
            reward=0.0,
            is_val=task.get("is_val", False),
            extra_fields={
                "model_tag": identifier,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
                "sft_batch_id": task.get("sft_batch_id"),
                # Normalize the complete mixed-SFT carrier key set for both
                # variants so list batching/reordering cannot detach metadata.
                "sample_ref": sample_ref,
                "datum_id": task.get("datum_id"),
                "input_token_count": input_token_count,
                # Server-owned catalog policy remains an internal carrier; it
                # is deliberately not added to the SampleRef itself.
                "content_visibility": task.get("content_visibility"),
                "loss_fn_inputs": dict(attached_loss_inputs),
            },
        )

        return self._put_trajectory(trajectory)
