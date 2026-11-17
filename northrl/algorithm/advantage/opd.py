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

"""OPD advantage estimation from prepared student and teacher logprobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from northrl.northrl_types import Trajectory
from northrl.utils.trajectory_alignment import align_logprobs_to_target, align_loss_mask_to_target
from northrl.utils.trajectory_sequences import iter_training_sequences

from .types import OPDInputs

_LOGGER = logging.getLogger(__name__)


def discounted_future_sum(advantages: np.ndarray, discount: float) -> np.ndarray:
    """Retain OPD's discount over every target position within one sequence."""
    if discount == 0:
        return advantages
    result = np.zeros_like(advantages)
    future_sum = 0.0
    for index in reversed(range(len(advantages))):
        future_sum = advantages[index] + discount * future_sum
        result[index] = future_sum
    return result


@dataclass(frozen=True)
class OPDEstimator:
    """Existing OPD math over explicitly prepared student and teacher scores."""

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    def estimate(
        self,
        trajectories: list[Trajectory],
        inputs: OPDInputs,
        metrics: dict[str, Any],
        *,
        logger: logging.Logger | None = None,
    ) -> list[Trajectory]:
        """Compute OPD from explicitly captured rollout and teacher logprob inputs.

        ``inputs`` has ``sequences`` (original trainable sequence objects),
        ``student_logprobs`` (their rollout-score arrays captured before actor
        recomputation), and ``teacher_logprobs`` (RPC outputs in the same order).
        Keep those score arrays alive by reference when later code replaces a
        sequence's ``old_log_probs``. If a producer mutates arrays in place instead,
        its preparation stage must snapshot the scores. Token buffers are not copied.

        Missing/misaligned rows retain the existing zero-advantage fallback. Discount
        counts masked context positions and resets at sequence boundaries, matching
        the existing OPD trainer rather than a cross-segment reward-to-go algorithm.
        """
        logger = logger or _LOGGER
        coefficient = float(self.kl_penalty_coef)
        discount = float(self.kl_discount_factor)
        sequences = list(iter_training_sequences(trajectories))
        if coefficient == 0:
            logger.warning(
                "kl_penalty_coef is 0, skipping teacher logprob computation. "
                "This is equivalent to no distillation."
            )
            for sequence in sequences:
                sequence["token_advantages"] = [0.0] * sum(sequence["loss_mask"])
            return trajectories

        for field, values in (
            ("sequences", inputs.get("sequences")),
            ("student_logprobs", inputs.get("student_logprobs")),
            ("teacher_logprobs", inputs.get("teacher_logprobs")),
        ):
            if values is None:
                raise ValueError(f"OPD estimation requires prepared {field}")
            if len(values) != len(sequences):
                raise ValueError(f"OPD {field} must align with the step's training sequences")
        if any(saved is not current for saved, current in zip(inputs["sequences"], sequences)):
            raise ValueError("OPD prepared sequences must match the original step sequence objects")

        reverse_kl_sums = []
        valid_token_counts = []
        for sequence, student_logprobs, teacher_logprobs in zip(
            sequences, inputs["student_logprobs"], inputs["teacher_logprobs"], strict=True
        ):
            loss_mask = sequence["loss_mask"]
            num_response_tokens = sum(loss_mask)
            if not teacher_logprobs or not student_logprobs:
                logger.warning("Missing logprobs, setting zero advantages")
                sequence["token_advantages"] = [0.0] * num_response_tokens
                continue
            try:
                target_student_lp = align_logprobs_to_target(
                    tokens=sequence["tokens"], logprobs=student_logprobs, name="old_log_probs"
                )
                target_teacher_lp = align_logprobs_to_target(
                    tokens=sequence["tokens"], logprobs=teacher_logprobs, name="teacher_logprobs"
                )
                target_loss_mask = align_loss_mask_to_target(
                    tokens=sequence["tokens"], loss_mask=loss_mask
                )
            except ValueError as exc:
                logger.warning("OPD logprob alignment failed (%s), setting zero advantages", exc)
                sequence["token_advantages"] = [0.0] * num_response_tokens
                continue
            mask_arr = np.array(target_loss_mask, dtype=float)
            reverse_kl_masked = (
                np.array(target_student_lp) - np.array(target_teacher_lp)
            ) * mask_arr
            kl_advantages = -coefficient * reverse_kl_masked
            if discount > 0:
                kl_advantages = discounted_future_sum(kl_advantages, discount)
            sequence["token_advantages"] = [
                float(kl_advantages[index])
                for index, mask in enumerate(target_loss_mask)
                if mask == 1
            ]
            valid_tokens = int(mask_arr.sum())
            if valid_tokens > 0:
                reverse_kl_sums.append(float(reverse_kl_masked.sum()))
                valid_token_counts.append(valid_tokens)
        if valid_token_counts:
            total_kl = sum(reverse_kl_sums)
            total_tokens = sum(valid_token_counts)
            average_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
            metrics["opd/teacher_kl"] = average_kl
            metrics["opd/valid_tokens"] = total_tokens
            logger.info("[OPD] Average teacher KL: %.4f over %d tokens", average_kl, total_tokens)
        else:
            logger.warning("[OPD] No valid tokens for KL computation")
        logger.info("[OPD] Trajectory preparation completed")
        return trajectories
