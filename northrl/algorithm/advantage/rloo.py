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

"""Leave-one-out reward baselines over complete comparison groups."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from northrl.algorithm.groups import _final_reward, validate_trajectory_groups
from northrl.northrl_types import Trajectory
from northrl.utils.logging_utils import log_advantage_distribution_metrics, log_grpo_metrics
from northrl.utils.trajectory_sequences import iter_training_sequences


class RlooEstimator:
    """Leave-one-out scalar advantage estimation over complete parent groups."""

    def estimate(
        self,
        trajectories: list[Trajectory],
        inputs: Mapping[str, Any],
        metrics: dict[str, Any],
        *,
        logger: logging.Logger | None = None,
    ) -> list[Trajectory]:
        """Write one float32 scalar per parent, independent of segment or batch sizes.

        The baseline for parent i is the mean reward of the other N-1 parents in its
        (model_tag, group_id) comparison group. Successful zero-sequence parents
        still contribute one reward. Singleton groups receive zero; failed/skipped
        groups retain the shared complete-group skip policy. No standard-deviation
        or whole-step whitening is applied.
        """
        del inputs
        for group_key, group in validate_trajectory_groups(trajectories).items():
            if group[0].get("group_skipped", False):
                continue
            rewards = torch.tensor(
                [_final_reward(parent.reward) for parent in group], dtype=torch.float32
            )
            if logger is not None:
                logger.info("group_id: %s, trajectory rewards: %s", group_key, rewards.tolist())
            advantages = torch.zeros_like(rewards)
            if len(group) > 1:
                advantages = (rewards - rewards.mean()) * (len(group) / (len(group) - 1))
            for parent, advantage in zip(group, advantages, strict=True):
                parent["advantage"] = advantage
        for sequence in iter_training_sequences(trajectories):
            if sequence.get("token_advantages") is not None:
                raise ValueError("RLOO estimator requires scalar advantages, not token_advantages")
        log_grpo_metrics(trajectories, metrics)
        log_advantage_distribution_metrics(trajectories, metrics)
        return trajectories
