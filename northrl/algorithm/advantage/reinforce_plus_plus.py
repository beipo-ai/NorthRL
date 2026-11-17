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

"""REINFORCE++ returns and global token whitening on complete trajectories."""

from __future__ import annotations

import logging
import math
from array import array
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import torch

from northrl.northrl_types import Trajectory
from northrl.utils.logging_utils import log_advantage_distribution_metrics, log_grpo_metrics
from northrl.utils.trajectory_sequences import iter_trajectory_sequences

from ..groups import _final_reward, _identity_value, validate_trajectory_groups

TrainingSequence = Trajectory | dict[str, Any]
_WHITEN_EPSILON = 1e-8


@dataclass
class _ParentData:
    trajectory: Trajectory
    sequences: list[tuple[TrainingSequence, int]]
    count: int
    reward: float
    group_key: tuple[Any, Any]


def _finite_scalar(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _response_count(sequence: TrainingSequence) -> int:
    tokens, mask = sequence.get("tokens"), sequence.get("loss_mask")
    if tokens is None or mask is None:
        raise ValueError("REINFORCE++ requires tokens and a token-aligned loss_mask")
    try:
        if len(tokens) != len(mask):
            raise ValueError("loss_mask must be token-aligned with tokens")
    except TypeError as exc:
        raise ValueError("tokens and loss_mask must be sequences") from exc
    values = np.asarray(mask) if type(mask) in (array, np.ndarray) else None
    if values is not None and values.ndim == 1 and values.dtype.kind in "biuf":
        if not ((values == 0) | (values == 1)).all():
            raise ValueError("loss_mask must contain only 0 or 1")
        if len(values) and values[0] != 0:
            raise ValueError("loss_mask[0] must be 0; the first token has no target logprob")
        return int(np.count_nonzero(values))
    # Preserve scalar-object inputs without accepting strings/complex numbers
    # through NumPy coercion. Typed numeric buffers take the vectorized path.
    count = 0
    for index, value in enumerate(mask):
        if hasattr(value, "item"):
            value = value.item()
        if not isinstance(value, Real) or value not in (0, 1):
            raise ValueError("loss_mask must contain only 0 or 1")
        if index == 0 and value != 0:
            raise ValueError("loss_mask[0] must be 0; the first token has no target logprob")
        if value == 1:
            count += 1
    return count


def _write_returns(parent: _ParentData, output: np.ndarray, gamma: float) -> None:
    """Discount the terminal reward across all effective tokens of one parent."""
    count = len(output)
    if not count:
        return
    if gamma == 1:
        output.fill(parent.reward)
    else:
        running = parent.reward
        output[-1] = running
        # Sequential multiplication avoids powers and division by gamma, keeping
        # long responses and very small (or zero) gamma numerically stable.
        for index in reversed(range(count - 1)):
            running = gamma * running
            output[index] = running
    if not np.isfinite(output).all():
        raise ValueError("discounted return must be a finite real scalar")


def compute_reinforce_plus_plus_advantages(
    trajectories: list[Trajectory],
    *,
    baseline: bool,
    gamma: float = 1.0,
) -> None:
    """Write response-only advantages from outcomes after complete-group validation.

    The caller must first repair complete groups. This function validates them.
    Successful non-skipped parents contribute exactly one outcome each. Vanilla
    REINFORCE++ places that outcome on the final effective response token and
    computes discounted returns across all of a parent's segments; masked context
    positions consume no discount step.

    The baseline variant instead subtracts the group's inclusive mean outcome
    from each parent's outcome and broadcasts the result to every effective token.
    A successful empty parent participates in that baseline. Singleton groups
    have zero centered score. Baseline mode requires gamma=1 and never divides by
    a group reward standard deviation.

    Both variants whiten all effective token returns across the complete step
    using unbiased variance (N-1) and epsilon=1e-8. N<=1 yields zero advantages.
    Each sequence receives its own response-only ``token_advantages`` vector;
    the parent's scalar ``advantage`` is their mean for diagnostics (zero if
    empty). Skipped parents remain untouched. Memory is linear in active tokens
    and sequences, without a padded batch tensor.
    """
    if not isinstance(baseline, bool):
        raise ValueError("baseline must be a bool")
    gamma = _finite_scalar(gamma, "gamma")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    if baseline and gamma != 1:
        raise ValueError("REINFORCE++ baseline requires gamma=1")

    validate_trajectory_groups(trajectories)

    parents = []
    for index, trajectory in enumerate(trajectories):
        if trajectory.status != "success" or trajectory.get("group_skipped", False):
            continue
        reward = _final_reward(trajectory.reward)
        sequences = []
        # Do not broadcast a stale parent-wide token_advantages vector into
        # nested segments: each segment owns its own token alignment.
        sequence_records = (
            [trajectory]
            if trajectory.segments is None and trajectory.tokens is not None
            else iter_trajectory_sequences(trajectory)
        )
        for sequence in sequence_records:
            sequences.append((sequence, _response_count(sequence)))
        group_key = (
            _identity_value(trajectory.get("model_tag")),
            _identity_value(trajectory.get("group_id", ("ungrouped", index))),
        )
        parents.append(
            _ParentData(
                trajectory,
                sequences,
                sum(count for _, count in sequences),
                reward,
                group_key,
            )
        )

    flat_values = np.empty(sum(parent.count for parent in parents), dtype=np.float64)
    if baseline:
        scores = [parent.reward for parent in parents]
        groups: dict[tuple[Any, Any], list[int]] = defaultdict(list)
        for index, parent in enumerate(parents):
            groups[parent.group_key].append(index)
        centered = [0.0] * len(parents)
        for indices in groups.values():
            mean = math.fsum(scores[index] / len(indices) for index in indices)
            for index in indices:
                centered[index] = _finite_scalar(scores[index] - mean, "centered reward")
        offset = 0
        for score, parent in zip(centered, parents, strict=True):
            flat_values[offset : offset + parent.count] = score
            offset += parent.count
    else:
        offset = 0
        for parent in parents:
            _write_returns(parent, flat_values[offset : offset + parent.count], gamma)
            offset += parent.count

    # Share the contiguous float64 return buffer with Torch: retain its existing
    # unbiased statistics/kernel precision without a flattened Python float list.
    flat_returns = torch.from_numpy(flat_values)
    if flat_returns.numel() <= 1:
        flat_returns.zero_()
    else:
        variance, global_mean = torch.var_mean(flat_returns, correction=1)
        if not torch.isfinite(variance) or not torch.isfinite(global_mean):
            raise ValueError("REINFORCE++ whitening statistics must remain finite")
        flat_returns.sub_(global_mean).mul_(torch.rsqrt(variance + _WHITEN_EPSILON))
    offset = 0
    for parent in parents:
        count = parent.count
        parent_advantages = flat_values[offset : offset + count]
        parent.trajectory["advantage"] = (
            math.fsum(memoryview(parent_advantages)) / count if count else 0.0
        )
        if parent.trajectory.segments is not None:
            parent.trajectory.extra_fields.pop("token_advantages", None)
        elif not parent.sequences:
            parent.trajectory["token_advantages"] = []
        for sequence, sequence_count in parent.sequences:
            values = flat_values[offset : offset + sequence_count]
            # Exact built-in arrays opt in; custom sequence subclasses retain
            # their previous list-output behavior.
            if type(sequence["tokens"]) is array:  # pylint: disable=unidiomatic-typecheck
                packed = array("d")
                packed.frombytes(memoryview(values).cast("B"))
                sequence["token_advantages"] = packed
            else:
                sequence["token_advantages"] = values.tolist()
            offset += sequence_count


@dataclass(frozen=True)
class ReinforcePlusPlusEstimator:
    """Compute terminal returns or centered group scores, then whiten the step."""

    gamma: float = 1.0
    baseline: bool = False

    def estimate(
        self,
        trajectories: list[Trajectory],
        inputs: Mapping[str, Any],
        metrics: dict[str, Any],
        *,
        logger: logging.Logger | None = None,
    ) -> list[Trajectory]:
        """Use rewards and masks already on trajectories; retain original objects."""
        del inputs, logger
        compute_reinforce_plus_plus_advantages(
            trajectories,
            baseline=self.baseline,
            gamma=self.gamma,
        )
        log_grpo_metrics(trajectories, metrics)
        log_advantage_distribution_metrics(trajectories, metrics)
        return trajectories
