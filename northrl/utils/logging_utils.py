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

from __future__ import annotations

import logging
import math
import os
from array import array
from collections import defaultdict
from numbers import Real
from typing import Any

import numpy as np

from northrl.algorithm.groups import _identity_value
from northrl.utils.step_metrics import finite_number, group_key, log_step_metrics
from northrl.utils.trajectory_sequences import iter_trajectory_sequences

from ..northrl_types import Trajectory  # pylint: disable=relative-beyond-top-level

logger = logging.getLogger(__name__)


def set_logging_basic_config():
    """
    This function sets the global logging format and level. It will be called when import northrl
    """
    import sys

    # Get logging level from environment variable, default to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Convert string level to logging level constant
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,  # Common alias
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,  # Common alias
    }

    level = level_mapping.get(log_level_str, logging.INFO)

    # Set the logging level while preserving the original format
    # Format: [timestamp][logger_name][level] - message
    log_stream = os.getenv("LOG_STREAM", "")

    # Explicitly set stream to stdout when requested by the runtime.
    if log_stream == "stdout":
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            stream=sys.stdout,
            force=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        )

    print(f"📋 Logging Configuration:")
    print(f"   └─ NorthRL logging level: {logging.getLevelName(level)} ({level})")


def _read_int_file(path: str) -> int | None:
    """Read a single integer from ``path``; return None on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_cgroup_memory() -> tuple[int | None, int | None]:
    """Return ``(usage_bytes, limit_bytes)`` for the current cgroup.

    Cgroup limits can constrain a local process independently of host-wide
    memory availability. Tries cgroup v2 first, then v1. ``limit`` is None when
    unlimited (v2 reports ``"max"``; v1 reports a near-2**63 sentinel).
    Either element is None if unreadable.
    """
    # cgroup v2: unified hierarchy at /sys/fs/cgroup
    usage = _read_int_file("/sys/fs/cgroup/memory.current")
    if usage is not None:
        limit: int | None = None
        try:
            with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as f:
                raw = f.read().strip()
            limit = None if raw == "max" else int(raw)
        except (OSError, ValueError):
            limit = None
        return usage, limit

    # cgroup v1
    usage = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    # v1 uses a huge sentinel (~PAGE_COUNTER_MAX) to mean "no limit".
    if limit is not None and limit >= 1 << 62:
        limit = None
    return usage, limit


def format_memory_usage() -> str:
    """Return a one-line summary of driver-process and cgroup memory.

    Intended for periodic logging so an OOM-kill of the driver process can be
    diagnosed after the fact: the kernel kills the process with SIGKILL, so no
    Python traceback is emitted and the last logged memory figure is the only
    on-disk evidence. ``driver_rss`` is the controller process alone, while
    the historical ``pod_mem`` label covers the current cgroup, including local
    processes subject to the kernel's memory limit. Never raises.
    """
    gib = 1024**3
    parts: list[str] = []

    try:
        import psutil

        rss = psutil.Process().memory_info().rss
        parts.append(f"driver_rss={rss / gib:.2f}GiB")
    except Exception as exc:
        parts.append(f"driver_rss=? ({exc})")

    usage, limit = _read_cgroup_memory()
    if usage is not None and limit:
        parts.append(
            f"pod_mem={usage / gib:.2f}/{limit / gib:.2f}GiB ({100.0 * usage / limit:.1f}%)"
        )
    elif usage is not None:
        parts.append(f"pod_mem={usage / gib:.2f}GiB (limit=unset)")

    return " ".join(parts)


def _metric_reward(trajectory: Trajectory) -> float | None:
    reward = trajectory.reward
    if reward is None:
        return None
    if hasattr(reward, "item"):
        reward = reward.item()
    if isinstance(reward, Real) and not isinstance(reward, bool) and math.isfinite(reward):
        return float(reward)
    if trajectory.status != "success" or trajectory.get("group_skipped", False):
        return None
    raise ValueError("A successful trajectory reward must be a finite scalar")


def _active_trajectories(trajectories: list[Trajectory]) -> list[Trajectory]:
    return [
        trajectory
        for trajectory in trajectories
        if trajectory.status == "success" and not trajectory.get("group_skipped", False)
    ]


def _trajectory_group_key(trajectory: Trajectory, index: int) -> tuple[Any, Any]:
    """Match GRPO identity normalization, including its ungrouped fallback."""
    return (
        _identity_value(trajectory.get("model_tag")),
        _identity_value(trajectory.get("group_id", ("ungrouped", index))),
    )


def log_rollout_metrics(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    """Summarize exactly this consumed batch using canonical object metrics."""
    log_step_metrics(trajectories, metrics)
    rewards = [
        reward for trajectory in trajectories if (reward := _metric_reward(trajectory)) is not None
    ]
    metrics["trajectory/reward_observed_count"] = len(rewards)
    if rewards:
        metrics["trajectory/reward_mean"] = float(np.mean(rewards))
        metrics["trajectory/reward_std"] = float(np.std(rewards))
        metrics["trajectory/reward_min"] = min(rewards)
        metrics["trajectory/reward_max"] = max(rewards)


def log_grpo_metrics(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    # Preparation may mark an entire group skipped after the input snapshot.
    if any(group_key(parent) is not None for parent in trajectories):
        metrics["batch/skipped_group_count"] = len(
            {
                key
                for parent in trajectories
                if parent.get("group_skipped", False) and (key := group_key(parent)) is not None
            }
        )
    group_to_rewards: dict[Any, list[float]] = defaultdict(list)
    for index, trajectory in enumerate(trajectories):
        if trajectory.status != "success" or trajectory.get("group_skipped", False):
            continue
        reward = _metric_reward(trajectory)
        if reward is not None:
            group_to_rewards[_trajectory_group_key(trajectory, index)].append(reward)
    group_stds = [
        float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
        for rewards in group_to_rewards.values()
    ]
    metrics["group/valid_group_count"] = len(group_to_rewards)
    if group_stds:
        metrics["group/reward_std_min"] = min(group_stds)
        metrics["group/reward_std_max"] = max(group_stds)
    metrics["group/singleton_group_count"] = sum(
        len(rewards) == 1 for rewards in group_to_rewards.values()
    )


def _response_token_count(mask: Any) -> int:
    if isinstance(mask, array) and mask.typecode in "bBhHiIlLqQfd":
        return int(np.count_nonzero(np.asarray(mask) > 0))
    return sum(value > 0 for value in mask)


def log_advantage_distribution_metrics(
    trajectories: list[Trajectory], metrics: dict[str, Any]
) -> None:
    """Log trajectory-level advantages and a separate token-weighted mean."""
    active = [
        trajectory
        for trajectory in _active_trajectories(trajectories)
        if finite_number(trajectory.get("advantage")) is not None
    ]
    if not active:
        return
    arr = np.array([float(trajectory.get("advantage", 0.0)) for trajectory in active])
    metrics["trajectory/advantage_mean"] = float(np.mean(arr))
    metrics["trajectory/advantage_std"] = float(np.std(arr))
    metrics["trajectory/advantage_min"] = float(np.min(arr))
    metrics["trajectory/advantage_max"] = float(np.max(arr))
    metrics["trajectory/advantage_abs_mean"] = float(np.mean(np.abs(arr)))

    token_counts = np.array(
        [
            sum(
                _response_token_count(segment["loss_mask"])
                for segment in iter_trajectory_sequences(trajectory)
            )
            for trajectory in active
        ],
        dtype=np.float64,
    )
    total = token_counts.sum()
    if total > 0:
        metrics["trajectory/advantage_token_weighted_mean"] = float(
            np.sum(arr * token_counts) / total
        )
    n = len(arr)
    n_pos = int(np.sum(arr > 0))
    n_neg = int(np.sum(arr < 0))
    n_zero = int(np.sum(arr == 0.0))
    metrics["trajectory/advantage_positive_count"] = n_pos
    metrics["trajectory/advantage_negative_count"] = n_neg
    metrics["trajectory/advantage_zero_count"] = n_zero
    metrics["trajectory/advantage_zero_percentage"] = 100 * n_zero / n
    logger.debug(
        "[advantage_distribution] trajectories=%d pos=%d neg=%d zero=%d mean=%.4f std=%.4f",
        n,
        n_pos,
        n_neg,
        n_zero,
        float(np.mean(arr)),
        float(np.std(arr)),
    )


def log_logprob_metrics(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    """Log sampling-logprob denominator statistics over response tokens.

    Very negative logprobs indicate the model assigned low probability to
    its own outputs at sampling time, which correlates with large importance
    sampling ratios when the policy changes.
    """
    all_response_logprobs: list[np.ndarray] = []
    segment_ppls: list[float] = []
    for trajectory in _active_trajectories(trajectories):
        for sequence in iter_trajectory_sequences(trajectory):
            logprobs = sequence.get("old_log_probs")
            if logprobs is None:
                continue
            loss_mask = sequence["loss_mask"]
            if (
                isinstance(logprobs, array)
                and logprobs.typecode in "fd"
                and isinstance(loss_mask, array)
                and loss_mask.typecode in "bBhHiIlLqQfd"
            ):
                count = min(len(logprobs), len(loss_mask))  # same truncation as zip
                values = np.asarray(logprobs, dtype=np.float64)[:count]
                valid = (np.asarray(loss_mask)[:count] > 0) & np.isfinite(values)
                sequence_logprobs = values[valid]
            else:
                sequence_logprobs = np.asarray(
                    [
                        number
                        for lp, mask in zip(logprobs, loss_mask)
                        if mask > 0 and (number := finite_number(lp)) is not None
                    ],
                    dtype=np.float64,
                )
            if sequence_logprobs.size:
                all_response_logprobs.append(sequence_logprobs)
                try:
                    ppl = math.exp(-float(np.mean(sequence_logprobs)))
                except OverflowError:
                    ppl = math.inf
                segment_ppls.append(ppl)

    if not all_response_logprobs:
        return

    if segment_ppls and all(math.isfinite(value) for value in segment_ppls):
        metrics["sampling/ppl_mean"] = float(np.mean(segment_ppls))
    arr = np.concatenate(all_response_logprobs)
    metrics["sampling/logprob_mean"] = float(np.mean(arr))
    metrics["sampling/logprob_std"] = float(np.std(arr))
    metrics["sampling/logprob_min"] = float(np.min(arr))
    metrics["sampling/logprob_max"] = float(np.max(arr))
    logger.debug(
        "[logprob] n_tokens=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
        len(arr),
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    )
