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

"""Group rollouts by model and prompt; check rollout IDs, counts and rewards."""

from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from typing import Any

from northrl.northrl_types import Trajectory


def _identity_value(value: Any) -> Any:
    """把分组、去重使用的 ID 转成按值比较的字典 key。

    例如 ``torch.tensor(7)`` 或 ``np.int64(7)`` 经 ``.item()`` 变成整数 7，
    避免把两个值相同的 Tensor 当作不同的字典 key。普通整数、字符串、
    可哈希的 tuple 和 None 保持原样，不会统一转成字符串。

    这里只用 ``hash(value)`` 检查能否作为字典 key 或 set 元素；
    本函数仍返回 ID，不返回它的哈希值。
    list、dict 等不可哈希的 ID 会触发 ValueError；这里不严格限制为标量类型。
    如果对象的 ``.item()`` 失败（例如多元素 Tensor），该异常直接向上传递。
    """
    if hasattr(value, "item"):
        value = value.item()
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"Trajectory identity must be scalar, got {value!r}") from exc
    return value


def _final_reward(value: Any) -> float:
    """读取一条完整 trajectory 的最终 reward，并转成有限的 Python float。

    支持普通实数以及能通过 ``.item()`` 取出实数的 Tensor/NumPy 对象。
    bool、非实数、NaN 和无穷大会触发 ValueError；``.item()`` 的异常直接传递。
    这里只校验单个最终 reward，不收集 segment，也不对 segment 的分数求平均。
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Trajectory reward must be a finite scalar, got {value!r}")
    reward = float(value)
    if not math.isfinite(reward):
        raise ValueError(f"Trajectory reward must be finite, got {value!r}")
    return reward


def validate_trajectory_groups(trajectories: list[Trajectory]) -> dict[Any, list[Trajectory]]:
    """Group rollouts and check for duplicates, missing slots and invalid rewards.

    Return groups keyed by (model_tag, group_id). Each trajectory is one complete
    rollout, including those with several segments or no training tokens. When
    expected_rollouts is supplied, require every expected run_id exactly once.
    Mark groups with failed rollouts to skip; validate finite rewards for the
    remaining groups. Leave advantage fields untouched. Callers that repair
    failed groups must do so before this check.
    """
    groups: dict[Any, list[Trajectory]] = defaultdict(list)
    rollout_ids: set[Any] = set()
    for index, trajectory in enumerate(trajectories):
        rollout_id = trajectory.get("rollout_id") or trajectory.get("e2b_rollout_id")
        if rollout_id is not None:
            rollout_id = _identity_value(rollout_id)
            if rollout_id in rollout_ids:
                raise ValueError(f"Duplicate complete trajectory rollout_id={rollout_id!r}")
            rollout_ids.add(rollout_id)
        group_id = trajectory.get("group_id", ("ungrouped", index))
        group_key = (_identity_value(trajectory.get("model_tag")), _identity_value(group_id))
        groups[group_key].append(trajectory)

    for group in groups.values():
        expected_values = [trajectory.get("expected_rollouts") for trajectory in group]
        if any(expected is not None for expected in expected_values):
            expected = expected_values[0]
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
                raise ValueError("expected_rollouts must be a positive integer on every trajectory")
            if any(value != expected for value in expected_values):
                raise ValueError("Inconsistent expected_rollouts within trajectory group")
            run_ids = [trajectory.get("run_id") for trajectory in group]
            if any(
                isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or not 0 <= run_id < expected
                for run_id in run_ids
            ):
                raise ValueError("run_id must identify one of the expected rollout slots")
            if len(group) != expected or set(run_ids) != set(range(expected)):
                raise ValueError(
                    "Incomplete trajectory group: expected each rollout slot exactly once"
                )

        skipped = [bool(trajectory.get("group_skipped", False)) for trajectory in group]
        if any(skipped) and not all(skipped):
            raise ValueError("Inconsistent group_skipped within trajectory group")
        if any(skipped) or any(trajectory.status != "success" for trajectory in group):
            for trajectory in group:
                trajectory["group_skipped"] = True
            continue
        for trajectory in group:
            _final_reward(trajectory.reward)
    return dict(groups)
