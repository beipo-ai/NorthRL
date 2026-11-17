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

"""GRPO reward normalization, trajectory ordering by prompt and run ID, and estimator."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from northrl.northrl_types import Trajectory
from northrl.utils.logging_utils import log_advantage_distribution_metrics, log_grpo_metrics
from northrl.utils.trajectory_sequences import iter_training_sequences

from ..groups import _final_reward, _identity_value, validate_trajectory_groups


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    run_ids: torch.Tensor | None = None,
    norm_adv_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """按完整 trajectory 的 reward 计算 GRPO advantage，再广播到各行的 token。

    这是张量接口：B 是输入序列行数，L 是补齐后的 response token 长度。
    下文的“第 row 行”均指二维张量 token_level_rewards 的第 row 行。
    普通 trajectory 占一行；多段 trajectory 展开输入时，每个 segment 占一行，
    所以 B 可能大于 trajectory 数量。训练中的完整对象接口见
    ``compute_grpo_advantage_for_trajectories``。

    Args:
        token_level_rewards: [B, L] 的奖励张量，每行求和得到 scores[row]。
            常见表示是在最后一个有效 response token 放最终 reward，其余填 0，
            但本函数只要求每行之和等于该 trajectory 的最终 reward。
            同一 trajectory 的各 segment 必须携带完全相同的有限最终 reward，
            不是把最终 reward 分摊给各段。padding 位置应为 0；求和不使用 eos_mask。
        eos_mask: [B, L] 的 0/1 掩码，1 表示需要输出 advantage 的 token 位置，
            0 表示忽略的位置。仅在最后广播 advantage 时使用。
        index: shape [B] 的一维组 ID 张量。第 row 个元素 index[row]
            表示 token_level_rewards 第 row 行所属的 prompt 组。
            可以把参数名理解为 group_ids；元素的值是组 ID，不是行号。
        epsilon: 标准差分母中的稳定项。
        run_ids: 可选的 shape [B] 一维 rollout ID 张量。第 row 个元素
            run_ids[row] 表示 token_level_rewards 第 row 行属于组内哪次 rollout。
            同组且 run_id 相同的多行属于同一 trajectory。不同组可以复用 run_id。
            不传时用行号作为 run_id，即每行都被视为独立 trajectory。
        norm_adv_by_std: 默认除以组内标准差；False 时仅减去组内均值（Dr.GRPO）。

    Example:
        下面用列表展示张量内容，共 4 条 trajectory、5 行输入；
        第 0、1 行是同一 trajectory 的两段::

            token_level_rewards = [
                [0., 0., 0., 1.],  # row = 0
                [0., 0., 0., 1.],  # row = 1
                [0., 0., 0., 3.],  # row = 2
                [0., 0., 0., 2.],  # row = 3
                [0., 0., 0., 4.],  # row = 4
            ]  # shape: [5, 4]

            # 下方各一维张量的第 row 个元素，对应上方二维张量的第 row 行。
            # row:      0   1   2   3   4
            index   = [10, 10, 10, 20, 20]  # shape: [5]，prompt 组 ID
            run_ids = [ 0,  0,  1,  0,  1]  # shape: [5]，组内 rollout ID
            scores  = [1., 1., 3., 2., 4.]  # token_level_rewards.sum(dim=-1)
            groups  = {
                10: {0: [0, 1], 1: [2]},
                20: {0: [3],    1: [4]},
            }

        index 横着写只是排版，shape 是 [5]，不是 [1, 5]。例如 index[2]=10、
        run_ids[2]=1，表示 token_level_rewards 第 2 行属于 prompt 10 的第 1 次 rollout。
        groups[group_id][run_id] 存的是输入行号，不是 reward 或 token。
        第 10 组只用 [1., 3.] 求统计量，均值为 2；不会用 [1., 1., 3.]。
        默认使用样本标准差（correction=1），计算 (reward - mean) / (std + epsilon)。
        norm_adv_by_std=False 时只计算 reward - mean。
        同一 trajectory 的所有行共享 advantage；只有一条 trajectory 的组取 0。

    Returns:
        (advantages, returns, group_stds)。前两项引用同一个 [B, L] 张量，
        每行的 advantage 广播到 token 后乘 eos_mask。group_stds 将组 ID
        映射到标量标准差 Tensor；只有一条 trajectory 的组，其标准差记为 0。
    """
    scores = token_level_rewards.sum(dim=-1)
    if not scores.is_floating_point():
        scores = scores.float()
    # 两层字典自动创建：prompt 组 -> rollout -> 属于这条 trajectory 的输入行号。
    groups: dict[Any, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in range(len(scores)):
        group_id = _identity_value(index[row])
        run_id = _identity_value(run_ids[row]) if run_ids is not None else row
        groups[group_id][run_id].append(row)

    group_stds: dict[Any, torch.Tensor] = {}
    advantages = torch.zeros_like(scores)
    with torch.no_grad():
        for group_id, runs in groups.items():
            rewards = []
            for rows in runs.values():
                # 同一 trajectory 的所有行先校验 reward 一致，再只计入一次。
                reward = scores[rows[0]]
                if not torch.isfinite(reward) or not torch.all(scores[rows] == reward):
                    raise ValueError("Segments of one trajectory must share a finite final reward")
                rewards.append(reward)
            if len(rewards) == 1:
                group_stds[group_id] = torch.zeros_like(rewards[0])
                continue
            std, mean = torch.std_mean(torch.stack(rewards))
            group_stds[group_id] = std
            for reward, rows in zip(rewards, runs.values(), strict=True):
                # rows 可有多个行号；这些 segment 共享所属 trajectory 的 advantage。
                advantage = reward - mean
                advantages[rows] = advantage / (std + epsilon) if norm_adv_by_std else advantage

        advantages = advantages.unsqueeze(-1).expand_as(eos_mask) * eos_mask
    return advantages, advantages, group_stds


def compute_grpo_advantage_for_trajectories(
    trajectories: list[Trajectory],
    epsilon: float = 1e-6,
    logger: logging.Logger | None = None,
    use_run_ids: bool = False,
    norm_adv_by_std: bool = True,
) -> list[Trajectory]:
    """按完整 trajectory 分组计算 GRPO advantage，并原地写回每个对象。

    与张量接口不同，这里的列表每个元素就是一次独立 rollout 的完整 trajectory：
    普通 trajectory 直接带 tokens，多段 trajectory 内含 segments。两者都只贡献
    一个最终 reward；即使成功的 trajectory 没有训练序列，也参与组内统计。
    segment 数量不影响 baseline、标准差或组大小，本函数也不展开 segments。

    分组结构是 ``groups[(model_tag, group_id)] = [trajectory, ...]``，
    字典里存完整对象，不存张量接口中的行号。同组的 reward 做均值和标准差归一化，
    每条 trajectory 得到一个标量 advantage，后续训练转换时才传给内部各段。
    例如同组 A 的 reward=1 且有 2 段，B 的 reward=3 且有 1 段，
    统计量使用 [1, 3]，baseline=2；不会使用 [1, 1, 3]。

    Args:
        trajectories: 完整 Trajectory 对象列表，有效组中各对象的 reward 是有限实数。
            依据 (model_tag, group_id) 分组；缺少 group_id 时，各对象单独成组。
            非空 rollout_id（回退到 e2b_rollout_id）在整个输入列表中不能重复。
            某组任一对象提供 expected_rollouts=R 时，全组必须提供相同的 R，
            并恰好包含 run_id 为 0 到 R-1 的 R 条 trajectory，否则报错。
        epsilon: 标准差分母中的稳定项。
        logger: 可选日志对象，记录各有效组的 trajectory rewards。
        use_run_ids: 历史参数名；当前不控制按 run_id 合并 segment。
            True 使用 float32、样本标准差（correction=1），写回零维 Tensor；
            False 使用 float64、总体标准差（correction=0），写回 Python float。
        norm_adv_by_std: 默认除以组内标准差；False 时只做 reward - mean，
            保留奖励尺度，用于 Dr.GRPO。

    Returns:
        原传入的列表，顺序不变，成功组的每个对象被写入 ``trajectory["advantage"]``。
        默认 advantage = (reward - 组内均值) / (组内标准差 + epsilon)；
        norm_adv_by_std=False 时 advantage = reward - 组内均值。
        只有一条 trajectory 的组直接写入 0。

        若组内有非 success 对象，或整组已标记 group_skipped，整组标记
        group_skipped=True 并跳过。这些对象仍留在列表中，不计算新的 advantage，
        也不清除已有的 advantage。同组 group_skipped 标记不一致会报错；
        组完整性检查在跳过之前执行。
    """
    groups = validate_trajectory_groups(trajectories)
    dtype = torch.float32 if use_run_ids else torch.float64
    for group_id, group in groups.items():
        if group[0].get("group_skipped", False):
            continue

        # 一条完整 trajectory 只取一个 reward，不读取内部 segments 的数量或内容。
        rewards = torch.tensor(
            [_final_reward(trajectory.reward) for trajectory in group], dtype=dtype
        )
        if logger is not None:
            logger.info("group_id: %s, trajectory rewards: %s", group_id, rewards.tolist())
        if len(group) == 1:
            advantages = torch.zeros_like(rewards)
        else:
            std, mean = torch.std_mean(rewards, correction=int(use_run_ids))
            advantages = rewards - mean
            if norm_adv_by_std:
                advantages = advantages / (std + epsilon)
        for trajectory, advantage in zip(group, advantages, strict=True):
            trajectory["advantage"] = advantage if use_run_ids else advantage.item()
    return trajectories


def _to_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pylint: disable=broad-exception-caught
            return value
    return value


def _stable_sort_key(value: Any) -> tuple[int, Any]:
    value = _to_python(value)
    return (0, value) if isinstance(value, int) else (1, str(value))


def _grpo_trajectory_key(trajectory: Trajectory) -> tuple[tuple[int, Any], tuple[int, Any], int]:
    group = trajectory.get("uid") or trajectory.get("group_id") or ""
    try:
        run = int(_to_python(trajectory.get("run_id", 0)))
    except Exception:  # pylint: disable=broad-exception-caught
        run = 0
    primary = trajectory.get("task_id")
    if primary is None:
        primary = trajectory.get("index")
    if primary is None:
        primary = group
    return _stable_sort_key(primary), _stable_sort_key(group), run


def prepare_grpo_trajectories(trajectories: list[Trajectory]) -> list[Trajectory]:
    """Order trajectories by task, group and run ID, then validate before service I/O."""
    trajectories = sorted(trajectories, key=_grpo_trajectory_key)
    validate_trajectory_groups(trajectories)
    return trajectories


@dataclass(frozen=True)
class GRPOEstimator:
    """GRPO scalar advantages with estimator-owned comparison groups."""

    norm_adv_by_std: bool = True

    def estimate(
        self,
        trajectories: list[Trajectory],
        inputs: Mapping[str, Any],
        metrics: dict[str, Any],
        *,
        logger: logging.Logger | None = None,
    ) -> list[Trajectory]:
        """Preserve GRPO ordering, sample-standard-deviation math and base metrics."""
        del inputs
        trajectories = compute_grpo_advantage_for_trajectories(
            sorted(trajectories, key=_grpo_trajectory_key),
            logger=logger,
            use_run_ids=True,
            norm_adv_by_std=self.norm_adv_by_std,
        )
        log_grpo_metrics(trajectories, metrics)
        log_advantage_distribution_metrics(trajectories, metrics)
        for sequence in iter_training_sequences(trajectories):
            if sequence.get("token_advantages") is not None:
                raise ValueError("GRPO estimator requires scalar advantages, not token_advantages")
        return trajectories
