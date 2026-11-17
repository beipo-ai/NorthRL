[English](../../../en/developer-guide/06-trainers/grpo.md) | 简体中文 | [文档索引](../../index.md)

# GRPO 组件

`northrl/algorithm/advantage/grpo.py` 包含张量和完整 trajectory 的 GRPO 公式、
`prepare_grpo_trajectories` 和 `GRPOEstimator`。准备和估计时，都会按题目和
生成编号排列 trajectory，具体依次比较任务标识、组标识和 `run_id`。
排序字段相同的记录保留原来的先后顺序。
显式工厂为 `mode: grpo` 构造该 estimator；它的
`estimate(trajectories, inputs, metrics)` 方法直接返回 trajectory 列表，
标量 advantage 写在原 trajectory 对象上。

`northrl/algorithm/groups.py` 按 `(model_tag, group_id)` 整理同一道题的多次 rollout，
检查 rollout ID 是否重复。数据带有 `expected_rollouts` 时，还检查预期的每次
rollout 是否恰好出现一次；有失败 rollout 的组会被标记为跳过。对未跳过的组，
检查 reward 是否为有限数值。一条 trajectory 就算一次完整 rollout，拆成多少个
训练 segment 都不增加 rollout 次数。

训练调用 `GRPOEstimator.estimate()`，再调用 `compute_grpo_advantage_for_trajectories()`。
它接收 `list[Trajectory]`，给未跳过组中的每条 trajectory 写入一个标量 advantage，再返回原列表；
后续转换训练数据时才把 advantage 分配给各段的 token。
张量接口 `compute_grpo_outcome_advantage()` 接收 `[序列数, token 数]` 的 reward
张量，直接返回已经展开到 token 的 advantage，当前只有测试调用。
这两个函数都统一从 `northrl.algorithm.advantage.grpo` 导入，
`validate_trajectory_groups` 则从 `northrl.algorithm.groups` 导入。
旧模块 `northrl.algorithm.core_algos` 已删除，调用方需要直接使用上述模块。OPD 实现仍独立。
此次整理不改变 reward、分组、排序、指标或服务准备。

每条成功的完整 trajectory 在 `(model_tag, group_id)` 组内贡献一次 reward，
没有训练序列的成功 trajectory 也参与统计。训练 estimator 使用样本标准差：
`A = (reward - group_mean) / (group_std + 1e-6)`。单元素组取零；
`params.norm_adv_by_std: false` 时只减去组内均值。各 segment 共享完整轨迹
的标量 advantage，不增加 reward 统计中的观测数。前面说明的 rollout 检查和
跳过规则仍在估计前执行。

已有训练配置、数据要求和启动方式见 [数学 recipe 指南](../../recipe/math/README.md)。
