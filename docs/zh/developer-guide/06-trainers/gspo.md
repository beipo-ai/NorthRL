[English](../../../en/developer-guide/06-trainers/gspo.md) | 简体中文 | [文档索引](../../index.md)

# GSPO

[返回算法组件总览](algorithm-components.md)

设置 `algorithm.type: grpo` 和 `algorithm.loss.mode: gspo` 即可使用
[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)。
示例使用 GRPO estimator 按完整 rollout 的奖励计算标量 advantage，并启用组内标准化
（`norm_adv_by_std: true`）；
GSPO 将 actor 的优化目标改为序列概率比及序列级裁剪。

<a id="two-levels-of-averaging"></a>

## 两层平均

记 `M_i` 为第 `i` 条完整轨迹所有 segment 中 `loss_mask` 选中的 response
token，`I` 为 optimizer batch 中满足 `|M_i| > 0` 的 trajectory 集合：

```text
log_r_i = sum(log_pi_new[t] - log_pi_old[t], t in M_i) / |M_i|
r_i     = exp(log_r_i)
L_i     = -min(r_i * A_i, clip(r_i, 1 - eps_low, 1 + eps_high) * A_i)
L       = sum(L_i, i in I) / |I|
```

第一层在一条完整轨迹内对 token logratio 求均值，再取指数、做一次序列级
裁剪，得到 `L_i`。序列概率比参与梯度计算。第二层对 optimizer batch 中各条
trajectory 的 `L_i` 求均值；`trajectory_mean` 指的是这第二层外部平均。

每条非空 trajectory 等权，不因 response 长度或 segment 数量不同而改变权重。
同一 trajectory 的所有 segment 共同形成一个 `log_r_i` 和一个 `L_i`；
`datum_mean` 则将实际生成的各段作为独立 Datum。advantage 为零的 trajectory
贡献零 loss，但仍计入 `|I|`。被 mask 的 prompt 和工具返回 token 不进入 `M_i`。
没有有效 response token 的 trajectory 不进入 `I`；optimizer batch 中没有非空
trajectory 时跳过更新。

<a id="run-the-example"></a>

## 运行示例

使用 `recipe/math/weaver_gspo.yaml`，按[快速开始](../01-getting-started/quick-start.md)
中的本地运行流程启动，并先准备数学数据、模型和 Weaver 凭据。
recipe 默认使用 console 和 W&B 日志；需要配置 W&B，或按快速开始切换为仅使用
console。关键配置如下：

```yaml
algorithm:
  type: grpo
  advantage:
    estimator: {mode: grpo, params: {norm_adv_by_std: true}}
    postprocessing: {mode: null}
  loss:
    mode: gspo
    params:
      clip:
        ratio_low: 0.0004
        ratio_high: 0.0006
    reduction:
      mode: trajectory_mean
  sampling_logprob_source: recomputed
  entropy_coeff: 0.0
  kl:
    mode: null
  mis:
    enabled: false
  optim:
    lr: 1.0e-4
    batch_size: null
    mini_batch_size: 16
    submit_ahead: 0
backend:
  type: weaver
```

裁剪宽度需要显式指定。未单独配置 `ratio_low` 或 `ratio_high` 时，可用
`loss.params.clip.ratio` 提供共同默认值。下侧宽度必须在 `[0, 1)` 内，上侧宽度必须有限且非负。
示例参数仅提供起点，不代表复现了 GSPO 论文结果。

<a id="execution-and-scope"></a>

## 执行方式与支持范围

NorthRL 通过 Weaver 的 `forward_backward_custom` 调用内置 GSPO callback，
不要求服务端提供名为 `gspo` 的 loss。callback 接收原始 advantage 和完整
optimizer batch，只做一次完整轨迹平均。即使 `mini_batch_size` 将 forward 和
surrogate backward 请求分块，也不会再次归一化。必须设置 `batch_size: null`，
确保同一完整轨迹的所有 segment 留在同一个 optimizer batch 中，同时避免
Datum 级 shuffle 和尾批补齐。

普通单轮轨迹和多 segment 的 agent 轨迹均受支持。同一条轨迹的各 segment 属于
同一次完整 rollout。NorthRL 为实际生成的每条 Datum 记录其所属完整轨迹和预期段数，
并保留 router replay 和 sampling
mask 元数据。把同一条已 mask 的 response
拆成更多 segment 不会改变 loss 和梯度。各段 mask 应使每个训练 response token
只计一次；重复的 prompt/context token 应被 mask。

callback 会拒绝缺失段、重复段和段间不一致的 advantage。段序号按实际转换出的
Datum 重新编号，converter 跳过无 target 或无有效 response 位置的段不会造成索引缺口。不支持逐 token
advantage；所有 segment 继承其所属完整轨迹的标量 advantage。

`sampling_logprob_source: recomputed` 在 optimizer 更新前冻结 trainer 的旧策略
logprob；设为 `inference` 时，分母使用 rollout 保存的采样策略 logprob。
GSPO 的当前策略/旧策略序列概率比参与梯度计算，与 rollout mismatch 校正
`algorithm.mis` 是不同机制。当前实现会拒绝启用 MIS 校正、entropy 正则、KL
正则、自定义 loss 覆盖、`submit_ahead`、双重裁剪 `loss.params.clip.ratio_c`、以及截断 IS
裁剪 `loss.params.clip.truncation_c`。只收集 mismatch 指标、不启用校正仍可使用。

日志包含 `algorithm/weaver/gspo/ratio_mean`、`algorithm/weaver/gspo/ratio_min`、`algorithm/weaver/gspo/ratio_max`、
`algorithm/weaver/gspo/clip_fraction`、`algorithm/weaver/gspo/active_trajectories`、`algorithm/weaver/gspo/empty_trajectories`。

ratio 均值和 clip fraction 按非空 trajectory 数加权，极值取 min/max，trajectory
计数求和。trainer 在调用 callback 前会丢弃无有效 response 位置的段，因此正常训练中
`empty_trajectories` 为零。直接调用 callback 时可传入全 mask 的 Datum；该指标统计
没有有效 response token 的完整轨迹数量。转换时被丢弃的完整轨迹和整个被跳过的
optimizer batch 不计入 callback 指标。

GSPO 使用 `algorithm.loss.mode: gspo` 和
`algorithm.loss.reduction.mode: trajectory_mean`。
每条完整轨迹必须有一个标量 advantage，由其所有 segment 共享。使用
GRPO 或 RLOO estimator，并关闭 advantage 后处理。两个 estimator 在返回标量估计
之前，都会拒绝训练 sequence 上非 `None` 的 `token_advantages`，包括空数组或常量数组。
OPD、REINFORCE++ estimator
和 tool-error rebalancing 会产生 token advantage，不能与 GSPO 组合。
clip 保持严格数值校验，不接受布尔值或字符串。
