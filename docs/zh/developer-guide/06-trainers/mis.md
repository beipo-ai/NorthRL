[English](../../../en/developer-guide/06-trainers/mis.md) | 简体中文 | [文档目录](../../index.md)

<a id="rollout-and-training-mismatch-correction-mis"></a>

# Rollout 与训练不匹配修正（MIS）

NorthRL 可衡量 rollout 与冻结训练对数概率之间的不匹配，
并可选择修正 rollout 样本对策略梯度的贡献。
诊断仅报告这种不匹配，不更改训练优势值或损失
掩码。修正在训练之前计算固定权重和拒绝掩码，
然后将生成的优势值和损失掩码传给 Weaver。

<a id="supported-configuration"></a>

## 支持的配置

生产环境中的不匹配诊断及 MIS 修正使用 `backend.type: weaver`；
测试也可使用 `mock` 后端。两种模式均要求：

- `algorithm.type: grpo`。
- 未经过滤的 rollout 采样：`top_p: 1.0`、`top_k <= 0` 和 `min_p: 0.0`。
- `rollout.sampling.temperature: 1.0` 和 `repetition_penalty: 1.0`。

`algorithm.mis.enabled: true` 启用修正，同时报告基本不匹配
指标。修正还要求
`algorithm.sampling_logprob_source: recomputed`、原生 `importance_sampling`
或 `ppo_clip` 损失（通过 `algorithm.loss.mode` 选择）；不支持 custom callback。

`algorithm.mis.enabled: false` 配合 `get_mismatch_metrics: true` 仅启用
诊断。此模式支持 `sampling_logprob_source: inference` 或
`recomputed`，且不限制所配置的训练损失或自定义损失。
它不附加修正权重或掩码，不更改训练分母，也不启用 MIS 拒绝。
无论是否启用 MIS，公共转换函数都会跳过 loss mask 全零的 sequence。

配置验证会拒绝启用的 top-p、top-k 或 min-p
过滤器。概率比较还要求轨迹不包含
`sampling_mask`。要支持这些过滤器，需验证 Weaver 的
`forward_logprob` 和训练损失应用相同的候选集
归一化；初始实现不假定此约定成立。

两种模式均拒绝非 1 的温度和重复惩罚：尚未验证当前
采样器接口能在这些变换下返回精确的行为策略
概率。这是接口限制，
并非重要性采样本身的限制。支持其他值
需要采样器提供包含所有采样变换的**实际行为策略**
对数概率。

将以下示例合并到符合上述支持条件的 Weaver 训练配方中。省略了模型、数据、
优化器和其他必需设置：

```yaml
backend:
  type: weaver

rollout:
  sampling:
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    min_p: 0.0
    repetition_penalty: 1.0

algorithm:
  type: grpo
  loss:
    mode: importance_sampling  # Also supports ppo_clip
  sampling_logprob_source: recomputed
  mis:
    enabled: true
    get_mismatch_metrics: false  # Basic mismatch metrics are also emitted when enabled=true.
    tis_mode: truncate
    tis_level: token
    tis_lower_bound: null
    tis_upper_bound: 2.0
    use_rs: false
    rs_level: token
    rs_lower_bound: null
    rs_upper_bound: null
    rs_veto_threshold: null
    tis_batch_normalize: false
```

此示例启用修正。`enabled` 和 `get_mismatch_metrics` 均默认
为 `false`；展示的其他 MIS 值也都是默认值。未配置
`algorithm.mis` 的现有配方保持原有行为。

仅记录不匹配而不修正损失时，使用：

```yaml
algorithm:
  sampling_logprob_source: recomputed  # Also supports inference for diagnostics alone.
  mis:
    enabled: false
    get_mismatch_metrics: true
```

使用 `recomputed` 时，诊断复用训练批次已有的重计算
前向。两条路径均按 `algorithm.optim.mini_batch_size` 拆分 actor 前向请求，
并基于整批计算诊断指标。使用 `inference` 时，诊断增加只读的训练模型前向计算；
训练 Datum 仍使用 rollout 对数概率作为分母。两种
情况下，诊断均在优化器更新之前运行，并使用原始损失掩码。

<a id="probability-sources"></a>

## 概率来源

对于每个响应 token：

- `rollout_prob` 是采样该 token 的行为策略给出的概率。
- `old_prob` 是训练模型在当前训练批次
  第一次优化器更新之前重新计算的概率。
- `current_prob` 是前向/反向计算期间的当前训练概率。

修正使用冻结的 rollout 到训练比率：

```text
log_ratio        = old_logprob - rollout_logprob
correction_ratio = old_prob / rollout_prob = exp(log_ratio)
```

启用修正后，Weaver 原生训练损失使用另一个独立比率：

```text
train_ratio = current_prob / old_prob
            = exp(current_logprob - old_logprob)
```

启用修正时，NorthRL 在重计算之前将采样器分数保留为
`trajectory.rollout_log_probs`。重计算后，
`trajectory.old_log_probs` 包含 `old_logprob`，并成为 Datum 的
`loss_fn_inputs.logprobs`。Weaver 在自身训练
前向中计算 `current_logprob`。用于 KL 的参考策略对数概率是独立字段，
不属于上述任一分母。

诊断将 rollout 分数与真实训练模型前向计算的结果比较。
它不替换训练分母：来源为 `recomputed` 时，
现有训练路径仍会替换分母；来源为 `inference` 时，额外的
诊断前向将分母保持为 rollout 分数。两种模式中的指标
名称 `training_*` 都指优化器更新之前的冻结前向计算。

对于不截断、不过滤、不归一化的 token 级修正，
`correction_ratio * train_ratio = current_prob / rollout_prob`。使用
所配置的修正后，Weaver 接收
`corrected_advantage = advantage * correction_weights`：
`importance_sampling` 在其代理目标中使用此优势值，`ppo_clip`
将其与常规裁剪的 `train_ratio` 配合使用。MIS 边界控制
`correction_ratio`；PPO 裁剪独立控制 `train_ratio`。

<a id="configuration-reference"></a>

## 配置参考

以下所有字段均位于 `algorithm.mis` 下。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `enabled` | `false` | 启用修正：严格的重计算验证、权重、可选拒绝，以及基本不匹配指标。 |
| `get_mismatch_metrics` | `false` | 在 `enabled=false` 时仅报告基本不匹配指标而不修正；`enabled=true` 时修正已会报告这些指标。 |
| `tis_mode` | `truncate` | `none`、`truncate`、`clip` 或 `mask`；选择权重和 TIS 拒绝规则。 |
| `tis_level` | `token` | `token`、`sequence` 或 `geometric`；选择 TIS 使用的比率聚合方式。 |
| `tis_lower_bound` | `null` | 有效下界 `L`；`null` 表示 `1 / tis_upper_bound`。用于 `clip` 和 `mask`，需要时由 RS 继承。 |
| `tis_upper_bound` | `2.0` | 上界 `U`；必须为有限正数，且不能为 `null`。 |
| `use_rs` | `false` | 使用独立的比率层级和区间启用确定性拒绝。 |
| `rs_level` | `token` | `token`、`sequence` 或 `geometric`；独立于 `tis_level`。 |
| `rs_lower_bound` | `null` | RS 下界；`null` 继承有效 TIS 下界。 |
| `rs_upper_bound` | `null` | RS 上界；`null` 继承 `tis_upper_bound`。 |
| `rs_veto_threshold` | `null` | 根据最小原始 token 比率可选地否决整个序列；仅在 `use_rs: true` 时生效。 |
| `tis_batch_normalize` | `false` | 使用原始损失掩码，将权重除以其在原始训练批次中的均值。 |

提供的所有边界和否决阈值必须是有限正数。启用
MIS 时，`clip` 和 `mask` 要求 `L < U`；启用 RS 时，要求
解析后的下界小于解析后的上界。TIS
下界在 `truncate` 或 `none` 模式下不裁剪权重。几何平均
TIS 权重不能与批次归一化结合；`tis_mode: none`
不受此限制，因为其权重为 1。

<a id="ratio-levels"></a>

### 比率层级

只有**原始**轨迹损失掩码选择的 token 才参与
聚合。令 `V_i` 为这些位置，`n_i = len(V_i)`：

| 层级 | 聚合后的对数比率 | 含义 |
| --- | --- | --- |
| `token` | `log_ratio[i,t]` | 每个 token 独立修正。 |
| `sequence` | `sum(log_ratio[i,t] for t in V_i)` | token 比率的乘积，广播到序列中的所有有效 token。 |
| `geometric` | `sum(log_ratio[i,t] for t in V_i) / n_i` | token 比率的几何平均值，广播到所有有效 token。 |

为保证数值稳定性，在**聚合之后**将聚合的对数比率限制到 `[-20, 20]`，
然后取指数。因此，较大的正负 token
差异可在此限制之前相互抵消。空序列
没有有效 token，产生零权重和零掩码。

<a id="weighting-modes"></a>

### 加权模式

令 `correction_ratio` 表示经过数值限制后 `tis_level` 层级上的比率。
以下规则适用于每个原始有效位置：

| `tis_mode` | 存储的权重 | TIS 修正掩码 |
| --- | --- | --- |
| `none` | `1` | `1` |
| `truncate` | `min(correction_ratio, U)` | `1` |
| `clip` | `max(L, min(correction_ratio, U))` | `1` |
| `mask` | `correction_ratio` | 若 `L <= correction_ratio <= U` 则为 `1`，否则为 `0` |

过滤包含区间边界。`mask` 即使在被拒绝位置也保留
原始权重，并单独记录拒绝。这对批次
归一化很重要，因为归一化仍包含这些原始位置。

`tis_mode: none` 配合 `use_rs: true` 仅启用拒绝。配合
`use_rs: false` 时，使用单位权重且不拒绝。两种情况下，
`enabled: true` 均仍强制执行 MIS 概率与重计算
约定。如需观察不匹配且不应用上述修正阶段行为，请改用
`enabled: false` 配合 `get_mismatch_metrics: true`。

<a id="rejection-and-veto"></a>

### 拒绝与否决

RS 根据原始 `log_ratio` 值在 `rs_level` 层级重新计算比率，独立于
TIS 裁剪或加权。它将现有修正掩码与
闭区间 `[rs_lower_bound, rs_upper_bound]` 的判定取交集。在序列
或几何平均层级，该区间判定适用于整个序列。
RS 修改掩码，不会在权重中乘入
额外比率。

设置 `use_rs: true` 和 `rs_veto_threshold: v` 后，任何原始有效
token 满足 `log_ratio[i,t] < log(v)` 都会拒绝整个序列。此检查
使用数值限制之前的原始 token 对数比率，包括
已被先前掩码拒绝的 token。等于阈值时
保留。`use_rs: false` 时，否决设置不生效。

这些都是确定性阈值过滤器，不生成随机数，
也不按与比率成比例的概率接受样本。

例如，以下覆盖配置保留单位权重，过滤
`[0.5, 2.0]` 之外的 token 比率，并附加序列否决：

```yaml
algorithm:
  mis:
    enabled: true
    tis_mode: none
    use_rs: true
    rs_level: token
    rs_lower_bound: 0.5
    rs_upper_bound: 2.0
    rs_veto_threshold: 0.0001
```

<a id="batch-normalization-and-loss-reduction"></a>

### 批次归一化与损失归约

`tis_batch_normalize: true` 时，归一化发生在加权
和拒绝判定之后，但使用**原始**掩码。被拒绝位置
存储的权重仍参与归一化因子 `Z` 的计算：

- token 层级，或 `tis_mode: none`：`Z` 是完整输入批次中所有原始
  有效 token 的平均权重。
- 序列层级：先计算每个序列原始有效 token 的平均权重，
  再在输入批次的全部序列间对这些均值求平均。
  空序列或全部被掩码屏蔽的序列对该序列平均值贡献零。

若 `Z > 1e-8`，将每个存储权重除以 `Z`；否则权重
保持不变。归一化后不再进行第二次裁剪，因此
最终归一化权重可能超过 `tis_upper_bound`。

NorthRL 在将批次拆分为优化器批次之前，基于完整准备好的训练批次
计算一次 `Z`。Slime 的 MIS 辅助函数则在
损失调用的微批次中计算因子。因此，批次划分不同时，
相同设置不一定产生相同的归一化权重。

权重归一化也独立于 Weaver 的损失归约。Weaver
接收过滤后的损失掩码，并使用该掩码重新计算
平均损失；`loss.reduction.mode: token_mean` 时，分母为保留的 token
数。Slime 的集成在归约器中保留原始掩码计数。
因此 NorthRL 不会复现 Slime 在拒绝后的损失缩放。
例如，拒绝两个 token 中的一个，会使 Weaver 的 token-mean 分母
从二变为一，而 Slime 保持为二。

<a id="comparison-with-slime"></a>

## 与 Slime 对比

此对比使用 Slime 提交
[`4c193f1`](https://github.com/THUDM/slime/tree/4c193f1f37509cca70f0e88807a9305b70f63f4e)，
通过 `custom_tis_function_path` 选择
[`examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/examples/train_infer_mismatch_helper/mis.py)。
损失对比指其使用默认 PG 归约器的 Megatron `policy_loss`
集成。Slime 内置的 `vanilla_tis_function`、自定义 PG 归约器和 OPSM
属于独立路径，不作为此处参考。

<a id="shared-formulas-and-switch-mapping"></a>

### 共有公式与开关映射

在对数概率、原始掩码、边界和归一化分组相同时，
共有的 `truncate`、`clip` 和 `mask` 模式使用相同修正公式。
两者都支持 token、序列和几何平均比率，在取指数之前将聚合的对数
比率限制到 `[-20, 20]`，继承未指定的下界/RS 边界，
并使用包含边界的拒绝区间。RS 是确定性阈值过滤。
可选否决检查原始有效 token 上的原始 token 对数比率；
它可以拒绝整个序列。两者均使用包含被拒绝位置的原始掩码
归一化，且不会在归一化后再次裁剪权重。

下表使用 `algorithm.mis` 下的 NorthRL 字段。未列出的开关均为 false；
仅在此表中，将 `get_mismatch_metrics` 简写为 `metrics`。

| 行为 | Slime MIS 辅助函数 | NorthRL |
| --- | --- | --- |
| 无修正或不匹配诊断 | `use_tis=false`、`metrics=false` | `enabled=false`、`metrics=false` |
| 仅不匹配诊断 | `use_tis=false`、`metrics=true` | `enabled=false`、`metrics=true` |
| TIS，包含 TIS `mask` 拒绝 | `use_tis=true`、`use_rs=false` | `enabled=true`、`use_rs=false`，相同 `tis_mode` |
| TIS 加 RS 和可选否决 | `use_tis=true`、`use_rs=true` | `enabled=true`、`use_rs=true`，相同 `tis_mode` |
| 单位权重下的 RS 和可选否决 | 没有生效的仅 RS 分支：`use_tis=false` 在 RS 之前返回；不支持 `tis_mode=none` | `enabled=true`、`tis_mode=none`、`use_rs=true` |

两种实现中，启用修正也会计算基本不匹配
指标。关闭修正开关即禁用 RS，即使 `use_rs=true`。
因此 NorthRL 的 `tis_mode=none` 需要 `enabled=true` 才能运行 RS；它仍
要求重计算对数概率及其他所有修正前提条件。

<a id="probability-sources-and-recompute-timing"></a>

### 概率来源与重计算时机

若可用，Slime 将 `batch["log_probs"]` 作为 `train_log_probs`
传给辅助函数。否则传入该次损失前向计算的
`current_logprob.detach()`。辅助函数的修正比率始终为
`exp(train_logprob - rollout_logprob)`。NorthRL 始终使用独立的
更新前重计算结果：`exp(old_logprob - rollout_logprob)`，在准备好的
批次及其派生的每个优化器批次中保持冻结。
[Slime 损失实现](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/loss.py#L964)。

启用 TIS 且 `compute_advantages_and_returns=true` 时，如果
`can_reuse_log_probs_in_loss` 的全部条件成立，Slime 可省略
独立旧策略前向：rollout 批次仅进行一次优化器更新
（`len(num_microbatches) == 1`，但仍可能包含多个
微批次）、`loss_type=policy_loss`、`kl_coef=0`，并禁用
`use_rollout_logprobs`、`get_mismatch_metrics`、`use_critic`、`keep_old_actor`
以及 `use_opd`。路由重放必须禁用或使用 rollout 路由
重放，且优势值估计器不能为 `gspo`。`use_tis` 本身
不会禁用此优化。NorthRL 修正没有等效的 detach 快速
路径；始终在训练之前重计算。
[Slime 重计算条件](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/actor.py#L428)。

对于受支持的 token 级策略损失，两者都使用
`exp(current_logprob - old_logprob)` 作为训练比率。在 Slime 的快速
路径中，`old_logprob=current_logprob.detach()`：比率值为一，
同时分子保留梯度。Slime 拒绝同时设置 `use_rollout_logprobs=true`
和 `use_tis=true`；NorthRL 修正同样要求 `recomputed`。
仅诊断时，两者都可保留 rollout 对数概率作为训练
分母，并通过独立训练模型前向进行
比较。[Slime 配置检查](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/utils/arguments.py#L1860)。

<a id="loss-integration-and-observable-differences"></a>

### 损失集成与可观测差异

| 方面 | Slime 参考集成 | NorthRL 配合 Weaver |
| --- | --- | --- |
| 应用权重 | 在损失辅助函数中乘到逐 token 的 `pg_loss` 上 | 在原生 `importance_sampling` 或 `ppo_clip` 之前乘到 Datum 优势值上；由 Weaver 计算损失 |
| 应用拒绝 | 返回修改后的响应掩码，用于 PG、熵和显式 KL 损失 | 乘到目标对齐的 Datum 损失掩码上；PG、熵和显式 KL 损失同样受掩码控制 |
| 拒绝后的平均 | 保留原始 `rollout_mask_sums` 和 token 计数归一化因子 | 使用过滤后的掩码归约；如上所述，token-mean 除以保留的 token 数 |
| 权重归一化范围 | 在上下文并行 rank 间收集序列位置后，对每个损失微批次计算；不对完整训练批次归一化 | 在优化器批次拆分之前，对完整准备好的训练批次计算一次 |
| 优化器批次中所有目标被拒绝 | 检查的训练循环中没有 MIS 专属的优化器步骤跳过 | 跳过前向/反向计算和优化器步骤，包括动量及权重衰减更新 |
| 支持的集成 | 辅助函数运行于 Slime 策略损失内部；不施加 NorthRL 的配置限制 | 修正要求 `algorithm.type: grpo`、Weaver、原生 IS/PPO 损失，以及上述采样限制 |

对于非负固定权重以及匹配的 token 级 IS/PPO 代理目标，
对优势值加权与对未归约 PG 损失加权等价。
两种实现中的权重均不直接乘到熵或 KL 项上。
掩码在这些损失项上的作用范围相同，但平均
分母不同会改变拒绝后的缩放。相同权重和掩码
并不意味着相同最终损失或优化器更新。Slime 的默认样本
归约器还支持同一 rollout 组内样本共享分母；
不应假定其与 Weaver 的 `sample-mean` 相同。
[Slime 归约器](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/cp_utils.py#L47)，
[Slime 优化器步骤](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/model.py#L681)。

Slime 还具有用于温度缩放和重放 rollout top-p 候选集的
训练对数概率路径。NorthRL 目前在 MIS 中拒绝这些采样设置，
因为相应的采样器/Weaver 概率约定尚未
验证；这是集成限制，并未改变
修正公式。[Slime 对数概率处理](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/loss.py#L83)。

11 个基本不匹配测量使用相同的逐序列公式，但
日志并非一一对应。Slime 通过原始损失归约器聚合辅助函数
张量；NorthRL 使用配置的 token/样本均值聚合完整准备好的
批次，并从诊断中排除空序列。
Slime 报告中间 TIS/RS 统计和归约后的逐序列极值；
NorthRL 报告下述 `algorithm/mis/` 字段，包括实际批次权重极值和
跳过的优化器批次。NorthRL 使用 Python 浮点数进行修正，
对非有限诊断结果会省略受影响的指标并报告
`algorithm/mis/nonfinite_metric_count`；Slime 辅助函数使用 Torch 张量算术，
未实现此省略策略。

<a id="training-lifecycle-and-alignment"></a>

## 训练生命周期与对齐

启用修正后：

1. 准备完整 trajectory，在任何优化器更新之前，快照保存实际 rollout
   对数概率，并对整个 step 重计算训练对数概率。
2. 按配置对整个 step 计算 advantage 和可选工具错误后处理，保留零 advantage
   trajectory。
3. 基于该 step 计算固定修正权重、掩码和指标。
4. 将轨迹转换为目标对齐的 Datum，应用 MIS 权重和 mask。最终 loss mask
   全零的 sequence 不生成 Datum 或来源记录。保留的 Datum 再按
   `algorithm.optim.batch_size` 拆分优化器批次。基于每个有效批次归一化，再按
   `algorithm.optim.mini_batch_size` 拆分前向/反向请求。
   该批次的全部请求成功后执行一次优化器更新。重计算分母、权重和
   掩码在所有更新过程中保持固定；不会根据每个新模型状态重新计算。

轨迹修正数组的长度与 `tokens` 相同。位置
零是无上下文占位符，权重和掩码均为零。其他
位于原始损失掩码之外的位置，其权重和掩码也为零。

转换器首先使用原始掩码对齐优势值、对数概率、参考对数概率和
响应字段，然后应用：

```text
input_tokens  = tokens[:-1]
target_tokens = tokens[1:]
Datum.advantages[t] *= correction_weights[t + 1]
Datum.loss_mask[t]  *= correction_mask[t + 1]
```

拒绝后不删除 token，也不重新编号。例如：

```text
tokens             = [10, 11, 12, 13]
original_loss_mask = [ 0,  0,  1,  1]
advantage          = 2
correction_weights = [ 0,  0, 0.5, 2]
correction_mask    = [ 0,  0,  1,  0]

Datum.input_tokens  = [10, 11, 12]
Datum.target_tokens = [11, 12, 13]
Datum.advantages    = [ 0,  1,  4]
Datum.loss_mask     = [ 0,  1,  0]
```

被拒绝 token 仍保留原始位置、分母和
参考对数概率。其掩码移除了损失贡献。修正权重
缩放策略梯度优势值，不直接缩放熵或 KL
系数。修改后的损失掩码还控制哪些位置参与
Weaver 带掩码的熵和 KL 项。

是否过滤零优势样本，由转换函数的 `drop_zero_advantage_datums` 参数决定。
本文训练入口 `RemoteApiGrpoTrainer` 传入 `False`：loss mask 非零的 token
即使 advantage 为零，仍参与基于掩码的归约及配置的熵或 KL 项。
其他调用方若使用默认值 `True`，则会过滤 advantage 全零的样本。
无论该参数取何值，`convert_trajectories_to_datums()` 都会在 optimizer 拆批之前
跳过最终 loss mask 全零的 sequence，不会重新计算 advantage，也不改变
baseline 和 whitening。没有剩余 Datum 时，
设置 `training/skipped_flag`，不发起前向/反向计算或优化器更新。

启用修正时，若有效位置的 rollout 对数概率缺失、为占位值、格式错误或非有限值，
训练批次会在重计算之前失败。
有效位置重计算对数概率的 MIS 验证在每次
重计算尝试中运行，因此无效训练器输出会在修正之前重试。

仅诊断时，advantage 准备、Datum 转换、损失输入和优化器拆批保留公共行为，
包括转换函数中的全零 mask 检查。训练器不添加修正数组。来源为 `recomputed` 时
复用已有重计算结果；来源为 `inference` 时执行一次只读
前向，并将结果与训练轨迹分开保留。
不会为每个优化器批次额外执行前向。

来源为 `recomputed` 时，训练分母重计算的每个前向数据块先尝试一次，
最多再重试三次，包括对返回的对数概率进行校验。
只重试失败块；成功块保留其结果和路由重放记录。
每次重试前记录 warning，包含该块的起始序列索引、大小、尝试次数和异常。
全部数据块成功后才写回结果。若某个块耗尽尝试次数，则在优化器更新之前
抛出异常，不受 MIS 开关或路由重放影响；不会回退到 rollout 对数概率。
取消异常直接传播，不重试。

来源为 `inference` 时，独立诊断前向失败会设置
`algorithm/mis/metrics_available=0` 并省略测量值，同时保留训练的
rollout 分母。两种模式中，纯指标计算失败同样与
训练隔离。采样器协议仍要求真实对数概率，
服务持有器仍对请求字段执行严格检查，包括
`old_logprobs`。

<a id="metrics-and-source-files"></a>

## 指标与源文件

下文使用正式 step 日志与 W&B 的输出字段名，统一加上
`algorithm/mis/` 前缀，使自动 workspace 将其归入 Algorithm 分区。
内部采集仍使用 `mis/` 名称，输出规范化时改名，不重复保留旧曲线。
内部 `mis/log_ppl_diff` 与 `mis/kl` 同义，对外只保留
`algorithm/mis/kl`。详见[训练 step 指标与 W&B 查看方式](../../metrics.md)。

`enabled` 或 `get_mismatch_metrics` 为 true 时输出
基本不匹配指标。比较具有有效概率
输入且至少有一个有效 token 时，`algorithm/mis/metrics_available` 为一，否则为零。测量在
修正或优化器批次拆分之前，使用完整准备好的训练批次
上的原始损失掩码。`algorithm/mis/mismatch_tokens` 和
`algorithm/mis/mismatch_sequences` 统计计算使用的原始有效 token 和非空
序列。没有有效 token 的批次不会
报告虚构的 PPL、KL 或卡方值。

以下 10 个去重后的基本测量首先对每个非空序列计算。
`train_logprob` 是冻结训练对数概率，`rollout_logprob` 是采样器
对数概率，`log_ratio = train_logprob - rollout_logprob`。
`mean_over_valid_tokens(...)` 和 `sum_over_valid_tokens(...)` 仅使用
该序列的原始有效位置。定义如下：

```text
training_log_ppl = -mean_over_valid_tokens(train_logprob)
rollout_log_ppl  = -mean_over_valid_tokens(rollout_logprob)
log_ppl_diff    = training_log_ppl - rollout_log_ppl
clamp_log_ratio(log_ratio) = clamp(log_ratio, -20, 20)
```

| 指标 | 逐序列值 |
| --- | --- |
| `algorithm/mis/training_log_ppl` | `training_log_ppl` |
| `algorithm/mis/training_ppl` | `exp(training_log_ppl)` |
| `algorithm/mis/rollout_log_ppl` | `rollout_log_ppl` |
| `algorithm/mis/rollout_ppl` | `exp(rollout_log_ppl)` |
| `algorithm/mis/kl` | `mean_over_valid_tokens(rollout_logprob - train_logprob)`，直接的 rollout 到训练 KL 估计 |
| `algorithm/mis/k3_kl` | `mean_over_valid_tokens(exp(log_ratio) - log_ratio - 1)` |
| `algorithm/mis/log_ppl_abs_diff` | `abs(log_ppl_diff)` |
| `algorithm/mis/ppl_ratio` | `exp(log_ppl_diff)` |
| `algorithm/mis/chi2_token` | `mean_over_valid_tokens(exp(2 * clamp_log_ratio(log_ratio))) - 1` |
| `algorithm/mis/chi2_seq` | `exp(2 * clamp_log_ratio(sum_over_valid_tokens(log_ratio))) - 1` |

这些序列值使用 `algorithm.loss.reduction.mode` 聚合：

- `token_mean`：按每个序列的原始
  有效 token 数加权。
- `datum_mean`：每个非空序列权重相等。

空序列不参与任一分母。对于 PPL，先取指数
再聚合；结果是序列 PPL 的平均，
而非聚合后 log PPL 的指数。同样，绝对差
先逐序列计算再求平均。这些指标描述 rollout 与
冻结训练概率的差异，独立于训练损失中的参考策略 KL 项。
`kl` 和 `chi2_*` 的有限样本估计可能为负。

PPL 和 K3 的指数不会被静默裁剪。仅卡方指标
使用显式 `[-20, 20]` 对数比率边界，与 Slime 公式一致。若任一
序列或最终聚合产生某指标的非有限值，
整个批次都会省略该指标；计算不会仅丢弃
溢出的序列。`algorithm/mis/nonfinite_metric_count` 统计省略的指标
名称数，而非 token 数。其他有限指标仍会报告，且
`algorithm/mis/metrics_available` 可保持为一。此数值策略不改变
修正权重，也不引入新的训练失败。来源为 `inference` 时，独立诊断前向的客户端
墙钟耗时报告为 `timing_train/mismatch_forward_seconds`。mismatch 统计仍正常计算，
不再单独生成统计计算耗时曲线。

以下修正统计仅在 `enabled=true` 时输出。
训练器输出 `algorithm/mis/original_tokens`、`algorithm/mis/kept_tokens`、
`algorithm/mis/original_sequences` 和 `algorithm/mis/kept_sequences`，以及 token 和序列的
拒绝比例。只要至少一个 token
保留，该序列就计为保留。`algorithm/mis/veto_fraction` 统计被完全否决的序列，
`algorithm/mis/clipped_fraction` 统计权重因
`truncate` 或 `clip` 而改变的原始 token。

`algorithm/mis/raw_weight_{mean,min,max}` 描述数值限制后、TIS 处理前的
比率。`algorithm/mis/final_weight_{mean,min,max}` 描述处理
和可选归一化后的存储权重。两者均覆盖原始有效
位置，包括被拒绝位置；min 和 max 是实际批次极值。
`algorithm/mis/batch_norm_factor` 记录除数，未应用归一化时为一。
转换后仍有训练数据时，`algorithm/mis/skipped_batches` 统计 optimizer 拆批后
移除的空 batch，`algorithm/mis/optimizer_steps` 统计保留的 batch 数。
转换函数跳过的 sequence 不计入 skipped batches。转换后没有 Datum 时，不产生
这两个指标；应看 `training/skipped_flag=1` 和 `training/optimizer_step_count=0`。
MIS 权重计算照常执行，不再单独生成计时曲线。

```text
northrl/
  algorithm/mis.py                    # Pure correction and mismatch metrics
  utils/config_schema.py              # MISConfig and support validation
  trainer/remote_api_trainer.py        # Snapshot, recompute, correction, updates
  utils/finetune_service_utils.py      # Target alignment and Datum conversion
```

有关完整训练流程及 advantage 计算，
请参阅[远程 API 训练器](remote-api-trainers.md)。
