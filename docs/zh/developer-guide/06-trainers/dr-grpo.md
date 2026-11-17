[English](../../../en/developer-guide/06-trainers/dr-grpo.md) | 简体中文 | [文档索引](../../index.md)

# Dr.GRPO 固定响应长度归约

Dr.GRPO 沿用 `algorithm.type: grpo` 训练流程，使用 `mode: grpo` estimator
及 `params.norm_adv_by_std: false`，并组合 `ppo_clip` 与
`loss.reduction.mode: fixed_response_budget`。它复用 `GRPOEstimator`，
没有独立的 `DrGRPOEstimator`，也没有 `dr_grpo` estimator mode。

## 目标与分母

每条完整 trajectory 在 `(model_tag, group_id)` 组内获得
`A = reward - group_mean`，不论包含多少个 segment，都只为均值贡献一次 reward。
单元素组或组内 reward 完全相同的组，其 advantage 为零。

clipped token loss 的总和除以 `D = N * L`：

- `N` 是完整 rollout step 中 `status == "success"`，且 `group_skipped` 为
  false 或未设置的 trajectory 数量。在失败修复和分组校验完成后、拆分训练请求前计数。
- `L` 是配置中的固定响应长度预算，
  即 `algorithm.loss.reduction.params.max_response_tokens`。

例如，一个 step 中有两条成功且未跳过的 trajectory，属于同一组，reward 分别为
`1` 和 `3`，那么 advantage 分别为 `-1` 和 `1`。当 `L = 8192` 时，
分母为 `16384`，与这两条 trajectory 实际生成了多少 token 无关。

成功且未跳过的 trajectory，即使响应为空、advantage 为零，或全部 token 都被
MIS 拒绝，也计入 `N`。失败或 `group_skipped` 的 trajectory 不计入。
一条 trajectory 包含更多 segment 不会增加 `N`。
没有剩余训练 token 时，trainer 仍按已有规则跳过优化器更新。

跳过标记由前面的分组 rollout 流程决定。pool 收齐同一道题的 rollout 后，
检查执行成功、reward 为有限数值、且没有已有跳过标记的 trajectory 数量。
有效数量不足一半时，整组标记跳过；否则 trainer 会复制组内成功 trajectory，
补上无效位置，然后进行分组校验和 advantage 计算。这是复制已有数据，
不会重新发起 rollout。例如，3 条有效、1 条失败，补齐后计入 4 条；
1 条有效、3 条失败，整组跳过，对 `N` 的贡献为 0。

## 数据流

1. trainer 接收整个 step 的 trajectory 列表，将其中的 segment 转为训练 datum。
   一个 datum 是一个 segment 发给训练后端的 token、logprob 和 advantage 数据。
2. `northrl/algorithm/reductions.py` 中的 `attach_fixed_response_budget_denominator()`
   从该 trajectory 列表统计 `N`，
   计算 `N * L`，将相同的整数存入每个 datum 的 `_dr_grpo_denominator` 字段。
   这里只增加这个元数据字段，不复制 token 或 advantage 缓冲区。
3. 同一模块中的 `get_loss_denominators()` 读取并校验已保存的分母。
   `northrl/weaver/loss_adapter.py` 用它对 advantage 的贡献做一次除法，
   使后端的 sum loss 得到所需的归一化结果。
4. trainer 可以将 datum 拆成多个 forward/backward 请求，再执行一次优化器更新。
   每个请求都保留原来的分母；请求大小、保留的 token 数和 datum 数都不会重新决定 `N`。

## 配置

在现有 [`recipe/math/weaver.yaml`](../../../../recipe/math/weaver.yaml) 中，
用下面的内容替换整个 `algorithm` 段。数据、模型、服务设置和启动步骤继续参考
[数学 recipe 指南](../../recipe/math/README.md)。

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: grpo
      params:
        norm_adv_by_std: false
    postprocessing:
      mode: null
  loss:
    mode: ppo_clip
    params:
      clip:
        ratio: 0.2
    reduction:
      mode: fixed_response_budget
      params:
        max_response_tokens: 8192
  sampling_logprob_source: inference
  optim:
    lr: 1.0e-4
    beta1: 0.9
    beta2: 0.95
    eps: 1.0e-8
    weight_decay: 0.01
    grad_clip_norm: 1.0
    batch_size: null
    mini_batch_size: 32
  entropy_coeff: 0.0
  kl:
    mode: null
```

`max_response_tokens` 必须是正整数，应与配置的生成长度上限一致。
这个数学配置省略了 `rollout.sampling.max_tokens`，因此生成上限由
`data.max_sequence_length - data.max_prompt_length = 12288 - 4096 = 8192`
推导。修改生成上限时，要同时调整这个归约预算；它不会根据实际生成的响应推导。
多轮 rollout 应采用完整 trajectory 的总 response token 预算，而不是单次生成的
长度上限。这个预算只缩放 loss，不截断生成。

`norm_adv_by_std: false` 表示只减去组内 reward 均值。
`postprocessing.mode: null` 表示不做 advantage 后处理。
`sampling_logprob_source: inference` 表示使用 rollout 时返回的 logprob。
当前实现要求使用原生 `ppo_clip`，设置有限正数的对称 clip ratio，
关闭 KL 和 custom loss，并设置 `entropy_coeff: 0`。
非对称 clipping、dual clipping 与 truncation clipping 会被拒绝。

`optim.batch_size` 必须省略或为 `null`，保证整个 step 只做一次优化器更新。
`mini_batch_size: 32` 只把这次更新中的每个 forward/backward 请求限制为最多
32 个 datum，不改变分母。

支持 `sync`、`batch-async` 和 `fully-async`。`no-sync` 要求显式 optimizer
batch size，因此不兼容。
