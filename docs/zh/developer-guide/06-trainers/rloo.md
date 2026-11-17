[English](../../../en/developer-guide/06-trainers/rloo.md) | 简体中文 | [文档索引](../../index.md)

# RLOO

[返回算法组件总览](algorithm-components.md)

RLOO（REINFORCE Leave-One-Out）将每条 rollout 的最终奖励与同一道题的其他
rollout 比较，得到标量 advantage。在 `algorithm.type: grpo` 奖励训练流程中，
设置 `algorithm.advantage.estimator.mode: rloo` 即可选择该估计器。

<a id="advantage-definition"></a>

## Advantage 定义

对于包含 `N > 1` 条完整轨迹的比较组，记各自的最终奖励为 `R_i`：

```text
A_i = R_i - mean(R_j for j != i)
    = (R_i - group_mean) * N / (N - 1)
```

按 `(model_tag, group_id)` 分组。每条成功的完整轨迹贡献一次奖励，包括没有
训练 sequence 的完整轨迹。同一完整轨迹的所有 segment 共享其标量 advantage；
增加 segment 不改变基线。RLOO 不除以标准差，也不对完整 step 做 whitening。
数值函数对单元素组返回零，但 recipe 校验要求每道题至少生成两条 rollout。

<a id="configuration"></a>

## 配置

将下面的 algorithm 块合并到奖励训练 recipe，例如
[数学任务 recipe](../../recipe/math/README.md)，并设置
`data.train.rollout_repeat_n >= 2`：

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: rloo
      params: {}
    postprocessing:
      mode: null
  loss:
    mode: importance_sampling
    reduction:
      mode: token_mean
  optim:
    lr: 1.0e-6
```

RLOO 没有 estimator 参数，GRPO 的 `norm_adv_by_std` 参数不适用。示例选择原生
importance sampling loss；estimator 和 loss 仍是独立组件。

<a id="data-and-scope"></a>

## 数据与计算范围

trainer 先修复或筛除失败组，随后
[`RlooEstimator`](../../../../northrl/algorithm/advantage/rloo.py) 校验完整比较组：
重复 rollout ID、无效奖励，以及未收齐声明的 `expected_rollouts` 都会报错；
跳过组保持跳过。估计接收本次交付的整个 step，但各比较组分别计算自己的基线。
optimizer batch 大小和请求分块不会改变基线。

estimator 为每条有效完整轨迹写入一个 float32 标量 `advantage`，并广播到该
轨迹的训练 segment。它不需要额外的 estimator 输入（`inputs: {}`）。
可选的[工具错误优势重分配](advantage-processing.md)
在估计之后执行，可以从新计算的标量生成 token advantage。
