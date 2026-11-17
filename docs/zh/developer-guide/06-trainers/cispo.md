[English](../../../en/developer-guide/06-trainers/cispo.md) | 简体中文 | [文档索引](../../index.md)

# CISPO

CISPO 使用停止梯度的裁剪 importance weight 缩放各 response token 的
log-probability 梯度：

```text
r = exp(clamp(logp - old_logp, -20, 20))
w = stop_gradient(clamp(r, 1 - ratio_low, 1 + ratio_high))
L = -sum_active_tokens(w * advantage * logp) / active_token_count
```

先将 log ratio 限制在 `[-20, 20]`，再将 ratio 裁剪到
`[1 - ratio_low, 1 + ratio_high]`。超出裁剪区间时仍保留
log-probability 梯度。

```yaml
algorithm:
  type: grpo
  advantage:
    estimator: {mode: grpo, params: {norm_adv_by_std: true}}
  loss:
    mode: cispo
    params:
      clip:
        ratio: 0.2
    reduction:
      mode: token_mean
  entropy_coeff: 0
  kl:
    mode: null
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
```

`ratio_low`、`ratio_high` 可分别覆盖 `ratio`。边界必须显式提供有限非负数，
不接受布尔值或字符串。不支持 `ratio_c`、`truncation_c`。

CISPO 是内置 loss mode。`LossAdapter.custom` 构造 Weaver custom-loss callback 参数。
Trainer 使用 callback 和 surrogate backward 路径。Converter 将完整轨迹的
标量 advantage 广播，或将 response-only token advantage 对齐到 `target_tokens=tokens[1:]`。Callback 读取这些 target 对齐的 Datum
数组，并负责 token 均值。
Host 不会再次缩放 advantage。

每个 optimizer batch 只调用一次 callback，即使 `mini_batch_size` 将远程
forward/backward 拆成多个请求。`batch_size` 可以将 step 分成多次 optimizer
更新；advantage 估计仍在分批前对整个 step 运行一次。跨更新的 loss 指标按
response token 数加权。

上例使用 GRPO advantage estimator。CISPO 要求 `algorithm.type: grpo`、
Weaver backend、零 entropy、关闭 reference KL、关闭 MIS 修正，以及
`optim.submit_ahead: 0`。
可选的
`advantage.postprocessing.mode: tool_error_rebalance` 在转换前对整个 step 运行。
有效位置上零 advantage token 仍计入 loss 分母；无有效 target 的序列不生成
Datum。REINFORCE++ estimator 保留原生 loss 限制，不能与 CISPO 组合。
Router replay 和 mismatch 指标仍遵循已有契约。参见 [recipe](../../../../recipe/math/weaver_cispo.yaml) 和
[算法组件](algorithm-components.md)。

完整 step 的流程直接写在 `RemoteApiTrainer.train()`：准备概率，调用
`self._advantage_estimator.estimate(...)`，可选调用
`self._advantage_postprocessor.process(...)`，把返回的 `list[Trajectory]`
转换成 Datum，最后划分 optimizer batch。Estimator 和 postprocessor
把 advantage 写回已有 trajectory/sequence。CISPO 读取预测位置对齐后的值。
