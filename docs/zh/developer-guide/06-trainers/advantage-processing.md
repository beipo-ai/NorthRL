[English](../../../en/developer-guide/06-trainers/advantage-processing.md) | 简体中文 | [文档索引](../../index.md)

# 工具错误优势重分配

工具错误优势重分配在 estimator 完成后、prepared dump 和 MIS 之前，对**当前 step 的全部完整
trajectory**调用一次。统计池跨 group、model tag 和 segment；与 optimizer batch、
mini batch 或远端请求分块无关。失败和 group-skipped 的 trajectory 不参与。

## 配置

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: grpo
      params:
        norm_adv_by_std: true
    postprocessing:
      mode: tool_error_rebalance
      params:
        negative_multiplier: 2.0
        positive_scale_max: 2.0
        negative_scale_min: 0.5
```

这是 algorithm 块的局部示例；loss、optim、rollout 等继续使用原 recipe 配置。
省略 postprocessing 或使用 `mode: null` 时，不做后处理。关闭时
`params` 必须为空。设置 `mode: tool_error_rebalance` 即可启用重分配。

| 参数 | 含义 | 范围 |
| --- | --- | --- |
| negative_multiplier | 错误轮次的 response token 的负优势惩罚倍数 κ | 有限且 > 1 |
| positive_scale_max | 正优势 clean token 的放大上限 | 有限且 ≥ 1 |
| negative_scale_min | 负优势 clean token 的保留比例下限 | 有限且 0 < 值 ≤ 1 |

## 流转数据

直接接口为 `ToolErrorRebalancePostprocessor.process(trajectories, metrics) -> list[Trajectory]`。
优先读取已有的 response-only `token_advantages`，没有 token 估计时才广播完整轨迹的
标量 `advantage`。返回同一个 trajectory 列表，在原 sequence 上保存 response-only
`token_advantages`，不复制 token buffer。普通 trajectory 的诊断字段放
`extra_fields`；分段轨迹则放在各自的 segment 字典中。一个 segment 示例：

```python
{
    "tokens": [10, 11, 12, 13],
    "loss_mask": [0, 1, 1, 1],
    "old_log_probs": [0.0, -0.2, -0.3, -0.4],
    "turn_annotations": [{
        "version": 1,
        "turn_id": "turn-1",
        "response_span": [1, 4],
        "source": "northgate",
        "finish_reason": "tool_calls",
        "truncated": False,
        "tools_declared": True,
        "diagnostics": {
            "checked": True,
            "malformed": True,
            "detected": 1,
            "invalid": 1,
            "reasons": ["invalid_arguments_json"],
        },
    }],
}
```

`response_span` 是本 sequence 的 tokens 下标 `[start, end)`，只能覆盖 response，
不能覆盖 context、token 0 或与其他轮重叠。**错误轮次的 response token 指被判为 error 的整轮 span
内所有 response token**，不是解析器定位出的几个错误字符。诊断传输与 span 定位由
rollout 链路完成，此模块只消费已有 annotation。

- `error`：可靠的工具调用格式错误，例如参数 JSON 错误、缺工具名、重复参数。
  undeclared_tool 要求当轮声明了工具；incomplete_tool_call 要求正常结束原因。
- `clean`：checked=true、malformed=false、reasons=[] 且 invalid 为 0 或缺省。
- `unknown`：缺诊断、不支持的版本/source、截断/外部停止、未知原因或证据不足。
  unknown 原值保留，既不提供也不接受补偿。

error/clean/unknown 是处理时从 annotation 临时推导的 token 状态。
原 scalar advantage 保留作统计；训练读取写出的
`token_advantages`，它只包含 `loss_mask[1:] == 1` 的 response 位置，顺序不变。
`base_token_advantages` 保存同样长度的原始优势以支持重复后处理；确实额外保存一份
**优势数组**。`RemoteApiTrainer.train()` 每次重新估计前调用
`algorithm/advantage/tool_error_rebalance.py` 的 `clear_postprocessing_cache(trajectories)`，清除
缓存和对应的旧 shaped 输出，即使本次后处理已关闭；没有 saved base 的 token
估计会保留。独立调用组件的调用方也须在重估前调用该函数；对同一次估计重复
`process()` 则不清缓存，直接复用 saved base。compact 输入
使用 `array("d")` 保存两份优势，prepared dump 同时记录 base、shaped 和 annotations。
两份数组保持独立，即使全部诊断都是 unknown 也保留这些字段。处理时先验证 span 并
计算所有输出，再统一写入 sequence；状态缓冲区只覆盖 response，逐段归约不保留
完整 token 长度的 coverage 数组和整个 step 的筛选副本。最终输出暂存后即释放
对应临时数组。

## 整个 step 的计算

按每个 token 的原始优势符号分池：Pe/Pc 是正优势 error/clean 的**优势之和**；
Ne/Nc 是负优势 error/clean 的**优势绝对值之和**，不是 token 数（e = error，c = clean）。

```text
alpha = min(1 + Pe / Pc, positive_scale_max)       # Pc=0 时 alpha=1
beta  = max(1 - (κ-1) * Ne / Nc, negative_scale_min) # Nc=0 时 beta=1
正 error → 0；正 clean → alpha*A
负 error → κ*A；负 clean → beta*A
零优势和 unknown → 原值
```

多轮 trajectory 中，每轮的 `response_span` 决定其 response token 的状态。
如果中间多轮出错，就把这些轮中各 token 的优势**累加到对应的错误池**，不再按出错轮数
额外乘倍数：每个正优势 error token 归零，每个负优势 error token 只乘一次 κ。
整个 step 内同符号的 clean token 共用一个 alpha 或 beta，包括其他 trajectory 和 segment 的 token。

补偿池为空或缩放被上下限截断时，优势之和未必守恒。
W&B 中这些指标的前缀为 `01_algorithm/extra/advantage_postprocess/`：

| 指标后缀 | 含义 |
| --- | --- |
| `zero_advantage_tokens` | 处理前优势为 0 的 response token 数 |
| `zero_advantage_error_tokens` | 处理前优势为 0 的 error response token 数 |
| `positive_error_advantage_sum`、`positive_clean_advantage_sum` | 处理前的 Pe 和 Pc |
| `negative_error_advantage_abs_sum`、`negative_clean_advantage_abs_sum` | 处理前的 Ne 和 Nc，均为非负数 |
| `positive_advantage_sum_before`、`positive_advantage_sum_after` | 处理前/后的正 response 优势之和 |
| `negative_advantage_abs_sum_before`、`negative_advantage_abs_sum_after` | 处理前/后的负 response 优势绝对值之和 |
| `positive_scale`、`negative_scale` | 实际使用的 alpha 和 beta，无量纲倍数 |
| `positive_scale_clamped` | 无单位的 0/1 标记：截断前的 alpha 大于 `positive_scale_max` 时为 1；恰好等于上限时为 0 |
| `positive_no_clean_tokens` | 无单位的 0/1 标记：Pe > 0 且 Pc = 0 时为 1，表示正优势 clean 池为空，无法接收被清零的正优势 error 补偿 |

`positive_no_clean_tokens` 不是 token 数，也不表示所有 clean token 都不存在，仍可能有
零优势或负优势 clean token。正优势 clean token 的数量看 `positive_clean_tokens`。
对应的 negative 标记分别表示截断前的 beta 低于下限，以及存在负优势 error 但负优势 clean 池为空。
处理前/后的优势之和包含保持原值的 unknown token；这里统计的是原始 response 优势，
不承诺经过 loss reduction、MIS 或 optimizer 更新后的梯度守恒。

`build_postprocessor()` 根据已验证配置创建 `ToolErrorRebalancePostprocessor`。
`RemoteApiTrainer.train()` 对完整 step 先调用 estimator，再调用后处理器。
重分配使用 estimator 的最终输出，包括估计阶段已经完成的归一化或 whitening。
各算法支持的组合见[算法指南](algorithm-components.md#algorithm-guides)。
SFT 没有 advantage estimator，不能启用后处理。
