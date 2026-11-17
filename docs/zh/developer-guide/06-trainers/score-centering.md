[English](../../../en/developer-guide/06-trainers/score-centering.md) | 简体中文 | [文档目录](../../index.md)

# Weaver Score centering（实验性）

Score centering（SC）需要配套的 SDK、服务端和 Megatron Bridge trainer 支持。目前仅支持使用内置 simple rollout worker 的单轮训练，不适用于 E2B Agent 训练配置。

<a id="configuration"></a>

## 配置

在常规 [GRPO recipe](../../recipe/math/README.md) 上覆盖以下配置。请替换整个 `algorithm.loss` 块，避免保留旧 loss 的参数。完整 recipe 仍需保留 optimizer、batch、数据、奖励和评测配置；下面不是可独立运行的完整配置。

```yaml
algorithm:
  type: grpo
  loss:
    mode: score_centering
    params:
      topk: 128
  sampling_logprob_source: inference
  entropy_coeff: 0
  kl:
    mode: null
  mis:
    enabled: false
    get_mismatch_metrics: false
  router_replay:
    enabled: false
rollout:
  type: simple
  sampling:
    temperature: 1
    top_p: 1
    top_k: -1
    min_p: 0
    min_tokens: 0
    repetition_penalty: 1
training:
  weight_sync:
    mode: sync
backend:
  type: weaver
```

权重同步必须设置 `mode: sync`，配置校验会强制检查。数据和奖励需要符合 `SimpleRolloutWorker` 的要求：其内置奖励会将 `<answer>...</answer>` 中的文本与 `ground_truth` 比较。

<a id="sampler-data"></a>

## 采样数据与训练对齐

service holder 为训练样本请求生成 token 的 top-k 概率。贪心评测和 warmup 不要求 SC 统计数据。

远程客户端和 simple worker 通过标准 `northrl_train` 封装保留 `sampler_distribution_data`。截断时，所有与 response 对齐的字段一起裁剪。转换为 Datum 时，这些数据会移到预测目标位置，prompt 对应行以 `mask=0` 填充，并保留最后一个 response token。

采样概率与 `old_log_probs` 相互独立。现有 advantage 归一化在 trainer 做求和归约前，将 token/sample 均值的分母计入 A；minibatch 补齐时重复的 Datum 也参与计算。参数更新边界见[完整 rollout 与 batch 语义](sync-rollout-semantics.md#optimizer-update-boundary)。

<a id="support-and-validation"></a>

## 支持范围与验证限制

初始支持范围是单轮任务、内置 simple worker 和纯 SC。暂不支持 MIS、自定义 loss、router replay、KL、熵正则，以及经过过滤或约束的采样；配置校验会拒绝这些组合。服务端或 SDK 缺少所需能力时，会明确报错。

现有验证仅覆盖 CPU 协议和梯度测试。原始实现记录中尚未进行真实 GPU optimizer 或训练效果实验。在完成这些检查之前，应继续将此功能视为实验性功能；配置示例不能作为端到端训练效果的证明。
