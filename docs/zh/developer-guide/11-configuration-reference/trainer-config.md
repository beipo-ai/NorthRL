[English](../../../en/developer-guide/11-configuration-reference/trainer-config.md) | 简体中文 | [文档目录](../../index.md)

<a id="trainer-configuration"></a>

# 训练器配置

本页介绍训练器选择、调度、优化器批次与训练控制。
算法组合与组件配置见[算法组件指南](../06-trainers/algorithm-components.md)。

<a id="trainer-selection"></a>

## 训练器选择

控制器直接根据 `algorithm.type` 选择训练器：

| `algorithm.type` | 训练器实现 |
|---|---|
| `grpo` | `RemoteApiGrpoTrainer` |
| `opd` | `RemoteApiOpdTrainer` |
| `sft` | `RemoteApiCrossEntropyTrainer` |

生产训练使用 `backend.type: weaver`；测试可以使用 `mock`。

<a id="grpo-and-opd"></a>

### 使用 advantage 的训练

通过 `algorithm.type` 选择 trainer 类型，并在对应配置块中设置 estimator、
可选后处理和 loss。

```yaml
algorithm:
  type: grpo
  # Configure optim, advantage, and loss here.

backend:
  type: weaver
```

<a id="supervised-fine-tuning-algorithmtype-sft"></a>

### 监督微调（`algorithm.type: "sft"`）

使用 Weaver 后端进行监督学习。控制器根据
`algorithm.type: sft` 在内部选择 `RemoteApiCrossEntropyTrainer`；
`cross_entropy` 是损失函数。

```yaml
algorithm:
  type: sft

backend:
  type: weaver

training:
  total_steps: 100
```

## Weaver metrics-v1

使用 Weaver 后端且启用 W&B 日志时，所有返回的 v1 指标会自动写入 NorthRL
现有 W&B run 的 `weaver_metrics_v1/` 命名空间。这是增量功能：现有 RL/旧版
指标的名称、数值及图表横轴不变。没有仅限默认指标的白名单；启用的诊断指标
（包括 `entropy/top10_mean`）也走同一路径，不会创建第二个 W&B run。
需要包含观察回调接口的 `nex-weaver>=1.19.4`。

本地持久化与 W&B 独立，复用 SDK 现有写入逻辑：

```yaml
backend:
  type: weaver
  metrics_store:
    enabled: true
    path: null
```

默认 `enabled: true`、`path: null`，相对于 NorthRL 客户端进程工作目录写入
`./weaver/.logs/metrics-<model-id>/`。自定义路径是 `metrics-<model-id>/` 的父目录。
`enabled: false` 仅关闭本地写入，不关闭采集或 W&B。文件按指标追加，不按步骤
或参数创建。如果不启用 W&B 且关闭本地保存，本集成不会持久保留这些观测。

可用指标及采集频率由 Weaver 服务决定。NorthRL 接收这些观测，
由 `backend.metrics_store` 控制本地保存；在 `runtime.logger.backends`
中启用 `wandb`，即可写入 W&B。

<a id="weaver-scheduling"></a>

## Weaver 调度

通常省略 `backend.scheduling_policy` 或设为 `null`，使用服务默认调度。
此可选映射会原样传给 Weaver SDK；仅在服务方提供了支持的策略，且所用 SDK
接受该参数时填写。Weaver SDK 1.18.0 不支持此参数。

<a id="weaver-performance-tier"></a>

## Weaver 性能档位

`backend.performance_tier` 用于选择 actor 训练模型的 Weaver 性能档位。
支持 `normal`、`fast` 和 `flash`；省略该字段或设为 `null` 时，使用 Weaver
服务端默认档位。该配置仅作用于 actor 训练模型。

```yaml
backend:
  type: weaver
  performance_tier: flash
```

<a id="optimizer-batches-and-forwardbackward-requests"></a>

## 优化器批次与前向/反向请求

`algorithm.optim.batch_size` 控制每次优化器更新使用的 Datum 数量。
可选的 `algorithm.optim.mini_batch_size` 控制每个 actor `forward-only`
（重算 logprobs 或 mismatch 诊断）及 `forward_backward` 请求最多包含多少个 Datum。
重算按当前交付批次展开后的 sequence 顺序拆分，不打乱、不补齐；所有分批
forward 完成并校验后才写回 logprobs，再进行优化器拆批和更新。R2 的 routing ref
及 sample index 按各自 forward 请求绑定。KL reference forward 仍由
`algorithm.kl.ref_logprobs_batch_size` 单独控制。

Weaver RL 内置损失可设置非负整数 `algorithm.optim.submit_ahead`（默认 `0`，串行）。
`1` 允许重算或 FB 阶段内最多两个未完成请求，不改变拆批、归一化或 OPT 顺序。
它不跨优化器更新预提交，也不控制 SFT 的流水线。

```yaml
algorithm:
  type: grpo
  optim:
    lr: 1.0e-6
    batch_size: 64
    mini_batch_size: 16
```

此例将 64 个 Datum 分为四次各 16 个 Datum 的前向/反向请求，
累积梯度后执行一次优化器更新。损失归一化的分母在拆分请求前
基于完整优化器批次计算；减小请求大小不会增加优化器更新次数。

- RL 使用 `training.weight_sync.mode: sync`、`batch-async` 或 `fully-async` 时，省略
  `batch_size` 或设为 `null`，会将已交付批次转换后的全部 Datum 作为一个优化器
  批次；所有请求累积梯度后只更新一次。被跳过的批次（包括没有可训练 Datum
  的批次）不更新参数。
- 显式 `batch_size` 必须为正整数，用于按固定 Datum 数拆批。
  `no-sync` 模式下的 RL 必须提供这个正整数。
- `mini_batch_size` 默认为 `null`，表示不按 Datum 数量拆分请求。
  提供时必须为正整数；大于实际优化器批次大小时，整个批次只发送一个请求。
- `data.train.batch_size` 控制 rollout 前的输入任务数；
  一个任务可能产生多个 Datum，因此与优化器批次大小独立。
- Weaver 服务端的微批次配置控制设备执行，与 NorthRL 的请求和更新边界独立。
- SFT 每次优化器更新保留完整 token 预算批次，不受
  `algorithm.optim.batch_size` 影响；该字段可省略或设为 `null`，
  `mini_batch_size` 可拆分其请求。

分组 RL 的 `sync`、`batch-async` 和 `fully-async` 都取 B 个完整组，
B = `data.train.batch_size`，每组 R = `data.train.rollout_repeat_n` 条完整轨迹。
不足 B 组返回 `None`，额外组继续排队，不另等 worker 静止或 loader buffer 耗尽。
调度器控制输入放行，包括 `batch-async` 的 staleness 策略。公开 RL loader
跨 epoch 循环填满 B 个输入；只有有限源耗尽且 worker 静止时才允许较小的
完整组尾批。所选 estimator 在已交付批次转换为 Datum 之前计算 advantage。
advantage 计算和可选后处理都作用于完整 step，先于 MIS 和 optimizer 拆批；
optimizer batch 大小不改变这两个阶段的输入。

每个已交付批次只更新一次的配置示例：

```yaml
algorithm:
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
training:
  weight_sync:
    mode: sync  # 或 batch-async / fully-async
```

省略指最终解析后的配置中没有该字段；若继承的父配置已设置整数
`batch_size`，需要显式使用 `null` 覆盖。

已交付批次按原顺序保留转换后的全部 Datum，不补齐。
显式整数小于已交付批次大小时，Datum 会被随机打乱并拆为固定大小的优化器批次，
末批不足时补齐。
前向/反向请求的拆分不会额外补齐或复制 Datum。
损失按 `algorithm.loss.reduction.mode` 归一化；每个已交付批次只更新一次
本身不意味着每条完整轨迹等权。
跨请求梯度累积要求 Weaver 后端保留梯度，直到 `optim_step` 完成。

<a id="prompt-mean-loss-aggregation"></a>

## Prompt-Mean 损失归一化

`algorithm.loss.reduction.mode: prompt_mean` 让每个完整比较组等权，组内所有有效
response token 等权。分组键是 `(model_tag, group_id)`，不是 `task_id`；
同一任务再次采样仍是新组。组内所有 rollout 及其所有 segment 共用 token
分母，因此把同一 trajectory 拆成更多 segment 不会增加其权重。

该聚合定义的参考来源：

- [Mercor / SkyRL 训练指南](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)：Step 4 的 “Token aggregation” 第 3 条，定义各组等权、组内 token 等权。
- [MiMo-V2.6 技术报告第 8 页 §4.1、公式 (1)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=8)：分母为同一 prompt 下所有 response 的长度之和；[第 21 页 §5.1](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=21) 将其称为 `prompt-mean`。

下述 loss 支持范围和 optimizer 拆批限制属于 NorthRL 当前实现的限制。

设一个 optimizer batch 中有 `G` 个有效组，`T_g` 是组 `g` 中原始有效 response
token 总数，每个 token 的策略损失权重为 `1 / (G * max(T_g, 1))`。
prompt、工具上下文和 padding 不计入。分母取 **MIS rejection 之前**的 mask；
被拒绝 token 贡献为零，不提高剩余 token 的权重，importance weight 也不参与分母。
零 advantage 组、成功但没有训练 token 的组仍计入 `G`；修复组按替换后成员计算，不可修复组排除。
整批没有可训练 token 时不执行 optimizer 更新。

例如，A 组两个回答分别有 10、30 个 token，平均 token loss 为 2、6；
B 组两个回答均有 10 个 token，平均 loss 为 1、3。每个回答恰好是一条序列时：

| 归一化方式 | 结果 |
|---|---|
| `token_mean` | `(10×2 + 30×6 + 10×1 + 10×3) / 60 = 4` |
| `datum_mean` | `(2 + 6 + 1 + 3) / 4 = 3` |
| `prompt_mean` | `((10×2 + 30×6) / 40 + (10×1 + 10×3) / 20) / 2 = 3.5` |

`prompt_mean` 要求 `algorithm.type: grpo`，使用原生 `loss.mode: importance_sampling`
或 `ppo_clip`，并设置 `entropy_coeff: 0`、`kl.mode: null`。`optim.batch_size` 必须省略或为
`null`，确保本批完整组只做一次 optimizer 更新；可用 `mini_batch_size` 限制
前向/反向请求大小，拆请求不会重新计算组分母。支持 `sync`、`batch-async`、
`fully-async`，不支持 `no-sync`。

```yaml
algorithm:
  type: grpo
  loss:
    mode: importance_sampling  # 或 ppo_clip
    reduction:
      mode: prompt_mean
  entropy_coeff: 0
  kl:
    mode: null
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
```

<a id="training-control"></a>

## 训练控制

<a id="trainingtotal_steps--trainingepochs"></a>

### training.total_steps / training.epochs

设置且仅设置一个正数训练上限：

```yaml
training:
  total_steps: 200

# Or, for SFT only:
training:
  epochs: 3
```

RL 的一个训练步处理一个准备好的 rollout 批次。`sync`、`batch-async` 或 `fully-async`
RL 中省略 `algorithm.optim.batch_size` 或设为 `null` 时，实际训练的已交付批次
只更新一次；显式整数配置可产生多次优化器更新。
SFT 的一个训练步处理一个 token 预算批次，并执行一次优化器更新。
`training.epochs` 仅支持 `algorithm.type: sft`。

<a id="trainingsave"></a>

### training.save

检查点保存频率和本地 Weaver 检查点索引目录位于
`training.save` 下：

```yaml
training:
  save:
    every_n_steps: 10  # 0 disables periodic saves
    dir: "${oc.env:EXPERIMENT_PATH}/ckpt"
```

`dir` 为必填项。最终保存行为独立于 `every_n_steps`。

<a id="trainingresume"></a>

### training.resume

恢复设置位于 `training.resume` 下：

- `mode`：`disable`、`auto` 或 `from_path`
- `path`：`from_path` 使用的检查点路径
- `skip_dataloader`：非 SFT 的跳过行为；SFT 会恢复其保存的精确游标
- `max_retries`：训练失败后控制器自动重试的次数
  （默认：`1`；设置为 `0` 可禁用）

重试时，NorthRL 先关闭失败的控制器，再构建新的
控制器，并强制使用 `mode: auto`，使新的尝试从
`training.save.dir/weaver_checkpoints.jsonl` 加载最新条目。首次尝试仍然
遵循配置的 `mode`。

```yaml
training:
  save:
    every_n_steps: 10
    dir: "${oc.env:EXPERIMENT_PATH}/ckpt"
  resume:
    mode: disable
    path: ""
    skip_dataloader: true
    max_retries: 1
```

键盘中断和终止信号不会触发重试。进程意外退出或
进程丢失后需要重启驱动；如果新进程应从检查点索引
恢复，请配置 `mode: auto`。

<a id="trainingweight_syncpause_mode"></a>

### training.weight_sync.pause_mode

可选配置，用于在替换采样器时暂停 Weaver 引擎：

| 值 | 行为 |
|---|---|
| `null`（默认）/ 省略 | 保持原有同步行为，不暂停引擎 |
| `retract` | 撤回正在执行的请求，恢复后重新调度 |
| `in_place` | 保留暂停的请求及其 KV 状态 |

非空值必须同时满足以下条件，否则配置验证失败：

- `algorithm.type: grpo` 或 `opd`，且 `training.weight_sync.mode: fully-async`。
- `backend.type: weaver`，且 `training.weight_sync.freeze_inflight: true`。
- `full_ft`：省略 `models.actor.lora`、设为 `null`，或设为 `enabled: false`
  且 rank 省略或为 `null`。暂停作用于整个引擎，因此不支持 LoRA 适配器。

```yaml
training:
  weight_sync:
    mode: fully-async
    freeze_inflight: true
    pause_mode: in_place
```

详见[可选的引擎暂停](../08-features/weight-synchronization.md#optional-engine-pause)。

<a id="trainingweight_sync-for-sft"></a>

### SFT 的 training.weight_sync

训练器级别没有独立的跳过标志。WeightSyncController 负责采样器
权重的发布与同步行为。

对于 `algorithm.type: sft`，省略该配置块时，NorthRL 会推导出 `training.weight_sync.mode: no-sync`。
如果显式配置了冲突的模式，则无法通过配置结构
验证。

<a id="related-documentation"></a>

## 相关文档

- [远程 API 训练器](../06-trainers/remote-api-trainers.md) - 训练器行为和 Weaver Datum 协议
- [SFT 数据](../03-data-loader/sft-data.md) - 流式输入、token 预算批处理和精确恢复
- [权重同步](../08-features/weight-synchronization.md) - 权重同步模式
