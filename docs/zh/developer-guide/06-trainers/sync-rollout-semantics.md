[English](../../../en/developer-guide/06-trainers/sync-rollout-semantics.md) | 简体中文 | [文档目录](../../index.md)

# 完整 rollout、同步训练与 SWE 指标

训练数据与分组遵循 [Task、rollout 与 trajectory](../04-trajectory-pool/trajectory-semantics.md)：一次 rollout 只产生一个完整 `Trajectory`，需要分段时把全部序列放在其 `segments` 内。任务执行阶段和实时观测见 [E2B 执行流程](../05-rollout-workers/e2b-runtime.md#lifecycle)与[实时指标](../08-features/live-metrics.md)。

<a id="optimizer-update-boundary"></a>

## 一次 optimizer 更新的范围

```yaml
data:
  train:
    batch_size: 8
    rollout_repeat_n: 8
algorithm:
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
training:
  weight_sync:
    mode: sync
```

每轮抽取 8 个输入任务，建立 8 个独立比较组；每组执行 8 次，共 64 个完整 rollout。假设过滤后留下 100 个 Datum，则执行 7 次 forward/backward，最后一次包含 4 个 Datum，累积梯度后只执行 1 次 optimizer update。全量模式不打乱、不复制补齐。

`algorithm.optim.batch_size` 省略或为 `null` 时，sync、batch-async 和 fully-async 都使用 pool 本次交付的完整批次。fully-async 按完整组交付，不等待整轮 worker 静止。`mini_batch_size` 只限制一次前向/反向请求的 Datum 数，不改变归一化或参数更新边界。no-sync RL 仍要求显式正整数；SFT 沿用 token-budget batch。显式正整数保留既有 Datum 级拆批和尾批补齐行为。

GRPO 每个完整 trajectory 读取一次 reward，再把同一个 advantage 应用于其所有内部序列。同一道任务再次抽样必须获得新的 `group_id`，不能按 `task_id` 合并跨轮比较组。`algorithm.loss.reduction.mode: prompt_mean` 按完整比较组等权，遵循相应的[配置约束和 token 归一化](../11-configuration-reference/trainer-config.md)；不会因 segment 数量变化而改变组权重。

<a id="identity-and-completion"></a>

## 身份与完成边界

| 字段 | 含义 |
| --- | --- |
| `task_id` | 原始任务标识 |
| `group_id` | 本次抽样建立的独立比较组；同题再次抽样也生成新 ID |
| `run_id` | 组内从 0 到 R−1 的执行槽位 |
| `expected_rollouts` | 本组预期的完整结果数 R |
| `rollout_id` | 一次逻辑执行的唯一 ID，环境重试沿用该 ID |
| `segment_index` | 一个分段结果内部的序列位置，不是独立 rollout |

worker 每次提交一个完整父结果，包含成功、失败、跳过或有效零序列结果。pool 保留同组 R 个原始结果。父状态非 `success`、reward 无效或预先标记跳过的成员有 F 个时，F ≤ R/2 则由 trainer 保留全部有效成员，并均匀随机、无放回抽取 F 个不同有效成员，各复制一次补齐失败槽位；超过一半失败才跳过整组。复制在记录原始指标之后、advantage 和 logprob 重算之前执行。没有可训练 Datum 的批次仍推进 training step，但不更新参数、不发布新权重。

序列的规范字段为 `segments`。部分内存 contract 接受历史别名 `trajectories`，但远端 trace/result 文件必须是含有 `segments` 的 gzip 压缩 compact schema v2；wire 读取器不会回退读取旧 JSON。trace 是过程快照，不代表 agent 或评测已经结束。

<a id="task-and-training-status"></a>

## SWE 任务状态与训练状态

| E2B 结果 | 父 `Trajectory.status` | 训练与指标语义 |
| --- | --- | --- |
| `success` | `success` | 有效任务结果，最终 reward 只计一次 |
| `task_failed` | `success` | 有效的 0 分任务结果，参与 GRPO；不因未解题而跳过全组 |
| `env_error` | `failed` | 环境或评测故障；没有有效 reward，丢弃部分训练序列，该失败槽位是否补齐由组内失败数量决定 |

命令失败且没有可信的本次 attempt 阶段标记时，按 `env_error` 处理，不能作为有效零分样本进入 GRPO。只有在 agent 命令派发前，才允许换新沙箱重试；一旦派发，执行器只尝试恢复本次已有结果，不会自动重新运行 agent。详见[执行时限与恢复](../11-configuration-reference/e2b-execution-budgets.md#recovery-behavior)。

原始任务状态保存在 `e2b_status`，不能与训练执行状态混用。成功的零 segment 结果仍参与组内 reward/advantage 计算，但不产生 Datum。上下文耗尽后仍可评测；如果 worker 实际截断了过长 segment，则整个 rollout 的训练 reward 和 `swebench_resolved` 清零。

<a id="metric-definitions"></a>

## 指标口径

下表使用当前规范日志名称。W&B 会添加固定排序编号，例如 `algorithm/trajectory/reward_mean` 显示为 `01_algorithm/01_trajectory/reward_mean`；其他日志后端保留规范名称。

| 指标 | 统计对象 |
| --- | --- |
| `batch/trajectory_count` | pool 交付的原始完整父结果，包含失败/跳过/有效零序列结果 |
| `batch/segment_count` | 原始父结果中保存的序列数，包含被跳过组内的序列 |
| `algorithm/trajectory/reward_observed_count`、`algorithm/trajectory/reward_mean` | 有效最终 reward 的父结果数及均值，每次原始执行只计一次 |
| `batch/failed_trajectory_count`、`batch/skipped_trajectory_count` | 训练执行终态，不是解题通过率 |
| `outcome/task_failed_count`、`outcome/env_error_count` | 任务未通过与环境/基础设施故障，分别计数 |
| `batch/empty_trajectory_count` | 本次 trainer 批次中没有保存序列的原始完整结果数 |
| `batch/group_count` | 本次 trainer 批次交付的原始比较组数；非分组算法省略 |
| `batch/resampled_group_count`、`batch/resampled_trajectory_count` | 通过替换失败槽位修复的组数，以及替换的 trajectory 数 |
| `batch/trainable_group_count`、`batch/trainable_trajectory_count` | 准备后至少贡献一个可训练 segment 的组数和父 trajectory 数 |
| `batch/trainable_segment_count` | Datum 转换与 mask 处理后具有有效 response 训练目标的 segment 数，optimizer 补齐前统计 |
| `batch/trainable_response_tokens_total` | 上述 Datum 中有效的 response 训练 token 总数，optimizer 补齐前统计 |
| `training/optimizer_step_count`、`training/forward_backward_call_count` | 实际完成的参数更新次数和前反向请求次数 |

例如一个 reward=1 的 rollout 保存了 3 段，另一个 reward=0 的 rollout 保存了 1 段，`batch/trajectory_count=2`、`batch/segment_count=4`、`algorithm/trajectory/reward_mean=0.5`。环境失败不产生有效评分，不作为伪造的 0 分加入 reward 均值。

原始输入数量和 reward 在失败槽位替换前统计。可训练数量在 Datum/mask 处理后统计，包含替换成员，但不包含单纯为 optimizer 补齐而复制的数据。这些指标的统计对象不同；原始保存的 segment 不一定可训练。

旧 `rollout/*`、`round/*`、`datum/*` 和 `training/*` 中的序列数量字段已退出训练步日志。尤其是 `round/task_count` 和 `round/accepted_sequences`，目前已不再作为图表名称。trainer 专用处理之前的资格数量不能直接改名为 mask 处理之后的可训练数量。更新次数方面，旧 producer 别名 `training/optimizer_steps` 和 `training/forward_backward_calls` 会映射到表中的当前名称。

批次统计既适用于同步批次，也适用于 fully-async 每次交付的批次；不能把 fully-async 的一次 trainer 批次误读为一个完整 dataloader round。运行时 `live/*` 快照保留在普通日志中，包含 attempt 重试工作量，不生成 W&B 历史行。详见[实时指标](../08-features/live-metrics.md)与[训练指标参考](../../metrics.md)。

<a id="tracing-and-dumps"></a>

## 追踪与保存

开启 `runtime.debug.enable_data_dump` 并设置 `dump_dir` 后：

- `rollout_traces/e2b_rollout_<rollout_id>.json.gz` 以 gzip 压缩的 schema v2 保存 compact remote segments、任务身份和采样版本。
- `rollout_rounds/step_<六位批次编号>.json` 保存本次 trainer 批次所有父结果，包括零序列、失败和跳过结果；同时保存训练 `status` 与任务 `e2b_status`。它通过 `DataDumper` 保存，沿用 `remote_api_dump_options.prepared_trajectories` 开关和 `dump_every_n_steps` 频率。文件编号及 JSON 的 `step` 与同批 prepared/datums dump 一致，从 0 开始；JSON 的 `training_step` 是完成轮次，从 1 开始。
- prepared trajectory dump 使用 `format_version=3`，父身份和 reward 只保存一次，序列嵌套在 `segments`；不退回旧 flat segment 格式。

训练清单里的 reward 是 worker 最终提交的训练评分；环境失败为 `null`。实时评测指标描述沙箱执行阶段，训练评分还会受到截断和完整组过滤影响，两者应分别解释。
