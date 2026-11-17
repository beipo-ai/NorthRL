[English](../../../en/developer-guide/04-trajectory-pool/trajectory-semantics.md) | 简体中文 | [文档目录](../../index.md)

# Task、rollout 与 trajectory

**一次 rollout 对应一个 `Trajectory`。** 普通 trajectory 直接保存 tokens、loss mask 和最终 reward；只有 E2B 等需要分段的结果，才在内部增加 `segments`。没有独立流转的 `Segment` 类型，也不要求单轮先包装成一段。

<a id="task-to-training"></a>

## 从任务到训练

| 术语 | 含义 |
| --- | --- |
| task | 一道题或一个待完成的任务，可以执行多次 |
| item | dataloader 发给 worker 的一次执行请求 |
| rollout | 模型或 agent 执行任务的一次尝试，可能包含多次模型调用 |
| trajectory | 一次 rollout 的完整结果，由一个 `Trajectory` 表示 |
| segment | 分段 trajectory 内的一条序列记录，不是独立执行结果 |
| group | 同一道任务在一次采样中产生、参与 GRPO 比较的一组完整 trajectories |
| Datum | 提交给训练服务的一条序列数据；过滤后数量可能减少 |
| sample | 泛称，讨论数量时应明确是输入任务、trajectory 还是 Datum |
| trace | 保存的执行过程记录；过程快照不代表 rollout 已完成 |

```text
Direct:    item → rollout → Trajectory(tokens, loss_mask, reward) → Datum
Segmented: item → rollout → Trajectory(reward, segments=[{...}, {...}]) → multiple Datum objects
```

多轮不一定需要分段。E2B recorder 可以把 token 前缀连续的多次调用接到同一序列；无法接续时才开始新段。模型调用数、内部段数和完整 trajectory 数是不同的计数。

<a id="sampling-groups"></a>

## 同一道任务与独立采样组

`task_id` 标识原始任务；`group_id` 标识抽到这道任务后建立的本次采样组。
`rollout_repeat_n=R` 表示在本次采样组内执行 R 次，各自生成一个完整 trajectory。
同一道任务再次被抽到时，仍可使用相同 `task_id`，但必须获得新的 `group_id`。

```text
task_id = 7, rollout_repeat_n = 2
First draw  → group A → trajectories A0, A1
Second draw → group B → trajectories B0, B1
```

A、B 分别比较各自的两个 reward，不能因为 `task_id` 相同就合成四成员组。
`run_id` 为组内的 0 到 R−1 槽位，`rollout_id` 唯一标识其中一次执行。
身份唯一不等于排序可复现：当前同一训练批有相同 `task_id` 的多个组时，随机
`group_id` 仍会参与排序；固定 seed 也可能得到不同的 optimizer 拆批。这个问题尚未修复，
详见[训练器的排序边界](../06-trainers/remote-api-trainers.md#trajectory-ordering)。

<a id="trajectory-storage"></a>

## 一个公共 Trajectory，两种存储形式

```python
from northrl.northrl_types import Trajectory

single = Trajectory(
    tokens=[1, 2, 3],
    loss_mask=[0, 1, 1],
    reward=1.0,
    extra_fields={"old_log_probs": [0.0, -0.2, -0.3]},
)
assert single.segments is None

agent = Trajectory(
    reward=0.5,
    segments=[
        {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1],
         "old_log_probs": [0.0, -0.2, -0.3]},
        {"tokens": [4, 5], "loss_mask": [0, 1],
         "old_log_probs": [0.0, -0.4]},
    ],
)
assert agent.tokens is None
```

普通形式的 tokens/loss_mask 是真实字段，不是到内部单段的代理。分段形式的 `segments` 是内部字典列表，保存 tokens、loss_mask、logprobs 等序列字段。两种形式互斥，不同时保存两份 token 数据。

以上描述 rollout 结果。内部 managed SFT 的 `SampleRef` 载体是例外：通过 `extra_fields["sample_ref"]`、`input_token_count` 和 `datum_id` 引用服务端样本，本地不保存 tokens、loss_mask 或 segments；由 Weaver 读取实际训练数据。

reward、status、is_val、执行身份、score 和 GRPO advantage 属于完整轨迹。内部段不保存独立 reward 或执行终态。worker 一次提交完整对象；pool、GRPO 和 trainer 准备接口始终传递 `list[Trajectory]`。

训练器在 reference KL、OPD teacher logprob、重算 logprob、MIS 和 Datum 转换内部遍历序列：普通 trajectory 使用自身，有 segments 时使用内部记录。它们保留原对象引用并写回序列字段；不会把单轮转换成独立 Segment，也不会把一个内部段重新当作完整 Trajectory。

远端 `RemoteSegment` 是 E2B wire 协议的序列记录类型，不表示完整 rollout。本地 worker 将 `RemoteRolloutResult` 中所有段的 token 数据放入同一个 `Trajectory` 的内部字典列表，并在完整轨迹对象上保存最终 reward。

`RemoteRolloutTrace` 保留 Trace 类名，表示远端 rollout 仍在运行时保存的过程快照；已有 trace 不代表执行完成。`RemoteRolloutResult` 表示向本地训练流程交付的远端结果，并携带最终 reward；正常执行结束后生成；超时恢复也可把最后一份有效快照附在 `env_error` 结果中，协议层 reward 为 0。这不表示任务已评测且得零分：worker 会记录 `status="failed"`、`reward=None`，并丢弃可训练 segments。

两者实际保存序列记录的字段都叫 `segments`，新代码使用 `trace.segments` / `result.segments`，默认 JSON 输出和 JSON Schema 也使用 `segments`。内存中的 Pydantic trace/result 合约仍接受旧输入字段 `trajectories`，重新输出统一为 `segments`；同时传入新旧字段会报错。紧凑的 schema-v2 wire 格式要求使用 `segments`，不支持此旧别名。旧 Python 属性 `.trajectories` 不再提供，读取新 JSON 的调用方也需要使用 `segments`。

<a id="grpo-statistics"></a>

## GRPO 的统计单位

GRPO 从同一 group 的 N 个完整 trajectory 各读取一次最终 reward：

```text
baseline = (reward_1 + ... + reward_N) / N
advantage_i = (reward_i - baseline) / (std(reward_1, ..., reward_N) + epsilon)
```

主 GRPO 路径在 N > 1 时使用样本标准差，N = 1 时 advantage 为 0。普通 trajectory 的 token 和分段 trajectory 的所有内部段使用所属完整轨迹的同一个 advantage。

主训练调用 `northrl.algorithm.advantage.grpo` 中的 `compute_grpo_advantage_for_trajectories()`，直接读取 trajectory 的标量 reward。
它的历史参数 `use_run_ids` 现在只控制标准差口径和相关数值类型，不再控制分组；主路径传入 `True`。
`compute_grpo_outcome_advantage()` 则是保留的张量接口，接收 `[序列数, token 数]` 的 reward 张量；当前只有测试调用，生产 trainer 不调用它。该函数也从 `northrl.algorithm.advantage.grpo` 导入。

`validate_trajectory_groups()` 从 `northrl.algorithm.groups` 导入，检查 rollout ID 是否重复，以及未跳过组中的 reward 是否有效；提供 `expected_rollouts` 时，还检查每个预期的 `run_id` 是否恰好出现一次。旧模块 `northrl.algorithm.core_algos` 已删除，调用方需要直接导入上述实现。

例如，同组两条 trajectory 的 reward 分别为 0 和 1；第一条直接保存一条序列，第二条内部有三段。**仍是两条 trajectory，baseline 为 0.5**，不会因为总共有四条训练序列而变成 0.75。

成功但没有可训练序列的结果可以显式提交 `Trajectory(segments=[], reward=有效值)`；它仍参与 baseline，只是不产生 Datum。同组 R 个成员收齐后，完整轨迹的状态非 `success`、reward 无效或已标记跳过的成员计为失败，数量为 F。F ≤ R/2 时保留全部有效成员，均匀随机、无放回抽取 F 个不同有效成员，各复制一次补齐失败槽位；F > R/2 时整组不训练。有效零段成员可被复制，零分、同分和已评分的 E2B `task_failed` 都不算失败。trainer 在原始指标记录后、advantage 和 logprob 重算前补齐，所有分组 GRPO/OPD 模式使用同一规则。

<a id="group-delivery"></a>

## 完整组的提交与交付

1. dataloader 为每次采样分配新 `group_id`；同组 R 次尝试各有唯一 `rollout_id`，`run_id` 为 0 到 R−1，`expected_rollouts` 为 R。`task_id` 标识原始任务。
2. worker 构造并提交一个完整 `Trajectory`。`execute_rollout` 暂存对象，待自定义 `rollout` 正常返回后才写入 pool。
3. 训练任务正常返回但没有结果时生成 skipped trajectory；返回 `"fail"` 或异常时记录 failed trajectory，异常仍继续传播，不自动重试。验证任务直接进入 `rollout()`，提交的结果交给 validator，不走训练 pool 的终态补全。没有独立的完成记录通道。
4. pool 等待同组 R 个槽位的原始对象，只检查可修复性。fully-async 还要求有效成员至少有一个训练目标；不合格组不占就绪名额，其他模式仍交付原始对象。
5. trainer 先记录原始指标，再按上述规则补齐失败槽位。GRPO 和准备阶段保留完整轨迹对象，token 计算及 Datum 转换在内部处理对应序列，batch 调度不变。

同一结果的重复提交在等待中、就绪以及最近 1024 个已消费或已丢弃 group 内保持幂等；冲突结果被拒绝，去重记录有界。

`GroupedTrajectoryStore` 用四个容器管理这段生命周期：

| 容器 | 保存的内容 |
| --- | --- |
| `_groups` | 未凑齐的组：`group_id → run_id → Trajectory` |
| `_ready_groups` | 已凑齐的组：`group_id → list[Trajectory]`，按凑齐顺序入队，由 `data.train.group_order` 选择 FIFO（默认）或 LIFO 交付 |
| `_rollout_signatures` | 等待中、就绪和最近已取走结果共用的 `rollout_id → 签名` 索引 |
| `_recent_groups` | 按 trainer 取走或 pool 丢弃顺序记录 `group_id → rollout ID 列表`，超过 1024 组时删除最旧组的签名 |

这里的“已消费”指 trainer 从 pool 取走，尚不意味着训练完成；丢弃的组也进入这份有界历史。签名保留到对应历史组过期，重复提交不会重新入队，也不会刷新队列或历史记录的顺序。

令 B = `data.train.batch_size`，R = `data.train.rollout_repeat_n`。分组 RL 的
`sync`、`batch-async` 和 `fully-async` 统一直接调用 `take_groups(num_groups=B)`，
正常交付 B 个完整组、B × R 条完整轨迹。不足 B 组返回 `None`；即使有
积压，一次也最多取 B 组。按 `data.train.group_order` 选择完整组，内部 segments 随完整轨迹对象保留。

常规取组不另等 loader buffer 耗尽或 worker 静止。`sync` / `batch-async`
通常由调度器每次放行一个 B 组输入批次；`fully-async` 可跨输入批次凑组，
其余组留在 pool。sync 在训练及权重同步后放行下一批，batch-async 是否提前
放行仍由 staleness 控制，与单批的 optimizer 更新次数独立。

公开 RL loader 会跨 epoch 循环填满 B 个输入。`sync` / `batch-async` 只有在底层有限源的
`is_finished()` 明确返回 `True` 且 worker 全部静止后，才允许不足 B 个完整组
的尾批；fully-async 分组 RL 会丢弃不足 B 组的尾批并记录排除计数。缺成员诊断不必等整个源结束：不足 B 组且当前 buffer 耗尽、worker
静止时，仍有 pending 组就报错。

`data.train.group_order` 由 Controller 传入内部 pool，`fifo`（默认）优先最早
凑齐并进入 ready 队列的组，`lifo` 优先最近凑齐的组。例如 A、B、C 依次凑齐，
取两组时分别得到 A、B 或 C、B；两者都不拆组，组内仍按 `run_id` 排序。
新旧仅指整组完成顺序，不代表发起时间或采样模型版本。持续使用 LIFO 时，
老组可能长期等待。就绪组超过 B 时，该策略影响批次成员，最常见于 fully-async
积压时；所有模式仍最多取 B 组。GRPO trainer 后续会按任务与 group 身份重新
排序，因此 pool 顺序不保证 optimizer 执行顺序。

以上是通用分组 pool 的调度规则。`E2BRolloutWorker` 接受 `weight_sync.mode="sync"` 和 `"fully-async"`，`RemoteRolloutTrace` 仍要求至少一个 segment，但经过评测的 `RemoteRolloutResult` 可以没有 segment；worker 会保留 `Trajectory(segments=[], reward=...)`，其 reward 仍在组内统计中计一次。

`get_trajectories()` 返回 `None` 才表示等待。会交付跳过组或零序列组的模式仍返回完整轨迹对象列表；trainer 跳过 optimizer 更新并推进已消费轮次。fully-async 会在入就绪队列前排除不合格组。

内部 `GroupedTrajectoryStore.take_groups(num_groups=...)` 按完整组取数，
`SimpleTrajectoryStore.take_trajectories(num_trajectories=...)` 按完整轨迹
取数；二者均要求具体正整数，不接受 `None`。数量不足时返回 `None`，不部分
消费。分组路径由 `take_groups(B)` 直接判断；Simple/SFT 则等当前 loader
整轮已领取完且 worker 静止后，传入实际整轮的 `ready_trajectory_count()`，
因此 SFT 可按 token budget 交付可变数量的 trajectory。

<a id="batch-units"></a>

## Batch 和 batch_size

`Batch` 是按行保存字段的通用容器，不是主训练流程的必经节点。`Batch.from_trajectories()` 一行保存一条完整 trajectory，嵌套 segments 不增加行数；`to_trajectories()` 保留这层关系。转换前必须先去掉 padding。没有独立的 segment batch 转换接口。

| 位置 | 单位与含义 |
| --- | --- |
| `data.train.batch_size`（RL） | repeat 前的一批原始输入任务数 |
| dataloader buffer | repeat 后等待领取的执行请求 |
| 分组 pool `groups_per_batch` | 所有模式常规出批的完整 group 数 B；积压时也最多取 B 组 |
| 分组 pool `group_size` | 每组预期的完整 trajectory 数 R |
| pool / trainer 的 `list[Trajectory]` | 完整执行结果数，包括失败、跳过及成功零序列结果 |
| `Batch.metadata["batch_size"]` | 这个容器当前的行数，包含 padding；通过 `from_trajectories` 构造时原始每行是一条完整 trajectory |
| `algorithm.optim.batch_size`（RL） | 正整数为一次 optimizer 更新的目标 Datum 数；sync / batch-async / fully-async 下省略或 `null` 表示已交付批次的全部 Datum |
| `algorithm.optim.mini_batch_size`（RL） | 一次 actor forward-only 或前向/反向请求的 Datum 数上限 |
| SFT 输入批次 | 以 token budget 组织独立 Datum，不是固定任务数 |

`algorithm.optim.batch_size` 的 `null` 表示整批更新，与内部取组/取 trajectory API
必须传具体数量是不同层次。sync / batch-async / fully-async RL 省略该配置或设为
`null` 时，每次交付的 B 个完整组经准备和序列转换得到的全部 Datum 组成一个
optimizer batch；只有前述有限源尾批可少于 B 组。调度器放行输入的时机不改变
这个取组数量，也不要求分组 pool 等待所有 worker 静止。

每个 group 先按完整轨迹计算 GRPO advantage，再转换整个已交付批次。`mini_batch_size` 只限制每次前向/反向请求的 Datum 数；全部请求累积梯度后执行一次 optimizer 更新，不打乱、不补齐。被跳过的批次（包括没有可训练 Datum 时）仍不更新参数。

显式正整数继续沿用 Datum 级 optimizer 拆批、随机打乱及末批补齐规则，可能让同一 trajectory 的序列进入不同 optimizer batch。`no-sync` RL 必须提供正整数；SFT 保留完整 token-budget batch，`batch_size` 可省略或设为 `null`。完整组交付与 optimizer 更新边界是两层独立语义；每个已交付批次一次更新本身不改变损失归一化，也不保证每条 trajectory 等权。

<a id="prompt-mean"></a>

## Prompt-Mean 的比较组权重

`algorithm.loss.reduction.mode: prompt_mean` 按 `(model_tag, group_id)` 定义完整比较组，
不是按 `task_id`、trajectory 或 segment 求平均。设本次 optimizer 更新有 `G`
个有效组，组 `g` 的所有 rollout/segment 合计 `T_g` 个原始有效 response token，
每个 token 的策略损失权重为 `1 / (G * max(T_g, 1))`。工具上下文、prompt 和
padding 不参与分母；MIS rejection 不改变 `T_g`，被拒绝 token 只清零其贡献。
因此拆分 segment 或前向/反向请求不会改变组权重。

参考：[Mercor / SkyRL 训练指南](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/) 的 Step 4、Token aggregation 第 3 条；
[MiMo-V2.6 技术报告第 8 页 §4.1、公式 (1)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=8) 和
[第 21 页 §5.1](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=21)。这些来源说明聚合定义；下述配置限制属于 NorthRL 当前实现。

零 advantage 组和成功零 token 组保留在 `G` 中；修复组按替换后成员计算，不可修复组排除。成功零序列
trajectory 仍按既有规则参与所属组的 reward/advantage 计算，不因无法生成 Datum
而从组成员中消失。整批没有可训练 token 时跳过 optimizer 更新。

此模式要求 `algorithm.optim.batch_size` 省略或为 `null`，每批完整组一起更新；
`mini_batch_size` 仅拆请求。首版仅支持 GRPO 原生 `importance_sampling` / `ppo_clip`，
关闭 entropy、KL 和 custom loss。原有 loss modes 不变。配置和数值对比见
[训练器配置](../11-configuration-reference/trainer-config.md#prompt-mean-loss-aggregation)。

<a id="counts-and-dumps"></a>

## 计数与保存

序列计数统计实际可处理的序列数：普通 trajectory 为 1，分段结果为内部段数，零序列结果为 0。这是计数单位，不是新数据类型。trajectory 数始终按完整轨迹对象计数。

| 指标 | 含义 |
| --- | --- |
| pool `ready_trajectories` / `ready_groups` | 就绪的完整执行数 / 完整比较组数；sync / batch-async 包含跳过组，fully-async 仅包含合格组 |
| pool `ready_sequences` | 就绪且可训练的序列数；旧 `ready_segments` / `finished_trajectories` 保留兼容 |
| `batch/trajectory_count` | 传入 trainer 的完整 rollout 次数，包含失败/跳过/空结果 |
| `batch/segment_count` | 输入完整轨迹对象保存的全部段数 |
| `batch/resampled_group_count` | 本批次修复的组数 |
| `batch/resampled_trajectory_count` | 本批次通过复制有效成员补齐的失败槽位数 |
| `algorithm/trajectory/reward_observed_count` / `algorithm/trajectory/reward_mean` | 有限最终 reward 的完整轨迹对象数 / 每条完整轨迹的 reward 只计一次的均值 |
| `batch/trainable_segment_count` | Datum/mask 后的有效训练段数，包括失败槽位替换成员，排除 optimizer 补齐复制 |


同步及 batch-async 的周期进度日志按当前 dataloader 输入轮次统计：

```text
Progress [sync]
  Rollouts: 28/64 finished | 36 in progress | 0 queued
  Groups: 3/8 complete
  Segments: 39 ready for training
```

`finished` 按 pool 已接收的完整 `Trajectory` 计数，包含失败、跳过及成功零序列结果；
`in progress` 是已领取但尚未交付完整结果的 rollout 数，包含等待和重试；`queued` 是未领取数。
三者之和等于本轮 repeat 后的执行请求数，不使用 worker 活跃数推算。
`Groups` 按本轮所有成员已返回的组计数，包含整组跳过的情况。
`Segments` 沿用序列计数单位：普通 trajectory 计 1，分段结果计内部段数，零序列计 0；
分组训练只计完整且未跳过的组，尚未经过 trainer 的最终 Datum 过滤。
这些计数从同一份按 `rollout_id` 去重的轮次快照读取，trainer 取走结果后仍保留，
下一输入批次放行时重置。日志观察不改变 pool 的完整组交付边界。
SFT 不显示 GRPO 的 `Groups` 行。

fully-async 跨输入批次连续统计执行进度，pool 库存单独展示：

```text
Progress [fully-async]
  Rollouts: 1240 finished total | 36 in progress | 20 queued
  Pool groups: 3 ready | 5 partially returned | 8 per batch
  Pool segments: 39 ready for training
```

`finished total` 是当前 loader 会话已交付完整结果的累计 rollout 数；新输入批次及
trainer 取走结果不会清零。`in progress` 包含来自较早输入批次的未完成执行，
重试不增加逻辑执行数。`queued` 只统计当前已加载但未领取的执行项，不代表整个数据集剩余量。
完成计数只在已领取的 `rollout_id` 第一次交付终态结果时递增，重复或过期结果不重复计数。
计数器只保存尚在执行的 ID 和累计数字，不永久保存所有已完成身份；loader 每次发出
新任务时生成新的 ID，内部重试沿用当前 ID。重启、显式 reset 或恢复时跳批会重新开始统计。

`Pool groups` 的 `partially returned` 仅统计已经收到部分结果但还未凑齐的组，
尚无任何结果返回的组不计入；`ready` 只包含合格完整组，fully-async 会在入队前排除不可修复组和无训练目标的组。
`per batch` 表示该模型每次常规交付所需的完整组数。`Pool segments` 只计完整有效组的序列。
这些是当前库存，trainer 取走后下降；多个模型分别展示各自库存及出批目标。
无身份统计接口的自定义 loader 保留 worker 活跃数及 pool 库存展示。

prepared trajectory dump 使用 `format_version=3`：普通 trajectory 保持顶层 tokens/loss_mask，分段 trajectory 保存内部 `segments` 字典列表，reward/status/身份保存一次。`num_trajectories` 是完整轨迹对象数，`num_sequences` 是序列数，`num_segments` 保留旧计数兼容。重放兼容先前 v2 嵌套格式与旧 flat dump，并拒绝无法恢复完整比较组的输入。

<a id="related-guides"></a>

## 对应开发指南

- [核心数据类型](../02-core-architecture/data-types.md)
- [轨迹池](trajectory-pool.md)
- [Rollout 工作进程](../05-rollout-workers/overview.md)
- [远端训练器](../06-trainers/remote-api-trainers.md)
- [同步 Rollout 与优化器更新](../06-trainers/sync-rollout-semantics.md)
- [运行进度与实时指标](../08-features/live-metrics.md)

新的字段字典、查找方式和口径见[训练 step 指标与 W&B 查看方式](../../metrics.md)。旧 rollout/datum/round 数量别名不再进入新 W&B 曲线。
