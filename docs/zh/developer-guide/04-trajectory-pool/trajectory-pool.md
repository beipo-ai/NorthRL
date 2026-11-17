[English](../../../en/developer-guide/04-trajectory-pool/trajectory-pool.md) | 简体中文 | [文档目录](../../index.md)

<a id="trajectory-pool"></a>

# 轨迹池

**完成约定：** 分组 pool 等待每个不同 rollout 的完整 `Trajectory`。`group_size` 按 rollout 数计数，`groups_per_batch` 按完成 group 数计数（fully-async GRPO/OPD 仅计有效组）；交付时保留每个 trajectory 及其全部嵌套 segments。进度区分 `ready_trajectories`、`ready_groups` 和可训练 `ready_sequences`；`ready_segments` 与 `finished_trajectories` 保留为序列计数的兼容名称。

**术语约定：** 一次 rollout 产出一个完整 `Trajectory`。普通轨迹直接保存 tokens；E2B 等分段结果才增加内部 `segments` 记录。worker、pool 和 trainer 准备接口传递完整 trajectory。详见[轨迹术语说明](trajectory-semantics.md)。

轨迹池收集、分组并批处理 rollout 工作进程提交的完整 `Trajectory` 对象。

<a id="overview"></a>

## 概览

**用途**：收集完整执行结果，等待完整比较组，并向训练器交付 trajectories

**主要特性**：
- 多存储区架构（每个模型标签对应一个存储区）
- 灵活的分组策略（简单、分组）
- 完整组交付与 SFT 整轮交付
- 权重同步协调
- 线程安全操作

<a id="trajectorypool"></a>

## TrajectoryPool

**位置**：`northrl/trajectory_pool.py`

完整 trajectory 收集的**主接口**。

<a id="constructor"></a>

### 构造函数

```python
def __init__(self, config: DictConfig)
```

**配置**：

下面是 Controller 派生的内部 pool 配置。分组 RL 中，B 来自 `data.train.batch_size`，
R 来自 `data.train.rollout_repeat_n`，`group_order` 来自 `data.train.group_order`（默认 `fifo`）。
Controller 仅为 `fully-async` 下的分组 GRPO/OPD 派生
`require_trainable_groups=true`；这是内部配置，不是公开 recipe 选项。
分组存储按完整组计数；简单存储按当前
loader 整轮实际产生的完整轨迹数交付。嵌套 segments 始终随完整轨迹对象保留。

```yaml
trajectory_pool:
  type: "default"
  groups_per_batch: 32  # B completed comparison groups per delivery
  group_size: 4  # R independent rollouts per group
  group_order: fifo  # 派生自 data.train.group_order，可选 fifo 或 lifo
  key_list: ["group_id"]  # A fresh comparison group for each sampling occurrence
```

<a id="core-methods"></a>

### 核心方法

<a id="put_trajectorytrajectory-trajectory---str"></a>

#### put_trajectory(trajectory: Trajectory) -> str

一次性存入一个完整 rollout：直接 token 字段或内部序列字典，以及最终 reward、终态和执行元数据。
返回 `"success"` 或 `"fail"`。pool 按 `model_tag` 路由，分组存储将一个完整对象
计为一次 rollout，与它包含多少 segment 无关。worker 不逐段提交。

```python
result = trajectory_pool.put_trajectory(trajectory)
```

成功零段且最终 reward 有限的 trajectory 是合法 GRPO 成员。
pool 保留原始结果，并按下文规则检查失败成员是否可补齐。
fully-async 分组 GRPO/OPD 排除不可修复或有效成员没有训练目标的组，
并单独记录拒绝统计；其他分组模式仍交付所有完整对象。
group 等待中、就绪待消费及最近 1024 个已消费 group 的窗口内，相同提交保持幂等，
冲突提交会被拒绝。缓存有界，不保证无限执行历史的去重。
紧凑数值数组直接对完整二进制缓冲区计算哈希，并包含类型和长度，
避免逐 token 展开为文本。其他字段的表示仍参与签名，
不会为了降低去重开销而省略训练数据或元数据。

权重同步不会拒绝完整 trajectory 提交。存储锁只保护共享数据的读取和修改。

<a id="get_trajectories"></a>

#### get_trajectories(model_tag: ModelTag | None = None) -> list[Trajectory] | None

从指定模型存储区获取完整 trajectories。`model_tag=None` 时尝试已有存储区；
显式指定的模型必须已有实例。只有 `None` 表示没有就绪批次。
fully-async 分组 GRPO/OPD 仅接收符合下文条件的有效组；其他分组模式仍为
失败组和成功零段组返回全部完整 Trajectory 对象。trainer 保留完整轨迹对象，
在内部遍历可训练序列；经过 trainer 侧过滤后没有可训练序列时，跳过 optimizer
更新，但推进已消费轮次。

分组 RL 的 `sync`、`batch-async` 和 `fully-async` 都直接调用
`take_groups(num_groups=B)`。常规批次包含 B 个完整组，即 B × R 条完整轨迹；不足 B 组返回 `None`，积压也不会使单次交付超过 B 组。
下一 loader 批次何时放行由调度器控制。组始终完整，所有嵌套 segments
都随完整轨迹一起交付。fully-async GRPO/OPD 的 B 组都须通过 pool
有效性检查；只有其他分组模式允许下文所述的较小有限源尾批。

```python
trajectories = trajectory_pool.get_trajectories(model_tag="actor")
```

<a id="is_emptymodel_tag-modeltag--none--none---bool"></a>

#### is_empty(model_tag: ModelTag | None = None) -> bool

检查轨迹池是否为空。

```python
if trajectory_pool.is_empty():
    print("No pending or ready trajectories")
```

**参数**：
- `model_tag`：要检查的特定模型（None 表示所有模型）

**返回值**：指定存储区（或所有存储区）没有等待中或就绪工作时为 `True`。被拒绝的组不影响 fully-async 分组 pool 判空；其他模式的零 segment 完成组在消费前仍不算空

<a id="get_model_tags---listmodeltag"></a>

#### get_model_tags() -> list[ModelTag]

获取所有活动模型标签。

```python
tags = trajectory_pool.get_model_tags()
# ["policy", "reference"]
```

**返回值**：具有活动存储区的模型标签列表

<a id="trajectorypoolinstance"></a>

## TrajectoryPoolInstance

**用途**：管理单个模型完整 trajectories 的独立池实例

**位置**：`northrl/trajectory_pool.py`

<a id="setup"></a>

### 设置

<a id="set_module_referencesdataloader-weight_sync_controller-activity_tracker"></a>

#### set_module_references(dataloader, weight_sync_controller, activity_tracker)

设置对其他模块的引用。

```python
instance.set_module_references(
    dataloader=dataloader,
    weight_sync_controller=weight_sync_controller,
    activity_tracker=activity_tracker
)
```

<a id="trajectory-management"></a>

### 轨迹管理

<a id="put_trajectorytrajectory-trajectory---str-1"></a>

#### put_trajectory(trajectory: Trajectory) -> str

将一个完整 trajectory 添加到此实例的存储区。

**返回值**：
- `"success"`：添加成功
- `"fail"`：添加失败

<a id="trajectory-store-types"></a>

## 轨迹存储区类型

轨迹池根据配置自动创建合适的存储区。
`key_list` 必须为空或仅包含一个分组键；指定多个键会抛出 `ValueError`。

内部取数 API 明确区分计数单位：

| 存储方法 | 数量单位 |
| --- | --- |
| `GroupedTrajectoryStore.take_groups(num_groups=...)` | 完整组 |
| `SimpleTrajectoryStore.take_trajectories(num_trajectories=...)` | 完整轨迹 |

二者都必须传入具体正整数，不支持 `None`；存储方法恰好取出指定数量，
不足时返回 `None` 且不消费任何对象。分组 RL 直接调用 `take_groups(num_groups=B)`，
不另做外层就绪判断。按 `group_order` 选择完整组，组内按 `run_id` 排列。
Simple/SFT 等当前 loader 整轮完成后，把实际 trajectory 数传给 `take_trajectories`。

<a id="simpletrajectorystore"></a>

### SimpleTrajectoryStore

**使用场景**：无需分组

**行为**：将完整 trajectory 追加到 `_ready_trajectories`，按完成的 loader 整轮交付；每轮 trajectory 数量可变

**配置**：
```yaml
trajectory_pool:
  key_list: []  # SimpleTrajectoryStore
```

**适用场景**：
- 样本相互独立的任务
- 无需轨迹分组
- 最简单、最快的选择

<a id="groupedtrajectorystore"></a>

### GroupedTrajectoryStore

**使用场景**：按本次采样的 `group_id` 组织完整 GRPO/OPD 比较组

**行为**：按 group 和 rollout 身份缓存原始 trajectories，`group_size` 个不同执行
全部到齐后组才完成。R 个成员中，完整轨迹的状态非 `success`、最终 reward 无效或已被标记
跳过的成员计为失败，数量为 F。F ≤ R/2 时保留全部有效成员，从中均匀随机、
无放回抽取 F 个不同成员，各复制一次补齐失败槽位；恰好一半失败也可修复。
F > R/2 时整组不训练。pool 只检查可修复性；所有分组 GRPO/OPD 模式都由
trainer 在记录原始指标之后、计算 advantage 或重算 logprob 之前执行复制。

**Fully-async 有效性条件**：组可修复，且有效成员中至少有一条普通或嵌套序列
长度大于 1，并且 `loss_mask[1:]` 中存在有效训练目标。
成功零段成员仍有效，也可被抽中复制。已评分的 E2B `task_failed` 会映射为完整轨迹的状态
`success`，不算失败。reward 为零或组内
相同均可接受。被拒绝的组单独汇总，不进入 ready 队列，也不占
`max_ready_groups` 背压名额，因此新 rollout 可继续补齐有效组，拒绝本身不消费
`train_step`。配置背压后，fully-async 的实际水位为
`max(rollout.max_ready_groups, data.train.batch_size)`，避免尚未收齐一批就停止领取任务。

丢弃指标包括 `pool/discarded_groups_total`、`pool/skipped_groups_total`、
`pool/empty_groups_total`、`pool/tail_groups_total`，以及被丢弃 trajectory、
response token 和 rollout 耗时的总量。这些计数在当前进程内累计，随完成的
训练步数上报；即使没有完成训练步数或丢弃的是最终尾批，仍可查看逐组丢弃
日志和实例进度。

这里保证每个 trainer 轮次在 pool 准入时有 B 个有效组。后续 MIS 拒绝和 OPD
依赖 advantage 的过滤仍可能减少训练数据；`algorithm.optim.batch_size` 也可能
把同一组的 Datum 分到多次 optimizer 更新，因此不保证每次 optimizer 更新的组数。

**配置**：
```yaml
trajectory_pool:
  key_list: ["group_id"]
  group_size: 4  # Four independent rollouts, regardless of segment counts
  groups_per_batch: 32  # Deliver 32 completed groups together
  group_order: fifo  # 最早完成优先；lifo 为最近完成优先
```

**GRPO 的语义结构**：抽到一道任务后，为本次采样建立一个包含 R 条完整 trajectory
的比较组。再次抽到同一道任务时，即使 `task_id` 相同，也会使用新的 `group_id`，
不能仅按题目身份合并比较组。普通 trajectory 直接保存 token 字段，需要分段的结果
才保存内部序列字典。

```text
Group A：本次抽到 task A 后执行四次
├── trajectory A0：tokens、loss_mask、reward
├── trajectory A1：segments=[A1.0, A1.1]、reward
├── trajectory A2：tokens、loss_mask、reward
└── trajectory A3：segments=[A3.0, A3.1, A3.2]、reward
```

这个组有 **4 条完整 trajectory、7 条 token 序列**：两条普通序列加五个内部 segment。
GRPO 对四个最终 reward 求平均；序列数和 segment 数都不能定义组成员或判断组已完成。

**交付条件**：A0、A1、A2、A3 四个完整对象必须全部到齐，交付时保留原始对象
用于统计。一到两个成员失败时，trainer 在训练前各复制一个不同的有效成员补齐
失败槽位；三到四个成员失败时整组不可修复。成功零序列成员仍贡献 reward，也可被复制。

**取组顺序**：公开字段 `data.train.group_order` 可选 `fifo`（默认）或 `lifo`。
FIFO 优先取最早进入 ready 队列的组，LIFO 优先取最近进入的组；只有预期的全部
rollout 结果到齐后才入队，不按任务发起或首条结果到达时间排序。例如 A、B、C
依次完成，一次取两个组时，FIFO 返回 A、B，LIFO 返回 C、B。每组仍按 `run_id`
顺序返回全部完整轨迹，内部 segments 原样保留。

就绪组超过 B 时，这会改变批次成员，最常见于 `fully-async` 积压时。
所有模式每次最多取 B 组；`sync` 和 `batch-async` 通常每次放行一个 B 组输入批次。
GRPO trainer 后续可能重新排序，因此这不是 optimizer 执行顺序的设置。
完成顺序不代表采样发起时间或模型权重的新旧；
持续使用 LIFO 消费新完成组时，老组可能长期等待。详见[公开配置](../11-configuration-reference/data-config.md#group_order)。

**适用场景**：
- GRPO 算法（按本次采样的比较组身份分组）
- 每个查询生成多个样本的任务
- 需要比较同一上下文中的样本

<a id="batch-delivery"></a>

## 批次交付

### 分组 RL

三种权重同步模式都直接请求 `take_groups(num_groups=B)`，
B = `data.train.batch_size`。存储区返回 B 个完整组，或在不足时返回 `None`，
不会部分消费。B = 32、R = 4 时，常规批次交付 128 条完整轨迹及其
全部嵌套 segments；额外就绪组继续留在队列中。

该路径不等待 loader buffer 耗尽或 worker 静止。`sync` 和 `batch-async`
通常由调度器每次放行一个包含 B 组的输入批次；`fully-async` 的 B 个有效
完成组可以来自不同输入批次，被拒绝的组不计入 B。不足 B 组时，如果当前
loader buffer 已耗尽且 worker 静止，仍有缺成员的 pending 组会报错，
无需等整个数据源结束。

### Simple/SFT

pool 等当前 loader 整轮已全部领取且所有 rollout worker 静止后，若有结果，
调用 `take_trajectories(num_trajectories=ready_trajectory_count())`。
交付数量是实际整轮的 trajectory 数，SFT 的 token budget 可使每轮数量不同；
无需等待数据集或 epoch 结束。

### 有限源尾批

不足 B 组时，pool 先确认底层 loader 的 `is_finished()` 明确返回 `True`
且所有 rollout worker 静止。fully-async 分组 GRPO/OPD 随后丢弃并记录不足
B 个有效组的尾批，不训练也不推进 `train_step`；其他分组模式仍交付较小尾批。
仍有缺少成员的 pending 组会报错。公开的按 step 训练 RL recipe 使用循环
loader，会跨 epoch 边界填满 B 个输入，因此 epoch 结束不会终止收集或触发尾批处理。

<a id="weight-synchronization-coordination"></a>

## 权重同步协调

以下模式描述共享 pool 和调度器。具体 worker 可以有更严格的要求：
`E2BRolloutWorker` 接受 `sync` 和 `fully-async`。
`RemoteRolloutTrace` 要求至少一个 segment，已评测的 `RemoteRolloutResult` 则允许零 segment。
Worker 保留该结果的奖励，提交完整的零序列 trajectory 参与组内统计。

轨迹池在训练、权重同步和
验证期间均接受完整 trajectory。训练 trajectory 进入轨迹池，验证 trajectory 则
路由至验证器。权重同步不会拒绝已完成的结果。

<a id="sync-and-batch-async"></a>

### sync 与 batch-async

1. 数据加载器通过 `keep_batch_order: true` 提供一批输入。
2. 工作进程为每次 rollout 提交一个完整 Trajectory，失败和零段结果也保留为完整对象。
3. 分组 RL 在 `take_groups(num_groups=B)` 成功时立即取 B 个完整组。
   Simple/SFT 则等待当前 loader 整轮已领完且 worker 静止。
4. 训练器取走批次时，轨迹池通知权重同步控制器。
5. 在 batch-async 模式下，若陈旧程度低于或等于阈值，控制器会在
   训练开始前释放下一批数据。
6. 否则，下一批数据保持不可用，直到训练和权重
   同步完成，并在需要时完成验证。如果训练继续，
   控制器会释放下一批数据。

无需单独控制轨迹池的写入：当前批次已经
提交，而在数据加载器释放下一批次前，工作进程无法领取
它。如果跳过某一训练轮次且训练将继续，则会释放被保留的批次，
而不执行权重同步。

<a id="fully-async"></a>

### fully-async

数据加载器使用 `keep_batch_order: false`，工作进程领取当前缓冲区
的所有数据项后，便立即加载另一批输入。轨迹池调用
`take_groups(num_groups=B)` 每次消费 B 个有效完成组，同时其他 rollout 继续执行。
不可修复的组，以及有效成员没有训练目标的组，在统计就绪数量前排除，
并单独记录拒绝统计。
不同组可以乱序完成，但不会交付半组；每个组的完整 Trajectory 对象及嵌套 segments、最终 reward 一起交付。
数据加载自动继续，训练器取走批次时无需另行放行下一批数据。
权重同步不会停止完整 trajectory 的提交。

设置 `freeze_inflight: true` 时，如果控制器报告正在同步，
推理请求会在客户端等待后再开始采样。此检查不会
拒绝正提交到轨迹池的完整 trajectory。

<a id="usage-patterns"></a>

## 使用模式

<a id="basic-usage-in-trainer"></a>

### 在训练器中的基本用法

```python
class MyTrainer(BaseTrainer):
    def train_step(self):
        # Get batch
        batch = execute(self.trajectory_pool.get_trajectories)

        if batch is None:
            # Not enough samples yet
            return None

        # Fully-async grouped RL has already excluded ineligible groups.
        # Record original outcomes, then replace eligible failed slots before training.
        # Successful empty members remain valid, including as replacement donors.
        # If no trainable segments remain, skip the optimizer but advance the round.
        metrics = self._train_on_batch(batch)

        return metrics
```

<a id="grpo-grouping"></a>

### GRPO 分组

```yaml
# R distinct rollout completions per group; B complete groups per delivery
trajectory_pool:
  key_list: ["group_id"]  # GRPO comparison key
  group_size: 4
  groups_per_batch: 32  # All segments from 32 complete groups
```

```python
# Preserve the dataloader-assigned comparison group and attempt number
trajectory = {
    "tokens": ...,
    "group_id": task["group_id"],  # Shared by attempts in this comparison group
    "run_id": task["run_id"],  # Attempt slot within this group
    "rollout_id": task["rollout_id"],  # Unique execution identity
    "expected_rollouts": task["expected_rollouts"],
    # ...
}
```

<a id="multiple-models"></a>

### 多模型

```python
# Separate stores for policy and reference models
trajectory_policy = {
    "tokens": ...,
    "model_tag": "policy",
}

trajectory_reference = {
    "tokens": ...,
    "model_tag": "reference",
}

# Get batches per model
policy_batch = pool.get_trajectories(model_tag="policy")
reference_batch = pool.get_trajectories(model_tag="reference")
```

<a id="configuration-examples"></a>

## 配置示例

<a id="simple-no-grouping"></a>

### 简单模式（不分组）

```yaml
trajectory_pool:
  type: "default"
  key_list: []  # Deliver the completed loader round
```

<a id="grpo-single-level-grouping"></a>

### GRPO（单层分组）

```yaml
trajectory_pool:
  type: "default"
  groups_per_batch: 32
  group_size: 4  # Four distinct terminal rollout records per group
  key_list: ["group_id"]  # Group by comparison group
```

<a id="best-practices"></a>

## 最佳实践

<a id="grouping"></a>

### 分组

1. **为 GRPO 使用分组** — 正确计算优势值所必需
2. **按完整 trajectory 计数** — `group_size` 是预期 rollout 数 R，每个唯一终态 rollout 只贡献一个最终 reward，增加 segment 不会改变 R
3. **保留本次采样身份** — 尝试共享本次新生成的 group_id，run_id 区分槽位，rollout_id 唯一标识执行；跨 epoch 再采样不复用该 group
4. **测试分组逻辑** — 验证轨迹分组正确

<a id="batch-sizes"></a>

### 批次大小

1. **平衡内存与吞吐量** — 更大的批次意味着更高的 GPU 利用率，但需要更多内存
2. **配置完成组数** — `groups_per_batch` 选取 B 个完整组，再交付它们的全部 segments；后续 optimizer batch/mini-batch 拆分保持不变
3. **明确交付单位** — 分组 RL 取 B 个完整组；Simple/SFT 取实际完成的 loader 整轮
4. **监控批次耗时** — 如果训练器等待批次过久，应进行调整

<a id="performance"></a>

### 性能

1. **减少锁竞争** — 轨迹存储区采用细粒度锁
2. **在工作进程中预先分组** — 提交前保留 group_id 和 run_id
3. **监控轨迹池大小** — 避免轨迹池无限增长
4. **使用合适的存储区类型** — 无需分组时，SimpleTrajectoryStore 最快

<a id="debugging"></a>

### 调试

1. **记录 segment 数量** — 监控 segment 流入和流出
2. **检查分组键** — 验证轨迹包含必需的键
3. **监控批次就绪状态** — 检查是否按预期形成批次
4. **检查轨迹** — 查看前几条轨迹，验证结构

<a id="troubleshooting"></a>

## 故障排查

<a id="issue-trainer-not-receiving-batches"></a>

### 问题："Trainer not receiving batches"

**症状**：训练器空闲，轨迹不断积累

**解决方法**：
1. 检查公开字段 `data.train.batch_size` 是否符合预期的完整组数
2. 检查所有预期 rollout 是否提交完整 Trajectory，以及 `groups_per_batch` 是否为期望的完成组数
3. 检查轨迹中是否存在 `key_list` 指定的键
4. 监控轨迹池中的就绪 segment 数量

<a id="issue-incomplete-groups"></a>

### 问题："Incomplete groups"

**症状**：部分分组一直无法完成，轨迹滞留

**解决方法**：
1. 验证所有轨迹都有分组键
2. 检查分组键名称是否存在拼写错误
3. 检查 R 个槽位是否都有唯一 rollout 身份和完整 Trajectory，失败或零段执行也不能遗漏
4. 不能强制释放缺少完整 trajectory 的组；不足 B 组且当前 buffer 耗尽、worker 静止后，仍有未完成组会报错

<a id="issue-weight-sync-blocking-too-long"></a>

### 问题："Weight sync blocking too long"

**症状**：工作进程等待下一批输入，或等待推理请求继续执行

**解决方法**：
1. 降低权重同步频率
2. 使用异步权重同步模式
3. 增大陈旧程度阈值
4. 优化同步操作速度

<a id="next-steps"></a>

## 后续步骤

- 了解[权重同步](../08-features/weight-synchronization.md)
