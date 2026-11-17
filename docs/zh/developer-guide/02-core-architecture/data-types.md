[English](../../../en/developer-guide/02-core-architecture/data-types.md) | 简体中文 | [文档目录](../../index.md)

<a id="core-data-types"></a>

# 核心数据类型

**术语约定：** 一次 rollout 对应一条完整 `Trajectory`。普通轨迹直接保存 tokens 和 loss mask，只有多段轨迹使用嵌套的 `segments` 字典列表。task、执行请求、group 和训练记录的区别见[轨迹术语说明](../04-trajectory-pool/trajectory-semantics.md)。



本文档介绍 NorthRL 框架中使用的基础数据类型。

<a id="overview"></a>

## 概览

NorthRL 在 `northrl/northrl_types.py` 中定义了若干核心数据类型，构成框架数据流和组件交互的基础。

<a id="type-aliases"></a>

## 类型别名

<a id="modeltag"></a>

### ModelTag

```python
ModelTag = str  # Type alias for model identification
```

**用途**：标识系统中的不同模型

**用法**：
```python
model_tag = "actor"  # Current rollout and training model
```

**常用值**：
- `"actor"`：当前 rollout、pool 和 trainer 路径使用的模型标签
- `"default"`：未提供标签时部分通用 Batch/pool 工具使用的缺省值

请求、trajectory 与目标 pool 的标签必须一致。KL reference client 的存在
不会自动创建一个名为 `"reference"` 的 trajectory-pool 标签。

<a id="data-classes"></a>

## 数据类

<a id="trajectory"></a>

### Trajectory

```python
@dataclass(init=False)
class Trajectory:
    tokens: list[int] | None
    loss_mask: list[int] | None
    reward: float | None
    is_val: bool
    extra_fields: dict[str, Any]
    segments: list[dict[str, Any]] | None
    status: str  # success, skipped, or failed
```

`Trajectory` 保存一次完整执行及最终结果。普通形式直接保存 `tokens` 和 `loss_mask`，
`segments=None`，不会创建任何嵌套包装对象。`old_log_probs`、sampling mask、
correction weight 等 token 元数据仍放在 `extra_fields`，保持原单轮接口。

需要分段的执行使用 `segments`：其中每项是带 `tokens`、`loss_mask` 和序列元数据的
扁平字典，完整轨迹对象的 `tokens`、`loss_mask` 均为 `None`。构造时同时提供两种形式会抛出
`ValueError`。reward、终态、验证标记、执行身份、score 和 advantage 均归完整轨迹对象。
已评分但没有训练序列的结果使用 `Trajectory(segments=[], reward=final_reward)`。

类字典接口支持 `obj[key]`、赋值、`get`、`keys`、`items` 和成员检查。
`group_id` 标识本次采样的比较组，`run_id` 是预期尝试槽位，`rollout_id` 标识一次执行。
`task_id` 标识原始任务：再次抽到同一道任务时，`task_id` 可以相同，但会生成新的
`group_id`。`rollout_repeat_n=R` 表示同一组内执行 R 次，不是创建 R 个组。

多次模型调用不一定需要多段：recorder 可把 token 前缀连续的调用接到同一序列。
E2B 总是返回 segment 列表，也可以只有一项。远端 `RemoteSegment` 是序列化协议类型；
本地内部记录仍是完整 `Trajectory` 所拥有的字典。

```python
from northrl.northrl_types import Trajectory

# Ordinary trajectory: these are its actual fields, with no nested object.
ordinary = Trajectory(
    tokens=[1, 2, 3],
    loss_mask=[0, 1, 1],
    reward=1.0,
    extra_fields={"old_log_probs": [0.0, -0.2, -0.3]},
)
assert ordinary.segments is None
assert ordinary.__dict__["tokens"] == [1, 2, 3]

# Multi-sequence trajectory: each nested record is a flat dictionary.
multi = Trajectory(
    segments=[
        {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1],
         "old_log_probs": [0.0, -0.2, -0.3]},
        {"tokens": [4, 5], "loss_mask": [0, 1],
         "old_log_probs": [0.0, -0.4]},
    ],
    reward=0.5,
    extra_fields={"group_id": "g", "rollout_id": "r0", "run_id": 0,
                  "expected_rollouts": 2},
)
assert multi.tokens is None and multi.loss_mask is None
multi["advantage"] = 0.5  # Computed once from the complete trajectory's reward.
```

普通构造将 tokens 和 mask 保存为直接字段，恢复后仍保持这一结构。
不提供新的公共序列类型或展平 API。内部 `iter_trajectory_sequences` 原样查看序列记录，
不修改 metadata。训练边界的 `iter_training_sequences` 过滤失败/被跳过的组，普通形式
直接返回原 Trajectory，多段形式返回原字典。完整轨迹身份和 advantage 覆盖子记录中的旧值；
其他 metadata 只补缺省，保留每段自己的 logprobs 和 mask，不下沉 reward 或终态。

<a id="batch"></a>

### Batch

```python
@dataclass
class Batch:
    values: dict[str, Any]      # Tensor or data arrays
    metadata: dict[str, Any]    # Batch metadata
```

**用途**：保存按行对齐的字段和批次元数据的通用容器，不将任意一行固定解释为 segment、trajectory 或 Datum

**结构**：
- `values`：将字段名称映射到批数据的字典
  - 张量：以第一维作为行维度的字段
  - 列表：非张量数据（字符串、字典等）
- `metadata`：批次级信息
  - `batch_size`：当前行数，包含 padding 补出的行
  - `model_tag`：此批次的模型标签
  - `temperature`：采样温度（如果一致）

**关键方法**：

<a id="__len__---int"></a>

#### `__len__() -> int`

返回批次大小：

```python
batch_size = len(batch)
```

<a id="copy---batch"></a>

#### `copy() -> Batch`

对 `values` 和 `metadata` 字典创建浅拷贝，内部值仍共享。当前方法不保留 `pad_size`，因此带 padding 的 batch 应先 unpad 再复制：

```python
batch_copy = batch.copy()
```

<a id="to_dict---dictstr-any"></a>

#### `to_dict() -> dict[str, Any]`

合并 values 和 metadata（metadata 会覆盖 values）：

```python
combined = batch.to_dict()
# Access both values and metadata in one dict
```

<a id="from_trajectoriestrajectories-model_tagnone---batch"></a>

#### `from_trajectories(trajectories, model_tag=None) -> Batch`

每行保存一条完整 Trajectory，保留普通直接字段或多段嵌套字典。
子序列数量不改变轨迹行数。各行元数据键相同时，保持历史平铺字段列；键不同时，
使用固定轨迹字段列加每行独立的 `extra_fields` 字典列。因此普通轨迹对象保存 logprobs、
多段仅在子字典中保存 logprobs 的结果也能同批，不会给其他行补出缺失元数据键。

<a id="to_trajectories---listtrajectory"></a>

#### `to_trajectories() -> list[Trajectory]`

每行还原一条完整 Trajectory，包括 reward、status 和嵌套序列。
还原前必须调用 `unpad()`，不能将 padding 行伪造成执行结果。
缺少必要 trajectory 字段的通用 Batch 不能使用这个转换接口。

当前远程 trainer 不创建 Batch：它接收完整 trajectories，准备其中的序列记录，
然后转成 Datums。Batch 转换方法是独立工具，不是这条训练主路径的额外阶段。

<a id="pad_to_world_sizeworld_size-int---batch"></a>

#### `pad_to_world_size(world_size: int) -> Batch`

填充批次，使其大小可被 world_size 整除（用于分布式训练）：

```python
batch = batch.pad_to_world_size(world_size=4)
# Batch size becomes multiple of 4
```

<a id="remove_redundant_left_paddingdata-pad_token_id-"></a>

#### `remove_redundant_left_padding(data, pad_token_id, ...)`

用于移除公共左侧填充的静态方法：

```python
batch = Batch.remove_redundant_left_padding(
    data=batch,
    pad_token_id=0,
    fields=["input_ids", "attention_mask"],
    anchor_field="input_ids",
    max_strip_threshold=512
)
```

**参数**：
- `pad_token_id`：填充 token 的 ID
- `fields`：要裁剪的字段（None 表示所有二维张量）
- `anchor_field`：用于确定填充长度的字段
- `max_strip_threshold`：最多裁剪的 token 数（-1 表示不限制）

<a id="remove_redundant_right_paddingdata-pad_token_id-"></a>

#### `remove_redundant_right_padding(data, pad_token_id, ...)`

用于移除公共右侧填充的静态方法：

```python
batch = Batch.remove_redundant_right_padding(
    data=batch,
    pad_token_id=0,
    fields=["input_ids", "attention_mask"],
    anchor_field="input_ids"
)
```

<a id="to_nextrainer_batch---dictstr-any"></a>

#### `to_nextrainer_batch() -> dict[str, Any]`

将批次转换为 NexTrainer 格式：

```python
nextrainer_batch = batch.to_nextrainer_batch()
# Returns: {
#     "batch": {tensor_fields...},
#     "non_tensor_batch": {non_tensor_fields...},
#     "meta_info": {...}
# }
```

**使用示例**：

```python
from northrl.northrl_types import Batch, Trajectory

trajectories = [
    Trajectory(tokens=[1, 2], loss_mask=[0, 1], reward=1.0),
    Trajectory(segments=[
        {"tokens": [3, 4], "loss_mask": [0, 1]},
        {"tokens": [5, 6], "loss_mask": [0, 1]},
    ], reward=0.5),
]
batch = Batch.from_trajectories(trajectories)
assert len(batch) == 2  # Two complete trajectories, not three sequence records.
batch.pad_to_world_size(4)
assert len(batch) == 4  # Includes two padding rows.
batch.unpad()
restored = batch.to_trajectories()
assert len(restored) == 2 and len(restored[1].segments) == 2
```

<a id="northrlrole"></a>

### NorthRLRole

```python
class NorthRLRole(Enum):
    ROLLOUT_WORKER = "rollout_worker"
    TRAINER = "trainer"
    TRAJECTORY_POOL = "trajectory_pool"
    WEIGHT_SYNC_CONTROLLER = "weight_sync_controller"
    DATA_LOADER = "data_loader"
    VALIDATE_DATALOADER = "validate_dataloader"
    VALIDATOR = "validator"
```

**用途**：标识本地组件，供控制器选择实现、初始化实例，并生成活动跟踪使用的模块名。

**用法**：

```python
from northrl.northrl_types import NorthRLRole

role = NorthRLRole.ROLLOUT_WORKER
module_name = f"{role.value}-0"  # rollout_worker-0
```

控制器在本地驱动进程中创建模块实例，通过直接引用连接各模块。
详见[模块初始化](controller.md#module-initialization)。

<a id="evaluation-system-types"></a>

## 评估系统类型

单轮奖励函数接收 `(task, result)` 字典。`northrl.rewards.evaluate_reward`
将返回值归一化为包含有限数值 `reward` 的字典；可选字段包括布尔值 `correct`、
标量指标 `metrics` 和文本 `reason`。详见[评估器](../05-rollout-workers/evaluators.md)。

<a id="remote-agent-and-evaluation-results"></a>

### 远程智能体与评估结果

E2B 任务钩子使用 `northrl.remote_rollout.contracts` 中的接口：

- `RolloutInput`：包含 `rollout_id`、通用 JSON `task` 和来源信息 `metadata`。
- `AgentResult`：包含执行 `exit_code`、可选 `final_answer`，以及从逻辑名称到路径的
  `artifacts` 映射。训练 token 由运行时单独记录。
- `EvaluationResult`：包含最终 `reward`、标量 `metrics` 和评估 `status`。

```python
from northrl.remote_rollout.contracts import AgentResult, EvaluationResult

agent_result = AgentResult(final_answer="42")
result = EvaluationResult(
    reward=1.0,
    metrics={"exact_match": 1.0, "answer_length": 2.0},
)
```

`reward` 可以是任意有限数值。`status="success"` 表示评估完成，不代表答案一定正确；
`status="task_failed"` 要求奖励为零。基础设施故障应抛出 `RolloutEnvironmentError`，
而不是返回虚构的任务分数。`run_agent`、`evaluate` 的接口及分阶段生命周期见
[E2B 任务接入](../05-rollout-workers/e2b-task-contract.md)。

<a id="type-conversions"></a>

## 类型转换

<a id="trajectory--batch"></a>

### Trajectory ↔ Batch

```python
# One complete trajectory per row, preserving nested sequence boundaries.
batch = Batch.from_trajectories([trajectory1, trajectory2])
trajectories = batch.to_trajectories()
```

<a id="batch--dictionary"></a>

### Batch ↔ 字典

```python
# Batch to dict
combined_dict = batch.to_dict()

# Dict to Batch (manual)
batch = Batch(
    values={"tokens": [...], "loss_mask": [...]},
    metadata={"batch_size": 32}
)
```

<a id="batch--nextrainer-format"></a>

### Batch → NexTrainer 格式

```python
nextrainer_batch = batch.to_nextrainer_batch()

# Result structure:
# {
#     "batch": {tensor_fields},
#     "non_tensor_batch": {non_tensor_fields},
#     "meta_info": {metadata}
# }
```

<a id="common-patterns"></a>

## 常见模式

<a id="creating-a-trajectory-in-rollout-worker"></a>

### 在 Rollout 工作进程中创建完整 Trajectory

```python
async def rollout(self, task: dict[str, Any]) -> str | None:
    # Run agent/LLM
    response = self.agent.run(task["query"])

    # Evaluate
    eval_result = self.evaluator.evaluate(task, response)

    # Ordinary construction stores tokens and loss_mask directly.
    trajectory = Trajectory(
        tokens=generated_tokens,
        loss_mask=compute_loss_mask(generated_tokens),
        reward=eval_result.reward,
        is_val=False,
        extra_fields={
            "model_tag": self._model_tag,
            "query": task["query"],
            "final_answer": response,
            "ground_truth": eval_result.ground_truth,
            **eval_result.metrics,
            **eval_result.extra_info,
        }
    )

    return self._put_trajectory(trajectory)
```

<a id="processing-a-batch-in-trainer"></a>

### 在训练器中处理批次

这个工具示例移除公共 token padding 并记录当前行数，不代表远程 trainer 的执行路径。GRPO 必须先使用完整 trajectory 的身份及最终 reward，再准备序列数据。

```python
def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
    metrics = {}

    # Remove padding
    batch = Batch.remove_redundant_left_padding(
        batch, pad_token_id=self._pad_token_id
    )

    # Report rows; their semantic unit depends on the constructor.
    metrics["batch/row_count"] = len(batch)

    return batch, metrics
```

<a id="grouping-trajectories"></a>

### 对完整 Trajectory 分组

```python
# Index complete rollout results by their comparison group.
grouped_trajectories = {}
for trajectory in trajectories:
    group_id = trajectory["group_id"]
    grouped_trajectories.setdefault(group_id, []).append(trajectory)
```

这里索引的是完整 rollout 对象，包括零段结果。pool 等待预期槽位中的不同 rollout 身份全部到齐后才交付组。GRPO 对每条完整轨迹的 reward 只计一次，segment 数不影响 baseline。

<a id="best-practices"></a>

## 最佳实践

<a id="for-trajectories"></a>

### Trajectory 与序列记录的最佳实践

1. 多模型场景下**始终包含 model_tag**
2. 在任务中**使用一致的 extra_fields**
3. 可用时**包含 ground_truth**
4. **将指标保存为浮点数**以便聚合
5. **将 group_id 和 rollout_id 保存在完整 trajectory 上**；普通 token 元数据放在 extra_fields，多段 token 元数据分别放在各嵌套字典中

<a id="for-batches"></a>

### 批次的最佳实践

1. **验证元数据中的 batch_size** 与实际大小一致
2. **移除填充**以优化内存和计算
3. 操作前**检查张量形状**
4. **先 unpad 再还原 trajectories**，嵌套序列仍属于同一轨迹行
5. 谨慎**使用 to_dict()**（会丢失结构）

<a id="for-type-safety"></a>

### 类型安全

1. 在代码中为这些类型**使用类型提示**
2. 从 extra_fields 访问数据前**验证字段**
3. 使用 `.get()` 妥善**处理缺失字段**
4. 在实现中**记录自定义 extra_fields**
5. **测试类型转换**（尤其是 Trajectory ↔ Batch）

<a id="next-steps"></a>

## 后续步骤

- 了解[控制器](controller.md)如何管理这些类型
- 了解 [Rollout 工作进程](../05-rollout-workers/overview.md)如何生成轨迹
- 查看[轨迹池](../04-trajectory-pool/trajectory-pool.md)的批处理策略
