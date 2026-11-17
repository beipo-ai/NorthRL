[English](../../../en/developer-guide/05-rollout-workers/overview.md) | 简体中文 | [文档目录](../../index.md)

<a id="rollout-workers-overview"></a>

# Rollout 工作进程概览

**术语约定：** 一次 rollout 产出一个完整 `Trajectory`。普通轨迹直接保存 `tokens` 和 `loss_mask`（`segments=None`）；需要保留独立训练序列时才使用内部 `segments` 字典列表。多轮交互不一定需要分段。worker、pool 和 trainer 准备接口传递完整 trajectory。详见[轨迹术语说明](../04-trajectory-pool/trajectory-semantics.md)。

Rollout 工作进程负责执行任务并与 LLM 服务交互，从而生成轨迹。它们构成向训练过程提供数据的数据生成流水线。

<a id="architecture"></a>

## 架构

```
DataLoader → RolloutWorker → TrajectoryPool
                ↓
        Reward calculation
```

Rollout 工作进程会：
1. 从数据加载器获取任务
2. 执行任务（与 LLM 或智能体交互）
3. 评估结果
4. 准备直接 token 字段；需要分段时才构造内部记录
5. 构造包含最终评估 reward 的一个完整 `Trajectory`
6. 只提交一次该对象；训练 wrapper 待 `rollout` 返回后才发布

<a id="worker-hierarchy"></a>

## 工作进程继承层级

```
NorthRLModule
    ↓
BaseRolloutWorker (abstract)
    ↓
    ├── SimpleRolloutWorker (basic LLM completion)
    ├── SingleTurnRolloutWorker (single response and reward)
    ├── SFTRolloutWorker (pass-through for SFT)
    ├── E2BRolloutWorker (remote agent execution; sync / fully-async)
    └── Custom Workers (task-specific, derived from BaseRolloutWorker)
```

<a id="when-to-use-each-worker"></a>

## 各工作进程的适用场景

<a id="simplerolloutworker"></a>

### SimpleRolloutWorker
- **使用场景**：基本的单轮 LLM 补全任务
- **特性**：提示 → LLM → 响应
- **示例**：问答、文本生成

<a id="singleturnrolloutworker"></a>

### SingleTurnRolloutWorker
- **使用场景**：单响应训练任务
- **特性**：生成响应、选择奖励并构建完整 trajectory
- **示例**：数学推理、指令遵循

<a id="sftrolloutworker"></a>

### SFTRolloutWorker
- **使用场景**：使用已分词数据或原始聊天消息进行监督微调
- **特性**：无 LLM 推理；`SFTStreamingDataLoader` 流式读取 JSONL，
  渲染消息，并验证按 token 对齐的损失掩码
- **打包**：NorthRL 将每行保留为独立 Datum；Weaver 执行
  实际的序列打包
- **数据流**：`messages` 或 `{input_ids, loss_mask}` → `{input_ids, loss_mask}` → `Trajectory(tokens=..., loss_mask=..., reward=...)`
- **选择方式**：设置 `algorithm.type: sft`，并省略整个顶层
  `rollout` 配置块。控制器只创建一个内部工作进程；
  如果显式提供该块，则发出 `UserWarning` 并忽略整个块
- **用法**：参见 [SFT 数据](../03-data-loader/sft-data.md)

<a id="e2brolloutworker"></a>

### E2BRolloutWorker

- **用例**：在 E2B sandbox 中执行智能体
- **边界**：当前要求 Weaver 后端和 `weight_sync.mode: sync`
- **任务输入**：每个 recipe 必须将 `rollout.execution.input_fields` 配置为不重复的非空字段名组成的列表；`[]` 表示不上传题目字段。worker 仅将 dataloader 行中存在且被选中的字段放入 `RolloutInput.task`。任务函数校验必需字段并构建 prompt，worker 不要求固定的任务字段名。例如，问答任务可选择 `[question, context]`。`template` 用于选择 sandbox，不会自动复制到 `task`。
- **SWE prompt**：SWE recipe 选择 `[instruction, interface]`。其任务函数要求非空 `instruction`，并用一个空行连接可选的非空 `interface`，保留两段原文且不添加标题。该任务的 Claude prompt 只使用这两个字段。
- **评测与来源信息**：`RolloutInput.metadata` 保存数据中存在的 `task_id`、`group_id`、`run_id`、`expected_rollouts`，以及 `weight_version`。评测使用 `metadata.task_id`，因此无需在 `input_fields` 中选择 `task_id`。metadata 不会自动加入 agent prompt。
- **输入隔离**：未被选中的数据字段不进入 `task`。recipe 应只选择解题输入，将隐藏答案和测试放进评测阶段才上传的材料中。SWE 不选择 `patch`、`test_patch`。runtime 核验请求输入文件属主并设置 `0600` 权限。任务函数以无法读取该文件的受限用户运行 agent。
- **结果**：一个完整 `Trajectory`，内部保存序列字典，完整轨迹对象保存一个 reward；远端协议要求至少一个 segment
- **Trace 和 Result**：`RemoteRolloutTrace` 是执行过程快照；`RemoteRolloutResult` 是交付结果，超时时可使用最后一份有效 partial trace，reward 为 0。两者字段均为 `segments`。
- **周期 trace 同步**：启用 trace 输出后，共享进程池最多启动两个 spawn 子进程，对 gzip 快照完整校验并原子保存。解析出的 token 对象留在子进程，不争用训练进程的 GIL。成功保存后，相同压缩内容跳过重复校验和写入。无效快照不会覆盖最后一份有效文件。后台仍最多准入四个处理回调；取消时等待写入结束后才允许重试，最后一个同步任务退出时回收进程池。最终结果可通过 `rollout.execution.result_processing: process` 启用独立进程；部分训练 trace 的恢复沿用现有解码路径。

<a id="custom-workers"></a>

### 自定义工作进程
- **使用场景**：任务专属行为
- **特性**：自定义查询格式、专门的评估、领域逻辑
- **示例**：任务专属提示、多步骤模型调用

<a id="core-workflow"></a>

## 核心工作流程

<a id="1-main-loop"></a>

### 1. 主循环

```python
while not self._stop_event.is_set():
    with self._activity_tracker.track(self._module_name, "rollout"):
        self._next_task = self._get_rollout_task()
        while self._next_task is not None and not self._stop_event.is_set():
            await self.execute_rollout(copy.deepcopy(self._next_task))
            if not self._stop_event.is_set():
                self._next_task = self._get_rollout_task()

    if self._next_task is None and not self._stop_event.is_set():
        await asyncio.sleep(0.1)
```

<a id="2-rollout-execution"></a>

### 2. Rollout 执行

主循环调用 `execute_rollout(task)`。自定义工作进程仍实现
`async rollout(task)`，并通过 `self._put_trajectory(trajectory)` 提交完整对象：

```python
async def rollout(self, task: dict[str, Any]) -> str | None:
    """
    Execute single rollout operation.

    Returns:
        - "success": Complete trajectory accepted for submission
        - "fail": Failed to submit
        - None: No explicit status; absent trajectory submission is marked skipped
    """
```

训练时，`execute_rollout` 负责整个请求的生命周期。自定义 `rollout` 构造一个完整
`Trajectory`：普通结果直接保存 token 字段，需要独立序列时使用内部字典，并在完整轨迹对象
保存最终 reward。然后只调用一次 `_put_trajectory`。
wrapper 填写并校验执行身份，先在 worker 本地暂存对象，待 `rollout` 返回后才一次性
写入 pool。同次执行第二次提交会报错；此接口不逐段流式提交，也没有单独的 finish 通知。

已评分但没有可训练段时，显式提交
`Trajectory(segments=[], reward=final_reward, extra_fields={"score": score})`。
正常返回但未提交对象会生成空的 skipped trajectory。返回 `"fail"` 会将已暂存对象
标为 failed；若未提交对象，则生成空的 failed 结果。从 `rollout` 逃出的异常会使
wrapper 用空的 failed trajectory 替换暂存结果，提交后继续抛出，不会自动重试。

验证任务（`task["is_val"]`）直接调用 `rollout`，`is_val=True` 的对象立即进入 validator。
训练 wrapper 的暂存、单次提交检查和自动 skipped/failed 结果均不适用于验证路径。

<a id="3-trajectory-creation"></a>

### 3. 创建完整 Trajectory

普通工作进程直接保存训练序列：

```python
trajectory = Trajectory(
    tokens=tokens,
    loss_mask=loss_mask,
    reward=final_reward,
    is_val=task.get("is_val", False),
    extra_fields={"old_log_probs": old_log_probs, "score": score},
)
assert trajectory.segments is None
```

多轮执行如果只保留一条有效 token 序列，也可以使用这种形式。需要保留独立序列时，
才将它们作为内部字典放进一个完整结果；不能为了避免分段而拼接无关的 prompt：

```python
Trajectory(
    segments=[
        {
            "tokens": [...],
            "loss_mask": [...],
            "old_log_probs": [...],
        },
        # Additional segments from the same execution, if any.
    ],
    reward=final_reward,
    is_val=task.get("is_val", False),
    extra_fields={
        "ground_truth": ground_truth,
        "score": score,
        # Execution identity is supplied/validated by the wrapper.
    },
)
```

<a id="4-per-turn-tool-diagnostics"></a>

### 4. 逐轮工具诊断

E2B NorthGate 路径为每次记录的模型响应保留服务端返回的
`tool_call_diagnostics`。无需新增 recipe 配置或诊断开关；NorthGate 返回诊断时，
本地才有对应证据。relay 在流式和非流式响应中保留该字段，recorder 再将它附到
拥有本轮响应的训练序列：

```text
NorthGate response → TrajectoryRecorder → RemoteSegment / compact trace
                  → E2BRolloutWorker → Trajectory → prepared dump / replay
```

每条序列使用 `turn_annotations` 列表保存记录，单条记录示例：

```json
{
  "version": 1,
  "turn_id": "turn-0",
  "response_span": [2, 4],
  "diagnostics": {
    "checked": true,
    "malformed": true,
    "detected": 1,
    "invalid": 1,
    "reasons": ["invalid_arguments_json"]
  },
  "finish_reason": "tool_calls",
  "truncated": false,
  "tools_declared": true,
  "source": "northgate"
}
```

`diagnostics` 原样保留 JSON 对象，包括未知字段和 reason 值；传输层不会从工具执行
结果或聚合指标反推诊断。服务端未返回该字段或返回值不是对象时，保存为 `None`
（JSON 中为 `null`）。`turn_annotations: []` 表示没有逐轮 annotation；
`diagnostics: null` 表示这一轮没有诊断证据。两者都不代表响应已经检查且没有错误。
`tools_declared` 表示请求的 `tools` 值不是 `None`；显式空列表也算已知声明。
`finish_reason` 保存 OpenAI 原始 `finish_reason` 或 Anthropic 原始 `stop_reason`，
无法取得时为 `None`。

`response_span` 使用从 0 开始的半开区间 `[start, end)`，坐标指向**完整训练序列的
token 数组**，包含 prompt 及中间的工具结果、上下文 token。它不是仅保留 response
token 后的压缩下标，也不是移位后的 `tokens[1:]` target 下标。例如 tokens 为
`[10, 11, 20, 21, 30, 40, 41]`、mask 为 `[0, 0, 1, 1, 0, 1, 1]` 时，
两轮响应区间可以是 `[2, 4]` 和 `[5, 7]`。远端校验要求
`1 <= start < end <= len(tokens)`、区间内 mask 全为 1、区间有序且不重叠，
同一 segment 内的 turn ID 非空且不重复。即使 compact 编码合并相邻的 response
mask 区间，相邻轮次仍保留各自独立的 annotation。

直接保存 token 的 trajectory（`segments=None`）将 annotation 放在
`extra_fields["turn_annotations"]`；分段 trajectory 则放在每个 child 字典，
与该 child 的 `tokens`、`loss_mask` 同级：

```python
# annotation is the record above; both examples own tokens at positions [2, 4).
sequence = {"tokens": [10, 11, 20, 21], "loss_mask": [0, 0, 1, 1]}
direct = Trajectory(
    **sequence, reward=1.0, extra_fields={"turn_annotations": [annotation]}
)
segmented = Trajectory(
    segments=[{**sequence, "turn_annotations": [annotation]}], reward=1.0
)
```

不要将某个 segment 的 annotation 放在分段完整轨迹对象的 `extra_fields`：各 child
拥有独立的坐标和响应归属。尚未提供此证据的 worker 可以省略该字段；远端 segment
协议默认使用 `[]`。

recorder 在一次 rollout 内依次分配 `turn-0`、`turn-1` 等 ID，每成功记录一轮后
递增。若新 prompt 精确续接最新 segment 的 tokens，只追加新的上下文后缀和响应，
保留已有 annotation 并新增一条。否则开启新 segment，其中只记录新响应的 annotation。
新 prompt 中重复出现的历史响应 token 是 mask 为 0 的上下文；对应 annotation
仍属于最初生成它们的 segment。

`truncated` 表示响应不完整或无法确认完整性。服务端以 `length` / `max_tokens`
结束，或请求使用自定义 stop 设置时，该标记为 true。Driver 按配置的最大长度裁剪
segment 时，会丢弃起点已超出保留 tokens 的 annotation；对跨越裁剪点的区间缩短
终点，并设置 `truncated=true`。此前完整保留的轮次维持原标记。即使 worker 将
segment 级别的 `finish_reason` 设为 `length` 或 `timeout`，annotation 内服务端
原始的 `finish_reason` 仍然保留。

Prepared trajectory dump 在 `.pt` 和 `.jsonl` 中都保留上述直接序列或 child 位置
的 annotation；`MockRolloutWorker` 文件 replay 恢复这些记录，不补造缺失证据。
是否写文件仍由已有 dump 设置控制，采集 annotation 本身不要求开启 dump。
参见[数据 dump 用法](../../debug_data_dump_usage.md)。

compact wire envelope 仍使用 `schema_version: 2`，annotation 的 `version: 1`
是独立字段。新 reader 可读取缺少 `turn_annotations` 的旧 v2 segment，并使用
默认值 `[]`。旧 reader 会因 schema 禁止未声明字段而拒绝新增字段，因此 Driver
和 sandbox 代码必须版本匹配。现有 executor 会为 sandbox 打包当前源码。
wire 边界详见 [E2B 任务集成](e2b-runtime.md#trace-schema)。

<a id="key-concepts"></a>

## 关键概念

<a id="activity-tracking"></a>

### 活动跟踪

工作进程跟踪自身活动以便监控：

```python
with self._activity_tracker.track(self._module_name, "rollout"):
    result = await self.execute_rollout(task)
```

这支持：
- 系统健康监控
- 静止状态检测（所有工作进程空闲）
- 性能分析

<a id="weight-synchronization-coordination"></a>

### 权重同步协调

轨迹池在整个权重同步期间都接受已完成的结果。
它不会因为正在同步而要求重新执行 rollout。

**同步模式：**
- **sync**：下一批输入等待训练、同步，以及需要时的验证
  完成。
- **fully-async**：输入加载自动继续。训练与 rollout 重叠执行；
  每轮更新后的训练结果都会在下一轮训练前保存并同步。
  设置 `freeze_inflight: true` 时，新推理请求在同步期间等待。
- **batch-async**：超过陈旧程度阈值时，下一批输入
  等待；否则，在当前批次训练前释放。

`execute_rollout()` 发布完整 trajectory 并返回后，工作进程会领取下一个任务，除非已收到
停止请求。任务专属的重试应在 `rollout()` 内部实现。

<a id="validation-mode"></a>

### 验证模式

工作进程同时支持训练和验证：

```python
def begin_validate(self):
    """Switch to validation mode"""
    self._is_running_validate = True

def end_validate(self):
    """Switch back to training mode"""
    self._is_running_validate = False
```

处于验证模式时：
- 任务来自验证数据加载器
- 轨迹提交至验证器（而非轨迹池）
- 收集完成后计算验证指标

<a id="inference-service-integration"></a>

### 推理服务集成

工作进程通过推理客户端访问 LLM 服务：

```python
# Set during initialization
self._inference_client = create_inference_service_client(
    backend=config.backend.type,
    config=config
)

# Use in rollout
result = await self._inference_client.completion(
    prompt=prompt  # Sampling settings come from configuration.
)
```

<a id="module-references"></a>

## 模块引用

工作进程需要引用其他组件：

```python
def set_module_references(
    self,
    trajectory_pool: TrajectoryPool,
    dataloader: BaseDataLoader,
    weight_sync_controller: WeightSyncController,
    validate_dataloader: BaseDataLoader,
    validator: Validator,
):
    """Set module references before run()"""
```

这些引用由控制器在初始化期间设置。

<a id="configuration"></a>

## 配置

基本工作进程配置：


<a id="common-patterns"></a>

## 常见模式

<a id="basic-completion-pattern"></a>

### 基本补全模式

下面的普通构造直接保存 token 字段，`segments` 保持 `None`。

```python
class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        # Get prompt
        prompt = task["prompt"]

        # Call LLM
        result = await self._inference_client.completion(prompt)
        train_fields = result.get("northrl_train", result)

        # Create trajectory
        trajectory = Trajectory(
            tokens=train_fields["prompt_tokens"] + train_fields["response_tokens"],
            loss_mask=[0] * len(train_fields["prompt_tokens"]) + [1] * len(train_fields["response_tokens"]),
            reward=self._evaluate(result, task),
            is_val=task.get("is_val", False),
            extra_fields={"ground_truth": task["answer"]}
        )

        # Submit
        return self._put_trajectory(trajectory)
```

<a id="sandbox-task-hooks"></a>

### 沙箱任务函数

沙箱任务在入口脚本中实现 `run_agent` 和 `evaluate`，由 `E2BRolloutWorker`
记录模型交互并提交最终结果。详见 [E2B 任务接入](e2b-task-integration.md)
和[奖励评估](evaluators.md)。

<a id="error-handling"></a>

## 错误处理

训练任务中，从 `rollout` 逃出的异常保留 fail-fast 行为。wrapper 会发布空的 failed Trajectory，
然后异常继续传播；不会吞掉异常后自动执行下一个任务：

```python
async def rollout(self, task):
    try:
        # Process task
        result = self._process(task)
        return self._put_trajectory(result)
    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        raise  # execute_rollout submits an empty failed Trajectory, then propagates it
```

`track()` 上下文会报告传播出来的异常。跳过无效 GRPO group 并不意味着
worker 的所有异常都会自动恢复。

<a id="performance-considerations"></a>

## 性能注意事项

<a id="parallel-workers"></a>

### 并行工作进程

多个 worker 协程可以重叠执行：


**指南：**
- 仅在实际吞吐量提升时增加并发
- 控制本地驱动的内存和 CPU 使用
- 遵守推理服务容量及速率限制

<a id="batching"></a>

### 批处理

工作进程逐个处理任务，但以下组件会进行批处理：
1. **推理服务**：对多个请求进行批处理
2. **轨迹池**：交付完整 trajectories/groups，trainer 的 token 操作在内部遍历对应序列

<a id="resource-management"></a>

### 资源管理

Worker 作为本地协程运行。通过 `rollout.num_workers` 和可选的 `rollout.max_concurrent_requests` 控制并发，使其符合驱动内存与远端服务容量限制。详见 [rollout 配置](../11-configuration-reference/rollout-config.md)。

<a id="next-steps"></a>

## 后续步骤

- [Rollout 工作进程基类](base-rollout-worker.md) — 核心工作进程接口和方法
- [自定义工作进程](custom-workers.md) — 创建任务专属工作进程
- [评估器](evaluators.md) — 实现评估逻辑

<a id="related-documentation"></a>

## 相关文档

- [数据加载器](../03-data-loader/data-loader.md) — 任务来源
- [轨迹池](../04-trajectory-pool/trajectory-pool.md) — 轨迹目标
- [活动跟踪](../02-core-architecture/activity-tracking.md) — 监控
- [配置参考](../11-configuration-reference/rollout-config.md) — 全部配置选项
