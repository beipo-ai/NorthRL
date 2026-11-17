[English](../../../en/developer-guide/05-rollout-workers/base-rollout-worker.md) | 简体中文 | [文档目录](../../index.md)

<a id="base-rollout-worker"></a>

# Rollout 工作进程基类

**术语约定：** 一次 rollout 产出一个完整 `Trajectory`。普通轨迹直接保存 `tokens` 和 `loss_mask`（`segments=None`）；需要保留独立训练序列时才使用内部 `segments` 字典列表。多轮交互不一定需要分段。worker、pool 和 trainer 准备接口传递完整 trajectory。详见[轨迹术语说明](../04-trajectory-pool/trajectory-semantics.md)。

`BaseRolloutWorker` 是 NorthRL 中所有 rollout 工作进程的抽象基类，提供任务处理、LLM 交互和轨迹生成的核心基础设施。

<a id="class-definition"></a>

## 类定义

**位置：** `northrl/rollout_worker/base_rollout_worker.py`

```python
from northrl.rollout_worker import BaseRolloutWorker

class BaseRolloutWorker(NorthRLModule, ABC):
    """Base class for rollout workers."""
```

<a id="constructor"></a>

## 构造函数

```python
def __init__(self, config: DictConfig):
    """
    Initialize the rollout worker.

    Args:
        config: Worker configuration containing:
            - inference_service: Inference service config
            - need_llm_inference: Whether LLM inference is needed
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens
            - ... worker-specific config
    """
```

构造函数会：
1. 初始化线程相关组件
2. 设置停止事件，以支持优雅关闭
3. 准备模块引用占位符（稍后设置）
4. 不初始化推理客户端（延迟至 `init_inference_service_client()`）

<a id="abstract-method"></a>

## 抽象方法

<a id="rollout"></a>

### rollout()

派生类必须实现的核心方法：

```python
@abstractmethod
async def rollout(self, task: dict[str, Any]) -> str | None:
    """
    Execute a single rollout operation.

    Args:
        task: Task dictionary from data loader

    Returns:
        - "success": Complete trajectory accepted for submission
        - "fail": Failed to submit trajectory
        - None: No explicit status; absent submission becomes a skipped trajectory
    """
```

**实现指南：**
1. 从 `task` 中提取必需字段
2. 执行任务专属处理（LLM 调用、智能体执行等）
3. 普通结果直接保存 token 字段；需要分段的结果使用内部字典记录
4. 普通结果构造 `Trajectory(tokens=tokens, loss_mask=loss_mask, reward=final_reward)`；需要独立序列时才构造 `Trajectory(segments=segments, reward=final_reward)`
5. 将完整结果传给 `self._put_trajectory(trajectory)`，只调用一次

训练任务由主循环通过 `execute_rollout` 填写执行身份并暂存对象，待 `rollout` 返回后
一次性发布。未提交对象就返回时，生成空的 skipped 结果；若返回 `"fail"`，则生成
failed 结果。已暂存对象后返回 `"fail"`，会将该对象标为 failed。已评分但没有训练序列
的成功结果必须显式提交 `Trajectory(segments=[], reward=final_reward)`。

异常从 `rollout` 逃出时，训练 wrapper 提交空的 failed trajectory 后继续抛出，不会自动
重试。验证任务绕过这层逻辑，直接调用 `rollout`；`_put_trajectory` 将 `is_val=True`
对象立即交给 validator，不自动补 skipped/failed 验证结果。
生命周期约定见 [worker 概览](overview.md#2-rollout-execution)。

<a id="setup-methods"></a>

## 设置方法

<a id="set_module_references"></a>

### set_module_references()

由控制器调用，建立模块之间的连接：

```python
def set_module_references(
    self,
    trajectory_pool: TrajectoryPool,
    dataloader: BaseDataLoader,
    weight_sync_controller: WeightSyncController,
    validate_dataloader: BaseDataLoader,
    validator: Validator,
):
    """Set module references for the worker."""
```

**调用时机：** 控制器初始化期间、`run()` 之前

<a id="init_inference_service_client"></a>

### init_inference_service_client()

初始化 LLM 推理客户端：

```python
def init_inference_service_client(self, service_holder=None):
    """
    Initialize inference service client.

    Args:
        service_holder: Shared service holder (Tinker/Weaver) if applicable
    """
```

**特性：**
- 仅在 `need_llm_inference=True` 时初始化
- 支持多个后端（vLLM、Tinker、Weaver 等）
- 设置权重同步协调

**调用时机：** 控制器初始化期间，设置模块引用之后

<a id="set_activity_tracker"></a>

### set_activity_tracker()

设置活动监控代理：

```python
def set_activity_tracker(self, tracker: ActivityTrackerProxy):
    """Set activity tracker for monitoring."""
```

继承自 `NorthRLModule`。

<a id="lifecycle-methods"></a>

## 生命周期方法

<a id="run"></a>

### run()

将 worker 协程提交到共享 rollout 事件循环：

```python
def run():
    """Schedule worker execution on the rollout event loop."""
```

**过程：**
1. 验证模块引用已设置
2. 清除停止事件
3. 调度 `_main_loop()` 并保留其 future

**前置条件：**
- 必须已调用 `set_module_references()`
- 必须已调用 `set_activity_tracker()`
- 必须已调用 `init_inference_service_client()`（如有需要）

<a id="stop"></a>

### stop()

优雅地停止工作进程：

```python
def stop():
    """Request stop, wait for the worker future, and cancel on timeout."""
```

**过程：**
1. 设置停止事件
2. 最多等待协程完成 30 秒，超时后请求取消

<a id="_main_loop"></a>

### _main_loop()

在共享 rollout 事件循环上调度的内部协程：

```python
async def _main_loop(self) -> None:
    """Main worker loop, scheduled as an asyncio task."""
```

**支持通知的 loader 的处理过程（简化）：**
```python
while not self._stop_event.is_set():
    ready.clear()  # Clear before checking for tasks, never after an empty result.
    with self._activity_tracker.track(self._module_name, "rollout"):
        self._next_task = self._get_rollout_task()
        while self._next_task is not None and not self._stop_event.is_set():
            await self.execute_rollout(copy.deepcopy(self._next_task))
            if not self._stop_event.is_set():
                self._next_task = self._get_rollout_task()

    if self._next_task is None and not self._stop_event.is_set():
        await ready.wait()
```

**主要特性：**
- 跟踪活动以便监控
- rollout 返回后领取下一个任务，除非已收到停止请求
- 领取任务前订阅 loader 的就绪通知；没有任务时在活动跟踪之外等待。
  Sequential loader 在开放下一批或 reset 后通知，消除空轮询及其与训练的 CPU/GIL 争用。
- 在检查任务之前清除通知。检查前发布的任务能从 loader 状态中读到；
  检查之后到达的通知会一直保留到 worker 开始等待，避免漏唤醒。
- 停止和验证模式切换也通过 rollout 事件循环的线程安全调度唤醒空闲 worker。
  每次运行创建独立事件，退出时取消订阅。
- 未支持 `subscribe_ready` 的 loader 保留一秒轮询兜底，停止和验证切换仍会立即唤醒。
  pool 容量限制继续使用原有的独立等待逻辑。

<a id="data-flow-methods"></a>

## 数据流方法

<a id="_get_rollout_task"></a>

### _get_rollout_task()

从对应数据加载器获取下一个任务：

```python
def _get_rollout_task(self) -> dict[str, Any] | None:
    """
    Get next rollout task from data loader.

    Returns:
        Task dictionary or None if no tasks available
    """
```

**行为：**
- `_is_running_validate=True` 时使用验证数据加载器
- 否则使用训练数据加载器
- 无任务时返回 `None`；就绪等待由主循环处理
- 非阻塞操作

<a id="_put_trajectory"></a>

### _put_trajectory()

每次 rollout 提交一个完整 trajectory：

```python
def _put_trajectory(self, trajectory: Trajectory) -> str:
    """
    Submit trajectory to pool or validator.

    Args:
        trajectory: One complete rollout result, with direct token fields or internal segments

    Returns:
        - "success": Submitted successfully
        - "fail": Failed to submit
    """
```

**路由规则：**
- **验证**（`trajectory.is_val`）：路由至验证器
- **训练**：先暂存完整对象，再由执行 wrapper 向 pool 发布

同次训练 `execute_rollout` 内第二次调用会报错；验证提交直接进入 validator，不使用
这项检查。此方法校验普通序列或每个内部序列的长度，截断和最终 reward 调整
必须提前完成。下方普通 `Trajectory(tokens=..., loss_mask=..., reward=...)` 构造直接保存 token 字段，`segments=None`。

<a id="validation-mode-methods"></a>

## 验证模式方法

<a id="begin_validate"></a>

### begin_validate()

将工作进程切换至验证模式：

```python
def begin_validate():
    """Switch to validation mode."""
```

**影响：**
- 将 `_is_running_validate` 设为 `True`
- 后续任务来自验证数据加载器
- 轨迹路由至验证器

**调用时机：** 控制器在验证周期开始时调用

<a id="end_validate"></a>

### end_validate()

将工作进程切回训练模式：

```python
def end_validate():
    """Switch back to training mode."""
```

**影响：**
- 将 `_is_running_validate` 设为 `False`
- 后续任务来自训练数据加载器
- 轨迹路由至轨迹池

**调用时机：** 控制器在验证周期结束时调用

<a id="inference-client-interface"></a>

## 推理客户端接口

工作进程通过 `_inference_client` 访问 LLM：

<a id="completion"></a>

### completion()

单轮补全：

```python
result = await self._inference_client.completion(
    prompt="Your prompt here"
)
```

**返回值：**
```python
{
    "choices": [{"text": "Generated text", "finish_reason": "stop"}],
    "northrl_train": {
        "prompt_tokens": [...],
        "response_tokens": [...],
        "response_logprobs": [...],
        "response_old_logprobs": [...],
    },
    "sampling_mask": None,  # May instead contain an inline mask or opaque reference
    "weight_version": "...",  # When reported by the sampling service
    # Other OpenAI-compatible completion fields
}
```

远端客户端返回上面的 OpenAI 风格 envelope：token 字段位于 `northrl_train`，
completion 文本位于 `choices[0]["text"]`；mock 或旧客户端可能返回扁平字段。
采样参数来自 `rollout.sampling`（验证时使用验证采样配置）；向 `completion()` 额外传入
`temperature`/`max_tokens` 关键字参数不会覆盖这些配置。

<a id="generate"></a>

### generate()

使用消息进行聊天式生成：

```python
result = await self._inference_client.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    tools=[...],  # Optional tool definitions
)
```

**返回值：**
```python
{
    "choices": [{
        "message": {"role": "assistant", "content": "Generated text"},
        "finish_reason": "stop",
    }],
    "northrl_train": {"prompt_tokens": [...], "response_tokens": [...]},
    # Other OpenAI-compatible chat completion fields
}
```

<a id="weight-sync-coordination"></a>

### 权重同步协调

推理客户端自动处理权重同步：
- `freeze_for_weight_sync=True` 时在权重同步期间阻塞
- 此行为对工作进程实现透明

<a id="example-implementation-simplerolloutworker"></a>

## 实现示例：SimpleRolloutWorker

下面的示例展示普通 trajectory 的数据形状和一次提交约定：

```python
from northrl.rollout_worker import BaseRolloutWorker
from northrl.northrl_types import Trajectory
from typing import Any
import logging
import re

logger = logging.getLogger(__name__)

class SimpleRolloutWorker(BaseRolloutWorker):
    """Simple worker for basic LLM completion tasks."""

    async def rollout(self, task: dict[str, Any]) -> str | None:
        # 1. Validate task
        if "prompt" not in task:
            logger.error(f"Task missing 'prompt' field: {task}")
            return None

        prompt = task["prompt"]

        # 2. Call LLM
        assert self._inference_client is not None
        completion_result = await self._inference_client.completion(prompt)

        # 3. Extract tokens from the remote envelope (or legacy flat fields).
        train_fields = completion_result.get("northrl_train", completion_result)
        prompt_tokens = train_fields["prompt_tokens"]
        response_tokens = train_fields["response_tokens"]
        response_logprobs = train_fields.get("response_old_logprobs")
        if response_logprobs is None:
            response_logprobs = train_fields["response_logprobs"]

        # 4. Create token sequence and loss mask
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        old_log_probs = [0.0] * len(prompt_tokens) + response_logprobs
        tokens, loss_mask, old_log_probs, is_truncated = self._check_and_truncate(
            tokens, loss_mask, old_log_probs
        )

        # 5. Extract answer (task-specific)
        choices = completion_result.get("choices") or [{}]
        response = completion_result.get("response", choices[0].get("text", ""))
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else ""

        # 6. Get ground truth
        ground_truth = task.get("ground_truth", "")

        # 7. Calculate reward
        reward = 1.0 if not is_truncated and extracted_answer == ground_truth else 0.0

        # 8. Create trajectory
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                "ground_truth": ground_truth,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
                "temperature": self._config.rollout.sampling.temperature,
                "finish_reason": completion_result.get("finish_reason", choices[0].get("finish_reason", "stop")),
                "old_log_probs": old_log_probs,
                "is_truncated": is_truncated,
                "response": response,
                "extracted_answer": extracted_answer,
            },
        )

        # 9. Submit trajectory
        return self._put_trajectory(trajectory)
```

<a id="configuration-example"></a>

## 配置示例


<a id="common-patterns"></a>

## 常见模式

<a id="pattern-1-basic-completion"></a>

### 模式 1：基本补全

```python
class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        prompt = task["prompt"]
        result = await self._inference_client.completion(prompt)

        trajectory = self._create_trajectory(result, task)
        return self._put_trajectory(trajectory)
```

<a id="pattern-2-multi-step-processing"></a>

### 模式 2：多步骤处理

```python
class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        # Step 1: Generate reasoning
        reasoning_prompt = self._format_reasoning_prompt(task)
        reasoning = await self._inference_client.completion(reasoning_prompt)

        # Step 2: Generate answer
        answer_prompt = self._format_answer_prompt(task, reasoning)
        answer = await self._inference_client.completion(answer_prompt)

        # Step 3: Evaluate and create trajectory
        trajectory = self._create_trajectory(answer, task)
        return self._put_trajectory(trajectory)
```

<a id="pattern-3-error-handling"></a>

### 模式 3：错误处理

```python
class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        try:
            result = self._process_task(task)
            trajectory = self._create_trajectory(result, task)
            return self._put_trajectory(trajectory)
        except ValueError as e:
            logger.error(f"Invalid task: {e}")
            return None  # No submitted object: training wrapper records a skipped outcome.
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise  # The wrapper publishes failure, then the exception propagates.
```

<a id="threading-considerations"></a>

## 线程注意事项

**线程安全：**
- 每个工作进程在自己的线程中运行
- 初始化后模块引用为只读
- 活动跟踪器提供线程安全的跟踪
- 推理客户端处理并发请求

**最佳实践：**
- 不要在 `rollout()` 中修改共享状态
- 不要在 `rollout()` 中无限期阻塞
- 对耗时操作使用活动跟踪器

<a id="performance-tips"></a>

## 性能建议

1. **保持 rollout() 高效**：尽量减少非必要工作
2. **尽可能批处理**：部分推理服务支持批处理
3. **使用异步模式**：适用于 I/O 密集型操作

<a id="debugging"></a>

## 调试

<a id="logging"></a>

### 日志记录

```python
import logging
logger = logging.getLogger(__name__)

class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        logger.debug(f"Processing task: {task.get('task_id')}")
        # ... processing
        logger.info(f"Generated trajectory with reward: {reward}")
```

<a id="activity-tracking"></a>

### 活动跟踪

主循环会自动跟踪活动：

```python
with self._activity_tracker.track(self._module_name, "rollout"):
    result = await self.execute_rollout(task)
```

检查工作进程活动：

```python
# In controller or monitoring
if activity_tracker.is_rollout_worker_quiescent():
    print("All workers idle")
```

<a id="common-issues"></a>

## 常见问题

<a id="issue-inference-client-not-initialized"></a>

### 问题：推理客户端未初始化

**症状：** `AttributeError: 'NoneType' object has no attribute 'completion'`

**解决方法：** 确保配置中设置了 `need_llm_inference=true`

<a id="issue-tasks-not-appearing"></a>

### 问题：任务未出现

**症状：** 工作进程空闲，但数据加载器中有数据

**解决方法：** 检查数据加载器的 `can_return_item()` 和 `is_finished()` 方法

<a id="next-steps"></a>

## 后续步骤

- [自定义工作进程](custom-workers.md) — 创建任务专属工作进程
- [评估器](evaluators.md) — 实现评估逻辑

<a id="related-documentation"></a>

## 相关文档

- [数据类型](../02-core-architecture/data-types.md) — 轨迹结构
- [活动跟踪](../02-core-architecture/activity-tracking.md) — 监控细节
- [权重同步](../08-features/weight-synchronization.md) — 同步协调
