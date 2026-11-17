[English](../../../en/developer-guide/02-core-architecture/activity-tracking.md) | 简体中文 | [文档目录](../../index.md)

<a id="activity-tracking-system"></a>

# 活动跟踪系统

活动跟踪系统在整个 NorthRL 框架中提供全面的进行中工作监控、错误报告和实验日志记录。

<a id="overview"></a>

## 概览

活动跟踪系统由两个主要组件组成：

1. **ActivityTracker**：监控所有模块活动的集中式跟踪器
2. **ActivityTrackerProxy**：将跟踪调用转发给集中式跟踪器的本地代理

<a id="activitytracker"></a>

## ActivityTracker

**位置**：`northrl/activity_tracker.py`

**用途**：集中跟踪所有模块中正在进行的工作

<a id="constructor"></a>

### 构造函数

```python
def __init__(self, config: DictConfig, max_errors: int = 1000)
```

**参数**：
- `config`：包含项目名称和实验名称的配置，用于日志记录
- `max_errors`：内存中保留的最大错误数量

**初始化内容**：
- 用于跟踪的线程同步原语
- 用于异常管理的错误报告器
- 用于 W&B 等后端的实验日志记录器（Tracking 实例）
- 用于进度指标的模块引用

<a id="activity-tracking-methods"></a>

### 活动跟踪方法

<a id="startmodule-str-work-str---str"></a>

#### start(module: str, work: str) -> str

登记工作的开始。

```python
token = activity_tracker.start("RolloutWorker-1", "processing_task")
```

**参数**：
- `module`：执行工作的模块名称
- `work`：正在执行的工作描述

**返回值**：此工作项的唯一令牌

**用法**：由 `ActivityTrackerProxy.track()` 上下文管理器自动调用

<a id="endtoken-str---none"></a>

#### end(token: str) -> None

登记工作的完成。

```python
activity_tracker.end(token)
```

**参数**：
- `token`：对应 `start()` 调用返回的令牌

**用法**：由 `ActivityTrackerProxy.track()` 上下文管理器自动调用

<a id="status-monitoring-methods"></a>

### 状态监控方法

<a id="is_quiescent---bool"></a>

#### is_quiescent() -> bool

检查所有工作是否已完成。

```python
if activity_tracker.is_quiescent():
    print("System is idle")
```

**返回值**：如果所有模块当前均无正在进行的工作，则为 `True`

**用法**：控制器使用此方法检测训练完成

<a id="is_rollout_worker_quiescent---bool"></a>

#### is_rollout_worker_quiescent() -> bool

检查所有 rollout 工作进程是否空闲。

```python
if activity_tracker.is_rollout_worker_quiescent():
    print("All rollout workers are idle")
```

**返回值**：如果所有已注册的 rollout 工作进程均无活动工作，则为 `True`

**用法**：验证器使用此方法检测验证完成

<a id="wait_quiescenttimeout-float--none--none---bool"></a>

#### wait_quiescent(timeout: float | None = None) -> bool

等待系统进入静止状态。

```python
if activity_tracker.wait_quiescent(timeout=30.0):
    print("System became quiescent")
else:
    print("Timeout waiting for quiescence")
```

**参数**：
- `timeout`：最长等待时间，单位为秒；`None` 表示无限等待

**返回值**：进入静止状态时为 `True`，超时时为 `False`

**用法**：控制器在优雅关闭期间使用此方法

<a id="get_running_status_summary---str"></a>

#### get_running_status_summary() -> str

获取当前活动的可读摘要。

```python
status = activity_tracker.get_running_status_summary()
print(status)
# Output: "RolloutWorker-1: 3 tasks, Trainer: 1 batch, ..."
```

**返回值**：按模块描述活动工作的字符串

**用法**：调试和监控

<a id="module-registration"></a>

### 模块注册

<a id="register_modulemodule_name-str-module_ref-any-is_rollout_worker-bool--false"></a>

#### register_module(module_name: str, module_ref: Any, is_rollout_worker: bool = False)

注册模块，用于进度指标和 rollout 活动跟踪。

```python
activity_tracker.register_module(
    module_name="RolloutWorker-1",
    module_ref=worker_actor,
    is_rollout_worker=True
)
```

**参数**：
- `module_name`：用于识别的名称
- `module_ref`：本地模块的引用
- `is_rollout_worker`：是否为 rollout 工作进程（用于专门监控）

**用法**：控制器在初始化期间注册所有模块


<a id="error-reporting"></a>

### 错误报告

<a id="report_exceptionmodule-str-work-str-exception-exception-severity-errorseverity--none--none---str"></a>

#### report_exception(module: str, work: str, exception: Exception, severity: ErrorSeverity | None = None) -> str

通过错误报告器报告异常。

```python
try:
    process_data()
except Exception as e:
    error_id = activity_tracker.report_exception(
        module="RolloutWorker-1",
        work="processing_task",
        exception=e,
        severity=ErrorSeverity.ERROR
    )
```

**参数**：
- `module`：发生异常的模块
- `work`：工作上下文
- `exception`：抛出的异常
- `severity`：错误严重程度（默认为 ERROR）

**返回值**：唯一错误 ID

**严重程度级别**：
- `ErrorSeverity.INFO`：信息
- `ErrorSeverity.WARNING`：警告
- `ErrorSeverity.ERROR`：错误（默认值）

<a id="get_error_health_status---dictstr-any"></a>

#### get_error_health_status() -> dict[str, Any]

获取错误健康状态信息。

```python
status = activity_tracker.get_error_health_status()
# {
#     "status": "healthy"|"warning"|"error",
#     "message": "...",
#     "recent_error_count": 0,
#     "error_level_count": 0,
#     "warning_level_count": 0
# }
```

**返回值**：包含健康状态和错误计数的字典

**用法**：控制器检查此信息以决定是否停止系统

<a id="training-step-tracking"></a>

### 训练步数跟踪

<a id="set_training_stepstep-int"></a>

#### set_training_step(step: int)

更新当前训练步数。

```python
activity_tracker.set_training_step(100)
```

**参数**：
- `step`：当前训练步数

**用法**：训练器在每个训练步骤结束后更新

<a id="get_training_step---int"></a>

#### get_training_step() -> int

获取当前训练步数。

```python
current_step = activity_tracker.get_training_step()
```

**返回值**：当前训练步数

**用法**：用于日志记录和实验跟踪

<a id="experiment-logging"></a>

### 实验日志记录

<a id="experiment_logger_postbackend-str-kwargs"></a>

#### experiment_logger_post(backend: str, **kwargs)

向实验日志后端提交指标或消息。

```python
# Log metrics
activity_tracker.experiment_logger_post(
    backend="wandb",
    data={"loss": 0.5, "reward": 1.0},
    step=100
)

# Log message
activity_tracker.experiment_logger_post(
    backend="feishu",
    content="Training completed!",
    title="Success"
)
```

**参数**：
- `backend`：日志后端（"wandb"、"feishu" 等）
- `**kwargs`：后端专属参数

**常用参数**：
- W&B：`data`（dict）、`step`（int）
- 飞书：`content`（str）、`title`（str）

**用法**：所有模块均可通过此统一接口记录日志

<a id="activitytrackerproxy"></a>

## ActivityTrackerProxy

**位置**：`northrl/activity_tracker.py`

**用途**：将跟踪调用转发给集中式 ActivityTracker 的本地代理

<a id="constructor-1"></a>

### 构造函数

```python
def __init__(self, central_tracker: Any)
```

**参数**：
- `central_tracker`：本地集中式 ActivityTracker 的引用

**用法**：控制器创建代理并分发给各模块

<a id="context-manager-pattern"></a>

### 上下文管理器模式

<a id="trackmodule-str-work-str-auto_report_errors-bool--true"></a>

#### track(module: str, work: str, auto_report_errors: bool = True)

用于自动跟踪活动的上下文管理器。

```python
with activity_tracker.track("MyModule", "processing_batch"):
    # Perform work
    process_batch(batch)
    # Automatic activity tracking and error reporting
```

**参数**：
- `module`：执行工作的模块名称
- `work`：工作描述
- `auto_report_errors`：是否自动报告异常（默认：True）

**特性**：
- 进入时自动调用 `start()`
- 退出时自动调用 `end()`
- `auto_report_errors=True` 时自动报告异常
- 安全地重新抛出异常

**手动处理错误的示例**：

```python
with activity_tracker.track("MyModule", "risky_work", auto_report_errors=False):
    try:
        risky_operation()
    except SpecificError as e:
        # Custom error handling
        logger.warning(f"Expected error: {e}")
        # Don't re-raise
```

<a id="forwarding-methods"></a>

### 转发方法

代理将以下方法转发给集中式跟踪器：

```python
# Status checking
is_quiescent = activity_tracker_proxy.is_quiescent()
is_workers_idle = activity_tracker_proxy.is_rollout_worker_quiescent()

# Training step
activity_tracker_proxy.set_training_step(step)
current_step = activity_tracker_proxy.get_training_step()

# Logging
activity_tracker_proxy.experiment_logger_post(
    backend="wandb",
    data=metrics,
    step=step
)
```

<a id="usage-patterns"></a>

## 使用模式

<a id="basic-activity-tracking"></a>

### 基本活动跟踪

```python
class MyModule(NorthRLModule):
    def process_work(self, item):
        with self._activity_tracker.track("MyModule", "process_work"):
            # Do work
            result = process(item)
            return result
```

<a id="multiple-work-items"></a>

### 多个工作项

```python
def process_batch(self, batch):
    with self._activity_tracker.track("MyModule", f"batch_{batch_id}"):
        for item in batch:
            # Each item tracked separately
            with self._activity_tracker.track("MyModule", f"item_{item_id}"):
                process_item(item)
```

<a id="error-reporting-with-context"></a>

### 带上下文的错误报告

```python
def risky_operation(self):
    with self._activity_tracker.track("MyModule", "risky_op"):
        try:
            dangerous_work()
        except CriticalError as e:
            # Manual reporting with specific severity
            self._activity_tracker.report_exception(
                module=self._module_name,
                work="risky_op",
                exception=e,
                severity=ErrorSeverity.ERROR
            )
            # Decide whether to re-raise
            raise
```

<a id="logging-metrics"></a>

### 记录指标

```python
def after_training_step(self, metrics, step):
    # Log to W&B
    self._activity_tracker.experiment_logger_post(
        backend="wandb",
        data={
            "train/loss": metrics["loss"],
            "train/reward": metrics["reward"],
        },
        step=step
    )

    # Update training step
    self._activity_tracker.set_training_step(step)
```

<a id="checking-quiescence"></a>

### 检查静止状态

```python
# In controller
def wait_for_completion(self):
    if self.activity_tracker.wait_quiescent(timeout=60.0):
        logger.info("All work completed")
    else:
        status = self.activity_tracker.get_running_status_summary()
        logger.warning(f"Timeout. Active work: {status}")
```

<a id="integration-with-controller"></a>

## 与控制器集成

控制器在整个系统中使用活动跟踪器：

```python
class NorthRLController:
    def __init__(self, config):
        # Create central tracker
        self.activity_tracker = ActivityTracker(config)

        # Create and distribute proxies to all modules
        for module in all_modules:
            proxy = ActivityTrackerProxy(self.activity_tracker)
            module.set_activity_tracker(proxy)

    def run(self):
        # Monitor system health
        while not self._check_finish():
            # Check quiescence
            if self.activity_tracker.is_quiescent():
                if self._all_data_processed():
                    break


            # Check for errors
            health = self.activity_tracker.get_error_health_status()
            if health["status"] == "error":
                logger.error("System errors detected")
                break
```

<a id="best-practices"></a>

## 最佳实践

<a id="activity-tracking"></a>

### 活动跟踪

1. **始终使用上下文管理器**进行自动跟踪
2. **选择描述清晰的工作名称**，便于调试
3. **跟踪主要工作**（避免跟踪琐碎操作）
4. **避免过度跟踪**（会影响性能）

<a id="error-reporting-1"></a>

### 错误报告

1. 多数情况下让 **auto_report_errors=True** 处理错误
2. 仅在特殊情况下**手动报告**
3. **设置合适的严重程度级别**
4. 在错误消息中**包含上下文**

<a id="logging"></a>

### 日志记录

1. **使用一致的指标名称**（例如使用 "train/loss"，而非 "loss"）
2. **以合适的间隔记录日志**（不要在每次迭代中记录）
3. 为时间序列**添加步数**
4. **先聚合再记录**（不要记录单个样本）

<a id="monitoring"></a>

### 监控

1. 关闭前**检查静止状态**
2. 定期**检查错误状态**
3. 针对关闭操作和外部工作**设置合适的超时时间**

<a id="configuration"></a>

## 配置


<a id="next-steps"></a>

## 工具错误遥测

E2B 按 `live/tool/error_<category>` 上报 `command_not_found`、
`dependency_missing`、`network`、`permission`、`resource`、`file_not_found`、
`edit`、`test_or_compile` 和 `other`。每个失败工具结果按调用 ID 去重，只分到
一个类别；失败由错误标记或明确的非零退出码识别，成功读取的文件正文中的报错
不会误计。前五类表示环境问题线索，不改变任务状态和奖励。

`live/tool/env_issue_rollout_rate` 的分子是受影响的已结束逻辑 rollout，分母是
至少观察到一个工具返回的已结束 rollout。`timeout_env_issue_rollout_rate` 使用
其中任意 attempt 曾超时的子集，表示共现而非因果；重试前的观测也会保留。
同一前缀下的 `observed_rollouts`、`unobserved_rollouts` 和 `observation_coverage`
展示分母及缺失观测。分母为零时不发布比例；覆盖率表示有观测的 rollout 占比，
不表示 trace 完整率。

attempt 结束前会用最终结果中的工具计数补齐实时轮询。未随后续模型请求返回的
工具结果仍可能不可观测。指标沿用实时刷新周期和时间横轴，进程重启后计数重置。

## 后续步骤

- 了解[控制器](controller.md)如何使用活动跟踪
- 阅读[错误报告](../08-features/error-handling.md)
- 探索[模块开发](../12-best-practices/module-development.md)，了解集成方法

## 耗时指标

`timing_weights/weight_export_seconds` 统计采样权重导出，
`timing_weights/checkpoint_seconds` 统计成功的 checkpoint 保存；二者均不计入
`timing_train/batch_training_seconds`。`timing_weights/weight_sync_seconds` 统计 sampler
发布与切换，不包含导出、validation 和日志上报；在 validation 或下一批放行前，
并入产生该权重的 `training/train_step`。未执行操作不补零；失败使用
`timing_weights/weight_export_failed_seconds` / `timing_weights/weight_sync_failed_seconds`，
并将对应的 `training/weight_export_success_flag` / `training/weight_sync_success_flag` 置为 0。
详见[训练步计时指标](../../metrics.md)。

E2B 的 `live/inference/*` 上报按结果分类的请求数、累计/平均/最大延迟、合并累计
直方图估算的 P95，以及最近快照中的活动请求数和最长等待时间。每次结束的
`generate` 调用都参与，包括错误、超时和取消；耗时包含客户端排队及后端重试，
不能解释为纯 GPU 时间或 TTFT。沿用 30 秒快照节奏；活动状态允许下降，旧心跳不能
恢复已经结束的请求。请求计数与桶保留在运行时遥测，不作为完整轨迹的训练 score 求均值。

开启 debug rollout trace 保存时，逐请求记录在沙箱清理前尽力回收为 trace 旁的
`e2b_rollout_<rollout_id>.requests.<attempt_id>.jsonl`，失败重试各自保留文件。
记录包含身份、耗时、结果和 token 数，不含模型内容。常规 PROFILE 日志降为 DEBUG，
失败和超过 60 秒的慢请求仍可见；`LOG_LEVEL=DEBUG` 会传入 E2B 沙箱。
