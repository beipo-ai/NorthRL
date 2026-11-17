[English](../../../en/developer-guide/02-core-architecture/controller.md) | 简体中文 | [文档目录](../../index.md)

<a id="northrl-controller"></a>

# NorthRL 控制器



`NorthRLController` 是 NorthRL 框架的主协调器，负责初始化、协调和监控所有框架组件。

<a id="overview"></a>

## 概览

控制器作为核心协调节点，负责：
- 初始化所有框架模块
- 设置模块引用和依赖关系
- 管理训练生命周期
- 监控上报的错误
- 协调验证周期
- 处理检查点保存和权重同步

<a id="class-definition"></a>

## 类定义

```python
class NorthRLController:
    """
    NorthRL controller is the main process of the RL training framework.
    Responsible for starting the entire experiment, initializing various modules,
    and synchronizing configuration information.
    """
```

**位置**：`northrl/controller.py`

<a id="constructor"></a>

## 构造函数

```python
def __init__(self, config: DictConfig)
```

**参数**：
- `config`：包含所有模块设置的 Hydra 配置

**初始化步骤**：
1. 设置日志配置
2. 设置随机数生成器种子
3. 初始化活动跟踪器和错误报告器
4. 初始化外部服务（如果配置了 Tinker/Weaver）
5. 初始化所有框架模块
6. 设置模块引用和依赖关系

**示例**：

```python
from omegaconf import DictConfig
from northrl import NorthRLController

config = DictConfig({...})
controller = NorthRLController(config)
```

<a id="module-registry"></a>

## 模块注册表

控制器为每种角色维护模块类型注册表。这支持动态加载模块，并兼容内置实现和自定义实现。

<a id="module_registry-structure"></a>

### MODULE_REGISTRY 结构

```python
MODULE_REGISTRY = {
    NorthRLRole.ROLLOUT_WORKER: {
        "mock": MockRolloutWorker,
        "simple": SimpleRolloutWorker,
        "single_turn": SingleTurnRolloutWorker,
        "e2b": E2BRolloutWorker,
        # Custom workers loaded dynamically
    },
    NorthRLRole.TRAINER: {
        "grpo": RemoteApiGrpoTrainer,
        "sft": RemoteApiCrossEntropyTrainer,
        "opd": RemoteApiOpdTrainer,
    },
    NorthRLRole.DATA_LOADER: {
        "mock": MockDataLoader,
        "torch": TorchDataLoader,
    },
    # ... other roles
}
```

SFT 会绕过公共 rollout 工作进程注册表。设置 `algorithm.type: sft` 时，
应省略顶层 `rollout` 配置块，控制器只创建一个
内部直通式 `SFTRolloutWorker`。如果显式提供 `rollout`
配置块，将发出 `UserWarning` 并忽略整个配置块。

<a id="dynamic-module-loading"></a>

### 动态模块加载

对于 rollout 工作进程，控制器支持从配方目录加载自定义实现：

**配置示例**：


**加载过程**：

1. 检查是否指定了 `rollout.custom_module_path`
2. 相对于配方 YAML 目录解析路径，或使用绝对路径
3. 使用 `importlib` 动态导入模块
4. 使用 `getattr` 提取 `rollout.custom_class_name` 指定的类（默认 `CustomRolloutWorker`）
5. 如果未指定，则回退到已注册的工作进程类型

**优点**：
- 配方包含自己的自定义逻辑，可独立使用
- 新任务无需修改框架核心代码
- 便于对任务专属实现进行版本控制和共享
- 同时支持绝对路径和相对路径

<a id="core-methods"></a>

## 核心方法

<a id="run"></a>

### run()

启动所有组件并进入监控循环，开始训练过程。

```python
def run() -> None
```

**过程**：

1. **初始化训练工作进程**
   ```python
   execute(self.trainer.initialize_workers)
   ```

2. **加载初始检查点**
   ```python
   self._load_initial_checkpoint()
   ```

3. **可选的训练前验证**
   ```python
   if self._config.validate.validate_before_train:
       self._start_validate(model_tag)
       # ... start workers ...
       self._end_validate(model_tag)
   ```

4. **启动所有组件**
   ```python
   execute(self.trainer.run)
   for worker in self.rollout_workers:
       execute(worker.run)
   ```

5. **进入监控循环**
   - 检查验证触发条件
   - 检查训练是否完成
   - 检查系统异常
   - 处理优雅关闭

**监控循环配置**：

```yaml
runtime:
  health:
    exception_policy: stop_on_error
    exception_check_interval_s: 1.0
    memory_log_interval_s: 10.0
```

**使用示例**：

```python
controller = NorthRLController(config)
controller.run()  # Blocks until training completes or error occurs
```

<a id="stop"></a>

### stop()

优雅地停止所有组件，并等待活动完成。

```python
def _stop()
```

**过程**：

1. 通知所有工作进程停止
2. 等待进入静止状态，并设置超时（默认：30 秒）
3. 如果超时，记录剩余活动
4. 清理资源

**特性**：
- 非阻塞的停止信号
- 可配置的超时时间
- 用于调试的活动日志

<a id="completion-checking"></a>

## 完成状态检查

<a id="_check_finish---bool"></a>

### _check_finish() -> bool

根据各种条件判断是否应停止训练。

```python
def _check_finish(self) -> bool
```

**判断条件**：

1. **达到最大训练步数**
   ```python
   if self.trainer.global_step >= self._config.trainer.total_train_steps:
       return True
   ```

2. **系统进入静止状态**（所有工作已完成）
   ```python
   if self.activity_tracker.is_quiescent():
       if self.dataloader.is_finished():
           if self.trajectory_pool.is_empty():
               return True
   ```

**返回值**：应停止训练时为 `True`，否则为 `False`

<a id="progress-reporting"></a>

### 进度报告

在训练期间、等待验证期间，以及等待训练器和 rollout 工作进程停止期间，
控制器会在主线程上刷新 rollout 进度。
到达更新时机时，`ActivityTracker.update_progress()` 会采集并显示一次更新；
`runtime.progress.update_interval_s`、`log_every_n_steps` 和 `use_tqdm` 继续
控制刷新频率和输出。健康检查与内存检查保留各自的
执行间隔。控制器进行其他阻塞调用时，进度输出可能暂停。

<a id="memory-usage-logging"></a>

### 内存使用日志

监控循环每隔 `memory_log_interval_s` 秒会记录一行
驱动进程的内存摘要：

```
Driver memory usage: driver_rss=12.34GiB pod_mem=96.50/128.00GiB (75.4%)
```

- `driver_rss` — 仅控制器进程的常驻内存。
- `pod_mem` — 进程所在整个 cgroup 的使用量 / 上限，
  从 cgroup v2（`/sys/fs/cgroup/memory.{current,max}`）或 v1
  （`/sys/fs/cgroup/memory/memory.{usage_in_bytes,limit_in_bytes}`）读取。这是
  触发内核 OOM-killer 的内存指标。

其目的是支持事后诊断：OOM-kill 会发送 SIGKILL，且没有
Python 回溯，因此最后记录的 `pod_mem` 往往是驱动所在 cgroup 触及内存上限的
唯一磁盘证据。辅助函数
（`northrl.utils.logging_utils.format_memory_usage`）不会抛出异常；如果 psutil 或
cgroup 文件不可用，则退化为部分信息或标记字符串。

<a id="_check_module_exceptions---bool"></a>

### _check_module_exceptions() -> bool

检查系统中的严重错误。

```python
def _check_module_exceptions(self) -> bool
```

**返回值**：系统健康时为 `True`，检测到严重错误时为 `False`

**过程**：

1. 从活动跟踪器获取错误健康状态
2. 检查错误策略配置
3. 判断错误是否要求关闭系统

**错误策略**：

- `"stop_on_error"`：遇到任何错误时停止
- `"stop_on_critical"`：仅遇到严重错误时停止
- `"continue"`：记录错误但继续运行

**配置**：

```yaml
runtime_monitor:
  exception_handling:
    enabled: true
    check_interval: 1.0
    policy: "stop_on_error"
```

<a id="checkpoint-management"></a>

## 检查点管理

<a id="_load_initial_checkpoint"></a>

### _load_initial_checkpoint()

加载初始检查点，或准备从头训练。

```python
def _load_initial_checkpoint()
```

**过程**：

1. 检查是否从检查点恢复
2. 如果恢复训练，则调用 `_load_resume_checkpoint()`
3. 如果从头开始，则创建权重同步缓冲区
4. 向推理服务执行初始权重同步

<a id="_load_resume_checkpoint"></a>

### _load_resume_checkpoint()

根据恢复配置加载检查点。

```python
def _load_resume_checkpoint()
```

**恢复模式**：

1. **禁用**：不恢复
   ```yaml
   resume:
     mode: "disable"
   ```

2. **自动**：自动查找最新检查点
   ```yaml
   resume:
     mode: "auto"
   ```

3. **指定路径**：从特定检查点恢复
   ```yaml
   resume:
     mode: "from_path"
     resume_path: "/path/to/checkpoint"
   ```

**过程**：

1. 根据模式确定检查点路径
2. 调用训练器的 `resume_checkpoint()` 方法
3. 更新活动跟踪器中的训练步数

<a id="_find_latest_checkpointcheckpoint_folder-str---str--none"></a>

### _find_latest_checkpoint(checkpoint_folder: str) -> str | None

在指定文件夹中查找最新检查点。

```python
def _find_latest_checkpoint(self, checkpoint_folder: str) -> str | None
```

**参数**：
- `checkpoint_folder`：包含检查点的文件夹路径

**返回值**：最新检查点的路径；如果未找到，则为 `None`

**过程**：

1. 列出所有 `global_step_*` 目录
2. 从目录名称解析步数
3. 返回步数最大的路径

**预期目录结构**：
```
checkpoint_folder/
├── global_step_100/
├── global_step_200/
├── global_step_300/
└── ...
```

<a id="validation-coordination"></a>

## 验证协调

<a id="_run_validatemodel_tag-modeltag"></a>

### _run_validate(model_tag: ModelTag)

在权重同步事件后运行完整的验证周期。

```python
def _run_validate(self, model_tag: ModelTag)
```

**过程**：

1. 调用 `_start_validate(model_tag)` 开始验证
2. 等待验证完成（validator.is_complete()）
3. 调用 `_end_validate(model_tag)` 结束验证

**阻塞行为**：此方法会阻塞，直到验证完成

<a id="_start_validatemodel_tag-modeltag"></a>

### _start_validate(model_tag: ModelTag)

将 rollout 工作进程切换至验证模式，开始验证。

```python
def _start_validate(self, model_tag: ModelTag)
```

**过程**：

1. 将 rollout 工作进程切换至验证模式
   ```python
   for worker in self.rollout_workers:
       execute(worker.begin_validate)
   ```

2. 工作进程随后会：
   - 使用验证数据加载器
   - 将轨迹提交给验证器，而不是轨迹池

<a id="_end_validatemodel_tag-modeltag"></a>

### _end_validate(model_tag: ModelTag)

结束验证并记录结果。

```python
def _end_validate(self, model_tag: ModelTag)
```

**过程**：

1. 计算并记录验证指标
   ```python
   metrics = execute(self.validator.compute_and_log_metrics)
   ```

2. 将工作进程切回训练模式
   ```python
   for worker in self.rollout_workers:
       execute(worker.end_validate)
   ```

3. 清空验证轨迹
   ```python
   execute(self.validator.clear)
   ```

4. 通知权重同步控制器
   ```python
   execute(self.weight_sync_controller.end_validate, model_tag)
   ```

<a id="module-initialization"></a>

## 模块初始化

控制器按特定顺序初始化模块，以满足依赖关系：

```python
def _init_modules(self):
    # 1. Data loaders (no dependencies)
    self.dataloader = self._create_module(NorthRLRole.DATA_LOADER)
    self.validate_dataloader = self._create_module(NorthRLRole.VALIDATE_DATALOADER)

    # 2. Trajectory management
    self.trajectory_pool = self._create_module(NorthRLRole.TRAJECTORY_POOL)
    self.validator = self._create_module(NorthRLRole.VALIDATOR)

    # 3. Weight synchronization
    self.weight_sync_controller = self._create_module(NorthRLRole.WEIGHT_SYNC_CONTROLLER)

    # 4. Trainer (depends on trajectory_pool)
    self.trainer = self._create_module(NorthRLRole.TRAINER)

    # 5. Rollout workers (depend on all above)
    self.rollout_workers = self._create_modules(NorthRLRole.ROLLOUT_WORKER)

    # 6. Set up module references
    self._set_module_references()
```

<a id="launch-modes"></a>
<a id="local-mode"></a>

## 本地执行

控制器、数据加载器、轨迹池和 rollout worker 运行在本地驱动进程中。Weaver 在远端执行模型训练与采样，E2B 任务使用配置好的沙箱服务。`rollout.num_workers` 控制 rollout 并发数，应根据本地驱动和服务容量设置。

通过 `python -m northrl.main` 启动，由入口校验配方、创建控制器并管理重试和清理。完整命令与环境要求见[本地运行](../01-getting-started/quick-start.md#run-locally)。

<a id="best-practices"></a>

## 最佳实践

<a id="configuration"></a>

### 配置

1. **使用 Hydra 组合配置**复用通用设置
2. 运行前**验证配置**
3. 在配方中**记录自定义模块**
4. 将配置纳入**版本控制**

<a id="error-handling"></a>

### 错误处理

1. 根据使用场景**配置合适的错误策略**
2. 在生产环境中**监控上报的错误**
3. 根据工作负载**设置合理的超时时间**
4. 定期**检查错误日志**

<a id="performance"></a>

### 性能

1. 根据驱动和远端服务容量**调整本地 rollout 并发数**
2. 适当**配置内存日志间隔**
3. **监控活动跟踪器**以发现瓶颈
4. 在恢复时间与开销之间**优化检查点频率**

<a id="validation"></a>

### 验证

1. **启用训练前验证**以检查设置
2. 根据训练速度**配置验证频率**
3. **监控验证指标**以评估模型质量
4. 根据评估需求**平衡验证开销**

<a id="example-complete-workflow"></a>

## 完整工作流程

安装、服务凭据、配方选择和日志说明统一维护在[本地快速入门](../01-getting-started/quick-start.md)中。公开入口负责配置校验和控制器生命周期，任务代码通过支持的扩展 hook 接入即可。

<a id="next-steps"></a>

## 后续步骤

- 了解控制器使用的[数据类型](data-types.md)
- 阅读[活动跟踪](activity-tracking.md)，了解监控方法
- 探索[模块开发](../12-best-practices/module-development.md)
