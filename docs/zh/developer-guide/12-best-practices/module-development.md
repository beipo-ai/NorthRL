[English](../../../en/developer-guide/12-best-practices/module-development.md) | 简体中文 | [文档目录](../../index.md)

<a id="module-development"></a>

# 模块开发

开发 NorthRL 模块的最佳实践。

<a id="module-base-class"></a>

## 模块基类

所有 NorthRL 组件都应继承 `NorthRLModule`：

```python
from northrl.northrl_module import NorthRLModule
from northrl.northrl_types import NorthRLRole
from omegaconf import DictConfig

class MyModule(NorthRLModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config
        # Initialize module-specific state

```

**优点：**
- 统一的本地模块接口
- 集成活动追踪
- 支持模块命名

<a id="configuration-handling"></a>

## 配置处理

<a id="accept-dictconfig"></a>

### 接受 DictConfig

始终接受 `DictConfig` 作为配置：

```python
def __init__(self, config: DictConfig):
    self._config = config

    # Access with defaults
    self._batch_size = config.get("batch_size", 32)

    # Environment variable interpolation
    self._data_path = config.data_files[0]  # Already interpolated
```

<a id="validate-configuration"></a>

### 验证配置

尽早验证关键配置：

```python
def __init__(self, config: DictConfig):
    super().__init__()

    # Validate required fields
    assert "batch_size" in config, "batch_size is required"
    assert config.batch_size > 0, "batch_size must be positive"

    # Validate paths exist
    if "data_files" in config:
        for file_path in config.data_files:
            assert os.path.exists(file_path), f"Data file not found: {file_path}"
```

<a id="module-references"></a>

## 模块引用

<a id="set-references-pattern"></a>

### 设置引用模式

使用 `set_module_references()` 处理模块间依赖：

```python
class TrajectoryPool(NorthRLModule):
    def set_module_references(
        self,
        dataloader: BaseDataLoader,
        weight_sync_controller: WeightSyncController,
        activity_tracker: ActivityTrackerProxy
    ):
        """Set references to other modules."""
        self._dataloader = dataloader
        self._weight_sync_controller = weight_sync_controller
        self._activity_tracker = activity_tracker
```

**优点：**
- 避免 `__init__` 中的循环依赖
- 清晰声明依赖
- 兼容本地模块执行

<a id="defer-initialization"></a>

### 延迟初始化

不要在 `__init__` 中启动工作，应显式使用 `run()`：

```python
class MyWorker(NorthRLModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config
        self._running = False
        # Don't start thread here

    def run(self):
        """Start worker thread."""
        assert self._activity_tracker is not None
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop)
        self._thread.start()
```

<a id="activity-tracking"></a>

## 活动追踪

<a id="track-long-operations"></a>

### 追踪长时间操作

对所有长时间运行的操作使用活动追踪：

```python
def process_batch(self, batch):
    with self._activity_tracker.track("MyModule", "process_batch"):
        # Processing logic
        result = self._process(batch)
        return result
```

**优点：**
- 自动报告错误
- 检测系统静止状态
- 支持调试

<a id="manual-tracking"></a>

### 手动追踪

如需手动控制：

```python
def process_item(self, item):
    token = self._activity_tracker.start("MyModule", "process_item")
    try:
        result = self._process(item)
        return result
    finally:
        self._activity_tracker.end(token)
```

<a id="error-handling"></a>

## 错误处理

<a id="use-activity-tracker"></a>

### 使用活动追踪器

让活动追踪器处理错误：

```python
def step(self, task):
    with self._activity_tracker.track("Worker", "step", auto_report_errors=True):
        # Errors automatically reported
        return self._process(task)
```

<a id="manual-error-reporting"></a>

### 手动报告错误

如需特定的错误处理：

```python
def critical_operation(self):
    try:
        result = self._perform_operation()
        return result
    except Exception as e:
        self._activity_tracker.report_exception(
            "MyModule",
            "critical_operation",
            e,
            severity=ErrorSeverity.ERROR
        )
        raise
```

<a id="lifecycle-management"></a>

## 生命周期管理

<a id="implement-stop"></a>

### 实现 stop()

始终实现优雅关闭：

```python
class MyWorker(NorthRLModule):
    def stop(self):
        """Stop worker gracefully."""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("Worker thread did not stop cleanly")
```

<a id="cleanup-resources"></a>

### 清理资源

在 `stop()` 中进行清理：

```python
def stop(self):
    self._running = False

    # Stop threads
    if self._thread:
        self._thread.join()

    # Close files/connections
    if self._file_handle:
        self._file_handle.close()

    # Clear caches
    self._cache.clear()
```

<a id="thread-safety"></a>

## 线程安全

<a id="use-locks-for-shared-state"></a>

### 使用锁保护共享状态

用锁保护共享状态：

```python
class MyModule(NorthRLModule):
    def __init__(self, config):
        super().__init__()
        self._lock = threading.RLock()
        self._shared_state = {}

    def update_state(self, key, value):
        with self._lock:
            self._shared_state[key] = value

    def get_state(self, key):
        with self._lock:
            return self._shared_state.get(key)
```

<a id="avoid-deadlocks"></a>

### 避免死锁

使用多个锁时要谨慎：

```python
# Bad: Lock ordering can deadlock
def method_a(self):
    with self._lock_a:
        with self._lock_b:
            # Do work
            pass

def method_b(self):
    with self._lock_b:  # Reversed order!
        with self._lock_a:
            pass

# Good: Consistent lock ordering
def method_a(self):
    with self._lock_a:
        with self._lock_b:  # Always A then B
            pass

def method_b(self):
    with self._lock_a:  # Always A then B
        with self._lock_b:
            pass
```

<a id="testing"></a>

## 测试

<a id="unit-tests"></a>

### 单元测试

独立测试模块：

```python
def test_my_module():
    config = DictConfig({"batch_size": 32})
    module = MyModule(config)

    # Test initialization
    assert module._batch_size == 32

    # Test methods
    result = module.process(test_data)
    assert result is not None
```

<a id="integration-tests"></a>

### 集成测试

使用已初始化的 `trajectory_pool` fixture，配置 `key_list: [group_id]`、
`group_size: 2` 和 `groups_per_batch: 1`，并连接正常的 dataloader、
activity tracker 和 weight-sync 引用。分组 pool 直接请求一个完整组；
`get_trajectories()` 不接收调用者指定的行数：

```python
from northrl.northrl_types import Trajectory

def test_integration(trajectory_pool):
    for run_id in range(2):
        trajectory = Trajectory(
            tokens=[1, 2], loss_mask=[0, 1], reward=float(run_id),
            extra_fields={
                "model_tag": "actor",
                "group_id": "test-group",
                "run_id": run_id,
                "rollout_id": f"test-rollout-{run_id}",
                "expected_rollouts": 2,
            },
        )
        assert trajectory_pool.put_trajectory(trajectory) == "success"

    trajectories = trajectory_pool.get_trajectories(model_tag="actor")
    assert trajectories is not None
    assert len(trajectories) == 2
    assert {trajectory["run_id"] for trajectory in trajectories} == {0, 1}
```

<a id="logging"></a>

## 日志记录

<a id="use-module-name"></a>

### 使用模块名

在日志中包含模块名：

```python
import logging

logger = logging.getLogger(__name__)

class MyModule(NorthRLModule):
    def process(self, data):
        logger.info(f"[{self.get_module_name()}] Processing data")
```

<a id="log-at-appropriate-levels"></a>

### 使用适当的日志级别

```python
logger.debug("Detailed debug information")
logger.info("Important state changes")
logger.warning("Recoverable issues")
logger.error("Errors requiring attention")
```

<a id="performance"></a>

## 性能

<a id="avoid-busy-waiting"></a>

### 避免忙等

使用事件或 sleep：

```python
# Bad: Busy wait
while not self._ready:
    pass

# Good: Event
self._ready_event.wait()

# Good: Sleep
while not self._ready:
    time.sleep(0.1)
```

<a id="batch-operations"></a>

### 批量操作

尽可能批量处理：

```python
# Bad: Process one at a time
for item in items:
    result = process_single(item)
    store(result)

# Good: Batch processing
batch = []
for item in items:
    batch.append(item)
    if len(batch) >= 32:
        results = process_batch(batch)
        store_batch(results)
        batch = []
```

<a id="common-patterns"></a>

## 常用模式

<a id="singleton-module"></a>

### 单例模块

```python
class SingletonModule(NorthRLModule):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

<a id="factory-registration"></a>

### 工厂注册

```python
MODULE_REGISTRY = {
    NorthRLRole.ROLLOUT_WORKER: {
        "simple": SimpleRolloutWorker,
        "single_turn": SingleTurnRolloutWorker,
    }
}

def create_module(role: NorthRLRole, module_type: str, config):
    cls = MODULE_REGISTRY[role][module_type]
    return cls(config)
```

<a id="related-documentation"></a>

## 相关文档

- [核心架构](../02-core-architecture/overview.md) - 系统架构
- [自定义 Worker](../05-rollout-workers/custom-workers.md) - Worker 实现
