English | [简体中文](../../../zh/developer-guide/12-best-practices/module-development.md) | [Documentation index](../../index.md)

# Module Development

Best practices for developing NorthRL modules.

## Module Base Class

All NorthRL components should inherit from `NorthRLModule`:

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

**Benefits:**
- Consistent local module interfaces
- Activity tracking integration
- Module naming support

## Configuration Handling

### Accept DictConfig

Always accept `DictConfig` for configuration:

```python
def __init__(self, config: DictConfig):
    self._config = config

    # Access with defaults
    self._batch_size = config.get("batch_size", 32)

    # Environment variable interpolation
    self._data_path = config.data_files[0]  # Already interpolated
```

### Validate Configuration

Validate critical configuration early:

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

## Module References

### Set References Pattern

Use `set_module_references()` for inter-module dependencies:

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

**Benefits:**
- Avoids circular dependencies in `__init__`
- Clear dependency declaration
- Compatible with local module execution

### Defer Initialization

Don't start work in `__init__`, use explicit `run()`:

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

## Activity Tracking

### Track Long Operations

Use activity tracking for all long-running operations:

```python
def process_batch(self, batch):
    with self._activity_tracker.track("MyModule", "process_batch"):
        # Processing logic
        result = self._process(batch)
        return result
```

**Benefits:**
- Automatic error reporting
- System quiescence detection
- Debugging support

### Manual Tracking

For manual control:

```python
def process_item(self, item):
    token = self._activity_tracker.start("MyModule", "process_item")
    try:
        result = self._process(item)
        return result
    finally:
        self._activity_tracker.end(token)
```

## Error Handling

### Use Activity Tracker

Let activity tracker handle errors:

```python
def step(self, task):
    with self._activity_tracker.track("Worker", "step", auto_report_errors=True):
        # Errors automatically reported
        return self._process(task)
```

### Manual Error Reporting

For specific error handling:

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

## Lifecycle Management

### Implement stop()

Always implement graceful shutdown:

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

### Cleanup Resources

Clean up in `stop()`:

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

## Thread Safety

### Use Locks for Shared State

Protect shared state with locks:

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

### Avoid Deadlocks

Be careful with multiple locks:

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

## Testing

### Unit Tests

Test modules in isolation:

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

### Integration Tests

Use an initialized `trajectory_pool` fixture with `key_list: [group_id]`,
`group_size: 2`, and `groups_per_batch: 1`. The fixture must attach the normal
dataloader, activity tracker and weight-sync references. The grouped pool directly
requests one complete group; `get_trajectories()` does not accept a requested row count:

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

## Logging

### Use Module Name

Include module name in logs:

```python
import logging

logger = logging.getLogger(__name__)

class MyModule(NorthRLModule):
    def process(self, data):
        logger.info(f"[{self.get_module_name()}] Processing data")
```

### Log at Appropriate Levels

```python
logger.debug("Detailed debug information")
logger.info("Important state changes")
logger.warning("Recoverable issues")
logger.error("Errors requiring attention")
```

## Performance

### Avoid Busy Waiting

Use events or sleep:

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

### Batch Operations

Process in batches when possible:

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

## Common Patterns

### Singleton Module

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

### Factory Registration

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

## Related Documentation

- [Core Architecture](../02-core-architecture/overview.md) - System architecture
- [Custom Workers](../05-rollout-workers/custom-workers.md) - Worker implementation
