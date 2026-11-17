English | [简体中文](../../../zh/developer-guide/02-core-architecture/controller.md) | [Documentation index](../../index.md)

# NorthRL Controller



The `NorthRLController` is the main orchestrator for the NorthRL framework, responsible for initializing, coordinating, and monitoring all framework components.

## Overview

The controller serves as the central coordination point that:
- Initializes all framework modules
- Sets up module references and dependencies
- Manages the training lifecycle
- Monitors reported errors
- Coordinates validation cycles
- Handles checkpointing and weight synchronization

## Class Definition

```python
class NorthRLController:
    """
    NorthRL controller is the main process of the RL training framework.
    Responsible for starting the entire experiment, initializing various modules,
    and synchronizing configuration information.
    """
```

**Location**: `northrl/controller.py`

## Constructor

```python
def __init__(self, config: DictConfig)
```

**Parameters**:
- `config`: Hydra configuration containing all module settings

**Initialization Steps**:
1. Set up logging configuration
2. Seed random number generators
3. Initialize activity tracker and error reporter
4. Initialize external services (Tinker/Weaver if configured)
5. Initialize all framework modules
6. Set up module references and dependencies

**Example**:

```python
from omegaconf import DictConfig
from northrl import NorthRLController

config = DictConfig({...})
controller = NorthRLController(config)
```

## Module Registry

The controller maintains a registry of module types for each role. This enables dynamic module loading and supports both built-in and custom implementations.

### MODULE_REGISTRY Structure

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

SFT bypasses the public rollout-worker registry. With `algorithm.type: sft`,
the top-level `rollout` block is omitted and the Controller creates exactly one
internal pass-through `SFTRolloutWorker`. An explicitly provided `rollout`
block emits a `UserWarning` and is ignored in full.

### Dynamic Module Loading

For rollout workers, the controller supports loading custom implementations from recipe directories:

**Configuration Example**:


**Loading Process**:

1. Check if `rollout.custom_module_path` is specified
2. Resolve the path relative to the recipe YAML directory, or use an absolute path
3. Dynamically import the module using `importlib`
4. Extract `rollout.custom_class_name` (default `CustomRolloutWorker`) using `getattr`
5. Fall back to registered worker type if not specified

**Benefits**:
- Recipes are self-contained with their custom logic
- No need to modify core framework code for new tasks
- Easy to version control and share task-specific implementations
- Supports both absolute and relative path specifications

## Core Methods

### run()

Starts the training process by launching all components and entering the monitoring loop.

```python
def run() -> None
```

**Process**:

1. **Initialize Training Workers**
   ```python
   execute(self.trainer.initialize_workers)
   ```

2. **Load Initial Checkpoint**
   ```python
   self._load_initial_checkpoint()
   ```

3. **Optional Pre-Training Validation**
   ```python
   if self._config.validate.validate_before_train:
       self._start_validate(model_tag)
       # ... start workers ...
       self._end_validate(model_tag)
   ```

4. **Start All Components**
   ```python
   execute(self.trainer.run)
   for worker in self.rollout_workers:
       execute(worker.run)
   ```

5. **Enter Monitoring Loop**
   - Check for validation triggers
   - Check for training completion
   - Check for system exceptions
   - Handle graceful shutdown

**Monitoring Loop Configuration**:

```yaml
runtime:
  health:
    exception_policy: stop_on_error
    exception_check_interval_s: 1.0
    memory_log_interval_s: 10.0
```

**Example Usage**:

```python
controller = NorthRLController(config)
controller.run()  # Blocks until training completes or error occurs
```

### stop()

Gracefully stops all components and waits for activity completion.

```python
def _stop()
```

**Process**:

1. Signal all workers to stop
2. Wait for quiescence with timeout (default: 30 seconds)
3. Log remaining activities if timeout exceeded
4. Clean up resources

**Features**:
- Non-blocking stop signals
- Configurable timeout
- Activity logging for debugging

## Completion Checking

### _check_finish() -> bool

Determines if training should stop based on various criteria.

```python
def _check_finish(self) -> bool
```

**Criteria**:

1. **Maximum training steps reached**
   ```python
   if self.trainer.global_step >= self._config.trainer.total_train_steps:
       return True
   ```

2. **System quiescence** (all work complete)
   ```python
   if self.activity_tracker.is_quiescent():
       if self.dataloader.is_finished():
           if self.trajectory_pool.is_empty():
               return True
   ```

**Returns**: `True` if training should stop, `False` otherwise

### Progress reporting

The controller refreshes rollout progress on its main thread during training,
validation waits, and waits for the trainer and rollout workers to stop.
`ActivityTracker.update_progress()` collects and displays one update when due;
`runtime.progress.update_interval_s`, `log_every_n_steps`, and `use_tqdm` retain
control over refresh frequency and output. Health and memory checks keep their
own intervals. Progress output can pause during other blocking controller calls.

### Memory usage logging

Every `memory_log_interval_s` seconds, the monitoring loop logs a one-line
memory summary for the driver:

```
Driver memory usage: driver_rss=12.34GiB pod_mem=96.50/128.00GiB (75.4%)
```

- `driver_rss` — resident memory of the controller process alone.
- `pod_mem` — usage / limit of the whole process cgroup, read from cgroup v2 (`/sys/fs/cgroup/memory.{current,max}`) or v1
  (`/sys/fs/cgroup/memory/memory.{usage_in_bytes,limit_in_bytes}`). This is the
  figure the kernel OOM-killer trips on.

The intent is post-mortem diagnosis: an OOM-kill arrives as SIGKILL with no
Python traceback, so the last logged `pod_mem` is often the only on-disk
evidence that the driver cgroup hit its memory limit. The helper
(`northrl.utils.logging_utils.format_memory_usage`) never raises; if psutil or the
cgroup files are unavailable it degrades to a partial/marker string.

### _check_module_exceptions() -> bool

Checks for critical errors in the system.

```python
def _check_module_exceptions(self) -> bool
```

**Returns**: `True` if system is healthy, `False` if critical errors detected

**Process**:

1. Get error health status from activity tracker
2. Check error policy configuration
3. Determine if errors require system shutdown

**Error Policies**:

- `"stop_on_error"`: Stop on any error
- `"stop_on_critical"`: Stop only on critical errors
- `"continue"`: Log errors but continue running

**Configuration**:

```yaml
runtime_monitor:
  exception_handling:
    enabled: true
    check_interval: 1.0
    policy: "stop_on_error"
```

## Checkpoint Management

### _load_initial_checkpoint()

Loads initial checkpoint or prepares for training from scratch.

```python
def _load_initial_checkpoint()
```

**Process**:

1. Check if resuming from checkpoint
2. If resuming, call `_load_resume_checkpoint()`
3. If starting fresh, create sync weight buffer
4. Perform initial weight sync to inference service

### _load_resume_checkpoint()

Loads checkpoint based on resume configuration.

```python
def _load_resume_checkpoint()
```

**Resume Modes**:

1. **Disable**: Don't resume
   ```yaml
   resume:
     mode: "disable"
   ```

2. **Auto**: Automatically find latest checkpoint
   ```yaml
   resume:
     mode: "auto"
   ```

3. **From Path**: Resume from specific checkpoint
   ```yaml
   resume:
     mode: "from_path"
     resume_path: "/path/to/checkpoint"
   ```

**Process**:

1. Determine checkpoint path based on mode
2. Call trainer's `resume_checkpoint()` method
3. Update training step in activity tracker

### _find_latest_checkpoint(checkpoint_folder: str) -> str | None

Finds the latest checkpoint in the given folder.

```python
def _find_latest_checkpoint(self, checkpoint_folder: str) -> str | None
```

**Parameters**:
- `checkpoint_folder`: Path to folder containing checkpoints

**Returns**: Path to latest checkpoint, or `None` if none found

**Process**:

1. List all `global_step_*` directories
2. Parse step numbers from directory names
3. Return path with highest step number

**Expected Directory Structure**:
```
checkpoint_folder/
├── global_step_100/
├── global_step_200/
├── global_step_300/
└── ...
```

## Validation Coordination

### _run_validate(model_tag: ModelTag)

Runs a complete validation cycle after a weight sync event.

```python
def _run_validate(self, model_tag: ModelTag)
```

**Process**:

1. Call `_start_validate(model_tag)` to begin validation
2. Wait for validation to complete (validator.is_complete())
3. Call `_end_validate(model_tag)` to finish validation

**Blocking**: This method blocks until validation completes

### _start_validate(model_tag: ModelTag)

Starts validation by switching rollout workers to validation mode.

```python
def _start_validate(self, model_tag: ModelTag)
```

**Process**:

1. Switch rollout workers to validation mode
   ```python
   for worker in self.rollout_workers:
       execute(worker.begin_validate)
   ```

2. Workers will now:
   - Use validation dataloader
   - Submit trajectories to validator instead of trajectory pool

### _end_validate(model_tag: ModelTag)

Ends validation and logs results.

```python
def _end_validate(self, model_tag: ModelTag)
```

**Process**:

1. Compute and log validation metrics
   ```python
   metrics = execute(self.validator.compute_and_log_metrics)
   ```

2. Switch workers back to training mode
   ```python
   for worker in self.rollout_workers:
       execute(worker.end_validate)
   ```

3. Clear validation trajectories
   ```python
   execute(self.validator.clear)
   ```

4. Notify weight sync controller
   ```python
   execute(self.weight_sync_controller.end_validate, model_tag)
   ```

## Module Initialization

The controller initializes modules in a specific order to respect dependencies:

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

## Local Execution

The controller, data loader, trajectory pool, and rollout workers run in the local driver process. Weaver performs model training and sampling remotely, and E2B tasks use their configured sandbox service. Rollout concurrency is controlled by `rollout.num_workers`; size it to the local driver and service capacity.

Start through `python -m northrl.main`, which validates the recipe, creates the controller, and manages retries and cleanup. See [Run Locally](../01-getting-started/quick-start.md#run-locally) for the complete command and required environment.

## Best Practices

### Configuration

1. **Use Hydra composition** for common settings
2. **Validate configurations** before running
3. **Document custom modules** in recipes
4. **Version control** configurations

### Error Handling

1. **Configure appropriate error policies** for your scenario
2. **Monitor reported errors** in production
3. **Set reasonable timeouts** based on workload
4. **Review error logs** regularly

### Performance

1. **Tune local rollout concurrency** to the driver and remote-service limits
2. **Configure the memory-log interval** appropriately
3. **Monitor activity tracker** for bottlenecks
4. **Optimize checkpoint frequency** vs. recovery time

### Validation

1. **Enable pre-training validation** to check setup
2. **Configure validation frequency** based on training speed
3. **Monitor validation metrics** for model quality
4. **Balance validation overhead** with evaluation needs

<a id="example-complete-workflow"></a>

## Complete Workflow

Use the maintained [local quick start](../01-getting-started/quick-start.md) for installation, service credentials, recipe selection, and logs. The public entry point owns configuration validation and controller lifecycle; task code only needs the supported extension hooks.

## Next Steps

- Learn about [Data Types](data-types.md) used by the controller
- Understand [Activity Tracking](activity-tracking.md) for monitoring
- Explore [Module Development](../12-best-practices/module-development.md)
