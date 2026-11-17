English | [简体中文](../../../zh/developer-guide/05-rollout-workers/base-rollout-worker.md) | [Documentation index](../../index.md)

# Base Rollout Worker

**Terminology:** One rollout produces one complete `Trajectory`. Ordinary trajectories store `tokens` and `loss_mask` directly (`segments=None`); results that need separate training sequences use `segments` as a list of dictionaries. Multiple interaction turns do not by themselves require segments. Workers, the pool and trainer preparation exchange complete trajectories. See [trajectory semantics](../04-trajectory-pool/trajectory-semantics.md).

The `BaseRolloutWorker` is the abstract base class for all rollout workers in NorthRL. It provides the core infrastructure for task processing, LLM interaction, and trajectory generation.

## Class Definition

**Location:** `northrl/rollout_worker/base_rollout_worker.py`

```python
from northrl.rollout_worker import BaseRolloutWorker

class BaseRolloutWorker(NorthRLModule, ABC):
    """Base class for rollout workers."""
```

## Constructor

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

The constructor:
1. Initializes threading components
2. Sets up stop event for graceful shutdown
3. Prepares module reference placeholders (set later)
4. Does NOT initialize inference client (deferred to `init_inference_service_client()`)

## Abstract Method

### rollout()

The core method that derived classes must implement:

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

**Implementation guidelines:**
1. Extract required fields from `task`
2. Perform task-specific processing (LLM calls, agent execution, etc.)
3. Keep ordinary token fields directly on the trajectory; segmented results use internal dictionaries
4. Construct `Trajectory(tokens=tokens, loss_mask=loss_mask, reward=final_reward)` for an ordinary result, or `Trajectory(segments=segments, reward=final_reward)` when separate sequences are needed
5. Call `self._put_trajectory(trajectory)` once with the complete result

For training tasks, the main loop uses `execute_rollout`: it stamps the execution
identity, holds the submitted object until `rollout` returns, then publishes it
atomically. Returning without an object records an empty skipped outcome, or a
failed outcome if the return value is `"fail"`. Returning `"fail"` after staging
an object marks that object failed. A scored success with no training sequences
explicitly submits `Trajectory(segments=[], reward=final_reward)`.

An exception escaping `rollout` causes the training wrapper to submit an empty
failed trajectory and then re-raise; it does not retry the task. Validation tasks
bypass this wrapper logic: `rollout` runs directly, and `_put_trajectory` sends
`is_val=True` objects to the validator immediately. No automatic skipped/failed
validation object is created. See the [worker overview](overview.md#2-rollout-execution).

## Setup Methods

### set_module_references()

Called by controller to establish inter-module connections:

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

**When called:** During controller initialization, before `run()`

### init_inference_service_client()

Initializes the LLM inference client:

```python
def init_inference_service_client(self, service_holder=None):
    """
    Initialize inference service client.

    Args:
        service_holder: Shared service holder (Tinker/Weaver) if applicable
    """
```

**Features:**
- Only initializes if `need_llm_inference=True`
- Supports multiple backends (vLLM, Tinker, Weaver, etc.)
- Sets up weight sync coordination

**When called:** During controller initialization, after module references are set

### set_activity_tracker()

Sets the activity monitoring proxy:

```python
def set_activity_tracker(self, tracker: ActivityTrackerProxy):
    """Set activity tracker for monitoring."""
```

Inherited from `NorthRLModule`.

## Lifecycle Methods

### run()

Schedules the worker coroutine on the shared rollout event loop:

```python
def run():
    """Schedule worker execution on the rollout event loop."""
```

**Process:**
1. Validates that module references are set
2. Clears stop event
3. Schedules `_main_loop()` and retains its future

**Preconditions:**
- `set_module_references()` must have been called
- `set_activity_tracker()` must have been called
- `init_inference_service_client()` must have been called (if needed)

### stop()

Gracefully stops the worker:

```python
def stop():
    """Request stop, wait for the worker future, and cancel on timeout."""
```

**Process:**
1. Sets stop event
2. Waits up to 30 seconds for the coroutine to finish, then requests cancellation

### _main_loop()

Internal coroutine scheduled on the shared rollout event loop:

```python
async def _main_loop(self) -> None:
    """Main worker loop, scheduled as an asyncio task."""
```

**Process with a notification-capable loader (simplified):**
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

**Key features:**
- Tracks activity for monitoring
- Claims the next task after rollout returns, unless stop was requested
- Subscribes to loader readiness before claiming tasks and waits outside activity
  tracking when no task is available. Sequential loaders notify after batch release
  or reset, eliminating empty polling and its CPU/GIL contention with training.
- Clears the notification before checking for work. A release before the check is
  visible in loader state; a later notification stays set until the worker waits.
- Stop and validation switches also wake idle workers through the rollout event
  loop's thread-safe scheduler. Each run owns a fresh event and unsubscribes on exit.
- Loaders without `subscribe_ready` support retain a one-second polling fallback;
  stop and validation switches still interrupt that wait. Pool-capacity throttling
  uses its existing separate wait.

## Data Flow Methods

### _get_rollout_task()

Fetches the next task from appropriate data loader:

```python
def _get_rollout_task(self) -> dict[str, Any] | None:
    """
    Get next rollout task from data loader.

    Returns:
        Task dictionary or None if no tasks available
    """
```

**Behavior:**
- Uses validation data loader if `_is_running_validate=True`
- Uses training data loader otherwise
- Returns `None` when no task is available; the main loop handles readiness waiting
- Non-blocking operation

### _put_trajectory()

Submit one complete trajectory per rollout:

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

**Routing:**
- **Validation** (`trajectory.is_val`): Routes to the validator
- **Training**: Holds the complete object until the execution wrapper publishes it to the pool

A second call during a training `execute_rollout` raises an error; validation
submissions go directly to the validator and do not use this guard. `_put_trajectory` validates
every training sequence's length; truncation and final reward adjustment must happen first.
The ordinary `Trajectory(tokens=..., loss_mask=..., reward=...)` constructor
used below stores token fields directly, with `segments=None`.

## Validation Mode Methods

### begin_validate()

Switches worker to validation mode:

```python
def begin_validate():
    """Switch to validation mode."""
```

**Effects:**
- `_is_running_validate` set to `True`
- Subsequent tasks from validation data loader
- Trajectories routed to validator

**When called:** By controller at start of validation cycle

### end_validate()

Switches worker back to training mode:

```python
def end_validate():
    """Switch back to training mode."""
```

**Effects:**
- `_is_running_validate` set to `False`
- Subsequent tasks from training data loader
- Trajectories routed to trajectory pool

**When called:** By controller at end of validation cycle

## Inference Client Interface

Workers access LLM through `_inference_client`:

### completion()

Single-turn completion:

```python
result = await self._inference_client.completion(
    prompt="Your prompt here"
)
```

**Returns:**
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

The remote client returns the OpenAI-style envelope above. Token fields are in
`northrl_train`, and completion text is in `choices[0]["text"]`; mock or legacy clients
may return flat fields. Sampling settings come from `rollout.sampling` (or the
validation sampling configuration). Passing `temperature`/`max_tokens` as extra
keyword arguments to `completion()` does not override those settings.

### generate()

Chat-style generation with messages:

```python
result = await self._inference_client.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    tools=[...],  # Optional tool definitions
)
```

**Returns:**
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

### Weight Sync Coordination

The inference client automatically handles weight sync:
- Blocks during weight synchronization if `freeze_for_weight_sync=True`
- This is transparent to the worker implementation

## Example Implementation: SimpleRolloutWorker

The following example shows the ordinary trajectory shape and one-submission contract:

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

## Configuration Example


## Common Patterns

### Pattern 1: Basic Completion

```python
class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        prompt = task["prompt"]
        result = await self._inference_client.completion(prompt)

        trajectory = self._create_trajectory(result, task)
        return self._put_trajectory(trajectory)
```

### Pattern 2: Multi-step Processing

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

### Pattern 3: Error Handling

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

## Threading Considerations

**Thread safety:**
- Each worker runs in its own thread
- Module references are read-only after initialization
- Activity tracker handles thread-safe tracking
- Inference client handles concurrent requests

**Best practices:**
- Don't modify shared state in `rollout()`
- Don't block indefinitely in `rollout()`
- Use activity tracker for long operations

## Performance Tips

1. **Keep rollout() fast**: Minimize non-essential work
2. **Batch when possible**: Some inference services support batching
3. **Use async patterns**: For I/O-heavy operations

## Debugging

### Logging

```python
import logging
logger = logging.getLogger(__name__)

class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        logger.debug(f"Processing task: {task.get('task_id')}")
        # ... processing
        logger.info(f"Generated trajectory with reward: {reward}")
```

### Activity Tracking

The main loop automatically tracks activity:

```python
with self._activity_tracker.track(self._module_name, "rollout"):
    result = await self.execute_rollout(task)
```

Check worker activity:

```python
# In controller or monitoring
if activity_tracker.is_rollout_worker_quiescent():
    print("All workers idle")
```

## Common Issues

### Issue: Inference client not initialized

**Symptoms:** `AttributeError: 'NoneType' object has no attribute 'completion'`

**Solution:** Ensure `need_llm_inference=true` in config

### Issue: Tasks not appearing

**Symptoms:** Worker idle, but data loader has data

**Solution:** Check data loader `can_return_item()` and `is_finished()` methods

## Next Steps

- [Custom Workers](custom-workers.md) - Creating task-specific workers
- [Evaluators](evaluators.md) - Implementing evaluation logic

## Related Documentation

- [Data Types](../02-core-architecture/data-types.md) - Trajectory structure
- [Activity Tracking](../02-core-architecture/activity-tracking.md) - Monitoring details
- [Weight Synchronization](../08-features/weight-synchronization.md) - Sync coordination
