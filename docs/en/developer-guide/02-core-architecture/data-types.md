English | [简体中文](../../../zh/developer-guide/02-core-architecture/data-types.md) | [Documentation index](../../index.md)

# Core Data Types

**Terminology:** One rollout produces one complete `Trajectory`. Ordinary trajectories store tokens and loss masks directly; only multi-sequence trajectories use nested `segments` dictionaries. See [trajectory semantics](../04-trajectory-pool/trajectory-semantics.md) for tasks, rollout attempts, groups and training records.



This document describes the fundamental data types used throughout the NorthRL framework.

## Overview

NorthRL defines several core data types in `northrl/northrl_types.py` that form the foundation of the framework's data flow and component interaction.

## Type Aliases

### ModelTag

```python
ModelTag = str  # Type alias for model identification
```

**Purpose**: Identifies different models within the system

**Usage**:
```python
model_tag = "actor"  # Current rollout and training model
```

**Common Values**:
- `"actor"`: The model tag used by the current rollout, pool and trainer path
- `"default"`: A fallback in generic Batch/pool utilities when no tag is supplied

Keep the request, trajectory and target pool tags consistent. A KL reference client
does not automatically create a separate `"reference"` trajectory-pool tag.

## Data Classes

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

A `Trajectory` stores one complete execution and its final outcome. Its ordinary
form stores `tokens` and `loss_mask` directly, with `segments=None`; it creates no
nested wrapper. Token metadata such as `old_log_probs`, sampling masks and
correction weights remains in `extra_fields`, as in the original single-turn API.

Segmented executions use `segments`, a list of flat dictionaries with
`tokens`, `loss_mask` and sequence metadata. The complete trajectory then has `tokens=None`
and `loss_mask=None`. Supplying both forms to the constructor raises `ValueError`.
Reward, status, validation mode, execution identity, score and advantage belong to
the complete trajectory. A scored result with no training sequence uses
`Trajectory(segments=[], reward=final_reward)`.

The dictionary interface supports `obj[key]`, assignment, `get`, `keys`, `items`
and membership. `group_id` identifies the sampling occurrence, `run_id` identifies
an expected attempt slot, and `rollout_id` identifies one execution. `task_id`
identifies the source task: drawing that task again creates a new `group_id`, even
though `task_id` stays the same. `rollout_repeat_n=R` creates R attempts within one
group, not R separate groups.

Multiple model calls do not necessarily require multiple segments: a recorder can
append calls with a continuous token prefix to one sequence. E2B always returns a
segment list, which can contain just one entry. Its remote `RemoteSegment` type is
a serialization contract; local nested records remain dictionaries owned by one
complete `Trajectory`.

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

The ordinary constructor stores tokens and masks as direct fields, including
after restoration. There is no public sequence class or flattening API.
Internally, `iter_trajectory_sequences` inspects original sequence records
without changing them. At the training boundary,
`iter_training_sequences` skips failed/skipped groups, yields an ordinary
Trajectory unchanged, and yields the original dictionaries for a multi-sequence
trajectory. Trajectory identity and advantage override stale child values; other
metadata fills missing keys without overwriting per-sequence logprobs or masks.
Reward and terminal fields are not copied onto the dictionaries.

### Batch

```python
@dataclass
class Batch:
    values: dict[str, Any]      # Tensor or data arrays
    metadata: dict[str, Any]    # Batch metadata
```

**Purpose**: A general container of row-aligned fields and batch metadata. It does not impose a segment, trajectory or Datum unit on arbitrary rows

**Structure**:
- `values`: Dictionary mapping field names to batched data
  - Tensors: Fields whose first dimension is the row dimension
  - Lists: Non-tensor data (strings, dicts, etc.)
- `metadata`: Batch-level information
  - `batch_size`: Current number of rows, including rows added by padding
  - `model_tag`: Model tag for this batch
  - `temperature`: Sampling temperature (if consistent)

**Key Methods**:

#### `__len__() -> int`

Returns the batch size:

```python
batch_size = len(batch)
```

#### `copy() -> Batch`

Creates new shallow `values` and `metadata` dictionaries; nested values remain shared. The current method does not preserve `pad_size`, so unpad a padded batch before copying it:

```python
batch_copy = batch.copy()
```

#### `to_dict() -> dict[str, Any]`

Merges values and metadata (metadata overwrites values):

```python
combined = batch.to_dict()
# Access both values and metadata in one dict
```

#### `from_trajectories(trajectories, model_tag=None) -> Batch`

Stores one complete Trajectory per row. Ordinary direct sequence fields and
multi-sequence nested dictionaries are both preserved; sequence counts do not
change the number of trajectory rows. When rows share the same metadata keys, their
fields keep the historical flat column layout. When keys differ, the batch stores
fixed trajectory columns plus an `extra_fields` dictionary per row. This supports
ordinary logprobs on the complete trajectory alongside multi-sequence logprobs inside nested
dictionaries, without adding missing metadata keys to other rows.

#### `to_trajectories() -> list[Trajectory]`

Restores each row as one complete Trajectory, including reward, status and any
nested sequences. Call `unpad()` before restoration; padding rows must not become
invented execution results. A generic Batch without the required trajectory
fields cannot use this conversion.

The current remote trainer does not construct Batch: it receives complete
trajectories, prepares their sequence records, and converts those to Datums.
These Batch conversion methods are separate utilities, not an extra stage in
that training path.

#### `pad_to_world_size(world_size: int) -> Batch`

Pads batch to be divisible by world_size (for distributed training):

```python
batch = batch.pad_to_world_size(world_size=4)
# Batch size becomes multiple of 4
```

#### `remove_redundant_left_padding(data, pad_token_id, ...)`

Static method to remove common left padding:

```python
batch = Batch.remove_redundant_left_padding(
    data=batch,
    pad_token_id=0,
    fields=["input_ids", "attention_mask"],
    anchor_field="input_ids",
    max_strip_threshold=512
)
```

**Parameters**:
- `pad_token_id`: ID of padding token
- `fields`: Which fields to strip (None = all 2D tensors)
- `anchor_field`: Field to determine padding length
- `max_strip_threshold`: Maximum tokens to strip (-1 = no limit)

#### `remove_redundant_right_padding(data, pad_token_id, ...)`

Static method to remove common right padding:

```python
batch = Batch.remove_redundant_right_padding(
    data=batch,
    pad_token_id=0,
    fields=["input_ids", "attention_mask"],
    anchor_field="input_ids"
)
```

#### `to_nextrainer_batch() -> dict[str, Any]`

Converts batch to NexTrainer format:

```python
nextrainer_batch = batch.to_nextrainer_batch()
# Returns: {
#     "batch": {tensor_fields...},
#     "non_tensor_batch": {non_tensor_fields...},
#     "meta_info": {...}
# }
```

**Example Usage**:

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

**Purpose**: Identifies local components so the controller can select their
implementation, initialize them, and name them for activity tracking.

**Usage**:

```python
from northrl.northrl_types import NorthRLRole

role = NorthRLRole.ROLLOUT_WORKER
module_name = f"{role.value}-0"  # rollout_worker-0
```

The controller creates module instances in the local driver and connects them
through direct references. See [Module Initialization](controller.md#module-initialization).

## Evaluation System Types

Single-turn reward functions receive `(task, result)` dictionaries. The
`northrl.rewards.evaluate_reward` dispatcher normalizes their return value to a
dictionary with a finite `reward`; optional fields include boolean `correct`,
scalar `metrics`, and text `reason`. See [Evaluators](../05-rollout-workers/evaluators.md).

### Remote Agent and Evaluation Results

E2B task hooks use the contracts in `northrl.remote_rollout.contracts`:

- `RolloutInput`: A `rollout_id`, generic JSON `task`, and provenance `metadata`.
- `AgentResult`: An execution `exit_code`, optional `final_answer`, and `artifacts`
  mapping logical names to paths. The runtime records training tokens separately.
- `EvaluationResult`: The final `reward`, scalar `metrics`, and evaluation `status`.

```python
from northrl.remote_rollout.contracts import AgentResult, EvaluationResult

agent_result = AgentResult(final_answer="42")
result = EvaluationResult(
    reward=1.0,
    metrics={"exact_match": 1.0, "answer_length": 2.0},
)
```

`reward` may be any finite number. `status="success"` means evaluation completed,
not necessarily that the answer is correct; `status="task_failed"` requires a
zero reward. Infrastructure failures raise `RolloutEnvironmentError` rather than
return a fabricated task score. See [E2B task integration](../05-rollout-workers/e2b-task-contract.md)
for the `run_agent` and `evaluate` hook signatures and staged lifecycle.

## Type Conversions

<a id="trajectory--batch"></a>

### Trajectory ↔ Batch

```python
# One complete trajectory per row, preserving nested sequence boundaries.
batch = Batch.from_trajectories([trajectory1, trajectory2])
trajectories = batch.to_trajectories()
```

### Batch ↔ Dictionary

```python
# Batch to dict
combined_dict = batch.to_dict()

# Dict to Batch (manual)
batch = Batch(
    values={"tokens": [...], "loss_mask": [...]},
    metadata={"batch_size": 32}
)
```

### Batch → NexTrainer Format

```python
nextrainer_batch = batch.to_nextrainer_batch()

# Result structure:
# {
#     "batch": {tensor_fields},
#     "non_tensor_batch": {non_tensor_fields},
#     "meta_info": {metadata}
# }
```

## Common Patterns

<a id="creating-a-trajectory-in-rollout-worker"></a>

### Creating a Complete Trajectory in Rollout Worker

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

### Processing a Batch in Trainer

This utility example removes common token padding and reports the current row
count. It is not the remote trainer's execution path. GRPO requires complete
trajectory identity and final reward before sequence-level preparation.

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

### Grouping Complete Trajectories

```python
# Index complete rollout results by their comparison group.
grouped_trajectories = {}
for trajectory in trajectories:
    group_id = trajectory["group_id"]
    grouped_trajectories.setdefault(group_id, []).append(trajectory)
```

This indexes complete rollout objects, including zero-segment results. The pool waits for all distinct rollout identities in the expected slots before delivering a group. GRPO reads one final reward per complete trajectory; segment count has no role in the baseline.

## Best Practices

### For Trajectories and Sequence Records

1. **Always include model_tag** for multi-model scenarios
2. **Use consistent extra_fields** across your task
3. **Include ground_truth** when available
4. **Store metrics as floats** for aggregation
5. **Keep group_id and rollout_id on the complete trajectory**; ordinary token metadata stays in extra_fields, while multi-sequence token metadata belongs to each nested dictionary

### For Batches

1. **Validate batch_size** in metadata matches actual size
2. **Use padding removal** to optimize memory and computation
3. **Check tensor shapes** before operations
4. **Unpad before restoring trajectories**; keep nested sequences attached to their trajectory row
5. **Use to_dict()** carefully (loses structure)

### For Type Safety

1. **Use type hints** with these types in your code
2. **Validate fields** before accessing from extra_fields
3. **Handle missing fields** gracefully with `.get()`
4. **Document custom extra_fields** in your implementation
5. **Test conversions** (especially Trajectory ↔ Batch)

## Next Steps

- Learn how [Controller](controller.md) manages these types
- Understand [Rollout Workers](../05-rollout-workers/overview.md) generate trajectories
- See [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) batching strategies
