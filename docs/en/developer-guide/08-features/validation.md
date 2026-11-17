English | [简体中文](../../../zh/developer-guide/08-features/validation.md) | [Documentation index](../../index.md)

# Validation

## Overview

`Validator` collects one complete `Trajectory` per validation rollout and aggregates trajectory-level scores. Internal sequence count does not give an execution extra weight. Validation uses its own collector and does not require training comparison groups.

**Location**: `northrl/validator.py`

## Key Concepts

### Validation Cycle

```
Weight Sync → Reset Validate DataLoader → Rollout Workers Generate → Collect Trajectories → Compute Metrics
```

Set `validate.enabled: true` and provide validation datasets. Periodic validation
is configured through `validate.schedule.every_n_steps`; WeightSyncController
uses this interval when publishing model versions. The schedule fragment is:

```yaml
validate:
  schedule:
    before_train: false
    every_n_steps: 5  # Validate every 5 published model versions
```

### Completion Criteria

Validation is complete when:
1. Validate dataloader is finished (all data consumed)
2. All rollout workers are quiescent (no active work)

## Core Methods

### Constructor

```python
def __init__(self, config: DictConfig)
```

Initializes the validator with configuration.

**Parameters**:
- `config`: Configuration dictionary

### set_module_references

```python
def set_module_references(self, validate_dataloader: BaseDataLoader)
```

Sets reference to the validation dataloader.

**Parameters**:
- `validate_dataloader`: DataLoader providing validation data

### put_trajectory

```python
def put_trajectory(self, trajectory: Trajectory) -> str
```

Store validation trajectory from rollout workers.

**Parameters**:
- `trajectory`: One complete `Trajectory`, with validation scores in `extra_fields["score"]`

**Returns**: `"success"` (matches TrajectoryPool interface)

### is_complete

```python
def is_complete(self) -> bool
```

Check if all validation trajectories have been collected.

**Returns**: `True` if validation complete

**Logic**:
```python
return (
    execute(self._validate_dataloader.is_finished)
    and self._activity_tracker.is_rollout_worker_quiescent()
)
```

### compute_and_log_metrics

```python
def compute_and_log_metrics(self) -> dict[str, float]
```

Compute validation metrics and log via activity tracker.

**Returns**: Dictionary of computed metrics with `"val/"` prefix

**Process**:
1. Collect score values from all trajectories
2. Group by score key and data source
3. Compute mean for each key
4. Add `"val/"` prefix to metric names
5. Log to experiment tracking (e.g., wandb)
6. Clear collected trajectories

**Metric Keys**:
- `val/{score_key}`: Mean across complete trajectories with a numeric value for that score key
- `val/{data_source}_{score_key}`: The same mean within each data source
- `val/trajectory_count`: Number of complete validation trajectories received
- `val/num_samples`: Compatibility name for that same trajectory count
- `val/sequence_count`: Number of their token sequences; an ordinary result contributes one, a segmented result contributes its internal sequence count, and an empty result contributes zero
- `val/segment_count`: Compatibility name for `val/sequence_count`

## Trajectory Score Structure

Validation submits complete objects. Put the `score` dictionary and `data_source` on the complete trajectory, including for results with several internal sequences:

```python
from northrl.northrl_types import Trajectory

trajectory = Trajectory(
    tokens=[1, 2, 3],
    loss_mask=[0, 1, 1],
    reward=0.85,
    is_val=True,
    extra_fields={
        "data_source": "dataset_name",
        "score": {"reward": 0.85, "accuracy": 1.0, "custom_metric": 0.9},
    },
)
validator.put_trajectory(trajectory)
```

## Validation Workflow

### 1. Controller Triggers Validation

When `rollout_model_version % validate_freq == 0`:

```python
# In WeightSyncController.train_worker_notify_weight_update
if self._validate_freq > 0 and rollout_model_version % self._validate_freq == 0:
    self._waiting_for_validation = True
    # Trainer waits; sync and batch-async leave the next data batch unreleased.
```

### 2. Controller Runs Validation

Before claiming another training batch, the trainer checks whether validation is
waiting. It waits outside activity tracking and is reported as idle during this
wait. This also applies to `fully-async`, where training batches may already be
queued when validation starts.

```python
# In NorthRLController._run_validate
def _run_validate(self, model_tag):
    self._start_validate(model_tag)
    self._end_validate(model_tag)
```

`_start_validate()` resets validation input before switching workers to validation.
`_end_validate()` waits for completion while refreshing progress and checking
worker errors, then computes metrics and switches workers back to training.
It also calls `WeightSyncController.end_validate()` to clear the validation wait.
In `fully-async`, the dataloader has `keep_batch_order: false` and fetches the
next batch automatically from `get_next_item()`; `release_next_batch()` is a
no-op, and `end_validate()` does not call it. The trainer's validation wait
prevents it from consuming queued training batches. When training should continue,
sync and batch-async call `_release_next_batch()` to make the next data batch
available to workers. The trajectory pool has no weight-sync write gate and
continues accepting completed results.
After the final training round, validation completion does not release another batch.

### 3. Rollout Workers Process Validation Data

Rollout workers generate complete trajectories from validation input. The custom worker sets `is_val` from the task and calls `_put_trajectory`; that method routes the object to the validator:

```python
# In a custom worker's async rollout(task), after generating/evaluating:
trajectory = Trajectory(
    tokens=tokens,
    loss_mask=loss_mask,
    reward=final_reward,
    is_val=task.get("is_val", False),
    extra_fields={"score": scores, "data_source": task.get("data_source", "")},
)
return self._put_trajectory(trajectory)
```

### 4. Validator Computes Metrics

After all trajectories collected:

```python
metrics = execute(self._validator.compute_and_log_metrics)
# Example output:
# {
#   "val/reward": 0.85,
#   "val/accuracy": 0.92,
#   "val/dataset1_reward": 0.83,
#   "val/dataset2_reward": 0.87,
#   "val/trajectory_count": 100,
#   "val/num_samples": 100,
#   "val/sequence_count": 140,
#   "val/segment_count": 140
# }
```

## Configuration

### Validator Config

Controller creates the built-in `Validator` from the root configuration.
Configure validation under `validate`; there is no `validator.type` selector.
The examples below are fragments to merge into an existing recipe.

### Validation DataLoader

```yaml
validate:
  enabled: true
  data:
    validation_repeat_n: 1
    datasets:
      dev:
        files: [/data/validation.parquet]
        batch_size: 1
```

### Validation Frequency

```yaml
validate:
  schedule:
    before_train: false
    every_n_steps: 5  # 0 disables periodic validation only
```

## Validation vs Training Dataloaders

Controller derives training input from `data.train` and validation input from
`validate.data.datasets`. `_derive_validate_dataloader_config()` maps each named
dataset's `files` and `batch_size` into an internal loader configuration. It
inherits `prompt_key` from `data.train`, the actor tokenizer, and
`data.max_prompt_length`, while fixing `shuffle`, `drop_last` and
`filter_prompts` to `false`.

Controller instantiates `MultiDatasetValidationDataLoader`, which creates a
`TorchDataLoader(..., is_validate=True)` for each dataset. Each validation cycle
is a finite pass; resetting starts that pass again. Do not instantiate the
abstract `BaseDataLoader` or add an `epochs` field under `validate.data`. Use
`validation_repeat_n` to control attempts per validation prompt. Training input
uses `TorchDataLoader` for RL and `SFTStreamingDataLoader` for SFT.

**Key Differences**:
- Training dataloader: RL cycles until `training.total_steps`; SFT uses its configured step or epoch limit
- Validation dataloader: Single pass, no shuffle; reset and selected by workers during validation

## Integration Example

### Recipe Configuration

```yaml
# Data and validation fragment for an existing RL recipe
data:
  train:
    files: [/data/train.parquet]
    batch_size: 1
    rollout_repeat_n: 2
    prompt_key: prompt
    shuffle: true
  max_prompt_length: 1024
  max_sequence_length: 2048

validate:
  enabled: true
  data:
    datasets:
      dev:
        files: [/data/validation.parquet]
        batch_size: 1
    validation_repeat_n: 1
  aggregation: mean
  schedule:
    before_train: false
    every_n_steps: 10
```

### Custom Evaluator with Validation Scores

An E2B evaluator returns scalar metrics with its reward:

```python
from northrl.remote_rollout.contracts import EvaluationResult

result = EvaluationResult(
    reward=1.0,
    metrics={"accuracy": 1.0, "length": 2.0},
)
```

The validator reads `Trajectory.extra_fields["score"]`, not the evaluator result directly. `E2BRolloutWorker` copies task metrics and the reward into that dictionary. A custom worker must explicitly populate it, for example `{"reward": result.reward, **result.metrics}`. With those keys present, validation reports:
- `val/reward`: Mean reward
- `val/accuracy`: Mean accuracy
- `val/length`: Mean answer length

## Related Documentation

- [Core Architecture](../02-core-architecture/overview.md) - System overview
- [Weight Synchronization](weight-synchronization.md) - Weight sync coordination
- [Activity Tracking](../02-core-architecture/activity-tracking.md) - Logging and monitoring
- [Data Loader](../03-data-loader/data-loader.md) - Data loading system
- [Evaluators](../05-rollout-workers/evaluators.md) - Evaluation patterns
