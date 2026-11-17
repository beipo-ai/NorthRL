English | [简体中文](../../../zh/developer-guide/08-features/weight-synchronization.md) | [Documentation index](../../index.md)

# Weight Synchronization

## Overview

`WeightSyncController` manages model weight synchronization between training and inference services in NorthRL. It coordinates when weights should be synchronized, blocks services during updates, and manages validation triggers.

**Location**: `northrl/weight_sync/weight_sync_controller.py`

## Key Concepts

### Synchronization Lifecycle

```
Training Complete → Check Sync Mode → Sync Weights → Update Version → Continue Training
                                           ↓
                                    Validation (if needed)
```

### Model Version Tracking

The synchronization controller maintains two process-local counters for each model:
- **train_model_version**: Number of completed training rounds that updated weights
- **rollout_model_version**: Training round count reached by the last successful synchronization

The trainer names exported weights using the completed training step, such as
`northrl_actor_v000001`. The holder keeps the active sampling client's path and name
together. Each request compares the server-reported version with the name captured
for that request, independently of the controller's counters.

Sampler snapshot labels use the completed `train_step`. A filtered round does
not export weights, so these labels may have gaps.

After a restart or resume, the two counters start again at `0`; the restored
checkpoint becomes that process's sampler baseline. Global training progress is
tracked separately by `train_step`.

### Service States

`RolloutServiceState` tracks the synchronization state:
- **running**: Service operational, no sync needed
- **need_sync**: Sync required after the current training round; next data batch remains unavailable
- **syncing**: Sync in progress

## Synchronization Modes

NorthRL supports four synchronization modes via `training.weight_sync.mode`:

### 1. no-sync (No Synchronization)

No weights are published to or synchronized with a rollout sampler.

**When to use**: Pure SFT or any pipeline where no inference service needs weight updates

**Trade-off**: No rollout model receives updated weights.

```yaml
training:
  weight_sync:
    mode: no-sync
```

For `algorithm.type: sft`, NorthRL derives `no-sync` when the `weight_sync` block
is omitted. An explicitly configured conflicting mode is rejected.

**Note**: The Controller derives `keep_batch_order: false` for non-SFT
`no-sync` pipelines. For SFT it derives `keep_batch_order: true` so only one
token-budget batch is active until its optimizer step commits the source cursor
and the SFT trainer calls `release_next_batch()`. SFT does not use the weight-sync
state machine for this data-batch gate.

### 2. sync (Synchronous)

All rollout workers block until weight sync completes after every training step.

**When to use**: Maximum training stability, immediate weight updates

**Trade-off**: Lower throughput due to frequent blocking

```yaml
weight:
  sync_mode: "sync"
```

### 3. fully-async (Fully Asynchronous)

Rollout overlaps training. After every round that updates weights, including the
final round, the trainer exports the weights and passes that round's path and
name to the controller for synchronization. Skipped rounds do not export or
synchronize weights. The trainer waits for export and sampling-client replacement
to finish before starting its next training round.

The controller derives `keep_batch_order: false` for fully-async training.
`get_next_item()` loads another batch whenever the current buffer is exhausted;
it does not wait for training or a batch-ready notification. For these loaders,
`release_next_batch()` is a no-op, so the batch-ready notification does not call
it. The trajectory pool accepts results throughout training and synchronization.

With `freeze_inflight: false`, new requests can keep using the old sampling client
until replacement succeeds. With `freeze_inflight: true`, requests check the
controller before sampling and wait if synchronization is in progress. Requests
already started compare their responses
with their own captured version, even if the active client changes meanwhile.

```yaml
training:
  weight_sync:
    mode: fully-async
    freeze_inflight: false
```

#### Optional engine pause

`training.weight_sync.pause_mode` accepts `retract`, `in_place`, or `null`
(default). Omission or `null` keeps the existing behavior without an engine pause.
Enabling either mode requires `fully-async` RL (`grpo` or `opd`),
`backend.type: weaver`, `freeze_inflight: true`, and `full_ft`: omit
`models.actor.lora`, set it to `null`, or set `enabled: false` with no LoRA rank.
Pause affects the entire inference engine, so it is restricted to full fine-tuning
and cannot be used for an individual LoRA adapter on a shared engine.

```yaml
models:
  actor:
    lora: null
backend:
  type: weaver
training:
  weight_sync:
    mode: fully-async
    freeze_inflight: true
    pause_mode: retract  # or in_place; null disables engine pause
```

After weight export completes, the holder calls `pause_generation(mode=...)`
on its active sampling client, replaces it with a client using the exported
weights, and calls `continue_generation()` on the **current** client. A `finally`
block attempts recovery even if pausing or replacement fails, using the old
client if replacement did not succeed. Synchronization failures propagate to
the weight-sync controller, which keeps its synchronization state blocked.

Weaver owns request queueing and scheduling during pause. A submitted sampling
operation can remain pending across the weight update; NorthRL continues waiting
for the result of that operation. Sampling retains its concurrency semaphore,
retries, and captured client/version behavior. The existing `freeze_inflight`
check remains at the inference entry point; requests already past that check can
overlap weight synchronization. The holder adds no submission/update lock and
does not wait for pending request submissions or generation results before pausing.

| `pause_mode` | Engine behavior |
|---|---|
| `null` / omitted | No engine pause; existing synchronization behavior |
| `retract` | Retract running requests for rescheduling after continuation; preservation of generated tokens or KV state is not guaranteed |
| `in_place` | Retain paused requests and their KV state |

`in_place` requires a backend that can update weights while retaining requests
and their KV state. There is no supported cache flush while these requests are
retained. NorthRL does not automatically fall back to `retract` or another mode.

### 4. batch-async (Batch Asynchronous)

Workers block only when staleness exceeds threshold.

**When to use**: Balance between throughput and weight freshness

**Trade-off**: Configurable staleness threshold

```yaml
weight:
  sync_mode: "batch-async"
  staleness_threshold: 2  # Block if behind by 2+ versions
```

## Weight Sync Methods

NorthRL supports multiple weight transfer methods via `sync_method` config:

### network

Transfer weights via HTTP to vLLM or SGLang backends.

```yaml
weight_sync_controller:
  sync_method: "network"
  sync_weight_path: "/path/to/weights"
```

**Backends**: vLLM, SGLang

### disk

Update weights from disk path. For SGLang, queries all workers via sglang-router and updates in parallel.

```yaml
weight_sync_controller:
  sync_method: "disk"
  sync_weight_path: "/path/to/weights"
```

**Backends**: SGLang (with sglang-router)

When `weight_type: "sglang_disk"`, the disk sync path calls SGLang
`/update_weights_from_dcp` and sends `weight_version` together with
`checkpoint_dir`. Other SGLang disk updates pass the same label to
`/update_weights_from_disk`.

## Rollout Weight Version Validation

Weaver requests capture the sampling client and its expected weight name before
sampling. The response's `weight_version` must match that captured name, even if
a newer client becomes active while the request is running. Missing or empty
response versions raise an error. Inference-only holders without a known expected
name still require a response version but skip the equality comparison.

A mismatch raises `WeightVersionMismatchError` through the inference client and
rollout worker. It is not converted into a warning, automatic weight repair, or
another rollout attempt. The worker reports the failure through the existing
activity tracker; the controller's default `stop_on_error` policy stops the run.
Startup warmup uses the same strict comparison. The main entry point's existing
`training.resume.max_retries` setting still controls restarting the whole training
run after a failure; this mode does not change that behavior.

### tinker

Use Tinker service backend for weight sync.

```yaml
weight_sync_controller:
  sync_method: "tinker"
```

### weaver

Use Weaver service backend for weight sync.

```yaml
weight_sync_controller:
  sync_method: "weaver"
```

## Core Methods

### Constructor

```python
def __init__(self, config: DictConfig)
```

Initializes weight sync controller with configuration.

**Parameters**:
- `config`: Configuration including sync_mode, sync_method, validate_freq

### check_rollout_service_status

```python
def check_rollout_service_status(self, model_tag: ModelTag) -> Literal["continue", "block"]
```

Called by `InferenceServiceClient` before each request to check if service should proceed or block.

**Returns**:
- `"continue"`: Service can proceed
- `"block"`: Service should block and retry

### trajectory_pool_notify_batch_ready

```python
def trajectory_pool_notify_batch_ready(self, model_tag: ModelTag) -> None
```

Called when the trainer takes a ready batch from TrajectoryPool. Checks whether
the next data batch can be released before training this batch.

**Logic**:
1. In `sync`, or `batch-async` when staleness exceeds the threshold, set state
   to `"need_sync"` and leave the next data batch unreleased
2. In `batch-async` below or at the threshold, call `_release_next_batch()` so
   workers can collect the next batch while this batch trains
3. In `fully-async`, leave data loading unchanged; it already continues automatically
4. In `no-sync`, return without changing data loading

### train_worker_notify_weight_update

```python
def train_worker_notify_weight_update(
    self,
    worker_name: str,
    model_tag: ModelTag,
    *,
    sampling_path: str | None,
    weight_version: str | None,
    weights_updated: bool = True,
    continue_training: bool = True,
) -> None
```

Called when a training round completes. Coordinates synchronous weight
publication when needed.

**Process**:
1. If `weights_updated` is false, keep both versions unchanged, restore a
   `"need_sync"` state to `"running"`, and proceed to the shared release decision
2. Otherwise, increment `train_model_version` after the parameter update
3. Synchronize when state is `"need_sync"` or the mode is `fully-async`, using
   the supplied `sampling_path` and `weight_version`
4. Update `rollout_model_version` only after synchronization succeeds
5. After synchronization, check validation frequency; if validation is due,
   leave release to `end_validate()`
6. Both updated and skipped rounds use the same final decision: if a batch was
   held for synchronization and training continues, call `_release_next_batch()`.
   The final round keeps the next batch blocked. `fully-async` does not need a release

### _release_next_batch

This helper makes the next ordered data batch available to workers:

```python
self._dataloader.release_next_batch()
```

The pool has no weight-sync write gate. In `sync` and `batch-async`,
`keep_batch_order: true` prevents workers from taking the next data batch until it
is released. Grouped RL directly requests B = `data.train.batch_size` complete
groups in every mode; the pool does not add a worker-idle or loader-drained check.
Simple/SFT delivery waits for the current loader round to drain and workers to
become quiescent, then returns the whole round. Regular pool/store locks still
protect shared data against concurrent reads and writes.

There are three release points: before training in `batch-async` when the threshold
is not exceeded; after an updated or skipped round that held the batch; and after
validation. The latter two release only when training will continue.

### sync_weight_to_rollout_service

```python
def sync_weight_to_rollout_service(
    self, model_tag: ModelTag, sampling_path: str | None, weight_version: str | None
) -> None
```

Publishes the weights saved by `WeaverServiceHolder` and updates the synchronized
training counter after publication succeeds. Weaver requires both the path and
name; the mock backend accepts `None` for its simulated update. Failures raise an
exception.

The controller keeps `train_model_version` and `rollout_model_version` for batch
scheduling and validation. `save_weights_for_sampler(name)` returns the exported
path without storing it in the holder. `RemoteApiTrainer.train()` returns
`(metrics, sampling_path, weight_version)`; `BaseTrainer` logs only the metrics and
passes the path and name to this controller. When synchronization is due, the
controller calls `holder.update_sampling_client(sampling_path, weight_version)`.

A round that does not require synchronization leaves the active sampler unchanged
and does not retain its exported path for a later round. The next training round
exports its own weights and passes those parameters instead. Initialization and
checkpoint resume also pass their export results directly to the update call.

The holder retains only the active sampling path/name pair. Creating a new client
uses the call's parameters, even if another export occurs in the meantime. It
replaces the active client and pair only after creation succeeds. A caller can
retry a failed update with the same path and name; retrying initialization after
failure performs a fresh export instead of reusing a cached pending export.

Each sampling request captures its client and expected version together. The
holder compares the returned version with that request's expected version and
raises `WeightVersionMismatchError` on a mismatch. The error propagates through
the inference client and rollout worker without automatic repair or retrying the task.
Startup warmup also fails on a mismatch. Inference-only holders with no known
expected version skip this comparison.

E2B accepts `sync` and `fully-async`. It passes the path captured at task start to
the remote executor and retains that task's version on its trajectories. A later
local sampler change does not invalidate the result. The remote result does not expose an actual weight
version, so this path does not verify which version the E2B service used.

## Validation Coordination

Weight sync controller coordinates validation runs triggered at regular intervals.

### Configuration

```yaml
weight_sync_controller:
  validate_freq: 5  # Run validation every 5 training steps
```

### Process

After a successful synchronization, when validation is enabled and
`rollout_model_version % validate_freq == 0`:
1. Set `_waiting_for_validation = True`
2. The trainer waits before claiming another training batch and remains idle in
   activity tracking while it waits; sync and batch-async leave the next data
   batch unreleased
3. Controller runs validation using the synchronized sampler
4. Controller calls `end_validate()` to let training resume

### end_validate

```python
def end_validate(self, model_tag: ModelTag) -> None
```

Called by the controller after validation completes. Clears the validation wait
and, when training continues, calls `_release_next_batch()` to start rollout
timing and release the next data batch. In `fully-async`, the dataloader fetches
batches automatically, so this method only clears the validation wait. The final
round does not release another batch.

## RolloutServiceState

Data structure tracking per-model sync state:

```python
@dataclass
class RolloutServiceState:
    # State tracking
    state: Literal["need_sync", "syncing", "running"] = "running"
    rollout_model_version: int = 0
    train_model_version: int = 0

    # Service info
    model_name: str = ""
    weight_type: str = ""
    weight_path: str = ""
    backend: str = ""
    base_url: str = ""
```

## Configuration Reference

```yaml
weight_sync_controller:
  type: "weight_sync_controller"

  # Synchronization mode
  sync_mode: "sync"  # "no-sync" | "sync" | "fully-async" | "batch-async"

  # Staleness threshold (for batch-async mode)
  staleness_threshold: 2

  # Weight sync method
  sync_method: "disk"  # "network" | "disk" | "tinker" | "weaver" | "mock"

  # Weight path (for network/disk methods)
  sync_weight_path: "/path/to/weights"

  # Validation frequency (0 = disabled)
  validate_freq: 5

  # Inference service config (model info)
  inference_service:
    model_tag: "default"
    model: "model-name"
    weight_type: "full"
    backend: "sglang"
    base_url: "http://localhost:8000"
```

**Derived behavior for `training.weight_sync.mode`:**

| mode | `keep_batch_order` | publish sampler weights |
|------|--------------------|-------------------------|
| `no-sync` | `false` | no |
| `sync` | `true` | yes |
| `batch-async` | `true` | yes |
| `fully-async` | `false` | yes |

## Integration Example

The controller initializes and wires the weight sync controller:

```python
# In NorthRLController.__init__
self.weight_sync_controller = WeightSyncController(config.weight_sync_controller)

# Set module references
self.weight_sync_controller.set_module_references(
    dataloader=self.dataloader,
    trajectory_pool=self.trajectory_pool
)

# For Weaver backend
if config.inference_service.backend == "weaver":
    self.weight_sync_controller.set_weaver_service_holder(self.weaver_service_holder)
```

## Related Documentation

- [Core Architecture](../02-core-architecture/overview.md) - System architecture
- [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) - Trajectory collection
- [Validation](validation.md) - Validation coordination
