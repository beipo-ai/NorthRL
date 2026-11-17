English | [简体中文](../../../zh/developer-guide/04-trajectory-pool/trajectory-pool.md) | [Documentation index](../../index.md)

# Trajectory Pool

**Completion contract:** The grouped pool waits for one complete `Trajectory` per distinct rollout. `group_size` counts rollouts; `groups_per_batch` counts completed groups (eligible groups in fully-async GRPO/OPD). Delivery preserves every trajectory and its nested segments. Progress distinguishes `ready_trajectories`, `ready_groups` and trainable `ready_sequences`; `ready_segments` and `finished_trajectories` remain compatibility aliases for the sequence count.

**Terminology:** One rollout produces one complete `Trajectory`. Ordinary trajectories store tokens directly; E2B and other segmented results add internal `segments` records. Workers, the pool and trainer preparation exchange complete trajectories. See [trajectory semantics](trajectory-semantics.md).

The Trajectory Pool collects, groups, and batches complete `Trajectory` objects from rollout workers.

## Overview

**Purpose**: Collect complete rollout results, wait for complete comparison groups, and provide trajectories to the trainer

**Key Features**:
- Multi-store architecture (one store per model tag)
- Flexible grouping strategies (simple, grouped)
- Complete-group delivery and whole-round SFT delivery
- Weight synchronization coordination
- Thread-safe operation

## TrajectoryPool

**Location**: `northrl/trajectory_pool.py`

**Main interface** for complete trajectory collection.

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Configuration**:

These are internal pool settings compiled by the Controller. For grouped RL,
B comes from `data.train.batch_size`, R from `data.train.rollout_repeat_n`, and
`group_order` from `data.train.group_order` (default `fifo`). The Controller derives
`require_trainable_groups=true` only for grouped GRPO/OPD in `fully-async`; it is
an internal setting, not a public recipe option.
Grouped stores count complete groups; simple stores deliver the actual number
of complete trajectories in a completed loader round. Nested segments remain
with their complete trajectory.

```yaml
trajectory_pool:
  type: "default"
  groups_per_batch: 32  # B completed comparison groups per delivery
  group_size: 4  # R independent rollouts per group
  group_order: fifo  # Derived from data.train.group_order; fifo or lifo
  key_list: ["group_id"]  # A fresh comparison group for each sampling occurrence
```

### Core Methods

#### put_trajectory(trajectory: Trajectory) -> str

Atomically stores one complete rollout: all nested segments, final reward,
terminal status and execution metadata. It returns `"success"` or `"fail"`.
The pool routes by `model_tag`; grouped stores count this object as one rollout,
regardless of its segment count. Workers do not submit segments separately.

```python
result = trajectory_pool.put_trajectory(trajectory)
```

A successful zero-segment trajectory with a finite final reward is valid and
participates in GRPO. The pool preserves original outcomes and checks whether
failed members can be replaced under the rule below. Fully-async grouped GRPO/OPD
excludes unrepairable groups and groups without valid training targets; other
grouped modes still deliver the complete objects.
Identical submissions are idempotent for pending/ready groups and the most recent
1024 consumed groups; conflicting duplicates are rejected. This bounded retry
cache does not promise deduplication across unlimited execution history.
Packed numeric arrays are hashed directly from their complete binary buffers,
including type and length, rather than expanded into token-by-token text.
Other field representations still participate in the signature; training
data and metadata are not omitted to make duplicate detection cheaper.

Weight synchronization does not reject completed trajectory submissions. Store
locks protect shared data only while it is read or modified.

#### get_trajectories(model_tag: ModelTag | None = None) -> list[Trajectory] | None

Returns complete trajectories from the selected model's store. `model_tag=None`
tries available stores; an explicit model tag must already have an instance.
Only `None` means that no batch is ready. Fully-async grouped GRPO/OPD admits only
eligible groups as defined below. Other grouped modes still return complete objects
for failed groups and successful zero-segment groups. The trainer keeps complete
trajectories and visits their trainable sequence records internally. If none remain
after trainer-side filtering, it skips the optimizer update while advancing the round.

For grouped RL, `sync`, `batch-async` and `fully-async` all call
`take_groups(num_groups=B)` directly. A normal delivery contains B complete
groups, or B × R complete trajectories; fewer ready groups return `None`, and
a backlog never increases the delivery beyond B. The scheduler controls when
new loader batches are released. Every group remains complete, and all nested
segments stay with their complete trajectories. In fully-async GRPO/OPD, all B groups
must pass pool eligibility checks. Only other grouped modes allow the smaller
finite-source delivery described below.

```python
trajectories = trajectory_pool.get_trajectories(model_tag="actor")
```

#### is_empty(model_tag: ModelTag | None = None) -> bool

Checks if trajectory pool is empty.

```python
if trajectory_pool.is_empty():
    print("No pending or ready trajectories")
```

**Parameters**:
- `model_tag`: Specific model to check (None = all models)

**Returns**: `True` if the selected store (or all stores) has no pending or ready work. Rejected groups do not keep a fully-async grouped pool nonempty; other modes retain completed zero-segment groups until consumption

#### get_model_tags() -> list[ModelTag]

Gets all active model tags.

```python
tags = trajectory_pool.get_model_tags()
# ["policy", "reference"]
```

**Returns**: List of model tags with active stores

## TrajectoryPoolInstance

**Purpose**: Individual pool instance managing complete trajectories for a single model

**Location**: `northrl/trajectory_pool.py`

### Setup

#### set_module_references(dataloader, weight_sync_controller, activity_tracker)

Sets references to other modules.

```python
instance.set_module_references(
    dataloader=dataloader,
    weight_sync_controller=weight_sync_controller,
    activity_tracker=activity_tracker
)
```

### Trajectory Management

#### put_trajectory(trajectory: Trajectory) -> str

Adds one complete trajectory to this instance's store.

**Returns**:
- `"success"`: Added successfully
- `"fail"`: Failed to add

## Trajectory Store Types

The pool automatically creates appropriate stores based on configuration.
`key_list` must be empty or contain a single grouping key; multiple keys raise a `ValueError`.

The internal consumption APIs use explicit units:

| Store method | Count unit |
| --- | --- |
| `GroupedTrajectoryStore.take_groups(num_groups=...)` | Complete groups |
| `SimpleTrajectoryStore.take_trajectories(num_trajectories=...)` | Complete trajectories |

Both require a concrete positive integer; `None` is not supported. Each method
consumes exactly that many items, or returns `None` without consuming any if too
few are ready. Grouped RL calls `take_groups(num_groups=B)` without a separate
readiness check. Group selection follows `group_order`, with members ordered by
`run_id`. Simple/SFT delivery waits for the current loader round to finish before
passing its actual trajectory count to `take_trajectories`.

### SimpleTrajectoryStore

**Use Case**: No grouping required

**Behavior**: Appends complete trajectories to `_ready_trajectories` and delivers the completed loader round; its trajectory count may vary

**Configuration**:
```yaml
trajectory_pool:
  key_list: []  # SimpleTrajectoryStore
```

**When to Use**:
- Tasks with independent samples
- No need for trajectory grouping
- Simplest and fastest option

### GroupedTrajectoryStore

**Use Case**: Complete GRPO/OPD comparison groups identified by a sampling-specific `group_id`

**Behavior**: Buffers original trajectories by group and rollout identity. The
group closes after all `group_size` distinct executions arrive. For R members,
count F failures: non-`success` complete trajectory status, invalid final reward or a pre-existing
skip flag. If F ≤ R/2, the trainer keeps every valid member and uniformly samples
F distinct valid members without replacement, copying each once into a failed
slot. Exactly half may fail. If F > R/2, the group does not train. The pool only
checks eligibility; copying happens after raw metrics and before advantage
calculation or logprob recomputation, in every grouped GRPO/OPD mode.

**Fully-async eligibility**: The group must be repairable, and at least one
ordinary or nested sequence belonging to a valid member must have more than one
token and an active target in `loss_mask[1:]`. Successful empty members remain
valid and may be copied. A scored E2B `task_failed` result has complete trajectory status
`success`; it is not a failed member. Zero rewards and equal rewards
are allowed. Rejected groups are summarized separately, do not enter the ready
queue, and do not occupy `max_ready_groups` backpressure slots. New rollouts can
therefore fill the missing eligible-group slots without consuming `train_step`.
The effective fully-async watermark is
`max(rollout.max_ready_groups, data.train.batch_size)` when configured, so
backpressure cannot stop collection below one complete training batch.

Discard metrics include `pool/discarded_groups_total`, `pool/skipped_groups_total`,
`pool/empty_groups_total`, `pool/tail_groups_total`, and discarded trajectory,
response-token and rollout-time totals. They are cumulative process-local
counters, emitted with completed training steps. Per-group discard logs and
instance progress remain available when no training step completes or a terminal
tail is discarded.

This guarantees B eligible groups at pool admission per trainer round. Later
MIS rejection and OPD advantage-dependent filtering can still remove training
data. `algorithm.optim.batch_size` can split one group's Datums across optimizer
updates, so the guarantee is not a fixed group count per optimizer update.

**Configuration**:
```yaml
trajectory_pool:
  key_list: ["group_id"]
  group_size: 4  # Four independent rollouts, regardless of segment counts
  groups_per_batch: 32  # Deliver 32 completed groups together
  group_order: fifo  # Oldest completed first; lifo selects newest completed first
```

**GRPO semantics**: One draw of a task creates one comparison group containing R
complete trajectories. Drawing the same task again creates a different `group_id`,
even if `task_id` is unchanged. Groups are never merged solely because they share
a task. Ordinary trajectories hold token fields directly; segmented trajectories
hold internal sequence dictionaries.

```text
Group A: four attempts from one draw of task A
├── trajectory A0: tokens, loss_mask, reward
├── trajectory A1: segments=[A1.0, A1.1], reward
├── trajectory A2: tokens, loss_mask, reward
└── trajectory A3: segments=[A3.0, A3.1, A3.2], reward
```

This group has **4 complete trajectories and 7 token sequences**: two ordinary
sequences plus five nested segments. GRPO averages four final rewards. Neither
sequence count nor segment count determines group membership or completion.

**Delivery**: All four complete objects A0–A3 must arrive. Delivery preserves the
original objects for metrics. With one or two failed members, the trainer replaces
each failed slot with a copy of a distinct valid member before training. Three or
four failures make the group unrepairable. Successful zero-sequence members still
contribute their rewards and may be copied.

**Group selection order**: Set public `data.train.group_order` to `fifo` (default)
or `lifo`. FIFO consumes the earliest group to enter the ready queue; LIFO consumes
the most recent. Entry occurs only when all expected rollout results have arrived,
not when a task starts or its first result arrives. For completion order A, B, C,
a two-group delivery returns A, B with FIFO or C, B with LIFO. Each selected group
still returns all its complete trajectories in `run_id` order, preserving their segments.

The choice changes batch membership whenever more than B ready groups are queued,
most commonly in `fully-async`. Every mode takes at most B groups; `sync` and
`batch-async` normally admit one B-group input batch at a time. Subsequent GRPO
trainer sorting can change the delivered order; this is not an optimizer ordering
setting. Completion order does not indicate sampling start time or model-weight freshness, and sustained LIFO traffic can leave older
groups waiting. See [public configuration](../11-configuration-reference/data-config.md#group_order).

**When to Use**:
- GRPO algorithm (group by sampling-specific comparison identity)
- Multi-sample per query tasks
- Need to compare samples from same context

## Batch Delivery

### Grouped RL

For all three weight-sync modes, the pool directly requests
`take_groups(num_groups=B)`, where B = `data.train.batch_size`. The store returns
B complete groups or `None`; it does not consume a partial batch. For B = 32
and R = 4, a normal delivery contains 128 complete trajectories and all their
nested segments. Any additional ready groups remain queued.

This path does not wait for the loader buffer to drain or workers to become
idle. With `sync` and `batch-async`, the scheduler normally admits one B-group
input batch at a time. With `fully-async`, the B eligible completed groups can
come from different input batches; rejected groups do not count toward B. If fewer
than B groups are ready after the current loader buffer is drained and workers
are quiescent, a pending group with missing
members raises an error even if the source can still provide future batches.

### Simple/SFT

The pool waits until the current loader round has been claimed and all rollout
workers are quiescent, then calls
`take_trajectories(num_trajectories=ready_trajectory_count())` when results exist.
It delivers the actual whole round; SFT's token budget can produce a variable
trajectory count. This does not require the dataset or epoch to end.

### Finite-source Tail

If fewer than B groups remain, the pool first requires the underlying loader to
report `is_finished() is True` and all rollout workers to be quiescent. Fully-async
grouped GRPO/OPD then discards and logs the undersized eligible tail without
training or advancing `train_step`. Other grouped modes deliver the smaller tail.
Any pending group missing members still raises an error. Public step-based RL
recipes use a cycling loader that fills B inputs across epoch boundaries, so
reaching an epoch boundary does not end collection or trigger tail handling.

## Weight Synchronization Coordination

These modes describe the shared pool and scheduler. Individual workers may have
stricter requirements: `E2BRolloutWorker` accepts `sync` and `fully-async`.
`RemoteRolloutTrace` requires at least one segment, while an evaluated
`RemoteRolloutResult` can have none. The worker preserves that result's reward
on a complete zero-sequence trajectory for group statistics.

The pool accepts complete trajectories during training, weight synchronization, and
validation. Training trajectories go to the pool; validation trajectories are
routed to the validator. Weight synchronization does not reject completed results.

### sync and batch-async

1. The dataloader provides one input batch with `keep_batch_order: true`.
2. Workers submit one complete Trajectory per rollout, including failed and zero-segment outcomes.
3. Grouped RL takes B complete groups as soon as `take_groups(num_groups=B)` succeeds.
   Simple/SFT instead waits for the current loader round to drain and workers to become quiescent.
4. When the trainer takes the batch, the pool notifies the weight-sync controller.
5. In batch-async below or at the staleness threshold, the controller releases
   the next data batch before training starts.
6. Otherwise, the next data batch stays unavailable until training and weight
   synchronization complete, plus validation when required. If training continues,
   the controller releases the next data batch.

No separate pool write gate is needed: the current batch has already been
submitted, and workers cannot claim the next batch until the dataloader releases
it. A skipped training round releases a held batch without synchronizing weights
when training will continue.

### fully-async

The dataloader uses `keep_batch_order: false` and loads another input batch as soon
as workers claim all items in the current buffer. The pool calls
`take_groups(num_groups=B)` to consume B eligible completed groups while other
rollouts continue. Unrepairable groups and groups whose valid members have no
active training targets are excluded before counting readiness, with rejection
statistics recorded separately.
Groups may finish out of input order; no partial group is delivered. Every valid
consumed group retains all of its trajectory objects, nested segments and final rewards.
Data loading continues automatically without a batch-release call when the trainer
takes a batch. Weight synchronization does not stop complete trajectory submissions.

With `freeze_inflight: true`, inference requests wait at the client before sampling
when the controller reports synchronization in progress. This check does not
reject completed trajectory submissions to the pool.

## Usage Patterns

### Basic Usage in Trainer

```python
class MyTrainer(BaseTrainer):
    def train_step(self):
        # Get batch
        batch = execute(self.trajectory_pool.get_trajectories)

        if batch is None:
            # Not enough samples yet
            return None

        # Fully-async grouped RL has already excluded ineligible groups.
        # Record original outcomes, then replace eligible failed slots before training.
        # Successful empty members remain valid, including as replacement donors.
        # If no trainable segments remain, skip the optimizer but advance the round.
        metrics = self._train_on_batch(batch)

        return metrics
```

### GRPO Grouping

```yaml
# R distinct rollout completions per group; B complete groups per delivery
trajectory_pool:
  key_list: ["group_id"]  # GRPO comparison key
  group_size: 4
  groups_per_batch: 32  # All segments from 32 complete groups
```

```python
# Preserve the dataloader-assigned comparison group and attempt number
trajectory = {
    "tokens": ...,
    "group_id": task["group_id"],  # Shared by attempts in this comparison group
    "run_id": task["run_id"],  # Attempt slot within this group
    "rollout_id": task["rollout_id"],  # Unique execution identity
    "expected_rollouts": task["expected_rollouts"],
    # ...
}
```

### Multiple Models

```python
# Separate stores for policy and reference models
trajectory_policy = {
    "tokens": ...,
    "model_tag": "policy",
}

trajectory_reference = {
    "tokens": ...,
    "model_tag": "reference",
}

# Get batches per model
policy_batch = pool.get_trajectories(model_tag="policy")
reference_batch = pool.get_trajectories(model_tag="reference")
```

## Configuration Examples

### Simple (No Grouping)

```yaml
trajectory_pool:
  type: "default"
  key_list: []  # Deliver the completed loader round
```

### GRPO (Single-Level Grouping)

```yaml
trajectory_pool:
  type: "default"
  groups_per_batch: 32
  group_size: 4  # Four distinct terminal rollout records per group
  key_list: ["group_id"]  # Group by comparison group
```

## Best Practices

### Grouping

1. **Use grouping for GRPO** - Essential for proper advantage computation
2. **Count complete trajectories** - `group_size` is the expected rollout count R, and each unique terminal rollout contributes one final reward; segment counts do not change R
3. **Preserve sampling identity** - Attempts share a fresh group_id for this sampling occurrence; run_id identifies slots and rollout_id uniquely identifies executions. Repeated epochs do not reuse the group
4. **Test grouping logic** - Verify trajectories grouped correctly

### Batch Sizes

1. **Balance memory and throughput** - Larger batches = better GPU utilization but more memory
2. **Configure groups, not flattened segments** - `groups_per_batch` selects B complete groups; deliver every segment they contain. Optimizer batch/mini-batch splitting happens later and is unchanged
3. **Keep delivery units explicit** - Grouped RL takes B groups; Simple/SFT takes the actual completed loader round
4. **Monitor batch timing** - Adjust if trainer waiting too long for batches

### Performance

1. **Minimize lock contention** - Trajectory stores use fine-grained locking
2. **Pre-group in workers** - Preserve group_id and run_id before submission
3. **Monitor pool size** - Don't let pools grow unbounded
4. **Use appropriate store type** - SimpleTrajectoryStore is fastest when grouping not needed

### Debugging

1. **Log segment counts** - Monitor incoming and outgoing segments
2. **Check grouping keys** - Verify trajectories have required keys
3. **Monitor batch readiness** - Check if batches forming as expected
4. **Inspect trajectories** - Look at first few trajectories to verify structure

## Troubleshooting

### Issue: "Trainer not receiving batches"

**Symptoms**: Trainer idle, trajectories accumulating

**Solutions**:
1. Check public `data.train.batch_size` matches the desired group count
2. Check that every expected rollout submitted a complete Trajectory and that `groups_per_batch` matches the desired complete-group count
3. Check `key_list` keys exist in trajectories
4. Monitor the ready segment count in the pool

### Issue: "Incomplete groups"

**Symptoms**: Some groups never complete, trajectories stuck

**Solutions**:
1. Verify all trajectories have grouping keys
2. Check for typos in group key names
3. Check all R run slots have unique rollout identities and complete Trajectory objects, even for failed or executions with no training sequences
4. Do not force-flush missing trajectories. When fewer than B groups are ready, a drained loader buffer with quiescent workers and unfinished groups reports an error

### Issue: "Weight sync blocking too long"

**Symptoms**: Workers wait for the next input batch or for inference requests to proceed

**Solutions**:
1. Reduce weight sync frequency
2. Use async weight sync mode
3. Increase staleness threshold
4. Optimize sync operation speed

## Next Steps

- Understand [Weight Synchronization](../08-features/weight-synchronization.md)
