English | [简体中文](../../../zh/developer-guide/06-trainers/remote-api-trainers.md) | [Documentation index](../../index.md)

# Remote API Trainers

See [Algorithm Components](algorithm-components.md) for the shared configuration
structure and the boundaries between estimation, postprocessing, loss, and reduction.
The trainer calls concrete components directly; the sections below describe its
numerical behavior and token-alignment contracts.

The pool and preparation APIs use complete `list[Trajectory]` objects throughout. Ordinary trajectories store tokens directly; only segmented results contain internal records. Token operations and Datum conversion iterate those sequences internally. `batch/segment_count` counts all saved input segments and `batch/trainable_segment_count` counts effective segments after Datum/mask processing; optimizer `batch_size` and `mini_batch_size` count Datums. See [trajectory semantics](../04-trajectory-pool/trajectory-semantics.md).

Remote API trainers use Weaver for training on external infrastructure. These trainers receive complete trajectories and convert their retained token sequences into service Datums.

> **Heads up — logprob shape conventions.** When working on trajectory →
> Datum routing, KL/ref_model, or anything that touches `logprobs` /
> `ref_logprobs` / `old_log_probs`, read
> [`.ai-instructions/developing/logprobs-conventions.md`](../../../../.ai-instructions/developing/logprobs-conventions.md)
> first. The full-length vs target-aligned distinction is silent if you
> get it wrong (no exception, just wrong gradients).

## Architecture

```python
BaseTrainer
    ↓
RemoteApiTrainer (abstract)
    ↓
    ┌────────────────────────────┐
    ↓                            ↓
RemoteApiGrpoTrainer    RemoteApiCrossEntropyTrainer
```

## RemoteApiTrainer

Base class for remote API training backends. Located in `northrl/trainer/remote_api_trainer.py`.

### Purpose

Provides core training loop for remote API backends:
1. Algorithm-specific trajectory preparation
2. Conversion to service datum format
3. Remote API communication
4. Metrics collection and logging


| Boundary | Remote trainer contract |
| --- | --- |
| Input and preparation | Complete `list[Trajectory]`, with direct token fields or internal sequence dictionaries |
| Tokenization | Workers supply token IDs; Datum conversion aligns inputs, targets and masks |
| Batching | NorthRL forms optimizer batches and requests; the service manages device microbatches |
| Preparation and mathematics | `_prepare_trajectories()` prepares complete trajectories; the estimator computes advantages later in `train()` |
| Service payload | Sequence-level Datums; device padding and packing are handled by the service |

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Key Initialization:**
- Extracts training hyperparameters (lr, beta1, beta2, etc.)
- Builds the concrete estimator, optional postprocessor, and backend loss adapter
- Service holder is set later via `set_service_holder()`

With an SDK supporting `tensor_output_format`, Weaver recompute requests opt into
CPU tensor outputs for HTTP-binary. NorthRL copies them into owned float64 arrays
and prepends the no-context zero in bulk for compact trajectories. Legacy SDKs
and inline JSON still work; target alignment, R2/R3 metadata and MIS checks are unchanged.

Compact trajectory buffers stay compact during target alignment, masking and
scalar-advantage expansion. NumPy computes numeric token statistics without
Python per-token objects; alignment checks, float64 values and estimator formulas stay unchanged.

With a supporting SDK, HTTP-binary Datum construction also keeps input tokens in
an owned numeric buffer. Public mutable token lists materialize on access;
inline JSON and older SDKs retain the existing path and validation.

### Optimizer and Request Batching

`algorithm.optim.batch_size` sets the target number of Datums per RL optimizer update.
The optional `algorithm.optim.mini_batch_size` sets the maximum number of Datums
per forward/backward request within that optimizer batch. All requests for a
batch complete before its single optimizer step; normalization is computed over
the complete optimizer batch, not independently per request. See
[Trainer Configuration](../11-configuration-reference/trainer-config.md#optimizer-batches-and-forwardbackward-requests)
for configuration details.

`algorithm.optim.submit_ahead: 1` allows one request ahead for Weaver built-in losses.
It overlaps next-request preparation/submission with remote execution in recompute
and FB phases. `0` (default) keeps serial requests. Results retain input order;
failures drain submitted work and prevent OPT. Whole-batch normalization and the
recompute-before-FB boundary are unchanged. Custom losses are not supported.
Initial preparation and the final request's completion remain on the critical path;
a phase with only one request has nothing to overlap. Whole-batch conversion and
normalization still precede FB. SFT retains its separate existing request pipeline.

Recompute retries only the failed forward chunk, at most three times after its
initial attempt, with a warning before each retry. Successful chunks are retained;
logprobs and router replay records are written back after all chunks pass validation.
Cancellation is not retried. This does not add retries to FB or optimizer updates.

For RL with `training.weight_sync.mode: sync`, `batch-async`, or `fully-async`, omitted or `null`
`batch_size` keeps all converted Datums from the delivered batch in one optimizer
batch, without shuffling or padding. Grouped RL uses the same B-group delivery
in all three modes, where B = `data.train.batch_size` and each group contains
R = `data.train.rollout_repeat_n` complete trajectories. Fewer than B ready groups
return `None`; a backlog stays queued beyond those B groups. The pool does not
separately wait for idle workers or a drained loader buffer. The scheduler still
controls input admission, including `batch-async`'s staleness policy. Public RL
loaders cycle across epochs to fill B inputs; only an exhausted finite source
with quiescent workers permits a smaller tail of complete groups. The selected
estimator receives the complete delivered step before Datum conversion.
`mini_batch_size` only splits forward/backward requests; all requests accumulate gradients before one
optimizer update. Skipped batches, including empty training batches, perform no
update or weight export.

An explicit positive integer retains the existing Datum-level batching rules:
when an RL batch is split, NorthRL shuffles Datums and pads a short final optimizer
batch by repeating its own records; a batch that does not need splitting is kept
as-is. `no-sync` RL requires a positive integer. Mini-batches preserve
their optimizer batch's order and add no padding. Advantage estimation and optional
postprocessing finish on the complete step before batching; integer batching may
distribute a trajectory's sequences or a comparison group's members across optimizer batches.
Using one optimizer batch per delivery does not change trajectory/segment
representation or loss weighting. SFT retains its complete token-budget batch per update and permits
omitted or `null` `batch_size`, as described below.

For `algorithm.loss.reduction.mode: prompt_mean`, the complete delivered groups must
share one optimizer update (`optim.batch_size: null`). The trainer counts original
response tokens across all rollouts and segments of each `(model_tag, group_id)`
and weights policy loss by `1 / (G * max(T_g, 1))`, before splitting requests.
MIS rejection does not shrink this denominator. Zero-advantage and successful
zero-token groups remain in `G`; repaired groups use their replacement members,
and unrepairable groups are excluded. This mode
requires native GRPO `importance_sampling` or `ppo_clip`, zero entropy coefficient,
disabled KL, and disabled custom loss. See [configuration and examples](../11-configuration-reference/trainer-config.md#prompt-mean-loss-aggregation).

<a id="trajectory-ordering"></a>

**Current ordering limitation:** If one delivered batch contains several groups
with the same `task_id`, ordering still uses random group UUIDs to break the tie.
Even with a fixed shuffle seed, changing those UUIDs can change Datum order and
optimizer-batch membership. This reproducibility issue remains unresolved; the
complete-trajectory model does not provide a stable sampling-order field.

### Main Training Flow

The order below is visible in
[`RemoteApiTrainer.train()`](../../../../northrl/trainer/remote_api_trainer.py).
It is an execution outline, not a replacement implementation:

```text
record complete outcomes -> repair/filter failed groups
order trajectories + prepare estimator-specific inputs; reference/teacher I/O
optional actor logprob recomputation / mismatch diagnostics
if estimator exists:
    clear_postprocessing_cache(trajectories)
    trajectories = estimator.estimate(trajectories, inputs, metrics)
    if postprocessor exists:
        trajectories = postprocessor.process(trajectories, metrics)
prepared-trajectory dump -> MIS -> sequence-to-Datum conversion
split optimizer batches -> loss_adapter.normalize_advantages_for_sum()
split requests -> forward/backward accumulation -> optimizer update
export weights when required -> return metrics, sampling_path, weight_version
```

`__init__()` calls `build_estimator()`, `build_postprocessor()`, and
`build_loss_adapter()`. The estimator and optional postprocessor see the complete
step before Datum splitting; no execution plan, registry lookup, or step
dispatcher sits between the trainer and those calls. SFT has no estimator.
The adapter handles native/custom payloads and supported sum-loss scaling;
[`algorithm/reductions.py`](../../../../northrl/algorithm/reductions.py) holds the
pure denominator calculations. The trainer still owns every service call.

`BaseTrainer._main_loop()` awaits `train()`, logs only `metrics`, and passes the
exported path and name to WSC before advancing the training-step counter. Filtered
batches return `(metrics, None, None)` without training or exporting weights.
Each Weaver holder coroutine is awaited through `await_coro(...)`, which keeps
the Trainer orchestration on its dedicated loop while running the event-loop-affine
Weaver client and operation handles on NorthRL's SDK loop.

### Key Methods

#### _prepare_trajectories() (Abstract)

```python
@abstractmethod
async def _prepare_trajectories(
    self,
    trajectories: list[Trajectory],
    metrics: dict[str, Any]
) -> list[Trajectory]
```

Trainer-specific preparation before probability I/O and advantage estimation.
Subclasses implement this hook; the estimator is called separately in `train()`.

**Parameters:**
- `trajectories`: Complete rollout results, including failed/skipped or successful zero-sequence outcomes
- `metrics`: Dictionary to populate with metrics

**Returns:** Prepared complete `list[Trajectory]`; ordinary token fields and any nested sequence records retain their original ownership.

**Purpose:** Prepare and validate trajectories, keeping internal sequences within their owning trajectory.
Advantage formulas run in the concrete estimator after the required probability
inputs are ready, rather than inside this hook.

#### set_service_holder()

```python
def set_service_holder(self, service_holder: WeaverServiceHolder) -> None
```

Sets the remote service holder for API communication.

**Called by:** Controller during initialization

## RemoteApiGrpoTrainer

`RemoteApiGrpoTrainer` prepares complete comparison groups and converts training
data for remote API backends. It is located in
`northrl/trainer/remote_api_grpo_trainer.py`. The base training flow calls the
estimator selected by `algorithm.advantage.estimator.mode`, followed by optional
postprocessing, MIS, and Datum conversion. See
[the component call chain](algorithm-components.md#step-data-flow-and-alignment)
for the shared data flow and available components.

<a id="_prepare_trajectories-implementation"></a>

### _prepare_trajectories() Implementation

This hook calls `prepare_grpo_trajectories()` to order trajectories by task,
group, and run ID. It checks for duplicate rollout IDs and, when
`expected_rollouts` is supplied, verifies that every expected run ID appears
exactly once. Groups containing failed rollouts are marked to skip; rewards
must be finite in groups that are not skipped. It also copies trajectory
metadata to internal sequences and records rollout-logprob diagnostics.
It does not calculate advantages.
The base training method later calls the selected estimator's `estimate()` with
the complete prepared step. See [the component call chain](algorithm-components.md#step-data-flow-and-alignment).

The formulas and scalar-advantage behavior below describe
`algorithm.advantage.estimator.mode: grpo`.

This API receives complete `Trajectory` objects. **GRPO baseline counts complete trajectories, once each**, rather than segments. For a comparison group of N distinct complete trajectories with final rewards R_1 through R_N:

```text
b_g     = (R_1 + ... + R_N) / N
sigma_g = std(R_1, ..., R_N)
A_i     = (R_i - b_g) / (sigma_g + epsilon)
```

The standard deviation uses those same N rewards. All segments of trajectory i share A_i; changing only its segment count changes neither R_i, b_g, nor A_i. Reward belongs only to the complete trajectory; segments do not store reward copies. The standard-deviation convention and single-trajectory case are specified below.

**Complete-trajectory statistics:** The estimator reads each complete trajectory's final `reward` once per unique `rollout_id` and stores its advantage on that same complete trajectory. Token operations read the complete trajectory advantage and apply it to any internal segments. For N > 1, the standard deviation is the sample standard deviation over those N rewards (denominator N−1), preserving the main GRPO path's convention. For N = 1, the trajectory's advantage is zero.

`use_run_ids` is a historical argument name. Both settings group by
`(model_tag, group_id)` and validate expected `run_id` slots when supplied; it
does not switch grouping on or off or average several segments into a rollout.
The GRPO estimator uses the `True` calculation:

| `use_run_ids` | Standard deviation for N > 1 | Reward calculation dtype | Complete trajectory `advantage` value |
| --- | --- | --- | --- |
| `True` (main path) | Sample, denominator N−1 | `torch.float32` | Scalar tensor |
| `False` | Population, denominator N | `torch.float64` | Python `float` |

Both settings assign zero advantage to a single-trajectory group and return the
complete trajectory list.

A completed group has R original members and F failures: non-`success` complete trajectory
status, invalid final reward or a pre-existing skip flag. If F ≤ R/2, keep all valid
members, uniformly sample F distinct valid members without replacement, and copy
each once into a failed slot. If F > R/2, skip the group. The grouped training
flow applies this rule after recording raw outcomes and before advantage
calculation or logprob recomputation. Valid zero-sequence members remain in the
comparison group and are eligible donors, but produce no Datum.
Zero or equal rewards remain
valid; scored E2B `task_failed` outcomes have complete trajectory status `success`.

The pool preserves original `list[Trajectory]` groups. Fully-async excludes
unrepairable groups and groups whose valid members have no training targets;
other modes still deliver those original outcomes. Only `None` means no batch is
ready. If trainer filtering leaves no trainable data, skip optimizer updates and
advance the training-step counter. Batch scheduling and optimizer boundaries
follow the rules above.

Raw rollout reward/score and input-count metrics describe original outcomes,
before replacement. `batch/trajectory_count` counts those outcomes and
`batch/segment_count` counts their saved sequences. Advantage statistics describe
the repaired training groups and the advantages produced by the selected estimator.
`batch/resampled_group_count` counts repaired groups and
`batch/resampled_trajectory_count` counts replaced slots.
`algorithm/trajectory/advantage_token_weighted_mean` retains token weighting;
`batch/trainable_segment_count` counts effective segments after Datum/mask
processing, including replacements but before optimizer padding. See
[step metrics in W&B](../../metrics.md).

This training flow retains samples with active training tokens even when their
advantages are zero. Optional KL or entropy terms may still contribute where
the mask permits. Responses stopped by the sampler with
`finish_reason: length` keep their rollout loss mask and the math agent judges
the visible response normally instead of forcing reward `0`.

After advantage estimation, optional postprocessing, and MIS computation, the
shared `convert_trajectories_to_datums()` applies MIS and checks the final
`loss_mask`. An all-zero mask produces neither a Datum nor a source entry,
including sequences fully rejected by MIS. Only the produced Datums are then
split into optimizer batches. This filtering does not recompute advantages.

For Slime-compatible math scoring, set
`rollout.reward_fn: math_answer_only` (or `reward_fn: math_answer_only` on a sample). It checks the complete
response for a boxed answer, applies the same MathD/SymPy equality rules as
Slime's `rm_type=math`, and returns exactly `0.0` or `1.0`; think tags do not
affect the result. The single-turn rollout worker enforces `data.max_prompt_length`
using token IDs returned by the inference service's chat template rather than
the rendered string's character count.

### Configuration Example


### Router Replay (Weaver MoE)

`RemoteApiGrpoTrainer` can transport MoE routing refs without loading or
validating routing tensors in NorthRL. The Weaver trainer remains authoritative
for routing shape, parallel placement, persistence, and replay validation.

- **R3 (rollout replay):** requires
  `sampling_logprob_source: recomputed`. The sampler offloads each sequence's
  routing and returns an opaque `moe_topk_indices_ref`. NorthRL keeps the ref on
  the corresponding sequence record and sends it as `datum.metadata.router_replay.value_ref` to
  both trainer forwards: the `forward_logprob` call that recomputes the IS
  denominator and the subsequent loss forward that computes the current-policy
  numerator. Both forwards therefore use the rollout-selected expert paths,
  while both sides of the ratio use trainer-computed logprobs.
- **R2 (recompute replay):** requires
  `sampling_logprob_source: recomputed`. The exact trainer forward that
  produces recomputed logprobs also runs `RECORD`. Its result explicitly
  contains both logprobs and the recorded routing-set ref; NorthRL immediately
  binds that ref and the original record index to each sequence before Datum
  conversion.

Both modes fail the step when an enabled replay path is missing its required
ref. Replay metadata is attached before debug datum dumps and optimizer-batch
splitting, so observability and training see the same payload.

```yaml
algorithm:
  type: grpo
  sampling_logprob_source: recomputed
  router_replay:
    enabled: true
    mode: R3                          # R2 or R3
```

## RemoteApiCrossEntropyTrainer

Implements supervised learning on the Weaver backend. Located in `northrl/trainer/remote_api_cross_entropy_trainer.py`.

### Purpose

Used for supervised fine-tuning tasks where:
- Trajectories contain reference outputs
- No advantage computation needed
- Uses cross-entropy loss directly

### _prepare_trajectories() Implementation

```python
async def _prepare_trajectories(
    self, trajectories: list[Trajectory], metrics: dict[str, Any]
) -> list[Trajectory]:
    # No reward/advantage computation; retain the sequence loss masks.
    return trajectories
```

**Rationale:** For supervised learning, the `loss_mask` field already indicates which tokens should contribute to the loss. No advantage computation or reward processing is needed.

### Configuration Example

```yaml
algorithm:
  type: sft

backend:
  type: weaver

validate:
  enabled: false

training:
  total_steps: 100
```

SFT may instead use an epoch limit:

```yaml
training:
  epochs: 3
```

`algorithm.type: sft` selects `RemoteApiCrossEntropyTrainer`, derives
`algorithm.loss.mode: cross_entropy`, and does not provision an inference
sampler. Omit the entire top-level `rollout` block; the Controller always
creates exactly one internal `SFTRolloutWorker`. If `rollout` is provided,
NorthRL emits a `UserWarning` and ignores the whole block. NorthRL also derives
`training.weight_sync.mode: no-sync` when it is omitted. `rollout_repeat_n`
is not configurable for SFT; an explicit value emits a warning and is ignored.
Sampler-backed validation must currently remain disabled.

Exactly one of `training.total_steps` and `training.epochs` is required.
`training.epochs` is currently supported only for SFT.

## SFT Service Datum Format

SFT uses `convert_trajectories_to_datums_cross_entropy()` to send Weaver
pre-tokenized next-token prediction arrays:

```python
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums_cross_entropy

datums = convert_trajectories_to_datums_cross_entropy(trajectories)
# Each datum has exactly this protocol:
{
    "input_tokens": [...],
    "loss_fn_inputs": {
        "target_tokens": [...],
        "loss_mask": [...],
    },
}
```

For a trajectory with token-aligned `tokens` and `loss_mask`, the resulting
arrays are target-aligned as follows:

```text
input_tokens = tokens[:-1]
target_tokens = tokens[1:]
loss_mask = trajectory.loss_mask[1:]
```

There are no prompt/response strings, advantages, or logprobs in an SFT Datum.
Weaver receives token IDs directly; it does not tokenize the SFT records.

Before this conversion, `SFTStreamingDataLoader` has already grouped
independent items under this per-request budget:

```text
token_budget = data.max_sequence_length * data.train.batch_size
datum contribution = len(input_ids) - 1
```

The trainer treats that whole token-budget group as one optimizer window and
uses the same `max_sequence_length * batch_size` budget for transport. A normal
loader window is therefore one `forward_backward` operation, allowing Weaver to
compute one loss denominator across every Datum, microbatch, and DP rank. Datum
order remains stable. If a custom caller supplies an oversized window, NorthRL
greedily divides it into whole-Datum transports, queues them in stable order,
collects every result, and submits exactly one `optim_step` only after all
transports succeed. A submission failure drains any already accepted
operations, while a compute failure prevents both the update and source-cursor
commit. `training/window_shifted_tokens` reports the public position total,
`training/loss_agg_denominator` reports the effective supervised token count,
and `training/num_transport_batches` reports the number of forward/backward
calls. The corresponding input/effective-token throughput is also logged.
NorthRL preserves each item as an independent Datum throughout.

Epoch-based SFT can apply a per-optimizer-window cosine schedule:

```yaml
algorithm:
  type: sft
  optim:
    lr: 2.0e-5  # peak LR
    schedule:
      style: cosine
      min_lr: 2.0e-6
      warmup_fraction: 0.03
training:
  epochs: 1
```

Before training or resume, the loader materializes the deterministic epoch plan
once to count its exact greedy token-budget windows. That count is the schedule
horizon; it is not estimated from row count or copied from another dataset. The
first update is zero-based and therefore uses LR 0 during a non-zero warmup.
`training/learning_rate` records the LR actually sent to Weaver, while
`training/lr_schedule_step` records the zero-based completed-update index used
to calculate it. An unsuccessful optimizer update does not commit the SFT
cursor or advance that index.

## Key Features

### Complete Trajectories to Datums

Preparation returns complete trajectories. Ordinary results keep direct tokens; only segmented results use internal dictionaries:

```python
trajectory = Trajectory(
    tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0,
    extra_fields={"old_log_probs": [0.0, -0.2, -0.3], "advantage": 0.5},
)
datums = convert_trajectories_to_datums([trajectory])
assert trajectory.segments is None
```

KL, OPD, logprob recomputation and MIS also visit the relevant sequences internally. They do not require a public segment object. The main path does not use `Batch`.

Prepared-trajectory dumps use `format_version: 3`: ordinary results retain top-level token fields; segmented results retain internal `segments` dictionaries. `num_trajectories` counts executions and `num_sequences` counts sequences; the loader accepts older formats.

### Weaver Service Holder

For `backend.type: weaver`, the Controller creates and injects the Weaver
service holder:

```python
if config.backend.type == "weaver":
    service_holder = WeaverServiceHolder(config)
    trainer.set_service_holder(service_holder)
```

SFT skips inference-sampler provisioning, but its optimizer requests still use
the Weaver training service.

### Automatic Trainer Selection

The Controller selects the trainer directly from `config.algorithm.type`; there
is no separate public trainer selector:

```python
trainer_registry = {
    "grpo": RemoteApiGrpoTrainer,
    "opd": RemoteApiOpdTrainer,
    "sft": RemoteApiCrossEntropyTrainer,
}
trainer = trainer_registry[config.algorithm.type](config)
```

Production training uses `backend.type: weaver`. In particular, supervised
training is selected only by `algorithm.type: sft`.

## SFT Usage Example

The supported supervised path uses the canonical top-level selectors:

```yaml
algorithm:
  type: sft

backend:
  type: weaver

training:
  total_steps: 100

validate:
  enabled: false
```

This is a focused selector example; model, data, optimizer, and save settings
are omitted. There is no `trainer.type`, `trainer.train_service`, Tinker, or

For custom algorithms, extend `RemoteApiTrainer`:

```python
from northrl.trainer import RemoteApiTrainer
from northrl.northrl_types import Trajectory

class MyCustomRemoteTrainer(RemoteApiTrainer):
    async def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict
    ) -> list[Trajectory]:
        # Your custom algorithm logic
        for traj in trajectories:
            # Add custom fields
            traj["custom_field"] = compute_custom_value(traj)

        return trajectories
```


<a id="comparison-grpo-vs-sft"></a>

## Trainer Comparison

| Feature | RemoteApiGrpoTrainer | RemoteApiCrossEntropyTrainer |
|---------|----------------------|------------------------------|
| **Training Type** | Reinforcement Learning | Supervised Learning |
| **Loss Function** | Configured policy loss | cross_entropy |
| **Advantage Computation** | Configured estimator | No |
| **Comparison Grouping** | By model_tag and group_id; run_id identifies an attempt slot | No grouping |
| **Use Case** | RL with reward signal | Supervised fine-tuning |
| **Requires reward** | Yes | No |

## Related Documentation

- [Mismatch Correction (MIS)](mis.md) - Recomputed logprobs, importance weights, and rejection masks
