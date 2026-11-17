English | [简体中文](../../../zh/developer-guide/11-configuration-reference/trainer-config.md) | [Documentation index](../../index.md)

# Trainer Configuration

Configuration options for trainer selection, scheduling, optimizer batches, and
training control. For algorithm composition and component settings, see the
[Algorithm Components guide](../06-trainers/algorithm-components.md).

## Trainer Selection

The Controller selects the trainer directly from `algorithm.type`:

| `algorithm.type` | Trainer implementation |
|---|---|
| `grpo` | `RemoteApiGrpoTrainer` |
| `opd` | `RemoteApiOpdTrainer` |
| `sft` | `RemoteApiCrossEntropyTrainer` |

Production training uses `backend.type: weaver`; `mock` is available for tests.

<a id="grpo-and-opd"></a>

### Advantage-Based Training

Select the trainer family with `algorithm.type` and configure the estimator,
optional postprocessor, and loss in their respective blocks.

```yaml
algorithm:
  type: grpo
  # Configure optim, advantage, and loss here.

backend:
  type: weaver
```

### Supervised Fine-Tuning (`algorithm.type: "sft"`)

Supervised learning using the Weaver backend. The Controller selects
`RemoteApiCrossEntropyTrainer` internally from `algorithm.type: sft`;
`cross_entropy` is the loss function.

```yaml
algorithm:
  type: sft

backend:
  type: weaver

training:
  total_steps: 100
```

## Weaver metrics-v1

With the Weaver backend, emitted v1 observations automatically appear in NorthRL's
existing W&B run under `weaver_metrics_v1/` when W&B logging is enabled. This is
additive: existing RL/legacy metrics, their values and chart axes are unchanged.
There is no default-only allowlist: enabled diagnostics, including
`entropy/top10_mean`, flow through the same path. No second W&B run is created.
Requires the SDK observation hook in `nex-weaver>=1.19.4`.

Local persistence is independent of W&B and uses the SDK's existing writer:

```yaml
backend:
  type: weaver
  metrics_store:
    enabled: true
    path: null
```

Defaults are `enabled: true`, `path: null`, writing beneath
`./weaver/.logs/metrics-<model-id>/` relative to the NorthRL client process's working
directory. A custom path is the parent of `metrics-<model-id>/`; `enabled: false`
disables only local writes, not collection or W&B. Files are appended per metric,
not per step or per parameter. Without W&B and with local storage disabled,
observations are not durably retained by this integration.

Available metrics and collection frequency are determined by the Weaver service.
NorthRL receives those observations. `backend.metrics_store` controls local
persistence; including `wandb` in `runtime.logger.backends` enables W&B logging.

## Weaver Scheduling

Normally omit `backend.scheduling_policy` or set it to `null` to use the service's
default scheduling. This optional mapping is forwarded unchanged to the Weaver
SDK. Set it only when your service specifies a supported policy and the installed
SDK accepts this argument. Weaver SDK 1.18.0 does not support it.

## Weaver Performance Tier

`backend.performance_tier` selects the Weaver performance tier for the actor
training model. Supported values are `normal`, `fast`, and `flash`; omitting the
field or setting it to `null` uses Weaver's server default. This setting applies
only to the actor training model.

```yaml
backend:
  type: weaver
  performance_tier: flash
```

## Optimizer Batches and Forward/Backward Requests

`algorithm.optim.batch_size` controls how many converted Datums contribute to
one optimizer update. `algorithm.optim.mini_batch_size` optionally limits the
number of Datums sent in each actor `forward-only` request (logprob recomputation
or mismatch diagnostics) and each `forward_backward` request. Recompute splits
the delivered batch's expanded sequences in order, without shuffling or padding.
All chunks finish and pass validation before logprobs are replaced and optimizer
batches are formed. R2 routing refs and sample indices remain local to each
forward request. KL reference forwards are independently controlled by
`algorithm.kl.ref_logprobs_batch_size`.

For Weaver RL built-in losses, `algorithm.optim.submit_ahead` is a nonnegative
integer (default `0`, serial). `1` allows at most two outstanding requests within
each recompute or FB phase, without changing batching, normalization or OPT order.
It does not pipeline across optimizer updates or control SFT's pipeline.

```yaml
algorithm:
  type: grpo
  optim:
    lr: 1.0e-6
    batch_size: 64
    mini_batch_size: 16
```

This example runs four forward/backward requests of 16 Datums, accumulates their
gradients, and then runs one optimizer step for the 64-Datum batch. The loss
normalization denominator is computed over the entire optimizer batch before
request splitting. A smaller request does not introduce another optimizer step.

- For RL with `training.weight_sync.mode: sync`, `batch-async`, or `fully-async`, omitted or
  `null` `batch_size` uses all converted Datums from the delivered batch as one
  optimizer batch. All its requests accumulate gradients before one optimizer
  update; skipped batches (including empty training batches) perform no update.
- An explicit `batch_size` must be a positive integer and selects fixed-size
  Datum batching. RL in `no-sync` mode requires this positive integer.
- `mini_batch_size` defaults to `null` (no count-based request splitting) and
  must be a positive integer when provided. If it exceeds the actual optimizer
  batch size, that batch is sent as one request.
- `data.train.batch_size` controls input-task batching before rollouts.
  A task may produce several Datums, so it is independent of optimizer batching.
- Weaver's server-side microbatch settings control execution on its devices;
  they are independent of these NorthRL request and update boundaries.
- SFT keeps one complete token-budget batch per optimizer update, regardless of
  `algorithm.optim.batch_size`, which may be omitted or `null`;
  `mini_batch_size` can split its requests.

For grouped RL, `sync`, `batch-async` and `fully-async` all take B complete groups,
where B = `data.train.batch_size` and each group contains
R = `data.train.rollout_repeat_n` complete trajectories. Fewer ready groups return
`None`; additional groups remain queued. No separate worker-idle or loader-drained
check is required. The scheduler controls input admission, including
`batch-async`'s staleness policy. The public RL loader cycles across epochs to fill
B inputs. Only an exhausted finite source with quiescent workers permits a smaller
complete-group tail. The selected estimator computes advantages before the
delivered batch is converted to Datums. Advantage estimation and optional
postprocessing operate on the complete step, before MIS and optimizer batching.
Optimizer batch size does not change the inputs to either stage.

For one optimizer update per delivered batch, use:

```yaml
algorithm:
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
training:
  weight_sync:
    mode: sync  # or batch-async / fully-async
```

Omission refers to the final resolved configuration. If a parent configuration
sets an integer `batch_size`, explicitly use `null` to override it.

The delivered batch keeps all converted Datums in order, without padding. With
an explicit integer smaller than the delivered batch, Datums are shuffled and
split into fixed-size optimizer batches; the final batch is padded if needed.
Forward/backward request splitting never adds padding or duplicates.
Normalization follows `algorithm.loss.reduction.mode`; using one update per delivered
batch alone does not give every complete trajectory equal weight.

Cross-request accumulation requires the Weaver backend to preserve gradients
between forward/backward requests until `optim_step` completes.

## Prompt-Mean Loss Aggregation

`algorithm.loss.reduction.mode: prompt_mean` gives each complete comparison group equal
weight, with equal weight per valid response token within that group. A group is
identified by `(model_tag, group_id)`, not `task_id`: repeated sampling of the same
task forms separate groups. All rollouts and all their segments share the group's
token denominator; splitting a trajectory into more segments adds no weight.

References for this aggregation definition:

- [Mercor / SkyRL training guide](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/): Step 4, “Token aggregation,” item 3 defines equal group weights and equal token weights within each group.
- [MiMo-V2.6 technical report, §4.1, Eq. (1), p. 8](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=8): the denominator sums the lengths of all responses to one prompt. [§5.1, p. 21](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=21) names this aggregation `prompt-mean`.

The restrictions on supported losses and optimizer batching below come from the current NorthRL implementation.

For an optimizer batch containing `G` eligible groups, let `T_g` be the number of
original valid response tokens across group `g`. Each token's policy-loss weight is
`1 / (G * max(T_g, 1))`. Prompt, tool-context, and padding tokens are excluded.
The denominator uses the mask **before MIS rejection**; rejected tokens contribute
zero without increasing the weights of surviving tokens. Importance weights do
not enter the denominator either. Zero-advantage groups and successful groups with
zero training tokens remain in `G`; repaired groups use their replacement members,
and unrepairable groups do not enter `G`. A batch with no
trainable tokens performs no optimizer update.

For example, group A has two responses: 10 tokens with average token loss 2, and
30 tokens with average loss 6. Group B has two 10-token responses with average
losses 1 and 3. With one sequence per response:

| Aggregation | Result |
|---|---|
| `token_mean` | `(10×2 + 30×6 + 10×1 + 10×3) / 60 = 4` |
| `datum_mean` | `(2 + 6 + 1 + 3) / 4 = 3` |
| `prompt_mean` | `((10×2 + 30×6) / 40 + (10×1 + 10×3) / 20) / 2 = 3.5` |

`prompt_mean` requires `algorithm.type: grpo`, native `loss.mode: importance_sampling`
or `ppo_clip`, `entropy_coeff: 0`, and `kl.mode: null`.
`optim.batch_size` must be omitted or `null` so the complete delivered groups share
one optimizer update. Use `mini_batch_size` to bound forward/backward requests;
request splitting does not recompute group denominators. This supports `sync`,
`batch-async`, and `fully-async`, but not `no-sync`.

```yaml
algorithm:
  type: grpo
  loss:
    mode: importance_sampling  # or ppo_clip
    reduction:
      mode: prompt_mean
  entropy_coeff: 0
  kl:
    mode: null
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
```

## Training Control

### training.total_steps / training.epochs

Set exactly one positive training limit:

```yaml
training:
  total_steps: 200

# Or, for SFT only:
training:
  epochs: 3
```

For RL, one training step completes one prepared rollout batch. In
`sync`, `batch-async`, or `fully-async` RL, omitted or `null` `algorithm.optim.batch_size` performs
one optimizer update for a non-skipped delivered batch; an explicit integer can
produce multiple updates.
For SFT, one step processes one token-budget batch and performs one optimizer
update. `training.epochs` is supported only for `algorithm.type: sft`.

### training.save

Checkpoint cadence and the local Weaver checkpoint-index directory live under
`training.save`:

```yaml
training:
  save:
    every_n_steps: 10  # 0 disables periodic saves
    dir: "${oc.env:EXPERIMENT_PATH}/ckpt"
```

`dir` is required. Final-save behavior is independent of `every_n_steps`.

### training.resume

Resume settings live under `training.resume`:

- `mode`: `disable`, `auto`, or `from_path`
- `path`: checkpoint path used by `from_path`; in `auto`, a fallback when the
  local index has no valid entries
- `skip_dataloader`: non-SFT skip behavior; SFT restores its exact saved cursor
- `max_retries`: number of automatic controller retries after a training failure
  (default: `1`; set `0` to disable)

On a retry, NorthRL first shuts down the failed controller, constructs a fresh
controller, and forces `mode: auto` so the new attempt loads the latest entry
from `training.save.dir/weaver_checkpoints.jsonl`. The first attempt still
honors the configured `mode`. If the index has no valid entries, retries retain
the original `from_path` or `auto` path as a fallback. A path configured with
`mode: disable` stays ignored. With neither an indexed checkpoint nor a fallback,
the retry starts from scratch. SFT still requires matching checkpoint metadata
with a saved dataloader cursor; a fallback path alone cannot resume SFT.

```yaml
training:
  save:
    every_n_steps: 10
    dir: "${oc.env:EXPERIMENT_PATH}/ckpt"
  resume:
    mode: disable
    path: ""
    skip_dataloader: true
    max_retries: 1
```

Keyboard interrupts and termination signals are not retried. A hard process or
process loss requires restarting the driver; configure `mode: auto`
when that new process should resume from the checkpoint index.

### training.weight_sync.pause_mode

Optional Weaver engine pause during sampler replacement:

| Value | Behavior |
|---|---|
| `null` (default) / omitted | Keep existing synchronization behavior without an engine pause |
| `retract` | Retract running requests and reschedule them after continuation |
| `in_place` | Retain paused requests and their KV state |

Non-null values require all of the following; other combinations fail schema
validation:

- `algorithm.type: grpo` or `opd` and `training.weight_sync.mode: fully-async`.
- `backend.type: weaver` and `training.weight_sync.freeze_inflight: true`.
- `full_ft`: `models.actor.lora` omitted, `null`, or `enabled: false` with rank
  omitted or `null`. Pause is engine-wide, so LoRA adapters are unsupported.

```yaml
training:
  weight_sync:
    mode: fully-async
    freeze_inflight: true
    pause_mode: in_place
```

See [Optional engine pause](../08-features/weight-synchronization.md#optional-engine-pause).

### training.weight_sync for SFT

There is no separate Trainer-level skip flag. WeightSyncController owns sampler
weight publication and synchronization behavior.

For `algorithm.type: sft`, NorthRL derives `training.weight_sync.mode: no-sync`
when the block is omitted. An explicit conflicting mode fails schema
validation.

## Related Documentation

- [Remote API Trainers](../06-trainers/remote-api-trainers.md) - Trainer behavior and Weaver Datum protocols
- [SFT Data](../03-data-loader/sft-data.md) - Streaming input, token-budget batching, and exact resume
- [Weight Synchronization](../08-features/weight-synchronization.md) - Weight-sync modes
