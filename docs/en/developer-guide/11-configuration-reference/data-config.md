English | [简体中文](../../../zh/developer-guide/11-configuration-reference/data-config.md) | [Documentation index](../../index.md)

# Data Loader Configuration

Public training input settings live under `data.train`. The Controller selects
`TorchDataLoader` for rollout-based training with `files`, `MixtureDataLoader`
for GRPO training with `sources`, and `SFTStreamingDataLoader` for
`algorithm.type: sft`; there is no public `data.type` selector.

## Rollout-based Training

This is a configuration fragment to merge into an existing RL recipe:

```yaml
experiment:
  seed: 42
models:
  actor:
    tokenizer: "${oc.env:NORTHRL_MODEL_PATH}/model"
data:
  train:
    files:
      - "${oc.env:NORTHRL_DATA_PATH}/train1.parquet"
      - "${oc.env:NORTHRL_DATA_PATH}/train2.parquet"
    batch_size: 32
    rollout_repeat_n: 8
    group_order: fifo
    prompt_key: prompt
    filter_prompts: false
    shuffle: true
    drop_last: true
  max_prompt_length: 4096
  max_sequence_length: 12288
training:
  total_steps: 3000
  weight_sync:
    mode: sync
```

### files

`data.train.files` is a nonempty list of input file paths, required when
`sources` is omitted. Set exactly one of `files` and `sources`. The RL loader
supports Parquet, CSV, and JSONL and concatenates the supplied files.
List each path once; step-based RL automatically cycles the dataset until the
training stop condition is reached. Use `${oc.env:NAME}` for environment
variables, and list shard paths explicitly.

### Mixing RL Task Sources

Use `data.train.sources` instead of `files` to sample task families by weight:

```yaml
data:
  train:
    sources:
      - type: local
        name: mimo_code
        files: [/data/mimo/code.jsonl]
        ratio: 1
        rollout_profile: mimo_code
      - type: local
        name: mimo_arvo
        files: [/data/mimo/arvo.jsonl]
        ratio: 1
        rollout_profile: mimo_arvo
      - type: local
        name: mimo_terminal
        files: [/data/mimo/terminal.jsonl]
        ratio: 1
        rollout_profile: mimo_terminal
      - type: local
        name: mimo_general
        files: [/data/mimo/general.jsonl]
        ratio: 1
        rollout_profile: mimo_general
      - type: local
        name: swe
        files: [/data/swe/train.jsonl]
        ratio: 1
        rollout_profile: swe
    batch_size: 8
    rollout_repeat_n: 8
    shuffle: true
```

RL source mixing requires `algorithm.type: grpo` and supports only
`type: local` sources. Each source requires a unique non-empty
`name`, non-empty `files`, and a positive `ratio`. Every original input draw first
selects a source with probability `ratio / sum(ratios)`, then takes the next input
from that source's independent shuffled cycle. This example gives each family
20% probability; for 50% SWE and 50% MiMo overall, set SWE's ratio to `4` and
leave each MiMo ratio at `1`. These are long-run sampling probabilities, not
per-batch quotas. File sizes do not determine the source probabilities.

`batch_size` still counts original inputs and `rollout_repeat_n` is shared by
all sources. Each selected input is expanded into one complete rollout group;
a group never mixes tasks. Sources cycle independently until
`training.total_steps` is reached. `total_samples` and `data.train.seed` are SFT
settings; RL uses `experiment.seed`. With `shuffle: false`, source selection is
still random but each source cycles in file order.

The loader injects `data_source: <source.name>` into each runtime task and
preserves the original `task_id`. An optional source `rollout_profile` is also
injected, so single-family JSONL files need no new field. If a row already
contains a different `rollout_profile`, loading fails. For a file containing
multiple execution profiles, omit the source-level value and set
`rollout_profile` on every row. See
[E2B execution profiles](rollout-config.md#execution-profiles) for worker routing.

Keep the seed, source order, files, and ratios fixed when resuming. Synchronous
training reconstructs source draws through `skip_batches(global_step)`; async
completion order and in-flight tasks are not checkpointed as sampler state.
Start with `training.weight_sync.mode: sync` when checking reproducibility.

### batch_size

`data.train.batch_size` is a required positive integer. For RL, let `B` denote
this value and `R` denote `data.train.rollout_repeat_n`:

| Stage | Unit and count |
| --- | --- |
| Loader source batch | `B` original dataset inputs, before repeat expansion |
| Loader task buffer | `B × R` independent rollout requests |
| GRPO/OPD sampling groups | `B` groups, each expecting `R` complete rollout outcomes |
| Training token sequences | One per ordinary trajectory, or the nested sequences of a multi-sequence trajectory; may differ from `B × R` |

The Controller derives the trajectory pool's internal `groups_per_batch=B`
and `group_size=R` from these public settings. Grouped RL uses
`take_groups(num_groups=B)` in `sync`, `batch-async` and `fully-async`: it returns
B complete groups (B × R complete trajectories), or `None` if fewer are ready.
There is no separate loader-drained or worker-idle check for normal grouped
delivery. Even with a backlog, each delivery takes at most B groups.

For grouped GRPO/OPD in `fully-async`, the Controller also derives the internal
`require_trainable_groups=true`. Only eligible complete groups count toward B:
at most half the members may fail, and a valid member must contain at least one
sequence with more than one token and an active target in `loss_mask[1:]`.
The trainer later replaces failed slots under the rule in the trajectory-pool
guide. Successful members with no training sequences remain valid, including as
replacement donors. Zero or equal rewards do not reject a group.
Rejected groups are summarized separately and do not occupy ready-group or
`max_ready_groups` backpressure slots. The pool keeps collecting until B eligible
groups are available; rejecting a group does not advance `train_step`. When
`rollout.max_ready_groups` is configured, fully-async uses at least B as the
watermark so backpressure cannot prevent filling one batch. See
[Trajectory Pool](../04-trajectory-pool/trajectory-pool.md#groupedtrajectorystore)
for discard metrics and logs.

The scheduler still controls input admission: `sync` and `batch-async` normally
release one B-group input batch at a time; `fully-async` can collect groups from
different input batches. The public RL loader cycles across epoch boundaries
to fill B inputs. For an underlying finite source, the pool confirms
`is_finished() is True` and all workers are quiescent. Fully-async grouped GRPO/OPD
then discards an undersized eligible tail without training or advancing
`train_step`; other grouped modes deliver the smaller complete-group tail.
Incomplete groups still raise an error when fewer than B groups are ready and
the current buffer is drained with workers quiescent.

Simple/SFT delivery instead waits for the current loader round to drain and
workers to become quiescent, then returns all trajectories from that round.
Its trajectory count can vary with the SFT token budget.

Other grouped modes still deliver failed/skipped groups and successful groups
without training sequences; these can consume a round without an optimizer
update. The fully-async guarantee is B eligible groups at pool admission per
trainer round. Later MIS rejection and OPD advantage-dependent filtering can
still reduce training data, and sequence/Datum counts need not equal B × R.

`algorithm.optim.batch_size` is a later optimizer boundary measured in converted
training Datums: a positive integer splits them into updates, while an omitted
value or `null` uses the entire delivered batch for one update in these three
RL modes. `algorithm.optim.mini_batch_size` limits each forward/backward request.
Neither setting defines loader input or GRPO group sizes. Optimizer splitting
can divide a group's Datums across updates; B is not a per-update group guarantee.
See [Remote API Trainers](../06-trainers/remote-api-trainers.md).

### rollout_repeat_n

`data.train.rollout_repeat_n` is a required positive integer for non-SFT
training. It is the number of independent rollout attempts for each draw of
an input, not the number of conversation turns or nested token sequences.

```yaml
data:
  train:
    rollout_repeat_n: 8  # Eight rollout attempts in each group
```

Each draw receives a fresh `group_id`. Its attempts have `run_id` slots
`0..R-1`, unique `rollout_id` values, and `expected_rollouts=R`.
`task_id` identifies the source task, so drawing that task again creates a
new group without reusing an earlier execution identity.

For SFT, omit this field; an explicit value emits a warning and is ignored.
Validation uses `validate.data.validation_repeat_n` independently.

### group_order

`data.train.group_order` controls which complete GRPO/OPD groups the trajectory
pool consumes first. Allowed values are `fifo` (default) and `lifo`. The Controller
passes this setting into the pool's internal `group_order`; configure the public
field in your recipe:

```yaml
data:
  train:
    group_order: lifo  # Prefer the most recently completed groups
```

Ordering starts when the last required rollout arrives and its whole group
enters the ready queue. If groups finish in the order A, B, C and the next batch
needs two groups, `fifo` selects A, B; `lifo` selects C, B. Neither option splits
groups or changes the `run_id` order within a group.

This changes batch membership whenever more than B ready groups are queued,
most commonly in `fully-async`. Every mode takes at most B groups; `sync` and
`batch-async` normally admit one B-group input batch at a time. The GRPO trainer
subsequently sorts delivered trajectories, so pool ordering does not guarantee
optimizer execution order.
"Most recently completed" does not mean most recently started or sampled with
the newest model weights. With a sustained backlog, `lifo` can leave older groups
waiting while newer completions are consumed. This setting does not affect the
ungrouped store used for SFT.

### keep_batch_order (internal)

This is derived by the Controller, not a public `data.train` setting:

| Mode | Internal value | When another source batch is exposed |
| --- | --- | --- |
| RL `sync` / `batch-async` | `true` | After the current round, through `release_next_batch()` |
| RL `fully-async` / `no-sync` | `false` | Automatically when `get_next_item()` exhausts the current task buffer |
| SFT | `true` | Through the SFT batch commit/release flow |

The first batch is fetched by `get_next_item()` without an explicit release.
Workers always claim one request at a time. Ordered loading does not mean one
worker receives an entire batch. Disabling this gate does not disable
GRPO/OPD group completeness in the trajectory pool.

### prompt_key and filter_prompts

`data.train.prompt_key` defaults to `prompt`. It selects the input column used
for loader prompt processing; it does not rename dataset fields for workers.
`data.train.filter_prompts` defaults to `false`. When enabled for RL, prompts
exceeding `data.max_prompt_length` are excluded before iteration.

For SFT, `filter_prompts` must remain `false`; overlong examples are truncated
using `data.max_sequence_length` instead.

### Length Limits and Tokenizer

- `data.max_prompt_length`: required positive prompt-token limit for RL; omit it for SFT.
- `data.max_sequence_length`: required positive total-sequence limit. For RL, it must exceed `max_prompt_length`.
- `models.actor.tokenizer`: tokenizer name or path; this is not `data.tokenizer_path`.
- `rollout.sampling.max_tokens`: optional generation limit. E2B preserves the agent's budget when omitted or null; an explicit value overrides it. Other inference paths derive the default from `max_sequence_length - max_prompt_length`. There is no `data.max_response_length` field.

### shuffle and seed

`data.train.shuffle` defaults to `true`. TorchDataLoader uses a dedicated
shuffle generator seeded with `experiment.seed + epoch_id` (`experiment.seed`
defaults to `42`). With `shuffle: false`, it cycles in source order. These
settings control dataset order; they do not promise identical asynchronous
completion order or identical training order across runs.

### drop_last

`data.train.drop_last` defaults to `true`. In a finite TorchDataLoader, it
drops a final source batch smaller than `batch_size`. Step-based RL uses an
internally derived `cycle` flag: it fills an epoch tail with inputs from the
next epoch, so every source batch is full and no tail inputs are discarded.
Do not add `cycle` to a public recipe.

Production validation loaders are finite and internally use
`shuffle=false`, `drop_last=false`, and `filter_prompts=false`.

## SFT Data

The Controller selects `SFTStreamingDataLoader` automatically:

```yaml
models:
  actor:
    name: Qwen/Qwen3-8B
    tokenizer: Qwen/Qwen3-8B
data:
  train:
    files:
      - ./train.jsonl
      - ./more-train.jsonl
    batch_size: 8
    shuffle: true
    drop_last: true
  max_sequence_length: 4096
algorithm:
  type: sft
validate:
  enabled: false
```

For SFT, `batch_size` multiplies the token budget
`data.max_sequence_length × data.train.batch_size`; it is not a fixed number
of rows, rollout groups, or Datums. The loader streams JSONL and requires
`shuffle=true` and `drop_last=true`. It drops the incomplete epoch tail rather
than filling it from the next epoch; in mixed mode (below), every non-empty EOF
optimizer-window tail is emitted instead, still without crossing the epoch
boundary. Weaver packs the independent Datums. For SFT, `max_sequence_length`
is the complete pre-shift model-bound limit and must be at least `2`.

Checkpoints save the committed source cursor. While batch N trains, the
loader prepares and tokenizes one lookahead batch. An optional
`data.train.render_chat` selects a renderer using `path/to/file.py:function`.

Omit the top-level `rollout` block. The Controller creates one internal
`SFTRolloutWorker`; an explicitly supplied block emits a warning and is ignored.
`algorithm.loss.mode` defaults to `cross_entropy` for SFT.
See [SFT Data](../03-data-loader/sft-data.md) for row formats and resume behavior.

### Mixing Local and Managed Sources

To mix local and authorized Weaver-managed data, replace `files` with
`total_samples` and `sources`:

```yaml
data:
  train:
    total_samples: 100000
    seed: 42
    managed_variant_ready_timeout_seconds: 300
    managed_variant_poll_initial_seconds: 1
    managed_variant_poll_max_seconds: 10
    sources:
      - type: local
        files: [./train.jsonl]
        ratio: 0.8
      - type: managed
        dataset: hq-math
        version: 2026-08
        ratio: 0.2
    batch_size: 8
    shuffle: true
    drop_last: true
  max_sequence_length: 4096
```

`ratio` is required for every source and is a positive explicit weight.
Catalog `recommended_ratio` remains a recommendation and is never silently
substituted into the run config. `total_samples` is the exact number of virtual
sample occurrences planned per epoch. Token-budget batching preserves every
non-empty final group. Managed `dataset` and `version` are always explicit;
NorthRL does not resolve `latest`. Each is preserved exactly and must be one safe URL path
segment: no boundary whitespace, slash, backslash, control character, `.` or
`..`; dataset names are limited to 160 characters and versions to 128. A
managed source is allowed only for SFT with the Weaver
backend. `render_chat`, when present, applies only to local sources.
Mixed mode requires `shuffle: true` because concat-then-global-shuffle is part
of the versioned deterministic plan.

The server returns one model-bound `input_token_count` for each managed
reference. NorthRL uses that integer to group complete samples and receives no
managed tokens, text, messages, masks, or segment metadata. Dataset name,
version, deterministic source plan, and committed cursor form the client-side
resume identity; the model remains fixed for the lifetime of one training run.
Preparation performs a catalog check and one opaque model-compatibility length
lookup for each managed source before iteration begins. Because a model-bound
variant is built asynchronously, the compatibility lookup waits for at most
`managed_variant_ready_timeout_seconds` (default `300`). It uses exponential
polling from `managed_variant_poll_initial_seconds` (default `1`) up to
`managed_variant_poll_max_seconds` (default `10`). Only Weaver errors with
`retryable=true` and the exact code `managed_dataset_variant_not_ready` or
`managed_datasets_unavailable` are retried; permission, missing-version,
protocol, and every other error fail immediately.
Managed references cannot supply the server-owned `model_input`, `target_tokens`,
`loss_mask`, or `weights`. Caller-owned inputs such as `sampling_mask` remain
attached to their original datum and preserve order through transport batching.

See [SFT Data](../03-data-loader/sft-data.md) for the mixing algorithm, resume,
and batching details.

## Validation Data

Validation uses named datasets under `validate.data.datasets`, with a separate
repeat count:

```yaml
validate:
  enabled: true
  schedule:
    before_train: true
    every_n_steps: 100
  data:
    datasets:
      held_out:
        files:
          - "${oc.env:NORTHRL_DATA_PATH}/test.parquet"
        batch_size: 16
    validation_repeat_n: 4
  aggregation: pass_at_k
```

Each dataset's `batch_size` counts original inputs before validation repeat
expansion. `validation_repeat_n` defaults to `1`; here it requests four attempts
per input for grouped evaluation. It does not use the training repeat count.
Validation shares the configured tokenizer and length limits. For SFT, set
`validate.enabled=false`; the current validation path requires an inference sampler.

## Troubleshooting

- **No tasks at startup:** call `get_next_item()` directly. `can_return_item()` only inspects the existing buffer and is initially `False`.
- **`None` between ordered batches:** check `is_finished()`; a pending release is not source exhaustion.
- **Unexpected rollout count:** compare `B` source inputs with `B × R` requests and complete outcomes.
- **Unexpected Datum count:** account for nested sequences, zero-sequence outcomes, and skipped groups. `drop_last` does not force a fixed number of training sequences.
- **File not found:** check `data.train.files`, environment variables, and explicit shard paths.

## Related Documentation

- [Data Loader](../03-data-loader/data-loader.md) — implementation and custom sources
- [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) — complete-group collection
- [Configuration Setup](../01-getting-started/configuration-setup.md) — environment variables
