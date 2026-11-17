English | [简体中文](../../../zh/developer-guide/08-features/rollout-timing-metrics.md) | [Documentation index](../../index.md)

# Batch rollout, trainer waits, and execution timing

This page explains the scope of durations in [training metrics](../../metrics.md). Each training step summarizes its claimed raw batch before GRPO filtering and optimizer padding. Each parent trajectory counts once, regardless of segment count. In fully asynchronous training, `training/train_step` identifies the completed batch that consumed the data, not when sampling started or finished. Sampling can overlap earlier training steps.

<a id="batch-timing"></a>

## Start with completion intervals and waiting

`timing/` contains only `batch_completion_interval_seconds` and `trainer_wait_for_batch_seconds`. The first is the interval between consecutive `train()` returns: previous synchronization, validation, logging, data waiting, and this step's full training and saving. The first return establishes a baseline without reporting zero. The second measures observed waiting from finding no complete batch until claiming one, including polling and scheduling delays. An immediate claim reports 0; initialization, validation pauses, training, and weight synchronization are excluded.

`timing_train/` describes training and its stages; `timing_weights/` covers export, checkpoints, and synchronization. `timing_rollout/` and `timing_val/` describe sampling and validation distributions. The fixed training timing set has 27 fields: two overview fields, eight training fields, five weight/save fields, and 12 rollout distribution fields. NorthRL configures personal workspace section ordering at startup; see [workspace order](../../metrics.md#workspace-order-at-startup) for its scope and retained settings.

<a id="execution-distributions"></a>

## Trajectory, group, and evaluation durations

The tables use canonical metric names; W&B adds stable display ranks as described in [step metrics](../../metrics.md). Reward-function and validation breakdowns are added where applicable. Timing names use one `/` between the section and name; distributions use `<object>_<mean|min|max>_seconds`. Raw worker metadata and `[timing-metrics]` logs can retain producer names.

| W&B metric | Scope |
| --- | --- |
| `timing_rollout/trajectory_mean_seconds` | Mean complete execution time per parent trajectory, including generation, tools, evaluation, conversion, cleanup, internal waiting, and retries. Excludes queues before the worker and pool waiting after completion. |
| `timing_rollout/group_mean_seconds` | Latest finish minus earliest start within each `(model_tag, group_id)` sampling group, averaged equally across valid complete groups. Includes staggered member starts; the same group ID under different models is separate. |
| `timing_rollout/verifier_mean_seconds` | One complete SingleTurn reward evaluation, including a reward function's LLM judge, sandbox creation, execution, cleanup, and retries. Excludes answer generation. |
| `timing_rollout/model_request_mean_seconds` | Observed answer-generation request duration, including internal waits, transport, and retries. Each observed request has equal weight. |
| `verifier/skipped_length_count` | Trajectories explicitly skipping evaluation because of `finish_reason=length`. They still contribute total execution time but not a zero-second verifier observation. |

All four duration distributions also publish `_min_seconds` and `_max_seconds`. Means use only actual, finite, nonnegative observations. Missing observations are omitted, not replaced with zero. Valid timing counts remain internal and in raw diagnostics as denominators; they are not extra W&B timing counters. Existing batch, group, model-request, and reward counts remain unchanged and cannot substitute for valid timing counts when calculating means.

A group is a sampling group, not a separate task abstraction. Group timing requires valid, comparable start/end timestamps for every member. If `expected_rollouts` is provided, it must match group size; provided `run_id` values must identify complete, unique sampling slots. Incomplete or duplicate groups are excluded. Missing, empty, or nonfinite numeric group IDs are not real groups. SingleTurn gives independent group identities to validation tasks using default single sampling without a valid group ID. `timing_rollout/group_mean_seconds` weights groups equally; group-total quantities such as `group/response_tokens_*` still publish only group minima/maxima, not a mean.

For example, trajectories running over `[100, 110]` and `[105, 120]` have a group duration of 20 seconds and a mean trajectory duration of 12.5 seconds. Neither their summed duration of 25 seconds nor batch span divided by trajectory count is the group duration.

Effective reward-function breakdowns stay in the rollout section as `timing_rollout/<reward_fn>_<object>_<mean|min|max>_seconds`, such as `timing_rollout/code_e2b_group_mean_seconds` or `timing_rollout/if_verify_group_max_seconds`. Batch-wide trajectory/verifier means weight actual observations equally; group means weight valid complete groups equally. They are not averages of per-reward-function means.

Validation uses the same short names under `timing_val/`, such as `timing_val/trajectory_mean_seconds`, `timing_val/group_mean_seconds`, and `timing_val/code_e2b_verifier_mean_seconds`. SingleTurn validation currently collects total and verifier durations, but does not aggregate `timing_val/model_request_*_seconds`. Results from workers without explicit verifier timing do not contribute to verifier means.

<a id="training-and-weights"></a>

## Training, weight saving, and overlap

| W&B metric | Scope |
| --- | --- |
| `timing_train/batch_training_seconds` | Standard trainer client wall time from entry until saving begins: data processing, recomputation, FB, and optimizer work. Excludes export, checkpoints, and synchronization. Custom trainers without standard stage timing fall back to the whole call. |
| `timing_train/recompute_logprobs_seconds` | Training-side log-probability recomputation client wall time, including request waits. |
| `timing_train/ref_logprobs_seconds` | Reference-model forward-pass client wall time. |
| `timing_train/mismatch_forward_seconds` | Separate mismatch-diagnostic forward-pass wall time, excluding subsequent statistics. |
| `timing_train/forward_backward_wall_seconds` | Total FB phase client wall time, including Datum preparation, submission, service queuing, network, and result polling. Pipelining uses the span from before first submission until all results complete, without double-counting overlapping waits. |
| `timing_train/forward_backward_compute_seconds` | Sum of complete, valid FB service `compute_ms` observations divided by 1000. |
| `timing_train/optimizer_step_wall_seconds` | Client wall time from optimizer submission, after all FB work succeeds, until results finish; multiple updates are summed. |
| `timing_train/optimizer_step_compute_seconds` | Sum of complete, valid optimizer service `compute_ms` observations divided by 1000. |
| `timing_weights/weight_export_seconds` | Client wall time for successful sampler-weight export. |
| `timing_weights/checkpoint_seconds` | Client wall time of a successfully completed checkpoint save in this step; omitted on save failure. |
| `timing_weights/weight_sync_seconds` | Sampling-weight synchronization duration after `train()` returns. |
| `timing_weights/weight_export_failed_seconds` | Time until a sampling-weight export fails. |
| `timing_weights/weight_sync_failed_seconds` | Time until sampling-weight synchronization fails. |

Total training time is not the sum of stages: other work such as data processing has no separate curve, some stages include other work, and sampling overlaps training. Service compute and client wall time describe the same operation differently and cannot be added. Service `compute_ms` is not guaranteed to be GPU-only time. Missing, nonfinite, negative, or incomplete observations are not zero-filled, and partial observations are not presented as the step's total compute time.

<a id="legacy-fields"></a>

## Legacy fields and missing observations

Batch rollout spans, old release/claim intervals, and separate MIS/mismatch-statistics, prepare, and other durations are retired and no longer collected or registered; their algorithms and statistics still run. Normalization drops these older producer fields. Legacy `timing/batch_training_seconds` and `timing/training_time` included saving, so they are dropped rather than renamed to the new training total. Legacy `timing/mismatch_forward` included subsequent statistics and is also dropped rather than mapped to forward-only timing. Other synonymous old names migrate into the current sections.

A long duration does not establish a timeout, and `finish_reason=length` is not a timeout. Generation-request timeouts use `timeout/model_request_count`; E2B reward-evaluation timeouts use `verifier/timeout_count`. An LLM judge belongs to evaluation rather than answer generation. See [generation and evaluation failures](verifier-metrics.md).

These are batch or validation snapshots, not process totals. Interrupted training might not produce a completed-step W&B row; initial validation can appear at step 0. Restart training processes to load changed code. Normalization maps raw producer names and earlier `timing/<object>/duration_seconds_*`, `timing/reward_fn/...`, and `val/timing/...` names into current sections, without duplicating retired duration names or historical `task` aliases. It does not backfill historical curves; startup workspace ordering is a separate operation described above.

See [synchronous training semantics](../06-trainers/sync-rollout-semantics.md) for batch boundaries and sampling versions, and [runtime monitoring](live-metrics.md) for process-wide cumulative observations.
