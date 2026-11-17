English | [简体中文](../zh/metrics.md) | [Documentation index](index.md)

# Step metrics in W&B

Existing NorthRL training charts use **`training/train_step`: the number of batches consumed and completed by the trainer**. One step can contain zero or multiple optimizer updates. Runtime `live/*` snapshots remain in ordinary `[rollout-monitor]` logs and do not create W&B history rows.

The [metric definitions](../../northrl/utils/metric_definitions.py) describe units, meanings and collection stages. The [catalog](../../northrl/utils/metric_catalog.py) normalizes fields and configures W&B axes and summaries. Registration does not imply support by every worker or backend; only valid observations are recorded.

## Find metrics in the ordinary Run / Workspace

Open the run or project workspace after training logs to W&B. No additional command is needed. The tables below use canonical metric names; W&B adds stable display ranks, for example `algorithm/trajectory/reward_mean` becomes `01_algorithm/01_trajectory/reward_mean` and `timing_rollout/` becomes `06_timing_rollout/`. Other logging backends keep canonical names.

| Prefix | Contents |
| --- | --- |
| `batch/` | Input and trainable groups, trajectories, segments, total tokens and response tokens; sandbox retries |
| `trajectory/` | Saved response length and segment/tool/request counts |
| `segment/` | Saved response length, reliably assigned tool/timeout events, actual clipping |
| `group/` | Min/max across group quantity totals, valid and singleton group counts |
| `algorithm/` | Reward, advantage, reference KL, entropy, PPO/IS, sampling logprobs/PPL, MIS and OPD |
| `training/` | Completed updates, forward/backward calls, skip and weight-publication flags |
| `model_request/` | Individual request input/output lengths and counts |
| `tool/` | Calls, observed results, errors and calls without observed results |
| `timeout/` | Tool, model-request and rollout timeouts and affected objects |
| `verifier/` | Reward evaluation outcomes, failures and E2B timeouts/retries |
| `reward_fn/` | Reward outcome counters by effective reward function |
| `outcome/` | Task/environment outcomes, context limits, clipping and additional task scores |
| `weaver/` | Service loss, gradient norm, token/sample counts and applicable update status |
| `timing/` | Completed-batch interval and trainer wait |
| `timing_rollout/` | Trajectory/group/verifier/model-request duration distributions, including reward-function cohorts |
| `timing_train/` | Training total, logprob/diagnostic forwards, FB and optimizer wall/compute time |
| `timing_weights/` | Weight export, checkpoint saving, synchronization and failure durations |
| `timing_val/` | Validation duration distributions, including reward-function cohorts |
| `val/` | Validation counts, scores and pass@k |

Automatic W&B workspaces group charts by the prefix before the first `/`, so numbered algorithm fields belong to `01_algorithm`; see [W&B section organization](https://docs.wandb.ai/models/track/workspaces#section-organization). Data fields are not hidden; only the `training/train_step` axis field is hidden from automatic charts. A small set of headline metrics keeps latest-value summaries; other metrics remain in history without additional summary aggregates.

### Workspace order at startup

After `wandb.init`, NorthRL configures **both the project page and Run page personal workspaces** for the user owning `WANDB_KEY`, within the current run's entity and project. These are two separate saved views. The run view is shared across that user's runs in the project; the change can therefore also appear when viewing older runs. Another user's personal workspace, saved named views and other projects are unaffected.

The initializer reserves the catalog's 25 numbered sections (`01_algorithm` through `25_group`) so later-arriving metrics fill their positions. It sorts visible sections by name, removes their pins, enables first-prefix grouping and alphabetical ordering inside flow sections, and hides empty sections. `Charts` and `System` are reserved too; W&B's hidden-panel bank stays last. Existing chart definitions, collapsed states, filters, run selections, hidden panels and custom grid geometry are retained. Existing manual chart-generation settings are respected. Mean/min/max remain separate charts.

Ordering runs automatically whenever W&B tracking starts; no configuration is needed.

No additional dependency or command is required. Offline/disabled runs skip the API; an unsupported server API, missing write permission or request failure logs a warning and training continues. Repeated startup only writes when ordering needs an update. This uses W&B's internal view API and does not change training configuration or history. It re-reads before saving and verifies afterward, but the API has no atomic revision check: a simultaneous edit or a stale browser tab's later autosave can overwrite the order. It is startup initialization, not a background layout enforcer.

### Timing sections

Timing uses five sections. The fixed training catalog contains 27 fields: two overview timers, eight training timers, five weight/save timers and 12 rollout distribution fields. Reward-function and validation distributions are added where applicable. Every timing field has one `/`; distribution `*` means `mean`, `min` or `max`.

| W&B field | Meaning |
| --- | --- |
| `timing/batch_completion_interval_seconds` | Time between consecutive `train()` returns: step `n` reports `finish(n) - finish(n-1)` using the trainer process's monotonic clock. |
| `timing/trainer_wait_for_batch_seconds` | Observed wait for a complete batch, including polling and scheduling delay. An immediate claim records 0. Initialization, initial validation, training, saving and synchronization are excluded. |
| `timing_train/batch_training_seconds` | Standard trainer wall time from entry into training until saving begins. Includes data processing, recomputation and FB/optimizer; excludes weight export, checkpoint saving and weight synchronization. |
| `timing_train/recompute_logprobs_seconds` | Client wall time for trainer logprob recomputation, including request waiting. |
| `timing_train/ref_logprobs_seconds` | Client wall time for reference-model logprobs. |
| `timing_train/mismatch_forward_seconds` | Client wall time for a separate mismatch diagnostic forward, excluding subsequent statistics. |
| `timing_train/forward_backward_wall_seconds` | Client wall time for all FB phases in this step, including Datum preparation, submission, queuing and result polling. Pipelined FB requests use the whole phase span. |
| `timing_train/forward_backward_compute_seconds` | Sum of complete, valid FB service `compute_ms` observations divided by 1000. |
| `timing_train/optimizer_step_wall_seconds` | Client wall time for optimizer submission and completion, summed across updates; starts after every FB succeeds. |
| `timing_train/optimizer_step_compute_seconds` | Sum of complete, valid optimizer service `compute_ms` observations divided by 1000. |
| `timing_weights/weight_export_seconds` | Client wall time for successful sampling-weight export. |
| `timing_weights/checkpoint_seconds` | Client wall time for a successful checkpoint save performed in this step; failed saves do not report this field. |
| `timing_weights/weight_sync_seconds` | Client wall time for sampling-weight synchronization after `train()` returns. |
| `timing_weights/weight_export_failed_seconds` | Time until a sampling-weight export fails. |
| `timing_weights/weight_sync_failed_seconds` | Time until sampling-weight synchronization fails. |
| `timing_rollout/trajectory_*_seconds` | Execution duration distribution for individual complete rollouts. |
| `timing_rollout/group_*_seconds` | Complete sampling-group execution-span distribution. |
| `timing_rollout/verifier_*_seconds` | Reward-evaluation duration distribution, excluding answer generation. |
| `timing_rollout/model_request_*_seconds` | Observed answer-generation request duration distribution. |

Completion interval has the same definition for sync, batch-async, fully-async, no-sync and SFT. It includes the previous round's synchronization, validation, logging and remaining data wait, plus this round's complete `train()` call, including export and checkpoint work. Work after this return belongs to the next interval. The first return after startup or resume establishes a baseline and omits the interval. Filtered/skipped rounds still complete a step; compare optimizer-update counts when measuring update throughput.

A validation pause during batch waiting is excluded from trainer wait; waiting observed before the pause is retained. Custom trainers without a separate training-phase timer fall back to the complete call duration because they do not expose standard saving boundaries.

The training total is not the sum of the displayed stages: it also includes data processing and other work without separate timing curves, and some stages overlap or contain other work. Service compute time and client wall time are separate observations of the same operation and must not be added together. `compute_ms` is service-reported computation time, not a verified pure-GPU timer. Missing, invalid or incomplete compute observations omit that step's compute total; no zero is invented. A successful skipped batch can report zero client work while providing no compute observation.

Reward-function duration cohorts remain under their execution stage, using `timing_rollout/<reward_fn>_<object>_<reducer>_seconds`; validation uses `timing_val/<object>_<reducer>_seconds` and `timing_val/<reward_fn>_<object>_<reducer>_seconds`. Valid observation counts remain internal denominator metadata and ordinary diagnostics, rather than separate W&B timing fields.

Batch rollout span and separate MIS/mismatch-statistics/preparation/other timers are retired; their algorithms still run. Same-meaning producer and previous timing names normalize to current sections. The old `timing/batch_training_seconds` and `timing/training_time` included saving, so normalization drops them instead of relabeling them as the new training-only total. Old `timing/mismatch_forward` also included subsequent statistics and is dropped instead of mapping to the narrower forward-only timer. Restart processes to use new collection code; historical runs are not backfilled. See [rollout and trainer timing](developer-guide/08-features/rollout-timing-metrics.md).

### Algorithm section

Algorithm metrics share the first-level prefix `algorithm/`. The next component preserves the measured object or source so that cohorts and collection stages remain clear:

| Fields | Contents and interpretation |
| --- | --- |
| `algorithm/trajectory/reward_*` | Finite final reward per complete trajectory: mean/min/max/std and observation count |
| `algorithm/group/reward_std_min,max` | Compute sample reward std within each valid GRPO group, then compare extrema across groups |
| `algorithm/trajectory/advantage_*` | Valid GRPO complete trajectory advantage distributions, positive/negative/zero counts, zero percentage and token-weighted mean |
| `algorithm/kl/actor_ref_mean`, `actor_ref_loss`, `coefficient` | Actual training-forward reference KL, explicit service-reported loss contribution and coefficient |
| `algorithm/weaver/entropy_coeff`, `entropy_loss` | Entropy coefficient used by the service and its reported entropy-loss term |
| `algorithm/weaver/entropy`, `entropy:sum`, `entropy:tokens` | Full-vocabulary policy entropy in nats at temperature 1, its sum and effective masked token count |
| `algorithm/weaver/ppo_clip/*` | PPO clipping fractions and current-policy/fixed-denominator KL estimate |
| `algorithm/weaver/is:*` | Service probability ratios, log ratios, squares, extrema and related statistics |
| `algorithm/weaver/ratio:*`, `algorithm/weaver/logprob_diff:*` | Custom-loss ratio and logprob-difference diagnostics, interpreted using each field's definition |
| `algorithm/sampling/*` | Original pre-update sampling logprobs and segment-level PPL; these do not measure post-update model quality |
| `algorithm/mis/*` | Pre-update trainer/original-rollout probability mismatch, weights and filtering |
| `algorithm/opd/*` | Aligned student/teacher probability diagnostics and observed token count |

Abbreviated suffixes in a row use that row's full prefix. The metric dictionary lists every exact field, unit and denominator. Only the new names are published, with no duplicate legacy curves; calculations and submission timing are unchanged. Formal step logs also use the new names, while raw service-response diagnostics retain their source fields. `weaver/loss`, `weaver/grad_norm` and `training/*` keep their names outside Algorithm.

Weaver's `entropy` diagnostic is independent of `entropy_coeff` and `loss.reduction.mode`; supported service versions return it even when the coefficient is zero. Across forward/backward requests and optimizer updates, NorthRL adds `entropy:sum` and `entropy:tokens`, then divides the totals instead of averaging request means. An explicitly reported empty mask retains all three zeros; missing observations remain missing. The W&B keys are `01_algorithm/56_weaver/entropy`, `01_algorithm/57_weaver/entropy:sum` and `01_algorithm/58_weaver/entropy:tokens`. Existing `entropy_loss` and `entropy_coeff` names and meanings remain unchanged.

Training started with the new code groups these metrics under `01_algorithm` when automatic charts are enabled. Startup workspace ordering also sorts the numbered algorithm panels. Already-running old processes do not pick up code changes; historical metric data is not backfilled.

See [W&B metric definitions](https://docs.wandb.ai/models/ref/python/experiments/run) for axis and visibility settings. They do not migrate historical runs or rearrange existing workspaces.

## Units, lengths and cohorts

`batch/tokens_total` (W&B: `15_batch/tokens_total`) sums the full lengths of all
locally saved sequences in the current step, including response tokens and
non-response positions such as prompts and tool context. Context repeated across
segments counts once per saved segment. Like `batch/response_tokens_total`, it
describes the original consumed batch before resampling, training filtering and
optimizer padding; it resets each step. Empty sequences and token-free managed
SampleRef carriers contribute zero local tokens, so this is not a managed-SFT
service workload counter.

- **Model request:** input includes that request's full history and tool results; output is only its newly generated output. Each observed request has equal weight. Missing usage is omitted, not zero.
- **Segment:** response length counts saved positions with `loss_mask > 0`.
- **Trajectory:** response length sums those counts across its saved segments; it is not all tokens ever generated during execution.
- **Group:** a model-scoped sampling group identified by `(model_tag, group_id)`. Sum data quantities within each group, then take min/max across completely observed groups; quantity totals such as response tokens have no group mean.
- **Trajectory duration:** first execution start to final result, including preparation, model/tools, evaluation, cleanup and retries; excludes initial queueing and subsequent pool waiting.
- **Group duration:** `timing_rollout/group_*_seconds` (`mean`, `min`, `max`) measures earliest member worker start to latest member finish using comparable clocks. Only complete expected groups with valid timestamps and, when supplied, complete unique `run_id` slots contribute. The mean weights valid groups equally; it does not sum member durations. Segments have no independent duration.

Group is the sampling abstraction, rather than a separate task object. In fully-async mode, the training-step axis identifies the completed batch that consumed a group or trajectory. Its execution can overlap earlier training steps, so execution durations do not measure the trainer's wait for data or a rollout phase within that step. Use `timing/trainer_wait_for_batch_seconds` for the wait.

Input counts and raw rollout metrics describe original complete trajectories, including failed,
skipped and empty results, before failed-slot replacement. A nonempty ordinary
trajectory contributes one segment. Advantage statistics use repaired GRPO groups.
`batch/resampled_group_count` counts groups repaired in this batch;
`batch/resampled_trajectory_count` counts their replaced failure slots. Trainable
counts include replacement members and are measured after Datum/mask processing,
before optimizer padding duplicates; these cohorts have different denominators.

`count` means objects/events; `seconds` means seconds. New `percentage` values range from 0 to 100; existing MIS `fraction`, custom-loss `frac` and validation ratios remain 0–1. A probability ratio is a multiplier, not a percentage. Missing/inapplicable fields and ratios without valid denominators are omitted. Confirmed applicable zero events remain zero. Reward, advantage and sampling-logprob standard deviations use `ddof=0`; within-group reward std uses `ddof=1`, with singleton groups assigned zero.

## Generation, reward evaluation and rollout timeouts

Answer generation, reward evaluation and the overall rollout deadline are separate stages. An LLM judge called by a reward function belongs to reward evaluation; its API failures do not count as answer-generation model-request timeouts.

| Stage | Metric | Coverage |
| --- | --- | --- |
| Answer-generation request | `timeout/model_request_count` | Observed logical `generate` calls whose final failure is recognized as a timeout. Recovered backend-internal timeouts do not count. SingleTurn validation does not aggregate this metric. |
| Reward E2B operation | `verifier/timeout_count` | Explicitly reported sandbox timeouts: CodeE2B reports `run`; IfVerify/VerInstruct hard checkers also report creation and upload. |
| Exhausted reward timeout retries | `verifier/timeout_after_retry_count` | Complete trajectories assigned reward 0 after reward timeout retries are exhausted, rather than timeout events. |
| Reward LLM judge | `verifier/judge_error_count` | Results that end in `judge_error`; Reasoning360 API timeout, HTTP 400 and network errors are not separately counted yet. |
| Overall execution deadline | `timeout/rollout_count` | Requires an explicit worker observation; it is not the sum of generation and reward timeouts. |

A reward `run` timeout identifies the stage, not the underlying cause: slow generated code, a stalled judge/checker or sandbox communication can all prevent completion. A program test-case TLE returned normally by the judge has no dedicated aggregate yet; inspect the original test results. `finish_reason=length` is a token limit, not a timeout. See [generation and reward failures](developer-guide/08-features/verifier-metrics.md) and [group/trajectory/verifier durations](developer-guide/08-features/rollout-timing-metrics.md) for definitions and limits. Validation reports `val/verifier/*`, but there is currently no `val/timeout/model_request_count`. Exceptions that stop training may prevent a completed-step W&B row; a missing metric is not zero.

Tools, model requests and their timeouts include only the execution that produced the delivered data. Sandbox retry counts and complete trajectory duration cover all attempts. Deduplication includes rollout, execution and call identity. Events without reliable segment attribution remain at the complete trajectory level; they are not copied across segments.

`timeout/tool_count`, `timeout/model_request_count` and `timeout/rollout_count` represent separate layers and must not be added. Affected complete trajectories/segments have distinct deduplicated counts. `segment_from_timed_out_rollout_count` counts saved segments carried by timed-out complete trajectories, not segment-local timeout events. Timeout totals remain in `timeout/*`; no duplicate `batch/*` aliases are recorded. Unobserved tool results are not assumed to be timeouts.

The runtime API adapters observe tool results from the history supplied to a subsequent model request. A terminal tool result that is never sent in a later request may remain unobserved. `timeout/tool_count` counts only confirmed timeouts in observed tool results; zero does not establish that every emitted call completed without a timeout. `tool/unobserved_call_count` counts emitted calls with no observed result and is not a timeout count. Workers without reliable metadata do not receive fabricated zero-valued tool or timeout curves.

The runtime checkpoints the trace during model-request processing and performs a final flush at shutdown. The final flush persists observations already collected; it cannot recover an unseen terminal tool result.

## NorthGate tool-call diagnostics

E2B rollouts consume the top-level `tool_call_diagnostics` in both JSON and SSE responses. `checked` and `malformed` feed `tool/format_unknown_turn_count` and `tool/format_error_turn_count`; native structural errors can also prove an error. This is parser/structure checking, not full JSON Schema validation. Responses without this field retain the legacy parser diagnostics path.

`detected` and `invalid` feed `tool/calls_detected_count` and `tool/calls_invalid_count`. These raw-output counts include repaired or discarded calls and are not added again from the returned tool list. A null count contributes no estimated calls and increments `tool/call_counts_unknown_turn_count` once per response, even when `malformed=true` already proves an error. This coverage counter applies only to NorthGate diagnostics; the call totals represent the known portion.

`tool/format_reason_<reason>_turn_count` counts responses containing each fixed reason: `incomplete_tool_call`, `invalid_arguments_json`, `missing_tool_name`, `invalid_tool_call_structure`, `arguments_not_object`, `duplicate_parameter`, or `undeclared_tool`. Each reason counts at most once per response; reasons can overlap and are not invalid-call counts. Unknown reasons do not create dynamic metrics. Existing `tool/arguments_json_error_count` and `tool/arguments_type_error_count` still describe local checks of returned arguments.

These step counters cover the consumed complete trajectories' final executions. The corresponding `live/tool/*_turns` counters cover all observed attempts and retries; see [live metrics](developer-guide/08-features/live-metrics.md). Diagnostics only affect metrics, with no reward, advantage, or tool acceptance changes.

## Service, MIS and validation metrics

Raw service responses remain in diagnostic logs. Formal aggregates require suitable reduction semantics: counts sum, extrema combine as extrema, means/ratios require weights or numerators and denominators, and std requires sufficient statistics. Multiple PPO/IS shards without explicit mean weights do not produce an average of shard averages; extrema can still be retained.

Reference KL uses actual target-aligned training-forward response positions and the configured estimator. `algorithm/kl/actor_ref_mean` is token-weighted across requests/updates, including padded or repeated data actually submitted to the service. This differs from Batch's unique-data counts. `algorithm/kl/actor_ref_loss` requires an explicit service-reported loss contribution; it is not inferred as coefficient × mean. Missing forward observations do not generate KL curves, and ambiguous raw `kl_loss` is not renamed. Entropy and PPO/IS use `algorithm/weaver/*` to preserve their service source; loss and gradient norm keep `weaver/*`. Multiple optimizer `grad_norm` observations are averaged equally over observed successful updates; pre/post-clipping interpretation belongs to the service contract.

Actor/reference KL, current/fixed-denominator PPO/IS, and pre-update-trainer/original-rollout MIS compare different policy pairs. PPO/IS aliases are merged only when policies, masks, weights and reducers are known to agree. The catalog does not infer equivalence from nearby numerical values. The verified internal alias `mis/log_ppl_diff` is published as `algorithm/mis/kl`; `val/<dataset>/avg@k` no longer adds a curve duplicating `pass@1`. Custom validation metrics remain supported without a global whitelist.

Validation belongs to the validation dataset and is located at its associated completed training step; initial validation can use step 0. Each finite score has its own observation denominator. MIS ratios and distributions without observations remain missing; `algorithm/mis/optimizer_steps` counts planned updates, while `training/optimizer_step_count` records completed updates. OPD observations require aligned student/teacher probabilities and the relevant algorithm path.

## Submission and logs

`timing/batch_release_to_claim_seconds` and `timing/batch_claim_interval_seconds` are retired: neither is collected or registered, and normalization drops either field if supplied by an old producer. Use `timing/batch_completion_interval_seconds` for completion throughput and `timing/trainer_wait_for_batch_seconds` for observed pool waiting, including the first batch. Missing timing from older processes is not backfilled.

Worker observations travel with their trajectories. Batch membership is fixed at consumption. Preparation, recompute/MIS, Datum and service metrics are merged into the completed step; weight synchronization adds to that pending row. Validation commits the row when scheduled; otherwise the trainer flushes it. Normal execution writes one history row per completed batch. Late metrics cannot reopen committed steps. Fully filtered batches still advance the step with zero optimizer updates.

Local summaries print integer counts and enough significant digits to retain tiny losses. Stage logs identify the completed-step number and do not advertise uncomputed trainable counts as final zeros. Approximately 30-second runtime snapshots count all attempts in the current process and are distinct from the consumed batch's final-execution observations. Debug dump filenames retain the historical zero-based batch index; the round manifest separately carries `training_step`.

Retired `rollout/*`, `datum/*`, `round/*`, ambiguous length fields, reference coverage/sync counters, recompute counters, MoE, scale, denominator internals and unknown raw backend fields do not become new formal step curves. Historical runs are not backfilled.

## Compatibility

| Mode | General workers | E2B |
| --- | --- | --- |
| sync | Batch-owned data and supported worker observations | Supported; final-execution events and all-attempt durations/retries stay distinct |
| batch-async | Belongs to the actually consumed batch | Rejected by E2BRolloutWorker |
| fully-async | Consumed complete groups determine step ownership; no cumulative-counter differencing | Accepted; step metrics use the complete groups consumed by training |
| SFT | Applicable Batch, Training, Timing and service metrics; no fabricated GRPO groups/advantages | Outside the current E2B recipe |

Group timestamps must share comparable clock domains. These statements describe metric paths and current restrictions, not completed end-to-end validation of every service configuration.

## Additive Weaver metrics-v1

With the Weaver backend and W&B enabled, every emitted v1 observation is also
published under `weaver_metrics_v1/`, reusing the SDK's scalar, boolean, status-table and expert-histogram formatting. These are not inputs to NorthRL's legacy
metric aggregation. Each completed operation retains its own observation row;
multiple optimizer windows are never averaged into one top-10% entropy value.

V1 charts use `weaver_metrics_v1/step` (Trainer optimizer attempts, including
skipped updates), not `training/train_step` (NorthRL completed batches). Forward-only
observations use an operation-index axis. Multiple models/restarted Trainers get
separate short prefixes, with their identities in run config. Existing NorthRL
charts remain unchanged. W&B's internal `_step` is a history-row cursor, not a
training counter; it can now advance more than once per NorthRL batch.

The SDK callback queues observations without W&B I/O. NorthRL publishes them after
its existing round commit (or on shutdown), so partial RL rows are not flushed
prematurely. No metric-name filter limits this to defaults. Local JSONL persistence
is independent and preserves all original names, values, labels and statuses.
See [configuration](developer-guide/11-configuration-reference/trainer-config.md#weaver-metrics-v1)
for `backend.metrics_store` and the registered Trainer's collection YAML.
