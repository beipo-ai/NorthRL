English | [简体中文](../../../zh/developer-guide/08-features/live-metrics.md) | [Documentation index](../../index.md)

# E2B runtime monitoring

`live/*` snapshots describe running work in the host process. They appear in ordinary logs as `[rollout-monitor] elapsed_seconds=... metrics={...}` and **are not sent to W&B**. See [step metrics in W&B](../../metrics.md) for training fields, aggregation rules, and chart navigation. E2B sandboxes do not need W&B credentials.

<a id="update-intervals"></a>

## Defaults and update intervals

Collection is enabled by default. The sandbox updates its heartbeat about every 30 seconds and on startup, stage transitions, and completion. Host polling attempts to read `metrics.json` immediately, waits five minutes after each read finishes, and reads once more during final cleanup. Each read has a five-second timeout. Monitoring logs still publish about every 30 seconds using the latest received snapshot. Background publication runs separately from the control thread; intermediate snapshots may be coalesced. Configuring a W&B backend does not change the log-only behavior of `Tracking.log_live`.

`elapsed_seconds` measures time since the monitor started. Cumulative values cover every attempt in this process, including failures and retries; active values describe the current snapshot. They do not belong to a `training/train_step` or create W&B history rows.

<a id="terminology"></a>

## Terminology

- **Rollout / trajectory:** one logical task identified by `rollout_id`. The [trajectory contract](../04-trajectory-pool/trajectory-semantics.md) submits one complete `Trajectory` containing zero or more `segments`.
- **Attempt:** one execution attempt. Retrying a rollout creates another attempt.
- **Active rollout:** a task whose first attempt has started but whose final result has not arrived. Preparation, agent execution, evaluation, cleanup, and retry waits all count. Success, task failure, environment error, or cancellation ends this interval.
- **Model turn:** one complete model result returned by the inference client.
- **Segment / sequence:** a token sequence returned as a `RemoteSegment` and saved in `Trajectory.segments`. Training eligibility depends on the parent trajectory status and validity of its whole group. One rollout can have several segments; one segment can contain several model turns.
- **Active attempt:** an attempt that has started and not ended. Only attempts with a received metrics snapshot contribute to active-length statistics.

<a id="sandbox-liveness"></a>

## Sandboxes and liveness observations

| Metric | Meaning |
| --- | --- |
| `live/e2b/active_sandboxes` | Created sandboxes whose cleanup is not confirmed and whose configured creation-time TTL has not expired. This does not prove they are reachable or running normally. |
| `live/e2b/active_rollouts` | Unfinished logical rollouts, deduplicated by `rollout_id`, from the first attempt until the whole rollout ends. |
| `live/e2b/unknown_sandboxes` | Sandboxes within their TTL with a failed metrics read or unconfirmed cleanup. A missing metrics file before the first heartbeat counts only as pending. |
| `live/e2b/heartbeat_pending_sandboxes` | Sandboxes without an observed initial runtime heartbeat, including those still preparing their environment. |
| `live/e2b/heartbeat_age_seconds_max` | Longest time since an observed heartbeat advanced. Reading the same unchanged file does not reset this age. |
| `live/e2b/sandbox_age_seconds_max` | Longest time since the host observed creation, among sandboxes without confirmed cleanup. |
| `live/e2b/ttl_remaining_seconds_min` | Smallest estimated remaining lifetime. Startup and environment preparation also consume the configured TTL. |
| `live/e2b/attempt_count` | All execution attempts, including those producing no trajectory. |
| `live/e2b/retry_count` | Additional attempts, excluding each rollout's first attempt. |
| `live/e2b/completed_rollouts` | Finished logical rollouts, including environment errors and cancellation; each counts once. |
| `live/e2b/successful_rollouts` | Rollouts with final E2B status `success`: execution completed, which does not imply SWE resolved. |
| `live/e2b/task_failed_rollouts` | Valid task-failure samples with final status `task_failed`. |
| `live/e2b/valid_rollouts` | Rollouts ending as `success` or `task_failed`, including zero-segment results. Their GRPO groups may still be ineligible for training. |
| `live/e2b/overlength_ratio` | Rollouts ending because of `context_length_exceeded` divided by all finished rollouts; not a token-clipping ratio. |
| `live/e2b/env_error_rate` | Rollouts ending in environment error divided by all finished rollouts. |
| `live/e2b/timeout_count` | Timed-out attempts, including timeouts before sandbox creation. |
| `live/e2b/timeout_sandbox` | Created sandboxes whose attempt ended in timeout, counted when the attempt ends. Retry sandboxes count separately, regardless of cleanup success. Pre-creation timeouts do not count. This includes execution or request timeouts, not just sandbox TTL expiry. |
| `live/e2b/timeout_rate` | Timed-out attempts divided by finished attempts; retries count separately. |

If a rollout's first attempt fails and waits for a retry, `active_rollouts` remains 1. Starting the retry adds 1 to both `attempt_count` and `retry_count`. Only the final rollout result reduces `active_rollouts`.

Confirmed shutdown or expiry of the creation-time TTL removes a sandbox from active counts, heartbeat ages, and remaining-TTL statistics. Cumulative `live/e2b/cleanup_error_count` survives that removal. A shutdown confirmation arriving after an attempt ends can still remove the sandbox record.

A missing heartbeat can mean the environment has not started or a read failed; it does not establish that the sandbox exited. Before any valid heartbeat is observed, `heartbeat_age_seconds_max` is also 0: check `heartbeat_pending_sandboxes` alongside it.

<a id="trace-and-tools"></a>

## Trace lengths and tool calls

| Metric | Meaning |
| --- | --- |
| `live/e2b/active_attempts_with_metrics` | Active attempts with a received snapshot: the sample count for the active-length statistics below. |
| `live/trace/active_model_turns_mean`, `live/trace/active_model_turns_max` | Mean and maximum completed model turns across those attempts. |
| `live/trace/active_generated_tokens_mean`, `live/trace/active_generated_tokens_max` | Mean and maximum generated response tokens across those attempts. |
| `live/trace/active_segments_mean`, `live/trace/active_segments_max` | Mean and maximum recorded segments across those attempts. Segments and model turns are different counts. |
| `live/trace/max_context_tokens` | Largest observed prompt-token plus response-token count for one model request. |
| `live/tool/active_calls_emitted_mean`, `live/tool/active_calls_emitted_max` | Mean and maximum emitted tool calls across those attempts. |
| `live/tool/format_unknown_turns` | Replies whose tool parsing/format status cannot be determined. Without raw-response diagnostics, an empty call list or structurally valid calls do not establish that parsing was error-free. |
| `live/tool/call_counts_unknown_turns` | Replies with unknown NorthGate `detected` or `invalid` counts. Independent of whether a format error is confirmed; not a complete coverage measure for legacy parsers. |
| `live/tool/format_reason_<reason>_turns` | Replies with that NorthGate diagnostic reason, at most once per reason per turn; not the number of invalid calls. Reasons are listed below. |
| `live/tool/format_error_turn_rate` | Replies with a detected parsing/format error divided by replies with known status: `live/trace/model_turns - live/tool/format_unknown_turns`. This says nothing about the unknown replies. |
| `live/tool/result_error_rate` | Explicitly failed tool results divided by observed tool results. |

Counters under `live/trace/model_turns`, `live/trace/generated_tokens`, `live/trace/segments`, and `live/tool/*` accumulate work across every attempt, including failures and retries. Repeated snapshots contribute only new increments. These cumulative values live in process memory and reset after a restart.

With no active attempt snapshots, active-length means and maxima are 0. Tool-error rates, environment-error rates, and attempt-timeout rates are omitted when their denominator is 0; `live/e2b/overlength_ratio` retains its initial 0.

For example, among 10 replies, two confirmed errors, three confirmed clean replies, and five unknown replies produce `format_unknown_turns=5` and `format_error_turn_rate=2/5=40%`. If every reply is unknown, the unknown count remains but no error rate is published.

<a id="tool-errors"></a>

## Tool errors and environment clues

`live/tool/error_<category>` accumulates observed tool errors in the ordinary 30-second snapshots, located by `elapsed_seconds`. A result is failed when it has `is_error=true`, an explicit nonzero `exit_code`, or a matched Bash result beginning with `Exit code N` for nonzero N. Error text inside a successfully read file is not itself a tool error.

| Category | Meaning |
| --- | --- |
| `command_not_found` | The shell cannot find a command, such as `cargo: command not found`. |
| `dependency_missing` | An explicitly missing module, package, or shared library. |
| `network` | Connection, download, or network-timeout failure. |
| `permission` | Insufficient permissions. |
| `resource` | Explicit disk-space, memory, or other resource shortage. |
| `file_not_found` | A missing file or path. |
| `edit` | A failed edit match or invalid edit. |
| `test_or_compile` | A failed test or compilation. |
| `other` | Confirmed failure outside the categories above. |

Each tool-call ID counts once and each error has one category; category totals equal `live/tool/result_errors`. Explicit command, dependency, network, permission, and resource errors take precedence over generic test or file failures. Classification is fixed on first observation so repeated history or later text cannot reduce a cumulative count. The first five categories are environment clues, not proof of the sole failure cause; they do not change reward, task status, or retry behavior. Metrics contain fixed categories and numbers; commands, paths, and original error text remain in traces.

| Metric | Meaning |
| --- | --- |
| `live/tool/observed_rollouts` | Finished rollouts with at least one observed tool result in any attempt. |
| `live/tool/unobserved_rollouts` | Finished rollouts without an observed tool result, including tasks that never executed a tool. |
| `live/tool/observation_coverage` | `observed_rollouts` divided by all finished rollouts; not completeness of an individual trace. |
| `live/tool/env_issue_rollouts` | Finished rollouts with an environment clue in any attempt. |
| `live/tool/env_issue_rollout_rate` | `env_issue_rollouts` divided by `observed_rollouts`. |
| `live/tool/timeout_observed_rollouts` | Finished rollouts with observed tool results and a timeout in any attempt. |
| `live/tool/timeout_env_issue_rollouts` | The preceding cohort with an environment clue in any attempt. |
| `live/tool/timeout_env_issue_rollout_rate` | `timeout_env_issue_rollouts` divided by `timeout_observed_rollouts`: co-occurrence, not causation. |

Error counts include failed attempts and retries; impact rates deduplicate logical rollouts and retain any attempt's clues until completion. Rates with a zero denominator are omitted. Unobserved rollouts are not treated as error-free in environment-issue rates. Completed-ID deduplication keeps a rolling window of 4096 entries; restarting resets cumulative values.

The HTTP relay observes tool results when later model requests include them in history. A final tool result never sent in another request may never be observed. Requests save traces and runtime exit performs a final flush. Final metrics collection before attempt completion preserves already collected observations missed by polling; it cannot recover results never seen by the runtime. Timeout or disconnection preserves cumulative values already received.

The training metric `timeout/tool_count` counts confirmed timeouts among observed tool results. `tool/unobserved_call_count` counts emitted calls with no observed result, which is different from timeout. A zero tool-timeout count does not mean every call finished.

These runtime totals appear only in monitoring logs. W&B `tool/*` and `timeout/*` summarize final executions in the trainer's consumed batch, with a different scope.

<a id="segment-distribution"></a>

## Segments per rollout

| Metric | Meaning |
| --- | --- |
| `live/trace/segments_per_trajectory_mean` | Mean final segment count for finished `success` or `task_failed` rollouts with an explicit count, including zero. |
| `live/trace/segments_per_rollout_max` | Maximum final count in that cohort. |
| `live/trace/segments_per_rollout_p95` | The cohort's 95th percentile. |
| `live/trace/split_trajectory_ratio` | Fraction of that cohort with more than one segment. |
| `live/trace/rollouts_with_segments` | Finished rollouts returning at least one segment and not ending in environment error. |
| `live/trace/rollouts_without_segments` | Finished rollouts explicitly returning zero segments and not ending in environment error. |

<a id="segment-counting"></a>

### Counting rules

- Use final `len(result.segments)`, once per logical `rollout_id`, even if the last heartbeat lags behind the result.
- Only the final attempt supplies the rollout's segment count. Earlier attempts still contribute to cumulative `live/trace/segments` workload.
- Valid `task_failed` results participate, including explicit zero-segment results.
- Environment errors, cancellation, and missing final counts do not participate and are not replaced with zero.
- Explicit zero-segment valid results increment `rollouts_without_segments` and contribute 0 to the mean, maximum, P95, and split ratio, preserving one sample per complete rollout.
- P95 uses nearest rank: sort valid counts and take item `ceil(0.95 × sample_count)`, weighting every rollout equally.

Final distributions persist after rollout completion and publish at the same monitoring interval. Counts and distributions cover the whole process lifetime. Only completed IDs used to reject duplicate terminal events and late starts are limited to the latest 4096 rollouts. Normal executor callbacks are ordered; replayed historical events outside that window are not guaranteed to deduplicate.

<a id="segment-example"></a>

### Example

A new segment starts when the next prompt no longer begins with the current segment's complete token sequence, for example after context compaction.

The distribution sample count is `live/trace/rollouts_with_segments + live/trace/rollouts_without_segments`; no redundant total is published. Normal executor results provide explicit counts, so this equals `live/e2b/valid_rollouts`. Missing or invalid counts still do not participate and are not filled with zero.

A rollout returning three `RemoteSegment` objects contributes **one sample with value 3**.

<a id="training-step-comparison"></a>

## Relationship to training-step metrics

Runtime monitoring measures executor activity and all attempts. W&B Batch / Trajectory / Segment / Group metrics describe parent objects and saved sequences consumed by the trainer. Worker clipping, reward changes, skipped groups, and Datum/mask filtering do not rewrite runtime totals. Differences between adjacent cumulative snapshots cannot substitute for consumed-batch statistics, especially in fully asynchronous training.

Training fields have migrated from earlier `rollout/*`, `critic/*`, and `advantage/*` names. See [step metrics](../../metrics.md) for the current definitions and navigation.

<a id="collection-scope"></a>

## Collection and observation boundaries

<a id="stage-metrics"></a>

### The `metrics` fields across stages

See the [task interface reference](../05-rollout-workers/e2b-task-contract.md) for result types. The three fields below have different sources and purposes:

| Field | Default | Source and purpose |
| --- | --- | --- |
| `EvaluationResult.metrics` | `{}` | Additional evaluation measurements from `evaluate`, such as `swebench_resolved` or `tests_passed`. The same result's `reward` supplies the training reward. |
| `RemoteRolloutTrace.metrics` | `{}` | Runtime snapshot saved with a trace, such as model turns, generated tokens, and tool calls. Evaluation restores agent observations from this field when a trace exists. |
| `AgentStageResult.metrics` | `{}` | Runtime snapshot when no trace exists, such as a first-request context-limit failure with no segment. The framework passes it to evaluation. When a trace exists, this field remains empty and `trace.metrics` is used. |

On failure, the final result prefers current runtime observations, even if the latest model result could not be converted, the trace has not yet updated, or no trace exists. An evaluation failure still preserves agent-stage observations.

All three fields use string keys and finite numeric values. The platform reads either agent `trace.metrics` or `stage.metrics`, then merges it with `EvaluationResult.metrics` into the final result. Runtime values win on duplicate keys. `AgentStageResult.metrics` describes execution observations; `EvaluationResult.status` and `reward` express evaluation outcomes.

<a id="heartbeats"></a>

### Sandbox heartbeats

The sandbox writes a small, access-restricted `metrics.json` beside the result file. Atomic replacement prevents partial reads. Heartbeats advance independently of model turns and continue during evaluation.

Host polling is independent of debug traces, so `runtime.debug.enable_data_dump` is not required. Read failures record observation errors without failing or restarting the rollout. Before the first valid snapshot, a missing file is normal preparation and remains pending. File disappearance after an initial snapshot, or connection, missing-sandbox, or JSON errors at any stage, increment `live/e2b/poll_error_count`. If a sandbox disappears suddenly, observations since the last successful read may be lost.

<a id="model-turns"></a>

### Model turns and tokens

Model turns count complete NorthGate results received by the relay, including results later rejected because training-token data is invalid. Generated-token counts use NorthGate usage rather than repeatedly counting historical prompt tokens. Streaming turns count after a complete terminal event; interrupted streams are not completed turns.

<a id="tool-format"></a>

### Tool calls and format diagnostics

Only newly emitted calls count; calls repeated in request history do not. Observed tool results deduplicate by call ID within an attempt. Results never sent back through the relay cannot be counted; emitting a call does not imply its execution finished.

The runtime prefers NorthGate's top-level `tool_call_diagnostics`, without an extra request flag. Non-streaming replies provide it directly; OpenAI streams provide it in a metadata chunk with `choices: []`, and Anthropic streams in the final `message_delta`. `checked=true` means the raw-response parser supplied a format check; `malformed=true` confirms an error, not complete argument JSON Schema validation. Without these diagnostics, the runtime reads `northrl_parse_diagnostics.tool_format_checked/tool_format_error`, or the legacy `nexrl_parse_diagnostics` when the newer field is absent. Without diagnostic evidence or a locally detected structural error, status is unknown; an empty tool list does not establish a clean parse.

Common structural checks still validate `function`, nonempty tool name and ID, and arguments that parse as a JSON object. Invalid argument JSON increments `live/tool/arguments_json_errors`; valid JSON of a non-object type increments `live/tool/arguments_type_errors`. A structural error establishes a bad reply even when raw-response diagnostics are unknown.

- NorthGate `detected/invalid` includes calls in the original response, including repaired or discarded calls, and populates `live/tool/calls_detected` and `live/tool/calls_invalid`. Returned-message structural errors are not added again when these counts are used.
- `detected/invalid=null` means unknown counts. The runtime does not estimate from the final call list: totals include only known counts, and `live/tool/call_counts_unknown_turns` increases. `malformed=true` still counts an erroneous turn.
- `live/tool/format_error_turns` counts replies with a confirmed parsing/format error, at most once per reply. Missing invalid-call counts are not invented.
- Only server-side parsing diagnostics can count discarded calls. Local structure checks are not full JSON Schema validation. Tool execution `is_error` contributes separately to `result_errors`.

Fixed `reasons` are `incomplete_tool_call`, `invalid_arguments_json`, `missing_tool_name`, `invalid_tool_call_structure`, `arguments_not_object`, `duplicate_parameter`, and `undeclared_tool`. One reply may match several, so their totals cannot be summed into bad-reply or invalid-call counts. Unknown reasons do not create dynamic metrics. `arguments_json_errors/arguments_type_errors` remain local checks of returned calls, separate from raw-response reasons.

Diagnostics do not change tool acceptance, reward, or advantage. Training fields `tool/format_unknown_turn_count`, `tool/call_counts_unknown_turn_count`, and `tool/format_reason_<reason>_turn_count` total final executions in the current batch; corresponding `live/tool/*_turns` accumulate all observed attempts in the process.

<a id="request-timing"></a>

## Weight synchronization and model-request timing

All durations use seconds and a monotonic clock.

| Training metric | Meaning |
| --- | --- |
| `timing_weights/weight_export_seconds` | Export of sampler weights, timed separately from `timing_train/batch_training_seconds`. |
| `timing_weights/checkpoint_seconds` | Client wall time of a successful checkpoint save in this step, excluded from total training time; omitted on failure. |
| `timing_weights/weight_sync_seconds` | Creating and switching sampling clients, including caller waits; excludes export, validation, and logging. |
| `training/weight_export_success_flag`, `training/weight_sync_success_flag` | Operation success: 1 for success, 0 for failure. Failed durations use `timing_weights/weight_export_failed_seconds` and `timing_weights/weight_sync_failed_seconds`, separate from successful curves. |

After publication and before validation or admitting the next batch, synchronization metrics join the pending row for the `training/train_step` that produced the weights. Skipped training, no-sync operation, or a step without publication produces no corresponding duration or zero placeholder. Timing-observation failures do not alter synchronization or batch admission.

| Runtime metric | Meaning |
| --- | --- |
| `live/inference/request_count`, `completed_count` | Started and finished model requests. |
| `live/inference/success_count`, `error_count`, `timeout_count`, `cancelled_count` | Mutually exclusive finished-request outcomes. Success requires a complete NorthGate response and recorded training-token data; streaming also requires a terminal event. |
| `live/inference/request_seconds_sum`, `request_seconds_mean`, `request_seconds_max` | Sum, request-weighted mean, and maximum duration of all finished requests, including errors, timeouts, and cancellation. Concurrent duration sums are not rollout wall time. |
| `live/inference/request_seconds_p95` | Approximate P95 from merged cumulative histograms, not an average of per-rollout P95s. Upper bounds are 1, 5, 10, 30, 60, 120, 300, 600, and 1800 seconds plus overflow. The estimate is the smaller of the selected bucket's upper bound and the observed maximum. |
| `live/inference/inflight`, `oldest_inflight_seconds` | Total in-flight requests and longest wait in the latest active-attempt snapshots. These are current gauges, not historical maxima, and become zero without active requests. Interpret stale snapshots alongside heartbeat ages. |

These values update with host reads of sandbox snapshots: five-minute polling plus a final read. They accumulate across all attempts, including environment failures and retries. Mean/P95 are omitted before any request completes. Duplicate or older heartbeats neither double-count work nor overwrite newer active state.

One request is one logical model call handled by the sandbox relay. Its duration includes network time, NorthGate templating/tokenization and output parsing, backend processing, and local training-token recording. Streaming lasts until a complete terminal event and confirmed training data; interruption, invalid metadata, or a missing terminal event is a failure. Relay, NorthGate, and backend internal retries remain the same observed request; a new agent HTTP retry is a new request. These metrics measure the complete call, without a separate first-token or GPU-compute timer. SSE events alone therefore do not turn these observations into true TTFT or GPU-only inference time.

Runtime totals use ordinary-log `live/inference/*`. Final-execution request observations also enter the batch: counts and lengths become `model_request/*`, and durations become `timing_rollout/model_request_*_seconds` (`mean`, `min`, `max`). Valid duration counts remain internal; existing request counters are unchanged. Raw buckets and active gauges remain runtime observations rather than training reward/score averages.

Each finished request also writes a private `requests.jsonl` row containing rollout/attempt/request identity, API, outcome, duration, token counts, and exception type, without prompt or response text. With debug rollout traces enabled, the host retrieves it before cleanup as `rollout_traces/e2b_rollout_<rollout_id>.requests.<attempt_id>.jsonl`, separately for each retry. Without trace saving, runtime statistics remain available but request details are not retrieved. Retrieval failures do not change the rollout result or prevent cleanup; sudden sandbox loss can lose final observations or details.

Normal `[PROFILE]` start/completion logs use DEBUG; failures remain ERROR, and a model request exceeding 60 seconds emits a WARNING with request identity. Synchronization retains start and duration summaries; backend retry warnings remain. `LOG_LEVEL=DEBUG`, also forwarded into E2B, restores detailed stage logs. Disabling verbose logs does not disable timing collection.
