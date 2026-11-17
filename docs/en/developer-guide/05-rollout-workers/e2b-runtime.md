English | [简体中文](../../../zh/developer-guide/05-rollout-workers/e2b-runtime.md) | [Documentation index](../../index.md)

# E2B execution and trace storage

This reference explains how the platform runs tasks, connects model services, saves traces, and retrieves evaluation logs. Start with the [integration guide](e2b-task-integration.md) for task setup or the [interface reference](e2b-task-contract.md) for hooks and return values.

For live sandbox and request counters, see [runtime monitoring](../08-features/live-metrics.md). Completed training-step charts are described in [training metrics](../../metrics.md).

<a id="components"></a>

## Component responsibilities

| Component | Location | Responsibility |
| --- | --- | --- |
| `e2b_rollout_worker.py` | Host worker | Build requests from training samples, obtain the published sampler gateway, call the executor, and convert the result into a training trajectory. |
| `e2b_executor.py` | Host | Create the sandbox, dispatch stages, upload evaluation assets, retrieve results, manage recovery, and terminate the sandbox. |
| `e2b_runtime.py` / `E2BRolloutRuntime` | Sandbox | Run platform stages, relay NorthGate HTTP traffic, record trajectories, invoke hooks, and persist stage/final results. |
| Task module, such as `swerebench_claude_code_rollout.py` | Sandbox | Prepare the task environment, run the agent, and evaluate its output. |
| `e2b_bootstrap.sh` / `e2b_bootstrap.py` | Sandbox | Prepare shared Python dependencies before the first stage. |
| `contracts.py` | Host and sandbox | Define inputs, agent outputs, stage records, and evaluation results. |

The worker calls the executor. Task authors do not copy or launch these platform scripts themselves.

<a id="sampler-gateway"></a>

## Sampler publication, version routing, and recovery

Model requests follow `agent → sandbox HTTP relay → NorthGate → Weaver sampler`. The host uses the Weaver SDK to manage training and samplers, waits for `gateway.state: published`, and obtains `gateway.base_url` and `gateway.model`. At startup, it centrally waits for publication after warming the shared sampler and before starting workers. For subsequent weight synchronization, it publishes the replacement sampler before switching the client/version and admitting the next tasks.

Successful publication results are cached for that session; ordinary rollouts do not repeatedly poll publication. `rollout.execution.gateway_publication_timeout_seconds` bounds each central wait and defaults to 300 seconds. The service must enable automatic sampler publication and configure the model's tokenizer, chat template, and response parsing. Recipes do not manually supply a NorthGate URL or sampler model name. Configure `WEAVER_API_KEY` and `NORTHGATE_API_KEY` on the host.

A successful cache does not detect idle retirement. To revive an idle retired sampler, the host can call `warmup_sampling(refresh_gateway=True)` for a direct short Weaver sample followed by publication confirmation. Alternatively, directly sample or compute log probabilities, then call `refresh_gateway_snapshot()`, or create a new sampler. NorthGate retries, sampler queries, and heartbeats do not revive the endpoint. Publication checks explicitly reject `retiring` / `retired` without issuing extra sampling requests. Expired or terminated samplers require a new sampler. Recovery affects later tasks; running rollouts retain their original session and version.

Agents use the supplied local OpenAI or Anthropic endpoint and send `model: northrl`. The relay rewrites it to the selected `weaver-<sampling-session-uuid>`. Anthropic responses, including streamed `message_start`, restore the model name originally requested by the client. Otherwise, Claude Code can remove thinking blocks after seeing a mismatched response model, changing the rendered history and forcing each turn into a separate training segment.

An explicit `rollout.sampling.max_tokens` (`validate.sampling` for validation) overrides the agent's response budget without clipping it to the remaining context. An omitted or null value preserves the agent budget. NorthGate owns tokenization and context-limit checks.

Training arrays come from NorthGate's `token_data`: actual prompt/response tokens, log probabilities, and any additional fields requested by the training configuration. Readable trace fields `message_format`, `message_system`, `messages`, and `tools` reflect the last successful forwarded request in its native protocol, with the response appended to `messages`. `tools` preserves full definitions, descriptions, and parameter schemas: absent tools become `null`, explicitly empty tools become `[]`. Protocol-level Anthropic `cache_control` is stripped before forwarding and matching, so it is also absent from these trace request fields.

<a id="lifecycle"></a>

## Execution order

```text
Wait for the sampler's NorthGate gateway to be published
  → Create the sandbox; upload platform code and the task module
  → Call prepare, if supplied
  → Start the relay and call run_agent
  → Clean up the agent and wait for it and its descendants to exit
  → Stop the relay; save the final trace and agent-stage record
  → Validate the stage record's rollout and attempt identities
  → Upload hidden evaluation assets into the same sandbox
  → Call evaluate
  → Combine the trace and reward; write the final result
  → Download the result and terminate the sandbox
```

`trace.json.gz` is also updated during execution; its existence does not prove agent completion. Evaluation upload requires a trusted completion record. Evaluation reuses the modified workspace without rerunning the agent or resetting the repository. See [execution budgets and recovery](../11-configuration-reference/e2b-execution-budgets.md) for stage deadlines and recovery behavior.

<a id="trace-schema"></a>

## Trace files and schema v2

`trace.json.gz`, `agent-stage.json.gz`, and `result.json.gz` require `schema_version: 2`. Unversioned files are unsupported; host executor and sandbox runtime must use compatible versions. In-memory `RemoteSegment` objects retain full arrays, while storage and transport use compact models in `northrl.remote_rollout.wire`.

Adding `turn_annotations` did not change the schema version. New readers accept older v2 records without the field as an empty list. Old readers reject the added field because they use `extra="forbid"`, even when the list is empty. This is one-way read compatibility, not permission to mix runtime versions; a rollback to old code cannot directly consume new artifacts.

| Segment field | Stored representation |
| --- | --- |
| `base_segment` | Index of an earlier segment in the same file, or `null` without a shared prefix. The encoder compares actual token IDs with the preceding segment. |
| `prefix_len` | Number of tokens reused from the start of the referenced segment; zero without a reference. |
| `suffix_tokens` | This segment's tokens following that shared prefix. |
| `response_spans` | Response intervals in full segment-token coordinates: zero-based, half-open `[start, end)`, possibly multiple spans. |
| `response_log_probs` | Original log probabilities in response-span order, omitting zeros at context positions. |
| `old_response_log_probs` | Values in the same response order, or `{"ref":"response_log_probs"}` when identical. |
| `sampling_mask` | Existing inline/ref representation, aligned with all response tokens in this segment. |
| `segment_metrics` | Segment counters. |
| `turn_annotations` | Raw diagnostics and completion information for each generated turn; absent means `[]`. |

<a id="turn-annotations"></a>

## Per-turn diagnostics

Each annotation has `version: 1`, a nonempty `turn_id` unique within its segment, `response_span: [start, end]`, `diagnostics`, `finish_reason`, `truncated`, `tools_declared`, and `source: "northgate"`. Spans use full, uncompressed segment-token coordinates, are ordered and non-overlapping, and cover only generated tokens. They are not indices into compressed tokens or response-logprob arrays. Adjacent turns retain separate annotations even if their spans are combined into one compact `response_spans` interval.

The recorder creates annotations automatically. `diagnostics` preserves NorthGate's original `tool_call_diagnostics` dictionary or `null`, not a reconstruction from accepted calls or aggregate counters. `finish_reason` is the original reason or `null`. `truncated` means truncated output or uncertain completion. `tools_declared` means tool declarations are known; an explicit empty list still sets it to `true`.

A new segment does not inherit annotations for earlier turns copied only as context. Training-sequence trimming preserves earlier complete turns, shortens the affected turn's span and marks it `truncated: true`, and removes turns entirely outside the retained range.

<a id="trace-decoding"></a>

## Decoding and compression

For example, S0 tokens `[10,21,22,23]` and S1 tokens `[10,21,22,30,41,50,61,62]` allow S1 to store `base_segment=0`, `prefix_len=3`, and `suffix_tokens=[30,41,50,61,62]`. If only `41` and `61,62` are response tokens, S1's `response_spans` are `[[4,5],[6,8]]`. Decoding sets `loss_mask=1` at those positions and fills context mask/log probabilities with zeros. Shared tokens may be responses in one segment and context in another, so masks and log probabilities cannot be inherited from the reference.

All references are local to one file: `result.json.gz` can be decoded without a separate `trace.json.gz`. Metadata, native messages, tool definitions, reward, and status retain their meaning. Periodic sync validates and saves the compact form without expanding training arrays; the executor restores and validates full segments when returning training data.

```python
from pathlib import Path
from northrl.remote_rollout.wire import decode_trace, load_trace_gzip

compact = load_trace_gzip(Path("trace.json.gz").read_bytes())
trace = decode_trace(compact)  # Expand only when full training arrays are needed.
```

Compression reduces storage and transfer size, not model computation or the size of expanded training arrays. All three files use gzip level 1 and binary downloads. Local traces are named `e2b_rollout_<rollout_id>.json.gz`. Only gzip-wrapped v2 data is accepted, without fallback to uncompressed files. Status, metrics, request journals, and evaluation logs remain text.

<a id="downloads"></a>

## Download concurrency and background work

Rollouts on the same event loop share one queue with **128 active downloads: 8 reserved for heartbeats/control files and 120 for ordinary downloads**. Periodic traces use ordinary slots without a separate download cap. Pending work is unbounded and not dropped when all slots are occupied; entries hold operations, not downloaded buffers. Final results, stage handoffs, and recovery reads take priority over periodic traces. Each rollout syncs its trace serially and waits for the next interval only after the previous sync finishes. Waiting for admission suspends the coroutine.

The individual I/O timeout begins when reading starts; queueing still counts against the overall stage/recovery deadline. Retry backoff releases the slot without resetting that deadline. Streaming evaluation logs retain their slot until the stream closes. See [execution budgets](../11-configuration-reference/e2b-execution-budgets.md) for ordinary and final-result read limits.

Host diagnostic work admits at most four background callbacks, separately from the two final-result processing slots. Periodic gzip parsing, full validation, and atomic saving use a shared pool of at most two spawned child processes. Only small timing values return, keeping parsed token objects outside the training process. Identical compressed snapshots skip repeated validation/writes after a successful save; invalid snapshots never replace the last valid file.

Final results default to thread decoding. `rollout.execution.result_processing: process` uses one shared spawned child process under the same two-slot result admission, returning packed training arrays. Partial trace recovery for training keeps its existing decoding path.

A download retains its slot until postprocessing finishes to bound downloaded buffers. Small heartbeat and stage-status JSON is parsed directly. Cancellation drains running work, and the last owner closes the relevant process pool, preventing old writes from racing subsequent attempts. These limits control concurrency and memory, not every scheduling delay. Threaded filesystem operations cannot be forcibly interrupted; a stalled filesystem can extend cleanup beyond a stage deadline.

Inside the sandbox, periodic trace writes use one background thread and retain only the latest pending snapshot. Serialization, compression, and atomic writes happen there. Agent finalization waits for the newest snapshot before publishing the final trace and stage record.

<a id="evaluator-logs"></a>

## Final evaluation stdout and stderr

When the evaluation package root contains `run-tests.sh`, the executor instruments the uploaded copy for log capture. Host originals and task entrypoints remain unchanged. Before scoring, the wrapper restores the original script at its installed path and runs Bash with the same arguments, working directory, and environment. It sends stdout/stderr to existing scoring logic while preserving separate byte-for-byte logs. Script `set -e`, exit codes, relative paths, and EXIT traps retain their behavior.

With `runtime.debug.enable_data_dump` enabled and `runtime.debug.dump_dir` set, logs are saved alongside E2B traces:

```text
<dump_dir>/rollout_traces/e2b_rollout_<rollout_id>.evaluation.<attempt_id>/
  stdout.log
  stderr.log
  status.json
```

`attempt_id` separates attempts without overwriting earlier failure logs. `status.json` records rollout/attempt identity, completion, `returncode` (negative for signal termination), collection errors, and both download states. `returncode: 0` means the script exited successfully, not that the task passed; the evaluator still assigns the task score. After timeout or cancellation, `finished: false` and `returncode: null` mean the final exit status is unknown; logs can be partial.

Success, task failure, scoring exceptions, timeouts, and cancellation all use the same best-effort log collection before sandbox termination. Downloads stream complete files without a 2,000-character tail limit. Collection adds at most 10 seconds, runs alongside final metric retrieval, and remains inside the cleanup allowance. Interrupted/budget-limited downloads preserve received bytes and mark incomplete `downloads`. Collection failure does not change reward or skip sandbox cleanup. Once the sandbox is unreachable, missing bytes cannot be recovered.

Custom evaluators with other filenames do not receive this automatic wrapper; existing outer command logs continue to work.

<a id="request-cancellation"></a>

## Model request cancellation

An agent can leave an unfinished model request when it exits, even after changing workspace files. The relay increments `model_requests_cancelled`; cancellation alone does not imply a model-service failure. Without other execution or cleanup errors, the workspace can still be evaluated.

A cancellation also counts as a model-service failure only if either condition holds:

1. The request encountered HTTP 429, 5xx, or a connection error, and its retry has not yet succeeded.
2. No successful response headers or streaming chunk has arrived for at least 300 seconds.

The 300-second timer starts with request handling and resets on successful response headers or each streaming chunk. It is checked when cancellation arrives; reaching 300 seconds does not itself terminate a request.
