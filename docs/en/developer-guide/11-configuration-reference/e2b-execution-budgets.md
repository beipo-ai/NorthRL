English | [简体中文](../../../zh/developer-guide/11-configuration-reference/e2b-execution-budgets.md) | [Documentation index](../../index.md)

# E2B execution time budgets

E2B execution allocates separate time budgets to preparation, agent execution,
agent finalization, evaluation preparation, evaluation, and data recovery. A slow
stage does not consume the time reserved for a later stage. Preparation, agent execution, finalization,
and evaluation are configurable. Data recovery and individual I/O attempts use
fixed framework limits.

Configure these fields under `rollout.execution`:

| Field | Default | Purpose |
| --- | ---: | --- |
| `preparation_timeout_seconds` | 600 | Prepare the sandbox, inputs, and runtime. |
| `agent_timeout_seconds` | 3600 | Run the agent and its tool calls. |
| `agent_finalize_timeout_seconds` | 600 | Finish agent output and persist the trace after agent execution stops. |
| `evaluation_timeout_seconds` | 600 | Run evaluation and produce its result after assets are ready. |

Each configurable stage budget must be a positive integer. Booleans, floating-point
numbers, strings, explicit nulls, zero, and negative values are rejected during
worker initialization, including values in execution profiles.

```yaml
rollout:
  execution:
    preparation_timeout_seconds: 600
    agent_timeout_seconds: 3600
    agent_finalize_timeout_seconds: 600
    evaluation_timeout_seconds: 600
```

Evaluation asset upload and extraction share a separate, fixed **600-second**
preparation budget, including retries and ready-marker observation. The configured
evaluation budget starts only once those assets are ready. Preparation exhaustion
ends the attempt as an environment failure without launching evaluation or rerunning
the agent. Profiling records `driver.evaluation-preparation`,
`driver.evaluation-upload`, `driver.evaluation-extract`, and
`driver.evaluator-command` separately (successful timings at DEBUG, failures at ERROR).
This preparation budget is internal and adds no YAML setting.

The framework reserves **900 seconds (15 minutes)** for recovering existing
results and trajectory data. An ordinary transfer or recovery I/O attempt is capped
at **60 seconds**. Downloading the final `result.json.gz` uses a **120-second**
read and SDK request limit because this file can be much larger than control files.
Both remain bounded by the current stage's remaining time. These limits are internal
constants; they cannot be changed through YAML or execution profiles.

Stage budgets include retries within that stage. Retrying a transfer does not
reset the recovery deadline or grant another full stage budget. Preparation
also shares one deadline across any pre-agent sandbox retries. The fixed
recovery allocation is for retrieving data already produced by the attempt; it
does not restart the agent. `env_error_retries` permits a new sandbox only for
failures **before the agent command is dispatched**. Once dispatched, a sample
that cannot be recovered ends as an environment failure without a scored
outcome; the executor does not automatically run the agent again. The worker
still publishes the terminal complete trajectory so its group can finish.

<a id="recovery-behavior"></a>

## Recovery behavior

- A timed-out driver download of the completion record emits a warning. After
  the agent command has completed, the evaluator still validates the record
  locally. Missing, malformed, or foreign records are not accepted.
- A disconnected agent command is observed in the original sandbox until its
  original deadline. Evaluation starts only after a trusted completion state
  for the current attempt has been observed. An observation timeout is not
  treated as a scored task failure.
- Evaluation uploads retry in place. An extractor with a confirmed failed exit
  may be retried; after a transport disconnect the executor observes its ready
  marker instead of launching overlapping extractors.
- The evaluator is launched once. If its command stream disconnects, the
  executor waits for that evaluator's result. A final result download can retry
  for the remaining recovery budget without restarting evaluation or the agent.
- A platform supervisor runs the evaluation runtime in its own process group.
  It kills that group when the evaluation deadline expires or the runtime exits,
  then waits at most 5 additional seconds for the runtime process to be reaped.
  This also stops leftover test processes when an unchanged task hook returns
  or is cancelled. The supervisor inherits output streams and does not wait for
  descendant-held pipes to close. A subprocess that escapes the group is covered
  by subsequent sandbox termination. An incomplete evaluation remains an
  environment failure, not a zero test score. Task entrypoints need no changes.
- Failure recovery resolves a trustworthy result or failure state before
  downloading trajectories. Trace or diagnostic failures cannot overwrite an
  already established outcome. An available local trace is used after a remote
  read timeout; an agent with no model turns need not wait 15 minutes for a
  trace file that does not exist.
- Sandbox termination gets an initial 10-second attempt. If it fails, the host
  retains the sandbox ID and schedules background control-plane checks and at
  most two further termination attempts, capped at 60 seconds in total. Successful
  termination or an explicit sandbox-not-found response confirms closure; network
  errors leave the status unknown. Late confirmation clears the unknown sandbox
  from live metrics without changing the rollout result. IDs are logged for
  investigation if retries exhaust or the worker shuts down before confirmation;
  background retries require the worker event loop to remain running.
- Extra logs, metrics, and sandbox cleanup remain best effort after the result
  has been obtained. Missing essential results or training data are reported;
  they are not replaced with invented scores or segments.

Agent finalization has its own 600-second default budget. Finishing and persisting
an agent trace therefore does not need to fit inside the agent's remaining
execution time. Recovery likewise has time to download a trace even when the
agent has exhausted its execution budget.

The sandbox runtime enforces the agent budget on the running hook, then waits
for process cleanup before committing completion. Budget exhaustion records
`agent_exit_code=124` and `e2b_timed_out=1`; evaluation can still score the modified
workspace. Failed or timed-out finalization does not authorize evaluation.
`agent_execution_seconds` and `agent_finalize_seconds` are included in runtime
metrics. The host bounds the entire command as a fallback for non-cooperative
hooks, and sandbox termination remains the final cleanup fallback.

The sandbox lifetime reserves the sum of the four configured stage budgets,
the fixed 600-second evaluation preparation and 900-second recovery allocations,
and 30 seconds for cleanup. With the defaults above this is 6930 seconds. The fixed
I/O limits cap individual attempts and are not added again to the sandbox lifetime.

<a id="existing-configurations-and-execution-profiles"></a>

## Existing configurations and execution profiles

Use `rollout.execution.agent_timeout_seconds` to set the Agent duration.
MiMo hooks do not read `agent_timeout_sec` from task data or impose a fallback
1800-second Agent deadline. Existing dataset rows can retain that unused field;
new recipes omit it from `input_fields`. The runtime cancels the Agent when the
configured budget expires, then gives cleanup its separate finalization budget.
Verifier-specific evaluation timeouts are unchanged.

The legacy `rollout.execution.timeout_seconds` field remains supported as the
**agent execution budget**. It no longer limits all stages through one shared
deadline. For example, `timeout_seconds: 3600` gives the agent one hour while
preparation, finalization, and evaluation retain their independent defaults,
and recovery retains its fixed 15-minute allocation.

When both fields are present, `agent_timeout_seconds` takes precedence over
`timeout_seconds`. Direct executor callers retain the `timeout_seconds` argument;
it supplies the agent budget when the explicit `agent_timeout_seconds`
configuration field is absent. If neither a configured timeout nor a caller
argument is provided, the agent budget defaults to 3600 seconds.

Execution profiles inherit common `rollout.execution` settings and can override
the four configurable budgets separately: preparation, agent execution, agent
finalization, and evaluation. Validation applies to every resolved profile
before any rollout starts. The selected profile supplies those stage budgets;
the recovery and I/O limits remain fixed for every profile.

See [Agent task integration](../05-rollout-workers/e2b-task-integration.md) for the
setup flow and [E2B runtime](../05-rollout-workers/e2b-runtime.md) for execution and
result recovery details.
