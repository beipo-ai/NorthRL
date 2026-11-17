English | [简体中文](../../../zh/developer-guide/08-features/verifier-metrics.md) | [Documentation index](../../index.md)

# Generation and reward-evaluation failures and timeouts

A SingleTurn execution first asks the training model to generate an answer, then calls a reward function to evaluate it. An evaluation LLM, such as the Reasoning360 STEM judge, belongs to the **reward stage**, separate from answer generation. Metrics identify the failing stage; error types, test results, and logs are still needed to diagnose the cause.

<a id="failure-stages"></a>

## Identify the failing stage

| Question | Metric or diagnostic source | Interpretation |
| --- | --- | --- |
| Did answer generation time out? | `timeout/model_request_count` | Observed logical `generate` calls whose final failure is recognized as timeout. Excludes reward LLM-judge and E2B requests. Does not distinguish inference-service, network, or client-wait causes. |
| Did generation reach its token limit? | `trajectory/finish_reason_length_count` | `finish_reason=length` directly receives zero and skips evaluation. This is a length limit, not a timeout. |
| Did E2B reward evaluation time out? | `verifier/timeout_count` | Sandbox operations that did not complete within budget, including timeouts recovered by retries. Excludes answer generation; supported stages are described below. |
| Did evaluation still fail after timeout retries? | `verifier/timeout_after_retry_count` | Trajectories receiving zero because evaluation timeouts exhausted retries. Not the number of timed-out requests or a diagnosis of server-side cause. |
| Did an LLM judge request fail? | `verifier/judge_error_count` | Final results with `judge_error`. Reasoning360 combines timeout, network errors, HTTP 400, and other failures here; it does not separately count judge API timeouts. |
| Did generated code time out on a test case (TLE)? | No separate aggregate; inspect CodeE2B `sandbox_result.tests`. | Normally returned failed cases contribute to `failed_tests`. `verifier/timeout_count` is not the program-TLE count; raw test fields depend on the sandbox judge. |
| Did the whole rollout reach its deadline? | `timeout/rollout_count` | Requires an explicit whole-execution timeout observation from the worker. Not the sum of model and reward timeouts; SingleTurn currently has no corresponding total-deadline observation. |
| Did training forward/backward fail? | Trainer / Weaver error logs | Occurs after generation and evaluation, outside these generation/evaluation timeout counters. |

An E2B `stage=run` timeout says only that the evaluation command did not complete in time. Slow generated code, a stalled checker/judge, or sandbox communication can all cause it. Inspect `stage`, `error_type`, sandbox probes, and raw test results. The historical name `verifier/service_error_count` means **evaluation did not return a normal result**; it does not establish that the reward service implementation caused the problem.

Training SingleTurn observes final timeouts of logical `generate` calls. An intermediate timeout recovered by Weaver's internal retries does not increment this count. Validation does not aggregate model-request timeouts, so there is no `val/timeout/model_request_count`; `val/verifier/*` below describes reward evaluation only. A missing field does not imply zero timeouts.

<a id="verifier-counters"></a>

## Evaluation results and retries

Each training step summarizes reward results from the raw batch before GRPO filtering. Validation uses the same rules with a `val/` prefix. Each parent trajectory counts once, regardless of segments. Aggregation emits `[verifier-metrics]` ordinary logs. W&B receives a completed training step or validation snapshot, which may not be submitted if execution stops early.

| Metric | Meaning |
| --- | --- |
| `verifier/result_count` | Recognizable reward scores or nonempty reasons received, including `finish_reason_length` skipped results. Not the number of actual evaluation calls. |
| `verifier/service_error_count` | Final results whose evaluation did not complete normally, with `judge_error` or `timeout_after_retry`. Does not diagnose root cause. Wrong answers, generation length limits, and missing code do not count. |
| `verifier/judge_error_count` | Failed evaluation results after API retries, reported as `judge_error`, including `reasoning360/.../judge_error`. Several internal requests remain one failed logical evaluation result. |
| `verifier/timeout_count` | E2B operation timeout events explicitly reported by reward functions, including recovered timeouts. Excludes answer generation, LLM-judge API timeouts, and normally returned test-case TLEs. |
| `verifier/retry_count` | Retries explicitly reported by reward functions. |
| `verifier/retried_trajectory_count` | Trajectories with evaluation retries, each counted once. |
| `verifier/retry_completed_count` | Trajectories returning a normal evaluation result after retrying. A correct answer scoring 1 and an incorrect answer scoring 0 both qualify. |
| `verifier/timeout_after_retry_count` | Trajectories receiving zero after timeout retries are exhausted. |
| `verifier/no_code_count` | Results receiving zero with `no_code` because no nonempty code could be extracted. |

For example, an IfVerify evaluation that times out once and then returns an incorrect verdict reports `timeout_count=1`, `retry_count=1`, `retried_trajectory_count=1`, and `retry_completed_count=1`, with `service_error_count=0`. If the checker command times out on all three permitted `run` attempts, it reports `timeout_count=3`, `retry_count=2`, `timeout_after_retry_count=1`, and `service_error_count=1`.

<a id="reward-coverage"></a>

## Coverage by reward function

One CodeE2B trajectory can contain several test batches, so timeout and retry event counts can exceed trajectory counts. Its generic `timeout_count` / `retry_count` describe judge-command timeouts and retries in the `run` stage. Recoverable creation, upload, and warmup failures use independent retry budgets. After recovery, their `create_*`, `upload_*`, and `warmup_*` diagnostics remain in raw `score.metrics` rather than contributing to generic evaluation timeout/retry counters. Exhausted budgets and unrecoverable errors raise exceptions. Some whole-task execution paths also retry recoverable non-timeout command failures, so `retry_count` is not a timeout count.

IfVerify and the VerInstruct hard checker report sandbox-creation, upload, and execution timeouts. Their generic retry counter covers these stages, with stage-specific counters alongside it. Only an exhausted `run` budget consisting entirely of command timeouts returns `timeout_after_retry`; unrecoverable infrastructure errors, an expired whole-attempt deadline, or an unresolved command during cleanup raise instead. Cleanup timeouts appear only in cleanup logs, not failed-evaluation or evaluation-retry counts.

LLM-judge API timeouts and internal request retries do not yet have separate counters. Reasoning360's final `judge_error` contributes a failed evaluation result. VerInstruct's soft judge uses Recast and raises when API retries are exhausted, without producing a normal reward result for these counters.

The same counters are broken down by effective reward function, such as `reward_fn/code_e2b/verifier/timeout_count` and `reward_fn/reasoning360/verifier/judge_error_count`. Validation examples are `val/verifier/service_error_count` and `val/reward_fn/if_verify/verifier/retry_completed_count`.

<a id="aggregation-limits"></a>

## Aggregation boundaries and missing data

These are counts for the current batch or validation snapshot, not process totals. They read parent `score.reason` and explicit `score.metrics.timeout_count/retry_count`, without expanding arbitrary numeric judge scores or mixing in remote-rollout `sandbox_retries`. Negative, non-integer, boolean, and nonfinite counters are ignored. Missing event counts are not inferred from a reason string.

These verifier counters require a recognizable parent reward score or reason. A training wrapper can preserve a failed empty trajectory and model-request observations after an exception, but does not invent a reward result. With early `stop_on_error`, a batch or step may never complete and W&B may not record that failure. Inspect ordinary error logs; missing curves do not establish that no timeout occurred. Restart running training processes to load worker or reward-function changes.

See [step metrics](../../metrics.md) for chart navigation and [execution timing](rollout-timing-metrics.md) for duration definitions. For agent sandboxes, see [runtime monitoring](live-metrics.md) for process totals and the [task interface reference](../05-rollout-workers/e2b-task-contract.md#scoring-and-failures) for completion and scoring rules.
