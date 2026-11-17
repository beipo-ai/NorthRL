English | [简体中文](../../../zh/developer-guide/05-rollout-workers/e2b-task-contract.md) | [Documentation index](../../index.md)

# E2B task interface reference

This reference covers task hooks, result types, evaluation, and failures. For a first integration, follow the [integration guide](e2b-task-integration.md) and [minimal task example](../../recipe/e2b_minimal/README.md).

<a id="hooks"></a>

## Task hooks and inputs

```python
def resolve_workspace(input_data: RolloutInput) -> Path: ...

async def prepare(input_data: RolloutInput, workspace: Path) -> None: ...

async def run_agent(
    input_data: RolloutInput,
    workspace: Path,
    model_environment: dict[str, str],
) -> AgentResult | None: ...

async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult: ...
```

Import the contract types from `northrl.remote_rollout.contracts` and `Path` from `pathlib`.

- `run_agent` and `evaluate` are required; `prepare` is optional.
- Supply the workspace through `resolve_workspace` or `entrypoint.workspace`. The configuration takes precedence. The path must be an **existing absolute directory inside the sandbox**.
- `input_data` is a `RolloutInput` object with `rollout_id`, `task`, and `metadata`, not an instruction string. The worker copies only existing data fields selected by `rollout.execution.input_fields` into `task`. Missing fields are skipped; hooks validate the fields they require and build their own prompts. `template` selects the sandbox and is not automatically included in `task`.
- `metadata` contains `task_id`, `group_id`, `run_id`, and `expected_rollouts` when present, plus `weight_version`. Evaluation uses `metadata.task_id`, so it need not appear in `input_fields`. Metadata is not automatically inserted into the agent prompt.
- Task field names are not fixed by the worker. SWE selects `[instruction, interface]`, requires a nonempty `instruction`, and appends an optional nonempty `interface` with one blank line, preserving both texts without adding headings. Its Claude prompt uses only these fields.
- `model_environment` supplies `OPENAI_BASE_URL` / `OPENAI_API_KEY` and `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`. URLs point to the sandbox relay; the local placeholder key is `northrl`. Task hooks do not need Weaver credentials or model weight paths.
- `AgentResult` carries task output to `evaluate`; modified workspace files remain available directly. Runtime records model tokens and log probabilities separately.
- `evaluation_assets` is the task's evaluation directory, **uploaded only in the second stage**. Return an `EvaluationResult` with a finite reward and numeric metrics.

The platform imports the task module. Its top level must not run the agent, start a runtime, or install hidden tests. The module itself is visible during the agent stage: put secret scoring rules in the later evaluation package.

Fields outside `input_fields` are excluded from `task`. SWE excludes `patch`, `test_patch`, `pr_description`, `install_config`, and `meta`. Other tasks should likewise select only problem inputs. Runtime verifies the request file owner and applies mode `0600`; the agent should run as a restricted system user that cannot read this file.

<a id="agent-result"></a>

## AgentResult

After agent cleanup, runtime saves this object in the stage record and passes it to `evaluate` in another process in the same sandbox.

| Field | Default | Meaning |
| --- | --- | --- |
| `exit_code` | `0` | Diagnostic agent exit status. After successful cleanup, evaluation runs for zero, nonzero, and timeout code `124`; the evaluator determines the reward. No final Claude `success` marker is required. |
| `final_answer` | `None` | Optional task-defined answer text. The platform does not extract it from model messages or score it automatically. Tasks evaluated from workspace files can omit it. |
| `artifacts` | `{}` | Artifact name to workspace-relative path, such as `{"patch": "fix.patch"}`. These are references, not file contents. The platform does not copy/upload/download these files or validate whether paths are relative or exist; the evaluator resolves them according to the task contract. |

```python
return AgentResult(
    exit_code=0,
    final_answer="42",
    artifacts={"answer": "output/answer.txt"},
)
```

The evaluator can read `agent_result.final_answer` or `workspace / agent_result.artifacts["answer"]`. SWE evaluates the modified repository and only needs the actual exit code. Returning `None` from `run_agent` is equivalent to returning `AgentResult()`; the final reward always comes from `EvaluationResult`.

`model_config = _MODEL_CONFIG` is shared Pydantic validation configuration, not a task output field. It requires strict types and rejects undeclared fields. `default_factory=dict` creates a separate empty artifacts mapping for each result.

<a id="evaluation-result"></a>

## EvaluationResult

`evaluate` returns the final task score. Runtime combines it with the agent trace into `RemoteRolloutResult`; task code does not construct training trajectories.

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `reward` | Required | Finite score for the complete rollout, on the task's chosen scale; it is not restricted to `[0, 1]`. All segments share this parent score rather than being scored separately. |
| `metrics` | `{}` | Additional measurements with string keys and finite numeric values, such as `{"tests_passed": 8.0}`. These supplement the reward; the framework does not combine them into a reward. |
| `status` | `"success"` | Either `success` or `task_failed`. `success` permits any finite reward, including zero, and need not mean the task was solved. `task_failed` requires `reward=0.0`. Both are valid evaluations and become successful parent trajectories for complete-group processing. |

```python
return EvaluationResult(
    reward=1.0 if passed else 0.0,
    status="success" if passed else "task_failed",
    metrics={"swebench_resolved": float(passed)},
)
```

An evaluator crash, missing hidden assets, or another environment failure should raise `RolloutEnvironmentError`. `EvaluationResult.status` does not accept `env_error`; do not disguise an evaluation failure as a valid zero score. Shared model validation rejects undeclared fields, invalid types, NaN, and Infinity.

<a id="agent-stage-result"></a>

## AgentStageResult

The framework creates this cross-process handoff; task code does not return it. After agent cleanup and relay shutdown, runtime writes `agent-stage.json.gz`. The executor validates its identity before uploading hidden evaluation assets, then starts the evaluation process. This record commits agent completion, not the task's final score.

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `rollout_id` | Required | Nonempty logical rollout ID from `RolloutInput`, retained across environment retries and checked by both executor and evaluator. |
| `attempt_id` | Required | Nonempty ID generated for each attempt. Retries receive a new ID so a stale stage record cannot authorize an evaluation upload. |
| `phase` | `"agent_completed"` | The only allowed value. A completion marker, not a live progress field or evaluation outcome. |
| `workspace` | Required | Absolute sandbox workspace path. Runtime also verifies that it exists and matches the agent stage. Evaluation reuses it. |
| `agent_result` | Required | The hook's `AgentResult`, or the default result if `run_agent` returned `None`. |
| `trace` | `None` | Recorded `RemoteRolloutTrace` containing segments, messages, and provenance. Its `rollout_id` must match. It may be absent if no model turn was recorded, for example an exit before the first completed request or an initial context-limit failure. |
| `metrics` | `{}` | Agent-stage runtime counters when no trace exists. With a trace, counters are stored in `trace.metrics` and this mapping is empty. These are not task evaluation scores. |
| `termination_reason` | `None` | Otherwise `context_length_exceeded`: generation stopped at the context budget, but the workspace can still be evaluated. It is not a task verdict. |
| `length_details` | `{}` | Context-stop details: `prompt_tokens`, `requested_max_tokens`, and `max_sequence_length` describe input length, requested response budget, and total context limit. Validation requires string keys and integer values, not that exact set of keys. |

Any `agent_result.exit_code` is allowed, including timeout code `124`. Shared strict validation applies. The next stage's `EvaluationResult` determines the score; it is not inferred from the phase, exit code, or context-stop reason.

<a id="process-cleanup"></a>

## Process cleanup and cross-stage data

When running Claude Code or another subprocess, `run_agent` must clean up the agent and its descendants and wait for them to exit on normal return, exceptions, and cancellation. A background agent must not survive to read evaluation assets or modify outputs after upload. The SWE adapter provides an example with a dedicated agent user and cleanup logic.

Task hooks do not write `agent-stage.json.gz` or `result.json.gz`; the platform generates and validates them. `--phase agent` and `--phase evaluate` are executor arguments to `e2b_runtime.py`, not a CLI that task modules implement. The phases use separate Python processes in the same sandbox: pass data through workspace files and `AgentResult`, not module globals.

`profile_block("task.stage", rollout_id=input_data.rollout_id)` is an optional timing helper, not a required hook.

<a id="hidden-assets"></a>

## Hidden evaluation assets and task files

`entrypoint.target` must be an absolute path to a host-side `.py` file. Relative paths and `~` are not expanded, and the project root is not prepended automatically. The platform imports the uploaded script without executing its `__main__`. Shared Python code under `northrl` travels with the platform package. An external script is uploaded alone, without neighboring helpers; install other public tools, data, and dependencies in the template.

`task_files_root`, when configured, must also be an absolute host path; it can lie outside the checkout but cannot use relative paths or `~`. Each `task_files_root/<task_id>/` is packaged recursively, preserving directories. Any evaluation files are allowed, not only SWE's fixed set. The directory must not contain symlinks. Each evaluator interprets the supplied directory itself.

Each task row needs a nonempty `template` to create its sandbox. Hooks validate selected `input_data.task` fields. With `task_files_root`, also provide a nonempty string `task_id`; the worker copies it to `metadata.task_id` to locate the evaluation package. Without external assets, omit `task_files_root`: `evaluate` still receives an empty directory. New examples of the same task usually need only new data, templates, or per-task evaluation files, not copies of the task module.

Do not preinstall hidden tests, answers, or secret scoring rules in agent-visible template files, task modules, initial code archives, or task inputs, including copies inside archives. Keep them in the delayed evaluation package; public tests may be provided earlier. Delayed upload cannot remove secret copies already present in a template, so inspect template contents as part of integration.

<a id="scoring-and-failures"></a>

## Scores and failures

- A correctly executed but unsolved task can return zero. `task_failed` requires `reward=0.0`; `success` supports continuous scores including zero. Both become `Trajectory.status="success"`, with the original task status retained as `e2b_status`.
- Evaluator crashes, missing assets, and unavailable environments raise `RolloutEnvironmentError`. The task's evaluator must distinguish a scoring program's nonzero exit meaning “task failed” from an execution failure.
- An ordinary agent exception or any returned exit code still permits evaluation after cleanup. Runtime retains the actual evaluator score and recorded trajectory; `agent_exit_code` is diagnostic and does not preemptively zero reward. Existing artifacts can be evaluated even with no model trace.
- Explicit environment errors, model-service failures, failed process cleanup, and exhausted sandbox lifetime remain environment/execution failures. Hidden assets cannot be uploaded until agent completion is confirmed.
- Evaluation does not change the model version used in the agent stage. The executor manages recovery. `env_error_retries` allows a new sandbox only before agent-command dispatch; after dispatch, it recovers the original attempt without automatically rerunning the agent. See [execution budgets and recovery](../11-configuration-reference/e2b-execution-budgets.md).
- The worker submits one complete `Trajectory`; its `segments` hold all training sequences, and reward/score belong to the parent. Sequence truncation zeros that rollout's training reward and `swebench_resolved`.
- An environment/evaluation failure ends as an empty `Trajectory(status="failed", reward=None)` for complete-group handling, never a fabricated zero score. A valid result with no training sequences retains its final reward and participates in group comparison.
- Remote trace/result files serialize `segments`; counts distinguish complete rollouts, internal sequences, and tokens. See [trajectory grouping](../04-trajectory-pool/trajectory-pool.md) and [training metrics](../../metrics.md).

Cancelling an unfinished model request as the agent exits does not itself invalidate evaluation. See [request cancellation](e2b-runtime.md#request-cancellation).

<a id="integration-checks"></a>

## Integration checks

Check that only `input_fields` reach the task; hooks validate them and build the intended prompt; `task_id` is metadata and stays out of the SWE prompt; the agent cannot read hidden assets or request files; evaluation upload follows agent cleanup and a trusted completion record; positive and negative scoring cases work; evaluator errors remain environment failures; a real rollout produces valid training data; and a small training batch completes.
