English | [简体中文](../../../zh/developer-guide/05-rollout-workers/e2b-task-integration.md) | [Documentation index](../../index.md)

<a id="e2b-task-integration"></a>

# Integrate an Agent Task

Use the E2B task interface to train the model behind your agent from task outcomes.
You provide task inputs, agent execution, and scoring. NorthRL runs each task in a sandbox,
records model interactions through NorthGate, and sends trajectories and rewards to training.

For a first run, follow the [minimal task walkthrough](../../recipe/e2b_minimal/README.md).
It supplies data preparation, a complete configuration, and local launch commands.
This guide explains how to adapt that workflow to your own task.

<a id="tasks-and-rewards"></a>

## 1. Prepare task inputs and evaluation materials

Decide what the agent receives, what it should produce, and how to score the
result. A task may return an answer, change a repository, or produce files.
Your evaluation decides which outcome earns which reward.

Each dataset record needs a `template` identifying an E2B template you can use.
Here is the record shape used by the minimal task:

```json
{"instruction": "Reply with exactly: READY", "template": "YOUR_E2B_TEMPLATE_ID", "task_id": "task-001"}
```

- `rollout.execution.input_fields` is a required list of unique, nonempty strings.
  These select the fields passed to `input_data.task`; `[]` selects none.
  For this task, use `[instruction]`.
- `template` selects the sandbox. `task_id` is also available as
  `input_data.metadata["task_id"]`; neither needs to be included in `input_fields`.
- Missing selected fields are skipped. Validate the fields your task requires
  before constructing the agent prompt.
- Set `data.train.prompt_key` to the input field used by the data loader, here
  `instruction`. Use a [supported dataset format](../03-data-loader/data-loader.md).

When evaluation needs hidden answers or tests, put them on the machine running
NorthRL under `task_files_root/<task_id>/`. For example:

```text
/data/my_task/evaluation/
└── task-001/
    └── expected.txt
```

For the record above, `expected.txt` contains `READY`. Use a task ID that is a
single directory name, such as `task-001`; evaluation directories cannot contain
symlinks. Keep hidden materials out of the task inputs, agent-visible template,
and task module. NorthRL uploads them only after agent execution and cleanup
have finished. Public tests can be available during execution.

If evaluation needs no external materials, omit `task_files_root`; `evaluate`
still runs and receives an empty evaluation directory.

<a id="task-hooks"></a>

## 2. Implement execution and scoring

Put the task hooks in one Python module. The two required hooks are `run_agent`
and `evaluate`; add `prepare` when the task needs setup. Their interfaces are:

```python
from pathlib import Path

from northrl.remote_rollout.contracts import AgentResult, EvaluationResult, RolloutInput


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

This is an interface outline. Use the
[minimal implementation](../../../../recipe/e2b_minimal/rollout.py) as a working
reference when writing the function bodies.

| Hook | Your responsibility |
| --- | --- |
| `prepare` | Prepare task files or dependencies before agent execution. |
| `run_agent` | Read task input, run the agent using `model_environment`, and return its answer or artifact references in `AgentResult`. |
| `evaluate` | Read the answer or modified workspace, use `evaluation_assets` if needed, and return `EvaluationResult`. |

`evaluate` runs in a separate process in the same sandbox. Pass results through
`AgentResult` or workspace files, not module globals. The framework does not
automatically extract `final_answer` from model messages or score an exit code.
Workspace-based tasks may return `AgentResult(exit_code=...)` without an answer.

Your reward must be finite. `status="success"` permits any finite score, including
zero; `status="task_failed"` requires `reward=0.0`. Missing evaluation materials,
broken evaluators, and unavailable environments should raise
`RolloutEnvironmentError`, rather than be reported as an ordinary zero score.

If the agent runs as a subprocess, `run_agent` must stop and wait for it and its
children on completion, error, or cancellation. Evaluation materials are uploaded
after cleanup; no background agent should remain active during evaluation.
The platform imports the module and calls its hooks: it does not execute a task
`main()`. See the [task contract](e2b-task-contract.md) for all fields and statuses.

<a id="sandbox-template"></a>

## 3. Prepare the E2B template and workspace

Choose an [E2B](https://e2b.dev/) template accessible with your E2B credentials.
It must contain the tools, task dependencies, and public files your agent needs.
Configure an existing, writable absolute directory such as `/workspace` as
`entrypoint.workspace`. Workspace resolution happens before `prepare`, so create
that directory in the template rather than in the hook.

For shared tools or data, see [`execution.volume_mounts`](../11-configuration-reference/rollout-config.md#execution).
Mounted data is shared and writable; the task decides whether to copy it locally.

Keep the locations distinct:

| Setting | Location |
| --- | --- |
| `entrypoint.target` | Absolute path to your task `.py` module on the machine running NorthRL. |
| `entrypoint.task_files_root` | Absolute path to evaluation materials on that same machine. |
| `entrypoint.workspace` | Existing directory inside the sandbox. |

The host paths do not accept `~` or relative paths. NorthRL uploads the task
module and its shared runtime; sibling files next to an external task module
are not automatically packaged. Put supporting code and agent dependencies in
the template. Run subprocess-based agents with the task's restricted user so
they cannot read the platform's private request and credential files.

<a id="model-services"></a>

## 4. Connect the model services

Choose a Weaver model you have permission to train. Export these variables in
the shell that launches NorthRL:

| Variable | Purpose |
| --- | --- |
| `WEAVER_BASE_URL` | Weaver service endpoint. |
| `WEAVER_API_KEY` | Training and sampling credentials. |
| `NORTHGATE_API_KEY` | Credentials for the model gateway used by agent calls. |
| `E2B_API_KEY` | Credentials to create sandboxes from your chosen template. |

For a private E2B service, use its SDK endpoint settings as provided by the
service operator. Weaver and NorthGate must publish the selected sampler and
return its training token data; confirm this service capability before running.

Inside `run_agent`, use the supplied `model_environment` for model calls:

| API | Supplied environment keys |
| --- | --- |
| OpenAI Chat Completions | `OPENAI_BASE_URL`, `OPENAI_API_KEY` |
| Anthropic Messages | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` |

Both APIs support ordinary and streaming responses. The OpenAI Responses API
is not supported by this task interface. Pass the supplied values to your model
client or agent subprocess and request `model: northrl`. NorthRL routes that
name to the selected sampler and records the actual tokens and log-probabilities;
your task does not construct training token arrays or set a remote model URL.

E2B does not use `execution.tokenizer`, `execution.chat_template`, or `rollout.parsers`.
NorthGate handles tokenization, chat templates, and response parsing. The host-side `models.actor.tokenizer`
setting remains available for data preparation and training.

<a id="training-recipe"></a>

## 5. Configure a recipe and start locally

Start from the [complete minimal recipe](../../recipe/e2b_minimal/README.md#training-config)
and replace its task data, model, and evaluation settings. The following fragment
shows the task-specific fields to merge into a complete recipe:

```yaml
backend:
  type: weaver
data:
  train:
    prompt_key: instruction
training:
  weight_sync:
    mode: sync
rollout:
  type: e2b
  execution:
    input_fields: [instruction]
    agent_timeout_seconds: 3600
    entrypoint:
      target: /path/to/NorthRL/recipe/my_task/rollout.py
      workspace: /workspace
      task_files_root: /data/my_task/evaluation
```

Replace the absolute host paths with your actual locations. This fragment omits
the model, dataset files, optimizer, and other training settings; keep those from
the complete recipe and adapt them to your task. Any enabled validation datasets
also need the task fields and evaluation materials used by their hooks.

Follow the example's [local launch steps](../../recipe/e2b_minimal/README.md#run)
after installing the `.[core]` dependencies. Set `TRAIN_CONFIG` and Hydra's
configuration arguments to the same completed recipe. Direct local execution
does not load a `.env` file automatically; export or source your variables first.
The [local running guide](../01-getting-started/quick-start.md#run-locally) explains the entry point and console logging.

<a id="verify-integration"></a>

## 6. Verify before scaling up

1. Check scoring on a known correct and incorrect result. Missing hidden
   reference answers must be an environment error, not a scored failure.
2. Verify that only selected task inputs reach the agent and that hidden
   evaluation materials become available only after agent cleanup.
3. Run a small task through the sandbox. Confirm the agent output, evaluation
   reward, and recorded model trajectory; printed agent text alone is not a
   training trajectory.
4. Complete a small training run using the
   [example's inspection steps](../../recipe/e2b_minimal/README.md#inspect).
   Check both task scores and model updates before increasing concurrency.

<a id="troubleshooting"></a>

## Troubleshooting

| Symptom | Check |
| --- | --- |
| A required task field is missing | Dataset keys, `input_fields`, and validation inside the task hook. |
| Task module or workspace cannot be loaded | Absolute host paths; the sandbox workspace must already exist. |
| Evaluation materials cannot be found | `task_id`, the matching directory under `task_files_root`, and files expected by `evaluate`. |
| Model calls fail or produce no training trace | The provided API environment, supported API, service credentials, and sampler publication. |
| A long task times out | `agent_timeout_seconds`, the agent client's own request timeout, and the task's cleanup behavior. |

<a id="further-reference"></a>

## Further reference

- [Task contract](e2b-task-contract.md): hook arguments, outputs, and scoring semantics.
- [Runtime lifecycle](e2b-runtime.md): execution, evaluation, and recovery behavior.
- [Advanced E2B configuration](../11-configuration-reference/e2b-advanced.md): optional settings and operational details.
