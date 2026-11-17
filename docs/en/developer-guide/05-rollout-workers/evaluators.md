English | [简体中文](../../../zh/developer-guide/05-rollout-workers/evaluators.md) | [Documentation index](../../index.md)

# Rewards and Evaluation

Scoring belongs to the task. The integration point depends on the rollout path:

| Path | Scoring interface | Result |
| --- | --- | --- |
| `single_turn` | A named function in `northrl.rewards.REWARD_FNS` | Finite scalar or dictionary containing `reward` |
| Custom worker | Task code called inside `rollout(task)` | Final reward on one complete `Trajectory` |
| `e2b` | Async `evaluate(input_data, workspace, agent_result, evaluation_assets)` | `EvaluationResult` |

<a id="single-turn-rewards"></a>

## Single-turn Rewards

Merge a reward selection into a complete recipe:

```yaml
rollout:
  type: single_turn
  reward_fn: math_answer_only
```

A sample's non-null `reward_fn` overrides `rollout.reward_fn`. Unknown names,
including an explicit empty string, raise an error. Functions receive `(task,
result)`, where `result["response"]` is the generated answer and
`result["finish_reason"]` records completion status. See the
[supported rewards and input fields](../11-configuration-reference/rollout-config.md#single-turn-reward-selection).

`evaluate_reward` normalizes a scalar to `{"reward": value}`. A dictionary must
contain `reward` and may include a boolean `correct`, a string `reason`, and
`metrics` mapping strings to finite numbers. Rewards must be finite; the API does
not require the range `[0, 1]`. Choose a scale appropriate for the task and keep
its meaning consistent across comparable samples.

Synchronous scoring runs in a worker thread; async functions and returned
awaitables run on the caller's loop. Cancelling a wait does not interrupt a
synchronous scorer that has already started. Judge failures propagate.
`finish_reason == "length"` returns zero without invoking the scorer, and
`SingleTurnRolloutWorker` also clears the reward if it truncates the training
sequence after evaluation.

<a id="custom-worker-scoring"></a>

## Custom Worker Scoring

A custom worker can call a task function directly or reuse `evaluate_reward` with
an explicit function mapping:

```python
from northrl.rewards import evaluate_reward


def exact_match(task, result):
    correct = result["response"].strip() == task["answer"].strip()
    return {"reward": float(correct), "correct": correct}


async def score_response(task, result):
    return await evaluate_reward(
        task,
        result,
        functions={"exact_match": exact_match},
        default_reward_fn="exact_match",
    )
```

The mapping is a Python API argument, not a YAML plugin setting. The stock
single-turn worker uses `REWARD_FNS`; setting an arbitrary file path as
`rollout.reward_fn` does not load it. To use recipe-local scoring without changing
the built-in mapping, select a [custom worker](custom-workers.md) and call the
function there. Preserve one final reward on the complete trajectory.

<a id="sandbox-evaluation"></a>

## Sandbox Evaluation

The task entrypoint implements `run_agent` and `evaluate`, plus optional
`prepare`. The platform finishes the agent stage, saves traces, uploads hidden
materials, and calls `evaluate` in a separate process in the same sandbox.
Model traces are captured by the runtime; `AgentResult` carries the optional
answer and artifact references. This scoring hook matches the minimal task in
`recipe/e2b_minimal/rollout.py`:

```python
from pathlib import Path
from northrl.remote_rollout.contracts import (
    AgentResult,
    EvaluationResult,
    RolloutEnvironmentError,
    RolloutInput,
)


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    try:
        expected = (evaluation_assets / "expected.txt").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise RolloutEnvironmentError("cannot read expected.txt evaluation asset") from exc
    answer = agent_result.final_answer
    matched = answer is not None and answer.strip() == expected
    return EvaluationResult(
        reward=1.0 if matched else 0.0,
        status="success" if matched else "task_failed",
    )
```

`EvaluationResult.reward` is finite; optional `metrics` are finite numbers.
`status="task_failed"` means an evaluated, unsuccessful task and requires reward
zero. `status="success"` accepts any finite reward, including zero. Missing
materials or evaluator failures should raise `RolloutEnvironmentError`, not
fabricate a scored answer. See [E2B task integration](e2b-task-contract.md#scoring-and-failures)
for entrypoint configuration and hidden-material handling.

<a id="testing-evaluation"></a>

## Testing Evaluation

Test correct and incorrect answers, missing or malformed fields, empty output,
multiple accepted references, and non-finite scores. Keep detailed diagnostics
separate from scalar metrics. When a scorer uses a service, distinguish service
failures from valid incorrect answers and make its timeout explicit.

## Related Documentation

- [Custom Workers](custom-workers.md)
- [Trajectory Semantics](../04-trajectory-pool/trajectory-semantics.md)
- [Rollout Debugging](../../debug_rollout.md)
