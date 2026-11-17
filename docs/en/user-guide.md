English | [简体中文](../zh/user-guide.md) | [Documentation index](index.md)

# NorthRL User Guide

NorthRL connects task execution, reward calculation, and training. Start with an
existing recipe, then customize the task boundary you need. Runtime components
handle task admission, trajectory collection, weight synchronization, and
training-service calls.

<a id="quick-start"></a>

## Quick Start

Follow [Quick Start](developer-guide/01-getting-started/quick-start.md) for
installation, credentials, and launch commands, and
[Local Configuration](developer-guide/01-getting-started/configuration-setup.md)
for service credentials and data paths. Run the driver in your prepared local Python environment.

Choose a starting point:

| Task | Entry point | Guide |
| --- | --- | --- |
| Single model response with a named reward | `rollout.type: single_turn` | [Math recipe](recipe/math/README.md) |
| Custom prompts, multiple calls, or custom scoring | `BaseRolloutWorker` subclass | [Custom workers](developer-guide/05-rollout-workers/custom-workers.md) |
| Agent or tool execution inside a sandbox | `rollout.type: e2b` | [E2B task integration](developer-guide/05-rollout-workers/e2b-task-integration.md) |
| Supervised fine-tuning | `algorithm.type: sft`, omit `rollout` | [SFT recipe](recipe/sft_quickstart/README.md) |

The built-in `simple` worker also supports completion with `<answer>` exact-match
scoring. It has its own fixed scoring behavior; use `single_turn` to select a
registered reward.

<a id="recipe-preparation"></a>

## Recipe Preparation

Keep a task's configuration and code together:

```text
recipe/my_task/
├── train.yaml
├── worker.py       # Only for a custom local worker
└── rollout.py      # Only for a sandbox task
```

A complete training recipe defines the `experiment`, `models`, `data`,
`validate`, `algorithm`, `rollout`, `training`, `backend`, and `runtime` blocks.
SFT omits `rollout`. Copy a maintained recipe and adjust its model, data paths,
training budget, and runtime environment; the snippets below are partial
configuration fragments, not standalone training configs.

For a single-turn math task, use:

```yaml
rollout:
  type: single_turn
  reward_fn: math_answer_only
```

The data loader expects a chat-message list in `prompt`. For example:

```json
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "label": "42"}
```

`label` supplies the reference answer for this reward. An optional sample-level
`reward_fn` overrides the configured default. Other scorers may need different
fields; see [data formats](developer-guide/03-data-loader/data-loader.md) and
[reward selection](developer-guide/11-configuration-reference/rollout-config.md#single-turn-reward-selection).

Custom worker paths resolve relative to the recipe YAML. E2B entrypoint targets
and configured `task_files_root` currently require absolute host paths; use the
[E2B configuration guide](developer-guide/05-rollout-workers/e2b-task-integration.md) when preparing them.
Keep credentials in the runtime environment, and keep hidden evaluation data
out of the problem inputs sent to sandbox agents.

<a id="implementing-rollout-workers"></a>

## Implementing Rollout Workers

A custom worker subclasses `BaseRolloutWorker`, implements
`async rollout(task)`, calls the initialized inference client, and submits one
complete `Trajectory` using `_put_trajectory`. Select it with:

```yaml
rollout:
  type: custom
  custom_module_path: worker.py
  custom_class_name: MyWorker
```

The controller imports `worker.py` relative to the recipe and constructs
`MyWorker` with the full configuration. Put task-specific settings under
`rollout.params`. The [custom worker guide](developer-guide/05-rollout-workers/custom-workers.md)
contains a complete implementation and explains module imports.

An ordinary trajectory stores `tokens` and `loss_mask` directly. When multiple
independent training sequences are needed, store sequence dictionaries in
`segments` on a single complete trajectory. Multiple interaction turns alone do
not require segments. Preserve the final reward on the complete trajectory, the task's
`is_val` flag, and token-aligned `old_log_probs` with prompt zero placeholders.
Process truncation before submission, invalidating a reward if its scored answer
was truncated. See [trajectory semantics](developer-guide/04-trajectory-pool/trajectory-semantics.md).

For training, the wrapper publishes the submitted result after `rollout` returns.
An absent submission records a skipped result; `"fail"` records failure; an
escaping exception publishes an empty failed result and propagates. Validation
routes `is_val=True` objects directly to the validator and does not create those
automatic outcomes. See [worker lifecycle](developer-guide/05-rollout-workers/base-rollout-worker.md).

<a id="implementing-evaluators"></a>

## Implementing Rewards and Evaluation

The stock single-turn path calls `evaluate_reward(task, result)`. Reward names
select ordinary Python callables from `northrl.rewards.REWARD_FNS`. Their result is
a finite scalar or a dictionary containing `reward` and optional `correct`,
`metrics`, and `reason`. Recipe-local scoring can run inside a custom worker;
`rollout.reward_fn` itself does not load arbitrary Python files.

E2B task scripts implement async `run_agent` and `evaluate`, with optional
`prepare`. The runtime records model interactions and uploads hidden evaluation
materials after the agent stage. The evaluation hook returns `EvaluationResult`;
an unsuccessful task can return `status="task_failed"` with reward zero.
Infrastructure failures raise `RolloutEnvironmentError` rather than producing a
fabricated score. See [rewards and evaluation](developer-guide/05-rollout-workers/evaluators.md).

<a id="trainer-architecture"></a>

## Trainer Architecture

Choose `algorithm.type: grpo`, `opd`, or `sft` for the corresponding trainer.
Task scoring usually requires no trainer changes. The trainer prepares complete
trajectories, computes the algorithm's training signals, and converts their
sequences to service Datums.

Start with [Algorithm Components](developer-guide/06-trainers/algorithm-components.md)
for estimator, loss, and postprocessing choices and links to their detailed guides.
Extend these components or the trainer only when changing optimization behavior,
and preserve complete trajectories until conversion. See
[Remote API Trainers](developer-guide/06-trainers/remote-api-trainers.md) for the
trainer interfaces and execution flow.

<a id="configuration-guide"></a>

## Configuration Guide

Use the [configuration reference](developer-guide/11-configuration-reference/rollout-config.md)
for accepted fields. Keep generation settings in `rollout.sampling`, task settings
in the selected worker's configuration, and optimizer settings in `algorithm`.
Prepare dependencies and environment variables in the local driver environment
before running `python -m northrl.main`; see [Local Configuration](developer-guide/01-getting-started/configuration-setup.md).

<a id="best-practices"></a>

## Best Practices

- Keep task prompts, reward rules, and helper code in the task recipe.
- Validate required input fields and distinguish a wrong answer from a failed service.
- Keep rewards and scalar metrics finite; document their scale and meaning.
- Preserve token IDs, masks, and sampling logprobs from the actual model calls.
- Submit one complete result and retain execution identity and validation routing.
- Test task logic before running a distributed training job.

<a id="debugging-tips"></a>

## Debugging Tips

Start with [rollout debugging](debug_rollout.md) to inspect generation and reward
without a full training run. Use [data dumps](debug_data_dump_usage.md) to inspect
tokens, masks, scores, and trajectory grouping. Test correct, incorrect, empty,
and truncated responses, then check reward distributions and non-finite values
during training. For launch and service failures, consult
[startup requirements](developer-guide/01-getting-started/quick-start.md).
