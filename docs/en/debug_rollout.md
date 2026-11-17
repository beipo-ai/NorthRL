English | [简体中文](../zh/debug_rollout.md) | [Documentation index](index.md)

# Debug one rollout batch

Run an existing training YAML through one rollout batch and save the completed
results, without starting a controller or trainer:

```bash
python -m northrl.debug_rollout --config recipe/math/weaver.yaml \
  data.train.batch_size=2 \
  data.train.rollout_repeat_n=2 \
  runtime.debug.dump_dir=/tmp/math-rollout-debug
```

Run from the repository root in your training Python environment. Set dataset
paths, Weaver credentials, and judge configuration in the current shell first,
or source your own environment script before running the command.
If `EXPERIMENT_PATH` is absent, the command supplies a fallback while resolving
the YAML. You can also put the overrides directly in your YAML.

The command runs the first `batch_size` prompts with `rollout_repeat_n` attempts
each, sequentially. The example makes four attempts. Training data-loader
shuffle, filtering, and `drop_last` settings apply, including for asynchronous
training configurations. It never moves to another batch or epoch. If your
dataset is smaller than the batch size, reduce it or set `data.train.drop_last=false`.

Each invocation prints a fresh directory below `runtime.debug.dump_dir`
(default `./debug_dump`). Successful rollouts use the existing worker dump:
`rollout_traces/*.json`. For `single_turn`, each file contains:

- `task`: original messages, ground truth, IDs, and metadata.
- `prompt_str`: the rendered prompt; `rollout_result.northrl_train.prompt_tokens`
  holds the prompt tokens returned by the sampler.
- `response` and `rollout_result`: parsed generation and its existing result fields.
- `reward` and `score`: final reward and scoring details. Truncation can set the
  final reward to zero while `score.reward_score` retains the evaluated score.
- `tokens`, `loss_mask`, `old_log_probs`, token counts, `finish_reason`, and
  `is_truncated`: the worker's trajectory data and completion status.

Generation or reward errors are logged, and remaining samples are still attempted.
A failed or skipped sample has no completed trace. There is no extra capture of
intermediate requests, template source, or judge calls, and no separate report.

Supported workers are `single_turn` and `simple`; `simple` retains its existing
answer-tag reward and trace fields. E2B, custom workers, mock, and SFT are
unsupported. Weaver is initialized for sampling only. For a base-model check,
leave `models.actor.checkpoint` unset and use `training.resume.mode=disable`.
For saved weights, set `runtime.debug.sampling_model_path` to a Weaver sampling
checkpoint and `models.actor.tokenizer` to its matching tokenizer. Training
checkpoints are not converted automatically: explicit checkpoint or existing
resume configurations require a sampling path to avoid sampling the wrong weights.
