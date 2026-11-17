English | [简体中文](../zh/debug_data_dump_usage.md) | [Documentation index](index.md)

<a id="debug-data-dump-使用指南"></a>

# Debug Data Dump Guide

Use `runtime.debug` in a complete Weaver training recipe to save data from a local driver run. To inspect a single rollout batch without training, use [Single-batch Rollout Debugging](debug_rollout.md).

<a id="enable-data-dump"></a>

## Enable Data Dumps

Prepare the local Python environment, data, and service credentials using [Quick Start](developer-guide/01-getting-started/quick-start.md#run-locally). Merge these settings into your recipe's existing `runtime` block:

```yaml
runtime:
  debug:
    enable_data_dump: true
    dump_dir: "${oc.env:EXPERIMENT_PATH}/debug_dump"
    dump_every_n_steps: 1
    remote_api_dump_options:
      prepared_trajectories: true
      datums: true
      training_metrics: true
```

Then run that recipe from the repository root, with `TRAIN_CONFIG` pointing to the edited YAML and `EXPERIMENT_PATH` set to a writable output directory:

```bash
python -m northrl.main \
  --config-path="$(dirname "$TRAIN_CONFIG")" \
  --config-name="$(basename "${TRAIN_CONFIG%.*}")" \
  'runtime.logger.backends=[console]'
```

This executes actual training through Weaver. Dump settings do not replace model access, data, or the rest of the recipe. `dump_every_n_steps` controls the step interval; each enabled data category is saved when its trainer stage runs.

<a id="dump-directory-structure"></a>

## Dump Contents

```text
debug_dump/
├── prepared_trajectories/
│   └── step_000000.pt
├── datums/
│   └── step_000000.pt
└── training_metrics/
    └── step_000000.pt
```

- `prepared_trajectories`: complete rollout results after advantage preparation, including their token data and training signals.
- `datums`: converted data sent to the Weaver training API.
- `training_metrics`: metrics returned by the training service.

Current prepared dumps use `format_version=3`: ordinary trajectories keep top-level token fields, while segmented results keep internal `segments` dictionaries. Treat these files as training data and keep them with the experiment that produced them.

<a id="trajectory-replay-contract"></a>

## Replay in Development Tools

The internal mock worker can load saved trajectories when constructed with a replay configuration. This is a developer API, not a replay option on the local training command: the public recipe schema does not expose `trajectory_load_path` or `trajectory_format`.

Replay restores complete trajectories and checks that every saved group contains exactly one trajectory for each expected `run_id` slot. A saved `expected_rollouts` value must match the trigger task. Missing members, duplicate slots, or inconsistent legacy outcomes are rejected. The reader accepts v2 nested dumps and older flat rows only when it can reconstruct complete executions and comparison groups.

The first trigger replays the loaded file once per worker instance. Groups receive new `group_id` values, trajectories receive new `rollout_id` values, and `run_id` slots remain unchanged. Later triggers drain the dataloader. Use one worker in development replay to avoid loading and replaying the same file independently in several workers.

<a id="troubleshooting"></a>

## Troubleshooting

- If directories are empty, check `enable_data_dump`, the category switches, the dump interval, and whether training reached the corresponding stage.
- A dump run still calls Weaver. Use the separate single-batch command when you only need rollout debugging.
- NorthRL's local dump options do not enable debug settings inside a remote service; those belong to the service operator.
