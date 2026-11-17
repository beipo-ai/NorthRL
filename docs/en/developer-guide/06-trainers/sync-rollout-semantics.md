English | [简体中文](../../../zh/developer-guide/06-trainers/sync-rollout-semantics.md) | [Documentation index](../../index.md)

# Complete rollouts, synchronized training, and SWE metrics

Training data and groups follow [Tasks, rollouts, and trajectories](../04-trajectory-pool/trajectory-semantics.md): one rollout produces one complete `Trajectory`. When it contains multiple sequences, all of them belong in its `segments`. For task execution stages and live observation, see the [E2B lifecycle](../05-rollout-workers/e2b-runtime.md#lifecycle) and [live metrics](../08-features/live-metrics.md).

<a id="optimizer-update-boundary"></a>

## What one optimizer update includes

```yaml
data:
  train:
    batch_size: 8
    rollout_repeat_n: 8
algorithm:
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
training:
  weight_sync:
    mode: sync
```

Each round samples 8 input tasks and creates 8 independent comparison groups. Each group runs 8 rollouts, producing 64 complete rollouts. If filtering leaves 100 Datums, training makes 7 forward/backward calls; the last contains 4 Datums. Gradients accumulate across these calls before a single optimizer update. Full-batch mode does not shuffle or duplicate Datums to pad the batch.

When `algorithm.optim.batch_size` is omitted or `null`, sync, batch-async, and fully-async modes all use the complete batch delivered by the pool. Fully-async delivers complete groups without waiting for every worker in a dataloader round to become idle. `mini_batch_size` limits the number of Datums in one forward/backward request; it does not change normalization or the optimizer update boundary. No-sync RL still requires an explicit positive integer; SFT retains token-budget batching. An explicit positive integer retains Datum-level batch splitting and final-batch padding.

GRPO reads one reward per complete trajectory and applies the resulting advantage to every sequence inside it. Sampling the same task again must create a new `group_id`; groups from different rounds must not be merged by `task_id`. `algorithm.loss.reduction.mode: prompt_mean` weights complete comparison groups equally, subject to its [configuration constraints and token normalization](../11-configuration-reference/trainer-config.md). The number of segments does not change a group's weight.

<a id="identity-and-completion"></a>

## Identity and completion boundaries

| Field | Meaning |
| --- | --- |
| `task_id` | Original task identifier |
| `group_id` | Independent comparison group created by this sampling event; resampling the same task creates a new ID |
| `run_id` | Execution slot within the group, from 0 to R−1 |
| `expected_rollouts` | Expected number of complete results in the group, R |
| `rollout_id` | Unique ID for one logical execution, retained across environment retries |
| `segment_index` | Sequence position inside a segmented result, not a separate rollout |

Each worker submission is one complete parent result: successful, failed, skipped, or a valid result with zero sequences. The pool retains all R original results for the group. Let F be the number of members whose parent status is not `success`, whose reward is invalid, or which are already marked to skip. If F ≤ R/2, the trainer keeps every valid member, uniformly samples F distinct valid members without replacement, and copies each once to fill the failed slots. If more than half fail, it skips the whole group. Replacement happens after recording original metrics and before advantage calculation and logprob recomputation. A batch with no trainable Datums still advances the training step, but does not update parameters or publish new weights.

The canonical sequence field is `segments`. Some in-memory contracts accept the historical `trajectories` alias, but remote trace/result files require gzip-wrapped compact schema v2 with `segments`; the wire reader does not fall back to legacy JSON. A trace is a progress snapshot; its existence does not mean the agent or evaluation has finished.

<a id="task-and-training-status"></a>

## SWE task status and training status

| E2B result | Parent `Trajectory.status` | Training and metric meaning |
| --- | --- | --- |
| `success` | `success` | Valid task result; the final reward is counted once |
| `task_failed` | `success` | Valid zero-score result that participates in GRPO; failing to solve the task does not discard the whole group |
| `env_error` | `failed` | Environment or evaluation failure; no valid reward, and partial training sequences are discarded. The group's failure count determines whether this slot is replaced |

A command failure without trustworthy stage markers for the current attempt is an `env_error`, not a valid zero-score sample. A fresh sandbox retry is permitted only before the agent command is dispatched. After dispatch, the executor tries to recover the existing attempt and does not automatically rerun the agent. See [execution time budgets and recovery](../11-configuration-reference/e2b-execution-budgets.md#recovery-behavior).

The original task status is retained in `e2b_status`; do not confuse it with training execution status. A successful result with zero segments still participates in group reward/advantage calculation, but produces no Datum. Evaluation can still run after context exhaustion. If the worker actually truncates an overlong segment, however, it resets the entire rollout's training reward and `swebench_resolved` to zero.

<a id="metric-definitions"></a>

## Metric definitions

The table uses current canonical logging names. W&B adds stable display ranks, for example `algorithm/trajectory/reward_mean` appears as `01_algorithm/01_trajectory/reward_mean`. Other logging backends retain canonical names.

| Metric | What it counts |
| --- | --- |
| `batch/trajectory_count` | Original complete parent results delivered by the pool, including failed/skipped/valid zero-sequence results |
| `batch/segment_count` | Saved sequences in those original results, including sequences in skipped groups |
| `algorithm/trajectory/reward_observed_count`, `algorithm/trajectory/reward_mean` | Count and mean of valid final parent rewards, counted once per original execution |
| `batch/failed_trajectory_count`, `batch/skipped_trajectory_count` | Training execution outcomes, not the task-solving pass rate |
| `outcome/task_failed_count`, `outcome/env_error_count` | Unsolved tasks and environment/infrastructure failures, counted separately |
| `batch/empty_trajectory_count` | Original complete results with zero saved sequences in this trainer batch |
| `batch/group_count` | Original comparison groups delivered in this trainer batch; omitted for ungrouped algorithms |
| `batch/resampled_group_count`, `batch/resampled_trajectory_count` | Groups repaired by replacing failed slots, and the number of replacement trajectories |
| `batch/trainable_group_count`, `batch/trainable_trajectory_count` | Groups and parent trajectories contributing at least one trainable segment after preparation |
| `batch/trainable_segment_count` | Segments with valid response training targets after Datum conversion and masking, before optimizer padding |
| `batch/trainable_response_tokens_total` | Valid response training tokens in those Datums, before optimizer padding |
| `training/optimizer_step_count`, `training/forward_backward_call_count` | Actual completed parameter updates and forward/backward requests |

For example, one rollout has reward 1 and 3 saved segments, and another has reward 0 and 1 saved segment. Then `batch/trajectory_count=2`, `batch/segment_count=4`, and `algorithm/trajectory/reward_mean=0.5`. Environment failures have no valid score and do not enter the reward mean as invented zeroes.

Original input counts and rewards are captured before failed-slot replacement. Trainable counts include replacement members after Datum/mask processing, but exclude copies added solely for optimizer padding. These are different populations; a raw saved segment is not necessarily trainable.

The old `rollout/*`, `round/*`, `datum/*`, and sequence-count fields under `training/*` are retired from step logging. In particular, `round/task_count` and `round/accepted_sequences` are not current chart names. A pre-trainer eligibility count cannot be mechanically renamed to a post-mask trainable count. For update counters, the old `training/optimizer_steps` and `training/forward_backward_calls` producer aliases map to the current names in the table.

Batch observations apply to both synchronous batches and each fully-async delivery. One fully-async trainer batch is not necessarily a complete dataloader round. Runtime `live/*` snapshots remain in ordinary logs, include attempt-level retry workload, and do not create W&B history rows. See [live metrics](../08-features/live-metrics.md) and the [training metrics reference](../../metrics.md).

<a id="tracing-and-dumps"></a>

## Tracing and saved data

With `runtime.debug.enable_data_dump` enabled and `dump_dir` configured:

- `rollout_traces/e2b_rollout_<rollout_id>.json.gz` saves compact remote segments, task identity, and sampling versions using gzip-wrapped schema v2.
- `rollout_rounds/step_<six-digit batch index>.json` saves every parent result in this trainer batch, including zero-sequence, failed, and skipped results. It retains both training `status` and task `e2b_status`. `DataDumper` uses the `remote_api_dump_options.prepared_trajectories` switch and `dump_every_n_steps` frequency for this file. Its filename and JSON `step` match the prepared/datums dump for the same batch, starting at 0. JSON `training_step` counts completed rounds, starting at 1.
- Prepared trajectory dumps use `format_version=3`: parent identity and reward appear once, with sequences nested in `segments`. They do not revert to the old flat-segment format.

The reward in the training records is the final training score submitted by the worker; environment failures have `null`. Live evaluation metrics describe sandbox execution, while the training score can also be affected by truncation and complete-group filtering. Interpret them separately.
