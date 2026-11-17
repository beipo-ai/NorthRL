English | [简体中文](../../../zh/developer-guide/04-trajectory-pool/trajectory-semantics.md) | [Documentation index](../../index.md)

# Tasks, rollouts, and trajectories

**One rollout produces one `Trajectory`.** Ordinary trajectories store tokens, a loss mask, and a final reward directly. Results that need multiple sequences, such as E2B results, add internal `segments`. There is no independently exchanged `Segment` type, and a single-turn result does not need a one-segment wrapper.

<a id="task-to-training"></a>

## From a task to training

| Term | Meaning |
| --- | --- |
| task | A question or task that can be attempted multiple times |
| item | One execution request that the dataloader sends to a worker |
| rollout | One model or agent attempt at a task; it may include multiple model calls |
| trajectory | The complete result of one rollout, represented by one `Trajectory` |
| segment | One sequence record inside a segmented trajectory, not an independent execution result |
| group | Complete trajectories sampled together for the same task and compared by GRPO |
| Datum | One sequence submitted to the training service; filtering can reduce the count |
| sample | A generic term; specify whether a count means input tasks, trajectories, or Datum objects |
| trace | A saved execution record; a progress snapshot does not mean the rollout has finished |

```text
Direct:    item → rollout → Trajectory(tokens, loss_mask, reward) → Datum
Segmented: item → rollout → Trajectory(reward, segments=[{...}, {...}]) → multiple Datum objects
```

Multiple turns do not necessarily require multiple segments. The E2B recorder can append calls with continuous token prefixes to the same sequence; it starts a new segment when continuation is not possible. Model calls, internal segments, and complete trajectories are different counts.

<a id="sampling-groups"></a>

## The same task in independent sampling groups

`task_id` identifies the original task. `group_id` identifies the sampling group created each time that task is drawn. `rollout_repeat_n=R` produces R attempts in this group, each with a complete trajectory. Drawing the same task again can reuse its `task_id`, but must create a new `group_id`.

```text
task_id = 7, rollout_repeat_n = 2
First draw  → group A → trajectories A0, A1
Second draw → group B → trajectories B0, B1
```

A and B each compare their own two rewards. Matching `task_id` values must not merge them into one four-member group. `run_id` identifies a slot from 0 to R−1 within a group; `rollout_id` uniquely identifies an attempt.

Unique identity does not guarantee reproducible ordering. When one training batch contains multiple groups with the same `task_id`, random `group_id` values still affect sorting. Even with a fixed seed, optimizer batch membership can differ. This remains unresolved; see the [trainer ordering boundary](../06-trainers/remote-api-trainers.md#trajectory-ordering).

<a id="trajectory-storage"></a>

## One public Trajectory, two storage forms

```python
from northrl.northrl_types import Trajectory

single = Trajectory(
    tokens=[1, 2, 3],
    loss_mask=[0, 1, 1],
    reward=1.0,
    extra_fields={"old_log_probs": [0.0, -0.2, -0.3]},
)
assert single.segments is None

agent = Trajectory(
    reward=0.5,
    segments=[
        {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1],
         "old_log_probs": [0.0, -0.2, -0.3]},
        {"tokens": [4, 5], "loss_mask": [0, 1],
         "old_log_probs": [0.0, -0.4]},
    ],
)
assert agent.tokens is None
```

In the direct form, tokens/loss_mask are actual fields, not proxies to an internal segment. In the segmented form, `segments` is a list of internal dictionaries with tokens, loss_mask, logprobs, and other sequence fields. These forms are mutually exclusive; they do not store two copies of the token data.

This describes rollout results. The internal managed-SFT `SampleRef` carrier is an exception: `extra_fields["sample_ref"]`, `input_token_count`, and `datum_id` refer to a server-side sample. It carries no local tokens, loss_mask, or segments; Weaver reads the training data.

Reward, status, is_val, execution identity, score, and GRPO advantage belong to the complete trajectory. Internal segments have no independent reward or execution terminal state. A worker submits one complete object; pool, GRPO, and trainer preparation interfaces exchange `list[Trajectory]` throughout.

The trainer traverses sequences internally for reference KL, OPD teacher logprobs, logprob recomputation, MIS, and Datum conversion. It uses the ordinary trajectory itself or its internal segment records, retaining references and writing sequence fields back into them. It does not convert a single-turn result into an independent Segment or treat an internal segment as a complete Trajectory.

A remote `RemoteSegment` is a sequence record in the E2B wire protocol, not a complete rollout. The local worker places all segment token data from a `RemoteRolloutResult` into the internal dictionaries of one `Trajectory`, with the final reward on that complete object.

`RemoteRolloutTrace` retains the Trace name for progress snapshots saved while the remote rollout is still running. An existing trace does not indicate completion. `RemoteRolloutResult` is the remote result delivered to the local training workflow and carries the final reward. It is normally created after execution finishes. Timeout recovery can also attach the last valid snapshot to an `env_error` result with wire-level reward 0. This is not an evaluated zero-score task: the worker records `status="failed"`, `reward=None`, and no trainable segments.

Both types store sequence records in `segments`. New code uses `trace.segments` / `result.segments`; default JSON output and JSON Schema also use `segments`. The in-memory Pydantic trace/result contracts still accept historical input named `trajectories`, but serialize it as `segments`; supplying both names is an error. The compact schema-v2 wire format requires `segments` and does not provide that legacy alias. The old Python `.trajectories` property is unavailable, and readers of new JSON must use `segments`.

<a id="grpo-statistics"></a>

## The statistical unit for GRPO

GRPO reads the final reward once from each of N complete trajectories in a group:

```text
baseline = (reward_1 + ... + reward_N) / N
advantage_i = (reward_i - baseline) / (std(reward_1, ..., reward_N) + epsilon)
```

The main GRPO path uses sample standard deviation for N > 1 and an advantage of 0 for N = 1. All tokens of an ordinary trajectory, or all internal segments of a segmented one, use the complete trajectory's advantage.

The main training path calls `compute_grpo_advantage_for_trajectories()` from `northrl.algorithm.advantage.grpo`, reading scalar trajectory rewards directly. Its historical `use_run_ids` parameter now controls the standard-deviation convention and numeric types, not grouping; the main path passes `True`. `compute_grpo_outcome_advantage()` remains a tensor interface accepting rewards shaped `[sequence count, token count]`. It is currently used only by tests, not production trainers, and is imported from the same module.

`validate_trajectory_groups()` from `northrl.algorithm.groups` checks duplicate rollout IDs and valid rewards in groups that are not skipped. When `expected_rollouts` is supplied, it also checks that every expected `run_id` occurs exactly once. The old `northrl.algorithm.core_algos` module was removed; import these implementations directly.

For example, two trajectories have rewards 0 and 1. The first stores one sequence directly; the second contains three segments. **There are still two trajectories, with baseline 0.5**, not four equally scored sequences with baseline 0.75.

A successful result without trainable sequences can explicitly submit `Trajectory(segments=[], reward=valid_value)`. It participates in the baseline but produces no Datum. Once all R members arrive, let F be the number whose complete-trajectory status is not `success`, whose reward is invalid, or which are marked skipped. If F ≤ R/2, keep all valid members and choose F distinct valid members uniformly without replacement, copying each once to fill a failed slot. If F > R/2, skip training for the whole group.

Valid zero-sequence members can be copied. Zero rewards, equal rewards, and evaluated E2B `task_failed` outcomes are not failures for this rule. The trainer repairs groups after recording original metrics and before advantage computation or logprob recomputation. All grouped GRPO/OPD modes use the same rule.

<a id="group-delivery"></a>

## Submitting and delivering complete groups

1. The dataloader assigns a fresh `group_id` per draw. The R attempts have unique `rollout_id` values, `run_id` values from 0 to R−1, and `expected_rollouts=R`. `task_id` identifies the original task.
2. A worker constructs and submits a complete `Trajectory`. `execute_rollout` stages the object and writes it to the pool only after the custom `rollout` returns normally.
3. A training task that returns normally without a result produces a skipped trajectory. Returning `"fail"` or raising records a failed trajectory; exceptions still propagate, with no automatic retry. Validation tasks enter `rollout()` directly and send results to the validator, without training-pool terminal completion. There is no separate completion-record channel.
4. The pool waits for the original objects in all R slots and checks repairability. Fully-async additionally requires at least one training target among valid members; ineligible groups do not occupy ready slots. Other modes still deliver the original objects.
5. The trainer records original metrics, then repairs failed slots as above. GRPO and preparation preserve complete trajectory objects; token operations and Datum conversion handle their sequences internally. Batch scheduling is unchanged.

Duplicate submissions remain idempotent while a group is waiting, ready, or among the latest 1,024 consumed or discarded groups. Conflicting results are rejected, and deduplication storage is bounded.

`GroupedTrajectoryStore` manages this lifecycle with four containers:

| Container | Contents |
| --- | --- |
| `_groups` | Incomplete groups: `group_id → run_id → Trajectory` |
| `_ready_groups` | Complete groups: `group_id → list[Trajectory]`, queued by completion time; `data.train.group_order` selects FIFO (default) or LIFO delivery |
| `_rollout_signatures` | Shared `rollout_id → signature` index for pending, ready, and recently taken results |
| `_recent_groups` | `group_id → rollout ID list` in consumption/discard order; signatures for the oldest group expire beyond 1,024 groups |

“Consumed” means the trainer has taken the group from the pool, not that training has finished; discarded groups also enter this bounded history. Signatures remain until their historical group expires. Duplicate submissions do not requeue a group or refresh queue/history ordering.

Let B = `data.train.batch_size` and R = `data.train.rollout_repeat_n`. Grouped RL in `sync`, `batch-async`, and `fully-async` calls `take_groups(num_groups=B)` directly, normally delivering B complete groups and B × R complete trajectories. Fewer than B groups returns `None`; even with a backlog, each call takes at most B groups. `data.train.group_order` selects groups, with internal segments preserved in complete trajectory objects.

Normal group delivery does not additionally wait for the loader buffer to empty or all workers to become idle. In `sync` / `batch-async`, the scheduler usually releases one B-group input batch at a time. `fully-async` can assemble groups across input batches, leaving excess groups in the pool. Sync releases the next batch after training and weight synchronization; batch-async can release it earlier subject to staleness, independently of the number of optimizer updates per delivered batch.

The public RL loader loops across epochs to fill B inputs. Sync / batch-async can deliver a smaller final batch of complete groups only when the underlying finite source explicitly reports `is_finished() == True` and all workers are idle. Fully-async grouped RL instead discards an undersized final batch and records its exclusion counters. Missing-member diagnostics need not wait for the entire source to end: if fewer than B groups are ready, the current buffer is empty, all workers are idle, and pending groups remain, this is an error.

The Controller passes `data.train.group_order` to the internal pool. `fifo` (default) selects the earliest completed ready groups; `lifo` selects the latest. If A, B, C finish in that order, taking two yields A, B or C, B. Neither splits a group, and members remain ordered by `run_id`. “Old” and “new” refer to group completion order, not start time or sampling model version. Continuous LIFO use can leave older groups waiting indefinitely.

When more than B groups are ready, this setting affects batch membership, most often with a fully-async backlog. Every mode still takes at most B groups. The GRPO trainer later sorts by task and group identity, so pool order does not guarantee optimizer execution order.

These are general grouped-pool rules. `E2BRolloutWorker` accepts `weight_sync.mode="sync"` and `"fully-async"`; `RemoteRolloutTrace` requires at least one segment, but an evaluated `RemoteRolloutResult` can have none. The worker preserves such a result as `Trajectory(segments=[], reward=...)`; its reward still participates once in group statistics.

Only `None` from `get_trajectories()` means “wait.” In modes that deliver skipped or zero-sequence groups, these still return complete trajectory objects; the trainer skips the optimizer update and advances the consumed round. Fully-async excludes ineligible groups before they enter the ready queue.

Internally, `GroupedTrajectoryStore.take_groups(num_groups=...)` takes complete groups, while `SimpleTrajectoryStore.take_trajectories(num_trajectories=...)` takes complete trajectories. Both require a concrete positive integer, reject `None`, and return `None` without partial consumption when insufficient data is available. The grouped path checks `take_groups(B)` directly. Simple/SFT waits for the current loader round to be fully claimed and workers to be idle, then passes its actual `ready_trajectory_count()`. SFT can therefore deliver variable trajectory counts according to its token budget.

<a id="batch-units"></a>

## Batch and batch_size

`Batch` is a generic row-oriented field container, not a mandatory stage in the main training path. Each row from `Batch.from_trajectories()` holds one complete trajectory; nested segments add no rows. `to_trajectories()` preserves that relationship. Remove padding before conversion. There is no separate segment-batch conversion interface.

| Location | Unit and meaning |
| --- | --- |
| `data.train.batch_size` (RL) | Number of original input tasks in a batch, before repeat |
| Dataloader buffer | Execution requests waiting to be claimed, after repeat |
| Grouped-pool `groups_per_batch` | B complete groups per normal delivery in every mode; at most B even with a backlog |
| Grouped-pool `group_size` | R expected complete trajectories per group |
| Pool / trainer `list[Trajectory]` | Complete execution results, including failed, skipped, and successful zero-sequence results |
| `Batch.metadata["batch_size"]` | Current row count including padding; each original `from_trajectories` row represents a complete trajectory |
| `algorithm.optim.batch_size` (RL) | A positive integer targets this many Datum objects per optimizer update; omitted or `null` in sync / batch-async / fully-async uses all Datum objects from the delivered batch |
| `algorithm.optim.mini_batch_size` (RL) | Maximum Datum objects in one actor forward-only or forward/backward request |
| SFT input batch | Independent Datum objects organized by token budget, not a fixed task count |

A `null` optimizer batch size means one update over the whole delivered batch. This is separate from the concrete count required by the internal group/trajectory retrieval APIs. In sync / batch-async / fully-async RL, omitted or `null` uses all Datum objects produced by preparing and converting the B complete groups into one optimizer batch. Only the finite-source final batch described above can contain fewer groups. Input-release timing does not change this delivery count or require the grouped pool to wait for all workers to become idle.

Each group computes GRPO advantages on complete trajectories before the delivered batch is converted. `mini_batch_size` only limits Datum objects per forward/backward request. Gradients from all requests accumulate into one optimizer update, without shuffling or padding. Skipped batches, including those with no trainable Datum objects, do not update parameters.

An explicit positive integer retains Datum-level optimizer splitting, shuffling, and final-batch padding, which can place sequences from one trajectory into different optimizer batches. `no-sync` RL requires a positive integer. SFT retains its full token-budget batch and accepts omitted or `null` batch_size. Complete-group delivery and optimizer-update boundaries are separate concepts: one update per delivered batch neither changes loss normalization nor guarantees equal weight per trajectory.

<a id="prompt-mean"></a>

## Comparison-group weights with Prompt-Mean

`algorithm.loss.reduction.mode: prompt_mean` defines complete comparison groups by `(model_tag, group_id)`, not by task_id, trajectory, or segment. Suppose an optimizer update contains G valid groups, and all rollouts/segments of group g contain T_g original valid response tokens. Each policy-loss token receives weight `1 / (G * max(T_g, 1))`.

Tool context, prompt, and padding tokens are excluded from the denominator. MIS rejection does not change T_g; it only zeroes the rejected token's contribution. Splitting segments or forward/backward requests therefore does not change a group's weight.

References: Step 4, Token aggregation item 3 in the [Mercor / SkyRL training guide](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/); [MiMo-V2.6 technical report, page 8, §4.1, equation (1)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=8) and [page 21, §5.1](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf#page=21). These sources describe the aggregation definition; the configuration restrictions below belong to NorthRL's implementation.

Zero-advantage groups and successful zero-token groups remain in G. Repaired groups use their replacement members; unrepairable groups are excluded. A successful zero-sequence trajectory still participates in its group's reward/advantage calculation even though it produces no Datum. If the entire batch has no trainable tokens, the optimizer update is skipped.

This mode requires omitted or `null` `algorithm.optim.batch_size`, updating complete groups together. `mini_batch_size` only splits requests. The initial implementation supports GRPO's native `importance_sampling` / `ppo_clip` with entropy, KL, and custom loss disabled. Existing loss modes remain unchanged. See [trainer configuration](../11-configuration-reference/trainer-config.md#prompt-mean-loss-aggregation) for configuration and numeric comparisons.

<a id="counts-and-dumps"></a>

## Counts and saved data

Sequence count measures processable sequences: 1 for an ordinary trajectory, the internal segment count for a segmented trajectory, and 0 for a zero-sequence result. This is a counting unit, not a new data type. Trajectory count always counts complete trajectory objects.

| Metric | Meaning |
| --- | --- |
| Pool `ready_trajectories` / `ready_groups` | Ready complete executions / comparison groups; sync / batch-async include skipped groups, while fully-async queues eligible groups only |
| Pool `ready_sequences` | Ready trainable sequences; `ready_segments` / `finished_trajectories` remain compatibility aliases |
| `batch/trajectory_count` | Complete rollouts passed to the trainer, including failed/skipped/empty results |
| `batch/segment_count` | All sequences stored in input complete trajectory objects |
| `batch/resampled_group_count` | Groups repaired in this batch |
| `batch/resampled_trajectory_count` | Failed slots filled by copying valid members in this batch |
| `algorithm/trajectory/reward_observed_count` / `algorithm/trajectory/reward_mean` | Complete trajectories with finite final rewards / mean with each complete trajectory's reward counted once |
| `batch/trainable_segment_count` | Trainable sequences after Datum/mask processing, including failed-slot replacements but excluding optimizer-padding copies |

Periodic sync and batch-async progress logs count the current dataloader input round:

```text
Progress [sync]
  Rollouts: 28/64 finished | 36 in progress | 0 queued
  Groups: 3/8 complete
  Segments: 39 ready for training
```

`finished` counts complete `Trajectory` objects received by the pool, including failed, skipped, and successful zero-sequence results. `in progress` counts claimed rollouts without a delivered complete result, including waiting and retries. `queued` counts unclaimed requests. Their sum is the round's execution-request count after repeat; it is not inferred from active-worker counts.

`Groups` counts groups whose members have all returned, including skipped groups. `Segments` uses the sequence unit: 1 for ordinary, internal segment count for segmented, and 0 for zero-sequence trajectories. For grouped training it counts only complete, non-skipped groups, before the trainer's final Datum filtering.

All counts come from the same round snapshot deduplicated by `rollout_id`. They remain after the trainer takes results and reset when the next input batch is released. Observing logs does not change the pool's complete-group delivery boundary. SFT omits the GRPO `Groups` line.

Fully-async progress accumulates across input batches and displays pool inventory separately:

```text
Progress [fully-async]
  Rollouts: 1240 finished total | 36 in progress | 20 queued
  Pool groups: 3 ready | 5 partially returned | 8 per batch
  Pool segments: 39 ready for training
```

`finished total` is the cumulative number of rollouts that delivered complete results in the current loader session. Neither new input batches nor trainer consumption resets it. `in progress` includes unfinished attempts from earlier input batches; retries do not add logical executions. `queued` counts loaded but unclaimed items, not the remaining dataset.

Completion increments only when a claimed `rollout_id` first delivers a terminal result; duplicate or stale results do not count twice. The counter stores active IDs and aggregate counts, not every completed identity forever. The loader gives newly issued tasks fresh IDs, while internal retries retain the current ID. Restarting, explicitly resetting, or skipping batches during restoration restarts this accounting.

`Pool groups` counts a group as `partially returned` only after some but not all results arrive; groups with no returned results are excluded. `ready` counts eligible complete groups only; fully-async excludes unrepairable or zero-target groups before queuing them. `per batch` is the complete-group target for normal delivery for this model. `Pool segments` counts sequences only in complete valid groups.

Pool values are current inventory and decrease when the trainer takes results. Multiple models show separate inventories and delivery targets. Custom loaders without identity-based statistics retain active-worker and pool-inventory reporting.

Prepared trajectory dumps use `format_version=3`. Ordinary trajectories keep top-level tokens/loss_mask; segmented trajectories keep internal `segments` dictionaries. Reward, status, and identity are stored once. `num_trajectories` counts complete objects, `num_sequences` counts sequences, and `num_segments` remains a compatibility count. Replay accepts earlier nested v2 and old flat dumps, but rejects input from which complete comparison groups cannot be recovered.

<a id="related-guides"></a>

## Related developer guides

- [Core data types](../02-core-architecture/data-types.md)
- [Trajectory pool](trajectory-pool.md)
- [Rollout workers](../05-rollout-workers/overview.md)
- [Remote trainers](../06-trainers/remote-api-trainers.md)
- [Synchronous rollout and optimizer updates](../06-trainers/sync-rollout-semantics.md)
- [Runtime progress and live metrics](../08-features/live-metrics.md)

See [training-step metrics and W&B](../../metrics.md) for the current field dictionary, lookup methods, and counting conventions. Old rollout/datum/round count aliases are no longer emitted as new W&B curves.
