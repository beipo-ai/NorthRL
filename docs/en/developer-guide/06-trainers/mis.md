English | [简体中文](../../../zh/developer-guide/06-trainers/mis.md) | [Documentation index](../../index.md)

# Rollout and Training Mismatch Correction (MIS)

NorthRL can measure the mismatch between rollout and frozen training logprobs,
and optionally correct the policy-gradient contribution of rollout samples.
Diagnostics report that mismatch without changing training advantages or loss
masks. Correction computes fixed weights and rejection masks before training,
then passes the resulting advantages and loss masks to Weaver.

## Supported Configuration

Production mismatch diagnostics and MIS correction use `backend.type: weaver`;
the `mock` backend is also available for tests. Both modes require:

- `algorithm.type: grpo`.
- Unfiltered rollout sampling: `top_p: 1.0`, `top_k <= 0`, and `min_p: 0.0`.
- `rollout.sampling.temperature: 1.0` and `repetition_penalty: 1.0`.

`algorithm.mis.enabled: true` enables correction and also reports basic mismatch
metrics. Correction additionally requires
`algorithm.sampling_logprob_source: recomputed`, the native `importance_sampling`
or `ppo_clip` selected with `algorithm.loss.mode`; custom callbacks are unsupported.

`algorithm.mis.enabled: false` with `get_mismatch_metrics: true` enables diagnostics
alone. This mode supports either `sampling_logprob_source: inference` or
`recomputed`, and does not restrict the configured training loss or custom loss.
It does not attach correction weights or masks, alter the training denominator,
or enable MIS rejection. The shared converter still skips sequences whose loss
mask is all zero, regardless of whether MIS is enabled.

Active top-p, top-k, or min-p filters are rejected during configuration
validation. The probability comparison also requires trajectories without
`sampling_mask`. Supporting these filters requires verifying that Weaver's
`forward_logprob` and training losses apply the same candidate-set
normalization; the initial implementation does not assume this contract.

Both modes reject non-unit temperature and repetition penalty: the current
sampler interface has not been verified to return the exact behavior-policy
probabilities under those transformations. This is an interface limitation,
not a restriction of importance sampling itself. Supporting other values
requires sampler log probabilities of the **actual behavior policy**, including
every sampling transformation.

Merge this example into an existing Weaver training recipe that meets the support
requirements above. Model, data, optimizer, and other required settings are omitted:

```yaml
backend:
  type: weaver

rollout:
  sampling:
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    min_p: 0.0
    repetition_penalty: 1.0

algorithm:
  type: grpo
  loss:
    mode: importance_sampling  # Also supports ppo_clip
  sampling_logprob_source: recomputed
  mis:
    enabled: true
    get_mismatch_metrics: false  # Basic mismatch metrics are also emitted when enabled=true.
    tis_mode: truncate
    tis_level: token
    tis_lower_bound: null
    tis_upper_bound: 2.0
    use_rs: false
    rs_level: token
    rs_lower_bound: null
    rs_upper_bound: null
    rs_veto_threshold: null
    tis_batch_normalize: false
```

The example enables correction. Both `enabled` and `get_mismatch_metrics` default
to `false`; all other MIS values shown are also defaults. Existing recipes that
omit `algorithm.mis` keep their previous behavior.

To record mismatch without correcting the loss, use:

```yaml
algorithm:
  sampling_logprob_source: recomputed  # Also supports inference for diagnostics alone.
  mis:
    enabled: false
    get_mismatch_metrics: true
```

With `recomputed`, diagnostics reuse the training batch's existing recompute
forward. With `inference`, diagnostics add a read-only training-model forward
pass; the training Datum still uses the rollout logprobs as its denominator.
Both paths split actor forward requests by `algorithm.optim.mini_batch_size`
and aggregate diagnostics over the full batch. Diagnostics run before optimizer
updates and use the original loss mask.

## Probability Sources

At each response token:

- `rollout_prob` is the probability under the behavior policy that sampled it.
- `old_prob` is the probability recomputed by the training model before this
  training batch's first optimizer update.
- `current_prob` is the current training probability during forward/backward.

The correction uses the frozen rollout-to-training ratio:

```text
log_ratio        = old_logprob - rollout_logprob
correction_ratio = old_prob / rollout_prob = exp(log_ratio)
```

With correction enabled, Weaver's native training loss uses a separate ratio:

```text
train_ratio = current_prob / old_prob
            = exp(current_logprob - old_logprob)
```

When correction is enabled, NorthRL preserves the sampler scores as
`trajectory.rollout_log_probs` before recomputation. After recomputation,
`trajectory.old_log_probs` contains `old_logprob` and becomes the Datum's
`loss_fn_inputs.logprobs`. Weaver computes `current_logprob` in its own training
forward. Reference-policy logprobs used for KL are a separate field and are not
either denominator above.

Diagnostics compare rollout scores with a genuine training-model forward.
They do not replace the training denominator: with source `recomputed`, the
existing training path still replaces it; with source `inference`, the additional
diagnostic forward leaves it as the rollout scores. In either mode, the metric
name `training_*` refers to that frozen forward, before optimizer updates.

For token-level correction without truncation, filtering, or normalization,
`correction_ratio * train_ratio = current_prob / rollout_prob`. With the
configured corrections, Weaver receives
`corrected_advantage = advantage * correction_weights`:
`importance_sampling` uses this advantage in its surrogate, and `ppo_clip`
uses it with its usual clipped `train_ratio`. MIS bounds control
`correction_ratio`; PPO clipping controls `train_ratio` independently.

## Configuration Reference

Every field below is under `algorithm.mis`.

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enables correction: strict recomputation validation, weights, optional rejection, and basic mismatch metrics. |
| `get_mismatch_metrics` | `false` | Reports basic mismatch metrics without correction when `enabled=false`; correction already reports them when `enabled=true`. |
| `tis_mode` | `truncate` | `none`, `truncate`, `clip`, or `mask`; selects the weighting and TIS rejection rule. |
| `tis_level` | `token` | `token`, `sequence`, or `geometric`; selects the ratio aggregation used by TIS. |
| `tis_lower_bound` | `null` | Effective lower bound `L`; `null` means `1 / tis_upper_bound`. Used by `clip` and `mask`, and inherited by RS when needed. |
| `tis_upper_bound` | `2.0` | Upper bound `U`; must be a finite positive number and cannot be `null`. |
| `use_rs` | `false` | Enables deterministic rejection using a separate ratio level and interval. |
| `rs_level` | `token` | `token`, `sequence`, or `geometric`; independent of `tis_level`. |
| `rs_lower_bound` | `null` | RS lower bound; `null` inherits the effective TIS lower bound. |
| `rs_upper_bound` | `null` | RS upper bound; `null` inherits `tis_upper_bound`. |
| `rs_veto_threshold` | `null` | Optional whole-sequence veto based on the smallest original token ratio; only active with `use_rs: true`. |
| `tis_batch_normalize` | `false` | Divides weights by their mean over the original training batch, using the original loss mask. |

All supplied bounds and veto thresholds must be finite and positive. When
MIS is enabled, `clip` and `mask` require `L < U`; enabled RS requires its
resolved lower bound to be smaller than its resolved upper bound. The TIS
lower bound does not clip weights in `truncate` or `none` mode. Geometric
TIS weights cannot be combined with batch normalization; `tis_mode: none`
is exempt because its weights are one.

### Ratio Levels

Only tokens selected by the **original** trajectory loss mask participate
in aggregation. Let `V_i` be those positions and `n_i = len(V_i)`:

| Level | Aggregated log ratio | Interpretation |
| --- | --- | --- |
| `token` | `log_ratio[i,t]` | Each token has its own correction. |
| `sequence` | `sum(log_ratio[i,t] for t in V_i)` | Product of the token ratios, broadcast to all valid tokens in the sequence. |
| `geometric` | `sum(log_ratio[i,t] for t in V_i) / n_i` | Geometric mean of token ratios, broadcast to all valid tokens. |

For numerical stability, the aggregated log ratio is clamped to `[-20, 20]`
**after aggregation**, then exponentiated. Large positive and negative token
differences can therefore cancel before this clamp. Empty sequences have
no effective tokens and produce zero weights and masks.

### Weighting Modes

Let `correction_ratio` denote the ratio at `tis_level` after the numerical clamp. The
following rules apply at each originally valid position:

| `tis_mode` | Stored weight | TIS correction mask |
| --- | --- | --- |
| `none` | `1` | `1` |
| `truncate` | `min(correction_ratio, U)` | `1` |
| `clip` | `max(L, min(correction_ratio, U))` | `1` |
| `mask` | `correction_ratio` | `1` if `L <= correction_ratio <= U`, otherwise `0` |

Bounds are inclusive for filtering. `mask` preserves the raw weight even
at rejected positions and records rejection separately. This matters for
batch normalization, which still includes those original positions.

`tis_mode: none` with `use_rs: true` enables rejection alone. With
`use_rs: false`, it applies unit weights and no rejection. In both cases,
`enabled: true` still enforces the MIS probability and recomputation
contracts. To observe mismatch without those correction-stage behaviors, use
`enabled: false` with `get_mismatch_metrics: true` instead.

### Rejection and Veto

RS recomputes ratios from the original `log_ratio` values at `rs_level`, independently
of TIS clipping or weighting. It intersects the existing correction mask
with the inclusive interval `[rs_lower_bound, rs_upper_bound]`. At sequence
or geometric level, that interval decision applies to the whole sequence.
RS changes the mask and does not multiply an additional ratio into the
weight.

When `use_rs: true` and `rs_veto_threshold: v` is set, any original valid
token satisfying `log_ratio[i,t] < log(v)` rejects the entire sequence. This check
uses the original token log ratio before numerical clamping, including
tokens already rejected by an earlier mask. Equality with the threshold
is retained. With `use_rs: false`, the veto setting has no effect.

These are deterministic threshold filters. They do not draw random numbers
or accept samples with a probability proportional to their ratio.

For example, this override keeps unit weights and filters token ratios
outside `[0.5, 2.0]`, with an additional sequence veto:

```yaml
algorithm:
  mis:
    enabled: true
    tis_mode: none
    use_rs: true
    rs_level: token
    rs_lower_bound: 0.5
    rs_upper_bound: 2.0
    rs_veto_threshold: 0.0001
```

### Batch Normalization and Loss Reduction

When `tis_batch_normalize: true`, normalization happens after the weighting
and rejection decisions, but uses the **original** masks. Rejected positions
still contribute their stored weights to the normalization factor `Z`:

- Token level, or `tis_mode: none`: `Z` is the mean weight across all original
  valid tokens in the full input batch.
- Sequence level: compute each sequence's mean weight over its original valid
  tokens, then average these means across every sequence in the input batch.
  An empty or all-masked sequence contributes zero to this sequence average.

If `Z > 1e-8`, every stored weight is divided by `Z`; otherwise weights are
left unchanged. There is no second clipping pass after normalization, so a
final normalized weight can exceed `tis_upper_bound`.

NorthRL computes `Z` once over the complete prepared training batch, before
splitting it into optimizer batches. Slime's MIS helper computes its
factor in the loss call's microbatch. Consequently, the same settings need
not produce the same normalized weights when batch partitioning differs.

Weight normalization is also separate from Weaver's loss reduction. Weaver
receives the filtered loss mask and recomputes its loss average using that
mask; with `loss.reduction.mode: token_mean`, the denominator is the retained-token
count. Slime's integration retains the original mask count in its reducer.
NorthRL therefore does not reproduce Slime's loss scaling after rejection.
For example, rejecting one of two tokens changes a token-mean denominator
from two to one in Weaver, while Slime retains two.

## Comparison with Slime

This comparison uses Slime commit
[`4c193f1`](https://github.com/THUDM/slime/tree/4c193f1f37509cca70f0e88807a9305b70f63f4e),
with `custom_tis_function_path` selecting
[`examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/examples/train_infer_mismatch_helper/mis.py).
Loss comparisons refer to its Megatron `policy_loss` integration with the default
PG reducer. Slime's built-in `vanilla_tis_function`, custom PG reducers, and OPSM
are separate paths and are not the reference here.

### Shared Formulas and Switch Mapping

For identical logprobs, original masks, bounds, and normalization groups, the
common `truncate`, `clip`, and `mask` modes use the same correction formulas.
Both support token, sequence, and geometric ratios, clamp the aggregated log
ratio to `[-20, 20]` before exponentiation, inherit unspecified lower/RS bounds,
and use inclusive rejection intervals. RS is deterministic threshold filtering.
The optional veto checks raw token log ratios over the original valid tokens;
it can reject an entire sequence. Both normalize using original masks, including
rejected positions, and do not clip weights again after normalization.

The table uses NorthRL fields under `algorithm.mis`. Unlisted switches are false;
`get_mismatch_metrics` is abbreviated as `metrics` in this table only.

| Behavior | Slime MIS helper | NorthRL |
| --- | --- | --- |
| No correction or mismatch diagnostics | `use_tis=false`, `metrics=false` | `enabled=false`, `metrics=false` |
| Mismatch diagnostics only | `use_tis=false`, `metrics=true` | `enabled=false`, `metrics=true` |
| TIS, including TIS `mask` rejection | `use_tis=true`, `use_rs=false` | `enabled=true`, `use_rs=false`, same `tis_mode` |
| TIS plus RS and optional veto | `use_tis=true`, `use_rs=true` | `enabled=true`, `use_rs=true`, same `tis_mode` |
| RS and optional veto with unit weights | No active RS-only branch: `use_tis=false` returns before RS; `tis_mode=none` is unsupported | `enabled=true`, `tis_mode=none`, `use_rs=true` |

In both implementations, enabling correction also computes basic mismatch
metrics. Disabling the correction switch disables RS even if `use_rs=true`.
NorthRL's `tis_mode=none` therefore requires `enabled=true` to run RS; it still
requires recomputed logprobs and all other correction prerequisites.

### Probability Sources and Recompute Timing

Slime passes `batch["log_probs"]` to the helper as `train_log_probs` when
available. Otherwise it passes `current_logprob.detach()` from that loss
forward. The helper's correction ratio is always
`exp(train_logprob - rollout_logprob)`. NorthRL always uses the separate,
pre-update recompute result: `exp(old_logprob - rollout_logprob)`, frozen for
the prepared batch and every optimizer batch derived from it.
[Slime loss implementation](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/loss.py#L964).

With TIS enabled and `compute_advantages_and_returns=true`, Slime can omit
its separate old-policy forward when all conditions in
`can_reuse_log_probs_in_loss` hold: one optimizer step for the
rollout batch (`len(num_microbatches) == 1`, which may still contain several
microbatches), `loss_type=policy_loss`, `kl_coef=0`, and all of
`use_rollout_logprobs`, `get_mismatch_metrics`, `use_critic`, `keep_old_actor`,
and `use_opd` disabled. Routing replay must be disabled or use rollout routing
replay, and the advantage estimator must not be `gspo`. `use_tis` itself does
not disable this optimization. NorthRL correction has no equivalent detach fast
path; it always recomputes before training.
[Slime recompute condition](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/actor.py#L428).

For the supported token-level policy loss, both use
`exp(current_logprob - old_logprob)` as the training ratio. In Slime's fast
path, `old_logprob=current_logprob.detach()`: the ratio's value is one while
the numerator retains its gradient. Slime rejects `use_rollout_logprobs=true`
together with `use_tis=true`; NorthRL similarly requires `recomputed` for
correction. Diagnostics alone may retain rollout logprobs as the training
denominator in either system, with a separate training-model forward for the
comparison. [Slime configuration checks](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/utils/arguments.py#L1860).

### Loss Integration and Observable Differences

| Aspect | Slime reference integration | NorthRL with Weaver |
| --- | --- | --- |
| Applying weights | Multiplies per-token `pg_loss` in the loss helper | Multiplies Datum advantages before native `importance_sampling` or `ppo_clip`; Weaver computes the loss |
| Applying rejection | Returns modified response masks used for PG, entropy, and explicit KL loss | Multiplies the target-aligned Datum loss mask; PG, entropy, and explicit KL loss are also masked |
| Average after rejection | Keeps original `rollout_mask_sums` and token-count normalizers | Uses the filtered mask for reduction; token-mean divides by retained tokens, as described above |
| Weight normalization scope | Each loss microbatch, after gathering sequence positions across context-parallel ranks; no full training-batch normalization | Once over the full prepared training batch, before optimizer-batch splitting |
| All targets rejected in an optimizer batch | No MIS-specific optimizer-step skip in the inspected training loop | Skips forward/backward and the optimizer step, including momentum and weight-decay updates |
| Supported integration | Helper runs inside Slime's policy loss; it does not impose NorthRL's configuration restrictions | Correction requires `algorithm.type: grpo`, Weaver, native IS/PPO loss, and the sampling restrictions listed above |

For nonnegative fixed weights and matching token-level IS/PPO surrogates,
weighting advantages and weighting the unreduced PG loss are equivalent.
Weights do not directly multiply entropy or KL terms in either implementation.
The mask has the same scope across these loss terms, but the different averaging
denominator changes their scale after rejection. Equal weights and masks do
not imply equal final losses or optimizer updates. Slime's default sample
reducer also supports shared denominators across samples in one rollout group;
it should not be assumed identical to Weaver's `sample-mean`.
[Slime reducers](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/cp_utils.py#L47),
[Slime optimizer step](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/model.py#L681).

Slime also has training-logprob paths for temperature scaling and replaying
rollout top-p candidate sets. NorthRL currently rejects those sampling settings
for MIS because the corresponding sampler/Weaver probability contract has not
been verified; this is an integration limitation rather than a change to the
correction formula. [Slime logprob processing](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/loss.py#L83).

The 11 basic mismatch measurements use the same per-sequence formulas, but
logging is not a one-to-one match. Slime aggregates helper tensors through its
original loss reducer; NorthRL aggregates the full prepared batch using the
configured token/sample mean and excludes empty sequences from diagnostics.
Slime reports intermediate TIS/RS statistics and reduced per-sequence extrema;
NorthRL reports the `algorithm/mis/` fields below, including actual batch weight extrema and
skipped optimizer batches. NorthRL uses Python floats for correction and handles
non-finite diagnostic results by omitting the affected metric and reporting
`algorithm/mis/nonfinite_metric_count`; Slime's helper uses Torch tensor arithmetic
and does not implement that omission policy.

## Training Lifecycle and Alignment

With correction enabled:

1. Prepare complete trajectories, snapshot actual rollout logprobs, and recompute
   training logprobs for the whole step before any optimizer update.
2. Run the configured advantage estimator and optional tool-error postprocessor
   over the whole step, keeping zero-advantage trajectories.
3. Compute fixed correction weights, masks, and metrics over that step.
4. Convert trajectories to target-aligned Datums, applying MIS weights and masks.
   Skip sequences whose final loss mask is all zero before creating their Datum
   and source entry. Split the retained Datums into optimizer batches using
   `algorithm.optim.batch_size`. Normalize over each active batch,
   split its forward/backward requests using `algorithm.optim.mini_batch_size`,
   and run one optimizer step after all requests for that batch succeed. The
   recomputed denominator, weights, and masks stay fixed across all these updates;
   they are not recomputed from each new model state.

A trajectory's correction arrays have the same length as `tokens`. Position
zero is a no-context placeholder with weight and mask zero. Other positions
outside the original loss mask also have zero weight and mask.

The converter first aligns advantages, logprobs, reference logprobs, and
response fields using the original mask, then applies:

```text
input_tokens  = tokens[:-1]
target_tokens = tokens[1:]
Datum.advantages[t] *= correction_weights[t + 1]
Datum.loss_mask[t]  *= correction_mask[t + 1]
```

No tokens are deleted or reindexed after rejection. For example:

```text
tokens             = [10, 11, 12, 13]
original_loss_mask = [ 0,  0,  1,  1]
advantage          = 2
correction_weights = [ 0,  0, 0.5, 2]
correction_mask    = [ 0,  0,  1,  0]

Datum.input_tokens  = [10, 11, 12]
Datum.target_tokens = [11, 12, 13]
Datum.advantages    = [ 0,  1,  4]
Datum.loss_mask     = [ 0,  1,  0]
```

The rejected token still has its original position, denominator, and
reference logprob. Its mask removes its loss contribution. Correction weights
scale policy-gradient advantages; they do not directly scale entropy or KL
coefficients. The modified loss mask also controls which positions contribute
to Weaver's masked entropy and KL terms.

Zero-advantage filtering is controlled by the converter's `drop_zero_advantage_datums`
argument. The training entry described here, `RemoteApiGrpoTrainer`, passes `False`:
tokens with a nonzero loss mask still participate in mask-based reduction and any
configured entropy or KL terms even when their advantage is zero. Other callers
that retain the converter's default `True` filter out all-zero-advantage samples.
Regardless of this argument, `convert_trajectories_to_datums()` omits sequences
whose final loss mask is all zero before optimizer batching, without recomputing
advantages or changing the baseline and whitening.
If no Datums remain, `training/skipped_flag` is set and no forward/backward or
optimizer update is issued.

With correction enabled, missing, placeholder, malformed, or non-finite rollout
logprobs at effective positions fail the training batch before recomputation.
MIS validation of recomputed logprobs at effective positions runs within each
recompute attempt, so invalid trainer outputs are retried before correction.

With diagnostics alone, advantage preparation, Datum conversion, loss inputs,
and optimizer batching retain their shared behavior, including the all-zero-mask
check in the converter. The trainer adds no correction arrays. Source `recomputed`
reuses the existing recompute result; source `inference` makes one read-only
forward and keeps its result separate from the training trajectories. There is
no additional forward per optimizer batch.

With source `recomputed`, each training-denominator forward chunk makes one initial
attempt plus at most three retries, including validation of its returned logprobs.
Only the failed chunk is retried; successful chunks retain their results and router
replay records. Each retry logs a warning with the chunk's starting sequence index,
size, attempt, and exception. Results are written back only after all chunks succeed.
If a chunk exhausts its attempts, recomputation raises before optimizer updates,
regardless of MIS switches or router replay; there is no fallback to rollout
logprobs. Cancellation propagates without retry.

With source `inference`, failure of the independent diagnostic forward sets
`algorithm/mis/metrics_available=0` and omits the measurements while preserving training's
rollout denominator. Pure metric-computation failures are likewise isolated from
training in both modes. The sampler protocol still requires real logprobs, and
the service holder retains its strict checks for requested fields, including
`old_logprobs`.

## Metrics and Source Files

The names below are the formal step-log and W&B fields. They use
`algorithm/mis/` so that automatic workspaces group them in Algorithm.
Internal producers retain `mis/` names; normalization changes the output
names without duplicating legacy curves. The internal `mis/log_ppl_diff`
alias equals `mis/kl` and is published only as `algorithm/mis/kl`.
See [step metrics in W&B](../../metrics.md).

Basic mismatch metrics are emitted when `enabled` or `get_mismatch_metrics` is
true. `algorithm/mis/metrics_available` is one when the comparison has valid probability
inputs and at least one effective token, otherwise zero. The measurements use
the original loss mask over the complete prepared training batch, before
correction or optimizer-batch splitting. `algorithm/mis/mismatch_tokens` and
`algorithm/mis/mismatch_sequences` count the original effective tokens and nonempty
sequences used by the computation. A batch without effective tokens does not
report fabricated PPL, KL, or chi-squared values.

The 10 distinct basic measurements below are computed first for each nonempty sequence.
`train_logprob` is the frozen training logprob, `rollout_logprob` is the sampler
logprob, and `log_ratio = train_logprob - rollout_logprob`.
`mean_over_valid_tokens(...)` and `sum_over_valid_tokens(...)` use only that
sequence's original valid positions. Define:

```text
training_log_ppl = -mean_over_valid_tokens(train_logprob)
rollout_log_ppl  = -mean_over_valid_tokens(rollout_logprob)
log_ppl_diff    = training_log_ppl - rollout_log_ppl
clamp_log_ratio(log_ratio) = clamp(log_ratio, -20, 20)
```

| Metric | Per-sequence value |
| --- | --- |
| `algorithm/mis/training_log_ppl` | `training_log_ppl` |
| `algorithm/mis/training_ppl` | `exp(training_log_ppl)` |
| `algorithm/mis/rollout_log_ppl` | `rollout_log_ppl` |
| `algorithm/mis/rollout_ppl` | `exp(rollout_log_ppl)` |
| `algorithm/mis/kl` | `mean_over_valid_tokens(rollout_logprob - train_logprob)`, the direct rollout-to-training KL estimate |
| `algorithm/mis/k3_kl` | `mean_over_valid_tokens(exp(log_ratio) - log_ratio - 1)` |
| `algorithm/mis/log_ppl_abs_diff` | `abs(log_ppl_diff)` |
| `algorithm/mis/ppl_ratio` | `exp(log_ppl_diff)` |
| `algorithm/mis/chi2_token` | `mean_over_valid_tokens(exp(2 * clamp_log_ratio(log_ratio))) - 1` |
| `algorithm/mis/chi2_seq` | `exp(2 * clamp_log_ratio(sum_over_valid_tokens(log_ratio))) - 1` |

These sequence values are aggregated using `algorithm.loss.reduction.mode`:

- `token_mean`: weight each sequence by its original
  effective-token count.
- `datum_mean`: give each nonempty sequence equal weight.

Empty sequences contribute to neither denominator. For PPL, exponentiation
happens before this aggregation; the result is an average of sequence PPLs,
not the exponential of the aggregated log PPL. Likewise, absolute differences
are taken per sequence before averaging. These metrics describe rollout versus
frozen training probabilities, independently of the reference-policy KL term
in the training loss. Finite-sample estimates of `kl` and `chi2_*` can be negative.

PPL and K3 exponentials are not silently clipped. Only the chi-squared metrics
use the explicit `[-20, 20]` log-ratio bound, matching Slime's formulas. If any
sequence or the final aggregation produces a non-finite value for a metric,
that metric is omitted for the whole batch; the computation does not drop just
the overflowing sequence. `algorithm/mis/nonfinite_metric_count` counts omitted metric
names, not tokens. Other finite metrics are still reported, and
`algorithm/mis/metrics_available` can remain one. This numerical policy does not alter
correction weights or introduce a new training failure. The separate forward for
source `inference` reports client wall time as `timing_train/mismatch_forward_seconds`.
Mismatch-statistics computation still runs without a separate timing curve.

The following correction statistics are emitted only when `enabled=true`.
The trainer emits `algorithm/mis/original_tokens`, `algorithm/mis/kept_tokens`,
`algorithm/mis/original_sequences`, and `algorithm/mis/kept_sequences`, plus token and sequence
rejection fractions. A sequence is counted as kept when at least one token
survives. `algorithm/mis/veto_fraction` counts wholly vetoed sequences, and
`algorithm/mis/clipped_fraction` counts original tokens whose weights changed under
`truncate` or `clip`.

`algorithm/mis/raw_weight_{mean,min,max}` describes ratios before TIS processing, after
the numerical clamp. `algorithm/mis/final_weight_{mean,min,max}` describes stored weights
after processing and optional normalization. Both cover original valid
positions, including rejected positions; min and max are actual batch extrema.
`algorithm/mis/batch_norm_factor` records the divisor, or one when normalization does
not apply. If Datum conversion leaves training data, `algorithm/mis/skipped_batches`
counts empty batches removed after optimizer batching, and `algorithm/mis/optimizer_steps`
counts the retained batches. Sequences omitted by the converter do not count as
skipped batches. If conversion leaves no Datums, these two metrics are absent;
use `training/skipped_flag=1` and `training/optimizer_step_count=0` instead.
MIS weight computation still runs without a separate timer.

```text
northrl/
  algorithm/mis.py                    # Pure correction and mismatch metrics
  utils/config_schema.py              # MISConfig and support validation
  trainer/remote_api_trainer.py        # Snapshot, recompute, correction, updates
  utils/finetune_service_utils.py      # Target alignment and Datum conversion
```

See [Remote API Trainers](remote-api-trainers.md) for the surrounding
training flow and advantage computation.
