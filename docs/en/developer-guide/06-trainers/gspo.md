English | [简体中文](../../../zh/developer-guide/06-trainers/gspo.md) | [Documentation index](../../index.md)

# GSPO

[Back to algorithm components](algorithm-components.md)

Set `algorithm.loss.mode: gspo` with `algorithm.type: grpo` to train with
[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071).
The example uses the GRPO estimator to compute one scalar advantage per complete
rollout, with group standardization enabled (`norm_adv_by_std: true`). GSPO changes
the actor objective to a sequence likelihood ratio with sequence-level clipping.

## Two levels of averaging

For complete trajectory `i`, let `M_i` be the response positions selected by
`loss_mask` across all of its segments. Let `I` contain the trajectories in the
optimizer batch for which `|M_i| > 0`:

```text
log_r_i = sum(log_pi_new[t] - log_pi_old[t], t in M_i) / |M_i|
r_i     = exp(log_r_i)
L_i     = -min(r_i * A_i, clip(r_i, 1 - eps_low, 1 + eps_high) * A_i)
L       = sum(L_i, i in I) / |I|
```

First, average the token log-ratios within one complete trajectory, take the
exponential, and apply sequence-level clipping once to obtain `L_i`. The sequence
ratio retains its gradient. Then average the `L_i` values across the optimizer
batch. `trajectory_mean` names this second, outer mean.

Each nonempty trajectory has equal weight regardless of response length or
segment count. All of its segments contribute to the same `log_r_i` and `L_i`;
`datum_mean` instead treats emitted segments as separate Datums. Zero-advantage
trajectories contribute zero loss and still count in `|I|`. Masked prompt and
tool-result tokens do not enter `M_i`. Trajectories with no effective response
tokens are excluded from `I`. If `I` is empty, the optimizer update is skipped.

## Run the example

Use `recipe/math/weaver_gspo.yaml` with the local execution instructions
in the [quick start](../01-getting-started/quick-start.md). Prepare the math data,
model, and Weaver credentials there before launching. The recipe defaults to
console and W&B logging; configure W&B or select console-only logging as shown
in the quick start. Its relevant settings are:

```yaml
algorithm:
  type: grpo
  advantage:
    estimator: {mode: grpo, params: {norm_adv_by_std: true}}
    postprocessing: {mode: null}
  loss:
    mode: gspo
    params:
      clip:
        ratio_low: 0.0004
        ratio_high: 0.0006
    reduction:
      mode: trajectory_mean
  sampling_logprob_source: recomputed
  entropy_coeff: 0.0
  kl:
    mode: null
  mis:
    enabled: false
  optim:
    lr: 1.0e-4
    batch_size: null
    mini_batch_size: 16
    submit_ahead: 0
backend:
  type: weaver
```

Clipping widths must be explicit. `loss.params.clip.ratio` supplies a shared width when an
individual `ratio_low` or `ratio_high` is omitted. The lower width must be in
`[0, 1)` and the upper width must be finite and nonnegative. The example values
are starting settings, not a claim that these reproduce the GSPO paper.

## Execution and scope

NorthRL maps the loss name to its built-in client-side GSPO callback through
Weaver's `forward_backward_custom`; it does not require a server loss named
`gspo`. The callback receives raw advantages and the entire optimizer batch.
It normalizes exactly once, including when `mini_batch_size` splits the
forward and surrogate-backward requests. `batch_size: null` is required to
keep all segments of each complete trajectory in one optimizer batch; it also avoids
Datum-level shuffle and tail padding.

Both ordinary single-turn trajectories and multi-segment agent trajectories
are supported. The segments of one trajectory belong to a single complete
rollout. NorthRL records the owning complete trajectory and the expected segment
count on each emitted Datum, preserving router-replay and sampling-mask metadata.
Splitting the same masked response into more segments preserves
the loss and gradients. Each trained response token must be selected once by
the segment masks; repeated prompt/context tokens should remain masked.

The callback rejects missing/duplicate segments and conflicting advantages.
It uses emitted segment ordinals, so segments with no targets or no active
response positions omitted by the converter do not create gaps. Per-token advantages are rejected; all segments
inherit their complete trajectory's scalar advantage.

`sampling_logprob_source: recomputed` freezes the trainer's old-policy scores
before optimizer updates. With `inference`, the denominator is the stored
rollout-policy logprob. GSPO's differentiable current/old-policy sequence ratio
is separate from rollout mismatch importance sampling (`algorithm.mis`). This
implementation rejects enabled MIS correction, entropy regularization, KL
regularization, custom-loss overrides, `submit_ahead`, dual clipping
(`loss.params.clip.ratio_c`), and truncated-IS clipping (`loss.params.clip.truncation_c`). Mismatch
metrics without correction can still be collected.

Training logs include `algorithm/weaver/gspo/ratio_mean`, `algorithm/weaver/gspo/ratio_min`, `algorithm/weaver/gspo/ratio_max`,
`algorithm/weaver/gspo/clip_fraction`, `algorithm/weaver/gspo/active_trajectories`, and `algorithm/weaver/gspo/empty_trajectories`.

Ratio means and clip fractions are weighted by active trajectory count; extrema
use min/max and trajectory counts sum. The trainer omits segments with no active
response positions before invoking the callback, so its `empty_trajectories`
metric is zero in normal training. Direct callback callers can pass masked-out
Datums; the metric counts complete trajectories with no active response tokens.
Complete trajectories omitted by conversion and entirely skipped optimizer
batches do not contribute callback metrics.

Select GSPO with
`algorithm.loss.mode: gspo` and `algorithm.loss.reduction.mode: trajectory_mean`.
GSPO requires one scalar advantage per complete trajectory, shared by all of its
segments. Use the GRPO or RLOO estimator and disable advantage postprocessing.
Both estimators reject non-`None` `token_advantages` on training sequences,
including empty or constant arrays, before returning scalar estimates.
OPD, REINFORCE++ estimators, and tool-error rebalancing produce token advantages
and cannot be combined with GSPO.
Clip values use strict numeric validation: booleans and strings are rejected.
