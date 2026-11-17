English | [简体中文](../../../zh/developer-guide/06-trainers/cispo.md) | [Documentation index](../../index.md)

# CISPO

CISPO uses detached clipped importance weights to scale each response token's
log-probability gradient:

```text
r = exp(clamp(logp - old_logp, -20, 20))
w = stop_gradient(clamp(r, 1 - ratio_low, 1 + ratio_high))
L = -sum_active_tokens(w * advantage * logp) / active_token_count
```

Ratios are clamped in log space to `[-20, 20]`, then
clipped to `[1 - ratio_low, 1 + ratio_high]`.
The gradient remains active outside the clipping interval.

```yaml
algorithm:
  type: grpo
  advantage:
    estimator: {mode: grpo, params: {norm_adv_by_std: true}}
  loss:
    mode: cispo
    params:
      clip:
        ratio: 0.2
    reduction:
      mode: token_mean
  entropy_coeff: 0
  kl:
    mode: null
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
```

`ratio_low` and `ratio_high` override `ratio` independently. Bounds must be
explicit finite nonnegative numbers; booleans and strings are rejected.
`ratio_c` and `truncation_c` are not supported.

CISPO is a built-in loss mode. `LossAdapter.custom` builds Weaver's custom-loss
callback parameters. The trainer uses the callback and surrogate backward path.
The converter broadcasts each complete
trajectory's scalar advantage or aligns response-only token advantages to
`target_tokens=tokens[1:]`. The callback receives the resulting
target-aligned arrays and owns the token mean. The host does not normalize
advantages a second time.

Each optimizer batch is evaluated by one callback, even when `mini_batch_size`
splits remote forward/backward requests. `batch_size` may split a step into
multiple optimizer updates. This differs from whole-step advantage estimation,
which occurs once before optimizer batching. Loss metrics across updates use
response-token weights.

The example above uses the GRPO advantage estimator. CISPO requires
`algorithm.type: grpo`, the Weaver backend, zero entropy, reference KL disabled,
MIS correction disabled, and `optim.submit_ahead: 0`.
Optional `advantage.postprocessing.mode: tool_error_rebalance` runs over the
complete step before conversion. Active tokens with zero advantages remain in
the loss denominator; sequences with no active target positions produce no
Datum. REINFORCE++ estimators retain their native-loss restriction and cannot
be combined with CISPO. Existing router-replay and mismatch-diagnostic
contracts remain in force.
See [the recipe](../../../../recipe/math/weaver_cispo.yaml) and
[algorithm components](algorithm-components.md).

The step remains explicit in `RemoteApiTrainer.train()`: prepare probabilities,
call `self._advantage_estimator.estimate(...)`, optionally call
`self._advantage_postprocessor.process(...)`, convert the returned
`list[Trajectory]` to Datums, then split optimizer batches. The estimator and
postprocessor write advantages onto the existing trajectory/sequence objects. CISPO consumes
these values after target-position alignment.
