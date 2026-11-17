English | [简体中文](../../../zh/developer-guide/06-trainers/advantage-processing.md) | [Documentation index](../../index.md)

# Tool error advantage rebalancing

Tool error advantage rebalancing runs once after estimation, before prepared dumps
and MIS, across **all complete trajectories in the current step**. Pools span
groups, model tags and segments;
optimizer batches, mini batches and request chunks do not define the scope.
Failed and group-skipped complete trajectories are excluded.

## Configuration

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: grpo
      params:
        norm_adv_by_std: true
    postprocessing:
      mode: tool_error_rebalance
      params:
        negative_multiplier: 2.0
        positive_scale_max: 2.0
        negative_scale_min: 0.5
```

This is a partial algorithm block; retain the recipe's loss, optimizer and rollout
settings. Missing postprocessing or `mode: null` is an exact no-op; disabled
postprocessing requires empty `params`.
Set `mode: tool_error_rebalance` to enable rebalancing.

| Parameter | Meaning | Range |
| --- | --- | --- |
| negative_multiplier | Penalty multiplier κ for negative error tokens | Finite, > 1 |
| positive_scale_max | Maximum multiplier for positive clean tokens | Finite, ≥ 1 |
| negative_scale_min | Minimum retained multiplier for negative clean tokens | Finite, (0, 1] |

## Data contract

The direct interface is
`ToolErrorRebalancePostprocessor.process(trajectories, metrics) -> list[Trajectory]`.
It reads existing response-only `token_advantages`, or broadcasts the trajectory's scalar `advantage` when no token estimate exists. The same trajectory
list is returned, with response-only `token_advantages` stored on its original
sequences. Token buffers are reused. An ordinary trajectory stores annotations in
its `extra_fields`; segmented trajectories store them in each segment dictionary:

```python
{
    "tokens": [10, 11, 12, 13],
    "loss_mask": [0, 1, 1, 1],
    "old_log_probs": [0.0, -0.2, -0.3, -0.4],
    "turn_annotations": [{
        "version": 1,
        "turn_id": "turn-1",
        "response_span": [1, 4],
        "source": "northgate",
        "finish_reason": "tool_calls",
        "truncated": False,
        "tools_declared": True,
        "diagnostics": {
            "checked": True, "malformed": True, "detected": 1,
            "invalid": 1, "reasons": ["invalid_arguments_json"],
        },
    }],
}
```

`response_span` is a half-open range of local token positions. It must contain only
responses, exclude token zero, and not overlap other turns. The error state covers **every
response token in the diagnosed turn**, rather than a parser-local character
range. Rollout transports diagnostics and computes spans; this module consumes them.

- `error`: supported, reliable tool errors such as invalid argument JSON, missing
  names or duplicate parameters. Undeclared tools require tools_declared=true;
  incomplete tool calls require a normal finish reason.
- `clean`: checked=true, malformed=false, empty reasons and invalid=0 or missing.
- `unknown`: missing diagnostics, unsupported source/version, truncation/external
  stops, unsupported reasons or insufficient evidence. These tokens remain unchanged
  and never donate or receive advantage compensation.

Token classifications are computed temporarily from the annotations. The trajectory's
scalar remains diagnostic. Training reads response-only
`token_advantages` in original `loss_mask[1:] == 1` order. `base_token_advantages`
saves a second advantage array for idempotent replay. Before each fresh estimation,
`RemoteApiTrainer.train()` calls `clear_postprocessing_cache(trajectories)` from
`algorithm/advantage/tool_error_rebalance.py`, clearing saved base and its shaped
output even when postprocessing is now disabled. Token estimates without a saved base are retained.
Standalone callers call the same function before re-estimating reused trajectories;
repeating `process()` on the same estimate reuses its saved base without clearing. Tokens are not copied. Compact input uses
`array("d")` for advantage storage. Prepared dumps include base, output and annotations.
The two stored arrays remain independent, including when every diagnostic is unknown.
Processing validates spans and computes all outputs before committing any sequence;
response-only state buffers and per-sequence reductions avoid full-token coverage
arrays and retaining selected token copies across the step. Temporary buffers are
released as final output buffers are staged.

## Step-wide computation

Classify signs using each original token advantage. Pe/Pc are the sums of positive
error/clean advantages; Ne/Nc are the sums of absolute negative error/clean advantages,
**not token counts**
(e = error, c = clean).

```text
alpha = min(1 + Pe/Pc, positive_scale_max)          # Pc=0: alpha=1
beta  = max(1 - (κ-1)*Ne/Nc, negative_scale_min)   # Nc=0: beta=1
positive error → 0; positive clean → alpha*A
negative error → κ*A; negative clean → beta*A
zero or unknown → unchanged
```

For multi-turn trajectories, each turn's `response_span` determines the states of
its response tokens. If several turns contain errors, **sum their token advantages**
into the appropriate error pools. The error count is not an extra multiplier:
each positive error token becomes zero and each negative error token is multiplied
by κ once. All clean tokens of the same sign across the complete step share one
alpha or beta, including tokens from other trajectories and segments.

Clamps or empty pools can prevent the advantage sums from being conserved.
W&B uses the prefix `01_algorithm/extra/advantage_postprocess/` for these metrics:

| Metric suffix | Meaning |
| --- | --- |
| `zero_advantage_tokens` | Number of response tokens whose advantage is zero before processing |
| `zero_advantage_error_tokens` | Number of error response tokens whose advantage is zero before processing |
| `positive_error_advantage_sum`, `positive_clean_advantage_sum` | Pe and Pc before processing |
| `negative_error_advantage_abs_sum`, `negative_clean_advantage_abs_sum` | Ne and Nc before processing; both nonnegative |
| `positive_advantage_sum_before`, `positive_advantage_sum_after` | Sum of positive response advantages before/after processing |
| `negative_advantage_abs_sum_before`, `negative_advantage_abs_sum_after` | Sum of absolute negative response advantages before/after processing |
| `positive_scale`, `negative_scale` | Applied alpha and beta; dimensionless multipliers |
| `positive_scale_clamped` | Dimensionless 0/1 flag: 1 only when the unclamped alpha exceeds `positive_scale_max`; equality gives 0 |
| `positive_no_clean_tokens` | Dimensionless 0/1 flag: 1 when Pe > 0 and Pc = 0; the positive clean pool is empty and cannot absorb the cleared positive error advantages |

`positive_no_clean_tokens` is not a token count and does not mean that all clean
tokens are absent; zero or negative clean tokens may still exist. Use
`positive_clean_tokens` for the positive clean token count. The corresponding
negative flags report the unclamped beta below its lower bound and an empty negative clean pool
when there are negative error advantages. Before/after sums include unchanged
unknown tokens. These sums describe raw response advantages, not gradients after
loss reduction, MIS or optimizer updates.

`build_postprocessor()` constructs `ToolErrorRebalancePostprocessor` from validated
configuration. `RemoteApiTrainer.train()` calls the estimator and then the
postprocessor on the complete step. Rebalancing uses the estimator's final output,
including any normalization or whitening performed during estimation. See
[algorithm guides](algorithm-components.md#algorithm-guides) for
compatible combinations. SFT has no advantage estimator and cannot enable postprocessing.
