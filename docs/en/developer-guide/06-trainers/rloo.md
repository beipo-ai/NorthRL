English | [简体中文](../../../zh/developer-guide/06-trainers/rloo.md) | [Documentation index](../../index.md)

# RLOO

[Back to algorithm components](algorithm-components.md)

RLOO (REINFORCE Leave-One-Out) estimates a scalar advantage by comparing each
rollout's final reward with the other rollouts of the same prompt. Select
`algorithm.advantage.estimator.mode: rloo` within the `algorithm.type: grpo`
reward-training flow.

## Advantage definition

For a comparison group of `N > 1` complete trajectories with final rewards `R_i`:

```text
A_i = R_i - mean(R_j for j != i)
    = (R_i - group_mean) * N / (N - 1)
```

Groups are identified by `(model_tag, group_id)`. Each successful complete
trajectory contributes one reward, including complete trajectories with no
training sequences. All segments of a complete trajectory share its scalar
advantage; adding segments does not change the baseline. RLOO does not divide by
a standard deviation or whiten the complete step. Its numerical singleton
fallback is zero, while recipe validation requires at least two rollouts per prompt.

## Configuration

Merge this algorithm block into a reward-training recipe, such as the
[math recipes](../../recipe/math/README.md), and set
`data.train.rollout_repeat_n >= 2`:

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: rloo
      params: {}
    postprocessing:
      mode: null
  loss:
    mode: importance_sampling
    reduction:
      mode: token_mean
  optim:
    lr: 1.0e-6
```

RLOO has no estimator parameters. In particular, the GRPO
`norm_adv_by_std` parameter does not apply. The example selects native importance
sampling loss; the estimator and loss remain separate components.

## Data and scope

The trainer repairs or filters failed groups before estimation.
[`RlooEstimator`](../../../../northrl/algorithm/advantage/rloo.py) then validates
complete comparison groups: duplicate rollout IDs, invalid rewards, and missing
declared `expected_rollouts` are rejected; skipped groups remain skipped.
Estimation receives the entire delivered step, but each comparison group has its
own baseline. Optimizer batch sizes and request chunks do not change it.

The estimator writes one float32 scalar `advantage` per eligible complete
trajectory and broadcasts it to that trajectory's training segments. It requires
no additional estimator inputs (`inputs: {}`). Optional
[tool error advantage rebalancing](advantage-processing.md) runs afterward and
can produce token advantages from the newly computed scalars.
