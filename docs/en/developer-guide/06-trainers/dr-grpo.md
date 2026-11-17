English | [简体中文](../../../zh/developer-guide/06-trainers/dr-grpo.md) | [Documentation index](../../index.md)

# Dr.GRPO fixed response budget

Dr.GRPO uses the existing `algorithm.type: grpo` training flow and `mode: grpo`
estimator with `params.norm_adv_by_std: false`, together with `ppo_clip` and
`loss.reduction.mode: fixed_response_budget`. It reuses `GRPOEstimator`; there
is no separate `DrGRPOEstimator` or `dr_grpo` estimator mode.

## Objective and denominator

Each complete trajectory receives `A = reward - group_mean` within its
`(model_tag, group_id)` group. It contributes once to that mean regardless of
its segment count. A singleton or constant-reward group receives zero advantage.

The clipped token-loss sum is divided by `D = N * L`:

- `N` counts every trajectory in the full rollout step whose `status` is
  `"success"` and whose `group_skipped` flag is false or absent. Counting happens
  after failure repair and group validation, before splitting training requests.
- `L` is the configured fixed response length budget,
  `algorithm.loss.reduction.params.max_response_tokens`.

For a step containing two successful, non-skipped trajectories in the same group,
with rewards `1` and `3`, the advantages are `-1` and `1`. With `L = 8192`, the
denominator is `16384`, regardless of their actual response lengths.

Successful, non-skipped trajectories with empty responses or zero advantage still
count toward `N`, as do those whose tokens are all rejected by MIS. Failed or
`group_skipped` trajectories do not count. More segments in one trajectory do not
increase `N`. If no training tokens remain, the trainer skips the optimizer update
under its existing rules.

The grouped rollout flow decides these flags before the denominator is counted.
After all rollouts for a prompt arrive, the pool checks for successful status,
a finite reward and no existing skip flag. If fewer than half are valid, it marks
the whole group skipped. Otherwise, the trainer fills invalid slots by copying
successful trajectories from that group before group validation and advantage
calculation. This copies existing data; it does not launch new rollouts.
For example, three valid rollouts and one failed rollout become four counted
trajectories after replacement. One valid rollout and three failed rollouts
cause the group to be skipped, contributing zero to `N`.

## Data flow

1. The trainer receives the full step's trajectory list and converts its segments
   to training datums. A datum is one segment's token, logprob and advantage data
   prepared for the training backend.
2. `attach_fixed_response_budget_denominator()` in `northrl/algorithm/reductions.py`
   counts `N` from that trajectory list,
   computes `N * L`, and stores the same integer as `_dr_grpo_denominator` on every
   datum. It adds only this metadata field; it does not copy token or advantage
   buffers.
3. `get_loss_denominators()` in the same module reads and validates the saved denominator.
   `northrl/weaver/loss_adapter.py` divides the advantage contributions by it once
   so that the backend's sum loss implements the required normalization.
4. The trainer may split datums into several forward/backward requests, then
   execute one optimizer update. Each request keeps the original denominator.
   Its size, retained token count and number of datums do not recalculate `N`.

## Configuration

In the existing [`recipe/math/weaver.yaml`](../../../../recipe/math/weaver.yaml),
replace the entire `algorithm` section with the following. Data, model, service
setup and startup instructions follow the [math recipe guide](../../recipe/math/README.md).

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: grpo
      params:
        norm_adv_by_std: false
    postprocessing:
      mode: null
  loss:
    mode: ppo_clip
    params:
      clip:
        ratio: 0.2
    reduction:
      mode: fixed_response_budget
      params:
        max_response_tokens: 8192
  sampling_logprob_source: inference
  optim:
    lr: 1.0e-4
    beta1: 0.9
    beta2: 0.95
    eps: 1.0e-8
    weight_decay: 0.01
    grad_clip_norm: 1.0
    batch_size: null
    mini_batch_size: 32
  entropy_coeff: 0.0
  kl:
    mode: null
```

`max_response_tokens` must be a positive integer and should match the configured
generation limit. In this math configuration, `rollout.sampling.max_tokens` is
omitted, so the limit is derived as `data.max_sequence_length -
data.max_prompt_length = 12288 - 4096 = 8192`. Update the reduction budget when
changing that limit; it is not inferred from generated responses. For multi-turn
rollouts, use the total response-token budget for the complete trajectory, not
the limit for one generation call. This budget scales the loss; it does not
truncate generation.

`norm_adv_by_std: false` subtracts only the group's mean reward.
`postprocessing.mode: null` disables advantage postprocessing.
`sampling_logprob_source: inference` uses the logprobs returned during rollout.
The current implementation requires native `ppo_clip` with a finite positive
symmetric clip ratio, KL and custom loss disabled, and `entropy_coeff: 0`.
Asymmetric, dual and truncation clipping are rejected.

`optim.batch_size` must be omitted or `null` so that the full step contributes to
one optimizer update. `mini_batch_size: 32` only limits each forward/backward
request to 32 datums within that update. It does not change the denominator.

The supported synchronization modes are `sync`, `batch-async`, and `fully-async`.
`no-sync` requires an explicit optimizer batch size and is incompatible.
