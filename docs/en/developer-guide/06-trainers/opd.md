English | [简体中文](../../../zh/developer-guide/06-trainers/opd.md) | [Documentation index](../../index.md)

# On-policy distillation (OPD)

[Back to algorithm components](algorithm-components.md)

OPD trains a student on its own rollouts using teacher logprobs as a token-level
learning signal. Select `algorithm.type: opd` and
`algorithm.advantage.estimator.mode: opd`. It produces response-only
`token_advantages`, rather than a reward-group scalar.

## Advantage definition

For each segment, let `t` index target positions (`tokens[1:]`), `m_t` be its
target loss mask, `c = kl_penalty_coef`, and `gamma = kl_discount_factor`:

```text
d_t = -c * (student_logprob_t - teacher_logprob_t) * m_t
A_t = d_t + gamma * A_(t+1)          # terminal future advantage is zero
```

With `gamma: 0`, each response advantage is just its local `d_t`. For a positive
gamma, the discounted sum advances over every target position, including masked
context positions. Those positions contribute zero local signal but still count
toward the discount distance. The recurrence restarts at each segment; it does
not carry future advantages across segments. Only values where `m_t == 1` are
stored in `token_advantages`, in response order.

## Configuration

The complete [Weaver OPD recipe](../../../../recipe/on_policy_distill/weaver_opd.yaml)
declares the student under `models.actor` and the teacher under `models.teachers`.
Its estimator and loss can be configured as follows; this is a partial recipe:

```yaml
algorithm:
  type: opd
  advantage:
    estimator:
      mode: opd
      params:
        kl_penalty_coef: 1.0
        kl_discount_factor: 0.0
    postprocessing:
      mode: null
  loss:
    mode: importance_sampling
    reduction:
      mode: token_mean
  opd:
    teacher_logprob_source: inference_service
    teacher_service_timeout_s: 1200.0
  sampling_logprob_source: inference
  kl:
    mode: null
  optim:
    lr: 1.0e-4
```

The two estimator parameter defaults are `kl_penalty_coef: 1.0` and
`kl_discount_factor: 0.0`. Setting the coefficient to zero skips teacher scoring
and produces zero advantages. Teacher service settings belong to `algorithm.opd`;
`teacher_logprob_source` accepts `inference_service` or `training_service`.
Reference-policy KL under `algorithm.kl` is separate from the OPD teacher signal.

## Data and scope

OPD requires a nonempty `models.teachers` list. The current scoring path uses its
first teacher; it does not route datasets to different teachers. Unlike RLOO,
OPD does not require repeated comparison groups; the example uses
`data.train.rollout_repeat_n: 1`.

[`RemoteApiOpdTrainer`](../../../../northrl/trainer/remote_api_opd_trainer.py)
prepares an `OPDInputs` dictionary for
[`OPDEstimator`](../../../../northrl/algorithm/advantage/opd.py):

| Field | Required value for a nonzero coefficient |
| --- | --- |
| `sequences` | Original training sequence objects in the step's traversal order. |
| `student_logprobs` | Rollout student-score arrays captured before actor recomputation. |
| `teacher_logprobs` | Teacher-score arrays in the same sequence order. |

Each probability row is aligned to the full token sequence, including the
position-zero placeholder; the estimator aligns it to targets before computing
advantages. Missing fields, row-count mismatches, or changed sequence identities
are errors. A missing or misaligned probability row produces zero advantages
for that sequence.

The recipe uses `sampling_logprob_source: inference`. If actor recomputation is
enabled, OPD still uses the captured rollout student scores for its advantages;
the recomputed `old_log_probs` become the loss denominator. Holding the captured
arrays by reference is sufficient while recomputation replaces the field. A
producer that mutates those arrays in place must snapshot them during preparation.
Teacher calls and their failures remain trainer responsibilities, while the
estimator only computes advantages. Optional
[tool error advantage rebalancing](advantage-processing.md) runs afterward.
