English | [简体中文](../../../zh/developer-guide/06-trainers/reinforce-plus-plus.md) | [Documentation index](../../index.md)

# REINFORCE++ and REINFORCE++-baseline

Select either estimator with `algorithm.advantage.estimator.mode` and set
`algorithm.type: grpo`. The factory constructs `ReinforcePlusPlusEstimator` from
`northrl.algorithm.advantage.reinforce_plus_plus`. The trainer directly calls
`estimate(trajectories, {}, metrics)`, which returns the
original `list[Trajectory]` with response-only advantages on each sequence.
Both use the native `importance_sampling` or `ppo_clip` loss.
In this configuration, omitting the estimator selects the default `grpo`
estimator. See [Algorithm Components](algorithm-components.md).

## Configuration

Copy `recipe/math/weaver.yaml` to `recipe/math/weaver_reinforce_plus_plus.yaml`.
Keep its model, data paths, rollout, backend, and runtime settings; replace its
`algorithm` block with the following. These are overrides for that complete
recipe, not a standalone training configuration.

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: reinforce_plus_plus
      params:
        gamma: 1.0
    postprocessing:
      mode: null
  loss:
    mode: ppo_clip
    params:
      clip:
        ratio: 0.2
    reduction:
      mode: token_mean
  sampling_logprob_source: recomputed
  entropy_coeff: 0.0
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
  kl:
    mode: null
```

The math recipe already sets `data.train.rollout_repeat_n: 8`. Vanilla
REINFORCE++ also permits `1`. To use the baseline variant, change only
`advantage.estimator.mode: reinforce_plus_plus_baseline`, keep its `params.gamma: 1.0`, and use
`data.train.rollout_repeat_n >= 2`. After configuring the same services and
environment variables required by the original recipe, launch with:

```bash
export TRAIN_CONFIG="$(pwd)/recipe/math/weaver_reinforce_plus_plus.yaml"
mkdir -p "$EXPERIMENT_PATH"
python -m northrl.main \
  --config-path="$(dirname "$TRAIN_CONFIG")" \
  --config-name="$(basename "${TRAIN_CONFIG%.*}")" \
  'runtime.logger.backends=[console]'
```

Both estimators require `loss.reduction.mode: token_mean` and native
`loss.mode: importance_sampling` or `ppo_clip`; custom callbacks are unsupported.
They are unavailable with `algorithm.type: opd` or `sft`. Only vanilla
REINFORCE++ accepts a non-default `gamma`; it must be finite and between 0 and 1.
Changing the advantage estimator does not change PPO clipping, entropy, or MIS
configuration.

Configure estimator `mode` and `params.gamma` under `algorithm.advantage.estimator`.
REINFORCE++ uses global token whitening, so its estimator parameters do not include
GRPO's `norm_adv_by_std` option.

## Optional advantage postprocessing

Both variants support [tool error advantage rebalancing](advantage-processing.md).
To enable it, replace only `algorithm.advantage.postprocessing` in the configuration above:

```yaml
postprocessing:
  mode: tool_error_rebalance
  params:
    negative_multiplier: 2.0
    positive_scale_max: 2.0
    negative_scale_min: 0.5
```

Rebalancing consumes the whitened response-only token advantages across the
complete step, before MIS or batching; it does not whiten them again. It saves
those base advantages for replay. Before each fresh estimate, the trainer clears
saved base and shaped values, including when postprocessing has been disabled.

## Returns and the group baseline

For a complete trajectory, order all effective response tokens across its
segments as `j = 1, ..., T`. Prompt, tool-observation, masked, and padding tokens
are excluded. Let `R` be its final outcome reward. The token reward is:

```text
r_j = R if j == T else 0
```

Vanilla REINFORCE++ computes reward-to-go:

```text
G_j = r_j + gamma * G_(j+1), with G_(T+1) = 0
```

The terminal outcome is added once per complete trajectory. Return propagation
crosses segment boundaries; masked observations neither reset the running return
nor consume a discount step. With `gamma: 1.0`, every
effective token of a complete trajectory starts with its outcome reward.

The baseline variant uses each complete trajectory's outcome reward and subtracts
the inclusive mean within its `(model_tag, group_id)` comparison group:

```text
B_g = mean_i_in_group(R_i)
G_ij = R_i - B_g
```

It does not use a leave-one-out baseline, divide by group reward standard
deviation, or discount returns. Each complete trajectory contributes one reward regardless of
its number of segments. Successful complete trajectories with no effective tokens still
contribute to the group baseline, but contribute no tokens to whitening or loss.
After postprocessing and MIS computation, the shared converter checks the final
`loss_mask`. A sequence with an all-zero mask produces neither a training Datum
nor a source entry. This training flow retains sequences with active tokens and
zero advantages. This does not change the completed baseline or whitening; see
the [shared data flow](algorithm-components.md#step-data-flow-and-alignment)
for conversion settings.

Both variants validate complete groups and repair failed groups before computing
returns. The baseline configuration requires at least two expected members; the
low-level estimator safely centers a singleton to zero. Raw rollout reward metrics
describe the original outcomes.

## Global token whitening and batching

Both variants whiten `G` once over all effective response tokens in the repaired,
delivered training step. For `N` effective tokens:

```text
mu = sum(G) / N
variance = sum((G - mu)^2) / (N - 1)
A = (G - mu) / sqrt(variance + 1e-8)
```

This uses unbiased sample variance and token weighting: longer responses
contribute more observations to these statistics. If `N <= 1`, advantages are
zero. Masked positions have zero training advantage. Each segment carries its
own response-token `token_advantages` vector; the complete trajectory's scalar `advantage`
is only the mean of its effective token advantages for diagnostics, or zero when
empty. Compact token buffers produce response-only `array('d')` advantages;
list inputs retain list outputs. Token and mask processing uses typed-buffer
operations where possible, and whitening retains float64 precision. Discounted
returns propagate across segments without dividing by powers of gamma;
long responses and very small gamma remain stable. The estimator reads the
original token and mask buffers, counts effective response tokens, and allocates
working buffers for mask validation, returns, and whitening.

Whitening uses the trajectory rewards and happens before
advantage postprocessing, MIS/rejection, Datum
shuffle, optimizer splitting, or tail padding. Rejected tokens therefore do not
cause the remaining advantages to be re-whitened. Explicit positive
`algorithm.optim.batch_size` values select fixed-size optimizer batches;
shuffling, splitting, and tail padding happen after whitening when the delivered
batch exceeds that size. Token-mean loss division is then performed separately
for each optimizer batch. Changing
`mini_batch_size` changes request size without changing whitening statistics or
the number of optimizer updates. The example uses `batch_size: null` to keep
one delivered batch in one update; see [batching details](../11-configuration-reference/trainer-config.md#optimizer-batches-and-forwardbackward-requests).

## KL modes

These estimators do not compute KL or read actor/reference logprobs.
`kl.mode: in_loss` adds an independent loss term and requires
`models.reference`; `kl.mode: in_reward` is not implemented and is rejected during
trainer initialization.

<a id="input-production-and-consumption"></a>

## Data flow from trajectories to advantages

Both REINFORCE++ variants read their data directly from `trajectories` and use
an empty `inputs` dictionary in the shared estimator interface:

```text
RemoteApiGrpoTrainer._prepare_advantage_inputs()
  returns {}
    ↓
RemoteApiTrainer.train()
  optionally recomputes actor logprobs for training; clears postprocessing caches
    ↓
ReinforcePlusPlusEstimator.estimate(trajectories, {}, metrics)
  reads gamma and baseline from the estimator instance
    ↓
compute_reinforce_plus_plus_advantages(trajectories, baseline=..., gamma=...)
  computes returns and step-wide whitening; writes token_advantages on each sequence
    ↓
optional tool-error postprocessing → compute MIS weights and rejection masks
    ↓
convert_trajectories_to_datums()
  apply MIS; check the final loss_mask; skip all-zero masks; create Datums and source entries
    ↓
split optimizer batches
```

The [trainer](../../../../northrl/trainer/remote_api_grpo_trainer.py) prepares no extra
inputs for these estimators. Actor recomputation and reference scoring, when
enabled, serve the training loss and do not change the estimator's inputs.
The [estimator](../../../../northrl/algorithm/advantage/reinforce_plus_plus.py) passes
the following data to `compute_reinforce_plus_plus_advantages()`:

| Data | Source | Use |
| --- | --- | --- |
| Outcome reward | Each complete trajectory's `reward` | Terminal return or group-centered baseline |
| Group identity | `model_tag` and `group_id` in the complete trajectory's `extra_fields` | Group baseline and validation |
| Tokens and mask | Original trajectory or its segment dictionaries | Select effective response positions |
| `gamma`, `baseline` | Estimator instance, built from estimator configuration | Discounting and variant selection |

The formula traverses the original sequences of successful, non-skipped complete trajectories,
without a separate sequence or probability payload. It writes
`token_advantages` on the original sequences and returns `None`. Then `estimate()`
records metrics and returns the original `list[Trajectory]` for postprocessing
and training-data conversion.
