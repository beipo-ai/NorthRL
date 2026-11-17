English | [简体中文](../../../zh/developer-guide/06-trainers/algorithm-components.md) | [Documentation index](../../index.md)

# Algorithm components

This page describes the shared component model and training data flow. Algorithm
formulas, runnable configurations, and supported combinations live in the
[algorithm guides](#algorithm-guides).

<a id="canonical-configuration"></a>

## Component responsibilities

The trainer constructs components from validated configuration and calls them in
order. The algorithm name does not necessarily identify a separate trainer:
estimation, postprocessing, the loss, and its reduction are separate choices.

| Configuration | Responsibility |
| --- | --- |
| `algorithm.type` | Select the trainer family and its input preparation. |
| `algorithm.advantage.estimator` | Compute advantages from trajectories and any additional scoring inputs. |
| `algorithm.advantage.postprocessing` | Optionally transform the estimated advantages. |
| `algorithm.loss` | Define the optimization objective. |
| `algorithm.loss.reduction` | Define how contributions are combined into the optimizer batch's scalar loss. |
| `algorithm.kl` | Configure reference-policy regularization. |
| `algorithm.optim` | Configure optimizer updates and request batching. |

Estimator, postprocessing, loss, reduction, and KL selections use a `mode` and a
`params` mapping. Parameters belong to the selected component. The configuration
schema validates the complete combination; independently selectable components
are not necessarily compatible. Omitting postprocessing or setting its mode to
`null` disables it. Reference KL is disabled with `algorithm.kl.mode: null`.

Model declarations belong under `models`. Backend settings, optimizer batching,
and training control are described in [Trainer Configuration](../11-configuration-reference/trainer-config.md).

<a id="step-data-flow-and-alignment"></a>

## Step data flow

Workers publish complete trajectories to the pool. The trainer claims a batch
and passes it to `train(trajectories)` as one training step:

```text
list[Trajectory] for the delivered step
  -> prepare trajectories and any required model scores
  -> estimate advantages over the complete prepared step
  -> optionally postprocess advantages
  -> apply configured sampling corrections and convert sequences to Datums
  -> form optimizer batches and apply the selected loss/reduction
  -> split service requests, accumulate gradients, and update the optimizer
```

The estimation scope is the prepared list received for this step. Splitting it
into optimizer batches or service requests does not rerun estimation on smaller
subsets. Group definitions and handling of unusable rollouts belong to the
selected estimator.

A complete trajectory (`Trajectory`) represents one rollout. Code sometimes calls
it a `parent` because it owns the saved training segments.

| Term | Meaning |
| --- | --- |
| Prompt group | Several rollouts sampled for the same prompt in one comparison group. |
| Complete trajectory | The result of one rollout, including its reward and status. |
| Segment | A saved token sequence within that trajectory. |
| Datum | A service training input produced from a retained sequence. |

For example, sampling two rollouts for one prompt produces two complete
trajectories in one group. If they contain two and one trainable segments,
respectively, conversion produces three Datums while the trajectory count stays
two. A multi-turn rollout, including its tool calls, still produces one complete
trajectory; a tool call does not necessarily start a new segment. Unsegmented
trajectories store tokens directly.

A loss over complete trajectories keeps all segments associated with their
original rollout. See [Remote API Trainers](remote-api-trainers.md) for conversion,
filtering, and batching, and [trajectory semantics](../04-trajectory-pool/trajectory-semantics.md)
for the data model.

<a id="data-passed-between-components"></a>

## Shared data interface

| Value | Meaning |
| --- | --- |
| `trajectories` | `list[Trajectory]` containing complete trajectories. |
| Estimator `inputs` | Additional data prepared by the trainer; `{}` when none is required. |
| Complete trajectory data | Reward, status, and rollout/group identity. |
| Sequence data | Tokens, masks, logprobs, and other fields for a flat trajectory or one segment. |
| Scalar advantage | One `advantage` on the complete trajectory, shared by its training sequences. |
| Token advantage | Response-only `token_advantages` on each sequence. |
| Component return value | A trajectory list with computed fields on the original objects; the list may be reordered. |
| `metrics` | A shared dictionary updated in place. |

An estimator determines the granularity of its advantage output. Optional
postprocessing consumes that output; without postprocessing, the estimator's
result is preserved. During conversion, explicit token advantages are aligned
to target positions; otherwise the trajectory's scalar is broadcast to the response
tokens. The selected loss determines which representation it accepts.

The public component calls are:

```python
trajectories = estimator.estimate(trajectories, inputs, metrics)
trajectories = postprocessor.process(trajectories, metrics)  # when enabled
```

These components compute over the supplied data. The trainer prepares additional
inputs and owns external service calls, request splitting, and optimizer updates.

<a id="loss-and-reduction"></a>

## Loss and reduction

A loss defines each contribution to the optimization objective. A reduction
specifies their weighting and aggregation into one scalar for an optimizer
batch. Tokens, Datums, complete trajectories, and prompt groups are distinct
aggregation units; splitting a trajectory into segments must not implicitly
change the unit chosen by the objective.

Native losses run through the service loss interface. Custom losses run through
a callback. The adapter translates the selected configuration into the matching
protocol and applies normalization where that protocol requires it. Splitting
requests within an optimizer batch does not define a new averaging population.

### Custom loss callbacks

Use `loss.mode: custom` to select a callback. Its `loss.params` block is:

```yaml
name: my_package.losses:build_loss
params: {}
base_loss_fn: importance_sampling
clip: {}
```

`name` is a registered alias or an importable `module:factory`; the inner
`params` mapping is passed to that factory. `base_loss_fn` and outer `clip` are
adapter fields, not callback arguments. A `module_path` does not load a Python
file relative to the recipe. Custom losses require zero entropy and disabled
submit-ahead. Use `loss.mode: native` with `params.name` to select a service-native
loss by name instead.

<a id="where-to-read-the-code"></a>

## Code and extension points

| File | Responsibility |
| --- | --- |
| [`trainer/remote_api_trainer.py`](../../../../northrl/trainer/remote_api_trainer.py) | Prepare inputs, invoke components, convert data, and orchestrate training. |
| [`algorithm/advantage/types.py`](../../../../northrl/algorithm/advantage/types.py) | Estimator and postprocessor interfaces. |
| [`algorithm/advantage/factory.py`](../../../../northrl/algorithm/advantage/factory.py) | Construct components from validated configuration. |
| [`weaver/loss_adapter.py`](../../../../northrl/weaver/loss_adapter.py) | Adapt loss configuration and data to the service protocol. |
| [`algorithm/reductions.py`](../../../../northrl/algorithm/reductions.py) | Compute shared normalization denominators. |

<a id="adding-an-algorithm-and-python-api-changes"></a>

Implement a new estimator or postprocessor through the corresponding interface,
add its factory branch and configuration validation, and document its behavior
in its own guide. Extend trainer input preparation when additional external data
is needed. Test output alignment, supported combinations, and invariance under
request splitting. Add the guide to the directory below.

<a id="supported-configurations"></a>
<a id="algorithm-guides"></a>

## Algorithm guides

- [GRPO](grpo.md)
- [RLOO](rloo.md)
- [REINFORCE++ and REINFORCE++-baseline](reinforce-plus-plus.md)
- [OPD](opd.md)
- [Dr.GRPO](dr-grpo.md)
- [CISPO](cispo.md)
- [GSPO](gspo.md)

<a id="future-integrations"></a>

## Related guides

- [Advantage postprocessing](advantage-processing.md)
- [Rollout/training mismatch correction](mis.md)
- [Remote API Trainers](remote-api-trainers.md)
- [Trainer Configuration](../11-configuration-reference/trainer-config.md)
