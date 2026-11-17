English | [简体中文](../../../zh/developer-guide/06-trainers/score-centering.md) | [Documentation index](../../index.md)

# Score centering with Weaver (experimental)

Score centering (SC) requires matching SDK, server, and Megatron Bridge trainer support. It currently supports single-turn training with the built-in simple rollout worker. It is not an E2B Agent training configuration.

<a id="configuration"></a>

## Configuration

Use a normal [GRPO recipe](../../recipe/math/README.md) with the following overrides. Replace the whole `algorithm.loss` block so parameters for the old loss are not retained. Existing optimizer, batching, data, reward, and evaluation settings remain part of your complete recipe; this is not a standalone configuration.

```yaml
algorithm:
  type: grpo
  loss:
    mode: score_centering
    params:
      topk: 128
  sampling_logprob_source: inference
  entropy_coeff: 0
  kl:
    mode: null
  mis:
    enabled: false
    get_mismatch_metrics: false
  router_replay:
    enabled: false
rollout:
  type: simple
  sampling:
    temperature: 1
    top_p: 1
    top_k: -1
    min_p: 0
    min_tokens: 0
    repetition_penalty: 1
training:
  weight_sync:
    mode: sync
backend:
  type: weaver
```

Weight synchronization must use `mode: sync`; configuration validation enforces this. Choose data and rewards appropriate to `SimpleRolloutWorker`: its built-in reward compares the text inside `<answer>...</answer>` with `ground_truth`.

<a id="sampler-data"></a>

## Sampler data and training alignment

The service holder requests generated-token top-k probabilities on training samples. Greedy evaluation and warmup do not require SC statistics.

The remote client and simple worker preserve `sampler_distribution_data` through the canonical `northrl_train` envelope. Truncation slices all response-aligned fields together. Datum conversion shifts them onto target positions, pads prompt rows with `mask=0`, and preserves the final response token.

Sampler probabilities remain independent of `old_log_probs`. The existing advantage normalization folds token/sample mean denominators into A before the trainer's sum reduction, including repeated Datums from minibatch padding. See [complete-rollout and batching semantics](sync-rollout-semantics.md#optimizer-update-boundary) for update boundaries.

<a id="support-and-validation"></a>

## Supported combinations and validation limits

Initial support is single-turn, built-in simple worker, and pure SC. MIS, custom loss, router replay, KL, entropy regularization, and filtered or constrained sampling are unsupported. Configuration validation rejects these combinations. A missing server or SDK capability fails explicitly.

Existing validation is limited to CPU protocol and gradient tests. The original implementation notes report no live GPU optimizer or training-quality experiment. Treat this feature as experimental until those checks are completed; the configuration example is not evidence of end-to-end training quality.
