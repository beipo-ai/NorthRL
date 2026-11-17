# Logprobs Shape Conventions (Weaver path)

**Read this before touching anything that produces, consumes, or routes
``logprobs`` / ``ref_logprobs`` / ``old_log_probs`` between rollout, the
trainer, ``convert_trajectories_to_datums``, and the Weaver loss
computers.** Position drift bugs here are silent — wrong gradients, no
exception — and they have already cost time once.

## TL;DR

Terminology:

- **Trajectory-level fields** are token-aligned: any trajectory logprob field
  (``old_log_probs``, ``ref_logprobs``, rollout logprobs) has
  ``len(field) == len(tokens)``. Position 0 is a no-context placeholder.
- **Target-level / Datum-level fields** are aligned to ``target_tokens``:
  ``target_tokens = tokens[1:]``. These are the model-predicted positions
  where per-token logprobs/losses can be computed. The actual objective still
  uses ``loss_mask`` to decide which target positions contribute.
- Weaver consumes target-level fields in ``loss_fn_inputs``; it should not see
  trajectory-length logprob tensors.

| Source | Length | Position 0 | Comment |
| --- | --- | --- | --- |
| ``sampling_client.compute_logprobs(prompt=ModelInput(tokens))`` | ``len(tokens)`` | **``None``** | flat ``list[float \| None]``; first slot has no context |
| ``training_client.forward(datums, "forward_logprob")`` (nex-weaver ≥1.7.0) | ``len(target_tokens)`` | real logprob | already target-aligned; no leading sentinel |
| ``training_client.forward(datums, "cross_entropy")`` | ``len(target_tokens)`` | real logprob | same shape; ``loss_fn_outputs[i]`` also has ``elementwise_loss`` |
| Rollout workers (``simple_rollout_worker.py``, ``single_turn_rollout_worker.py``) | ``len(tokens)`` | **``0.0``** | written as ``[0.0]*len(prompt) + response_logprobs`` |
| ``_recompute_logprobs`` (in ``remote_api_trainer.py``) | ``len(tokens)`` | **``0.0``** | calls ``compute_target_logprobs_batch`` → ``training_client.forward(..., "forward_logprob")`` and prepends the no-context slot before storing |

So: the only two on-disk shapes for any logprob field are
**full-length (``len(tokens)``, position 0 is ``None`` or ``0.0``)** or
**target-aligned (``len(tokens) - 1``, every position carries info)**.

## Wire format from ``training_client.forward``

The returned `logprobs` field is a **dict**, not a flat list. Always go
through ``.get("data")`` — the existing converter already handles both
shapes, do not strip this defensively elsewhere:

```python
result = training_client.forward(datums, loss_fn="forward_logprob", wait=True)
outputs = result["result"]["loss_fn_outputs"]
for o in outputs:
    lp = o["logprobs"]
    if isinstance(lp, dict):       # {"data": [...], "dtype": "float32", "shape": [N]}
        lp = lp["data"]
    # lp is now a list[float] of length len(target_tokens)
```

## Loss-side contract (weaver-trainer)

**Every length-N tensor that a Datum hands to weaver-trainer's loss
computer must be aligned to ``target_tokens``** (length = N-1 for input of
length N). This includes ``logprobs``, ``sampling_logprobs``,
``ref_logprobs``, ``advantages``, ``loss_mask``, ``weights``. The model's
own forward output (``logprobs`` tensor in compute_loss) is also at target
positions, so KL / IS ratio compare element-wise without any internal
shift on the trainer side.

The NorthRL converter enforces this contract before sending datums to the
training service. See ``northrl/utils/finetune_service_utils.py`` and the
alignment cases in ``tests/unittests/utils/test_finetune_service_utils.py``:
``len(ref_logprobs) == len(target_tokens)``, with no additional shift.

## Converter contract (``northrl/utils/finetune_service_utils.py``)

``convert_trajectories_to_datums`` is the boundary between the two shapes. It
expects trajectory-level logprob fields (``old_log_probs``, ``ref_logprobs``)
to be full-length and emits target-level datums for Weaver:

```python
target = align_logprobs_to_target(tokens=tokens, logprobs=arr, name=name)
```

``align_logprobs_to_target`` verifies that position 0 is ``None`` or
``|float| < 1e-9`` before dropping it. Position 0 is the no-context slot.

This guard catches the silent-shift bug class: a target-aligned array of
length N (e.g. someone padded to ``max_len = len(tokens)`` upstream)
would otherwise be trimmed and silently mis-aligned by 1.

## Rules for new code

1. **Producing logprobs:** pick **one** of the two on-disk shapes and
   document it. If full-length, position 0 must be ``None`` or ``0.0``. Do
   not invent a third shape (e.g. "length N with leading real logprob"
   means no one knows what position 0 means).

2. **Consuming logprobs going to a Datum:** route through
   ``convert_trajectories_to_datums``; do not roll your own ``[1:]``.

3. **Calling ``training_client.forward`` for ref/teacher logprobs:** build
   datums with ``model_input = ModelInput.from_ints(tokens=tokens[:-1])``
   and ``loss_fn_inputs = {"target_tokens": tokens[1:]}``. The returned
   ``logprobs.data`` is target-aligned; prepend a ``0.0`` placeholder before
   storing it on the trajectory as ``ref_logprobs`` so trajectory-level
   logprob fields stay full-length.

4. **Recomputing ``old_log_probs``:** must use the trainer-forward path,
   i.e. ``WeaverServiceHolder.compute_target_logprobs_batch`` →
   ``training_client.forward(..., "forward_logprob")``. That call returns
   target-aligned logprobs. Before writing them back to a trajectory, convert
   through ``align_target_logprobs_to_tokens`` so ``old_log_probs`` remains
   token-aligned with a ``0.0`` no-context slot at position 0. Do not route
   recompute through ``sampling_client.compute_logprobs``.

5. **Comparing sampling_client vs training_client logprobs:** they are the
   same logprob in principle but typically differ by 1e-2 to 1e-4 due to
   bf16/fp32 and sampler-vs-training kernel differences (measured on
   Qwen3-8B). For anything where numerical fidelity matters (KL, ref
   policy, recomputed ``old_log_probs``), prefer
   ``training_client.forward`` — same kernel as the loss computation.

6. **``ref_logprobs`` in particular:** must come from ``training_client``
   (or be a frozen snapshot in the same dtype). Preserve the trainer-forward
   path and the explicit token/target alignment when refactoring reference
   model code. Validate those shapes using the converter tests above.
