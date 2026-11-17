English | [简体中文](../../../zh/developer-guide/03-data-loader/sft-data.md) | [Documentation index](../../index.md)

# Supervised Fine-Tuning Data

NorthRL's SFT path uses `SFTStreamingDataLoader` and `SFTRolloutWorker`. It
performs no LLM inference. Local records become token-aligned `input_ids` and
`loss_mask`; managed records remain opaque `SampleRef` values. The worker
passes either variant to Weaver as an independent `Datum`. NorthRL does not
locally pack SFT samples; Weaver owns packing for training.

## Configuration

Set `algorithm.type` to `sft`. The following excerpt shows the SFT-specific
fields; keep the other required model, optimizer, backend, and runtime fields
from your recipe. Omit the entire top-level `rollout` block for SFT. The
Controller always creates exactly one internal pass-through
`SFTRolloutWorker`; if `rollout` is explicitly provided, NorthRL emits a
`UserWarning` and ignores the whole block.

```yaml
models:
  actor:
    name: Qwen/Qwen3-8B
    tokenizer: Qwen/Qwen3-8B

data:
  train:
    files:
      - ./train.jsonl
    batch_size: 4
    shuffle: true
    drop_last: true
    # Optional; paths are resolved relative to this recipe YAML.
    # render_chat: ./custom_sft.py:render_chat
  max_sequence_length: 8192

algorithm:
  type: sft

training:
  # Set exactly one of total_steps or epochs.
  epochs: 3

validate:
  enabled: false
```

For SFT, `algorithm.loss.mode` is optional and is derived as `cross_entropy`;
selecting any other `algorithm.loss.mode`, including `custom`, fails
validation.

For SFT, omit `data.train.rollout_repeat_n`; it is not configurable. If it is
present, NorthRL emits a warning and ignores it. Also omit
`training.weight_sync`: SFT derives weight sync mode `no-sync`.

SFT requires `drop_last: true`. Mixed sources also require `shuffle: true`;
with `files`, `shuffle: false` reads rows in file order every epoch, which lets a
pre-ordered stream replay another trainer's exact sample order. Other values fail
schema validation. `filter_prompts` is unsupported for SFT; length handling always
uses the truncation behavior described below.

`data.train.window_samples` (files only) closes each optimizer window after that
many samples, so a step has a fixed sample count like a global batch size. The
token budget below still bounds every window.

SFT training length can be specified in either steps or epochs, but not both:

```yaml
training:
  total_steps: 1000  # token-budget optimizer updates
# or: epochs: 3      # complete shuffled passes over the source
```

`models.actor.tokenizer` is required for `messages` input. It is not used to
re-tokenize rows that already contain `input_ids` and `loss_mask`.

Relative entries in `data.train.files` and relative custom-renderer paths are
resolved from the recipe YAML, not from the process working directory.
An existing local tokenizer path is resolved the same way; other tokenizer
strings remain remote model IDs.

### Mixing local and Weaver-managed sources

Use `sources` instead of `files` to opt into deterministic mixing. The two
forms are mutually exclusive. Mixed mode requires an explicit number of
logical samples per epoch:

```yaml
data:
  train:
    total_samples: 100000
    seed: 42  # optional; defaults to experiment.seed
    managed_variant_ready_timeout_seconds: 300
    managed_variant_poll_initial_seconds: 1
    managed_variant_poll_max_seconds: 10
    sources:
      - type: local
        files: [./train.jsonl]
        ratio: 0.8
      - type: managed
        dataset: hq-math
        version: 2026-08
        ratio: 0.2
    batch_size: 8
    shuffle: true
    drop_last: true
  max_sequence_length: 4096
```

`ratio` is required on every source. Ratios are explicit positive weights and
are normalized. Catalog `recommended_ratio` is visible as a recommendation,
but NorthRL never silently substitutes it for configured input. NorthRL converts
the configured weights to integer quotas with the largest-remainder method;
equal fractional remainders are assigned in source declaration order. For each
source, the plan repeats every logical index `quota // source_size` times and
deterministically samples the remainder without replacement. It concatenates
those virtual indices and then applies one global shuffle. Safe plan logs show
the fractional planned and integer realized count for each source.
Mixed mode therefore requires `shuffle: true`; disabling the versioned global
shuffle is rejected instead of being silently ignored.

Mixed permutations use the versioned
`sha256-counter-fisher-yates-v1` algorithm. Its SHA-256 counter stream,
rejection-sampled `randbelow`, Fisher-Yates swaps, and source/global domain
separation are part of the resume contract; changing them requires a new plan
version. Epoch and source ordinal are included in the random domain.

Managed records remain opaque. NorthRL sees only `dataset`, explicit immutable
`version`, stable `sample_idx`, and the model-bound `input_token_count`. The
dataset and version are preserved exactly as configured and must each be one
safe URL path segment (no boundary whitespace, slash, backslash, control
character, `.` or `..`; limits are 160 and 128 characters respectively). The
count is the final shifted Datum input length after the model/run's rendering,
tokenization, and truncation policy. NorthRL never receives managed messages,
loss masks, or token IDs, and never constructs `-8` placeholder inputs. Every
mixed occurrence, including repeated references and local rows, receives a
stable `datum_id`; one managed occurrence becomes one
`Datum.from_sample_ref(...)` at the Weaver boundary.

The model is fixed for the lifetime of the training run, and each length is
resolved for that model together with the explicit dataset name and version.
NorthRL binds resume to the deterministic source plan and committed cursor; it
does not expose or persist a server-private materialization identifier.

Catalog metadata and one opaque length lookup per managed source are resolved
during loader preparation. This validates publication, access, and the pinned
model profile before iteration. Model-bound variants are prepared asynchronously,
so NorthRL polls this initial lookup with bounded exponential backoff. The default
timeout/initial/max delays are 300/1/10 seconds and can be configured with the
three `managed_variant_*_seconds` fields shown above. Only retryable errors whose
exact code is `managed_dataset_variant_not_ready` or
`managed_datasets_unavailable` are retried; authorization, not-found, protocol,
and other failures are propagated immediately. Polling logs contain only safe
dataset/version identities plus readiness status and timing.

Phase-one managed references support only Weaver's built-in `cross_entropy`
`forward_backward` operation. A managed occurrence may carry caller-owned
`loss_fn_inputs` and request metadata, and a mixed batch may pass the built-in
cross-entropy configuration. Those values retain datum order through every
transport batch. `model_input`, `target_tokens`, `loss_mask`, and `weights` are
server-owned and cannot be overridden by the caller. Custom losses remain out
of scope. These rules apply equally to `public` and `protected` datasets;
content visibility controls response disclosure, not the allowed training
operation.

NorthRL pins `data.max_sequence_length` once as
`training_max_sequence_length` when it creates the Weaver model. Length lookup
and later materialization use that model-bound value; it cannot be overridden
per lookup. SFT therefore requires `data.max_sequence_length >= 2`: at least
two complete tokens are needed to produce one shifted training position.

## Supported Input Formats

Only JSONL (`.jsonl`) files are supported. Records are read on demand instead
of materializing the complete dataset. A dataset may
use either of the following row formats.

### Tokenized Rows

Provide both `input_ids` and `loss_mask`:

```json
{"input_ids": [151643, 872, 198, 9707], "loss_mask": [0, 0, 0, 1]}
```

The two arrays must:

- contain integers;
- have the same non-zero length;
- use only `0` and `1` in `loss_mask`;
- select at least one trainable token with `1`.

`loss_mask[i]` describes whether `input_ids[i]` contributes to the training
loss. NorthRL always sets `loss_mask[0]` to `0`, because the first token has no
preceding token from which to predict it.

`labels` cannot substitute for `loss_mask`. A row with `input_ids` but no
`loss_mask` fails immediately, even if it contains `labels`.

### Raw Message Rows

Provide an OpenAI-style `messages` array:

```json
{"messages": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4"}]}
```

When `data.train.render_chat` is omitted, NorthRL reads `row["messages"]` and
splits the conversation into independent rounds, each ending with one
`assistant` message. For every complete round it renders two token sequences:

```python
prompt_ids = tokenizer.apply_chat_template(
    round_messages[:-1],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=False,
)
round_ids = tokenizer.apply_chat_template(
    round_messages,
    add_generation_prompt=False,
    tokenize=True,
    return_dict=False,
)
```

`prompt_ids` must be a strict prefix of `round_ids`. NorthRL assigns `0` to
that entire prompt prefix, including the assistant header added by
`add_generation_prompt=True`. It assigns `1` to the remaining suffix, so the
assistant completion and its end-of-turn token contribute to the loss. A
trailing round without an assistant response is rendered with an all-zero
mask. The chat template is responsible for putting the assistant header in
`prompt_ids` and any end-of-turn marker in `round_ids`.

Each round is rendered independently and then appended to the sample. A BOS
token belongs only at the beginning of the combined sequence, so a repeated
leading BOS produced while rendering a later round is not appended again.

This role-based construction does not use Transformers'
`return_assistant_tokens_mask` and does not require the chat template to
contain `{% generation %}` blocks. It supports the usual sequence in which a
non-assistant prompt is followed by one assistant response. Conversations
that start with `assistant` or contain consecutive assistant messages require
a custom renderer.

## Custom Chat Rendering and Loss Masks

Set `data.train.render_chat` to a Python file and function. A relative module
path is resolved relative to the recipe YAML:

```yaml
data:
  train:
    render_chat: ./custom_sft.py:render_chat
```

The function must be synchronous and have this contract:

```python
def render_chat(row, tokenizer) -> tuple[list[int], list[int]]:
    return input_ids, loss_mask
```

`row` is the complete dictionary loaded from the current JSONL record, not
only its `messages` field. This allows custom renderers to use
fields such as `prompt`, `answer`, or metadata. The default renderer requires
`row["messages"]` and uses the assistant-terminated round behavior described
above. Use a custom renderer for another message layout or model-specific loss
boundary. Its returned `loss_mask` follows the same binary, length, and
non-empty rules as tokenized input.

## Length Handling

NorthRL truncates `input_ids` and `loss_mask` together to
`data.max_sequence_length`. It then sets the first mask value to `0` and
requires at least one remaining `1`. Truncation emits a warning rather than
failing. A row that has no trainable token after truncation is still invalid.

## Token-Budget Batches and Weaver Packing

For each training request, NorthRL uses the token budget:

```text
token_budget = data.max_sequence_length * data.train.batch_size
```

Each Datum contributes `len(input_ids) - 1` next-token prediction positions to
that budget.

Prepared rows are accumulated while they fit that budget. If adding the next
row would exceed it, the current batch is sent and that row starts the next
batch. Every row remains an independent `Datum`; NorthRL never concatenates
`input_ids` or `loss_mask`. Weaver receives the complete group and performs the
actual packing.

For a managed row, the same comparison uses its authorized
`input_token_count`. The complete `SampleRef` is carried to the next batch on
overflow; V1 never splits or windows it.

At the end of a mixed epoch, every non-empty optimizer-window tail is emitted
even if variable sample lengths leave unused token budget. This keeps a larger
window multiplier from dropping the final few planned samples. Legacy
local-only mode retains its token-budget-tail behavior. An EOF tail is never
combined with records from the next epoch, and the source is deterministically
reshuffled before every epoch.

With `training.epochs`, the loader stops after handling the final configured
epoch's tail. With `training.total_steps`, it continues into new epochs until
the step limit.

## Prefetch and Exact Resume

While Weaver trains batch N, one background worker reads, renders/tokenizes,
and prepares batch N+1. There is exactly one lookahead batch, so preparation
cannot run arbitrarily far ahead of training.

Only a successfully trained batch advances the committed source cursor.
Checkpoints record the committed cursor (epoch and position). Mixed checkpoints
also record the deterministic plan schema, permutation algorithm, total, source
identities/sizes/quotas, and plan identity. Resume reconstructs the same epoch
order and continues from the exact committed source position; a merely
prefetched batch is replayed rather than skipped. The model remains fixed for
the lifetime of a training run, while managed rendering and tokenization stay
entirely server-side. Resume state does not add content, tokenizer, renderer,
or configuration hashes; the mixed-plan identity exists only to reproduce the
agreed virtual order and stable occurrence IDs.

## Data Flow

```text
messages ── default/custom render_chat ──┐
                                        ├─ input_ids + loss_mask
tokenized row ───────────────────────────┘
                                                   │
                                                   ▼
                               SFTStreamingDataLoader token-budget group
managed dataset ── virtual SampleRef + safe length ─┘
                                                   │
                                                   ▼
              SFTRolloutWorker → independent Trajectories → Weaver packing/training
```

Invalid rows report their dataset index. Common causes are a missing
`loss_mask`, an unsupported assistant-message layout, a round prompt that is
not a token prefix of the complete round, unequal token and mask lengths,
non-binary mask values, or a mask with no trainable tokens.
