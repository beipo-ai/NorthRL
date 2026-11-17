English | [简体中文](../../../zh/developer-guide/03-data-loader/data-loader.md) | [Documentation index](../../index.md)

# Data Loader

The Data Loader provides input data for rollout workers during training and validation. NorthRL includes built-in data loaders and an abstract interface for custom implementations.

## Overview

**Purpose**: Manage dataset loading, shuffling, and distribution to rollout workers

**Key Features**:
- Sequential task delivery with optional source shuffling
- Explicit release of the next input batch for ordered loading
- Validation and training data separation
- Threadsafe operation
- Configurable shuffling and batching

## BaseDataLoader

**Location**: `northrl/data_loader/data_loader.py`

**Base class** for all data loaders providing the core interface.

### Constructor

```python
def __init__(self, config: DictConfig, is_validate: bool = False)
```

**Parameters**:
- `config`: Configuration for the data loader
- `is_validate`: Whether this dataloader is for validation (affects behavior and tracking)

**Attributes**:
- `_is_validate`: Flag indicating validation mode

### Abstract Methods

Subclasses must implement these methods:

#### `get_next_item() -> dict[str, Any] | None`

Get the next data item in sequence.

```python
item = dataloader.get_next_item()
if item is not None:
    process(item)
```

**Returns**: One rollout request as a dictionary, or `None` when no request can be issued now. For an ordered loader, `None` can mean that the current input batch has been claimed and the next batch is waiting for `release_next_batch()`; it does not by itself mean the dataset is exhausted. Use `is_finished()` to distinguish source exhaustion from this temporary wait.

#### `is_finished() -> bool`

For TorchDataLoader, check whether the finite source is exhausted and every buffered request has been claimed. This does not mean all claimed rollouts have finished. Cycling RL loaders remain unfinished while training continues.

```python
if dataloader.is_finished():
    print("All source inputs have been claimed")
```

**Returns**: `True` when the source and task buffer are exhausted

#### `can_return_item() -> bool`

For `SequentialDataLoader`, this only checks whether the current buffer has an
unclaimed item. It does not read the data source or fill an empty buffer.

```python
has_buffered_item = dataloader.can_return_item()  # A buffer snapshot
item = dataloader.get_next_item()                # May fetch the first/next batch
```

Before the first fetch, `can_return_item()` is `False` even for a nonempty
dataset. Do not require it to be `True` before calling `get_next_item()`.
Rollout workers call `get_next_item()` directly; the loader handles the batch gate.

#### `reset()`

Reset the dataloader to initial state.

```python
dataloader.reset()
# Can iterate through data again
```

**Usage**: Called between validation cycles

### Convenience Methods

#### `release_next_batch()`

For an ordered loader, fetch and expose the next input batch once the current buffer has been claimed. The caller is responsible for waiting until the current round has been processed. In RL, the weight sync controller performs this release after the applicable training/synchronization or validation boundary. Fully asynchronous loaders fetch the next batch from `get_next_item()` and need no explicit release.

```python
dataloader.release_next_batch()
```

#### `subscribe_ready(callback) -> Callable[[], None] | None`

Optional readiness subscription. Return an unsubscribe function when supported,
or `None` to keep the worker's polling fallback. `SequentialDataLoader` notifies
after `release_next_batch()` and `reset()`, outside its data lock. Callbacks may
run on the publishing thread and should only schedule a wakeup, not claim tasks.
Subscribers must recheck loader state after subscribing and after each wakeup.
Custom implementations must notify after every change that can make an empty
loader return work; notifications are hints, not task reservations.

## TorchDataLoader

**Location**: `northrl/data_loader/torch_data_loader.py`

**Concrete implementation** using PyTorch DataLoader for file-based datasets.

### Features

- Loads Parquet, CSV, and JSONL files into one filtered dataset
- Uses a dedicated deterministic shuffle generator
- Automatically cycles step-based RL training across epochs
- Keeps validation finite
- Joins an epoch tail with the next epoch head to keep training batches full
- Supports prompt-length filtering before iteration

### Single-turn Reward Data

For `rollout.type: single_turn`, each dataset row contains the prompt,
an optional reward name, and any reference data required by that reward. Use JSONL (one JSON object per
line) or Parquet with the same columns to preserve nested messages and metadata.
The loader also reads CSV, but does not decode JSON strings inside CSV cells into
message lists or dictionaries.

| Field | Required | Meaning |
| --- | --- | --- |
| `prompt` | Yes | A list of chat messages, such as `[{"role": "user", "content": "What is 40 + 2?"}]`. Use the literal top-level key `prompt` for this agent. |
| `label` | Reference answer for rewards that compare against a target | Use a non-empty string, such as `"42"` for math or `"D"` for `rule_nemotron_mcqa`. `reasoning360` requires a string, ordered string list, table dictionary, or integer grid according to `metadata.data_source`. `code_e2b` requires a prepared judge-request dictionary, including all test cases. `rule_reasoning_gym`, `synlogic`, `ifeval`, `if_verify`, `recast`, and `verinstruct` do not require a label. Legacy math answer fields are also supported below. |
| `reward_fn` | Unless a default is configured | A short registered name, such as `math_answer_only`. A value on this row overrides `rollout.reward_fn`. |
| `metadata` | Depends on reward | A dictionary of task-specific information, passed to the reward function as `task["metadata"]`. `rule_reasoning_gym`, `reasoning360`, `synlogic`, `ifeval`, `if_verify`, and `verinstruct` read task state, cases, constraints, or checker source from this dictionary; see below. |

For example, save these five lines as `train.jsonl`:

```jsonl
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "label": "42", "reward_fn": "math_answer_only", "metadata": {"source": "arithmetic"}}
{"prompt": [{"role": "user", "content": "What is 6 * 7? Put the final answer in \\boxed{}."}], "label": "42", "reward_fn": "math_with_format"}
{"prompt": [{"role": "user", "content": "What is 20 + 22? Put the final answer in \\boxed{}."}], "label": "42"}
{"prompt": [{"role": "user", "content": "Give exactly three suggestions for learning Python, in Chinese."}], "reward_fn": "recast"}
{"prompt": [{"role": "user", "content": "Which option is correct? A. First B. Second. End with Answer: \\boxed{A/B}."}], "label": "B", "reward_fn": "rule_nemotron_mcqa"}
```

Set these fields in an existing training recipe:

```yaml
data:
  train:
    files:
      - ./train.jsonl
    prompt_key: prompt

rollout:
  reward_fn: math_answer_only
  type: single_turn
```

The first row uses `math_answer_only`, the second uses `math_with_format`, and
the third uses the configured `math_answer_only` default. The fourth uses
`recast` to judge instruction following without a reference answer; configure its
[LLM judge](../11-configuration-reference/rollout-config.md#recast-reward) before
using this mixed dataset. Each generated response is scored independently with its row's selected function. No generated response
or numeric reward needs to be stored in the input dataset.

Reward selection follows these rules:

1. A non-null top-level `reward_fn` selects that exact name.
2. An omitted or `null` value uses `rollout.reward_fn`. If every row specifies a
   reward, the configured default can be omitted.
3. If neither location supplies a name, evaluation raises `KeyError("reward_fn")`.
4. An unknown name or an explicit empty string raises an error, even when a
   default exists. CSV empty cells follow pandas' missing-value parsing and are
   treated as absent; JSONL `"reward_fn": ""` remains an explicit empty string.

Use the top-level `reward_fn` to select a registered reward.
Setting `data.train.prompt_key` to a different name does not rename that
column to `prompt` for this agent.

See [Single-turn Reward Selection](../11-configuration-reference/rollout-config.md#single-turn-reward-selection)
for registered rewards, scoring, and LLM requirements. ReasoningGym rows use
`rule_reasoning_gym` and put the concrete task name in
`metadata.metadata.source_dataset`. Nemotron MCQA rows use
`rule_nemotron_mcqa`. IFEval rows use `ifeval` and put their
structured constraints in `metadata.instructions`.

Code datasets use `reward_fn: code_e2b` with the complete judge request in
`label`. See [Code E2B Reward](../11-configuration-reference/rollout-config.md#code-e2b-reward)
for label fields, examples, sandbox configuration, and scoring.

Reasoning360 rows use the single public name `reasoning360` and put one of the
seven supported source names in `metadata.data_source`. That field selects the
internal graph, zebra, ordering, shared ARC/BARC grid, or STEM LLM verifier; it
does not replace the top-level `reward_fn`. `stem_web` rows additionally require
the original question in `metadata.question`.

SynLogic rows use the single public name `synlogic`. They put one of 35 exact
game names in `metadata.data_source` and a serialized SynLogic task object in
`metadata.game_data_str`. The task object contains the puzzle state, reference
answer, and game-specific metadata used by the selected local verifier. The
top-level `label` may be an empty string and is not used. For example:

```jsonl
{"prompt": [{"role": "user", "content": "Complete this logic puzzle."}], "label": "", "reward_fn": "synlogic", "metadata": {"data_source": "sudoku", "game_data_str": "{\"question\": \"...\", \"answer\": \"...\", \"difficulty\": 1, \"metadata\": {\"solution\": \"...\"}}", "rm_type": "synlogic"}}
```

The 35 game verifiers are bundled with NorthRL and return strict binary rewards.
They do not call an LLM, E2B, or a bp-rl-judge service. The `operation` game
uses the pinned `math-verify` dependency; no SynLogic-specific recipe option is
required. See [SynLogic Reward](../11-configuration-reference/rollout-config.md#synlogic-reward).

For `verinstruct`, put Python hard-constraint checkers in
`metadata.hard_functions`. Each string defines
`check_following(instruction, response)` and returns a truthy value when the
response satisfies that constraint. For example, this is one JSONL row; `\n`
inside the checker string represents a newline:

```jsonl
{"prompt": [{"role": "user", "content": "Write a greeting using exactly three words."}], "reward_fn": "verinstruct", "metadata": {"instruction": "Write a greeting using exactly three words.", "hard_functions": ["def check_following(instruction, response):\n    return len(response.split()) == 3"]}}
```

Use `metadata.instruction` to supply the instruction to the checkers;
`metadata.prompt_text` is also accepted. The fallback is a plain-string
`prompt`, or an empty string for a chat-message list. Set `metadata.instruction`
explicitly for single-turn rollout worker data. Missing, null, or empty `hard_functions`
skips hard checks and runs only the LLM check. Parquet arrays of checker strings
are supported. `label` is optional and unused.

The checkers run in a fresh E2B sandbox with network access disabled. NorthRL
creates and cleans up the sandbox for each row with hard checks; rows without
hard checks do not require E2B credentials. Only when all hard checks pass does
the LLM judge instruction following. See [VerInstruct Reward](../11-configuration-reference/rollout-config.md#verinstruct-reward)
for E2B and judge configuration, response normalization, and failure handling.

For `if_verify`, put one Python checker in `metadata.checker`. It must define
`evaluate(response)` and return a truthy value only when the generated response
satisfies the row-specific constraint:

```jsonl
{"prompt": [{"role": "user", "content": "Answer without using the word but."}], "label": null, "reward_fn": "if_verify", "metadata": {"checker": "def evaluate(response):\n    return 'but' not in response.lower()"}}
```

The checker runs in a fresh E2B sandbox with network access disabled, never in
the data loader or driver process. It receives only the normalized final-answer
text; `label` and `prompt` are not checker arguments. Configure the same
`E2B_API_KEY`, optional `E2B_TEMPLATE`, and custom endpoint variables described
for VerInstruct. IfVerify does not call an LLM and adds no recipe parameter. See
[IfVerify Reward](../11-configuration-reference/rollout-config.md#ifverify-reward)
for execution limits and error behavior.

For legacy math data, the adapter first uses a non-null `label`; otherwise it
reads `ground_truth`, then `reward_model.ground_truth` if `ground_truth` is absent
or empty. Prefer `label` in new datasets. An explicit empty `label` does not fall
back to legacy fields. These rows are also accepted and use the configured
reward default:

```jsonl
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "ground_truth": "42"}
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "reward_model": {"ground_truth": "42"}}
```

This contract describes the current single-turn reward path. SimpleRolloutWorker
and E2B keep their existing evaluation paths; SFT has its own
[data contract](sft-data.md).

### Constructor

```python
def __init__(self, config: DictConfig, is_validate: bool = False)
```

**Training configuration**:

```yaml
data:
  train:
    files:
      - /path/to/train.jsonl
    batch_size: 32
    rollout_repeat_n: 16
    prompt_key: prompt
    filter_prompts: true
    shuffle: true
    drop_last: true
  max_prompt_length: 4096

training:
  total_steps: 3000
```

`cycle` is not a user-facing option. The Controller derives it for non-SFT,
step-based training. A dataset path should be listed once; the loader starts a
new epoch automatically until `training.total_steps` is reached.

### Implementation Details

#### Shuffling

- Epoch 0 uses `experiment.seed + 0`
- Epoch 1 uses `experiment.seed + 1`, and so on
- Each epoch permutes the complete filtered dataset exactly once
- `shuffle: false` cycles in the original dataset order

#### Epoch Boundaries

Training is an effectively infinite prompt stream whose stopping condition is
`training.total_steps`. If an epoch has fewer than `batch_size` prompts left,
the loader increments `epoch_id`, reshuffles, and fills the batch from the next
epoch. For example, a 29-item tail with `batch_size: 32` is joined with the
first 3 items of the new epoch. No partial training batch is emitted and no
epoch-tail prompt is discarded.

Validation loaders do not cycle. They retain ordinary finite `drop_last`
semantics and report `is_finished()` after their single pass.

#### Resume

Non-SFT resume continues to use `skip_batches(global_step)`. Because every
epoch order is a deterministic function of `experiment.seed + epoch_id`, the
skip operation can cross epoch boundaries and reconstruct the same data cursor.

#### Filtering

- Filters samples exceeding `max_prompt_length`
- Logs filtered samples count
- Ensures prompts fit in model context window

## MixtureDataLoader

**Location**: `northrl/data_loader/mixture_data_loader.py`

The Controller selects this loader for GRPO training with
`data.train.sources`. It loads each local source through `FileDataset`, draws
a source by its positive `ratio`, and advances that source's independent
shuffled cycle. `experiment.seed` controls source draws and within-source
orders. Sources need not appear in every batch; configured ratios describe
sampling probabilities over time.

The loader returns original inputs to `SequentialDataLoader`, which assigns a
new group and expands each input into the global `rollout_repeat_n` requests.
It injects `data_source` from the source's name and optional `rollout_profile`
from its configuration, preserving the original `task_id`. All attempts in a
group therefore use the same task and execution profile.

Step-based training cycles until `training.total_steps`. Reset and batch
skipping reconstruct the deterministic source stream; async completion order
and in-flight tasks are outside this replay contract. Source settings and the
five-family MiMo/SWE configuration are documented in
[Mixing RL Task Sources](../11-configuration-reference/data-config.md#mixing-rl-task-sources).

## SFTStreamingDataLoader

**Location**: `northrl/data_loader/sft_streaming_data_loader.py`

The Controller selects this training-only loader when `algorithm.type: sft`.
Omit the entire top-level `rollout` block: NorthRL always creates exactly one
internal `SFTRolloutWorker`, and warns then ignores the whole block if it is
provided. See [SFT Data](sft-data.md) for its row contract, configuration,
and custom `render_chat` callbacks.

### Features

- Streams JSONL source records on demand
- Deterministically reshuffles records at every epoch
- Forms requests with a token budget of
  `max_sequence_length * batch_size`
- Keeps every sample as an independent `Datum` for Weaver-side packing
- Drops an incomplete local-only EOF tail; in mixed mode, emits every
  non-empty EOF optimizer-window tail instead. Neither carries into the next epoch
- Commits an exact source cursor only after successful training and resumes
  from that cursor
- Prepares and tokenizes exactly one lookahead batch while the current batch is
  training in Weaver

## Mock DataLoader

**Location**: `northrl/mock/mock_data_loader.py`

**Testing implementation** with generated mock data.

### Features

- Generates fake data on-the-fly
- Configurable number of samples
- No external file dependencies
- Useful for testing and development

This is an internal testing class, instantiated directly by tests. Production
recipes do not select it with `data.type`. Like other sequential loaders, it
supplies `_fetch_batch_data()` rather than replacing task-buffer handling.

## Usage Patterns

### Basic Data Loading

The worker's existing `_get_rollout_task()` calls the selected loader directly.
For a local loader, the basic operation is:

```python
item = dataloader.get_next_item()
if item is not None:
    process(item)
elif dataloader.is_finished():
    print("Source exhausted")
else:
    print("Waiting for the next input batch to be released")
```

Do not override the worker's scheduling logic just to add a `can_return_item()`
check. `get_next_item()` fetches the first batch itself. With `sync` or
`batch-async`, it returns `None` after all requests in that batch have been
claimed, until the weight sync controller calls `release_next_batch()`.
With `fully-async`, the next call fetches the next input batch automatically.
Each call claims one rollout request, not a whole group or a completed trajectory.

### Validation Cycle

The Controller resets the validation loader inside `_start_validate()` before
workers can claim validation tasks. `_end_validate()` waits for completion,
logs results, returns workers to training, and notifies the weight sync controller.

```python
# NorthRLController
def _run_validate(self, model_tag):
    self._start_validate(model_tag)
    self._end_validate(model_tag)
```

## Custom Data Loader Implementation

### Minimal Implementation

Subclass `SequentialDataLoader` to supply the data source while reusing its
locking, task buffer, repeat expansion, reset, resume, and batch-release logic.
This example reads JSONL in file order; it does not implement TorchDataLoader's
shuffling or prompt filtering. Its internal `cycle` flag supports a continuing
RL input stream, while validation stays finite.

```python
import json
from typing import Any

from northrl.data_loader import SequentialDataLoader
from omegaconf import DictConfig


class MyDataLoader(SequentialDataLoader):
    """JSONL source with the standard task buffer and repeat handling."""

    def __init__(self, config: DictConfig, is_validate: bool = False):
        super().__init__(config, is_validate)
        self._data: list[dict[str, Any]] = []
        for filename in config.files:
            with open(filename, encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        item = json.loads(line)
                        item.setdefault("task_id", len(self._data))
                        self._data.append(item)
        self._cycle = bool(config.get("cycle", False)) and not is_validate
        self._drop_last = bool(config.get("drop_last", False))
        self._index = 0

    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        # Called under SequentialDataLoader's lock; return original inputs.
        batch = []
        while self._data and len(batch) < self._batch_size:
            if self._index == len(self._data):
                if not self._cycle:
                    break
                self._index = 0
            batch.append(self._data[self._index].copy())
            self._index += 1
        if not self._cycle and self._drop_last and len(batch) < self._batch_size:
            return []
        return batch

    def _reset_iterator(self) -> None:
        self._index = 0

    def is_finished(self) -> bool:
        with self._lock:
            source_done = not self._data or (
                not self._cycle and self._index >= len(self._data)
            )
            return source_done and self._buffer_index >= len(self._data_buffer)
```

`_fetch_batch_data()` returns up to `batch_size` original inputs. Do not repeat
or assign execution identities here: `_try_fetch_batch()` expands each input
into `rollout_repeat_n` requests and supplies one fresh `group_id`, `run_id`
slots `0..R-1`, unique `rollout_id` values, and `expected_rollouts=R`.
The same dataset item can therefore be drawn again as a separate group.
`_reset_iterator()` resets only the source cursor; the inherited `reset()` also
clears the task buffer and its indices. `is_finished()` checks both source
exhaustion and whether all buffered requests have been claimed.

### Integrating a Custom Loader

The current Controller selects `TorchDataLoader` for RL with `files`,
`MixtureDataLoader` for GRPO with `sources`, and `SFTStreamingDataLoader` for SFT
in `_get_module_class_for_role()`. There is no
public `data.type` selector or DataLoader `MODULE_REGISTRY`. Integrating this
example requires a code change to that selection point, or direct construction
in an embedding application. Pass the internal loader configuration derived
by `_derive_train_dataloader_config()`, which includes `files` or `sources`, `batch_size`,
`rollout_repeat_n`, `cycle`, and `keep_batch_order`.

Keep the inherited `get_next_item()` and `release_next_batch()` methods: they
allow the same source implementation to work with both the ordered sync flow
and fully asynchronous task fetching.

## Configuration Reference

The following is a fragment to merge into an existing RL recipe:

```yaml
experiment:
  seed: 42
models:
  actor:
    tokenizer: /path/to/model
data:
  train:
    files:
      - /path/to/train.jsonl
    batch_size: 32
    rollout_repeat_n: 8
    shuffle: true
    drop_last: true
  max_prompt_length: 4096
  max_sequence_length: 8192
training:
  total_steps: 3000
  weight_sync:
    mode: sync
```

`data.train.batch_size=B` counts original inputs, and `rollout_repeat_n=R`
expands them into `B × R` rollout requests. For GRPO/OPD the Controller derives
`groups_per_batch=B` and `group_size=R` for the trajectory pool. These are
complete rollout groups; a multi-sequence trajectory does not count as several
rollouts. The number of token sequences/Datums sent to training may therefore
differ from `B × R`.

In `fully-async`, grouped GRPO/OPD waits for B eligible complete groups across
input batches. Groups with more than half their members failed, or with no active
training target among valid members, do not fill those slots; workers keep
claiming new requests. The trainer repairs eligible groups before training. The
pool preserves successful empty members, which remain valid replacement donors. Reward
zero or equality is not a rejection condition. This fixes group count at pool
admission per trainer round, before later MIS/OPD filtering or optimizer splitting.
A finite exhausted source with idle workers discards an undersized eligible tail
without training or advancing `train_step`; other grouped modes keep their
complete-group delivery and finite-tail behavior.

`keep_batch_order` and `cycle` are internal derived fields, not recipe options.
The tokenizer comes from `models.actor.tokenizer` and the shuffle seed from
`experiment.seed`. See [Data Loader Configuration](../11-configuration-reference/data-config.md)
for batch units, mode differences, and validation configuration.

## Best Practices

### Data Organization

1. **Use Parquet format** for efficient storage and loading
2. **Split large datasets** into multiple files for parallel loading
3. **Include all required fields** (prompt, answer, id, etc.)
4. **Validate data format** before training

### Configuration

1. **Set appropriate max_prompt_length** to filter oversized prompts
2. **Set `experiment.seed`** for reproducible dataset shuffling
3. **Enable shuffling** for better training dynamics
4. **Choose `data.train.batch_size` and `rollout_repeat_n` together** to size rollout work

### Performance

1. **Monitor buffer size** (avoid excessive memory use)
2. **Balance file count** vs file size
3. **Use SSD/fast storage** for data files
4. **Preprocess data** offline when possible

### Error Handling

1. **Validate file paths** before training starts
2. **Handle missing files** gracefully
3. **Check data format** compatibility
4. **Log loading errors** clearly

## Troubleshooting

### Issue: "No data items available"

**Symptoms**: Workers idle, no tasks being processed

**Solutions**:
1. Check `data.train.files` paths are correct
2. Verify files are not empty
3. Check `max_prompt_length` filter isn't too restrictive
4. Ensure file format is compatible

### Issue: "Data loading is slow"

**Symptoms**: Low rollout worker utilization

**Solutions**:
1. Profile file reading and prompt filtering before changing rollout concurrency
2. Use faster storage (SSD, parallel filesystem)
3. Reduce file size through compression
4. Preload data into buffer

### Issue: "Out of memory"

**Symptoms**: System crashes during data loading

**Solutions**:
1. Reduce `batch_size` for PyTorch DataLoader
2. Process smaller files
3. Filter or preprocess oversized records
4. Filter samples by length more aggressively

## Next Steps

- Understand [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) for trajectory batching
- Learn about [Rollout Workers](../05-rollout-workers/overview.md) that consume data
- Explore [Configuration Reference](../11-configuration-reference/data-config.md) for all options
- Review [Best Practices](../12-best-practices/module-development.md) for custom loaders
