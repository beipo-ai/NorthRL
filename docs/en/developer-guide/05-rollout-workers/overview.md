English | [简体中文](../../../zh/developer-guide/05-rollout-workers/overview.md) | [Documentation index](../../index.md)

# Rollout Workers Overview

**Terminology:** One rollout produces one complete `Trajectory`. Ordinary trajectories store `tokens` and `loss_mask` directly (`segments=None`); results that need separate training sequences use `segments` as a list of dictionaries. Multiple interaction turns do not by themselves require segments. Workers, the pool and trainer preparation exchange complete trajectories. See [trajectory semantics](../04-trajectory-pool/trajectory-semantics.md).

Rollout workers are the components responsible for generating trajectories by executing tasks and interacting with LLM services. They form the data generation pipeline that feeds the training process.

## Architecture

```
DataLoader → RolloutWorker → TrajectoryPool
                ↓
        Reward calculation
```

Rollout workers:
1. Fetch tasks from the data loader
2. Execute the task (interact with LLM/agent)
3. Evaluate the result
4. Prepare direct token fields, or internal records when the result is segmented
5. Construct one complete `Trajectory` with its final evaluation reward
6. Submit that object once; the training wrapper publishes it after `rollout` returns

## Worker Hierarchy

```
NorthRLModule
    ↓
BaseRolloutWorker (abstract)
    ↓
    ├── SimpleRolloutWorker (basic LLM completion)
    ├── SingleTurnRolloutWorker (single response and reward)
    ├── SFTRolloutWorker (pass-through for SFT)
    ├── E2BRolloutWorker (remote agent execution; sync / fully-async)
    └── Custom Workers (task-specific, derived from BaseRolloutWorker)
```

## When to Use Each Worker

### SimpleRolloutWorker
- **Use case**: Basic single-turn LLM completion tasks
- **Features**: Prompt → LLM → Response
- **Example**: Question answering, text generation

### SingleTurnRolloutWorker
- **Use case**: Single-response training tasks
- **Features**: Generate a response, select its reward, and build a trajectory
- **Example**: Math reasoning, instruction following

### SFTRolloutWorker
- **Use case**: Supervised fine-tuning with tokenized data or raw chat messages
- **Features**: No LLM inference; `SFTStreamingDataLoader` streams JSONL,
  renders messages, and validates token-aligned loss masks
- **Packing**: NorthRL keeps each row as an independent Datum; Weaver performs
  the actual sequence packing
- **Data flow**: `messages` or `{input_ids, loss_mask}` → `{input_ids, loss_mask}` → `Trajectory(tokens=..., loss_mask=..., reward=...)`
- **Selection**: Set `algorithm.type: sft` and omit the entire top-level
  `rollout` block. The Controller creates exactly one internal worker; an
  explicitly provided block emits a `UserWarning` and is ignored in full
- **Usage**: See [SFT Data](../03-data-loader/sft-data.md)

### E2BRolloutWorker

- **Use case**: Agent execution inside an E2B sandbox
- **Boundary**: Currently requires the Weaver backend and `weight_sync.mode: sync`
- **Task input**: Each recipe must set `rollout.execution.input_fields` to a list of unique, non-empty field names; `[]` sends no task fields. The worker copies only selected fields present in the dataloader row into `RolloutInput.task`. Task hooks validate required fields and build the prompt; the worker requires no fixed task field names. For example, a question-answering task can select `[question, context]`. `template` selects the sandbox and is not automatically copied into `task`.
- **SWE prompt**: The SWE recipe selects `[instruction, interface]`. Its task hook requires a non-empty `instruction` and joins an optional non-empty `interface` with a blank line, preserving both texts without adding a heading. These are the only fields used in its Claude prompt.
- **Evaluation and provenance**: `RolloutInput.metadata` carries `task_id`, `group_id`, `run_id`, and `expected_rollouts` when present, plus `weight_version`. Evaluation uses `metadata.task_id`, so `task_id` need not be selected in `input_fields`. Metadata is not automatically included in the agent prompt.
- **Input isolation**: Unselected dataloader fields are excluded from `task`. Recipes should select only problem inputs and keep hidden answers and tests in evaluation materials uploaded in the evaluation stage. SWE excludes `patch` and `test_patch`. The runtime checks the request input file's owner and applies mode `0600`. Task hooks run the agent as a restricted user that cannot read that file.
- **Result**: One complete `Trajectory` with internal sequence dictionaries and one reward; the remote contract requires at least one segment
- **Trace and result**: `RemoteRolloutTrace` is an in-progress checkpoint. `RemoteRolloutResult` is the delivered result; on timeout it may use the last valid partial trace with reward 0. Both expose `segments`.
- **Periodic trace sync**: With trace output enabled, gzip snapshots are fully validated and atomically saved by a shared pool of at most two spawned child processes. Parsed token objects stay outside the training process's GIL. Identical compressed snapshots skip repeated validation/writes after a successful save. Invalid snapshots never replace the last valid file. Background admission still permits at most four processing callbacks; cancellation drains writes before retries, and the last sync task joins the pool. Final results optionally use `rollout.execution.result_processing: process`; partial training-trace recovery keeps its existing decode path.

### Custom Workers
- **Use case**: Task-specific behavior
- **Features**: Custom query formatting, specialized evaluation, domain logic
- **Example**: Task-specific prompts, multi-step model calls

## Core Workflow

### 1. Main Loop

```python
while not self._stop_event.is_set():
    with self._activity_tracker.track(self._module_name, "rollout"):
        self._next_task = self._get_rollout_task()
        while self._next_task is not None and not self._stop_event.is_set():
            await self.execute_rollout(copy.deepcopy(self._next_task))
            if not self._stop_event.is_set():
                self._next_task = self._get_rollout_task()

    if self._next_task is None and not self._stop_event.is_set():
        await asyncio.sleep(0.1)
```

### 2. Rollout Execution

The main loop calls `execute_rollout(task)`. Custom workers continue to implement
`async rollout(task)` and submit a complete trajectory through `self._put_trajectory(trajectory)`:

```python
async def rollout(self, task: dict[str, Any]) -> str | None:
    """
    Execute single rollout operation.

    Returns:
        - "success": Complete trajectory accepted for submission
        - "fail": Failed to submit
        - None: No explicit status; absent trajectory submission is marked skipped
    """
```

For training, `execute_rollout` owns the request lifecycle. Custom `rollout`
implementations construct one complete `Trajectory`: direct token fields for an
ordinary result, or internal dictionaries for separate sequences, plus the final
reward. They call `_put_trajectory` once. The wrapper stamps
and validates the execution identity, keeps the object pending locally, and
publishes it to the pool only after `rollout` returns. A second submission within
the same rollout is an error; segment streaming and separate finish notifications
are not part of this interface.

A scored execution with no training segments explicitly submits
`Trajectory(segments=[], reward=final_reward, extra_fields={"score": score})`.
Normal return without a submitted object produces an empty skipped trajectory.
Returning `"fail"` marks the staged object failed, or creates an empty failed object
if none was submitted. An exception escaping `rollout` replaces any staged result
with an empty failed trajectory and then propagates. The wrapper does not retry.

Validation tasks (`task["is_val"]`) call `rollout` directly. Objects marked
`is_val=True` go to the validator immediately; the training wrapper's staging,
single-submission guard, and automatic skipped/failed outcomes do not apply.

<a id="3-trajectory-creation"></a>

### 3. Complete Trajectory Creation

Ordinary workers store their training sequence directly:

```python
trajectory = Trajectory(
    tokens=tokens,
    loss_mask=loss_mask,
    reward=final_reward,
    is_val=task.get("is_val", False),
    extra_fields={"old_log_probs": old_log_probs, "score": score},
)
assert trajectory.segments is None
```

A multi-turn execution can still use this form when it retains one valid token
sequence. When independent sequences must be retained, submit them as dictionaries
inside one complete result. Do not concatenate unrelated prompts just to avoid
segments:

```python
Trajectory(
    segments=[
        {
            "tokens": [...],
            "loss_mask": [...],
            "old_log_probs": [...],
        },
        # Additional segments from the same execution, if any.
    ],
    reward=final_reward,
    is_val=task.get("is_val", False),
    extra_fields={
        "ground_truth": ground_truth,
        "score": score,
        # Execution identity is supplied/validated by the wrapper.
    },
)
```

### 4. Per-turn Tool Diagnostics

The E2B NorthGate path preserves the provider's `tool_call_diagnostics` for each
recorded model response. It adds no recipe configuration or diagnostics switch:
evidence is available when NorthGate returns it. The relay retains it for both
streaming and non-streaming responses, and the recorder attaches it to the
response's owning sequence:

```text
NorthGate response → TrajectoryRecorder → RemoteSegment / compact trace
                  → E2BRolloutWorker → Trajectory → prepared dump / replay
```

Each sequence has a `turn_annotations` list. An individual record looks like:

```json
{
  "version": 1,
  "turn_id": "turn-0",
  "response_span": [2, 4],
  "diagnostics": {
    "checked": true,
    "malformed": true,
    "detected": 1,
    "invalid": 1,
    "reasons": ["invalid_arguments_json"]
  },
  "finish_reason": "tool_calls",
  "truncated": false,
  "tools_declared": true,
  "source": "northgate"
}
```

`diagnostics` is the raw JSON object, including unknown fields and reason values;
the transport does not infer it from tool execution results or aggregate metrics.
A missing or non-object provider value becomes `None` (`null` in JSON).
`turn_annotations: []` means there are no recorded turn annotations, and
`diagnostics: null` means that turn has no diagnostic evidence. Neither establishes
that a response was checked and found clean. `tools_declared` means the request's
`tools` value was not `None`; an explicit empty list also counts as a known
declaration. `finish_reason` retains the original OpenAI `finish_reason` or
Anthropic `stop_reason`, or `None` when unavailable.

`response_span` is a zero-based, half-open interval `[start, end)` in the **full
sequence's token array**, including its prompt and intervening tool/context
tokens. It is not an index into the compressed list of response tokens or the
shifted `tokens[1:]` targets. For example, tokens `[10, 11, 20, 21, 30, 40, 41]`
with mask `[0, 0, 1, 1, 0, 1, 1]` can own two turns at `[2, 4]` and `[5, 7]`.
Remote validation requires `1 <= start < end <= len(tokens)`, response-only mask
values within each span, ordered non-overlapping spans, and unique non-empty
turn IDs within the segment. Adjacent turns remain separate annotations even
when compact encoding combines their response-mask spans.

For a direct trajectory (`segments=None`), store annotations in
`extra_fields["turn_annotations"]`. For a segmented trajectory, store them on
each child dictionary, alongside that child's `tokens` and `loss_mask`:

```python
# annotation is the record above; both examples own tokens at positions [2, 4).
sequence = {"tokens": [10, 11, 20, 21], "loss_mask": [0, 0, 1, 1]}
direct = Trajectory(
    **sequence, reward=1.0, extra_fields={"turn_annotations": [annotation]}
)
segmented = Trajectory(
    segments=[{**sequence, "turn_annotations": [annotation]}], reward=1.0
)
```

Do not put segment-specific annotations on the segmented complete trajectory's `extra_fields`:
each child's coordinates and response ownership are independent. Existing
workers without this evidence may omit the field; remote segment contracts
default it to `[]`.

The recorder assigns `turn-0`, `turn-1`, and so on across one rollout, incrementing
after each successfully recorded response. If a new prompt extends the latest
segment's exact tokens, it appends only the new context suffix and response,
retaining previous annotations and adding one new record. Otherwise it starts a
new segment with only the new response's annotation. Historical response tokens
repeated in the new prompt are context with mask 0; their annotations remain on
the segment that originally generated them.

`truncated` records incomplete or uncertain response completion. Provider
`length` / `max_tokens` termination and custom stop settings mark it true. When
the Driver clips a segment to its configured maximum length, it drops annotations
whose start is outside the retained tokens and clips crossing spans to the new
end, setting their `truncated` flag. Earlier fully retained turns keep their
original flags. The annotation's provider `finish_reason` remains intact even
if the worker sets a segment-level `finish_reason` such as `length` or `timeout`.

Prepared trajectory dumps preserve annotations in both `.pt` and `.jsonl`, in
the direct/child locations above. `MockRolloutWorker` file replay restores those
records; it does not reconstruct missing evidence. Existing dump settings still
control whether files are written; they are not required to collect annotations.
See [data dump usage](../../debug_data_dump_usage.md).

The compact wire envelope remains `schema_version: 2`; annotation `version: 1`
is a separate field. New readers accept older v2 segments without
`turn_annotations` and use `[]`. Older readers reject the new field because their
schema forbids undeclared fields, so Driver and sandbox code must use matching
versions. The existing executor packages the current source for the sandbox.
See [E2B task integration](e2b-runtime.md#trace-schema) for the wire boundary.

## Key Concepts

### Activity Tracking

Workers track their activity for monitoring:

```python
with self._activity_tracker.track(self._module_name, "rollout"):
    result = await self.execute_rollout(task)
```

This enables:
- System health monitoring
- Quiescence detection (all workers idle)
- Performance profiling

### Weight Synchronization Coordination

The trajectory pool accepts completed results throughout weight synchronization.
It does not request another rollout because synchronization is in progress.

**Sync modes:**
- **sync**: The next input batch waits for training, synchronization, and validation
  when required.
- **fully-async**: Input loading continues automatically. Training overlaps rollout;
  each updated training round is saved and synchronized before the next training
  round. With `freeze_inflight: true`, new inference requests wait during sync.
- **batch-async**: The next input batch waits when the staleness threshold is
  exceeded; otherwise it is released before the current batch trains.

After `execute_rollout()` returns and publishes the complete trajectory, the worker
claims the next task unless stop was requested. Any task-specific retries belong
inside `rollout()` itself.

### Validation Mode

Workers support both training and validation:

```python
def begin_validate(self):
    """Switch to validation mode"""
    self._is_running_validate = True

def end_validate(self):
    """Switch back to training mode"""
    self._is_running_validate = False
```

When in validation mode:
- Tasks come from validation data loader
- Complete trajectories go to the validator (not the trajectory pool)
- Validation metrics computed after collection

### Inference Service Integration

Workers access LLM services through inference client:

```python
# Set during initialization
self._inference_client = create_inference_service_client(
    backend=config.backend.type,
    config=config
)

# Use in rollout
result = await self._inference_client.completion(
    prompt=prompt  # Sampling settings come from configuration.
)
```

## Module References

Workers need references to other components:

```python
def set_module_references(
    self,
    trajectory_pool: TrajectoryPool,
    dataloader: BaseDataLoader,
    weight_sync_controller: WeightSyncController,
    validate_dataloader: BaseDataLoader,
    validator: Validator,
):
    """Set module references before run()"""
```

These are set by the controller during initialization.

## Configuration

Basic worker configuration:


## Common Patterns

### Basic Completion Pattern

The ordinary constructor below stores token fields directly; `segments` stays `None`.

```python
class MyWorker(BaseRolloutWorker):
    async def rollout(self, task):
        # Get prompt
        prompt = task["prompt"]

        # Call LLM
        result = await self._inference_client.completion(prompt)
        train_fields = result.get("northrl_train", result)

        # Create trajectory
        trajectory = Trajectory(
            tokens=train_fields["prompt_tokens"] + train_fields["response_tokens"],
            loss_mask=[0] * len(train_fields["prompt_tokens"]) + [1] * len(train_fields["response_tokens"]),
            reward=self._evaluate(result, task),
            is_val=task.get("is_val", False),
            extra_fields={"ground_truth": task["answer"]}
        )

        # Submit
        return self._put_trajectory(trajectory)
```

### Sandbox Task Hooks

For sandbox tasks, keep `run_agent` and `evaluate` in a task entrypoint and use
`E2BRolloutWorker` to record model interactions and submit the final result.
See [E2B task integration](e2b-task-integration.md) and
[reward evaluation](evaluators.md).

## Error Handling

For training tasks, exceptions escaping `rollout` retain fail-fast behavior. The wrapper publishes an empty
failed Trajectory, then the exception propagates; it does not swallow
the exception and continue with the next task:

```python
async def rollout(self, task):
    try:
        # Process task
        result = self._process(task)
        return self._put_trajectory(result)
    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        raise  # execute_rollout submits an empty failed Trajectory, then propagates it
```

The `track()` context reports propagated errors. Skipping an invalid GRPO group
does not imply automatic recovery from every worker exception.

## Performance Considerations

### Parallel Workers

Multiple worker coroutines can overlap their work:


**Guidelines:**
- Increase concurrency only while it improves measured throughput
- Keep within local driver memory and CPU capacity
- Respect inference service capacity and rate limits

### Batching

Workers process tasks individually, but batching happens in:
1. **Inference service**: Batches multiple requests
2. **Trajectory pool**: Delivers complete trajectories/groups; token operations visit their sequences internally

### Resource Management

Workers run as local coroutines. Configure `rollout.num_workers` and, when needed, `rollout.max_concurrent_requests` to stay within driver memory and remote-service limits. See [rollout configuration](../11-configuration-reference/rollout-config.md).

## Next Steps

- [Base Rollout Worker](base-rollout-worker.md) - Core worker interface and methods
- [Custom Workers](custom-workers.md) - Creating task-specific workers
- [Evaluators](evaluators.md) - Implementing evaluation logic

## Related Documentation

- [Data Loader](../03-data-loader/data-loader.md) - Task source
- [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) - Trajectory destination
- [Activity Tracking](../02-core-architecture/activity-tracking.md) - Monitoring
- [Configuration Reference](../11-configuration-reference/rollout-config.md) - Full config options
