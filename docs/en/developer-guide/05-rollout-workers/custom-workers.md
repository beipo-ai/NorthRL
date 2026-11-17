English | [简体中文](../../../zh/developer-guide/05-rollout-workers/custom-workers.md) | [Documentation index](../../index.md)

# Creating Custom Rollout Workers

Implement `async rollout(task)` on `BaseRolloutWorker` when a task needs custom
prompt construction, multiple model calls, or its own scoring logic. For a
single response with a built-in reward, use `rollout.type: single_turn`. For
sandbox execution, use the task hooks described in
[E2B task integration](e2b-task-integration.md).

<a id="recipe-integration"></a>

## Recipe Integration

Keep task code beside its recipe:

```text
recipe/my_task/
├── train.yaml
└── worker.py
```

Merge this selection into a complete training configuration:

```yaml
rollout:
  type: custom
  custom_module_path: worker.py
  custom_class_name: MySimpleWorker
```

`custom_module_path` resolves relative to the recipe YAML. The default class name
is `CustomRolloutWorker`; set `custom_class_name` when using another name.
The controller imports that file and constructs the class with the full config.
Use `config.rollout.params` for task settings. Sibling directories are not added
to `sys.path` automatically: install shared helper packages in the execution
environment or keep small helpers in the worker module.

<a id="single-response-worker"></a>

## Single-response Worker

This implementation uses the existing completion interface, retains the actual
token IDs and sampling logprobs, and scores an answer against the task's `answer`.
Put it in `worker.py`:

```python
from northrl.rollout_worker import BaseRolloutWorker
from northrl.northrl_types import Trajectory
from typing import Any
import logging

logger = logging.getLogger(__name__)

class MySimpleWorker(BaseRolloutWorker):
    """Custom worker for simple tasks."""

    async def rollout(self, task: dict[str, Any]) -> str | None:
        # 1. Extract and validate task data
        if "question" not in task:
            logger.error("Task missing 'question' field")
            return None

        question = task["question"]
        context = task.get("context", "")

        # 2. Format prompt
        prompt = self._format_prompt(question, context)

        # 3. Call LLM
        # Sampling parameters come from rollout.sampling in the configuration.
        result = await self._inference_client.completion(prompt)
        train_fields = result.get("northrl_train", result)
        choices = result.get("choices") or [{}]
        response = result.get("response", choices[0].get("text", ""))

        # 4. Process response
        answer = self._extract_answer(response)

        tokens = train_fields["prompt_tokens"] + train_fields["response_tokens"]
        loss_mask = [0] * len(train_fields["prompt_tokens"]) + [1] * len(train_fields["response_tokens"])
        response_log_probs = train_fields.get("response_old_logprobs")
        if response_log_probs is None:
            response_log_probs = train_fields["response_logprobs"]
        old_log_probs = [0.0] * len(train_fields["prompt_tokens"]) + response_log_probs
        tokens, loss_mask, old_log_probs, is_truncated = self._check_and_truncate(
            tokens, loss_mask, old_log_probs
        )

        # 5. Evaluate the retained result
        reward = (
            0.0
            if is_truncated or result.get("finish_reason", choices[0].get("finish_reason")) == "length"
            else self._compute_reward(answer, task)
        )

        # 6. Create one ordinary trajectory with direct token fields
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                "ground_truth": task.get("answer", ""),
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "extracted_answer": answer,
                "old_log_probs": old_log_probs,
                "is_truncated": is_truncated,
            }
        )

        # 7. Submit
        return self._put_trajectory(trajectory)

    def _format_prompt(self, question: str, context: str) -> str:
        """Format task-specific prompt."""
        if context:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        # Task-specific extraction logic
        import re
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    def _compute_reward(self, answer: str, task: dict) -> float:
        """Compute reward."""
        ground_truth = task.get("answer", "")
        return 1.0 if answer.strip() == ground_truth.strip() else 0.0
```

The controller initializes the inference client and component references before
running the worker. Sampling parameters come from `rollout.sampling`; validation
uses its configured sampling settings. Keep `tokens`, `loss_mask`, and
`old_log_probs` aligned. Prompt logprobs are zero placeholders, followed by the
returned response logprobs; do not shift them to target-token positions here.

<a id="multi-step-processing"></a>

## Multi-step Processing

Multiple calls do not automatically require multiple training sequences. To train
only the final call, retain its actual prompt and response as one ordinary
`Trajectory`. If several independent sequences should train, retain them as
plain dictionaries under one complete trajectory:

```python
trajectory = Trajectory(
    segments=[
        {"tokens": first_tokens, "loss_mask": first_mask, "old_log_probs": first_log_probs},
        {"tokens": second_tokens, "loss_mask": second_mask, "old_log_probs": second_log_probs},
    ],
    reward=final_reward,
    is_val=task.get("is_val", False),
)
return self._put_trajectory(trajectory)
```

The snippet assumes each sequence has already been processed and truncated.
Keep one final reward on the complete trajectory. Do not concatenate unrelated prompts or
submit separate trajectories for individual turns of the same execution.
If truncation removes part of an answer that was scored, invalidate that reward
before submission. See [trajectory semantics](../04-trajectory-pool/trajectory-semantics.md).

<a id="error-handling"></a>

## Submission and Errors

For training, `execute_rollout` stages one `_put_trajectory` call and publishes
that object after `rollout` returns. A second submission is an error. Returning
`None` without submitting records a skipped result; returning `"fail"` records
a failure. An escaping exception replaces the staged result with an empty failed
trajectory and propagates. The wrapper does not retry task execution.

Validation calls `rollout` directly and routes `is_val=True` results to the
validator. It does not add automatic skipped/failed objects. Always preserve the
task's `is_val` flag and submit only after sequence processing and scoring finish.

<a id="testing"></a>

## Testing and Debugging

Check answer extraction and reward calculation independently, then verify that a
worker preserves token/mask/logprob lengths, validation routing, and one final
submission. Cover truncated output, missing task data, and failed model calls.
Use the [rollout debugging guide](../../debug_rollout.md) and
[data dumps](../../debug_data_dump_usage.md) to inspect actual model responses.

## Related Documentation

- [Base Rollout Worker](base-rollout-worker.md)
- [Rewards and Evaluation](evaluators.md)
- [Rollout Configuration](../11-configuration-reference/rollout-config.md)
