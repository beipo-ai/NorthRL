# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data exchanged between the Driver and an E2B rollout."""

from pathlib import Path
from typing import Literal, Self, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

_MODEL_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RolloutEnvironmentError(RuntimeError):
    """Infrastructure failure without a scored outcome."""

    rollout_status = "env_error"


def _validate_rollout_id(value: str) -> str:
    if not value.strip():
        raise ValueError("rollout_id must be a non-empty string")
    return value


class RolloutInput(BaseModel):
    """Input shared by the platform and task hooks for one rollout.

    ``rollout_id`` identifies this execution; ``task`` is a generic JSON
    dictionary. The E2B worker selects dataloader fields configured in
    ``rollout.execution.input_fields``; task hooks validate their contents and
    build the agent prompt. ``metadata`` carries ``task_id`` for evaluation
    and training provenance into the persisted trace; it is not automatically
    included in the agent prompt.
    """

    model_config = _MODEL_CONFIG

    rollout_id: str
    task: dict[str, JsonValue]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _rollout_id_must_not_be_empty = field_validator("rollout_id")(_validate_rollout_id)


class TurnAnnotation(BaseModel):
    """Raw diagnostics for one generated response, in segment token coordinates."""

    model_config = _MODEL_CONFIG

    version: Literal[1]
    turn_id: str = Field(min_length=1)
    response_span: list[int] = Field(min_length=2, max_length=2)
    diagnostics: dict[str, JsonValue] | None
    finish_reason: str | None
    truncated: bool
    tools_declared: bool
    source: Literal["northgate"]


def validate_turn_annotations(annotations: list[dict[str, JsonValue]]) -> None:
    """Validate annotation shape and ownership without expanding token buffers."""
    previous_stop = 1
    identities: set[str] = set()
    for value in annotations:
        annotation = TurnAnnotation.model_validate(value)
        start, stop = annotation.response_span
        if not 1 <= start < stop or start < previous_stop:
            raise ValueError("turn_annotations spans must be nonempty, ordered and non-overlapping")
        if not annotation.turn_id.strip() or annotation.turn_id in identities:
            raise ValueError("turn_annotations require unique, nonempty turn_id values")
        previous_stop = stop
        identities.add(annotation.turn_id)


class RemoteSegment(BaseModel):
    """One token-faithful training segment of a complete rollout produced inside E2B.

    ``log_probs`` preserves Weaver's response log probabilities, while
    ``old_log_probs`` is the behavior-policy denominator used for training.
    Both are token-aligned and use ``0.0`` for context positions.
    """

    model_config = _MODEL_CONFIG

    tokens: list[int] = Field(min_length=1)
    loss_mask: list[int]
    log_probs: list[float]
    old_log_probs: list[float]
    sampling_mask: list[list[int]] | dict[str, JsonValue] | None = None
    segment_metrics: dict[str, int] = Field(default_factory=dict)
    turn_annotations: list[dict[str, JsonValue]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_alignment(self) -> Self:
        token_count = len(self.tokens)
        if (
            len(self.loss_mask) != token_count
            or len(self.log_probs) != token_count
            or len(self.old_log_probs) != token_count
        ):
            raise ValueError(
                "tokens, loss_mask, log_probs, and old_log_probs must have equal length"
            )
        if any(token < 0 for token in self.tokens):
            raise ValueError("tokens must contain non-negative token IDs")
        if any(value not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss_mask must contain only 0 or 1")
        if self.loss_mask[0] != 0:
            raise ValueError("loss_mask[0] must be 0; the first token has no target logprob")
        if not any(self.loss_mask):
            raise ValueError("a segment must contain at least one response token")
        validate_turn_annotations(self.turn_annotations)
        for annotation in self.turn_annotations:
            start, stop = cast(list[int], annotation["response_span"])
            if stop > token_count or not all(self.loss_mask[start:stop]):
                raise ValueError("turn_annotations spans must contain only response tokens")
        for name, log_probs in (
            ("log_probs", self.log_probs),
            ("old_log_probs", self.old_log_probs),
        ):
            if log_probs[0] != 0.0:
                raise ValueError(f"{name}[0] must be the 0.0 placeholder")
            if any(
                log_prob != 0.0
                for selected, log_prob in zip(self.loss_mask, log_probs)
                if selected == 0
            ):
                raise ValueError(f"context token {name} must use the 0.0 placeholder")

        if self.sampling_mask is None:
            return self

        if isinstance(self.sampling_mask, dict):
            from northrl.utils.sampling_mask_refs import align_sampling_mask_ref

            align_sampling_mask_ref(self.tokens, self.loss_mask, self.sampling_mask)
            return self

        sampled_tokens = [
            token for token, selected in zip(self.tokens, self.loss_mask) if selected == 1
        ]
        if len(self.sampling_mask) != len(sampled_tokens):
            raise ValueError("sampling_mask must have one entry for each response token")
        for index, (candidates, sampled_token) in enumerate(
            zip(self.sampling_mask, sampled_tokens)
        ):
            if not candidates:
                raise ValueError(f"sampling_mask[{index}] must not be empty")
            if any(candidate < 0 for candidate in candidates):
                raise ValueError(f"sampling_mask[{index}] must contain non-negative token IDs")
            if sampled_token not in candidates:
                raise ValueError(
                    f"sampling_mask[{index}] must contain sampled token {sampled_token}"
                )
        return self


class RemoteRolloutTrace(BaseModel):
    """Partial rollout state checkpointed while an E2B harness is still running."""

    model_config = _MODEL_CONFIG

    rollout_id: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    segments: list[RemoteSegment] = Field(
        min_length=1, validation_alias=AliasChoices("segments", "trajectories")
    )
    message_format: Literal["openai", "anthropic"] | None = None
    message_system: JsonValue | None = None
    tools: list[dict[str, JsonValue]] | None = None
    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    _rollout_id_must_not_be_empty = field_validator("rollout_id")(_validate_rollout_id)


class AgentResult(BaseModel):
    """Task output returned by ``run_agent`` and passed to ``evaluate``.

    The platform saves this value in the completed agent-stage record so the
    evaluator can consume it in a separate process in the same sandbox. Files
    modified in the workspace remain available directly; this object carries
    only the execution status, optional answer text, and artifact references.
    Model traces and training tokens are recorded separately by the runtime.

    Returning ``None`` from ``run_agent`` is equivalent to ``AgentResult()``.
    Tasks evaluated from their workspace, such as SWE, can leave both optional
    output fields empty. The evaluator returns the task's reward separately in
    ``EvaluationResult``.
    """

    # Use the shared strict field types and reject undeclared fields.
    model_config = _MODEL_CONFIG

    # Agent execution status: 0 means normal completion, not a correct answer.
    # Nonzero codes are diagnostics; every completed agent stage is evaluated.
    exit_code: int = 0

    # Optional task-defined answer text for evaluate(), e.g. a generated answer.
    # The platform does not extract it from model messages or score it itself.
    final_answer: str | None = None

    # Logical artifact name -> path relative to workspace, e.g. {"patch": "fix.patch"}.
    # These are references only: evaluate() resolves/reads them as needed. The
    # platform does not validate file paths/existence or copy/upload these files.
    artifacts: dict[str, str] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Task score returned by ``evaluate`` after evaluation assets are uploaded.

    The runtime combines this outcome with the completed agent's trace to build
    ``RemoteRolloutResult``. Both statuses represent evaluated task outcomes;
    ``task_failed`` is a valid zero-reward sample, not an infrastructure failure.
    Evaluator crashes or missing assets should raise ``RolloutEnvironmentError``
    instead of returning a fabricated task score.
    """

    # Require the declared field types and finite numeric values; reject extra fields.
    model_config = _MODEL_CONFIG

    # Required final score for this complete rollout, counted once across its segments.
    # The task defines the scale; there is no built-in [0, 1] restriction.
    reward: float

    # Optional task-defined evaluation measurements, e.g. {"swebench_resolved": 1.0}.
    # Values must be finite numbers. These supplement, rather than determine, reward.
    metrics: dict[str, float] = Field(default_factory=dict)

    # "success" accepts any finite reward, including zero; it need not mean solved.
    # "task_failed" explicitly reports an unsuccessful task and requires reward=0.0.
    # Infrastructure failures are reported by exceptions, not by this status field.
    status: Literal["success", "task_failed"] = "success"

    @model_validator(mode="after")
    def _validate_reward(self) -> Self:
        if self.status == "task_failed" and self.reward != 0.0:
            raise ValueError("a failed evaluation must have reward 0.0")
        return self


class AgentStageResult(BaseModel):
    """Framework-owned handoff from the agent process to the evaluation process.

    The runtime writes ``agent-stage.json.gz`` after agent cleanup and model-runtime
    shutdown. The executor validates the rollout/attempt identities before it
    uploads hidden evaluation assets; the evaluator process then reads the same
    record to recover the workspace, task output, and training trace.
    Task hooks return ``AgentResult`` or ``EvaluationResult``, not this record.

    This marks agent cleanup completion only: evaluation has not produced a score
    yet. Any exit code is allowed, including failures before a model turn was
    recorded. The evaluator scores the available workspace and task output.
    """

    # Use the shared strict contract validation and reject undeclared fields.
    model_config = _MODEL_CONFIG

    # Required logical execution ID from RolloutInput; unchanged across environment retries.
    rollout_id: str

    # Required ID generated by the executor for this individual attempt. Each retry
    # gets a new ID so a stale stage record cannot authorize evaluation assets.
    attempt_id: str

    # Fixed commit marker, not a mutable progress/status field or an evaluation result.
    phase: Literal["agent_completed"] = "agent_completed"

    # Required absolute workspace path inside the sandbox, reused by evaluate().
    # The runtime also checks that this directory exists and matches the agent stage.
    workspace: str

    # Required task output returned by run_agent(), or AgentResult() if it returned None.
    agent_result: AgentResult

    # Captured segments, model transcript, and provenance, not the final task score.
    # The trace can be absent when no model turn was recorded. A present trace
    # must belong to the same rollout_id.
    trace: RemoteRolloutTrace | None = None

    # Runtime measurements carried into evaluation when no trace was produced,
    # e.g. an agent stop before the first recorded model result. With a
    # trace, the runtime stores them in trace.metrics and leaves this mapping empty.
    # These are agent-stage counters, not task-owned evaluation scores.
    metrics: dict[str, float] = Field(default_factory=dict)

    # None for ordinary completion; context_length_exceeded stops generation while
    # still allowing evaluation of the current workspace. It is not a task verdict.
    termination_reason: Literal["context_length_exceeded"] | None = None

    # Token-budget details for a context-limit stop, otherwise an empty mapping:
    # prompt_tokens = input length, requested_max_tokens = requested response budget,
    # max_sequence_length = applicable total context limit. The model validates
    # string keys/integer values; it does not require this exact set of keys.
    length_details: dict[str, int] = Field(default_factory=dict)

    _rollout_id_must_not_be_empty = field_validator("rollout_id")(_validate_rollout_id)

    @field_validator("attempt_id")
    @classmethod
    def _validate_attempt_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("attempt_id must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _validate_completion(self) -> Self:
        if self.trace is not None and self.trace.rollout_id != self.rollout_id:
            raise ValueError("agent stage trace rollout_id must match rollout_id")
        if not Path(self.workspace).is_absolute():
            raise ValueError("agent stage workspace must be an absolute path")
        return self


class RemoteRolloutResult(BaseModel):
    """Completed rollout or classified failure returned to the Driver."""

    model_config = _MODEL_CONFIG

    rollout_id: str
    status: Literal["success", "task_failed", "env_error"] = "success"
    error_type: str | None = None
    error_message: str | None = None
    termination_reason: Literal["context_length_exceeded"] | None = None
    length_details: dict[str, int] = Field(default_factory=dict)
    segments: list[RemoteSegment] = Field(
        default_factory=list, validation_alias=AliasChoices("segments", "trajectories")
    )
    message_format: Literal["openai", "anthropic"] | None = None
    message_system: JsonValue | None = None
    tools: list[dict[str, JsonValue]] | None = None
    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)

    _rollout_id_must_not_be_empty = field_validator("rollout_id")(_validate_rollout_id)

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if self.status != "success" and self.reward != 0.0:
            raise ValueError("a failed rollout must have reward 0.0")
        return self
