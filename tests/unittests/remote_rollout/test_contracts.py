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

import json

import pytest
from pydantic import ValidationError

from northrl.remote_rollout import (
    RemoteRolloutResult,
    RemoteRolloutTrace,
    RemoteSegment,
    RolloutInput,
)


def _segment(**overrides) -> RemoteSegment:
    values = {
        "tokens": [10, 20, 30],
        "loss_mask": [0, 1, 1],
        "log_probs": [0.0, -0.1, -0.2],
        "old_log_probs": [0.0, -0.2, -0.3],
    }
    values.update(overrides)
    return RemoteSegment(**values)


def test_rollout_input_is_json_round_trip_safe():
    input_data = RolloutInput(
        rollout_id="rollout-1",
        task={"problem": "fix the bug", "attempt": 1, "flags": [True, None]},
        metadata={"group_id": "group-1", "run_id": 0, "weight_version": "v1"},
    )

    assert RolloutInput.model_validate_json(input_data.model_dump_json()) == input_data


def test_legacy_input_and_trace_accept_missing_metadata():
    input_data = RolloutInput.model_validate({"rollout_id": "rollout-1", "task": {}})
    trace = RemoteRolloutTrace.model_validate(
        {"rollout_id": "rollout-1", "segments": [_segment().model_dump()]}
    )

    assert input_data.metadata == {}
    assert trace.metadata == {}
    assert trace.tools is None


@pytest.mark.parametrize(
    "task",
    [
        {"not_json": (1, 2)},
        {"not_json": float("nan")},
        {1: "not a JSON object key"},
    ],
)
def test_rollout_input_rejects_non_json_task_values(task):
    with pytest.raises(ValidationError):
        RolloutInput(rollout_id="rollout-1", task=task)


def test_segment_accepts_response_aligned_sampling_mask():
    trajectory = _segment(sampling_mask=[[20, 21], [30]])

    assert trajectory.sampling_mask == [[20, 21], [30]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"response_span": [0, 2]},
        {"response_span": [2, 2]},
        {"response_span": [2, 4]},
        {"response_span": [True, 3]},
        {"response_span": [1, 2, 3]},
        {"turn_id": " "},
        {"source": "metrics"},
        {"version": 2},
        {"truncated": 1},
    ],
)
def test_segment_rejects_invalid_turn_annotations(overrides):
    annotation = {
        "version": 1,
        "turn_id": "turn-0",
        "response_span": [1, 3],
        "diagnostics": None,
        "finish_reason": None,
        "truncated": False,
        "tools_declared": False,
        "source": "northgate",
        **overrides,
    }
    with pytest.raises(ValidationError):
        _segment(turn_annotations=[annotation])


def test_legacy_segment_has_no_invented_turn_annotations():
    assert _segment().turn_annotations == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loss_mask": [0, 1]}, "equal length"),
        ({"log_probs": [0.0, -0.1]}, "equal length"),
        ({"old_log_probs": [0.0, -0.2]}, "equal length"),
        ({"loss_mask": [1, 1, 1]}, r"loss_mask\[0\] must be 0"),
        ({"loss_mask": [0, 0, 0]}, "at least one response token"),
        ({"log_probs": [-0.1, -0.2, -0.3]}, r"log_probs\[0\]"),
        ({"old_log_probs": [-0.1, -0.2, -0.3]}, r"old_log_probs\[0\]"),
        (
            {
                "loss_mask": [0, 0, 1],
                "log_probs": [0.0, -0.1, -0.2],
                "old_log_probs": [0.0, 0.0, -0.3],
            },
            "context token log_probs",
        ),
        (
            {
                "loss_mask": [0, 0, 1],
                "log_probs": [0.0, 0.0, -0.2],
                "old_log_probs": [0.0, -0.2, -0.3],
            },
            "context token old_log_probs",
        ),
        ({"sampling_mask": [[20]]}, "one entry for each response token"),
        ({"sampling_mask": [[21], [30]]}, "must contain sampled token 20"),
    ],
)
def test_segment_rejects_invalid_alignment(overrides, message):
    with pytest.raises(ValidationError, match=message):
        _segment(**overrides)


def test_result_supports_multiple_segments_and_json_round_trip():
    result = RemoteRolloutResult(
        rollout_id="rollout-1",
        segments=[
            _segment(),
            _segment(
                tokens=[40, 50],
                loss_mask=[0, 1],
                log_probs=[0.0, -0.3],
                old_log_probs=[0.0, -0.4],
            ),
        ],
        message_format="anthropic",
        message_system="Follow the instructions.",
        messages=[
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        reward=1,
        metrics={"tests_passed": 2},
    )

    restored = RemoteRolloutResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert len(restored.segments) == 2


def test_partial_trace_is_json_round_trip_safe():
    trace = RemoteRolloutTrace(
        rollout_id="rollout-1",
        segments=[_segment()],
        metadata={"group_id": "group-1", "task_id": "task-1", "run_id": 2},
        message_format="openai",
        messages=[{"role": "assistant", "content": "still working"}],
    )
    assert RemoteRolloutTrace.model_validate_json(trace.model_dump_json()) == trace


@pytest.mark.parametrize("model_class", [RemoteRolloutTrace, RemoteRolloutResult])
@pytest.mark.parametrize(
    "tools",
    [
        None,
        [],
        [
            {
                "name": "shell",
                "description": "Execute shell commands.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                "cache_control": {"type": "ephemeral"},
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Execute shell commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                    "strict": True,
                },
            }
        ],
    ],
)
def test_native_tool_definitions_are_json_round_trip_safe(model_class, tools):
    payload = {"rollout_id": "rollout-1", "segments": [_segment()], "tools": tools}
    if model_class is RemoteRolloutResult:
        payload["reward"] = 1.0
    value = model_class(**payload)

    restored = model_class.model_validate_json(value.model_dump_json())

    assert restored.tools == tools
    assert restored == value


@pytest.mark.parametrize("segments", [None, []])
def test_successful_result_accepts_no_recorded_segments(segments):
    values = {"rollout_id": "rollout-1", "reward": 1.0}
    if segments is not None:
        values["segments"] = segments
    result = RemoteRolloutResult(**values)
    assert result.segments == []
    assert result.reward == 1.0


@pytest.mark.parametrize("status", ["task_failed", "env_error"])
@pytest.mark.parametrize("recorded_segment", [False, True])
def test_failure_result_preserves_error_and_optional_segments(status, recorded_segment):
    result = RemoteRolloutResult(
        rollout_id="rollout-1",
        status=status,
        reward=0.0,
        segments=[_segment()] if recorded_segment else [],
        error_type="RuntimeError",
        error_message="operation failed",
    )

    restored = RemoteRolloutResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.status == status
    assert restored.error_message == "operation failed"


@pytest.mark.parametrize("status", ["task_failed", "env_error"])
def test_failure_result_defaults_to_no_segments(status):
    result = RemoteRolloutResult(rollout_id="rollout-1", status=status, reward=0.0)

    assert result.segments == []


@pytest.mark.parametrize("status", ["task_failed", "env_error"])
@pytest.mark.parametrize("reward", [-1.0, 1.0])
def test_failure_result_rejects_nonzero_reward(status, reward):
    with pytest.raises(ValidationError, match="must have reward 0.0"):
        RemoteRolloutResult(
            rollout_id="rollout-1", status=status, segments=[_segment()], reward=reward
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0"])
def test_result_rejects_invalid_reward(value):
    with pytest.raises(ValidationError):
        RemoteRolloutResult(
            rollout_id="rollout-1",
            segments=[_segment()],
            reward=value,
        )


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RolloutInput(
            rollout_id="rollout-1",
            task={},
            provider="openai",
        )


@pytest.mark.parametrize("exit_code", [0, 1, 124])
@pytest.mark.parametrize("has_trace", [True, False])
def test_agent_stage_round_trip_preserves_exit_status_and_optional_trace(
    tmp_path, exit_code, has_trace
):
    from northrl.remote_rollout.contracts import AgentResult, AgentStageResult

    stage = AgentStageResult(
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        workspace=str(tmp_path),
        agent_result=AgentResult(
            exit_code=exit_code, final_answer="finished", artifacts={"patch": "answer.patch"}
        ),
        trace=(
            RemoteRolloutTrace(rollout_id="rollout-1", segments=[_segment()]) if has_trace else None
        ),
    )
    assert AgentStageResult.model_validate_json(stage.model_dump_json()) == stage
    for updates in (
        {"attempt_id": " "},
        {"rollout_id": " "},
        {"workspace": "relative/workspace"},
        {"trace": {"rollout_id": "another-rollout", "segments": [_segment().model_dump()]}},
        {"phase": "agent"},
    ):
        with pytest.raises(ValidationError):
            AgentStageResult.model_validate({**stage.model_dump(), **updates})


def test_agent_output_and_evaluation_contracts_are_strict():
    from northrl.remote_rollout.contracts import AgentResult, EvaluationResult

    assert AgentResult().exit_code == 0
    assert EvaluationResult(reward=0.25).reward == 0.25
    assert EvaluationResult(reward=0.0, status="task_failed").status == "task_failed"
    for values in ({"exit_code": True}, {"artifacts": {"answer": 3}}, {"model_path": "secret"}):
        with pytest.raises(ValidationError):
            AgentResult(**values)
    for values in (
        {"reward": 0.5, "status": "task_failed"},
        {"reward": float("nan")},
        {"reward": 0.0, "status": "env_error"},
        {"reward": 0.0, "metrics": {"score": float("inf")}},
    ):
        with pytest.raises(ValidationError):
            EvaluationResult(**values)


def test_context_stop_without_model_turns_can_be_evaluated(tmp_path):
    from northrl.remote_rollout.contracts import AgentResult, AgentStageResult

    details = {"prompt_tokens": 262144, "requested_max_tokens": 8192, "max_sequence_length": 262144}
    stage = AgentStageResult(
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        workspace=str(tmp_path),
        agent_result=AgentResult(),
        trace=None,
        termination_reason="context_length_exceeded",
        length_details=details,
    )
    assert AgentStageResult.model_validate_json(stage.model_dump_json()) == stage
    result = RemoteRolloutResult(
        rollout_id="rollout-1",
        reward=1.0,
        termination_reason=stage.termination_reason,
        length_details=stage.length_details,
    )
    assert RemoteRolloutResult.model_validate_json(result.model_dump_json()).reward == 1.0
    assert result.segments == []
    assert (
        AgentStageResult.model_validate({**stage.model_dump(), "termination_reason": None}).trace
        is None
    )


@pytest.mark.parametrize("model_class", [RemoteRolloutTrace, RemoteRolloutResult])
def test_legacy_json_is_read_but_only_segments_are_written(model_class):
    payload = {
        "rollout_id": "rollout-1",
        "trajectories": [_segment().model_dump()],
        "message_format": "openai",
        "messages": [{"role": "assistant", "content": "done"}],
    }
    if model_class is RemoteRolloutResult:
        payload.update(reward=0.5, metrics={"tests_passed": 2.0})

    result = model_class.model_validate_json(json.dumps(payload))
    expected = {
        **payload,
        "segments": payload["trajectories"],
        "message_system": None,
        "tools": None,
    }
    del expected["trajectories"]
    if model_class is RemoteRolloutTrace:
        expected.update(metadata={}, metrics={})
    else:
        expected.update(
            status="success",
            error_type=None,
            error_message=None,
            termination_reason=None,
            length_details={},
        )

    assert isinstance(result.segments[0], RemoteSegment)
    assert result.tools is None
    assert result.model_dump() == expected
    assert json.loads(result.model_dump_json()) == expected
    assert result.model_dump(by_alias=True) == expected
    assert model_class.model_validate_json(result.model_dump_json()) == result
    assert not hasattr(result, "trajectories")


@pytest.mark.parametrize("model_class", [RemoteRolloutTrace, RemoteRolloutResult])
@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_remote_schema_exposes_only_the_canonical_segments_field(model_class, mode):
    schema = model_class.model_json_schema(mode=mode)

    assert "segments" in schema["properties"]
    if model_class is RemoteRolloutTrace:
        assert "segments" in schema["required"]
        assert schema["properties"]["segments"]["minItems"] == 1
    else:
        # Evaluation can complete without recorded model tokens.
        assert "segments" not in schema["required"]
    assert "trajectories" not in schema["properties"]
    assert "trajectories" not in schema["required"]
    assert "segments" in model_class.model_fields
    assert "trajectories" not in model_class.model_fields


@pytest.mark.parametrize("model_class", [RemoteRolloutTrace, RemoteRolloutResult])
@pytest.mark.parametrize("same_values", [False, True])
def test_remote_contract_rejects_both_sequence_field_names(model_class, same_values):
    canonical = [_segment().model_dump()]
    legacy = canonical if same_values else [_segment(), _segment()]
    legacy = [value.model_dump() if isinstance(value, RemoteSegment) else value for value in legacy]
    payload = {"rollout_id": "rollout-1", "segments": canonical, "trajectories": legacy}
    if model_class is RemoteRolloutResult:
        payload["reward"] = 0.5

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_class.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("field", ["segments", "trajectories"])
def test_trace_requires_a_segment(field):
    payload = {"rollout_id": "rollout-1", field: []}
    with pytest.raises(ValidationError):
        RemoteRolloutTrace.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("field", ["segments", "trajectories"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_evaluated_result_can_preserve_reward_without_model_tokens(field, reward):
    result = RemoteRolloutResult.model_validate_json(
        json.dumps({"rollout_id": "rollout-1", field: [], "reward": reward})
    )
    assert result.segments == []
    assert result.status == "success"
    assert result.reward == reward
