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

"""Tests for trajectory recording."""

from copy import deepcopy

import pytest

from northrl.remote_rollout import TrajectoryRecorder


def _record_first_turn(
    recorder: TrajectoryRecorder,
    *,
    sampling_mask=None,
) -> None:
    recorder.record_model_turn(
        prompt_tokens=[10, 11],
        response_tokens=[20, 21],
        response_log_probs=[-0.1, -0.15],
        response_old_log_probs=[-0.2, -0.3],
        sampling_mask=sampling_mask,
    )


def test_records_context_responses_and_tool_results():
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)

    recorder.record_model_turn(
        prompt_tokens=[10, 11, 20, 21, 30, 31],
        response_tokens=[40],
        response_log_probs=[-0.35],
        response_old_log_probs=[-0.4],
    )

    [trajectory] = recorder.to_remote_segments()
    assert trajectory.tokens == [10, 11, 20, 21, 30, 31, 40]
    assert trajectory.loss_mask == [0, 0, 1, 1, 0, 0, 1]
    assert trajectory.log_probs == [0.0, 0.0, -0.1, -0.15, 0.0, 0.0, -0.35]
    assert trajectory.old_log_probs == [0.0, 0.0, -0.2, -0.3, 0.0, 0.0, -0.4]
    assert trajectory.sampling_mask is None


def test_records_response_aligned_sampling_masks():
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder, sampling_mask=[[20, 22], [21]])

    recorder.record_model_turn(
        prompt_tokens=[10, 11, 20, 21, 30],
        response_tokens=[40],
        response_log_probs=[-0.35],
        response_old_log_probs=[-0.4],
        sampling_mask=[[40, 41]],
    )

    assert recorder.to_remote_segments()[0].sampling_mask == [
        [20, 22],
        [21],
        [40, 41],
    ]


@pytest.mark.parametrize(
    "prompt_tokens",
    [
        [10, 11, 99, 21],
        [10, 11],
    ],
)
def test_starts_new_segment_when_prompt_does_not_continue(prompt_tokens):
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)

    recorder.record_model_turn(
        prompt_tokens=prompt_tokens,
        response_tokens=[40],
        response_log_probs=[-0.35],
        response_old_log_probs=[-0.4],
        sampling_mask=[[40]],
    )

    first, second = recorder.to_remote_segments()
    assert first.tokens == [10, 11, 20, 21]
    assert second.tokens == [*prompt_tokens, 40]
    assert second.loss_mask == [*([0] * len(prompt_tokens)), 1]
    assert second.log_probs == [*([0.0] * len(prompt_tokens)), -0.35]
    assert second.old_log_probs == [*([0.0] * len(prompt_tokens)), -0.4]
    assert second.sampling_mask == [[40]]


def test_rejects_misaligned_response_without_mutating_trace():
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)

    with pytest.raises(ValueError, match="must have equal length"):
        recorder.record_model_turn(
            prompt_tokens=[10, 11, 20, 21],
            response_tokens=[40, 41],
            response_log_probs=[-0.3, -0.35],
            response_old_log_probs=[-0.4],
        )

    assert recorder.to_remote_segments()[0].tokens == [10, 11, 20, 21]


@pytest.mark.parametrize(
    ("first_mask", "second_mask"),
    [
        (None, [[40]]),
        ([[20], [21]], None),
    ],
)
def test_requires_sampling_mask_consistently(first_mask, second_mask):
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder, sampling_mask=first_mask)

    with pytest.raises(ValueError, match="every model turn or none"):
        recorder.record_model_turn(
            prompt_tokens=[10, 11, 20, 21, 30],
            response_tokens=[40],
            response_log_probs=[-0.35],
            response_old_log_probs=[-0.4],
            sampling_mask=second_mask,
        )

    assert recorder.to_remote_segments()[0].tokens == [10, 11, 20, 21]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prompt_tokens": []}, "prompt_tokens must not be empty"),
        ({"response_tokens": []}, "response_tokens must not be empty"),
        ({"response_tokens": [-1]}, "non-negative token IDs"),
        ({"response_log_probs": []}, "equal length"),
        ({"response_log_probs": [float("nan")]}, "finite number"),
        ({"response_old_log_probs": [float("nan")]}, "finite number"),
        ({"sampling_mask": [[21]]}, "must contain sampled token 20"),
    ],
)
def test_rejects_invalid_turn_data(overrides, message):
    values = {
        "prompt_tokens": [10],
        "response_tokens": [20],
        "response_log_probs": [-0.1],
        "response_old_log_probs": [-0.2],
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        TrajectoryRecorder().record_model_turn(**values)


def test_requires_a_model_turn_before_export():
    with pytest.raises(ValueError, match="before recording a model turn"):
        TrajectoryRecorder().to_remote_segments()


def test_notifies_without_decorating_provider_transcript():
    snapshots = []

    def snapshot(current: TrajectoryRecorder) -> None:
        snapshots.append(
            (
                current.message_format,
                current.to_messages(),
                current.to_remote_segments()[0].tokens,
            )
        )

    recorder = TrajectoryRecorder(on_update=snapshot)
    recorder.record_inference_result(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "working",
                        "reasoning_content": "private reasoning",
                    }
                }
            ],
            "northrl_train": {
                "prompt_tokens": [10],
                "response_tokens": [20],
                "response_logprobs": [-0.1],
            },
        },
        message_format="openai",
        messages=[{"role": "assistant", "content": "working"}],
    )

    assert snapshots == [("openai", [{"role": "assistant", "content": "working"}], [10, 20])]


def test_tool_definitions_are_copied_with_the_latest_successful_transcript():
    recorder = TrajectoryRecorder()
    result = {
        "northrl_train": {
            "prompt_tokens": [10],
            "response_tokens": [20],
            "response_logprobs": [-0.1],
        }
    }
    tools = [
        {
            "name": "shell",
            "description": "Execute commands.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        }
    ]
    expected_tools = deepcopy(tools)
    messages = [{"role": "user", "content": "first request"}]
    recorder.record_inference_result(
        result,
        message_format="anthropic",
        message_system="first system",
        tools=tools,
        messages=messages,
    )

    tools[0]["input_schema"]["properties"]["command"]["type"] = "number"
    exported_tools = recorder.tools
    exported_tools[0]["description"] = "changed"
    assert recorder.tools == expected_tools

    invalid_result = deepcopy(result)
    invalid_result["northrl_train"]["response_logprobs"] = []
    with pytest.raises(ValueError, match="equal length"):
        recorder.record_inference_result(
            invalid_result,
            message_format="anthropic",
            message_system="failed system",
            tools=[],
            messages=[{"role": "user", "content": "failed request"}],
        )
    assert recorder.tools == expected_tools
    assert recorder.message_system == "first system"
    assert recorder.to_messages() == messages

    for latest_tools in ([], None):
        latest_messages = [{"role": "user", "content": f"request with {latest_tools!r}"}]
        recorder.record_inference_result(
            result,
            message_format="anthropic",
            message_system="latest system",
            tools=latest_tools,
            messages=latest_messages,
        )
        assert recorder.tools == latest_tools
        assert recorder.message_system == "latest system"
        assert recorder.to_messages() == latest_messages


def test_legacy_export_method_keeps_the_segment_copy_contract():
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)

    legacy_segments = recorder.to_remote_trajectories()
    segments = recorder.to_remote_segments()

    assert legacy_segments == segments
    assert legacy_segments[0] is not segments[0]
    legacy_segments[0].tokens.append(99)
    assert recorder.to_remote_segments() == segments


def test_compact_export_caches_finalized_segments_and_refreshes_only_changed_turn(monkeypatch):
    from northrl.remote_rollout import trajectory_recorder
    from northrl.remote_rollout.wire import decode_segments

    encoded = []
    encode_segment = trajectory_recorder.encode_segment

    def record_encoding(segment, previous_segment=None, *, base_segment=None):
        encoded.append(segment)
        return encode_segment(segment, previous_segment, base_segment=base_segment)

    monkeypatch.setattr(trajectory_recorder, "encode_segment", record_encoding)
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)
    first_snapshot = recorder.to_compact_segments()
    assert len(encoded) == 1

    recorder.record_model_turn(
        prompt_tokens=[10, 11, 99],
        response_tokens=[40],
        response_log_probs=[-0.3],
        response_old_log_probs=[-0.4],
    )
    second_snapshot = recorder.to_compact_segments()
    assert len(encoded) == 2
    assert second_snapshot[0] is first_snapshot[0]
    assert decode_segments(second_snapshot) == recorder.to_remote_segments()

    recorder.record_model_turn(
        prompt_tokens=[10, 11, 99, 40, 50],
        response_tokens=[60],
        response_log_probs=[-0.5],
        response_old_log_probs=[-0.6],
    )
    third_snapshot = recorder.to_compact_segments()
    assert len(encoded) == 3
    assert third_snapshot[0] is first_snapshot[0]
    assert third_snapshot[1] is not second_snapshot[1]
    assert decode_segments(second_snapshot)[1].tokens == [10, 11, 99, 40]
    assert decode_segments(third_snapshot) == recorder.to_remote_segments()
    assert recorder.to_compact_segments() == third_snapshot
    assert len(encoded) == 3


def test_compact_export_keeps_response_spans_independent_of_shared_tokens():
    from northrl.remote_rollout.wire import decode_segments

    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)
    recorder.record_model_turn(
        prompt_tokens=[10, 11, 20],
        response_tokens=[21],
        response_log_probs=[-0.5],
        response_old_log_probs=[-0.5],
    )
    compact = recorder.to_compact_segments()
    assert compact[0].response_spans == [[2, 4]]
    assert compact[1].response_spans == [[3, 4]]
    assert decode_segments(compact) == recorder.to_remote_segments()


def test_compact_export_refreshes_late_metrics_without_reencoding(monkeypatch):
    from northrl.remote_rollout import trajectory_recorder
    from northrl.remote_rollout.wire import decode_segments

    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)
    before = recorder.to_compact_segments()
    observations = {"tool_calls": 1, "tool_results": 0}

    class Metrics:
        def segment_snapshot(self, _index):
            return dict(observations)

    def fail_encoding(*_args, **_kwargs):
        raise AssertionError("unchanged token arrays must not be encoded again")

    monkeypatch.setattr(trajectory_recorder, "encode_segment", fail_encoding)
    recorder.set_metrics(Metrics())
    initial = recorder.to_compact_segments()
    observations["tool_results"] = 1
    updated = recorder.to_compact_segments()
    assert before[0].segment_metrics == {}
    assert initial[0].segment_metrics["tool_results"] == 0
    assert updated[0].segment_metrics["tool_results"] == 1
    assert decode_segments(updated) == recorder.to_remote_segments()


def test_turn_annotations_preserve_unknown_evidence_and_own_only_generated_tokens():
    from northrl.remote_rollout.wire import decode_segments

    recorder = TrajectoryRecorder()
    diagnostics = {
        "checked": False,
        "malformed": True,
        "detected": None,
        "invalid": None,
        "reasons": ["invalid_arguments_json"],
    }
    recorder.record_inference_result(
        {
            "choices": [{"finish_reason": "tool_calls"}],
            "tool_call_diagnostics": diagnostics,
            "northrl_train": {
                "prompt_tokens": [10, 11],
                "response_tokens": [20, 21],
                "response_logprobs": [-0.1, -0.2],
            },
        },
        message_format="openai",
        tools=[],
    )
    before = recorder.to_compact_segments()
    diagnostics["reasons"].clear()
    recorder.record_inference_result(
        {
            "stop_reason": "max_tokens",
            "northrl_train": {
                "prompt_tokens": [10, 11, 20, 21, 30],
                "response_tokens": [40, 41],
                "response_logprobs": [-0.3, -0.4],
            },
        },
        message_format="anthropic",
    )
    # A compacted history retains old response tokens only as context.
    recorder.record_model_turn(
        prompt_tokens=[10, 20, 21],
        response_tokens=[50],
        response_log_probs=[-0.5],
        response_old_log_probs=[-0.5],
    )
    compact = recorder.to_compact_segments()
    first, second = decode_segments(compact)
    assert [item["turn_id"] for item in first.turn_annotations] == ["turn-0", "turn-1"]
    assert [item["response_span"] for item in first.turn_annotations] == [[2, 4], [5, 7]]
    assert first.turn_annotations[0] == {
        "version": 1,
        "turn_id": "turn-0",
        "response_span": [2, 4],
        "diagnostics": {
            "checked": False,
            "malformed": True,
            "detected": None,
            "invalid": None,
            "reasons": ["invalid_arguments_json"],
        },
        "finish_reason": "tool_calls",
        "truncated": False,
        "tools_declared": True,
        "source": "northgate",
    }
    assert first.turn_annotations[1]["diagnostics"] is None
    assert first.turn_annotations[1]["finish_reason"] == "max_tokens"
    assert first.turn_annotations[1]["truncated"] is True
    assert first.turn_annotations[1]["tools_declared"] is False
    assert [item["turn_id"] for item in second.turn_annotations] == ["turn-2"]
    assert second.turn_annotations[0]["response_span"] == [3, 4]
    assert second.turn_annotations[0]["diagnostics"] is None
    assert second.turn_annotations[0]["finish_reason"] is None
    assert len(before[0].turn_annotations) == 1
    assert decode_segments(compact, packed=True)[0].turn_annotations == first.turn_annotations
    first.turn_annotations[0]["diagnostics"]["reasons"].clear()
    assert recorder.to_remote_segments()[0].turn_annotations[0]["diagnostics"]["reasons"]


def test_adjacent_turns_remain_distinct_when_compact_response_spans_merge():
    recorder = TrajectoryRecorder()
    _record_first_turn(recorder)
    recorder.record_model_turn(
        prompt_tokens=[10, 11, 20, 21],
        response_tokens=[40],
        response_log_probs=[-0.3],
        response_old_log_probs=[-0.4],
    )
    [segment] = recorder.to_compact_segments()
    assert segment.response_spans == [[2, 5]]
    assert [annotation["response_span"] for annotation in segment.turn_annotations] == [
        [2, 4],
        [4, 5],
    ]


def test_compact_export_requires_a_model_turn():
    with pytest.raises(ValueError, match="before recording a model turn"):
        TrajectoryRecorder().to_compact_segments()
