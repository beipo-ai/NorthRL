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

import gzip
import json
import math
from array import array

import pytest
from pydantic import ValidationError

from northrl.remote_rollout import wire
from northrl.remote_rollout.contracts import (
    AgentResult,
    AgentStageResult,
    RemoteRolloutResult,
    RemoteRolloutTrace,
    RemoteSegment,
)
from northrl.remote_rollout.wire import (
    CompactRolloutTrace,
    ResponseLogProbsRef,
    decode_agent_stage,
    decode_result,
    decode_segments,
    decode_trace,
    encode_agent_stage,
    encode_result,
    encode_segment,
    encode_segments,
    encode_trace,
    load_agent_stage,
    load_result,
    load_trace,
)


def _segment(tokens, positions, *, old_different=False, sampling_mask=None):
    mask = [int(index in positions) for index in range(len(tokens))]
    probabilities = [-(index + 1) / 10 if selected else 0.0 for index, selected in enumerate(mask)]
    old = [
        value - 0.25 if old_different and selected else value
        for value, selected in zip(probabilities, mask)
    ]
    return RemoteSegment(
        tokens=tokens,
        loss_mask=mask,
        log_probs=probabilities,
        old_log_probs=old,
        sampling_mask=sampling_mask,
        segment_metrics={"model_turns": 2},
    )


def _trace():
    return RemoteRolloutTrace(
        rollout_id="rollout-1",
        metadata={"task_id": "task-1", "group_id": "group-1", "run_id": 3},
        segments=[
            _segment([10, 11, 12, 13, 14], {2, 3, 4}, sampling_mask=[[12, 90], [13], [14]]),
            # Previously trained prefix tokens are context in this new segment.
            _segment([10, 11, 12, 13, 14, 15, 16, 17], {6, 7}, old_different=True),
            # One segment can contain several response spans separated by context.
            _segment([10, 11, 12, 13, 40, 41, 42, 43], {1, 2, 6}),
        ],
        message_format="anthropic",
        message_system=[{"type": "text", "text": "Be exact."}],
        tools=[{"name": "shell", "input_schema": {"type": "object"}}],
        messages=[{"role": "assistant", "content": [{"type": "text", "text": "done"}]}],
        metrics={"model_turns": 7.0},
    )


def test_trace_round_trip_preserves_supervision_and_all_metadata():
    trace = _trace()
    compact = encode_trace(trace)

    assert compact.schema_version == 2
    assert compact.segments[1].base_segment == 0
    assert compact.segments[1].prefix_len == 5
    assert compact.segments[1].suffix_tokens == [15, 16, 17]
    assert compact.segments[1].response_spans == [[6, 8]]
    assert compact.segments[2].prefix_len == 4
    assert compact.segments[2].response_spans == [[1, 3], [6, 7]]
    assert isinstance(compact.segments[0].old_response_log_probs, ResponseLogProbsRef)
    assert isinstance(compact.segments[1].old_response_log_probs, list)
    assert decode_trace(load_trace(compact.model_dump_json())) == trace
    assert CompactRolloutTrace.model_validate(compact.model_dump(mode="json")) == compact
    assert "loss_mask" not in compact.model_dump_json()


def test_packed_driver_segments_preserve_values_without_python_scalar_storage():
    trace = _trace()
    compact = encode_trace(trace)
    packed = decode_trace(load_trace(compact.model_dump_json()), packed=True)
    assert packed.metrics == trace.metrics
    for actual, expected in zip(packed.segments, trace.segments, strict=True):
        for name in ("tokens", "loss_mask", "log_probs", "old_log_probs"):
            values = getattr(actual, name)
            assert isinstance(values, array)
            assert values.tolist() == getattr(expected, name)
        assert actual.loss_mask.itemsize == 1
        assert actual.log_probs.itemsize == 8
        assert actual.sampling_mask == expected.sampling_mask
        assert actual.segment_metrics == expected.segment_metrics
    # Inherited contexts still own independent buffers.
    packed.segments[1].tokens[0] = 99
    assert packed.segments[0].tokens[0] == trace.segments[0].tokens[0]
    assert compact.segments[0].suffix_tokens[0] == trace.segments[0].tokens[0]


def test_packed_result_preserves_outcome_and_signed_zero():
    segment = RemoteSegment(
        tokens=[1, 2],
        loss_mask=[0, 1],
        log_probs=[0.0, -0.0],
        old_log_probs=[0.0, 0.0],
    )
    source = RemoteRolloutResult(rollout_id="packed", reward=0.25, segments=[segment])
    result = decode_result(encode_result(source), packed=True)
    assert result.status == source.status
    assert result.reward == source.reward
    assert math.copysign(1, result.segments[0].log_probs[1]) == -1
    assert math.copysign(1, result.segments[0].old_log_probs[1]) == 1


def test_same_token_sequence_can_have_different_response_positions():
    segments = [_segment([1, 2, 3, 4], {1, 2}), _segment([1, 2, 3, 4], {3})]
    compact = encode_segments(segments)

    assert compact[1].prefix_len == 4
    assert compact[1].suffix_tokens == []
    assert decode_segments(compact) == segments


@pytest.mark.parametrize("packed", [False, True])
def test_turn_annotations_survive_compact_json_round_trip(packed):
    trace = _trace()
    annotation = {
        "version": 1,
        "turn_id": "turn-1",
        "response_span": [6, 8],
        "diagnostics": {
            "checked": False,
            "malformed": False,
            "detected": None,
            "invalid": None,
            "reasons": [],
        },
        "finish_reason": None,
        "truncated": False,
        "tools_declared": False,
        "source": "northgate",
    }
    trace.segments[1].turn_annotations = [annotation]
    compact = encode_trace(trace)
    restored = decode_trace(load_trace(compact.model_dump_json()), packed=packed)
    assert restored.segments[1].turn_annotations == [annotation]
    assert restored.segments[0].turn_annotations == []
    restored.segments[1].turn_annotations[0]["diagnostics"]["reasons"].append("unknown_reason")
    assert annotation["diagnostics"]["reasons"] == []
    assert compact.segments[1].turn_annotations[0]["diagnostics"]["reasons"] == []


@pytest.mark.parametrize("legacy", [False, True], ids=["annotated-v2", "legacy-v2"])
@pytest.mark.parametrize(
    "artifact,packed",
    [
        ("trace", False),
        ("trace", True),
        ("result", False),
        ("result", True),
        ("agent-stage", False),  # Stage handoff decoding only exposes the list form.
    ],
)
def test_gzip_artifacts_preserve_annotations_and_accept_legacy_v2(artifact, packed, legacy):
    trace = _trace()
    for index, span in enumerate(([2, 5], [6, 8])):
        trace.segments[index].turn_annotations = [
            {
                "version": 1,
                "turn_id": f"turn-{index}",
                "response_span": span,
                "diagnostics": (
                    None
                    if index == 0
                    else {
                        "checked": False,
                        "malformed": True,
                        "detected": None,
                        "invalid": None,
                        "reasons": ["invalid_arguments_json"],
                    }
                ),
                "finish_reason": "tool_calls" if index else None,
                "truncated": index == 0,
                "tools_declared": index == 1,
                "source": "northgate",
            }
        ]
    if artifact == "trace":
        source = trace
        compact = encode_trace(source)
        loader = wire.load_trace_gzip
    elif artifact == "result":
        source = RemoteRolloutResult(
            **trace.model_dump(exclude={"metadata", "segments"}),
            segments=trace.segments,
            reward=0.75,
        )
        compact = encode_result(source)
        loader = wire.load_result_gzip
    else:
        source = AgentStageResult(
            rollout_id=trace.rollout_id,
            attempt_id="attempt-2",
            workspace="/workspace/task",
            agent_result=AgentResult(final_answer="answer"),
            trace=trace,
        )
        compact = encode_agent_stage(source)
        loader = wire.load_agent_stage_gzip

    payload = wire.dump_artifact(compact)
    if legacy:
        # Real old v2 JSON omitted the field; new model defaults must not add it back.
        document = json.loads(gzip.decompress(payload))
        segment_document = document["trace"] if artifact == "agent-stage" else document
        for segment in segment_document["segments"]:
            del segment["turn_annotations"]
        payload = gzip.compress(json.dumps(document).encode("utf-8"))
    loaded = loader(payload)
    assert loaded.schema_version == 2
    if artifact == "agent-stage":
        assert loaded.trace.schema_version == 2
        restored = decode_agent_stage(loaded)
        assert restored.model_dump(exclude={"trace"}) == source.model_dump(exclude={"trace"})
        restored, source = restored.trace, trace
    else:
        decoder = decode_trace if artifact == "trace" else decode_result
        restored = decoder(loaded, packed=packed)
    assert restored.model_dump(exclude={"segments"}) == source.model_dump(exclude={"segments"})
    for actual, expected in zip(restored.segments, source.segments, strict=True):
        for name in ("tokens", "loss_mask", "log_probs", "old_log_probs"):
            values = getattr(actual, name)
            assert isinstance(values, array if packed else list)
            assert list(values) == getattr(expected, name)
        assert actual.sampling_mask == expected.sampling_mask
        assert actual.segment_metrics == expected.segment_metrics
        assert actual.turn_annotations == ([] if legacy else expected.turn_annotations)


@pytest.mark.parametrize("span", [[0, 1], [2, 5], [5, 7], [7, 9]])
def test_compact_turn_annotations_cannot_claim_inherited_context(span):
    compact = encode_trace(_trace()).model_dump(mode="json")
    compact["segments"][1]["turn_annotations"] = [
        {
            "version": 1,
            "turn_id": "turn-1",
            "response_span": span,
            "diagnostics": None,
            "finish_reason": "stop",
            "truncated": False,
            "tools_declared": False,
            "source": "northgate",
        }
    ]
    with pytest.raises(ValidationError, match="turn_annotations"):
        CompactRolloutTrace.model_validate(compact)


def test_prefix_match_is_exact_and_only_compares_previous_segment():
    segments = [
        _segment([1, 2, 3], {2}),
        _segment([9, 2, 3], {2}),
        _segment([1, 2, 3, 4], {3}),
    ]
    compact = encode_segments(segments)

    assert compact[1].base_segment is None
    assert compact[2].base_segment is None  # No search back to segment 0.
    assert decode_segments(compact) == segments


def test_encoder_accepts_a_shorter_exact_prefix_without_retokenizing():
    segments = [_segment([100, 200, 300, 400], {3}), _segment([100, 200], {1})]
    compact = encode_segments(segments)
    assert compact[1].prefix_len == 2
    assert compact[1].suffix_tokens == []
    assert decode_segments(compact) == segments


def test_probability_reference_preserves_signed_zero_and_differences():
    segment = RemoteSegment(
        tokens=[1, 2],
        loss_mask=[0, 1],
        log_probs=[0.0, -0.0],
        old_log_probs=[0.0, 0.0],
    )
    compact = encode_segment(segment)
    assert isinstance(compact.old_response_log_probs, list)
    restored = decode_segments([compact])[0]
    assert math.copysign(1, restored.log_probs[1]) == -1
    assert math.copysign(1, restored.old_log_probs[1]) == 1


def test_sampling_mask_refs_keep_slices_and_multiple_response_spans():
    sampling_mask = {
        "sampling_mask_refs": [
            {
                "sampling_mask_ref": {"opaque_id": "mask-1", "token_count": 5},
                "sampling_mask_slice": {"start": 2, "stop": 4},
            },
            {"sampling_mask_ref": {"opaque_id": "mask-2", "token_count": 1}},
        ]
    }
    segment = _segment([10, 11, 12, 13, 14, 15], {1, 2, 5}, sampling_mask=sampling_mask)
    compact = encode_segments([segment])
    assert compact[0].sampling_mask == sampling_mask
    assert decode_segments(compact) == [segment]
    # No aliasing between compact payload, source records, or decoded records.
    compact[0].segment_metrics["model_turns"] = 99
    assert segment.segment_metrics["model_turns"] == 2


@pytest.mark.parametrize("status", ["success", "task_failed", "env_error"])
@pytest.mark.parametrize("with_segments", [False, True])
def test_result_round_trip_preserves_all_outcome_fields(status, with_segments):
    trace = _trace()
    result = RemoteRolloutResult(
        **trace.model_dump(exclude={"metadata", "segments"}),
        segments=trace.segments if with_segments else [],
        status=status,
        reward=0.75 if status == "success" else 0.0,
        error_type="ExampleError" if status != "success" else None,
        error_message="example" if status != "success" else None,
        termination_reason="context_length_exceeded",
        length_details={"prompt_tokens": 100, "requested_max_tokens": 20},
    )
    compact = encode_result(result)
    assert decode_result(load_result(compact.model_dump_json())) == result


@pytest.mark.parametrize("with_trace", [False, True])
def test_agent_stage_round_trip_preserves_nested_trace_and_handoff(with_trace):
    stage = AgentStageResult(
        rollout_id="rollout-1",
        attempt_id="attempt-2",
        workspace="/workspace/task",
        agent_result=AgentResult(exit_code=7, final_answer="answer", artifacts={"patch": "x.diff"}),
        trace=_trace() if with_trace else None,
        metrics={"model_turns": 4.0},
        termination_reason="context_length_exceeded",
        length_details={"prompt_tokens": 100},
    )
    compact = encode_agent_stage(stage)
    assert compact.schema_version == 2
    assert decode_agent_stage(load_agent_stage(compact.model_dump_json())) == stage


@pytest.mark.parametrize("version", [None, 1, 3, "2", 2.0, True, False])
def test_only_explicit_integer_version_two_is_accepted(version):
    payload = encode_trace(_trace()).model_dump(mode="json")
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version
    with pytest.raises(ValidationError):
        load_trace(json.dumps(payload))


@pytest.mark.parametrize(
    ("payload", "loader"),
    [
        (_trace().model_dump_json(), load_trace),
        (RemoteRolloutResult(rollout_id="r", reward=1).model_dump_json(), load_result),
        (
            AgentStageResult(
                rollout_id="r", attempt_id="a", workspace="/workspace", agent_result=AgentResult()
            ).model_dump_json(),
            load_agent_stage,
        ),
    ],
)
def test_unversioned_dense_records_are_rejected(payload, loader):
    with pytest.raises(ValidationError):
        loader(payload)


@pytest.mark.parametrize(
    ("index", "updates", "message"),
    [
        (0, {"base_segment": 0, "prefix_len": 1}, "earlier segment"),
        (1, {"base_segment": 2}, "earlier segment"),
        (1, {"base_segment": -1}, "greater than or equal"),
        (1, {"base_segment": True}, "valid integer"),
        (1, {"prefix_len": 99}, "referenced segment"),
        (1, {"base_segment": None}, "nonempty prefix"),
        (1, {"prefix_len": 0}, "empty prefix"),
        (0, {"suffix_tokens": [1, -2, 3, 4, 5]}, "non-negative"),
    ],
)
def test_invalid_prefix_graph_is_rejected_without_expansion(index, updates, message):
    payload = encode_trace(_trace()).model_dump(mode="json")
    payload["segments"][index].update(updates)
    with pytest.raises(ValidationError, match=message):
        load_trace(json.dumps(payload))


@pytest.mark.parametrize(
    "updates",
    [
        {"response_spans": []},
        {"response_spans": [[2]]},
        {"response_spans": [[2, 4, 5]]},
        {"response_spans": [[0, 3]]},
        {"response_spans": [[2, 2]]},
        {"response_spans": [[2, 8]]},
        {"response_spans": [[3, 5], [2, 3]]},
        {"response_spans": [[2, 4], [3, 4]]},
        {"response_spans": [[True, 4]]},
        {"response_log_probs": [-0.1]},
        {"response_log_probs": [-0.1, float("inf"), -0.2]},
        {"old_response_log_probs": [-0.1]},
        {"old_response_log_probs": {"ref": "previous.log_probs"}},
        {"old_response_log_probs": {"ref": "response_log_probs", "segment": 0}},
        {"old_response_log_probs": "response_log_probs"},
        {"sampling_mask": [[12]]},
        {"sampling_mask": [[], [13], [14]]},
        {"sampling_mask": {"sampling_mask_ref": {"token_count": 1}}},
        {"sampling_mask": {"sampling_mask_ref": {"token_count": True}}},
        {"sampling_mask": {"unrecognized": True}},
    ],
)
def test_invalid_response_supervision_is_rejected(updates):
    payload = encode_trace(_trace()).model_dump(mode="json")
    payload["segments"][0].update(updates)
    with pytest.raises(ValidationError):
        load_trace(json.dumps(payload))


def test_compact_read_checks_sampling_candidates_against_inherited_tokens():
    payload = encode_trace(_trace()).model_dump(mode="json")
    payload["segments"][1]["response_spans"] = [[2, 4]]
    payload["segments"][1]["sampling_mask"] = [[900], [901]]
    with pytest.raises(ValidationError, match="must contain sampled token"):
        load_trace(json.dumps(payload))


@pytest.mark.parametrize("branch_after_unqueried_prefix", [False, True])
def test_sampling_validation_resolves_shared_prefix_positions_only_once(
    branch_after_unqueried_prefix,
):
    """Count graph accesses rather than relying on noisy wall-clock timings."""

    class CountingSegments(list):
        def __init__(self, values):
            super().__init__(values)
            self.lookups = 0

        def __getitem__(self, index):
            self.lookups += 1
            return super().__getitem__(index)

    segment_count = 160
    response_count = 256
    tokens = list(range(response_count + 1))
    unqueried_prefix_count = segment_count // 2 if branch_after_unqueried_prefix else 0
    segments = CountingSegments(
        [
            wire.CompactSegment(
                base_segment=(
                    unqueried_prefix_count - 1
                    if branch_after_unqueried_prefix and index >= unqueried_prefix_count
                    else index - 1 if index else None
                ),
                prefix_len=len(tokens) if index else 0,
                suffix_tokens=[] if index else tokens,
                response_spans=[[1, len(tokens)]],
                response_log_probs=[-0.5] * response_count,
                old_response_log_probs=ResponseLogProbsRef(ref="response_log_probs"),
                sampling_mask=(
                    [[token] for token in tokens[1:]] if index >= unqueried_prefix_count else None
                ),
            )
            for index in range(segment_count)
        ]
    )

    wire._validate_segment_graph(segments)  # pylint: disable=protected-access

    # Every queried (segment, position) needs at most one graph lookup; graph
    # bounds checks add one access per non-root segment. The branching case
    # requires path compression through ancestors that have no sampling mask.
    assert segments.lookups <= segment_count * response_count + segment_count - 1


def test_loading_and_resaving_compact_trace_never_decodes_dense_arrays(monkeypatch):
    payload = encode_trace(_trace()).model_dump_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("compact I/O must not decode dense arrays")

    monkeypatch.setattr(wire, "decode_segments", forbidden)
    restored = load_trace(payload)
    assert load_trace(restored.model_dump_json()) == restored


def test_long_shared_context_is_stored_once_without_changing_training_values():
    context = list(range(10_000))
    segments = [
        _segment(context + [10_001], {10_000}),
        _segment(context + [10_002], {10_000}),
        _segment(context + [10_003], {10_000}),
    ]
    trace = RemoteRolloutTrace(rollout_id="long", segments=segments)
    compact = encode_trace(trace)
    assert sum(len(segment.suffix_tokens) for segment in compact.segments) == 10_003
    assert len(compact.model_dump_json()) < len(trace.model_dump_json()) / 5
    assert decode_trace(load_trace(compact.model_dump_json())) == trace


def test_agent_stage_rejects_wrong_trace_identity_and_relative_workspace():
    stage = AgentStageResult(
        rollout_id="rollout-1",
        attempt_id="a",
        workspace="/workspace",
        agent_result=AgentResult(),
        trace=_trace(),
    )
    payload = encode_agent_stage(stage).model_dump(mode="json")
    payload["trace"]["rollout_id"] = "another-rollout"
    with pytest.raises(ValidationError, match="rollout_id must match"):
        load_agent_stage(json.dumps(payload))
    payload["trace"]["rollout_id"] = "rollout-1"
    payload["workspace"] = "relative/workspace"
    with pytest.raises(ValidationError, match="absolute path"):
        load_agent_stage(json.dumps(payload))
