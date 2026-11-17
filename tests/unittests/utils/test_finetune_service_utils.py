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

"""Test logprob and loss-mask alignment at the trajectory-to-Datum boundary."""

import struct
from array import array

import pytest

from northrl.northrl_types import Trajectory
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums


def _make_traj(extra_fields=None) -> Trajectory:
    """Use token-aligned old_log_probs with a no-context placeholder at index 0.

    The first two tokens are prompt context; the remaining three, including the
    final token, contribute to training. Datum arrays align with tokens[1:].
    """
    return Trajectory(
        tokens=[1, 2, 3, 4, 5],
        loss_mask=[0, 0, 1, 1, 1],
        reward=0.5,
        is_val=False,
        extra_fields={"advantage": 1.0, "old_log_probs": [0.0, 0.0, -0.3, -0.4, -0.5]}
        | (extra_fields or {}),
    )


def _convert_one(traj: Trajectory) -> dict:
    datums = convert_trajectories_to_datums([traj])
    assert len(datums) == 1
    return datums[0]


def test_no_ref_logprobs_preserves_baseline_keys():
    """Shift all fields to prediction targets and keep the last response token."""
    datum = _convert_one(_make_traj())

    assert datum["input_tokens"] == [1, 2, 3, 4]
    assert datum["loss_fn_inputs"] == {
        "target_tokens": [2, 3, 4, 5],
        "logprobs": [0.0, -0.3, -0.4, -0.5],
        "loss_mask": [0.0, 1.0, 1.0, 1.0],
        "advantages": [0.0, 1.0, 1.0, 1.0],
    }


@pytest.mark.parametrize("dtype_name", ["python", "float32", "float64"])
@pytest.mark.parametrize("value", [0.1, -0.1, 0.0, -0.0, 1e-38, 1e38])
def test_scalar_advantage_expansion_preserves_dtype_rounding_and_signed_zero(dtype_name, value):
    """Caching repeated masks must retain the original scalar product exactly."""
    if dtype_name == "python":
        advantage = value
    else:
        torch = pytest.importorskip("torch")
        advantage = torch.tensor(value, dtype=getattr(torch, dtype_name))
    # Alignment converts bools to ints; nonbinary and negative integer masks
    # are also accepted and must not be collapsed to a binary response mask.
    masks = [False, True, 2, 7, -2, 0, 1, 7, -2]
    traj = Trajectory(
        tokens=list(range(len(masks) + 1)),
        loss_mask=[0, *masks],
        reward=0.0,
        extra_fields={"advantage": advantage, "old_log_probs": [0.0] * (len(masks) + 1)},
    )

    datum = convert_trajectories_to_datums([traj], drop_zero_advantage_datums=False)[0]
    actual = datum["loss_fn_inputs"]["advantages"]
    expected = [float(advantage * int(mask)) for mask in masks]

    # Ordinary float equality misses negative zero and approximate comparisons
    # can hide float32 rounding changes from converting the advantage too early.
    assert [struct.pack("!d", item) for item in actual] == [
        struct.pack("!d", item) for item in expected
    ]


@pytest.mark.parametrize("head", [None, 0.0], ids=["none-placeholder", "zero-placeholder"])
def test_full_length_ref_logprobs_are_shifted(head):
    """Reference scores use the same token alignment and mask as old_log_probs."""
    ref = [head, -1.1, -1.2, -1.3, -1.4]
    datum = _convert_one(_make_traj({"ref_logprobs": ref}))

    assert datum["loss_fn_inputs"]["ref_logprobs"] == [0.0, -1.2, -1.3, -1.4]
    assert datum["loss_fn_inputs"]["loss_mask"] == [0.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("field", ["old_log_probs", "ref_logprobs"])
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0.0, -0.1], id="too-short"),
        pytest.param([-0.1, -0.2, -0.3, -0.4], id="target-aligned"),
        pytest.param([0.0, -0.1, -0.2, -0.3, -0.4, -0.5], id="too-long"),
    ],
)
def test_logprob_length_mismatch_is_rejected(field, values):
    """Require one score per original token instead of guessing or truncating."""
    with pytest.raises(ValueError, match=rf"{field} must be token-aligned with tokens"):
        _convert_one(_make_traj({field: values}))


# Every Datum field must align with target_tokens=tokens[1:]. The converter
# accepts token-aligned inputs only; mismatched inputs are rejected above.
SHAPE_KEYS_BASELINE = ("logprobs", "advantages", "loss_mask")
SHAPE_KEYS_WITH_REF = SHAPE_KEYS_BASELINE + ("ref_logprobs",)


def _assert_shape_consistent(datum: dict, expected_len: int, keys: tuple) -> None:
    inp = datum["input_tokens"]
    tgt = datum["loss_fn_inputs"]["target_tokens"]
    assert len(inp) == expected_len, f"input_tokens len={len(inp)}, want {expected_len}"
    assert len(tgt) == expected_len, f"target_tokens len={len(tgt)}, want {expected_len}"
    for key in keys:
        arr = datum["loss_fn_inputs"][key]
        assert len(arr) == expected_len, f"{key} len={len(arr)} != input_tokens len={expected_len}"


@pytest.mark.parametrize("old_head", [None, 0.0], ids=["none-placeholder", "zero-placeholder"])
def test_shape_consistent_no_ref(old_head):
    """Both supported first-position placeholders produce identical targets."""
    traj = _make_traj({"old_log_probs": [old_head, -0.2, -0.3, -0.4, -0.5]})
    datum = _convert_one(traj)

    _assert_shape_consistent(datum, expected_len=4, keys=SHAPE_KEYS_BASELINE)
    assert datum["loss_fn_inputs"]["logprobs"] == [0.0, -0.3, -0.4, -0.5]


@pytest.mark.parametrize("old_head", [None, 0.0], ids=["old-none", "old-zero"])
@pytest.mark.parametrize("ref_head", [None, 0.0], ids=["ref-none", "ref-zero"])
def test_shape_consistent_with_ref(old_head, ref_head):
    """Old and reference scores retain matching target positions after shifting."""
    traj = _make_traj(
        {
            "old_log_probs": [old_head, -0.2, -0.3, -0.4, -0.5],
            "ref_logprobs": [ref_head, -1.1, -1.2, -1.3, -1.4],
        }
    )
    datum = _convert_one(traj)

    _assert_shape_consistent(datum, expected_len=4, keys=SHAPE_KEYS_WITH_REF)
    assert datum["loss_fn_inputs"]["logprobs"] == [0.0, -0.3, -0.4, -0.5]
    assert datum["loss_fn_inputs"]["ref_logprobs"] == [0.0, -1.2, -1.3, -1.4]


def test_converter_uses_old_log_probs_as_denominator():
    """The denominator comes from old_log_probs without a source-selection flag."""
    traj = _make_traj(
        {
            "old_log_probs": [0.0, -0.7, -0.8, -0.9, -1.0],
            # Distinct stale metadata must not override the training denominator.
            "logprobs": [0.0, -1.1, -1.2, -1.3, -1.4],
            "rollout_log_probs": [0.0, -2.1, -2.2, -2.3, -2.4],
        }
    )
    datum = _convert_one(traj)

    _assert_shape_consistent(datum, expected_len=4, keys=SHAPE_KEYS_BASELINE)
    assert datum["loss_fn_inputs"]["logprobs"] == [0.0, -0.8, -0.9, -1.0]


@pytest.mark.parametrize("alternative_field", ["logprobs", "rollout_log_probs"])
def test_missing_old_log_probs_is_rejected(alternative_field):
    """Other score fields cannot silently substitute for a missing denominator."""
    traj = _make_traj()
    traj.extra_fields[alternative_field] = traj.extra_fields.pop("old_log_probs")

    with pytest.raises(KeyError, match="missing required old_log_probs"):
        _convert_one(traj)


@pytest.mark.parametrize("seq_len", [3, 5, 8, 16])
def test_shape_consistent_varying_seq_len(seq_len):
    """All original response positions survive conversion at different lengths."""
    tokens = list(range(1, seq_len + 1))
    loss_mask = [0] + [1] * (seq_len - 1)
    old_lp = [0.0] + [-0.1 * i for i in range(1, seq_len)]
    ref_lp = [0.0] + [-0.2 * i for i in range(1, seq_len)]
    traj = Trajectory(
        tokens=tokens,
        loss_mask=loss_mask,
        reward=0.0,
        is_val=False,
        extra_fields={
            "advantage": 1.0,
            "old_log_probs": old_lp,
            "ref_logprobs": ref_lp,
        },
    )
    datum = _convert_one(traj)

    _assert_shape_consistent(datum, expected_len=seq_len - 1, keys=SHAPE_KEYS_WITH_REF)
    assert datum["loss_fn_inputs"]["logprobs"] == old_lp[1:]
    assert datum["loss_fn_inputs"]["ref_logprobs"] == ref_lp[1:]
    assert datum["loss_fn_inputs"]["loss_mask"] == [1.0] * (seq_len - 1)


def test_shape_consistent_with_token_advantages():
    """Response-only OPD advantages align with targets, including the last token."""
    traj = _make_traj({"token_advantages": [0.4, 0.5, 0.6]})
    datum = _convert_one(traj)

    _assert_shape_consistent(datum, expected_len=4, keys=SHAPE_KEYS_BASELINE)
    assert datum["loss_fn_inputs"]["advantages"] == [0.0, 0.4, 0.5, 0.6]


@pytest.mark.parametrize("drop_zero_advantage_datums", [False, True])
@pytest.mark.parametrize("mis_rejected", [False, True], ids=["original-mask", "mis-mask"])
def test_converter_omits_sequences_without_final_training_targets(
    drop_zero_advantage_datums, mis_rejected
):
    """No-target samples must disappear even when zero advantages are retained."""
    trajectory = _make_traj()
    if mis_rejected:
        trajectory["correction_weights"] = [0.0, 1.0, 1.0, 1.0, 1.0]
        trajectory["correction_mask"] = [0, 0, 0, 0, 0]
    else:
        trajectory.loss_mask = [0, 0, 0, 0, 0]
    sources = []

    datums = convert_trajectories_to_datums(
        [trajectory],
        drop_zero_advantage_datums=drop_zero_advantage_datums,
        datum_sources=sources,
    )

    assert datums == []
    assert sources == []


@pytest.mark.parametrize("compact", [False, True], ids=["lists", "compact-buffers"])
def test_converter_filtering_preserves_source_indices_and_metadata(compact):
    """Filter final masks without renumbering replay samples or nested sources."""

    def sequence(first_token, mask, correction_mask=None):
        result = {
            "tokens": [first_token, first_token + 1, first_token + 2],
            "loss_mask": mask,
            "old_log_probs": [0.0, -0.1, -0.2],
        }
        if correction_mask is not None:
            result["correction_weights"] = [0.0, 1.0, 1.0]
            result["correction_mask"] = correction_mask
        if compact:
            for name, values in result.items():
                typecode = (
                    "I" if name == "tokens" else "d" if "prob" in name or "weights" in name else "b"
                )
                result[name] = array(typecode, values)
        return result

    empty = _make_traj()
    empty.loss_mask = [0] * len(empty.tokens)
    segments = [
        sequence(20, [0, 1, 1], [0, 0, 0]),
        sequence(30, [0, 1, 1], [0, 0, 1]),
        sequence(40, [0, 0, 0]),
    ]
    nested = Trajectory(segments=segments, reward=1.0, extra_fields={"advantage": 0.0})
    active = _make_traj({"advantage": 0.0})
    sources = []
    metadata_calls = []

    def metadata_builder(training_sequence, sample_index):
        metadata_calls.append((training_sequence, sample_index))
        return {"sample_index": sample_index, "first_token": training_sequence["tokens"][0]}

    datums = convert_trajectories_to_datums(
        [empty, nested, active],
        drop_zero_advantage_datums=False,
        datum_sources=sources,
        metadata_builder=metadata_builder,
    )

    assert sources == [(1, 1), (2, 0)]
    assert len(datums) == len(metadata_calls) == 2
    assert metadata_calls[0][0] is segments[1]
    assert metadata_calls[1][0] is active
    assert [index for _, index in metadata_calls] == [2, 4]
    assert [datum["metadata"] for datum in datums] == [
        {"sample_index": 2, "first_token": 30},
        {"sample_index": 4, "first_token": 1},
    ]
    assert [list(datum["loss_fn_inputs"]["loss_mask"]) for datum in datums] == [
        [0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ]
    assert all(not any(datum["loss_fn_inputs"]["advantages"]) for datum in datums)
    assert list(segments[1]["loss_mask"]) == [0, 1, 1]
    assert list(segments[1]["correction_mask"]) == [0, 0, 1]


@pytest.mark.parametrize("field", ["old_log_probs", "ref_logprobs"])
def test_full_length_logprobs_with_nonzero_head_are_rejected(field):
    """A real score in the no-context slot must not be silently discarded."""
    values = [-0.1, -0.2, -0.3, -0.4, -0.5]
    with pytest.raises(ValueError, match=rf"{field}\[0\] must be None or 0\.0"):
        _convert_one(_make_traj({field: values}))


@pytest.mark.parametrize("field", ["old_log_probs", "ref_logprobs"])
def test_logprobs_with_none_after_the_placeholder_are_rejected(field):
    """Only the first no-context slot may omit a numeric logprob."""
    values = [0.0, -0.2, None, -0.4, -0.5]
    with pytest.raises(ValueError, match=rf"{field} may only contain None at the first token"):
        _convert_one(_make_traj({field: values}))


@pytest.mark.parametrize("cross_entropy", [False, True])
def test_converter_keeps_plain_trajectory_and_expands_only_nested_sequences(cross_entropy):
    from northrl.northrl_types import Trajectory
    from northrl.utils.finetune_service_utils import convert_trajectories_to_datums_cross_entropy

    tokens = [10, 11]
    plain = Trajectory(
        tokens=tokens,
        loss_mask=[0, 1],
        reward=1.0,
        extra_fields={"old_log_probs": [0.0, -0.1], "advantage": 0.5},
    )
    nested_sequences = [
        {"tokens": [20, 21], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.2]},
        {"tokens": [30, 31, 32], "loss_mask": [0, 1, 1], "old_log_probs": [0.0, -0.3, -0.4]},
    ]
    nested = Trajectory(
        segments=nested_sequences,
        reward=0.0,
        extra_fields={"advantage": -0.5, "rollout_id": "nested"},
    )
    ignored = Trajectory(tokens=[40, 41], loss_mask=[0, 1], reward=None, status="failed")
    convert = (
        convert_trajectories_to_datums_cross_entropy
        if cross_entropy
        else convert_trajectories_to_datums
    )
    datums = convert([plain, nested, ignored])

    assert [datum["input_tokens"] for datum in datums] == [[10], [20], [30, 31]]
    assert [datum["loss_fn_inputs"]["target_tokens"] for datum in datums] == [[11], [21], [31, 32]]
    assert plain.tokens is tokens
    assert plain.segments is None
    assert nested.segments is nested_sequences
    assert all("reward" not in sequence for sequence in nested_sequences)
    if not cross_entropy:
        assert [datum["loss_fn_inputs"]["advantages"] for datum in datums] == [
            [0.5],
            [-0.5],
            [-0.5, -0.5],
        ]


@pytest.mark.parametrize("cross_entropy", [False, True])
def test_converter_rejects_a_standalone_sequence_dictionary(cross_entropy):
    from northrl.utils.finetune_service_utils import convert_trajectories_to_datums_cross_entropy

    sequence = {
        "tokens": [10, 11],
        "loss_mask": [0, 1],
        "old_log_probs": [0.0, -0.1],
        "advantage": 1.0,
    }
    convert = (
        convert_trajectories_to_datums_cross_entropy
        if cross_entropy
        else convert_trajectories_to_datums
    )
    with pytest.raises(TypeError, match="complete Trajectory"):
        convert([sequence])
