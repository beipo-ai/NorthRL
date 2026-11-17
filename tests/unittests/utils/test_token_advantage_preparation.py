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

"""Response-only expansion preserves values, lazy reads and buffer ownership."""

from array import array
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from northrl.northrl_types import Trajectory
from northrl.utils.finetune_service_utils import (
    _expand_response_advantages,
    _has_nonzero_advantages,
    convert_trajectories_to_datums,
)


def _legacy_expand(values, masks):
    output = []
    index = 0
    for mask in masks:
        if mask == 1 and index < len(values):
            output.append(float(values[index]))
            index += 1
        else:
            output.append(0.0)
    return output


def _edge_values(typecode):
    values = array(typecode, [0.0, -0.0, 0.1, -0.1, float("inf"), -float("inf")])
    # Include positive/negative quiet NaNs with payloads and a signaling NaN.
    bits = (
        np.array([0x7FF8000000000001, 0xFFF8000000000003, 0x7FF0000000000001], dtype=np.uint64)
        if typecode == "d"
        else np.array([0x7FC00001, 0xFFC00003, 0x7F800001], dtype=np.uint32)
    )
    values.frombytes(bits.tobytes())
    return values


@pytest.mark.parametrize("typecode", ["f", "d"])
@pytest.mark.parametrize("length", [0, 1, 4, 9, 13])
@pytest.mark.parametrize("pattern", [[], [0], [1], [0, 1], [1, 0, -1, 2, 1]])
def test_compact_expansion_is_bitwise_equal(typecode, length, pattern):
    values = _edge_values(typecode)
    values.extend([1e-30] * 4)
    values = values[:length]
    mask = array("b", pattern * 9)
    before = values.tobytes(), mask.tobytes()
    expected = array("d", _legacy_expand(values, mask))
    with np.errstate(all="raise"):
        actual = _expand_response_advantages(values, mask)
        assert _has_nonzero_advantages(actual) == any(value != 0.0 for value in expected)
    assert type(actual) is array and actual.typecode == "d"
    assert actual.tobytes() == expected.tobytes()
    assert (values.tobytes(), mask.tobytes()) == before
    # No exported NumPy views or shared output/source storage remain alive.
    actual.append(99)
    values.append(42)
    mask.append(1)
    if expected:
        actual[0] = 123
        assert values[:-1].tobytes() == before[0]


@pytest.mark.parametrize("typecode", ["f", "d"])
def test_random_float_bits_are_preserved(typecode):
    rng = np.random.default_rng(22)
    dtype, maximum = (np.uint64, 2**64 - 1) if typecode == "d" else (np.uint32, 2**32 - 1)
    values = array(typecode)
    values.frombytes(rng.integers(0, maximum, 1024, dtype=dtype).tobytes())
    masks = array("b", rng.choice([-2, 0, 1, 1, 2], 3000))
    with np.errstate(all="raise"):
        actual = _expand_response_advantages(values, masks)
    assert actual.tobytes() == array("d", _legacy_expand(values, masks)).tobytes()


@pytest.mark.parametrize("values", [[], [0.0, -0.0], [float("nan")], [float("inf")]])
def test_compact_nonzero_scan_matches_python(values):
    packed = array("d", values)
    assert _has_nonzero_advantages(packed) == any(value != 0.0 for value in values)
    packed.append(0)


def test_custom_sequence_keeps_indexed_float_conversion_and_ignores_tail():
    reads = []

    class FloatValue:
        def __float__(self):
            return -0.25

    class IndexedOnly(Sequence):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            assert type(index) is int
            reads.append(index)
            return [FloatValue(), float("inf"), None][index]

        def __iter__(self):
            raise AssertionError("The original converter only indexes response values")

        def __array__(self, *args, **kwargs):
            raise AssertionError("Custom conversion hooks must not be used")

    actual = _expand_response_advantages(IndexedOnly(), array("b", [0, 1, 2, 1, -1]))
    assert actual == [0.0, -0.25, 0.0, float("inf"), 0.0]
    assert reads == [0, 1]
    with pytest.raises(TypeError):
        _expand_response_advantages(IndexedOnly(), array("b", [1, 1, 1]))


def test_array_subclasses_keep_overridden_index_and_iteration_behavior():
    class IndexedArray(array):
        def __getitem__(self, index):
            return 2 * super().__getitem__(index)

    class IteratedMask(array):
        def __iter__(self):
            return iter([1, 0, 1])

    assert _expand_response_advantages(IndexedArray("d", [1, 3]), array("b", [1, 0, 1])) == [
        2,
        0,
        6,
    ]
    assert _expand_response_advantages(array("d", [1, 3]), IteratedMask("b", [0, 0, 0])) == [
        1,
        0,
        3,
    ]


@pytest.mark.parametrize(
    "factory", [list, tuple, lambda values: np.array(values, dtype=np.float32)]
)
def test_noncompact_values_keep_fallback(factory):
    values = factory([0.0, -0.0, 0.1, float("inf")])
    masks = array("b", [0, 1, 2, 1, 1, 1, 1])
    assert (
        array("d", _expand_response_advantages(values, masks)).tobytes()
        == array("d", _legacy_expand(values, masks)).tobytes()
    )


def _trajectory(compact, values, *, mis=False):
    masks = [0, 0, 1, -2, 1, 2, 0, 1, 1]
    fields = {"old_log_probs": [0.0] + [-0.2] * 8, "token_advantages": values}
    if compact:
        fields["old_log_probs"] = array("d", fields["old_log_probs"])
    if mis:
        fields.update(
            correction_weights=[0, 0, 0.5, 2, 1, 1, 1, 2, 1],
            correction_mask=[0, 0, 1, 0, 1, 1, 1, 0, 1],
        )
    return Trajectory(
        tokens=array("q", range(9)) if compact else list(range(9)),
        loss_mask=array("b", masks) if compact else masks,
        reward=1,
        extra_fields=fields,
    )


@pytest.mark.parametrize("mis", [False, True])
@pytest.mark.parametrize("values", [[], [0.0, -0.0], [0.1, -0.2], [0.1, 0.2, -0.3, 0.4, 99]])
def test_converter_payload_and_gradient_match_original_expansion(values, mis):
    gradients = []
    payloads = []
    for compact in (False, True):
        source = array("d", values)
        trajectory = _trajectory(compact, source, mis=mis)
        converted = convert_trajectories_to_datums([trajectory], drop_zero_advantage_datums=False)[
            0
        ]
        loss_inputs = converted["loss_fn_inputs"]
        if compact:
            assert type(loss_inputs["advantages"]) is array
        payloads.append({key: array("d", value).tobytes() for key, value in loss_inputs.items()})
        current = torch.linspace(-0.4, -0.1, 8, dtype=torch.float64, requires_grad=True)
        denominator = torch.tensor(loss_inputs["logprobs"], dtype=torch.float64)
        advantages = torch.tensor(loss_inputs["advantages"], dtype=torch.float64)
        mask = torch.tensor(loss_inputs["loss_mask"], dtype=torch.float64)
        loss = -(torch.exp(current - denominator) * advantages * mask).sum()
        loss.backward()
        gradients.append(current.grad)
        assert source.tobytes() == array("d", values).tobytes()
        source.append(0)
    assert payloads[0] == payloads[1]
    assert torch.equal(*gradients)


@pytest.mark.parametrize("values", [[], [0.0, -0.0], [0.0, 0.0, 0.0, 0.0, 99]])
def test_unused_nonzero_tail_does_not_retain_zero_datum(values):
    assert convert_trajectories_to_datums([_trajectory(True, array("d", values))]) == []


def test_nonfinite_response_values_keep_datum_and_payload_bits():
    source = _edge_values("d")
    trajectory = _trajectory(True, source[6:])
    actual = convert_trajectories_to_datums([trajectory])[0]["loss_fn_inputs"]["advantages"]
    expected = _legacy_expand(source[6:], trajectory.loss_mask[1:])
    assert actual.tobytes() == array("d", expected).tobytes()
