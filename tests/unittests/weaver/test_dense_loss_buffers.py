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

"""Dense buffer conversion must retain the original tensor construction contract."""

from array import array

import pytest
import torch

from northrl.weaver.datum_preparation import prepare_dense_loss_inputs


def _assert_same_tensor(actual, expected):
    assert (actual.dtype, actual.device, actual.shape) == (
        expected.dtype,
        expected.device,
        expected.shape,
    )
    assert actual.requires_grad == expected.requires_grad
    if actual.device.type != "meta":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        if actual.is_floating_point():
            assert torch.equal(
                torch.signbit(actual[actual == 0]), torch.signbit(expected[expected == 0])
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_buffer_values_and_default_dtypes_match_original_conversion(dtype):
    data = {
        "target_tokens": array("q", [0, 1024, -100, 2**63 - 1, -(2**63)]),
        "logprobs": array("d", [-0.0, 1e-300, -1e300, float("inf"), float("nan")]),
        "advantages": array("d", [1e-8, 0, -2, 3.141592653589793, -float("inf")]),
        "loss_mask": array("b", [-128, -1, 0, 1, 127]),
        "ref_logprobs": array("d", [0.0, -4.25]),
    }
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        actual = prepare_dense_loss_inputs(data)
        for key, value in data.items():
            expected = torch.tensor(value, dtype=torch.int64 if key == "target_tokens" else dtype)
            _assert_same_tensor(actual[key], expected)
            assert isinstance(data[key], array)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "key, code", [("target_tokens", "q"), ("logprobs", "d"), ("loss_mask", "b")]
)
def test_buffer_copies_have_independent_resizable_storage(key, code):
    source = array(code, [1, 2, 3])
    result = prepare_dense_loss_inputs({key: source})[key]
    source[0] = 9
    assert result[0].item() == 1
    result[1] = 8
    assert source[1] == 2
    source.append(4)
    result.resize_(8)
    assert result[:3].tolist() == [1, 8, 3]


@pytest.mark.parametrize("length", [16383, 16384, 16385, 65536, 67415])
def test_large_buffers_match_across_copy_boundaries(length):
    data = {
        "target_tokens": array("q", range(length)),
        "advantages": array("d", [-4.25]) * length,
        "loss_mask": array("b", [1]) * length,
    }
    actual = prepare_dense_loss_inputs(data)
    for key, source in data.items():
        expected = torch.tensor(
            source, dtype=torch.int64 if key == "target_tokens" else torch.get_default_dtype()
        )
        _assert_same_tensor(actual[key], expected)


@pytest.mark.parametrize("key", ["target_tokens", "advantages", "loss_mask"])
@pytest.mark.parametrize("code", ["q", "d", "b", "f", "i"])
def test_empty_buffers_match_original_conversion(key, code):
    source = array(code)
    dtype = torch.int64 if key == "target_tokens" else torch.get_default_dtype()
    _assert_same_tensor(
        prepare_dense_loss_inputs({key: source})[key], torch.tensor(source, dtype=dtype)
    )


@pytest.mark.parametrize(
    "key, source",
    [
        ("target_tokens", array("d", [1.5, -100.5])),
        ("target_tokens", array("d", [float("nan")])),
        ("target_tokens", array("d", [float("inf")])),
        ("target_tokens", array("Q", [2**64 - 1])),
        ("target_tokens", array("i", [0, 1, -100])),
        ("advantages", array("q", [2**60 + 1, 2**63 - 1])),
        ("advantages", array("f", [0, 1, -0.5])),
    ],
)
def test_unsupported_buffers_preserve_conversion_and_errors(key, source):
    dtype = torch.int64 if key == "target_tokens" else torch.get_default_dtype()
    try:
        expected = torch.tensor(source, dtype=dtype)
    except Exception as error:
        with pytest.raises(type(error)) as actual:
            prepare_dense_loss_inputs({key: source})
        assert str(actual.value) == str(error)
    else:
        _assert_same_tensor(prepare_dense_loss_inputs({key: source})[key], expected)


def test_array_subclasses_keep_custom_indexing():
    class CustomArray(array):
        def __getitem__(self, index):
            return super().__getitem__(index) * 2

    source = CustomArray("q", [1, 2, 3])
    _assert_same_tensor(
        prepare_dense_loss_inputs({"target_tokens": source})["target_tokens"],
        torch.tensor(source, dtype=torch.int64),
    )


def test_non_cpu_device_context_and_default_are_preserved():
    data = {"target_tokens": array("q", [1, 2]), "logprobs": array("d", [-4.0, -2.0])}

    def check():
        for key, result in prepare_dense_loss_inputs(data).items():
            expected = torch.tensor(
                data[key],
                dtype=torch.int64 if key == "target_tokens" else torch.get_default_dtype(),
            )
            _assert_same_tensor(result, expected)
            assert result.device.type == "meta"

    with torch.device("meta"):
        check()
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        check()
    finally:
        torch.set_default_device(previous)


def test_other_inputs_and_custom_fields_keep_existing_behavior():
    tensor = torch.arange(4)
    jagged = [[1], [2, 3]]
    custom = array("q", [1, 2])
    data = {"target_tokens": tensor, "advantages": jagged, "custom": custom}
    actual = prepare_dense_loss_inputs(data)
    assert actual is not data
    for key, value in data.items():
        assert actual[key] is value
    for source in ([1, 2, 3], (1.5, -0.0)):
        _assert_same_tensor(
            prepare_dense_loss_inputs({"logprobs": source})["logprobs"], torch.as_tensor(source)
        )
