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

"""Compare prepared loss inputs with the real Weaver SDK's normalization contract."""

from array import array
from copy import deepcopy

import numpy as np
import pytest
import torch

from northrl.weaver.datum_preparation import prepare_dense_loss_inputs
from northrl.weaver.weaver_service_holder import WeaverServiceHolder

sdk_types = pytest.importorskip("weaver.types")

_DENSE_FIELDS = ("target_tokens", "logprobs", "advantages", "loss_mask", "ref_logprobs")


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_packed_driver_buffers_match_list_payloads(default_dtype):
    previous = torch.get_default_dtype()
    raw = {
        "target_tokens": [11, 12, 13],
        "logprobs": [0.0, -0.0, -0.123456789012345],
        "advantages": [0.0, -0.25, 0.75],
        "loss_mask": [0.0, 1.0, 1.0],
        "ref_logprobs": [0.0, -0.0, -0.23456789012345],
    }
    packed = {
        key: array("q" if key == "target_tokens" else "d", values) for key, values in raw.items()
    }
    packed["loss_mask"] = array("b", [0, 1, 1])
    try:
        torch.set_default_dtype(default_dtype)
        expected = _datum(prepare_dense_loss_inputs(raw))
        holder = object.__new__(WeaverServiceHolder)
        [actual] = holder._build_weaver_datums(
            [
                {
                    "input_tokens": array("q", [10, 11, 12]),
                    "loss_fn_inputs": packed,
                    "datum_id": "test-datum",
                    "metadata": {"sequence_index": 2},
                }
            ]
        )
        assert actual.to_payload() == expected.to_payload()
        for key in raw:
            torch.testing.assert_close(
                actual.loss_fn_inputs[key],
                expected.loss_fn_inputs[key],
                rtol=0,
                atol=0,
                check_dtype=True,
            )
    finally:
        torch.set_default_dtype(previous)


def test_packed_forward_tokens_match_list_payloads():
    tokens = [10, 11, 12, 13]
    kwargs = {
        "sampling_masks": None,
        "router_replay_record": None,
        "datum_metadata_batch": None,
    }
    expected, expected_indices = WeaverServiceHolder._build_forward_datums([tokens], **kwargs)
    actual, actual_indices = WeaverServiceHolder._build_forward_datums(
        [array("q", tokens)], **kwargs
    )
    assert actual_indices == expected_indices
    assert actual[0].to_payload() == expected[0].to_payload()


def _datum(loss_fn_inputs):
    return sdk_types.Datum.from_raw(
        model_input=sdk_types.ModelInput.from_ints([10, 11, 12]),
        loss_fn_inputs=loss_fn_inputs,
        datum_id="test-datum",
        metadata={"sequence_index": 2},
    )


def _assert_sdk_equivalence(raw_inputs):
    baseline = _datum(raw_inputs)
    prepared = prepare_dense_loss_inputs(raw_inputs)
    optimized = _datum(prepared)
    assert baseline.to_payload() == optimized.to_payload()
    assert baseline.loss_fn_inputs.keys() == optimized.loss_fn_inputs.keys()
    for name, expected in baseline.loss_fn_inputs.items():
        actual = optimized.loss_fn_inputs[name]
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    return prepared, optimized


@pytest.mark.parametrize("field", _DENSE_FIELDS)
@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3],
        (1, 2, 3),
        [True, False, True],
        [1.25, 2.5, 3.75],
        [1, 2.5, True],
        [],
    ],
)
def test_known_inputs_preserve_sdk_payload_dtype_and_shape(field, values):
    raw = {field: deepcopy(values)}
    original = deepcopy(raw)
    prepared, _ = _assert_sdk_equivalence(raw)
    assert isinstance(prepared[field], torch.Tensor)
    assert raw == original


@pytest.mark.parametrize("field", _DENSE_FIELDS[1:])
@pytest.mark.parametrize("values", [[[], []], [[1, 2], [3, 4]]])
def test_rectangular_nested_inputs_keep_the_sdk_shape(field, values):
    prepared, optimized = _assert_sdk_equivalence({field: values})
    assert prepared[field] is values
    assert optimized.loss_fn_inputs[field].shape == torch.as_tensor(values).shape


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_float_and_empty_inputs_follow_current_torch_default_dtype(default_dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        prepared, _ = _assert_sdk_equivalence(
            {"logprobs": [0.25, -0.5], "advantages": [], "target_tokens": []}
        )
        assert all(value.dtype == default_dtype for value in prepared.values())
    finally:
        torch.set_default_dtype(previous)


def test_all_training_fields_match_the_original_complete_sdk_payload():
    raw = {
        "target_tokens": [11, -100, 13],
        "logprobs": [-0.1, -1.5, -2.0],
        "advantages": [1.5, -0.25, 0.0],
        "loss_mask": [True, False, True],
        "ref_logprobs": [-0.2, -1.25, -2.5],
        "sampling_mask": [[11], [12, 13], []],
        "custom_weights": [0.125, 0.5, 1.0],
    }
    original = deepcopy(raw)
    _assert_sdk_equivalence(raw)
    assert raw == original


def test_existing_tensors_arrays_tensor_data_and_scalars_keep_identity():
    tensor = torch.tensor([0.25, 0.5], dtype=torch.float64)
    array = np.asarray([1, 0], dtype=np.int32)
    tensor_data = sdk_types.TensorData.from_array([0.5, 0.25])
    raw = {
        "logprobs": tensor,
        "loss_mask": array,
        "ref_logprobs": tensor_data,
        "advantages": 0.5,
        "target_tokens": (11, 12),
    }
    prepared, optimized = _assert_sdk_equivalence(raw)
    for field in ("logprobs", "loss_mask", "ref_logprobs", "advantages"):
        assert prepared[field] is raw[field]
    assert optimized.loss_fn_inputs["logprobs"] is tensor
    assert torch.as_tensor(prepared["loss_mask"]).data_ptr() == torch.as_tensor(array).data_ptr()


@pytest.mark.parametrize("field", _DENSE_FIELDS)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64, np.bool_])
def test_numpy_inputs_keep_identity_and_sdk_dtype(field, dtype):
    values = np.asarray([1, 0, 1], dtype=dtype)
    prepared, _ = _assert_sdk_equivalence({field: values})
    assert prepared[field] is values


@pytest.mark.parametrize("field", _DENSE_FIELDS[1:])
@pytest.mark.parametrize("values", [[[1], [2, 3]], [[1], 2]])
def test_jagged_known_fields_fall_back_to_the_original_sdk_behavior(field, values):
    prepared, optimized = _assert_sdk_equivalence({field: values})
    assert prepared[field] is values
    assert optimized.loss_fn_inputs[field] is values


@pytest.mark.parametrize("field", _DENSE_FIELDS[1:])
@pytest.mark.parametrize("nested_first", [True, False])
@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_mixed_nested_containers_preserve_order_dependent_sdk_behavior(
    field, nested_first, array_type
):
    array = np.asarray([3, 4]) if array_type == "numpy" else torch.tensor([3, 4])
    values = [[1, 2], array] if nested_first else [array, [1, 2]]
    raw = {field: values}
    prepared = prepare_dense_loss_inputs(raw)
    assert prepared[field] is values

    # The SDK's jagged predicate depends on container order: a Python list
    # followed by an ndarray/tensor stays raw, while reversed inputs may be
    # normalized or rejected. Preserve the actual SDK outcome in every case.
    try:
        baseline = _datum(raw)
    except Exception as baseline_error:
        with pytest.raises(type(baseline_error)) as optimized_error:
            _datum(prepared)
        assert str(optimized_error.value) == str(baseline_error)
        return

    optimized = _datum(prepared)
    expected = baseline.loss_fn_inputs[field]
    actual = optimized.loss_fn_inputs[field]
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert optimized.to_payload() == baseline.to_payload()
    else:
        assert expected is values
        assert actual is values
        assert baseline.to_payload()["loss_fn_inputs"][field] is values
        assert optimized.to_payload()["loss_fn_inputs"][field] is values


def test_sampling_masks_and_unknown_fields_remain_untouched():
    raw = {
        "sampling_mask": [[11], [12, 13]],
        "custom_dense": [0.25, 0.5],
        "custom_jagged": [[1], [2, 3]],
        "custom_scalar": 0.5,
    }
    prepared, _ = _assert_sdk_equivalence(raw)
    assert prepared is not raw
    assert all(prepared[field] is value for field, value in raw.items())


@pytest.mark.parametrize(
    "values",
    [[-8, 12], [-1, 12], [-8.5, 12], [[1], [2, 3]], [[], []], [[1, 2], [3, 4]]],
)
def test_invalid_targets_raise_the_same_sdk_exception(values):
    raw = {"target_tokens": values}
    with pytest.raises(Exception) as baseline:
        _datum(raw)
    prepared = prepare_dense_loss_inputs(raw)
    with pytest.raises(type(baseline.value)) as optimized:
        _datum(prepared)
    assert str(optimized.value) == str(baseline.value)


def test_ignored_targets_and_existing_float_target_semantics_are_preserved():
    _assert_sdk_equivalence({"target_tokens": [-100, 12]})
    _assert_sdk_equivalence({"target_tokens": [-0.5, 12.5]})


@pytest.mark.parametrize("values", [[None], [1, [2, 3]], [1, "invalid"]])
def test_invalid_known_inputs_fall_back_to_the_original_sdk_error(values):
    raw = {"advantages": values}
    with pytest.raises(Exception) as baseline:
        _datum(raw)
    prepared = prepare_dense_loss_inputs(raw)
    assert prepared["advantages"] is values
    with pytest.raises(type(baseline.value)) as optimized:
        _datum(prepared)
    assert str(optimized.value) == str(baseline.value)
