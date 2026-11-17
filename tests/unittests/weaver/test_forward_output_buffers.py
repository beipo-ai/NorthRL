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

"""CPU regression tests for compact forward outputs and legacy compatibility."""

import asyncio
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from northrl.utils.trajectory_alignment import align_target_logprobs_to_tokens
from northrl.weaver.weaver_service_holder import (
    WeaverServiceHolder,
    _extract_forward_logprobs_batch_from_result,
)


def _result(values):
    return {"result": {"loss_fn_outputs": [{"logprobs": {"data": values}}]}}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length", [0, 1, 7, 65537])
def test_compact_output_matches_legacy_bits_and_owns_storage(dtype, length):
    values = torch.arange(length, dtype=dtype) / -7
    if length >= 7:
        values[:7] = torch.tensor(
            [0.0, -0.0, float("nan"), float("inf"), -float("inf"), -1e-20, -2.5], dtype=dtype
        )
    legacy = _extract_forward_logprobs_batch_from_result(_result(values.tolist()))[0]
    compact = _extract_forward_logprobs_batch_from_result(_result(values))[0]
    assert isinstance(compact, array) and compact.typecode == "d"
    expected = array("d", legacy)
    assert compact.tobytes() == expected.tobytes()
    tokens = array("q", [1]) * (length + 1)
    aligned = align_target_logprobs_to_tokens(tokens=tokens, target_logprobs=compact)
    assert isinstance(aligned, array)
    assert aligned.tobytes() == array("d", [0.0, *legacy]).tobytes()
    values.fill_(5)
    assert compact.tobytes() == expected.tobytes()
    if length:
        compact[0] = 9
        assert aligned[1] != 9
    # List callers keep their existing return type.
    assert isinstance(
        align_target_logprobs_to_tokens(tokens=list(tokens), target_logprobs=compact), list
    )


@pytest.mark.parametrize("length", [0, 1])
def test_empty_or_single_token_alignment(length):
    aligned = align_target_logprobs_to_tokens(
        tokens=array("q", [1]) * length, target_logprobs=array("d")
    )
    assert aligned == array("d", [0.0] * length)
    with pytest.raises(ValueError, match="target-aligned"):
        align_target_logprobs_to_tokens(tokens=array("q", [1, 2]), target_logprobs=array("d"))


@pytest.mark.parametrize(
    "values",
    [torch.zeros((2, 1)), torch.tensor(1.0), torch.tensor([True]), torch.empty(2, device="meta")],
)
def test_reject_invalid_compact_vectors(values):
    with pytest.raises(ValueError, match="CPU vector|numeric tensor"):
        _extract_forward_logprobs_batch_from_result(_result(values))


@pytest.mark.parametrize("new_sdk", [False, True])
def test_holder_selects_compact_output_only_when_sdk_supports_it(new_sdk):
    calls = []

    async def old_forward(datums, loss_fn, *, wait):
        calls.append("list")
        return _result([-0.25, -0.5])

    async def new_forward(datums, loss_fn, *, wait, tensor_output_format="list"):
        calls.append(tensor_output_format)
        return _result(torch.tensor([-0.25, -0.5]))

    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._closed = False
    holder._ensure_provisioned = AsyncMock()
    holder._training_client = SimpleNamespace(forward=new_forward if new_sdk else old_forward)
    result = asyncio.run(holder.compute_target_logprobs_batch([[], [10, 11, 12], [10]]))
    assert calls == ["torch" if new_sdk else "list"]
    assert result[0] == result[2] == []
    assert list(result[1]) == [-0.25, -0.5]
    assert isinstance(result[1], array if new_sdk else list)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_binary_outputs_keep_legacy_float_conversion(dtype):
    values = torch.tensor([1, -2, 2**30 + 1], dtype=dtype)
    compact = _extract_forward_logprobs_batch_from_result(_result(values))[0]
    legacy = _extract_forward_logprobs_batch_from_result(_result(values.tolist()))[0]
    assert compact.tobytes() == array("d", legacy).tobytes()
