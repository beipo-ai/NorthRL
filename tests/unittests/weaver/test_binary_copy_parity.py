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

"""Bounded copies retain dtype bits, independent storage and binary payload parity."""

from array import array

import numpy as np
import pytest
import torch

from northrl.weaver.datum_preparation import prepare_dense_loss_inputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length", [1, 16383, 16384, 16385, 32768, 65536, 262144])
def test_buffer_copy_exact_float_bits_and_ownership(dtype, length):
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2**64, size=length, dtype=np.uint64)
    bits[: min(length, 8)] = np.array(
        [
            0,
            1 << 63,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000000001,
            0xFFF8000000000042,
            0x7FF0000000000001,
            1,
        ],
        dtype=np.uint64,
    )[: min(length, 8)]
    source = array("d")
    source.frombytes(bits.tobytes())
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        reference = torch.frombuffer(source, dtype=torch.float64).to(dtype=dtype, copy=True)
        result = prepare_dense_loss_inputs({"advantages": source})["advantages"]
        integer = torch.int32 if dtype == torch.float32 else torch.int64
        assert torch.equal(result.view(integer), reference.view(integer))
        assert result.untyped_storage().resizable()
        source.append(2.0)
        result.resize_(length + 10)
    finally:
        torch.set_default_dtype(old)


def test_dense_copy_retains_bounded_kernel_sizes(monkeypatch):
    original = torch.Tensor.copy_
    calls = []

    def record(self, source, *args, **kwargs):
        calls.append(source.numel())
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record)
    prepare_dense_loss_inputs({"target_tokens": array("q", range(65536))})
    assert calls == [16384] * 4


@pytest.mark.parametrize("compression", ["raw", "zstd"])
@pytest.mark.parametrize(
    "loss", ["cross_entropy", "forward_logprob", "importance_sampling", "ppo_clip"]
)
def test_large_binary_request_matches_legacy_construction(compression, loss):
    from weaver.tensor_transport import serialize_training_data

    from northrl.weaver.weaver_service_holder import WeaverServiceHolder

    size = 65536
    source = {
        "input_tokens": array("q", range(size)),
        "loss_fn_inputs": {
            "target_tokens": array("q", range(1, size + 1)),
            "logprobs": array("d", [-0.123456789, -0.0]) * (size // 2),
            "advantages": array("d", [-0.5, 0.5]) * (size // 2),
            "loss_mask": array("b", [0, 1]) * (size // 2),
        },
    }
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    requests = []
    try:
        for transport in ("default", "http-binary"):
            data = holder._build_weaver_datums([source], tensor_transport=transport)
            requests.append(
                serialize_training_data(
                    data, loss_fn=loss, transport="http-binary", compression=compression
                )
            )
        assert requests[0].data == requests[1].data
        assert (
            requests[0].tensor_pack.path.read_bytes() == requests[1].tensor_pack.path.read_bytes()
        )
    finally:
        for request in requests:
            request.tensor_pack.close()
