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

"""Transport-gated token buffers do not change holder data or old-SDK behavior."""

import asyncio
from array import array
from types import SimpleNamespace

import pytest
import torch
from weaver.types import ModelInput

from northrl.weaver.weaver_service_holder import WeaverServiceHolder, _prepare_model_input


def assert_transport_path(model, transport):
    supported = hasattr(ModelInput, "_from_int_buffer")
    buffered = getattr(model, "_binary_token_buffer", lambda: None)() is not None
    assert buffered == (supported and transport == "http-binary")


@pytest.mark.parametrize("transport", ["default", "http-binary"])
def test_transport_gate_and_input_ownership(transport):
    source = array("q", [1, 2, 3])
    model = _prepare_model_input(source, tensor_transport=transport)
    assert_transport_path(model, transport)
    source[0] = 99
    assert model.to_ints() == [1, 2, 3]


def test_old_sdk_falls_back(monkeypatch):
    monkeypatch.delattr(ModelInput, "_from_int_buffer", raising=False)
    assert _prepare_model_input(array("q", [1, 2]), tensor_transport="http-binary").to_ints() == [
        1,
        2,
    ]


@pytest.mark.parametrize("transport", ["default", "http-binary"])
def test_recompute_target_alignment_and_metadata(transport):
    tokens = array("q", [10, 11, 12, 13])
    data, indices = WeaverServiceHolder._build_forward_datums(
        [tokens],
        sampling_masks=None,
        router_replay_record=None,
        datum_metadata_batch=[{"router_replay": "ref"}],
        tensor_transport=transport,
    )
    assert indices == [0]
    assert data[0].metadata == {"router_replay": "ref"}
    assert_transport_path(data[0].model_input, transport)
    assert data[0].model_input.to_ints() == [10, 11, 12]
    assert data[0].loss_fn_inputs["target_tokens"].tolist() == [11, 12, 13]


@pytest.mark.parametrize("transport", ["default", "http-binary"])
def test_training_uses_service_transport_and_retains_dense_fields(transport):
    async def run():
        holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
        holder._service_client = SimpleNamespace(tensor_transport=transport)
        inputs = dict(
            input_tokens=array("q", [1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": array("q", [2, 3, 4]),
                "logprobs": array("d", [-0.0, -1.0, -2.0]),
                "advantages": array("d", [1.0, 0.0, -1.0]),
                "loss_mask": array("b", [1, 0, 1]),
            },
            metadata={"router_replay": "ref"},
        )
        data = await holder._prepare_training_datums([inputs])
        assert_transport_path(data[0].model_input, transport)
        expected = holder._build_weaver_datums([inputs])[0]
        assert data[0].model_input == expected.model_input
        assert data[0].metadata == expected.metadata
        for key in expected.loss_fn_inputs:
            actual, reference = (
                data[0].loss_fn_inputs[key],
                expected.loss_fn_inputs[key],
            )
            assert actual.dtype == reference.dtype and torch.equal(actual, reference)

    asyncio.run(asyncio.wait_for(run(), 30))
