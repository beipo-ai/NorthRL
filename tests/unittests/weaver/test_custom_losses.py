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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from northrl.weaver.custom_losses import parse_custom_loss_spec, resolve_custom_loss


def import_path_loss_factory(params):
    scale = float(params["scale"])

    def callback(data, logprob_tensors):
        del data
        loss = scale * torch.stack([tensor.mean() for tensor in logprob_tensors]).mean()
        return loss, {"loss": float(loss.detach().cpu()), "scale": scale}

    return callback


def import_path_direct_callback(data, logprob_tensors):
    del data
    loss = torch.stack([tensor.sum() for tensor in logprob_tensors]).mean()
    return loss, {"loss": float(loss.detach().cpu())}


def _make_datum(loss_fn_inputs):
    return SimpleNamespace(
        model_input=SimpleNamespace(to_ints=lambda: [1, 2]),
        loss_fn_inputs=loss_fn_inputs,
    )


def test_parse_custom_loss_spec_accepts_string_and_mapping():
    spec_from_string = parse_custom_loss_spec("direct_logprob")
    assert spec_from_string is not None
    assert spec_from_string.name == "direct_logprob"
    assert spec_from_string.params == {}

    spec_from_mapping = parse_custom_loss_spec(
        {"name": "importance_sampling", "params": {"truncation_c": 3.0}}
    )
    assert spec_from_mapping is not None
    assert spec_from_mapping.name == "importance_sampling"
    assert spec_from_mapping.params == {"truncation_c": 3.0}


def test_resolve_custom_loss_supports_import_path_factory():
    callback = resolve_custom_loss(
        {
            "name": "tests.unittests.weaver.test_custom_losses:import_path_loss_factory",
            "params": {"scale": 2.0},
        }
    )

    loss, metrics = callback([_make_datum({})], [torch.tensor([1.0, 3.0], requires_grad=True)])

    assert torch.is_tensor(loss)
    assert float(loss.detach().cpu()) == pytest.approx(4.0)
    assert metrics["scale"] == 2.0


def test_resolve_custom_loss_supports_direct_imported_callback_without_params():
    callback = resolve_custom_loss(
        {
            "name": "tests.unittests.weaver.test_custom_losses:import_path_direct_callback",
            "params": {},
        }
    )

    loss, metrics = callback([_make_datum({})], [torch.tensor([1.0, 2.0], requires_grad=True)])

    assert float(loss.detach().cpu()) == pytest.approx(3.0)
    assert metrics["loss"] == pytest.approx(3.0)


def test_importance_sampling_custom_loss_computes_expected_loss():
    callback = resolve_custom_loss(
        {"name": "importance_sampling", "params": {"entropy_coeff": 0.0}}
    )
    datum = _make_datum(
        {
            "target_tokens": [10, 11],
            "logprobs": [0.0, 0.0],
            "advantages": [1.0, 2.0],
            "loss_mask": [1.0, 1.0],
        }
    )

    logprob_value = float(torch.log(torch.tensor(2.0)).item())
    loss, metrics = callback(
        [datum],
        [torch.tensor([0.0, logprob_value], requires_grad=True)],
    )

    assert float(loss.detach().cpu()) == pytest.approx(-2.5)
    assert metrics["loss"] == pytest.approx(-2.5)
    assert metrics["ratio:max"] == pytest.approx(2.0)
    assert metrics["tokens"] == 2


def test_importance_sampling_custom_loss_requires_loss_mask():
    callback = resolve_custom_loss({"name": "importance_sampling", "params": {}})
    datum = _make_datum(
        {
            "target_tokens": [10, 11],
            "logprobs": [0.0, 0.0],
            "advantages": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="loss_mask"):
        callback([datum], [torch.tensor([0.0, 0.0], requires_grad=True)])


def test_importance_sampling_custom_loss_rejects_entropy_coeff():
    with pytest.raises(ValueError, match="entropy regularization"):
        resolve_custom_loss({"name": "importance_sampling", "params": {"entropy_coeff": 0.1}})
