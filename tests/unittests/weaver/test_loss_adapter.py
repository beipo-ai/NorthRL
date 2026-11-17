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

"""Exercise objective-to-Weaver adaptation through actual payloads and gradients."""

from array import array
from copy import deepcopy

import pytest
import torch

from northrl.utils.config_schema import AlgorithmConfig
from northrl.weaver.loss_adapter import build_loss_adapter, normalize_advantages_for_sum


def adapter(**algorithm):
    return build_loss_adapter(
        AlgorithmConfig.model_validate({"type": "grpo", "optim": {"lr": 1e-6}, **algorithm})
    )


def test_canonical_config_emits_expected_backend_payload():
    new = adapter(
        advantage={"estimator": {"mode": "grpo", "params": {"norm_adv_by_std": False}}},
        loss={
            "mode": "ppo_clip",
            "params": {"clip": {"ratio": 0.2}},
            "reduction": {
                "mode": "fixed_response_budget",
                "params": {"max_response_tokens": 16384},
            },
        },
        entropy_coeff=0,
    )
    assert new.backend_name == "ppo_clip"
    assert new.runtime_reduction == "dr-grpo"
    assert new.build_loss_fn_config(enable_debug_dump=True) == {
        "entropy_coeff": 0,
        "enable_debug_dump": True,
        "loss_agg_mode": "sum",
        "clip_ratio": 0.2,
    }


def test_optional_ratio_parameters_keep_exact_wire_names():
    configured = adapter(
        loss={
            "mode": "ppo_clip",
            "params": {
                "clip": {
                    "ratio": 0.2,
                    "ratio_low": 0.1,
                    "ratio_high": 0.3,
                    "ratio_c": 3,
                    "truncation_c": 6,
                }
            },
        },
        entropy_coeff=0.01,
    )
    assert configured.build_loss_fn_config(enable_debug_dump=False) == {
        "entropy_coeff": 0.01,
        "enable_debug_dump": False,
        "loss_agg_mode": "sum",
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.1,
        "clip_ratio_high": 0.3,
        "clip_ratio_c": 3,
        "truncation_c": 6,
    }
    assert adapter(entropy_coeff=0).build_loss_fn_config(enable_debug_dump=False) == {
        "entropy_coeff": 0,
        "enable_debug_dump": False,
        "loss_agg_mode": "sum",
    }


def test_sft_uses_native_cross_entropy():
    configured = adapter(type="sft")
    assert configured.backend_name == "cross_entropy"
    assert configured.custom is None


def test_custom_callback_preserves_its_parameters_and_base_loss():
    configured = adapter(
        loss={
            "mode": "custom",
            "params": {"name": "module:factory", "params": {"scale": 2.0}, "base_loss_fn": None},
            "reduction": {"mode": "datum_mean"},
        }
    )
    assert configured.backend_name is None
    assert configured.custom == {
        "name": "module:factory",
        "params": {"scale": 2.0},
    }
    assert configured.runtime_reduction == "sample-mean"


def test_native_extension_preserves_backend_name():
    configured = adapter(loss={"mode": "native", "params": {"name": "extension_objective"}})
    assert configured.backend_name == "extension_objective"
    assert configured.custom is None


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "mode", ["token_mean", "datum_mean", "prompt_mean", "fixed_response_budget", "sum"]
)
def test_sum_protocol_preserves_ppo_loss_and_gradient_without_mutation(mode, packed):
    advantages = [[2.0, -3.0, 0.0], [-2.0, 4.0, 0.0], [5.0, 0.0, 0.0]]
    masks = [[1, 1, 1], [1, 1, 0], [0, 0, 0]]
    datums = [
        {
            "input_tokens": [10, 11, 12],
            "loss_fn_inputs": {
                "advantages": array("d", a) if packed else a,
                "loss_mask": m,
                "logprobs": [-0.1, -0.2, -0.3],
            },
            "_prompt_mean_denominator": d,
            "_dr_grpo_denominator": 48,
        }
        for a, m, d in zip(advantages, masks, [6, 12, 12], strict=True)
    ]
    original = deepcopy(datums)
    normalized = normalize_advantages_for_sum(
        datums, loss_fn="ppo_clip", mode=mode, token_counts=[3, 2, 0]
    )
    ratios = torch.tensor(
        [[0.6, 4.0, 1.1], [1.05, 1.4, 1.0], [0.9, 1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask_tensor = torch.tensor(masks)

    def objective(values):
        values = torch.tensor(values, dtype=torch.float64)
        losses = torch.maximum(-values * ratios, -values * ratios.clamp(0.8, 1.2))
        return torch.where(values < 0, torch.minimum(losses, -values * 3), losses) * mask_tensor

    raw_loss = objective(advantages)
    if mode == "token_mean":
        expected = raw_loss.sum() / 5
    elif mode == "datum_mean":
        expected = (raw_loss[0].sum() / 3 + raw_loss[1].sum() / 2) / 2
    elif mode == "prompt_mean":
        expected = raw_loss[0].sum() / 6 + raw_loss[1].sum() / 12
    elif mode == "fixed_response_budget":
        expected = raw_loss.sum() / 48
    else:
        expected = raw_loss.sum()
    actual = objective([list(d["loss_fn_inputs"]["advantages"]) for d in normalized]).sum()
    expected_grad = torch.autograd.grad(expected, ratios, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, ratios)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert datums == original
    for before, after in zip(datums, normalized, strict=True):
        assert before["input_tokens"] is after["input_tokens"]
        assert before["loss_fn_inputs"]["loss_mask"] is after["loss_fn_inputs"]["loss_mask"]
        assert before["loss_fn_inputs"]["logprobs"] is after["loss_fn_inputs"]["logprobs"]
        assert isinstance(after["loss_fn_inputs"]["advantages"], array) is packed


def test_padding_occurrences_receive_independent_advantage_buffers():
    datum = {"loss_fn_inputs": {"advantages": array("d", [3, 0]), "loss_mask": [1, 1]}}
    normalized = normalize_advantages_for_sum(
        [datum, datum], loss_fn="importance_sampling", mode="token_mean"
    )
    first, second = [d["loss_fn_inputs"]["advantages"] for d in normalized]
    assert list(first) == list(second) == [0.75, 0]
    assert first is not second
    assert list(datum["loss_fn_inputs"]["advantages"]) == [3, 0]


@pytest.mark.parametrize(
    "loss", ["cross_entropy", "custom_extension", "truncated_importance_sampling", None]
)
def test_other_objectives_keep_their_own_normalization(loss):
    datums = [{"loss_fn_inputs": {"loss_mask": [1]}}]
    assert normalize_advantages_for_sum(datums, loss_fn=loss, mode="token_mean") is datums
