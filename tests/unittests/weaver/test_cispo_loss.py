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

from types import SimpleNamespace

import pytest
import torch

from northrl.weaver.custom_losses import resolve_custom_loss


def _datum(old, advantages, mask):
    return SimpleNamespace(
        loss_fn_inputs={"logprobs": old, "advantages": advantages, "loss_mask": mask}
    )


def _callback(**params):
    return resolve_custom_loss({"name": "cispo", "params": {"clip_ratio": 0.2, **params}})


def test_clipped_weights_have_no_gradient_but_clipped_tokens_still_update():
    current = (torch.tensor([0.25, 0.9, 1.5, 2.0], dtype=torch.float64).log() - 3).requires_grad_()
    datum = _datum([-3.0] * 4, [2.0, -3.0, 1.0, -2.0], [1] * 4)
    loss, metrics = _callback()([datum], [current])
    loss.backward()
    # Analytic dL/dlogp = -clip(r) * A / N, including both clipped tails.
    expected_gradient = torch.tensor([-0.4, 0.675, -0.3, 0.6], dtype=torch.float64)
    torch.testing.assert_close(current.grad, expected_gradient)
    torch.testing.assert_close(loss, (expected_gradient * current).sum())
    assert metrics["ppo_clip/clipfrac"] == pytest.approx(0.75)


def test_global_token_mean_is_invariant_to_segment_boundaries():
    current = torch.tensor([-2.0, -3.0, -4.0, -2.5], dtype=torch.float64, requires_grad=True)
    old, advantages = [-3.0] * 4, [1.0, -2.0, 0.0, 4.0]
    whole, _ = _callback()([_datum(old, advantages, [1] * 4)], [current])
    split, metrics = _callback()(
        [_datum(old[:1], advantages[:1], [1]), _datum(old[1:], advantages[1:], [1] * 3)],
        [current[:1], current[1:]],
    )
    torch.testing.assert_close(whole, split)
    torch.testing.assert_close(
        torch.autograd.grad(whole, current, retain_graph=True)[0],
        torch.autograd.grad(split, current)[0],
    )
    assert metrics["tokens"] == 4  # Zero-advantage targets remain in the denominator.


def test_masked_nonfinite_values_and_empty_sequences_do_not_pollute_loss():
    first = torch.tensor([float("nan"), -2.0, float("inf")], requires_grad=True)
    empty = torch.tensor([float("nan")], requires_grad=True)
    datums = [
        _datum([float("nan"), -2.0, 0.0], [float("nan"), 2.0, float("inf")], [0, 1, 0]),
        _datum([float("nan")], [float("nan")], [0]),
    ]
    loss, metrics = _callback()(datums, [first, empty])
    loss.backward()
    assert loss.item() == pytest.approx(4.0)
    assert metrics["tokens"] == 1
    torch.testing.assert_close(first.grad, torch.tensor([0.0, -2.0, 0.0]))
    torch.testing.assert_close(empty.grad, torch.tensor([0.0]))


def test_all_masked_batch_returns_connected_zero():
    current = torch.tensor([float("nan")], requires_grad=True)
    loss, metrics = _callback()([_datum([0.0], [1.0], [0])], [current])
    loss.backward()
    assert loss.item() == metrics["tokens"] == 0
    assert current.grad.item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_extreme_ratios_remain_finite_in_low_precision(dtype):
    current = torch.tensor([-1.0, -100.0], dtype=dtype, requires_grad=True)
    loss, _ = _callback()([_datum([-100.0, -1.0], [1.0, -1.0], [1, 1])], [current])
    loss.backward()
    assert torch.isfinite(loss)
    torch.testing.assert_close(
        current.grad.float(), torch.tensor([-0.6, 0.4]), rtol=0.01, atol=0.001
    )


def test_asymmetric_clipping_and_null_fallback():
    current = torch.tensor([-1.0, -6.0], requires_grad=True)
    loss, _ = _callback(clip_ratio_low=0.5, clip_ratio_high=None)(
        [_datum([-3.0, -3.0], [1.0, 1.0], [1, 1])], [current]
    )
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([-0.6, -0.25]))


@pytest.mark.parametrize(
    "params",
    [
        {"clip_ratio": None},
        {"clip_ratio": -0.2},
        {"clip_ratio_high": float("nan")},
        {"clip_ratio_low": float("inf")},
        {"entropy_coeff": 0.1},
        {"loss_agg_mode": "sample-mean"},
    ],
)
def test_invalid_parameters_fail_before_training(params):
    with pytest.raises(ValueError, match="CISPO"):
        _callback(**params)


def test_nonfinite_active_values_fail():
    with pytest.raises(ValueError, match="finite"):
        _callback()([_datum([-2.0], [1.0], [1])], [torch.tensor([float("nan")])])
