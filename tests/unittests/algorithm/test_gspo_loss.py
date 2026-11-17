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

"""Numerical and gradient checks against the sequence-level GSPO objective."""

import math
from types import SimpleNamespace

import pytest
import torch

from northrl.utils.metric_catalog import normalize_step_metrics, prepare_wandb_metrics
from northrl.utils.service_metrics import aggregate_service_metrics
from northrl.weaver.custom_losses import resolve_custom_loss


def _datum(old, advantages, mask):
    return SimpleNamespace(
        loss_fn_inputs={"logprobs": old, "advantages": advantages, "loss_mask": mask}
    )


def _loss(low=0.2, high=0.3):
    return resolve_custom_loss(
        {"name": "gspo", "params": {"clip_ratio_low": low, "clip_ratio_high": high}}
    )


def test_gspo_matches_independent_sequence_reference_and_gradients():
    old = [[-2.0, -1.5, -3.0], [-1.0, -4.0], [-1.0]]
    # Token ratios vary within each response; token PPO and detached MIS would
    # both produce different gradients. The third row must stay in the mean.
    new = [
        torch.tensor(
            [-2.0 + math.log(1.8), -1.5 + math.log(0.7), -3.0],
            dtype=torch.float64,
            requires_grad=True,
        ),
        torch.tensor(
            [-1.0 + math.log(0.6), -4.0 + math.log(1.8)], dtype=torch.float64, requires_grad=True
        ),
        torch.tensor([-0.9], dtype=torch.float64, requires_grad=True),
    ]
    advantages = [2.0, -1.5, 0.0]
    masks = [[1, 1, 0], [1, 1], [1]]
    datums = [_datum(o, [a] * len(o), m) for o, a, m in zip(old, advantages, masks, strict=True)]

    loss, metrics = _loss()(datums, new)
    expected_losses = []
    expected_grads = []
    for values, before, advantage, mask in zip(new, old, advantages, masks, strict=True):
        selected = [i for i, enabled in enumerate(mask) if enabled]
        ratio = math.exp(
            sum(float(values[i].detach()) - before[i] for i in selected) / len(selected)
        )
        clipped_ratio = min(1.3, max(0.8, ratio))
        expected_losses.append(-min(ratio * advantage, clipped_ratio * advantage))
        clipped = ratio * advantage > clipped_ratio * advantage
        gradient = 0.0 if clipped else -advantage * ratio / len(selected) / len(datums)
        expected_grads.append([gradient if enabled else 0.0 for enabled in mask])

    assert loss.item() == pytest.approx(sum(expected_losses) / len(datums), abs=1e-12)
    loss.backward()
    for values, expected in zip(new, expected_grads, strict=True):
        torch.testing.assert_close(values.grad, torch.tensor(expected, dtype=torch.float64))
    assert metrics["gspo/active_trajectories"] == 3
    assert metrics["tokens"] == 5


@pytest.mark.parametrize("ratio,advantage,expected", [(1.7, 2.0, -2.6), (0.6, -2.0, 1.6)])
def test_clipped_positive_and_negative_sequences_have_zero_gradient(ratio, advantage, expected):
    values = torch.tensor([math.log(ratio)] * 3, requires_grad=True)
    loss, metrics = _loss()([_datum([0.0] * 3, [advantage] * 3, [1] * 3)], [values])
    loss.backward()
    assert loss.item() == pytest.approx(expected)
    assert values.grad is not None
    assert values.grad.tolist() == [0.0] * 3
    assert metrics["gspo/clip_fraction"] == 1.0


def test_sequence_mean_weights_short_and_long_responses_equally_and_ignores_masked_nan():
    short = torch.tensor([float("nan"), math.log(1.1)], requires_grad=True)
    long = torch.tensor([math.log(0.9)] * 4, requires_grad=True)
    empty = torch.tensor([float("nan")], requires_grad=True)
    data = [
        _datum([float("nan"), 0.0], [float("nan"), 2.0], [0, 1]),
        _datum([0.0] * 4, [-1.0] * 4, [1] * 4),
        _datum([float("nan")], [float("nan")], [0]),
    ]
    loss, metrics = _loss()(data, [short, long, empty])
    assert loss.item() == pytest.approx((-2.2 + 0.9) / 2)
    loss.backward()
    torch.testing.assert_close(short.grad, torch.tensor([0.0, -1.1]))
    torch.testing.assert_close(long.grad, torch.full((4,), 0.9 / 8))
    assert empty.grad is not None
    assert empty.grad.tolist() == [0.0]
    assert metrics["gspo/empty_trajectories"] == 1


def test_all_empty_rows_return_differentiable_zero():
    values = torch.tensor([float("nan")], requires_grad=True)
    loss, metrics = _loss()([_datum([], [], [0])], [values])
    loss.backward()
    assert loss.item() == 0.0
    assert values.grad is not None
    assert values.grad.tolist() == [0.0]
    assert metrics["gspo/active_trajectories"] == 0


def test_gspo_passes_autograd_gradcheck():
    values = torch.tensor([-0.95, -1.05], dtype=torch.float64, requires_grad=True)
    datum = _datum([-1.0, -1.0], [2.0, 2.0], [1, 1])
    assert torch.autograd.gradcheck(lambda v: _loss()([datum], [v])[0], (values,))


@pytest.mark.parametrize(
    "data,match",
    [
        (_datum([0, 0], [1, 2], [1, 1]), "constant scalar"),
        (_datum([0, 0], [1, 1], [1, 0.5]), "only 0 or 1"),
        (_datum([0, float("nan")], [1, 1], [1, 1]), "finite active"),
        (_datum([0], [1, 1], [1, 1]), "shape"),
    ],
)
def test_gspo_rejects_invalid_active_inputs(data, match):
    with pytest.raises(ValueError, match=match):
        _loss()([data], [torch.zeros(2, requires_grad=True)])


@pytest.mark.parametrize("low,high", [(-0.1, 0.2), (1.0, 0.2), (0.2, -0.1), (0.2, float("inf"))])
def test_gspo_rejects_invalid_clip_bounds(low, high):
    with pytest.raises(ValueError, match="GSPO clip"):
        _loss(low, high)


def _owned(datum, parent, index, count):
    datum.metadata = {
        "northrl_gspo": {"trajectory_id": parent, "segment_index": index, "segment_count": count}
    }
    return datum


def test_segmented_trajectory_matches_unsplit_loss_and_gradients():
    raw = [math.log(2.0), math.log(0.5), math.log(1.5)]
    full = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    other = torch.tensor([math.log(0.8), math.log(1.3)], dtype=torch.float64, requires_grad=True)
    expected, _ = _loss()(
        [_datum([0.0] * 3, [2.0] * 3, [1] * 3), _datum([0.0] * 2, [-1.0] * 2, [1] * 2)],
        [full, other],
    )
    expected.backward()

    first = torch.tensor(raw[:1], dtype=torch.float64, requires_grad=True)
    second = torch.tensor(raw[1:], dtype=torch.float64, requires_grad=True)
    empty = torch.tensor([float("nan")], dtype=torch.float64, requires_grad=True)
    other_split = other.detach().clone().requires_grad_(True)
    # Interleaving parents also catches an implementation that groups adjacent rows only.
    data = [
        _owned(_datum([0.0], [2.0], [1]), "a", 0, 3),
        _owned(_datum([0.0] * 2, [-1.0] * 2, [1] * 2), "b", 0, 1),
        _owned(_datum([0.0], [0.0], [0]), "a", 1, 3),
        _owned(_datum([0.0] * 2, [2.0] * 2, [1] * 2), "a", 2, 3),
    ]
    actual, metrics = _loss()(data, [first, other_split, empty, second])
    actual.backward()
    torch.testing.assert_close(actual, expected)
    assert first.grad is not None and second.grad is not None
    torch.testing.assert_close(torch.cat([first.grad, second.grad]), full.grad)
    torch.testing.assert_close(other_split.grad, other.grad)
    assert empty.grad is not None
    assert empty.grad.tolist() == [0.0]
    assert metrics["gspo/active_trajectories"] == 2
    assert metrics["gspo/empty_trajectories"] == 0


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "inconsistent_count", "mixed_metadata", "different_advantage"]
)
def test_invalid_trajectory_grouping_fails_closed(case):
    first = _owned(_datum([0.0], [1.0], [1]), "parent", 0, 2)
    second = _owned(_datum([0.0], [1.0], [1]), "parent", 1, 2)
    data = [first, second]
    if case == "missing":
        data.pop()
    elif case == "duplicate":
        second.metadata["northrl_gspo"]["segment_index"] = 0
    elif case == "inconsistent_count":
        second.metadata["northrl_gspo"]["segment_count"] = 3
    elif case == "mixed_metadata":
        second.metadata = {}
    else:
        second.loss_fn_inputs["advantages"] = [2.0]
    with pytest.raises(ValueError, match="GSPO"):
        _loss()(data, [torch.zeros(1, requires_grad=True) for _ in data])


def test_gspo_metrics_pool_by_trajectory_count_and_survive_logging():
    records = []
    for ratios, lengths in [([1.1, 0.9], [1, 4]), ([1.5], [1])]:
        data = [_datum([0.0] * length, [1.0] * length, [1] * length) for length in lengths]
        values = [
            torch.full((length,), math.log(ratio), requires_grad=True)
            for ratio, length in zip(ratios, lengths, strict=True)
        ]
        _, metrics = _loss()(data, values)
        records.append(metrics)
    combined = aggregate_service_metrics(records)
    assert combined["gspo/ratio_mean"] == pytest.approx((1.1 + 0.9 + 1.5) / 3)
    assert combined["gspo/ratio_min"] == pytest.approx(0.9)
    assert combined["gspo/ratio_max"] == pytest.approx(1.5)
    assert combined["gspo/clip_fraction"] == pytest.approx(1 / 3)
    assert combined["gspo/active_trajectories"] == 3
    assert combined["gspo/empty_trajectories"] == 0
    assert combined["loss"] == pytest.approx(-(1.1 + 0.9 + 1.3) / 3)
    logged = normalize_step_metrics({f"weaver/{key}": value for key, value in combined.items()})
    assert logged["algorithm/weaver/gspo/active_trajectories"] == 3
    assert logged["algorithm/weaver/gspo/ratio_mean"] == combined["gspo/ratio_mean"]
    assert any("gspo/ratio_mean" in key for key in prepare_wandb_metrics(logged))
