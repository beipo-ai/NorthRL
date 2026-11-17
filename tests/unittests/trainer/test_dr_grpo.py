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

"""Dr.GRPO keeps reward scale and a fixed denominator across complete rollouts."""

from array import array
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from northrl.algorithm.advantage.grpo import (
    compute_grpo_advantage_for_trajectories,
    compute_grpo_outcome_advantage,
)
from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import AlgorithmConfig, KLConfig
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyDataDumper,
    _FakeWeaverHolder,
    _FakeWeightSyncController,
    _make_config,
)
from tests.unittests.trainer.test_remote_api_trainer_loss_aggregation import _AccumulatingHolder


def _config(mini_batch_size=None):
    config = _make_config(batch_size=None, mini_batch_size=mini_batch_size)
    config.algorithm.loss.mode = "ppo_clip"
    config.algorithm.loss.reduction = {
        "mode": "fixed_response_budget",
        "params": {"max_response_tokens": 8},
    }
    config.algorithm.advantage = {
        "estimator": {"mode": "grpo", "params": {"norm_adv_by_std": False}}
    }
    config.algorithm.loss.params.clip.ratio = 0.2
    config.algorithm.kl = KLConfig().model_dump()
    config.models = {"reference": None}
    return config


def _trainer(monkeypatch, mini_batch_size=None):
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    trainer = RemoteApiGrpoTrainer(_config(mini_batch_size))
    trainer._weight_sync_controller = _FakeWeightSyncController(sync_mode="sync")
    return trainer


def _trajectory(reward, lengths, run_id=0, group_id="group"):
    return Trajectory(
        reward=reward,
        segments=[
            {
                "tokens": [1] + [2] * length,
                "loss_mask": [0] + [1] * length,
                "old_log_probs": [0.0] + [-0.1] * length,
            }
            for length in lengths
        ],
        extra_fields={"group_id": group_id, "run_id": run_id},
    )


@pytest.mark.parametrize("use_run_ids", [False, True])
def test_centering_retains_reward_scale_and_ignores_segment_counts(use_run_ids):
    trajectories = [_trajectory(1, [1, 1]), _trajectory(3, [1], 1), _trajectory(8, [], 2)]
    compute_grpo_advantage_for_trajectories(
        trajectories, use_run_ids=use_run_ids, norm_adv_by_std=False
    )
    assert [float(t["advantage"]) for t in trajectories] == pytest.approx([-3, -1, 4])
    scaled = [_trajectory(10 * t.reward, [], i) for i, t in enumerate(trajectories)]
    compute_grpo_advantage_for_trajectories(scaled, use_run_ids=use_run_ids, norm_adv_by_std=False)
    assert [float(t["advantage"]) for t in scaled] == pytest.approx([-30, -10, 40])


def test_tensor_centering_deduplicates_segments_and_preserves_mask_and_std_metrics():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 3.0], [0.0, 9.0]])
    mask = torch.tensor([[1, 1], [0, 1], [1, 0], [1, 1]])
    advantages, returns, stds = compute_grpo_outcome_advantage(
        rewards,
        mask,
        torch.tensor([0, 0, 0, 1]),
        run_ids=torch.tensor([0, 0, 1, 0]),
        norm_adv_by_std=False,
    )
    torch.testing.assert_close(advantages, torch.tensor([[-1.0, -1.0], [0, -1], [1, 0], [0, 0]]))
    assert returns is advantages
    assert stds[0].item() == pytest.approx(2**0.5)
    assert stds[1].item() == 0


@pytest.mark.parametrize("packed", [False, True])
def test_fixed_denominator_matches_clipped_objective_and_gradient(monkeypatch, packed):
    trainer = _trainer(monkeypatch)
    advantages = torch.tensor([[2.0, -3.0, 0.0], [-2.0, 4.0, 0.0]], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    ratios = torch.tensor([[0.6, 1.4, 1.1], [1.05, 0.7, 1.2]], dtype=torch.float64)
    log_ratios = ratios.log().requires_grad_()
    datums = [
        {
            "input_tokens": [1, 2, 3],
            "loss_fn_inputs": {
                "advantages": array("d", adv) if packed else adv,
                "loss_mask": row,
            },
            "_dr_grpo_denominator": 2 * 8,
        }
        for adv, row in zip(advantages.tolist(), mask.tolist(), strict=True)
    ]
    original = deepcopy(datums)
    normalized = trainer._normalize_advantages_for_sum(datums)
    scaled = torch.tensor(
        [list(d["loss_fn_inputs"]["advantages"]) for d in normalized], dtype=torch.float64
    )

    def objective(adv):
        ratio = log_ratios.exp()
        return (torch.maximum(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)) * mask).sum()

    expected = objective(advantages) / 16
    actual = objective(scaled)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected_gradient = torch.autograd.grad(expected, log_ratios, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, log_ratios)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert actual_gradient.count_nonzero() > 0
    assert datums == original
    assert isinstance(normalized[0]["loss_fn_inputs"]["advantages"], array if packed else list)


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("mini_batch_size", [None, 1])
def test_train_uses_parent_count_including_empty_responses(monkeypatch, segmented, mini_batch_size):
    trainer = _trainer(monkeypatch, mini_batch_size)
    holder = _AccumulatingHolder()
    trainer.set_service_holder(holder)
    trajectories = [
        _trajectory(1, [2]),
        _trajectory(3, [1, 3] if segmented else [4], 1),
        _trajectory(7, [], 0, "empty"),
        _trajectory(7, [], 1, "empty"),
    ]
    metrics, _, _ = run_coro(trainer.train(trajectories))
    datums = [datum for call in holder.builtin_calls for datum in call["datums_data"]]
    assert all(datum["_dr_grpo_denominator"] == 4 * 8 for datum in datums)
    assert [value for datum in datums for value in datum["loss_fn_inputs"]["advantages"]] == (
        [-1 / 32] * 2 + [1 / 32] * 4
    )
    assert all(call["loss_fn"] == "ppo_clip" for call in holder.builtin_calls)
    assert all(call["loss_fn_config"]["loss_agg_mode"] == "sum" for call in holder.builtin_calls)
    assert all(call["loss_fn_config"]["clip_ratio"] == 0.2 for call in holder.builtin_calls)
    assert metrics["training/optimizer_step_count"] == 1
    assert len(holder.optim_calls) == 1
    assert holder.gradients == pytest.approx([-torch.exp(torch.tensor(0.2)).item() * 2 / 32])


@pytest.mark.parametrize(
    ("failed_runs", "expected_count", "expected_replacements"),
    [
        ({1, 2}, 2, 0),  # More failures than donors: exclude the entire mixed group.
        ({2}, 5, 1),  # One donor repairs the mixed group: count every rollout slot.
    ],
)
def test_denominator_tracks_group_eligibility(
    monkeypatch, failed_runs, expected_count, expected_replacements
):
    trainer = _trainer(monkeypatch)
    holder = _AccumulatingHolder()
    trainer.set_service_holder(holder)
    trajectories = [_trajectory(1, [2], 0, "good"), _trajectory(3, [2], 1, "good")]
    for run_id in range(3):
        trajectory = _trajectory(5 + run_id, [2], run_id, "mixed")
        if run_id in failed_runs:
            trajectory.status = "failed"
        trajectories.append(trajectory)
    for trajectory in trajectories:
        trajectory["rollout_id"] = f"{trajectory['group_id']}-{trajectory['run_id']}"

    metrics, _, _ = run_coro(trainer.train(trajectories))

    datums = [datum for call in holder.builtin_calls for datum in call["datums_data"]]
    assert len(datums) == expected_count
    assert {datum["_dr_grpo_denominator"] for datum in datums} == {expected_count * 8}
    assert metrics["batch/resampled_trajectory_count"] == expected_replacements
    assert metrics["training/optimizer_step_count"] == 1


def test_rejection_does_not_rescale_surviving_tokens_or_trajectories(monkeypatch):
    trainer = _trainer(monkeypatch)
    trajectories = [_trajectory(1, [2]), _trajectory(3, [4], 1)]
    prepared = run_coro(trainer._prepare_trajectories(trajectories, {}))
    prepared = trainer._advantage_estimator.estimate(prepared, {}, {})
    prepared[0].segments[0].update(correction_weights=[0.0, 1.0, 1.0], correction_mask=[0, 0, 0])
    prepared[1].segments[0].update(
        correction_weights=[0.0, 1.0, 1.0, 1.0, 1.0], correction_mask=[0, 1, 0, 1, 0]
    )
    datums = trainer._normalize_advantages_for_sum(trainer._convert_to_datums(prepared))
    assert len(datums) == 1
    assert trainer._last_datum_sources == [(1, 0)]
    assert datums[0]["loss_fn_inputs"]["loss_mask"] == [1, 0, 1, 0]
    assert datums[0]["_dr_grpo_denominator"] == 16
    assert datums[0]["loss_fn_inputs"]["advantages"] == [1 / 16] * 4


def test_all_masked_batch_skips_optimizer(monkeypatch):
    trainer = _trainer(monkeypatch)
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    trajectories = [_trajectory(1, [1]), _trajectory(3, [1], 1)]
    for trajectory in trajectories:
        trajectory.segments[0]["loss_mask"] = [0, 0]
    metrics, _, _ = run_coro(trainer.train(trajectories))
    assert metrics["training/skipped_flag"] == 1
    assert not holder.builtin_calls and not holder.optim_calls


def test_dr_grpo_mismatch_diagnostics_keep_token_mean(monkeypatch):
    trainer = _trainer(monkeypatch)
    sequences = [_trajectory(1, [2]).segments[0], _trajectory(3, [4], 1).segments[0]]
    metrics = {}
    trainer._record_mismatch_metrics(
        sequences,
        [sequence["old_log_probs"] for sequence in sequences],
        [[-0.2] * 2, [-0.4] * 4],
        metrics,
    )
    assert metrics["mis/metrics_available"] == 1
    assert metrics["mis/training_log_ppl"] == pytest.approx((2 * 0.2 + 4 * 0.4) / 6)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss.reduction.params.max_response_tokens", None),
        ("loss.reduction.params.max_response_tokens", 0),
        ("loss.reduction.params.max_response_tokens", -1),
        ("loss.reduction.params.max_response_tokens", True),
        ("loss.reduction.params.max_response_tokens", 1.5),
        ("loss.reduction.params.max_response_tokens", "8"),
        ("advantage.estimator.params.norm_adv_by_std", True),
        ("loss.params.clip", {}),
        ("loss.params.clip.ratio", None),
        ("loss.params.clip.ratio", 0.0),
        ("loss.params.clip.ratio", -0.2),
        ("loss.params.clip.ratio", float("nan")),
        ("loss.params.clip.ratio", float("inf")),
        ("loss.params.clip.ratio", float("-inf")),
        ("loss.params.clip.ratio_low", 0.1),
        ("loss.params.clip.ratio_high", 0.3),
        ("loss.params.clip.ratio_c", 3.0),
        ("loss.params.clip.truncation_c", 2.0),
        ("type", "opd"),
        ("loss.mode", "importance_sampling"),
        ("entropy_coeff", 1e-4),
        ("kl.mode", "in_loss"),
        ("loss.mode", "custom"),
        ("optim.batch_size", 8),
    ],
)
def test_invalid_dr_grpo_configuration_rejected_before_setup(monkeypatch, field, value):
    config = _config()
    OmegaConf.update(config, "algorithm." + field, value, merge=False)
    dumper = Mock()
    monkeypatch.setattr(trainer_module, "get_data_dumper", dumper)
    with pytest.raises(ValidationError):
        RemoteApiGrpoTrainer(config)
    dumper.assert_not_called()


@pytest.mark.parametrize("ratio", [0.2, 1.5])
def test_dr_grpo_accepts_finite_positive_symmetric_clip(ratio):
    config = _config()
    config.algorithm.loss.params.clip.ratio = ratio
    validated = AlgorithmConfig.model_validate(
        OmegaConf.to_container(config.algorithm, resolve=True)
    )
    assert validated.loss.params["clip"]["ratio"] == ratio


@pytest.mark.parametrize("algorithm_type", ["opd", "sft"])
def test_non_grpo_rejects_group_std_normalization_parameter(algorithm_type):
    with pytest.raises(ValidationError, match="norm_adv_by_std"):
        AlgorithmConfig(
            type=algorithm_type,
            optim={"lr": 1e-6},
            advantage={
                "estimator": {
                    "mode": "opd" if algorithm_type == "opd" else None,
                    "params": {"norm_adv_by_std": False},
                }
            },
        )


def test_grpo_can_disable_group_std_without_dr_grpo_aggregation():
    config = AlgorithmConfig(
        type="grpo",
        optim={"lr": 1e-6},
        advantage={"estimator": {"mode": "grpo", "params": {"norm_adv_by_std": False}}},
    )
    assert config.loss.reduction.mode == "token_mean"
    assert config.advantage.estimator.params["norm_adv_by_std"] is False


@pytest.mark.parametrize("algorithm_type", ["grpo", "opd", "sft"])
def test_token_mean_has_no_unused_response_budget(algorithm_type):
    config = AlgorithmConfig(type=algorithm_type, optim={"lr": 1e-6})
    assert config.loss.reduction.params == {}
    with pytest.raises(ValidationError, match="max_response_tokens"):
        AlgorithmConfig(
            type=algorithm_type,
            optim={"lr": 1e-6},
            loss={
                "mode": "cross_entropy" if algorithm_type == "sft" else "importance_sampling",
                "reduction": {"mode": "token_mean", "params": {"max_response_tokens": 8}},
            },
        )
