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

"""RLOO formula, component configuration and full-step execution contracts."""

from copy import deepcopy

import pytest
import torch
from pydantic import ValidationError

from northrl.algorithm.advantage import build_estimator
from northrl.algorithm.advantage.rloo import RlooEstimator
from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import AlgorithmConfig, NorthRLConfig
from northrl.utils.trajectory_sequences import iter_training_sequences
from tests.unittests.trainer.test_group_resampling import _group
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyActivityTracker,
    _DummyDataDumper,
    _FakeWeaverHolder,
    _FakeWeightSyncController,
    _make_config,
)
from tests.unittests.utils.test_config_schema import _load_reference_recipe


def _parents(rewards=(1.0, 2.0, 4.0), *, model="actor"):
    return [
        Trajectory(
            segments=[
                {"tokens": [10 + run, 20], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.1]}
                for _ in range(run)
            ],
            reward=reward,
            extra_fields={
                "model_tag": model,
                "group_id": "a",
                "run_id": run,
                "rollout_id": f"{model}-{run}",
                "expected_rollouts": len(rewards),
            },
        )
        for run, reward in enumerate(rewards)
    ]


def test_rloo_counts_each_parent_once_including_empty_and_separates_models():
    parents = _parents() + _parents((10.0, 14.0), model="other")
    buffers = [(seq, seq["tokens"], seq["loss_mask"]) for seq in iter_training_sequences(parents)]
    result = RlooEstimator().estimate(parents, {}, {})
    assert result is parents
    assert [float(parent["advantage"]) for parent in parents] == pytest.approx(
        [-2, -0.5, 2.5, -4, 4]
    )
    assert all(parent["advantage"].dtype == torch.float32 for parent in parents)
    for seq, tokens, mask in buffers:
        assert seq["tokens"] is tokens and seq["loss_mask"] is mask
        assert "token_advantages" not in seq
    assert [float(seq["advantage"]) for seq in parents[2].segments] == [2.5, 2.5]


@pytest.mark.parametrize("rewards", [(3.0,), (2.0, 2.0, 2.0)])
def test_rloo_constant_and_singleton_groups_have_zero_advantages(rewards):
    parents = _parents(rewards)
    RlooEstimator().estimate(parents, {}, {})
    assert all(float(parent["advantage"]) == 0 for parent in parents)


@pytest.mark.parametrize("failure", ["duplicate", "incomplete", "reward", "mixed_skip"])
def test_rloo_preserves_complete_group_validation_before_estimation(failure):
    parents = _parents()
    if failure == "duplicate":
        parents[1]["rollout_id"] = parents[0]["rollout_id"]
    elif failure == "incomplete":
        parents.pop()
    elif failure == "reward":
        parents[0].reward = float("nan")
    else:
        parents[0]["group_skipped"] = True
    with pytest.raises(ValueError):
        RlooEstimator().estimate(parents, {}, {})
    assert all("advantage" not in parent for parent in parents)


def test_rloo_skips_failed_groups_and_repaired_rewards_define_baselines():
    failed = _parents()
    failed[0].status = "failed"
    RlooEstimator().estimate(failed, {}, {})
    assert all(parent["group_skipped"] and "advantage" not in parent for parent in failed)
    repaired = RemoteApiGrpoTrainer._resample_failed_groups(_group(), {})
    RlooEstimator().estimate(repaired, {}, {})
    assert [float(parent["advantage"]) for parent in repaired[:2]] == pytest.approx([2 / 3, -2 / 3])


def test_rloo_canonical_round_trip_does_not_mutate_input():
    raw = {"type": "grpo", "advantage": {"estimator": {"mode": "rloo"}}, "optim": {"lr": 1e-6}}
    original = deepcopy(raw)
    config = AlgorithmConfig.model_validate(raw)
    assert config.advantage.estimator.model_dump() == {"mode": "rloo", "params": {}}
    assert AlgorithmConfig.model_validate(config.model_dump()) == config
    assert raw == original
    assert isinstance(build_estimator(config.advantage.estimator), RlooEstimator)


@pytest.mark.parametrize(
    "legacy",
    [
        {"adv_estimator": "rloo"},
        {"norm_adv_by_std_in_grpo": True},
        {"norm_adv_by_std_in_grpo": False},
    ],
)
def test_rloo_rejects_legacy_estimator_settings(legacy):
    with pytest.raises(ValidationError, match="Extra inputs"):
        AlgorithmConfig(
            type="grpo", optim={"lr": 1e-6}, advantage={"estimator": {"mode": "rloo"}}, **legacy
        )


@pytest.mark.parametrize("flow", ["opd", "sft"])
def test_rloo_requires_reward_training_flow(flow):
    with pytest.raises(ValidationError, match="requires advantage.estimator"):
        AlgorithmConfig(type=flow, optim={"lr": 1e-6}, advantage={"estimator": {"mode": "rloo"}})


def test_rloo_rejects_unused_grpo_params_and_requires_two_configured_rollouts():
    with pytest.raises(ValidationError, match="norm_adv_by_std"):
        AlgorithmConfig(
            type="grpo",
            optim={"lr": 1e-6},
            advantage={"estimator": {"mode": "rloo", "params": {"norm_adv_by_std": True}}},
        )
    recipe = _load_reference_recipe()
    recipe["algorithm"]["advantage"] = {"estimator": {"mode": "rloo"}}
    recipe["data"]["train"]["rollout_repeat_n"] = 2
    assert NorthRLConfig.model_validate(recipe).algorithm.advantage.estimator.mode == "rloo"
    for count in (None, 1):
        recipe["data"]["train"]["rollout_repeat_n"] = count
        with pytest.raises(ValidationError, match="rollout_repeat_n"):
            NorthRLConfig.model_validate(recipe)


@pytest.mark.parametrize("postprocessing", [None, "tool_error_rebalance"])
@pytest.mark.parametrize("batch_size,mini_batch_size", [(None, None), (1, 1), (2, 1)])
def test_rloo_estimates_once_on_the_full_step_before_optimizer_splitting(
    monkeypatch, batch_size, mini_batch_size, postprocessing
):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    config = _make_config(batch_size=batch_size, mini_batch_size=mini_batch_size)
    config.algorithm.advantage = {
        "estimator": {"mode": "rloo"},
        "postprocessing": {"mode": postprocessing},
    }
    config.models = {"reference": None}
    trainer = RemoteApiGrpoTrainer(config)
    trainer._activity_tracker = _DummyActivityTracker()
    trainer._weight_sync_controller = _FakeWeightSyncController()
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    parents = _parents()
    # Prepared trajectories can retain postprocessing from a previous round.
    # A fresh estimate must discard both cached bases and shaped values.
    for sequence in iter_training_sequences(parents):
        sequence["base_token_advantages"] = [999.0]
        sequence["token_advantages"] = [888.0]
    calls = []
    original = trainer._advantage_estimator

    class RecordingEstimator:
        def estimate(self, trajectories, inputs, metrics, *, logger=None):
            calls.append(trajectories)
            assert holder.builtin_calls == holder.optim_calls == []
            return original.estimate(trajectories, inputs, metrics, logger=logger)

    trainer._advantage_estimator = RecordingEstimator()
    run_coro(trainer.train(list(reversed(parents))))
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0], parents, strict=True))
    assert [float(parent["advantage"]) for parent in parents] == pytest.approx([-2, -0.5, 2.5])
    assert holder.builtin_calls
    for sequence in iter_training_sequences(parents):
        if postprocessing:
            assert sequence["base_token_advantages"] == pytest.approx(
                [float(sequence["advantage"])]
            )
        else:
            assert "base_token_advantages" not in sequence
            assert "token_advantages" not in sequence
    denominator = batch_size or 3
    for call in holder.builtin_calls:
        assert call["loss_fn_config"]["loss_agg_mode"] == "sum"
        for datum in call["datums_data"]:
            scalar = {11: -0.5, 12: 2.5}[datum["input_tokens"][0]]
            assert datum["loss_fn_inputs"]["advantages"] == pytest.approx([scalar / denominator])
            assert datum["loss_fn_inputs"]["logprobs"] == [-0.1]
            assert datum["loss_fn_inputs"]["loss_mask"] == [1]


def test_rloo_preserves_cross_entropy_rejection_without_execution_specs():
    with pytest.raises(ValidationError, match="cross_entropy loss does not consume advantages"):
        AlgorithmConfig(
            type="grpo",
            optim={"lr": 1e-6},
            advantage={"estimator": {"mode": "rloo"}},
            loss={"mode": "cross_entropy"},
        )
