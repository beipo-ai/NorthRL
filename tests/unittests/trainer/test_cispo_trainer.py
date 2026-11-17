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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import AlgorithmConfig, KLConfig, NorthRLConfig
from northrl.weaver.custom_losses import resolve_custom_loss
from northrl.weaver.loss_adapter import build_loss_adapter
from tests.unittests.algorithm.test_tool_error_rebalance import PREFIX, _annotation
from tests.unittests.trainer.test_prompt_mean import _round
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyDataDumper,
    _FakeWeightSyncController,
    _make_config,
)
from tests.unittests.trainer.test_remote_api_trainer_loss_aggregation import _AccumulatingHolder
from tests.unittests.utils.test_config_schema import _load_recipe, _load_reference_recipe


def _config(mini_batch_size=None):
    config = _make_config(batch_size=None, mini_batch_size=mini_batch_size)
    config.algorithm.loss = {
        "mode": "cispo",
        "params": {"clip": {"ratio": 0.2}},
        "reduction": {"mode": "token_mean"},
    }
    config.algorithm.kl = KLConfig().model_dump()
    config.models = {"reference": None}
    config.backend = {"type": "weaver"}
    config.training.weight_sync = {"mode": "sync"}
    return config


class _CispoHolder(_AccumulatingHolder):
    def __init__(self):
        super().__init__()
        self.observations = []

    async def forward_backward_custom(self, *, datums_data, custom_loss_spec, **kwargs):
        self.custom_calls.append(
            {"datums_data": datums_data, "custom_loss_spec": custom_loss_spec, **kwargs}
        )
        datums = [SimpleNamespace(loss_fn_inputs=d["loss_fn_inputs"]) for d in datums_data]
        logprobs = [
            self.parameter
            * torch.tensor(d.loss_fn_inputs["target_tokens"], dtype=torch.float64)
            / 100
            - 2
            for d in datums
        ]
        loss, metrics = resolve_custom_loss(custom_loss_spec)(datums, logprobs)
        self.observations.append(metrics)
        loss.backward()
        return {"loss": loss.item(), "metrics": metrics}


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("mini_batch_size", [None, 1, 3])
def test_cispo_real_grpo_pipeline_matches_analytic_gradient(monkeypatch, split, mini_batch_size):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    trainer = RemoteApiGrpoTrainer(_config(mini_batch_size))
    trainer._weight_sync_controller = _FakeWeightSyncController()
    holder = _CispoHolder()
    trainer.set_service_holder(holder)
    trajectories = _round(split=split)
    metrics, _, _ = run_coro(trainer.train(trajectories))

    # All current/old ratios are below 0.8 here. The analytic gradient still
    # includes every response token, with one denominator over the entire batch.
    terms = []
    expected_loss = 0.0
    for trajectory in trajectories:
        advantage = float(trajectory["advantage"])
        for segment in trajectory.segments:
            for token, keep in zip(segment["tokens"][1:], segment["loss_mask"][1:]):
                if keep:
                    terms.append(-0.8 * advantage * token / 100)
                    expected_loss += -0.8 * advantage * (0.2 * token / 100 - 2)
    gradient = sum(terms) / len(terms)
    assert abs(gradient) > 0.01
    assert holder.gradients == pytest.approx([gradient])
    assert holder.parameter.item() == pytest.approx(0.2 - 0.1 * gradient)
    assert metrics["weaver/loss"] == pytest.approx(expected_loss / len(terms))
    assert holder.builtin_calls == []
    assert len(holder.custom_calls) == len(holder.optim_calls) == 1
    call = holder.custom_calls[0]
    assert call["custom_loss_spec"]["name"] == "cispo"
    if mini_batch_size and len(call["datums_data"]) > mini_batch_size:
        assert max(call["mini_batch_sizes"]) <= mini_batch_size


def test_cispo_reports_weighted_loss_and_clipping_across_optimizer_batches(monkeypatch):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda items: None)
    config = _config(mini_batch_size=1)
    config.algorithm.optim.batch_size = 2
    trainer = RemoteApiGrpoTrainer(config)
    trainer._weight_sync_controller = _FakeWeightSyncController()
    holder = _CispoHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(_round()))
    assert len(holder.optim_calls) == 2
    tokens = sum(record["tokens"] for record in holder.observations)
    for key in ("loss", "ppo_clip/clipfrac", "ppo_clip/ppo_kl"):
        expected = sum(record[key] * record["tokens"] for record in holder.observations) / tokens
        metric_name = f"weaver/{key}" if key == "loss" else f"algorithm/weaver/{key}"
        assert metrics[metric_name] == pytest.approx(expected)


@pytest.mark.parametrize(
    "path, value, message",
    [
        ("entropy_coeff", 0.01, "entropy_coeff=0"),
        ("kl.mode", "in_loss", "KL disabled"),
        ("mis.enabled", True, "mis.enabled=true"),
        ("loss.reduction.mode", "datum_mean", "loss.reduction.mode='token_mean'"),
        ("loss.params.clip", {}, "requires explicit clip"),
        ("loss.params.clip.ratio", -0.2, "greater than or equal to 0"),
        ("loss.params.clip.ratio", float("nan"), "finite"),
        ("loss.params.clip.ratio_c", 3.0, "ratio_c"),
        ("type", "opd", "type='grpo'"),
        ("optim.submit_ahead", 1, "submit_ahead=0"),
    ],
)
def test_config_rejects_unsupported_or_ambiguous_cispo_options(path, value, message):
    config = _config().algorithm
    OmegaConf.update(config, path, value, merge=False)
    with pytest.raises(ValueError, match=message):
        AlgorithmConfig.model_validate(OmegaConf.to_container(config))


@pytest.mark.parametrize("selector", ["name", "base_loss_fn"])
def test_cispo_cannot_bypass_contract_via_custom_loss(selector):
    config = _config().algorithm
    config.loss = {"mode": "custom", "params": {selector: "cispo"}}
    with pytest.raises(ValueError, match="Select CISPO"):
        AlgorithmConfig.model_validate(OmegaConf.to_container(config))


def test_cispo_backend_checked_by_schema():
    root = _load_reference_recipe()
    root["algorithm"].update(
        loss={"mode": "cispo", "params": {"clip": {"ratio": 0.2}}}, entropy_coeff=0.0
    )
    NorthRLConfig.model_validate(root)
    root["backend"]["type"] = "mock"
    with pytest.raises(ValueError, match="CISPO requires backend"):
        NorthRLConfig.model_validate(root)


@pytest.mark.parametrize("value", [True, False, "0.2", float("inf"), -0.1])
def test_canonical_cispo_clip_is_strict_and_finite(value):
    with pytest.raises(ValueError):
        AlgorithmConfig.model_validate(
            {
                "type": "grpo",
                "optim": {"lr": 1e-6},
                "entropy_coeff": 0,
                "loss": {"mode": "cispo", "params": {"clip": {"ratio": value}}},
            }
        )


def test_cispo_adapter_selects_callback_without_custom_loss_switch():
    algorithm = AlgorithmConfig.model_validate(
        {
            "type": "grpo",
            "optim": {"lr": 1e-6},
            "entropy_coeff": 0,
            "loss": {"mode": "cispo", "params": {"clip": {"ratio": 0.2}}},
        }
    )
    loss = build_loss_adapter(algorithm)
    assert loss.custom == {
        "enabled": True,
        "name": "cispo",
        "params": {
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "entropy_coeff": 0.0,
            "loss_agg_mode": "token-mean",
        },
    }
    assert loss.runtime_reduction == "token-mean"
    assert algorithm.loss.mode == "cispo"


@pytest.mark.parametrize("split", [False, True])
def test_cispo_uses_actual_token_advantages_from_direct_postprocessor(monkeypatch, split):
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    trainer = RemoteApiGrpoTrainer(_config(mini_batch_size=1))
    trainer._weight_sync_controller = _FakeWeightSyncController()
    holder = _CispoHolder()
    trainer.set_service_holder(holder)
    visits = []
    terms = []

    class TokenShaping:
        def process(self, trajectories, metrics, **kwargs):
            visits.append(len(trajectories))
            for trajectory in trajectories:
                for sequence in trajectory.segments:
                    values = []
                    for token, keep in zip(sequence["tokens"][1:], sequence["loss_mask"][1:]):
                        if keep:
                            # Includes zero, positive and negative token advantages.
                            value = (len(terms) % 3) - 1.0
                            values.append(value)
                            terms.append(-0.8 * value * token / 100)
                    sequence["token_advantages"] = values
            return trajectories

    trainer._advantage_postprocessor = TokenShaping()
    run_coro(trainer.train(_round(split=split)))
    assert visits == [4]  # One whole-step call, before request splitting.
    assert holder.gradients == pytest.approx([sum(terms) / len(terms)])
    assert holder.observations[0]["tokens"] == len(terms)


@pytest.mark.parametrize("split", [False, True])
def test_cispo_real_tool_postprocessing_preserves_token_advantages_and_zero_rows(
    monkeypatch, split
):
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    config = _config(mini_batch_size=1)
    config.algorithm.advantage = {
        "estimator": {"mode": "grpo", "params": {"norm_adv_by_std": False}},
        "postprocessing": {"mode": "tool_error_rebalance"},
    }
    trainer = RemoteApiGrpoTrainer(config)
    trainer._weight_sync_controller = _FakeWeightSyncController()
    holder = _CispoHolder()
    trainer.set_service_holder(holder)
    trajectories = _round(split=split)
    for trajectory in trajectories:
        for sequence in trajectory.segments:
            sequence["turn_annotations"] = [
                _annotation(
                    index, index + 1, reasons=["missing_tool_name"] if token in {101, 104} else []
                )
                for index, (token, keep) in enumerate(
                    zip(sequence["tokens"], sequence["loss_mask"])
                )
                if keep
            ]
    metrics, _, _ = run_coro(trainer.train(trajectories))
    submitted = {
        token: advantage
        for datum in holder.custom_calls[0]["datums_data"]
        for token, advantage, keep in zip(
            datum["loss_fn_inputs"]["target_tokens"],
            datum["loss_fn_inputs"]["advantages"],
            datum["loss_fn_inputs"]["loss_mask"],
        )
        if keep
    }
    expected = {101: -1.0, 102: -1 / 3, 103: -1 / 3, 104: 0.0, 105: -1 / 3}
    expected.update({token: 7 / 12 for token in range(106, 112)})
    assert submitted == pytest.approx(expected)
    assert metrics[PREFIX + "response_tokens"] == 11
    assert holder.observations[0]["tokens"] == 11
    assert holder.gradients == pytest.approx(
        [sum(-0.8 * advantage * token / 100 for token, advantage in expected.items()) / 11]
    )


def test_cispo_recipe_and_documented_algorithm_block_validate():
    root = Path(__file__).resolve().parents[3]
    recipe = NorthRLConfig.model_validate(_load_recipe(root / "recipe/math/weaver_cispo.yaml"))
    assert recipe.algorithm.loss.mode == "cispo"
    assert recipe.algorithm.advantage.estimator.mode == "grpo"
    assert recipe.algorithm.optim.submit_ahead == 0
    for language in ("en", "zh"):
        page = root / f"docs/{language}/developer-guide/06-trainers/cispo.md"
        block = page.read_text().split("```yaml\n", 1)[1].split("```", 1)[0]
        algorithm = AlgorithmConfig.model_validate(yaml.safe_load(block)["algorithm"])
        assert algorithm.loss.mode == recipe.algorithm.loss.mode
        assert algorithm.loss.params == recipe.algorithm.loss.params


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_cispo_retains_reinforce_native_loss_restriction(estimator):
    config = _config()
    config.algorithm.advantage = {"estimator": {"mode": estimator}}
    with pytest.raises(ValueError, match=r"REINFORCE\+\+ requires native"):
        RemoteApiGrpoTrainer(config)
