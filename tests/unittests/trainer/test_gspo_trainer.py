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

"""Validate GSPO configuration and execute its real Weaver surrogate path offline."""

# Tests inspect the trainer and SDK boundary directly.
# pylint: disable=protected-access

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import AlgorithmConfig, NorthRLConfig
from northrl.weaver.loss_adapter import build_loss_adapter
from tests.unittests.trainer.test_grpo_training_temperature import _holder, _trajectories
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import _DummyDataDumper
from tests.unittests.utils.test_config_schema import _load_recipe

RECIPE = Path(__file__).resolve().parents[3] / "recipe/math/weaver_gspo.yaml"


def _config():
    return NorthRLConfig.model_validate(_load_recipe(RECIPE))


def _trainer(monkeypatch, mini_batch_size):
    config = _config()
    config.algorithm.optim.mini_batch_size = mini_batch_size
    config.rollout.sampling.temperature = 0.7
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    trainer = RemoteApiGrpoTrainer(OmegaConf.create(config.model_dump(by_alias=True)))
    trainer._need_sampling = False
    return trainer


def test_gspo_recipe_validates_and_selects_client_side_loss(monkeypatch):
    trainer = _trainer(monkeypatch, 2)
    assert trainer._custom_loss_spec.to_payload() == {
        "name": "gspo",
        "params": {"clip_ratio_low": 0.0004, "clip_ratio_high": 0.0006, "entropy_coeff": 0.0},
    }
    assert trainer._loss_fn == "gspo"


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("type", "opd", "type='grpo'"),
        ("loss.reduction.mode", "token_mean", "trajectory_mean"),
        ("entropy_coeff", 0.01, "entropy_coeff=0"),
        ("advantage.estimator", {"mode": "reinforce_plus_plus"}, "scalar trajectory estimator"),
        (
            "advantage.estimator",
            {"mode": "reinforce_plus_plus_baseline"},
            "scalar trajectory estimator",
        ),
        ("kl.mode", "in_loss", "KL disabled"),
        ("mis.enabled", True, "mis.enabled=true"),
        ("optim.submit_ahead", 1, "submit_ahead=0"),
        ("optim.batch_size", 4, "optim.batch_size=null"),
        ("loss.params.clip.ratio_c", 2.0, "ratio_c"),
        ("loss.params.clip.truncation_c", 2.0, "truncation_c"),
        ("loss.params.clip.ratio_low", 1.0, "less than 1"),
        ("loss.params.clip.ratio_high", float("nan"), "finite"),
    ],
)
def test_gspo_rejects_unsupported_or_invalid_options(path, value, match):
    config = OmegaConf.create(_config().algorithm.model_dump())
    OmegaConf.update(config, path, value, merge=False)
    with pytest.raises(ValueError, match=match):
        AlgorithmConfig.model_validate(OmegaConf.to_container(config))


def test_gspo_requires_explicit_clipping_and_supports_symmetric_fallback():
    algorithm = _config().algorithm.model_dump()
    algorithm["loss"]["params"]["clip"] = {}
    with pytest.raises(ValueError, match="requires explicit clip.ratio"):
        AlgorithmConfig.model_validate(algorithm)
    algorithm["loss"]["params"]["clip"] = {"ratio": 0.1}
    assert AlgorithmConfig.model_validate(algorithm).loss.params["clip"]["ratio"] == 0.1


def test_gspo_config_rejects_mock_backend():
    config = _config().model_dump(by_alias=True)
    config["backend"]["type"] = "mock"
    with pytest.raises(ValueError, match="backend.type='weaver'"):
        NorthRLConfig.model_validate(config)


@pytest.mark.parametrize("selector", ["name", "base_loss_fn"])
def test_gspo_cannot_bypass_contract_via_custom_loss(selector):
    algorithm = _config().algorithm.model_dump()
    algorithm["loss"] = {"mode": "custom", "params": {selector: "gspo"}}
    with pytest.raises(ValueError, match="Select GSPO"):
        AlgorithmConfig.model_validate(algorithm)


@pytest.mark.parametrize("segmented", [False, True])
def test_gspo_mini_batches_preserve_global_normalization_and_surrogate_gradients(
    monkeypatch, segmented
):
    gradients = []
    for mini_batch_size in (None, 2):
        trainer = _trainer(monkeypatch, mini_batch_size)
        holder = _holder()
        trainer.set_service_holder(holder)
        trajectories = _trajectories()
        # Unequal response lengths catch a second token normalization. The SDK
        # fake evaluates all active logprobs at -0.25, so recomputed ratio = 1.
        trajectories[0].loss_mask = [0, 0, 1]
        if segmented:
            parent = trajectories[0]
            trajectories[0] = Trajectory(
                reward=parent.reward,
                extra_fields=deepcopy(parent.extra_fields),
                segments=[
                    {"tokens": [10, 20], "loss_mask": [0, 0], "old_log_probs": [0.0, -0.5]},
                    # A discarded no-target segment leaves a source-index gap.
                    {"tokens": [10], "loss_mask": [0], "old_log_probs": [0.0]},
                    {"tokens": [20, 30], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.5]},
                    {"tokens": [30, 40], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.5]},
                ],
            )
        trainer._normalize_advantages_for_sum = lambda *a, **kw: pytest.fail(
            "GSPO must not pre-normalize the raw scalar advantages"
        )
        metrics, _, _ = run_coro(trainer.train(trajectories))
        client = holder._training_client
        assert client.optim_count == 1
        assert metrics["training/optimizer_step_count"] == 1
        backward = [request for stage, request in client.requests if stage == "backward"]
        assert len(backward) == (1 if mini_batch_size is None else (3 if segmented else 2))
        assert metrics["algorithm/weaver/gspo/active_trajectories"] == 4
        assert metrics["algorithm/weaver/gspo/ratio_mean"] == pytest.approx(1.0)
        assert all(request["loss_fn"] == "surrogate" for request in backward)
        assert all(
            request["loss_fn_config"] == {"temperature": 0.7} for _, request in client.requests
        )
        backward_datums = [
            datum
            for (stage, _), chunk in zip(client.requests, client.datums, strict=True)
            if stage == "backward"
            for datum in chunk
        ]
        # The SDK uses the callback gradients as surrogate weights. Verify
        # actual payloads, not only that the custom callback was invoked.
        grads = [datum.loss_fn_inputs["surrogate_weights"].tolist() for datum in backward_datums]
        parent_tokens: dict[str, int] = {}
        for datum in backward_datums:
            parent = datum.metadata["northrl_gspo"]["trajectory_id"]
            parent_tokens[parent] = parent_tokens.get(parent, 0) + sum(
                datum.loss_fn_inputs["loss_mask"].tolist()
            )
        for datum, grad in zip(backward_datums, grads, strict=True):
            parent = datum.metadata["northrl_gspo"]["trajectory_id"]
            active = datum.loss_fn_inputs["loss_mask"].tolist()
            advantages = datum.loss_fn_inputs["advantages"].tolist()
            # Weaver surrogate weights are d(loss)/d(logprob).
            expected = [
                -advantage / parent_tokens[parent] / 4 if mask else 0.0
                for advantage, mask in zip(advantages, active, strict=True)
            ]
            torch.testing.assert_close(torch.tensor(grad), torch.tensor(expected))
        if segmented:
            ownership = [datum.metadata["northrl_gspo"] for datum in backward_datums[:2]]
            assert ownership == [
                {"trajectory_id": "0", "segment_index": 0, "segment_count": 2},
                {"trajectory_id": "0", "segment_index": 1, "segment_count": 2},
            ]
        gradients.append(grads)
    assert gradients[0] == gradients[1]


def test_gspo_accepts_one_nested_complete_sequence_and_retains_zero_advantage(monkeypatch):
    trainer = _trainer(monkeypatch, 2)
    trajectory = Trajectory(
        segments=[{"tokens": [1, 2], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.2]}],
        reward=1.0,
        extra_fields={"group_id": "g", "run_id": 0, "advantage": 0.0},
    )
    datums = trainer._convert_to_datums([trajectory])
    assert len(datums) == 1
    assert datums[0]["loss_fn_inputs"]["advantages"] == [0.0]


def test_gspo_skips_optimizer_batch_with_no_response_tokens(monkeypatch):
    trainer = _trainer(monkeypatch, 2)
    holder = _holder()
    trainer.set_service_holder(holder)
    trajectories = _trajectories()
    for trajectory in trajectories:
        trajectory.loss_mask = [0, 0, 0]
    metrics, _, _ = run_coro(trainer.train(trajectories))
    assert metrics["training/skipped_flag"] == 1
    assert holder._training_client.optim_count == 0
    assert not any(stage == "backward" for stage, _ in holder._training_client.requests)


def test_gspo_rejects_per_token_advantages_during_estimation_before_backward(monkeypatch):
    trainer = _trainer(monkeypatch, 2)
    holder = _holder()
    trainer.set_service_holder(holder)
    trajectories = _trajectories()
    trajectories[0]["token_advantages"] = [0.2, 0.2]
    with pytest.raises(ValueError, match="token_advantages"):
        run_coro(trainer.train(trajectories))
    assert not any(stage == "backward" for stage, _ in holder._training_client.requests)
    assert holder._training_client.optim_count == 0


@pytest.mark.parametrize("value", [True, False, "0.0004", float("inf")])
def test_gspo_canonical_clip_rejects_non_numeric_or_non_finite(value):
    algorithm = _config().algorithm.model_dump()
    algorithm["loss"]["params"]["clip"]["ratio_low"] = value
    with pytest.raises(ValueError):
        AlgorithmConfig.model_validate(algorithm)


def test_gspo_uses_explicit_complete_trajectory_reduction():
    canonical = _config().algorithm.model_dump()
    adapter = build_loss_adapter(_config().algorithm)
    assert adapter.custom is not None
    assert adapter.custom["name"] == "gspo"
    assert adapter.runtime_reduction == "trajectory-mean"
    invalid = deepcopy(canonical)
    invalid["loss"]["mode"] = "ppo_clip"
    with pytest.raises(ValueError, match="only by GSPO"):
        AlgorithmConfig.model_validate(invalid)


def test_gspo_rejects_legacy_loss_selector():
    algorithm = _config().algorithm.model_dump()
    loss = algorithm.pop("loss")
    algorithm.update(loss_fn="gspo", clip=loss["params"]["clip"], loss_agg_mode="sample-mean")
    with pytest.raises(ValidationError) as error:
        AlgorithmConfig.model_validate(algorithm)
    assert {
        tuple(item["loc"]) for item in error.value.errors() if item["type"] == "extra_forbidden"
    } == {
        ("loss_fn",),
        ("clip",),
        ("loss_agg_mode",),
    }


def test_gspo_does_not_reinterpret_legacy_sample_mean():
    algorithm = _config().algorithm.model_dump()
    algorithm["loss"].pop("reduction")
    algorithm["loss_agg_mode"] = "sample-mean"
    with pytest.raises(ValidationError) as error:
        AlgorithmConfig.model_validate(algorithm)
    assert any(
        item["loc"] == ("loss_agg_mode",) and item["type"] == "extra_forbidden"
        for item in error.value.errors()
    )


def test_gspo_rejects_configured_postprocessing():
    algorithm = _config().algorithm.model_dump()
    algorithm["advantage"]["postprocessing"] = {"mode": "tool_error_rebalance"}
    with pytest.raises(ValueError, match="disabled advantage postprocessing"):
        AlgorithmConfig.model_validate(algorithm)
