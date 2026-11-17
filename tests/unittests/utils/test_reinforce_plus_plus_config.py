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

"""Validate REINFORCE++ recipes through the complete public config schema."""

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from northrl.algorithm.advantage import build_estimator, build_postprocessor
from northrl.algorithm.advantage.reinforce_plus_plus import ReinforcePlusPlusEstimator
from northrl.algorithm.advantage.tool_error_rebalance import ToolErrorRebalancePostprocessor
from northrl.utils.config_schema import AlgorithmConfig, NorthRLConfig

ESTIMATORS = ("reinforce_plus_plus", "reinforce_plus_plus_baseline")


@pytest.fixture
def math_recipe() -> dict[str, Any]:
    """Use the shipped recipe without depending on or changing environment variables."""
    path = Path(__file__).resolve().parents[3] / "recipe/math/weaver.yaml"
    raw = path.read_text(encoding="utf-8")
    for interpolation, replacement in (
        ("${oc.env:NORTHRL_DATA_PATH}", "/data"),
        ("${oc.env:EXPERIMENT_PATH}", "/exp"),
        ("${oc.env:FEISHU_WEBHOOK_URL,}", ""),
    ):
        raw = raw.replace(interpolation, replacement)
    recipe = yaml.safe_load(raw)
    return recipe


def test_existing_recipe_keeps_grpo_defaults(math_recipe):
    config = NorthRLConfig.model_validate(math_recipe)
    assert config.algorithm.advantage.estimator.mode == "grpo"
    assert "gamma" not in config.algorithm.advantage.estimator.params


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_documented_recipe_round_trips(math_recipe, estimator):
    math_recipe["algorithm"].update(
        advantage={"estimator": {"mode": estimator}},
        loss={"mode": "ppo_clip", "params": {"clip": {"ratio": 0.2}}},
        entropy_coeff=0.0,
        optim={"lr": 1e-6, "batch_size": None, "mini_batch_size": 16},
    )
    math_recipe["data"]["train"]["rollout_repeat_n"] = 8
    config = NorthRLConfig.model_validate(math_recipe)
    assert config.algorithm.advantage.estimator.mode == estimator
    assert config.algorithm.advantage.estimator.params["gamma"] == 1.0
    assert "adv_estimator" not in config.algorithm.model_dump()
    assert "gamma" not in config.algorithm.model_dump()
    assert config.algorithm.optim.batch_size is None
    assert config.algorithm.optim.mini_batch_size == 16
    assert config.algorithm.loss.params["clip"]["ratio"] == 0.2
    assert NorthRLConfig.model_validate(config.model_dump()).algorithm == config.algorithm


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
def test_vanilla_accepts_single_rollout_and_discount(math_recipe, gamma):
    math_recipe["algorithm"].update(
        advantage={"estimator": {"mode": "reinforce_plus_plus", "params": {"gamma": gamma}}}
    )
    math_recipe["data"]["train"]["rollout_repeat_n"] = 1
    config = NorthRLConfig.model_validate(math_recipe)
    assert config.algorithm.advantage.estimator.params["gamma"] == gamma
    assert config.data.train.rollout_repeat_n == 1


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("inf"), -float("inf"), float("nan")])
def test_discount_must_be_finite_and_in_range(math_recipe, gamma):
    math_recipe["algorithm"].update(
        advantage={"estimator": {"mode": "reinforce_plus_plus", "params": {"gamma": gamma}}}
    )
    with pytest.raises(ValidationError, match="gamma"):
        NorthRLConfig.model_validate(math_recipe)


@pytest.mark.parametrize("estimator", ["grpo", "reinforce_plus_plus_baseline"])
def test_discount_is_only_configurable_for_vanilla(math_recipe, estimator):
    math_recipe["algorithm"].update(
        advantage={"estimator": {"mode": estimator, "params": {"gamma": 0.9}}}
    )
    with pytest.raises(ValidationError, match="gamma"):
        NorthRLConfig.model_validate(math_recipe)


@pytest.mark.parametrize("repeat", [None, 1])
def test_baseline_requires_multiple_rollouts(math_recipe, repeat):
    math_recipe["algorithm"]["advantage"] = {"estimator": {"mode": "reinforce_plus_plus_baseline"}}
    math_recipe["data"]["train"]["rollout_repeat_n"] = repeat
    with pytest.raises(ValidationError, match="rollout_repeat_n>=2"):
        NorthRLConfig.model_validate(math_recipe)


def test_baseline_accepts_two_rollouts(math_recipe):
    math_recipe["algorithm"]["advantage"] = {"estimator": {"mode": "reinforce_plus_plus_baseline"}}
    math_recipe["data"]["train"]["rollout_repeat_n"] = 2
    assert NorthRLConfig.model_validate(math_recipe).data.train.rollout_repeat_n == 2


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
def test_native_losses_and_token_aggregations_are_supported(math_recipe, estimator, loss_fn):
    math_recipe["algorithm"].update(
        advantage={"estimator": {"mode": estimator}},
        loss={"mode": loss_fn, "reduction": {"mode": "token_mean"}},
    )
    assert NorthRLConfig.model_validate(math_recipe).algorithm.loss.mode == loss_fn


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("batch_size", [None, 1, 32])
def test_optimizer_batch_splitting_is_supported(math_recipe, estimator, batch_size):
    math_recipe["algorithm"]["advantage"] = {"estimator": {"mode": estimator}}
    math_recipe["algorithm"]["optim"].update(batch_size=batch_size, mini_batch_size=16)
    config = NorthRLConfig.model_validate(math_recipe)
    assert config.algorithm.optim.batch_size == batch_size
    assert config.algorithm.optim.mini_batch_size == 16


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize(
    "loss_fn, message",
    [
        ("cross_entropy", "requires native importance_sampling or ppo_clip"),
        ("gspo", "scalar trajectory estimator"),
        ("cispo", "requires native importance_sampling or ppo_clip"),
        ("score_centering", "requires native importance_sampling or ppo_clip"),
    ],
)
def test_non_native_losses_are_rejected(math_recipe, estimator, loss_fn, message):
    math_recipe["algorithm"].update(
        advantage={"estimator": {"mode": estimator}}, loss={"mode": loss_fn}
    )
    if loss_fn in {"gspo", "cispo"}:
        # Keep callback loss settings valid to isolate estimator incompatibility.
        math_recipe["algorithm"].update(
            entropy_coeff=0.0,
            loss={
                "mode": loss_fn,
                "params": {"clip": {"ratio": 0.2}},
                "reduction": {"mode": "trajectory_mean" if loss_fn == "gspo" else "token_mean"},
            },
        )
        if loss_fn == "gspo":
            math_recipe["algorithm"]["optim"]["batch_size"] = None
    with pytest.raises(ValidationError, match=message):
        NorthRLConfig.model_validate(math_recipe)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_custom_loss_is_rejected(math_recipe, estimator):
    math_recipe["algorithm"].update(advantage={"estimator": {"mode": estimator}}, entropy_coeff=0.0)
    math_recipe["algorithm"]["loss"] = {"mode": "custom", "params": {"name": "example"}}
    with pytest.raises(ValidationError, match="requires native importance_sampling or ppo_clip"):
        NorthRLConfig.model_validate(math_recipe)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("aggregation", ["datum_mean", "prompt_mean", "sum"])
def test_other_loss_aggregations_are_rejected(math_recipe, estimator, aggregation):
    math_recipe["algorithm"].update(
        advantage={"estimator": {"mode": estimator}},
        loss={"mode": "ppo_clip", "reduction": {"mode": aggregation}},
    )
    with pytest.raises(ValidationError, match="requires loss.reduction.mode='token_mean'"):
        NorthRLConfig.model_validate(math_recipe)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("algorithm_type", ["opd", "sft"])
def test_other_trainer_types_are_rejected(math_recipe, estimator, algorithm_type):
    math_recipe["algorithm"].update(
        type=algorithm_type, advantage={"estimator": {"mode": estimator}}
    )
    if algorithm_type == "sft":
        math_recipe.pop("rollout")
        math_recipe["data"]["train"].pop("rollout_repeat_n")
    with pytest.raises(ValidationError, match="require algorithm.type='grpo'"):
        NorthRLConfig.model_validate(math_recipe)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("coefficient", [0.0, 0.01])
def test_existing_loss_kl_config_remains_available(math_recipe, estimator, coefficient):
    math_recipe["algorithm"]["advantage"] = {"estimator": {"mode": estimator}}
    math_recipe["algorithm"]["kl"] = {"mode": "in_loss", "params": {"coef": coefficient}}
    math_recipe["models"]["reference"] = {"name": None, "checkpoint": None}
    config = NorthRLConfig.model_validate(math_recipe)
    assert config.algorithm.kl.mode == "in_loss"
    assert config.algorithm.kl.params.coef == coefficient


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_existing_loss_kl_requires_a_reference_model(math_recipe, estimator):
    math_recipe["algorithm"]["advantage"] = {"estimator": {"mode": estimator}}
    math_recipe["algorithm"]["kl"] = {"mode": "in_loss", "params": {"coef": 0.01}}
    with pytest.raises(ValidationError, match=r"models\.reference"):
        NorthRLConfig.model_validate(math_recipe)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_mis_and_rejection_remain_available(math_recipe, estimator):
    math_recipe["algorithm"]["advantage"] = {"estimator": {"mode": estimator}}
    math_recipe["algorithm"]["mis"] = {"enabled": True, "use_rs": True}
    config = NorthRLConfig.model_validate(math_recipe)
    assert config.algorithm.mis.enabled
    assert config.algorithm.mis.use_rs


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_canonical_estimator_round_trip_preserves_estimator(estimator):
    canonical = AlgorithmConfig(
        type="grpo",
        optim={"lr": 1e-6},
        advantage={"estimator": {"mode": estimator, "params": {"gamma": 1.0}}},
    )
    assert AlgorithmConfig.model_validate(canonical.model_dump()) == canonical
    actual = build_estimator(canonical.advantage.estimator)
    assert isinstance(actual, ReinforcePlusPlusEstimator)
    assert actual.gamma == 1.0
    assert actual.baseline == (estimator == "reinforce_plus_plus_baseline")


@pytest.mark.parametrize("legacy", [{"adv_estimator": "reinforce_plus_plus"}, {"gamma": 1.0}])
def test_legacy_estimator_fields_are_rejected_with_canonical_config(legacy):
    with pytest.raises(ValidationError, match="Extra inputs"):
        AlgorithmConfig(
            type="grpo",
            optim={"lr": 1e-6},
            **legacy,
            advantage={"estimator": {"mode": "reinforce_plus_plus", "params": {"gamma": 1.0}}},
        )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_estimator_and_tool_postprocessing_round_trip_together(math_recipe, estimator):
    params = {
        "negative_multiplier": 3.0,
        "positive_scale_max": 4.0,
        "negative_scale_min": 0.25,
    }
    canonical_recipe = deepcopy(math_recipe)
    canonical_recipe["algorithm"]["advantage"] = {
        "estimator": {"mode": estimator, "params": {"gamma": 1.0}},
        "postprocessing": {"mode": "tool_error_rebalance", "params": params},
    }
    canonical = NorthRLConfig.model_validate(canonical_recipe)
    restored = NorthRLConfig.model_validate(canonical.model_dump())
    assert restored == canonical
    assert build_estimator(restored.algorithm.advantage.estimator) == ReinforcePlusPlusEstimator(
        gamma=1.0, baseline=estimator == "reinforce_plus_plus_baseline"
    )
    assert build_postprocessor(
        restored.algorithm.advantage.postprocessing
    ) == ToolErrorRebalancePostprocessor(**params)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("normalize", [False, True])
def test_legacy_grpo_normalization_is_rejected(estimator, normalize):
    with pytest.raises(ValidationError, match="Extra inputs"):
        AlgorithmConfig(
            type="grpo",
            optim={"lr": 1e-6},
            norm_adv_by_std_in_grpo=normalize,
            advantage={"estimator": {"mode": estimator}},
        )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_canonical_grpo_normalization_remains_invalid_for_reinforce(estimator):
    canonical = {"estimator": {"mode": estimator, "params": {"norm_adv_by_std": True}}}
    with pytest.raises(ValidationError, match="norm_adv_by_std"):
        AlgorithmConfig(type="grpo", optim={"lr": 1e-6}, advantage=canonical)


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("postprocessing", [False, True])
def test_documented_canonical_yaml_validates_as_recipe_override(
    math_recipe, language, postprocessing
):
    page = (
        Path(__file__).resolve().parents[3]
        / f"docs/{language}/developer-guide/06-trainers/reinforce-plus-plus.md"
    )
    blocks = [
        yaml.safe_load(block) for block in re.findall(r"```yaml\n(.*?)```", page.read_text(), re.S)
    ]
    math_recipe["algorithm"] = next(
        block["algorithm"] for block in blocks if "advantage" in block.get("algorithm", {})
    )
    if postprocessing:
        math_recipe["algorithm"]["advantage"]["postprocessing"] = next(
            block["postprocessing"] for block in blocks if "postprocessing" in block
        )
    config = NorthRLConfig.model_validate(math_recipe)
    assert config.algorithm.advantage.estimator.mode == "reinforce_plus_plus"
    assert config.algorithm.kl.mode is None
    assert config.algorithm.advantage.postprocessing.mode == (
        "tool_error_rebalance" if postprocessing else None
    )
