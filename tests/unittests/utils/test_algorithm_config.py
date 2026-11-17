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

"""Canonical algorithm configuration is explicit, strict, and safe to reload."""

from copy import deepcopy

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from northrl.utils.config_schema import AlgorithmConfig, KLConfig


def _algorithm(**fields):
    return {"type": "grpo", "optim": {"lr": 1e-6}, **fields}


def _resolve(raw):
    before = deepcopy(raw)
    resolved = AlgorithmConfig.model_validate(raw).model_dump()
    assert raw == before
    assert AlgorithmConfig.model_validate(resolved).model_dump() == resolved
    through_omegaconf = OmegaConf.to_container(OmegaConf.create(resolved), resolve=True)
    assert AlgorithmConfig.model_validate(through_omegaconf).model_dump() == resolved
    return resolved


@pytest.mark.parametrize(
    "flow,estimator,loss",
    [
        ("grpo", "grpo", "importance_sampling"),
        ("opd", "opd", "importance_sampling"),
        ("sft", None, "cross_entropy"),
    ],
)
def test_default_components_follow_the_training_flow(flow, estimator, loss):
    resolved = _resolve(_algorithm(type=flow))
    assert resolved == _resolve(
        _algorithm(type=flow, advantage={"estimator": {"mode": estimator}}, loss={"mode": loss})
    )
    assert resolved["advantage"]["postprocessing"] == {"mode": None, "params": {}}
    assert resolved["loss"]["reduction"] == {"mode": "token_mean", "params": {}}
    assert resolved["kl"] == {
        "mode": None,
        "params": {"coef": 0.0, "estimator": "k3"},
        "actor_to_ref_sync_freq": 0,
        "ref_logprobs_batch_size": 0,
    }


@pytest.mark.parametrize(
    "fields,location",
    [
        ({"loss_fn": "ppo_clip"}, ("loss_fn",)),
        ({"loss_fn": "cispo"}, ("loss_fn",)),
        ({"loss_fn": None}, ("loss_fn",)),
        ({"clip": {}}, ("clip",)),
        ({"loss_agg_mode": "token-mean"}, ("loss_agg_mode",)),
        ({"loss_normalizer": None}, ("loss_normalizer",)),
        ({"custom_loss": {"enabled": False}}, ("custom_loss",)),
        ({"custom_loss": "my_loss"}, ("custom_loss",)),
        ({"norm_adv_by_std_in_grpo": True}, ("norm_adv_by_std_in_grpo",)),
        ({"score_centering_topk": 128}, ("score_centering_topk",)),
        ({"adv_estimator": "rloo"}, ("adv_estimator",)),
        ({"gamma": 1.0}, ("gamma",)),
        ({"advantage_postprocessing": {"mode": None}}, ("advantage_postprocessing",)),
        ({"opd": {"kl_penalty_coef": 1.0}}, ("opd", "kl_penalty_coef")),
        ({"opd": {"kl_discount_factor": 0.0}}, ("opd", "kl_discount_factor")),
        ({"kl": {"enabled": False}}, ("kl", "enabled")),
        ({"kl": {"coef": 0.0}}, ("kl", "coef")),
        ({"kl": {"estimator": "k3"}}, ("kl", "estimator")),
        ({"loss_fn": "ppo_clip", "loss": {"mode": "ppo_clip"}}, ("loss_fn",)),
        ({"clip": {}, "loss": {"params": {"clip": {}}}}, ("clip",)),
    ],
)
def test_old_fields_are_rejected_even_when_disabled_or_equal_to_new_fields(fields, location):
    raw = _algorithm(**fields)
    before = deepcopy(raw)
    with pytest.raises(ValidationError) as error:
        AlgorithmConfig.model_validate(raw)
    assert any(
        e["type"] == "extra_forbidden" and e["loc"] == location for e in error.value.errors()
    )
    assert raw == before


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "sft", "advantage": {"estimator": {"mode": ""}}},
        {"advantage": {"postprocessing": {"mode": ""}}},
        {"kl": {"mode": ""}},
    ],
)
def test_disabled_components_require_null_instead_of_empty_string(fields):
    with pytest.raises(ValidationError):
        AlgorithmConfig.model_validate(_algorithm(**fields))


@pytest.mark.parametrize("postprocessing", [{}, {"mode": None}])
def test_disabled_postprocessing_has_no_parameters(postprocessing):
    resolved = _resolve(_algorithm(advantage={"postprocessing": postprocessing}))
    assert resolved["advantage"]["postprocessing"] == {"mode": None, "params": {}}


@pytest.mark.parametrize("estimator", ["grpo", "rloo"])
@pytest.mark.parametrize("loss, reduction", [("cispo", "token_mean"), ("gspo", "trajectory_mean")])
def test_callback_losses_and_scalar_estimators_roundtrip(estimator, loss, reduction):
    resolved = _resolve(
        _algorithm(
            entropy_coeff=0.0,
            advantage={"estimator": {"mode": estimator}},
            loss={
                "mode": loss,
                "params": {"clip": {"ratio": 0.1}},
                "reduction": {"mode": reduction},
            },
        )
    )
    assert resolved["advantage"]["estimator"]["mode"] == estimator
    assert resolved["loss"]["mode"] == loss
    assert resolved["loss"]["reduction"] == {"mode": reduction, "params": {}}


def _dr_grpo(clip):
    return _algorithm(
        entropy_coeff=0.0,
        advantage={"estimator": {"mode": "grpo", "params": {"norm_adv_by_std": False}}},
        loss={
            "mode": "ppo_clip",
            "params": {"clip": clip},
            "reduction": {"mode": "fixed_response_budget", "params": {"max_response_tokens": 2048}},
        },
    )


def test_dr_grpo_symmetric_bounds_preserve_the_explicit_clip_and_budget():
    resolved = _resolve(_dr_grpo({"ratio_low": 0.2, "ratio_high": 0.2}))
    assert resolved == _resolve(_dr_grpo({"ratio": 0.2}))
    assert resolved["loss"]["reduction"]["params"]["max_response_tokens"] == 2048


@pytest.mark.parametrize(
    "clip",
    [
        {"ratio_low": 0.1, "ratio_high": 0.2},
        {"ratio": 0.2, "ratio_low": 0.2, "ratio_high": 0.2},
        {"ratio_low": 0.2},
        {"ratio_low": 0.0, "ratio_high": 0.0},
        {"ratio_low": float("inf"), "ratio_high": float("inf")},
        {"ratio_low": 0.2, "ratio_high": 0.2, "truncation_c": 2.0},
        {"ratio_low": 0.2, "ratio_high": 0.2, "ratio_c": 3.0},
    ],
)
def test_dr_grpo_rejects_ambiguous_or_unsupported_bounds(clip):
    with pytest.raises(ValidationError):
        AlgorithmConfig.model_validate(_dr_grpo(clip))


def test_opd_math_is_separate_from_teacher_service_settings():
    io = {"teacher_service_timeout_s": 90.0, "teacher_logprob_source": "training_service"}
    params = {"kl_penalty_coef": 0.7, "kl_discount_factor": 0.95}
    resolved = _resolve(
        _algorithm(type="opd", opd=io, advantage={"estimator": {"mode": "opd", "params": params}})
    )
    assert resolved["opd"] == io
    assert resolved["advantage"]["estimator"]["params"] == params


def test_custom_loss_keeps_callback_params_separate_from_adapter_params():
    callback = {"scale": 2.5, "nested": {"arbitrary_callback_option": True}}
    resolved = _resolve(
        _algorithm(
            entropy_coeff=0.0,
            loss={
                "mode": "custom",
                "params": {
                    "module_path": "loss.py",
                    "name": "my_loss",
                    "params": callback,
                    "base_loss_fn": "backend_extension",
                    "clip": {"ratio": 0.3},
                },
            },
        )
    )
    assert resolved["loss"]["params"]["params"] == callback
    assert resolved["loss"]["params"]["base_loss_fn"] == "backend_extension"
    assert resolved["loss"]["params"]["clip"]["ratio"] == 0.3


@pytest.mark.parametrize("base", [None, "importance_sampling"])
def test_custom_loss_preserves_explicit_base_loss_name(base):
    resolved = _resolve(
        _algorithm(
            loss={
                "mode": "custom",
                "params": {
                    "name": "my_loss",
                    "base_loss_fn": base,
                },
            }
        )
    )
    assert resolved["loss"]["params"]["base_loss_fn"] == base


def test_prompt_mean_keeps_the_complete_group_requirement():
    _resolve(_algorithm(loss={"reduction": {"mode": "prompt_mean"}}, entropy_coeff=0.0))
    with pytest.raises(ValidationError, match="optim.batch_size=null"):
        AlgorithmConfig.model_validate(
            _algorithm(
                loss={"reduction": {"mode": "prompt_mean"}},
                entropy_coeff=0.0,
                optim={"lr": 1e-6, "batch_size": 4},
            )
        )


@pytest.mark.parametrize(
    "config",
    [
        {"advantage": {"estimator": {"mode": "unknown_estimator"}}},
        {"advantage": {"postprocessing": {"mode": "not-implemented"}}},
        {"advantage": {"postprocessing": {"mode": None, "params": {"negative_multiplier": 2}}}},
        {"advantage": {"estimator": {"mode": "grpo", "params": {"typo": True}}}},
        {"loss": {"mode": "mimo"}},
        {"loss": {"mode": "native", "params": {"name": "cispo"}}},
        {"loss": {"mode": "native", "params": {"name": "gspo"}}},
        {"loss": {"mode": "native", "params": {"name": "ppo_clip"}}},
        {"loss": {"mode": "ppo_clip", "params": {"clipp": 0.2}}},
        {"loss": {"mode": "score_centering", "params": {"topk": 129}}},
        {"loss": {"reduction": {"mode": "token_mean", "params": {"max_response_tokens": 2}}}},
        {"loss": {"reduction": {"mode": "typo"}}},
        {"loss": {"reduction": {"mode": "sample-mean"}}},
    ],
)
def test_unknown_modes_and_component_parameters_fail(config):
    with pytest.raises(ValidationError):
        AlgorithmConfig.model_validate(_algorithm(**config))


def test_native_extension_requires_explicit_mode_and_name():
    resolved = _resolve(
        _algorithm(
            loss={
                "mode": "native",
                "params": {
                    "name": "backend_extension",
                    "clip": {"truncation_c": 3.0},
                },
            }
        )
    )
    assert resolved["loss"]["params"]["name"] == "backend_extension"
    with pytest.raises(ValidationError):
        AlgorithmConfig.model_validate(_algorithm(loss={"mode": "backend_extension"}))


def test_kl_mode_controls_activation_and_parameters_roundtrip():
    assert KLConfig.model_validate({}).mode is None
    assert KLConfig.model_validate({"mode": "in_loss"}).enabled is True
    raw = {"mode": None, "params": {"coef": 0.3, "estimator": "k2"}}
    resolved = KLConfig.model_validate(raw).model_dump()
    assert resolved["mode"] is None
    assert resolved["params"] == raw["params"]
    assert KLConfig.model_validate(resolved).model_dump() == resolved


@pytest.mark.parametrize("flow,estimator", [("grpo", None), ("opd", "grpo"), ("sft", "opd")])
def test_estimator_modes_cannot_silently_change_training_flow(flow, estimator):
    with pytest.raises(ValidationError, match="requires advantage.estimator.mode"):
        AlgorithmConfig.model_validate(
            _algorithm(type=flow, advantage={"estimator": {"mode": estimator}})
        )


@pytest.mark.parametrize("flow", ["grpo", "opd"])
def test_cross_entropy_rejects_advantage_estimators_during_config_validation(flow):
    with pytest.raises(ValidationError, match="cross_entropy loss does not consume advantages"):
        AlgorithmConfig.model_validate(_algorithm(type=flow, loss={"mode": "cross_entropy"}))
