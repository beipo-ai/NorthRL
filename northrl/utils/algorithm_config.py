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

"""Strict canonical configuration for algorithm components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NATIVE_LOSS_MODES = frozenset(
    {
        "importance_sampling",
        "truncated_importance_sampling",
        "ppo_clip",
        "score_centering",
        "cross_entropy",
    }
)
UNIMPLEMENTED_MODES = frozenset(
    {
        "rloo",
        "reinforce",
        "reinforce++",
        "reinforce_plus_plus",
        "reinforce_plus_plus_baseline",
        "mimo",
        "tool_error_rebalance",
    }
)
BUILTIN_CALLBACK_LOSS_MODES = frozenset({"cispo", "gspo"})


class _ComponentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClipParams(_ComponentModel):
    ratio: float | None = None
    ratio_low: float | None = None
    ratio_high: float | None = None
    ratio_c: float | None = None
    truncation_c: float | None = None


class _EmptyParams(_ComponentModel):
    pass


class _GrpoParams(_ComponentModel):
    norm_adv_by_std: bool = True


class _OpdParams(_ComponentModel):
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0


class _ReinforceParams(_ComponentModel):
    gamma: float = Field(1.0, ge=0, le=1, allow_inf_nan=False)


class _ReinforceBaselineParams(_ReinforceParams):
    @model_validator(mode="after")
    def _check_gamma(self) -> _ReinforceBaselineParams:
        if self.gamma != 1:
            raise ValueError("reinforce_plus_plus_baseline requires gamma=1")
        return self


class EstimatorConfig(_ComponentModel):
    mode: (
        Literal["grpo", "rloo", "opd", "reinforce_plus_plus", "reinforce_plus_plus_baseline"] | None
    ) = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> EstimatorConfig:
        schema: type[_ComponentModel] = {
            "grpo": _GrpoParams,
            "rloo": _EmptyParams,
            "opd": _OpdParams,
            "reinforce_plus_plus": _ReinforceParams,
            "reinforce_plus_plus_baseline": _ReinforceBaselineParams,
            None: _EmptyParams,
        }[self.mode]
        self.params = schema.model_validate(self.params).model_dump()
        return self


class _ToolErrorRebalanceParams(_ComponentModel):
    negative_multiplier: float = Field(2.0, gt=1, allow_inf_nan=False)
    positive_scale_max: float = Field(2.0, ge=1, allow_inf_nan=False)
    negative_scale_min: float = Field(0.5, gt=0, le=1, allow_inf_nan=False)


class PostprocessingConfig(_ComponentModel):
    mode: Literal["tool_error_rebalance"] | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> PostprocessingConfig:
        schema = _ToolErrorRebalanceParams if self.mode == "tool_error_rebalance" else _EmptyParams
        self.params = schema.model_validate(self.params).model_dump()
        return self


class AdvantageConfig(_ComponentModel):
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)


class _FixedResponseBudgetParams(_ComponentModel):
    max_response_tokens: int = Field(..., strict=True, gt=0)


class ReductionConfig(_ComponentModel):
    mode: Literal[
        "token_mean", "datum_mean", "trajectory_mean", "prompt_mean", "fixed_response_budget", "sum"
    ] = "token_mean"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> ReductionConfig:
        schema = (
            _FixedResponseBudgetParams if self.mode == "fixed_response_budget" else _EmptyParams
        )
        self.params = schema.model_validate(self.params).model_dump()
        return self


class _PolicyLossParams(_ComponentModel):
    clip: ClipParams = Field(default_factory=ClipParams)


class _CallbackClipParams(_ComponentModel):
    """Callback bounds reject booleans, strings and non-finite numbers."""

    ratio: float | None = Field(None, strict=True, ge=0, allow_inf_nan=False)
    ratio_low: float | None = Field(None, strict=True, ge=0, allow_inf_nan=False)
    ratio_high: float | None = Field(None, strict=True, ge=0, allow_inf_nan=False)
    ratio_c: None = None
    truncation_c: None = None


class _CispoLossParams(_ComponentModel):
    clip: _CallbackClipParams

    @model_validator(mode="after")
    def _check_bounds(self) -> _CispoLossParams:
        for name in ("ratio_low", "ratio_high"):
            if getattr(self.clip, name) is None and self.clip.ratio is None:
                raise ValueError(f"CISPO requires explicit clip.{name} or clip.ratio")
        return self


class _GspoLossParams(_ComponentModel):
    clip: _CallbackClipParams

    @model_validator(mode="after")
    def _check_bounds(self) -> _GspoLossParams:
        for name in ("ratio_low", "ratio_high"):
            if getattr(self.clip, name) is None and self.clip.ratio is None:
                raise ValueError(f"GSPO requires explicit clip.{name} or clip.ratio")
        low = self.clip.ratio_low if self.clip.ratio_low is not None else self.clip.ratio
        if low is not None and low >= 1:
            raise ValueError("GSPO clip.ratio_low (or ratio) must be less than 1")
        return self


class _ScoreCenteringParams(_PolicyLossParams):
    topk: int = Field(128, ge=1, le=128)


class _CustomLossParams(_PolicyLossParams):
    module_path: str | None = None
    name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    base_loss_fn: str | None = "importance_sampling"


class _NativeLossParams(_PolicyLossParams):
    name: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_unimplemented(self) -> _NativeLossParams:
        if self.name in UNIMPLEMENTED_MODES:
            raise ValueError(f"Loss mode {self.name!r} is not implemented")
        if self.name in NATIVE_LOSS_MODES | BUILTIN_CALLBACK_LOSS_MODES:
            raise ValueError(f"Use algorithm.loss.mode={self.name!r} for a built-in loss")
        return self


class LossConfig(_ComponentModel):
    mode: Literal[
        "importance_sampling",
        "truncated_importance_sampling",
        "ppo_clip",
        "score_centering",
        "cross_entropy",
        "custom",
        "native",
        "cispo",
        "gspo",
    ] = "importance_sampling"
    params: dict[str, Any] = Field(default_factory=dict)
    reduction: ReductionConfig = Field(default_factory=ReductionConfig)

    @model_validator(mode="after")
    def _validate_params(self) -> LossConfig:
        schema: type[_ComponentModel] = {
            "cispo": _CispoLossParams,
            "gspo": _GspoLossParams,
            "score_centering": _ScoreCenteringParams,
            "custom": _CustomLossParams,
            "native": _NativeLossParams,
        }.get(self.mode, _PolicyLossParams)
        self.params = schema.model_validate(self.params).model_dump()
        return self


class KLParams(_ComponentModel):
    coef: float = 0.0
    estimator: Literal["k1", "k2", "k3"] = "k3"
