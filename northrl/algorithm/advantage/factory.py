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

"""Explicit construction of the implemented advantage components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .grpo import GRPOEstimator
from .opd import OPDEstimator
from .reinforce_plus_plus import ReinforcePlusPlusEstimator
from .rloo import RlooEstimator
from .tool_error_rebalance import ToolErrorRebalancePostprocessor
from .types import AdvantageEstimator, AdvantagePostprocessor

if TYPE_CHECKING:
    from northrl.utils.algorithm_config import EstimatorConfig, PostprocessingConfig


def build_estimator(config: EstimatorConfig) -> AdvantageEstimator | None:
    """Build one estimator from validated canonical configuration."""
    if config.mode is None:
        return None
    if config.mode == "grpo":
        return GRPOEstimator(norm_adv_by_std=config.params["norm_adv_by_std"])
    if config.mode == "rloo":
        return RlooEstimator()
    if config.mode in {"reinforce_plus_plus", "reinforce_plus_plus_baseline"}:
        return ReinforcePlusPlusEstimator(
            gamma=config.params["gamma"],
            baseline=config.mode == "reinforce_plus_plus_baseline",
        )
    if config.mode == "opd":
        return OPDEstimator(
            kl_penalty_coef=config.params["kl_penalty_coef"],
            kl_discount_factor=config.params["kl_discount_factor"],
        )
    raise ValueError(f"Unknown advantage estimator: {config.mode!r}")


def build_postprocessor(config: PostprocessingConfig) -> AdvantagePostprocessor | None:
    """Build the selected postprocessor; a disabled component needs no object."""
    if config.mode is None:
        return None
    if config.mode == "tool_error_rebalance":
        return ToolErrorRebalancePostprocessor(**config.params)
    raise ValueError(f"Unknown advantage postprocessor: {config.mode!r}")
