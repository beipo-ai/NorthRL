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

"""Inputs and outputs shared by concrete advantage components."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

from northrl.northrl_types import Trajectory


class OPDInputs(TypedDict, total=False):
    """OPD scores prepared by the trainer, aligned with original sequences.

    OPD with a nonzero coefficient requires all three fields. Student rows retain
    rollout scores captured before actor recomputation; teacher rows are
    token-aligned service outputs. Rows may be missing for OPD's existing
    zero-advantage fallback. Holding these references
    does not copy tokens or scores; in-place score mutations require a snapshot
    in the preparation stage.
    """

    sequences: Sequence[Trajectory | dict[str, Any]]
    student_logprobs: Sequence[Sequence[float | None] | None]
    teacher_logprobs: Sequence[Sequence[float | None] | None]


class AdvantageEstimator(Protocol):
    """Estimate once over the complete step, retaining original trajectory objects.

    Advantages remain on their existing parents or sequences. Response-only
    token arrays follow ``loss_mask[1:] == 1`` order; scalar advantages are not
    expanded into token arrays at this boundary. Each concrete estimator declares
    its own input payload type; the common caller passes that payload through.
    """

    def estimate(
        self,
        trajectories: list[Trajectory],
        inputs: Any,
        metrics: dict[str, Any],
        *,
        logger: logging.Logger | None = None,
    ) -> list[Trajectory]: ...


class AdvantagePostprocessor(Protocol):
    """Process estimated advantages on the complete step's original trajectories."""

    def process(
        self,
        trajectories: list[Trajectory],
        metrics: dict[str, Any],
        *,
        logger: logging.Logger | None = None,
    ) -> list[Trajectory]: ...
