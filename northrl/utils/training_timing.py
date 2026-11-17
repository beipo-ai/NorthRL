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

"""Keep client wall times separate from service-reported training compute times."""

import math
from typing import Any

_TRAINING_STAGE_KEYS = (
    "timing_train/forward_backward_wall_seconds",
    "timing_train/optimizer_step_wall_seconds",
    "timing_train/forward_backward_compute_seconds",
    "timing_train/optimizer_step_compute_seconds",
)


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _complete_sum(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    values = [_finite_nonnegative(record.get(key)) for record in records]
    if any(value is None for value in values):
        return None
    total = sum(value for value in values if value is not None)
    return total if math.isfinite(total) else None


def training_stage_metrics(
    *,
    forward_backward_metrics: list[dict[str, Any]],
    optimizer_metrics: dict[str, Any],
    forward_backward_wall_seconds: float,
    optimizer_step_wall_seconds: float,
) -> dict[str, float]:
    """Describe one optimizer batch without merging same-named compute metrics.

    Forward/backward compute is the sum of every request's reported ``compute_ms``;
    it is omitted if any request lacks a valid value. Its service-defined scope
    need not match GPU kernel time or elapsed wall time when requests overlap.
    """
    result: dict[str, float] = {}
    wall_times = (
        (_TRAINING_STAGE_KEYS[0], forward_backward_wall_seconds),
        (_TRAINING_STAGE_KEYS[1], optimizer_step_wall_seconds),
    )
    for key, value in wall_times:
        seconds = _finite_nonnegative(value)
        if seconds is not None:
            result[key] = seconds

    compute_times = (
        (_TRAINING_STAGE_KEYS[2], _complete_sum(forward_backward_metrics, "compute_ms")),
        (_TRAINING_STAGE_KEYS[3], _finite_nonnegative(optimizer_metrics.get("compute_ms"))),
    )
    for key, milliseconds in compute_times:
        if milliseconds is not None:
            result[key] = milliseconds / 1000.0
    return result


def sum_training_stage_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Sum a training step only where every optimizer batch reported the field."""
    result: dict[str, float] = {}
    for key in _TRAINING_STAGE_KEYS:
        seconds = _complete_sum(records, key)
        if seconds is not None:
            result[key] = seconds
    return result
