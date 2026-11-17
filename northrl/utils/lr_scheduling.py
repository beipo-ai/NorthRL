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

"""Pure per-update learning-rate schedules used by remote trainers."""

from __future__ import annotations

import math


def cosine_learning_rate(
    *,
    update_step: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_fraction: float,
) -> float:
    """Match Megatron's zero-based linear-warmup plus cosine schedule."""

    if update_step < 0:
        raise ValueError(f"update_step must be non-negative, got {update_step}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= warmup_fraction < 1:
        raise ValueError(f"warmup_fraction must be in [0, 1), got {warmup_fraction}")
    if not 0 <= min_lr <= peak_lr:
        raise ValueError(f"expected 0 <= min_lr <= peak_lr, got {min_lr} and {peak_lr}")

    warmup_steps = warmup_fraction * total_steps
    if warmup_steps > 0 and update_step <= warmup_steps:
        return peak_lr * float(update_step) / float(warmup_steps)
    if update_step > total_steps:
        return min_lr

    decay_ratio = float(update_step - warmup_steps) / float(total_steps - warmup_steps)
    coefficient = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
    return min_lr + coefficient * (peak_lr - min_lr)
