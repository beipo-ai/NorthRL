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

import pytest

from northrl.utils.lr_scheduling import cosine_learning_rate


def test_cosine_schedule_matches_fractional_megatron_warmup() -> None:
    values = [
        cosine_learning_rate(
            update_step=step,
            total_steps=85,
            peak_lr=2.0e-5,
            min_lr=2.0e-6,
            warmup_fraction=0.03,
        )
        for step in (0, 1, 2, 3, 84, 85)
    ]

    assert values == pytest.approx(
        [
            0.0,
            7.84313725490196e-6,
            1.568627450980392e-5,
            1.999867704416579e-5,
            2.006532484604231e-6,
            2.0e-6,
        ]
    )


def test_cosine_schedule_without_warmup_starts_at_peak_and_clamps_after_horizon() -> None:
    assert cosine_learning_rate(
        update_step=0,
        total_steps=10,
        peak_lr=2.0e-5,
        min_lr=2.0e-6,
        warmup_fraction=0.0,
    ) == pytest.approx(2.0e-5)
    assert cosine_learning_rate(
        update_step=11,
        total_steps=10,
        peak_lr=2.0e-5,
        min_lr=2.0e-6,
        warmup_fraction=0.0,
    ) == pytest.approx(2.0e-6)
