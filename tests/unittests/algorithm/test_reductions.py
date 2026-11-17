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

"""Canonical reduction denominators preserve their intended population."""

from copy import deepcopy

import pytest

from northrl.algorithm.reductions import (
    attach_fixed_response_budget_denominator,
    get_fixed_response_budget_denominators,
    get_loss_denominators,
)
from northrl.northrl_types import Trajectory


def test_token_and_datum_means_count_active_zero_advantage_tokens():
    # Counts already include valid zero advantages and exclude masked targets.
    datums = [{}, {}, {}]
    assert get_loss_denominators(datums, [3, 2, 0], "token_mean") == [5, 5, 5]
    assert get_loss_denominators(datums, [3, 2, 0], "datum_mean") == [6, 4, 1]


@pytest.mark.parametrize(
    "mode,key",
    [
        ("prompt_mean", "_prompt_mean_denominator"),
        ("fixed_response_budget", "_dr_grpo_denominator"),
    ],
)
def test_complete_population_survives_rejected_tokens(mode, key):
    datums = [{key: 24}, {key: 48}]
    # MIS can reject some/all tokens without changing the original divisor.
    assert get_loss_denominators(datums, [3, 2], mode) == [24, 48]
    assert get_loss_denominators(datums, [0, 0], mode) == [24, 48]


@pytest.mark.parametrize("bad", [None, 0, -1, True, 1.0])
@pytest.mark.parametrize(
    "mode,key",
    [
        ("prompt_mean", "_prompt_mean_denominator"),
        ("fixed_response_budget", "_dr_grpo_denominator"),
    ],
)
def test_complete_population_requires_integer_metadata(mode, key, bad):
    scope = "full-step" if mode == "fixed_response_budget" else "complete-group"
    with pytest.raises(ValueError, match=f"positive {scope} denominator"):
        get_loss_denominators([{key: bad}], [1], mode)


@pytest.mark.parametrize("mode", ["token_mean", "datum_mean", "sum"])
def test_one_count_is_required_per_datum(mode):
    with pytest.raises(ValueError, match="Every Datum"):
        get_loss_denominators([{}, {}], [2], mode)


def test_empty_and_sum_reductions():
    assert get_loss_denominators([], [], "token_mean") == []
    assert get_loss_denominators([{}, {}], [0, 0], "token_mean") == [1, 1]
    assert get_loss_denominators([{}, {}], [0, 0], "datum_mean") == [1, 1]
    assert get_loss_denominators([{}], [2], "sum") is None
    with pytest.raises(ValueError, match="Unknown loss reduction"):
        get_loss_denominators([], [], "sample-mean")


def _sequence(mask=None):
    return {
        "tokens": [1, 2, 3],
        "loss_mask": [0, 1, 1],
        "old_log_probs": [0.0, -0.1, -0.1],
        "correction_mask": [0, 1, 1] if mask is None else mask,
    }


def test_denominator_counts_empty_zero_advantage_and_rejected_trajectories():
    trajectories = [
        Trajectory(segments=[_sequence(), _sequence()], reward=1.0),
        Trajectory(segments=[_sequence()], reward=2.0, extra_fields={"advantage": 0.0}),
        Trajectory(segments=[], reward=3.0),
        Trajectory(segments=[_sequence([0, 0, 0])], reward=4.0),
        Trajectory(segments=[_sequence()], reward=None, status="failed"),
        Trajectory(segments=[_sequence()], reward=9.0, extra_fields={"group_skipped": True}),
    ]
    inputs = {"advantages": [1.0, 1.0], "loss_mask": [1, 1]}
    datums = [{"loss_fn_inputs": inputs} for _ in range(4)]
    before = deepcopy([trajectory.segments for trajectory in trajectories])
    attach_fixed_response_budget_denominator(trajectories, datums, max_response_tokens=8)
    assert get_fixed_response_budget_denominators(datums) == [32] * 4
    assert get_fixed_response_budget_denominators(datums[2:]) == [32] * 2
    assert all(datum["loss_fn_inputs"] is inputs for datum in datums)
    assert [trajectory.segments for trajectory in trajectories] == before


def test_empty_datum_batch_does_not_require_a_positive_trajectory_count():
    datums = []
    attach_fixed_response_budget_denominator([], datums, max_response_tokens=8)
    assert get_fixed_response_budget_denominators(datums) == []


@pytest.mark.parametrize("value", [None, 0, -1, True, 2.0, "2"])
def test_missing_or_invalid_step_denominator_is_not_recomputed(value):
    datum = {"_dr_grpo_denominator": value, "loss_fn_inputs": {"loss_mask": [1, 1]}}
    with pytest.raises(ValueError, match="positive full-step denominator"):
        get_fixed_response_budget_denominators([datum])


@pytest.mark.parametrize("value", [0, -1, True, 2.0])
def test_invalid_response_budget_fails_before_writing_datums(value):
    datums = [{}]
    with pytest.raises(ValueError, match="positive integer max_response_tokens"):
        attach_fixed_response_budget_denominator([], datums, max_response_tokens=value)
    assert datums == [{}]
