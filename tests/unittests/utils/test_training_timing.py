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

"""Training stages require complete service observations, independent of wall time."""

import pytest

from northrl.utils.training_timing import sum_training_stage_metrics, training_stage_metrics


def _stage_metrics(forward_metrics, optimizer_metrics):
    return training_stage_metrics(
        forward_backward_metrics=forward_metrics,
        optimizer_metrics=optimizer_metrics,
        forward_backward_wall_seconds=3.0,
        optimizer_step_wall_seconds=2.0,
    )


def test_compute_totals_preserve_request_and_optimizer_boundaries():
    result = _stage_metrics([{"compute_ms": 1000}, {"compute_ms": 4500}], {"compute_ms": 50})
    assert result == {
        "timing_train/forward_backward_wall_seconds": 3.0,
        "timing_train/optimizer_step_wall_seconds": 2.0,
        "timing_train/forward_backward_compute_seconds": 5.5,
        "timing_train/optimizer_step_compute_seconds": 0.05,
    }


@pytest.mark.parametrize("invalid", [None, True, False, "4", -1, float("nan"), float("inf")])
def test_invalid_request_compute_omits_entire_forward_total_only(invalid):
    result = _stage_metrics([{"compute_ms": 1000}, {"compute_ms": invalid}], {"compute_ms": 50})
    assert "timing_train/forward_backward_compute_seconds" not in result
    assert result["timing_train/optimizer_step_compute_seconds"] == 0.05
    assert result["timing_train/forward_backward_wall_seconds"] == 3.0


def test_missing_or_empty_request_compute_does_not_become_zero():
    for records in ([], [{}], [{"compute_ms": 1000}, {}]):
        result = _stage_metrics(records, {})
        assert set(result) == {
            "timing_train/forward_backward_wall_seconds",
            "timing_train/optimizer_step_wall_seconds",
        }


@pytest.mark.parametrize("invalid", [None, True, False, "4", -1, float("nan"), float("inf")])
def test_invalid_optimizer_compute_does_not_remove_valid_forward_compute(invalid):
    result = _stage_metrics([{"compute_ms": 1000}], {"compute_ms": invalid})
    assert result["timing_train/forward_backward_compute_seconds"] == 1.0
    assert "timing_train/optimizer_step_compute_seconds" not in result


def test_zero_compute_and_wall_are_valid_observations():
    result = training_stage_metrics(
        forward_backward_metrics=[{"compute_ms": 0}],
        optimizer_metrics={"compute_ms": 0},
        forward_backward_wall_seconds=0.0,
        optimizer_step_wall_seconds=0.0,
    )
    assert len(result) == 4
    assert all(value == 0.0 for value in result.values())


def test_nonfinite_totals_and_unrepresentable_integers_are_omitted():
    result = _stage_metrics([{"compute_ms": 1e308}, {"compute_ms": 1e308}], {"compute_ms": 10**400})
    assert set(result) == {
        "timing_train/forward_backward_wall_seconds",
        "timing_train/optimizer_step_wall_seconds",
    }


def test_step_totals_require_every_optimizer_batch_for_each_field():
    first = _stage_metrics([{"compute_ms": 1000}], {"compute_ms": 50})
    second = _stage_metrics([{}], {"compute_ms": 100})
    result = sum_training_stage_metrics([first, second])
    assert result == {
        "timing_train/forward_backward_wall_seconds": 6.0,
        "timing_train/optimizer_step_wall_seconds": 4.0,
        "timing_train/optimizer_step_compute_seconds": pytest.approx(0.15),
    }
    assert sum_training_stage_metrics([first]) == first
    assert sum_training_stage_metrics([first, {}]) == {}
    assert sum_training_stage_metrics([]) == {}


def test_step_totals_reject_invalid_values_and_ignore_unrelated_metrics():
    result = sum_training_stage_metrics(
        [
            {"timing_train/forward_backward_wall_seconds": 1.0, "loss": 3.0},
            {"timing_train/forward_backward_wall_seconds": True, "loss": 4.0},
        ]
    )
    assert result == {}
