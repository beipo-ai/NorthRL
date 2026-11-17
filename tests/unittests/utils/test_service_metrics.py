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

"""Service summaries retain their denominator and use actual training outputs."""

import math

import pytest

from northrl.utils.service_metrics import aggregate_service_metrics, reference_kl_from_outputs


def _summary(values, *, loss=0.0, loss_weight=None):
    mean = sum(values) / len(values)
    result = {
        "tokens": len(values),
        "_response_tokens": len(values),
        "loss": loss,
        "ratio:mean": mean,
        "ratio:min": min(values),
        "ratio:max": max(values),
        "ratio:std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
    }
    if loss_weight is not None:
        result["_loss_weight"] = loss_weight
    return result


def test_unequal_token_weights_extrema_and_pooled_standard_deviation():
    left = _summary([1, 3], loss=2)
    right = _summary([10], loss=8)
    result = aggregate_service_metrics([left, right])
    assert result["tokens"] == 3
    assert result["loss"] == 4
    assert result["ratio:mean"] == pytest.approx(14 / 3)
    assert result["ratio:min"] == 1
    assert result["ratio:max"] == 10
    expected_std = math.sqrt((1 + 9 + 100) / 3 - (14 / 3) ** 2)
    assert result["ratio:std"] == pytest.approx(expected_std)
    assert result["ratio:std"] != pytest.approx((left["ratio:std"] + right["ratio:std"]) / 2)


def test_distribution_merge_does_not_depend_on_request_partitioning():
    records = [_summary([1, 3]), _summary([2]), _summary([8, 10, 12])]
    direct = aggregate_service_metrics(records)
    nested = aggregate_service_metrics([aggregate_service_metrics(records[:2]), records[2]])
    for key in ("tokens", "ratio:mean", "ratio:min", "ratio:max", "ratio:std"):
        assert nested[key] == pytest.approx(direct[key])


def _entropy_summary(total, count, **other):
    return {
        "entropy": total / count if count else 0.0,
        "entropy:sum": total,
        "entropy:tokens": count,
        **other,
    }


@pytest.mark.parametrize("accumulate", [False, True])
def test_policy_entropy_uses_its_effective_token_count_independent_of_loss(accumulate):
    records = [
        _entropy_summary(3.0, 2.0, tokens=100, _response_tokens=100, _loss_weight=1),
        _entropy_summary(9.0, 3.0, tokens=1, _response_tokens=1, _loss_weight=1),
    ]
    for record in records:
        record["entropy_coeff"] = 0.0
    result = aggregate_service_metrics(records, accumulate=accumulate)
    assert result["entropy"] == pytest.approx(12 / 5)
    assert result["entropy:sum"] == 12.0
    assert result["entropy:tokens"] == 5.0
    assert result["entropy_coeff"] == 0.0
    assert records[0]["entropy"] == 1.5  # Input observations are untouched.


def test_policy_entropy_is_preserved_across_nested_request_and_optimizer_aggregation():
    records = [_entropy_summary(3.0, 2), _entropy_summary(9.0, 3), _entropy_summary(4.0, 1)]
    direct = aggregate_service_metrics(records)
    nested = aggregate_service_metrics(
        [
            aggregate_service_metrics(records[:2], accumulate=True),
            aggregate_service_metrics(records[2:], accumulate=True),
        ]
    )
    assert (
        nested
        == direct
        == {
            "entropy": pytest.approx(16 / 6),
            "entropy:sum": 16.0,
            "entropy:tokens": 6,
        }
    )


def test_policy_entropy_distinguishes_missing_data_from_empty_masks_and_real_zero():
    empty = _entropy_summary(0.0, 0.0)
    observed = _entropy_summary(0.0, 5.0)
    assert aggregate_service_metrics([empty]) == empty
    assert aggregate_service_metrics([empty, empty]) == empty
    assert aggregate_service_metrics([empty, observed]) == observed
    assert aggregate_service_metrics([{}, {}]) == {}
    assert aggregate_service_metrics([{}, empty]) == empty


@pytest.mark.parametrize(
    "invalid",
    [
        {"entropy": 8.0},
        {"entropy:sum": 8.0},
        {"entropy:tokens": 2.0},
        {"entropy:sum": float("nan"), "entropy:tokens": 2.0},
        {"entropy:sum": 8.0, "entropy:tokens": float("inf")},
        {"entropy:sum": 8.0, "entropy:tokens": -2.0},
        {"entropy:sum": 8.0, "entropy:tokens": 0.0},
    ],
)
def test_policy_entropy_omits_incomplete_or_invalid_pairs_together(invalid):
    observed = _entropy_summary(3.0, 2.0)
    assert aggregate_service_metrics([invalid, observed]) == observed
    assert aggregate_service_metrics([invalid, {}]) == {}


def test_policy_entropy_does_not_pair_statistics_from_different_requests():
    assert aggregate_service_metrics([{"entropy:sum": 3.0}, {"entropy:tokens": 2.0}]) == {}


def test_loss_reduction_weight_is_distinct_from_token_distribution_weight():
    records = [_summary([1] * 100, loss=2, loss_weight=1), _summary([3] * 2, loss=8, loss_weight=3)]
    result = aggregate_service_metrics(records)
    assert result["loss"] == 6.5
    assert result["ratio:mean"] == pytest.approx(106 / 102)
    assert aggregate_service_metrics(records, accumulate=True)["loss"] == 10


def test_unknown_custom_loss_weight_does_not_fallback_to_tokens():
    records = [
        dict(_summary([1], loss=2), _loss_weight=None),
        dict(_summary([2], loss=8), _loss_weight=None),
    ]
    result = aggregate_service_metrics(records)
    assert "loss" not in result
    assert result["ratio:mean"] == 1.5


def test_native_is_requires_its_own_weights_instead_of_guessing_token_mask():
    records = [
        {
            "tokens": 100,
            "is:count": 100,
            "is:ratio": 2,
            "is:ratio_min": 1,
            "is:ratio_max": 3,
            "ppo_clip/clipfrac": 0.8,
        },
        {
            "tokens": 1,
            "is:count": 1,
            "is:ratio": 10,
            "is:ratio_min": 10,
            "is:ratio_max": 10,
            "ppo_clip/clipfrac": 0.1,
        },
    ]
    result = aggregate_service_metrics(records)
    assert "is:ratio" not in result
    assert "ppo_clip/clipfrac" not in result
    assert "is:count" not in result
    assert result["is:ratio_min"] == 1
    assert result["is:ratio_max"] == 10
    for record, weight in zip(records, (1, 3)):
        record["metric_weights"] = {"is:ratio": weight, "ppo_clip/clipfrac": weight}
    weighted = aggregate_service_metrics(records)
    assert weighted["is:ratio"] == 8
    assert weighted["ppo_clip/clipfrac"] == pytest.approx(0.275)


def test_native_logratio_std_uses_weighted_second_moment():
    records = [
        {
            "is:logratio": 2,
            "is:logratio_sq": 5,
            "metric_weights": {"is:logratio": 2, "is:logratio_sq": 2},
        },
        {
            "is:logratio": 8,
            "is:logratio_sq": 64,
            "metric_weights": {"is:logratio": 1, "is:logratio_sq": 1},
        },
    ]
    result = aggregate_service_metrics(records)
    assert result["is:logratio"] == 4
    assert result["is:logratio_std"] == pytest.approx(math.sqrt(74 / 3 - 16))


def _datum(reference, mask):
    return {
        "loss_fn_inputs": {
            "ref_logprobs": reference,
            "loss_mask": mask,
            "logprobs": [-99] * len(mask),
        }
    }


@pytest.mark.parametrize(
    "estimator,expected",
    [
        ("k1", 1.5),
        ("k2", 1.25),
        ("k3", (math.exp(-1) + math.exp(-2) + 1) / 2),
    ],
)
def test_reference_kl_uses_target_aligned_current_forward_and_effective_mask(estimator, expected):
    result = {"result": {"loss_fn_outputs": [{"logprobs": {"data": [-8, -2, -1]}}]}}
    # Original sampling denominator=-99 must not substitute for current actor.
    metrics = reference_kl_from_outputs(
        result, [_datum([-99, -3, -3], [0, 1, 1])], {"kl_coeff": 0.2, "kl_estimator": estimator}
    )
    assert metrics["actor_ref_mean"] == pytest.approx(expected)
    assert metrics["_ref_tokens"] == 2
    assert "actor_ref_loss" not in metrics  # Backend loss scaling cannot be inferred.


def test_reference_kl_is_token_weighted_across_datums_and_excludes_empty_targets():
    result = {"loss_fn_outputs": [{"logprobs": [-1]}, {"logprobs": [-1, -1]}, {"logprobs": [-100]}]}
    datums = [_datum([-2], [1]), _datum([-3, -4], [1, 1]), _datum([-1], [0])]
    metrics = reference_kl_from_outputs(result, datums, {"kl_coeff": 1, "kl_estimator": "k1"})
    assert metrics == {"actor_ref_mean": 2, "_ref_tokens": 3}


@pytest.mark.parametrize(
    "result,datums",
    [
        ({"loss_fn_outputs": []}, [_datum([-2], [1])]),
        ({"loss_fn_outputs": [{"logprobs": [0, -1]}]}, [_datum([-2], [1])]),
        ({"loss_fn_outputs": [{"logprobs": [-1]}]}, [_datum([-2], [0])]),
        ({"loss_fn_outputs": [{"logprobs": [float("nan")]}]}, [_datum([-2], [1])]),
        ({"loss_fn_outputs": [{"logprobs": [-10000]}]}, [_datum([0], [1])]),
    ],
)
def test_reference_kl_missing_misaligned_or_nonfinite_outputs_are_not_fake_zero(result, datums):
    assert reference_kl_from_outputs(result, datums, {"kl_coeff": 1, "kl_estimator": "k3"}) == {}


def test_missing_service_loss_stays_missing_and_real_zero_is_retained():
    from northrl.weaver.weaver_service_holder import WeaverServiceHolder

    assert WeaverServiceHolder._extract_loss_value({}, {}) is None
    assert WeaverServiceHolder._extract_loss_value({}, {"loss": 0.0}) == 0.0


def test_nested_service_loss_respects_requested_sum_reduction():
    from northrl.weaver.weaver_service_holder import _extract_metrics_from_result

    result = {
        "result": {
            "forward_backward_task_results": {
                "a": {"metrics": {"loss": 2.0, "tokens": 1}},
                "b": {"metrics": {"loss": 4.0, "tokens": 3}},
            }
        }
    }
    assert _extract_metrics_from_result(result)["loss"] == 3.5
    assert _extract_metrics_from_result(result, accumulate_loss=True)["loss"] == 6.0
