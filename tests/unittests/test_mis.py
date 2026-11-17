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

"""Numerical contracts for frozen rollout correction and rejection."""

from __future__ import annotations

import copy
import math

import pytest

from northrl.algorithm.mis import compute_mis_weights, compute_mismatch_metrics
from northrl.utils.config_schema import MISConfig


def _compute(ratios, *, masks=None, **settings):
    rows = [[0.0, *[math.log(value) for value in row]] for row in ratios]
    return compute_mis_weights(
        MISConfig(enabled=True, **settings),
        rollout_log_probs=[[0.0] * len(row) for row in rows],
        recomputed_log_probs=rows,
        loss_masks=masks if masks is not None else [[0, *[1] * (len(row) - 1)] for row in rows],
    )


@pytest.mark.parametrize(
    ("mode", "weights", "mask", "clipped"),
    [
        ("none", [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 0),
        ("truncate", [0.25, 0.5, 1, 2, 2], [1, 1, 1, 1, 1], 0.2),
        ("clip", [0.5, 0.5, 1, 2, 2], [1, 1, 1, 1, 1], 0.4),
        ("mask", [0.25, 0.5, 1, 2, 4], [0, 1, 1, 1, 0], 0),
    ],
)
def test_modes_preserve_inclusive_bounds_and_separate_weights_from_rejection(
    mode, weights, mask, clipped
):
    result = _compute([[0.25, 0.5, 1, 2, 4]], tis_mode=mode)
    assert result.correction_weights[0] == pytest.approx([0, *weights])
    assert result.correction_mask == [[0, *mask]]
    assert result.metrics["clipped_fraction"] == clipped
    assert result.metrics["raw_weight_min"] == 0.25
    assert result.metrics["raw_weight_max"] == 4


@pytest.mark.parametrize(
    ("level", "expected"),
    [("token", [2, 8]), ("sequence", [16, 16]), ("geometric", [4, 4])],
)
def test_levels_aggregate_only_original_valid_tokens(level, expected):
    result = _compute([[2, 1e10, 8]], masks=[[0, 1, 0, 1]], tis_level=level, tis_upper_bound=100)
    assert result.correction_weights[0] == pytest.approx([0, expected[0], 0, expected[1]])
    assert result.correction_mask == [[0, 1, 0, 1]]


def test_sequence_logratio_is_bounded_after_aggregation_and_cancellation():
    result = compute_mis_weights(
        MISConfig(enabled=True, tis_level="sequence", tis_upper_bound=1e20),
        rollout_log_probs=[[0, -100, 0]],
        recomputed_log_probs=[[0, 0, -99]],
        loss_masks=[[0, 1, 1]],
    )
    assert result.correction_weights[0] == pytest.approx([0, math.e, math.e])
    bounded = compute_mis_weights(
        MISConfig(enabled=True, tis_upper_bound=1e20),
        rollout_log_probs=[[0, -1000, 0]],
        recomputed_log_probs=[[0, 0, -1000]],
        loss_masks=[[0, 1, 1]],
    )
    assert bounded.correction_weights[0] == pytest.approx([0, math.exp(20), math.exp(-20)])


@pytest.mark.parametrize(
    ("level", "expected_mask"),
    [("token", [0, 0, 1]), ("sequence", [0, 0, 0]), ("geometric", [0, 1, 1])],
)
def test_rs_level_is_independent_of_tis_and_supports_unit_weight_rs(level, expected_mask):
    result = _compute(
        [[0.25, 2]],
        tis_mode="none",
        rs_level=level,
        use_rs=True,
        rs_lower_bound=0.6,
        rs_upper_bound=3,
    )
    assert result.correction_weights == [[0, 1, 1]]
    assert result.correction_mask == [expected_mask]


def test_rs_uses_unclipped_ratio_and_inherits_tis_bounds():
    result = _compute([[0.25, 1, 4]], tis_mode="clip", use_rs=True)
    assert result.correction_weights == [[0, 0.5, 1, 2]]
    assert result.correction_mask == [[0, 0, 1, 0]]
    assert result.metrics["token_rejection_fraction"] == pytest.approx(2 / 3)


def test_tis_mask_and_rs_mask_intersect_without_multiplying_two_ratios():
    result = _compute(
        [[0.25, 1, 4]],
        tis_mode="mask",
        tis_lower_bound=0.1,
        tis_upper_bound=2,
        use_rs=True,
        rs_lower_bound=0.5,
        rs_upper_bound=10,
    )
    assert result.correction_weights == [[0, 0.25, 1, 4]]
    assert result.correction_mask == [[0, 0, 1, 0]]


def test_veto_reads_raw_logratios_even_when_safety_bound_or_rs_would_hide_them():
    result = compute_mis_weights(
        MISConfig(
            enabled=True,
            tis_mode="none",
            use_rs=True,
            rs_lower_bound=0.5,
            rs_upper_bound=2,
            rs_veto_threshold=math.exp(-30),
        ),
        rollout_log_probs=[[0, 0, 0], [0, 0, 0]],
        recomputed_log_probs=[[0, -40, 0], [0, -40, 0]],
        loss_masks=[[0, 1, 1], [0, 0, 1]],
    )
    assert result.correction_mask == [[0, 0, 0], [0, 0, 1]]
    assert result.metrics["veto_fraction"] == 0.5
    assert result.metrics["sequence_rejection_fraction"] == 0.5


def test_veto_threshold_is_inclusive_and_requires_rs():
    retained = _compute([[0.5]], tis_mode="none", use_rs=True, rs_veto_threshold=0.5)
    assert retained.correction_mask == [[0, 1]]
    inactive = _compute([[0.25]], tis_mode="none", rs_veto_threshold=0.5)
    assert inactive.correction_mask == [[0, 1]]


def test_batch_normalization_uses_full_original_batch_including_rejected_weights():
    result = _compute([[0.25], [1, 4]], tis_mode="mask", tis_batch_normalize=True)
    assert result.metrics["batch_norm_factor"] == pytest.approx(1.75)
    assert result.correction_weights[0] == pytest.approx([0, 1 / 7])
    assert result.correction_weights[1] == pytest.approx([0, 4 / 7, 16 / 7])
    assert result.correction_mask == [[0, 0], [0, 1, 0]]
    assert result.metrics["final_weight_mean"] == pytest.approx(1)
    assert result.metrics["final_weight_max"] == pytest.approx(16 / 7)
    assert result.metrics["kept_sequences"] == 1


def test_sequence_normalization_averages_rows_including_empty_rows():
    result = _compute(
        [[0.5], [2, 2], []],
        tis_level="sequence",
        tis_upper_bound=10,
        tis_batch_normalize=True,
    )
    assert result.metrics["batch_norm_factor"] == pytest.approx((0.5 + 4 + 0) / 3)
    assert result.correction_weights[0] == pytest.approx([0, 1 / 3])
    assert result.correction_weights[1] == pytest.approx([0, 8 / 3, 8 / 3])
    assert result.correction_weights[2] == [0]
    assert result.metrics["original_sequences"] == 2


def test_truncate_then_rs_is_not_mask_mode_when_normalizing_rejected_weights():
    clipped = _compute([[0.25, 1, 4]], use_rs=True, tis_batch_normalize=True)
    masked = _compute([[0.25, 1, 4]], tis_mode="mask", tis_batch_normalize=True)
    assert clipped.correction_mask == masked.correction_mask == [[0, 0, 1, 0]]
    assert clipped.correction_weights[0][2] == pytest.approx(12 / 13)
    assert masked.correction_weights[0][2] == pytest.approx(4 / 7)


def test_extrema_are_over_actual_valid_tokens_not_averages_of_sequence_extrema():
    result = _compute([[0.25, 0.5], [2, 4]], tis_upper_bound=10)
    assert result.metrics["final_weight_min"] == 0.25
    assert result.metrics["final_weight_max"] == 4
    assert result.metrics["final_weight_mean"] == pytest.approx(6.75 / 4)


def test_disabled_returns_identity_and_does_not_modify_inputs():
    rollout = [[None, -10, float("nan")]]
    recomputed = [[float("inf"), -2, None]]
    masks = [[0, 1, 0]]
    before = copy.deepcopy((rollout, recomputed, masks))
    result = compute_mis_weights(
        MISConfig(enabled=False, use_rs=True),
        rollout_log_probs=rollout,
        recomputed_log_probs=recomputed,
        loss_masks=masks,
    )
    assert result.correction_weights == [[0, 1, 0]]
    assert result.correction_mask == [[0, 1, 0]]
    assert rollout[0][0:2] == before[0][0][0:2]
    assert math.isnan(rollout[0][2])
    assert recomputed == before[1]
    assert masks == before[2]
    result.correction_mask[0][1] = 0
    result.correction_weights[0][1] = 2
    assert masks == [[0, 1, 0]]
    assert recomputed[0][1] == -2


@pytest.mark.parametrize("rows", [[], [[]], [[None, float("nan")]]])
def test_empty_and_fully_masked_inputs_are_finite(rows):
    result = compute_mis_weights(
        MISConfig(enabled=True, tis_batch_normalize=True, use_rs=True),
        rollout_log_probs=rows,
        recomputed_log_probs=rows,
        loss_masks=[[0] * len(row) for row in rows],
    )
    assert result.correction_weights == [[0] * len(row) for row in rows]
    assert result.correction_mask == [[0] * len(row) for row in rows]
    assert all(math.isfinite(value) for value in result.metrics.values())
    assert result.metrics["original_tokens"] == result.metrics["kept_tokens"] == 0


@pytest.mark.parametrize(
    ("rollout", "recomputed", "masks", "error"),
    [
        ([[0, 0]], [], [[0, 1]], "number of sequences"),
        ([[0, 0]], [[0]], [[0, 1]], "lengths must match"),
        ([[0, 0]], [[0, 0]], [[0, 0.5]], "only 0 or 1"),
        ([[0, 0]], [[0, 0]], [[1, 1]], "masked placeholder"),
        ([[0, float("nan")]], [[0, 0]], [[0, 1]], "must be finite"),
        ([[0, 0]], [[0, float("inf")]], [[0, 1]], "must be finite"),
        ([[0, None]], [[0, 0]], [[0, 1]], "must be finite"),
    ],
)
def test_invalid_active_data_is_rejected(rollout, recomputed, masks, error):
    with pytest.raises(ValueError, match=error):
        compute_mis_weights(
            MISConfig(enabled=True),
            rollout_log_probs=rollout,
            recomputed_log_probs=recomputed,
            loss_masks=masks,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"tis_mode": "invalid"},
        {"tis_level": "invalid"},
        {"tis_upper_bound": 0},
        {"tis_lower_bound": float("inf")},
        {"tis_mode": "clip", "tis_lower_bound": 2, "tis_upper_bound": 2},
        {"use_rs": True, "rs_lower_bound": 3, "rs_upper_bound": 2},
        {"rs_veto_threshold": -1},
        {"tis_level": "geometric", "tis_batch_normalize": True},
    ],
)
def test_algorithm_validates_config_even_when_schema_validation_was_bypassed(settings):
    config = MISConfig(enabled=True).model_copy(update=settings)
    with pytest.raises(ValueError):
        compute_mis_weights(config, rollout_log_probs=[], recomputed_log_probs=[], loss_masks=[])


def test_none_mode_allows_geometric_setting_with_normalization_for_pure_rs():
    result = _compute(
        [[0.25, 1, 4]],
        tis_mode="none",
        tis_level="geometric",
        tis_batch_normalize=True,
        use_rs=True,
    )
    assert result.correction_weights == [[0, 1, 1, 1]]
    assert result.correction_mask == [[0, 0, 1, 0]]


def test_enabled_computation_preserves_all_inputs_and_output_is_independent():
    rollout = [[0.0, -3.0, 1000.0, -2.0]]
    recomputed = [[0.0, -2.0, -1000.0, -3.0]]
    masks = [[0, 1, 0, 1]]
    before = copy.deepcopy((rollout, recomputed, masks))
    result = compute_mis_weights(
        MISConfig(enabled=True, tis_mode="mask", use_rs=True, tis_batch_normalize=True),
        rollout_log_probs=rollout,
        recomputed_log_probs=recomputed,
        loss_masks=masks,
    )
    assert (rollout, recomputed, masks) == before
    result.correction_weights[0][1] = 123.0
    result.correction_mask[0][1] = 1
    assert (rollout, recomputed, masks) == before


def test_disabled_stage_settings_need_not_form_an_active_configuration():
    config = MISConfig(enabled=False, tis_level="geometric", tis_batch_normalize=True)
    result = compute_mis_weights(
        config,
        rollout_log_probs=[[0.0, -3.0]],
        recomputed_log_probs=[[0.0, -2.0]],
        loss_masks=[[0, 1]],
    )
    assert result.correction_weights == [[0, 1]]
    assert result.correction_mask == [[0, 1]]


def test_large_finite_bounds_do_not_overflow_normalization_or_metrics():
    result = _compute(
        [[1, 1]],
        tis_mode="clip",
        tis_lower_bound=1e308,
        tis_upper_bound=1.5e308,
        tis_batch_normalize=True,
    )
    assert result.correction_weights == [[0, 1, 1]]
    assert all(math.isfinite(value) for value in result.metrics.values())


@pytest.mark.parametrize("aggregation", ["token-mean", "sample-mean", "dapo"])
def test_mismatch_metrics_known_unequal_sequence_lengths(aggregation):
    log_two = math.log(2)
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[0, -log_two], [0, -2 * log_two, -2 * log_two]],
        recomputed_log_probs=[[0, -2 * log_two], [0, -log_two, -log_two]],
        loss_masks=[[0, 1], [0, 1, 1]],
        loss_agg_mode=aggregation,
    )
    sample_mean = aggregation == "sample-mean"
    assert metrics == pytest.approx(
        {
            "mismatch_tokens": 3,
            "mismatch_sequences": 2,
            "nonfinite_metric_count": 0,
            "training_log_ppl": 1.5 * log_two if sample_mean else 4 * log_two / 3,
            "rollout_log_ppl": 1.5 * log_two if sample_mean else 5 * log_two / 3,
            "training_ppl": 3 if sample_mean else 8 / 3,
            "rollout_ppl": 3 if sample_mean else 10 / 3,
            "kl": 0 if sample_mean else -log_two / 3,
            "k3_kl": 0.25 if sample_mean else 0.5 - log_two / 3,
            "log_ppl_diff": 0 if sample_mean else -log_two / 3,
            "log_ppl_abs_diff": log_two,
            "ppl_ratio": 1.25 if sample_mean else 1,
            "chi2_token": 1.125 if sample_mean else 1.75,
            "chi2_seq": 7.125 if sample_mean else 9.75,
        }
    )


@pytest.mark.parametrize("aggregation", ["token-mean", "sample-mean"])
def test_mismatch_metrics_ignore_masked_values_and_empty_sequences(aggregation):
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[float("nan"), -2, None, -4], [], [float("inf")]],
        recomputed_log_probs=[[None, -2, float("nan"), -4], [], [None]],
        loss_masks=[[0, 1, 0, 1], [], [0]],
        loss_agg_mode=aggregation,
    )
    assert metrics["mismatch_tokens"] == 2
    assert metrics["mismatch_sequences"] == 1
    assert metrics["training_log_ppl"] == metrics["rollout_log_ppl"] == 3
    assert metrics["training_ppl"] == pytest.approx(math.exp(3))
    assert metrics["kl"] == metrics["k3_kl"] == metrics["chi2_seq"] == 0
    assert metrics["ppl_ratio"] == 1


@pytest.mark.parametrize("rows", [[], [[]], [[None, float("nan")]]])
def test_mismatch_metrics_empty_input_has_only_empty_statistics(rows):
    assert compute_mismatch_metrics(
        rollout_log_probs=rows,
        recomputed_log_probs=rows,
        loss_masks=[[0] * len(row) for row in rows],
    ) == {"mismatch_tokens": 0, "mismatch_sequences": 0, "nonfinite_metric_count": 0}


@pytest.mark.parametrize(
    ("rollout", "recomputed", "masks", "error"),
    [
        ([[0, 0]], [], [[0, 1]], "number of sequences"),
        ([[0, 0]], [[0]], [[0, 1]], "lengths must match"),
        ([[0, 0]], [[0, 0]], [[0, 0.5]], "only 0 or 1"),
        ([[0, 0]], [[0, 0]], [[1, 1]], "masked placeholder"),
        ([[0, float("nan")]], [[0, 0]], [[0, 1]], "must be finite"),
        ([[0, 0]], [[0, float("inf")]], [[0, 1]], "must be finite"),
        ([[0, None]], [[0, 0]], [[0, 1]], "must be finite"),
        ([[0, -1e308]], [[0, 1e308]], [[0, 1]], "difference overflowed"),
    ],
)
def test_mismatch_metrics_reject_invalid_effective_inputs(rollout, recomputed, masks, error):
    with pytest.raises(ValueError, match=error):
        compute_mismatch_metrics(
            rollout_log_probs=rollout, recomputed_log_probs=recomputed, loss_masks=masks
        )


def test_mismatch_metrics_reject_unsupported_loss_aggregation():
    with pytest.raises(ValueError, match="loss_agg_mode"):
        compute_mismatch_metrics(
            rollout_log_probs=[], recomputed_log_probs=[], loss_masks=[], loss_agg_mode="unknown"
        )


def test_mismatch_metrics_omit_overflowing_exponentials_and_keep_finite_diagnostics():
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[0, -1000]],
        recomputed_log_probs=[[0, -1]],
        loss_masks=[[0, 1]],
    )
    assert "rollout_ppl" not in metrics
    assert "k3_kl" not in metrics
    assert metrics["nonfinite_metric_count"] == 2
    assert metrics["rollout_log_ppl"] == 1000
    assert metrics["training_ppl"] == pytest.approx(math.e)
    assert metrics["log_ppl_diff"] == -999
    assert metrics["chi2_token"] == pytest.approx(math.exp(40) - 1)
    assert all(math.isfinite(value) for value in metrics.values())


def test_mismatch_metrics_omit_unrepresentable_training_ppl_and_ppl_ratio():
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[0, -1]],
        recomputed_log_probs=[[0, -1000]],
        loss_masks=[[0, 1]],
    )
    assert "training_ppl" not in metrics
    assert "ppl_ratio" not in metrics
    assert metrics["nonfinite_metric_count"] == 2
    assert metrics["training_log_ppl"] == 1000
    assert metrics["k3_kl"] == 998
    assert all(math.isfinite(value) for value in metrics.values())


def test_mismatch_metrics_keep_true_large_exponentials_and_stable_small_k3():
    large = compute_mismatch_metrics(
        rollout_log_probs=[[0, -31]],
        recomputed_log_probs=[[0, -1]],
        loss_masks=[[0, 1]],
    )
    assert large["rollout_ppl"] == pytest.approx(math.exp(31))
    assert large["k3_kl"] == pytest.approx(math.exp(30) - 31)
    tiny = compute_mismatch_metrics(
        rollout_log_probs=[[0, -1e-8]],
        recomputed_log_probs=[[0, 0]],
        loss_masks=[[0, 1]],
    )
    assert tiny["k3_kl"] == pytest.approx(5e-17, rel=1e-7, abs=0)


def test_mismatch_metrics_large_finite_means_do_not_overflow_intermediate_sums():
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[0, -1e308, -1e308]],
        recomputed_log_probs=[[0, -1e308, -1e308]],
        loss_masks=[[0, 1, 1]],
    )
    assert metrics["training_log_ppl"] == metrics["rollout_log_ppl"] == 1e308
    assert metrics["kl"] == metrics["k3_kl"] == 0
    assert metrics["nonfinite_metric_count"] == 2
    assert all(math.isfinite(value) for value in metrics.values())


def test_mismatch_metrics_never_modify_inputs():
    rollout = [[0.0, -3.0, 1000.0, -2.0]]
    recomputed = [[0.0, -2.0, -1000.0, -3.0]]
    masks = [[0, 1, 0, 1]]
    before = copy.deepcopy((rollout, recomputed, masks))
    result = compute_mismatch_metrics(
        rollout_log_probs=rollout, recomputed_log_probs=recomputed, loss_masks=masks
    )
    result["kl"] = 123
    assert (rollout, recomputed, masks) == before


def test_mismatch_prompt_mean_weights_tokens_within_groups_and_groups_equally():
    log_two = math.log(2)
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[
            [0, -log_two],
            [0, -2 * log_two, -2 * log_two],
            [0, *[-3 * log_two] * 4],
        ],
        recomputed_log_probs=[
            [0, -2 * log_two],
            [0, -log_two, -log_two],
            [0, *[-3 * log_two] * 4],
        ],
        loss_masks=[[0, 1], [0, 1, 1], [0, 1, 1, 1, 1]],
        loss_agg_mode="prompt-mean",
        group_ids=[("actor", "prompt"), ("actor", "prompt"), ("other", "prompt")],
    )
    assert metrics == pytest.approx(
        {
            "mismatch_tokens": 7,
            "mismatch_sequences": 3,
            "mismatch_groups": 2,
            "nonfinite_metric_count": 0,
            "training_log_ppl": 13 * log_two / 6,
            "rollout_log_ppl": 7 * log_two / 3,
            "training_ppl": 16 / 3,
            "rollout_ppl": 17 / 3,
            "kl": -log_two / 6,
            "k3_kl": 0.25 - log_two / 6,
            "log_ppl_diff": -log_two / 6,
            "log_ppl_abs_diff": log_two / 2,
            "ppl_ratio": 1,
            "chi2_token": 7 / 8,
            "chi2_seq": 39 / 8,
        }
    )


def test_mismatch_prompt_mean_ignores_empty_groups_and_masked_values():
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[None, -2, float("nan")], [], [float("inf")], []],
        recomputed_log_probs=[[float("nan"), -2, None], [], [None], []],
        loss_masks=[[0, 1, 0], [], [0], []],
        loss_agg_mode="prompt-mean",
        group_ids=["nonempty", "nonempty", "empty", "empty"],
    )
    assert metrics["mismatch_tokens"] == metrics["mismatch_sequences"] == 1
    assert metrics["mismatch_groups"] == 1
    assert metrics["training_log_ppl"] == metrics["rollout_log_ppl"] == 2
    assert metrics["training_ppl"] == pytest.approx(math.exp(2))
    assert metrics["kl"] == metrics["k3_kl"] == metrics["chi2_seq"] == 0
    assert metrics["ppl_ratio"] == 1


@pytest.mark.parametrize("rows", [[], [[]], [[None, float("nan")]]])
def test_mismatch_prompt_mean_empty_input_has_only_empty_statistics(rows):
    assert compute_mismatch_metrics(
        rollout_log_probs=rows,
        recomputed_log_probs=rows,
        loss_masks=[[0] * len(row) for row in rows],
        loss_agg_mode="prompt-mean",
        group_ids=["empty"] * len(rows),
    ) == {
        "mismatch_tokens": 0,
        "mismatch_sequences": 0,
        "mismatch_groups": 0,
        "nonfinite_metric_count": 0,
    }


@pytest.mark.parametrize(
    ("group_ids", "error"),
    [
        (None, "require group_ids"),
        ([], "same number of sequences"),
        (["first", "second"], "same number of sequences"),
        ([["unhashable"]], "hashable group keys"),
    ],
)
def test_mismatch_prompt_mean_rejects_invalid_group_keys(group_ids, error):
    with pytest.raises(ValueError, match=error):
        compute_mismatch_metrics(
            rollout_log_probs=[[0, -1]],
            recomputed_log_probs=[[0, -1]],
            loss_masks=[[0, 1]],
            loss_agg_mode="prompt-mean",
            group_ids=group_ids,
        )


@pytest.mark.parametrize("aggregation", ["token-mean", "sample-mean", "dapo"])
def test_mismatch_legacy_aggregation_does_not_require_group_keys(aggregation):
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[0, -1]],
        recomputed_log_probs=[[0, -1]],
        loss_masks=[[0, 1]],
        loss_agg_mode=aggregation,
        group_ids=[["ignored-unhashable-key"]],
    )
    assert "mismatch_groups" not in metrics
    assert metrics["training_ppl"] == pytest.approx(math.e)


def test_mismatch_prompt_mean_omits_only_overflowing_metrics():
    metrics = compute_mismatch_metrics(
        rollout_log_probs=[[0, -1000], [0, -2]],
        recomputed_log_probs=[[0, -1], [0, -2]],
        loss_masks=[[0, 1], [0, 1]],
        loss_agg_mode="prompt-mean",
        group_ids=["first", "second"],
    )
    assert "rollout_ppl" not in metrics
    assert "k3_kl" not in metrics
    assert metrics["nonfinite_metric_count"] == 2
    assert metrics["mismatch_groups"] == 2
    assert metrics["rollout_log_ppl"] == 501
    assert metrics["training_log_ppl"] == 1.5
    assert all(math.isfinite(value) for value in metrics.values())
