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

"""Metric cohort labels, missing-data policy, and explicit presentation contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from northrl.utils.logger.aggregate_logger import concat_dict_to_str
from northrl.utils.metric_catalog import (
    ALGORITHM_METRICS,
    METRIC_SPECS,
    STEP_METRIC,
    define_wandb_metrics,
    normalize_step_metrics,
    prepare_wandb_metrics,
    wandb_metric_name,
)


def test_normalization_preserves_unknown_validation_and_distinguishes_missing_from_zero():
    raw = {
        "timeout/tool_count": 0,
        "trajectory/duration_seconds_mean": None,
        "trajectory/reward_mean": float("nan"),
        "val/custom/coverage": 0.75,
        "val/custom_score": 0.125,
        "my_plugin/useful_diagnostic": 7,
        "weaver/new_unknown_counter": 100,
        "weaver/is:count": 100,
        "weaver/is:abs_logratio_gt_1": 0.2,
    }
    result = normalize_step_metrics(raw)
    assert result == {
        "timeout/tool_count": 0,
        "val/custom/coverage": 0.75,
        "val/custom_score": 0.125,
        "my_plugin/useful_diagnostic": 7,
    }
    assert "trajectory/duration_seconds_mean" in raw  # caller's snapshot is untouched
    assert normalize_step_metrics(result) == result


def test_normalization_does_not_invent_probability_units_or_unverified_aliases():
    raw = {
        "trajectory/advantage_zero_percentage": 50,
        "mis/token_rejection_fraction": 0.5,
        "weaver/is:ratio": 1.2,
        "weaver/is:logratio": -0.1,
        "weaver/ppo_clip/ppo_kl": 0.1,
        "weaver/ratio:mean": 1.1,
    }
    assert normalize_step_metrics(raw) == {f"algorithm/{key}": value for key, value in raw.items()}
    assert METRIC_SPECS["algorithm/trajectory/advantage_zero_percentage"].unit == "%"
    assert METRIC_SPECS["algorithm/mis/token_rejection_fraction"].unit == "0–1"
    assert METRIC_SPECS["algorithm/weaver/is:ratio"].unit == "无量纲"


def test_batch_token_totals_are_registered_and_reach_the_wandb_batch_section():
    metrics = {"batch/tokens_total": 13, "batch/response_tokens_total": 3}

    assert METRIC_SPECS["batch/tokens_total"].unit == "token"
    assert METRIC_SPECS["batch/tokens_total"].section == "Batch"
    assert normalize_step_metrics(metrics) == metrics
    assert prepare_wandb_metrics(metrics) == {
        "15_batch/tokens_total": 13,
        "15_batch/response_tokens_total": 3,
    }


def test_length_finish_reason_metrics_are_registered_and_preserved():
    metrics = {
        "trajectory/finish_reason_length_count": 3,
        "trajectory/finish_reason_length_fraction": 0.25,
        "val/finish_reason_length_count": 1,
        "val/finish_reason_length_fraction": 0.1,
        "val/code/trajectory_count": 5,
        "val/code/finish_reason_length_count": 1,
        "val/code/finish_reason_length_fraction": 0.2,
    }
    assert normalize_step_metrics(metrics) == metrics
    assert METRIC_SPECS["trajectory/finish_reason_length_count"].unit == "条"
    assert METRIC_SPECS["trajectory/finish_reason_length_fraction"].unit == "0–1"
    assert METRIC_SPECS["val/<dataset>/finish_reason_length_fraction"].unit == "0–1"


def test_duration_metrics_use_stage_sections_without_publishing_timing_counts():
    metrics = {
        "group/duration_seconds_mean": 20.0,
        "group/duration_seconds_observed_count": 2,
        "trajectory/duration_seconds_mean": 12.5,
        "trajectory/duration_seconds_observed_count": 4,
        "verifier/duration_seconds_mean": 5.0,
        "verifier/duration_seconds_observed_count": 3,
        "verifier/skipped_length_count": 1,
        "reward_fn/code_e2b/group/duration_seconds_mean": 20.0,
        "reward_fn/code_e2b/verifier/duration_seconds_mean": 9.0,
        "val/group/duration_seconds_mean": 21.0,
        "val/verifier/duration_seconds_observed_count": 3,
        "val/reward_fn/ifeval/group/duration_seconds_mean": 7.0,
        "val/reward_fn/ifeval/verifier/duration_seconds_mean": 0.1,
    }
    assert normalize_step_metrics(metrics) == {
        "timing_rollout/group_mean_seconds": 20.0,
        "timing_rollout/trajectory_mean_seconds": 12.5,
        "timing_rollout/verifier_mean_seconds": 5.0,
        "verifier/skipped_length_count": 1,
        "timing_rollout/code_e2b_group_mean_seconds": 20.0,
        "timing_rollout/code_e2b_verifier_mean_seconds": 9.0,
        "timing_val/group_mean_seconds": 21.0,
        "timing_val/ifeval_group_mean_seconds": 7.0,
        "timing_val/ifeval_verifier_mean_seconds": 0.1,
    }
    for key in ("timing_rollout/group_mean_seconds", "timing_rollout/verifier_mean_seconds"):
        assert METRIC_SPECS[key].unit == "秒"


@pytest.mark.parametrize(
    ("prefix", "canonical_prefix"),
    [
        ("", "timing_rollout/"),
        ("val/", "timing_val/"),
        ("reward_fn/code_e2b/", "timing_rollout/code_e2b_"),
        ("val/reward_fn/ifeval/", "timing_val/ifeval_"),
    ],
)
def test_group_duration_replaces_legacy_task_timing_in_owned_namespaces(prefix, canonical_prefix):
    current = {
        f"{prefix}group/duration_seconds_{reducer}": value
        for reducer, value in (("mean", 13.5), ("min", 7.0), ("max", 20.0), ("observed_count", 2))
    }
    legacy = {
        key.replace("group/duration_seconds_", "task/duration_seconds_"): 99 for key in current
    }
    custom = {
        f"{prefix}task/custom_score": 0.75,
        f"{prefix}task/duration_seconds_custom": 9.0,
    }

    normalized = normalize_step_metrics({**current, **legacy, **custom})

    expected = {
        f"{canonical_prefix}group_{reducer}_seconds": value
        for reducer, value in (("mean", 13.5), ("min", 7.0), ("max", 20.0))
    }
    assert normalized == {**expected, **custom}
    assert normalize_step_metrics(normalized) == normalized


def test_only_group_duration_mean_moves_to_rollout_while_other_group_means_stay_retired():
    metrics = {
        "group/duration_seconds_mean": 20.0,
        "group/tool_calls_mean": 2.0,
        "group/response_tokens_mean": 128.0,
        "group/tool_calls_min": 1.0,
        "group/tool_calls_max": 3.0,
    }

    assert normalize_step_metrics(metrics) == {
        "timing_rollout/group_mean_seconds": 20.0,
        "group/tool_calls_min": 1.0,
        "group/tool_calls_max": 3.0,
    }


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("prefix", "canonical_prefix"),
    [
        ("", "timing_rollout/"),
        ("timing/", "timing_rollout/"),
        ("val/", "timing_val/"),
        ("val/timing/", "timing_val/"),
        ("reward_fn/new_reward_v2/", "timing_rollout/new_reward_v2_"),
        ("timing/reward_fn/new_reward_v2/", "timing_rollout/new_reward_v2_"),
        ("val/reward_fn/new_reward_v2/", "timing_val/new_reward_v2_"),
        ("val/timing/reward_fn/new_reward_v2/", "timing_val/new_reward_v2_"),
    ],
)
def test_duration_aliases_are_idempotent_and_canonical_values_win(
    prefix, canonical_prefix, reverse
):
    raw = {}
    expected = {}
    for obj in ("trajectory", "group", "verifier", "model_request"):
        reducers = (("mean", 13.5), ("min", 0.0), ("max", 20.0), ("observed_count", 2))
        for reducer, value in reducers:
            suffix = f"{obj}/duration_seconds_{reducer}"
            raw[f"{prefix}{suffix}"] = value
            if reducer != "observed_count":
                expected[f"{canonical_prefix}{obj}_{reducer}_seconds"] = value
    snapshot = dict(raw)
    assert normalize_step_metrics(raw) == expected
    assert raw == snapshot

    current = {key: value + 100 for key, value in expected.items()}
    items = list({**raw, **current}.items())
    normalized = normalize_step_metrics(dict(reversed(items) if reverse else items))

    assert normalized == current
    assert normalize_step_metrics(normalized) == current


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("raw_prefix", "old_prefix", "new_prefix"),
    [
        ("", "timing/", "timing_rollout/"),
        ("val/", "val/timing/", "timing_val/"),
        ("reward_fn/if_verify/", "timing/reward_fn/if_verify/", "timing_rollout/if_verify_"),
        ("val/reward_fn/if_verify/", "val/timing/reward_fn/if_verify/", "timing_val/if_verify_"),
    ],
)
def test_old_timing_values_win_over_raw_producers_in_either_order(
    raw_prefix, old_prefix, new_prefix, reverse
):
    items = [
        (f"{raw_prefix}group/duration_seconds_max", 10.0),
        (f"{old_prefix}group/duration_seconds_max", 20.0),
    ]
    assert normalize_step_metrics(dict(reversed(items) if reverse else items)) == {
        f"{new_prefix}group_max_seconds": 20.0
    }


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
@pytest.mark.parametrize("reverse", [False, True])
def test_missing_canonical_timing_does_not_fall_back_to_an_alias(missing, reverse):
    items = [
        ("group/duration_seconds_mean", 10.0),
        ("timing/group/duration_seconds_mean", 20.0),
        ("timing_rollout/group_mean_seconds", missing),
    ]
    assert normalize_step_metrics(dict(reversed(items) if reverse else items)) == {}


def test_duration_aliases_preserve_custom_fields_and_other_metric_cohorts():
    unchanged = {
        "my_plugin/trajectory/duration_seconds_mean": 12.0,
        "val/custom/duration_seconds_mean": 7.0,
        "reward_fn/custom/unknown/duration_seconds_mean": 3.0,
        "trajectory/duration_seconds_p95": 50.0,
        "timing/trajectory/duration_seconds_p95": 51.0,
        "trajectory/response_tokens_mean": 128,
        "model_request/count": 4,
        "timeout/rollout_count": 0,
        "verifier/skipped_length_count": 1,
        "timing/custom_measurement_seconds": 2.0,
        "my_plugin/trajectory/duration_seconds_observed_count": 3,
        "val/custom/duration_seconds_observed_count": 4,
        "outcome/score_accuracy_observed_count": 5,
        "algorithm/trajectory/reward_mean": 0.5,
    }
    normalized = normalize_step_metrics({**unchanged, "model_request/duration_seconds_mean": 2.5})

    assert normalized == {**unchanged, "timing_rollout/model_request_mean_seconds": 2.5}
    assert normalize_step_metrics(normalized) == normalized


@pytest.mark.parametrize(
    "removed",
    [
        "timing/batch_release_to_claim_seconds",
        "timing/batch_claim_interval_seconds",
        "timing/batch_rollout_span_seconds",
        "timing/batch_training_seconds",
        "timing/training_time",
        "timing/mismatch_forward",
        "timing/mismatch_metrics",
        "timing_train/mismatch_metrics_seconds",
        "timing/mis",
        "timing_train/mis_seconds",
        "timing_train/prepare_seconds",
        "timing_train/other_seconds",
    ],
)
def test_retired_timings_are_neither_registered_nor_published(removed):
    retained = {
        "timing/batch_completion_interval_seconds": 30.0,
        "timing_train/batch_training_seconds": 12.0,
        "timing_train/mismatch_forward_seconds": 3.0,
        "timing/trainer_wait_for_batch_seconds": 2.0,
    }

    assert removed not in METRIC_SPECS
    assert normalize_step_metrics({removed: 17.0, **retained}) == retained


def test_verified_aliases_prefer_canonical_values_and_remove_old_curves():
    result = normalize_step_metrics(
        {
            "training/global_step": 1,
            STEP_METRIC: 2,
            "mis/log_ppl_diff": 0.2,
            "mis/kl": 0.3,
            "val/sequence_count": 4,
            "val/segment_count": 5,
            "val/math/avg@8": 0.5,
            "val/math/pass@1": 0.5,
            "timing/weight_sync_time": 0.01,
            "rollout/trajectory_count": 3,
            "batch/trajectory_count": 3,
            "recompute_logprobs/replaced": 2,
            "timing/recompute_logprobs": 1.0,
            "kl/ref_logprobs_total": 8,
            "weaver/scale": 1,
            "weaver/moe_load_balancing_loss": 0.2,
            "weaver/loss_agg_denominator": 2,
        }
    )
    assert result == {
        STEP_METRIC: 2,
        "algorithm/mis/kl": 0.3,
        "val/segment_count": 5,
        "val/math/pass@1": 0.5,
        "timing_weights/weight_sync_seconds": 0.01,
        "batch/trajectory_count": 3,
        "timing_train/recompute_logprobs_seconds": 1.0,
    }


def test_all_data_series_are_visible_with_one_training_axis_and_selective_summaries():
    run = SimpleNamespace(define_metric=MagicMock())
    define_wandb_metrics(run)
    definitions = {call.args[0]: call.kwargs for call in run.define_metric.call_args_list}
    assert definitions["*"]["summary"] == "none"
    assert definitions["01_algorithm/01_trajectory/reward_mean"]["summary"] == "last"
    assert definitions["01_algorithm/09_trajectory/advantage_mean"]["summary"] == "last"
    assert definitions["01_algorithm/22_weaver/ppo_clip/clipfrac"]["summary"] == "none"
    assert "trajectory/reward_mean" not in definitions
    assert "weaver/ppo_clip/clipfrac" not in definitions
    assert definitions["06_timing_rollout/trajectory_mean_seconds"]["summary"] == "last"
    assert definitions["06_timing_rollout/trajectory_max_seconds"]["summary"] == "none"
    assert "trajectory/duration_seconds_mean" not in definitions
    assert "group/duration_seconds_mean" not in definitions
    assert "verifier/duration_seconds_mean" not in definitions
    assert "model_request/duration_seconds_mean" not in definitions
    for key in (
        "timing/batch_completion_interval_seconds",
        "timing/trainer_wait_for_batch_seconds",
    ):
        assert METRIC_SPECS[key].unit == "秒"
        assert METRIC_SPECS[key].section == "Timing"
        assert definitions[wandb_metric_name(key)]["summary"] == "none"
    assert all(
        not definition["hidden"] for key, definition in definitions.items() if key != STEP_METRIC
    )
    assert definitions[STEP_METRIC]["hidden"] is True
    assert all(
        d["step_metric"] == STEP_METRIC and d["step_sync"] is False for d in definitions.values()
    )
    assert not any(key.startswith("live/") for key in definitions)


def test_timing_catalog_has_five_sections_with_seconds_and_one_slash():
    timing = {key: spec for key, spec in METRIC_SPECS.items() if key.startswith("timing")}
    assert {key.split("/")[0] for key in timing} == {
        "timing",
        "timing_rollout",
        "timing_train",
        "timing_weights",
        "timing_val",
    }
    assert {key for key in timing if key.startswith("timing/")} == {
        "timing/batch_completion_interval_seconds",
        "timing/trainer_wait_for_batch_seconds",
    }
    fixed_training = {key for key in timing if "<" not in key and not key.startswith("timing_val/")}
    assert len(fixed_training) == 27
    assert sum(key.startswith("timing_train/") for key in fixed_training) == 8
    assert sum(key.startswith("timing_weights/") for key in fixed_training) == 5
    assert sum(key.startswith("timing_rollout/") for key in fixed_training) == 12
    assert all(key.count("/") == 1 and key.endswith("_seconds") for key in timing)
    assert all(spec.unit == "秒" for spec in timing.values())
    assert not any("duration_seconds_observed_count" in key for key in METRIC_SPECS)
    assert not any(key.startswith("val/timing/") for key in METRIC_SPECS)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        (
            "timing_train/batch_completion_interval_seconds",
            "timing/batch_completion_interval_seconds",
        ),
        ("timing/weight_export_seconds", "timing_weights/weight_export_seconds"),
        ("timing_train/weight_export_seconds", "timing_weights/weight_export_seconds"),
        ("timing/weight_sync_seconds", "timing_weights/weight_sync_seconds"),
        ("timing_train/weight_sync_failed_seconds", "timing_weights/weight_sync_failed_seconds"),
        ("timing/weight_export_failed_seconds", "timing_weights/weight_export_failed_seconds"),
        ("timing/weight_sync_failed_seconds", "timing_weights/weight_sync_failed_seconds"),
        ("timing/recompute_logprobs", "timing_train/recompute_logprobs_seconds"),
        ("timing/ref_logprobs", "timing_train/ref_logprobs_seconds"),
        ("timing/weight_export_time", "timing_weights/weight_export_seconds"),
        ("timing/weight_export_failed_time", "timing_weights/weight_export_failed_seconds"),
        ("timing/weight_sync_failed_time", "timing_weights/weight_sync_failed_seconds"),
    ],
)
def test_training_timing_aliases_publish_once_with_canonical_priority(old, canonical, reverse):
    assert normalize_step_metrics({old: 0.0}) == {canonical: 0.0}
    items = [(old, 1.0), (canonical, 2.0)]
    normalized = normalize_step_metrics(dict(reversed(items) if reverse else items))
    assert normalized == {canonical: 2.0}
    assert normalize_step_metrics(normalized) == normalized


@pytest.mark.parametrize("reverse", [False, True])
def test_current_stage_alias_wins_over_deprecated_time_alias(reverse):
    items = [("timing/weight_export_time", 1.0), ("timing/weight_export_seconds", 2.0)]
    assert normalize_step_metrics(dict(reversed(items) if reverse else items)) == {
        "timing_weights/weight_export_seconds": 2.0
    }


def test_log_format_distinguishes_integer_count_tiny_loss_and_fraction():
    line = concat_dict_to_str(
        {
            "batch/trajectory_count": 64.0,
            "weaver/loss": 1.23456e-8,
            "mis/token_rejection_fraction": 0.125,
            "missing": None,
            "bad": float("nan"),
        },
        step=2,
    )
    assert (
        line
        == "step:2 - batch/trajectory_count:64 - weaver/loss:1.23456e-08 - mis/token_rejection_fraction:0.125"
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_algorithm_aliases_are_idempotent_and_prefer_canonical_values(reverse):
    items = [
        ("trajectory/reward_mean", 0.1),
        ("algorithm/trajectory/reward_mean", 0.8),
        ("mis/log_ppl_diff", 0.2),
        ("mis/kl", 0.3),
        ("algorithm/mis/kl", 0.4),
        ("weaver/entropy_coeff", 0.01),
        ("sampling/ppl_mean", 5.0),
        ("opd/teacher_kl", 0.05),
        ("group/reward_std_min", 0.0),
        ("trajectory/response_tokens_mean", 128),
        ("weaver/loss", -0.2),
    ]
    raw = dict(reversed(items) if reverse else items)
    expected = {
        "algorithm/trajectory/reward_mean": 0.8,
        "algorithm/mis/kl": 0.4,
        "algorithm/weaver/entropy_coeff": 0.01,
        "algorithm/sampling/ppl_mean": 5.0,
        "algorithm/opd/teacher_kl": 0.05,
        "algorithm/group/reward_std_min": 0.0,
        "trajectory/response_tokens_mean": 128,
        "weaver/loss": -0.2,
    }
    normalized = normalize_step_metrics(raw)
    assert normalized == expected
    assert normalize_step_metrics(normalized) == expected
    assert raw["trajectory/reward_mean"] == 0.1
    assert normalize_step_metrics(dict(reversed(items[2:4]) if reverse else items[2:4])) == {
        "algorithm/mis/kl": 0.3
    }
    assert normalize_step_metrics({"mis/log_ppl_diff": 0.2}) == {"algorithm/mis/kl": 0.2}


def test_algorithm_namespace_preserves_raw_service_and_missing_value_exclusions():
    assert normalize_step_metrics(
        {
            "weaver/new_unknown_counter": 2,
            "algorithm/weaver/new_unknown_counter": 2,
            "algorithm/weaver/scale": 1,
            "algorithm/kl/ref_logprobs_attached": 4,
            "weaver/entropy_loss": float("nan"),
            "algorithm/kl/actor_ref_mean": None,
            "weaver/is:ratio_min": 0.0,
        }
    ) == {"algorithm/weaver/is:ratio_min": 0.0}


@pytest.mark.parametrize("entropy,total,tokens", [(2.4, 12.0, 5), (0.0, 0.0, 0)])
def test_weaver_entropy_statistics_reach_wandb_without_changing_legacy_loss_names(
    entropy, total, tokens
):
    raw = {
        "weaver/entropy": entropy,
        "weaver/entropy:sum": total,
        "weaver/entropy:tokens": tokens,
        "weaver/entropy_coeff": 0.0,
        "weaver/entropy_loss": 0.25,
    }
    expected = {
        "01_algorithm/25_weaver/entropy_loss": 0.25,
        "01_algorithm/26_weaver/entropy_coeff": 0.0,
        "01_algorithm/56_weaver/entropy": entropy,
        "01_algorithm/57_weaver/entropy:sum": total,
        "01_algorithm/58_weaver/entropy:tokens": tokens,
    }

    assert prepare_wandb_metrics(raw) == expected
    assert prepare_wandb_metrics(normalize_step_metrics(raw)) == expected
    assert prepare_wandb_metrics(expected) == expected


def test_wandb_payload_separates_scores_timing_counts_and_status_without_mutating_input():
    raw = {
        "trajectory/reward_mean": 0.25,
        "algorithm/trajectory/reward_mean": 0.75,
        "weaver/is:ratio": 1.05,
        "mis/kl": 0.02,
        "reward_fn/org/code_v2/reward_mean": 0.5,
        "reward_fn/org/code_v2/reward_observed_count": 2,
        "reward_fn/org/code_v2/verifier/retry_count": 1,
        "reward_fn/org/code_v2/verifier/duration_seconds_mean": 4.0,
        "val/reward_fn/org/code_v2/group/duration_seconds_mean": 12.0,
        "val/reward_fn/org/code_v2/verifier/result_count": 3,
        "val/org/math/pass@1": 0.6,
        "val/org/math/num_prompts": 3,
        "val/org/math/observed_generations/mean": 2.0,
        "val/org/math/custom_observed_count": 0,
        "val/org/math/complete_group_ratio": 0.5,
        "val/finish_reason_length_fraction": 0.1,
        "val/k": 4,
        "timing_rollout/trajectory_mean_seconds": 8.0,
        "training/skipped_flag": 1,
        "training/train_step": 7,
        "weaver/scale": 99.0,
        "trajectory/advantage_std": None,
    }
    snapshot = dict(raw)
    expected = {
        "01_algorithm/01_trajectory/reward_mean": 0.75,
        "01_algorithm/32_weaver/is:ratio": 1.05,
        "12_mis/kl": 0.02,
        "03_reward_fn/01_reward_mean/org/code_v2": 0.5,
        "16_count_reward/org/code_v2/reward_observed_count": 2,
        "16_count_reward/org/code_v2/verifier/retry_count": 1,
        "07_timing_reward/org/code_v2_verifier_mean_seconds": 4.0,
        "10_timing_val/org/code_v2_group_mean_seconds": 12.0,
        "17_count_val/reward_fn/org/code_v2/verifier/result_count": 3,
        "02_val/org/math/pass@1": 0.6,
        "17_count_val/org/math/num_prompts": 3,
        "17_count_val/org/math/observed_generations/mean": 2.0,
        "17_count_val/org/math/custom_observed_count": 0,
        "18_status_val/org/math/complete_group_ratio": 0.5,
        "18_status_val/finish_reason_length_fraction": 0.1,
        "18_status_val/k": 4,
        "06_timing_rollout/trajectory_mean_seconds": 8.0,
        "14_training/skipped_flag": 1,
        "training/train_step": 7,
    }

    result = prepare_wandb_metrics(raw)

    assert result == expected
    assert list(result) == sorted(expected)
    assert prepare_wandb_metrics(result) == result
    assert raw == snapshot
    assert normalize_step_metrics(raw)["training/skipped_flag"] == 1
    assert "algorithm/trajectory/reward_mean" in normalize_step_metrics(raw)


def test_wandb_algorithm_names_cover_the_catalog_and_sort_by_stable_ranks():
    registered = {
        key for key in METRIC_SPECS if key.startswith("algorithm/") and "/mis/" not in key
    }
    assert set(ALGORITHM_METRICS) == registered
    assert len(ALGORITHM_METRICS) == 64
    names = [wandb_metric_name(key) for key in ALGORITHM_METRICS]
    assert names == sorted(names)
    assert len(set(names)) == 64
    partial = prepare_wandb_metrics(
        {"algorithm/opd/valid_tokens": 1, "trajectory/advantage_zero_percentage": 0}
    )
    assert list(partial) == [
        "01_algorithm/06_trajectory/advantage_zero_percentage",
        "01_algorithm/55_opd/valid_tokens",
    ]


def test_wandb_definitions_match_every_published_fixed_catalog_metric():
    metrics = {key: 1 for key in METRIC_SPECS if "<" not in key}
    published = prepare_wandb_metrics(metrics)
    run = SimpleNamespace(define_metric=MagicMock())
    define_wandb_metrics(run)
    definitions = {call.args[0]: call.kwargs for call in run.define_metric.call_args_list}
    assert set(definitions) - {"*"} == set(published) | {STEP_METRIC}
    assert len(run.define_metric.call_args_list) == len(definitions)
    assert all(definition["step_metric"] == STEP_METRIC for definition in definitions.values())
    assert definitions[STEP_METRIC]["hidden"] is True


def test_custom_namespaces_do_not_collide_with_regrouped_validation_counts():
    raw = {
        "val/custom_observed_count": 0,
        "count_val/custom_observed_count": 10,
        "plugin/metric": 3,
        "99_custom/value": 4,
        "plain_value": 5,
    }
    assert prepare_wandb_metrics(raw) == {
        "17_count_val/custom_observed_count": 0,
        "count_val/custom_observed_count": 10,
        "plugin/metric": 3,
        "99_custom/value": 4,
        "plain_value": 5,
    }


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("explicit", [None, float("nan"), float("inf"), 0.0, 0.9])
def test_explicit_numbered_observations_win_before_missing_values_are_filtered(reverse, explicit):
    key = "01_algorithm/01_trajectory/reward_mean"
    pairs = [
        ("trajectory/reward_mean", 0.1),
        ("algorithm/trajectory/reward_mean", 0.2),
        (key, explicit),
    ]
    result = prepare_wandb_metrics(dict(reversed(pairs) if reverse else pairs))
    expected = {key: explicit} if explicit in (0.0, 0.9) else {}
    assert result == expected
    assert prepare_wandb_metrics(result) == result
