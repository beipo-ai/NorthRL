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

"""Canonical step metric definitions, filtering, and W&B history settings.

Training code uses stable canonical keys. At the W&B boundary, section and
chart ranks become part of the logged names, separating scores, timing, counts,
and MIS without a separate workspace tool. Metric definitions use those same
names and retain the hidden training-step axis. The tracking startup separately
configures workspace ordering through wandb_workspace. Missing data is never
filled with zero, and other logging backends keep their canonical metric names.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from northrl.utils.metric_definitions import METRIC_DEFINITIONS

STEP_METRIC = "training/train_step"
# Keep selected latest values in the run summary; all data series remain visible.
SUMMARY_METRICS = frozenset(
    {
        "batch/trajectory_count",
        "batch/trainable_segment_count",
        "batch/trainable_response_tokens_total",
        "algorithm/trajectory/reward_mean",
        "algorithm/trajectory/advantage_mean",
        "timing_rollout/trajectory_mean_seconds",
        "trajectory/response_tokens_mean",
        "segment/response_tokens_mean",
        "group/response_tokens_max",
        "training/optimizer_step_count",
        "weaver/loss",
        "timeout/tool_count",
        "timeout/model_request_count",
        "timeout/rollout_count",
    }
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    section: str
    unit: str
    description: str
    stage: str


METRIC_SPECS = {
    cells[0]: MetricSpec(*cells)
    for line in METRIC_DEFINITIONS.splitlines()
    if (cells := line.split("\t")) and len(cells) == 5
}


WANDB_SECTION_ORDER = (
    "algorithm",
    "val",
    "reward_fn",
    "outcome",
    "timing",
    "timing_rollout",
    "timing_reward",
    "timing_train",
    "timing_weights",
    "timing_val",
    "throughput",
    "mis",
    "weaver",
    "training",
    "batch",
    "count_reward",
    "count_val",
    "status_val",
    "verifier",
    "timeout",
    "trajectory",
    "model_request",
    "tool",
    "segment",
    "group",
)

# Stable display ranks are intentional: a missing observation must not renumber
# every later panel, and new algorithm fields must not claim an agreed rank.
ALGORITHM_METRICS = tuple(
    f"algorithm/{suffix}"
    for suffix in (
        "trajectory/reward_mean",
        "trajectory/reward_min",
        "trajectory/reward_max",
        "trajectory/reward_std",
        "trajectory/reward_observed_count",
        "trajectory/advantage_zero_percentage",
        "trajectory/advantage_abs_mean",
        "trajectory/advantage_std",
        "trajectory/advantage_mean",
        "trajectory/advantage_token_weighted_mean",
        "trajectory/advantage_min",
        "trajectory/advantage_max",
        "trajectory/advantage_positive_count",
        "trajectory/advantage_negative_count",
        "trajectory/advantage_zero_count",
        "group/reward_std_min",
        "group/reward_std_max",
        "kl/actor_ref_mean",
        "kl/actor_ref_loss",
        "kl/coefficient",
        "weaver/ppo_clip/ppo_kl",
        "weaver/ppo_clip/clipfrac",
        "weaver/ppo_clip/clipfrac_high",
        "weaver/ppo_clip/clipfrac_low",
        "weaver/entropy_loss",
        "weaver/entropy_coeff",
        "sampling/ppl_mean",
        "sampling/logprob_mean",
        "sampling/logprob_min",
        "sampling/logprob_max",
        "sampling/logprob_std",
        "weaver/is:ratio",
        "weaver/is:ratio_min",
        "weaver/is:ratio_max",
        "weaver/is:ratio_sq",
        "weaver/is:logratio",
        "weaver/is:logratio_min",
        "weaver/is:logratio_max",
        "weaver/is:logratio_std",
        "weaver/is:logratio_sq",
        "weaver/is:abs_logratio",
        "weaver/is:abs_logratio_max",
        "weaver/ratio:mean",
        "weaver/ratio:min",
        "weaver/ratio:max",
        "weaver/ratio:std",
        "weaver/ratio:gt1_frac",
        "weaver/ratio:gt2_frac",
        "weaver/ratio:gt5_frac",
        "weaver/logprob_diff:mean",
        "weaver/logprob_diff:min",
        "weaver/logprob_diff:max",
        "weaver/logprob_diff:std",
        "opd/teacher_kl",
        "opd/valid_tokens",
        # Append new fields so existing W&B history names retain their ranks.
        "weaver/entropy",
        "weaver/entropy:sum",
        "weaver/entropy:tokens",
        "weaver/gspo/ratio_mean",
        "weaver/gspo/ratio_min",
        "weaver/gspo/ratio_max",
        "weaver/gspo/clip_fraction",
        "weaver/gspo/active_trajectories",
        "weaver/gspo/empty_trajectories",
    )
)
_SECTION_NAMES = {name: f"{rank:02d}_{name}" for rank, name in enumerate(WANDB_SECTION_ORDER, 1)}
# Output-only section names are not producer namespaces: leave custom fields
# using those names untouched instead of merging them with regrouped metrics.
_PRODUCER_SECTIONS = frozenset(WANDB_SECTION_ORDER) - {
    "mis",
    "timing_reward",
    "count_reward",
    "count_val",
    "status_val",
}
_ALGORITHM_RANKS = {key: rank for rank, key in enumerate(ALGORITHM_METRICS, start=1)}
_REWARD_RANKS = {reducer: rank for rank, reducer in enumerate(("mean", "min", "max", "std"), 1)}
_REWARD_SCORE = re.compile(r"reward_fn/(?P<name>.+)/reward_(?P<reducer>mean|min|max|std)")
_REWARD_COUNT = re.compile(r"reward_fn/.+/(?:reward_observed_count|verifier/[^/]+_count)")
_REWARD_TIMING = re.compile(
    r"timing_rollout/(?P<name>.+)_(?:trajectory|group|verifier)_(?:mean|min|max)_seconds"
)
_VERIFIER_COUNTS = frozenset(
    (
        "result_count",
        "service_error_count",
        "judge_error_count",
        "timeout_count",
        "retry_count",
        "retried_trajectory_count",
        "retry_completed_count",
        "timeout_after_retry_count",
        "no_code_count",
        "skipped_length_count",
    )
)
_VAL_GLOBAL_COUNTS = frozenset(
    ("trajectory_count", "segment_count", "finish_reason_length_count", "duplicate_run_count")
)
_VAL_DATASET_COUNTS = frozenset(
    ("trajectory_count", "finish_reason_length_count", "num_prompts", "num_generations")
)


def _validation_section(key: str) -> str:
    """Recognize owned suffixes while preserving arbitrary cohort paths."""
    path = key.removeprefix("val/")
    cohort, _, field = path.rpartition("/")
    if path in {"k", "finish_reason_length_fraction"} or (
        cohort and field in {"finish_reason_length_fraction", "complete_group_ratio"}
    ):
        return "status_val"
    if path in _VAL_GLOBAL_COUNTS or (cohort and field in _VAL_DATASET_COUNTS):
        return "count_val"
    if re.fullmatch(r".+/observed_generations/mean", path):
        return "count_val"
    if field in _VERIFIER_COUNTS and (
        cohort == "verifier" or re.fullmatch(r"reward_fn/.+/verifier", cohort)
    ):
        return "count_val"
    # Registered score denominators also exist when no finite score was
    # observed: the validator emits count=0 and omits the corresponding mean.
    if re.fullmatch(r".+_observed_count", field):
        return "count_val"
    return "val"


def _wandb_metric_location(key: str) -> tuple[str, str]:
    if key.startswith("algorithm/mis/"):
        return "mis", key.removeprefix("algorithm/mis/")
    if key.startswith("algorithm/"):
        suffix = key.removeprefix("algorithm/")
        rank = _ALGORITHM_RANKS.get(key)
        return "algorithm", f"{rank:02d}_{suffix}" if rank else f"extra/{suffix}"
    if match := _REWARD_SCORE.fullmatch(key):
        reducer = match["reducer"]
        return "reward_fn", f"{_REWARD_RANKS[reducer]:02d}_reward_{reducer}/{match['name']}"
    if _REWARD_COUNT.fullmatch(key):
        return "count_reward", key.removeprefix("reward_fn/")
    if _REWARD_TIMING.fullmatch(key):
        return "timing_reward", key.removeprefix("timing_rollout/")
    if key.startswith("val/"):
        return _validation_section(key), key.removeprefix("val/")
    prefix, separator, suffix = key.partition("/")
    if separator and prefix in _PRODUCER_SECTIONS:
        return prefix, suffix
    return "", key


def wandb_metric_name(key: str) -> str:
    """Number a canonical metric at the W&B boundary, leaving the axis untouched.

    Numbers are real history keys, not panel titles. They sort by the agreed
    section and algorithm ranks when the workspace uses alphabetical ordering.
    Unknown namespaces and already numbered names retain their identities.
    """
    if key == STEP_METRIC:
        return key
    section, suffix = _wandb_metric_location(key)
    return f"{_SECTION_NAMES[section]}/{suffix}" if section else key


def prepare_wandb_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and number one W&B payload without changing producer data.

    Explicit numbered observations win over their unnumbered spelling, including
    explicit missing values. This keeps repeated preparation idempotent and
    prevents an older observation from filling a deliberately missing metric.
    """
    selected: dict[str, Any] = {}
    for key, value in sorted(normalize_step_metrics(metrics).items()):
        numbered = wandb_metric_name(key)
        if numbered != key and numbered in metrics:
            continue
        selected[numbered] = value
    return dict(sorted(selected.items()))


# Keep producer names and their aggregation contracts unchanged. Rename only
# registered algorithm fields at the logging boundary, preserving the original
# cohort/source below the common workspace section prefix.
ALGORITHM_ALIASES = {
    key.removeprefix("algorithm/"): key for key in METRIC_SPECS if key.startswith("algorithm/")
}

TIMING_ALIASES = {
    "timing_train/batch_completion_interval_seconds": "timing/batch_completion_interval_seconds",
    **{
        f"{section}/{stage}_seconds": f"timing_weights/{stage}_seconds"
        for section in ("timing", "timing_train")
        for stage in ("weight_export", "weight_export_failed", "weight_sync", "weight_sync_failed")
    },
    **{
        f"timing/{stage}": f"timing_train/{stage}_seconds"
        for stage in ("recompute_logprobs", "ref_logprobs")
    },
}

# Only synonyms with unchanged semantics are rewritten. Data/algorithm cohorts
# changed during this migration; their old fields must not be blindly renamed.
ALIASES = {
    "training/global_step": STEP_METRIC,
    "training/optimizer_steps": "training/optimizer_step_count",
    "training/forward_backward_calls": "training/forward_backward_call_count",
    "training/skipped": "training/skipped_flag",
    "training/weight_export_success": "training/weight_export_success_flag",
    "training/weight_sync_success": "training/weight_sync_success_flag",
    "timing/weight_export_time": "timing/weight_export_seconds",
    "timing/weight_export_failed_time": "timing/weight_export_failed_seconds",
    "timing/weight_sync_time": "timing/weight_sync_seconds",
    "timing/weight_sync_failed_time": "timing/weight_sync_failed_seconds",
    "mis/log_ppl_diff": "mis/kl",
    "val/sequence_count": "val/segment_count",
    "val/num_samples": "val/trajectory_count",
}
# Resolve existing aliases to their final presentation names in one pass.
ALIASES = {
    key: TIMING_ALIASES.get(value, ALGORITHM_ALIASES.get(value, value))
    for key, value in ALIASES.items()
}
ALIASES.update(ALGORITHM_ALIASES)
ALIASES.update(TIMING_ALIASES)
# Match owned duration fields only. Reward-function names are dynamic, while
# unrelated custom timing metrics keep their existing spelling and semantics.
DURATION_METRIC = re.compile(
    r"(?P<validation>val/)?(?P<timing>timing/)?"
    r"(?:reward_fn/(?P<reward_fn>.+)/)?"
    r"(?P<object>trajectory|group|verifier|model_request)/"
    r"duration_seconds_(?P<reducer>mean|min|max|observed_count)"
)
RETIRED_PREFIXES = (
    "live/",
    "rollout/",
    "round/",
    "datum/",
    "response_length/",
    "response_length_raw/",
    "prompt_length/",
    "critic/",
    "advantage/",
    "logprob/",
    "ppl/",
    "truncation/",
    "recompute_logprobs/",
)
RETIRED_KEYS = frozenset(
    {
        "training/rollout_count",
        "training/sequence_count",
        "training/trainable_sequences",
        "training/segment_count",
        "training/prompt_count",
        "training/batch_size",
        "training/num_datums",
        "training/num_batches",
        "timing/rollout_time",
        "timing/batch_release_to_claim_seconds",
        "timing/batch_claim_interval_seconds",
        "timing/batch_rollout_span_seconds",
        # These old totals include saving and cannot name the new training-only interval.
        "timing/batch_training_seconds",
        "timing/training_time",
        # The old diagnostic interval also includes statistics after the forward.
        "timing/mismatch_forward",
        *(
            f"{section}/{stage}{suffix}"
            for section in ("timing", "timing_train")
            for stage in ("mismatch_metrics", "mis", "prepare", "other")
            for suffix in ("", "_seconds", "_wall_seconds", "_time")
        ),
        "kl/ref_logprobs_attached",
        "kl/ref_logprobs_total",
        "kl/actor_to_ref_sync_time",
        "kl/actor_to_ref_sync_step",
    }
)
ALLOWED_WEAVER_KEYS = frozenset(
    source
    for key in METRIC_SPECS
    if (source := key.removeprefix("algorithm/")).startswith("weaver/")
)


def _is_retired(key: str) -> bool:
    key = key.removeprefix("algorithm/")
    if key in RETIRED_KEYS or key.startswith(RETIRED_PREFIXES):
        return True
    if key.startswith("weaver/") and key not in ALLOWED_WEAVER_KEYS:
        return True
    if key.startswith("val/") and re.search(r"/avg@\d+$", key):
        return True
    # Sampling-group timing replaces the legacy task namespace in all four scopes.
    if re.fullmatch(
        r"(?:val/)?(?:timing/)?(?:reward_fn/.+/)?"
        r"task/duration_seconds_(?:mean|min|max|observed_count)",
        key,
    ):
        return True
    # Restrict these presentation exclusions to owned namespaces. A custom
    # validation metric named coverage, for example, must remain loggable.
    if key.startswith(("batch/", "trajectory/", "segment/", "group/", "timeout/")):
        if any(
            word in key
            for word in ("coverage", "sequence_tokens", "context_tokens", "generated_tokens")
        ):
            return True
        if key.startswith("segment/") and ("duration" in key or "time_seconds" in key):
            return True
        if key.startswith("group/") and key.endswith("_mean"):
            return True
        if key.startswith("timeout/") and key.endswith(("_percentage", "_ratio")):
            return True
    return False


def normalize_step_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Remove retired/raw-only fields and verified aliases, without a global whitelist.

    Producers own aggregation and denominator validation. In particular, merely
    being in this catalog does not authorize averaging a raw service metric.
    Unknown custom metrics, including task-specific validation scores, survive.
    """
    selected: dict[str, tuple[int, Any]] = {}
    for key, value in metrics.items():
        canonical = ALIASES.get(key, key)
        priority = 3 if canonical == key else 1
        if key in TIMING_ALIASES:
            priority = 2
        elif key.startswith("timing/") and key in ALIASES:
            priority = 0
        if match := DURATION_METRIC.fullmatch(canonical):
            if match["reducer"] == "observed_count":
                continue
            section = "timing_val" if match["validation"] else "timing_rollout"
            cohort = f"{match['reward_fn']}_" if match["reward_fn"] else ""
            canonical = f"{section}/{cohort}{match['object']}_{match['reducer']}_seconds"
            priority = 2 if match["timing"] else 1
        if _is_retired(canonical):
            continue
        # Retain the existing MIS spelling preference when neither spelling has
        # reached the algorithm namespace yet; mapping must not depend on order.
        if key == "mis/log_ppl_diff" and "mis/kl" in metrics:
            continue
        if canonical not in selected or priority > selected[canonical][0]:
            selected[canonical] = (priority, value)

    # Resolve aliases before filtering so an explicit missing canonical value
    # cannot be replaced by a stale value submitted under an older spelling.
    normalized = {}
    for canonical, (_, value) in selected.items():
        if value is None or (isinstance(value, Real) and not math.isfinite(value)):
            continue
        normalized[canonical] = value
    return normalized


def define_wandb_metrics(run: Any) -> None:
    """Bind all visible data series to the completed training batch axis."""
    run.define_metric("*", step_metric=STEP_METRIC, step_sync=False, hidden=False, summary="none")
    run.define_metric(
        STEP_METRIC, step_metric=STEP_METRIC, step_sync=False, hidden=True, summary="last"
    )
    for key in sorted(METRIC_SPECS, key=wandb_metric_name):
        if key == STEP_METRIC or "<" in key:
            continue
        run.define_metric(
            wandb_metric_name(key),
            step_metric=STEP_METRIC,
            step_sync=False,
            hidden=False,
            summary="last" if key in SUMMARY_METRICS else "none",
        )
