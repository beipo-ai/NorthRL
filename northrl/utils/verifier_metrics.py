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

"""Count reported verifier outcomes separately from task-quality scores."""

import logging
import math
from collections import defaultdict
from collections.abc import Mapping
from numbers import Real
from typing import Any

from northrl.northrl_types import Trajectory

logger = logging.getLogger(__name__)

_COUNTERS = (
    "result_count",
    "service_error_count",
    "judge_error_count",
    "timeout_count",
    "retry_count",
    "retried_trajectory_count",
    "retry_completed_count",
    "timeout_after_retry_count",
    "no_code_count",
)


def _finite_number(value: Any) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _count(value: Any) -> int:
    """Ignore malformed counters rather than guessing or treating bools as counts."""
    if _finite_number(value) and value >= 0 and float(value).is_integer():
        return int(value)
    return 0


def _observed_score(score: Any) -> bool:
    if not isinstance(score, Mapping):
        return False
    reason = score.get("reason")
    return (
        (isinstance(reason, str) and bool(reason.strip()))
        or isinstance(score.get("correct"), bool)
        or any(
            _finite_number(score.get(name)) for name in ("reward", "reward_score", "format_correct")
        )
    )


def _summary(trajectories: list[Trajectory]) -> dict[str, int]:
    counts = dict.fromkeys(_COUNTERS, 0)
    for parent in trajectories:
        score = parent.get("score")
        if not _observed_score(score):
            continue
        counts["result_count"] += 1
        reason = score.get("reason")
        judge_error = reason == "judge_error" or (
            isinstance(reason, str)
            and reason.startswith("reasoning360/")
            and reason.endswith("/judge_error")
        )
        exhausted = reason == "timeout_after_retry"
        service_error = judge_error or exhausted
        counts["judge_error_count"] += judge_error
        counts["timeout_after_retry_count"] += exhausted
        counts["service_error_count"] += service_error
        counts["no_code_count"] += reason == "no_code"

        reported = score.get("metrics")
        reported = reported if isinstance(reported, Mapping) else {}
        timeout_count = _count(reported.get("timeout_count"))
        retry_count = _count(reported.get("retry_count"))
        counts["timeout_count"] += timeout_count
        counts["retry_count"] += retry_count
        retried = retry_count > 0 or exhausted
        counts["retried_trajectory_count"] += retried
        counts["retry_completed_count"] += retried and not service_error
    return {f"verifier/{name}": value for name, value in counts.items()}


def log_verifier_metrics(
    trajectories: list[Trajectory], metrics: dict[str, Any], *, prefix: str = ""
) -> None:
    """Publish parent-result counts for this batch and each effective reward_fn.

    Service errors count terminal judge errors or exhausted sandbox timeouts,
    not ordinary incorrect answers, missing code, or length short-circuits.
    Timeout/retry totals sum only explicitly reported integer counters. A retry
    that completes an ordinary verdict counts even when that verdict is wrong.
    No segment scores, arbitrary numeric judge metrics, or runtime retry fields
    are used. The returned task reward and correctness are unchanged.
    """
    summary = _summary(trajectories)
    cohorts: dict[str, list[Trajectory]] = defaultdict(list)
    for parent in trajectories:
        name = parent.get("reward_fn")
        if isinstance(name, str) and name:
            cohorts[name].append(parent)
    for name, parents in cohorts.items():
        summary.update(
            {f"reward_fn/{name}/{key}": value for key, value in _summary(parents).items()}
        )
    published = {f"{prefix}{key}": value for key, value in summary.items()}
    metrics.update(published)
    logger.info("[verifier-metrics] %s", published)
