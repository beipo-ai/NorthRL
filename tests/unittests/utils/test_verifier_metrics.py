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

"""Separate verifier service failures from ordinary zero-reward outcomes."""

from copy import deepcopy
from types import MappingProxyType

import pytest

from northrl.northrl_types import Trajectory
from northrl.utils.verifier_metrics import log_verifier_metrics

COUNTERS = (
    "result_count",
    "service_error_count",
    "judge_error_count",
    "timeout_after_retry_count",
    "no_code_count",
    "timeout_count",
    "retry_count",
    "retried_trajectory_count",
    "retry_completed_count",
)


def _parent(score=None, reward_fn=None, **fields):
    extra = {**fields}
    if score is not None:
        extra["score"] = score
    if reward_fn is not None:
        extra["reward_fn"] = reward_fn
    return Trajectory(tokens=[1, 2], loss_mask=[0, 1], reward=0, extra_fields=extra)


def test_mixed_outcomes_count_service_failures_and_retry_events_separately():
    parents = [
        _parent({"reason": "ok", "correct": True}, "code_e2b"),
        _parent({"reason": "wrong_answer", "correct": False}, "code_e2b"),
        _parent({"reason": "no_code"}, "code_e2b"),
        _parent(
            {"reason": "timeout_after_retry", "metrics": {"timeout_count": 5, "retry_count": 3}},
            "code_e2b",
        ),
        # A recovered evaluation can finish successfully while awarding zero.
        _parent(
            {
                "reason": "wrong_answer",
                "correct": False,
                "metrics": {"timeout_count": 2, "retry_count": 2},
            },
            "code_e2b",
        ),
        _parent({"reason": "reasoning360/stem/judge_error"}, "reasoning360"),
        _parent({"reason": "judge_error"}, "math"),
        _parent(
            {"reason": "finish_reason_length", "correct": False}, "ifeval", finish_reason="length"
        ),
        _parent({"correct": True}),
    ]
    metrics = {}
    log_verifier_metrics(parents, metrics)

    expected = {
        "result_count": 9,
        "service_error_count": 3,
        "judge_error_count": 2,
        "timeout_after_retry_count": 1,
        "no_code_count": 1,
        "timeout_count": 7,
        "retry_count": 5,
        "retried_trajectory_count": 2,
        "retry_completed_count": 1,
    }
    assert {name: metrics[f"verifier/{name}"] for name in COUNTERS} == expected
    assert metrics["reward_fn/code_e2b/verifier/result_count"] == 5
    assert metrics["reward_fn/code_e2b/verifier/service_error_count"] == 1
    assert metrics["reward_fn/code_e2b/verifier/no_code_count"] == 1
    assert metrics["reward_fn/code_e2b/verifier/retry_completed_count"] == 1
    assert metrics["reward_fn/reasoning360/verifier/judge_error_count"] == 1
    assert metrics["reward_fn/ifeval/verifier/service_error_count"] == 0
    assert metrics["reward_fn/ifeval/verifier/timeout_count"] == 0


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        True,
        False,
        -1,
        1.5,
        float("nan"),
        float("inf"),
        "2",
        pytest.param(10**400, id="overflow"),
    ],
)
def test_invalid_nested_event_counters_are_ignored_without_guessing_from_reason(invalid):
    parents = [
        _parent(
            {
                "reason": "timeout_after_retry",
                "metrics": {"timeout_count": invalid, "retry_count": invalid},
            },
            "code_e2b",
        )
    ]
    metrics = {}
    log_verifier_metrics(parents, metrics)

    assert metrics["verifier/result_count"] == 1
    assert metrics["verifier/service_error_count"] == 1
    assert metrics["verifier/timeout_after_retry_count"] == 1
    assert metrics["verifier/retried_trajectory_count"] == 1
    assert metrics["verifier/timeout_count"] == 0
    assert metrics["verifier/retry_count"] == 0
    assert metrics["verifier/retry_completed_count"] == 0


def test_only_nested_whitelisted_counters_and_parent_results_are_aggregated():
    parent = Trajectory(
        segments=[
            {
                "tokens": [1, 2],
                "loss_mask": [0, 1],
                "score": {"reason": "judge_error", "metrics": {"retry_count": 99}},
            }
            for _ in range(3)
        ],
        reward=0,
        extra_fields={
            "reward_fn": "code_e2b",
            "score": {
                "reason": "wrong_answer",
                "timeout_count": 100,
                "retry_count": 100,
                "metrics": {"timeout_count": 2.0, "retry_count": 1.0, "sandbox_retries": 100},
            },
            "runtime_metrics": {"sandbox_retries": 50, "verifier_duration_seconds": 12},
        },
    )
    original = deepcopy(dict(parent.items()))
    metrics = {"verifier/duration_seconds_mean": 12, "batch/sandbox_retry_count": 50}
    log_verifier_metrics([parent], metrics)

    assert metrics["verifier/result_count"] == 1
    assert metrics["verifier/judge_error_count"] == 0
    assert metrics["verifier/timeout_count"] == 2
    assert metrics["verifier/retry_count"] == 1
    assert metrics["verifier/retried_trajectory_count"] == 1
    assert metrics["verifier/retry_completed_count"] == 1
    assert metrics["verifier/duration_seconds_mean"] == 12
    assert metrics["batch/sandbox_retry_count"] == 50
    assert dict(parent.items()) == original


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (None, 0),
        ({}, 0),
        ([], 0),
        ({"metrics": {"retry_count": 1}}, 0),
        ({"reason": None}, 0),
        ({"reason": " "}, 0),
        ({"reason": 1}, 0),
        ({"correct": None, "reward": float("nan")}, 0),
        ({"reward": True}, 0),
        pytest.param({"reward_score": 10**400}, 0, id="overflow"),
        ({"duration_seconds": 4}, 0),
        ({"reason": "ok"}, 1),
        ({"correct": True}, 1),
        ({"correct": False}, 1),
        ({"reward": 0}, 1),
        ({"reward_score": 0.5}, 1),
        ({"format_correct": 1}, 1),
        (MappingProxyType({"correct": False}), 1),
    ],
)
def test_result_count_requires_an_observed_outcome(score, expected):
    metrics = {}
    log_verifier_metrics([_parent(score, "ifeval")], metrics)

    assert metrics["verifier/result_count"] == expected
    assert metrics["reward_fn/ifeval/verifier/result_count"] == expected


def test_prefix_empty_cohorts_and_snapshots_are_independent():
    metrics = {}
    log_verifier_metrics(
        [_parent({"reason": "judge_error"}, "reasoning360"), _parent({}, "ifeval")],
        metrics,
        prefix="val/",
    )
    assert all(key.startswith("val/") for key in metrics)
    assert metrics["val/verifier/service_error_count"] == 1
    assert all(metrics[f"val/reward_fn/ifeval/verifier/{name}"] == 0 for name in COUNTERS)

    next_metrics = {}
    log_verifier_metrics([_parent({"correct": False})], next_metrics)
    assert next_metrics["verifier/result_count"] == 1
    assert next_metrics["verifier/service_error_count"] == 0
    assert not any(key.startswith("reward_fn/") for key in next_metrics)

    empty_metrics = {}
    log_verifier_metrics([], empty_metrics)
    assert empty_metrics == {f"verifier/{name}": 0 for name in COUNTERS}
