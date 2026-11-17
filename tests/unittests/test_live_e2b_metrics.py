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

"""Live metrics remain accurate while rollout workers run concurrently."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import northrl.activity_tracker as tracker_module
from northrl.activity_tracker import ActivityTracker, ActivityTrackerProxy


@pytest.fixture
def live_tracker(monkeypatch, rollout_worker_config):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(tracker_module, "Tracking", MagicMock())
    rollout_worker_config.runtime.progress.update_interval_s = 60.0
    return ActivityTracker(rollout_worker_config), clock


def emit(tracker, kind, rollout_id="r1", attempt=1, **kwargs):
    tracker.record_e2b_event({"kind": kind, "rollout_id": rollout_id, "attempt": attempt, **kwargs})


def test_concurrent_snapshots_aggregate_deltas_and_context_max(live_tracker):
    tracker, _ = live_tracker

    def worker(index):
        rollout_id = f"r{index}"
        emit(tracker, "attempt_started", rollout_id)
        for _ in range(2):
            emit(
                tracker,
                "snapshot",
                rollout_id,
                metrics={
                    "model_turns": 2,
                    "generated_tokens": 20,
                    "max_context_tokens": 100 + index,
                    "tool_calls_emitted": 3,
                },
            )
        emit(
            tracker,
            "snapshot",
            rollout_id,
            metrics={"model_turns": 1, "generated_tokens": 10},
        )
        emit(
            tracker,
            "snapshot",
            rollout_id,
            metrics={"model_turns": 3, "generated_tokens": 30},
        )
        emit(tracker, "attempt_finished", rollout_id, status="success", duration_seconds=5)
        emit(tracker, "snapshot", rollout_id, metrics={"model_turns": 100})
        emit(tracker, "rollout_finished", rollout_id, status="success", attempts=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(32)))
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/model_turns"] == 96
    assert metrics["live/trace/generated_tokens"] == 960
    assert metrics["live/trace/max_context_tokens"] == 131
    assert metrics["live/tool/calls_emitted"] == 96
    assert metrics["live/e2b/completed_rollouts"] == 32
    assert metrics["live/e2b/active_rollouts"] == 0
    assert tracker._e2b_attempts == {}


def test_stale_snapshot_does_not_refresh_heartbeat(live_tracker):
    tracker, clock = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "sandbox_created", sandbox_id="sandbox", timeout_seconds=60)
    assert tracker.get_e2b_metrics()["live/e2b/heartbeat_pending_sandboxes"] == 1
    emit(tracker, "snapshot", heartbeat=1, metrics={})
    clock.now = 115
    emit(tracker, "snapshot", heartbeat=1, metrics={})
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/heartbeat_age_seconds_max"] == 15
    assert metrics["live/e2b/ttl_remaining_seconds_min"] == 45
    emit(tracker, "poll_error")
    assert tracker.get_e2b_metrics()["live/e2b/unknown_sandboxes"] == 1
    emit(tracker, "snapshot", heartbeat=2, metrics={})
    assert tracker.get_e2b_metrics()["live/e2b/heartbeat_age_seconds_max"] == 0
    assert tracker.get_e2b_metrics()["live/e2b/unknown_sandboxes"] == 0


def test_retry_failure_retains_metrics_and_unknown_sandbox(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "sandbox_created", timeout_seconds=60)
    emit(tracker, "snapshot", metrics={"tool_format_error_turns": 2})
    emit(tracker, "sandbox_closed", confirmed=False)
    emit(tracker, "attempt_finished", status="env_error", timed_out=True, duration_seconds=5)
    # The rollout remains active while transitioning to its retry.
    assert tracker.get_e2b_metrics()["live/e2b/active_rollouts"] == 1
    emit(tracker, "attempt_started", attempt=2)
    assert tracker.get_e2b_metrics()["live/e2b/active_rollouts"] == 1
    emit(tracker, "sandbox_created", attempt=2, timeout_seconds=60)
    emit(tracker, "snapshot", attempt=2, metrics={"tool_format_error_turns": 1})
    emit(tracker, "sandbox_closed", attempt=2, confirmed=True)
    emit(tracker, "attempt_finished", attempt=2, status="success", duration_seconds=15)
    emit(tracker, "rollout_finished", status="success", attempts=2)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/tool/format_error_turns"] == 3
    assert metrics["live/e2b/attempt_count"] == 2
    assert metrics["live/e2b/retry_count"] == 1
    assert metrics["live/e2b/completed_rollouts"] == 1
    assert metrics["live/e2b/active_rollouts"] == 0
    assert metrics["live/e2b/env_error_count"] == 0
    assert metrics["live/e2b/env_error_attempts"] == 1
    assert metrics["live/e2b/timeout_sandbox"] == 1
    assert metrics["live/e2b/timeout_rate"] == 0.5
    assert metrics["live/e2b/attempt_duration_seconds_mean"] == 10
    assert metrics["live/e2b/active_sandboxes"] == 1
    assert metrics["live/e2b/unknown_sandboxes"] == 1
    assert metrics["live/e2b/cleanup_error_count"] == 1


def test_unconfirmed_sandbox_expires_without_resetting_failure_totals(live_tracker):
    tracker, clock = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "sandbox_created", timeout_seconds=60)
    emit(tracker, "snapshot", heartbeat=1, metrics={"model_turns": 2})
    emit(tracker, "sandbox_closed", confirmed=False)
    emit(tracker, "attempt_finished", status="env_error", timed_out=True)
    emit(tracker, "rollout_finished", status="env_error")
    clock.now = 159
    assert tracker.get_e2b_metrics()["live/e2b/unknown_sandboxes"] == 1
    clock.now = 160
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/active_sandboxes"] == 0
    assert metrics["live/e2b/unknown_sandboxes"] == 0
    assert metrics["live/e2b/heartbeat_age_seconds_max"] == 0
    assert "live/e2b/ttl_remaining_seconds_min" not in metrics
    assert metrics["live/e2b/cleanup_error_count"] == 1
    assert metrics["live/e2b/timeout_sandbox"] == 1
    assert metrics["live/trace/model_turns"] == 2
    assert tracker._e2b_sandboxes == {}


def test_expired_sandbox_still_records_attempt_timeout_and_cleanup_failure(live_tracker):
    tracker, clock = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "sandbox_created", timeout_seconds=60)
    clock.now = 160
    assert tracker.get_e2b_metrics()["live/e2b/active_sandboxes"] == 0
    emit(tracker, "sandbox_closed", confirmed=False)
    emit(tracker, "attempt_finished", status="env_error", timed_out=True)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/cleanup_error_count"] == 1
    assert metrics["live/e2b/timeout_sandbox"] == 1


def test_late_cleanup_confirmation_removes_finished_attempt_sandbox(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "sandbox_created", timeout_seconds=60)
    emit(tracker, "sandbox_closed", confirmed=False)
    emit(tracker, "attempt_finished", status="success")
    emit(tracker, "rollout_finished", status="success", attempts=1)
    assert tracker.get_e2b_metrics()["live/e2b/unknown_sandboxes"] == 1
    emit(tracker, "sandbox_closed", confirmed=True)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/active_sandboxes"] == 0
    assert metrics["live/e2b/cleanup_error_count"] == 1
    assert metrics["live/e2b/successful_rollouts"] == 1
    assert tracker._e2b_sandboxes == {}


@pytest.mark.parametrize("cleanup_confirmed", [True, False])
def test_timeout_sandboxes_count_retries_once_after_cleanup(live_tracker, cleanup_confirmed):
    tracker, _ = live_tracker
    assert tracker.get_e2b_metrics()["live/e2b/timeout_sandbox"] == 0
    for attempt in (1, 2):
        emit(tracker, "attempt_started", attempt=attempt)
        emit(tracker, "sandbox_created", attempt=attempt, timeout_seconds=60)
        emit(tracker, "sandbox_closed", attempt=attempt, confirmed=cleanup_confirmed)
        assert tracker.get_e2b_metrics()["live/e2b/timeout_sandbox"] == attempt - 1
        for _ in range(2):
            emit(tracker, "attempt_finished", attempt=attempt, status="env_error", timed_out=True)
        assert tracker.get_e2b_metrics()["live/e2b/timeout_sandbox"] == attempt
    emit(tracker, "rollout_finished", status="env_error", attempts=2)
    tracker.flush_e2b_metrics(force=True)
    published = tracker.experiment_logger.log_live.call_args.args[0]
    assert published["live/e2b/timeout_sandbox"] == 2
    assert published["live/e2b/timeout_count"] == 2
    assert published["live/e2b/timeout_rate"] == 1


@pytest.mark.parametrize("timed_out", [True, False])
@pytest.mark.parametrize("sandbox_created", [True, False])
def test_timeout_sandboxes_require_creation_and_timeout(live_tracker, sandbox_created, timed_out):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    if sandbox_created:
        emit(tracker, "sandbox_created", timeout_seconds=60)
        emit(tracker, "sandbox_closed", confirmed=True)
    emit(tracker, "attempt_finished", status="env_error", timed_out=timed_out)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/timeout_sandbox"] == int(sandbox_created and timed_out)
    assert metrics["live/e2b/timeout_count"] == int(timed_out)


def test_runtime_monitoring_writes_all_categories_to_logs_without_metric_backends(
    monkeypatch, rollout_worker_config, caplog
):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: clock.now)
    rollout_worker_config.runtime.logger.backends = []
    tracker = ActivityTracker(rollout_worker_config)
    tracker.experiment_logger.log = MagicMock()
    emit(tracker, "attempt_started")
    emit(tracker, "sandbox_created", timeout_seconds=60)
    emit(
        tracker,
        "snapshot",
        heartbeat=1,
        metrics={"model_turns": 2, "tool_calls_emitted": 3, "model_requests_started": 4},
    )

    try:
        with caplog.at_level(logging.INFO, logger="northrl.utils.tracking"):
            assert tracker.flush_e2b_metrics() == 130
            clock.now = 129
            assert tracker.flush_e2b_metrics() == 130
            emit(tracker, "attempt_finished", status="success")
            emit(tracker, "rollout_finished", status="success", trajectory_segments=1)
            tracker.flush_e2b_metrics(force=True)

        records = [record for record in caplog.records if "[rollout-monitor]" in record.message]
        assert len(records) == 2
        first = json.loads(records[0].message.split("metrics=", 1)[1])
        final = json.loads(records[1].message.split("metrics=", 1)[1])
        assert first["live/e2b/active_sandboxes"] == 1
        assert first["live/trace/model_turns"] == 2
        assert first["live/tool/calls_emitted"] == 3
        assert first["live/inference/request_count"] == 4
        assert final["live/e2b/active_rollouts"] == 0
        assert final["live/e2b/completed_rollouts"] == 1
        assert tracker.get_training_step() == 0
        tracker.experiment_logger.log.assert_not_called()
    finally:
        tracker.experiment_logger.close()


def test_live_flush_runs_before_training_and_independently_of_progress(live_tracker):
    tracker, clock = live_tracker
    logger = tracker.experiment_logger
    emit(tracker, "attempt_started")
    assert tracker.update_progress() == 130
    assert logger.log_live.call_count == 1
    assert tracker.get_training_step() == 0
    clock.now = 129
    assert tracker.update_progress() == 130
    assert logger.log_live.call_count == 1
    clock.now = 130
    assert tracker.update_progress() == 160
    assert logger.log_live.call_count == 2
    assert logger.log_live.call_args.kwargs["elapsed_seconds"] == 30
    clock.now = 131
    tracker.flush_e2b_metrics(force=True)
    assert logger.log_live.call_count == 3


def test_flush_failure_does_not_stop_rollouts_or_spin(live_tracker):
    tracker, clock = live_tracker
    emit(tracker, "attempt_started")
    tracker.experiment_logger.log_live.side_effect = RuntimeError("unavailable")
    assert tracker.update_progress() == 130
    clock.now = 129
    assert tracker.update_progress() == 130
    assert tracker.experiment_logger.log_live.call_count == 1
    tracker.experiment_logger.log_live.side_effect = None
    clock.now = 130
    assert tracker.update_progress() == 160
    assert tracker.experiment_logger.log_live.call_count == 2


def test_proxy_records_events_without_logging(live_tracker):
    tracker, _ = live_tracker
    proxy = ActivityTrackerProxy(tracker)
    proxy.record_e2b_event({"kind": "attempt_started", "rollout_id": "r1", "attempt": 1})
    assert tracker.get_e2b_metrics()["live/e2b/attempt_count"] == 1
    tracker.experiment_logger.log_live.assert_not_called()


def test_live_metrics_collect_and_publish_without_configuration(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    tracker.update_progress()
    assert tracker.get_e2b_metrics()["live/e2b/attempt_count"] == 1
    tracker.experiment_logger.log_live.assert_called_once()


def test_active_trace_lengths_and_rates_exclude_finished_attempts(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "attempt_started", "r2")
    emit(tracker, "attempt_started", "r3")
    assert "live/tool/format_error_turn_rate" not in tracker.get_e2b_metrics()
    assert "live/tool/result_error_rate" not in tracker.get_e2b_metrics()
    emit(
        tracker,
        "snapshot",
        metrics={
            "model_turns": 2,
            "generated_tokens": 20,
            "trace_segments": 1,
            "tool_calls_emitted": 4,
            "tool_format_error_turns": 1,
            "tool_results_observed": 2,
            "tool_result_errors": 1,
        },
    )
    for _ in range(2):
        emit(
            tracker,
            "snapshot",
            "r2",
            metrics={
                "model_turns": 4,
                "generated_tokens": 40,
                "trace_segments": 3,
                "tool_calls_emitted": 6,
                "tool_results_observed": 2,
            },
        )
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/active_attempts_with_metrics"] == 2
    assert metrics["live/trace/active_model_turns_mean"] == 3
    assert metrics["live/trace/active_model_turns_max"] == 4
    assert metrics["live/trace/active_generated_tokens_mean"] == 30
    assert metrics["live/trace/active_generated_tokens_max"] == 40
    assert metrics["live/trace/active_segments_mean"] == 2
    assert metrics["live/trace/active_segments_max"] == 3
    assert metrics["live/tool/active_calls_emitted_mean"] == 5
    assert metrics["live/tool/active_calls_emitted_max"] == 6
    assert metrics["live/tool/format_error_turn_rate"] == pytest.approx(1 / 6)
    assert metrics["live/tool/result_error_rate"] == 0.25
    emit(tracker, "attempt_finished", status="success")
    emit(tracker, "attempt_finished", "r2", status="success")
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/active_attempts_with_metrics"] == 0
    assert metrics["live/trace/active_model_turns_mean"] == 0
    assert metrics["live/trace/active_generated_tokens_max"] == 0
    assert metrics["live/tool/active_calls_emitted_max"] == 0
    assert metrics["live/trace/model_turns"] == 6


def test_invalid_or_unknown_metrics_do_not_break_later_publication(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    emit(
        tracker,
        "snapshot",
        metrics={
            "unknown_key": 10,
            "model_turns": float("nan"),
            "generated_tokens": -1,
            "tool_calls_emitted": "bad",
        },
    )
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/model_turns"] == 0
    assert metrics["live/trace/generated_tokens"] == 0
    assert metrics["live/tool/calls_emitted"] == 0
    tracker.update_progress()
    tracker.experiment_logger.log_live.assert_called_once()


def test_final_segment_distribution_uses_result_once_after_retries(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "snapshot", metrics={"trace_segments": 7})
    emit(tracker, "attempt_finished", status="env_error")
    emit(tracker, "attempt_started", attempt=2)
    emit(tracker, "snapshot", attempt=2, metrics={"trace_segments": 1})
    assert "live/trace/segments_per_trajectory_mean" not in tracker.get_e2b_metrics()
    emit(tracker, "attempt_finished", attempt=2, status="success")
    # The result can be ahead of the last polled snapshot.
    for _ in range(2):
        emit(
            tracker,
            "rollout_finished",
            attempt=2,
            status="success",
            attempts=2,
            trajectory_segments=3,
        )
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/completed_rollouts"] == 1
    assert metrics["live/trace/segments"] == 8
    assert metrics["live/trace/rollouts_with_segments"] == 1
    assert metrics["live/trace/segments_per_trajectory_mean"] == 3
    assert metrics["live/trace/segments_per_rollout_max"] == 3
    assert metrics["live/trace/segments_per_rollout_p95"] == 3
    assert metrics["live/trace/split_trajectory_ratio"] == 1
    assert metrics["live/trace/active_segments_max"] == 0
    tracker.flush_e2b_metrics(force=True)
    logged = tracker.experiment_logger.log_live.call_args.args[0]
    assert logged["live/trace/segments_per_trajectory_mean"] == 3


def test_segment_distribution_weights_complete_rollouts_including_zero(live_tracker):
    tracker, _ = live_tracker
    for index, count in enumerate([1] * 19 + [3, 5]):
        emit(
            tracker,
            "rollout_finished",
            f"r{index}",
            status="task_failed",
            trajectory_segments=count,
        )
    emit(tracker, "rollout_finished", "zero", status="task_failed", trajectory_segments=0)
    # Environment failures cannot contribute training trajectories, and missing
    # observations must not become artificial zero-segment samples.
    emit(tracker, "rollout_finished", "env", status="env_error", trajectory_segments=9)
    emit(tracker, "rollout_finished", "cancelled", status="cancelled")
    emit(tracker, "rollout_finished", "missing", status="task_failed")
    for index, invalid in enumerate([True, -1, 2.5, "3"]):
        emit(
            tracker,
            "rollout_finished",
            f"invalid-{index}",
            status="success",
            trajectory_segments=invalid,
        )
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/rollouts_with_segments"] == 21
    assert metrics["live/e2b/valid_rollouts"] == 27
    assert metrics["live/trace/rollouts_without_segments"] == 1
    assert metrics["live/trace/segments_per_trajectory_mean"] == pytest.approx(27 / 22)
    assert metrics["live/trace/segments_per_rollout_max"] == 5
    assert metrics["live/trace/segments_per_rollout_p95"] == 3
    assert metrics["live/trace/split_trajectory_ratio"] == pytest.approx(2 / 22)


def test_zero_segment_rollout_contributes_a_complete_rollout_to_length_distribution(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "rollout_finished", status="task_failed", trajectory_segments=0)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/rollouts_without_segments"] == 1
    assert metrics["live/trace/rollouts_with_segments"] == 0
    assert metrics["live/e2b/valid_rollouts"] == 1
    assert metrics["live/trace/segments_per_trajectory_mean"] == 0
    assert metrics["live/trace/segments_per_rollout_max"] == 0
    assert metrics["live/trace/segments_per_rollout_p95"] == 0
    assert metrics["live/trace/split_trajectory_ratio"] == 0


def test_model_metadata_does_not_publish_version_metrics(live_tracker):
    tracker, _ = live_tracker
    emit(
        tracker,
        "attempt_started",
        model_path="weaver://weights/100",
        weight_version="northrl_actor_v000100",
    )
    emit(
        tracker,
        "snapshot",
        model_path="weaver://weights/100",
        weight_version="northrl_actor_v000100",
        metrics={"model_turns": 2, "generated_tokens": 20, "trace_segments": 1},
    )
    tracker.flush_e2b_metrics()
    metrics = tracker.experiment_logger.log_live.call_args.args[0]
    assert metrics["live/e2b/active_rollouts"] == 1
    assert metrics["live/e2b/attempt_count"] == 1
    assert metrics["live/trace/model_turns"] == 2
    assert metrics["live/trace/generated_tokens"] == 20
    assert metrics["live/trace/segments"] == 1
    assert not any(key.startswith("live/model/") for key in metrics)


@pytest.mark.parametrize("status", ["success", "task_failed", "env_error", "cancelled"])
def test_active_rollouts_count_logical_lifecycle_once(live_tracker, status):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started", "r1")
    emit(tracker, "attempt_started", "r1")
    emit(tracker, "attempt_started", "r2")
    assert tracker.get_e2b_metrics()["live/e2b/active_rollouts"] == 2

    emit(tracker, "attempt_finished", "r1", status=status)
    assert tracker.get_e2b_metrics()["live/e2b/active_rollouts"] == 2
    emit(tracker, "rollout_finished", "r1", status=status)
    assert tracker.get_e2b_metrics()["live/e2b/active_rollouts"] == 1

    # Duplicate terminal events and delayed start events cannot revive r1.
    emit(tracker, "rollout_finished", "r1", status=status)
    emit(tracker, "attempt_started", "r1")
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/active_rollouts"] == 1
    assert metrics["live/e2b/attempt_count"] == 2
    assert metrics["live/e2b/completed_rollouts"] == 1
    assert "live/e2b/active_attempts" not in metrics


def test_overlength_counts_final_rollouts_once_including_no_trajectories(live_tracker):
    tracker, _ = live_tracker
    initial = tracker.get_e2b_metrics()
    assert initial["live/e2b/overlength_count"] == 0
    assert initial["live/e2b/overlength_ratio"] == 0.0

    emit(
        tracker,
        "rollout_finished",
        "empty",
        status="success",
        termination_reason="context_length_exceeded",
        trajectory_segments=0,
    )
    for attempt in (1, 2):
        emit(
            tracker,
            "rollout_finished",
            "segmented",
            attempt=attempt,
            status="success",
            termination_reason="context_length_exceeded",
            trajectory_segments=3,
        )
    emit(tracker, "rollout_finished", "normal", status="success", trajectory_segments=1)
    emit(tracker, "rollout_finished", "failed", status="env_error", trajectory_segments=0)

    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/completed_rollouts"] == 4
    assert metrics["live/e2b/overlength_count"] == 2
    assert metrics["live/e2b/overlength_ratio"] == 0.5


def test_normal_rollout_reports_zero_overlength_metrics(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "rollout_finished", status="success", trajectory_segments=2)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/overlength_count"] == 0
    assert metrics["live/e2b/overlength_ratio"] == 0.0


def test_live_task_failures_are_valid_outcomes_separate_from_environment_errors(live_tracker):
    tracker, _ = live_tracker
    for rollout_id, status in [
        ("success", "success"),
        ("unsolved", "task_failed"),
        ("infra", "env_error"),
        ("cancelled", "cancelled"),
    ]:
        for _ in range(2):
            emit(tracker, "rollout_finished", rollout_id, status=status, trajectory_segments=0)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/completed_rollouts"] == 4
    assert metrics["live/e2b/valid_rollouts"] == 2
    assert metrics["live/e2b/successful_rollouts"] == 1
    assert metrics["live/e2b/task_failed_rollouts"] == 1
    assert metrics["live/e2b/env_error_count"] == 1
    assert metrics["live/e2b/env_error_rate"] == 0.25
    assert "live/trace/rollouts_with_segment_counts" not in metrics
    assert metrics["live/trace/rollouts_without_segments"] == 2
    assert metrics["live/trace/segments_per_trajectory_mean"] == 0


def test_finished_rollout_identity_window_is_bounded_without_resetting_totals(live_tracker):
    tracker, _ = live_tracker
    limit = tracker._MAX_RECENT_E2B_ROLLOUTS
    for index in range(limit + 1):
        emit(tracker, "rollout_finished", f"r{index}", status="success", trajectory_segments=1)
    assert len(tracker._e2b_finished_rollout_ids) == limit
    assert "r0" not in tracker._e2b_finished_rollout_ids
    assert tracker.get_e2b_metrics()["live/e2b/completed_rollouts"] == limit + 1
    assert tracker.get_e2b_metrics()["live/e2b/valid_rollouts"] == limit + 1

    # Recent duplicate terminal events and delayed starts remain idempotent.
    emit(tracker, "rollout_finished", "r1", status="success", trajectory_segments=1)
    emit(tracker, "attempt_started", "r1")
    assert tracker.get_e2b_metrics()["live/e2b/completed_rollouts"] == limit + 1
    assert tracker.get_e2b_metrics()["live/e2b/active_rollouts"] == 0

    # Beyond the FIFO window, identity is deliberately forgotten. The executor's
    # synchronous callbacks cannot replay these old events, but external callers
    # must not assume lifetime deduplication after eviction.
    emit(tracker, "attempt_started", "r0")
    assert tracker.get_e2b_metrics()["live/e2b/active_rollouts"] == 1
    emit(tracker, "rollout_finished", "r0", status="task_failed", trajectory_segments=0)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/active_rollouts"] == 0
    assert metrics["live/e2b/completed_rollouts"] == limit + 2
    assert metrics["live/e2b/valid_rollouts"] == limit + 2
    assert len(tracker._e2b_finished_rollout_ids) == limit
    assert "r1" not in tracker._e2b_finished_rollout_ids


def test_late_attempt_completion_after_rollout_terminal_is_counted_once(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "rollout_finished", status="success", trajectory_segments=0)
    emit(tracker, "attempt_started")
    for _ in range(2):
        emit(tracker, "attempt_finished", status="success", duration_seconds=3)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/e2b/attempt_count"] == 1
    assert metrics["live/e2b/completed_attempts"] == 1
    assert metrics["live/e2b/completed_rollouts"] == 1
    assert metrics["live/e2b/attempt_duration_seconds_mean"] == 3
    assert tracker._e2b_attempts == {}


def test_unknown_format_turns_do_not_publish_an_error_rate(live_tracker):
    from northrl.remote_rollout.metrics import RolloutMetrics

    tracker, _ = live_tracker
    observed = RolloutMetrics()
    observed.record_model_result({"choices": [{"message": {"content": "answer"}}]})
    emit(tracker, "attempt_started")
    for _ in range(2):
        emit(tracker, "snapshot", metrics=observed.snapshot())
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/model_turns"] == 1
    assert metrics["live/tool/format_unknown_turns"] == 1
    assert metrics["live/tool/format_error_turns"] == 0
    assert "live/tool/format_error_turn_rate" not in metrics

    observed.record_model_result({"northrl_parse_diagnostics": {"tool_format_checked": 1}})
    for _ in range(2):
        emit(tracker, "snapshot", metrics=observed.snapshot())
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/model_turns"] == 2
    assert metrics["live/tool/format_unknown_turns"] == 1
    assert metrics["live/tool/format_error_turn_rate"] == 0.0


def test_format_error_rate_uses_known_turns_across_repeated_attempt_snapshots(live_tracker):
    from northrl.remote_rollout.metrics import RolloutMetrics

    tracker, _ = live_tracker
    responses_by_attempt = [
        [
            {},
            {"northrl_parse_diagnostics": {"tool_format_checked": 1}},
            {
                "northrl_parse_diagnostics": {
                    "tool_format_checked": 0,
                    "tool_format_error": 1,
                    "tool_calls_invalid": 2,
                }
            },
        ],
        [
            {},
            {"northrl_parse_diagnostics": {"tool_format_checked": 0}},
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "call", "function": {"name": "Read", "arguments": "{bad"}}
                            ]
                        }
                    }
                ]
            },
        ],
    ]
    for attempt, responses in enumerate(responses_by_attempt, start=1):
        observed = RolloutMetrics()
        emit(tracker, "attempt_started", attempt=attempt)
        for response in responses:
            observed.record_model_result(response)
            for _ in range(2):
                emit(tracker, "snapshot", attempt=attempt, metrics=observed.snapshot())
        emit(tracker, "attempt_finished", attempt=attempt, status="env_error")
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/model_turns"] == 6
    assert metrics["live/tool/format_unknown_turns"] == 3
    assert metrics["live/tool/format_error_turns"] == 2
    assert metrics["live/tool/format_error_turn_rate"] == pytest.approx(2 / 3)
    assert metrics["live/tool/calls_invalid"] == 3
    assert metrics["live/tool/arguments_json_errors"] == 1


def test_northgate_diagnostics_reach_live_metrics_once_per_turn(live_tracker):
    # pylint: disable=redefined-outer-name
    from northrl.remote_rollout.metrics import TOOL_FORMAT_REASONS, RolloutMetrics

    tracker, _ = live_tracker
    observed = RolloutMetrics()
    emit(tracker, "attempt_started")
    diagnostics = [
        {"checked": False, "malformed": False, "detected": None, "invalid": None, "reasons": []},
        {"checked": True, "malformed": False, "detected": 0, "invalid": 0, "reasons": []},
        {
            "checked": False,
            "malformed": True,
            "detected": None,
            "invalid": None,
            "reasons": list(TOOL_FORMAT_REASONS) * 2 + ["future_reason"],
        },
    ]
    for diagnostic in diagnostics:
        observed.record_model_result({"tool_call_diagnostics": diagnostic})
        for _ in range(2):
            emit(tracker, "snapshot", metrics=observed.snapshot())
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/trace/model_turns"] == 3
    assert metrics["live/tool/format_unknown_turns"] == 1
    assert metrics["live/tool/format_error_turns"] == 1
    assert metrics["live/tool/format_error_turn_rate"] == 0.5
    assert metrics["live/tool/call_counts_unknown_turns"] == 2
    assert metrics["live/tool/calls_detected"] == metrics["live/tool/calls_invalid"] == 0
    for reason in TOOL_FORMAT_REASONS:
        assert metrics[f"live/tool/format_reason_{reason}_turns"] == 1
    assert not any("future_reason" in name for name in metrics)


def test_request_latency_merges_request_counts_and_histograms(live_tracker):
    from northrl.remote_rollout.metrics import REQUEST_DURATION_BUCKET_METRICS

    tracker, _ = live_tracker
    for rollout_id, durations in (("short", [1.0] * 19), ("long", [100.0])):
        emit(tracker, "attempt_started", rollout_id)
        snapshot = {
            "model_requests_started": len(durations),
            "model_requests_completed": len(durations),
            "model_requests_success": len(durations),
            "model_request_seconds_sum": sum(durations),
            "model_request_seconds_max": max(durations),
            **{
                name: sum(duration <= upper for duration in durations)
                for name, upper in REQUEST_DURATION_BUCKET_METRICS.items()
            },
        }
        emit(tracker, "snapshot", rollout_id, heartbeat=1, metrics=snapshot)
        emit(tracker, "snapshot", rollout_id, heartbeat=1, metrics=snapshot)
        emit(tracker, "snapshot", rollout_id, heartbeat=2, metrics=snapshot)
        emit(tracker, "attempt_finished", rollout_id, status="success")
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/inference/request_count"] == 20
    assert metrics["live/inference/completed_count"] == 20
    assert metrics["live/inference/request_seconds_mean"] == pytest.approx(119 / 20)
    assert metrics["live/inference/request_seconds_p95"] == 1
    assert metrics["live/inference/request_seconds_max"] == 100
    assert not any("bucket" in name for name in metrics)


def test_request_gauges_decrease_and_ignore_stale_snapshots(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    busy = {"model_request_inflight": 2, "model_request_oldest_inflight_seconds": 30}
    emit(tracker, "snapshot", heartbeat=1, metrics=busy)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/inference/inflight"] == 2
    assert metrics["live/inference/oldest_inflight_seconds"] == 30
    emit(
        tracker,
        "snapshot",
        heartbeat=2,
        metrics={"model_request_inflight": 0, "model_request_oldest_inflight_seconds": 0},
    )
    emit(tracker, "snapshot", heartbeat=1, metrics=busy)
    assert tracker.get_e2b_metrics()["live/inference/inflight"] == 0
    assert tracker.get_e2b_metrics()["live/inference/oldest_inflight_seconds"] == 0
    emit(tracker, "snapshot", heartbeat=3, metrics=busy)
    emit(tracker, "attempt_finished", status="env_error")
    assert tracker.get_e2b_metrics()["live/inference/inflight"] == 0
    assert "live/inference/request_seconds_mean" not in tracker.get_e2b_metrics()


def test_request_failures_and_retries_remain_in_latency_totals(live_tracker):
    tracker, _ = live_tracker
    for attempt, outcome in ((1, "timeout"), (2, "cancelled")):
        emit(tracker, "attempt_started", attempt=attempt)
        snapshot = {
            "model_requests_started": 1,
            "model_requests_completed": 1,
            f"model_requests_{outcome}": 1,
            "model_request_seconds_sum": 2000,
            "model_request_seconds_max": 2000,
            "model_request_seconds_bucket_le_inf": 1,
        }
        emit(tracker, "snapshot", attempt=attempt, heartbeat=1, metrics=snapshot)
        emit(tracker, "attempt_finished", attempt=attempt, status="env_error")
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/inference/request_count"] == 2
    assert metrics["live/inference/timeout_count"] == 1
    assert metrics["live/inference/cancelled_count"] == 1
    assert metrics["live/inference/request_seconds_mean"] == 2000
    assert metrics["live/inference/request_seconds_p95"] == 2000


def test_tool_error_counts_and_rollout_flags_survive_retries_without_double_counting(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    for count in (1, 1, 0, 2, 2):
        emit(
            tracker,
            "snapshot",
            metrics={
                "tool_results_observed": count,
                "tool_result_errors": count,
                "tool_error_command_not_found": count,
            },
        )
    # Active work contributes error counts, but not completed-rollout ratios.
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/tool/error_command_not_found"] == 2
    assert metrics["live/tool/env_issue_rollouts"] == 0
    assert "live/tool/env_issue_rollout_rate" not in metrics
    emit(tracker, "attempt_finished", status="env_error", timed_out=True)
    emit(tracker, "attempt_started", attempt=2)
    for _ in range(2):
        emit(
            tracker,
            "snapshot",
            attempt=2,
            metrics={"tool_results_observed": 3, "tool_result_errors": 1, "tool_error_edit": 1},
        )
    emit(tracker, "attempt_finished", attempt=2, status="success")
    for _ in range(2):
        emit(tracker, "rollout_finished", attempt=2, status="success", attempts=2)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/tool/results_observed"] == 5
    assert metrics["live/tool/result_errors"] == 3
    assert metrics["live/tool/error_command_not_found"] == 2
    assert metrics["live/tool/error_edit"] == 1
    assert metrics["live/tool/result_error_rate"] == pytest.approx(3 / 5)
    assert metrics["live/tool/observed_rollouts"] == 1
    assert metrics["live/tool/unobserved_rollouts"] == 0
    assert metrics["live/tool/env_issue_rollouts"] == 1
    assert metrics["live/tool/env_issue_rollout_rate"] == 1
    assert metrics["live/tool/timeout_observed_rollouts"] == 1
    assert metrics["live/tool/timeout_env_issue_rollouts"] == 1
    assert metrics["live/tool/timeout_env_issue_rollout_rate"] == 1
    assert metrics["live/e2b/env_error_count"] == 0
    assert tracker._e2b_tool_rollout_flags == {}


def test_tool_error_categories_only_mark_environment_clues_for_environment_errors(live_tracker):
    from northrl.remote_rollout.tool_errors import TOOL_ERROR_CATEGORIES

    tracker, _ = live_tracker
    environment_categories = {
        "command_not_found",
        "dependency_missing",
        "network",
        "permission",
        "resource",
    }
    for category in TOOL_ERROR_CATEGORIES:
        emit(tracker, "attempt_started", category)
        emit(
            tracker,
            "snapshot",
            category,
            metrics={
                "tool_results_observed": 1,
                "tool_result_errors": 1,
                f"tool_error_{category}": 1,
            },
        )
        emit(tracker, "attempt_finished", category, status="task_failed", timed_out=True)
        emit(tracker, "rollout_finished", category, status="task_failed")
    metrics = tracker.get_e2b_metrics()
    for category in TOOL_ERROR_CATEGORIES:
        assert metrics[f"live/tool/error_{category}"] == 1
    assert metrics["live/tool/observed_rollouts"] == len(TOOL_ERROR_CATEGORIES)
    assert metrics["live/tool/env_issue_rollouts"] == len(environment_categories)
    assert metrics["live/tool/env_issue_rollout_rate"] == pytest.approx(5 / 9)
    assert metrics["live/tool/timeout_env_issue_rollout_rate"] == pytest.approx(5 / 9)


def test_tool_observation_coverage_excludes_missing_and_invalid_observations(live_tracker):
    tracker, _ = live_tracker
    initial = tracker.get_e2b_metrics()
    assert initial["live/tool/error_command_not_found"] == 0
    assert initial["live/tool/observed_rollouts"] == 0
    for name in (
        "observation_coverage",
        "env_issue_rollout_rate",
        "timeout_env_issue_rollout_rate",
    ):
        assert f"live/tool/{name}" not in initial
    unobserved_snapshots = (
        {},
        {"model_turns": 1},
        {"tool_results_observed": 0},
        {"tool_results_observed": True},
        {"tool_results_observed": -1},
        {"tool_results_observed": float("nan")},
        {"tool_results_observed": "1"},
    )
    for index, snapshot in enumerate(unobserved_snapshots):
        rollout_id = f"unobserved-{index}"
        emit(tracker, "attempt_started", rollout_id)
        emit(tracker, "snapshot", rollout_id, metrics=snapshot)
        emit(tracker, "attempt_finished", rollout_id, status="env_error", timed_out=True)
        emit(tracker, "rollout_finished", rollout_id, status="env_error")
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/tool/unobserved_rollouts"] == len(unobserved_snapshots)
    assert metrics["live/tool/observation_coverage"] == 0
    assert "live/tool/env_issue_rollout_rate" not in metrics
    assert "live/tool/timeout_env_issue_rollout_rate" not in metrics
    assert "live/tool/result_error_rate" not in metrics
    assert metrics["live/tool/timeout_observed_rollouts"] == 0

    emit(tracker, "attempt_started", "observed")
    emit(tracker, "snapshot", "observed", metrics={"tool_results_observed": 1})
    emit(tracker, "attempt_finished", "observed", status="success")
    emit(tracker, "rollout_finished", "observed", status="success")
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/tool/observed_rollouts"] == 1
    assert metrics["live/tool/observation_coverage"] == pytest.approx(1 / 8)
    assert metrics["live/tool/env_issue_rollout_rate"] == 0
    assert "live/tool/timeout_env_issue_rollout_rate" not in metrics


def test_tool_timeout_cooccurrence_joins_all_attempts_but_only_completed_rollouts(live_tracker):
    tracker, _ = live_tracker
    # Timeout and an environment clue can occur in different attempts. This
    # measures cooccurrence across one logical task, not proof of causation.
    emit(tracker, "attempt_started")
    emit(tracker, "attempt_finished", status="env_error", timed_out=True)
    emit(tracker, "attempt_started", attempt=2)
    emit(
        tracker,
        "snapshot",
        attempt=2,
        metrics={"tool_results_observed": 1, "tool_result_errors": 1, "tool_error_network": 1},
    )
    emit(tracker, "attempt_finished", attempt=2, status="success")
    emit(tracker, "rollout_finished", attempt=2, status="success")
    emit(tracker, "attempt_started", "timeout-without-env")
    emit(tracker, "snapshot", "timeout-without-env", metrics={"tool_results_observed": 4})
    emit(tracker, "attempt_finished", "timeout-without-env", status="task_failed", timed_out=True)
    emit(tracker, "rollout_finished", "timeout-without-env", status="task_failed")
    emit(tracker, "attempt_started", "active-env")
    emit(
        tracker,
        "snapshot",
        "active-env",
        metrics={"tool_results_observed": 1, "tool_result_errors": 1, "tool_error_network": 1},
    )
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/tool/error_network"] == 2
    assert metrics["live/tool/observed_rollouts"] == 2
    assert metrics["live/tool/env_issue_rollouts"] == 1
    assert metrics["live/tool/env_issue_rollout_rate"] == 0.5
    assert metrics["live/tool/timeout_observed_rollouts"] == 2
    assert metrics["live/tool/timeout_env_issue_rollouts"] == 1
    assert metrics["live/tool/timeout_env_issue_rollout_rate"] == 0.5


def test_tool_rollout_completion_ignores_late_snapshots_and_duplicate_events(live_tracker):
    tracker, _ = live_tracker
    emit(tracker, "attempt_started")
    emit(tracker, "snapshot", metrics={"tool_results_observed": 1})
    # Even an out-of-order terminal event closes the observation cohort. A late
    # attempt completion still updates existing attempt counters, not this cohort.
    emit(tracker, "rollout_finished", status="success")
    for _ in range(2):
        emit(tracker, "attempt_started")
        emit(
            tracker,
            "snapshot",
            metrics={"tool_results_observed": 2, "tool_result_errors": 1, "tool_error_resource": 1},
        )
        emit(tracker, "rollout_finished", status="env_error")
        emit(tracker, "attempt_finished", status="env_error", timed_out=True)
    metrics = tracker.get_e2b_metrics()
    assert metrics["live/tool/results_observed"] == 1
    assert metrics["live/tool/result_errors"] == 0
    assert metrics["live/tool/error_resource"] == 0
    assert metrics["live/tool/observed_rollouts"] == 1
    assert metrics["live/tool/env_issue_rollouts"] == 0
    assert metrics["live/tool/timeout_observed_rollouts"] == 0
    assert metrics["live/e2b/completed_attempts"] == 1
    assert tracker._e2b_tool_rollout_flags == {}


def test_classified_tool_errors_reach_live_logger(live_tracker):
    from northrl.remote_rollout.metrics import RolloutMetrics

    tracker, _ = live_tracker
    observed = RolloutMetrics()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "cargo",
                    "name": "Bash",
                    "input": {"command": "cargo test"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "cargo",
                    "is_error": True,
                    "content": "Exit code 127\n/bin/bash: line 1: cargo: command not found",
                }
            ],
        },
    ]
    emit(tracker, "attempt_started")
    for _ in range(2):
        observed.observe_messages("anthropic", messages)
        emit(tracker, "snapshot", metrics=observed.snapshot())
    emit(tracker, "attempt_finished", status="task_failed", timed_out=True)
    emit(tracker, "rollout_finished", status="task_failed")
    tracker.flush_e2b_metrics(force=True)
    logged = tracker.experiment_logger.log_live.call_args.args[0]
    assert logged["live/tool/error_command_not_found"] == 1
    assert logged["live/tool/results_observed"] == 1
    assert logged["live/tool/result_errors"] == 1
    assert logged["live/tool/env_issue_rollout_rate"] == 1
    assert logged["live/tool/timeout_env_issue_rollout_rate"] == 1
    assert logged["live/tool/observation_coverage"] == 1
