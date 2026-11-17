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

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import northrl.activity_tracker as activity_tracker_module
from northrl.activity_tracker import ActivityTracker


@pytest.fixture
def progress_tracker(monkeypatch, rollout_worker_config):
    rollout_worker_config.runtime.progress.update_interval_s = 5.0
    rollout_worker_config.runtime.progress.log_every_n_steps = 3
    monkeypatch.setattr(activity_tracker_module, "Tracking", MagicMock())
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(activity_tracker_module.time, "monotonic", lambda: clock.now)
    return ActivityTracker(rollout_worker_config), clock


def test_progress_refreshes_immediately_and_obeys_interval(progress_tracker, monkeypatch):
    tracker, clock = progress_tracker
    collect = MagicMock(return_value={"active_rollout_workers": 2})
    display = MagicMock()
    monkeypatch.setattr(tracker, "get_progress_metrics", collect)
    monkeypatch.setattr(tracker, "_update_progress_display", display)

    assert tracker.update_progress() == 105.0
    collect.assert_called_once_with()
    display.assert_called_once_with({"active_rollout_workers": 2})

    clock.now = 104.9
    assert tracker.update_progress() == 105.0
    assert collect.call_count == 1

    clock.now = 105.0
    assert tracker.update_progress() == 110.0
    assert collect.call_count == 2

    # A delayed controller tick refreshes once, without replaying missed intervals.
    clock.now = 1000.0
    assert tracker.update_progress() == 1005.0
    assert tracker.update_progress() == 1005.0
    assert collect.call_count == 3
    assert display.call_count == 3


def test_progress_interval_starts_after_collection_finishes(progress_tracker, monkeypatch):
    tracker, clock = progress_tracker

    def collect_slow_metrics():
        clock.now += 10.0
        return {}

    collect = MagicMock(side_effect=collect_slow_metrics)
    monkeypatch.setattr(tracker, "get_progress_metrics", collect)
    tracker.update_progress()

    clock.now = 114.9
    tracker.update_progress()
    assert collect.call_count == 1

    clock.now = 115.0
    tracker.update_progress()
    assert collect.call_count == 2


def test_progress_preserves_log_cadence(progress_tracker, caplog):
    tracker, clock = progress_tracker

    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        for now in [100.0, 105.0, 110.0, 115.0]:
            clock.now = now
            tracker.update_progress()

    assert caplog.messages == ["Progress\n  Workers: 0 active"] * 2


def test_progress_preserves_tqdm_output(progress_tracker, monkeypatch):
    from tqdm import tqdm

    tracker, _ = progress_tracker
    tracker._progress_use_tqdm = True
    write = MagicMock()
    monkeypatch.setattr(tqdm, "write", write)

    tracker.update_progress()

    write.assert_called_once_with("Progress\n  Workers: 0 active")


@pytest.mark.parametrize("failing_method", ["get_progress_metrics", "_update_progress_display"])
def test_progress_failure_is_logged_and_retried_next_interval(
    progress_tracker, monkeypatch, caplog, failing_method
):
    tracker, clock = progress_tracker
    method = MagicMock(side_effect=[RuntimeError("metrics unavailable"), {}])
    monkeypatch.setattr(tracker, failing_method, method)

    with caplog.at_level(logging.ERROR, logger=activity_tracker_module.__name__):
        assert tracker.update_progress() == 105.0
        clock.now = 104.9
        assert tracker.update_progress() == 105.0
        assert method.call_count == 1

        clock.now = 105.0
        assert tracker.update_progress() == 110.0

    assert method.call_count == 2
    assert caplog.messages == ["Error updating rollout progress: metrics unavailable"]


def test_progress_runs_without_creating_a_thread(monkeypatch, rollout_worker_config):
    thread = MagicMock(side_effect=AssertionError("progress must run on the calling thread"))
    monkeypatch.setattr(activity_tracker_module.threading, "Thread", thread)
    monkeypatch.setattr(activity_tracker_module, "Tracking", MagicMock())

    tracker = ActivityTracker(rollout_worker_config)
    tracker.update_progress()

    thread.assert_not_called()


@pytest.mark.parametrize("sync_mode", ["sync", "batch-async"])
@pytest.mark.parametrize("use_tqdm", [False, True])
def test_ordered_round_display_uses_terminal_results(
    progress_tracker, caplog, monkeypatch, sync_mode, use_tqdm
):
    from tqdm import tqdm

    tracker, _ = progress_tracker
    tracker._progress_use_tqdm = use_tqdm
    write = MagicMock()
    monkeypatch.setattr(tqdm, "write", write)
    # Worker activity and drained pool inventory deliberately disagree with
    # completion, which must come from the preserved round snapshot.
    metrics = {
        "sync_mode": sync_mode,
        "active_rollout_workers": 99,
        "dataloader": {"batch_processed": 64, "batch_size": 64, "batch_remaining": 0},
        "trajectory_pool": {"ready_sequences": 0, "ready_groups": 0},
        "rollout_round": {
            "total": 64,
            "finished": 28,
            "in_progress": 36,
            "queued": 0,
            "groups_complete": 3,
            "groups_total": 8,
            "ready_segments": 39,
        },
    }
    expected = (
        f"Progress [{sync_mode}]\n"
        "  Rollouts: 28/64 finished | 36 in progress | 0 queued\n"
        "  Groups: 3/8 complete\n"
        "  Segments: 39 ready for training"
    )
    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        tracker._update_progress_display(metrics)
    if use_tqdm:
        write.assert_called_once_with(expected)
        assert caplog.messages == []
    else:
        assert caplog.messages == [expected]
        write.assert_not_called()


def test_continuous_sampling_reports_pool_inventory_without_round_fraction(
    progress_tracker, caplog
):
    tracker, _ = progress_tracker
    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        tracker._update_progress_display(
            {
                "sync_mode": "fully-async",
                "active_rollout_workers": 4,
                "trajectory_pool": {"ready_groups": 10, "ready_sequences": 73},
                "rollout_round": {"total": 8, "finished": 1},
            }
        )
    assert caplog.messages == [
        "Progress [fully-async]\n  Workers: 4 active\n"
        "  Pool groups: 10 ready\n  Pool segments: 73 ready for training"
    ]


def test_sft_round_omits_grpo_groups(progress_tracker, caplog):
    tracker, _ = progress_tracker
    tracker._config.algorithm.type = "sft"
    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        tracker._update_progress_display(
            {
                "sync_mode": "sync",
                "rollout_round": {
                    "total": 3,
                    "finished": 2,
                    "in_progress": 0,
                    "queued": 1,
                    "groups_complete": 2,
                    "groups_total": 3,
                    "ready_segments": 2,
                },
            }
        )
    assert caplog.messages == [
        "Progress [sync]\n  Rollouts: 2/3 finished | 0 in progress | 1 queued\n"
        "  Segments: 2 ready for training"
    ]


@pytest.mark.parametrize("algorithm, grouped", [("grpo", True), ("opd", True), ("sft", False)])
def test_metrics_collect_round_snapshot_from_loader(progress_tracker, algorithm, grouped):
    tracker, _ = progress_tracker
    tracker._config.algorithm.type = algorithm
    loader = MagicMock()
    loader.get_rollout_progress.return_value = {"total": 4, "finished": 1}
    tracker.register_module("data_loader-0", [loader])
    metrics = tracker.get_progress_metrics()
    assert metrics["rollout_round"] == {"total": 4, "finished": 1}
    loader.get_rollout_progress.assert_called_once_with(grouped=grouped)


@pytest.mark.parametrize("use_tqdm", [False, True])
def test_fully_async_display_distinguishes_cumulative_work_and_pool_inventory(
    progress_tracker, caplog, monkeypatch, use_tqdm
):
    from tqdm import tqdm

    tracker, _ = progress_tracker
    tracker._progress_use_tqdm = use_tqdm
    write = MagicMock()
    monkeypatch.setattr(tqdm, "write", write)
    metrics = {
        "sync_mode": "fully-async",
        "active_rollout_workers": 99,
        "rollout_round": {"total": 8, "finished": 1},
        "rollout_continuous": {"finished_total": 1240, "in_progress": 36, "queued": 20},
        "trajectory_pool": {
            "instances": {
                "actor": {
                    "ready_groups": 3,
                    "incomplete_groups": 5,
                    "groups_per_batch": 8,
                    "ready_sequences": 39,
                }
            }
        },
    }
    expected = (
        "Progress [fully-async]\n"
        "  Rollouts: 1240 finished total | 36 in progress | 20 queued\n"
        "  Pool groups: 3 ready | 5 partially returned | 8 per batch\n"
        "  Pool segments: 39 ready for training"
    )
    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        tracker._update_progress_display(metrics)
    if use_tqdm:
        write.assert_called_once_with(expected)
        assert caplog.messages == []
    else:
        assert caplog.messages == [expected]
        write.assert_not_called()


def test_fully_async_pools_have_separate_group_targets(progress_tracker, caplog):
    tracker, _ = progress_tracker
    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        tracker._update_progress_display(
            {
                "sync_mode": "fully-async",
                "rollout_continuous": {"finished_total": 64, "in_progress": 4, "queued": 0},
                "trajectory_pool": {
                    "ready_groups": 8,
                    "instances": {
                        "actor": {
                            "ready_groups": 2,
                            "incomplete_groups": 1,
                            "groups_per_batch": 8,
                            "ready_sequences": 17,
                        },
                        "other": {
                            "ready_groups": 6,
                            "incomplete_groups": 0,
                            "groups_per_batch": 4,
                            "ready_sequences": 29,
                        },
                    },
                },
            }
        )
    assert "Pool [actor] groups: 2 ready | 1 partially returned | 8 per batch" in caplog.text
    assert "Pool [other] groups: 6 ready | 0 partially returned | 4 per batch" in caplog.text
    assert "Pool [actor] segments: 17 ready for training" in caplog.text
    assert "Pool [other] segments: 29 ready for training" in caplog.text
    assert "Pool groups: 8 ready" not in caplog.text


def test_fully_async_startup_reports_zero_counts_before_pool_creation(progress_tracker, caplog):
    tracker, _ = progress_tracker
    tracker._config.data.train.batch_size = 8
    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        tracker._update_progress_display(
            {
                "sync_mode": "fully-async",
                "rollout_continuous": {"finished_total": 0, "in_progress": 0, "queued": 0},
                "trajectory_pool": {"instances": {}},
            }
        )
    assert caplog.messages == [
        "Progress [fully-async]\n"
        "  Rollouts: 0 finished total | 0 in progress | 0 queued\n"
        "  Pool groups: 0 ready | 0 partially returned | 8 per batch\n"
        "  Pool segments: 0 ready for training"
    ]


def test_metrics_collect_continuous_snapshot_independently_of_rounds(progress_tracker):
    tracker, _ = progress_tracker
    loader = MagicMock()
    loader.get_rollout_progress.return_value = None
    loader.get_continuous_rollout_progress.return_value = {
        "finished_total": 100,
        "in_progress": 5,
        "queued": 3,
    }
    tracker.register_module("data_loader-0", [loader])
    metrics = tracker.get_progress_metrics()
    assert metrics["rollout_round"] is None
    assert metrics["rollout_continuous"] == {"finished_total": 100, "in_progress": 5, "queued": 3}
    loader.get_continuous_rollout_progress.assert_called_once_with()


def test_fully_async_unavailable_pool_does_not_report_zero_inventory(progress_tracker, caplog):
    tracker, _ = progress_tracker
    with caplog.at_level(logging.INFO, logger=activity_tracker_module.__name__):
        tracker._update_progress_display(
            {
                "sync_mode": "fully-async",
                "rollout_continuous": {"finished_total": 100, "in_progress": 5, "queued": 3},
                "trajectory_pool": None,
            }
        )
    assert caplog.messages == [
        "Progress [fully-async]\n"
        "  Rollouts: 100 finished total | 5 in progress | 3 queued\n"
        "  Pool: unavailable"
    ]
