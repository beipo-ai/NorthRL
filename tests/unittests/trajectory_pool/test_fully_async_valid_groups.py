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

"""Fully async batches count eligible groups instead of completed attempts."""

import copy
import gc
import logging
import weakref

import pytest

from northrl.algorithm.advantage.grpo import compute_grpo_advantage_for_trajectories

from .test_complete_groups import _instance, _put, _trajectory


def _async_instance(**options):
    return _instance(require_trainable_groups=True, **options)


def _complete(instance, group, *, reward=1.0, status="success", count=1):
    return [
        _put(instance, _trajectory(group, run, count, reward=reward, status=status))
        for run in range(2)
    ]


@pytest.mark.parametrize(
    "reward,status",
    [
        (1.0, "failed"),
        (1.0, "skipped"),
        (None, "success"),
        (float("nan"), "success"),
        (float("inf"), "success"),
        (float("-inf"), "success"),
    ],
)
def test_half_failed_or_unscored_group_counts_toward_batch_and_watermark(reward, status):
    instance, _ = _async_instance(groups_per_batch=2)
    valid = _complete(instance, "valid")
    donor = _put(instance, _trajectory("repairable", 0, 2))
    invalid = _put(instance, _trajectory("repairable", 1, 1, reward=reward, status=status))

    assert instance.ready_group_count() == 2
    progress = instance.get_progress_info()
    assert progress["ready_groups"] == 2
    assert progress["ready_trajectories"] == 4
    assert progress["ready_sequences"] == 4
    assert progress["incomplete_groups"] == 0
    assert instance.get_trajectories() == valid + [donor, invalid]
    assert [donor["group_skipped"], invalid["group_skipped"]] == [False, True]
    assert instance.is_empty()


def test_preexisting_skip_flag_excludes_only_that_member_when_group_is_repairable():
    instance, _ = _async_instance()
    marked = _trajectory("already-skipped", 0, 1)
    marked["group_skipped"] = True
    _put(instance, marked)
    valid = _put(instance, _trajectory("already-skipped", 1, 1))

    assert instance.ready_group_count() == 1
    assert instance.get_trajectories() == [marked, valid]
    assert [marked["group_skipped"], valid["group_skipped"]] == [True, False]
    assert instance.is_empty()
    assert instance.get_progress_info()["skipped_groups_total"] == 0
    assert instance.get_progress_info()["failed_trajectories_total"] == 0
    assert instance.get_progress_info()["skipped_trajectories_total"] == 0


@pytest.mark.parametrize("size,failed", [(4, 1), (4, 2), (4, 3), (5, 2), (5, 3), (1, 1)])
@pytest.mark.parametrize("fully_async", [False, True])
def test_failure_majority_boundary_preserves_slots_and_controls_group_admission(
    size, failed, fully_async
):
    instance, _ = _instance(group_size=size, require_trainable_groups=fully_async)
    originals = [
        _trajectory(
            "boundary",
            run,
            1,
            expected=size,
            reward=None if run < failed else 0.0,
            status="failed" if run < failed else "success",
        )
        for run in range(size)
    ]
    for trajectory in reversed(originals):
        _put(instance, trajectory)

    majority = failed * 2 > size
    if fully_async and majority:
        assert instance.get_trajectories() is None
        assert instance.get_progress_info()["skipped_groups_total"] == 1
        assert instance.is_empty()
    else:
        assert instance.get_trajectories() == originals
        assert [item["group_skipped"] for item in originals] == [
            majority or run < failed for run in range(size)
        ]
        assert [item.status for item in originals].count("failed") == failed


def test_failed_members_cannot_supply_training_targets_for_an_otherwise_empty_group():
    instance, _ = _async_instance()
    _put(instance, _trajectory("empty-survivor", 0, 0, reward=0.0))
    _put(instance, _trajectory("empty-survivor", 1, 3, reward=None, status="failed"))

    assert instance.get_trajectories() is None
    assert instance.is_empty()
    assert instance.get_progress_info()["empty_groups_total"] == 1
    assert instance.get_progress_info()["skipped_groups_total"] == 0


@pytest.mark.parametrize(
    "sequences",
    [
        [],
        [{"tokens": [], "loss_mask": []}],
        [{"tokens": [1], "loss_mask": [1]}],
        [{"tokens": [1, 2], "loss_mask": [0, 0]}],
        [{"tokens": [1, 2], "loss_mask": [1, 0]}],
    ],
    ids=["no-sequences", "empty-sequence", "one-token", "fully-masked", "first-token-only"],
)
def test_successful_group_without_predictable_training_targets_does_not_become_ready(sequences):
    instance, _ = _async_instance()
    for run in range(2):
        trajectory = _trajectory("no-targets", run, 0, reward=float(run))
        trajectory.segments = copy.deepcopy(sequences)
        _put(instance, trajectory)

    assert instance.ready_group_count() == 0
    assert instance.get_progress_info()["incomplete_groups"] == 0
    assert instance.get_progress_info()["empty_groups_total"] == 1
    assert instance.get_progress_info()["skipped_groups_total"] == 0
    assert instance.get_trajectories() is None
    assert instance.is_empty()

    valid = _complete(instance, "valid")
    assert instance.get_trajectories() == valid


def test_zero_reward_and_zero_advantage_groups_still_fill_a_batch():
    instance, _ = _async_instance(groups_per_batch=2)
    zero_rewards = _complete(instance, "zero", reward=0.0)
    equal_rewards = _complete(instance, "equal", reward=1.0)

    trajectories = instance.get_trajectories()
    assert trajectories == zero_rewards + equal_rewards
    compute_grpo_advantage_for_trajectories(trajectories)
    assert [trajectory["advantage"] for trajectory in trajectories] == [0.0] * 4
    assert instance.is_empty()


def test_empty_parent_is_retained_in_an_eligible_group_for_reward_comparison():
    instance, _ = _async_instance()
    empty = _put(instance, _trajectory("mixed", 0, 0, reward=0.0))
    nonempty = _put(instance, _trajectory("mixed", 1, 3, reward=1.0))

    trajectories = instance.get_trajectories()
    assert trajectories == [empty, nonempty]
    assert [trajectory.reward for trajectory in trajectories] == [0.0, 1.0]
    compute_grpo_advantage_for_trajectories(trajectories)
    expected = 0.5 / (0.5 + 1e-6)
    assert [trajectory["advantage"] for trajectory in trajectories] == pytest.approx(
        [-expected, expected]
    )


@pytest.mark.parametrize("group_order", ["fifo", "lifo"])
def test_valid_batches_keep_completion_order_and_retries_do_not_reorder_groups(group_order):
    instance, _ = _async_instance(groups_per_batch=2, group_order=group_order)
    groups = {group: [_trajectory(group, run, 1) for run in range(2)] for group in ("a", "b", "c")}
    _put(instance, groups["a"][1])
    _put(instance, copy.deepcopy(groups["a"][1]))
    for trajectory in reversed(groups["b"]):
        _put(instance, trajectory)
    rejected = _complete(instance, "rejected", status="skipped")
    _put(instance, groups["a"][0])
    for trajectory in groups["c"]:
        _put(instance, trajectory)
    for trajectory in groups["b"] + rejected:
        _put(instance, copy.deepcopy(trajectory))

    expected = ("c", "a") if group_order == "lifo" else ("b", "a")
    remaining = "b" if group_order == "lifo" else "c"
    assert instance.ready_group_count() == 3
    assert instance.get_trajectories() == groups[expected[0]] + groups[expected[1]]
    for trajectory in groups[expected[0]] + rejected:
        _put(instance, copy.deepcopy(trajectory))
    assert instance.ready_group_count() == 1
    assert instance.get_trajectories() is None
    later = _complete(instance, "later")
    expected_tail = (
        later + groups[remaining] if group_order == "lifo" else groups[remaining] + later
    )
    assert instance.get_trajectories() == expected_tail
    assert instance.is_empty()


def test_rejected_group_retries_remain_idempotent_and_conflicts_are_detected():
    instance, _ = _async_instance()
    rejected = _complete(instance, "rejected", status="failed")
    for trajectory in rejected:
        _put(instance, copy.deepcopy(trajectory))
    assert instance.is_empty()
    assert instance.ready_group_count() == 0

    conflict = copy.deepcopy(rejected[0])
    conflict.reward = 0.5
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        _put(instance, conflict)
    new_rollout = copy.deepcopy(rejected[0])
    new_rollout["rollout_id"] = "replacement-id-in-completed-group"
    with pytest.raises(ValueError, match="Cannot add a new rollout to a completed group"):
        _put(instance, new_rollout)


def test_continuous_rejected_arrivals_do_not_accumulate_ready_inventory_or_full_payloads(
    monkeypatch,
):
    instance, _ = _async_instance(groups_per_batch=2)
    monkeypatch.setattr(instance._store, "_MAX_RECENT_GROUPS", 4)
    for index in range(20):
        _complete(instance, f"rejected-{index}", status="skipped", count=3)
        assert instance.ready_group_count() == 0
        assert instance.get_trajectories() is None
        assert instance.is_empty()

    assert not instance._store._groups
    assert not instance._store._ready_groups
    assert len(instance._store._recent_groups) <= 4
    assert len(instance._store._rollout_signatures) <= 8
    assert instance.get_progress_info()["discarded_groups_total"] == 20
    assert instance.get_progress_info()["discarded_trajectories_total"] == 40


def test_discarded_outcomes_are_logged_and_counted_once_without_retaining_payloads(caplog):
    instance, _ = _async_instance()
    rejected = [_trajectory("runtime-failed", run, 2) for run in range(2)]
    for trajectory, status, duration in zip(
        rejected, ("failed", "skipped"), (1.25, 2.5), strict=True
    ):
        trajectory.status = status
        trajectory["task_id"] = "failed-task"
        trajectory["runtime_metrics"] = {"duration_seconds": duration}
        trajectory["e2b_status"] = "error"
        trajectory["e2b_termination_reason"] = "download_failed"
        trajectory.segments[0]["tokens"] = [9134671, 9876543]
    references = [weakref.ref(trajectory) for trajectory in rejected]

    with caplog.at_level(logging.INFO, logger="northrl.trajectory_pool"):
        for trajectory in rejected:
            _put(instance, trajectory)
        for trajectory in rejected:
            _put(instance, copy.deepcopy(trajectory))
        assert instance.get_trajectories() is None

    expected = {
        "discarded_groups_total": 1,
        "skipped_groups_total": 1,
        "empty_groups_total": 0,
        "tail_groups_total": 0,
        "discarded_trajectories_total": 2,
        "failed_trajectories_total": 1,
        "skipped_trajectories_total": 1,
        "discarded_response_tokens_total": 4,
        "discarded_rollout_seconds_total": 3.75,
    }
    progress = instance.get_progress_info()
    assert {key: progress[key] for key in expected} == expected
    assert instance.get_discarded_metrics() == {
        f"pool/{key}": value for key, value in expected.items()
    }
    discard_logs = [
        record.getMessage() for record in caplog.records if "[group_discarded]" in record.message
    ]
    assert len(discard_logs) == 1
    outcome_log = discard_logs[0]
    for value in (
        "runtime-failed",
        "runtime-failed-0",
        "runtime-failed-1",
        "failed-task",
        "download_failed",
        "failed",
        "skipped",
        "3.75",
    ):
        assert value in outcome_log
    assert "9134671" not in outcome_log
    assert "9876543" not in outcome_log

    # Even the most recent rejected group keeps only compact retry signatures.
    del trajectory, rejected
    gc.collect()
    assert all(reference() is None for reference in references)


def test_discard_diagnostics_are_absent_when_the_fully_async_policy_is_disabled():
    instance, _ = _instance()
    rejected = _complete(instance, "legacy-skipped", status="failed")
    assert instance.get_trajectories() == rejected
    assert instance.get_discarded_metrics() == {}
    assert "discarded_groups_total" not in instance.get_progress_info()


@pytest.mark.parametrize("group_order", ["fifo", "lifo"])
@pytest.mark.parametrize("tail_size", [1, 3])
def test_incomplete_batch_is_dropped_only_after_source_exhaustion_and_worker_quiescence(
    group_order, tail_size
):
    instance, state = _async_instance(groups_per_batch=4, group_order=group_order)
    tail = [
        trajectory
        for index in range(tail_size)
        for trajectory in _complete(instance, f"tail-{index}")
    ]
    _complete(instance, "rejected", status="failed")
    state.has_items = False
    state.idle = True
    assert instance.get_trajectories() is None
    assert instance.ready_group_count() == tail_size

    state.exhausted = True
    state.idle = False
    assert instance.get_trajectories() is None
    assert instance.ready_group_count() == tail_size

    state.idle = True
    assert instance.get_trajectories() is None
    assert instance.is_empty()
    assert instance.ready_group_count() == 0
    assert instance.get_trajectories() is None
    progress = instance.get_progress_info()
    assert progress["discarded_groups_total"] == 1 + tail_size
    assert progress["skipped_groups_total"] == 1
    assert progress["tail_groups_total"] == tail_size
    assert progress["discarded_trajectories_total"] == 2 * (1 + tail_size)
    assert progress["failed_trajectories_total"] == 2
    metrics = instance.get_discarded_metrics()
    for trajectory in tail:
        _put(instance, copy.deepcopy(trajectory))
    assert instance.is_empty()
    assert instance.get_discarded_metrics() == metrics


@pytest.mark.parametrize("pending_status", ["success", "failed"])
def test_missing_rollout_slots_still_raise_instead_of_discarding_the_partial_group(pending_status):
    instance, state = _async_instance(groups_per_batch=2)
    _complete(instance, "valid")
    _put(instance, _trajectory("unfinished", 0, 1, status=pending_status))
    state.has_items = False
    state.exhausted = state.idle = True
    with pytest.raises(RuntimeError, match="unfinished rollout groups"):
        instance.get_trajectories()
    assert instance.ready_group_count() == 1
    assert instance.get_progress_info()["incomplete_groups"] == 1
