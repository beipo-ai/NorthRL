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

"""Complete trajectories define the GRPO batching boundary."""

import copy
from array import array
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import (
    GroupedTrajectoryStore,
    SimpleTrajectoryStore,
    TrajectoryPoolInstance,
)
from northrl.utils.trajectory_sequences import iter_trajectory_sequences


def _instance(*, groups_per_batch=1, group_size=2, **pool_options):
    instance = TrajectoryPoolInstance(
        OmegaConf.create(
            {
                "key_list": ["group_id"],
                "group_size": group_size,
                "groups_per_batch": groups_per_batch,
                "batch_size": groups_per_batch * group_size,
                "need_sampling": False,
                **pool_options,
            }
        ),
        "actor",
    )
    state = SimpleNamespace(has_items=True, idle=False, exhausted=False)
    instance.set_module_references(
        SimpleNamespace(
            can_return_item=lambda: state.has_items, is_finished=lambda: state.exhausted
        ),
        None,
        SimpleNamespace(is_rollout_worker_quiescent=lambda: state.idle),
    )
    return instance, state


def _trajectory(group, run, count, *, reward=1.0, status="success", expected=2):
    sequences = [{"tokens": [1, 2 + index], "loss_mask": [0, 1]} for index in range(count)]
    fields = sequences[0] if count == 1 else {"segments": sequences}
    return Trajectory(
        **fields,
        reward=reward,
        status=status,
        extra_fields={
            "group_id": group,
            "run_id": run,
            "rollout_id": f"{group}-{run}",
            "expected_rollouts": expected,
            "model_tag": "actor",
        },
    )


def _put(instance, trajectory):
    assert instance.put_trajectory(trajectory) == "success"
    return trajectory


@pytest.mark.parametrize(
    "store_type,method_name,count_name,valid_count",
    [
        (GroupedTrajectoryStore, "take_groups", "num_groups", 1),
        (SimpleTrajectoryStore, "take_trajectories", "num_trajectories", 2),
    ],
)
@pytest.mark.parametrize("invalid_count", [None, 0, -1, True, 1.0, "1"])
def test_store_take_rejects_invalid_counts_without_consuming(
    store_type, method_name, count_name, valid_count, invalid_count
):
    store = store_type(OmegaConf.create({"group_size": 2}), "actor")
    trajectories = [_trajectory("a", run, count) for run, count in enumerate([3, 0])]
    for trajectory in trajectories:
        assert store.put_trajectory(trajectory)
    take = getattr(store, method_name)
    with pytest.raises(ValueError, match=count_name):
        take(**{count_name: invalid_count})
    assert take(**{count_name: valid_count}) == trajectories
    assert store.is_empty()


def test_take_groups_requires_explicit_group_count_and_never_consumes_a_partial_batch():
    store = GroupedTrajectoryStore(OmegaConf.create({"group_size": 2}), "actor")
    groups = [
        [_trajectory(group, run, count) for run, count in enumerate(counts)]
        for group, counts in [("a", [3, 0]), ("b", [1, 1])]
    ]
    for group in groups:
        for trajectory in reversed(group):
            assert store.put_trajectory(trajectory)

    with pytest.raises(TypeError, match="num_groups"):
        store.take_groups()
    with pytest.raises(TypeError):
        store.take_groups(1)
    assert store.take_groups(num_groups=3) is None
    assert store.ready_group_count() == 2
    assert store.take_groups(num_groups=1) == groups[0]
    assert store.take_groups(num_groups=2) is None
    assert store.take_groups(num_groups=1) == groups[1]
    assert store.take_groups(num_groups=1) is None
    assert store.is_empty()


def test_take_trajectories_counts_parents_and_never_consumes_a_partial_batch():
    store = SimpleTrajectoryStore(OmegaConf.create({}), "actor")
    trajectories = [_trajectory("a", run, count) for run, count in enumerate([3, 0, 1])]
    for trajectory in trajectories:
        assert store.put_trajectory(trajectory)

    with pytest.raises(TypeError, match="num_trajectories"):
        store.take_trajectories()
    with pytest.raises(TypeError):
        store.take_trajectories(1)
    assert store.take_trajectories(num_trajectories=4) is None
    assert store.ready_trajectory_count() == 3
    assert store.take_trajectories(num_trajectories=2) == trajectories[:2]
    assert store.take_trajectories(num_trajectories=2) is None
    assert store.take_trajectories(num_trajectories=1) == trajectories[2:]
    assert store.take_trajectories(num_trajectories=1) is None
    assert store.is_empty()


def test_many_segments_do_not_complete_a_group_without_all_trajectories():
    instance, _ = _instance()
    first = _put(instance, _trajectory("a", 0, 3))
    assert instance.get_trajectories() is None
    assert instance.get_progress_info()["ready_segments"] == 0
    assert instance.get_progress_info()["ready_trajectories"] == 0
    second = _put(instance, _trajectory("a", 1, 1, reward=0.0))
    progress = instance.get_progress_info()
    assert (
        progress["ready_sequences"]
        == progress["ready_segments"]
        == progress["finished_trajectories"]
        == 4
    )
    assert second.segments is None
    assert second.tokens == [1, 2]
    assert progress["ready_trajectories"] == 2
    trajectories = instance.get_trajectories()
    assert trajectories == [first, second]
    assert [
        sum(1 for _ in iter_trajectory_sequences(trajectory)) for trajectory in trajectories
    ] == [3, 1]
    assert all(not trajectory["group_skipped"] for trajectory in trajectories)
    assert instance.get_trajectories() is None


@pytest.mark.parametrize("group_order", [None, "fifo", "lifo"], ids=["default", "fifo", "lifo"])
def test_grouped_batches_count_complete_groups_and_leave_backlog_queued(group_order):
    options = {} if group_order is None else {"group_order": group_order}
    instance, state = _instance(groups_per_batch=2, **options)
    groups = {
        group: [_trajectory(group, run, count) for run, count in enumerate(counts)]
        for group, counts in (("a", [1, 4]), ("b", [2, 3]), ("c", [1, 1]))
    }
    # A starts first, but B completes first. Runs also arrive out of slot order.
    for group, run in (("a", 1), ("b", 1), ("c", 0), ("b", 0), ("a", 0), ("c", 1)):
        _put(instance, groups[group][run])
    # Retries must not make the oldest ready group look newly completed.
    for trajectory in groups["b"]:
        _put(instance, copy.deepcopy(trajectory))
    assert instance.get_progress_info()["ready_groups"] == 3
    selected = ("c", "a") if group_order == "lifo" else ("b", "a")
    remaining = "b" if group_order == "lifo" else "c"
    assert instance.get_trajectories() == groups[selected[0]] + groups[selected[1]]
    assert instance.get_progress_info()["ready_segments"] == sum(
        sum(1 for _ in iter_trajectory_sequences(trajectory)) for trajectory in groups[remaining]
    )
    assert instance.get_progress_info()["ready_trajectories"] == 2
    assert instance.get_trajectories() is None

    # The smaller finite-source tail still drains with exactly the remaining group.
    state.has_items = False
    state.exhausted = state.idle = True
    assert instance.get_trajectories() == groups[remaining]
    assert instance.is_empty()
    assert instance.get_trajectories() is None


@pytest.mark.parametrize(
    "reward,status,count",
    [
        (None, "success", 0),
        (float("nan"), "success", 0),
        (float("inf"), "success", 0),
        (1.0, "failed", 1),
        (1.0, "skipped", 0),
    ],
)
def test_half_invalid_group_preserves_original_results_for_trainer_resampling(
    reward, status, count
):
    instance, _ = _instance()
    valid = _put(instance, _trajectory("a", 0, 2))
    invalid = _put(instance, _trajectory("a", 1, count, reward=reward, status=status))
    assert not instance.is_empty()
    assert instance.get_progress_info()["ready_groups"] == 1
    assert instance.get_progress_info()["ready_segments"] == 2
    assert instance.get_progress_info()["ready_trajectories"] == 2
    trajectories = instance.get_trajectories()
    assert trajectories == [valid, invalid]
    assert [trajectory["group_skipped"] for trajectory in trajectories] == [False, True]
    assert len(valid.segments) == 2  # Diagnostics retain the original payload.
    assert instance.is_empty()
    assert instance.get_trajectories() is None


def test_successful_zero_segment_trajectory_preserves_reward_for_baseline():
    instance, _ = _instance()
    empty = _put(instance, _trajectory("a", 0, 0, reward=0.25))
    nonempty = _put(instance, _trajectory("a", 1, 2, reward=1.0))
    trajectories = instance.get_trajectories()
    assert trajectories == [empty, nonempty]
    assert [trajectory.reward for trajectory in trajectories] == [0.25, 1.0]
    assert all(not trajectory["group_skipped"] for trajectory in trajectories)


def test_all_successful_zero_segment_group_is_delivered_once():
    instance, _ = _instance()
    expected = [_put(instance, _trajectory("a", run, 0, reward=float(run))) for run in range(2)]
    assert instance.get_progress_info()["ready_segments"] == 0
    assert instance.get_progress_info()["ready_trajectories"] == 2
    assert instance.get_trajectories() == expected
    assert instance.get_trajectories() is None


def test_duplicate_trajectory_is_idempotent_before_and_after_consumption():
    instance, _ = _instance()
    trajectories = [_trajectory("a", run, 1) for run in range(2)]
    for trajectory in trajectories:
        _put(instance, trajectory)
        _put(instance, copy.deepcopy(trajectory))
    assert instance.get_progress_info()["ready_groups"] == 1
    assert instance.get_trajectories() == trajectories
    for trajectory in trajectories:
        _put(instance, copy.deepcopy(trajectory))
    assert instance.get_trajectories() is None
    different_reward = copy.deepcopy(trajectories[0])
    different_reward.reward = 0.5
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        _put(instance, different_reward)
    different_segment = copy.deepcopy(trajectories[0])
    different_segment["tokens"][1] += 1
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        _put(instance, different_segment)


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("field", ["tokens", "loss_mask", "old_log_probs"])
def test_packed_signature_detects_late_token_changes_before_and_after_consumption(segmented, field):
    instance, _ = _instance()
    trajectories = [_trajectory("packed", run, 2 if segmented else 1) for run in range(2)]
    for trajectory in trajectories:
        for sequence in iter_trajectory_sequences(trajectory):
            sequence["tokens"] = array("q", range(10000))
            sequence["loss_mask"] = array("b", [0, 1]) * 5000
            sequence["old_log_probs"] = array("d", [0.0, -0.25]) * 5000
        _put(instance, trajectory)
    for consumed in [False, True]:
        if consumed:
            assert instance.get_trajectories() == trajectories
        _put(instance, copy.deepcopy(trajectories[0]))
        changed = copy.deepcopy(trajectories[0])
        sequence = list(iter_trajectory_sequences(changed))[-1]
        sequence[field][-2] += 1
        with pytest.raises(ValueError, match="Conflicting duplicate"):
            _put(instance, changed)


def test_signature_hashes_packed_buffers_without_repr(monkeypatch):
    import northrl.trajectory_pool as pool_module

    calls = []

    def checked_repr(value):
        assert not isinstance(value, array)
        calls.append(value)
        return repr(value)

    monkeypatch.setattr(pool_module, "repr", checked_repr, raising=False)
    instance, _ = _instance()
    trajectory = _trajectory("packed", 0, 1)
    trajectory["tokens"] = array("q", [1, 2])
    trajectory["loss_mask"] = array("b", [0, 1])
    _put(instance, trajectory)
    _put(instance, copy.deepcopy(trajectory))
    assert "tokens" in calls


def test_signature_frames_array_types_values_and_metadata():
    store = GroupedTrajectoryStore(OmegaConf.create({"group_size": 2}), "actor")
    trajectory = _trajectory("framed", 0, 1)
    trajectory["tokens"] = array("q", [1, 2])
    original = store._signature(trajectory, store._identity(trajectory))
    for field, value in [("tokens", array("i", [1, 2])), ("tokens", [1, 2]), ("note", "new")]:
        changed = copy.deepcopy(trajectory)
        changed[field] = value
        assert store._signature(changed, store._identity(changed)) != original
    reordered = copy.deepcopy(trajectory)
    reordered.extra_fields = dict(reversed(list(reordered.extra_fields.items())))
    assert store._signature(reordered, store._identity(reordered)) == original


@pytest.mark.parametrize("consumed", [False, True])
def test_completed_group_rejects_new_rollout_ids(consumed):
    instance, _ = _instance()
    trajectories = [_put(instance, _trajectory("a", run, 1)) for run in range(2)]
    if consumed:
        assert instance.get_trajectories() == trajectories
    new_rollout = copy.deepcopy(trajectories[0])
    new_rollout["rollout_id"] = "new-rollout"
    with pytest.raises(ValueError, match="Cannot add a new rollout to a completed group"):
        _put(instance, new_rollout)
    assert instance.get_trajectories() == (None if consumed else trajectories)
    assert instance.get_trajectories() is None


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("rollout_id", "", "rollout_id"),
        ("expected_rollouts", 3, "expected_rollouts"),
        ("run_id", 2, "run_id"),
        ("run_id", True, "run_id"),
        ("group_id", "", "group identity"),
        ("model_tag", "other", "model_tag"),
    ],
)
def test_identity_validation_prevents_untrackable_trajectories(field, value, message):
    instance, _ = _instance()
    trajectory = _trajectory("a", 0, 1)
    trajectory[field] = value
    with pytest.raises(ValueError, match=message):
        _put(instance, trajectory)
    assert instance.is_empty()


def test_run_slot_and_rollout_id_cannot_alias_different_executions():
    instance, _ = _instance()
    first = _put(instance, _trajectory("a", 0, 1))
    different_rollout = copy.deepcopy(first)
    different_rollout["rollout_id"] = "different"
    with pytest.raises(ValueError, match="multiple independent rollouts"):
        _put(instance, different_rollout)
    different_group = copy.deepcopy(first)
    different_group["group_id"] = "b"
    with pytest.raises(ValueError, match="different group or run slot"):
        _put(instance, different_group)


def test_same_task_in_two_sampling_groups_remains_separate():
    instance, _ = _instance(groups_per_batch=2)
    for group in ("occurrence-1", "occurrence-2"):
        for run in range(2):
            trajectory = _trajectory(group, run, 1)
            trajectory["task_id"] = "same-task"
            _put(instance, trajectory)
    trajectories = instance.get_trajectories()
    assert len(trajectories) == len({trajectory["rollout_id"] for trajectory in trajectories}) == 4


@pytest.mark.parametrize("group_order", [None, "fifo", "lifo"], ids=["default", "fifo", "lifo"])
@pytest.mark.parametrize("num_groups", [1, 3])
def test_finite_tail_requires_exhaustion_and_worker_quiescence(group_order, num_groups):
    options = {} if group_order is None else {"group_order": group_order}
    instance, state = _instance(groups_per_batch=4, **options)
    groups = [
        [_trajectory(f"group-{index}", run, count) for run, count in enumerate([3, 1])]
        for index in range(num_groups)
    ]
    # Complete groups out of creation order, with runs out of slot order.
    completion_order = groups[1:2] + groups[:1] + groups[2:]
    for group in completion_order:
        for trajectory in reversed(group):
            _put(instance, trajectory)
    state.has_items = False
    state.idle = True
    assert instance.get_trajectories() is None
    state.exhausted = True
    state.idle = False
    assert instance.get_trajectories() is None
    state.idle = True
    order = completion_order[::-1] if group_order == "lifo" else completion_order
    assert instance.get_trajectories() == [trajectory for group in order for trajectory in group]
    assert instance.is_empty()
    assert instance.get_trajectories() is None


def test_empty_pool_does_not_probe_completion_during_next_batch_release(monkeypatch):
    instance, state = _instance(groups_per_batch=2)
    state.has_items = False
    state.idle = True
    completion_probes = []

    def buffer_snapshot_with_concurrent_release():
        completion_probes.append("loader")
        # A snapshot of the previous empty buffer can precede a release and the
        # first result of the next batch. Empty-pool polling must avoid this probe.
        state.has_items = True
        state.idle = False
        _put(instance, _trajectory("next-batch", 0, 1))
        return False

    def source_finished():
        completion_probes.append("source")
        return False

    def workers_quiescent():
        completion_probes.append("workers")
        return state.idle

    monkeypatch.setattr(
        instance._dataloader, "can_return_item", buffer_snapshot_with_concurrent_release
    )
    monkeypatch.setattr(instance._dataloader, "is_finished", source_finished)
    monkeypatch.setattr(
        instance._activity_tracker, "is_rollout_worker_quiescent", workers_quiescent
    )
    assert instance.get_trajectories() is None
    assert completion_probes == []
    assert instance.is_empty()


def test_finite_tail_rechecks_worker_activity_after_observing_source_exhaustion(monkeypatch):
    instance, state = _instance(groups_per_batch=2)
    complete = [_put(instance, _trajectory("complete", run, 1)) for run in range(2)]
    pending = _put(instance, _trajectory("pending", 0, 1))
    state.idle = True

    def finish_source_while_claiming_last_task():
        # The source becomes exhausted when a worker claims its last buffered task;
        # an idle snapshot taken before this point no longer describes that worker.
        state.has_items = False
        state.exhausted = True
        state.idle = False
        return True

    monkeypatch.setattr(instance._dataloader, "is_finished", finish_source_while_claiming_last_task)
    assert instance.get_trajectories() is None
    assert instance.get_progress_info()["ready_groups"] == 1
    assert not state.idle

    last = _put(instance, _trajectory("pending", 1, 1))
    assert instance.get_trajectories() == complete + [pending, last]
    assert instance.is_empty()


def test_finite_tail_caps_delivery_when_groups_complete_after_first_take(monkeypatch):
    instance, state = _instance(groups_per_batch=2)
    groups = {group: [_trajectory(group, run, 1) for run in range(2)] for group in ("a", "b", "c")}
    for trajectory in groups["a"]:
        _put(instance, trajectory)
    state.has_items = False
    state.exhausted = True
    first_take = True
    take_groups = instance._store.take_groups

    def take_with_late_completions(*, num_groups):
        nonlocal first_take
        result = take_groups(num_groups=num_groups)
        if first_take:
            first_take = False
            assert result is None
            # Both other groups finish after the failed B-group take, before
            # the pool observes the exhausted source and quiescent workers.
            for group in ("b", "c"):
                for trajectory in groups[group]:
                    _put(instance, trajectory)
            state.idle = True
        return result

    monkeypatch.setattr(instance._store, "take_groups", take_with_late_completions)
    assert instance.get_trajectories() == groups["a"] + groups["b"]
    assert instance.get_progress_info()["ready_groups"] == 1
    assert instance.get_trajectories() == groups["c"]
    assert instance.is_empty()


@pytest.mark.parametrize("has_items,idle", [(True, False), (False, False), (True, True)])
def test_full_group_batch_does_not_wait_for_loader_or_worker_barriers(has_items, idle):
    instance, state = _instance(groups_per_batch=2)
    state.has_items = has_items
    state.idle = idle
    trajectories = []
    for group in ("a", "b"):
        for run, count in enumerate([3, 1]):
            trajectories.append(_put(instance, _trajectory(group, run, count)))
            if len(trajectories) < 4:
                assert instance.get_trajectories() is None

    assert instance.get_trajectories() == trajectories
    assert instance.is_empty()
    assert instance.get_trajectories() is None


@pytest.mark.parametrize("exhausted", [False, True])
def test_missing_trajectory_is_reported_only_after_all_workers_and_buffer_are_drained(exhausted):
    instance, state = _instance(groups_per_batch=2)
    for run in range(2):
        _put(instance, _trajectory("complete", run, 1))
    _put(instance, _trajectory("pending", 0, 1))
    state.idle = True
    assert instance.get_trajectories() is None  # Unclaimed tasks may complete the group.
    state.has_items = False
    state.idle = False
    assert instance.get_trajectories() is None  # A worker still owns a claimed task.
    state.exhausted = exhausted
    state.idle = True
    with pytest.raises(RuntimeError, match="unfinished rollout groups"):
        instance.get_trajectories()
    assert instance.get_progress_info()["ready_groups"] == 1


def test_simple_store_batches_complete_trajectories_without_flattening():
    instance = TrajectoryPoolInstance(
        OmegaConf.create(
            {
                "batch_size": 1,
                "need_sampling": False,
            }
        ),
        "actor",
    )
    instance.set_module_references(
        SimpleNamespace(can_return_item=lambda: False),
        None,
        SimpleNamespace(is_rollout_worker_quiescent=lambda: True),
    )
    trajectory = _put(instance, _trajectory("a", 0, 3))
    assert instance.get_progress_info()["ready_trajectories"] == 1
    assert instance.get_progress_info()["ready_segments"] == 3
    assert instance.get_trajectories() == [trajectory]


@pytest.mark.parametrize("group_order", ["fifo", "lifo"])
def test_consumed_group_retry_metadata_is_bounded(group_order):
    instance, _ = _instance(group_order=group_order)
    store = instance._store
    store._MAX_RECENT_GROUPS = 2
    pending = _put(instance, _trajectory("pending", 0, 1))
    ready = [_trajectory("ready", run, 1) for run in range(2)]
    # Both modes consume a, b, c, d while leaving "ready" in the active store.
    # Completion order is deliberately different from consumption order in LIFO.
    if group_order == "lifo":
        for trajectory in ready:
            _put(instance, trajectory)
    completion_order = ("d", "c", "b", "a") if group_order == "lifo" else ("a", "b", "c", "d")
    for group in completion_order:
        for run in range(2):
            _put(instance, _trajectory(group, run, 0))
    if group_order == "fifo":
        for trajectory in ready:
            _put(instance, trajectory)
    for group in ("a", "b", "c", "d"):
        assert {trajectory["group_id"] for trajectory in instance.get_trajectories()} == {group}

    # Retire old history without losing duplicate detection for active groups.
    assert list(store._recent_groups) == ["c", "d"]
    assert set(store._rollout_signatures) == {
        "pending-0",
        "ready-0",
        "ready-1",
        "c-0",
        "c-1",
        "d-0",
        "d-1",
    }
    for trajectory in [pending, *ready]:
        _put(instance, copy.deepcopy(trajectory))
    # A retry of an older consumed group must not refresh its eviction order.
    for run in range(2):
        _put(instance, _trajectory("c", run, 0))
    assert instance.get_progress_info()["ready_groups"] == 1
    assert instance.get_trajectories() == ready
    assert list(store._recent_groups) == ["d", "ready"]
    assert instance.get_trajectories() is None

    pending_second = _put(instance, _trajectory("pending", 1, 1))
    assert instance.get_trajectories() == [pending, pending_second]
    assert instance.is_empty()
    assert list(store._recent_groups) == ["ready", "pending"]
    assert set(store._rollout_signatures) == {"ready-0", "ready-1", "pending-0", "pending-1"}


def test_nonterminal_trajectory_is_rejected_before_creating_pending_state():
    instance, _ = _instance()
    trajectory = _trajectory("a", 0, 1)
    trajectory.status = "running"
    with pytest.raises(ValueError, match="terminal"):
        _put(instance, trajectory)
    assert instance.is_empty()


def test_skipped_groups_count_toward_async_batch_without_polluting_valid_groups():
    instance, _ = _instance(groups_per_batch=2)
    for run in range(2):
        _put(instance, _trajectory("skipped", run, 0, status="skipped"))
    assert instance.get_trajectories() is None
    for run in range(2):
        _put(instance, _trajectory("valid", run, 2))
    progress = instance.get_progress_info()
    assert progress["ready_trajectories"] == progress["ready_segments"] == 4
    trajectories = instance.get_trajectories()
    assert [trajectory["group_skipped"] for trajectory in trajectories] == [
        True,
        True,
        False,
        False,
    ]
    assert instance.is_empty()


@pytest.mark.parametrize("group_order", ["random", "FIFO", "", None, 1, ["fifo"]])
def test_invalid_internal_group_order_is_rejected(group_order):
    with pytest.raises(ValueError, match="group_order"):
        _instance(group_order=group_order)
