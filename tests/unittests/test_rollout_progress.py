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

"""Progress counts executions and completed groups independently of pool inventory."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from northrl.northrl_types import Trajectory
from northrl.utils.rollout_progress import ContinuousRolloutProgress, RolloutProgress


def _items(groups: int = 2, repeats: int = 2) -> list[dict]:
    return [
        {"rollout_id": f"g{group}-r{run}", "group_id": f"g{group}"}
        for group in range(groups)
        for run in range(repeats)
    ]


def _result(rollout_id: str, segments: int = 1, **kwargs) -> Trajectory:
    return Trajectory(
        segments=[{"tokens": [1, 2], "loss_mask": [0, 1]} for _ in range(segments)],
        reward=kwargs.pop("reward", 1.0),
        extra_fields={"rollout_id": rollout_id},
        **kwargs,
    )


def test_partial_and_complete_groups_count_executions_and_segments_separately() -> None:
    progress = RolloutProgress(_items())
    for item in _items():
        progress.claim(item["rollout_id"])
    progress.finish(_result("g0-r0", segments=3))
    assert progress.snapshot() == {
        "total": 4,
        "finished": 1,
        "in_progress": 3,
        "queued": 0,
        "groups_total": 2,
        "groups_complete": 0,
        "ready_segments": 0,
    }

    progress.finish(Trajectory([1], [1], 0.5, extra_fields={"rollout_id": "g0-r1"}))
    assert progress.snapshot()["groups_complete"] == 1
    assert progress.snapshot()["ready_segments"] == 4

    progress.finish(_result("g1-r0", segments=0))
    progress.finish(_result("g1-r1", segments=2))
    assert progress.snapshot() == {
        "total": 4,
        "finished": 4,
        "in_progress": 0,
        "queued": 0,
        "groups_total": 2,
        "groups_complete": 2,
        "ready_segments": 6,
    }


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_terminal_failures_complete_groups_without_exposing_their_segments(status: str) -> None:
    progress = RolloutProgress(_items(groups=1))
    progress.finish(_result("g0-r0", segments=3))
    progress.finish(_result("g0-r1", segments=0, status=status, reward=None))
    counts = progress.snapshot()
    assert counts["finished"] == 2
    assert counts["groups_complete"] == 1
    assert counts["ready_segments"] == 3
    assert counts["queued"] == counts["in_progress"] == 0


@pytest.mark.parametrize("reward", [None, True, float("nan"), float("inf"), "1.0"])
def test_invalid_reward_excludes_that_member_from_repairable_group_progress(reward) -> None:
    progress = RolloutProgress(_items(groups=1))
    progress.finish(_result("g0-r0", segments=3))
    progress.finish(_result("g0-r1", reward=reward))
    assert progress.snapshot()["groups_complete"] == 1
    assert progress.snapshot()["ready_segments"] == 3


@pytest.mark.parametrize("size,failures,ready", [(4, 2, 2), (4, 3, 0), (5, 2, 3), (5, 3, 0)])
def test_group_progress_obeys_majority_boundary_without_counting_replacement_copies(
    size, failures, ready
):
    progress = RolloutProgress(_items(groups=1, repeats=size))
    for run in range(size):
        progress.finish(_result(f"g0-r{run}", status="failed" if run < failures else "success"))
    assert progress.snapshot()["finished"] == size
    assert progress.snapshot()["groups_complete"] == 1
    assert progress.snapshot()["ready_segments"] == ready


def test_duplicate_old_and_nonterminal_results_do_not_change_counts() -> None:
    progress = RolloutProgress(_items(groups=1))
    progress.claim("g0-r0")
    progress.claim("g0-r0")
    progress.claim("old-id")
    progress.finish(_result("old-id", segments=100))
    progress.finish(_result("g0-r0", status="running"))
    assert progress.snapshot()["in_progress"] == 1
    assert progress.snapshot()["finished"] == 0
    assert progress.snapshot()["queued"] == 1

    progress.finish(_result("g0-r0", segments=3))
    progress.finish(_result("g0-r0", segments=100))
    progress.claim("g0-r0")
    progress.finish(_result("g0-r1"))
    assert progress.snapshot()["finished"] == 2
    assert progress.snapshot()["in_progress"] == 0
    assert progress.snapshot()["ready_segments"] == 4


def test_finish_before_claim_is_neither_queued_nor_in_progress() -> None:
    progress = RolloutProgress(_items(groups=1))
    progress.finish(_result("g0-r0"))
    progress.claim("g0-r0")
    assert progress.snapshot()["finished"] == 1
    assert progress.snapshot()["in_progress"] == 0
    assert progress.snapshot()["queued"] == 1
    progress.claim("g0-r1")
    assert progress.snapshot()["queued"] == 0
    assert progress.snapshot()["in_progress"] == 1


def test_simple_store_exposes_successful_segments_before_group_completion_without_rewards() -> None:
    progress = RolloutProgress(_items(groups=1, repeats=4))
    progress.finish(_result("g0-r0", segments=3, reward=None))
    progress.finish(_result("g0-r1", segments=7, status="failed"))
    skipped = _result("g0-r2", segments=9)
    skipped["group_skipped"] = True
    progress.finish(skipped)
    assert progress.snapshot()["ready_segments"] == 0
    counts = progress.snapshot(grouped=False)
    assert counts["finished"] == 3
    assert counts["groups_complete"] == 0
    assert counts["ready_segments"] == 3


def test_snapshots_do_not_retain_or_follow_mutable_batch_and_trajectory_objects() -> None:
    items = _items(groups=1)
    progress = RolloutProgress(items)
    items[0]["rollout_id"] = "changed"
    items.clear()
    trajectory = _result("g0-r0", segments=3)
    reference = weakref.ref(trajectory)
    progress.finish(trajectory)
    trajectory.segments.clear()
    trajectory.reward = None
    trajectory.status = "failed"
    del trajectory
    gc.collect()
    assert reference() is None
    progress.finish(_result("g0-r1"))
    assert progress.snapshot()["ready_segments"] == 4
    assert progress.snapshot()["total"] == 2


def test_concurrent_claims_and_duplicate_results_keep_a_consistent_partition() -> None:
    items = _items(groups=8, repeats=4)
    progress = RolloutProgress(items)

    def complete(item: dict) -> None:
        rollout_id = item["rollout_id"]
        progress.claim(rollout_id)
        progress.finish(_result(rollout_id))
        snapshot = progress.snapshot()
        assert snapshot["finished"] + snapshot["in_progress"] + snapshot["queued"] == len(items)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(complete, items * 3))
    counts = progress.snapshot()
    assert counts["finished"] == counts["ready_segments"] == 32
    assert counts["groups_complete"] == 8
    assert counts["in_progress"] == counts["queued"] == 0


def test_empty_and_ungrouped_inputs() -> None:
    assert RolloutProgress([]).snapshot() == {
        "total": 0,
        "finished": 0,
        "in_progress": 0,
        "queued": 0,
        "groups_total": 0,
        "groups_complete": 0,
        "ready_segments": 0,
    }
    progress = RolloutProgress([{"rollout_id": "r0"}, {"rollout_id": "r1"}])
    progress.finish(_result("r0", reward=None))
    assert progress.snapshot(grouped=False)["ready_segments"] == 1
    assert progress.snapshot()["groups_total"] == 2


def test_continuous_progress_preserves_inflight_and_total_across_loaded_batches() -> None:
    progress = ContinuousRolloutProgress()
    assert progress.snapshot() == {"finished_total": 0, "in_progress": 0, "queued": 0}
    progress.set_queued(3)
    for rollout_id in ("a", "b", "c"):
        progress.claim(rollout_id)
    progress.finish(_result("a"))
    progress.set_queued(2)
    assert progress.snapshot() == {"finished_total": 1, "in_progress": 2, "queued": 2}

    progress.claim("d")
    progress.finish(_result("b"))  # A previous batch can finish after the next batch loads.
    progress.finish(_result("d"))
    assert progress.snapshot() == {"finished_total": 3, "in_progress": 1, "queued": 1}
    progress.claim("e")
    progress.set_queued(0)  # Exhausting the source leaves in-flight executions intact.
    progress.finish(_result("c"))
    progress.finish(_result("e"))
    assert progress.snapshot() == {"finished_total": 5, "in_progress": 0, "queued": 0}


def test_continuous_progress_deduplicates_active_claims_and_late_results() -> None:
    progress = ContinuousRolloutProgress()
    progress.set_queued(2)
    progress.claim("a")
    progress.claim("a")
    progress.finish(_result("unknown"))
    progress.finish(_result("a", status="running"))
    assert progress.snapshot() == {"finished_total": 0, "in_progress": 1, "queued": 1}
    progress.finish(_result("a"))
    progress.finish(_result("a", status="failed"))
    progress.claim("b")
    progress.set_queued(3)
    progress.finish(_result("a"))
    assert progress.snapshot() == {"finished_total": 1, "in_progress": 1, "queued": 3}


@pytest.mark.parametrize("status", ["success", "failed", "skipped"])
def test_continuous_progress_counts_terminal_results_without_segments_or_reward(
    status: str,
) -> None:
    progress = ContinuousRolloutProgress()
    progress.set_queued(1)
    progress.claim("a")
    progress.finish(_result("a", segments=0, reward=None, status=status))
    assert progress.snapshot() == {"finished_total": 1, "in_progress": 0, "queued": 0}


def test_continuous_progress_concurrent_snapshots_preserve_issued_item_count() -> None:
    progress = ContinuousRolloutProgress()
    progress.set_queued(32)

    def complete(index: int) -> None:
        rollout_id = str(index)
        progress.claim(rollout_id)
        progress.claim(rollout_id)
        assert sum(progress.snapshot().values()) == 32
        progress.finish(_result(rollout_id))
        progress.finish(_result(rollout_id))
        assert sum(progress.snapshot().values()) == 32

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(complete, range(32)))
    assert progress.snapshot() == {"finished_total": 32, "in_progress": 0, "queued": 0}


def test_continuous_progress_releases_completed_id_history_and_trajectory_objects() -> None:
    progress = ContinuousRolloutProgress()
    references = []
    for batch in range(8):
        progress.set_queued(4)
        for index in range(4):
            rollout_id = f"{batch}-{index}"
            progress.claim(rollout_id)
            result = _result(rollout_id)
            references.append(weakref.ref(result))
            progress.finish(result)
    del result
    gc.collect()
    assert all(reference() is None for reference in references)
    # Cumulative totals must not require a collection of finished execution IDs.
    retained_collections = [
        value for value in vars(progress).values() if isinstance(value, (dict, list, set))
    ]
    assert all(not collection for collection in retained_collections)
    assert progress.snapshot() == {"finished_total": 32, "in_progress": 0, "queued": 0}
