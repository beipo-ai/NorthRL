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

"""Bounded progress accounting for ordered and continuous rollout execution."""

import math
import threading
from dataclasses import dataclass
from numbers import Real
from typing import Any

from northrl.northrl_types import Trajectory
from northrl.utils.trajectory_sequences import iter_trajectory_sequences


@dataclass(frozen=True, slots=True)
class _RolloutResult:
    """Retain outcome counts without retaining trajectories or their token arrays."""

    segments: int
    grouped_eligible: bool
    simple_eligible: bool


class RolloutProgress:
    """Track requested executions independently of worker state and pool consumption.

    The owner replaces this object when the next ordered input batch is fetched.
    Groups complete when all their requested rollout IDs have terminal results;
    Failed results complete their slots; groups with at least half valid outcomes
    remain eligible for resampling. Segment counts describe real valid rollouts.
    """

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._lock = threading.Lock()
        members: set[str] = set()
        groups: dict[Any, set[str]] = {}
        for item in items:
            rollout_id = item.get("rollout_id")
            if not isinstance(rollout_id, str) or not rollout_id:
                raise ValueError("Rollout progress requires a non-empty rollout_id")
            members.add(rollout_id)
            group_id = item.get("group_id")
            group_key = ("rollout", rollout_id) if group_id is None else ("group", group_id)
            groups.setdefault(group_key, set()).add(rollout_id)
        self._members = frozenset(members)
        self._groups = tuple(frozenset(group) for group in groups.values())
        self._claimed: set[str] = set()
        self._finished: dict[str, _RolloutResult] = {}

    def claim(self, rollout_id: str) -> None:
        """Mark a requested rollout as running, without counting duplicate claims."""
        with self._lock:
            if rollout_id in self._members and rollout_id not in self._finished:
                self._claimed.add(rollout_id)

    def finish(self, trajectory: Trajectory) -> None:
        """Record the first terminal result for a member of this input batch."""
        rollout_id = trajectory.get("rollout_id")
        with self._lock:
            if (
                rollout_id not in self._members
                or rollout_id in self._finished
                or trajectory.status not in {"success", "failed", "skipped"}
            ):
                return
            reward = trajectory.reward
            successful = trajectory.status == "success"
            self._finished[rollout_id] = _RolloutResult(
                segments=sum(1 for _ in iter_trajectory_sequences(trajectory)),
                grouped_eligible=(
                    successful
                    and not trajectory.get("group_skipped", False)
                    and isinstance(reward, Real)
                    and not isinstance(reward, bool)
                    and math.isfinite(reward)
                ),
                simple_eligible=successful and not trajectory.get("group_skipped", False),
            )
            self._claimed.discard(rollout_id)

    def snapshot(self, grouped: bool = True) -> dict[str, int]:
        """Return one consistent snapshot; finished counts survive pool consumption."""
        with self._lock:
            complete_groups = [group for group in self._groups if group <= self._finished.keys()]
            if grouped:
                ready_segments = sum(
                    sum(
                        self._finished[rollout_id].segments
                        for rollout_id in group
                        if self._finished[rollout_id].grouped_eligible
                    )
                    for group in complete_groups
                    if sum(self._finished[rollout_id].grouped_eligible for rollout_id in group) * 2
                    >= len(group)
                )
            else:
                ready_segments = sum(
                    result.segments for result in self._finished.values() if result.simple_eligible
                )
            total = len(self._members)
            finished = len(self._finished)
            in_progress = len(self._claimed)
            return {
                "total": total,
                "finished": finished,
                "in_progress": in_progress,
                "queued": total - finished - in_progress,
                "groups_total": len(self._groups),
                "groups_complete": len(complete_groups),
                "ready_segments": ready_segments,
            }


class ContinuousRolloutProgress:
    """Track cumulative completions and current work across asynchronous batches.

    Only active rollout IDs are retained. The loader must assign a fresh ID to
    each execution and claim it once; retries may repeat a claim while that
    execution is active, but completed IDs must never be claimed again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._finished_total = 0
        self._queued = 0

    def set_queued(self, count: int) -> None:
        """Replace the loaded buffer count without resetting executions in flight."""
        with self._lock:
            self._queued = count

    def claim(self, rollout_id: str) -> None:
        """Move one uniquely identified item from the loaded buffer into execution."""
        with self._lock:
            if rollout_id not in self._active:
                self._active.add(rollout_id)
                self._queued -= 1

    def finish(self, trajectory: Trajectory) -> None:
        """Count one terminal result for an active execution and forget its ID."""
        rollout_id = trajectory.get("rollout_id")
        with self._lock:
            if rollout_id in self._active and trajectory.status in {"success", "failed", "skipped"}:
                self._active.remove(rollout_id)
                self._finished_total += 1

    def snapshot(self) -> dict[str, int]:
        """Return cumulative completions and current active/buffered work atomically."""
        with self._lock:
            return {
                "finished_total": self._finished_total,
                "in_progress": len(self._active),
                "queued": self._queued,
            }
