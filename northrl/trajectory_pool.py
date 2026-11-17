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

import hashlib
import logging
import math
import threading
from abc import ABC, abstractmethod
from array import array
from collections import OrderedDict
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from .base_module import NorthRLModule
from .northrl_types import ModelTag, Trajectory
from .utils.trajectory_sequences import iter_trajectory_sequences

if TYPE_CHECKING:
    from northrl.activity_tracker import ActivityTrackerProxy

    from .data_loader import BaseDataLoader
    from .weight_sync.weight_sync_controller import WeightSyncController


logger = logging.getLogger(__name__)


def _hash_sequence_field(digest, value: Any) -> None:
    """Hash packed training buffers without formatting every token as text."""
    if type(value) is array:  # Preserve custom representations on subclasses.
        header = f"array:{value.typecode}:{value.itemsize}:{len(value)}:".encode()
        payload = memoryview(value).cast("B")
    else:
        header = b"repr:"
        payload = repr(value).encode()
    # Length framing keeps field/sequence boundaries unambiguous. hashlib can
    # release the GIL while hashing a large buffer, without a tobytes() copy.
    digest.update(header)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


class TrajectoryStoreBase(ABC):
    """Store complete rollout results, including zero-segment terminal outcomes."""

    model_tag: ModelTag

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        self._config = config
        self._lock = threading.Lock()
        self._ready_trajectories: list[Trajectory] = []
        self.model_tag = model_tag

    @abstractmethod
    def put_trajectory(self, trajectory: Trajectory) -> bool:
        """Add one complete trajectory to the store."""

    @staticmethod
    def _ready_sequence_count(trajectories: list[Trajectory]) -> int:
        return sum(
            sum(1 for _ in iter_trajectory_sequences(trajectory))
            for trajectory in trajectories
            if trajectory.status == "success" and not trajectory.get("group_skipped", False)
        )

    def is_empty(self) -> bool:
        with self._lock:
            return not self._ready_trajectories and self._is_active_storage_empty()

    def progress_counts(self) -> tuple[int, int, int]:
        """Return a consistent snapshot of ready sequences, trajectories, and groups."""
        ready_sequences, ready_trajectories, ready_groups, _ = self.progress_snapshot()
        return ready_sequences, ready_trajectories, ready_groups

    def progress_snapshot(self) -> tuple[int, int, int, int]:
        """Return ready sequences, trajectories, groups, and incomplete groups atomically."""
        with self._lock:
            return (
                self._ready_sequence_count(self._ready_trajectories),
                len(self._ready_trajectories),
                0,
                0,
            )

    def ready_trajectory_count(self) -> int:
        return self.progress_counts()[1]

    @abstractmethod
    def _is_active_storage_empty(self) -> bool:
        """Check storage outside the ready trajectory queue, with the store lock held."""


class SimpleTrajectoryStore(TrajectoryStoreBase):
    """Store complete trajectories without GRPO grouping."""

    def put_trajectory(self, trajectory: Trajectory) -> bool:
        if not isinstance(trajectory, Trajectory):
            raise TypeError("The trajectory pool accepts complete Trajectory objects")
        with self._lock:
            self._ready_trajectories.append(trajectory)
        return True

    def take_trajectories(self, *, num_trajectories: int) -> list[Trajectory] | None:
        """Consume exactly num_trajectories complete results, or leave them queued.

        The caller supplies an explicit count after checking batch readiness.
        For a completed SFT token-budget batch, it passes the actual number of
        ready trajectories, which can differ from data.train.batch_size.
        """
        if (
            isinstance(num_trajectories, bool)
            or not isinstance(num_trajectories, int)
            or num_trajectories < 1
        ):
            raise ValueError("num_trajectories must be a positive integer")
        with self._lock:
            if len(self._ready_trajectories) < num_trajectories:
                return None
            trajectories = self._ready_trajectories[:num_trajectories]
            del self._ready_trajectories[:num_trajectories]
            return trajectories

    def _is_active_storage_empty(self) -> bool:
        return True


class GroupedTrajectoryStore(TrajectoryStoreBase):
    """Publish a GRPO group only after all R complete trajectories have arrived.

    Ready groups are consumed in completion order (``group_order="fifo"``) or
    reverse completion order (``"lifo"``). Completion means the final expected
    rollout arrived, regardless of when sampling started or which weights it used.
    Each group retains ascending run_id order and complete trajectory containers.
    Groups with at most half invalid outcomes can be resampled by the trainer.
    Fully async excludes groups with a majority of invalid outcomes or no valid
    prediction targets before they enter the ready queue or its backpressure count.

    Duplicate submissions are idempotent while pending/ready and for the last
    1024 consumed groups. Only compact signatures are retained after consumption;
    history is always evicted in consumption/discard order, independently of group_order.
    """

    _MAX_RECENT_GROUPS = 1024

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        super().__init__(config, model_tag)
        self._group_key = config.get("group_key", "group_id")
        self._group_size = int(config.get("group_size", 1))
        self._require_trainable_groups = bool(config.get("require_trainable_groups", False))
        if self._group_size < 1:
            raise ValueError("group_size must be a positive rollout count")
        self._group_order = config.get("group_order", "fifo")
        if self._group_order not in ("fifo", "lifo"):
            raise ValueError("group_order must be 'fifo' or 'lifo'")
        # 尚未凑齐：group_id -> run_id -> 完整 trajectory。
        self._groups: dict[Any, dict[int, Trajectory]] = {}
        # 已凑齐：按完成顺序入队，交付时按 group_order 从队首或队尾取整组。
        self._ready_groups: OrderedDict[Any, list[Trajectory]] = OrderedDict()
        # 统一保存等待中、就绪及最近已取走的结果签名；签名包含 group_id/run_id。
        self._rollout_signatures: dict[str, tuple[Any, ...]] = {}
        # 按训练取走或丢弃顺序记录 rollout ID，用于整组淘汰历史签名。
        self._recent_groups: OrderedDict[Any, tuple[str, ...]] = OrderedDict()
        # Counters survive payload disposal without retaining rejected trajectories.
        self._discarded_counts: dict[str, int | float] = dict.fromkeys(
            (
                "discarded_groups_total",
                "skipped_groups_total",
                "empty_groups_total",
                "tail_groups_total",
                "discarded_trajectories_total",
                "failed_trajectories_total",
                "skipped_trajectories_total",
                "discarded_response_tokens_total",
                "discarded_rollout_seconds_total",
            ),
            0,
        )

    @staticmethod
    def _has_training_targets(trajectories: list[Trajectory]) -> bool:
        # A scored zero-segment member still participates in the reward comparison.
        # Only a group with no active prediction targets anywhere is empty.
        return any(
            len(sequence["tokens"]) > 1 and any(mask > 0 for mask in sequence["loss_mask"][1:])
            for trajectory in trajectories
            for sequence in iter_trajectory_sequences(trajectory)
        )

    def _remember_consumed_group(self, group_id: Any, group: list[Trajectory]) -> None:
        """Retain bounded retry signatures for trained and discarded groups alike."""
        self._recent_groups[group_id] = tuple(trajectory["rollout_id"] for trajectory in group)
        while len(self._recent_groups) > self._MAX_RECENT_GROUPS:
            _, expired = self._recent_groups.popitem(last=False)
            for rollout_id in expired:
                self._rollout_signatures.pop(rollout_id, None)

    def _discard_group(self, group_id: Any, group: list[Trajectory], reason: str) -> None:
        """Record excluded outcomes while holding the store lock, then release payloads."""
        counts = self._discarded_counts
        counts["discarded_groups_total"] += 1
        counts[f"{reason}_groups_total"] += 1
        counts["discarded_trajectories_total"] += len(group)
        outcomes = []
        for trajectory in group:
            if trajectory.status in ("failed", "skipped"):
                counts[f"{trajectory.status}_trajectories_total"] += 1
            response_tokens = sum(
                sum(mask > 0 for mask in sequence["loss_mask"])
                for sequence in iter_trajectory_sequences(trajectory)
            )
            counts["discarded_response_tokens_total"] += response_tokens
            runtime_metrics = trajectory.get("runtime_metrics")
            duration = (
                runtime_metrics.get("duration_seconds")
                if isinstance(runtime_metrics, Mapping)
                else None
            )
            duration_seconds = (
                float(duration)
                if isinstance(duration, Real) and not isinstance(duration, bool)
                else None
            )
            if (
                duration_seconds is not None
                and math.isfinite(duration_seconds)
                and duration_seconds >= 0
            ):
                counts["discarded_rollout_seconds_total"] += duration_seconds
            outcomes.append(
                {
                    **{
                        key: trajectory[key]
                        for key in (
                            "task_id",
                            "rollout_id",
                            "run_id",
                            "e2b_status",
                            "e2b_termination_reason",
                        )
                        if key in trajectory
                    },
                    "status": trajectory.status,
                    "reward": trajectory.reward,
                    "response_tokens": response_tokens,
                    "duration_seconds": duration,
                }
            )
        self._remember_consumed_group(group_id, group)
        logger.info(
            "[group_discarded] model_tag=%s group_id=%s reason=%s outcomes=%s totals=%s",
            self.model_tag,
            group_id,
            reason,
            outcomes,
            counts,
        )

    def discarded_counts(self) -> dict[str, int | float]:
        """Cumulative exclusions in this process, including an exhausted-source tail."""
        with self._lock:
            return dict(self._discarded_counts) if self._require_trainable_groups else {}

    def discard_ready_groups(self) -> None:
        """Dispose of a confirmed finite-source tail without creating a training round."""
        with self._lock:
            while self._ready_groups:
                group_id, group = self._ready_groups.popitem(last=self._group_order == "lifo")
                self._discard_group(group_id, group, "tail")

    def _identity(self, trajectory: Trajectory) -> tuple[Any, int, str]:
        group_id = trajectory.get(self._group_key)
        if group_id is None or group_id == "":
            raise ValueError(f"Missing rollout group identity: {self._group_key}")
        try:
            hash(group_id)
        except TypeError as exc:
            raise ValueError("Rollout group identity must be hashable") from exc
        expected = trajectory.get("expected_rollouts")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected != self._group_size
        ):
            raise ValueError("expected_rollouts must equal the configured group_size")
        run_id = trajectory.get("run_id")
        if isinstance(run_id, bool) or not isinstance(run_id, int) or not 0 <= run_id < expected:
            raise ValueError("run_id must identify one of the expected rollout slots")
        rollout_id = trajectory.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("rollout_id must be a non-empty string")
        if trajectory.get("model_tag", self.model_tag) != self.model_tag:
            raise ValueError("Rollout model_tag does not match its pool instance")
        return group_id, run_id, rollout_id

    @staticmethod
    def _finite_reward(reward: Any) -> bool:
        return isinstance(reward, Real) and not isinstance(reward, bool) and math.isfinite(reward)

    def _signature(self, trajectory: Trajectory, identity: tuple[Any, int, str]) -> tuple[Any, ...]:
        reward = trajectory.reward
        reward_signature = (
            float(reward) if reward is not None and self._finite_reward(reward) else repr(reward)
        )
        digest = hashlib.sha256()
        sequence_count = 0
        for sequence in iter_trajectory_sequences(trajectory):
            fields = sorted(
                (key, value)
                for key, value in sequence.items()
                if key not in {"reward", "is_val", "status", "group_skipped", "segments"}
            )
            digest.update(len(fields).to_bytes(8, "big"))
            for key, value in fields:
                _hash_sequence_field(digest, key)
                _hash_sequence_field(digest, value)
            sequence_count += 1
        return (
            identity[0],
            identity[1],
            trajectory.status,
            trajectory.is_val,
            reward_signature,
            sequence_count,
            digest.digest(),
        )

    def put_trajectory(self, trajectory: Trajectory) -> bool:
        """Atomically publish one terminal rollout result; never count segments as rollouts."""
        if not isinstance(trajectory, Trajectory):
            raise TypeError("The trajectory pool accepts complete Trajectory objects")
        identity = self._identity(trajectory)
        group_id, run_id, rollout_id = identity
        if trajectory.status not in {"success", "skipped", "failed"}:
            raise ValueError("Trajectory requires a terminal success/skipped/failed status")
        signature = self._signature(trajectory, identity)
        with self._lock:
            previous = self._rollout_signatures.get(rollout_id)
            if previous is not None:
                # 从签名读取归属，保留尚未取走结果的具体身份冲突报错。
                if previous[0] not in self._recent_groups and previous[:2] != (group_id, run_id):
                    raise ValueError(
                        "rollout_id was already assigned to a different group or run slot"
                    )
                if previous != signature:
                    raise ValueError("Conflicting duplicate trajectory submission")
                return True
            if group_id in self._ready_groups or group_id in self._recent_groups:
                raise ValueError("Cannot add a new rollout to a completed group")
            group = self._groups.setdefault(group_id, {})
            if run_id in group:
                raise ValueError("A run slot cannot contain multiple independent rollouts")
            group[run_id] = trajectory
            self._rollout_signatures[rollout_id] = signature
            if len(group) == self._group_size:
                trajectories = [group[slot] for slot in range(self._group_size)]
                valid = [
                    item.status == "success"
                    and self._finite_reward(item.reward)
                    and not item.get("group_skipped", False)
                    for item in trajectories
                ]
                skipped = sum(valid) * 2 < self._group_size
                for item, usable in zip(trajectories, valid, strict=True):
                    item["group_skipped"] = skipped or not usable
                del self._groups[group_id]
                if self._require_trainable_groups and skipped:
                    self._discard_group(group_id, trajectories, "skipped")
                elif self._require_trainable_groups and not self._has_training_targets(
                    [item for item, usable in zip(trajectories, valid, strict=True) if usable]
                ):
                    self._discard_group(group_id, trajectories, "empty")
                else:
                    self._ready_groups[group_id] = trajectories
        return True

    def take_groups(self, *, num_groups: int) -> list[Trajectory] | None:
        """Consume exactly num_groups queued groups in group_order, returning trajectories.

        Normal grouped RL batches have B = data.train.batch_size groups, each
        with R = data.train.rollout_repeat_n trajectories. Availability and
        consumption are checked together under the store lock, so the caller
        does not need a readiness precheck. If fewer groups are ready, nothing
        is taken. The caller may request a smaller confirmed finite-source tail.
        Groups keep run_id order and each trajectory retains all its segments.
        With require_trainable_groups, only eligible complete groups are queued.
        """
        if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 1:
            raise ValueError("num_groups must be a positive integer")
        with self._lock:
            if len(self._ready_groups) < num_groups:
                return None
            trajectories: list[Trajectory] = []
            for _ in range(num_groups):
                group_id, group = self._ready_groups.popitem(last=self._group_order == "lifo")
                trajectories.extend(group)
                self._remember_consumed_group(group_id, group)
            return trajectories

    def progress_snapshot(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                sum(self._ready_sequence_count(group) for group in self._ready_groups.values()),
                sum(len(group) for group in self._ready_groups.values()),
                len(self._ready_groups),
                len(self._groups),
            )

    def ready_group_count(self) -> int:
        with self._lock:
            return len(self._ready_groups)

    def has_pending_groups(self) -> bool:
        with self._lock:
            return bool(self._groups)

    def _is_active_storage_empty(self) -> bool:
        return not self._groups and not self._ready_groups


class TrajectoryPoolInstance:
    """
    Instance for a single ModelTag that manages a store, lock, and ready batches
    Each instance is responsible for a single model.
    """

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        self.model_tag = model_tag
        self._config = config
        self._lock = threading.RLock()
        self._store = self._create_store(config, model_tag)
        self._need_sampling = config.get("need_sampling", True)
        self._require_trainable_groups = bool(config.get("require_trainable_groups", False))

        self._activity_tracker: "ActivityTrackerProxy" = None  # type: ignore   # Will be set by trajectory pool
        self._dataloader: "BaseDataLoader" = None  # type: ignore   # Will be set by trajectory pool
        self._weight_sync_controller: "WeightSyncController" = None  # type: ignore   # Will be set by trajectory pool

    def _create_store(
        self, config: DictConfig, model_tag: ModelTag
    ) -> SimpleTrajectoryStore | GroupedTrajectoryStore:
        """Create appropriate store based on configuration"""
        key_list = config.get("key_list", [])

        if not key_list:
            logger.info(f"No grouping keys specified, using simple store for model_tag {model_tag}")
            # No grouping keys specified, use simple store
            return SimpleTrajectoryStore(config, model_tag)
        elif len(key_list) == 1:
            logger.info(
                f"Single grouping key specified, using grouped store for model_tag {model_tag}"
            )
            # Single grouping key, use grouped stor
            config_copy = config.copy()
            config_copy["group_key"] = key_list[0]
            return GroupedTrajectoryStore(config_copy, model_tag)
        else:
            raise ValueError("Trajectory pool supports at most one grouping key")

    def put_trajectory(self, trajectory: Trajectory) -> str:
        """Add one complete rollout result to the store.

        Returns:
            'success': Trajectory added successfully
            'fail': Failed to add trajectory
        """
        # Publish the progress observation before a consumer can take the result
        # and release the next ordered input batch. This never gates readiness.
        with self._lock:
            success = self._store.put_trajectory(trajectory)
            if not success:
                return "fail"
            record_result = getattr(self._dataloader, "record_rollout_result", None)
            if record_result is not None:
                try:
                    record_result(trajectory)
                except Exception:  # Progress reporting must not fail an accepted rollout.
                    logger.warning("Could not record rollout progress", exc_info=True)

        return "success"

    def _groups_per_batch(self) -> int:
        """Return B, derived by the Controller from data.train.batch_size."""
        count = int(self._config.get("groups_per_batch", 1))
        if count < 1:
            raise ValueError("groups_per_batch must be positive")
        return count

    def ready_group_count(self) -> int:
        """Return queued groups, counting eligible groups only in fully async."""
        with self._lock:
            if not isinstance(self._store, GroupedTrajectoryStore):
                return 0
            return self._store.ready_group_count()

    def _prepare_trajectories_if_ready(self) -> list[Trajectory] | None:
        """Consume one training batch, or return None until it is complete.

        Grouped RL requests B = data.train.batch_size groups. Fully async queues
        only eligible groups and discards an undersized finite-source tail;
        other modes may flush fewer complete groups at source exhaustion.
        Worker activity does not gate a full batch. Unfinished groups with no
        buffered tasks or active workers indicate missing rollout results and
        raise instead of waiting forever.

        SFT uses the simple store and token-budget input batches with variable
        trajectory counts. Wait for the current input round to finish, then
        consume all of its complete parent trajectories, keeping segments attached.
        """
        if isinstance(self._store, GroupedTrajectoryStore):
            trajectories = self._store.take_groups(num_groups=self._groups_per_batch())
            if trajectories is not None:
                return trajectories
            if self._store.is_empty():
                return None
            if self._dataloader is None or self._activity_tracker is None:
                return None
            dataloader_drained = not self._dataloader.can_return_item()
            source_finished = self._dataloader.is_finished() is True
            # Observe worker completion after source exhaustion: claiming the
            # last buffered tasks can finish the source while work is in flight.
            if not self._activity_tracker.is_rollout_worker_quiescent():
                return None
            if dataloader_drained and self._store.has_pending_groups():
                raise RuntimeError("Rollout workers are idle with unfinished rollout groups")
            if not source_finished:
                return None
            if self._require_trainable_groups:
                self._store.discard_ready_groups()
                return None
            num_groups = min(self._store.ready_group_count(), self._groups_per_batch())
            return self._store.take_groups(num_groups=num_groups) if num_groups else None

        if self._store.is_empty():
            return None
        if self._dataloader.can_return_item():
            return None
        if not self._activity_tracker.is_rollout_worker_quiescent():
            return None
        return self._store.take_trajectories(num_trajectories=self._store.ready_trajectory_count())

    def get_trajectories(self) -> list[Trajectory] | None:
        """
        Get a list of trajectories if possible

        Returns:
            A list of ready trajectories if available, None otherwise
        """
        trajectories = None
        with self._lock:
            trajectories = self._prepare_trajectories_if_ready()

        if trajectories is not None:
            logger.info("TrajectoryPoolInstance get trajectories successfully!")

            if self._need_sampling:
                self._weight_sync_controller.trajectory_pool_notify_batch_ready(self.model_tag)
        return trajectories

    def is_empty(self) -> bool:
        """Check if store are empty"""
        with self._lock:
            return self._store.is_empty()

    def set_module_references(
        self,
        dataloader: "BaseDataLoader",
        weight_sync_controller: "WeightSyncController",
        activity_tracker: "ActivityTrackerProxy",
    ) -> None:
        """Set reference to dataloader, weight sync controller, and activity tracker for coordination"""
        self._dataloader = dataloader
        self._weight_sync_controller = weight_sync_controller
        self._activity_tracker = activity_tracker

    def get_progress_info(self) -> dict[str, Any]:
        """
        Get progress information for this trajectory pool instance.

        Returns:
            dict with keys:
                - ready_segments: Compatibility alias for ready_sequences
                - ready_sequences: Number of training sequences in ready trajectories
                - ready_trajectories: Queued trajectories; fully async excludes discarded groups
                - finished_trajectories: Compatibility alias for ready_segments
                - incomplete_groups: Groups with some results received, awaiting remaining rollouts
                - groups_per_batch: Queued groups required per batch, or zero without grouping
                - *_total: Cumulative fully-async exclusions, retained without trajectory payloads
        """
        with self._lock:
            ready_count, ready_trajectories, ready_groups, incomplete_groups = (
                self._store.progress_snapshot()
            )
            return {
                "ready_sequences": ready_count,
                "ready_segments": ready_count,  # Legacy sequence-count name
                "ready_trajectories": ready_trajectories,
                "finished_trajectories": ready_count,  # Legacy name; also counts training sequences
                "ready_groups": ready_groups,
                "incomplete_groups": incomplete_groups,
                "groups_per_batch": (
                    self._groups_per_batch()
                    if isinstance(self._store, GroupedTrajectoryStore)
                    else 0
                ),
                **(
                    self._store.discarded_counts()
                    if isinstance(self._store, GroupedTrajectoryStore)
                    else {}
                ),
            }

    def get_discarded_metrics(self) -> dict[str, int | float]:
        if not isinstance(self._store, GroupedTrajectoryStore):
            return {}
        return {f"pool/{key}": value for key, value in self._store.discarded_counts().items()}


class TrajectoryPool(NorthRLModule):
    """
    Multi-instance trajectory pool that manages separate TrajectoryPoolInstance objects for different models.
    Route trajectory operations to the appropriate instance according to the model_tag.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config
        self._instances: dict[ModelTag, TrajectoryPoolInstance] = {}
        self._lock = threading.Lock()
        self._weight_sync_controller: "WeightSyncController" = None  # type: ignore # will be set by controller

    def _get_or_create_instance(self, model_tag: ModelTag) -> TrajectoryPoolInstance:
        """Get existing instance or create new one for the given model_tag"""
        if model_tag not in self._instances:
            logger.info(f"Creating new instance for model_tag: {model_tag}")
            instance = TrajectoryPoolInstance(self._config, model_tag)
            instance.set_module_references(
                dataloader=self._dataloader,
                weight_sync_controller=self._weight_sync_controller,
                activity_tracker=self._activity_tracker,
            )
            self._instances[model_tag] = instance
        return self._instances[model_tag]

    def put_trajectory(self, trajectory: Trajectory) -> str:
        """
        Add trajectory to appropriate instance based on ModelTag
        Args:
            trajectory: Trajectory to add

        Returns:
            'success' if the trajectory is added successfully, 'fail' otherwise
        """
        model_tag = trajectory.get("model_tag", "default")

        with self._lock:
            instance = self._get_or_create_instance(model_tag)

        return instance.put_trajectory(trajectory)

    def get_trajectories(self, model_tag: ModelTag | None = None) -> list[Trajectory] | None:
        """
        Get trajectories from specified model_tag instance, or from default if not specified

        Args:
            model_tag: ModelTag to get trajectories from. If None, uses default model_tag

        Returns:
            List of trajectories from the specified instance, or None if not enough samples
        """
        if model_tag is None:
            return self._get_trajectories_any()

        assert model_tag in self._instances, f"Model tag {model_tag} not found in instances"

        return self._instances[model_tag].get_trajectories()

    def _get_trajectories_any(self) -> list[Trajectory] | None:
        """
        Get trajectories from any available instance that has enough trajectories

        Returns:
            List of trajectories from any instance, or None if no instance has enough
        """
        for model_tag in self._instances:
            trajectories = self.get_trajectories(model_tag)
            if trajectories is not None:
                return trajectories

        return None

    def is_empty(self, model_tag: ModelTag | None = None) -> bool:
        """
        Check if specified instance is empty, or if all instances are empty

        Args:
            model_tag: ModelTag to check. If None, checks if all instances are empty

        Returns:
            True if specified instance (or all instances) is empty
        """
        with self._lock:
            if model_tag is not None:
                if model_tag not in self._instances:
                    return True
                return self._instances[model_tag].is_empty()
            else:
                # Check if all instances are empty
                return all(instance.is_empty() for instance in self._instances.values())

    def get_model_tags(self) -> list[ModelTag]:
        """Get list of all ModelTags that have instances"""
        with self._lock:
            return list(self._instances.keys())

    def ready_group_count(self, model_tag: ModelTag | None = None) -> int:
        """Return queued groups (eligible groups only in fully async) for backpressure."""
        with self._lock:
            if model_tag is not None:
                instance = self._instances.get(model_tag)
                return instance.ready_group_count() if instance is not None else 0
            return sum(instance.ready_group_count() for instance in self._instances.values())

    def get_discarded_metrics(self) -> dict[str, int | float]:
        """Snapshot cumulative sampling exclusions without consuming a training step."""
        metrics: dict[str, int | float] = {}
        with self._lock:
            for instance in self._instances.values():
                for key, value in instance.get_discarded_metrics().items():
                    metrics[key] = metrics.get(key, 0) + value
        return metrics

    def set_module_references(
        self, dataloader: "BaseDataLoader", weight_sync_controller: "WeightSyncController"
    ) -> None:
        """Set reference to weight manager for coordination"""
        self._dataloader = dataloader
        self._weight_sync_controller = weight_sync_controller

    def get_progress_info(self) -> dict[str, Any]:
        """
        Get aggregated progress information across all trajectory pool instances.

        Returns:
            dict with keys:
                - ready_segments: Compatibility alias for ready_sequences
                - ready_sequences: Current ready training-sequence count across all instances
                - ready_trajectories: Current ready trajectory count across all instances
                - incomplete_groups: Groups awaiting remaining rollouts across all instances
                - finished_trajectories: Compatibility alias for ready_segments
                - instances: Dict mapping model_tag to per-instance progress info
        """
        with self._lock:
            total_ready = 0
            total_ready_groups = 0
            total_incomplete_groups = 0
            total_ready_trajectories = 0
            instances_info = {}

            for model_tag, instance in self._instances.items():
                instance_info = instance.get_progress_info()
                instances_info[model_tag] = instance_info
                total_ready += instance_info["ready_segments"]
                total_ready_groups += instance_info["ready_groups"]
                total_incomplete_groups += instance_info["incomplete_groups"]
                total_ready_trajectories += instance_info["ready_trajectories"]

            return {
                "ready_sequences": total_ready,
                "ready_segments": total_ready,  # Legacy sequence-count name
                "ready_trajectories": total_ready_trajectories,
                "ready_groups": total_ready_groups,
                "incomplete_groups": total_incomplete_groups,
                "finished_trajectories": total_ready,  # Legacy name; also counts training sequences
                "instances": instances_info,
            }
