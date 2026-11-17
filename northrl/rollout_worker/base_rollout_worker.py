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

"""
Base Rollout Worker for NorthRL framework
"""

import asyncio
import copy
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable

from omegaconf import DictConfig

from ..base_module import NorthRLModule
from ..data_loader import BaseDataLoader
from ..northrl_types import Trajectory
from ..trajectory_pool import TrajectoryPool
from ..utils.async_utils import submit_rollout_coro
from ..utils.profiling import result_profile_block
from ..utils.trajectory_sequences import iter_trajectory_sequences
from ..validator import Validator
from ..weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


class BaseRolloutWorker(NorthRLModule, ABC):
    """
    Rollout worker is used to execute rollout tasks.
    BaseRolloutWorker is an abstract class for various RolloutWorker classes to inherit from.
    Since RolloutWorker is user-implemented code, to ensure clean and readable code,
    the worker does not interact with any other modules in NorthRL, only with worker_manager.
    Functions are called through worker_manager.
    Note: The worker does not actively establish connection with worker_manager when created.
    The worker_manager is responsible for establishing connection with the worker.
    """

    def __init__(self, config: DictConfig):
        """
        All necessary initialization work.

        Note: Inference service client initialization is deferred to init_inference_service_client()
        which is called by the controller after all dependencies are set up.

        Args:
            config: Configuration file
        """
        super().__init__()
        self._config = config
        # New schema: max_sequence_length lives at data.max_sequence_length (SSoT);
        # the worker truncates trajectories whose total length exceeds it.
        self._max_sequence_length = config.data.max_sequence_length
        self._stop_event = threading.Event()
        self._idle_wakeup: Callable[[], None] | None = None
        self._task_future: Future[None] | None = None
        # Inference service client will be initialized via init_inference_service_client() if needed
        self._inference_client = None  # type: ignore  # Will be set if need_llm_inference=True

        self._trajectory_pool: TrajectoryPool = None  # type: ignore  # Set via set_module_references()
        self._dataloader: BaseDataLoader = None  # type: ignore  # Set via set_module_references()
        self._weight_sync_controller: WeightSyncController = None  # type: ignore  # Set via set_module_references()
        self._validate_dataloader: BaseDataLoader = None  # type: ignore  # Set via set_module_references()
        self._validator: Validator = None  # type: ignore  # Set via set_module_references()

        self._next_task: dict[str, Any] | None = None
        self._active_task: dict[str, Any] | None = None
        self._pending_trajectory: Trajectory | None = None

        self._is_running_validate: bool = False
        self._pool_capacity_waiting = False
        self._rollout_trace_dir: str | None = None
        runtime_debug = getattr(getattr(config, "runtime", None), "debug", None)
        if runtime_debug is not None and runtime_debug.enable_data_dump and runtime_debug.dump_dir:
            self._rollout_trace_dir = os.path.join(str(runtime_debug.dump_dir), "rollout_traces")
            os.makedirs(self._rollout_trace_dir, exist_ok=True)

    def set_module_references(
        self,
        trajectory_pool: TrajectoryPool,
        dataloader: BaseDataLoader,
        weight_sync_controller: WeightSyncController,
        validate_dataloader: BaseDataLoader,
        validator: Validator,
    ):
        """
        Set the module references for the worker.
        """
        self._trajectory_pool = trajectory_pool
        self._dataloader = dataloader
        self._weight_sync_controller = weight_sync_controller
        self._validate_dataloader = validate_dataloader
        self._validator = validator
        if self._inference_client is not None:
            self._inference_client.set_weight_sync_controller(weight_sync_controller)

    def init_inference_service_client(self, service_holder=None):
        """
        Initialize inference service client for all backends.
        Called by controller after dependencies are set up.

        Args:
            service_holder: Shared backend-specific service holder (Tinker/Weaver)
        """
        # SFT tokenizes training data directly and does not sample from an LLM.
        need_llm_inference = self._config.algorithm.type != "sft"
        if not need_llm_inference:
            logger.info("LLM inference not needed for this worker, skipping initialization")
            return

        from ..utils.init_utils import create_inference_service_client

        self._inference_client = create_inference_service_client(
            backend=self._config.backend.type,
            config=self._config,
            weaver_service_holder=service_holder,
        )

    def run(self):
        """Start the worker's rollout coroutine on NorthRL's dedicated rollout loop."""
        assert self._trajectory_pool is not None, "TrajectoryPool is not set"
        assert self._dataloader is not None, "DataLoader is not set"
        assert self._weight_sync_controller is not None, "WeightSyncController is not set"

        if self._task_future is not None and not self._task_future.done():
            logger.warning("Rollout worker %s is already running", self._module_name)
            return

        self._stop_event.clear()
        self._task_future = submit_rollout_coro(self._main_loop())
        self._task_future.add_done_callback(self._log_task_failure)

    def _log_task_failure(self, future: Future[None]) -> None:
        if future.cancelled():
            return
        try:
            exc = future.exception()
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Failed to inspect rollout worker task state: %s", err)
            return
        if exc is not None:
            logger.error(
                "Rollout worker %s task failed",
                self._module_name,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    @abstractmethod
    async def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Single rollout operation, defined by user. Derived worker classes should override
        this function to implement user-defined worker operations.

        The implementation should call _put_trajectory() and return its result.

        Returns:
            str: 'success' or 'fail' from _put_trajectory, or None if processing failed
        """

    def request_stop(self) -> None:
        """Request coroutine shutdown without waiting for it to finish."""

        self._stop_event.set()
        self._wake_idle_worker()

    def _wake_idle_worker(self) -> None:
        wakeup = self._idle_wakeup
        if wakeup is not None:
            wakeup()

    def wait_stopped(
        self,
        timeout: float = 30.0,
        *,
        on_wait: Callable[[], object] | None = None,
        wait_interval: float = 1.0,
    ) -> None:
        """Wait for shutdown, optionally refreshing progress, and cancel on timeout."""

        if on_wait is not None and wait_interval <= 0:
            raise ValueError("wait_interval must be positive")
        future = self._task_future
        if future is None:
            return
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    future.result(
                        timeout=remaining if on_wait is None else min(wait_interval, remaining)
                    )
                    break
                except FutureTimeoutError:
                    if future.done():
                        # Distinguish an exception raised by the task from a
                        # timeout waiting for it, including completion races.
                        future.result()
                        break
                    if on_wait is None or time.monotonic() >= deadline:
                        logger.warning(
                            "Timed out waiting for rollout worker %s to stop", self._module_name
                        )
                        future.cancel()
                        break
                    try:
                        on_wait()
                    except Exception:  # pylint: disable=broad-exception-caught
                        logger.exception("Failed to refresh progress while stopping rollout worker")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Rollout worker %s stopped after task failure: %s", self._module_name, exc)
        finally:
            # Keep a concurrently started replacement task intact. Sequential
            # repeated stop/wait calls become no-ops after the first wait.
            if self._task_future is future:
                self._task_future = None

    def stop(self) -> None:
        """Request worker shutdown and wait up to 30 seconds for completion."""

        self.request_stop()
        self.wait_stopped()

    async def _main_loop(self) -> None:
        """Main rollout loop, scheduled as an asyncio task."""
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()

        def wakeup() -> None:
            try:
                loop.call_soon_threadsafe(ready.set)
            except RuntimeError:
                if not loop.is_closed():
                    raise

        self._idle_wakeup = wakeup
        subscriptions: dict[int, Callable[[], None]] = {}
        try:
            for loader in (self._dataloader, self._validate_dataloader):
                if isinstance(loader, BaseDataLoader) and id(loader) not in subscriptions:
                    unsubscribe = loader.subscribe_ready(wakeup)
                    if unsubscribe is not None:
                        subscriptions[id(loader)] = unsubscribe

            while not self._stop_event.is_set():
                # Clear BEFORE checking for work, never between an empty claim
                # and wait: a concurrent release must leave its notification set.
                ready.clear()
                # A worker waiting for capacity owns no task and must be idle.
                await self._wait_for_trajectory_pool_capacity()
                if self._stop_event.is_set():
                    break
                # Register ownership before claiming so quiescence checks cannot
                # observe an already-claimed task as idle.
                with self._activity_tracker.track(self._module_name, "rollout"):
                    self._next_task = self._get_rollout_task()
                    while self._next_task is not None and not self._stop_event.is_set():
                        await self.execute_rollout(copy.deepcopy(self._next_task))
                        if not self._stop_event.is_set():
                            await self._wait_for_trajectory_pool_capacity()
                            if self._stop_event.is_set():
                                break
                            self._next_task = self._get_rollout_task()

                if self._next_task is None and not self._stop_event.is_set():
                    loader = (
                        self._validate_dataloader if self._is_running_validate else self._dataloader
                    )
                    if id(loader) in subscriptions:
                        await ready.wait()
                    else:
                        # Custom loaders can retain polling until they implement
                        # notifications. Stop/validation still wake them promptly.
                        try:
                            await asyncio.wait_for(ready.wait(), timeout=1.0)
                        except TimeoutError:
                            pass
        finally:
            if self._idle_wakeup is wakeup:
                self._idle_wakeup = None
            for unsubscribe in subscriptions.values():
                unsubscribe()

    async def _wait_for_trajectory_pool_capacity(self) -> None:
        """Pause rollout admission while too many complete groups are buffered."""
        max_ready_groups = self._config.rollout.get("max_ready_groups", None)
        if max_ready_groups is None or self._config.algorithm.type not in ("grpo", "opd"):
            return
        # Validation must never be throttled by the training pool watermark: the
        # trainer waits for validation to finish before it consumes the pool, so
        # gating here would deadlock training and validation against each other.
        if self._is_running_validate:
            return
        max_ready_groups = int(max_ready_groups)
        if self._config.training.weight_sync.mode == "fully-async":
            # Admission must be able to fill one eligible training batch even
            # when the configured watermark is smaller than the batch target.
            max_ready_groups = max(max_ready_groups, int(self._config.data.train.batch_size))
        wait_started = False
        while (
            not self._stop_event.is_set()
            and not self._is_running_validate
            and self._trajectory_pool.ready_group_count("actor") >= max_ready_groups
        ):
            if not wait_started:
                wait_started = True
                self._pool_capacity_waiting = True
                logger.info(
                    "Rollout worker %s paused by pool watermark: ready_groups=%d limit=%d",
                    self._module_name,
                    self._trajectory_pool.ready_group_count("actor"),
                    max_ready_groups,
                )
            await asyncio.sleep(0.5)
        if wait_started:
            reason = "validation_started" if self._is_running_validate else "pool_below_limit"
            logger.info(
                "Rollout worker %s resumed after pool watermark wait: reason=%s ready_groups=%d",
                self._module_name,
                reason,
                self._trajectory_pool.ready_group_count("actor"),
            )
            self._pool_capacity_waiting = False

    async def execute_rollout(self, task: dict[str, Any]) -> str | None:
        """Publish one complete trajectory when the worker has finished its attempt.

        Existing workers still implement rollout() and call _put_trajectory().
        Hold that complete object until rollout() returns so a later exception
        cannot leave a successful result in the pool. Failures retain fail-fast
        propagation and publish an explicit failed outcome for the group.
        """
        if task.get("is_val", False):
            return await self.rollout(task)
        if getattr(self, "_active_task", None) is not None:
            raise RuntimeError("A worker cannot execute concurrent rollouts")
        task.setdefault("group_id", uuid.uuid4().hex)
        task.setdefault("run_id", 0)
        task.setdefault("rollout_id", uuid.uuid4().hex)
        task.setdefault("expected_rollouts", 1)
        self._active_task = task
        self._pending_trajectory = None
        from northrl.remote_rollout.metrics import RolloutMetrics

        from .runtime_metrics import ObservedInferenceClient

        started_at = time.time()
        started_monotonic = time.monotonic()
        original_client = self._inference_client
        observations = None
        if original_client is not None:
            observations = RolloutMetrics(rollout_id=task["rollout_id"], observe_tools=False)
            self._inference_client = ObservedInferenceClient(original_client, observations)

        def attach_runtime(trajectory: Trajectory) -> None:
            if (
                observations is not None
                and trajectory.tokens is not None
                and not trajectory.get("is_truncated", False)
            ):
                observations.assign_single_segment()
            runtime: dict[str, Any] = (
                observations.runtime_snapshot() if observations is not None else {}
            )
            known_local_requests = any(
                cls.__module__
                in {
                    "northrl.rollout_worker.single_turn_rollout_worker",
                    "northrl.rollout_worker.simple_rollout_worker",
                }
                for cls in type(self).__mro__
            )
            if not known_local_requests and runtime.get("model_requests", 0) == 0:
                # A custom harness may have bypassed this client entirely.
                # No observed call does not prove that no model request occurred.
                for key in (
                    "model_requests",
                    "model_requests_completed",
                    "model_request_timeouts",
                    "request_usage_count",
                    "request_stats",
                ):
                    runtime.pop(key, None)
            runtime.update(trajectory.get("runtime_metrics", {}))
            # The worker execution boundary also includes final conversion/cleanup.
            runtime.update(
                started_at=started_at,
                finished_at=time.time(),
                duration_seconds=max(0.0, time.monotonic() - started_monotonic),
            )
            trajectory["runtime_metrics"] = runtime
            if (
                observations is not None
                and trajectory.tokens is not None
                and not trajectory.get("is_truncated", False)
            ):
                # Ordinary one-sequence workers have an unambiguous single segment.
                # Multi-turn/nested workers must explicitly map event identities.
                segment_metrics = {}
                for source, target in (
                    ("tool_calls", "tool_calls"),
                    ("tool_timeouts", "tool_timeouts"),
                    ("model_request_timeouts", "model_request_timeouts"),
                ):
                    if source in runtime:
                        segment_metrics[target] = runtime[source]
                if segment_metrics:
                    trajectory["segment_metrics"] = segment_metrics

        try:
            try:
                result = await self.rollout(task)
            except BaseException:
                failed = Trajectory(segments=[], reward=None, status="failed")
                self._stamp_rollout_identity(failed)
                attach_runtime(failed)
                if self._trajectory_pool is not None:
                    self._publish_trajectory(failed)
                raise
            trajectory = self._pending_trajectory
            if trajectory is None:
                trajectory = Trajectory(
                    segments=[], reward=None, status="failed" if result == "fail" else "skipped"
                )
            elif result == "fail":
                trajectory.status = "failed"
            self._stamp_rollout_identity(trajectory)
            attach_runtime(trajectory)
            if self._trajectory_pool is None:
                return result
            return self._publish_trajectory(trajectory)
        finally:
            self._inference_client = original_client
            self._active_task = None
            self._pending_trajectory = None

    def _stamp_rollout_identity(self, trajectory: Trajectory) -> None:
        task = getattr(self, "_active_task", None)
        if task is None:
            return
        for key in ("group_id", "run_id", "rollout_id", "expected_rollouts", "model_tag"):
            value = task.get(key, "actor")
            if key in trajectory.extra_fields and trajectory[key] != value:
                raise ValueError(f"Trajectory {key} does not match its rollout request")
            trajectory[key] = value
        if "task_id" in task:
            trajectory.extra_fields.setdefault("task_id", task["task_id"])

    def _put_trajectory(self, trajectory: Trajectory) -> str:
        """Submit a complete trajectory, with all segments and its final outcome.

        Single-turn workers keep the same call site. Multi-turn workers assemble
        all segments before this call; a worker may submit only one trajectory
        per execution. Truncation and reward adjustment must already be complete.
        """
        if not isinstance(trajectory, Trajectory):
            raise TypeError("_put_trajectory requires a complete Trajectory")
        for sequence in iter_trajectory_sequences(trajectory):
            # Managed SFT trajectories deliberately carry no token IDs. Their
            # public ``input_token_count`` is the causal-shifted length
            # (``len(input_ids) - 1``), so add the omitted first token when
            # applying the same max-sequence check used for token sequences.
            if isinstance(sequence, Trajectory) and sequence.is_managed_sample_ref:
                sequence_length = sequence.extra_fields["input_token_count"] + 1
            else:
                sequence_length = len(sequence["tokens"])
            if sequence_length > self._max_sequence_length:
                raise ValueError(
                    f"Trajectory sequence exceeds max_sequence_length at _put_trajectory: "
                    f"sequence_length={sequence_length}, max_sequence_length={self._max_sequence_length}. "
                    "Truncation and reward adjustment must be done before insertion "
                    "(via _check_and_truncate)."
                )
        if trajectory.is_val:
            result = self._validator.put_trajectory(trajectory)
            logger.info(
                "Rollout worker %s submitted validation result: task_id=%s dataset=%s status=%s",
                self._module_name,
                trajectory.get("task_id"),
                trajectory.get("dataset_name"),
                trajectory.status,
            )
            return result
        self._stamp_rollout_identity(trajectory)
        if getattr(self, "_active_task", None) is not None:
            if self._pending_trajectory is not None:
                raise RuntimeError("Submit one complete Trajectory per rollout, with all segments")
            self._pending_trajectory = trajectory
            return "success"
        return self._publish_trajectory(trajectory)

    def _publish_trajectory(self, trajectory: Trajectory) -> str:
        # _put_trajectory normally only stages the object. Measure the actual
        # publication, including pool locks, grouping and duplicate detection.
        with result_profile_block(
            "driver.result-pool-insert", rollout_id=trajectory.get("rollout_id")
        ):
            return self._trajectory_pool.put_trajectory(trajectory)

    def _check_and_truncate(
        self,
        tokens: list,
        loss_mask: list,
        logprobs: list | None = None,
    ) -> tuple[list, list, list | None, bool]:
        """
        Check if a token sequence exceeds max_sequence_length and truncate if needed.

        Prefer calling this before evaluation so the evaluator can factor
        truncation into reward. A caller that only receives tokens after remote
        evaluation must invalidate the reward before emitting the trajectory.

        Returns:
            (tokens, loss_mask, logprobs, is_truncated)
        """
        if len(tokens) <= self._max_sequence_length:
            return tokens, loss_mask, logprobs, False

        logger.warning(
            f"Sequence truncated from {len(tokens)} to {self._max_sequence_length} tokens"
        )
        tokens = tokens[: self._max_sequence_length]
        loss_mask = loss_mask[: self._max_sequence_length]
        if logprobs is not None:
            logprobs = logprobs[: self._max_sequence_length]
        return tokens, loss_mask, logprobs, True

    def _save_rollout_trace(
        self,
        kind: str,
        task: dict[str, Any],
        logs: dict[str, Any],
    ) -> None:
        """Persist rollout-side trace/debug data when runtime.debug is enabled."""
        if self._rollout_trace_dir is None:
            return

        step = 0
        if self._activity_tracker is not None:
            try:
                step = self._activity_tracker.get_training_step()
            except Exception:  # pylint: disable=broad-exception-caught
                step = int(task.get("step", 0) or 0)

        group_id = task.get("group_id", "")
        run_id = task.get("run_id", 0)
        task_id = task.get("task_id", 0)
        filename = (
            f"{kind}_group_{group_id}_run_{run_id}_task_{task_id}_"
            f"step_{step}-uuid-{uuid.uuid4().hex}.json"
        )
        filepath = os.path.join(self._rollout_trace_dir, filename)
        payload = {
            "kind": kind,
            "module": self._module_name,
            "is_validate": self._is_running_validate,
            "training_step": step,
            "task": task,
            **logs,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._make_json_serializable(payload), f, ensure_ascii=False, indent=2)

    def _make_json_serializable(self, value: Any) -> Any:
        """Best-effort conversion for rollout trace objects before JSON dump."""
        if isinstance(value, dict):
            return {str(k): self._make_json_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._make_json_serializable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "tolist"):
            return self._make_json_serializable(value.tolist())
        if hasattr(value, "model_dump"):
            return self._make_json_serializable(value.model_dump(mode="json", exclude_none=True))
        if hasattr(value, "to_dict"):
            return self._make_json_serializable(value.to_dict())
        if hasattr(value, "__dict__"):
            return self._make_json_serializable(value.__dict__)
        return str(value)

    def _get_rollout_task(self) -> dict[str, Any] | None:
        """
        Get a rollout task from the DataLoader with weight-aware coordination.
        """
        if self._is_running_validate:
            task = self._validate_dataloader.get_next_item()
            if task is not None:
                logger.info(
                    "Rollout worker %s claimed validation task: task_id=%s dataset=%s",
                    self._module_name,
                    task.get("task_id"),
                    task.get("dataset_name"),
                )
            return task
        else:
            return self._dataloader.get_next_item()

    def begin_validate(self) -> None:
        """
        Begin validation mode - switch to using validation dataloader and collector.
        """
        self._is_running_validate = True
        logger.info(
            "Rollout worker %s entered validation mode (pool_waiting=%s)",
            self._module_name,
            self._pool_capacity_waiting,
        )
        if self._inference_client is not None:
            self._inference_client.set_validation_mode(True)
        self._wake_idle_worker()

    def end_validate(self) -> None:
        """
        End validation mode - switch back to training dataloader and trajectory pool.
        """
        self._is_running_validate = False
        logger.info("Rollout worker %s exited validation mode", self._module_name)
        if self._inference_client is not None:
            self._inference_client.set_validation_mode(False)
        self._wake_idle_worker()


def validate_trajectory(trajectory: Trajectory) -> bool:  # pylint: disable=unused-argument
    """
    Validate the trajectory.
    """
    return True
