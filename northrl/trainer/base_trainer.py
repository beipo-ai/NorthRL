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
BaseTrainer - Abstract base class for trainers
"""

import asyncio
import logging
import threading
import time
from abc import abstractmethod
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, Callable

from omegaconf import DictConfig

from ..base_module import NorthRLModule
from ..northrl_types import Trajectory
from ..utils.async_utils import submit_trainer_coro
from ..utils.data_dumper import DataDumper
from ..utils.metric_catalog import normalize_step_metrics
from ..utils.trajectory_sequences import iter_trajectory_sequences

if TYPE_CHECKING:
    from ..data_loader import BaseDataLoader
    from ..trajectory_pool import TrajectoryPool
    from ..weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


class BaseTrainer(NorthRLModule):
    """
    Abstract base class for trainers.

    Derived classes should implement the `train()` method which takes a list of
    trajectories and returns training metrics.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the base trainer

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self._config = config
        self._stop_event: threading.Event = threading.Event()
        self._stop_event.clear()
        self._task_future: Future[None] | None = None

        # Training state
        self._train_step: int = 0
        # Process-local observation; checkpoint resume starts a new interval series.
        self._last_train_completed_at: float | None = None
        self._total_train_steps: int | None = config.training.total_steps
        self._need_sampling = config.algorithm.type != "sft"

        # Module references (set via set_module_references)
        self._trajectory_pool: "TrajectoryPool" = None  # type: ignore
        self._weight_sync_controller: "WeightSyncController" = None  # type: ignore
        self._dataloader: "BaseDataLoader" = None  # type: ignore

        # ModelTag is hardcoded to "actor" in the new schema: identifier as a
        # user-facing field is gone (single trainable model).
        self._identifier = "actor"
        logger.info(f"Trainer using identifier: {self._identifier}")

    def set_module_references(
        self,
        trajectory_pool: "TrajectoryPool",
        weight_sync_controller: "WeightSyncController",
        dataloader: "BaseDataLoader | None" = None,
    ) -> None:
        """
        Set the module references for the trainer.

        Args:
            trajectory_pool: Reference to the trajectory pool
            weight_sync_controller: Reference to the weight sync controller
        """
        self._trajectory_pool = trajectory_pool
        self._weight_sync_controller = weight_sync_controller
        if dataloader is not None:
            self._dataloader = dataloader

    async def initialize_workers(self) -> None:
        """
        Initialize backend workers. Override in derived classes if needed.
        """

    async def _on_training_complete(self) -> None:
        """Run after the training loop exits normally."""

    async def _on_weight_sync_failed(self, completed_step: int) -> None:
        """Preserve a completed training step when publishing its weights fails."""

    def run(self) -> None:
        """Start the training coroutine on NorthRL's dedicated trainer loop."""
        assert self._trajectory_pool is not None, "TrajectoryPool is not set"
        assert self._weight_sync_controller is not None, "WeightSyncController is not set"

        if self._task_future is not None and not self._task_future.done():
            logger.warning("Trainer %s is already running", self._module_name)
            return

        self._stop_event.clear()
        self._task_future = submit_trainer_coro(self._main_loop())
        self._task_future.add_done_callback(self._log_task_failure)

    def _log_task_failure(self, future: Future[None]) -> None:
        if future.cancelled():
            return
        try:
            exc = future.exception()
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Failed to inspect trainer task state: %s", err)
            return
        if exc is not None:
            logger.error(
                "Trainer %s task failed",
                self._module_name,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def stop(
        self, *, on_wait: Callable[[], object] | None = None, wait_interval: float = 1.0
    ) -> None:
        """Drain the current training step, optionally refreshing progress while waiting."""
        if on_wait is not None and wait_interval <= 0:
            raise ValueError("wait_interval must be positive")
        self._stop_event.set()
        future = self._task_future
        if future is not None:
            try:
                # Do not cancel an in-flight optimizer step. The stop flag is
                # observed at the next batch boundary.
                if on_wait is None:
                    future.result()
                else:
                    while True:
                        try:
                            future.result(timeout=wait_interval)
                            break
                        except FutureTimeoutError:
                            if future.done():
                                # A task can itself raise TimeoutError. Retrieve
                                # its result so it follows normal failure handling.
                                future.result()
                                break
                            try:
                                on_wait()
                            except Exception:  # pylint: disable=broad-exception-caught
                                logger.exception(
                                    "Failed to refresh progress while stopping trainer"
                                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Trainer %s stopped after task failure: %s", self._module_name, exc)
            finally:
                self._task_future = None
        logger.info("Trainer stopped")

    async def _main_loop(self) -> None:
        """
        Main training loop:
        1. Get trajectories from trajectory pool
        2. Call train() to process and train
        3. Log metrics and notify weight sync controller
        """
        if self._total_train_steps is not None and self._train_step >= self._total_train_steps:
            logger.info(
                "Trainer already completed %d/%d rounds; nothing to resume",
                self._train_step,
                self._total_train_steps,
            )
            self._stop_event.set()
            await self._on_training_complete()
            return

        # Count insufficient-data polling, pausing for validation and stopping at claim.
        batch_wait_seconds = 0.0
        batch_wait_started_at: float | None = None
        while not self._stop_event.is_set():
            # Validation must finish before claiming another batch, including
            # batches already queued by asynchronous rollout workers. Stay
            # outside activity tracking while waiting so the trainer is idle.
            if self._need_sampling and self._weight_sync_controller.is_waiting_for_validation():
                if batch_wait_started_at is not None:
                    batch_wait_seconds += time.monotonic() - batch_wait_started_at
                    batch_wait_started_at = None
                await asyncio.sleep(0.1)
                continue

            trajectories = None
            with self._activity_tracker.track(self._module_name, "training_step"):
                # Claim while active so completion checks cannot observe a batch
                # that has left the pool but is not yet owned by the trainer.
                claim_started_at = time.monotonic()
                trajectories = self._get_trajectories()
                if trajectories is None and batch_wait_started_at is None:
                    batch_wait_started_at = claim_started_at
                if trajectories is not None:
                    if batch_wait_started_at is not None:
                        batch_wait_seconds += time.monotonic() - batch_wait_started_at
                        batch_wait_started_at = None
                    logger.info("Trainer get trajectories successfully!")

                    records = self._rollout_round_records(trajectories)
                    # Call the train method implemented by derived class
                    start_time = time.monotonic()
                    metrics, sampling_path, weight_version = await self.train(trajectories)
                    metrics.update(self._trajectory_pool.get_discarded_metrics())
                    train_completed_at = time.monotonic()
                    total_time = train_completed_at - start_time
                    completed_step = self._train_step + 1
                    self._record_rollout_round(records, metrics, completed_step)

                    # Emit one experiment-log record per completed rollout/training
                    # round. A filtered batch still advances the training progress.
                    # Remote trainers stop this timer before weight persistence.
                    # Trainers without a separate persistence phase use the call span.
                    training_time = metrics.setdefault(
                        "timing_train/batch_training_seconds", total_time
                    )
                    metrics["timing/trainer_wait_for_batch_seconds"] = batch_wait_seconds
                    batch_wait_seconds = 0.0
                    if self._last_train_completed_at is not None:
                        metrics["timing/batch_completion_interval_seconds"] = (
                            train_completed_at - self._last_train_completed_at
                        )
                    self._last_train_completed_at = train_completed_at
                    if training_time > 0:
                        shifted_positions = metrics.get("training/window_shifted_tokens")
                        if isinstance(shifted_positions, (int, float)):
                            metrics["throughput/input_tokens_per_second"] = (
                                float(shifted_positions) / training_time
                            )
                        effective_tokens = metrics.get("training/loss_agg_denominator")
                        if isinstance(effective_tokens, (int, float)):
                            metrics["throughput/effective_tokens_per_second"] = (
                                float(effective_tokens) / training_time
                            )
                    metrics["training/train_step"] = completed_step
                    metrics = normalize_step_metrics(metrics)
                    logger.info(
                        "[step_completed] train_step=%d metrics=%s", completed_step, metrics
                    )

                    self._activity_tracker.experiment_logger_post(
                        backend="wandb",
                        data=metrics,
                        step=completed_step,
                    )

                    # Validation metrics emitted by WeightSyncController must use the
                    # completed round's step. Keep the trainer's public step unchanged
                    # until weight sync has finished (or has latched validation), so the
                    # controller cannot declare the final round complete too early.
                    self._activity_tracker.set_training_step(completed_step)

                    continue_training = not self._stop_event.is_set() and (
                        self._total_train_steps is None or completed_step < self._total_train_steps
                    )

                    if self._need_sampling:
                        # identifier serves as model_tag for weight sync coordination
                        try:
                            self._weight_sync_controller.train_worker_notify_weight_update(
                                worker_name=self._module_name,
                                model_tag=self._identifier,
                                sampling_path=sampling_path,
                                weight_version=weight_version,
                                weights_updated=not bool(
                                    metrics.get(
                                        "training/skipped_flag", metrics.get("training/skipped", 0)
                                    )
                                ),
                                continue_training=continue_training,
                                completed_step=completed_step,
                            )
                        except Exception:
                            try:
                                await self._on_weight_sync_failed(completed_step)
                            except Exception:
                                logger.exception(
                                    "Failed to save checkpoint after weight synchronization failed"
                                )
                            raise

                    # Keep training and synchronization metrics in one W&B row.
                    # Validation commits this row itself before allowing the next
                    # round; otherwise publish it as soon as synchronization ends.
                    if (
                        not self._need_sampling
                        or not self._weight_sync_controller.is_waiting_for_validation()
                    ):
                        self._activity_tracker.experiment_logger_flush(completed_step)

                    self._train_step = completed_step

            if trajectories is None:
                await asyncio.sleep(0.1)
                continue

            # Check if training is complete
            if self._total_train_steps is not None and self._train_step >= self._total_train_steps:
                self._stop_event.set()
                break

        await self._on_training_complete()

    @staticmethod
    def _rollout_round_records(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
        """Snapshot complete outcomes without adding a second completion protocol.

        Accepted sequences are eligible after the pool's complete-group filtering,
        before trainer-specific filtering or optimizer tail padding.
        """
        records = []
        for trajectory in trajectories:
            sequences = list(iter_trajectory_sequences(trajectory))
            accepted = trajectory.status == "success" and not trajectory.get("group_skipped", False)
            record = {
                key: trajectory[key]
                for key in (
                    "task_id",
                    "group_id",
                    "run_id",
                    "rollout_id",
                    "model_tag",
                    "weight_version",
                    "e2b_status",
                    "e2b_termination_reason",
                )
                if key in trajectory
            }
            record.update(
                status=trajectory.status,
                reward=trajectory.reward,
                group_skipped=trajectory.get("group_skipped", False),
                sequence_count=len(sequences),
                accepted_sequences=len(sequences) if accepted else 0,
                # Managed SampleRef masks are server-private and count as zero.
                response_tokens=sum(sum(sequence["loss_mask"] or ()) for sequence in sequences),
            )
            records.append(record)
        return records

    def _record_rollout_round(
        self, records: list[dict[str, Any]], metrics: dict[str, Any], completed_step: int
    ) -> None:
        """Save the trace-to-task index.

        Rollout totals, outcome counts, sequence counts and response token totals
        are already emitted by log_rollout_metrics; keep only distinct admission
        information here. The manifest retains per-rollout details for diagnosis.
        """
        del metrics  # Counts are owned by the batch aggregator, not the dump index.
        if not records:
            return
        data_dumper = getattr(self, "_data_dumper", None)
        if data_dumper is None:
            data_dumper = self._data_dumper = DataDumper(self._config)
        data_dumper.dump_rollout_round(self._train_step, completed_step, records)

    def _get_trajectories(self) -> list[Trajectory] | None:
        """
        Get trajectories from the trajectory pool.

        Returns:
            List of trajectories or None if not available
        """
        return self._trajectory_pool.get_trajectories()

    @abstractmethod
    async def train(self, trajectories: list[Trajectory]) -> tuple[dict, str | None, str | None]:
        """
        Train on a list of trajectories.

        This is the core method that derived classes must implement.
        It should:
        1. Convert trajectories to batch format if needed
        2. Process the batch (e.g., compute advantages)
        3. Train on the processed batch
        4. Return training metrics and any exported sampling path/name

        The main loop advances the completed-round counter after this method
        returns successfully.

        Args:
            trajectories: List of trajectories to train on

        Returns:
            Training metrics, exported sampling path, and exported weight name.
            Return None for both path and name when no sampling weights are exported.
        """

    # ==================== Public Interface ====================

    def get_train_step(self) -> int:
        """Get the current train step"""
        return self._train_step

    def set_train_step(self, step: int) -> None:
        """
        Set the current train step (used for resuming from checkpoint).

        Args:
            step: The number of completed rollout/training rounds to restore.
        """
        self._train_step = step
        logger.info("Train step set to %d", step)
