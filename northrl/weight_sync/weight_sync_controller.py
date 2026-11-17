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
Weight Manager for NorthRL framework
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from omegaconf import DictConfig

from ..base_module import NorthRLModule
from ..northrl_types import ModelTag
from ..utils.async_utils import run_coro

if TYPE_CHECKING:
    from ..data_loader import BaseDataLoader
    from ..trajectory_pool import TrajectoryPool
    from ..weaver.weaver_service_holder import WeaverServiceHolder

logger = logging.getLogger(__name__)


@dataclass
class RolloutServiceState:
    """State tracking for model weight synchronization (weaver/mock backends)."""

    # State tracking
    state: Literal["need_sync", "syncing", "running"] = "running"
    rollout_model_version: int = 0
    train_model_version: int = 0

    # Per-state lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)


class WeightSyncController(NorthRLModule):
    """
    Weight Manager - Manages model weights and synchronization coordination

    Supports three synchronization modes:
    - sync: Block all workers until all sync to newest version
    - fully-async: Continuous rollout, sampler synchronization after each weight update
    - batch-async: Block individual workers when staleness exceeds threshold
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config

        # New schema: WSC receives the root NorthRLConfig and pulls everything
        # it needs from training.weight_sync.* / validate.schedule.* / backend.*.
        weight_sync = config.training.weight_sync
        self.sync_mode = weight_sync.mode
        self._staleness_threshold = weight_sync.staleness_threshold
        self._pause_mode = weight_sync.get("pause_mode", None)

        # backend.type selects the rollout sync implementation (weaver | mock).
        self._backend_type = config.backend.type

        # Validation coordination — `every_n_steps == 0` means disabled.
        self._validate_freq = config.validate.schedule.every_n_steps
        self._waiting_for_validation = False
        self._continue_training_after_validation = True

        # ModelTag is hardcoded to "actor" (single trainable model in new schema).
        self._identifier: ModelTag = "actor"
        self._rollout_services: dict[ModelTag, RolloutServiceState] = {
            self._identifier: RolloutServiceState(),
        }

        # References to other components (will be set via dependency injection)
        self._trajectory_pool: "TrajectoryPool" = None  # type: ignore  # Will be set by controller
        self._dataloader: "BaseDataLoader" = None  # type: ignore  # Will be set by controller
        self._weaver_service_holder: "WeaverServiceHolder | None" = None  # For Weaver backend

        logger.info(
            f"WeightSyncController initialized: sync_mode={self.sync_mode}, "
            f"backend_type={self._backend_type}, "
            f"staleness_threshold={self._staleness_threshold}, "
            f"validate_freq={self._validate_freq}"
        )

    def set_module_references(
        self, dataloader: "BaseDataLoader", trajectory_pool: "TrajectoryPool"
    ) -> None:
        """Set references to other modules for coordination"""
        self._trajectory_pool = trajectory_pool
        self._dataloader = dataloader

    def set_weaver_service_holder(self, weaver_service_holder: "WeaverServiceHolder") -> None:
        """Set reference to Weaver service holder (for Weaver backend only)"""
        self._weaver_service_holder = weaver_service_holder
        logger.info("WeaverServiceHolder reference set in WeightSyncController")

    def check_rollout_service_status(self, model_tag: ModelTag) -> Literal["continue", "block"]:
        """
        Check if rollout service should continue or block for the given model.
        Called by InferenceServiceClient before generate and completion calls.

        Args:
            model_tag: Model tag to check status for

        Returns:
            'continue' | 'block'
            continue: rollout service can proceed
            block: rollout service should block and retry
        """
        assert (
            model_tag in self._rollout_services
        ), f"Rollout service not found for model tag: {model_tag}"

        rollout_service = self._rollout_services[model_tag]

        with rollout_service._lock:  # pylint: disable=protected-access
            if rollout_service.state in ["need_sync", "syncing"]:
                return "block"
            else:
                return "continue"

    def is_waiting_for_validation(self) -> bool:
        """
        Check if weight sync is waiting for controller to handle validation

        Returns:
            True if waiting for validation to complete
        """
        return self._waiting_for_validation

    def end_validate(self, model_tag: ModelTag) -> None:
        """
        Controller signals validation is done so the next training batch can start.

        Args:
            model_tag: Model tag for validation
        """
        continue_training = self._continue_training_after_validation
        self._waiting_for_validation = False
        self._continue_training_after_validation = True

        if continue_training and self.sync_mode != "fully-async":
            self._release_next_batch()
            logger.info("Validation complete for %s - next batch released", model_tag)
        elif not continue_training:
            logger.info("Validation complete for %s at the final training round", model_tag)

    def _release_next_batch(self) -> None:
        """Make the next batch available to workers."""
        self._dataloader.release_next_batch()

    def get_rollout_model_version(self, model_tag: ModelTag) -> int:
        """
        Get current rollout model version

        Args:
            model_tag: Model tag to query

        Returns:
            Current rollout model version
        """
        rollout_service = self._rollout_services[model_tag]
        with rollout_service._lock:  # pylint: disable=protected-access
            return rollout_service.rollout_model_version

    def trajectory_pool_notify_batch_ready(self, model_tag: ModelTag) -> None:
        """
        Called when a batch becomes ready in TrajectoryPoolInstance.
        Keep the next data batch waiting when weight synchronization is needed.

        Args:
            model_tag: Model tag for which batch is ready
        """
        # Get or create the rollout service state
        assert (
            model_tag in self._rollout_services
        ), f"Rollout service not found for model tag: {model_tag}"

        rollout_service = self._rollout_services[model_tag]

        if self.sync_mode == "no-sync":
            return

        logger.debug(f"Trajectory pool batch ready for {model_tag}, sync mode: {self.sync_mode}")

        with rollout_service._lock:  # pylint: disable=protected-access
            # Decide whether the next data batch must wait for training and weight sync.
            if self.sync_mode == "sync" or (
                self.sync_mode == "batch-async"
                and rollout_service.train_model_version + 1 - rollout_service.rollout_model_version
                > self._staleness_threshold
            ):
                assert (
                    rollout_service.state == "running"
                ), "Rollout service should be running when weight sync is needed"

                logger.debug(
                    f"Batch ready for {model_tag}, weight sync needed. "
                    f"Rollout version: {rollout_service.rollout_model_version}, "
                    f"Incoming train version: {rollout_service.train_model_version + 1}"
                )

                rollout_service.state = "need_sync"
                logger.debug("Next data batch for %s will wait for weight sync", model_tag)
            else:
                logger.debug(f"Batch ready for {model_tag}, weight sync not needed. ")
                if self.sync_mode == "batch-async":
                    self._release_next_batch()

    def train_worker_notify_weight_update(
        self,
        worker_name: str,
        model_tag: ModelTag,
        *,
        sampling_path: str | None,
        weight_version: str | None,
        weights_updated: bool = True,
        continue_training: bool = True,
        completed_step: int | None = None,
    ) -> None:
        """
        Handle training completion and coordinate weight synchronization.
        This method synchronously performs weight synchronization.

        Args:
            worker_name: Name of the training worker
            model_tag: Model tag that was trained
            sampling_path: Sampler weights exported by this training round, if any.
            weight_version: Name used when exporting those sampler weights, if any.
            weights_updated: Whether this round actually changed model parameters.
            continue_training: Whether another rollout/training round is expected.
            completed_step: Global completed training round for validation scheduling and
                synchronization timing metrics. None preserves version-based scheduling
                for legacy direct callers; BaseTrainer always supplies the global round.
        """
        if self.sync_mode == "no-sync":
            return

        assert (
            model_tag in self._rollout_services
        ), f"Rollout service not found for model tag: {model_tag}"

        rollout_service = self._rollout_services[model_tag]

        if weights_updated:
            with rollout_service._lock:  # pylint: disable=protected-access
                rollout_service.train_model_version += 1
                logger.debug(
                    f"Training completed by {worker_name} for {model_tag}, new weight version: {rollout_service.train_model_version}"
                )

                # Fully-async publishes every update, including the final round.
                need_sync = self.sync_mode == "fully-async" or rollout_service.state == "need_sync"
                if need_sync:
                    rollout_service.state = "syncing"

            if not need_sync:
                return  # Batch-async already released data before this training round.

            logger.info("Starting weight synchronization for %s", model_tag)
            sync_started = time.monotonic()
            try:
                self.sync_weight_to_rollout_service(model_tag, sampling_path, weight_version)
            except Exception:
                self._log_sync_timing(
                    time.monotonic() - sync_started, completed_step=completed_step, success=False
                )
                raise
            sync_time = time.monotonic() - sync_started
            with rollout_service._lock:  # pylint: disable=protected-access
                rollout_service.state = "running"

            logger.info(
                "Weight synchronization completed for %s to version %s in %.2fs",
                model_tag,
                rollout_service.rollout_model_version,
                sync_time,
            )
            # Publish before validation can observe its latch or the next batch
            # is released. Export, logging, and validation are outside this timer.
            self._log_sync_timing(sync_time, completed_step=completed_step, success=True)

            # Sync versions restart at zero after resume. Use the trainer's global
            # round when available; keep the old behavior for legacy direct callers.
            validation_step = (
                completed_step
                if completed_step is not None
                else rollout_service.rollout_model_version
            )
            if (
                self._validate_freq > 0
                and validation_step > 0
                and validation_step % self._validate_freq == 0
            ):
                logger.info(
                    "Validation required: global_step=%s sync_version=%s every_n_steps=%s "
                    "schedule_source=%s; waiting before the next training round",
                    completed_step,
                    rollout_service.rollout_model_version,
                    self._validate_freq,
                    "global_step" if completed_step is not None else "legacy_sync_version",
                )
                # Store the decision before the controller can observe the wait.
                self._continue_training_after_validation = continue_training
                self._waiting_for_validation = True
                return
        else:
            logger.info(
                "Training skipped by %s for %s; keeping current weights",
                worker_name,
                model_tag,
            )
            with rollout_service._lock:  # pylint: disable=protected-access
                if rollout_service.state != "need_sync":
                    return  # This round did not leave a batch waiting for synchronization.
                rollout_service.state = "running"

        # Successful synchronization and a skipped waiting round release data here.
        # Fully-async loaders fetch the next batch automatically without a release.
        if continue_training and self.sync_mode != "fully-async":
            self._release_next_batch()
        elif not continue_training:
            logger.info("Training round completed for %s; no next batch requested", model_tag)

    def _log_sync_timing(
        self, elapsed: float, *, completed_step: int | None, success: bool
    ) -> None:
        if completed_step is None or self._activity_tracker is None:
            return
        key = "timing/weight_sync_seconds" if success else "timing/weight_sync_failed_seconds"
        try:
            self._activity_tracker.experiment_logger_post(
                backend="wandb",
                data={key: elapsed, "training/weight_sync_success_flag": int(success)},
                step=completed_step,
            )
        except Exception:
            # Observability must not change batch release or mask a sync failure.
            logger.warning("Failed to publish weight synchronization metrics", exc_info=True)

    def sync_weight_to_rollout_service(
        self, model_tag: ModelTag, sampling_path: str | None, weight_version: str | None
    ) -> None:
        """Publish the supplied weights and record the synchronized update count."""
        rollout_service = self._rollout_services[model_tag]
        with rollout_service._lock:  # pylint: disable=protected-access
            train_model_version = rollout_service.train_model_version

        if self._backend_type == "weaver":
            if sampling_path is None or weight_version is None:
                raise ValueError(
                    "Sampling path and weight version are required for weight synchronization"
                )
            assert self._weaver_service_holder is not None, "WeaverServiceHolder not set"
            if self._pause_mode is None:
                run_coro(
                    self._weaver_service_holder.update_sampling_client(
                        sampling_path, weight_version
                    )
                )
            else:
                run_coro(
                    self._weaver_service_holder.update_sampling_client(
                        sampling_path, weight_version, pause_mode=self._pause_mode
                    )
                )
        elif self._backend_type != "mock":
            raise ValueError(f"Unsupported backend type: {self._backend_type}")

        with rollout_service._lock:  # pylint: disable=protected-access
            rollout_service.rollout_model_version = train_model_version
