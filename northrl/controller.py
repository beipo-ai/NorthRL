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
NorthRL Controller - Main controller for the RL training framework
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .activity_tracker import ActivityTracker, ActivityTrackerProxy
from .data_loader import (
    BaseDataLoader,
    MixtureDataLoader,
    MultiDatasetValidationDataLoader,
    SFTStreamingDataLoader,
    TorchDataLoader,
)
from .mock.mock_rollout_worker import MockRolloutWorker
from .northrl_types import ModelTag, NorthRLRole
from .rollout_worker import (
    BaseRolloutWorker,
    E2BRolloutWorker,
    SFTRolloutWorker,
    SimpleRolloutWorker,
    SingleTurnRolloutWorker,
)
from .trainer import BaseTrainer
from .trainer.remote_api_cross_entropy_trainer import RemoteApiCrossEntropyTrainer
from .trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from .trainer.remote_api_opd_trainer import RemoteApiOpdTrainer
from .trajectory_pool import TrajectoryPool
from .utils.async_utils import run_coro
from .utils.config_utils import get_derived_sampling_flags, use_weaver
from .utils.logging_utils import format_memory_usage, set_logging_basic_config
from .validator import Validator
from .weight_sync.versioning import ACTOR_MODEL_TAG as _ACTOR_MODEL_TAG
from .weight_sync.versioning import (
    format_weight_version,
)
from .weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """
    Set the seed of each random module.
    `torch.manual_seed` will set seed on all devices.
    Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NorthRLController:
    """
    NorthRL controller is the main process of the RL training framework.
    Responsible for starting the entire experiment, initializing various modules,
    and synchronizing configuration information.

    Components are initialized locally and get references to the modules they
    need to interact with through the controller.

    Interaction relationship: DataLoader <- RolloutWorker -> TrajectoryPool <- Trainer
    """

    # Public interfaces - accessible to external components
    trajectory_pool: TrajectoryPool
    weight_sync_controller: WeightSyncController
    dataloader: BaseDataLoader
    validate_dataloader: BaseDataLoader
    validator: Validator
    trainer: BaseTrainer
    rollout_workers: list[BaseRolloutWorker]
    activity_tracker: ActivityTracker

    def __init__(self, config: DictConfig):
        """
        Initialize all members of NorthRL.
        Calls all other init() functions of all other modules in `_init_modules()`.

        Args:
            config: Configuration file
        """
        set_logging_basic_config()

        seed_everything(config.experiment.seed)

        self._config = config

        logger.info("Controller initializing in local mode")

        # Initialize activity tracker and error reporter
        self._init_activity_tracker()

        self.use_weaver = use_weaver(config)
        if self.use_weaver:
            self._init_weaver_service()

        # Initialize modules using unified approach
        self._init_modules()

        logger.info("NorthRL Controller initialized successfully")

    def run(self) -> None:
        """Run training and always stop initialized components."""
        run_error: BaseException | None = None
        close_weaver_first = False
        try:
            self._run_training_process()
        except BaseException as exc:
            run_error = exc
            # SIGINT is raised as KeyboardInterrupt and northrl.main converts
            # SIGTERM to SystemExit. On either external shutdown path, close
            # Weaver before waiting for an in-flight trainer operation: that
            # operation may itself be blocked on Weaver and otherwise prevent
            # the remote server resources from ever being released.
            close_weaver_first = isinstance(exc, (KeyboardInterrupt, SystemExit))
            raise
        finally:
            if run_error is None:
                logger.info("Training loop finished; entering controller shutdown")
            else:
                logger.warning(
                    "Training loop exited with %s; entering controller shutdown%s",
                    type(run_error).__name__,
                    " with Weaver closed first" if close_weaver_first else "",
                )
            shutdown_succeeded = False
            try:
                if close_weaver_first:
                    self._stop(close_weaver_first=True)
                else:
                    self._stop()
                shutdown_succeeded = True
            except Exception:  # pylint: disable=broad-exception-caught
                if run_error is None:
                    raise
                logger.error(
                    "Training cleanup also failed; preserving the original error",
                    exc_info=True,
                )
            finally:
                # Drain training before finishing its run, and finish before
                # main_task can construct the next controller. Tracker/module
                # reference cycles make destructor timing nondeterministic.
                try:
                    self.activity_tracker.flush_e2b_metrics(force=True)
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.warning("Failed to flush final E2B metrics", exc_info=True)
                self.activity_tracker.experiment_logger.close(
                    exit_code=0 if run_error is None and shutdown_succeeded else 1
                )

        logger.info("NorthRL training process finished")

    def _run_training_process(self) -> None:
        """Initialize, start, and monitor training until completion or failure."""
        logger.info("Starting NorthRL training process...")

        logger.info("Initializing trainers...")
        run_coro(self.trainer.initialize_workers())

        # Managed SFT sources need an authenticated catalog lookup before an
        # exact mixing plan (and therefore a resume fingerprint) can exist.
        # The hook is a no-op for every existing dataloader.
        run_coro(self.dataloader.prepare())

        # Epoch-based SFT learns its exact optimizer-step horizon from the
        # model-bound tokenized plan. Initialize the client-side LR schedule
        # before resume so restored completed steps select the next LR exactly.
        self._initialize_lr_schedule_horizon()

        # Load initial checkpoint
        logger.info("Loading initial checkpoint...")
        self._load_initial_checkpoint()

        # Compression and filesystem reads must finish before concurrent E2B
        # requests share the rollout event loop.
        self._prepare_e2b_artifacts()

        # Provision and exercise the shared sampler once before rollout fan-out.
        self._warmup_rollout_service()

        # Schema: `validate.enabled` is the master switch; per-event flags live on schedule.
        validate_cfg = self._config.validate
        run_before_train = validate_cfg.enabled and validate_cfg.schedule.before_train

        if run_before_train:
            # identifier serves as model_tag for weight sync coordination
            logger.info("Starting validation...")
            self._start_validate(_ACTOR_MODEL_TAG)

        # Start all components
        self.trainer.run()
        for worker in self.rollout_workers:
            worker.run()

        self.activity_tracker.update_progress()

        if run_before_train:
            # identifier serves as model_tag for weight sync coordination
            self._end_validate(_ACTOR_MODEL_TAG)

        # Main monitoring loop
        exception_check_interval = self._config.runtime.health.exception_check_interval_s
        next_exception_check_at = time.monotonic()
        memory_log_interval = self._config.runtime.health.memory_log_interval_s
        next_memory_log_at = time.monotonic()
        failure_reason: str | None = None

        logger.info("Starting training loop...")
        while True:
            next_progress_at = self.activity_tracker.update_progress()
            # Check if weight sync is waiting for validation
            # identifier serves as model_tag for weight sync coordination
            identifier = _ACTOR_MODEL_TAG
            if self.weight_sync_controller.is_waiting_for_validation():
                self._run_validate(identifier)

            # Check immediately, then by elapsed time regardless of progress cadence.
            if time.monotonic() >= next_exception_check_at:
                is_system_healthy = self._check_module_exceptions()
                if not is_system_healthy:
                    failure_reason = "System health check detected critical issues"
                    logger.error("%s. Terminating system.", failure_reason)
                    self.activity_tracker.experiment_logger_post(
                        backend="feishu",
                        content=f'"Experiment: {self._config.experiment.project}/{self._config.experiment.name}"'
                        f"training failed, {failure_reason.lower()}. wandb: {self.activity_tracker.experiment_logger.wandb_url} ",
                        title="Error!",
                    )
                    break
                next_exception_check_at = time.monotonic() + exception_check_interval

            if self._check_finish():
                # Close the race where the final worker exception is reported
                # between the periodic health check and the finish check.
                if not self._check_module_exceptions():
                    failure_reason = "System health check detected critical issues"
                    logger.error("%s. Terminating system.", failure_reason)
                    self.activity_tracker.experiment_logger_post(
                        backend="feishu",
                        content=f'"Experiment: {self._config.experiment.project}/{self._config.experiment.name}"'
                        f"training failed, {failure_reason.lower()}. wandb: {self.activity_tracker.experiment_logger.wandb_url} ",
                        title="Error!",
                    )
                    break
                self.activity_tracker.experiment_logger_post(
                    backend="feishu",
                    content=f'"Experiment: {self._config.experiment.project}/{self._config.experiment.name}"'
                    f"training completed, wandb: {self.activity_tracker.experiment_logger.wandb_url} ",
                    title="Complete!",
                )
                break

            if time.monotonic() >= next_memory_log_at:
                # Log driver/cgroup memory so a silent OOM-kill (SIGKILL, no
                # traceback) can be diagnosed from the last on-disk figure.
                logger.info("Driver memory usage: %s", format_memory_usage())
                next_memory_log_at = time.monotonic() + memory_log_interval

            next_check_at = min(next_progress_at, next_exception_check_at, next_memory_log_at)
            time.sleep(max(0.0, min(1.0, next_check_at - time.monotonic())))

        if failure_reason is not None:
            raise RuntimeError(failure_reason)

    def _initialize_lr_schedule_horizon(self) -> None:
        """Resolve the finite update horizon required by an SFT LR schedule."""

        optim = getattr(self._config.algorithm, "optim", None)
        schedule = optim.get("schedule") if optim is not None else None
        if schedule is None:
            return

        configured_steps = self._config.training.total_steps
        if configured_steps is not None:
            total_steps = int(configured_steps)
        else:
            count_windows = getattr(self.dataloader, "count_optimizer_windows", None)
            if not callable(count_windows):
                raise RuntimeError("Configured LR schedule requires a finite SFT window planner")
            total_steps = int(run_coro(count_windows()))

        set_total_steps = getattr(self.trainer, "set_lr_schedule_total_steps", None)
        if not callable(set_total_steps):
            raise RuntimeError("Configured LR schedule is unsupported by the selected trainer")
        set_total_steps(total_steps)

    def _prepare_e2b_artifacts(self) -> None:
        """Prepare shared source and task archives before starting E2B workers."""

        workers = [
            worker for worker in self.rollout_workers if isinstance(worker, E2BRolloutWorker)
        ]
        if not workers:
            return

        from .remote_rollout.e2b_executor import prepare_e2b_artifacts
        from .remote_rollout.execution_profiles import get_e2b_execution_configs

        logger.info("Preparing shared E2B rollout artifacts before starting workers...")
        configs = get_e2b_execution_configs(self._config)
        artifacts_by_profile = {}
        for name, config in configs.items():
            started = time.monotonic()
            logger.info("Preparing E2B artifacts: profile=%s", name or "default")
            artifacts = prepare_e2b_artifacts(config)
            artifacts_by_profile[name] = artifacts
            logger.info(
                "Prepared E2B artifacts: profile=%s source_bytes=%d evaluation_tasks=%d "
                "evaluation_bytes=%d elapsed_seconds=%.1f",
                name or "default",
                len(artifacts.source_archive),
                len(artifacts.evaluation_archives),
                sum(len(archive) for archive in artifacts.evaluation_archives.values()),
                time.monotonic() - started,
            )
        for worker in workers:
            if "" in artifacts_by_profile:
                worker.set_prepared_artifacts(artifacts_by_profile[""])
            else:
                worker.set_prepared_artifacts_by_profile(artifacts_by_profile)

    def _warmup_rollout_service(self) -> None:
        """Warm the shared Weaver sampler once before rollout workers start."""

        if not self.use_weaver or self._config.algorithm.type == "sft":
            return

        logger.info("Warming up rollout sampling service before starting rollout workers...")
        run_coro(self._weaver_service_holder.warmup_sampling())
        logger.info("Rollout sampling service warmup completed")

    def _stop(self, *, close_weaver_first: bool = False):
        """Stop all components gracefully.

        External process termination uses ``close_weaver_first`` so a blocked
        trainer cannot delay releasing server-side Weaver resources. Normal
        completion still drains the trainer before closing the shared client.
        """
        worker_count = len(self.rollout_workers)
        logger.info(
            "Stopping NorthRL training process (rollout_workers=%d, close_weaver_first=%s)",
            worker_count,
            close_weaver_first,
        )

        # Stop accepting new rollout work before any blocking cleanup.
        logger.info("Requesting shutdown from %d rollout worker(s)", worker_count)
        for worker in self.rollout_workers:
            worker.request_stop()
        logger.info("Shutdown requested from all rollout workers")
        rollout_stop_deadline = time.monotonic() + 30.0

        weaver_close_attempted = False
        if close_weaver_first and self.use_weaver:
            logger.info("Closing shared Weaver service client before interrupt cleanup...")
            weaver_close_attempted = True
            try:
                run_coro(self._weaver_service_holder.close())
                logger.info(
                    "WeaverServiceHolder close coroutine completed during interrupt cleanup"
                )
            except Exception:  # pylint: disable=broad-exception-caught
                logger.error(
                    "WeaverServiceHolder close coroutine failed during interrupt cleanup",
                    exc_info=True,
                )

        # Do not cancel an in-flight optimizer step; BaseTrainer.stop() drains
        # it before returning.
        logger.info("Stopping trainer; waiting for active training work to finish")
        self.trainer.stop(
            on_wait=self.activity_tracker.update_progress,
            wait_interval=self._config.runtime.progress.update_interval_s,
        )
        logger.info("Trainer stop completed")

        # All workers have been stopping concurrently since the broadcast.
        # Give the whole group one shared timeout budget instead of waiting up
        # to 30 seconds independently for every worker.
        logger.info(
            "Waiting for %d rollout worker(s) to stop (shared timeout: 30.0s)", worker_count
        )
        for worker_index, worker in enumerate(self.rollout_workers):
            remaining = max(0.0, rollout_stop_deadline - time.monotonic())
            worker_name = getattr(worker, "_module_name", f"rollout_worker[{worker_index}]")
            logger.info(
                "Waiting for rollout worker %s to stop (remaining timeout: %.1fs)",
                worker_name,
                remaining,
            )
            worker.wait_stopped(
                timeout=remaining,
                on_wait=self.activity_tracker.update_progress,
                wait_interval=self._config.runtime.progress.update_interval_s,
            )
            logger.info("Rollout worker %s stop wait completed", worker_name)
        logger.info("All rollout worker stop waits completed")

        self.activity_tracker.update_progress()

        # Wait for all activities to complete with timeout
        logger.info("Waiting for all activities to complete...")
        logger.info(f"Current status: {self.activity_tracker.get_running_status_summary()}")

        if self.activity_tracker.wait_quiescent(timeout=30.0):
            logger.info("All activities completed gracefully")
        else:
            logger.warning("Timeout reached - some activities may not have completed")
            logger.warning(
                f"Remaining activities: {self.activity_tracker.get_running_status_summary()}"
            )
            logger.warning("Workers should eventually stop when they check their stop_event")

        try:
            if self.use_weaver and not weaver_close_attempted:
                logger.info("Closing shared Weaver service client...")
                try:
                    run_coro(self._weaver_service_holder.close())
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "WeaverServiceHolder close coroutine failed during normal cleanup",
                        exc_info=True,
                    )
                    raise
                logger.info("WeaverServiceHolder close coroutine completed during normal cleanup")
        finally:
            logger.info("Closing dataloader")
            self.dataloader.close()
            logger.info("Dataloader closed")
        logger.info("NorthRL training process stopped")

    # ---- Load checkpoint functions ----
    def _load_initial_checkpoint(self):
        """Load initial checkpoint. Only the weaver backend has a checkpoint
        path; backend.type=="mock" is a no-op."""
        if self.use_weaver:
            self._load_initial_checkpoint_weaver()

    def _load_initial_checkpoint_weaver(self):
        """Resume Weaver state and, for SFT, its exact committed source cursor."""
        if self._config.training.resume.mode == "disable":
            logger.info("Resume mode is disabled, training from scratch (weaver)")
            return

        is_sft = self._config.algorithm.type == "sft"

        resume_mode = self._config.training.resume.mode
        resume_entry: dict[str, Any] | None = None
        if resume_mode == "auto":
            resume_entry = self._find_latest_weaver_checkpoint_entry()
            if resume_entry is not None:
                resume_path = str(resume_entry["path"])
                logger.info("Auto resume: latest Weaver checkpoint -> %s", resume_path)
            elif self._config.training.resume.path:
                resume_path = self._config.training.resume.path
                logger.info(
                    "No Weaver checkpoint index entry found for auto resume; "
                    "falling back to configured checkpoint -> %s",
                    resume_path,
                )
            else:
                logger.info(
                    "No Weaver checkpoint index entry found for auto resume, "
                    "training from scratch"
                )
                return
        elif resume_mode == "from_path":
            resume_path = self._config.training.resume.path
            if not resume_path:
                raise ValueError(
                    "resume.resume_path must be specified for Weaver resume "
                    "(use the path returned by save_weight_and_optimizer in a previous run)"
                )
            resume_entry = self._find_weaver_checkpoint_entry(resume_path)
        else:
            raise ValueError(f"Unsupported resume mode: {resume_mode}")

        if is_sft:
            if resume_entry is None or not isinstance(resume_entry.get("dataloader_state"), dict):
                raise ValueError(
                    "SFT resume requires a matching Weaver checkpoint index entry "
                    "with dataloader_state"
                )
            load_state_dict: Any = getattr(self.dataloader, "load_state_dict", None)
            if not callable(load_state_dict):
                raise RuntimeError("SFT resume requires a cursor-aware streaming dataloader")
            load_state_dict(resume_entry["dataloader_state"])
            train_step = int(resume_entry["step"])
        else:
            if resume_entry is not None:
                train_step = int(resume_entry["step"])
            else:
                train_step = self._extract_global_step_from_path(resume_path)

        logger.info("Loading Weaver checkpoint from: %s", resume_path)
        run_coro(self._weaver_service_holder.load_state_with_optimizer(resume_path))
        self.trainer.set_train_step(train_step)
        self.activity_tracker.set_training_step(train_step)
        if is_sft:
            logger.info(
                "Resumed SFT at train_step=%d with an exact committed data cursor",
                train_step,
            )
            return

        if train_step > 0:
            logger.info("Resuming from training step: %d", train_step)
        else:
            logger.warning(
                "Could not extract global step from Weaver checkpoint path, "
                "starting step counter from 0"
            )
        self._maybe_skip_dataloader(train_step)

        # Non-SFT pipelines may own a rollout sampler which must follow the
        # restored training weights.
        weight_version = format_weight_version(_ACTOR_MODEL_TAG, train_step)
        sampling_path = run_coro(
            self._weaver_service_holder.save_weights_for_sampler(name=weight_version)
        )
        # Weight-sync versions are process-local. The restored checkpoint is
        # the new sampler baseline for this process.
        logger.info("Syncing resumed Weaver checkpoint weights to rollout service...")
        self.weight_sync_controller.sync_weight_to_rollout_service(
            _ACTOR_MODEL_TAG, sampling_path, weight_version
        )

    @staticmethod
    def _extract_global_step_from_path(path: str) -> int:
        """Best-effort recovery of the training step from a checkpoint path."""
        if "global_step_" not in path:
            return 0
        try:
            tail = path.split("global_step_")[-1]
            # Accept both ``.../global_step_500`` and ``.../global_step_500/...``.
            token = tail.split("/")[0]
            return int(token)
        except ValueError:
            return 0

    def _read_weaver_checkpoint_entries(self) -> list[dict[str, Any]]:
        """Read valid local records that bind Weaver state to client state."""
        import json

        checkpoint_dir = self._config.training.save.dir or ""
        if not checkpoint_dir:
            logger.warning("training.save.dir is empty; cannot discover Weaver checkpoints")
            return []
        index_path = os.path.join(checkpoint_dir, "weaver_checkpoints.jsonl")
        if not os.path.exists(index_path):
            logger.info("Weaver checkpoint index does not exist: %s", index_path)
            return []

        entries: list[dict[str, Any]] = []
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        if not isinstance(raw, dict):
                            raise TypeError("checkpoint index entry must be an object")
                        path = raw.get("path")
                        step = int(raw.get("step", 0))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        logger.warning(
                            "Skipping malformed line %d in %s: %s",
                            line_no,
                            index_path,
                            exc,
                        )
                        continue
                    if not path:
                        continue
                    entry = dict(raw)
                    entry["path"] = str(path)
                    entry["step"] = step
                    entries.append(entry)
        except OSError as exc:
            logger.error("Failed to read Weaver checkpoint index %s: %s", index_path, exc)
            return []
        return entries

    def _find_latest_weaver_checkpoint_entry(self) -> dict[str, Any] | None:
        entries = self._read_weaver_checkpoint_entries()
        if not entries:
            return None
        _, entry = max(
            enumerate(entries),
            key=lambda indexed: (indexed[1]["step"], indexed[0]),
        )
        logger.info(
            "Found latest Weaver checkpoint in index: step=%d path=%s",
            entry["step"],
            entry["path"],
        )
        return entry

    def _find_weaver_checkpoint_entry(self, path: str) -> dict[str, Any] | None:
        matches = [
            entry for entry in self._read_weaver_checkpoint_entries() if entry["path"] == path
        ]
        return matches[-1] if matches else None

    def _find_latest_weaver_checkpoint(self) -> str | None:
        """Compatibility helper returning only the latest remote path."""
        entry = self._find_latest_weaver_checkpoint_entry()
        return str(entry["path"]) if entry is not None else None

    def _maybe_skip_dataloader(self, global_step: int) -> None:
        """Skip already-consumed batches in the dataloader after a resume."""
        resume_dataloader = self._config.training.resume.skip_dataloader
        if resume_dataloader and global_step > 0:
            logger.info(
                f"Resuming dataloader: skipping first {global_step} batches "
                f"(corresponding to {global_step} training steps)"
            )
            try:
                self.dataloader.skip_batches(global_step)
                logger.info(f"Successfully resumed dataloader from step {global_step}")
            except Exception as e:
                logger.error(
                    f"Failed to skip batches in dataloader: {e}. "
                    f"Training will continue but may process duplicate data.",
                    exc_info=True,
                )
        elif not resume_dataloader:
            logger.info(
                "Dataloader resume is disabled (resume.resume_dataloader=false). "
                "Dataloader will start from the beginning - data may be processed again."
            )
        else:
            logger.info("No batches to skip (global_step=0)")

    # ---- Check functions ----
    def _check_finish(self) -> bool:
        """
        Check if the training process should be stopped.

        Returns:
            True if training should be stopped, False otherwise
        """
        # Hard stop: reached or exceeded max training steps
        total_steps = self._config.training.total_steps
        steps_reached = total_steps is not None and self.trainer.get_train_step() >= total_steps
        if steps_reached:
            if self.weight_sync_controller.is_waiting_for_validation():
                logger.info("Finish check: final validation is still pending")
                return False
            logger.info("Finish check: steps_reached=True -> stopping")
            return True

        # Quiescence stop: only when all sources/sinks are drained and no work is in-flight
        dataloader_done = self.dataloader.is_finished()
        # Rollout workers produce into the pool, so observe them idle before
        # taking the pool snapshot. The trainer consumes from the pool, so
        # confirm activity again after that snapshot.
        tracker_idle_before_pool = self.activity_tracker.is_quiescent()
        rollout_pool_empty = self.trajectory_pool.is_empty()
        tracker_idle_after_pool = self.activity_tracker.is_quiescent()
        validation_pending = self.weight_sync_controller.is_waiting_for_validation()

        logger.debug(
            f"Finish check: rollout_pool_empty={rollout_pool_empty}, "
            f"dataloader_done={dataloader_done}, "
            f"tracker_idle_before_pool={tracker_idle_before_pool}, "
            f"tracker_idle_after_pool={tracker_idle_after_pool}, "
            f"validation_pending={validation_pending}"
        )

        should_stop = (
            dataloader_done
            and tracker_idle_before_pool
            and rollout_pool_empty
            and tracker_idle_after_pool
            and not validation_pending
        )
        if should_stop:
            logger.info("Finish check: input exhausted and pipeline drained -> stopping")
        return should_stop

    def _check_module_exceptions(self) -> bool:
        """
        Check system health based on exceptions and return whether the system is healthy.

        Returns:
            bool: True if system is healthy, False if system has critical issues
        """
        policy = self._config.runtime.health.exception_policy

        # Get recent errors from the error reporter
        try:
            health_status = self.activity_tracker.get_error_health_status()
            error_count = health_status.get("error_level_count", 0)

            # Apply policy to determine system health
            if policy == "stop_on_error" and error_count > 0:
                logger.error(
                    f"System health check: policy '{policy}' detected {error_count} errors. System is unhealthy."
                )
                return False
            elif policy == "stop_on_critical" and error_count > 0:
                # For now, treat any error as critical - can be extended later
                # For example, if only one of hundreds of rollout workers died and the system is still healthy, we should continue the system.
                logger.error(
                    f"System health check: policy '{policy}' detected {error_count} critical errors. System is unhealthy."
                )
                return False
            elif policy == "continue":
                # Always consider system healthy regardless of exceptions
                if error_count > 0:
                    logger.warning(
                        f"Detected {error_count} errors, but policy is 'continue'. System remains healthy."
                    )
                return True

            return True

        except Exception as e:
            logger.error(f"Error while checking system health: {e}")
            return False  # If we can't check health, assume unhealthy

    def _start_validate(self, model_tag: ModelTag):
        """Start validation"""
        current_version = self.weight_sync_controller.get_rollout_model_version(model_tag)
        logger.info(
            "Starting validation: global_step=%s sync_version=%s "
            "ready_groups=%s pool_segments=%s active_workers=%s",
            self.activity_tracker.get_training_step(),
            current_version,
            self.trajectory_pool.get_progress_info().get("ready_groups"),
            self.trajectory_pool.get_progress_info().get("ready_segments"),
            self.activity_tracker.is_rollout_worker_quiescent(),
        )

        # Refill validation data before workers are allowed to claim from it.
        self.validate_dataloader.reset()

        # Switch workers to validation mode
        for worker in self.rollout_workers:
            worker.begin_validate()
        logger.info("Validation mode enabled for %d rollout workers", len(self.rollout_workers))

    def _end_validate(self, model_tag: ModelTag):
        """End validation"""
        # Keep progress visible while validation uses the existing rollout workers.
        last_validation_log = 0.0
        while True:
            validation_complete = self.validator.is_complete()
            # Check after completion: the last worker can fail and become idle
            # without producing a validation result.
            if not self._check_module_exceptions():
                raise RuntimeError("System health check detected critical issues during validation")
            if validation_complete:
                break
            now = time.monotonic()
            if now - last_validation_log >= 10.0:
                last_validation_log = now
                logger.info(
                    "Validation waiting: dataloader_finished=%s can_return_item=%s workers_quiescent=%s",
                    self.validate_dataloader.is_finished(),
                    self.validate_dataloader.can_return_item(),
                    self.activity_tracker.is_rollout_worker_quiescent(),
                )
            next_progress_at = self.activity_tracker.update_progress()
            time.sleep(max(0.0, min(1.0, next_progress_at - time.monotonic())))
        logger.info("Validator is complete")

        self.validator.compute_and_log_metrics()

        # Prepare training data before workers can claim it. Otherwise, after
        # initial validation they may auto-load and drain the first batch before
        # this release, which would incorrectly expose a second rollout batch.
        self.weight_sync_controller.end_validate(model_tag)

        # Switch back to training mode only after the release decision is complete.
        for worker in self.rollout_workers:
            worker.end_validate()

        logger.info("Validation complete")

    # ---- Validation functions ----
    def _run_validate(self, model_tag: ModelTag):
        """Run validation right after a weight sync"""
        self._start_validate(model_tag)
        self._end_validate(model_tag)

    # ---- Module initialization functions ----
    def _load_custom_module(
        self, module_path: str, class_name: str, module_type: str = "module"
    ) -> type:
        """
        Dynamically load a custom module class from a recipe directory.

        Args:
            module_path: Path to the Python module (e.g., "rollout_workspace/my_custom.py")
            class_name: Name of the class to load
            module_type: Type of module being loaded (for logging/error messages)

        Returns:
            The loaded class
        """
        import importlib.util
        import sys

        # Resolve the full path relative to config file
        config_file_path = getattr(self._config, "_config_file_path", None)
        if not config_file_path:
            raise ValueError(
                "Config file path not available for resolving custom module path. "
                "Ensure the config has _config_file_path attribute set."
            )

        from .utils.path_utils import resolve_path_from_config

        full_path_str = resolve_path_from_config(module_path, config_file_path)

        if full_path_str is None:
            raise ValueError(f"Could not resolve module path: {module_path}")

        full_path = Path(full_path_str)

        if not full_path.exists():
            raise FileNotFoundError(f"Custom {module_type} module not found: {full_path}")

        # Create a unique module name
        module_name = f"northrl.custom_{module_type}.{full_path.stem}"

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {full_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the class
        if not hasattr(module, class_name):
            raise AttributeError(f"Module {module_path} does not have class {class_name}")

        loaded_class = getattr(module, class_name)
        logger.info(f"Loaded custom {module_type}: {class_name} from {module_path}")

        return loaded_class

    def _get_module_class_for_role(self, role: NorthRLRole) -> type:
        """Resolve the module class to instantiate for a given role.

        New schema: each role pulls its `type` selector from the
        canonical SSoT path (no `type` field re-injected at module config
        level). See CONFIG_REFACTOR.md for the registry rationale.

        Fixed-implementation roles (default-only):
          DATA_LOADER, VALIDATE_DATALOADER, TRAJECTORY_POOL,
          WEIGHT_SYNC_CONTROLLER, VALIDATOR
        Roles with a registry:
          ROLLOUT_WORKER   ← internal SFT bridge when algorithm.type == "sft";
                            otherwise config.rollout.type
                            (+ optional config.rollout.custom_module_path)
          TRAINER          ← config.algorithm.type (+ backend dispatch)
        """
        if role == NorthRLRole.DATA_LOADER:
            if self._config.algorithm.type == "sft":
                return SFTStreamingDataLoader
            if getattr(self._config.data.train, "sources", None) is not None:
                return MixtureDataLoader
            return TorchDataLoader
        if role == NorthRLRole.VALIDATE_DATALOADER:
            return MultiDatasetValidationDataLoader
        if role == NorthRLRole.TRAJECTORY_POOL:
            return TrajectoryPool
        if role == NorthRLRole.WEIGHT_SYNC_CONTROLLER:
            return WeightSyncController
        if role == NorthRLRole.VALIDATOR:
            return Validator

        if role == NorthRLRole.ROLLOUT_WORKER:
            if self._config.algorithm.type == "sft":
                return SFTRolloutWorker
            rollout_cfg = self._config.rollout
            if rollout_cfg.type is None:
                raise ValueError("rollout.type is required unless algorithm.type='sft'")
            custom_path = rollout_cfg.get("custom_module_path", None)
            if custom_path:
                custom_class_name = rollout_cfg.get("custom_class_name", "CustomRolloutWorker")
                return self._load_custom_module(custom_path, custom_class_name, "rollout_worker")
            rollout_type = rollout_cfg.type
            rollout_registry: dict[str, type] = {
                "mock": MockRolloutWorker,
                "simple": SimpleRolloutWorker,
                "e2b": E2BRolloutWorker,
                "single_turn": SingleTurnRolloutWorker,
            }
            if rollout_type not in rollout_registry:
                raise ValueError(
                    f"Unsupported rollout.type: {rollout_type!r}. "
                    f"Available: {sorted(rollout_registry)}"
                )
            return rollout_registry[rollout_type]

        if role == NorthRLRole.TRAINER:
            algo_type = self._config.algorithm.type
            trainer_registry: dict[str, type] = {
                "grpo": RemoteApiGrpoTrainer,
                "sft": RemoteApiCrossEntropyTrainer,
                "opd": RemoteApiOpdTrainer,
            }
            # Router replay (algorithm.router_replay.enabled) is handled inside
            # RemoteApiGrpoTrainer, gated on the flag — no separate trainer class.
            if algo_type not in trainer_registry:
                raise ValueError(
                    f"Unsupported algorithm.type: {algo_type!r}. "
                    f"Available: {sorted(trainer_registry)}"
                )
            return trainer_registry[algo_type]

        raise ValueError(f"Unsupported role: {role}")

    def _init_module_for_role(
        self, role: NorthRLRole, config: DictConfig, num_workers: int, *args, **kwargs
    ) -> list[Any]:
        """
        Generic local module instantiation.

        Args:
            role: The NorthRL role
            config: The configuration for the module
            num_workers: The number of workers to initialize
            *args, **kwargs: Arguments to pass to the module constructor
        Returns:
            list of module instances
        """
        module_class = self._get_module_class_for_role(role)

        def init_worker(_):
            return module_class(config, *args, **kwargs)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            ret = list(executor.map(init_worker, range(num_workers)))

        for i, module in enumerate(ret):
            proxy = ActivityTrackerProxy(self.activity_tracker)
            module.set_activity_tracker(proxy)
            module_name = role.value + f"-{i}"
            module.set_module_name(module_name)
            self.activity_tracker.register_module(
                module_name=module_name,
                module_ref=ret,
                is_rollout_worker=(role == NorthRLRole.ROLLOUT_WORKER),
            )
        logger.info(f"Module {role.value} initialization completed successfully")

        return ret

    def _init_rollout_worker_inference_clients(self):
        """Initialize inference service clients for all rollout workers."""
        backend = self._config.backend.type
        for worker in self.rollout_workers:
            if backend == "weaver":
                worker.init_inference_service_client(self._weaver_service_holder)
            else:
                # backend.type == "mock" (only other Literal value); no holder needed.
                worker.init_inference_service_client()

        logger.info(
            f"Initialized inference service clients for {len(self.rollout_workers)} rollout workers"
        )

    def _derive_trajectory_pool_config(self) -> DictConfig:
        """Derive pool delivery units from the public data and algorithm config.

        Grouped RL takes data.train.batch_size complete groups, each containing
        rollout_repeat_n trajectories. SFT uses a simple store and delivers the
        actual trajectory count of its completed token-budget input round.
        Fully async requires B eligible groups; other modes count completed groups.
        """
        data_train = self._config.data.train
        algo_type = self._config.algorithm.type
        is_sft = algo_type == "sft"
        repeat_n = int(data_train.rollout_repeat_n or 1)

        key_list = ["group_id"] if algo_type in ("grpo", "opd") else []

        return DictConfig(
            {
                "type": "default",
                "groups_per_batch": data_train.batch_size,
                "group_order": data_train.get("group_order", "fifo"),
                "require_trainable_groups": bool(key_list)
                and self._config.training.weight_sync.mode == "fully-async",
                "need_sampling": not is_sft,
                "group_size": repeat_n,
                "key_list": key_list,
            }
        )

    def _derive_train_dataloader_config(self) -> DictConfig:
        """Project the new-schema fields needed by the selected train loader.

        Schema enforces all the upstream invariants; here we just pluck the
        ones the dataloader reads and add the two derived bits:
          * keep_batch_order — SFT or a synchronizing weight mode
          * seed             — SFT's optional data seed, otherwise experiment.seed
          * cycle            — step-based non-SFT training continues across epochs
        See CONFIG_REFACTOR.md derivation table.
        """
        data = self._config.data
        train = data.train
        actor = self._config.models.actor
        weight_sync_mode = self._config.training.weight_sync.mode
        is_sft = self._config.algorithm.type == "sft"

        return DictConfig(
            {
                "files": list(train.files) if train.files is not None else None,
                "sources": (
                    [dict(source) for source in train.sources]
                    if train.sources is not None
                    else None
                ),
                "total_samples": train.total_samples,
                "managed_variant_ready_timeout_seconds": (
                    train.managed_variant_ready_timeout_seconds
                ),
                "managed_variant_poll_initial_seconds": (
                    train.managed_variant_poll_initial_seconds
                ),
                "managed_variant_poll_max_seconds": train.managed_variant_poll_max_seconds,
                "batch_size": train.batch_size,
                "rollout_repeat_n": int(train.rollout_repeat_n or 1),
                "prompt_key": train.prompt_key,
                "filter_prompts": train.filter_prompts,
                "shuffle": train.shuffle,
                "drop_last": train.drop_last,
                "window_samples": train.get("window_samples"),
                "max_prompt_length": data.max_prompt_length,
                "tokenizer": actor.tokenizer,
                "is_sft": is_sft,
                "render_chat": train.render_chat,
                "max_sequence_length": data.max_sequence_length,
                "config_file_path": getattr(self._config, "_config_file_path", None),
                "epochs": self._config.training.epochs if is_sft else None,
                "seed": (
                    train.seed
                    if is_sft and train.seed is not None
                    else self._config.experiment.seed
                ),
                # Non-SFT training is step-based, so the source must keep
                # producing deterministically reshuffled epochs until
                # training.total_steps is reached. This is internal derived
                # state, not a user-facing data.train option.
                "cycle": not is_sft and self._config.training.total_steps is not None,
                "keep_batch_order": is_sft or weight_sync_mode in ("sync", "batch-async"),
            }
        )

    def _derive_validate_dataloader_config(self) -> DictConfig:
        """Build the validate-side multi-dataset config.

        validate.data.datasets is the only user-writable form (see §14).
        Per-dataset shape we emit:
          { <name>: {files, batch_size, prompt_key, max_prompt_length, tokenizer,
                     shuffle: false, drop_last: false, filter_prompts: false} }
        Three constants are written here (schema forbids overriding them).
        """
        data = self._config.data
        actor = self._config.models.actor
        train = data.train
        validate_data = self._config.validate.data
        if validate_data is None:
            return DictConfig({"datasets": {}})

        per_dataset: dict[str, dict[str, Any]] = {}
        for name, ds in validate_data.datasets.items():
            per_dataset[name] = {
                "files": list(ds.files),
                "batch_size": ds.batch_size,
                # Inherited / constant fields (validate is reproducible, full-coverage, raw).
                "prompt_key": train.prompt_key,
                "max_prompt_length": data.max_prompt_length,
                "tokenizer": actor.tokenizer,
                "shuffle": False,
                "drop_last": False,
                "filter_prompts": False,
                "rollout_repeat_n": 1,  # validation never uses rollout_repeat_n
                # k for grouped pass@k
                "validation_repeat_n": validate_data.validation_repeat_n,
                "seed": self._config.experiment.seed,
            }
        return DictConfig({"datasets": per_dataset})

    def _init_modules(self):
        """Initialize all modules locally.

        New schema: every module receives the full root NorthRLConfig and reads
        whichever fields it needs (see CONFIG_REFACTOR.md). The old "inject
        service.* subtrees + slice per-module sub-configs" pipeline is gone —
        each module's own constructor is the single boundary that knows what
        it consumes.

        The one exception is TRAJECTORY_POOL, which still takes a synthesized
        sub-config (its fields are fully derived; see
        :meth:`_derive_trajectory_pool_config`).
        """
        logger.info("Initializing modules locally...")

        config_file_path = getattr(self._config, "_config_file_path", None)
        if not config_file_path:
            raise ValueError(
                "Config file path not available. Ensure the config has "
                "_config_file_path attribute set."
            )

        trajectory_pool_cfg = self._derive_trajectory_pool_config()
        train_dataloader_cfg = self._derive_train_dataloader_config()
        validate_dataloader_cfg = self._derive_validate_dataloader_config()

        # Most modules receive the root config; the four exceptions
        # (trajectory_pool, both dataloaders) get derived sub-configs because
        # the field set they need is narrow and partly derived.
        self.dataloader = self._init_module_for_role(
            NorthRLRole.DATA_LOADER,
            train_dataloader_cfg,
            num_workers=1,
            is_validate=False,
        )[0]
        self.trajectory_pool = self._init_module_for_role(
            NorthRLRole.TRAJECTORY_POOL,
            trajectory_pool_cfg,
            num_workers=1,
        )[0]
        self.weight_sync_controller = self._init_module_for_role(
            NorthRLRole.WEIGHT_SYNC_CONTROLLER,
            self._config,
            num_workers=1,
        )[0]
        self.trainer = self._init_module_for_role(
            NorthRLRole.TRAINER,
            self._config,
            num_workers=1,
        )[0]

        self.rollout_workers = self._init_module_for_role(
            NorthRLRole.ROLLOUT_WORKER,
            self._config,
            num_workers=self._config.rollout.num_workers,
        )
        self.validate_dataloader = self._init_module_for_role(
            NorthRLRole.VALIDATE_DATALOADER,
            validate_dataloader_cfg,
            num_workers=1,
            is_validate=True,
        )[0]
        self.validator = self._init_module_for_role(
            NorthRLRole.VALIDATOR,
            self._config,
            num_workers=1,
        )[0]

        # Initialize inference service clients
        self._init_rollout_worker_inference_clients()

        # Set module references
        self._set_module_references()

        logger.info("Module initialization completed successfully")

    def _set_module_references(self):
        """Set module references consistently across local modules."""

        for worker in self.rollout_workers:
            worker.set_module_references(
                trajectory_pool=self.trajectory_pool,
                dataloader=self.dataloader,
                weight_sync_controller=self.weight_sync_controller,
                validate_dataloader=self.validate_dataloader,
                validator=self.validator,
            )
        self.trajectory_pool.set_module_references(
            dataloader=self.dataloader,
            weight_sync_controller=self.weight_sync_controller,
        )
        self.trainer.set_module_references(
            trajectory_pool=self.trajectory_pool,
            weight_sync_controller=self.weight_sync_controller,
            dataloader=self.dataloader,
        )
        self.weight_sync_controller.set_module_references(
            dataloader=self.dataloader,
            trajectory_pool=self.trajectory_pool,
        )
        self.validator.set_module_references(
            validate_dataloader=self.validate_dataloader,
        )
        if self.use_weaver:
            self.weight_sync_controller.set_weaver_service_holder(self._weaver_service_holder)
            set_dataloader_service_holder = getattr(self.dataloader, "set_service_holder", None)
            if callable(set_dataloader_service_holder):
                set_dataloader_service_holder(self._weaver_service_holder)
            # Set service holder for RemoteApiTrainer
            if hasattr(self.trainer, "set_service_holder"):
                self.trainer.set_service_holder(self._weaver_service_holder)

    def _init_activity_tracker(self):
        """Initialize activity tracker and error reporter."""
        self.activity_tracker = ActivityTracker(max_errors=1000, config=self._config)

    def _init_weaver_service(self):
        """Initialize Weaver service holder for Weaver backend"""
        from .weaver import WeaverServiceHolder

        # New schema: connection config (base_url, api_key) collapsed into the
        # top-level `backend` block; legacy `service.weaver_service.*` block is
        # gone. ENV still wins (see CONFIG_REFACTOR.md §11).
        backend_cfg = self._config.backend
        scheduling_policy = backend_cfg.get("scheduling_policy")
        if scheduling_policy is not None:
            scheduling_policy = OmegaConf.to_container(scheduling_policy, resolve=True)

        base_url = os.environ.get("WEAVER_BASE_URL", None)
        if base_url is None:
            base_url = backend_cfg.base_url

        weaver_api_key = os.environ.get("WEAVER_API_KEY", None)
        if weaver_api_key is None:
            weaver_api_key = backend_cfg.api_key

        # Report error if weaver_api_key is not provided in either environment or config
        if not weaver_api_key:
            error_msg = (
                "WEAVER_API_KEY is required but not found in environment variables or config"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Set weaver_api_key in environment for downstream usage
        os.environ["WEAVER_API_KEY"] = weaver_api_key

        # New schema: model spec lives under models.actor; lora rank moved off
        # weaver_service block onto models.actor.lora. See CONFIG_REFACTOR.md.
        actor_model = self._config.models.actor
        base_model = actor_model.name
        tokenizer_path = actor_model.tokenizer
        lora_rank = actor_model.lora.rank if actor_model.lora and actor_model.lora.enabled else 32

        # training_mode derived from models.actor.lora.enabled (see CONFIG_REFACTOR.md
        # derivation table). Weaver's create_model expects "full_ft" (not "full")
        # for full fine-tuning — every full-FT recipe uses that literal.
        training_mode = "lora" if (actor_model.lora and actor_model.lora.enabled) else "full_ft"

        # SFT tokenizes training data directly and does not provision a sampler.
        # Derived from algorithm.type per CONFIG_REFACTOR.md §6
        need_sampling = self._config.algorithm.type != "sft"

        # New schema: sampler weight TTL moved off legacy weaver_service.* onto
        # training.weight_sync.sampler_ttl_s (see CONFIG_REFACTOR.md §11/§G).
        sampler_ttl_seconds = self._config.training.weight_sync.sampler_ttl_s

        max_concurrent_requests = self._config.rollout.get("max_concurrent_requests", None)
        if max_concurrent_requests is None:
            max_concurrent_requests = self._config.rollout.num_workers

        logger.info(
            "Initializing WeaverServiceHolder: base_model=%s, rank=%s, "
            "max_concurrent_requests=%s",
            base_model,
            lora_rank,
            max_concurrent_requests,
        )

        self._weaver_service_holder = WeaverServiceHolder(
            base_model=base_model,
            lora_rank=lora_rank,
            base_url=base_url,
            tokenizer_path=tokenizer_path,
            training_mode=training_mode,
            need_sampling=need_sampling,
            sampler_ttl_seconds=sampler_ttl_seconds,
            max_concurrent_requests=max_concurrent_requests,
            allow_weight_version_mismatch=self._config.training.weight_sync.mode == "fully-async",
            max_sequence_length=self._config.data.max_sequence_length,
            performance_tier=backend_cfg.get("performance_tier", None),
            training_max_sequence_length=self._config.data.max_sequence_length,
            scheduling_policy=scheduling_policy,
            metrics_store=dict(backend_cfg.get("metrics_store", {})),
            metric_observer=self.activity_tracker.experiment_logger.queue_weaver_metrics,
            gateway_publication_timeout_seconds=(
                self._config.rollout.execution.get("gateway_publication_timeout_seconds", 300)
                if self._config.rollout.type == "e2b"
                else None
            ),
        )

        # Configure sampling flags from explicit mask retention and training needs.
        sampling_flags = get_derived_sampling_flags(self._config)
        return_sampling_mask = sampling_flags["return_sampling_mask"]
        return_old_logprob = sampling_flags["return_old_logprob"]
        return_moe_topk_indices = sampling_flags["return_moe_topk_indices"]
        if return_sampling_mask or return_old_logprob or return_moe_topk_indices:
            logger.info(
                f"Configuring sampling flags: return_sampling_mask={return_sampling_mask}, "
                f"return_old_logprob={return_old_logprob}, "
                f"return_moe_topk_indices={return_moe_topk_indices}"
            )
            sampling_flag_kwargs = {
                "return_sampling_mask": return_sampling_mask,
                "return_old_logprob": return_old_logprob,
                "return_moe_topk_indices": return_moe_topk_indices,
            }
            run_coro(self._weaver_service_holder.configure_sampling_flags(**sampling_flag_kwargs))

        logger.info("WeaverServiceHolder initialized successfully")
