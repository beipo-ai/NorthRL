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
RemoteApiTrainer - Base trainer for the Weaver training service
Works with complete trajectories and explicitly ordered algorithm stages.
"""

import asyncio
import copy
import hashlib
import json
import logging
import math
import random
import time
from abc import abstractmethod
from array import array
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError as FutureCancelledError
from typing import Any, Callable, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..algorithm.advantage import build_estimator, build_postprocessor
from ..algorithm.advantage.tool_error_rebalance import clear_postprocessing_cache
from ..algorithm.gspo_loss import attach_gspo_trajectory_metadata
from ..algorithm.mis import compute_mis_weights, compute_mismatch_metrics
from ..northrl_types import Trajectory
from ..utils.async_utils import await_coro
from ..utils.config_schema import AlgorithmConfig
from ..utils.data_dumper import DataDumper, get_data_dumper
from ..utils.finetune_service_utils import convert_trajectories_to_datums
from ..utils.logging_utils import log_rollout_metrics
from ..utils.lr_scheduling import cosine_learning_rate
from ..utils.metric_catalog import normalize_step_metrics
from ..utils.prompt_mean import prompt_group_key
from ..utils.request_pipeline import run_request_pipeline
from ..utils.service_metrics import aggregate_service_metrics
from ..utils.step_metrics import datum_response_token_count, group_key, log_trainable_metrics
from ..utils.training_timing import sum_training_stage_metrics, training_stage_metrics
from ..utils.trajectory_alignment import (
    align_logprobs_to_target,
    align_response_sampling_mask_to_target,
    align_target_logprobs_to_tokens,
    validate_loss_mask_for_target_alignment,
)
from ..utils.trajectory_sequences import iter_training_sequences
from ..weaver.custom_losses import CustomLossSpec, parse_custom_loss_spec
from ..weaver.loss_adapter import build_loss_adapter, normalize_advantages_for_sum
from ..weaver.weaver_service_holder import ForwardLogprobsResult, WeaverServiceHolder
from ..weight_sync.versioning import ACTOR_MODEL_TAG, format_weight_version
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

# Keep persisted padding IDs stable across the package rename.
_PADDING_DATUM_ID_DOMAIN = "nexrl-mini-batch-padding-v1"


def _padding_occurrence_datum_id(datum_id: str, *, mini_batch_index: int, slot_index: int) -> str:
    """Derive a stable ID for one repeated mini-batch padding occurrence."""

    payload = json.dumps(
        {
            "domain": _PADDING_DATUM_ID_DOMAIN,
            "datum_id": datum_id,
            "mini_batch_index": mini_batch_index,
            "slot_index": slot_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"nexrl-pad-{hashlib.sha256(payload).hexdigest()}"


class RemoteApiTrainer(BaseTrainer):
    """
    Base trainer implementation for the Weaver training service.

    This class provides common training logic for API-backed trainers
    that use Weaver for distributed training.

    Key features:
    - Works directly with trajectories (no Batch conversion)
    - Step-wide estimator and postprocessor calls before optimizer batching
    - Uses service holder pattern for forward_backward and optim_step
    - No padding removal needed (service handles variable-length sequences)

    Derived classes should:
    - Set _service_holder in their set_service_holder() method
    - Optionally override methods for custom behavior
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the remote API trainer.

        Args:
            config: Root NorthRLConfig. Trainer reads `algorithm.*`, `training.*`,
                `runtime.debug.*` from it directly (new 9-block schema).
        """
        # Direct trainer construction validates the same canonical components
        # as the YAML entry point. Keep the caller's config untouched.
        algorithm_input = OmegaConf.to_container(config.algorithm, resolve=True)
        assert isinstance(algorithm_input, dict)
        self._algorithm_config = AlgorithmConfig.model_validate(algorithm_input)
        super().__init__(config)

        # Service holder will be set via set_service_holder() in derived class
        self._service_holder: WeaverServiceHolder | None = None

        # ---- algorithm.* hyperparameters ----
        algorithm = self._algorithm_config
        optim = algorithm.optim
        # Resolve concrete implementations once. The step below calls these
        # objects directly; importing an algorithm module never registers work.
        self._advantage_estimator = (
            build_estimator(algorithm.advantage.estimator)
            if self._uses_advantage_estimator
            else None
        )
        self._advantage_postprocessor = build_postprocessor(algorithm.advantage.postprocessing)
        self._loss_adapter = build_loss_adapter(algorithm)
        if self._advantage_estimator is None and self._advantage_postprocessor is not None:
            raise ValueError("Advantage postprocessing requires an advantage estimator")
        loss = self._loss_adapter
        custom_loss_cfg = loss.custom

        self._entropy_coeff = algorithm.entropy_coeff
        self._custom_loss_spec = self._build_custom_loss_spec(custom_loss_cfg)
        self._loss_fn = loss.backend_name
        self._learning_rate = optim.lr
        self._lr_schedule = optim.schedule
        self._lr_schedule_total_steps = (
            int(config.training.total_steps)
            if self._lr_schedule is not None and config.training.total_steps is not None
            else None
        )
        self._beta1 = optim.beta1
        self._beta2 = optim.beta2
        self._weight_decay = optim.weight_decay
        self._eps = optim.eps
        self._grad_clip_norm = optim.grad_clip_norm

        self._sampling_logprob_source = algorithm.sampling_logprob_source
        self._mis_config = algorithm.mis

        # Normalize policy advantages in NorthRL; CE delegates aggregation to Weaver.
        self._loss_agg_mode = loss.runtime_reduction

        # Optimizer batches control updates; mini-batches bound actor forward/FB requests.
        batch_size = optim.batch_size
        self._batch_size = int(batch_size) if batch_size is not None else None
        mini_batch_size = optim.mini_batch_size
        self._mini_batch_size = int(mini_batch_size) if mini_batch_size is not None else None
        self._submit_ahead = int(optim.submit_ahead)
        if self._submit_ahead and (
            config.backend.type != "weaver" or self._custom_loss_spec is not None
        ):
            raise ValueError("submit_ahead requires Weaver built-in losses")

        self._last_full_checkpoint_step: int | None = None

        # Initialize DataDumper for debug data collection. Trainer now gets the
        # root config so runtime.debug is a direct path; no parent-chain walk.
        self._data_dumper: DataDumper = get_data_dumper(config, rank=0, key="remote_api_trainer")
        self._enable_debug_dump = config.runtime.debug.enable_data_dump

        logger.info(
            f"RemoteApiTrainer initialized: loss_fn={self._loss_fn}, "
            f"lr={self._learning_rate}, beta1={self._beta1}, beta2={self._beta2}, "
            f"weight_decay={self._weight_decay}, eps={self._eps}, grad_clip_norm={self._grad_clip_norm}, "
            f"truncation_c={loss.clip.get('truncation_c')}, "
            f"sampling_logprob_source={self._sampling_logprob_source}, "
            f"debug_dump_enabled={self._data_dumper.enabled}, "
            f"custom_loss={self._custom_loss_spec.name if self._custom_loss_spec else None}, "
            f"lr_schedule={self._lr_schedule}, submit_ahead={self._submit_ahead}"
        )

    def set_train_step(self, step: int) -> None:
        """Restore completed rounds after resume."""
        super().set_train_step(step)
        self._last_full_checkpoint_step = step

    @property
    def _mis_enabled(self) -> bool:
        return bool(getattr(getattr(self, "_mis_config", None), "enabled", False))

    @property
    def _mis_metrics_enabled(self) -> bool:
        return self._mis_enabled or bool(
            getattr(getattr(self, "_mis_config", None), "get_mismatch_metrics", False)
        )

    @staticmethod
    def _read_rollout_logprobs(
        trajectories: list[Trajectory | dict[str, Any]],
    ) -> list[list[float]]:
        """Read validated sampler scores without changing training trajectories."""
        snapshots = []
        for index, trajectory in enumerate(trajectories):
            if trajectory.get("sampling_mask") is not None:
                raise ValueError(
                    "algorithm.mis does not yet support sampling_mask: filtered recompute "
                    "and training probability contracts must be verified first."
                )
            if trajectory.get("rollout_logprobs_valid") is False:
                raise ValueError(f"MIS trajectory {index} has placeholder rollout logprobs.")
            validate_loss_mask_for_target_alignment(
                tokens=trajectory["tokens"], loss_mask=trajectory["loss_mask"]
            )
            if any(mask not in (0, 1) for mask in trajectory["loss_mask"]):
                raise ValueError(f"MIS trajectory {index} loss_mask must contain only 0 or 1.")
            rollout_logprobs = trajectory.get("rollout_log_probs")
            if rollout_logprobs is None:
                rollout_logprobs = trajectory.get("old_log_probs")
            if rollout_logprobs is None:
                raise ValueError(f"MIS trajectory {index} is missing rollout logprobs.")
            targets = align_logprobs_to_target(
                tokens=trajectory["tokens"], logprobs=rollout_logprobs, name="rollout_log_probs"
            )
            if any(
                not math.isfinite(value)
                for value, mask in zip(targets, trajectory["loss_mask"][1:], strict=True)
                if mask
            ):
                raise ValueError(f"MIS trajectory {index} has non-finite rollout logprobs.")
            snapshots.append(([0.0] if trajectory["tokens"] else []) + targets)
        return snapshots

    def _capture_rollout_logprobs(self, trajectories: list[Trajectory | dict[str, Any]]) -> None:
        """Keep immutable sampler scores before replacing the training denominator."""
        snapshots = self._read_rollout_logprobs(trajectories)
        for trajectory, snapshot in zip(trajectories, snapshots, strict=True):
            trajectory["rollout_log_probs"] = snapshot

    def _read_mismatch_rollout_logprobs(
        self, trajectories: list[Trajectory | dict[str, Any]]
    ) -> list[list[float]] | None:
        try:
            return self._read_rollout_logprobs(trajectories)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Skipping mismatch metrics: invalid rollout logprobs: %s", exc)
            return None

    def _record_mismatch_metrics(
        self,
        trajectories: list[Trajectory | dict[str, Any]],
        rollout_log_probs: list[list[float]],
        target_log_probs: Sequence[Sequence[float] | None],
        metrics: dict[str, Any],
    ) -> None:
        """Observe actual forward outputs, never fallback training denominators."""
        metrics["mis/metrics_available"] = 0
        try:
            if len(target_log_probs) != len(trajectories) or any(
                values is None for values in target_log_probs
            ):
                raise ValueError("incomplete trainer-forward logprobs")
            recomputed_log_probs = [
                align_target_logprobs_to_tokens(
                    tokens=trajectory["tokens"],
                    target_logprobs=values,
                    name="mismatch_log_probs",
                )
                for trajectory, values in zip(trajectories, target_log_probs, strict=True)
                if values is not None
            ]
            result = compute_mismatch_metrics(
                rollout_log_probs=rollout_log_probs,
                recomputed_log_probs=recomputed_log_probs,
                loss_masks=[trajectory["loss_mask"] for trajectory in trajectories],
                # Dr.GRPO's fixed loss scale is not a diagnostic mean;
                # retain token-weighted mismatch statistics.
                loss_agg_mode=(
                    "token-mean" if self._loss_agg_mode == "dr-grpo" else self._loss_agg_mode
                ),
                group_ids=(
                    [prompt_group_key(trajectory) for trajectory in trajectories]
                    if self._loss_agg_mode == "prompt-mean"
                    else None
                ),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Skipping mismatch metrics: invalid trainer-forward logprobs: %s", exc)
        else:
            metrics.update({f"mis/{name}": value for name, value in result.items()})
            metrics["mis/metrics_available"] = int(result["mismatch_tokens"] > 0)

    async def _compute_inference_mismatch_metrics(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> None:
        """Run a diagnostic forward while retaining rollout training denominators."""
        sequences = list(iter_training_sequences(trajectories))
        metrics["mis/metrics_available"] = 0
        if not sequences:
            return
        rollout_log_probs = self._read_mismatch_rollout_logprobs(sequences)
        if rollout_log_probs is None:
            return
        start = time.monotonic()
        try:
            # The inference-denominator mode does not use router replay. Do not
            # call _recompute_logprobs or its hooks, which mutate training state.
            assert self._service_holder is not None
            target_log_probs = []
            chunk_size = self._mini_batch_size or len(sequences)
            # Subclasses override the base hook's None default.
            # pylint: disable-next=assignment-from-none
            forward_config = self._build_forward_loss_fn_config()
            forward_kwargs: dict[str, Any] = (
                {"loss_fn_config": forward_config} if forward_config is not None else {}
            )
            for start_index in range(0, len(sequences), chunk_size):
                chunk = sequences[start_index : start_index + chunk_size]
                chunk_log_probs = await await_coro(
                    self._service_holder.compute_target_logprobs_batch(
                        [trajectory["tokens"] for trajectory in chunk],
                        sampling_masks=None,
                        **forward_kwargs,
                    )
                )
                if len(chunk_log_probs) != len(chunk):
                    raise ValueError(
                        "Mismatch diagnostic forward returned an unexpected batch size."
                    )
                target_log_probs.extend(chunk_log_probs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Skipping mismatch metrics: diagnostic forward failed: %s", exc)
            metrics["timing_train/mismatch_forward_seconds"] = time.monotonic() - start
            return
        metrics["timing_train/mismatch_forward_seconds"] = time.monotonic() - start
        self._record_mismatch_metrics(sequences, rollout_log_probs, target_log_probs, metrics)

    def _compute_mis(self, trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
        """Compute fixed corrections before conversion and optimizer-step splitting."""
        sequences = list(iter_training_sequences(trajectories))
        result = compute_mis_weights(
            self._mis_config,
            rollout_log_probs=[trajectory["rollout_log_probs"] for trajectory in sequences],
            recomputed_log_probs=[trajectory["old_log_probs"] for trajectory in sequences],
            loss_masks=[trajectory["loss_mask"] for trajectory in sequences],
        )
        for trajectory, weights, mask in zip(
            sequences, result.correction_weights, result.correction_mask, strict=True
        ):
            trajectory["correction_weights"] = weights
            trajectory["correction_mask"] = mask
        metrics.update({f"mis/{name}": value for name, value in result.metrics.items()})

    def set_lr_schedule_total_steps(self, total_steps: int) -> None:
        """Set the finite optimizer-window horizon resolved before training."""

        if total_steps <= 0:
            raise ValueError(f"LR schedule total_steps must be positive, got {total_steps}")
        self._lr_schedule_total_steps = int(total_steps)
        logger.info("LR schedule optimizer-window horizon set to %d", total_steps)

    def _effective_learning_rate(self) -> float:
        """Return the LR for the next optimizer update (zero-based)."""

        if self._lr_schedule is None:
            return float(self._learning_rate)
        if self._lr_schedule_total_steps is None:
            raise RuntimeError("LR schedule optimizer-window horizon has not been initialized")
        style = str(self._lr_schedule.style)
        if style != "cosine":  # Config validation should make this unreachable.
            raise ValueError(f"Unsupported LR schedule style: {style!r}")
        return cosine_learning_rate(
            update_step=self._train_step,
            total_steps=self._lr_schedule_total_steps,
            peak_lr=float(self._learning_rate),
            min_lr=float(self._lr_schedule.min_lr),
            warmup_fraction=float(self._lr_schedule.warmup_fraction),
        )

    def set_service_holder(self, service_holder: WeaverServiceHolder) -> None:
        """
        Set the service holder.

        Args:
            service_holder: WeaverServiceHolder instance.
        """
        self._service_holder = service_holder

    @abstractmethod
    async def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories for training (e.g., compute advantages).

        Default implementation is a no-op. Override in derived classes
        to perform algorithm-specific trajectory preparation.

        Args:
            trajectories: Complete rollout results with direct or optional nested training sequences
            metrics: Dictionary to store preparation metrics

        Returns:
            Complete trajectories with algorithm-specific fields prepared in place
        """

    @staticmethod
    def _resample_failed_groups(
        trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """Replace at most half a group's outcomes after logging the real rollouts.

        Retain each valid member and copy distinct randomly chosen donors into
        invalid slots. Slot identities stay unique; source IDs identify the
        sampled data. Copies must own their segments because preparation mutates
        sequence metadata, advantages, and logprobs in place.
        """
        groups: dict[Any, list[int]] = {}
        for index, trajectory in enumerate(trajectories):
            key = group_key(trajectory)
            if key is not None:
                groups.setdefault(key, []).append(index)

        result = trajectories
        repaired_groups = replacements = 0
        for indices in groups.values():
            valid, invalid = [], []
            for index in indices:
                trajectory = trajectories[index]
                if trajectory.status == "success" and not trajectory.get("group_skipped", False):
                    valid.append(trajectory)
                else:
                    invalid.append(index)
            if not invalid or len(invalid) > len(valid):
                continue
            if result is trajectories:
                result = list(trajectories)
            for index, donor in zip(invalid, random.sample(valid, len(invalid)), strict=True):
                replacement = copy.deepcopy(donor)
                replacement["run_id"] = trajectories[index]["run_id"]
                replacement["rollout_id"] = trajectories[index]["rollout_id"]
                replacement["resampled_from_rollout_id"] = donor["rollout_id"]
                result[index] = replacement
            repaired_groups += 1
            replacements += len(invalid)

        metrics["batch/resampled_group_count"] = repaired_groups
        metrics["batch/resampled_trajectory_count"] = replacements
        if groups:
            metrics["batch/skipped_group_count"] = sum(
                all(result[index].get("group_skipped", False) for index in indices)
                for indices in groups.values()
            )
        return result

    # Built-in RL trainers opt into independent estimators. Third-party subclasses
    # retaining their own _prepare_trajectories contract are not re-estimated.
    _uses_advantage_estimator = False

    async def _prepare_advantage_inputs(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> Mapping[str, Any]:
        """Prepare service outputs before actor recompute and pure estimation."""
        del trajectories, metrics
        return {}

    def _build_forward_loss_fn_config(self) -> dict[str, Any] | None:
        """Optional actor probability settings; non-GRPO trainers keep their defaults."""
        return None

    def _build_loss_fn_config(self) -> dict[str, Any]:
        """Build the loss_fn_config dict forwarded to the remote training service.

        Includes entropy coefficient and any ratio-clipping parameters that were
        set in the train-service config.  Unknown keys are silently ignored by
        Weaver loss computers that do not read them, so this is safe for forward
        compatibility.
        """
        return self._loss_adapter.build_loss_fn_config(enable_debug_dump=self._enable_debug_dump)

    async def _recompute_logprobs(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """Recompute logprobs via the trainer model.

        Replaces old_log_probs stored in each trajectory with trainer-computed
        logprobs. When a trajectory carries a rollout sampling_mask, the mask is
        passed to the trainer-side computation so the denominator is computed
        over the same constrained action space as the rollout server.

        Weaver routes this through the training client's ``forward`` API with
        ``loss_fn="forward_logprob"``, so recompute uses the trainer backend
        rather than the rollout engine. Each request is bounded by
        ``algorithm.optim.mini_batch_size`` before optimizer-batch splitting.
        """
        sequences = list(iter_training_sequences(trajectories))
        if not sequences:
            metrics["timing/recompute_logprobs"] = 0.0
            logger.info("[recompute_logprobs] skipped empty trajectory batch")
            return trajectories

        assert self._service_holder is not None

        mismatch_rollout_log_probs = None
        if self._mis_metrics_enabled:
            metrics["mis/metrics_available"] = 0
        if self._mis_enabled:
            self._capture_rollout_logprobs(sequences)
            mismatch_rollout_log_probs = [
                trajectory["rollout_log_probs"] for trajectory in sequences
            ]
        elif self._mis_metrics_enabled:
            mismatch_rollout_log_probs = self._read_mismatch_rollout_logprobs(sequences)

        recompute_start = time.monotonic()
        sampling_masks = [self._sampling_mask_for_compute_logprobs(traj) for traj in sequences]
        has_sampling_mask = [mask is not None for mask in sampling_masks]
        use_sampling_masks = any(has_sampling_mask)
        assert (
            all(has_sampling_mask) or not use_sampling_masks
        ), "sampling_mask must be present for all sequences in a recompute batch or none."

        chunk_size = self._mini_batch_size or len(sequences)
        caller_task = asyncio.current_task()

        async def submit_forward(start_index):
            # Forward requests are read-only; keep their result/chunk association.
            return start_index

        async def collect_forward(start_index, _handle):
            chunk = sequences[start_index : start_index + chunk_size]
            # Retry only this read-only request, retaining already validated chunks.
            for attempt in range(4):
                try:
                    result = await self._compute_recompute_logprobs_batch(
                        chunk,
                        sampling_masks=(
                            sampling_masks[start_index : start_index + chunk_size]
                            if use_sampling_masks
                            else None
                        ),
                    )
                    if len(result.logprobs) != len(chunk):
                        raise ValueError(
                            f"compute_logprobs batch size mismatch: got {len(result.logprobs)}, "
                            f"expected {len(chunk)}"
                        )
                    if self._mis_enabled and any(values is None for values in result.logprobs):
                        raise ValueError("MIS recompute returned missing logprobs.")
                    recomputed: list[Sequence[float]] = [
                        align_target_logprobs_to_tokens(
                            tokens=traj["tokens"], target_logprobs=values, name="old_log_probs"
                        )
                        for traj, values in zip(chunk, result.logprobs, strict=True)
                    ]
                    if self._mis_enabled:
                        for index, (trajectory, values) in enumerate(
                            zip(chunk, recomputed, strict=True), start=start_index
                        ):
                            if isinstance(values, array):
                                mask = np.asarray(trajectory["loss_mask"], dtype=bool)
                                if mask.ndim != 1 or len(mask) != len(values):
                                    raise ValueError("MIS loss_mask and logprobs length mismatch.")
                                invalid = bool(np.any(~np.isfinite(np.asarray(values)) & mask))
                            else:
                                invalid = any(
                                    not math.isfinite(value)
                                    for value, mask in zip(
                                        values, trajectory["loss_mask"], strict=True
                                    )
                                    if mask
                                )
                            if invalid:
                                raise ValueError(
                                    "MIS recompute returned non-finite logprobs "
                                    f"for trajectory {index}."
                                )
                    for index, trajectory in enumerate(chunk):
                        if isinstance(trajectory["tokens"], array):
                            if not isinstance(recomputed[index], array):
                                recomputed[index] = array("d", recomputed[index])
                            if result.logprobs[index] is not None and not isinstance(
                                result.logprobs[index], array
                            ):
                                result.logprobs[index] = array("d", result.logprobs[index])
                    return chunk, result, recomputed
                except FutureCancelledError:
                    raise
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # The pipeline shields accepted requests while draining cancellation.
                    if caller_task is not None and caller_task.cancelling():
                        raise asyncio.CancelledError from exc
                    if attempt == 3:
                        raise
                    logger.warning(
                        "[recompute_logprobs] chunk_start=%d chunk_size=%d attempt=%d/4 "
                        "failed; retrying chunk (error_type=%s): %s",
                        start_index,
                        len(chunk),
                        attempt + 1,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )

        forward_batches = []
        if self._submit_ahead:
            logger.info(
                "[submit_ahead] forward requests=%d max_outstanding=%d",
                (len(sequences) + chunk_size - 1) // chunk_size,
                self._submit_ahead + 1,
            )
            forward_batches = await run_request_pipeline(
                range(0, len(sequences), chunk_size),
                submit_forward,
                collect_forward,
                submit_ahead=self._submit_ahead,
            )
        else:
            for start_index in range(0, len(sequences), chunk_size):
                forward_batches.append(await collect_forward(start_index, None))

        # Publish only after every chunk succeeds; keep R2 refs with their own request.
        results = []
        for chunk, forward_result, recomputed in forward_batches:
            results.extend(forward_result.logprobs)
            for traj, values in zip(chunk, recomputed, strict=True):
                traj["old_log_probs"] = values
        for chunk, forward_result, _recomputed in forward_batches:
            self._after_recompute_logprobs(chunk, forward_result, metrics)

        replaced = len(sequences)
        failed = 0
        recompute_time = time.monotonic() - recompute_start
        metrics["timing/recompute_logprobs"] = recompute_time
        logger.info(
            "[recompute_logprobs] replaced=%d/%d failed=%d time=%.1fs",
            replaced,
            len(sequences),
            failed,
            recompute_time,
        )
        if mismatch_rollout_log_probs is not None:
            self._record_mismatch_metrics(sequences, mismatch_rollout_log_probs, results, metrics)
        return trajectories

    async def _compute_recompute_logprobs_batch(
        self,
        trajectories: list[Trajectory | dict[str, Any]],
        *,
        sampling_masks: list[list[list[int]] | dict[str, Any] | None] | None,
    ) -> ForwardLogprobsResult:
        """Compute target logprobs for the batch (extension seam).

        Default is a plain trainer ``forward_logprob``. Features that must
        piggyback on this exact forward override it — R2 records the routing
        chosen by this forward, while R3 replays rollout routing during it
        (see ``RemoteApiGrpoTrainer``).
        """
        assert self._service_holder is not None
        # pylint: disable-next=assignment-from-none
        forward_config = self._build_forward_loss_fn_config()
        forward_kwargs: dict[str, Any] = (
            {"loss_fn_config": forward_config} if forward_config is not None else {}
        )
        logprobs = await await_coro(
            self._service_holder.compute_target_logprobs_batch(
                [trajectory["tokens"] for trajectory in trajectories],
                sampling_masks=sampling_masks,
                **forward_kwargs,
            )
        )
        return ForwardLogprobsResult(logprobs=logprobs)

    def _after_recompute_logprobs(
        self,
        trajectories: list[Trajectory | dict[str, Any]],
        forward_result: ForwardLogprobsResult,
        metrics: dict[str, Any],
    ) -> None:
        """Hook run after ``old_log_probs`` are replaced (default: no-op)."""
        del trajectories, forward_result, metrics

    @staticmethod
    def _sampling_mask_for_compute_logprobs(
        traj: Trajectory | dict[str, Any],
    ) -> list[list[int]] | dict[str, Any] | None:
        """Return target-token-aligned sampling mask for trainer recompute."""
        sampling_mask = traj.get("sampling_mask")
        if sampling_mask is None:
            return None
        if isinstance(sampling_mask, dict):
            from northrl.utils.sampling_mask_refs import align_sampling_mask_ref

            return align_sampling_mask_ref(traj["tokens"], traj["loss_mask"], sampling_mask)

        return align_response_sampling_mask_to_target(
            tokens=traj["tokens"],
            loss_mask=traj["loss_mask"],
            sampling_mask=sampling_mask,
        )

    def _convert_to_datums(self, trajectories: list[Trajectory]) -> list[dict]:
        """
        Convert complete trajectories to the Datum format expected by the training service.

        Default implementation uses ``convert_trajectories_to_datums`` (IS loss).
        Override in derived classes for different loss functions.

        Args:
            trajectories: Prepared trajectories

        Returns:
            List of serializable datum dicts
        """
        self._last_datum_sources: list[tuple[int, int]] = []
        return convert_trajectories_to_datums(trajectories, datum_sources=self._last_datum_sources)

    def _build_custom_loss_spec(self, raw_spec: dict[str, Any] | None) -> CustomLossSpec | None:
        """Resolve the canonical custom objective's callback payload."""
        if raw_spec is None:
            return None
        spec = parse_custom_loss_spec({"name": raw_spec["name"], "params": raw_spec["params"]})
        assert spec is not None
        params = dict(spec.params)
        params.setdefault("entropy_coeff", self._entropy_coeff)
        return CustomLossSpec(name=spec.name, params=params)

    def _split_into_batches(self, datums_data: list[dict]) -> list[list[dict]]:
        """Use the whole delivered batch when unset; otherwise keep fixed-size batches."""
        batch_size = self._batch_size
        if (
            self._loss_agg_mode in {"prompt-mean", "dr-grpo", "trajectory-mean"}
            and batch_size is not None
        ):
            raise ValueError(f"{self._loss_agg_mode} requires algorithm.optim.batch_size=null")
        if batch_size is None or batch_size <= 0 or batch_size >= len(datums_data):
            return [datums_data]

        shuffled_datums = list(datums_data)
        random.shuffle(shuffled_datums)
        batches = [
            shuffled_datums[i : i + batch_size] for i in range(0, len(shuffled_datums), batch_size)
        ]

        if len(batches[-1]) < batch_size:
            last = list(batches[-1])
            original_last = list(last)
            while len(last) < batch_size:
                source = original_last[len(last) % len(original_last)]
                repeated = copy.deepcopy(source)
                datum_id = repeated.get("datum_id")
                if datum_id is not None:
                    if not isinstance(datum_id, str) or not datum_id:
                        raise ValueError("datum_id must be a non-empty string when provided")
                    repeated["datum_id"] = _padding_occurrence_datum_id(
                        datum_id,
                        mini_batch_index=len(batches) - 1,
                        slot_index=len(last),
                    )
                last.append(repeated)
            batches[-1] = last

        logger.info(
            "[train] optimizer batch split: %d datums -> %d batches of ~%d",
            len(datums_data),
            len(batches),
            batch_size,
        )
        return batches

    def _split_into_mini_batches(self, datums_data: list[dict]) -> list[list[dict]]:
        """Split one normalized optimizer batch without shuffling or padding."""
        size = self._mini_batch_size
        if size is None or size >= len(datums_data):
            return [datums_data]
        return [datums_data[i : i + size] for i in range(0, len(datums_data), size)]

    def _normalize_advantages_for_sum(
        self, datums_data: list[dict], *, token_counts: Sequence[int] | None = None
    ) -> list[dict]:
        """Adapt loss reduction to the backend's supported sum objectives."""
        return normalize_advantages_for_sum(
            datums_data,
            loss_fn=self._loss_fn,
            mode=self._algorithm_config.loss.reduction.mode,
            token_counts=token_counts,
            logger=logger,
            train_step=self._train_step + 1,
        )

    def _loss_fn_config_for_microbatch(
        self,
        datums_data: list[dict],
        configured_loss_fn_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return the normal built-in CE configuration for managed SFT data.

        Managed and local SFT datums must use the same aggregation mode in a
        mixed microbatch.  In particular, forwarding ``loss_agg_mode`` is
        required for the server-materialized datum to reproduce ms-swift's
        active-token mean.
        """

        managed_datums = [datum for datum in datums_data if datum.get("kind") == "sample_ref"]
        if not managed_datums:
            return configured_loss_fn_config

        if self._custom_loss_spec is not None or self._loss_fn != "cross_entropy":
            raise ValueError(
                "Managed SampleRef microbatches require Weaver's built-in cross_entropy loss"
            )
        return configured_loss_fn_config

    def _on_optimizer_step_succeeded(self, trajectories: list[Trajectory]) -> None:
        """Hook for committing client-side state after the remote optimizer succeeds."""

    def _aggregate_batch_metrics(
        self, all_batch_metrics: list[dict[str, Any]], *, accumulate: bool = False
    ) -> dict[str, Any]:
        return aggregate_service_metrics(all_batch_metrics, accumulate=accumulate)

    async def _run_optimizer_batch(
        self,
        batch: list[dict],
        mini_batches: list[list[dict]],
        loss_fn_config: dict[str, Any] | None,
        *,
        token_counts: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Accumulate all FB contributions, then perform exactly one update."""
        assert self._service_holder is not None, "ServiceHolder not set"
        if token_counts is None:
            token_counts = [datum_response_token_count(datum) for datum in batch]
        if len(token_counts) != len(batch):
            raise ValueError("Every Datum requires its final response-token count")
        # Counts are scoped to this optimizer batch, never cached across calls
        # on mutable masks. Normalization only changes advantages.
        counts_by_datum = dict(zip(map(id, batch), token_counts, strict=True))
        forward_backward_metrics = []
        forward_backward_started = time.monotonic()
        if self._custom_loss_spec is not None:
            # Fail closed: managed SampleRef datums need the built-in CE loss.
            self._loss_fn_config_for_microbatch(batch, None)
            kwargs: dict[str, Any] = {}
            # pylint: disable-next=assignment-from-none
            forward_config = self._build_forward_loss_fn_config()
            if forward_config is not None:
                kwargs["loss_fn_config"] = forward_config
            if len(mini_batches) > 1:
                kwargs["mini_batch_sizes"] = [len(chunk) for chunk in mini_batches]
            result = await await_coro(
                self._service_holder.forward_backward_custom(
                    datums_data=batch,
                    custom_loss_spec=self._custom_loss_spec.to_payload(),
                    **kwargs,
                )
            )
            forward_backward_completed = time.monotonic()
            training_metrics = dict(result.get("metrics", {}))
            logger.info("Raw Weaver custom forward metrics: %s", training_metrics)
            if "loss" not in training_metrics and "loss" in result:
                training_metrics["loss"] = result["loss"]
        else:
            loss_name = self._loss_fn
            assert loss_name is not None, "Native execution requires a loss name"
            chunk_metrics = []
            pipelined_results = None
            if self._submit_ahead:
                service_holder = self._service_holder

                async def submit(chunk):
                    return await await_coro(
                        service_holder.submit_forward_backward(
                            datums_data=chunk,
                            loss_fn=loss_name,
                            loss_fn_config=self._loss_fn_config_for_microbatch(
                                chunk, loss_fn_config
                            ),
                        )
                    )

                async def collect(chunk, handle):
                    return await await_coro(
                        service_holder.collect_forward_backward_result(
                            handle,
                            datums_data=chunk,
                            loss_fn_config=self._loss_fn_config_for_microbatch(
                                chunk, loss_fn_config
                            ),
                        )
                    )

                logger.info(
                    "[submit_ahead] FB requests=%d max_outstanding=%d",
                    len(mini_batches),
                    self._submit_ahead + 1,
                )
                pipelined_results = await run_request_pipeline(
                    mini_batches, submit, collect, submit_ahead=self._submit_ahead
                )
                forward_backward_completed = time.monotonic()
            for index, chunk in enumerate(mini_batches, start=1):
                logger.info(
                    "[train] forward_backward: train_step=%d mini_batch=%d/%d num_datums=%d",
                    self._train_step + 1,
                    index,
                    len(mini_batches),
                    len(chunk),
                )
                if pipelined_results is None:
                    result = await await_coro(
                        self._service_holder.forward_backward(
                            datums_data=chunk,
                            loss_fn=loss_name,
                            loss_fn_config=self._loss_fn_config_for_microbatch(
                                chunk, loss_fn_config
                            ),
                        )
                    )
                    forward_backward_completed = time.monotonic()
                else:
                    result = pipelined_results[index - 1]
                current = dict(result.get("metrics", {}))
                if "loss" not in current and "loss" in result:
                    current["loss"] = result["loss"]
                logger.info("Raw Weaver forward metrics: %s", current)
                current["_response_tokens"] = sum(counts_by_datum[id(d)] for d in chunk)
                chunk_metrics.append(current)
                forward_backward_metrics.append(current)
            training_metrics = self._aggregate_batch_metrics(chunk_metrics, accumulate=True)

        forward_backward_wall = forward_backward_completed - forward_backward_started
        # Never enqueue OPT before every FB has completed successfully.
        optimizer_started = time.monotonic()
        optim_result = await await_coro(
            self._service_holder.optim_step(
                learning_rate=self._effective_learning_rate(),
                beta1=self._beta1,
                beta2=self._beta2,
                weight_decay=self._weight_decay,
                eps=self._eps,
                grad_clip_norm=self._grad_clip_norm,
            )
        )
        optimizer_wall = time.monotonic() - optimizer_started
        optim_metrics = optim_result.get("metrics", {})
        stage_timings = training_stage_metrics(
            forward_backward_metrics=forward_backward_metrics,
            optimizer_metrics=optim_metrics if isinstance(optim_metrics, dict) else {},
            forward_backward_wall_seconds=forward_backward_wall,
            optimizer_step_wall_seconds=optimizer_wall,
        )
        if isinstance(optim_metrics, dict):
            logger.info("Raw Weaver optimizer metrics: %s", optim_metrics)
            training_metrics.update(optim_metrics)
        training_metrics["_training_timing"] = stage_timings
        training_metrics["_response_tokens"] = sum(token_counts)
        if self._custom_loss_spec is not None:
            # Built-ins reduce over all samples, including zero-mask rows.
            known_sample_mean = self._custom_loss_spec.name in {
                "importance_sampling",
                "truncated_importance_sampling",
                "direct_logprob",
            }
            training_metrics["_loss_weight"] = len(batch) if known_sample_mean else None
        elif self._loss_agg_mode in {"token-mean", "dapo"}:
            training_metrics["_loss_weight"] = training_metrics["_response_tokens"]
        elif self._loss_agg_mode in {"prompt-mean", "dr-grpo"}:
            # These modes require one complete optimizer batch.
            training_metrics["_loss_weight"] = 1
        else:
            training_metrics["_loss_weight"] = sum(count > 0 for count in token_counts)
        return training_metrics

    async def _save_full_checkpoint(self, step: int) -> str | None:
        """Save a resumable Weaver checkpoint and update the local index."""
        service_holder = self._service_holder
        assert service_holder is not None
        try:
            checkpoint_path = await await_coro(
                service_holder.save_weight_and_optimizer(name=f"global_step_{step:06d}")
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception(
                "[checkpoint] Failed to save weights and optimizer at step %d",
                step,
            )
            return None

        self._last_full_checkpoint_step = step
        self._append_remote_checkpoint_index(step, checkpoint_path)
        logger.info(
            "[checkpoint] Saved weights and optimizer at step %d -> %s "
            "(set resume.resume_path to this value to resume)",
            step,
            checkpoint_path,
        )
        return checkpoint_path

    async def _on_training_complete(self) -> None:
        """Save the final completed training step unless it was already saved."""
        if self._train_step <= 0 or not self._config.training.save.get("final", True):
            return
        if self._last_full_checkpoint_step == self._train_step:
            logger.info("[checkpoint] Final step %d is already saved", self._train_step)
            return
        await self._save_full_checkpoint(self._train_step)

    async def _on_weight_sync_failed(self, completed_step: int) -> None:
        """Save the trained state before synchronization advances the public step."""
        if self._last_full_checkpoint_step != completed_step:
            await self._save_full_checkpoint(completed_step)

    def _append_remote_checkpoint_index(self, step: int, checkpoint_path: str) -> None:
        """Append a checkpoint entry to ``checkpoint_path/weaver_checkpoints.jsonl``.

        Failures here are non-fatal: training continues and the path is still
        available in logs / wandb, but ``resume.mode=auto`` may not see this
        save until a future entry succeeds.
        """
        index_dir = self._config.training.save.dir or ""
        if not index_dir:
            logger.debug("[checkpoint] training.save.dir is empty; skipping index update")
            return

        import datetime
        import os

        index_path = os.path.join(index_dir, "weaver_checkpoints.jsonl")
        entry = {
            "step": int(step),
            "path": checkpoint_path,
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        dataloader = getattr(self, "_dataloader", None)
        state_dict: Any = getattr(dataloader, "state_dict", None)
        if callable(state_dict):
            # pylint: disable=not-callable
            entry["dataloader_state"] = cast(Callable[[], dict[str, Any]], state_dict)()
            # pylint: enable=not-callable
            entry["schema_version"] = 1
        try:
            os.makedirs(index_dir, exist_ok=True)
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            logger.info("[checkpoint] Index updated: %s <- step %d", index_path, step)
        except OSError as e:
            logger.warning(
                "[checkpoint] Failed to update index %s: %s "
                "(checkpoint itself was saved; resume.mode=auto may miss this entry)",
                index_path,
                e,
            )

    async def train(self, trajectories: list[Trajectory]) -> tuple[dict, str | None, str | None]:
        """
        Train on complete trajectories, keeping parent results through preparation and conversion.

        Steps:
        1. Log rollout metrics
        2. Prepare probabilities, then call the estimator and optional postprocessor
        3. Convert trajectories to service Datum format
        4. Execute forward_backward + optim_step
        5. Export sampling weights and return their path and name

        Args:
            trajectories: Complete rollout results with direct tokens or optional nested sequences

        Returns:
            Training metrics, exported sampling path, and exported weight name.
            The path and name are None when no sampling weights are exported.
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        training_started = time.monotonic()
        metrics: dict[str, Any] = {
            "batch/trainable_group_count": 0,
            "batch/trainable_trajectory_count": 0,
            "batch/trainable_segment_count": 0,
            "batch/trainable_response_tokens_total": 0,
            "training/optimizer_step_count": 0,
            "training/forward_backward_call_count": 0,
        }

        # Step 1: Every complete trajectory contributes one outcome to metrics.
        log_rollout_metrics(trajectories, metrics)
        if "batch/group_count" not in metrics:
            metrics.pop("batch/trainable_group_count", None)
        if not trajectories:
            # A complete round may contain only failed/skipped groups or scored
            # zero-segment rollouts. Advance the round without an optimizer step.
            metrics["training/skipped_flag"] = 1
            metrics["timing_train/batch_training_seconds"] = time.monotonic() - training_started
            return metrics, None, None

        # Resample after outcome metrics, before advantages and actor recomputation.
        if self._config.algorithm.type in ("grpo", "opd"):
            trajectories = self._resample_failed_groups(trajectories, metrics)

        logger.info(
            "[batch_received] completed_step=%d input_metrics=%s",
            self._train_step + 1,
            {
                key: value
                for key, value in metrics.items()
                if not key.startswith(("training/", "batch/trainable_"))
            },
        )

        # Validate/order complete parents before any service I/O. Preparation
        # retains estimator-specific probability sources (notably OPD rollout
        # scores) before actor recompute can replace the loss denominator.
        trajectories = await self._prepare_trajectories(trajectories, metrics)
        advantage_inputs = await self._prepare_advantage_inputs(trajectories, metrics)
        num_sequences = sum(1 for _ in iter_training_sequences(trajectories))

        if num_sequences:
            if self._mis_metrics_enabled:
                metrics["mis/metrics_available"] = 0
            if (
                self._sampling_logprob_source == "recomputed"
                and self._algorithm_config.type != "sft"
            ):
                trajectories = await self._recompute_logprobs(trajectories, metrics)
            elif self._mis_metrics_enabled:
                await self._compute_inference_mismatch_metrics(trajectories, metrics)

        # Every estimator and postprocessor sees this complete rollout step.
        # Optimizer batches and forward/backward requests are split only below.
        if self._advantage_estimator is not None:
            clear_postprocessing_cache(trajectories)
            trajectories = self._advantage_estimator.estimate(
                trajectories, advantage_inputs, metrics, logger=logger
            )
            if self._advantage_postprocessor is not None:
                trajectories = self._advantage_postprocessor.process(
                    trajectories, metrics, logger=logger
                )
        if self._data_dumper.should_dump("prepared_trajectories", self._train_step):
            self._data_dumper.dump_prepared_trajectories(
                step=self._train_step, trajectories=trajectories
            )

        if not num_sequences:
            metrics["training/skipped_flag"] = 1
            metrics["timing_train/batch_training_seconds"] = time.monotonic() - training_started
            return metrics, None, None

        # Estimator normalization and postprocessing both see original response
        # masks. Rejection never changes their step statistics retroactively.
        if self._mis_enabled:
            self._compute_mis(trajectories, metrics)

        # Step 3: Convert to service Datum format
        self._last_datum_sources = []
        preparation_started = time.monotonic()
        datums_data = self._convert_to_datums(trajectories)
        if self._loss_fn == "gspo":
            attach_gspo_trajectory_metadata(datums_data, self._last_datum_sources)
        logger.info(
            "[training_preparation] train_step=%d stage=conversion num_datums=%d wall_seconds=%.6f",
            self._train_step + 1,
            len(datums_data),
            time.monotonic() - preparation_started,
        )

        preparation_started = time.monotonic()
        if len(self._last_datum_sources) == len(datums_data):
            token_counts = log_trainable_metrics(
                trajectories, datums_data, metrics, datum_sources=self._last_datum_sources
            )
        else:
            # Third-party converters may not expose parent identities. Do not guess them.
            token_counts = [datum_response_token_count(datum) for datum in datums_data]
            metrics["batch/trainable_segment_count"] = sum(count > 0 for count in token_counts)
            metrics["batch/trainable_response_tokens_total"] = sum(token_counts)
            metrics.pop("batch/trainable_trajectory_count", None)
            metrics.pop("batch/trainable_group_count", None)
            logger.warning("Datum converter omitted source identities; parent counts unavailable")
        logger.info(
            "[training_preparation] train_step=%d stage=metrics num_datums=%d wall_seconds=%.6f",
            self._train_step + 1,
            len(datums_data),
            time.monotonic() - preparation_started,
        )
        if len(datums_data) == 0:
            logger.warning("No valid datums after filtering, skipping training step")
            metrics["training/skipped_flag"] = 1
            metrics["timing_train/batch_training_seconds"] = time.monotonic() - training_started
            return metrics, None, None

        metrics["training/skipped_flag"] = 0

        # Step 4: Each optimizer batch contains one or more FB requests.
        # Normalize using the FULL optimizer batch before splitting requests.
        loss_fn_config = None if self._custom_loss_spec else self._build_loss_fn_config()
        batches = self._split_into_batches(datums_data)
        if self._mis_enabled or self._loss_agg_mode in {
            "prompt-mean",
            "dr-grpo",
            "trajectory-mean",
        }:
            active_batches = [
                batch
                for batch in batches
                if any(any(datum["loss_fn_inputs"]["loss_mask"]) for datum in batch)
            ]
            if self._mis_enabled:
                metrics["mis/skipped_batches"] = len(batches) - len(active_batches)
                metrics["mis/optimizer_steps"] = len(active_batches)
            batches = active_batches
            if not batches:
                logger.info("No effective training targets; skipping optimizer updates")
                metrics["training/skipped_flag"] = 1
                metrics["training/optimizer_step_count"] = 0
                metrics["training/forward_backward_call_count"] = 0
                metrics["timing_train/batch_training_seconds"] = time.monotonic() - training_started
                return metrics, None, None
        # Preserve final-mask counts through shuffling and optimizer padding.
        # Copied padding datums have new identities and are counted once here.
        counts_by_datum = dict(zip(map(id, datums_data), token_counts, strict=True))
        batch_token_counts = [
            [
                (
                    counts_by_datum[id(datum)]
                    if id(datum) in counts_by_datum
                    else datum_response_token_count(datum)
                )
                for datum in batch
            ]
            for batch in batches
        ]
        preparation_started = time.monotonic()
        if self._custom_loss_spec is None:
            batches = [
                self._normalize_advantages_for_sum(batch, token_counts=counts)
                for batch, counts in zip(batches, batch_token_counts, strict=True)
            ]
        logger.info(
            "[training_preparation] train_step=%d stage=normalization num_batches=%d wall_seconds=%.6f",
            self._train_step + 1,
            len(batches),
            time.monotonic() - preparation_started,
        )

        mini_batches_by_batch = [self._split_into_mini_batches(batch) for batch in batches]
        if self._data_dumper.should_dump("datums", self._train_step):
            dump_loss_fn_config = self._loss_fn_config_for_microbatch(
                datums_data,
                self._build_loss_fn_config(),
            )
            self._data_dumper.dump_datums(
                step=self._train_step,
                datums_data=[datum for batch in batches for datum in batch],
                extra_info={
                    "loss_fn": self._loss_fn,
                    # This is the fail-closed effective config for the whole
                    # dumped batch. SFT currently submits it as one microbatch;
                    # if splitting is later enabled, recording None remains a
                    # safe standalone replay default whenever managed refs
                    # occur anywhere in the dump.
                    "loss_fn_config": dump_loss_fn_config,
                    "learning_rate": self._effective_learning_rate(),
                    "beta1": self._beta1,
                    "beta2": self._beta2,
                    "weight_decay": self._weight_decay,
                    "eps": self._eps,
                    "grad_clip_norm": self._grad_clip_norm,
                    "batch_size": self._batch_size,
                    "batch_sizes": [len(batch) for batch in batches],
                    "mini_batch_size": self._mini_batch_size,
                },
            )

        all_batch_metrics = []
        all_batch_timings = []
        for index, (batch, mini_batches, counts) in enumerate(
            zip(batches, mini_batches_by_batch, batch_token_counts, strict=True), start=1
        ):
            logger.info(
                "[train] optimizer batch: train_step=%d batch=%d/%d num_datums=%d forward_backward_calls=%d",
                self._train_step + 1,
                index,
                len(batches),
                len(batch),
                len(mini_batches),
            )
            batch_metrics = await self._run_optimizer_batch(
                batch, mini_batches, loss_fn_config, token_counts=counts
            )
            all_batch_timings.append(batch_metrics.pop("_training_timing", {}))
            all_batch_metrics.append(batch_metrics)
        self._on_optimizer_step_succeeded(trajectories)
        metrics.update(sum_training_stage_metrics(all_batch_timings))

        training_metrics = (
            self._aggregate_batch_metrics(all_batch_metrics)
            if len(all_batch_metrics) > 1
            else all_batch_metrics[0]
        )
        metrics["training/optimizer_step_count"] = len(batches)
        metrics["training/forward_backward_call_count"] = sum(map(len, mini_batches_by_batch))

        # Sanity checks on training metrics returned from weaver
        _loss = training_metrics.get("loss")
        _grad_norm = training_metrics.get("grad_norm")
        _tokens = training_metrics.get("tokens")
        _ratio_mean = training_metrics.get("ratio:mean")
        if _loss is not None and isinstance(_loss, float):
            if abs(_loss) > 100:
                logger.warning(
                    "[train] Abnormal loss=%.4f (expected |loss| < 100). "
                    "Possible gradient_scaling_factor or loss aggregation issue.",
                    _loss,
                )
        if _grad_norm is not None and isinstance(_grad_norm, float):
            if _grad_norm > 1e6:
                logger.warning(
                    "[train] Abnormal grad_norm=%.4e (expected < 1e6). "
                    "Possible gradient_scaling_factor=1.0 (calculate_per_token_loss=True) "
                    "without proper finalize_model_grads normalization.",
                    _grad_norm,
                )
        if _tokens is not None and isinstance(_tokens, (int, float)):
            if _tokens <= 0:
                logger.warning("[train] tokens=%s — no valid tokens in training batch!", _tokens)
        if _ratio_mean is not None and isinstance(_ratio_mean, float):
            if abs(_ratio_mean - 1.0) > 0.5:
                logger.warning(
                    "[train] ratio:mean=%.4f — far from 1.0. "
                    "Model may have drifted significantly from old policy.",
                    _ratio_mean,
                )

        # Dump training metrics if enabled (DataDumper)
        if self._data_dumper.should_dump("training_metrics", self._train_step):
            self._data_dumper.dump_training_metrics(
                step=self._train_step,
                metrics=training_metrics,
            )

        # Stop the training timer before any sampler export or resumable checkpoint.
        metrics["timing_train/batch_training_seconds"] = time.monotonic() - training_started

        # Step 5: Export weights; WeightSyncController decides when to publish them. Name
        # weights after the completed round. Filtered rounds return before this
        # point, so gaps in weight names are expected.
        completed_step = self._train_step + 1
        sampling_path = None
        weight_version = None
        if self._need_sampling and self._weight_sync_controller.sync_mode != "no-sync":
            weight_version = format_weight_version(ACTOR_MODEL_TAG, completed_step)
            export_started = time.monotonic()
            try:
                sampling_path = await await_coro(
                    self._service_holder.save_weights_for_sampler(name=weight_version)
                )
            except Exception:
                # train() cannot return its metrics after an export failure. Publish
                # a separate failure sample without masking the original error.
                failed_time = time.monotonic() - export_started
                try:
                    self._activity_tracker.experiment_logger_post(
                        backend="wandb",
                        data={
                            "timing_weights/weight_export_failed_seconds": failed_time,
                            "training/weight_export_success_flag": 0,
                        },
                        step=completed_step,
                    )
                except Exception:
                    logger.warning("Failed to publish weight export failure metrics", exc_info=True)
                raise
            metrics["timing_weights/weight_export_seconds"] = time.monotonic() - export_started
            metrics["training/weight_export_success_flag"] = 1

            logger.debug(
                f"Training step {completed_step} completed, "
                f"exported sampling path: {sampling_path}"
            )
        else:
            logger.info(
                "Training step %s completed without publishing sampler weights",
                completed_step,
            )

        # Periodic full-checkpoint save for resume support.
        # Weaver checkpoint paths are server-managed; we surface the
        # returned path via logs and metrics so users can plug it into
        # ``resume.resume_path`` for a future run.
        save_freq = int(self._config.training.save.every_n_steps or 0)
        if save_freq > 0:
            if completed_step % save_freq == 0:
                checkpoint_started = time.monotonic()
                checkpoint_path = await self._save_full_checkpoint(completed_step)
                if checkpoint_path:
                    metrics["timing_weights/checkpoint_seconds"] = (
                        time.monotonic() - checkpoint_started
                    )
                    metrics["training/checkpoint_path"] = checkpoint_path

        # Service observations retain their provenance; catalog filters raw diagnostics.
        metrics.update({f"weaver/{key}": value for key, value in training_metrics.items()})
        for key in ("actor_ref_mean", "actor_ref_loss"):
            if key in training_metrics:
                metrics[f"kl/{key}"] = training_metrics[key]
        if getattr(self, "_kl_enabled", False) and "kl/actor_ref_mean" not in metrics:
            logger.warning(
                "Reference KL unavailable: backend did not provide target-aligned training-forward logprobs or explicit actor_ref_mean; attachment counts are not KL"
            )
        metrics = normalize_step_metrics(metrics)
        logger.info("Training step %d metrics: %s", completed_step, metrics)
        return metrics, sampling_path, weight_version
