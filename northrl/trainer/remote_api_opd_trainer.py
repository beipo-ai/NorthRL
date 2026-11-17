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
RemoteApiOpdTrainer - Remote API trainer with On-Policy Distillation

This trainer implements on-policy distillation (OPD) on the weaver remote API.
It creates a teacher weaver client — a sampling client (``inference_service``)
or a training client (``training_service``), mirroring the ref-model handling in
``RemoteApiGrpoTrainer`` — computes the per-token reverse KL between student and
teacher, and adjusts per-token advantages before training with the standard
importance_sampling loss.

Key features:
- Creates a teacher weaver client — ``inference_service`` (sampling) or
  ``training_service`` (training) (``algorithm.opd.teacher_logprob_source``)
- Computes per-token KL advantages (not scalar like GRPO)
- Supports multiple teachers (one per dataset)
"""

import asyncio
import logging
from typing import Any, cast

from omegaconf import DictConfig

from ..algorithm.advantage.types import OPDInputs
from ..northrl_types import Trajectory
from ..utils.async_utils import await_coro
from ..utils.trajectory_alignment import align_target_logprobs_to_tokens
from ..utils.trajectory_sequences import iter_training_sequences
from .remote_api_trainer import RemoteApiTrainer

logger = logging.getLogger(__name__)

# Bound the parallel teacher logprob wait so a hung/dead teacher service crashes
# the step instead of stalling the training loop forever (see issue #205).
# Default is generous enough to absorb teacher service cold-start (provisioning,
# image download, model load, and warmup); configure algorithm.opd.teacher_service_timeout_s.
_DEFAULT_TEACHER_SERVICE_TIMEOUT_S = 1200.0
_MAX_TEACHER_LOGPROB_CONCURRENCY = 32


class RemoteApiOpdTrainer(RemoteApiTrainer):
    """
    Remote API trainer with On-Policy Distillation (OPD) algorithm.

    This adapter prepares teacher scores and freezes the student rollout scores.
    The common train method directly calls the OPD estimator in
    ``northrl.algorithm.advantage`` and stores response-only ``token_advantages``.
    Everything else -- the training step, datum conversion (which already handles
    ``token_advantages``), mini-batch handling, checkpointing -- is inherited from
    ``RemoteApiTrainer``.

    The objective matches tinker-cookbook on-policy distillation: importance
    sampling over sampler (inference) logprobs with no recompute, so the recipe
    must set ``algorithm.sampling_logprob_source: inference``.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Remote API OPD trainer.

        Args:
            config: Root NorthRLConfig. Reads ``algorithm.opd.*`` for the OPD
                hyperparameters and ``models.teachers`` for the teacher list.
                Schema validators already require ``models.teachers`` to be
                non-empty when ``algorithm.type == "opd"``.
        """
        super().__init__(config)

        opd = self._algorithm_config.opd
        params = self._algorithm_config.advantage.estimator.params
        self._kl_penalty_coef = float(params["kl_penalty_coef"])
        self._kl_discount_factor = float(params["kl_discount_factor"])
        self._teacher_service_timeout = float(opd.teacher_service_timeout_s)
        self._teacher_logprob_source = str(opd.teacher_logprob_source)

        # Teacher configurations — schema validator already guarantees non-empty.
        self._teacher_configs = list(config.models.teachers or [])

        # Teacher weaver client(s) (raw SDK clients), created in initialize_workers.
        self._teacher_clients: list[Any] = []

        logger.info("RemoteApiOpdTrainer initialized with:")
        logger.info(f"  kl_penalty_coef={self._kl_penalty_coef}")
        logger.info(f"  kl_discount_factor={self._kl_discount_factor}")
        logger.info(f"  num_teachers={len(self._teacher_configs)}")
        logger.info(f"  teacher_service_timeout={self._teacher_service_timeout:.1f}s")

    def set_service_holder(self, service_holder) -> None:
        """
        Store the service holder. Teacher clients are created by initialize_workers().

        The teacher is a plain weaver SDK client — a training client for
        ``training_service`` (``create_model``) or a sampling client for
        ``inference_service`` (``create_sampling_client``) — mirroring the
        ref-model creation in ``RemoteApiGrpoTrainer``. No wrapper.

        Args:
            service_holder: Service holder instance (WeaverServiceHolder, etc.)
        """
        super().set_service_holder(service_holder)
        if self._teacher_logprob_source not in {"training_service", "inference_service"}:
            raise ValueError(f"unknown teacher_logprob_source: {self._teacher_logprob_source!r}")

    async def initialize_workers(self) -> None:
        """Create teacher clients on the Weaver SDK event loop."""
        await super().initialize_workers()
        if self._teacher_clients:
            logger.info("Teacher clients are already initialized")
            return

        service_holder = self._service_holder
        assert service_holder is not None, "ServiceHolder not set"
        teacher_clients: list[Any] = []
        logger.info(f"Creating {len(self._teacher_configs)} teacher(s)...")
        for i, teacher_cfg in enumerate(self._teacher_configs):
            try:
                if self._teacher_logprob_source == "training_service":
                    client = await await_coro(
                        service_holder.create_training_client(base_model=teacher_cfg.name)
                    )
                    if teacher_cfg.checkpoint:
                        await await_coro(client.load_state(teacher_cfg.checkpoint))
                elif self._teacher_logprob_source == "inference_service":
                    kwargs: dict[str, Any] = {"base_model": teacher_cfg.name}
                    if teacher_cfg.checkpoint:
                        kwargs["model_path"] = teacher_cfg.checkpoint
                    client = await await_coro(service_holder.create_sampling_client(**kwargs))
                else:
                    raise ValueError(
                        f"unknown teacher_logprob_source: {self._teacher_logprob_source!r}"
                    )
                teacher_clients.append(client)
                logger.info(
                    f"Teacher {i+1}/{len(self._teacher_configs)} created "
                    f"({self._teacher_logprob_source}): {teacher_cfg.name}"
                )
            except Exception as e:
                logger.error(f"Failed to create teacher {i}: {e}")
                raise RuntimeError(f"Teacher creation failed: {e}") from e

        self._teacher_clients = teacher_clients
        logger.info(f"All {len(self._teacher_clients)} teacher(s) created successfully")

    async def _compute_teacher_logprobs_parallel(
        self, trajectories: list[Trajectory | dict[str, Any]]
    ) -> list[list[float | None]]:
        """Token-aligned teacher logprobs for all trajectories.

        The two sources are equal alternatives selected by
        ``algorithm.opd.teacher_logprob_source``; each has its own handler.
        """
        if not self._teacher_clients:
            logger.warning("No teacher clients available, returning empty logprobs")
            return [[] for _ in trajectories]

        teacher_client = self._teacher_clients[0]
        if self._teacher_logprob_source == "training_service":
            return await self._teacher_logprobs_from_training_service(teacher_client, trajectories)
        if self._teacher_logprob_source == "inference_service":
            return await self._teacher_logprobs_from_inference_service(teacher_client, trajectories)
        raise ValueError(f"unknown teacher_logprob_source: {self._teacher_logprob_source!r}")

    async def _teacher_logprobs_from_training_service(
        self, teacher_client: Any, trajectories: list[Trajectory | dict[str, Any]]
    ) -> list[list[float | None]]:
        """training_service: ONE batched ``forward_logprob`` over the whole batch.

        Mirrors the ref-model forward in ``RemoteApiGrpoTrainer``.
        """
        # local imports keep weaver optional at module load
        from weaver import types as weaver_types

        from ..weaver.weaver_service_holder import _extract_forward_logprobs_batch_from_result

        results: list[list[float | None]] = [[] for _ in trajectories]
        datums: list[Any] = []
        active: list[int] = []
        for idx, traj in enumerate(trajectories):
            if len(traj["tokens"]) <= 1:
                continue
            datums.append(
                weaver_types.Datum.from_raw(
                    model_input=weaver_types.ModelInput.from_ints(tokens=traj["tokens"][:-1]),
                    loss_fn_inputs={"target_tokens": list(traj["tokens"][1:])},
                )
            )
            active.append(idx)
        if not datums:
            return results

        handle = await await_coro(teacher_client.forward(datums, "forward_logprob", wait=False))
        result = await await_coro(handle.result()) if hasattr(handle, "result") else handle
        batch_logprobs = _extract_forward_logprobs_batch_from_result(result)
        if len(batch_logprobs) != len(datums):
            raise ValueError(
                f"teacher forward_logprob returned {len(batch_logprobs)} outputs "
                f"for {len(datums)} datums"
            )
        for idx, logprobs in zip(active, batch_logprobs):
            results[idx] = cast(
                "list[float | None]",
                align_target_logprobs_to_tokens(
                    tokens=trajectories[idx]["tokens"], target_logprobs=logprobs
                ),
            )
        return results

    async def _teacher_logprobs_from_inference_service(
        self, teacher_client: Any, trajectories: list[Trajectory | dict[str, Any]]
    ) -> list[list[float | None]]:
        """inference_service: one ``compute_logprobs`` (prefill) request per trajectory,
        fanned out asynchronously (the inference server continuous-batches them).

        The timeout bounds the total wait so a hung/dead teacher surfaces as a
        TimeoutError instead of stalling the loop (#205).
        """
        # local import keeps weaver optional at module load
        from weaver import types as weaver_types

        semaphore = asyncio.Semaphore(_MAX_TEACHER_LOGPROB_CONCURRENCY)

        async def _prefill_logprobs(tokens: list[int]) -> list[float | None]:
            async with semaphore:
                model_input = weaver_types.ModelInput.from_ints(tokens=tokens)
                logprobs = await await_coro(teacher_client.compute_logprobs(prompt=model_input))
                return list(logprobs) if logprobs else []

        pending = [_prefill_logprobs(traj["tokens"]) for traj in trajectories]
        gathered = await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=self._teacher_service_timeout,
        )

        results: list[list[float | None]] = []
        for idx, result in enumerate(gathered):
            if isinstance(result, BaseException):
                logger.error(
                    "Teacher logprob computation failed for trajectory %d: %s",
                    idx,
                    result,
                )
                results.append([])
            else:
                results.append(result)
        return results

    _uses_advantage_estimator = True

    async def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """OPD consumes existing sequence masks and preserves parent ordering."""
        return trajectories

    async def _prepare_advantage_inputs(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> OPDInputs:
        """Freeze rollout score references and fetch teacher scores for the step.

        Actor recompute replaces old_log_probs later. OPD retains the original
        student denominator without duplicating token buffers or probability
        arrays. Estimator inputs are local to this train call, never cached on
        the trainer or serialized into rollout data.
        """
        if self._kl_penalty_coef == 0:
            return {}
        sequences = list(iter_training_sequences(trajectories))
        student_logprobs = [sequence.get("old_log_probs", []) for sequence in sequences]
        teacher_logprobs = await self._compute_teacher_logprobs_parallel(sequences)
        return {
            "sequences": sequences,
            "student_logprobs": student_logprobs,
            "teacher_logprobs": teacher_logprobs,
        }
