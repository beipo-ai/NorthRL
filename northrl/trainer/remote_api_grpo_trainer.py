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
RemoteApiGrpoTrainer - Remote API trainer with GRPO algorithm.

Optionally adds a KL term against a reference policy. Gating rules:

  * ``algorithm.use_kl`` is the single switch.
  * When ``use_kl=true`` the trainer requires a ``role: ref_model`` entry in
    ``train_service`` and creates a Weaver ``TrainingClient`` for it.
  * ``use_kl=false`` (or unset) with a ``ref_model`` entry present is a
    config conflict and is rejected fail-fast — silent fallthrough turned
    out to be a footgun (no kl/* metrics, no obvious reason).

Note that ``ref_model`` is **not** the same role as OPD's ``teacher``.
OPD's teacher is a stronger model being distilled into the student, and is
read by ``RemoteApiOpdTrainer`` from ``trainer.teachers``. The KL ref model
here is a frozen (or periodically refreshed) copy of the policy used to
constrain how far the actor can drift; it has nothing to do with OPD.

Ref logprobs come from a single batched
``training_client.forward(datums, "forward_logprob", wait=False)`` call
(async in nex-weaver v1.9.0). When ``algorithm.actor_to_ref_sync_freq > 0``
the actor's weights are periodically loaded into the ref via ``load_state``.
"""

import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..algorithm.advantage import prepare_grpo_trajectories
from ..algorithm.reductions import attach_fixed_response_budget_denominator
from ..northrl_types import Trajectory
from ..utils.async_utils import await_coro
from ..utils.finetune_service_utils import convert_trajectories_to_datums
from ..utils.trajectory_alignment import align_target_logprobs_to_tokens
from ..utils.trajectory_sequences import iter_training_sequences
from ..weaver.weaver_service_holder import ForwardLogprobsResult, WeaverServiceHolder
from .remote_api_trainer import RemoteApiTrainer
from .router_replay_handler import RouterReplayHandler

logger = logging.getLogger(__name__)


class RemoteApiGrpoTrainer(RemoteApiTrainer):
    """
    Remote API trainer with GRPO (Group Relative Policy Optimization) advantage
    computation, plus an opt-in KL term against a reference policy.

    Gating: ``algorithm.use_kl`` is the single switch.
      * ``use_kl=true`` requires a ``role: ref_model`` entry in
        ``train_service``; missing → fail-fast.
      * ``use_kl=false`` with a ``ref_model`` entry present is rejected as a
        config conflict (rather than silently ignoring the entry).

    The KL path currently requires the Weaver backend (it relies on
    ``training_client.forward`` from nex-weaver >= 1.9.0).
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Remote API GRPO trainer.

        Args:
            config: Root NorthRLConfig. Reads ``algorithm.kl.*`` and
                ``models.reference`` for the KL ref model (when enabled).
                Cross-block schema validators already ensure
                ``algorithm.kl.enabled <=> models.reference`` is consistent.
        """
        super().__init__(config)

        # One source for actor forward and backward; validation sampling is independent.
        self._training_temperature = float(
            OmegaConf.select(config, "rollout.sampling.temperature", default=1.0)
        )
        if not math.isfinite(self._training_temperature) or self._training_temperature < 0:
            raise ValueError("rollout.sampling.temperature must be finite and >= 0")
        logger.info(
            "GRPO training temperature=%s (source=rollout.sampling.temperature)",
            self._training_temperature,
        )

        algorithm_config = OmegaConf.create(self._algorithm_config.model_dump())

        # Router replay: opt-in MoE routing-index transport. The handler keeps
        # immutable config and owns the trajectory/datum metadata boundaries.
        self._router_replay = RouterReplayHandler(algorithm_config)

        kl = self._algorithm_config.kl
        self._kl_enabled = bool(kl.enabled)
        self._kl_mode = kl.mode  # "in_loss" or "in_reward"
        self._kl_coeff = float(kl.params.coef)
        self._kl_estimator = str(kl.params.estimator)
        self._actor_to_ref_sync_freq = int(kl.actor_to_ref_sync_freq)
        # 0 means "single batch": send every trajectory to the ref model in one
        # forward call. Set to a positive int to chunk if the batch is too
        # large for one server-side forward.
        self._ref_logprobs_batch_size = int(kl.ref_logprobs_batch_size)

        # Schema validators have already enforced the pairing rules, so we just
        # carry the ref model spec (may be None when kl.mode=null).
        self._ref_model_cfg = config.models.reference

        # in_reward mode requires per-token reward editing in
        # ``_prepare_trajectories``; that path isn't implemented yet (see
        # CONFIG_REFACTOR.md "KL 两种模式" — in_loss is the current Weaver path).
        if self._kl_enabled and self._kl_mode == "in_reward":
            raise NotImplementedError(
                "algorithm.kl.mode='in_reward' is not yet implemented; use 'in_loss' "
                "(see CONFIG_REFACTOR.md KL section for the planned advantage-rewrite path)."
            )

        if self._kl_enabled and self._kl_coeff <= 0.0:
            logger.warning(
                "[KL] KL is enabled but algorithm.kl.params.coef=%.4g; the KL term will "
                "have no effect on the loss (ref_logprobs are still computed and "
                "shipped for diagnostics).",
                self._kl_coeff,
            )

        # Created in initialize_workers via WeaverServiceHolder async factories.
        self._ref_training_client: Any = None

        logger.info(
            "RemoteApiGrpoTrainer initialized: kl_enabled=%s kl_mode=%s kl_coef=%g "
            "kl_estimator=%s actor_to_ref_sync_freq=%d",
            self._kl_enabled,
            self._kl_mode,
            self._kl_coeff,
            self._kl_estimator,
            self._actor_to_ref_sync_freq,
        )

    # ------------------------------------------------------------------ #
    # Service holder + ref model creation
    # ------------------------------------------------------------------ #
    def set_service_holder(self, service_holder) -> None:
        """Store the holder and validate that the configured KL path is supported."""
        super().set_service_holder(service_holder)

        if not self._kl_enabled:
            return

        if not isinstance(service_holder, WeaverServiceHolder):
            raise NotImplementedError(
                "algorithm.kl.mode is currently only supported with the Weaver backend "
                "(the KL path uses training_client.forward, added in nex-weaver 1.7.0)."
            )

    async def initialize_workers(self) -> None:
        """Create the optional reference model on the Weaver SDK event loop."""
        await super().initialize_workers()
        if not self._kl_enabled:
            return
        if self._ref_training_client is not None:
            logger.info("[KL] Ref-model TrainingClient is already initialized")
            return

        service_holder = self._service_holder
        assert isinstance(service_holder, WeaverServiceHolder)
        assert self._ref_model_cfg is not None  # guaranteed by __init__ fail-fast
        ref_cfg = self._ref_model_cfg

        # In the new schema only `name` (HF/weaver model name) and `checkpoint`
        # are user-facing on models.reference. tokenizer / lora / training_mode
        # are intentionally not exposed — refs share the actor's service and
        # never train. Default name = actor.name (same architecture).
        # pylint: disable=protected-access
        holder_base_model = service_holder._base_model
        holder_lora_rank = service_holder._lora_rank
        # pylint: enable=protected-access
        base_model = ref_cfg.name or holder_base_model

        ref_training_client = await await_coro(
            service_holder.create_training_client(
                base_model=base_model,
                training_mode="lora",
                lora_config={"rank": holder_lora_rank},
            )
        )
        logger.info(
            "[KL] Ref-model TrainingClient created via Weaver service: "
            "base_model=%s lora_rank=%s model_id=%s",
            base_model,
            holder_lora_rank,
            getattr(ref_training_client, "model_id", None),
        )

        # Optionally seed ref weights from a checkpoint. Useful when the
        # reference policy is a separately-trained SFT checkpoint rather than
        # the actor's own initial weights.
        ref_checkpoint = ref_cfg.checkpoint
        if ref_checkpoint:
            logger.info("[KL] Loading ref-model checkpoint: %s", ref_checkpoint)
            await await_coro(ref_training_client.load_state(ref_checkpoint))

        self._ref_training_client = ref_training_client

    _uses_advantage_estimator = True

    async def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """Order trajectories by task, group and run ID; check rollouts before I/O."""
        from ..utils.logging_utils import log_logprob_metrics

        trajectories = prepare_grpo_trajectories(trajectories)
        list(iter_training_sequences(trajectories))
        # These diagnostics historically observe rollout scores, before actor
        # recompute replaces old_log_probs. Estimation does not own this I/O.
        log_logprob_metrics(trajectories, metrics)
        return trajectories

    async def _prepare_advantage_inputs(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> Mapping[str, Any]:
        """Reference scoring is I/O; advantage computation stays in algorithm."""
        if self._kl_enabled:
            metrics["kl/coefficient"] = self._kl_coeff
            await self._compute_and_attach_ref_logprobs(trajectories, metrics)
        return {}

    # ------------------------------------------------------------------ #
    # Ref-model forward: ref logprobs via training_client.forward
    # ------------------------------------------------------------------ #
    async def _compute_and_attach_ref_logprobs(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> None:
        """Run a (possibly chunked) forward pass on the ref model and attach
        token-aligned ``ref_logprobs`` to each original training sequence.

        Uses ``training_client.forward(datums, "forward_logprob", wait=False)``
        — single batched server-side forward, one logprob per target position.
        The returned target-aligned values are stored back on the trajectory as
        ``[0.0] + ref_lp`` so all trajectory-level logprob fields have
        ``len(field) == len(tokens)`` and position 0 is the no-context slot.
        """
        from weaver import (
            types as weaver_types,  # local import keeps weaver optional at module load
        )

        sequences = list(iter_training_sequences(trajectories))
        if self._ref_training_client is None:
            logger.error("[KL] _kl_enabled but _ref_training_client is None; skipping.")
            return
        if not sequences:
            return

        start = time.monotonic()

        # Build a Datum per trajectory: input = tokens[:-1], target = tokens[1:].
        # No advantages / loss_mask needed — forward_logprob only reads target_tokens.
        datums: list[Any] = []
        for traj in sequences:
            tokens = traj["tokens"]
            if len(tokens) < 2:
                # Degenerate sequence — emit a placeholder datum to keep indices aligned.
                datums.append(
                    weaver_types.Datum.from_raw(
                        model_input=weaver_types.ModelInput.from_ints(tokens=tokens),
                        loss_fn_inputs={"target_tokens": tokens},
                    )
                )
                continue
            datums.append(
                weaver_types.Datum.from_raw(
                    model_input=weaver_types.ModelInput.from_ints(tokens=tokens[:-1]),
                    loss_fn_inputs={"target_tokens": tokens[1:]},
                )
            )

        chunk_size = self._ref_logprobs_batch_size
        if chunk_size <= 0 or chunk_size >= len(datums):
            chunks = [datums]
        else:
            chunks = [datums[i : i + chunk_size] for i in range(0, len(datums), chunk_size)]

        all_logprobs: list[list[float]] = []
        for chunk in chunks:
            handle = await await_coro(
                self._ref_training_client.forward(chunk, loss_fn="forward_logprob", wait=False)
            )
            result = await await_coro(handle.result()) if hasattr(handle, "result") else handle
            inner = result.get("result") if isinstance(result, dict) else None
            outputs = (inner or {}).get("loss_fn_outputs", []) if isinstance(inner, dict) else []
            if len(outputs) != len(chunk):
                logger.error(
                    "[KL] ref-model forward returned %d outputs for %d datums; "
                    "skipping this chunk.",
                    len(outputs),
                    len(chunk),
                )
                all_logprobs.extend([[] for _ in chunk])
                continue
            for output in outputs:
                lp = output.get("logprobs") or output.get("Logprobs")
                if isinstance(lp, dict):
                    lp = lp.get("data")
                if lp is None:
                    all_logprobs.append([])
                else:
                    all_logprobs.append([float(x) for x in lp])

        attached = 0
        for traj, ref_lp in zip(sequences, all_logprobs):
            if not ref_lp:
                continue
            try:
                token_aligned_ref_lp = align_target_logprobs_to_tokens(
                    tokens=traj["tokens"],
                    target_logprobs=ref_lp,
                    name="ref_logprobs",
                )
            except ValueError:
                logger.warning(
                    "[KL] ref_logprobs length mismatch: got %d, expected %d. Skipping.",
                    len(ref_lp),
                    max(len(traj["tokens"]) - 1, 0),
                )
                continue
            traj["ref_logprobs"] = token_aligned_ref_lp
            attached += 1

        elapsed = time.monotonic() - start
        metrics["timing/ref_logprobs"] = elapsed
        logger.info(
            "[KL] ref logprobs: attached=%d/%d in %.1fs (chunks=%d)",
            attached,
            len(sequences),
            elapsed,
            len(chunks),
        )

    # ------------------------------------------------------------------ #
    # Actor probability settings and built-in loss configuration.
    # ------------------------------------------------------------------ #
    def _build_forward_loss_fn_config(self) -> dict[str, Any]:
        """Use the rollout temperature for actor logprobs, including an explicit zero."""
        return {"temperature": self._training_temperature}

    def _build_loss_fn_config(self) -> dict[str, Any]:
        config = super()._build_loss_fn_config()
        config.update(self._build_forward_loss_fn_config())
        if self._kl_enabled:
            config["kl_coeff"] = self._kl_coeff
            config["kl_estimator"] = self._kl_estimator
        return config

    # ------------------------------------------------------------------ #
    # Periodic actor -> ref-model weight refresh
    # ------------------------------------------------------------------ #
    async def train(self, trajectories: list[Trajectory]) -> tuple[dict, str | None, str | None]:
        metrics, sampling_path, weight_version = await super().train(trajectories)

        if (
            not bool(metrics.get("training/skipped_flag", 0))
            and self._kl_enabled
            and self._actor_to_ref_sync_freq > 0
            and self._ref_training_client is not None
        ):
            # _train_step is incremented in BaseTrainer._main_loop AFTER train()
            # returns, so the post-increment value is (self._train_step + 1).
            next_step = self._train_step + 1
            if next_step % self._actor_to_ref_sync_freq == 0:
                await self._sync_actor_to_ref(next_step, metrics)

        return metrics, sampling_path, weight_version

    async def _sync_actor_to_ref(self, step: int, metrics: dict) -> None:
        """Save the actor's weights and load them into the ref model (blocking)."""
        del metrics  # Reference synchronization diagnostics stay in ordinary logs.
        assert self._service_holder is not None

        logger.info(
            "[KL] Syncing actor weights to ref model at step %d (freq=%d)",
            step,
            self._actor_to_ref_sync_freq,
        )
        sync_start = time.time()

        checkpoint_path = await await_coro(
            self._service_holder.save_state(name=f"ref_sync_step_{step}")
        )
        logger.info("[KL] Actor checkpoint saved to: %s", checkpoint_path)

        await await_coro(self._ref_training_client.load_state(checkpoint_path))

        sync_time = time.time() - sync_start
        logger.info("[KL] Actor->ref sync completed in %.2fs", sync_time)

    async def _compute_recompute_logprobs_batch(
        self,
        trajectories: list[Trajectory | dict[str, Any]],
        *,
        sampling_masks: list[list[list[int]] | dict[str, Any] | None] | None,
    ) -> ForwardLogprobsResult:
        """Apply the configured router-replay action to the denominator forward."""
        if not self._router_replay.enabled:
            return await super()._compute_recompute_logprobs_batch(
                trajectories, sampling_masks=sampling_masks
            )
        service_holder = self._service_holder
        if service_holder is None:
            raise RuntimeError("Router replay requires an initialized Weaver service holder.")

        tokens_batch = [trajectory["tokens"] for trajectory in trajectories]
        if self._router_replay.mode == "R3":
            datum_metadata_batch = [
                self._router_replay.build_datum_metadata(trajectory, trajectory_index)
                for trajectory_index, trajectory in enumerate(trajectories)
            ]
            logger.info(
                "[router_replay] R3 REPLAY denominator: replaying %d rollout routing refs",
                len(datum_metadata_batch),
            )
            logprobs = await await_coro(
                service_holder.compute_target_logprobs_batch(
                    tokens_batch,
                    sampling_masks=sampling_masks,
                    datum_metadata_batch=datum_metadata_batch,
                    loss_fn_config=self._build_forward_loss_fn_config(),
                )
            )
            return ForwardLogprobsResult(logprobs=logprobs)

        if not hasattr(service_holder, "compute_target_logprobs_batch_with_record"):
            raise NotImplementedError(
                "algorithm.router_replay with mode=R2 requires the Weaver backend."
            )
        # The holder omits sequences with no target tokens from Weaver Datums.
        # Keep R2 manifest indices aligned with those actual Datums, including
        # when minibatch splitting leaves a chunk with no targets at all.
        active_indices = [index for index, tokens in enumerate(tokens_batch) if len(tokens) > 1]
        if not active_indices:
            return ForwardLogprobsResult(logprobs=[[] for _ in trajectories])
        forward_result = await await_coro(
            service_holder.compute_target_logprobs_batch_with_record(
                [tokens_batch[index] for index in active_indices],
                sampling_masks=(
                    [sampling_masks[index] for index in active_indices]
                    if sampling_masks is not None
                    else None
                ),
                loss_fn_config=self._build_forward_loss_fn_config(),
            )
        )
        if not isinstance(forward_result.router_replay_record_ref, dict):
            raise ValueError("[router_replay] R2 recompute did not return a recorded routing ref.")
        r2_logprobs: list[Sequence[float]] = [[] for _ in trajectories]
        for index, values in zip(active_indices, forward_result.logprobs, strict=True):
            r2_logprobs[index] = values
        return ForwardLogprobsResult(
            logprobs=r2_logprobs,
            router_replay_record_ref=forward_result.router_replay_record_ref,
        )

    def _after_recompute_logprobs(
        self,
        trajectories: list[Trajectory | dict[str, Any]],
        forward_result: ForwardLogprobsResult,
        metrics: dict[str, Any],
    ) -> None:
        del metrics
        if self._router_replay.records_during_recompute:
            active_trajectories = [traj for traj in trajectories if len(traj["tokens"]) > 1]
            if active_trajectories:
                self._router_replay.bind_recorded_routing_ref(
                    active_trajectories,
                    forward_result.router_replay_record_ref,
                )

    def _convert_to_datums(self, trajectories: list[Trajectory]) -> list[dict]:
        """Convert GRPO trajectories and attach optional replay metadata."""
        self._last_datum_sources = []
        datums = convert_trajectories_to_datums(
            trajectories,
            datum_sources=self._last_datum_sources,
            metadata_builder=(
                self._router_replay.build_datum_metadata if self._router_replay.enabled else None
            ),
            # Slime keeps constant-reward groups and other zero-advantage
            # samples in the training batch.
            drop_zero_advantage_datums=False,
            prompt_mean=self._loss_agg_mode == "prompt-mean",
        )
        if self._loss_agg_mode == "dr-grpo":
            attach_fixed_response_budget_denominator(
                trajectories,
                datums,
                max_response_tokens=self._algorithm_config.loss.reduction.params[
                    "max_response_tokens"
                ],
            )
        return datums
