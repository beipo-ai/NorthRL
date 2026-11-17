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
RemoteApiCrossEntropyTrainer - Remote API trainer with Cross Entropy loss
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any, Callable, cast

from omegaconf import DictConfig

from ..northrl_types import Trajectory
from ..utils.async_utils import await_coro
from ..utils.finetune_service_utils import convert_trajectories_to_datums_cross_entropy
from ..utils.training_timing import training_stage_metrics
from ..utils.trajectory_sequences import iter_training_sequences, iter_trajectory_sequences
from .remote_api_trainer import RemoteApiTrainer

logger = logging.getLogger(__name__)


class RemoteApiCrossEntropyTrainer(RemoteApiTrainer):
    """Remote API cross-entropy trainer used by SFT.

    Local SFT trajectories contain ground-truth labels via ``loss_mask``.
    Managed trajectories keep labels server-private behind ``SampleRef``.
    Neither variant requires advantage computation.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Remote API Cross Entropy trainer.

        Args:
            config: Configuration dictionary
        """
        if config.algorithm.type != "sft":
            raise ValueError("RemoteApiCrossEntropyTrainer supports only algorithm.type='sft'")
        super().__init__(config)

        # The SFT loader has already formed one optimizer window under this
        # exact shifted-position budget. Reuse it for transport so a normal
        # window is one forward/backward operation. Besides avoiding redundant
        # service round trips, this lets Weaver compute one global loss
        # denominator across every Datum and every DP rank in the window.
        self._sft_transport_position_budget = int(config.data.max_sequence_length) * int(
            config.data.train.batch_size
        )

        # SFT uses Weaver's built-in cross-entropy loss. Local datums provide
        # targets/masks; managed SampleRefs receive server-owned targets/masks
        # during server-side materialization.
        self._loss_fn = "cross_entropy"
        self._last_datums: list[dict] = []
        self._last_datum_positions: list[int] = []
        # Cross-entropy never consumes policy logprobs, so a trainer-side
        # recompute would be pure overhead and another failure mode.
        self._sampling_logprob_source = "inference"

        logger.info(f"RemoteApiCrossEntropyTrainer initialized with loss_fn={self._loss_fn}")

    async def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories for cross entropy training.

        For supervised learning with cross entropy, no trajectory preparation
        (like advantage computation) is needed. For local trajectories the
        loss_mask indicates which tokens contribute; managed masks stay in
        Weaver.

        Args:
            trajectories: Local-token or managed-SampleRef SFT trajectories
            metrics: Dictionary to store preparation metrics

        Returns:
            Original trajectories, ready for cross-entropy conversion
        """
        metrics["training/window_shifted_tokens"] = sum(
            self._trajectory_input_token_count(trajectory) for trajectory in trajectories
        )
        return trajectories

    def _build_loss_fn_config(self) -> dict[str, Any]:
        """Let Weaver aggregate CE: there are no advantages to pre-normalize for ``sum``."""
        config = super()._build_loss_fn_config()
        config["loss_agg_mode"] = self._loss_agg_mode
        return config

    @staticmethod
    def _sequence_input_token_count(sequence: Trajectory | dict[str, Any]) -> int:
        """Return one sequence's causal-shifted input length (``len(tokens) - 1``)."""

        count = sequence.get("input_token_count")
        if count is not None:
            return int(count)
        tokens = sequence["tokens"]
        assert tokens is not None
        return len(tokens) - 1

    @staticmethod
    def _trajectory_input_token_count(trajectory: Trajectory) -> int:
        return sum(
            RemoteApiCrossEntropyTrainer._sequence_input_token_count(sequence)
            for sequence in iter_trajectory_sequences(trajectory)
        )

    def _split_into_batches(self, datums_data: list[dict]) -> list[list[dict]]:
        # The SFT loader already formed exactly one optimizer window.
        return [datums_data]

    def _split_into_mini_batches(self, datums_data: list[dict]) -> list[list[dict]]:
        """Split one optimizer window into stable, whole-Datum transports."""

        positions = self._last_datum_positions
        if len(positions) != len(datums_data) or any(
            datum is not converted for datum, converted in zip(datums_data, self._last_datums)
        ):
            raise RuntimeError("SFT transport split requires the converted optimizer window")

        transport_batches: list[list[dict]] = []
        current: list[dict] = []
        current_positions = 0
        for datum, datum_positions in zip(datums_data, positions):
            if (
                current
                and current_positions + datum_positions > self._sft_transport_position_budget
            ):
                transport_batches.append(current)
                current = []
                current_positions = 0
            current.append(datum)
            current_positions += datum_positions
        if current:
            transport_batches.append(current)

        logger.info(
            "[train] SFT optimizer window split: datums=%d shifted_positions=%d "
            "transport_batches=%d transport_budget=%d",
            len(datums_data),
            sum(positions),
            len(transport_batches),
            self._sft_transport_position_budget,
        )
        return transport_batches

    def _aggregate_optimizer_window_metrics(
        self,
        forward_backward_metrics: list[dict[str, Any]],
        optim_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one metric set without presenting an inexact loss as pooled."""

        if len(forward_backward_metrics) == 1:
            single_metrics = dict(forward_backward_metrics[0])
            single_metrics.update(optim_metrics)
            single_metrics["num_transport_batches"] = 1
            return single_metrics

        combined: dict[str, Any] = {
            "num_mini_batches": len(forward_backward_metrics),
            "num_transport_batches": len(forward_backward_metrics),
            "loss_aggregation_exact": 0,
        }
        denominators = self._numeric_metric_values(
            forward_backward_metrics,
            "loss_agg_denominator",
        )
        if denominators is not None:
            total_denominator = sum(denominators)
            if total_denominator > 0:
                if self._loss_agg_mode == "token-mean":
                    numerators = self._numeric_metric_values(
                        forward_backward_metrics,
                        "loss:sum",
                    )
                    if numerators is not None:
                        total_numerator = sum(numerators)
                        combined["loss"] = total_numerator / total_denominator
                        combined["loss:sum"] = total_numerator
                        combined["loss_agg_denominator"] = total_denominator
                        combined["loss_aggregation_exact"] = 1
                else:
                    losses = self._numeric_metric_values(
                        forward_backward_metrics,
                        "loss",
                    )
                    if losses is not None:
                        total_numerator = sum(
                            loss * denominator for loss, denominator in zip(losses, denominators)
                        )
                        combined["loss"] = total_numerator / total_denominator
                        combined["loss:sum"] = total_numerator
                        combined["loss_agg_denominator"] = total_denominator
                        combined["loss_aggregation_exact"] = 1
        if not combined["loss_aggregation_exact"]:
            transport_losses = [
                metrics["loss"]
                for metrics in forward_backward_metrics
                if isinstance(metrics.get("loss"), (int, float))
            ]
            if transport_losses:
                combined["transport_batch_loss_mean"] = sum(transport_losses) / len(
                    transport_losses
                )

        compute_times = [
            metrics["compute_ms"]
            for metrics in forward_backward_metrics
            if isinstance(metrics.get("compute_ms"), (int, float))
        ]
        if compute_times:
            combined["compute_ms"] = sum(compute_times)
        combined.update(optim_metrics)
        return combined

    @staticmethod
    def _numeric_metric_values(
        metrics_list: list[dict[str, Any]],
        key: str,
    ) -> list[float] | None:
        values: list[float] = []
        for metrics in metrics_list:
            value = metrics.get(key)
            if not isinstance(value, (int, float)):
                return None
            values.append(float(value))
        return values

    async def _run_optimizer_batch(
        self,
        batch: list[dict],
        mini_batches: list[list[dict]],
        loss_fn_config: dict[str, Any] | None,
        *,
        token_counts: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Queue the SFT backwards, collect all, then update once per window."""

        del batch, token_counts
        assert self._service_holder is not None, "ServiceHolder not set"
        service_holder = self._service_holder
        forward_backward_metrics: list[dict[str, Any]] = []
        # (handle, datums, loss_fn_config) for every accepted submission.
        pending: list[tuple[Any, list[dict], dict[str, Any] | None]] = []

        def collect(item: tuple[Any, list[dict], dict[str, Any] | None]) -> Any:
            handle, datums_data, mb_loss_fn_config = item
            return await_coro(
                service_holder.collect_forward_backward_result(
                    handle, datums_data=datums_data, loss_fn_config=mb_loss_fn_config
                )
            )

        forward_backward_start = time.perf_counter()
        try:
            for mini_batch_idx, mb_datums in enumerate(mini_batches, start=1):
                mb_loss_fn_config = self._loss_fn_config_for_microbatch(
                    mb_datums,
                    loss_fn_config,
                )
                logger.info(
                    "[train] Sending SFT transport batch to Weaver: train_step=%d "
                    "transport_batch=%d/%d num_datums=%d",
                    self._train_step + 1,
                    mini_batch_idx,
                    len(mini_batches),
                    len(mb_datums),
                )
                handle = await await_coro(
                    service_holder.submit_forward_backward(
                        datums_data=mb_datums,
                        loss_fn=self._loss_fn,
                        loss_fn_config=mb_loss_fn_config,
                    )
                )
                pending.append((handle, mb_datums, mb_loss_fn_config))
        except BaseException:
            # A later HTTP submission can fail after earlier operations were
            # accepted. Drain those handles before propagating the submission
            # error so no completed operation is abandoned by the client.
            await asyncio.gather(*[collect(item) for item in pending], return_exceptions=True)
            raise

        # All forward/backward operations are now queued in stable seq_id
        # order. Collect every result before submitting the optimizer step, so
        # a failed transport cannot cause a partial-gradient update.
        collected = await asyncio.gather(
            *[collect(item) for item in pending], return_exceptions=True
        )
        forward_backward_wall_seconds = time.perf_counter() - forward_backward_start
        failures = [result for result in collected if isinstance(result, BaseException)]
        if failures:
            raise failures[0]

        for result in collected:
            assert isinstance(result, dict)
            result_metrics = result.get("metrics", {})
            metrics = dict(result_metrics) if isinstance(result_metrics, dict) else {}
            if "loss" not in metrics and isinstance(result.get("loss"), (int, float)):
                metrics["loss"] = result["loss"]
            forward_backward_metrics.append(metrics)

        effective_learning_rate = self._effective_learning_rate()
        optimizer_step_start = time.perf_counter()
        optim_result = await await_coro(
            service_holder.optim_step(
                learning_rate=effective_learning_rate,
                beta1=self._beta1,
                beta2=self._beta2,
                weight_decay=self._weight_decay,
                eps=self._eps,
                grad_clip_norm=self._grad_clip_norm,
            )
        )
        optimizer_step_wall_seconds = time.perf_counter() - optimizer_step_start
        raw_optim_metrics = optim_result.get("metrics", {})
        optim_metrics = dict(raw_optim_metrics) if isinstance(raw_optim_metrics, dict) else {}
        if not bool(optim_metrics.get("update_successful", False)):
            raise RuntimeError(
                "Weaver SFT optimizer update was not successful; refusing to commit the data cursor"
            )

        window_metrics = self._aggregate_optimizer_window_metrics(
            forward_backward_metrics,
            optim_metrics,
        )
        window_metrics["_training_timing"] = training_stage_metrics(
            forward_backward_metrics=forward_backward_metrics,
            optimizer_metrics=optim_metrics,
            forward_backward_wall_seconds=forward_backward_wall_seconds,
            optimizer_step_wall_seconds=optimizer_step_wall_seconds,
        )
        window_metrics["learning_rate"] = effective_learning_rate
        window_metrics["lr_schedule_step"] = self._train_step
        if self._lr_schedule_total_steps is not None:
            window_metrics["lr_schedule_total_steps"] = self._lr_schedule_total_steps
        return window_metrics

    def _on_optimizer_step_succeeded(self, trajectories: list[Trajectory]) -> None:
        batch_ids = {trajectory.get("sft_batch_id") for trajectory in trajectories}
        if len(batch_ids) != 1:
            raise RuntimeError(f"SFT optimizer step mixed token-budget batches: {batch_ids!r}")
        batch_id = next(iter(batch_ids))
        if not isinstance(batch_id, str):
            raise RuntimeError(f"SFT optimizer step has invalid batch ID: {batch_id!r}")
        commit_batch = getattr(self._dataloader, "commit_batch", None)
        release_next_batch = getattr(self._dataloader, "release_next_batch", None)
        if not callable(commit_batch) or not callable(release_next_batch):
            raise RuntimeError("SFT trainer requires a cursor-aware streaming dataloader")
        cast(Callable[[str], None], commit_batch)(batch_id)  # pylint: disable=not-callable
        cast(Callable[[], None], release_next_batch)()  # pylint: disable=not-callable

    def _convert_to_datums(self, trajectories: list[Trajectory]) -> list[dict]:
        """
        Convert trajectories to datums for cross-entropy loss.

        Local datums use target_tokens/loss_mask. Managed datums preserve the
        SampleRef variant and any attached auxiliary loss inputs. The
        model_input, target_tokens, loss_mask, and weights fields remain
        server-owned.
        """
        self._last_datum_sources = []
        datums_data = convert_trajectories_to_datums_cross_entropy(
            trajectories, datum_sources=self._last_datum_sources
        )
        sequences_by_parent = [list(iter_training_sequences([parent])) for parent in trajectories]
        self._last_datums = list(datums_data)
        self._last_datum_positions = [
            self._sequence_input_token_count(sequences_by_parent[parent_index][segment_index])
            for parent_index, segment_index in self._last_datum_sources
        ]
        return datums_data
