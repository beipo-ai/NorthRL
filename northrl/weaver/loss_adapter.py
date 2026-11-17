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

"""Translate the selected objective into Weaver's existing loss protocol."""

from __future__ import annotations

import logging
from array import array
from collections.abc import Sequence
from typing import Any

import numpy as np

from northrl.algorithm.reductions import get_loss_denominators
from northrl.utils.config_schema import AlgorithmConfig
from northrl.utils.step_metrics import datum_response_token_count

# Names expected by existing trainer reductions and the Weaver protocol.
RUNTIME_REDUCTION_NAMES = {
    "token_mean": "token-mean",
    "datum_mean": "sample-mean",
    "trajectory_mean": "trajectory-mean",
    "prompt_mean": "prompt-mean",
    "fixed_response_budget": "dr-grpo",
    "sum": "sum",
}


class LossAdapter:
    """Translate validated loss configuration; service I/O stays in the trainer."""

    def __init__(self, algorithm: AlgorithmConfig):
        self._algorithm = algorithm

    @property
    def backend_name(self) -> str | None:
        loss = self._algorithm.loss
        if loss.mode == "custom":
            return loss.params["base_loss_fn"]
        if loss.mode == "native":
            return loss.params["name"]
        return loss.mode

    @property
    def clip(self) -> dict[str, Any]:
        return dict(self._algorithm.loss.params.get("clip", {}))

    @property
    def custom(self) -> dict[str, Any] | None:
        if self._algorithm.loss.mode == "cispo":
            clip = self.clip
            return {
                "enabled": True,
                "name": "cispo",
                "params": {
                    "clip_ratio": clip["ratio"],
                    "clip_ratio_low": clip["ratio_low"],
                    "clip_ratio_high": clip["ratio_high"],
                    "entropy_coeff": 0.0,
                    "loss_agg_mode": "token-mean",
                },
            }
        if self._algorithm.loss.mode == "gspo":
            clip = self.clip
            return {
                "enabled": True,
                "name": "gspo",
                "params": {
                    "clip_ratio_low": (
                        clip["ratio_low"] if clip["ratio_low"] is not None else clip["ratio"]
                    ),
                    "clip_ratio_high": (
                        clip["ratio_high"] if clip["ratio_high"] is not None else clip["ratio"]
                    ),
                },
            }
        if self._algorithm.loss.mode != "custom":
            return None
        return {key: self._algorithm.loss.params[key] for key in ("name", "params")}

    @property
    def runtime_reduction(self) -> str:
        """Existing trainer/service labels for the canonical reduction."""
        return RUNTIME_REDUCTION_NAMES[self._algorithm.loss.reduction.mode]

    def build_loss_fn_config(self, *, enable_debug_dump: bool) -> dict[str, Any]:
        """Translate native clip options and request sum for normalized inputs.

        Trainer subclasses append objective-specific options. Unset parameters
        are omitted, not sent null.
        """
        config: dict[str, Any] = {
            "entropy_coeff": self._algorithm.entropy_coeff,
            "enable_debug_dump": enable_debug_dump,
            "loss_agg_mode": "sum",
        }
        clip = self.clip
        for name, key in (
            ("truncation_c", "truncation_c"),
            ("ratio", "clip_ratio"),
            ("ratio_low", "clip_ratio_low"),
            ("ratio_high", "clip_ratio_high"),
            ("ratio_c", "clip_ratio_c"),
        ):
            value = clip.get(name)
            if value is not None:
                config[key] = value
        return config


def build_loss_adapter(algorithm: AlgorithmConfig) -> LossAdapter:
    """Construct the objective-to-service adapter for one trainer instance."""
    return LossAdapter(algorithm)


def normalize_advantages_for_sum(
    datums: list[dict],
    *,
    loss_fn: str | None,
    mode: str,
    token_counts: Sequence[int] | None = None,
    logger: logging.Logger | None = None,
    train_step: int | None = None,
) -> list[dict]:
    """Implement supported policy reductions through Weaver's sum protocol.

    The caller supplies a complete optimizer batch, after padding and before
    request splitting. Prompt/fixed-budget metadata was attached before Datum
    filtering. Other objectives retain their own aggregation semantics.

    Padding can repeat a Datum. Each occurrence gets a separate advantage
    buffer; tokens, masks and probability arrays stay shared and unmodified.
    """
    if loss_fn not in {"importance_sampling", "ppo_clip", "score_centering"}:
        return datums
    if token_counts is None:
        token_counts = [datum_response_token_count(datum) for datum in datums]
    denominators = get_loss_denominators(datums, token_counts, mode)
    if denominators is None:
        return datums

    normalized = []
    for datum, denominator in zip(datums, denominators, strict=True):
        inputs = dict(datum["loss_fn_inputs"])
        advantages = inputs["advantages"]
        if (
            type(advantages) is array  # pylint: disable=unidiomatic-typecheck
            and advantages.typecode == "d"
            and type(denominator) is int  # pylint: disable=unidiomatic-typecheck
            and 1 <= denominator <= 2**53
        ):
            scaled = array("d", [0.0]) * len(advantages)
            # Retain division and Python float behavior for underflow/NaNs.
            with np.errstate(all="ignore"):
                np.divide(np.asarray(advantages), denominator, out=np.asarray(scaled))
            inputs["advantages"] = scaled
        else:
            values = (adv / denominator for adv in advantages)
            inputs["advantages"] = (
                array("d", values) if isinstance(advantages, array) else list(values)
            )
        normalized.append({**datum, "loss_fn_inputs": inputs})

    if logger is not None:
        logger.info(
            "[advantage_normalization] done: train_step=%s mode=%s num_datums=%d "
            "valid_tokens=%d denominator_min=%d denominator_max=%d",
            train_step,
            mode,
            len(normalized),
            sum(token_counts),
            min(denominators, default=0),
            max(denominators, default=0),
        )
    return normalized
