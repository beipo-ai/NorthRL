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

"""Merge service observations without averaging extrema or unknown reductions.

Private weights are client observations. Native PPO/IS means require explicit
metric_weights from the service; tokens and is:count do not establish their mask.
Unknown service diagnostics remain in ordinary logs.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, TypeGuard, cast

logger = logging.getLogger(__name__)


def _number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _weight(record: dict, key: str) -> float | None:
    weights = record.get("metric_weights", {})
    if isinstance(weights, Mapping) and _number(weights.get(key)):
        return float(weights[key]) if weights[key] > 0 else None
    if key.startswith(("ratio:", "logprob_diff:")):
        value = record.get("_response_tokens", record.get("tokens"))
    elif key in {"loss", "actor_ref_loss"}:
        value = record.get("_loss_weight", record.get("tokens"))
    elif key == "actor_ref_mean":
        value = record.get("_ref_tokens")
    else:
        return None
    return float(value) if _number(value) and value > 0 else None


def aggregate_service_metrics(
    records: list[dict[str, Any]], *, accumulate: bool = False
) -> dict[str, Any]:
    """Sum native FB losses; weight optimizer objectives by their reduction unit.

    A service-native mean is exported across multiple results only when every
    result provides its explicit weight. Custom-loss token summaries have a
    known mask contract. Policy entropy carries its own sum and effective-token
    count, independent of the loss reduction. Missing observations never become zero.
    """
    if not records:
        return {}
    if len(records) == 1:
        return dict(records[0])
    result: dict[str, Any] = {}
    # Keep each numerator paired with its own denominator. Incomplete/nonfinite
    # observations must not contribute just one half of the pooled statistic.
    entropy_records = [
        record
        for record in records
        if _number(record.get("entropy:sum"))
        and _number(record.get("entropy:tokens"))
        and record["entropy:tokens"] >= 0
        and (record["entropy:tokens"] > 0 or record["entropy:sum"] == 0)
    ]
    if entropy_records:
        entropy_sum = sum(record["entropy:sum"] for record in entropy_records)
        entropy_tokens = sum(record["entropy:tokens"] for record in entropy_records)
        result.update(
            {
                "entropy": entropy_sum / entropy_tokens if entropy_tokens > 0 else 0.0,
                "entropy:sum": entropy_sum,
                "entropy:tokens": entropy_tokens,
            }
        )
    keys = set().union(*(record.keys() for record in records)) - {
        "entropy",
        "entropy:sum",
        "entropy:tokens",
    }
    additive = {
        "tokens",
        "num_samples",
        "_response_tokens",
        "_loss_weight",
        "_ref_tokens",
        "loss_agg_denominator",
        "gspo/active_trajectories",
        "gspo/empty_trajectories",
    }
    for key in keys:
        observed = [record for record in records if _number(record.get(key))]
        if not observed:
            continue
        values = [record[key] for record in observed]
        if key in additive or (accumulate and key == "loss"):
            result[key] = sum(values)
        elif key.endswith((":min", "_min")):
            result[key] = min(values)
        elif key.endswith((":max", "_max")):
            result[key] = max(values)
        elif key == "grad_norm":
            # One observation per successful optimizer update.
            result[key] = sum(values) / len(values)
        elif key in {"entropy_coeff", "update_successful"} and len(set(values)) == 1:
            result[key] = values[0]
        elif key.endswith((":std", "_std")):
            # Reconstructed below from sufficient statistics, never mean(std).
            continue
        else:
            weights = [_weight(record, key) for record in observed]
            if all(weight is not None for weight in weights):
                valid_weights = cast(list[float], weights)
                total = sum(valid_weights)
                result[key] = (
                    sum(value * weight for value, weight in zip(values, valid_weights)) / total
                )
                result.setdefault("metric_weights", {})[key] = total
            else:
                logger.debug("Omit service aggregate without reduction contract: %s", key)
    for std_key, mean_key, square_key in (
        ("ratio:std", "ratio:mean", None),
        ("logprob_diff:std", "logprob_diff:mean", None),
        ("is:logratio_std", "is:logratio", "is:logratio_sq"),
    ):
        if mean_key not in result:
            continue
        observed = [record for record in records if _number(record.get(mean_key))]
        weights = [_weight(record, mean_key) for record in observed]
        if not all(weight is not None for weight in weights):
            continue
        valid_weights = cast(list[float], weights)
        seconds = []
        for record in observed:
            if square_key and _number(record.get(square_key)):
                seconds.append(record[square_key])
            elif _number(record.get(std_key)):
                seconds.append(record[std_key] ** 2 + record[mean_key] ** 2)
            else:
                break
        if len(seconds) == len(observed):
            second = sum(value * weight for value, weight in zip(seconds, valid_weights)) / sum(
                valid_weights
            )
            result[std_key] = math.sqrt(max(0.0, second - result[mean_key] ** 2))
    return result


def reference_kl_from_outputs(
    result: Any, datums: list[dict], config: dict | None
) -> dict[str, float]:
    """Observe the configured reference-KL estimator on actual training forwards.

    Only a complete target-aligned actor logprob response is accepted. This is
    a token mean diagnostic, not an assertion about the backend's loss scaling.
    The loss contribution is exported only via an explicit actor_ref_loss field.
    """
    if not config or not config.get("kl_coeff") or not isinstance(result, dict):
        return {}
    payload = result.get("result", result)
    if not isinstance(payload, dict):
        return {}
    outputs = payload.get("loss_fn_outputs")
    if not isinstance(outputs, list) or len(outputs) != len(datums):
        return {}
    estimator = config.get("kl_estimator", "k3")
    if estimator not in {"k1", "k2", "k3"}:
        return {}
    values = []
    for datum, output in zip(datums, outputs):
        if not isinstance(output, dict):
            return {}
        logprobs = output.get("logprobs", output.get("Logprobs"))
        if isinstance(logprobs, dict):
            logprobs = logprobs.get("data")
        inputs = datum["loss_fn_inputs"]
        reference, mask = inputs.get("ref_logprobs"), inputs.get("loss_mask")
        if (
            not isinstance(logprobs, list)
            or reference is None
            or mask is None
            or len(logprobs) != len(reference)
            or len(mask) != len(reference)
        ):
            return {}
        for actor, ref, active in zip(logprobs, reference, mask):
            if not active:
                continue
            if not _number(actor) or not _number(ref):
                return {}
            difference = actor - ref
            try:
                value = (
                    difference
                    if estimator == "k1"
                    else (
                        difference**2 / 2
                        if estimator == "k2"
                        else math.expm1(-difference) + difference
                    )
                )
            except OverflowError:
                return {}
            if not math.isfinite(value):
                return {}
            values.append(value)
    return (
        {"actor_ref_mean": sum(values) / len(values), "_ref_tokens": len(values)} if values else {}
    )
