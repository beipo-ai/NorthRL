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

"""Loss denominator math, without training-service calls.

Fixed-budget attachment receives the complete repaired rollout step before
request splitting. Denominator lookup receives the datums being normalized;
token and datum means use their final active-token counts in the optimizer batch.
Stored prompt and fixed-budget denominators retain their original population
when tokens are rejected or requests are split.
"""

from __future__ import annotations

from collections.abc import Sequence

from northrl.northrl_types import Trajectory

_FIXED_RESPONSE_BUDGET_DENOMINATOR_KEY = "_dr_grpo_denominator"


def attach_fixed_response_budget_denominator(
    trajectories: Sequence[Trajectory],
    datums: Sequence[dict],
    *,
    max_response_tokens: int,
) -> None:
    """Attach N * L before request splitting, without changing token buffers.

    Count each successful trajectory with group_skipped=False once across the step.
    Empty responses, zero advantages and MIS-rejected trajectories still count;
    segments and surviving response-token counts do not enter this denominator.
    Failure repair and group eligibility must already have been decided by the
    training flow. An empty datum batch stays empty and does not trigger an update.
    """
    if (
        isinstance(max_response_tokens, bool)
        or not isinstance(max_response_tokens, int)
        or max_response_tokens < 1
    ):
        raise ValueError("fixed_response_budget requires positive integer max_response_tokens")
    count = sum(
        trajectory.status == "success" and not trajectory.get("group_skipped", False)
        for trajectory in trajectories
    )
    denominator = count * max_response_tokens
    for datum in datums:
        datum[_FIXED_RESPONSE_BUDGET_DENOMINATOR_KEY] = denominator


def get_fixed_response_budget_denominators(datums: Sequence[dict]) -> list[int]:
    """Read the full-step denominator; never infer it from retained datums.

    Keep each datum's explicit metadata through minibatch requests. Shared
    advantage scaling is performed by the host loss adapter, exactly once.
    """
    denominators = []
    for datum in datums:
        denominator = datum.get(_FIXED_RESPONSE_BUDGET_DENOMINATOR_KEY)
        if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator < 1:
            raise ValueError("dr-grpo requires a positive full-step denominator")
        denominators.append(denominator)
    return denominators


def get_loss_denominators(
    datums: list[dict], token_counts: Sequence[int], mode: str
) -> list[int] | None:
    """Return each Datum's divisor for a canonical reduction mode.

    ``datum_mean`` averages each nonempty Datum first, then those Datum means.
    ``token_mean`` averages over all active tokens in this optimizer batch.
    Complete-prompt and fixed-response-budget counts are supplied as metadata
    because they cannot be recovered from the filtered Datum list.
    ``sum`` requires no scaling and returns ``None``.
    """
    if len(token_counts) != len(datums):
        raise ValueError("Every Datum requires its final response-token count")
    if mode == "fixed_response_budget":
        return get_fixed_response_budget_denominators(datums)
    if mode == "prompt_mean":
        key = "_prompt_mean_denominator"
        denominators = []
        for datum in datums:
            denominator = datum.get(key)
            if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator < 1:
                raise ValueError(f"{mode} requires a positive complete-group denominator")
            denominators.append(denominator)
        return denominators
    if mode == "token_mean":
        return [max(sum(token_counts), 1)] * len(datums)
    if mode == "datum_mean":
        num_samples = sum(count > 0 for count in token_counts)
        return [max(num_samples * count, 1) for count in token_counts]
    if mode == "sum":
        return None
    raise ValueError(f"Unknown loss reduction mode: {mode!r}")
