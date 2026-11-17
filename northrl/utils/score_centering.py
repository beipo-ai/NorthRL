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

"""Preserve behavior distributions independently of recomputed IS logprobs."""

from __future__ import annotations

from typing import Any


def align_sampler_distribution(tokens, loss_mask, distribution: dict[str, Any]):
    """Align a single contiguous response to shifted target positions."""
    from weaver.score_centering import SAMPLER_FIELDS, validate_sampler_sequence

    active = [i for i, value in enumerate(loss_mask) if value]
    if not active or active[0] == 0 or active != list(range(active[0], len(tokens))):
        raise ValueError("score_centering requires a single response after a nonempty prompt")
    n = len(active)
    if "sampler_distribution_ref" in distribution:
        ref = distribution["sampler_distribution_ref"]
        view = distribution.get(
            "sampler_distribution_slice", {"start": 0, "stop": ref["token_count"]}
        )
        start, stop = view.get("start"), view.get("stop")
        if (
            any(not isinstance(bound, int) or isinstance(bound, bool) for bound in (start, stop))
            or not 0 <= start <= stop <= ref["token_count"]
            or stop - start != n
        ):
            raise ValueError(
                "sampler distribution reference range must align with retained response tokens"
            )
        if len(distribution["sampler_logprobs"]) != n:
            raise ValueError("sampler logprobs must align with retained response tokens")
        return {}, {
            "sampler_distribution": distribution["sampler_distribution"],
            "sampler_distribution_ref": dict(ref),
            "sampler_distribution_slice": {"start": start, "stop": stop},
        }
    if any(len(distribution[key]) != n for key in SAMPLER_FIELDS[:-1]):
        raise ValueError("sampler distribution must align with retained response tokens")
    k = len(distribution["sampler_topk_ids"][0])
    validate_sampler_sequence({"tokens": tokens[active[0] :], **distribution}, k)
    prompt_targets = active[0] - 1
    fields = {}
    for key in SAMPLER_FIELDS[:-1]:
        filler = [0] * k if key.startswith("sampler_topk_") else 0.0
        fields[key] = [
            list(filler) if isinstance(filler, list) else filler for _ in range(prompt_targets)
        ] + distribution[key]
    return fields, {"sampler_distribution": distribution["sampler_distribution"]}


def trim_sampler_distribution(distribution, response_length):
    """Keep all fields in lockstep when a simple rollout is truncated."""
    if distribution is None:
        return None
    from weaver.score_centering import SAMPLER_FIELDS

    result = dict(distribution)
    if "sampler_distribution_ref" in distribution:
        ref = distribution["sampler_distribution_ref"]
        view = distribution.get(
            "sampler_distribution_slice", {"start": 0, "stop": ref["token_count"]}
        )
        if (
            not isinstance(response_length, int)
            or isinstance(response_length, bool)
            or not 0 <= response_length <= view["stop"] - view["start"]
        ):
            raise ValueError("cannot extend a sampler distribution reference by truncating it")
        result["sampler_logprobs"] = distribution["sampler_logprobs"][:response_length]
        result["sampler_distribution_slice"] = {
            "start": view["start"],
            "stop": view["start"] + response_length,
        }
        return result
    for key in SAMPLER_FIELDS[:-1]:
        result[key] = distribution[key][:response_length]
    return result
