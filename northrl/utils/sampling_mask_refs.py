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

"""Keep response support opaque through truncation and target alignment."""
from __future__ import annotations

import copy


def mask_parts(mask):
    parts = mask.get("sampling_mask_refs", [mask])
    if not isinstance(parts, list) or not parts:
        raise ValueError("sampling mask ref list must be nonempty")
    for part in parts:
        ref = part["sampling_mask_ref"]
        view = part.get("sampling_mask_slice", {"start": 0, "stop": ref["token_count"]})
        start, stop, count = view["start"], view["stop"], ref["token_count"]
        if (
            any(not isinstance(v, int) or isinstance(v, bool) for v in (start, stop, count))
            or not 0 <= start <= stop <= count
        ):
            raise ValueError("invalid sampling mask reference slice")
        yield {"sampling_mask_ref": dict(ref), "sampling_mask_slice": dict(view)}


def join_sampling_masks(first, second):
    if first is None:
        return copy.deepcopy(second)
    if isinstance(first, dict) != isinstance(second, dict):
        raise ValueError("cannot mix inline and referenced masks across model turns")
    if isinstance(first, dict):
        return {"sampling_mask_refs": [*mask_parts(first), *mask_parts(second)]}
    return [*first, *second]


def trim_sampling_mask(mask, response_length):
    if mask is None:
        return None
    if (
        not isinstance(response_length, int)
        or isinstance(response_length, bool)
        or response_length < 0
    ):
        raise ValueError("invalid sampling mask response length")
    if not isinstance(mask, dict):
        if response_length > len(mask):
            raise ValueError("sampling mask is shorter than the retained response")
        return [list(row) for row in mask[:response_length]]
    remaining = response_length
    result = []
    for part in mask_parts(mask):
        view = part["sampling_mask_slice"]
        used = min(remaining, view["stop"] - view["start"])
        view["stop"] = view["start"] + used
        result.append(part)
        remaining -= used
        if remaining == 0:
            break
    if remaining:
        raise ValueError("cannot extend a sampling mask reference by truncation")
    return result[0] if len(result) == 1 else {"sampling_mask_refs": result}


def align_sampling_mask_ref(tokens, loss_mask, mask):
    from northrl.utils.trajectory_alignment import validate_loss_mask_for_target_alignment

    validate_loss_mask_for_target_alignment(tokens=tokens, loss_mask=loss_mask)
    positions = [i for i, value in enumerate(loss_mask[1:]) if value]
    parts = list(mask_parts(mask))
    offset = 0
    for part in parts:
        view = part["sampling_mask_slice"]
        count = view["stop"] - view["start"]
        part["sampling_mask_positions"] = positions[offset : offset + count]
        offset += count
    if offset != len(positions):
        raise ValueError("sampling mask ref must align with the retained response")
    return parts[0] if len(parts) == 1 else {"sampling_mask_refs": parts}
