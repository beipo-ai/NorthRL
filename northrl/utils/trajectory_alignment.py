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

"""Helpers for crossing the trajectory -> Weaver Datum alignment boundary.

Training-sequence logprob fields are token-aligned and keep ``len(field) == len(tokens)``
with position 0 as the no-context placeholder. Weaver Datum fields are
target-aligned: ``target_tokens = tokens[1:]``. Target positions are the
model-predicted positions where per-token logprobs/losses can be computed;
``loss_mask`` decides which target positions actually contribute to the
objective.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import Literal, TypeVar, overload

import numpy as np

T = TypeVar("T")


def drop_leading_placeholder_for_targets(
    *,
    name: str,
    tokens: Sequence[int],
    token_aligned_values: Sequence[T],
) -> list[T]:
    """Drop the no-context first slot so values align with ``target_tokens``.

    Sequence-level per-token fields share the same length as ``tokens``. The
    first position is a placeholder because no previous token predicts
    ``tokens[0]``. Trainer datums use causal-LM target alignment
    (``target_tokens = tokens[1:]``), so every per-token field is shifted to
    match target length.
    """
    if len(token_aligned_values) != len(tokens):
        raise ValueError(
            f"{name} must be token-aligned with tokens. "
            f"Got {name}={len(token_aligned_values)}, tokens={len(tokens)}."
        )
    return list(token_aligned_values[1:])


def validate_loss_mask_for_target_alignment(
    *,
    tokens: Sequence[int],
    loss_mask: Sequence[int],
) -> None:
    if len(loss_mask) != len(tokens):
        raise ValueError(
            "loss_mask must be token-aligned with tokens. "
            f"Got loss_mask={len(loss_mask)}, tokens={len(tokens)}."
        )
    if tokens and int(loss_mask[0]) != 0:
        raise ValueError("loss_mask[0] must be 0; the first token has no target logprob.")


@overload
def align_logprobs_to_target(
    *,
    tokens: Sequence[int],
    logprobs: Sequence[float | None],
    name: str = "logprobs",
    compact: Literal[False] = False,
) -> list[float]: ...


@overload
def align_logprobs_to_target(
    *,
    tokens: Sequence[int],
    logprobs: Sequence[float | None],
    name: str = "logprobs",
    compact: bool,
) -> list[float] | array: ...


def align_logprobs_to_target(
    *,
    tokens: Sequence[int],
    logprobs: Sequence[float | None],
    name: str = "logprobs",
    compact: bool = False,
) -> list[float] | array:
    """Return token-length trajectory logprobs aligned with ``target_tokens``.

    ``logprobs[0]`` is a placeholder and is intentionally dropped. All
    target positions must contain real numeric logprobs.
    ``compact`` preserves numeric array inputs in an owned float64 buffer.
    """
    if len(logprobs) != len(tokens):
        raise ValueError(
            f"{name} must be token-aligned with tokens. "
            f"Got {name}={len(logprobs)}, tokens={len(tokens)}."
        )
    if tokens:
        head = logprobs[0]
        if head is not None and abs(float(head)) >= 1e-9:
            raise ValueError(
                f"{name}[0] must be None or 0.0; the first token has no target logprob."
            )
    if compact and isinstance(logprobs, array) and logprobs.typecode in "fd":
        target = array("d")
        target.frombytes(memoryview(np.asarray(logprobs, dtype=np.float64)[1:]).cast("B"))
        return target
    target_logprobs = drop_leading_placeholder_for_targets(
        name=name,
        tokens=tokens,
        token_aligned_values=logprobs,
    )
    if any(value is None for value in target_logprobs):
        raise ValueError(f"{name} may only contain None at the first token position.")
    return [float(value) for value in target_logprobs if value is not None]


def align_target_logprobs_to_tokens(
    *,
    tokens: Sequence[int],
    target_logprobs: Sequence[float],
    name: str = "logprobs",
) -> list[float] | array:
    """Prepend the no-context slot so forward outputs align with ``tokens``.

    ``training_client.forward(..., "forward_logprob")`` returns one value per
    target position. Trajectories store logprobs at token length, so callers use
    this before writing values such as ``old_log_probs`` or ``ref_logprobs``.
    """
    expected_len = max(len(tokens) - 1, 0)
    if len(target_logprobs) != expected_len:
        raise ValueError(
            f"{name} must be target-aligned with tokens[1:]. "
            f"Got {name}={len(target_logprobs)}, target_tokens={expected_len}."
        )
    if isinstance(tokens, array) and isinstance(target_logprobs, array):
        aligned = array("d", [0.0] if tokens else [])
        aligned.extend(target_logprobs)
        return aligned
    return ([0.0] if tokens else []) + [float(lp) for lp in target_logprobs]


@overload
def align_loss_mask_to_target(
    *,
    tokens: Sequence[int],
    loss_mask: Sequence[int],
    compact: Literal[False] = False,
) -> list[int]: ...


@overload
def align_loss_mask_to_target(
    *,
    tokens: Sequence[int],
    loss_mask: Sequence[int],
    compact: bool,
) -> list[int] | array: ...


def align_loss_mask_to_target(
    *,
    tokens: Sequence[int],
    loss_mask: Sequence[int],
    compact: bool = False,
) -> list[int] | array:
    validate_loss_mask_for_target_alignment(tokens=tokens, loss_mask=loss_mask)
    if compact and isinstance(loss_mask, array) and loss_mask.typecode == "b":
        return loss_mask[1:]
    return [int(value) for value in loss_mask[1:]]


def align_response_sampling_mask_to_target(
    *,
    tokens: Sequence[int],
    loss_mask: Sequence[int],
    sampling_mask: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Expand response-aligned sampling_mask to trainer target alignment.

    Sequence ``sampling_mask`` is response-aligned: one candidate-token list
    for each token where ``loss_mask == 1``. Weaver trainer forward_logprob
    expects one candidate-token list for each target token. Non-response target
    positions are filled with a singleton ``[target_token]`` so constrained
    logprob is a no-op there.
    """
    validate_loss_mask_for_target_alignment(tokens=tokens, loss_mask=loss_mask)

    expected_response_count = sum(1 for value in loss_mask[1:] if int(value) > 0)
    if len(sampling_mask) != expected_response_count:
        raise ValueError(
            "sampling_mask must be response-aligned. "
            f"Got sampling_mask={len(sampling_mask)}, response_tokens={expected_response_count}."
        )

    response_masks = iter(sampling_mask)
    target_sampling_mask: list[list[int]] = []
    for token_idx, token in enumerate(tokens[1:], start=1):
        if int(loss_mask[token_idx]) > 0:
            token_mask = [int(token_id) for token_id in next(response_masks)]
            if not token_mask:
                raise ValueError("Each sampling_mask entry must contain at least one token id.")
            if int(token) not in token_mask:
                raise ValueError(
                    "sampling_mask must contain the sampled response token. "
                    f"token_idx={token_idx}, token={int(token)}."
                )
            target_sampling_mask.append(token_mask)
        else:
            target_sampling_mask.append([int(token)])

    return target_sampling_mask
