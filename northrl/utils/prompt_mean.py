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

"""Compute group identities and original token counts for prompt-mean policy loss.

A comparison group contains all rollouts sampled for one prompt occurrence and
is identified by ``(model_tag, group_id)``. Prompt-mean averages token losses
within each group, then averages those group means. Individual trajectories or
segments are not the units of the outer mean: a longer rollout accounts for a
larger share of its group's tokens, while every group mean is weighted equally.

For ``G`` eligible groups and ``T_g`` original response tokens in group ``g``,
each token receives weight ``1 / (G * max(T_g, 1))``. These helpers supply the
group keys and ``T_g`` values; they do not compute advantages or perform an
optimizer update. The Datum converter attaches ``G * max(T_g, 1)`` as local
normalization data. The remote trainer then divides policy advantages by that
value and requests a sum loss from the backend.

Counts use original loss masks across complete, prepared parent trajectories.
MIS rejection, importance weights, and zero advantages do not alter these
counts. Successful groups without response tokens remain in ``G`` with zero
loss contribution. The trainer separately skips batches with no effective
training targets. See
``docs/en/developer-guide/04-trajectory-pool/trajectory-semantics.md`` for the
aggregation references and the current implementation limits.
"""

from collections.abc import Hashable
from typing import Any, cast

from northrl.northrl_types import Trajectory
from northrl.utils.trajectory_alignment import validate_loss_mask_for_target_alignment
from northrl.utils.trajectory_sequences import iter_trajectory_sequences


def prompt_group_key(sequence: Trajectory | dict[str, Any]) -> tuple[Hashable, Hashable]:
    """Return the model-scoped comparison-group identity used by prompt-mean.

    The key follows GRPO's value-based ``(model_tag, group_id)`` grouping.
    Model tags keep equal group IDs from different models separate. A task
    sampled again belongs to a new group when it has a new ``group_id``; neither
    ``task_id`` nor a rollout/segment index is used as a substitute for that ID.

    Values exposing ``item()`` are converted to their scalar values before
    hashing, so scalar tensor identities compare by value. Errors from
    ``item()`` propagate unchanged; tensor identities must represent scalars.
    A missing model tag is represented by ``None``; this helper does not supply
    a default model. A group ID is required even when the trajectory contains
    no training tokens.

    Args:
        sequence: A complete parent ``Trajectory`` or a training-sequence dict
            carrying its parent's identity. For nested sequence dicts, callers
            must first propagate the authoritative parent fields, as
            ``iter_training_sequences`` does. This helper only reads the fields
            on the supplied object and does not inspect a parent or its siblings.

    Returns:
        A hashable ``(model_tag, group_id)`` tuple after scalar conversion.

    Raises:
        ValueError: ``group_id`` is missing, ``None``, or an empty string, or
            either identity value is unhashable after conversion.
    """
    if sequence.get("group_id") is None or sequence.get("group_id") == "":
        raise ValueError("prompt-mean requires group_id on every training trajectory")
    values = []
    for name in ("model_tag", "group_id"):
        value = sequence.get(name)
        if value is not None and hasattr(value, "item"):
            value = value.item()
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"prompt-mean requires a hashable {name}, got {value!r}") from exc
        values.append(cast(Hashable, value))
    return values[0], values[1]


def prompt_group_token_counts(
    trajectories: list[Trajectory],
) -> dict[tuple[Hashable, Hashable], int]:
    """Count original response tokens across complete, prepared comparison groups.

    Each successful parent that is not marked ``group_skipped`` registers its
    group, even if it has no sequences or its masks are entirely zero. Parents
    with another status or a truthy ``group_skipped`` flag are ignored. Callers
    must prepare complete groups first: this helper does not validate expected
    group membership or propagate one member's failure to the other members.

    Counts always use the parent's ``(model_tag, group_id)``. For a segmented
    trajectory, all nested sequences contribute to that parent group regardless
    of any identity fields on the segments. An ordinary trajectory contributes
    its own token sequence. An explicitly empty ``segments`` list contributes
    no sequences.

    Each sequence's original ``loss_mask`` must match its token length, contain
    only binary values, and mask out the first token because that position has
    no causal prediction target. Binary floats are accepted and counted as
    Python integers. Context or padding positions contribute nothing when their
    mask is zero. Active positions are counted as supplied, without deduplication;
    overlapping segments must mask repeated context to avoid counting it again.

    Only original masks are read. Zero advantages, MIS importance weights, and
    MIS rejection masks do not change the result. In particular, a group whose
    tokens are later all rejected still retains its original token count. The
    caller must not replace the original masks with effective rejection masks.
    This function does not mutate trajectories or their nested sequence records.

    Args:
        trajectories: Complete parent trajectories for one optimizer batch,
            after group eligibility has been determined and before Datum
            filtering or splitting into forward/backward requests. Passing
            only a subset of a group's rollouts or segments changes its count.

    Returns:
        A mapping from group keys to nonnegative integer token counts. Its
        length is the number ``G`` of eligible groups, including zero-token
        groups. For example, ``{("actor", "a"): 40, ("actor", "b"): 20}``
        yields per-token denominators of 80 and 40, respectively. Adding a
        successful zero-token group increases ``G`` to 3 while leaving those
        token counts unchanged. An input with no eligible parents returns ``{}``.
        Clamping zero counts and skipping empty training updates are left to
        the converter and trainer, respectively.

    Raises:
        ValueError: An eligible parent's group identity is invalid, or a
            sequence's loss mask has the wrong length, a nonzero first entry,
            or nonbinary values. See ``prompt_group_key`` for identity checks.
        KeyError: A nested sequence lacks required ``tokens`` or ``loss_mask``
            fields. Inputs must already satisfy the training-sequence contract.
    """
    counts: dict[tuple[Hashable, Hashable], int] = {}
    for trajectory in trajectories:
        if trajectory.status != "success" or trajectory.get("group_skipped", False):
            continue
        key = prompt_group_key(trajectory)
        counts.setdefault(key, 0)
        for sequence in iter_trajectory_sequences(trajectory):
            mask = sequence["loss_mask"]
            validate_loss_mask_for_target_alignment(tokens=sequence["tokens"], loss_mask=mask)
            if any(value not in (0, 1) for value in mask):
                raise ValueError("prompt-mean requires binary trajectory loss_mask values")
            counts[key] += sum(int(value) for value in mask)
    return counts
