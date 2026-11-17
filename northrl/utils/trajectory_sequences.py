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

"""Internal iteration over the sequence data owned by complete trajectories."""

from collections.abc import Iterable, Iterator
from typing import Any

from northrl.northrl_types import Trajectory


def iter_trajectory_sequences(trajectory: Trajectory) -> Iterator[Trajectory | dict[str, Any]]:
    """Yield original sequence records without filtering or modifying metadata."""
    if not isinstance(trajectory, Trajectory):
        raise TypeError("Sequence iteration requires a complete Trajectory object")
    if trajectory.segments is not None:
        yield from trajectory.segments
    elif trajectory.tokens or trajectory.is_managed_sample_ref:
        # A managed-SFT SampleRef carrier is one trainable sequence whose
        # tokens and loss_mask stay server-private (both are None here).
        yield trajectory


def iter_training_sequences(
    trajectories: Iterable[Trajectory],
) -> Iterator[Trajectory | dict[str, Any]]:
    """Yield trainable original records and broadcast authoritative parent metadata.

    Ordinary trajectories remain the sequence object themselves. Nested records
    retain their token-level metadata, while parent execution identity and
    advantage override stale values. Outcome fields stay on the parent.
    """
    authoritative_fields = {
        "group_id",
        "run_id",
        "rollout_id",
        "expected_rollouts",
        "task_id",
        "model_tag",
        "advantage",
    }
    outcome_fields = {
        "reward",
        "status",
        "is_val",
        "group_skipped",
        "segment_count",
        "accepted_segments",
        "tokens",
        "loss_mask",
        "segments",
        "extra_fields",
        "runtime_metrics",
        "segment_metrics",
        "metric_score_names",
    }
    for trajectory in trajectories:
        if not isinstance(trajectory, Trajectory):
            raise TypeError("Training requires complete Trajectory objects")
        if trajectory.status != "success" or trajectory.get("group_skipped", False):
            continue
        for sequence in iter_trajectory_sequences(trajectory):
            if isinstance(sequence, dict):
                for key, value in trajectory.extra_fields.items():
                    if key in authoritative_fields:
                        sequence[key] = value
                    elif key not in outcome_fields:
                        sequence.setdefault(key, value)
            yield sequence
