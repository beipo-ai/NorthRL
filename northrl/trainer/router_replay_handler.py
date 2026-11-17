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

"""Stateless Router Replay metadata transport for the GRPO trainer."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..northrl_types import Trajectory

logger = logging.getLogger(__name__)


class RouterReplayHandler:
    """Validate Router Replay config and attach opaque refs at data boundaries.

    The handler retains only immutable configuration. Per-step R2 RECORD output
    is bound to its trajectories immediately, so there is no pending ref that a
    later conversion or failed step could accidentally consume.
    """

    def __init__(self, algorithm_config: Any):
        config = algorithm_config or {}
        replay = (config.get("router_replay", {}) if config else {}) or {}
        self.enabled = bool(replay.get("enabled", False))
        self.mode = str(replay.get("mode", "R3"))

        if not self.enabled:
            return
        if self.mode not in ("R2", "R3"):
            raise ValueError(
                f"algorithm.router_replay.mode must be 'R2' or 'R3', got '{self.mode}'"
            )
        if (
            self.mode in ("R2", "R3")
            and str(config.get("sampling_logprob_source", "recomputed")) != "recomputed"
        ):
            raise ValueError(
                f"algorithm.router_replay.mode={self.mode} requires "
                "algorithm.sampling_logprob_source='recomputed' so trainer-side "
                "logprob recompute and the training forward use matching routing."
            )
        logger.info("[router_replay][config] enabled: mode=%s", self.mode)

    @property
    def records_during_recompute(self) -> bool:
        """Whether this trainer must RECORD routing during logprob recompute."""
        return self.enabled and self.mode == "R2"

    def bind_recorded_routing_ref(
        self,
        trajectories: Sequence[Trajectory | dict[str, Any]],
        recorded_routing_ref: dict[str, Any] | None,
    ) -> None:
        """Bind one R2 RECORD handle and its original index to each trajectory."""
        if not isinstance(recorded_routing_ref, dict):
            raise ValueError("[router_replay] R2 recompute did not return a recorded routing ref.")
        for sample_index, trajectory in enumerate(trajectories):
            trajectory["router_replay_ref"] = {
                **recorded_routing_ref,
                "sample_index": sample_index,
            }

    def build_datum_metadata(
        self,
        trajectory: Trajectory | dict[str, Any],
        trajectory_index: int,
    ) -> dict[str, Any]:
        """Build replay metadata for one retained trajectory."""
        source = "rollout" if self.mode == "R3" else "recompute"
        ref_field = "moe_topk_indices_ref" if self.mode == "R3" else "router_replay_ref"

        datum_routing_ref = trajectory.get(ref_field)
        if not isinstance(datum_routing_ref, dict):
            raise ValueError(
                f"[router_replay] trajectory {trajectory_index} is missing {ref_field} "
                f"while mode={self.mode} is active."
            )

        return {
            "router_replay": {
                "schema": "weaver.router_replay.datum.v1",
                "mode": self.mode,
                "source": source,
                "action": "REPLAY",
                "fail_fast": True,
                # Preserve the producer's opaque contract, including fields a
                # newer server/trainer may add after this NorthRL version ships.
                "value_ref": dict(datum_routing_ref),
            }
        }
