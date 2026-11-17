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
NorthRL - A large-scale reinforcement learning training framework
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base_module import NorthRLModule
    from .controller import NorthRLController
    from .data_loader import BaseDataLoader
    from .northrl_types import Batch, ModelTag, Trajectory
    from .rollout_worker import BaseRolloutWorker, SimpleRolloutWorker, SingleTurnRolloutWorker
    from .trajectory_pool import TrajectoryPool
    from .weight_sync.weight_sync_controller import WeightSyncController

__version__ = "0.1.0"

_EXPORTS = {
    "NorthRLController": (".controller", "NorthRLController"),
    "NorthRLModule": (".base_module", "NorthRLModule"),
    "BaseRolloutWorker": (".rollout_worker", "BaseRolloutWorker"),
    "SingleTurnRolloutWorker": (".rollout_worker", "SingleTurnRolloutWorker"),
    "SimpleRolloutWorker": (".rollout_worker", "SimpleRolloutWorker"),
    "TrajectoryPool": (".trajectory_pool", "TrajectoryPool"),
    "WeightSyncController": (
        ".weight_sync.weight_sync_controller",
        "WeightSyncController",
    ),
    "BaseDataLoader": (".data_loader", "BaseDataLoader"),
    "Batch": (".northrl_types", "Batch"),
    "Trajectory": (".northrl_types", "Trajectory"),
    "ModelTag": (".northrl_types", "ModelTag"),
}

__all__ = [
    "NorthRLController",
    "NorthRLModule",
    "BaseRolloutWorker",
    "SingleTurnRolloutWorker",
    "SimpleRolloutWorker",
    "TrajectoryPool",
    "WeightSyncController",
    "BaseDataLoader",
    "Batch",
    "Trajectory",
    "ModelTag",
]


def __getattr__(name: str) -> Any:
    """Load public training APIs only when callers request them."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
