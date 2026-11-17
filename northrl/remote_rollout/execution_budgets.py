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

"""Independent host-controlled budgets for one E2B execution attempt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

_CONFIG_KEYS = (
    ("preparation_seconds", "preparation_timeout_seconds"),
    ("agent_seconds", "agent_timeout_seconds"),
    ("finalize_seconds", "agent_finalize_timeout_seconds"),
    ("evaluation_seconds", "evaluation_timeout_seconds"),
)


def _positive_integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"rollout.execution.{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class ExecutionBudgets:
    """Stage budgets in seconds; I/O is capped within the current stage.

    ``timeout_seconds`` remains a legacy alias for the agent budget, not a
    shared deadline for preparation, finalization, evaluation and recovery.
    """

    preparation_seconds: int = 600
    agent_seconds: int = 3600
    finalize_seconds: int = 600
    evaluation_seconds: int = 600
    # Transfer limits are framework policy, not task configuration.
    evaluation_preparation_seconds: ClassVar[int] = 600
    recovery_seconds: ClassVar[int] = 900
    io_seconds: ClassVar[int] = 60
    result_download_seconds: ClassVar[int] = 120

    def __post_init__(self) -> None:
        for field, key in _CONFIG_KEYS:
            _positive_integer(getattr(self, field), key)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        agent_timeout_seconds: int | None = None,
    ) -> ExecutionBudgets:
        """Resolve an effective root config after execution-profile merging.

        An explicit ``agent_timeout_seconds`` config key wins. Otherwise use
        the supplied agent timeout, then the legacy ``timeout_seconds`` config
        key, and finally the default. Explicit nulls are invalid configuration.
        """
        execution = config.get("rollout", {}).get("execution", {})
        values = {
            field: execution.get(key, getattr(cls, field))
            for field, key in _CONFIG_KEYS
            if field != "agent_seconds"
        }
        if "agent_timeout_seconds" in execution:
            agent_value = execution["agent_timeout_seconds"]
            agent_key = "agent_timeout_seconds"
        elif agent_timeout_seconds is not None:
            agent_value = agent_timeout_seconds
            agent_key = "agent_timeout_seconds"
        else:
            agent_value = execution.get("timeout_seconds", cls.agent_seconds)
            agent_key = "timeout_seconds"
        values["agent_seconds"] = _positive_integer(agent_value, agent_key)
        return cls(**values)

    @property
    def sandbox_timeout_seconds(self) -> int:
        """Reserve all stage budgets plus 30 seconds for sandbox cleanup."""
        return (
            self.preparation_seconds
            + self.agent_seconds
            + self.finalize_seconds
            + self.evaluation_preparation_seconds
            + self.evaluation_seconds
            + self.recovery_seconds
            + 30
        )
