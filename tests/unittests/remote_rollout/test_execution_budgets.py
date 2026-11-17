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

"""Budget defaults, precedence and fail-fast configuration validation."""

from dataclasses import FrozenInstanceError

import pytest
from omegaconf import OmegaConf

from northrl.remote_rollout.execution_budgets import ExecutionBudgets

_KEYS = (
    "preparation_timeout_seconds",
    "agent_timeout_seconds",
    "agent_finalize_timeout_seconds",
    "evaluation_timeout_seconds",
    "timeout_seconds",
)


@pytest.mark.parametrize("config", [{}, {"rollout": {}}, {"rollout": {"execution": {}}}])
def test_defaults_reserve_independent_stages_and_cleanup(config):
    budgets = ExecutionBudgets.from_config(config)

    assert budgets == ExecutionBudgets(600, 3600, 600, 600)
    assert budgets.evaluation_preparation_seconds == 600
    assert budgets.recovery_seconds == 900
    assert budgets.io_seconds == 60
    assert budgets.result_download_seconds == 120
    assert budgets.sandbox_timeout_seconds == 6930
    with pytest.raises(FrozenInstanceError):
        budgets.agent_seconds = 1


@pytest.mark.parametrize("as_omegaconf", [False, True])
def test_custom_stage_budgets_preserve_internal_transfer_limits(as_omegaconf):
    raw = {
        "rollout": {
            "execution": {
                "preparation_timeout_seconds": 10,
                "agent_timeout_seconds": 20,
                "agent_finalize_timeout_seconds": 30,
                "evaluation_timeout_seconds": 40,
            }
        }
    }
    config = OmegaConf.create(raw) if as_omegaconf else raw

    budgets = ExecutionBudgets.from_config(config)

    assert budgets == ExecutionBudgets(10, 20, 30, 40)
    assert budgets.evaluation_preparation_seconds == 600
    assert budgets.recovery_seconds == 900
    assert budgets.io_seconds == 60
    assert budgets.result_download_seconds == 120
    # I/O is a per-operation cap within a stage, not another stage allocation.
    assert budgets.sandbox_timeout_seconds == 1630
    assert dict(config["rollout"]["execution"]) == raw["rollout"]["execution"]


@pytest.mark.parametrize(
    ("execution", "argument", "expected"),
    [
        ({}, None, 3600),
        ({"timeout_seconds": 120}, None, 120),
        ({"timeout_seconds": 120}, 240, 240),
        ({}, 240, 240),
        ({"agent_timeout_seconds": 360, "timeout_seconds": 120}, None, 360),
        ({"agent_timeout_seconds": 360, "timeout_seconds": 120}, 240, 360),
        ({"agent_timeout_seconds": 360, "timeout_seconds": False}, 240, 360),
        ({"agent_timeout_seconds": 360}, False, 360),
    ],
)
def test_agent_timeout_precedence(execution, argument, expected):
    budgets = ExecutionBudgets.from_config(
        {"rollout": {"execution": execution}}, agent_timeout_seconds=argument
    )

    assert budgets.agent_seconds == expected


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "60"])
def test_invalid_configured_budget_is_rejected(key, value):
    config = OmegaConf.create({"rollout": {"execution": {key: value}}})

    with pytest.raises(ValueError, match=rf"rollout\.execution\.{key} must be a positive integer"):
        ExecutionBudgets.from_config(config)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "60"])
def test_invalid_agent_argument_is_rejected(value):
    with pytest.raises(ValueError, match="agent_timeout_seconds must be a positive integer"):
        ExecutionBudgets.from_config({}, agent_timeout_seconds=value)


@pytest.mark.parametrize(
    "field",
    [
        "preparation_seconds",
        "agent_seconds",
        "finalize_seconds",
        "evaluation_seconds",
    ],
)
def test_direct_construction_cannot_bypass_positive_integer_validation(field):
    with pytest.raises(ValueError, match="must be a positive integer"):
        ExecutionBudgets(**{field: False})
