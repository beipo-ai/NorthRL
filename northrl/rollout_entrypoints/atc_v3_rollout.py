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

"""Adapt the template-installed ATC runtime to NorthRL's staged rollout hooks."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import re
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn

from northrl.remote_rollout.contracts import (
    AgentResult,
    EvaluationResult,
    RolloutEnvironmentError,
    RolloutInput,
)

logger = logging.getLogger(__name__)

WORKSPACE = Path("/workspace/atc")
_RUNTIME_ROOT = Path("/opt/atc-v3-runtime")


def _load_runtime() -> ModuleType:
    """Import only the root-owned, read-only runtime installed in the template."""
    package = _RUNTIME_ROOT / "atc_v3_runtime"
    initializer = package / "__init__.py"
    try:
        if os.geteuid() != 0:
            raise ValueError("ATC lifecycle hooks require the sandbox root account")
        paths = [*_RUNTIME_ROOT.parents, _RUNTIME_ROOT, package, *package.rglob("*")]
        for path in paths:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                raise ValueError(f"ATC runtime path must be root-owned and read-only: {path}")
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError(f"ATC runtime path is not a regular file or directory: {path}")
        module = sys.modules.get("atc_v3_runtime")
        if module is None:
            spec = importlib.util.spec_from_file_location(
                "atc_v3_runtime", initializer, submodule_search_locations=[str(package)]
            )
            if spec is None or spec.loader is None:
                raise ValueError("ATC runtime module could not be loaded")
            module = importlib.util.module_from_spec(spec)
            sys.modules["atc_v3_runtime"] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop("atc_v3_runtime", None)
                raise
        elif getattr(module, "__file__", None) != str(initializer):
            raise ValueError("ATC runtime was imported from an unexpected path")
        if getattr(module, "API_VERSION", None) != 1 or not all(
            callable(getattr(module, name, None)) for name in ("prepare", "run_agent", "evaluate")
        ):
            raise ValueError("ATC runtime must implement API_VERSION 1")
        return module
    except Exception as exc:
        raise RolloutEnvironmentError("ATC template runtime is missing or untrusted") from exc


def resolve_workspace(input_data: RolloutInput) -> Path:
    del input_data
    return WORKSPACE


def _task(input_data: RolloutInput) -> dict:
    task_id = input_data.metadata.get("task_id")
    if not isinstance(task_id, str) or re.fullmatch(r"task_[0-9]{5}", task_id) is None:
        raise RolloutEnvironmentError("ATC metadata.task_id must have the form task_00000")
    if input_data.task.get("task_id", task_id) != task_id:
        raise RolloutEnvironmentError("ATC task_id does not match rollout metadata")
    return {**input_data.task, "task_id": task_id}


def _runtime_error(exc: Exception) -> NoReturn:
    status = getattr(exc, "rollout_status", None)
    if status == "env_error":
        raise RolloutEnvironmentError(str(exc)) from exc
    raise RolloutEnvironmentError("ATC runtime could not complete the lifecycle stage") from exc


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    task = _task(input_data)
    runtime = _load_runtime()
    try:
        runtime.prepare(task, workspace)
    except Exception as exc:
        _runtime_error(exc)


async def run_agent(
    input_data: RolloutInput, workspace: Path, model_environment: dict[str, str]
) -> AgentResult:
    task = _task(input_data)
    runtime = _load_runtime()
    try:
        # The runtime returns only after agent/service cleanup and final DB snapshots.
        result = await runtime.run_agent(task, workspace, model_environment)
        return AgentResult.model_validate(result)
    except Exception as exc:
        _runtime_error(exc)


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    task = _task(input_data)
    runtime = _load_runtime()
    try:
        result = runtime.evaluate(task, workspace, agent_result.model_dump(), evaluation_assets)
        reward = result["score"]
        passed, total = result["passed_checks"], result["check_count"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("verifier score must be numeric")
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("verifier score must be finite and between zero and one")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (passed, total)):
            raise ValueError("verifier check counts must be integers")
        if total <= 0 or not 0 <= passed <= total:
            raise ValueError("verifier check counts are invalid")
    except Exception as exc:
        if getattr(exc, "rollout_status", None) == "env_error":
            raise RolloutEnvironmentError(str(exc)) from exc
        logger.warning(
            "ATC evaluation failed for %s; assigning reward 0", task["task_id"], exc_info=True
        )
        return EvaluationResult(
            reward=0.0,
            status="task_failed",
            metrics={"atc_agent_exit_code": float(agent_result.exit_code)},
        )
    return EvaluationResult(
        reward=float(reward),
        status="task_failed" if reward == 0 else "success",
        metrics={
            "atc_passed_checks": float(passed),
            "atc_check_count": float(total),
            "atc_agent_exit_code": float(agent_result.exit_code),
        },
    )
