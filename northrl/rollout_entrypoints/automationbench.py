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

"""AutomationBench hooks: prepare the task, run its installed agent, score its state."""

import asyncio
import importlib
import json
import os
import pwd
import signal
import sys
from pathlib import Path

from northrl.remote_rollout.contracts import (
    AgentResult,
    EvaluationResult,
    RolloutEnvironmentError,
    RolloutInput,
)
from northrl.remote_rollout.network_policy import (
    agent_network_command,
    agent_network_environment,
    network_policy_enabled,
    prepare_network_policy,
)

_BUNDLE_ROOT = Path("/opt/automation_01_agent_only")
_AGENT_USER = "user"  # The public template validates its runner under this account.


def _agent_identity() -> pwd.struct_passwd:
    try:
        identity = pwd.getpwnam(_AGENT_USER)
    except KeyError as exc:
        raise RolloutEnvironmentError("AutomationBench template requires its user account") from exc
    if identity.pw_uid == 0 or identity.pw_gid == 0:
        raise RolloutEnvironmentError("AutomationBench agent account must be non-root")
    return identity


def _prepare_agent_workspace(workspace: Path, identity: pwd.struct_passwd) -> None:
    # Only grant the runner its output directory. Never follow a workspace link
    # or recursively change ownership of the template's public task resources.
    descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchown(descriptor, identity.pw_uid, identity.pw_gid)
        os.fchmod(descriptor, 0o750)
    finally:
        os.close(descriptor)


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    """Check that the installed task matches the offline dataset."""
    sys.path[:0] = [str(_BUNDLE_ROOT), str(_BUNDLE_ROOT / "runtime")]
    try:
        runner = importlib.import_module("agent_runner")
        task = runner.load_task(_BUNDLE_ROOT, input_data.metadata["task_id"])
        if task["task_sha256"] != input_data.task["task_sha256"]:
            raise ValueError("template task hash does not match the dataset")
        if any(
            (workspace / name).exists() for name in ("automationbench-output", "agent-stop.json")
        ):
            raise ValueError("this rollout requires a fresh workspace")
        if network_policy_enabled():
            identity = _agent_identity()
            _prepare_agent_workspace(workspace, identity)
            await prepare_network_policy(_AGENT_USER)
    except Exception as exc:
        raise RolloutEnvironmentError("Cannot prepare AutomationBench task") from exc


async def run_agent(
    input_data: RolloutInput, workspace: Path, model_environment: dict[str, str]
) -> AgentResult:
    """Run the template's agent through the platform relay; checkpoint before stopping."""
    command = [
        "/bin/bash",
        str(_BUNDLE_ROOT / "run_rollout.sh"),
        "--task-id",
        str(input_data.metadata["task_id"]),
        "--model",
        "northrl",
        "--output-dir",
        str(workspace / "automationbench-output"),
        "--max-steps",
        str(input_data.task.get("max_steps", 50)),
        "--max-retries",
        "0",
    ]
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENAI_BASE_URL": model_environment["OPENAI_BASE_URL"],
        "OPENAI_API_KEY": model_environment["OPENAI_API_KEY"],
    }
    if network_policy_enabled():
        identity = _agent_identity()
        environment.update(HOME=identity.pw_dir, USER=_AGENT_USER, LOGNAME=_AGENT_USER)
        environment = agent_network_environment(environment, _AGENT_USER)
        command = agent_network_command(
            ["runuser", "--user", _AGENT_USER, "--preserve-environment", "--", *command],
            _AGENT_USER,
        )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=workspace,
        env=environment,
        start_new_session=True,
    )
    stopped = False
    try:
        code = await process.wait()
        await asyncio.sleep(0)  # Allow a relay context-limit cancellation to arrive.
    except asyncio.CancelledError:
        stopped = True
        raise
    finally:
        cancelled = False
        for stop_signal, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, None)):
            if stop_signal == signal.SIGTERM and process.returncode is not None:
                continue
            try:
                os.killpg(process.pid, stop_signal)
            except ProcessLookupError:
                pass
            # Keep the same wait and grace deadline when another cancellation arrives.
            waiter = asyncio.create_task(asyncio.wait_for(process.wait(), timeout=timeout))
            while True:
                try:
                    await asyncio.shield(waiter)
                    break
                except asyncio.CancelledError:
                    stopped = cancelled = True
                except TimeoutError:
                    break
        if stopped:
            # The runner may have saved a request error or only its last running
            # checkpoint as the relay shut down. Mark only after process cleanup.
            (workspace / "agent-stop.json").write_text(
                json.dumps(
                    [
                        input_data.rollout_id,
                        input_data.metadata["task_id"],
                        input_data.task["task_sha256"],
                    ]
                ),
                encoding="utf-8",
            )
        if cancelled:
            raise asyncio.CancelledError
    return AgentResult(exit_code=code)


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    """Restore the saved state and call the original rubric uploaded after the agent."""
    del agent_result
    sys.path[:0] = [str(_BUNDLE_ROOT), str(_BUNDLE_ROOT / "runtime")]
    try:
        runner = importlib.import_module("agent_runner")
        task = runner.load_task(_BUNDLE_ROOT, input_data.metadata["task_id"])
        output = workspace / "automationbench-output"
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        assets = json.loads((evaluation_assets / "evaluation.json").read_text(encoding="utf-8"))
        if task["task_sha256"] != input_data.task["task_sha256"]:
            raise ValueError("template task hash mismatch")
        for key in ("task_id", "source_task_id", "task_sha256"):
            if record[key] != task[key] or assets[key] != task[key]:
                raise ValueError("snapshot or evaluation assets belong to another task")
        if assets["initial_state"] != task["initial_state"]:
            raise ValueError("evaluation initial state mismatch")
        stop_file = workspace / "agent-stop.json"
        stopped = stop_file.exists()
        if stopped and json.loads(stop_file.read_text(encoding="utf-8")) != [
            input_data.rollout_id,
            task["task_id"],
            task["task_sha256"],
        ]:
            raise ValueError("platform stop belongs to another rollout")
        statuses = {"completed", "max_steps", "token_limit", "content_filter", "interrupted"}
        if stopped:
            statuses.update(("running", "model_error"))
        if record["status"] not in statuses or (not record.get("finished_at") and not stopped):
            raise ValueError("agent failed without a valid checkpoint")
        world = runner.restore_world(
            json.loads((output / "final_world.json").read_text(encoding="utf-8")),
            json.loads((output / "runtime_state.json").read_text(encoding="utf-8")),
        )
        # NorthRL starts evaluate in a new process, so no previous task's rubric is loaded.
        package = importlib.import_module("automationbench")
        package.__path__.insert(0, str(evaluation_assets / "runtime" / "automationbench"))
        os.environ["AUTOMATIONBENCH_STRICT_ASSERTIONS"] = "1"
        rubric = importlib.import_module("automationbench.rubric")
        state = {
            "world": world,
            "initial_state": runner.strip_none_values(assets["initial_state"]),
            "info": {"assertions": assets["assertions"]},
        }
        reward = float(rubric.partial_credit(state))
        return EvaluationResult(
            reward=reward,
            status="success" if reward > 0 else "task_failed",
            metrics={
                "automationbench_task_completed_correctly": float(
                    rubric.task_completed_correctly(state)
                )
            },
        )
    except Exception as exc:
        raise RolloutEnvironmentError("AutomationBench evaluation failed") from exc
