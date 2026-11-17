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

"""MiMo Terminal Bench hooks using the original image and hidden test package."""

from __future__ import annotations

import asyncio
import json
import math
import os
import pwd
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, Sequence

from northrl.remote_rollout.contracts import (
    AgentResult,
    EvaluationResult,
    RolloutEnvironmentError,
    RolloutInput,
)
from northrl.remote_rollout.network_policy import (
    agent_network_command,
    agent_network_environment,
    prepare_network_policy,
)

_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_STATE_ROOT = Path("/root/northrl-mimo")
_ROOT_UID = 0
_WORKSPACE = Path("/app")
_TESTS = Path("/tests")
_LOGS = Path("/logs")
_RUNTIME_METADATA = Path("/etc/mimo-e2b")


class CommandTimeoutError(RolloutEnvironmentError):
    """A subprocess exceeded the task's configured execution deadline."""


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RolloutEnvironmentError(f"MiMo {name} must be a positive finite number")
    return float(value)


def _agent_identity(input_data: RolloutInput) -> pwd.struct_passwd:
    user = input_data.task.get("agent_user")
    if not isinstance(user, str) or not re.fullmatch(r"[a-z_][a-z0-9_-]{0,30}", user):
        raise RolloutEnvironmentError("MiMo requires a validated agent_user")
    try:
        identity = pwd.getpwnam(user)
    except KeyError as exc:
        raise RolloutEnvironmentError("MiMo template agent user is missing") from exc
    if identity.pw_uid == 0 or identity.pw_dir != f"/home/{user}":
        raise RolloutEnvironmentError("MiMo agent must use its non-root template account")
    return identity


def resolve_workspace(input_data: RolloutInput) -> Path:
    raw = input_data.task.get("cwd")
    if not isinstance(raw, str) or not raw.startswith("/") or ".." in Path(raw).parts:
        raise RolloutEnvironmentError("MiMo task.cwd must be an absolute template directory")
    path = Path(raw)
    if not path.is_dir() or path.resolve() != path:
        raise RolloutEnvironmentError("MiMo workspace is missing or traverses a symlink")
    return path


def _require_private_path(path: Path) -> None:
    if path.is_symlink():
        raise RolloutEnvironmentError("MiMo trusted path must not be a symlink")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo trusted path is missing") from exc
    if metadata.st_uid != _ROOT_UID or metadata.st_mode & 0o022:
        raise RolloutEnvironmentError("MiMo trusted path must be root-owned and non-writable")


def _evaluation_file(evaluation_assets: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RolloutEnvironmentError("Invalid MiMo evaluation asset path")
    _require_private_path(evaluation_assets)
    evaluation_assets.chmod(0o700)
    path = evaluation_assets / relative
    for parent in [*path.parents]:
        if parent == evaluation_assets:
            break
        if parent.is_symlink():
            raise RolloutEnvironmentError("MiMo evaluation asset traverses a symlink")
    _require_private_path(path)
    if not path.is_file():
        raise RolloutEnvironmentError("MiMo evaluation asset must be a file")
    return path


async def _finish_cleanup(cleanup: asyncio.Task) -> None:
    """Complete cleanup even if the caller is cancelled repeatedly."""
    cancelled = False
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError:
            if cleanup.done():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def _run_command(
    argv: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Capture output privately; errors deliberately omit possibly secret output."""
    arguments = [str(arg) for arg in argv]
    if not arguments:
        raise RolloutEnvironmentError("MiMo subprocess command is empty")
    # These hooks run as root: never resolve a utility via the agent's PATH.
    if "/" not in arguments[0]:
        executable = shutil.which(arguments[0], path=_SYSTEM_PATH)
        if executable is None:
            raise RolloutEnvironmentError("MiMo subprocess utility is unavailable")
        arguments[0] = executable
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=cwd,
            env={"PATH": _SYSTEM_PATH} if env is None else env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo subprocess could not be started") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except (asyncio.CancelledError, TimeoutError) as exc:
        await _finish_cleanup(asyncio.create_task(_kill_process_group(process)))
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise CommandTimeoutError("MiMo subprocess exceeded its timeout") from exc
    returncode = process.returncode
    if returncode is None:
        raise RolloutEnvironmentError("MiMo subprocess did not terminate")
    return subprocess.CompletedProcess(
        list(argv),
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _agent_processes_running(uid: int) -> bool:
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            fields = dict(
                line.split(":", 1)
                for line in (directory / "status").read_text().splitlines()
                if ":" in line
            )
        except FileNotFoundError:
            continue
        if int(fields["Uid"].split()[0]) == uid and fields["State"].split()[0] != "Z":
            return True
    return False


async def _stop_agent_processes(input_data: RolloutInput) -> None:
    identity = _agent_identity(input_data)
    result = await _run_command(["pkill", "-KILL", "-u", str(identity.pw_uid)], timeout=10)
    if result.returncode not in (0, 1):
        raise RolloutEnvironmentError("MiMo agent process cleanup failed")
    for _ in range(100):
        if not _agent_processes_running(identity.pw_uid):
            return
        await asyncio.sleep(0.05)
    raise RolloutEnvironmentError("MiMo agent processes remained alive after cleanup")


def _state_file(input_data: RolloutInput) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", input_data.rollout_id) or input_data.rollout_id in {
        ".",
        "..",
    }:
        raise RolloutEnvironmentError("Invalid MiMo rollout state identity")
    for path in (_STATE_ROOT, _STATE_ROOT / input_data.rollout_id):
        if path.is_symlink():
            raise RolloutEnvironmentError("MiMo private state must not traverse a symlink")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_path(path)
        path.chmod(0o700)
    return _STATE_ROOT / input_data.rollout_id / "state.json"


def parse_reward(text: str) -> float:
    """The verifier file, rather than its shell exit code, defines the score."""
    try:
        reward = float(text.strip())
    except (ValueError, TypeError) as exc:
        raise RolloutEnvironmentError("Terminal verifier reward.txt is missing or invalid") from exc
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise RolloutEnvironmentError("Terminal verifier reward must be 0 or 1")
    return reward


def _trusted_file(path: Path) -> Path:
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise RolloutEnvironmentError(f"Terminal runtime file must be a regular file: {path}")
    metadata = path.stat()
    if metadata.st_uid != _ROOT_UID or metadata.st_mode & 0o022:
        raise RolloutEnvironmentError(f"Terminal runtime file must be root-owned: {path}")
    return path


def _check_workspace(input_data: RolloutInput, workspace: Path) -> None:
    if workspace != _WORKSPACE or resolve_workspace(input_data) != workspace:
        raise RolloutEnvironmentError("Terminal tasks require their original /app workspace")


def _tests_absent() -> None:
    if _TESTS.exists() or _TESTS.is_symlink():
        raise RolloutEnvironmentError("Terminal hidden /tests must be absent before evaluation")


def _fresh_verifier_logs() -> Path:
    if _LOGS.is_symlink() or _LOGS.resolve() != _LOGS.absolute():
        raise RolloutEnvironmentError("Terminal log directory must not traverse symlinks")
    _LOGS.mkdir(mode=0o755, exist_ok=True)
    if _LOGS.stat().st_uid != _ROOT_UID or _LOGS.stat().st_mode & 0o022:
        raise RolloutEnvironmentError("Terminal log directory must be owned by root")
    verifier_logs = _LOGS / "verifier"
    if verifier_logs.is_symlink():
        raise RolloutEnvironmentError("Terminal verifier directory must not be a symlink")
    if verifier_logs.exists():
        shutil.rmtree(verifier_logs)
    verifier_logs.mkdir(mode=0o700)
    return verifier_logs


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    _check_workspace(input_data, workspace)
    identity = _agent_identity(input_data)
    _tests_absent()
    verifier = _trusted_file(_RUNTIME_METADATA / "verifier-python").read_text().strip()
    interpreter = Path(verifier)
    if (
        not interpreter.is_absolute()
        or ".." in interpreter.parts
        or any(character.isspace() for character in verifier)
        or not interpreter.is_file()
        or not os.access(interpreter, os.X_OK)
    ):
        raise RolloutEnvironmentError("Terminal template has an invalid verifier Python")
    image_path = _trusted_file(_RUNTIME_METADATA / "image-path").read_text().strip()
    if not image_path or any(not Path(part).is_absolute() for part in image_path.split(os.pathsep)):
        raise RolloutEnvironmentError("Terminal template has an invalid original PATH")
    state_path = _state_file(input_data)
    if state_path.exists() or state_path.is_symlink():
        _require_private_path(state_path)
        state_path.unlink()
    state_path.write_text(
        json.dumps(
            {
                "verifier_python": verifier,
                "image_path": image_path,
                "rollout_id": input_data.rollout_id,
            }
        ),
        encoding="utf-8",
    )
    state_path.chmod(0o600)
    await prepare_network_policy(identity.pw_name)


async def run_agent(
    input_data: RolloutInput, workspace: Path, model_environment: dict[str, str]
) -> AgentResult:
    """Run the template Claude CLI and finish all processes owned by its user."""
    instruction = input_data.task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RolloutEnvironmentError("MiMo task.instruction must be non-empty text")
    max_turns = input_data.task.get("max_turns", 80)
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise RolloutEnvironmentError("MiMo task.max_turns must be a positive integer")
    for name in ("runuser", "pkill"):
        if shutil.which(name, path=_SYSTEM_PATH) is None:
            raise RolloutEnvironmentError(f"MiMo template requires the {name} utility")
    identity = _agent_identity(input_data)
    try:
        binary = Path("/opt/mimo-agents/claude").resolve(strict=True)
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo template Claude binary is missing") from exc
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RolloutEnvironmentError("MiMo template Claude binary is not executable")
    _require_private_path(binary)
    if binary.is_relative_to(resolve_workspace(input_data)):
        raise RolloutEnvironmentError("MiMo Claude binary must be outside the task workspace")
    image_path_file = _RUNTIME_METADATA / "image-path"
    image_path = (
        image_path_file.read_text(encoding="utf-8") if image_path_file.is_file() else _SYSTEM_PATH
    )
    paths = [f"{identity.pw_dir}/.local/bin", *image_path.split(":"), *_SYSTEM_PATH.split(":")]
    environment = {
        "HOME": identity.pw_dir,
        "USER": identity.pw_name,
        "LOGNAME": identity.pw_name,
        "PATH": ":".join(dict.fromkeys(path for path in paths if path.startswith("/"))),
    }
    for name in ("LANG", "LC_ALL", "TZ"):
        if value := os.environ.get(name):
            environment[name] = value
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        value = model_environment.get(name)
        if not isinstance(value, str) or not value:
            raise RolloutEnvironmentError("MiMo Claude requires the rollout model gateway")
        environment[name] = value
    environment.update(
        {
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_UPDATES": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        }
    )
    command: list[str] = [
        "runuser",
        "--user",
        identity.pw_name,
        "--preserve-environment",
        "--",
        str(binary),
        "--bare",
        "--print",
        "--model",
        "northrl",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Bash,Read,Edit,Write,Glob,Grep",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--max-turns",
        str(max_turns),
        "--",
        instruction,
    ]
    environment = agent_network_environment(environment, identity.pw_name)
    command = agent_network_command(command, identity.pw_name)
    try:
        result = await _run_command(
            command,
            cwd=workspace,
            env=environment,
        )
        return AgentResult(exit_code=result.returncode)
    except CommandTimeoutError:
        return AgentResult(exit_code=124)
    finally:
        await _finish_cleanup(asyncio.create_task(_stop_agent_processes(input_data)))


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    del agent_result
    _check_workspace(input_data, workspace)
    _tests_absent()
    try:
        config = json.loads(_evaluation_file(evaluation_assets, "evaluation.json").read_text())
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("Invalid MiMo evaluation config") from exc
    if not isinstance(config, dict):
        raise RolloutEnvironmentError("MiMo evaluation config must be an object")
    timeout = _positive_number(config.get("verifier_timeout_sec"), "verifier timeout")
    source_tests = evaluation_assets / "tests"
    if source_tests.is_symlink() or not source_tests.is_dir():
        raise RolloutEnvironmentError("Terminal evaluation is missing its hidden tests directory")
    _evaluation_file(evaluation_assets, "tests/test.sh")
    files = []
    for source in sorted(source_tests.rglob("*")):
        if source.is_symlink() or not (source.is_file() or source.is_dir()):
            raise RolloutEnvironmentError("Terminal hidden tests contain an unsupported file")
        if source.is_file():
            relative = source.relative_to(source_tests)
            files.append((relative, _evaluation_file(evaluation_assets, "tests/" + str(relative))))
    verifier_logs = _fresh_verifier_logs()
    _TESTS.mkdir(mode=0o700)
    for relative, source in files:
        destination = _TESTS / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o600)
    state_path = _state_file(input_data)
    _require_private_path(state_path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("Invalid MiMo private rollout state") from exc
    if not isinstance(state, dict) or state.get("rollout_id") != input_data.rollout_id:
        raise RolloutEnvironmentError("MiMo private state belongs to another rollout")
    verifier_python = state["verifier_python"]
    environment = {
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "PATH": str(Path(verifier_python).parent) + os.pathsep + state["image_path"],
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    tested = await _run_command(
        ["/bin/bash", str(_TESTS / "test.sh")],
        cwd=workspace,
        env=environment,
        timeout=float(timeout),
    )
    reward_file = _trusted_file(verifier_logs / "reward.txt")
    return EvaluationResult(
        reward=parse_reward(reward_file.read_text()),
        metrics={"verifier_exit_code": float(tested.returncode)},
    )
