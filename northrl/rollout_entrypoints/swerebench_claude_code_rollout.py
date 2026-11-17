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

"""SWE-rebench task hooks: prepare Claude, execute it, and score its workspace.

The platform owns Python installation, the model gateway, trace/result files,
and the two-stage command protocol. Hidden evaluator assets are supplied only
to ``evaluate`` after the agent stage has completed.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from northrl.remote_rollout.network_policy import (
    agent_network_command,
    agent_network_environment,
    prepare_network_policy,
)

if TYPE_CHECKING:
    from northrl.remote_rollout.contracts import (
        AgentResult,
        EvaluationResult,
        RolloutInput,
    )

_CLAUDE = Path("/usr/local/bin/claude")
_CLAUDE_RUNTIME_ROOT = Path("/opt/northrl-claude")
_AGENT_USER = "northrl-agent"
_EVALUATOR = "/opt/northrl/evaluator/run-tests.sh"
_EVALUATOR_DESTINATIONS = {
    "run-tests.sh": Path(_EVALUATOR),
    "tests/config.json": Path("/tests/config.json"),
    "log_parsers.py": Path("/opt/swe_eval/log_parsers.py"),
    "swe_constants.py": Path("/opt/swe_eval/swe_constants.py"),
}
_RESULT_PATTERN = re.compile(
    r"SWEBench results starts here\s*(PASSED|FAILED)\s*SWEBench results ends here"
)
_ELF_MAGIC = b"\x7fELF"
_ROOT_UID = 0
_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    command_environment = os.environ.copy() if env is None else env.copy()
    command_environment["PATH"] = _SYSTEM_PATH
    subprocess.run(command, check=True, env=command_environment)


def _require_private_path(path: Path, *, description: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink: {path}")
    metadata = path.stat()
    if metadata.st_uid != _ROOT_UID:
        raise PermissionError(f"{description} must be owned by root: {path}")
    if metadata.st_mode & 0o022:
        raise PermissionError(f"{description} must not be group/world-writable: {path}")


def _trusted_runtime_path(workspace: Path, claude_bin: Path) -> str:
    workspace = workspace.resolve()
    raw_candidates = [
        str(claude_bin),
        *os.getenv("PATH", "").split(os.pathsep),
        *_SYSTEM_PATH.split(os.pathsep),
    ]
    trusted: list[str] = []
    for raw_candidate in raw_candidates:
        if not raw_candidate:
            continue
        candidate = Path(raw_candidate)
        try:
            if not candidate.is_absolute():
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_dir() or resolved.is_relative_to(workspace):
                continue
            _require_private_path(resolved, description="runtime PATH directory")
        except (OSError, RuntimeError):
            continue
        candidate_text = str(resolved)
        if candidate_text not in trusted:
            trusted.append(candidate_text)
    return os.pathsep.join(trusted)


def _is_elf(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(len(_ELF_MAGIC)) == _ELF_MAGIC


def _prepare_claude(workspace: Path | None = None) -> Path:
    """Copy Claude and any required Node runtime to an agent-accessible directory."""

    if not _CLAUDE.is_file():
        raise FileNotFoundError(f"Claude Code not found at {_CLAUDE}")
    claude_entrypoint = _CLAUDE.resolve()
    if not claude_entrypoint.is_file():
        raise FileNotFoundError("generic SWE-rebench template has a broken Claude install")
    if workspace is not None and claude_entrypoint.is_relative_to(workspace.resolve()):
        raise RuntimeError("Claude Code entrypoint must not come from the task workspace")
    _require_private_path(claude_entrypoint, description="Claude Code entrypoint")

    node_executable: Path | None = None
    if not _is_elf(claude_entrypoint):
        node_executable = next(
            (
                candidate.resolve()
                for parent in claude_entrypoint.parents
                if (candidate := parent / "bin/node").is_file()
            ),
            None,
        )
        if node_executable is None or not node_executable.is_file():
            raise FileNotFoundError(
                "generic SWE-rebench template does not provide Node beside Claude Code"
            )
        if workspace is not None and node_executable.is_relative_to(workspace.resolve()):
            raise RuntimeError("Node executable must not come from the task workspace")
        _require_private_path(node_executable, description="Node executable")
    if _CLAUDE_RUNTIME_ROOT.exists() or _CLAUDE_RUNTIME_ROOT.is_symlink():
        raise FileExistsError(f"Claude runtime destination already exists: {_CLAUDE_RUNTIME_ROOT}")

    runtime_bin = _CLAUDE_RUNTIME_ROOT / "bin"
    package_destination = _CLAUDE_RUNTIME_ROOT / "package"
    runtime_bin.mkdir(parents=True, mode=0o755)
    shutil.copytree(claude_entrypoint.parent, package_destination, symlinks=True)
    copied_entrypoint = package_destination / claude_entrypoint.name
    copied_launcher = runtime_bin / "claude"
    copied_node: Path | None = None
    if node_executable is not None:
        copied_node = runtime_bin / "node"
        shutil.copy2(node_executable, copied_node)
    launcher_command = (
        f'exec "{copied_entrypoint}" "$@"\n'
        if copied_node is None
        else f'exec "{copied_node}" "{copied_entrypoint}" "$@"\n'
    )
    copied_launcher.write_text(
        "#!/bin/sh\n" + launcher_command,
        encoding="utf-8",
    )

    for copied_path in _CLAUDE_RUNTIME_ROOT.rglob("*"):
        if copied_path.is_symlink():
            if not copied_path.resolve().is_relative_to(_CLAUDE_RUNTIME_ROOT):
                raise RuntimeError(f"copied Claude package symlink escapes runtime: {copied_path}")
            continue
        copied_mode = copied_path.stat().st_mode
        copied_path.chmod(0o555 if copied_path.is_dir() or copied_mode & 0o111 else 0o444)
    _CLAUDE_RUNTIME_ROOT.chmod(0o555)
    if copied_node is not None:
        copied_node.chmod(0o555)
    copied_entrypoint.chmod(0o555)
    copied_launcher.chmod(0o555)

    _CLAUDE.unlink()
    _CLAUDE.symlink_to(copied_launcher)
    return runtime_bin


def _resolve_workspace() -> Path:
    """Find the actual task checkout without creating a workspace alias."""

    candidates = sorted(
        {path.parent.resolve() for path in Path("/").glob("*/.git") if path.exists()}
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one top-level task checkout, "
            f"found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def _ensure_agent_user(workspace: Path) -> None:
    """Prepare the sandbox user and workspace permissions for Claude Code.

    Called as root during ``prepare``. Create ``northrl-agent`` if needed and
    assign task files and its home directory to that user. Keep ``.git``
    root-owned and non-writable by other users; the workspace's sticky bit
    prevents the agent from deleting or replacing that directory. Register
    the checkout as a Git safe directory to allow its mixed ownership.

    The dedicated user also lets ``_stop_agent_processes`` clean up remaining
    agent processes before hidden evaluation assets are uploaded.

    Args:
        workspace: Existing task checkout inside the sandbox.
    """
    try:
        pwd.getpwnam(_AGENT_USER)
    except KeyError:
        _run(["useradd", "--create-home", "--shell", "/bin/bash", _AGENT_USER])

    for child in workspace.iterdir():
        if child.name != ".git":
            _run(
                [
                    "chown",
                    "--recursive",
                    "--no-dereference",
                    f"{_AGENT_USER}:{_AGENT_USER}",
                    "--",
                    str(child),
                ]
            )
    git_dir = workspace / ".git"
    if git_dir.exists():
        if git_dir.is_symlink():
            raise RuntimeError(f"git metadata must not be a symlink: {git_dir}")
        _run(["chown", "--recursive", "--no-dereference", "root:root", "--", str(git_dir)])
        _run(["chmod", "--recursive", "go-w", "--", str(git_dir)])
    _run(["chown", f"root:{_AGENT_USER}", "--", str(workspace)])
    workspace.chmod(0o1775)
    _run(
        [
            "chown",
            "--recursive",
            "--no-dereference",
            f"{_AGENT_USER}:{_AGENT_USER}",
            "--",
            f"/home/{_AGENT_USER}",
        ]
    )
    _run(["git", "config", "--system", "--add", "safe.directory", str(workspace)])


async def _run_claude_code(
    input_data: RolloutInput,
    model_environment: dict[str, str],
    *,
    workspace: Path,
) -> int:
    from northrl.utils.profiling import profile_block  # pylint: disable=import-outside-toplevel

    instruction = input_data.task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("SWE-bench task.instruction must be a non-empty string")
    interface = input_data.task.get("interface")
    if interface is not None and not isinstance(interface, str):
        raise ValueError("SWE-bench task.interface must be a string when provided")
    prompt = instruction
    if interface and interface.strip():
        prompt += "\n\n" + interface

    environment = os.environ.copy()
    for name in (
        "WEAVER_API_KEY",
        "WEAVER_BASE_URL",
        "NORTHGATE_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        environment.pop(name, None)
    environment["PATH"] = _trusted_runtime_path(workspace, _CLAUDE_RUNTIME_ROOT / "bin")
    environment.update(
        {
            "HOME": f"/home/{_AGENT_USER}",
            "USER": _AGENT_USER,
            "LOGNAME": _AGENT_USER,
            "ANTHROPIC_BASE_URL": model_environment["ANTHROPIC_BASE_URL"],
            "ANTHROPIC_API_KEY": model_environment["ANTHROPIC_API_KEY"],
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    environment = agent_network_environment(environment, _AGENT_USER)

    with profile_block(
        "harness.claude-code",
        rollout_id=input_data.rollout_id,
    ):
        process = await asyncio.create_subprocess_exec(
            *agent_network_command(
                [
                    "runuser",
                    "--user",
                    _AGENT_USER,
                    "--preserve-environment",
                    "--",
                    str(_CLAUDE),
                    "--print",
                    "--bare",
                    "--model",
                    "northrl",
                    "--dangerously-skip-permissions",
                    "--no-session-persistence",
                    prompt,
                ],
                _AGENT_USER,
            ),
            cwd=workspace,
            env=environment,
        )
        return await process.wait()


def _agent_processes_running(uid: int) -> bool:
    """Check for live agent processes; zombies cannot inspect uploaded assets."""

    for process_directory in Path("/proc").iterdir():
        if not process_directory.name.isdecimal():
            continue
        try:
            status = (process_directory / "status").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue  # The process exited between listing /proc and reading it.
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        if int(fields["Uid"].split()[0]) == uid and fields["State"].split()[0] != "Z":
            return True
    return False


async def _stop_agent_processes() -> None:
    from northrl.remote_rollout.contracts import RolloutEnvironmentError

    process = await asyncio.create_subprocess_exec(
        "/usr/bin/pkill",
        "-KILL",
        "-u",
        _AGENT_USER,
    )
    return_code = await process.wait()
    if return_code not in (0, 1):
        raise RolloutEnvironmentError(f"pkill exited with status {return_code}")
    if return_code == 1:
        return  # No matching processes were present.
    uid = pwd.getpwnam(_AGENT_USER).pw_uid
    for _ in range(100):
        if not _agent_processes_running(uid):
            return
        await asyncio.sleep(0.05)
    raise RolloutEnvironmentError("agent processes remained alive after SIGKILL")


def _evaluator_environment() -> dict[str, str]:
    environment = {
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "PATH": os.environ.get("PATH", _SYSTEM_PATH),
    }
    for name in ("LANG", "LC_ALL", "TZ"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


async def _run_evaluator(workspace: Path) -> float:
    """Run the trusted evaluator as root with an explicit environment allowlist."""

    environment = _evaluator_environment()
    environment["PATH"] = _trusted_runtime_path(workspace, _CLAUDE_RUNTIME_ROOT / "bin")
    process = await asyncio.create_subprocess_exec(
        "/bin/bash",
        _EVALUATOR,
        cwd=workspace,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output_bytes, _ = await process.communicate()
    output = output_bytes.decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise RuntimeError(
            f"SWE-bench evaluator exited with status {process.returncode}:\n{output[-2000:]}"
        )

    results = _RESULT_PATTERN.findall(output)
    if results == ["PASSED"]:
        return 1.0
    if results == ["FAILED"]:
        return 0.0
    raise RuntimeError(f"SWE-bench evaluator returned invalid result markers:\n{output[-2000:]}")


def resolve_workspace(input_data: RolloutInput) -> Path:
    """Locate the checkout provided by this task's template."""

    del input_data
    return _resolve_workspace()


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    """Prepare Claude and agent workspace permissions."""

    del input_data
    _prepare_claude(workspace)
    _ensure_agent_user(workspace)
    await prepare_network_policy(_AGENT_USER)


async def run_agent(
    input_data: RolloutInput,
    workspace: Path,
    model_environment: dict[str, str],
) -> AgentResult:
    """Execute Claude and finish process cleanup on success, error, or cancellation."""

    from northrl.remote_rollout.contracts import AgentResult
    from northrl.utils.profiling import profile_block

    try:
        exit_code = await _run_claude_code(input_data, model_environment, workspace=workspace)
    finally:
        with profile_block("harness.stop-agent-processes", rollout_id=input_data.rollout_id):
            cleanup = asyncio.create_task(_stop_agent_processes())
            cancelled = False
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    if cleanup.done():
                        raise
                    # Keep cleanup alive even if cancellation is requested again.
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
    return AgentResult(exit_code=exit_code)


def _install_evaluator_assets(evaluation_assets: Path, *, task_id: str) -> None:
    """Install the evaluation-only upload; preserve its original relative paths."""

    if evaluation_assets.is_symlink() or not evaluation_assets.is_dir():
        raise RuntimeError(f"invalid evaluator asset directory: {evaluation_assets}")
    _require_private_path(evaluation_assets, description="evaluator asset directory")
    evaluation_assets.chmod(0o700)
    root = evaluation_assets.resolve()
    sources = {}
    for name in _EVALUATOR_DESTINATIONS:
        source = evaluation_assets / name
        if source.is_symlink() or source.resolve() != root / name:
            raise RuntimeError(f"evaluator asset must not traverse a symlink: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"missing SWE-rebench evaluator asset: {source}")
        _require_private_path(source, description="evaluator asset")
        sources[name] = source

    config_path = sources["tests/config.json"]
    try:
        task_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid SWE-rebench evaluator config: {config_path}") from exc
    actual_instance = task_config.get("instance_id") if isinstance(task_config, dict) else None
    if actual_instance != task_id:
        raise ValueError(
            "SWE-rebench evaluator config instance_id does not match task_id: "
            f"expected={task_id!r}, actual={actual_instance!r}"
        )

    for name, destination in _EVALUATOR_DESTINATIONS.items():
        directory = destination.parent
        if directory.is_symlink() or directory.resolve() != directory.absolute():
            raise RuntimeError(f"evaluator directory must not traverse a symlink: {directory}")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_path(directory, description="evaluator directory")
        directory.chmod(0o700)
        if destination.is_symlink():
            raise RuntimeError(f"evaluator destination must not be a symlink: {destination}")
        if destination.exists():
            _require_private_path(destination, description="evaluator destination")
            # Do not overwrite a hard-linked file from the agent workspace.
            destination.unlink()
        shutil.copyfile(sources[name], destination)
        destination.chmod(0o700 if name == "run-tests.sh" else 0o600)


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    """Install the delayed evaluator package and score the modified checkout."""

    from northrl.remote_rollout.contracts import EvaluationResult
    from northrl.utils.profiling import profile_block

    del agent_result
    task_id = input_data.metadata.get("task_id", input_data.task.get("task_id"))
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("SWE-rebench evaluation requires a non-empty metadata.task_id")
    with profile_block("entrypoint.evaluator", rollout_id=input_data.rollout_id):
        _install_evaluator_assets(evaluation_assets, task_id=task_id.strip())
        reward = await _run_evaluator(workspace)
    return EvaluationResult(
        reward=reward,
        status="success" if reward == 1.0 else "task_failed",
        metrics={"swebench_resolved": reward},
    )
