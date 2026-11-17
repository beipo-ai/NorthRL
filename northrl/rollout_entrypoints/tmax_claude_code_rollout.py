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

"""tmax terminal tasks: run Claude Code in the task template and grade with ``run-tests.sh``.

Each tmax task has its own E2B template (Ubuntu 22.04, default user ``user`` with
passwordless sudo, work in ``/home/user``). The templates bake the grader into
``/tests``; ``prepare`` deletes it, installs Claude Code from an E2B volume onto
tmpfs, and records the grading toolchain hashes that ``run-tests.sh`` re-checks.
The hidden ``run-tests.sh`` (tests embedded, prints the SWEBench result block)
arrives only after the agent stage and is run as root by ``evaluate``.
"""

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
_ROOT_UID = 0
_WORKSPACE = Path("/home/user")
_AGENT_USER = "user"
# The recipe mounts the shared Claude Code volume here (rollout.execution.volume_mounts).
_CLAUDE_VOLUME = Path("/mnt/ccbin")
_CLAUDE_SOURCE = _CLAUDE_VOLUME / "claude"
# tmpfs: a quarter of the templates have under 600 MB free on their root partition.
_CLAUDE_ROOT = Path("/dev/shm/northrl-claude")
_CLAUDE = _CLAUDE_ROOT / "claude"
# Claude Code's own config, kept out of the task workspace (tests inspect /home/user).
_CLAUDE_CONFIG = Path("/var/lib/northrl-tmax/claude")
_STATE_ROOT = Path("/root/northrl-tmax")
_INTEGRITY_RECORD = Path("/root/.tmax_integrity")
_SYSTEM_PYTHON = "/usr/bin/python3"
_LEGACY_GRADER_PATHS = (Path("/tests"), Path("/logs/verifier"))
_RESULT_PATTERN = re.compile(
    r"SWEBench results starts here\s*(PASSED|FAILED)\s*SWEBench results ends here"
)

# Claude Code's run time is the platform agent budget (rollout.execution.agent_timeout_seconds):
# when it expires the runtime cancels run_agent, which kills Claude Code's process group, and
# the rollout is still graded.
_DEFAULTS: dict[str, Any] = {
    "evaluator_timeout_sec": 1200,
    "install_timeout_sec": 900,
    # Sent as Anthropic output_config.effort; NorthGate renders effort=high as forced
    # thinking (generation prompt ends in "<think>\n"). Empty string disables it.
    "effort": "high",
    # Proactive compaction window: Claude Code in --print mode only honours the
    # environment variable, and reactive compaction on a 400 is not in external builds.
    "compact_window": 128000,
    "max_output_tokens": 16384,
}

# Boot-time record of the grading toolchain, verified again by run-tests.sh.
_INTEGRITY_RECORDER = r"""
import glob, hashlib, os, sys, sysconfig

def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

py = os.path.realpath(sys.executable)
files = [py, "/bin/bash", "/bin/sh", "/usr/bin/env", os.path.join(sysconfig.get_paths()["stdlib"], "site.py")]
try:
    import pytest, _pytest  # noqa: E401
    files.append(os.path.realpath(pytest.__file__))
    files += sorted(glob.glob(os.path.join(os.path.dirname(os.path.realpath(_pytest.__file__)), "*.py")))
except Exception:
    pass
site_dirs = list(dict.fromkeys(sys.path + sysconfig.get_paths().get("purelib", "").split(":")))
for d in site_dirs:
    if not d or not os.path.isdir(d):
        continue
    for n in ("sitecustomize.py", "usercustomize.py"):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            files.append(p)
    files += sorted(glob.glob(os.path.join(d, "*.pth")))
lines = ["PYBIN=" + py, "SITEDIRS=" + ":".join(d for d in site_dirs if d and os.path.isdir(d))]
lines += [f"{sha(p)}  {p}" for p in dict.fromkeys(files) if os.path.isfile(p)]
os.makedirs("/root", exist_ok=True)
with open("/root/.tmax_integrity", "w") as f:
    f.write("\n".join(lines) + "\n")
os.chmod("/root/.tmax_integrity", 0o600)
print(f"[tmax-integrity] recorded {len(lines) - 2} files, python={py}")
"""


class CommandTimeoutError(RolloutEnvironmentError):
    """A trusted subprocess exceeded its deadline."""


def _option(input_data: RolloutInput, name: str) -> Any:
    return input_data.task.get(name, _DEFAULTS[name])


def _positive_number(input_data: RolloutInput, name: str) -> float:
    value = _option(input_data, name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RolloutEnvironmentError(f"tmax {name} must be a positive finite number")
    return float(value)


def _positive_int(input_data: RolloutInput, name: str) -> int:
    value = _option(input_data, name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RolloutEnvironmentError(f"tmax {name} must be a positive integer")
    return value


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


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


async def _run_command(
    argv: Sequence[str | Path],
    *,
    timeout: float | None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a trusted command in its own process group; kill the group on timeout or cancellation."""
    arguments = [str(arg) for arg in argv]
    if "/" not in arguments[0]:
        # These hooks run as root: never resolve a utility through the agent's PATH.
        executable = shutil.which(arguments[0], path=_SYSTEM_PATH)
        if executable is None:
            raise RolloutEnvironmentError(f"tmax template lacks the {arguments[0]} utility")
        arguments[0] = executable
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=cwd,
            env={"PATH": _SYSTEM_PATH} if env is None else env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise RolloutEnvironmentError(f"tmax could not start {arguments[0]}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout)
    except (asyncio.CancelledError, TimeoutError) as exc:
        await _finish_cleanup(asyncio.create_task(_kill_process_group(process)))
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise CommandTimeoutError(f"tmax command exceeded {timeout:.0f}s: {arguments[0]}") from exc
    assert process.returncode is not None
    return subprocess.CompletedProcess(
        arguments,
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _require_private_path(path: Path, *, description: str) -> None:
    if path.is_symlink():
        raise RolloutEnvironmentError(f"{description} must not be a symlink: {path}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise RolloutEnvironmentError(f"{description} is missing: {path}") from exc
    if metadata.st_uid != _ROOT_UID or metadata.st_mode & 0o022:
        raise RolloutEnvironmentError(f"{description} must be root-owned and not writable: {path}")


def _agent_identity() -> pwd.struct_passwd:
    try:
        identity = pwd.getpwnam(_AGENT_USER)
    except KeyError as exc:
        raise RolloutEnvironmentError("tmax template has no 'user' account") from exc
    if identity.pw_uid == _ROOT_UID or Path(identity.pw_dir) != _WORKSPACE:
        raise RolloutEnvironmentError("tmax agent account must be non-root with home /home/user")
    return identity


def _state_dir(input_data: RolloutInput) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", input_data.rollout_id) or input_data.rollout_id in {
        ".",
        "..",
    }:
        raise RolloutEnvironmentError("Invalid tmax rollout identity")
    for path in (_STATE_ROOT, _STATE_ROOT / input_data.rollout_id):
        if path.is_symlink():
            raise RolloutEnvironmentError("tmax private state must not traverse a symlink")
        path.mkdir(mode=0o700, exist_ok=True)
        _require_private_path(path, description="tmax private state")
        path.chmod(0o700)
    return _STATE_ROOT / input_data.rollout_id


def _remove_baked_grader() -> None:
    """Templates ship the original grader world-readable; the agent must not see it."""
    for path in _LEGACY_GRADER_PATHS:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)


async def _install_claude(input_data: RolloutInput) -> None:
    """Copy Claude Code off the shared volume, then unmount it.

    The copy keeps every later exec off the shared volume, and the unmount keeps the
    agent (which has sudo) from modifying a binary that all other rollouts read.
    """
    timeout = _positive_number(input_data, "install_timeout_sec")
    mounted = await _run_command(["mountpoint", "-q", _CLAUDE_VOLUME], timeout=60)
    if mounted.returncode != 0 or not _CLAUDE_SOURCE.is_file():
        raise RolloutEnvironmentError(
            "Claude Code volume is not mounted at /mnt/ccbin; "
            "set rollout.execution.volume_mounts in the recipe"
        )
    if _CLAUDE_ROOT.exists() or _CLAUDE_ROOT.is_symlink():
        raise RolloutEnvironmentError(f"Claude runtime destination already exists: {_CLAUDE_ROOT}")
    _CLAUDE_ROOT.mkdir(mode=0o755, parents=True)
    copied = await _run_command(["cp", _CLAUDE_SOURCE, _CLAUDE], timeout=timeout)
    if copied.returncode != 0:
        raise RolloutEnvironmentError(f"Claude Code copy failed: {copied.stderr[-500:]}")
    _CLAUDE.chmod(0o555)
    _CLAUDE_ROOT.chmod(0o555)
    unmounted = await _run_command(["umount", _CLAUDE_VOLUME], timeout=60)
    if unmounted.returncode != 0:
        unmounted = await _run_command(["umount", "-l", _CLAUDE_VOLUME], timeout=60)
    if unmounted.returncode != 0:
        raise RolloutEnvironmentError(
            f"Claude Code volume unmount failed: {unmounted.stderr[-300:]}"
        )
    version = await _run_command([_CLAUDE, "--version"], timeout=timeout, env=_minimal_env())
    if version.returncode != 0:
        raise RolloutEnvironmentError(f"Claude Code does not start: {version.stderr[-300:]}")


def _minimal_env() -> dict[str, str]:
    environment = {"PATH": _SYSTEM_PATH, "HOME": "/root", "USER": "root", "LOGNAME": "root"}
    for name in ("LANG", "LC_ALL", "TZ"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


async def _record_integrity() -> None:
    """Hash the grading toolchain with the template's own interpreter, as root."""
    if not Path(_SYSTEM_PYTHON).is_file():
        raise RolloutEnvironmentError("tmax template has no /usr/bin/python3")
    # Clean environment: the hook process carries the NorthRL runtime's PYTHONPATH.
    recorded = await _run_command(
        [_SYSTEM_PYTHON, "-"],
        timeout=120,
        env=_minimal_env(),
        stdin=_INTEGRITY_RECORDER.encode("utf-8"),
    )
    if recorded.returncode != 0 or not _INTEGRITY_RECORD.is_file():
        raise RolloutEnvironmentError(f"tmax integrity record failed: {recorded.stderr[-500:]}")


def _prepare_agent_dirs(identity: pwd.struct_passwd) -> None:
    for path in (_CLAUDE_CONFIG.parent, _CLAUDE_CONFIG):
        if path.is_symlink():
            raise RolloutEnvironmentError("Claude Code config directory must not be a symlink")
    _CLAUDE_CONFIG.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chown(_CLAUDE_CONFIG, identity.pw_uid, identity.pw_gid)
    _CLAUDE_CONFIG.chmod(0o700)


def resolve_workspace(input_data: RolloutInput) -> Path:
    """All tmax tasks work in the template user's home directory."""
    del input_data
    return _WORKSPACE


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    """Remove the baked grader, install Claude Code, and record the grading toolchain."""
    from northrl.utils.profiling import profile_block  # pylint: disable=import-outside-toplevel

    if workspace != _WORKSPACE or not workspace.is_dir():
        raise RolloutEnvironmentError("tmax tasks require the /home/user workspace")
    identity = _agent_identity()
    _remove_baked_grader()
    with profile_block("harness.claude-install", rollout_id=input_data.rollout_id):
        await _install_claude(input_data)
    _prepare_agent_dirs(identity)
    # After the NorthRL bootstrap, so its own installs are part of the recorded baseline.
    await _record_integrity()
    _state_dir(input_data)
    await prepare_network_policy(identity.pw_name)


def _agent_environment(
    input_data: RolloutInput, identity: pwd.struct_passwd, model_environment: dict[str, str]
) -> dict[str, str]:
    environment = {
        "HOME": identity.pw_dir,
        "USER": identity.pw_name,
        "LOGNAME": identity.pw_name,
        "PATH": f"{_CLAUDE_ROOT}:{_SYSTEM_PATH}",
    }
    for name in ("LANG", "LC_ALL", "TZ"):
        if value := os.environ.get(name):
            environment[name] = value
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        value = model_environment.get(name)
        if not isinstance(value, str) or not value:
            raise RolloutEnvironmentError("tmax Claude Code requires the rollout model gateway")
        environment[name] = value
    effort = _option(input_data, "effort")
    if not isinstance(effort, str):
        raise RolloutEnvironmentError("tmax effort must be a string")
    if effort:
        environment["CLAUDE_CODE_EXTRA_BODY"] = json.dumps({"output_config": {"effort": effort}})
    environment.update(
        {
            "CLAUDE_CONFIG_DIR": str(_CLAUDE_CONFIG),
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(_positive_int(input_data, "compact_window")),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(_positive_int(input_data, "max_output_tokens")),
            # Side requests use the same sampler; any other model name is rejected by the relay.
            "ANTHROPIC_SMALL_FAST_MODEL": "northrl",
            "IS_SANDBOX": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_UPDATES": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        }
    )
    return environment


def _final_result_event(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return {}


async def run_agent(
    input_data: RolloutInput, workspace: Path, model_environment: dict[str, str]
) -> AgentResult:
    """Run Claude Code as the template user until it exits or the agent budget cancels it.

    Background processes the agent started deliberately (services the tests may
    probe) are left running after a normal exit, as in the slime tmax harness.
    Claude Code's stream-json output stays in this root process and its private
    state directory, never in a file the agent can read.
    """
    from northrl.utils.profiling import profile_block  # pylint: disable=import-outside-toplevel

    instruction = input_data.task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RolloutEnvironmentError("tmax task.instruction must be non-empty text")
    identity = _agent_identity()
    environment = _agent_environment(input_data, identity, model_environment)
    command: list[str | Path] = [
        "runuser",
        "--user",
        identity.pw_name,
        "--preserve-environment",
        "--",
        _CLAUDE,
        "--print",
        "--model",
        "northrl",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--disallowedTools",
        "WebFetch,WebSearch",
        "--output-format",
        "stream-json",
        "--verbose",
        "--",
        instruction,
    ]
    with profile_block("harness.claude-code", rollout_id=input_data.rollout_id):
        # Cancellation by the agent budget kills the process group inside _run_command.
        completed = await _run_command(
            agent_network_command([str(argument) for argument in command], identity.pw_name),
            cwd=workspace,
            env=agent_network_environment(environment, identity.pw_name),
            timeout=None,
        )
    exit_code, stdout = completed.returncode, completed.stdout
    result_event = _final_result_event(stdout)
    summary = {
        "exit_code": exit_code,
        "num_turns": result_event.get("num_turns"),
        "result_subtype": result_event.get("subtype"),
        "is_error": result_event.get("is_error"),
    }
    state = _state_dir(input_data)
    (state / "agent.json").write_text(json.dumps(summary), encoding="utf-8")
    (state / "agent.json").chmod(0o600)
    (state / "claude-stdout.jsonl").write_text(stdout[-2_000_000:], encoding="utf-8")
    (state / "claude-stdout.jsonl").chmod(0o600)
    return AgentResult(exit_code=exit_code)


def _evaluator_script(evaluation_assets: Path) -> Path:
    if evaluation_assets.is_symlink() or not evaluation_assets.is_dir():
        raise RolloutEnvironmentError(f"invalid tmax evaluation directory: {evaluation_assets}")
    _require_private_path(evaluation_assets, description="tmax evaluation directory")
    evaluation_assets.chmod(0o700)
    script = evaluation_assets / "run-tests.sh"
    if script.is_symlink() or not script.is_file():
        raise RolloutEnvironmentError("tmax evaluation is missing run-tests.sh")
    _require_private_path(script, description="tmax run-tests.sh")
    return script


def _agent_metrics(input_data: RolloutInput) -> dict[str, float]:
    path = _state_dir(input_data) / "agent.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    # The runtime records agent_exit_code (124 when the agent budget stopped Claude Code).
    metrics: dict[str, float] = {}
    if isinstance(summary.get("num_turns"), int):
        metrics["claude_num_turns"] = float(summary["num_turns"])
    return metrics


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    """Run the hidden run-tests.sh as root; its result block is the binary reward."""
    from northrl.utils.profiling import profile_block  # pylint: disable=import-outside-toplevel

    del workspace, agent_result
    script = _evaluator_script(evaluation_assets)
    with profile_block("entrypoint.evaluator", rollout_id=input_data.rollout_id):
        try:
            graded = await _run_command(
                ["/bin/bash", script],
                cwd=Path("/root"),
                env=_minimal_env(),
                timeout=_positive_number(input_data, "evaluator_timeout_sec"),
            )
        except CommandTimeoutError as exc:
            raise RolloutEnvironmentError("tmax run-tests.sh exceeded its timeout") from exc
    output = graded.stdout + graded.stderr
    results = _RESULT_PATTERN.findall(output)
    if results not in (["PASSED"], ["FAILED"]):
        raise RolloutEnvironmentError(
            f"tmax run-tests.sh returned invalid result markers (exit {graded.returncode}):\n"
            f"{output[-2000:]}"
        )
    reward = 1.0 if results == ["PASSED"] else 0.0
    metrics = {
        "tmax_passed": reward,
        "tmax_integrity_mismatch": 1.0 if "INTEGRITY MISMATCH" in output else 0.0,
        **_agent_metrics(input_data),
    }
    return EvaluationResult(
        reward=reward,
        status="success" if reward == 1.0 else "task_failed",
        metrics=metrics,
    )
