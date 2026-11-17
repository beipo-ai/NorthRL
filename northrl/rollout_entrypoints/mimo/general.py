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

"""MiMo General Agent hooks preserving live MCP and the supplied rubric verifier."""

from __future__ import annotations

import asyncio
import json
import math
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

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

_ROOT_UID = 0
_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_IMAGE_PATH = Path("/etc/mimo-e2b/image-path")
_WORK = Path("/work")
_LOGS = Path("/logs")
_SESSION_ROOT = Path("/tmp/agent_output")
_RUNTIME_METADATA = Path("/etc/mimo-e2b")
_GRADER_PYTHON = Path("/opt/openai-agents-venv/bin/python")
_VERIFIER_NAMES = ("verify.py", "verifier_meta.json", "_helpers.py", "run_verify.py")


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
    check: bool = True,
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
    result = subprocess.CompletedProcess(
        list(argv),
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
    if check and result.returncode:
        raise RolloutEnvironmentError(f"MiMo subprocess exited with status {result.returncode}")
    return result


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
    result = await _run_command(
        ["pkill", "-KILL", "-u", str(identity.pw_uid)], check=False, timeout=10
    )
    if result.returncode not in (0, 1):
        raise RolloutEnvironmentError("MiMo agent process cleanup failed")
    for _ in range(100):
        if not _agent_processes_running(identity.pw_uid):
            return
        await asyncio.sleep(0.05)
    raise RolloutEnvironmentError("MiMo agent processes remained alive after cleanup")


def _evaluation_file(evaluation_assets: Path, name: str) -> Path:
    _require_private_path(evaluation_assets)
    evaluation_assets.chmod(0o700)
    path = evaluation_assets / name
    _require_private_path(path)
    if not path.is_file():
        raise RolloutEnvironmentError("MiMo evaluation asset must be a file")
    return path


def parse_reward(text: str) -> float:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise RolloutEnvironmentError("General verifier reward.json is missing or invalid") from exc
    if not isinstance(payload, dict) or "reward_error" in payload:
        raise RolloutEnvironmentError("General verifier failed or its judge was unavailable")
    reward = payload.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise RolloutEnvironmentError("General verifier reward must be a finite number in [0, 1]")
    return float(reward)


def _trusted_file(path: Path) -> Path:
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise RolloutEnvironmentError(f"General runtime file must be a regular file: {path}")
    metadata = path.stat()
    if metadata.st_uid != _ROOT_UID or metadata.st_mode & 0o022:
        raise RolloutEnvironmentError(f"General runtime file must be root-owned: {path}")
    return path


def _check_workspace(input_data: RolloutInput, workspace: Path) -> None:
    if workspace != _WORK / "workspace" or resolve_workspace(input_data) != workspace:
        raise RolloutEnvironmentError("General tasks require their original /work/workspace")


def _servers() -> list[dict[str, str]]:
    try:
        payload = json.loads(_trusted_file(_RUNTIME_METADATA / "mcp-servers.json").read_text())
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("General MCP server manifest is invalid") from exc
    if not isinstance(payload, list) or not payload:
        raise RolloutEnvironmentError("General template is missing its MCP server manifest")
    names = set()
    for row in payload:
        if not isinstance(row, dict):
            raise RolloutEnvironmentError("General template has an invalid MCP server manifest")
        name, url = row.get("name"), row.get("url")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None
            or name in names
            or not isinstance(url, str)
            or re.fullmatch(r"http://127\.0\.0\.1:[0-9]+/mcp", url) is None
        ):
            raise RolloutEnvironmentError("General template has an invalid MCP server manifest")
        names.add(name)
    return payload


async def _health() -> None:
    command = _trusted_file(_RUNTIME_METADATA / "general-ready.py")
    await _run_command([str(_GRADER_PYTHON), "-I", "-B", str(command)], timeout=30)


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    _check_workspace(input_data, workspace)
    identity = _agent_identity(input_data)
    servers = _servers()
    try:
        client_config = json.loads(_trusted_file(Path(identity.pw_dir) / "mcp.json").read_text())
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("General agent MCP configuration is invalid") from exc
    expected = {"mcpServers": {row["name"]: {"type": "http", "url": row["url"]} for row in servers}}
    if client_config != expected:
        raise RolloutEnvironmentError("General agent MCP configuration differs from its template")
    for name in _VERIFIER_NAMES:
        if (_WORK / name).exists() or (_WORK / name).is_symlink():
            raise RolloutEnvironmentError(
                "General hidden grading files must be absent before agent"
            )
    await _health()
    await _run_command(
        [
            str(_GRADER_PYTHON),
            "-I",
            "-B",
            "-c",
            "import requests, openpyxl, docx, pptx, fitz, pdfplumber",
        ],
        timeout=30,
    )
    hashes = _trusted_file(_RUNTIME_METADATA / "general-input-hashes.py")
    await _run_command([str(_GRADER_PYTHON), "-I", "-B", str(hashes), "check"], timeout=120)
    await prepare_network_policy(identity.pw_name)


def _final_answer(events: list[dict[str, Any]]) -> str | None:
    final = next((event for event in reversed(events) if event.get("type") == "result"), {})
    answer = final.get("result")
    if isinstance(answer, str) and answer.strip():
        return answer
    for event in reversed(events):
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].strip()
            ]
            if texts:
                return "\n".join(texts)
    return None


async def run_agent(
    input_data: RolloutInput, workspace: Path, model_environment: dict[str, str]
) -> AgentResult:
    identity = _agent_identity(input_data)
    servers = _servers()
    instruction = input_data.task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RolloutEnvironmentError("MiMo task.instruction must be non-empty text")
    max_turns = input_data.task.get("max_turns", 80)
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise RolloutEnvironmentError("MiMo task.max_turns must be a positive integer")
    for name in ("runuser", "pkill"):
        if shutil.which(name, path=_SYSTEM_PATH) is None:
            raise RolloutEnvironmentError(f"MiMo template requires the {name} utility")
    try:
        binary = Path("/opt/mimo-agents/claude").resolve(strict=True)
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo template Claude binary is missing") from exc
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RolloutEnvironmentError("MiMo template Claude binary is not executable")
    _require_private_path(binary)
    if binary.is_relative_to(resolve_workspace(input_data)):
        raise RolloutEnvironmentError("MiMo Claude binary must be outside the task workspace")
    image_path = _IMAGE_PATH.read_text(encoding="utf-8") if _IMAGE_PATH.is_file() else _SYSTEM_PATH
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
        ",".join(
            (
                "Bash",
                "Read",
                "Edit",
                "Write",
                "Glob",
                "Grep",
                *("mcp__" + row["name"] + "__*" for row in servers),
            )
        ),
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--max-turns",
        str(max_turns),
        "--mcp-config",
        str(Path(identity.pw_dir) / "mcp.json"),
        "--strict-mcp-config",
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
            check=False,
        )
    except CommandTimeoutError:
        result = subprocess.CompletedProcess(command, 124, "", "")
    finally:
        await _finish_cleanup(asyncio.create_task(_stop_agent_processes(input_data)))
    events = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    initial = [
        event
        for event in events
        if event.get("type") == "system" and event.get("subtype") == "init"
    ]
    connected_servers = initial[-1].get("mcp_servers", []) if initial else []
    statuses = (
        {row.get("name"): row.get("status") for row in connected_servers if isinstance(row, dict)}
        if isinstance(connected_servers, list)
        else {}
    )
    if initial and any(statuses.get(row["name"]) != "connected" for row in servers):
        raise RolloutEnvironmentError("MiMo agent MCP servers did not connect")
    return AgentResult(
        exit_code=result.returncode,
        final_answer=_final_answer(events),
    )


def _judge_environment() -> dict[str, str]:
    """Only evaluator-injected judge configuration reaches the grading subprocess."""
    environment = {}
    for suffix in ("", "2", "3"):
        names = ["GA_JUDGE_" + name + suffix for name in ("URL", "KEY", "MODEL", "API")]
        values = {name: os.environ.get(name, "").strip() for name in names}
        if suffix and not any(values.values()):
            continue
        values["GA_JUDGE_API" + suffix] = values["GA_JUDGE_API" + suffix] or "chat"
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RolloutEnvironmentError(
                "Missing evaluator judge configuration: " + ", ".join(missing)
            )
        if values["GA_JUDGE_API" + suffix] not in {"chat", "responses", "gemini"}:
            raise RolloutEnvironmentError("GA_JUDGE_API must be chat, responses, or gemini")
        try:
            url = urlsplit(values["GA_JUDGE_URL" + suffix])
        except ValueError as exc:
            raise RolloutEnvironmentError("GA_JUDGE_URL must be an HTTP(S) API base URL") from exc
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or any((url.username, url.password, url.query, url.fragment))
            or any(character.isspace() for character in values["GA_JUDGE_URL" + suffix])
        ):
            raise RolloutEnvironmentError("GA_JUDGE_URL must be an HTTP(S) API base URL")
        environment.update(values)
    for name, default in (("GA_JUDGE_PHASE_BUDGET", "600"), ("GRADE_VOTES", "1")):
        value = os.environ.get(name, default)
        try:
            valid = int(value) > 0
        except ValueError:
            valid = False
        if not valid:
            raise RolloutEnvironmentError(name + " must be a positive integer")
        environment[name] = value
    for name in ("GA_JUDGE_EFFORT", "GA_JUDGE_ROTATE"):
        if optional_value := os.environ.get(name):
            environment[name] = optional_value
    return environment


def _workspace_handoff(workspace: Path) -> None:
    def failed_walk(error: OSError) -> None:
        raise RolloutEnvironmentError("General workspace could not be inspected") from error

    for parent, directories, files in os.walk(workspace, followlinks=False, onerror=failed_walk):
        for name in directories + files:
            path = Path(parent) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                try:
                    target = path.resolve(strict=True)
                    target.relative_to(workspace)
                    mode = target.stat().st_mode
                except (ValueError, RuntimeError, OSError) as exc:
                    raise RolloutEnvironmentError(
                        "General workspace symlink escapes its workspace"
                    ) from exc
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise RolloutEnvironmentError(
                    "General workspace contains an unsupported special file"
                )


def _fresh_private_directory(path: Path) -> None:
    if path.is_symlink() or path.resolve() != path.absolute():
        raise RolloutEnvironmentError("General evaluator directory must not traverse symlinks")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(mode=0o700)


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    _check_workspace(input_data, workspace)
    judge_environment = _judge_environment()
    try:
        config = json.loads(_evaluation_file(evaluation_assets, "evaluation.json").read_text())
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("Invalid MiMo evaluation config") from exc
    if not isinstance(config, dict):
        raise RolloutEnvironmentError("MiMo evaluation config must be an object")
    timeout = _positive_number(config.get("verifier_timeout_sec"), "verifier timeout")
    verifier_files = {name: _evaluation_file(evaluation_assets, name) for name in _VERIFIER_NAMES}
    await _health()
    _workspace_handoff(workspace)
    if _LOGS.is_symlink() or _LOGS.resolve() != _LOGS.absolute():
        raise RolloutEnvironmentError("General log directory must not traverse symlinks")
    _LOGS.mkdir(mode=0o755, exist_ok=True)
    if _LOGS.stat().st_uid != _ROOT_UID or _LOGS.stat().st_mode & 0o022:
        raise RolloutEnvironmentError("General log directory must be owned by root")
    verifier_logs = _LOGS / "verifier"
    _fresh_private_directory(verifier_logs)
    (verifier_logs / "tmp").mkdir(mode=0o700)
    _fresh_private_directory(_SESSION_ROOT)
    sessions = _SESSION_ROOT / "sessions"
    sessions.mkdir(mode=0o700)
    for name, source in verifier_files.items():
        destination = _WORK / name
        if destination.is_symlink():
            raise RolloutEnvironmentError("General verifier destination must not be a symlink")
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o600)
    session = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": agent_result.final_answer or ""}],
        },
    }
    session_file = sessions / "general-final.jsonl"
    session_file.write_text(json.dumps(session, ensure_ascii=False) + "\n", encoding="utf-8")
    session_file.chmod(0o600)
    await _health()
    environment = {
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "PATH": str(_GRADER_PYTHON.parent) + os.pathsep + "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(_WORK),
        "PYTHONHOME": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(verifier_logs / "tmp"),
        **judge_environment,
    }
    # The supplied grader can print key prefixes. Do not forward its output or
    # include it in failures; reward.json is the only scoring contract.
    tested = await _run_command(
        [str(_GRADER_PYTHON), "-B", str(_WORK / "run_verify.py")],
        cwd=_WORK,
        env=environment,
        timeout=float(timeout),
        check=False,
    )
    reward_file = _trusted_file(verifier_logs / "reward.json")
    return EvaluationResult(
        reward=parse_reward(reward_file.read_text()),
        metrics={"verifier_exit_code": float(tested.returncode), "general_mcp_live": 1.0},
    )
