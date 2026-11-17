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

"""ARVO task hooks and its live submission verifier, served with --serve-verifier."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib import error, request

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
_IMAGE_PATH = Path("/etc/mimo-e2b/image-path")
_ROOT_UID = 0
_ROOT = Path("/root")
_PORT = 8666


class CommandTimeoutError(RolloutEnvironmentError):
    """A subprocess exceeded the task's configured execution deadline."""


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


def _agent_binary(input_data: RolloutInput) -> Path:
    try:
        path = Path("/opt/mimo-agents/claude").resolve(strict=True)
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo template Claude binary is missing") from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RolloutEnvironmentError("MiMo template Claude binary is not executable")
    _require_private_path(path)
    if path.is_relative_to(resolve_workspace(input_data)):
        raise RolloutEnvironmentError("MiMo Claude binary must be outside the task workspace")
    return path


def _runtime_environment(input_data: RolloutInput, *, user: str | None = None) -> dict[str, str]:
    identity = _agent_identity(input_data)
    selected_user = identity.pw_name if user is None else user
    home = "/root" if selected_user == "root" else identity.pw_dir
    image_path = _IMAGE_PATH.read_text(encoding="utf-8") if _IMAGE_PATH.is_file() else _SYSTEM_PATH
    # Preserve the source image's toolchain paths without inheriting model/judge credentials.
    paths = [*image_path.split(":"), *_SYSTEM_PATH.split(":")]
    if selected_user != "root":
        paths.insert(0, f"{identity.pw_dir}/.local/bin")
    else:
        trusted_paths = []
        for raw_path in paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                continue
            try:
                candidate = candidate.resolve(strict=True)
                for parent in (candidate, *candidate.parents):
                    _require_private_path(parent)
            except (OSError, RolloutEnvironmentError):
                continue
            if candidate.is_dir():
                trusted_paths.append(str(candidate))
        paths = trusted_paths
    environment = {
        "HOME": home,
        "USER": selected_user,
        "LOGNAME": selected_user,
        "PATH": ":".join(dict.fromkeys(path for path in paths if path.startswith("/"))),
    }
    for name in ("LANG", "LC_ALL", "TZ"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


def _private_directory(input_data: RolloutInput) -> Path:
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
    return _STATE_ROOT / input_data.rollout_id


def _write_state(input_data: RolloutInput, state: dict[str, Any]) -> None:
    path = _private_directory(input_data) / "state.json"
    if path.exists() or path.is_symlink():
        _require_private_path(path)
        path.unlink()
    path.write_text(json.dumps({**state, "rollout_id": input_data.rollout_id}), encoding="utf-8")
    path.chmod(0o600)


def _read_state(input_data: RolloutInput) -> dict[str, Any]:
    path = _private_directory(input_data) / "state.json"
    _require_private_path(path)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("Invalid MiMo private rollout state") from exc
    if not isinstance(state, dict) or state.get("rollout_id") != input_data.rollout_id:
        raise RolloutEnvironmentError("MiMo private state belongs to another rollout")
    return state


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
    result = subprocess.CompletedProcess(
        list(argv),
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
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
    result = await _run_command(["pkill", "-KILL", "-u", str(identity.pw_uid)], timeout=10)
    if result.returncode not in (0, 1):
        raise RolloutEnvironmentError("MiMo agent process cleanup failed")
    for _ in range(100):
        if not _agent_processes_running(identity.pw_uid):
            return
        await asyncio.sleep(0.05)
    raise RolloutEnvironmentError("MiMo agent processes remained alive after cleanup")


async def run_agent(
    input_data: RolloutInput,
    workspace: Path,
    model_environment: dict[str, str],
) -> AgentResult:
    """Run the template CLI, preserve its exit status and finish agent cleanup."""
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
    binary = _agent_binary(input_data)
    environment = _runtime_environment(input_data)
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


def _expected_config(input_data: RolloutInput) -> dict[str, str | int]:
    description = input_data.task.get("description") or input_data.task.get("instruction", "")
    if not isinstance(description, str):
        raise RolloutEnvironmentError("ARVO requires its public vulnerability description")
    function = re.search(r"in function `([^`]+)`", description)
    file = re.search(r"in file `([^`]+)`", description)
    kind = re.search(r"(\S+):\s+(\S+)\s+in function", description)
    if function is None or kind is None:
        raise RolloutEnvironmentError(
            "ARVO description must specify function, sanitizer and error type"
        )
    return {
        "function": function[1],
        "file": file[1] if file else "",
        "sanitizer": kind[1],
        "error_type": kind[2],
        "max_submits": 0,
    }


def _probe() -> None:
    """GET is queued behind POST, so 501 also confirms submission completion."""
    try:
        with request.urlopen(f"http://127.0.0.1:{_PORT}/", timeout=90):
            pass
    except error.HTTPError as exc:
        if exc.code == 501:
            return
        raise RolloutEnvironmentError("Unexpected ARVO verifier HTTP response") from exc
    raise RolloutEnvironmentError("Unexpected service on the ARVO verifier port")


async def _wait_for_verifier() -> None:
    for attempt in range(40):
        try:
            await asyncio.to_thread(_probe)
            return
        except (error.URLError, OSError) as exc:
            if attempt == 39:
                raise RolloutEnvironmentError("ARVO verifier is unavailable") from exc
            await asyncio.sleep(0.25)


def _verifier_identity() -> pwd.struct_passwd:
    try:
        identity = pwd.getpwnam("verify")
    except KeyError as exc:
        raise RolloutEnvironmentError("ARVO template verifier user is missing") from exc
    if identity.pw_uid == 0:
        raise RolloutEnvironmentError("ARVO trusted binary must run as the verifier user")
    return identity


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    identity = _agent_identity(input_data)
    _agent_binary(input_data)
    verifier = _verifier_identity()
    if (
        identity.pw_name != "agent"
        or workspace != Path("/home/agent")
        or verifier.pw_uid == identity.pw_uid
    ):
        raise RolloutEnvironmentError(
            "ARVO requires the template's distinct agent and verify users"
        )
    _require_private_path(_ROOT)
    if _ROOT.stat().st_mode & 0o077:
        raise RolloutEnvironmentError("ARVO template /root must be private")
    _require_private_path(_ROOT / "binary")
    _require_private_path(_ROOT / "binary/run.sh")
    _require_private_path(Path(__file__).resolve())
    # Fail if a baked-in or unrelated verifier occupies the port.
    import socket  # pylint: disable=import-outside-toplevel

    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", _PORT))
        except OSError as exc:
            raise RolloutEnvironmentError("ARVO verifier port is already occupied") from exc
    for name in (
        "last_result.json",
        "last_poc",
        "submit_count",
        "expected_func.json",
        "arvo-server.log",
    ):
        path = _ROOT / name
        if path.is_symlink():
            raise RolloutEnvironmentError("ARVO verifier state must not be symlinked")
        path.unlink(missing_ok=True)
    config_path = _ROOT / "expected_func.json"
    config_path.write_text(json.dumps(_expected_config(input_data)), encoding="utf-8")
    config_path.chmod(0o600)
    with (_ROOT / "arvo-server.log").open("wb") as log:
        # The verifier must outlive this interpreter: asyncio transports can
        # kill it on shutdown, and a Popen context manager would wait for it.
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-u", Path(__file__).resolve(), "--serve-verifier"],
            cwd=_ROOT,
            env=_runtime_environment(input_data, user="root"),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    _write_state(input_data, {"verifier_pid": process.pid})
    await _wait_for_verifier()
    await prepare_network_policy(identity.pw_name)


async def evaluate(
    input_data: RolloutInput, workspace: Path, agent_result: AgentResult, evaluation_assets: Path
) -> EvaluationResult:
    del workspace, agent_result, evaluation_assets
    state = _read_state(input_data)
    pid = state.get("verifier_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RolloutEnvironmentError("ARVO verifier process identity is invalid")
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise RolloutEnvironmentError("ARVO verifier process stopped before evaluation") from exc
    await _wait_for_verifier()
    path = _ROOT / "last_result.json"
    if not path.exists():
        return EvaluationResult(
            reward=0.0,
            status="task_failed",
            metrics={"mimo_arvo_resolved": 0.0, "mimo_arvo_submitted": 0.0},
        )
    _require_private_path(path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("ARVO verifier produced invalid JSON") from exc
    if not isinstance(result, dict) or any(
        not isinstance(result.get(name), bool) for name in ("crash", "match", "timeout")
    ):
        raise RolloutEnvironmentError("ARVO verifier produced an invalid result object")
    reward = float(result["crash"] and result["match"])
    metrics = {
        "mimo_arvo_resolved": reward,
        "mimo_arvo_submitted": 1.0,
        "mimo_arvo_verifier_timeout": float(result["timeout"]),
    }
    count = result.get("submit_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        metrics["mimo_arvo_submit_count"] = float(count)
    return EvaluationResult(
        reward=reward, status="success" if reward else "task_failed", metrics=metrics
    )


# The submission service below is adapted from the MIT-licensed MiMo verifier.
# MIT License
#
# Copyright (c) 2026 Xiaomi Corporation
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez (mini-swe-agent)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

_ROOT_BINARY = Path("/root/binary")
_VERIFY_HOME = Path("/home/verify")
_LAST_POC = Path("/root/last_poc")
_LAST_RESULT = Path("/root/last_result.json")
_EXPECTED_FUNC_FILE = Path("/root/expected_func.json")
_SUBMIT_COUNT_FILE = Path("/root/submit_count")

_INFRA_DIRS = ["llvm-project/", "llvm/", "libfuzzer/", "compiler-rt/", "aflplusplus/"]


def _extract_crash_function(stderr):
    """Extract the first application-level function from crash stack trace."""
    seen_frames = False
    for line in stderr.split("\n"):
        m = re.match(r"\s*#(\d+) 0x[0-9a-f]+ in (.+)\s+(\S+:\d+:\d+)\s*$", line)
        if not m:
            m = re.match(r"\s*#(\d+) 0x[0-9a-f]+ in (.+)\s+(\S+:\d+)\s*$", line)
        if not m:
            m = re.match(r"\s*#(\d+) 0x[0-9a-f]+ in (.+)\s+(/\S+)\s*$", line)
        if not m:
            continue

        frame_num = int(m.group(1))
        if frame_num == 0 and seen_frames:
            break
        seen_frames = True

        func = m.group(2).strip()
        raw_path = m.group(3)

        # Extract file path (remove :line:col)
        parts = raw_path.split(":")
        file_parts: list[str] = []
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0:
                file_parts = parts[:i]
                break
        else:
            file_parts = parts
        file_path = ":".join(file_parts)

        # Normalize and check starts with /src/
        normed = os.path.normpath(file_path)
        if not normed.startswith("/src/"):
            continue
        after_src = normed[5:]

        # Skip infra
        if any(after_src.startswith(d) for d in _INFRA_DIRS):
            continue
        if "LLVMFuzzerTestOneInput" in func:
            continue
        if "MSan" in func or "ASan" in func or "UBSan" in func or "Sanitizer" in func:
            continue

        return func, after_src

    return "", ""


def _extract_crash_type(stderr):
    """Extract sanitizer and error_type from SUMMARY line."""
    m = re.search(r"SUMMARY:\s+(\S+):\s+(\S+)", stderr)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def _load_expected():
    """Load expected config from file."""
    if _EXPECTED_FUNC_FILE.exists():
        try:
            data = json.loads(_EXPECTED_FUNC_FILE.read_text(encoding="utf-8"))
            return (
                data.get("function", ""),
                data.get("file", ""),
                data.get("sanitizer", ""),
                data.get("error_type", ""),
            )
        except Exception:
            pass
    return "", "", "", ""


# Defer private filesystem access until server startup; pure parser imports are
# useful to the adapter's local tests and require no access to /root.
_EXPECTED_FUNC, _EXPECTED_FILE, _EXPECTED_SANITIZER, _EXPECTED_ERROR_TYPE = "", "", "", ""


class _VerifyHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args, **_kwargs):
        pass

    def do_POST(self):
        if self.path != "/submit":
            self.send_error(404)
            return

        # Submit counting + limit enforcement
        try:
            count = int(_SUBMIT_COUNT_FILE.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            count = 0
        count += 1
        _SUBMIT_COUNT_FILE.write_text(str(count), encoding="utf-8")

        try:
            cfg = json.loads(_EXPECTED_FUNC_FILE.read_text(encoding="utf-8"))
            max_submits = int(cfg.get("max_submits", 0))
        except Exception:
            max_submits = 0

        if 0 < max_submits < count:
            resp = json.dumps(
                {"error": "submit_limit_exceeded", "count": count, "max_submits": max_submits}
            )
            _LAST_RESULT.write_text(resp, encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.encode())
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        boundary = content_type.split("boundary=")[1].encode()
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        parts = body.split(b"--" + boundary)
        file_data = None
        for part in parts:
            if b"filename=" in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    file_data = part[header_end + 4 :]
                    if file_data.endswith(b"\r\n"):
                        file_data = file_data[:-2]
                break

        if file_data is None:
            self.send_error(400, "No file in request")
            return

        _LAST_POC.write_bytes(file_data)

        # Setup verify environment
        verify_dir = _VERIFY_HOME / "run"
        if verify_dir.exists():
            shutil.rmtree(verify_dir)
        verify_dir.mkdir(parents=True)

        shutil.copytree(str(_ROOT_BINARY), str(verify_dir / "binary"))
        poc_path = verify_dir / "poc"
        poc_path.write_bytes(file_data)

        subprocess.run(["chown", "-R", "verify:verify", str(verify_dir)], capture_output=True)

        cmd = f"unset LC_CTYPE && cd {verify_dir} && bash {verify_dir}/binary/run.sh {poc_path}"
        try:
            result = subprocess.run(
                ["su", "verify", "-c", cmd],
                capture_output=True,
                timeout=60,
            )
            exit_code = result.returncode
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = -1
            stdout = ""
            stderr = "Timeout (60s)"
            timed_out = True

        shutil.rmtree(verify_dir, ignore_errors=True)

        crash = exit_code != 0 and not timed_out

        # Function + error type match check
        match = False
        actual_func = ""
        actual_sanitizer = ""
        actual_error_type = ""
        if crash and _EXPECTED_FUNC:
            actual_func, _ = _extract_crash_function(stderr)
            actual_sanitizer, actual_error_type = _extract_crash_type(stderr)
            match = (
                actual_func == _EXPECTED_FUNC
                and actual_sanitizer == _EXPECTED_SANITIZER
                and actual_error_type == _EXPECTED_ERROR_TYPE
            )

        result_data = {
            "exit_code": exit_code,
            "crash": crash,
            "match": match,
            "timeout": timed_out,
            "expected_func": _EXPECTED_FUNC,
            "actual_func": actual_func,
            "expected_error_type": f"{_EXPECTED_SANITIZER}: {_EXPECTED_ERROR_TYPE}",
            "actual_error_type": (
                f"{actual_sanitizer}: {actual_error_type}" if actual_sanitizer else ""
            ),
            "submit_count": count,
            "stdout": stdout,
            "stderr": stderr,
        }
        _LAST_RESULT.write_text(json.dumps(result_data, ensure_ascii=False), encoding="utf-8")

        response = json.dumps(result_data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARVO live submission verifier")
    parser.add_argument("--serve-verifier", action="store_true", required=True)
    parser.parse_args()
    _EXPECTED_FUNC, _EXPECTED_FILE, _EXPECTED_SANITIZER, _EXPECTED_ERROR_TYPE = _load_expected()
    if _EXPECTED_FUNC:
        print(
            f"Verify server: expecting `{_EXPECTED_SANITIZER}: {_EXPECTED_ERROR_TYPE}` in function `{_EXPECTED_FUNC}` in file `{_EXPECTED_FILE}`"
        )
    else:
        print("Verify server: no expected function configured (match always false)")
    server = HTTPServer(("127.0.0.1", _PORT), _VerifyHandler)
    print(f"Listening on 127.0.0.1:{_PORT}")
    server.serve_forever()
