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

"""Execute one remote rollout in an E2B sandbox."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import shlex
import sys
import tarfile
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any, BinaryIO, Literal

from omegaconf import DictConfig, OmegaConf

from ..utils import profiling
from ..utils.e2b_transport import E2B_TCP_USER_TIMEOUT_MS, set_e2b_tcp_user_timeout
from ..utils.e2b_volume_mounts import validate_volume_mounts
from ..utils.profiling import profile_block, profile_event
from .contracts import RemoteRolloutResult, RolloutInput
from .evaluator_logs import wrap_evaluator_script
from .execution_budgets import ExecutionBudgets
from .network_policy_config import resolve_network_policy
from .result_processing import get_result_processor, result_processing, result_size_fields
from .sandbox_cleanup import schedule_sandbox_cleanup
from .trace_processing import TraceSnapshotWriter, trace_snapshot_writer
from .wire import (
    CompactRolloutTrace,
    decode_result,
    decode_trace,
    get_download_queue,
    load_result_gzip,
    load_trace_gzip,
    run_background_io,
    run_blocking,
)

logger = logging.getLogger(__name__)

_INPUT_PATH = "/tmp/northrl-rollout-input.json"
_RESULT_PATH = "/opt/northrl-result/result.json.gz"
_TRACE_PATH = "/opt/northrl-result/trace.json.gz"
_AGENT_STAGE_PATH = "/opt/northrl-result/agent-stage.json.gz"
_EVALUATION_ARCHIVE_PATH = "/tmp/northrl-evaluation.tar.gz"
_EVALUATION_ROOT = "/opt/northrl-evaluation"
_EVALUATION_READY_PATH = "/opt/northrl-result/evaluation-ready"
_EVALUATION_TERMINAL_PATH = "/opt/northrl-result/evaluation-terminal.json"
_EVALUATOR_LOG_ROOT = "/opt/northrl-result/evaluator"
_EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS = 10.0
_EVALUATION_COMMAND_CLEANUP_SECONDS = 5.0
_SANDBOX_KILL_TIMEOUT_SECONDS = 10.0
_IO_RETRY_INITIAL_DELAY_SECONDS = 1.0
_IO_RETRY_MAX_DELAY_SECONDS = 10.0
_TRACE_SYNC_INTERVAL_SECONDS = 180
_DEFAULT_ENV_ERROR_RETRIES = 3
_SANDBOX_CREATE_RETRY_INITIAL_DELAY_SECONDS = 1.0
_SANDBOX_CREATE_RETRY_MAX_DELAY_SECONDS = 60.0
_STATE_PATH = "/opt/northrl-result/result.json.state.json"
_METRICS_PATH = "/opt/northrl-result/metrics.json"
_REQUESTS_PATH = "/opt/northrl-result/requests.jsonl"
_LIVE_METRICS_POLL_INTERVAL_SECONDS = 300.0
_LIVE_METRICS_READ_TIMEOUT_SECONDS = 5.0
_E2B_TCP_USER_TIMEOUT_MS = E2B_TCP_USER_TIMEOUT_MS
_ARTIFACT_PROGRESS_INTERVAL_SECONDS = 5.0
_SOURCE_ARCHIVE_PATH = "/tmp/northrl-source.tar.gz"
_SOURCE_ROOT = "/opt/northrl-src"
_BOOTSTRAP_INTERPRETER = "/opt/northrl-runtime/bin/python"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_SCRIPT_PATH = _PACKAGE_ROOT / "remote_rollout" / "e2b_bootstrap.sh"
_BOOTSTRAP_SCRIPT_ARCHIVE_PATH = (
    Path("northrl-src") / "northrl" / "remote_rollout" / "e2b_bootstrap.sh"
)
_BOOTSTRAP_SCRIPT_RUNTIME_PATH = f"{_SOURCE_ROOT}/northrl/remote_rollout/e2b_bootstrap.sh"
_BOOTSTRAP_COMMAND = f"/bin/sh {_BOOTSTRAP_SCRIPT_RUNTIME_PATH}"
_UPLOADED_ENTRYPOINT_ARCHIVE_PATH = Path("northrl-src") / "entrypoint.py"
_UPLOADED_ENTRYPOINT_PATH = f"{_SOURCE_ROOT}/entrypoint.py"
_TASK_INSTANCE_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_TASK_INSTANCE_MAX_LENGTH = 255
_DOWNLOAD_BUDGET: ContextVar[tuple[float, float | None]] = ContextVar(
    "northrl_download_budget", default=(60.0, None)
)


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _trace_stem(destination: Path) -> str:
    return Path(destination.name.removesuffix(".gz")).stem


async def _run_blocking_until(deadline: float, function, *args, **kwargs):
    """Bound admission and processing while still draining an active worker."""
    remaining = _remaining(deadline)
    if remaining <= 0:
        raise TimeoutError("artifact processing deadline expired")
    return await asyncio.wait_for(run_background_io(function, *args, **kwargs), remaining)


async def _gather_and_drain(*awaitables):
    """Cancellation must finish every child's active I/O before cleanup returns."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        drained = asyncio.gather(*tasks, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        drained.result()
        raise


async def _download_file(
    sandbox,
    path: str,
    *,
    decode: Callable[[Any], Any] | None = None,
    blocking_decode: bool = True,
    file_format: str | None = None,
    priority: Literal["critical", "control", "background"] = "critical",
    io_seconds: float | None = None,
    deadline: float | None = None,
    request_timeout: float | None = None,
    quiet_success: bool = False,
) -> Any:
    """Hold one shared download slot through network I/O and bounded processing."""
    inherited_seconds, inherited_deadline = _DOWNLOAD_BUDGET.get()
    if io_seconds is None:
        io_seconds = inherited_seconds
    if deadline is None:
        deadline = inherited_deadline

    queue = get_download_queue()
    started_at = time.monotonic()
    admitted_at = read_finished_at = processing_queued_at = None
    processing_started_at = processing_finished_at = None
    byte_count = None
    phase = "queue"
    submitted_active, submitted_pending = queue.active_count, queue.pending_count

    def process(raw):
        nonlocal processing_started_at, processing_finished_at, phase
        processing_started_at = time.monotonic()
        phase = "processing"
        try:
            assert decode is not None
            return decode(raw)
        finally:
            processing_finished_at = time.monotonic()

    async def process_download(raw):
        nonlocal processing_queued_at, phase
        processing_queued_at = time.monotonic()
        phase = "processing_queue"
        if blocking_decode:
            runner = run_blocking if path == _RESULT_PATH else run_background_io
            return await runner(process, raw)
        return process(raw)

    async def operation():
        nonlocal admitted_at, read_finished_at, byte_count, phase
        admitted_at = time.monotonic()
        phase = "read"
        kwargs: dict[str, Any] = {"user": "root"}
        if file_format is not None:
            kwargs["format"] = file_format
        timeout = io_seconds if deadline is None else min(io_seconds, _remaining(deadline))
        if request_timeout is not None:
            kwargs["request_timeout"] = min(request_timeout, timeout)
        try:
            raw = await asyncio.wait_for(sandbox.files.read(path, **kwargs), timeout)
        finally:
            read_finished_at = time.monotonic()
        if isinstance(raw, (bytes, bytearray)):
            byte_count = len(raw)
        elif isinstance(raw, memoryview):
            byte_count = raw.nbytes
        elif isinstance(raw, str):
            byte_count = len(raw.encode("utf-8"))
        # Diagnostics retain their deadline. Final results are processed after
        # the remote timeout ends, while still owning their download slot.
        if decode is None or path == _RESULT_PATH:
            return raw
        return await process_download(raw)

    def emit_profile():
        finished_at = time.monotonic()
        duration = finished_at - started_at
        if error is not None:
            quiet = isinstance(error, asyncio.CancelledError) or (
                priority in {"background", "control"} and _is_missing_file(error)
            )
            level = logging.DEBUG if quiet else logging.WARNING
        elif quiet_success:
            level = logging.DEBUG
        else:
            level = logging.INFO if path == _RESULT_PATH or duration >= 5.0 else logging.DEBUG
        if not profiling.logger.isEnabledFor(level):
            return

        fields = {
            "sandbox_id": str(getattr(sandbox, "sandbox_id", "")),
            "path": path,
            "priority": priority,
            "phase": phase,
            "queue_wait_seconds": (admitted_at if admitted_at is not None else finished_at)
            - started_at,
            "read_seconds": (
                (read_finished_at if read_finished_at is not None else finished_at) - admitted_at
                if admitted_at is not None
                else 0.0
            ),
            "bytes": byte_count,
            "processing_queue_wait_seconds": (
                (processing_started_at if processing_started_at is not None else finished_at)
                - processing_queued_at
                if processing_queued_at is not None
                else 0.0
            ),
            "processing_seconds": (
                (processing_finished_at if processing_finished_at is not None else finished_at)
                - processing_started_at
                if processing_started_at is not None
                else 0.0
            ),
            "duration_seconds": duration,
            "queue_active_at_submit": submitted_active,
            "queue_pending_at_submit": submitted_pending,
        }
        if error is not None:
            fields["error_type"] = type(error).__name__
            fields["error_message"] = str(error)[:512]
        profile_event(
            "driver.file-download",
            "completed" if error is None else "failed",
            level=level,
            **fields,
        )

    error = None
    try:
        result = await queue.run(
            operation,
            priority=priority,
            deadline=deadline,
            postprocess=process_download if decode is not None and path == _RESULT_PATH else None,
        )
        phase = "completed"
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            emit_profile()
        except Exception:
            # Diagnostics must not replace a transfer failure or invalidate an
            # artifact when formatting or a logging handler is unavailable.
            pass


_EXTRACT_COMMAND = f"tar --no-same-owner -xzf {_SOURCE_ARCHIVE_PATH} -C /opt"
_EXTRACT_EVALUATION_COMMAND = (
    f"umask 077; test ! -L {_EVALUATION_ROOT} && mkdir -p {_EVALUATION_ROOT} && "
    f"chmod 700 {_EVALUATION_ROOT} && "
    f"tar --no-same-owner --no-same-permissions -xzf {_EVALUATION_ARCHIVE_PATH} "
    f"-C {_EVALUATION_ROOT} && : > {_EVALUATION_READY_PATH} && rm -f {_EVALUATION_ARCHIVE_PATH}"
)


@dataclass(frozen=True)
class _EntrypointSpec:
    script_path: Path
    task_files_root: Path | None = None
    workspace: str | None = None


@dataclass(frozen=True)
class PreparedE2BArtifacts:
    """Immutable host assets shared by all workers and environment retries."""

    entrypoint: _EntrypointSpec
    source_archive: bytes
    evaluation_archives: Mapping[str | None, bytes]


def _resolve_entrypoint(config: DictConfig) -> _EntrypointSpec:
    """Resolve and validate the configured sandbox entrypoint."""

    entrypoint_config = OmegaConf.select(config, "rollout.execution.entrypoint")
    if entrypoint_config is None:
        raise ValueError("rollout.execution.entrypoint is required for the E2B rollout executor")
    if not isinstance(entrypoint_config, (dict, DictConfig)):
        raise ValueError("rollout.execution.entrypoint must be a mapping")

    target = OmegaConf.select(config, "rollout.execution.entrypoint.target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("rollout.execution.entrypoint.target must be a non-empty string")

    configured_task_files_root = OmegaConf.select(
        config,
        "rollout.execution.entrypoint.task_files_root",
    )
    task_files_root = None
    if configured_task_files_root is not None:
        if (
            not isinstance(configured_task_files_root, str)
            or not configured_task_files_root.strip()
        ):
            raise ValueError(
                "rollout.execution.entrypoint.task_files_root must be a non-empty string"
            )
        task_files_root = Path(configured_task_files_root.strip())
        if not task_files_root.is_absolute():
            raise ValueError(
                "rollout.execution.entrypoint.task_files_root must be an absolute host path"
            )
        task_files_root = task_files_root.resolve()

    target = target.strip()
    workspace = OmegaConf.select(config, "rollout.execution.entrypoint.workspace")
    if workspace is not None:
        if not isinstance(workspace, str) or not Path(workspace.strip()).is_absolute():
            raise ValueError("rollout.execution.entrypoint.workspace must be an absolute path")
        workspace = workspace.strip()
    script_path = Path(target)
    if not script_path.is_absolute():
        raise ValueError("rollout.execution.entrypoint.target must be an absolute host path")
    script_path = script_path.resolve()
    if script_path.suffix != ".py":
        raise ValueError("rollout.execution.entrypoint.target must point to a .py file")
    if not script_path.is_file():
        raise FileNotFoundError(f"Rollout entrypoint script not found: {script_path}")
    return _EntrypointSpec(
        script_path=script_path,
        task_files_root=task_files_root,
        workspace=workspace,
    )


def _resolve_selected_environment(selector: str, reserved: set[str]) -> dict[str, str]:
    """Select explicit host variables without persisting their values."""
    configured = os.environ.get(selector, "").strip()
    names = [name.strip() for name in configured.split(",")] if configured else []
    seen: set[str] = set()
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"{selector} must contain comma-separated environment variable names")
        if name in reserved:
            raise ValueError(f"{selector} cannot override reserved runtime variable {name}")
        if name in seen:
            raise ValueError(f"{selector} contains duplicate variable {name}")
        seen.add(name)
    return {name: os.environ[name] for name in names if name in os.environ}


def _resolve_evaluator_environment() -> dict[str, str]:
    """Select host variables for evaluation without persisting their values."""
    return _resolve_selected_environment(
        "NORTHRL_E2B_EVALUATOR_ENV", {"PYTHONPATH", "PYTHONSAFEPATH", "LOG_LEVEL"}
    )


def _resolve_agent_environment() -> dict[str, str]:
    """Forward optional tool settings only to the agent command, never bootstrap."""
    return _resolve_selected_environment(
        "NORTHRL_E2B_AGENT_ENV",
        {
            "PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSAFEPATH",
            "LOG_LEVEL",
            "NORTHGATE_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_API_KEY",
        },
    )


def _resolve_task_network_policy(config: DictConfig, input_data: RolloutInput) -> dict[str, Any]:
    """Select only host-configured rules using the rollout's metadata identity."""
    policy = config.rollout.execution.network_policy
    if OmegaConf.is_config(policy):
        policy = OmegaConf.to_container(policy, resolve=True)
    task_id = input_data.metadata.get("task_id")
    if task_id is not None and not isinstance(task_id, str):
        raise ValueError("network_policy requires metadata.task_id to be a nonblank string")
    return resolve_network_policy(policy, task_id)


def _build_entrypoint_command(entrypoint: _EntrypointSpec, *, phase: str = "agent") -> str:
    """Start the platform runtime, which imports the task's hooks inside E2B."""

    runner = f"import runpy, sys; sys.path.insert(0, {_SOURCE_ROOT!r}); "
    runner += "runpy.run_module('northrl.remote_rollout.e2b_runtime', run_name='__main__')"
    arguments = [
        "--phase",
        phase,
        "--entrypoint",
        _UPLOADED_ENTRYPOINT_PATH,
        "--input",
        _INPUT_PATH,
        "--output",
        _RESULT_PATH,
        "--trace-output",
        _TRACE_PATH,
        "--stage-path",
        _AGENT_STAGE_PATH,
        "--evaluation-assets",
        _EVALUATION_ROOT,
    ]
    if entrypoint.workspace is not None:
        arguments.extend(["--workspace", entrypoint.workspace])
    # The platform bootstrap installs all gateway dependencies before either phase.
    return shlex.join([_BOOTSTRAP_INTERPRETER, "-I", "-c", runner, *arguments])


# This supervisor stays outside the evaluator's process group. It must remain
# alive after the runtime exits so inherited pipes or TERM-ignoring descendants
# cannot outlive a completed evaluation command. New sessions still require
# the executor's final sandbox termination.
_EVALUATION_SUPERVISOR_CODE = """import json
import os
import signal
import subprocess
import sys
import tempfile
import time


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
deadline = time.monotonic() + float(sys.argv[1])
cleanup_seconds = float(sys.argv[2])
terminal_path, rollout_id, attempt_id = sys.argv[3:6]
process = None
exit_code = 1
try:
    process = subprocess.Popen(sys.argv[6:], start_new_session=True)
    try:
        exit_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        exit_code = 124
except SystemExit as exc:
    exit_code = int(exc.code)
finally:
    if process is not None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=cleanup_seconds)
        except subprocess.TimeoutExpired:
            pass
        else:
            # The evaluator cannot publish another result after its process
            # group has been killed and its main process has been reaped.
            temporary = None
            try:
                descriptor, temporary = tempfile.mkstemp(
                    prefix='.evaluation-terminal.', dir=os.path.dirname(terminal_path)
                )
                with os.fdopen(descriptor, 'w') as stream:
                    json.dump({
                        'rollout_id': rollout_id,
                        'attempt_id': attempt_id,
                        'phase': 'evaluation_finished',
                        'exit_code': exit_code if exit_code >= 0 else 128 - exit_code,
                    }, stream)
                os.replace(temporary, terminal_path)
                temporary = None
            except OSError as exc:
                print('Evaluation terminal record unavailable: ' + str(exc), file=sys.stderr)
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
sys.exit(exit_code if exit_code >= 0 else 128 - exit_code)
"""


class _EvaluationCommandTimeout(TimeoutError):
    """A supervisor-confirmed timeout; no evaluator is still publishing a result."""

    exit_code = 124


def _supervise_evaluation_command(
    command: str,
    timeout_seconds: float,
    *,
    rollout_id: str,
    attempt_id: str,
    terminal_path: str = _EVALUATION_TERMINAL_PATH,
) -> str:
    """Bound evaluation and clean its process group without changing task hooks."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise TimeoutError("evaluation deadline expired before command dispatch")
    return shlex.join(
        [
            _BOOTSTRAP_INTERPRETER,
            "-I",
            "-c",
            _EVALUATION_SUPERVISOR_CODE,
            str(timeout_seconds),
            str(_EVALUATION_COMMAND_CLEANUP_SECONDS),
            terminal_path,
            rollout_id,
            attempt_id,
            *shlex.split(command),
        ]
    )


def _e2b_log(message: str, *, stderr: bool = False) -> None:
    """Print lifecycle logs even when the caller has not configured logging."""

    print(f"[E2B] {message}", file=sys.stderr if stderr else sys.stdout, flush=True)


def _emit_event(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    """Keep observability failures independent from rollout outcomes."""

    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _e2b_log(f"metrics callback failed ({type(exc).__name__}: {exc})", stderr=True)


@dataclass
class _LiveMetricsState:
    """Share first-heartbeat state between periodic polling and the final read."""

    received_snapshot: bool = False


async def _read_live_metrics(
    sandbox,
    *,
    rollout_id: str,
    event_callback: Callable[[dict[str, Any]], None],
    timeout_seconds: float,
    state: _LiveMetricsState | None = None,
) -> None:
    """Read only the small heartbeat file, with a bounded recovery cost."""

    def decode(raw):
        snapshot = json.loads(raw)
        if not isinstance(snapshot, dict):
            raise ValueError("live metrics must be an object")
        snapshot_id = snapshot.get("rollout_id")
        if not isinstance(snapshot_id, str):
            raise ValueError("live metrics rollout_id must be a string")
        _validate_rollout_id(snapshot_id, rollout_id, "live metrics")
        heartbeat = snapshot.get("heartbeat")
        if isinstance(heartbeat, bool) or not isinstance(heartbeat, int) or heartbeat < 0:
            raise ValueError("live metrics heartbeat must be a non-negative integer")
        phase = snapshot.get("phase")
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("live metrics phase must be a non-empty string")
        metrics = snapshot.get("metrics")
        if not isinstance(metrics, dict) or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for key, value in metrics.items()
        ):
            raise ValueError("live metrics values must be finite numbers")
        return snapshot

    try:
        snapshot = await _download_file(
            sandbox,
            _METRICS_PATH,
            decode=decode,
            # Small runtime-generated heartbeat metadata must not wait behind
            # large gzip/trace jobs in the two-worker processing pool.
            blocking_decode=False,
            priority="control",
            io_seconds=timeout_seconds,
            deadline=time.monotonic() + timeout_seconds,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        file_missing = isinstance(exc, FileNotFoundError) or any(
            cls.__module__ == "e2b.exceptions" and cls.__name__ == "FileNotFoundException"
            for cls in type(exc).__mro__
        )
        if state is not None and not state.received_snapshot and file_missing:
            # The runtime creates this file after environment preparation. A
            # missing sandbox or failed connection still reports a poll error.
            return
        _emit_event(event_callback, kind="poll_error", error_type=type(exc).__name__)
        return
    if state is not None:
        state.received_snapshot = True
    _emit_event(
        event_callback,
        kind="snapshot",
        heartbeat=snapshot["heartbeat"],
        phase=snapshot["phase"],
        metrics={key: float(value) for key, value in snapshot["metrics"].items()},
    )


async def _poll_live_metrics(
    sandbox,
    *,
    rollout_id: str,
    event_callback: Callable[[dict[str, Any]], None],
    state: _LiveMetricsState,
) -> None:
    while True:
        await _read_live_metrics(
            sandbox,
            rollout_id=rollout_id,
            event_callback=event_callback,
            timeout_seconds=_LIVE_METRICS_READ_TIMEOUT_SECONDS,
            state=state,
        )
        await asyncio.sleep(_LIVE_METRICS_POLL_INTERVAL_SECONDS)


def _stream_e2b_stdout(output: str) -> None:
    """Forward an E2B stdout chunk without buffering it locally."""

    sys.stdout.write(output)
    sys.stdout.flush()


def _stream_e2b_stderr(output: str) -> None:
    """Forward an E2B stderr chunk without buffering it locally."""

    sys.stderr.write(output)
    sys.stderr.flush()


def _set_e2b_tcp_user_timeout(sandbox) -> None:
    """Configure established connections through the shared transport helper."""

    set_e2b_tcp_user_timeout(sandbox, timeout_ms=_E2B_TCP_USER_TIMEOUT_MS)


async def _download_request_journal(
    sandbox,
    destination: Path,
    *,
    rollout_id: str,
    attempt_id: str,
    log_context: str,
) -> None:
    """Recover completed request records before cleanup, including failed attempts."""

    def save(raw):
        # A still-running process can be interrupted mid-append; retain complete lines.
        lines = [line for line in raw.splitlines(keepends=True) if line.endswith("\n")]
        for line in lines:
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or record.get("rollout_id") != rollout_id
                or record.get("attempt_id") != attempt_id
            ):
                raise ValueError("request journal identity does not match this attempt")
        if not lines:
            return
        # Separate filenames keep all retry attempts without duplicate appends.
        target = destination.with_name(f"{_trace_stem(destination)}.requests.{attempt_id}.jsonl")
        _atomic_write_bytes(target, "".join(lines).encode("utf-8"))

    try:
        await _download_file(
            sandbox,
            _REQUESTS_PATH,
            decode=save,
            priority="background",
            io_seconds=_LIVE_METRICS_READ_TIMEOUT_SECONDS,
            deadline=time.monotonic() + _LIVE_METRICS_READ_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        missing = isinstance(exc, FileNotFoundError) or any(
            cls.__module__ == "e2b.exceptions" and cls.__name__ == "FileNotFoundException"
            for cls in type(exc).__mro__
        )
        if not missing:
            _e2b_log(
                f"{log_context} request journal unavailable ({type(exc).__name__}: {exc})",
                stderr=True,
            )


async def _download_evaluator_logs(
    sandbox,
    destination: Path,
    *,
    rollout_id: str,
    attempt_id: str,
    log_context: str,
) -> None:
    """Stream private evaluator output to the host, retaining partial downloads."""
    target = destination.with_name(f"{_trace_stem(destination)}.evaluation.{attempt_id}")
    status: dict[str, Any] | None = None
    downloads: dict[str, Any] = {}

    deadline = time.monotonic() + _EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS
    # Leave part of the existing grace period to record partial-transfer status.
    transfer_deadline = deadline - min(1.0, _EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS / 10)

    def open_output(name, opened):
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.chmod(0o700)
        output = (target / f"{name}.log").open("wb")
        opened.append(output)
        os.chmod(output.name, 0o600)

    async def download_stream(name: str) -> None:
        info: dict[str, Any] = {"bytes": 0, "complete": False}
        downloads[name] = info

        async def consume():
            stream = await asyncio.wait_for(
                sandbox.files.read(
                    f"{_EVALUATOR_LOG_ROOT}/{name}.log", format="stream", user="root"
                ),
                _remaining(transfer_deadline),
            )
            async with aclosing(stream):
                opened: list[BinaryIO] = []
                try:
                    # Publish ownership inside the worker so cancellation after
                    # open() still reaches the close in this finally block.
                    await _run_blocking_until(transfer_deadline, open_output, name, opened)
                    output = opened[0]
                    iterator = aiter(stream)
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                anext(iterator), _remaining(transfer_deadline)
                            )
                        except StopAsyncIteration:
                            break
                        await _run_blocking_until(transfer_deadline, output.write, chunk)
                        info["bytes"] += len(chunk)
                finally:
                    if opened:
                        # An already-open resource must be closed even after
                        # admission for new diagnostic work has expired.
                        await run_background_io(opened[0].close)
            expected = status.get("bytes", {}).get(name) if status is not None else None
            if (
                status
                and status.get("finished")
                and isinstance(expected, int)
                and info["bytes"] != expected
            ):
                raise ValueError("Evaluator log download size does not match completed capture")
            info["complete"] = True

        try:
            await get_download_queue().run(
                consume, priority="background", deadline=transfer_deadline
            )
        except asyncio.CancelledError:
            info["error"] = "download interrupted"
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            info["error"] = f"{type(exc).__name__}: {exc}"

    try:
        # Share a fixed grace budget across status, both streams and cleanup.
        async with asyncio.timeout(_EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS):
            candidate = await _download_file(
                sandbox,
                f"{_EVALUATOR_LOG_ROOT}/status.json",
                decode=json.loads,
                priority="background",
                deadline=transfer_deadline,
                io_seconds=_EVALUATOR_LOG_DOWNLOAD_TIMEOUT_SECONDS,
            )
            if candidate is None:
                return
            if (
                not isinstance(candidate, dict)
                or candidate.get("rollout_id") != rollout_id
                or candidate.get("attempt_id") != attempt_id
            ):
                raise ValueError("Evaluator log identity does not match this attempt")
            status = candidate
            await _gather_and_drain(download_stream("stdout"), download_stream("stderr"))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        missing = isinstance(exc, FileNotFoundError) or any(
            cls.__module__ == "e2b.exceptions" and cls.__name__ == "FileNotFoundException"
            for cls in type(exc).__mro__
        )
        if not missing:
            _e2b_log(
                f"{log_context} evaluator logs unavailable ({type(exc).__name__}: {exc})",
                stderr=True,
            )
    finally:
        if status is not None:
            try:
                status["downloads"] = downloads
                status_path = target / "status.json"
                await _run_blocking_until(
                    deadline,
                    _atomic_write_bytes,
                    status_path,
                    (json.dumps(status, separators=(",", ":")) + "\n").encode("utf-8"),
                )
                _e2b_log(f"{log_context} evaluator logs saved path={str(target)!r}")
            except (OSError, TimeoutError) as exc:
                _e2b_log(
                    f"{log_context} evaluator logs could not be saved ({type(exc).__name__}: {exc})",
                    stderr=True,
                )


async def _download_trace(
    sandbox,
    destination: Path | None,
    *,
    log_context: str,
    rollout_id: str,
    quiet: bool = False,
    read_without_destination: bool = False,
    raise_read_errors: bool = False,
    priority: Literal["critical", "background"] = "critical",
    io_seconds: float | None = None,
    deadline: float | None = None,
    snapshot_writer: TraceSnapshotWriter | None = None,
) -> CompactRolloutTrace | None:
    if destination is None and not read_without_destination:
        return None

    decode_seconds = local_save_seconds = 0.0
    local_save_status = "not_started" if destination is not None else "not_requested"
    processing_status = None

    def decode_and_save(payload):
        nonlocal processing_status, decode_seconds, local_save_seconds, local_save_status
        processing_status = "failed"
        if snapshot_writer is not None:
            decode_seconds, local_save_seconds, local_save_status = snapshot_writer.save(payload)
            processing_status = "completed"
            return None
        decode_started = time.perf_counter()
        try:
            trace = load_trace_gzip(payload)
            _validate_rollout_id(trace.rollout_id, rollout_id, "partial trace")
        finally:
            decode_seconds = time.perf_counter() - decode_started
        if destination is not None:
            save_started = time.perf_counter()
            try:
                # Keep the received compact gzip bytes unchanged. The I/O worker
                # drains cancellation before a retry can clear or reuse this path.
                _atomic_write_bytes(destination, payload)
                local_save_status = "saved"
            except OSError as exc:
                local_save_status = "failed"
                if not quiet:
                    _e2b_log(f"{log_context} partial trace could not be saved ({exc})", stderr=True)
            finally:
                local_save_seconds = time.perf_counter() - save_started
        processing_status = "completed"
        return trace

    try:
        trace = await _download_file(
            sandbox,
            _TRACE_PATH,
            decode=decode_and_save,
            file_format="bytes",
            priority=priority,
            io_seconds=io_seconds,
            deadline=deadline,
            quiet_success=snapshot_writer is not None,
        )
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        if not isinstance(exc, Exception) or raise_read_errors:
            raise
        if not quiet:
            _e2b_log(
                f"{log_context} partial trace unavailable ({type(exc).__name__}: {exc})",
                stderr=True,
            )
        return None
    finally:
        # Network outcomes are reported by _download_file. Only add details
        # here when processing actually ran, after its worker has drained.
        if processing_status is not None:
            if processing_status == "failed":
                level = logging.WARNING
            elif snapshot_writer is not None:
                level = logging.DEBUG
            else:
                level = (
                    logging.INFO if decode_seconds + local_save_seconds >= 5.0 else logging.DEBUG
                )
            if profiling.logger.isEnabledFor(level):
                try:
                    profile_event(
                        "driver.trace-processing",
                        processing_status,
                        level=level,
                        sandbox_id=str(getattr(sandbox, "sandbox_id", "unknown")),
                        rollout_id=rollout_id,
                        decode_seconds=decode_seconds,
                        local_save_seconds=local_save_seconds,
                        local_save_status=local_save_status,
                    )
                except Exception:
                    pass  # Diagnostics must not change the trace outcome.
    if trace is not None and not quiet:
        _e2b_log(
            f"{log_context} partial trace loaded segments={len(trace.segments)} messages={len(trace.messages)}"
        )
    return trace


def _load_locally_synced_trace(
    destination: Path | None,
    *,
    log_context: str,
    rollout_id: str,
) -> CompactRolloutTrace | None:
    """Load the last trace saved by the periodic sync task, if available."""

    if destination is None:
        return None
    try:
        trace = load_trace_gzip(destination.read_bytes())
        _validate_rollout_id(trace.rollout_id, rollout_id, "locally synced partial trace")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _e2b_log(
            f"{log_context} locally synced partial trace unavailable "
            f"path={str(destination)!r} ({type(exc).__name__}: {exc})",
            stderr=True,
        )
        return None

    _e2b_log(
        f"{log_context} using locally synced partial trace "
        f"path={str(destination.resolve())!r} "
        f"segments={len(trace.segments)} messages={len(trace.messages)}"
    )
    return trace


async def _load_local_trace_until(
    destination: Path | None, *, deadline: float, log_context: str, rollout_id: str
) -> CompactRolloutTrace | None:
    if destination is None or _remaining(deadline) <= 0:
        return None
    try:
        return await _run_blocking_until(
            deadline,
            _load_locally_synced_trace,
            destination,
            log_context=log_context,
            rollout_id=rollout_id,
        )
    except TimeoutError:
        return None


async def _recover_trace(
    sandbox,
    destination: Path | None,
    *,
    log_context: str,
    rollout_id: str,
    deadline: float,
    io_seconds: float,
) -> CompactRolloutTrace | None:
    delay = _IO_RETRY_INITIAL_DELAY_SECONDS
    missing_reads = 0
    while True:
        try:
            return await _download_trace(
                sandbox,
                destination,
                log_context=log_context,
                rollout_id=rollout_id,
                read_without_destination=True,
                quiet=True,
                raise_read_errors=True,
                io_seconds=io_seconds,
                deadline=deadline,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            local_trace = await _load_local_trace_until(
                destination,
                deadline=deadline,
                log_context=log_context,
                rollout_id=rollout_id,
            )
            if local_trace is not None:
                return local_trace
            missing = _is_missing_file(exc)
            missing_reads += int(missing)
            # Zero-turn agents may never publish a trace. Allow one publication
            # retry without spending the entire recovery budget on absent data.
            retryable = _is_retryable_io_error(exc) or (missing and missing_reads < 2)
            if not retryable or _remaining(deadline) <= 0:
                logger.warning(
                    "%s trace recovery unavailable (%s: %s)", log_context, type(exc).__name__, exc
                )
                return None
            logger.warning(
                "%s trace read failed (%s: %s); retrying in the same sandbox",
                log_context,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(min(delay, _remaining(deadline)))
            delay = min(delay * 2, _IO_RETRY_MAX_DELAY_SECONDS)


async def _sync_trace_periodically(
    sandbox,
    destination: Path,
    *,
    log_context: str,
    rollout_id: str,
) -> None:
    try:
        async with trace_snapshot_writer(destination, rollout_id) as writer:
            while True:
                await asyncio.sleep(_TRACE_SYNC_INTERVAL_SECONDS)
                await _download_trace(
                    sandbox,
                    destination,
                    log_context=log_context,
                    rollout_id=rollout_id,
                    quiet=True,
                    priority="background",
                    snapshot_writer=writer,
                )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Optional diagnostics must not turn a known agent outcome into a
        # rollout failure if the process pool cannot start or shut down.
        _e2b_log(f"{log_context} periodic trace sync unavailable ({exc})", stderr=True)
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError from None


async def _run_entrypoint_with_trace_sync(
    sandbox,
    *,
    entrypoint_command: str,
    timeout_seconds: float,
    sandbox_env: dict[str, str],
    trace_destination: Path | None,
    log_context: str,
    rollout_id: str,
) -> None:
    trace_sync_task = (
        asyncio.create_task(
            _sync_trace_periodically(
                sandbox,
                trace_destination,
                log_context=log_context,
                rollout_id=rollout_id,
            )
        )
        if trace_destination is not None
        else None
    )
    try:
        await sandbox.commands.run(
            entrypoint_command,
            user="root",
            timeout=timeout_seconds,
            request_timeout=0,
            envs={**sandbox_env, "PYTHONPATH": _SOURCE_ROOT, "PYTHONSAFEPATH": "1"},
            cwd=_SOURCE_ROOT,
            on_stdout=_stream_e2b_stdout,
            on_stderr=_stream_e2b_stderr,
        )
    except BaseException as exc:
        _e2b_log(
            f"{log_context} entrypoint failed; full traceback follows",
            stderr=True,
        )
        traceback.print_exception(exc, file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if trace_sync_task is not None:
            trace_sync_task.cancel()
            try:
                await trace_sync_task
            except asyncio.CancelledError:
                # Awaiting the cancelled sync task can also deliver a caller's
                # cancellation that arrived during cleanup. Preserve that request.
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise


def _add_task_assets(
    archive: tarfile.TarFile,
    *,
    task_files_root: Path,
    task_id: str,
) -> None:
    """Package private assets and tee the conventional run-tests.sh evaluator."""

    if (
        not isinstance(task_id, str)
        or len(task_id) > _TASK_INSTANCE_MAX_LENGTH
        or _TASK_INSTANCE_PATTERN.fullmatch(task_id) is None
    ):
        raise ValueError("metadata.task_id must be a safe single directory name")
    resolved_root = task_files_root.resolve(strict=True)
    task_dir = resolved_root / task_id
    if task_dir.is_symlink() or task_dir.resolve(strict=True) != task_dir:
        raise ValueError(f"Evaluation task directory must not be a symlink: {task_dir}")
    if not task_dir.is_dir():
        raise ValueError(f"Evaluation task directory does not exist: {task_dir}")
    for source_path in sorted(task_dir.rglob("*")):
        if source_path.is_symlink():
            raise ValueError(f"Rollout task asset must not be a symlink: {source_path}")
        if not source_path.is_file() and not source_path.is_dir():
            raise ValueError(f"Unsupported rollout task asset: {source_path}")
        # Host asset permissions must not make evaluation code writable by the agent.
        member = archive.gettarinfo(
            str(source_path), arcname=str(source_path.relative_to(task_dir))
        )
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        member.mode = 0o700 if member.isdir() else 0o600 | (member.mode & 0o100)
        if member.isdir():
            archive.addfile(member)
        elif source_path.relative_to(task_dir) == Path("run-tests.sh"):
            wrapped = wrap_evaluator_script(
                source_path.read_bytes(),
                interpreter=_BOOTSTRAP_INTERPRETER,
                helper=f"{_SOURCE_ROOT}/northrl/remote_rollout/evaluator_logs.py",
                input_path=_INPUT_PATH,
                log_dir=_EVALUATOR_LOG_ROOT,
            )
            member.size = len(wrapped)
            archive.addfile(member, BytesIO(wrapped))
        else:
            with source_path.open("rb") as asset:
                archive.addfile(member, asset)


def _build_evaluation_archive(entrypoint: _EntrypointSpec, *, task_id: str | None) -> bytes:
    """Build on the driver; upload only after the agent completion record is verified."""

    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        if entrypoint.task_files_root is not None:
            if task_id is None:
                raise ValueError("rollout metadata.task_id is required when task_files_root is set")
            _add_task_assets(archive, task_files_root=entrypoint.task_files_root, task_id=task_id)
    return output.getvalue()


def _build_source_archive(entrypoint: _EntrypointSpec) -> bytes:
    """Upload public code only, excluding the entire declared evaluation asset root."""

    private_root = (
        entrypoint.task_files_root.resolve() if entrypoint.task_files_root is not None else None
    )

    def is_private(path: Path) -> bool:
        return private_root is not None and path.resolve().is_relative_to(private_root)

    if is_private(entrypoint.script_path):
        raise ValueError("Task hook script must not be inside the private evaluation asset root")
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            if path.is_file() and not path.is_symlink() and not is_private(path):
                archive.add(
                    path,
                    arcname=Path("northrl-src") / "northrl" / path.relative_to(_PACKAGE_ROOT),
                    recursive=False,
                )
        archive.add(_BOOTSTRAP_SCRIPT_PATH, arcname=_BOOTSTRAP_SCRIPT_ARCHIVE_PATH, recursive=False)
        archive.add(
            entrypoint.script_path,
            arcname=_UPLOADED_ENTRYPOINT_ARCHIVE_PATH,
            recursive=False,
        )
    return output.getvalue()


def _evaluation_task_id(entrypoint: _EntrypointSpec, task_id: Any) -> str | None:
    if entrypoint.task_files_root is None:
        return None
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(
            "rollout metadata.task_id must be a non-empty string when "
            "entrypoint.task_files_root is set"
        )
    return task_id.strip()


def prepare_e2b_artifacts(
    config: DictConfig, task_ids: Iterable[Any] | None = None
) -> PreparedE2BArtifacts:
    """Pack host assets before worker fan-out.

    By default, prepare each task directory under task_files_root. Standalone
    callers can select their task explicitly. Private evaluation archives stay
    separate from public source and upload only after verified agent completion.
    """

    started_at = time.monotonic()
    entrypoint = _resolve_entrypoint(config)
    if entrypoint.task_files_root is None:
        selected_ids: dict[str | None, None] = {None: None}
    else:
        if task_ids is None:
            task_ids = (
                path.name
                for path in sorted(entrypoint.task_files_root.iterdir())
                if path.is_dir()
                and not path.is_symlink()
                and _TASK_INSTANCE_PATTERN.fullmatch(path.name)
            )
        selected_ids = dict.fromkeys(
            _evaluation_task_id(entrypoint, task_id) for task_id in task_ids
        )
    task_count = len(selected_ids)
    logger.info(
        "Preparing E2B artifacts: task_files_root=%s tasks=%d",
        entrypoint.task_files_root,
        task_count,
    )
    evaluation_archives = {}
    evaluation_bytes = 0
    last_progress_at = started_at
    with profile_block("driver.prepare-artifacts", tasks=task_count):
        source_archive = _build_source_archive(entrypoint)
        for completed, task_id in enumerate(selected_ids, start=1):
            archive = _build_evaluation_archive(entrypoint, task_id=task_id)
            evaluation_archives[task_id] = archive
            evaluation_bytes += len(archive)
            now = time.monotonic()
            if (
                completed < task_count
                and now - last_progress_at >= _ARTIFACT_PROGRESS_INTERVAL_SECONDS
            ):
                logger.info(
                    "Preparing E2B artifacts: task_files_root=%s tasks=%d/%d "
                    "archive_bytes=%d elapsed=%.1fs",
                    entrypoint.task_files_root,
                    completed,
                    task_count,
                    len(source_archive) + evaluation_bytes,
                    now - started_at,
                )
                last_progress_at = now
    logger.info(
        "Prepared E2B artifacts: task_files_root=%s tasks=%d "
        "source_bytes=%d evaluation_bytes=%d elapsed=%.1fs",
        entrypoint.task_files_root,
        task_count,
        len(source_archive),
        evaluation_bytes,
        time.monotonic() - started_at,
    )
    return PreparedE2BArtifacts(
        entrypoint=entrypoint,
        source_archive=source_archive,
        evaluation_archives=MappingProxyType(evaluation_archives),
    )


async def _verify_agent_completion(
    sandbox,
    input_data: RolloutInput,
    attempt_id: str,
    *,
    deadline: float,
    io_seconds: float,
    log_context: str,
) -> None:
    """Verify the small commit record; evaluation validates the full stage locally."""
    deadline = min(deadline, time.monotonic() + io_seconds)
    try:
        record = await _retry_io(
            lambda: _download_file(sandbox, _STATE_PATH, priority="control"),
            deadline=deadline,
            io_seconds=io_seconds,
            log_context=log_context,
            operation="agent completion verification",
            queued_download=True,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if not _is_timeout_exception(exc):
            raise
        logger.warning(
            "[E2B] rollout=%r sandbox=%r agent completion record read timed out (%s: %s); "
            "continuing to evaluation with sandbox-local validation",
            input_data.rollout_id,
            getattr(sandbox, "sandbox_id", "unknown"),
            type(exc).__name__,
            exc,
        )
        return

    # A malformed or foreign commit is never a transport-timeout fallback.
    state = json.loads(record)
    if not isinstance(state, dict):
        raise ValueError("agent completion record must be an object")
    _validate_rollout_id(state.get("rollout_id"), input_data.rollout_id, "agent completion")
    if state.get("attempt_id") != attempt_id:
        raise ValueError("agent completion attempt_id does not match this execution attempt")
    if state.get("phase") != "agent_completed":
        raise ValueError("agent completion record has no committed completion phase")
    if state.get("status") is not None:
        raise ValueError("agent completion record contains a failure status")


def _is_timeout_exception(exc: BaseException) -> bool:
    """Return whether an exception or its causal chain represents a timeout."""

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if any("timeout" in cls.__name__.lower() for cls in type(current).__mro__):
            return True
        code = getattr(current, "code", None)
        if getattr(code, "value", code) == "deadline_exceeded" and any(
            cls.__module__.split(".", 1)[0] == "connectrpc" for cls in type(current).__mro__
        ):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _is_missing_file(exc: BaseException) -> bool:
    return isinstance(exc, FileNotFoundError) or any(
        cls.__module__ == "e2b.exceptions" and cls.__name__ == "FileNotFoundException"
        for cls in type(exc).__mro__
    )


def _is_retryable_io_error(exc: BaseException) -> bool:
    if isinstance(exc, (_EvaluationResultUnavailable, _EvaluationCommandTimeout)):
        # A confirmed terminal outcome is not a transport failure, even when
        # its exception context contains the SDK timeout that started recovery.
        return False
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500
    if _is_missing_file(exc) or any(
        cls.__module__ == "e2b.exceptions"
        and cls.__name__
        in ("AuthenticationException", "NotFoundException", "InvalidArgumentException")
        for cls in type(exc).__mro__
    ):
        return False
    return (
        _is_timeout_exception(exc)
        or isinstance(exc, ConnectionError)
        or _is_command_transport_error(exc)
    )


async def _retry_io(
    call: Callable[[], Awaitable[Any]],
    *,
    deadline: float,
    io_seconds: float,
    log_context: str,
    operation: str,
    retry_missing: bool = False,
    queued_download: bool = False,
) -> Any:
    """Retry transfers against one deadline without rerunning remote commands."""
    delay = _IO_RETRY_INITIAL_DELAY_SECONDS
    while True:
        try:
            if queued_download:
                token = _DOWNLOAD_BUDGET.set((io_seconds, deadline))
                try:
                    # The queue applies the deadline to waiting and processing;
                    # per-read timing starts only after admission.
                    return await call()
                finally:
                    _DOWNLOAD_BUDGET.reset(token)
            return await asyncio.wait_for(call(), min(io_seconds, _remaining(deadline)))
        except Exception as exc:
            missing = _is_missing_file(exc)
            transient = _is_retryable_io_error(exc)
            if not (transient or (retry_missing and missing)) or _remaining(deadline) <= 0:
                raise
            logger.warning(
                "%s %s failed (%s: %s); retrying in the same sandbox, remaining_seconds=%.3f",
                log_context,
                operation,
                type(exc).__name__,
                exc,
                _remaining(deadline),
            )
            await asyncio.sleep(min(delay, _remaining(deadline)))
            delay = min(delay * 2, _IO_RETRY_MAX_DELAY_SECONDS)


class _EvaluationResultUnavailable(RuntimeError):
    """The evaluator stopped and its final result is confirmed missing."""


async def _read_pending_evaluation_result(
    sandbox,
    input_data: RolloutInput,
    attempt_id: str,
) -> RemoteRolloutResult:
    """Stop missing-file polling only after this evaluator has actually exited."""
    try:
        return await _read_result(sandbox, input_data)
    except Exception as exc:
        if not _is_missing_file(exc):
            raise
        missing_result = exc

    io_seconds, deadline = _DOWNLOAD_BUDGET.get()
    terminal_deadline = time.monotonic() + 5.0
    if deadline is not None:
        terminal_deadline = min(terminal_deadline, deadline)
    try:
        terminal = await _download_file(
            sandbox,
            _EVALUATION_TERMINAL_PATH,
            decode=json.loads,
            blocking_decode=False,
            priority="control",
            io_seconds=min(io_seconds, 5.0),
            deadline=terminal_deadline,
        )
        if not isinstance(terminal, dict):
            raise ValueError("evaluation terminal record must be an object")
        _validate_rollout_id(
            terminal.get("rollout_id"), input_data.rollout_id, "evaluation terminal"
        )
        if terminal.get("attempt_id") != attempt_id:
            raise ValueError("evaluation terminal attempt_id does not match this attempt")
        if terminal.get("phase") != "evaluation_finished":
            raise ValueError("evaluation terminal record has no valid phase")
        exit_code = terminal.get("exit_code")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool) or exit_code < 0:
            raise ValueError("evaluation terminal record has no valid exit_code")
    except Exception as terminal_exc:
        # Missing, stale, malformed, or temporarily unreadable evidence cannot
        # establish that the original evaluator has stopped publishing results.
        if not _is_missing_file(terminal_exc):
            logger.debug("Evaluation terminal record unavailable: %s", terminal_exc)
        raise missing_result from terminal_exc

    # Result publication is atomic on the same sandbox filesystem, and the
    # supervisor publishes only after the writer exits. Re-read immediately in
    # case publication raced the first read, retaining the full transfer budget.
    try:
        return await _read_result(sandbox, input_data)
    except Exception as exc:
        if not _is_missing_file(exc):
            raise
        message = (
            f"Evaluator exited with status {exit_code} but its result is missing; "
            "stopping result publication polling"
        )
        logger.warning("rollout=%r %s", input_data.rollout_id, message)
        if exit_code == 124:
            raise _EvaluationCommandTimeout(message) from exc
        raise _EvaluationResultUnavailable(message) from exc


async def _wait_for_agent_completion(
    sandbox,
    input_data: RolloutInput,
    attempt_id: str,
    *,
    deadline: float,
    io_seconds: float,
    log_context: str,
) -> None:
    """Observe a disconnected agent; never launch a second copy of its command."""
    while True:
        state = await _retry_io(
            lambda: _download_file(
                sandbox, _STATE_PATH, decode=json.loads, blocking_decode=False, priority="control"
            ),
            deadline=deadline,
            io_seconds=io_seconds,
            log_context=log_context,
            operation="agent completion observation",
            retry_missing=True,
            queued_download=True,
        )
        if not isinstance(state, dict):
            raise ValueError("agent state must be an object")
        state_rollout_id = state.get("rollout_id")
        if not isinstance(state_rollout_id, str):
            raise ValueError("agent state must contain a string rollout_id")
        _validate_rollout_id(state_rollout_id, input_data.rollout_id, "agent state")
        if state.get("attempt_id") != attempt_id:
            raise ValueError("agent state attempt_id does not match this attempt")
        if state.get("status") in ("env_error", "task_failed"):
            raise RuntimeError("agent reported failure while its command was disconnected")
        if state.get("phase") == "agent_completed":
            return
        if state.get("phase") not in ("environment", "agent"):
            raise ValueError("agent state has no valid phase")
        if _remaining(deadline) <= 0:
            raise TimeoutError("agent observation deadline expired")
        await asyncio.sleep(min(_IO_RETRY_MAX_DELAY_SECONDS, _remaining(deadline)))


def _validate_rollout_id(actual: object, expected: str, artifact: str) -> None:
    if actual != expected:
        raise ValueError(
            f"E2B {artifact} ID does not match rollout input: "
            f"expected={expected!r}, actual={actual!r}"
        )


def _failure_result(
    input_data: RolloutInput,
    exc: Exception,
    *,
    status: Literal["env_error", "task_failed"],
    trace: CompactRolloutTrace | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    termination_reason: Literal["context_length_exceeded"] | None = None,
    length_details: dict[str, int] | None = None,
) -> RemoteRolloutResult:
    if trace is not None:
        _validate_rollout_id(trace.rollout_id, input_data.rollout_id, "partial trace")
    observed_metrics = dict(trace.metrics) if trace is not None else {}
    if status == "env_error":
        # The saved trace remains useful for diagnosis, but infrastructure
        # failures must not supply training segments.
        trace = None
    # Expand only a terminal task failure that can supply training data. Periodic
    # downloads and environment-error diagnostics keep the compact wire format.
    training_trace = decode_trace(trace, packed=True) if trace is not None else None
    return RemoteRolloutResult(
        rollout_id=input_data.rollout_id,
        status=status,
        error_type=error_type or type(exc).__name__,
        error_message=error_message or str(exc),
        termination_reason=termination_reason,
        length_details=length_details or {},
        segments=training_trace.segments if training_trace is not None else [],
        message_format=training_trace.message_format if training_trace is not None else None,
        message_system=training_trace.message_system if training_trace is not None else None,
        tools=training_trace.tools if training_trace is not None else None,
        messages=training_trace.messages if training_trace is not None else [],
        reward=0.0,
        metrics={
            **observed_metrics,
            **({"e2b_timed_out": 1.0} if _is_timeout_exception(exc) else {}),
        },
    )


async def _read_result(sandbox, input_data: RolloutInput) -> RemoteRolloutResult:
    processor = get_result_processor()

    def decode(payload):
        if processor is not None:
            with profile_block("driver.result-process", rollout_id=input_data.rollout_id):
                return processor.decode(payload, input_data.rollout_id)
        timings: dict[str, Any] | None = {} if profiling.result_profiling_enabled() else None
        with profiling.measure_cpu_wall(timings, "parse"):
            with profile_block("driver.result-parse", rollout_id=input_data.rollout_id):
                result = load_result_gzip(payload)
                _validate_rollout_id(result.rollout_id, input_data.rollout_id, "rollout result")
        with profiling.measure_cpu_wall(timings, "restore"):
            with profile_block("driver.result-expand", rollout_id=input_data.rollout_id):
                expanded = decode_result(result, packed=True)
        if timings is not None:
            timings.update(result_size_fields(payload, expanded))
            profiling.result_profile_event(
                "driver.result-decode",
                mode="thread",
                rollout_id=input_data.rollout_id,
                **timings,
            )
        return expanded

    # Final results can be much larger than control files. Extend both the
    # driver read limit and the SDK's default 60-second request timeout, while
    # retaining the enclosing stage/recovery deadline inherited by _download_file.
    return await _download_file(
        sandbox,
        _RESULT_PATH,
        decode=decode,
        file_format="bytes",
        io_seconds=ExecutionBudgets.result_download_seconds,
        request_timeout=ExecutionBudgets.result_download_seconds,
    )


def _is_command_transport_error(exc: BaseException) -> bool:
    """Distinguish failed API connections from remote process exits and timeouts."""

    if str(exc) == "Command ended without an end event":
        return True
    if getattr(exc, "exit_code", None) is not None or isinstance(exc, TimeoutError):
        # A process exit or our outer wait_for budget is a command outcome.
        # A timeout buried inside a transport exception is not that budget.
        return False
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ConnectionError):
            return True
        is_sdk_timeout = any(
            cls.__module__ == "e2b.exceptions" and cls.__name__ == "TimeoutException"
            for cls in type(current).__mro__
        )
        for cls in type(current).__mro__:
            module = cls.__module__.split(".", 1)[0]
            if module == "connectrpc":
                code = getattr(current, "code", None)
                # The SDK may wrap an UNAVAILABLE transport failure around a
                # TimeoutError. Only the explicit RPC deadline means the command
                # used up its execution budget.
                if getattr(code, "value", code) == "deadline_exceeded":
                    return False
                return True
            if module in {"httpx", "httpcore", "pyqwest"}:
                return True
            if (
                not is_sdk_timeout
                and cls.__module__ == "e2b.exceptions"
                and cls.__name__
                in {
                    "AuthenticationException",
                    "RateLimitException",
                    "SandboxException",
                }
            ):
                return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _with_failure_trace(
    result: RemoteRolloutResult,
    trace: CompactRolloutTrace | None,
) -> RemoteRolloutResult:
    if result.status == "env_error":
        return result.model_copy(
            update={
                "segments": [],
                "message_format": None,
                "message_system": None,
                "tools": None,
                "messages": [],
            }
        )
    if result.status == "task_failed" and not result.segments and trace is not None:
        training_trace = decode_trace(trace, packed=True)
        return result.model_copy(
            update={
                "segments": training_trace.segments,
                "message_format": training_trace.message_format,
                "message_system": training_trace.message_system,
                "tools": training_trace.tools,
                "messages": training_trace.messages,
                "metrics": {**trace.metrics, **result.metrics},
            }
        )
    return result


async def _with_failure_trace_until(result, trace, *, deadline: float | None):
    if result.status != "task_failed" or result.segments or trace is None:
        return _with_failure_trace(result, trace)
    if deadline is None:
        return await run_blocking(_with_failure_trace, result, trace)
    try:
        return await _run_blocking_until(deadline, _with_failure_trace, result, trace)
    except TimeoutError:
        return result


async def _command_failure_result(
    sandbox,
    input_data: RolloutInput,
    exc: Exception,
    *,
    trace: CompactRolloutTrace | None,
    log_context: str,
    allow_final_result: bool = True,
    default_status: Literal["env_error", "task_failed"] = "task_failed",
    attempt_id: str | None = None,
    deadline: float | None = None,
    io_seconds: float = 60.0,
    retry_missing_result: bool = False,
) -> RemoteRolloutResult:
    """Use trusted phase state; an unscored agent result can never finish the rollout."""

    async def read(call, operation, *, retry_missing=False):
        if deadline is None:
            return await call()
        return await _retry_io(
            call,
            deadline=deadline,
            io_seconds=io_seconds,
            log_context=log_context,
            operation=operation,
            retry_missing=retry_missing,
            queued_download=True,
        )

    try:
        result = await read(
            (
                lambda: (
                    _read_pending_evaluation_result(sandbox, input_data, attempt_id)
                    if retry_missing_result and attempt_id is not None
                    else _read_result(sandbox, input_data)
                )
            ),
            "failure result recovery",
            retry_missing=retry_missing_result,
        )
        if not allow_final_result and result.status == "success":
            raise ValueError("agent phase cannot publish a final successful result")
    except Exception as result_exc:  # pylint: disable=broad-exception-caught
        _e2b_log(
            f"{log_context} final result unavailable ({type(result_exc).__name__}: {result_exc})",
            stderr=True,
        )
    else:
        result = await _with_failure_trace_until(result, trace, deadline=deadline)
        if _is_timeout_exception(exc):
            result = result.model_copy(update={"metrics": {**result.metrics, "e2b_timed_out": 1.0}})
        return result

    transport_error = _is_command_transport_error(exc)
    trusted_reported_status = False
    status: Literal["env_error", "task_failed"] = "env_error" if transport_error else default_status
    error_type = None
    error_message = None
    termination_reason: Literal["context_length_exceeded"] | None = None
    length_details: dict[str, int] = {}
    try:
        state = await read(
            lambda: _download_file(
                sandbox, _STATE_PATH, decode=json.loads, blocking_decode=False, priority="control"
            ),
            "failure state recovery",
        )
        if not isinstance(state, dict):
            raise ValueError("rollout phase marker must be a JSON object")
        state_rollout_id = state.get("rollout_id")
        if not isinstance(state_rollout_id, str):
            raise ValueError("rollout phase marker must contain a string rollout_id")
        _validate_rollout_id(state_rollout_id, input_data.rollout_id, "phase marker")
        if attempt_id is not None and state.get("attempt_id") != attempt_id:
            raise ValueError("phase marker attempt_id does not match this attempt")
        phase = state.get("phase")
        if phase not in ("environment", "agent", "evaluator", "agent_completed", "finished"):
            raise ValueError("rollout phase marker has no valid phase")
        reported_status = state.get("status")
        if reported_status is not None:
            if reported_status not in ("env_error", "task_failed"):
                raise ValueError("rollout phase marker has no valid failure status")
            status = reported_status
            trusted_reported_status = True
        elif phase != "agent":
            status = "env_error"
        if state.get("termination_reason") == "context_length_exceeded":
            termination_reason = "context_length_exceeded"
            details = state.get("length_details")
            if isinstance(details, dict):
                length_details = {
                    key: value
                    for key, value in details.items()
                    if isinstance(key, str)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                }
        if isinstance(state.get("error_type"), str):
            error_type = state["error_type"]
        if isinstance(state.get("error_message"), str):
            error_message = state["error_message"]
    except Exception as state_exc:  # pylint: disable=broad-exception-caught
        # Without a valid marker for this attempt, the command may have failed
        # before the agent started. Do not manufacture a scored task failure.
        status = "env_error"
        _e2b_log(
            f"{log_context} phase marker unavailable ({type(state_exc).__name__}: {state_exc}); "
            f"classifying command failure as {status}",
            stderr=True,
        )
    if transport_error and not trusted_reported_status:
        status = "env_error"
        error_type = None
        error_message = None
    result = _failure_result(
        input_data,
        exc,
        status=status,
        trace=trace if status == "env_error" else None,
        error_type=error_type,
        error_message=error_message,
        termination_reason=termination_reason,
        length_details=length_details,
    )
    if trace is not None:
        result = result.model_copy(update={"metrics": {**trace.metrics, **result.metrics}})
    return await _with_failure_trace_until(result, trace, deadline=deadline)


def _env_error_retries(config: DictConfig) -> int:
    execution = OmegaConf.select(config, "rollout.execution", default={})
    if "timeout_retries" in execution:
        raise ValueError(
            "rollout.execution.timeout_retries is no longer supported; "
            "use env_error_retries for environment failures"
        )
    retries = execution.get("env_error_retries", _DEFAULT_ENV_ERROR_RETRIES)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("rollout.execution.env_error_retries must be a non-negative integer")
    return retries


@dataclass
class _SandboxCreationRetryBudget:
    remaining_seconds: float


class _SandboxCreationRetryExhausted(Exception):
    def __init__(self, original_error: Exception) -> None:
        super().__init__(f"Sandbox creation retry budget exhausted: {original_error}")
        self.original_error = original_error


async def _create_sandbox_with_backoff(
    create: Callable[..., Any],
    *,
    template: str,
    sandbox_timeout: int,
    rollout_id: str,
    retry_budget: _SandboxCreationRetryBudget,
    volume_mounts: Mapping[str, str] | None = None,
) -> Any:
    """Retry creation HTTP 500s within a budget shared by all rollout attempts."""

    mounts = validate_volume_mounts(volume_mounts)
    started = time.monotonic()
    deadline = started + retry_budget.remaining_seconds
    delay = _SANDBOX_CREATE_RETRY_INITIAL_DELAY_SECONDS
    last_error: Exception | None = None
    attempt = 0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _SandboxCreationRetryExhausted(
                    last_error or TimeoutError("Sandbox creation retry budget exhausted")
                )
            attempt += 1
            try:
                options: dict[str, Any] = {"template": template, "timeout": sandbox_timeout}
                if mounts:
                    options["volume_mounts"] = dict(mounts)
                return await asyncio.wait_for(create(**options), timeout=remaining)
            except Exception as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _SandboxCreationRetryExhausted(last_error or exc) from exc
                if not (
                    any(cls.__name__ == "SandboxException" for cls in type(exc).__mro__)
                    and str(exc).startswith("500:")
                ):
                    raise
                last_error = exc
                wait = min(random.uniform(delay / 2, delay), remaining)
                _e2b_log(
                    f"rollout={rollout_id!r} sandbox creation retry attempt={attempt + 1} "
                    f"wait_seconds={wait:.2f} remaining_seconds={remaining:.2f} "
                    f"error_message={str(exc)!r}",
                    stderr=True,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, _SANDBOX_CREATE_RETRY_MAX_DELAY_SECONDS)
    finally:
        retry_budget.remaining_seconds = max(
            0.0, retry_budget.remaining_seconds - (time.monotonic() - started)
        )


async def run_rollout_in_e2b(
    input_data: RolloutInput,
    config: DictConfig,
    *,
    gateway: dict[str, str],
    template: str,
    timeout_seconds: int,
    trace_output_path: str | Path | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    prepared_artifacts: PreparedE2BArtifacts | None = None,
) -> RemoteRolloutResult:
    """Own optional result-process isolation across all attempts and recovery."""
    mode = config.rollout.execution.get("result_processing", "thread")
    async with result_processing(mode):
        return await _run_rollout_with_retries(
            input_data,
            config,
            gateway=gateway,
            template=template,
            timeout_seconds=timeout_seconds,
            trace_output_path=trace_output_path,
            event_callback=event_callback,
            prepared_artifacts=prepared_artifacts,
        )


async def _run_rollout_with_retries(
    input_data: RolloutInput,
    config: DictConfig,
    *,
    gateway: dict[str, str],
    template: str,
    timeout_seconds: int,
    trace_output_path: str | Path | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    prepared_artifacts: PreparedE2BArtifacts | None = None,
) -> RemoteRolloutResult:
    """Retry setup failures; preserve an agent's work once its command is dispatched."""

    rollout_started_at = time.time()
    rollout_started_monotonic = time.monotonic()
    try:
        if trace_output_path is not None:
            Path(trace_output_path).unlink(missing_ok=True)
        max_attempts = _env_error_retries(config) + 1
        budgets = ExecutionBudgets.from_config(config, agent_timeout_seconds=timeout_seconds)
        # Reject invalid host configuration before artifact preparation or retries.
        validate_volume_mounts(OmegaConf.select(config, "rollout.execution.volume_mounts"))
        if "network_policy" in config.rollout.execution:
            _resolve_task_network_policy(config, input_data)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _emit_event(
            event_callback,
            kind="rollout_finished",
            rollout_id=input_data.rollout_id,
            attempt=1,
            status="env_error",
            attempts=0,
        )
        result = _failure_result(input_data, exc, status="env_error")
        return result.model_copy(
            update={
                "metrics": {
                    **result.metrics,
                    "sandbox_retries": 0.0,
                    "rollout_started_at": rollout_started_at,
                    "rollout_finished_at": time.time(),
                    "rollout_duration_seconds": max(
                        0.0, time.monotonic() - rollout_started_monotonic
                    ),
                }
            }
        )

    creation_retry_budget = _SandboxCreationRetryBudget(budgets.preparation_seconds)
    preparation_deadline: float | None = None
    for attempt in range(1, max_attempts + 1):
        started_at = time.monotonic()
        final_snapshot: dict[str, float] = {}
        creation_retries_exhausted = False
        agent_dispatched = False

        def attempt_event(event: dict[str, Any], attempt_number: int = attempt) -> None:
            nonlocal final_snapshot, agent_dispatched
            if event.get("kind") == "stage" and event.get("phase") == "agent":
                agent_dispatched = True
            if event.get("kind") == "snapshot" and isinstance(event.get("metrics"), dict):
                candidate = event["metrics"]
                if candidate.get("runtime_revision", -1) >= final_snapshot.get(
                    "runtime_revision", -1
                ):
                    final_snapshot = dict(candidate)
            _emit_event(
                event_callback,
                **{**event, "rollout_id": input_data.rollout_id, "attempt": attempt_number},
            )

        _emit_event(attempt_event, kind="attempt_started")
        try:
            # Only files synced during this attempt count, even though retries share an ID.
            if trace_output_path is not None:
                Path(trace_output_path).unlink(missing_ok=True)
            if prepared_artifacts is None:
                # Standalone callers also keep compression off the rollout loop.
                # Retain the result for subsequent environment attempts.
                task_id = input_data.metadata.get("task_id", input_data.task.get("task_id"))
                prepared_artifacts = await asyncio.to_thread(
                    prepare_e2b_artifacts, config, [task_id]
                )
            if preparation_deadline is None:
                preparation_deadline = time.monotonic() + budgets.preparation_seconds
            if _remaining(preparation_deadline) <= 0:
                raise TimeoutError("preparation budget exhausted before sandbox retry")
            creation_retry_budget.remaining_seconds = min(
                creation_retry_budget.remaining_seconds, _remaining(preparation_deadline)
            )
            event_kwargs: dict[str, Any] = {
                "event_callback": attempt_event,
                "prepared_artifacts": prepared_artifacts,
                "creation_retry_budget": creation_retry_budget,
                "preparation_deadline": preparation_deadline,
            }
            result = await _run_rollout_in_e2b_once(
                input_data,
                config,
                gateway=gateway,
                template=template,
                timeout_seconds=timeout_seconds,
                trace_output_path=trace_output_path,
                **event_kwargs,
            )
        except asyncio.CancelledError:
            _emit_event(
                attempt_event,
                kind="attempt_finished",
                status="cancelled",
                duration_seconds=time.monotonic() - started_at,
                timed_out=False,
            )
            _emit_event(
                attempt_event, kind="rollout_finished", status="cancelled", attempts=attempt
            )
            raise
        except _SandboxCreationRetryExhausted as exc:
            creation_retries_exhausted = True
            result = _failure_result(input_data, exc.original_error, status="env_error")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # The command path classifies its own errors; earlier failures are environmental.
            result = _failure_result(input_data, exc, status="env_error")
        if final_snapshot.get("runtime_revision", -1) > result.metrics.get("runtime_revision", -1):
            result.metrics.update(final_snapshot)
        # The final result can contain tool receipts newer than the last heartbeat.
        # Merge them while this attempt is still active, including failed retries.
        tool_metrics = {
            key: value for key, value in result.metrics.items() if key.startswith("tool_")
        }
        if tool_metrics:
            _emit_event(attempt_event, kind="snapshot", metrics=tool_metrics)
        _emit_event(
            attempt_event,
            kind="attempt_finished",
            status=result.status,
            duration_seconds=time.monotonic() - started_at,
            timed_out=result.metrics.get("e2b_timed_out") == 1.0,
        )
        preparation_exhausted = (
            preparation_deadline is not None and _remaining(preparation_deadline) <= 0
        )
        if (
            creation_retries_exhausted
            or preparation_exhausted
            or agent_dispatched
            or result.status != "env_error"
            or attempt == max_attempts
        ):
            if result.status == "env_error":
                _e2b_log(
                    f"rollout={input_data.rollout_id!r} environment failure is terminal "
                    f"agent_dispatched={agent_dispatched} "
                    f"attempt={attempt}/{max_attempts} error_type={result.error_type} "
                    f"error_message={result.error_message!r}",
                    stderr=True,
                )
            _emit_event(
                attempt_event,
                kind="rollout_finished",
                status=result.status,
                attempts=attempt,
                trajectory_segments=len(result.segments),
                termination_reason=result.termination_reason,
                length_details=result.length_details,
            )
            return result.model_copy(
                update={
                    "metrics": {
                        **result.metrics,
                        "sandbox_retries": float(attempt - 1),
                        "rollout_started_at": rollout_started_at,
                        "rollout_finished_at": time.time(),
                        "rollout_duration_seconds": max(
                            0.0, time.monotonic() - rollout_started_monotonic
                        ),
                    }
                }
            )
        _e2b_log(
            f"rollout={input_data.rollout_id!r} environment failed "
            f"attempt={attempt}/{max_attempts} error_type={result.error_type} "
            f"error_message={result.error_message!r}; "
            f"retrying attempt={attempt + 1}/{max_attempts} with a fresh sandbox",
            stderr=True,
        )

    raise AssertionError("unreachable")


async def _run_rollout_in_e2b_once(
    input_data: RolloutInput,
    config: DictConfig,
    *,
    gateway: dict[str, str],
    template: str,
    timeout_seconds: int,
    trace_output_path: str | Path | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    prepared_artifacts: PreparedE2BArtifacts | None = None,
    creation_retry_budget: _SandboxCreationRetryBudget | None = None,
    preparation_deadline: float | None = None,
) -> RemoteRolloutResult:
    """Upload the current NorthRL source and run one rollout in E2B.

    The template provides system dependencies and the task checkout, without
    private evaluation assets. The local ``northrl`` Python package is uploaded
    for every sandbox so source-only changes do not require rebuilding the
    template. ``rollout.execution.entrypoint.target`` selects a host-side Python
    script uploaded with the package. The platform runtime loads the script's task
    hooks in both phases.
    """

    gateway_fields = ("base_url", "model", "sampling_session_id", "weight_version")
    for field in gateway_fields:
        if not isinstance(gateway.get(field), str) or not gateway[field].strip():
            raise ValueError(f"gateway.{field} must be a non-empty string")
    gateway = {field: gateway[field] for field in gateway_fields}
    if not template.strip():
        raise ValueError("template must not be empty")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise ValueError("timeout_seconds must be a positive integer")
    budgets = ExecutionBudgets.from_config(config, agent_timeout_seconds=timeout_seconds)
    volume_mounts = validate_volume_mounts(
        OmegaConf.select(config, "rollout.execution.volume_mounts")
    )
    task_id = input_data.metadata.get("task_id", input_data.task.get("task_id"))
    if prepared_artifacts is None:
        prepared_artifacts = await asyncio.to_thread(prepare_e2b_artifacts, config, [task_id])
    entrypoint = prepared_artifacts.entrypoint
    task_id = _evaluation_task_id(entrypoint, task_id)
    source_archive = prepared_artifacts.source_archive
    try:
        evaluation_archive = prepared_artifacts.evaluation_archives[task_id]
    except KeyError as exc:
        raise ValueError(f"Evaluation archive was not prepared for task_id={task_id!r}") from exc
    agent_command = _build_entrypoint_command(entrypoint, phase="agent")
    evaluation_command = _build_entrypoint_command(entrypoint, phase="evaluate")
    attempt_id = uuid.uuid4().hex
    evaluator_environment = _resolve_evaluator_environment()
    agent_environment = _resolve_agent_environment()
    try:
        from e2b import AsyncSandbox  # pylint: disable=import-error,import-outside-toplevel
    except ImportError as exc:
        raise ImportError("The e2b package is required to execute remote rollouts") from exc

    api_key = os.getenv("NORTHGATE_API_KEY")
    if not api_key:
        raise ValueError("NORTHGATE_API_KEY is required for E2B rollout inference")

    config_data: dict[str, Any] = {
        "models": {"actor": {"name": config.models.actor.name}},
        "rollout": {
            "num_workers": config.rollout.num_workers,
            "max_concurrent_requests": config.rollout.get("max_concurrent_requests"),
            "keep_sampling_mask": config.rollout.get("keep_sampling_mask", False),
            "sampling": OmegaConf.to_container(
                config.rollout.sampling, resolve=True, throw_on_missing=True
            ),
            "execution": {
                "token_prefix_reuse": config.rollout.execution.get("token_prefix_reuse", False),
                "agent_timeout_seconds": budgets.agent_seconds,
                "preparation_timeout_seconds": budgets.preparation_seconds,
                "agent_finalize_timeout_seconds": budgets.finalize_seconds,
                "evaluation_timeout_seconds": budgets.evaluation_seconds,
            },
        },
        "algorithm": OmegaConf.to_container(config.algorithm, resolve=True, throw_on_missing=True),
        "data": {
            "max_prompt_length": config.data.max_prompt_length,
            "max_sequence_length": config.data.max_sequence_length,
        },
    }
    if "network_policy" in config.rollout.execution:
        config_data["rollout"]["execution"]["network_policy"] = _resolve_task_network_policy(
            config, input_data
        )
    validation_sampling = OmegaConf.select(config, "validate.sampling")
    if validation_sampling is not None:
        config_data["validate"] = {
            "sampling": OmegaConf.to_container(
                validation_sampling, resolve=True, throw_on_missing=True
            )
        }

    sandbox_env = {**agent_environment, "NORTHGATE_API_KEY": str(api_key)}
    if log_level := os.getenv("LOG_LEVEL"):
        sandbox_env["LOG_LEVEL"] = log_level

    _e2b_log(
        f"rollout={input_data.rollout_id!r} creating sandbox "
        f"template={template!r} timeout_seconds={timeout_seconds} "
        f"entrypoint={str(entrypoint.script_path)!r}"
    )
    if preparation_deadline is None:
        preparation_deadline = time.monotonic() + budgets.preparation_seconds
    try:
        with profile_block(
            "driver.sandbox-create",
            rollout_id=input_data.rollout_id,
            template=template,
        ):
            # Phase and result recovery also need a live sandbox when callers do
            # not request a local trace file.
            sandbox_timeout = budgets.sandbox_timeout_seconds
            sandbox = await asyncio.wait_for(
                _create_sandbox_with_backoff(
                    AsyncSandbox.create,
                    template=template,
                    sandbox_timeout=sandbox_timeout,
                    rollout_id=input_data.rollout_id,
                    volume_mounts=volume_mounts,
                    retry_budget=(
                        creation_retry_budget
                        if creation_retry_budget is not None
                        else _SandboxCreationRetryBudget(budgets.preparation_seconds)
                    ),
                ),
                _remaining(preparation_deadline),
            )
    except BaseException as exc:
        _e2b_log(
            f"rollout={input_data.rollout_id!r} sandbox creation failed "
            f"({type(exc).__name__}) error_message={str(exc)!r}",
            stderr=True,
        )
        traceback.print_exception(exc, file=sys.stderr)
        sys.stderr.flush()
        raise

    deadline = preparation_deadline
    recovery_deadline: float | None = None

    def remaining() -> float:
        # wait_for(0) cancels an expired operation before it starts; do not
        # raise here after the caller has already constructed its coroutine.
        return _remaining(deadline)

    def recovery_remaining_deadline() -> float:
        nonlocal recovery_deadline
        if recovery_deadline is None:
            recovery_deadline = time.monotonic() + budgets.recovery_seconds
        return recovery_deadline

    async def transfer(call, operation, *, queued_download=False):
        return await _retry_io(
            call,
            deadline=deadline,
            io_seconds=budgets.io_seconds,
            log_context=log_context,
            operation=operation,
            queued_download=queued_download,
        )

    sandbox_id = str(getattr(sandbox, "sandbox_id", "unknown"))
    log_context = f"rollout={input_data.rollout_id!r} sandbox={sandbox_id!r}"
    trace_destination = Path(trace_output_path) if trace_output_path is not None else None
    _e2b_log(f"{log_context} created")
    _emit_event(
        event_callback,
        kind="sandbox_created",
        sandbox_id=sandbox_id,
        timeout_seconds=sandbox_timeout,
    )
    live_metrics_state = _LiveMetricsState()
    live_metrics_task = (
        asyncio.create_task(
            _poll_live_metrics(
                sandbox,
                rollout_id=input_data.rollout_id,
                event_callback=event_callback,
                state=live_metrics_state,
            )
        )
        if event_callback is not None
        else None
    )
    stage = "preparation"
    evaluation_result_pending = False
    _emit_event(event_callback, kind="stage", phase=stage)
    try:
        _e2b_log(
            f"{log_context} uploading source path={_SOURCE_ARCHIVE_PATH!r} "
            f"bytes={len(source_archive)}"
        )
        with profile_block(
            "driver.upload-source",
            rollout_id=input_data.rollout_id,
            sandbox_id=sandbox_id,
        ):
            await transfer(
                lambda: sandbox.files.write(_SOURCE_ARCHIVE_PATH, source_archive, user="root"),
                "source upload",
            )

        with profile_block(
            "driver.source-extract", rollout_id=input_data.rollout_id, sandbox_id=sandbox_id
        ):
            await asyncio.wait_for(
                sandbox.commands.run(
                    _EXTRACT_COMMAND,
                    user="root",
                    timeout=remaining(),
                    on_stdout=_stream_e2b_stdout,
                    on_stderr=_stream_e2b_stderr,
                ),
                remaining(),
            )
        # DNS lookup and descriptor inspection must not block other rollouts.
        await asyncio.wait_for(asyncio.to_thread(_set_e2b_tcp_user_timeout, sandbox), remaining())
        stage = "bootstrap"
        _emit_event(event_callback, kind="stage", phase=stage)
        with profile_block(
            "driver.sandbox-bootstrap", rollout_id=input_data.rollout_id, sandbox_id=sandbox_id
        ):
            await asyncio.wait_for(
                sandbox.commands.run(
                    _BOOTSTRAP_COMMAND,
                    user="root",
                    timeout=remaining(),
                    request_timeout=0,
                    on_stdout=_stream_e2b_stdout,
                    on_stderr=_stream_e2b_stderr,
                ),
                remaining(),
            )
        # Share preparation time with the runtime without deducting it from the agent.
        config_data["rollout"]["execution"]["preparation_timeout_seconds"] = max(
            1, math.ceil(remaining())
        )
        input_json = json.dumps(
            {
                "input_data": input_data.model_dump(mode="json"),
                "attempt_id": attempt_id,
                "config": config_data,
                "gateway": gateway,
            },
            ensure_ascii=False,
        )
        with profile_block(
            "driver.upload-input", rollout_id=input_data.rollout_id, sandbox_id=sandbox_id
        ):
            await transfer(
                lambda: sandbox.files.write(_INPUT_PATH, input_json, user="root"), "input upload"
            )
        if remaining() <= 0:
            raise TimeoutError("preparation deadline expired before agent dispatch")
        agent_command = f"{agent_command} --preparation-timeout-seconds {remaining()}"
        deadline = preparation_deadline + budgets.agent_seconds + budgets.finalize_seconds
        stage = "agent"
        _emit_event(event_callback, kind="stage", phase=stage)
        _e2b_log(f"{log_context} stage=agent starting; live output follows")
        with profile_block(
            "driver.agent-command", rollout_id=input_data.rollout_id, sandbox_id=sandbox_id
        ):
            try:
                await asyncio.wait_for(
                    _run_entrypoint_with_trace_sync(
                        sandbox,
                        entrypoint_command=agent_command,
                        timeout_seconds=remaining(),
                        sandbox_env=sandbox_env,
                        trace_destination=trace_destination,
                        log_context=log_context,
                        rollout_id=input_data.rollout_id,
                    ),
                    remaining(),
                )
            except Exception as command_exc:
                if not _is_command_transport_error(command_exc):
                    raise
                logger.warning(
                    "%s agent command disconnected; observing the original execution", log_context
                )
                try:
                    await _wait_for_agent_completion(
                        sandbox,
                        input_data,
                        attempt_id,
                        deadline=deadline,
                        io_seconds=budgets.io_seconds,
                        log_context=log_context,
                    )
                except Exception as observation_exc:
                    raise command_exc from observation_exc

        # A periodically updated trace is not a completion signal. The trusted runtime
        # commits this record only after the agent and gateway have both shut down.
        stage = "agent-verification"
        _emit_event(event_callback, kind="stage", phase=stage)
        deadline = min(deadline, time.monotonic() + budgets.finalize_seconds)
        await _verify_agent_completion(
            sandbox,
            input_data,
            attempt_id,
            deadline=deadline,
            io_seconds=budgets.io_seconds,
            log_context=log_context,
        )
        deadline = time.monotonic() + budgets.evaluation_preparation_seconds
        evaluation_profile = {"rollout_id": input_data.rollout_id, "sandbox_id": sandbox_id}
        with profile_block("driver.evaluation-preparation", **evaluation_profile):
            stage = "evaluation-upload"
            _emit_event(event_callback, kind="stage", phase=stage)
            with profile_block("driver.evaluation-upload", **evaluation_profile):
                await transfer(
                    lambda: sandbox.files.write(
                        _EVALUATION_ARCHIVE_PATH, evaluation_archive, user="root"
                    ),
                    "evaluation upload",
                )
            stage = "evaluation-extract"
            _emit_event(event_callback, kind="stage", phase=stage)
            with profile_block("driver.evaluation-extract", **evaluation_profile):
                while True:
                    try:
                        await asyncio.wait_for(
                            sandbox.commands.run(
                                _EXTRACT_EVALUATION_COMMAND,
                                user="root",
                                timeout=remaining(),
                                request_timeout=0,
                                on_stdout=_stream_e2b_stdout,
                                on_stderr=_stream_e2b_stderr,
                            ),
                            remaining(),
                        )
                        break
                    except Exception as extract_exc:
                        if getattr(extract_exc, "exit_code", None) is not None and remaining() > 0:
                            # A confirmed exit can be retried without overlapping extractors.
                            logger.warning(
                                "%s evaluation extraction failed; retrying in place", log_context
                            )
                            await asyncio.sleep(min(_IO_RETRY_INITIAL_DELAY_SECONDS, remaining()))
                            continue
                        if not _is_command_transport_error(extract_exc):
                            raise
                        await _retry_io(
                            lambda: _download_file(
                                sandbox, _EVALUATION_READY_PATH, priority="control"
                            ),
                            deadline=deadline,
                            io_seconds=budgets.io_seconds,
                            log_context=log_context,
                            operation="evaluation extraction observation",
                            retry_missing=True,
                            queued_download=True,
                        )
                        break
        # Evaluation gets its full budget only after the assets are ready.
        deadline = time.monotonic() + budgets.evaluation_seconds
        stage = "evaluator"
        _emit_event(event_callback, kind="stage", phase=stage)
        # Runtime cancellation reports a failure; the supervisor additionally
        # stops evaluator subprocesses before recovery can outlive this budget.
        evaluation_seconds = remaining()
        evaluation_command = _supervise_evaluation_command(
            f"{evaluation_command} --timeout-seconds {evaluation_seconds}",
            evaluation_seconds,
            rollout_id=input_data.rollout_id,
            attempt_id=attempt_id,
        )
        # This grace only permits process-group cleanup and delivery of the
        # supervisor's exit. The evaluator retains its original stage budget.
        evaluation_wait_seconds = remaining() + _EVALUATION_COMMAND_CLEANUP_SECONDS
        result = None
        with profile_block(
            "driver.evaluator-command", rollout_id=input_data.rollout_id, sandbox_id=sandbox_id
        ):
            try:
                await asyncio.wait_for(
                    sandbox.commands.run(
                        evaluation_command,
                        user="root",
                        timeout=evaluation_wait_seconds,
                        request_timeout=0,
                        envs={
                            **evaluator_environment,
                            "PYTHONPATH": _SOURCE_ROOT,
                            "PYTHONSAFEPATH": "1",
                            **({"LOG_LEVEL": log_level} if log_level else {}),
                        },
                        cwd=_SOURCE_ROOT,
                        on_stdout=_stream_e2b_stdout,
                        on_stderr=_stream_e2b_stderr,
                    ),
                    evaluation_wait_seconds,
                )
            except Exception as evaluation_exc:
                if getattr(evaluation_exc, "exit_code", None) == 124:
                    raise _EvaluationCommandTimeout(
                        "Evaluation supervisor stopped the process group after its deadline"
                    ) from evaluation_exc
                evaluation_result_pending = getattr(evaluation_exc, "exit_code", None) is None and (
                    _is_command_transport_error(evaluation_exc)
                    or _is_timeout_exception(evaluation_exc)
                )
                if not _is_command_transport_error(evaluation_exc):
                    raise
                logger.warning(
                    "%s evaluator disconnected; waiting for its original result", log_context
                )
                try:
                    result = await _retry_io(
                        lambda: _read_pending_evaluation_result(sandbox, input_data, attempt_id),
                        deadline=deadline,
                        io_seconds=budgets.io_seconds,
                        log_context=log_context,
                        operation="evaluation result observation",
                        retry_missing=True,
                        queued_download=True,
                    )
                except (_EvaluationResultUnavailable, _EvaluationCommandTimeout):
                    evaluation_result_pending = False
                    raise
                evaluation_result_pending = False
        stage = "result-download"
        _emit_event(event_callback, kind="stage", phase=stage)
        deadline = recovery_remaining_deadline()
        _e2b_log(f"{log_context} stage=rollout completed")
        _e2b_log(f"{log_context} reading result path={_RESULT_PATH!r}")
        with profile_block(
            "driver.result-download",
            rollout_id=input_data.rollout_id,
            sandbox_id=sandbox_id,
        ):
            if result is None:
                result = await transfer(
                    lambda: _read_result(sandbox, input_data),
                    "result download",
                    queued_download=True,
                )
        _e2b_log(
            f"{log_context} result validated status={result.status!r} reward={result.reward!r} "
            f"segments={len(result.segments)}"
        )
        if result.status == "task_failed" and not result.segments:
            _emit_event(event_callback, kind="stage", phase="trace-recovery")
            trace = await _recover_trace(
                sandbox,
                trace_destination,
                log_context=log_context,
                rollout_id=input_data.rollout_id,
                deadline=deadline,
                io_seconds=budgets.io_seconds,
            )
            return await _with_failure_trace_until(result, trace, deadline=deadline)
        diagnostic_deadline = min(deadline, time.monotonic() + budgets.io_seconds)
        await _download_trace(
            sandbox,
            trace_destination,
            log_context=log_context,
            rollout_id=input_data.rollout_id,
            priority="background",
            io_seconds=budgets.io_seconds,
            deadline=diagnostic_deadline,
        )
        return _with_failure_trace(result, None)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Resolve the trustworthy outcome BEFORE attempting potentially slow
        # trace downloads. Recovery must not turn a known result into a retry.
        deadline = recovery_remaining_deadline()
        _emit_event(event_callback, kind="stage", phase="result-recovery")
        if stage in ("agent", "evaluator"):
            try:
                # Each recovery transfer observes the shared remote deadline.
                # Do not cancel local processing of an already downloaded result.
                failure_result = await _command_failure_result(
                    sandbox,
                    input_data,
                    exc,
                    trace=None,
                    log_context=log_context,
                    allow_final_result=stage == "evaluator",
                    default_status=(
                        "env_error"
                        if stage == "evaluator" or _is_timeout_exception(exc)
                        else "task_failed"
                    ),
                    attempt_id=attempt_id,
                    deadline=deadline,
                    io_seconds=budgets.io_seconds,
                    retry_missing_result=stage == "evaluator" and evaluation_result_pending,
                )
            except Exception as recovery_exc:  # pylint: disable=broad-exception-caught
                logger.warning("%s result recovery unavailable: %s", log_context, recovery_exc)
                failure_result = _failure_result(input_data, exc, status="env_error")
        else:
            failure_result = _failure_result(input_data, exc, status="env_error")
        if failure_result.status == "task_failed" and not failure_result.segments:
            _emit_event(event_callback, kind="stage", phase="trace-recovery")
            try:
                partial_trace = await asyncio.wait_for(
                    _recover_trace(
                        sandbox,
                        trace_destination,
                        log_context=log_context,
                        rollout_id=input_data.rollout_id,
                        deadline=deadline,
                        io_seconds=budgets.io_seconds,
                    ),
                    remaining(),
                )
            except Exception as trace_exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "%s trace recovery failed; preserving outcome: %s", log_context, trace_exc
                )
                partial_trace = await _load_local_trace_until(
                    trace_destination,
                    deadline=deadline,
                    log_context=log_context,
                    rollout_id=input_data.rollout_id,
                )
            failure_result = await _with_failure_trace_until(
                failure_result, partial_trace, deadline=deadline
            )
        elif trace_destination is not None and remaining() > 0:
            # Diagnostic traces are optional once the outcome and training data
            # are known (or the sample has no evaluated outcome).
            diagnostic_deadline = min(deadline, time.monotonic() + budgets.io_seconds)
            try:
                await asyncio.wait_for(
                    _download_trace(
                        sandbox,
                        trace_destination,
                        log_context=log_context,
                        rollout_id=input_data.rollout_id,
                        priority="background",
                        io_seconds=budgets.io_seconds,
                        deadline=diagnostic_deadline,
                    ),
                    _remaining(diagnostic_deadline),
                )
            except Exception as trace_exc:  # pylint: disable=broad-exception-caught
                logger.warning("%s optional trace unavailable: %s", log_context, trace_exc)
        _e2b_log(
            f"{log_context} failed stage={stage} status={failure_result.status} "
            f"error_type={failure_result.error_type}; killing sandbox",
            stderr=True,
        )
        return failure_result
    finally:
        try:
            if live_metrics_task is not None:
                # Preserve a new caller cancellation while suppressing our own child cancellation.
                current_task = asyncio.current_task()
                cancellation_count = current_task.cancelling() if current_task is not None else 0
                live_metrics_task.cancel()
                try:
                    await live_metrics_task
                except asyncio.CancelledError:
                    if current_task is not None and current_task.cancelling() > cancellation_count:
                        raise
            final_observations = []
            if event_callback is not None:
                final_observations.append(
                    _read_live_metrics(
                        sandbox,
                        rollout_id=input_data.rollout_id,
                        event_callback=event_callback,
                        timeout_seconds=_LIVE_METRICS_READ_TIMEOUT_SECONDS,
                        state=live_metrics_state,
                    )
                )
            if trace_destination is not None:
                final_observations.append(
                    _download_request_journal(
                        sandbox,
                        trace_destination,
                        rollout_id=input_data.rollout_id,
                        attempt_id=attempt_id,
                        log_context=log_context,
                    )
                )
                if stage in ("evaluator", "result-download"):
                    final_observations.append(
                        _download_evaluator_logs(
                            sandbox,
                            trace_destination,
                            rollout_id=input_data.rollout_id,
                            attempt_id=attempt_id,
                            log_context=log_context,
                        )
                    )
            # Read concurrently: metrics/journal get 5s, evaluator files get 10s.
            if final_observations:
                observations = await asyncio.gather(*final_observations, return_exceptions=True)
                for observation in observations:
                    if isinstance(observation, asyncio.CancelledError):
                        raise observation
                    if isinstance(observation, Exception):
                        logger.warning(
                            "%s final observation unavailable: %s", log_context, observation
                        )
        finally:
            confirmed_closed = False
            try:
                with profile_block(
                    "driver.sandbox-kill",
                    rollout_id=input_data.rollout_id,
                    sandbox_id=sandbox_id,
                ):
                    await asyncio.wait_for(sandbox.kill(), _SANDBOX_KILL_TIMEOUT_SECONDS)
                confirmed_closed = True
                _e2b_log(f"{log_context} killed")
            except Exception as kill_exc:  # pylint: disable=broad-exception-caught
                _e2b_log(
                    f"{log_context} sandbox kill failed ({type(kill_exc).__name__}: {kill_exc})",
                    stderr=True,
                )
            finally:
                _emit_event(
                    event_callback,
                    kind="sandbox_closed",
                    sandbox_id=sandbox_id,
                    confirmed=confirmed_closed,
                )
                if not confirmed_closed:
                    schedule_sandbox_cleanup(
                        sandbox,
                        sandbox_id=sandbox_id,
                        on_confirmed=lambda: _emit_event(
                            event_callback,
                            kind="sandbox_closed",
                            sandbox_id=sandbox_id,
                            confirmed=True,
                        ),
                    )
