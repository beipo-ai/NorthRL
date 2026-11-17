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

"""Run a rollout harness through a local NorthGate relay inside E2B."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import logging
import math
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import ModuleType
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI
from omegaconf import DictConfig, OmegaConf

from ..exceptions import ContextLengthExceededError
from ..utils.profiling import profile_block, profile_event
from .contracts import (
    AgentResult,
    EvaluationResult,
    RemoteRolloutResult,
    RemoteRolloutTrace,
    RolloutEnvironmentError,
    RolloutInput,
)
from .execution_budgets import ExecutionBudgets
from .metrics import RolloutMetrics
from .network_policy import assert_network_policy_prepared, network_policy_context
from .northgate_relay import NorthGateRelay
from .request_journal import RequestJournal
from .telemetry import RolloutTelemetry
from .trajectory_recorder import TrajectoryRecorder
from .wire import (
    CoalescingArtifactWriter,
    CompactAgentStageResult,
    CompactRolloutResult,
    CompactRolloutTrace,
    encode_trace,
    load_agent_stage_gzip,
    load_trace_gzip,
    run_artifact_io,
    write_artifact,
)

logger = logging.getLogger(__name__)

HarnessRunner = Callable[
    [RolloutInput, dict[str, str]],
    Awaitable[tuple[float, dict[str, float]]],
]


class E2BRolloutRuntime:
    """Serve local model APIs while one E2B rollout harness runs."""

    def __init__(
        self,
        config: DictConfig,
        *,
        gateway: dict[str, Any],
        port: int = 8000,
        trace_path: str | Path | None = None,
        metrics: RolloutMetrics | None = None,
    ) -> None:
        self._config = config
        self._gateway = gateway
        self._port = port
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._metrics = metrics if metrics is not None else RolloutMetrics()
        self._trace_sink: CoalescingArtifactWriter | None = None

    async def run(
        self,
        input_data: RolloutInput,
        run_harness: HarnessRunner,
        *,
        preparation_deadline: float | None = None,
        on_agent_stop: Callable[[], None] | None = None,
    ) -> RemoteRolloutResult:
        """Run the harness and return exact tokens recorded from NorthGate.

        The async harness receives ``input_data`` plus the local OpenAI and
        Anthropic environment variables. It returns ``(reward, metrics)``.
        """

        budgets = ExecutionBudgets.from_config(self._config)
        loop = asyncio.get_running_loop()
        if preparation_deadline is None:
            preparation_deadline = loop.time() + budgets.preparation_seconds
        finalize_started_at: float | None = None
        agent_execution_seconds = 0.0
        agent_timed_out = False
        externally_cancelled = False

        async def finish_tasks(tasks: tuple[asyncio.Future[Any], ...], *, stage: str) -> list[Any]:
            """Drain cleanup within one shared finalization budget."""
            nonlocal finalize_started_at, externally_cancelled
            if finalize_started_at is None:
                finalize_started_at = loop.time()
            deadline = finalize_started_at + budgets.finalize_seconds
            cancelled = False
            pending = {task for task in tasks if not task.done()}
            while pending:
                try:
                    _, pending = await asyncio.wait(
                        pending, timeout=max(0.0, deadline - loop.time())
                    )
                except asyncio.CancelledError:
                    # Repeated caller cancellation must not abandon process cleanup,
                    # and must still propagate after cleanup has been drained.
                    cancelled = True
                    externally_cancelled = True
                    continue
                if pending:
                    for task in pending:
                        task.cancel()
                        task.add_done_callback(
                            lambda completed: (
                                completed.exception() if not completed.cancelled() else None
                            )
                        )
                    raise RolloutEnvironmentError(
                        f"Agent finalization timed out during {stage}; "
                        "agent completion cannot be trusted"
                    )
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            if cancelled:
                raise asyncio.CancelledError
            return outcomes

        service_errors: dict[str, Exception] = {}
        length_error: ContextLengthExceededError | None = None
        length_stop = asyncio.Event()

        def record_service_call(operation: str, error: Exception | None) -> None:
            nonlocal length_error
            # Keep the stop reason even if another in-flight call later succeeds.
            if isinstance(error, ContextLengthExceededError):
                if length_error is None:
                    length_error = error
                length_stop.set()
                return
            if (
                (length_error is not None or agent_timed_out)
                and error is not None
                and isinstance(error.__cause__, asyncio.CancelledError)
            ):
                # An intentional agent stop deliberately interrupts other
                # in-flight calls while the agent and local server shut down.
                return
            service_errors.pop(operation, None)
            if error is not None:
                service_errors[operation] = error

        recorder = TrajectoryRecorder(on_update=self._trace_writer(input_data))
        relay = NorthGateRelay(
            self._config,
            self._gateway,
            recorder=recorder,
            metrics=self._metrics,
            on_service_call=record_service_call,
            is_validation=bool(input_data.metadata.get("is_validation", False)),
        )
        try:
            app = FastAPI()
            app.include_router(relay.router)

            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=self._port,
                    access_log=False,
                    log_config=None,
                    timeout_graceful_shutdown=10,
                )
            )
            server_task = asyncio.create_task(server.serve())
            try:
                with profile_block(
                    "runtime.server-start",
                    rollout_id=input_data.rollout_id,
                ):
                    async with asyncio.timeout_at(preparation_deadline):
                        await self._wait_until_started(server, server_task)
                try:
                    with profile_block(
                        "runtime.harness",
                        rollout_id=input_data.rollout_id,
                    ):
                        harness_task = asyncio.ensure_future(
                            run_harness(input_data, self._model_environment())
                        )
                        stop_task = asyncio.create_task(length_stop.wait())
                        agent_started_at = loop.time()
                        try:
                            await asyncio.wait(
                                (harness_task, stop_task),
                                timeout=budgets.agent_seconds,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            agent_execution_seconds = loop.time() - agent_started_at
                            agent_timed_out = length_error is None and not harness_task.done()
                            if length_error is None and not agent_timed_out:
                                reward, metrics = await harness_task
                            else:
                                # A context stop is an agent completion, not an evaluation
                                # outcome. Finish process cleanup before committing the stage.
                                was_running = not harness_task.done()
                                if was_running:
                                    if on_agent_stop is not None:
                                        on_agent_stop()
                                    harness_task.cancel()
                                outcome = (
                                    await finish_tasks((harness_task,), stage="agent cleanup")
                                )[0]
                                if isinstance(outcome, Exception) and (
                                    was_running or _failure_status(outcome) == "env_error"
                                ):
                                    raise outcome
                                if agent_timed_out:
                                    profile_event(
                                        "runtime.agent-budget",
                                        "stopped",
                                        rollout_id=input_data.rollout_id,
                                        timeout_seconds=budgets.agent_seconds,
                                    )
                                    reward, metrics = 0.0, {
                                        "agent_exit_code": 124.0,
                                        "e2b_timed_out": 1.0,
                                    }
                                else:
                                    assert length_error is not None
                                    profile_event(
                                        "runtime.context-limit",
                                        "stopped",
                                        rollout_id=input_data.rollout_id,
                                        termination_reason="context_length_exceeded",
                                        **length_error.length_details,
                                    )
                                    reward, metrics = 0.0, {}
                        except asyncio.CancelledError:
                            externally_cancelled = True
                            raise
                        finally:
                            for task in (harness_task, stop_task):
                                if not task.done():
                                    if task is harness_task and on_agent_stop is not None:
                                        on_agent_stop()
                                    task.cancel()
                            await finish_tasks((harness_task, stop_task), stage="agent cleanup")
                    result = RemoteRolloutResult(
                        rollout_id=input_data.rollout_id,
                        termination_reason=(
                            "context_length_exceeded" if length_error is not None else None
                        ),
                        length_details=(
                            length_error.length_details if length_error is not None else {}
                        ),
                        segments=(recorder.to_remote_segments() if recorder.segment_count else []),
                        message_format=recorder.message_format,
                        message_system=recorder.message_system,
                        tools=recorder.tools,
                        messages=recorder.to_messages(),
                        reward=reward,
                        metrics={**metrics, **self._metrics.snapshot()},
                    )
                except Exception as exc:
                    if service_errors:
                        operation = next(reversed(service_errors))
                        error = service_errors[operation]
                        raise RolloutEnvironmentError(
                            f"Model service {operation} failed: {type(error).__name__}: {error}"
                        ) from exc
                    raise
            except asyncio.CancelledError:
                externally_cancelled = True
                raise
            finally:
                with profile_block(
                    "runtime.server-stop",
                    rollout_id=input_data.rollout_id,
                ):
                    server.should_exit = True
                    try:
                        outcome = (await finish_tasks((server_task,), stage="server shutdown"))[0]
                        if isinstance(outcome, BaseException):
                            raise outcome
                    except Exception as exc:
                        raise RolloutEnvironmentError(f"Model API shutdown failed: {exc}") from exc
        finally:
            with profile_block(
                "runtime.relay-close",
                rollout_id=input_data.rollout_id,
            ):
                try:
                    close_task = asyncio.create_task(relay.close())
                    outcome = (await finish_tasks((close_task,), stage="relay shutdown"))[0]
                    if isinstance(outcome, BaseException):
                        raise outcome
                except Exception as exc:
                    raise RolloutEnvironmentError(f"Model service cleanup failed: {exc}") from exc
                finally:
                    if recorder.segment_count:
                        try:
                            self._trace_writer(input_data)(recorder)
                        except Exception:
                            logger.warning(
                                "Could not checkpoint final runtime observations", exc_info=True
                            )
                    if self._trace_sink is not None:
                        try:
                            # Unlike async service cleanup, a filesystem call in
                            # a thread cannot be forcibly cancelled. Join before
                            # publishing the final trace/stage; a stalled mount
                            # can therefore extend the finalization budget.
                            await self._trace_sink.aclose()
                        except asyncio.CancelledError:
                            externally_cancelled = True
                        except Exception:
                            logger.warning("Could not flush final runtime trace", exc_info=True)
                    if externally_cancelled:
                        # Service cleanup remains bounded; artifact workers are
                        # drained above. Preserve cancellation instead of scoring.
                        raise asyncio.CancelledError
        if service_errors:
            operation = next(reversed(service_errors))
            error = service_errors[operation]
            raise RolloutEnvironmentError(
                f"Model service {operation} failed: {type(error).__name__}: {error}"
            )
        # Include native requests that completed while the local server drained.
        if recorder.segment_count:
            result.segments = recorder.to_remote_segments()
            result.message_format = recorder.message_format
            result.message_system = recorder.message_system
            result.tools = recorder.tools
            result.messages = recorder.to_messages()
        result.metrics.update(self._metrics.snapshot())  # pylint: disable=no-member
        result.metrics.update(  # pylint: disable=no-member
            agent_execution_seconds=agent_execution_seconds,
            agent_finalize_seconds=(
                loop.time() - finalize_started_at if finalize_started_at is not None else 0.0
            ),
        )
        return result

    def _trace_writer(
        self,
        input_data: RolloutInput,
    ) -> Callable[[TrajectoryRecorder], None]:
        trace_path = self._trace_path

        def write_trace(recorder: TrajectoryRecorder) -> None:
            self._metrics.set_trace_segments(recorder.segment_count)
            if trace_path is None:
                return
            trace = CompactRolloutTrace(
                schema_version=2,
                rollout_id=input_data.rollout_id,
                metadata=input_data.metadata,
                segments=recorder.to_compact_segments(),
                message_format=recorder.message_format,
                message_system=recorder.message_system,
                tools=recorder.tools,
                messages=recorder.to_messages(),
                metrics=self._metrics.snapshot(),
            )
            if self._trace_sink is None:
                self._trace_sink = CoalescingArtifactWriter(trace_path, write=write_artifact)
            self._trace_sink.submit(trace)

        return write_trace

    def _model_environment(self) -> dict[str, str]:
        base_url = f"http://127.0.0.1:{self._port}"
        return {
            "OPENAI_BASE_URL": f"{base_url}/v1",
            "OPENAI_API_KEY": "northrl",
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": "northrl",
        }

    @staticmethod
    async def _wait_until_started(
        server: uvicorn.Server,
        server_task: asyncio.Task[None],
    ) -> None:
        async with asyncio.timeout(10):
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError("E2B model API exited before startup")
                await asyncio.sleep(0.01)


def _prepare_private_directory(directory: Path) -> None:
    if directory.is_symlink():
        raise RuntimeError(f"platform directory must not be a symlink: {directory}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.resolve() != directory.absolute() or directory.stat().st_uid != os.geteuid():
        raise RuntimeError(f"platform directory must be owned by the runtime: {directory}")
    directory.chmod(0o700)


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a platform artifact in an agent-inaccessible directory."""
    directory = path.parent
    _prepare_private_directory(directory)
    if path.is_symlink():
        raise RuntimeError(f"rollout output must not be a symlink: {path}")
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_input(path: Path) -> tuple[RolloutInput, dict[str, Any], dict[str, Any], str]:
    """Keep the complete task payload private to the runtime and its task hooks."""
    if path.is_symlink():
        raise RuntimeError(f"rollout input must not be a symlink: {path}")
    if path.resolve() != path.absolute() or path.stat().st_uid != os.geteuid():
        raise RuntimeError(f"rollout input must be owned by the runtime: {path}")
    path.chmod(0o600)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("rollout input must be a JSON object")
    input_data = RolloutInput.model_validate(payload.get("input_data"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError("rollout input config must be a JSON object")
    gateway = payload.get("gateway")
    if not isinstance(gateway, dict):
        raise ValueError("rollout input gateway must be a JSON object")
    for name in ("base_url", "model", "sampling_session_id", "weight_version"):
        if not isinstance(gateway.get(name), str) or not gateway[name].strip():
            raise ValueError(f"rollout input gateway.{name} must be a non-empty string")
    attempt_id = payload.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError("rollout input attempt_id must be a non-empty string")
    return input_data, config, gateway, attempt_id


def _load_adapter(entrypoint: str) -> ModuleType:
    script = Path(entrypoint).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_northrl_task_adapter", script)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load task adapter: {script}")
    adapter = importlib.util.module_from_spec(spec)
    # A regular module name deliberately avoids running a script's __main__ block.
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    for name, argument_count in (("run_agent", 3), ("evaluate", 4), ("prepare", 2)):
        if name == "prepare" and not hasattr(adapter, name):
            continue
        hook = getattr(adapter, name, None)
        if not inspect.iscoroutinefunction(hook):
            raise TypeError(f"task adapter must expose async {name}")
        try:
            inspect.signature(hook).bind(*(None for _ in range(argument_count)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"task adapter {name} must accept {argument_count} positional arguments"
            ) from exc
    return adapter


async def _call_hook(adapter: ModuleType, name: str, *args: Any) -> Any:
    hook = getattr(adapter, name)
    result = hook(*args)
    if not inspect.isawaitable(result):
        raise RolloutEnvironmentError(f"task adapter {name} must be async")
    return await result


def _existing_workspace(value: Any) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("task workspace must be an explicit absolute directory")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("task workspace must be an absolute directory")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"task workspace is not a directory: {path}")
    return path


async def _resolve_workspace(
    adapter: ModuleType, input_data: RolloutInput, workspace: Path | None
) -> Path:
    if workspace is not None:
        return _existing_workspace(workspace)
    resolver = getattr(adapter, "resolve_workspace", None)
    if resolver is not None:
        value = resolver(input_data)
        if inspect.isawaitable(value):
            value = await value
        return _existing_workspace(value)
    return _existing_workspace(input_data.task.get("workspace"))


def _write_state(
    output_path: Path,
    rollout_id: str | None,
    attempt_id: str | None,
    phase: str,
    **details: Any,
) -> None:
    _write_private_json(
        Path(f"{str(output_path).removesuffix('.gz')}.state.json"),
        {"rollout_id": rollout_id, "attempt_id": attempt_id, "phase": phase, **details},
    )


def _failure_status(
    error: BaseException,
    *,
    default_status: Literal["env_error", "task_failed"] = "env_error",
) -> Literal["env_error", "task_failed"]:
    pending = [error]
    seen: set[int] = set()
    status = default_status
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        reported = getattr(current, "rollout_status", None)
        if reported == "env_error":
            return "env_error"
        if reported == "task_failed":
            status = "task_failed"
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return status


def _report_failure(
    input_path: Path,
    output_path: Path,
    trace_path: Path,
    phase: str,
    error: Exception,
    *,
    termination_reason: Literal["context_length_exceeded"] | None = None,
    length_details: dict[str, int] | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    rollout_id = attempt_id = None
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        candidate = payload["input_data"]["rollout_id"]
        if isinstance(candidate, str) and candidate.strip():
            rollout_id = candidate
        candidate = payload.get("attempt_id")
        if isinstance(candidate, str) and candidate.strip():
            attempt_id = candidate
    except (OSError, ValueError, KeyError, TypeError):
        pass
    status = _failure_status(error)
    _write_state(
        output_path,
        rollout_id,
        attempt_id,
        phase,
        status=status,
        error_type=type(error).__name__,
        error_message=str(error)[-2000:],
        termination_reason=termination_reason,
        length_details=length_details or {},
    )
    if rollout_id is None:
        return
    trace = None
    try:
        candidate = load_trace_gzip(trace_path.read_bytes())
        if candidate.rollout_id == rollout_id:
            trace = candidate
    except (OSError, ValueError):
        pass
    trainable_trace = trace if status == "task_failed" else None
    result = CompactRolloutResult(
        schema_version=2,
        rollout_id=rollout_id,
        status=status,
        reward=0.0,
        error_type=type(error).__name__,
        error_message=str(error)[-2000:],
        termination_reason=termination_reason,
        length_details=length_details or {},
        segments=trainable_trace.segments if trainable_trace else [],
        message_format=trainable_trace.message_format if trainable_trace else None,
        message_system=trainable_trace.message_system if trainable_trace else None,
        tools=trainable_trace.tools if trainable_trace else None,
        messages=trainable_trace.messages if trainable_trace else [],
        metrics={**(trace.metrics if trace else {}), **(metrics or {})},
    )
    write_artifact(output_path, result)


async def _run_phase(
    *,
    phase: str,
    input_path: Path,
    output_path: Path,
    trace_path: Path,
    stage_path: Path,
    entrypoint: str,
    evaluation_assets: Path | None = None,
    workspace: Path | None = None,
    phase_timeout_seconds: float | None = None,
    preparation_timeout_seconds: float | None = None,
) -> None:
    """Run one half of a rollout; only the host may insert hidden evaluation assets."""
    phase_started_at = asyncio.get_running_loop().time()
    active_phase = "environment"
    termination_reason: Literal["context_length_exceeded"] | None = None
    length_details: dict[str, int] = {}
    active_telemetry: RolloutTelemetry | None = None
    failure_metrics: dict[str, float] = {}
    try:
        if phase not in {"agent", "evaluate"}:
            raise ValueError("phase must be agent or evaluate")
        if phase_timeout_seconds is not None:
            if (
                phase != "evaluate"
                or isinstance(phase_timeout_seconds, bool)
                or not isinstance(phase_timeout_seconds, (int, float))
                or not math.isfinite(phase_timeout_seconds)
                or phase_timeout_seconds <= 0
            ):
                raise ValueError(
                    "phase timeout must be positive and finite, and is only for evaluate"
                )
        if preparation_timeout_seconds is not None:
            if (
                phase != "agent"
                or isinstance(preparation_timeout_seconds, bool)
                or not isinstance(preparation_timeout_seconds, (int, float))
                or not math.isfinite(preparation_timeout_seconds)
                or preparation_timeout_seconds <= 0
            ):
                raise ValueError(
                    "preparation timeout must be positive and finite, and is only for agent"
                )
        input_data, config, gateway, attempt_id = _load_input(input_path)
        budgets = ExecutionBudgets.from_config(config)
        network_policy = config.get("rollout", {}).get("execution", {}).get("network_policy")
        network_task_id = input_data.metadata.get("task_id")
        if not isinstance(network_task_id, str):
            network_task_id = None
        await run_artifact_io(
            _write_state, output_path, input_data.rollout_id, attempt_id, active_phase
        )
        if phase == "agent":
            preparation_deadline = phase_started_at + min(
                budgets.preparation_seconds,
                (
                    preparation_timeout_seconds
                    if preparation_timeout_seconds is not None
                    else budgets.preparation_seconds
                ),
            )
            if evaluation_assets is None:
                raise ValueError("evaluation assets directory must be specified")
            _prepare_private_directory(evaluation_assets)
            if any(evaluation_assets.iterdir()):
                raise RolloutEnvironmentError(
                    "hidden evaluation assets must be absent before the agent"
                )
            for path in (
                output_path,
                trace_path,
                stage_path,
                output_path.with_name("requests.jsonl"),
            ):
                _prepare_private_directory(path.parent)
                if path.is_symlink():
                    raise RuntimeError(f"rollout artifact must not be a symlink: {path}")
                path.unlink(missing_ok=True)
            adapter = _load_adapter(entrypoint)
            async with asyncio.timeout_at(preparation_deadline):
                workspace = await _resolve_workspace(adapter, input_data, workspace)
            async with RolloutTelemetry(
                input_data.rollout_id,
                output_path,
                attempt_id=attempt_id,
                completion_phase="agent_completed",
            ) as telemetry:
                active_telemetry = telemetry
                telemetry.metrics.set_request_observer(
                    RequestJournal(
                        output_path.with_name("requests.jsonl"), input_data.rollout_id, attempt_id
                    )
                )
                with network_policy_context(network_policy, task_id=network_task_id):
                    if hasattr(adapter, "prepare"):
                        async with asyncio.timeout_at(preparation_deadline):
                            await _call_hook(adapter, "prepare", input_data, workspace)
                    assert_network_policy_prepared()
                agent_result = AgentResult()
                agent_stopping = False

                def mark_agent_stopping() -> None:
                    nonlocal agent_stopping
                    agent_stopping = True

                async def run_agent(
                    agent_input_data: RolloutInput, model_environment: dict[str, str]
                ) -> tuple[float, dict[str, float]]:
                    nonlocal agent_result, active_phase
                    active_phase = "agent"
                    await run_artifact_io(
                        _write_state, output_path, input_data.rollout_id, attempt_id, active_phase
                    )
                    telemetry.set_phase(active_phase)
                    try:
                        try:
                            with network_policy_context(network_policy, task_id=network_task_id):
                                value = await _call_hook(
                                    adapter,
                                    "run_agent",
                                    agent_input_data,
                                    workspace,
                                    model_environment,
                                )
                        except Exception as exc:
                            if _failure_status(exc, default_status="task_failed") == "env_error":
                                raise
                            if agent_stopping:
                                # The runtime is waiting for a cancelled agent to prove
                                # it has stopped. A cleanup error is not a normal exit.
                                raise RolloutEnvironmentError(
                                    f"Agent cleanup failed after cancellation: {exc}"
                                ) from exc
                            # The adapter has finished its cleanup. Its execution outcome
                            # is diagnostic; only evaluate() determines the task reward.
                            value = AgentResult(
                                exit_code=124 if isinstance(exc, TimeoutError) else 1
                            )
                            logger.warning(
                                "Agent execution failed; continuing to evaluation: rollout_id=%s",
                                input_data.rollout_id,
                                exc_info=True,
                            )
                        if value is not None:
                            if not isinstance(value, AgentResult):
                                raise RolloutEnvironmentError(
                                    "run_agent must return AgentResult or None"
                                )
                            try:
                                agent_result = AgentResult.model_validate(
                                    value.model_dump(warnings=False)
                                )
                            except ValueError as exc:
                                raise RolloutEnvironmentError(
                                    f"invalid AgentResult: {exc}"
                                ) from exc
                    finally:
                        active_phase = "environment"
                        active_error = sys.exc_info()[1]
                        try:
                            await run_artifact_io(
                                _write_state,
                                output_path,
                                input_data.rollout_id,
                                attempt_id,
                                active_phase,
                            )
                        except asyncio.CancelledError as exc:
                            # A context stop can cancel this bookkeeping after the
                            # adapter already failed cleanup. Preserve that failure.
                            if isinstance(active_error, Exception):
                                raise active_error from exc
                            raise
                        telemetry.set_phase("runtime_cleanup")
                    return 0.0, {"agent_exit_code": float(agent_result.exit_code)}

                result = await E2BRolloutRuntime(
                    OmegaConf.create(config),
                    gateway=gateway,
                    trace_path=trace_path,
                    metrics=telemetry.metrics,
                ).run(
                    input_data,
                    run_agent,
                    preparation_deadline=preparation_deadline,
                    on_agent_stop=mark_agent_stopping,
                )
                # The result is assembled before gateway shutdown. Include requests
                # that finished or were cancelled while shutdown drained the server,
                # so evaluation cannot restore an older request-metrics checkpoint.
                result.metrics.update(telemetry.metrics.snapshot())
                if result.metrics.get("e2b_timed_out") == 1.0:
                    agent_result = agent_result.model_copy(update={"exit_code": 124})
                telemetry.update_metrics(result.metrics)
                termination_reason = result.termination_reason
                length_details = result.length_details
                trace = (
                    encode_trace(
                        RemoteRolloutTrace(
                            rollout_id=input_data.rollout_id,
                            metadata=input_data.metadata,
                            segments=result.segments,
                            message_format=result.message_format,
                            message_system=result.message_system,
                            tools=result.tools,
                            messages=result.messages,
                            metrics=result.metrics,
                        )
                    )
                    if result.segments
                    else None
                )
                stage = CompactAgentStageResult(
                    schema_version=2,
                    rollout_id=input_data.rollout_id,
                    attempt_id=attempt_id,
                    workspace=str(workspace),
                    agent_result=agent_result,
                    trace=trace,
                    metrics=result.metrics if trace is None else {},
                    termination_reason=termination_reason,
                    length_details=length_details,
                )
                # The runtime has stopped its local server and HTTP client before this commit.
                if trace is not None:
                    await run_artifact_io(write_artifact, trace_path, trace)
                await run_artifact_io(write_artifact, stage_path, stage)
                active_phase = "agent_completed"
                await run_artifact_io(
                    _write_state,
                    output_path,
                    input_data.rollout_id,
                    attempt_id,
                    active_phase,
                    termination_reason=termination_reason,
                    length_details=length_details,
                )
            return

        stage = await run_artifact_io(lambda: load_agent_stage_gzip(stage_path.read_bytes()))
        if stage.rollout_id != input_data.rollout_id or stage.attempt_id != attempt_id:
            raise ValueError("agent stage identity does not match this rollout attempt")
        trace_metrics = stage.trace.metrics if stage.trace is not None else stage.metrics
        failure_metrics.update(trace_metrics)
        termination_reason = stage.termination_reason
        length_details = stage.length_details
        stage_workspace = _existing_workspace(stage.workspace)
        if str(stage_workspace) != stage.workspace:
            raise ValueError("agent stage workspace changed after agent completion")
        if workspace is not None and _existing_workspace(workspace) != stage_workspace:
            raise ValueError("evaluation workspace must match the completed agent stage")
        if evaluation_assets is None or not evaluation_assets.is_dir():
            raise ValueError("evaluation assets must be an uploaded directory")
        _prepare_private_directory(evaluation_assets)
        adapter = _load_adapter(entrypoint)
        active_phase = "evaluator"
        await run_artifact_io(
            _write_state,
            output_path,
            input_data.rollout_id,
            attempt_id,
            active_phase,
            termination_reason=termination_reason,
            length_details=length_details,
        )
        async with RolloutTelemetry(
            input_data.rollout_id,
            output_path,
            attempt_id=attempt_id,
            phase=active_phase,
            initial_metrics=trace_metrics,
            resume=True,
        ) as telemetry:
            active_telemetry = telemetry
            evaluation_seconds = min(
                budgets.evaluation_seconds,
                (
                    phase_timeout_seconds
                    if phase_timeout_seconds is not None
                    else budgets.evaluation_seconds
                ),
            )
            try:
                async with asyncio.timeout_at(phase_started_at + evaluation_seconds):
                    with network_policy_context(network_policy, task_id=network_task_id):
                        evaluation = await _call_hook(
                            adapter,
                            "evaluate",
                            input_data,
                            stage_workspace,
                            stage.agent_result,
                            evaluation_assets,
                        )
            except TimeoutError:
                failure_metrics["e2b_timed_out"] = 1.0
                raise
            if not isinstance(evaluation, EvaluationResult):
                raise TypeError("evaluate must return EvaluationResult")
            evaluation = EvaluationResult.model_validate(evaluation.model_dump(warnings=False))
            wire_result = CompactRolloutResult(
                schema_version=2,
                rollout_id=input_data.rollout_id,
                status=evaluation.status,
                segments=stage.trace.segments if stage.trace is not None else [],
                message_format=stage.trace.message_format if stage.trace is not None else None,
                message_system=stage.trace.message_system if stage.trace is not None else None,
                tools=stage.trace.tools if stage.trace is not None else None,
                messages=stage.trace.messages if stage.trace is not None else [],
                reward=evaluation.reward,
                termination_reason=termination_reason,
                length_details=length_details,
                metrics={**evaluation.metrics, **trace_metrics},
            )
            telemetry.update_metrics(wire_result.metrics)
            await run_artifact_io(write_artifact, output_path, wire_result)
            active_phase = "finished"
            await run_artifact_io(
                _write_state,
                output_path,
                input_data.rollout_id,
                attempt_id,
                active_phase,
                termination_reason=termination_reason,
                length_details=length_details,
            )
    except Exception as exc:
        if active_telemetry is not None:
            try:
                failure_metrics.update(active_telemetry.metrics_snapshot())
            except Exception:  # pylint: disable=broad-exception-caught
                # Diagnostics must not replace the error that determines retry
                # and whether the partial trace is a valid training sample.
                logger.warning("Could not collect final rollout metrics", exc_info=True)
        await run_artifact_io(
            _report_failure,
            input_path,
            output_path,
            trace_path,
            active_phase,
            exc,
            termination_reason=termination_reason,
            length_details=length_details,
            metrics=failure_metrics,
        )
        raise


def main() -> None:
    # This CLI runs without the training controller's logging configuration.
    level = logging.getLevelNamesMapping().get(os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    # Imported SDKs may have already installed a handler before this entrypoint.
    logging.getLogger().setLevel(level)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("agent", "evaluate"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-output", required=True, type=Path)
    parser.add_argument("--stage-path", required=True, type=Path)
    parser.add_argument("--evaluation-assets", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--preparation-timeout-seconds", type=float)
    args = parser.parse_args()
    asyncio.run(
        _run_phase(
            phase=args.phase,
            input_path=args.input.absolute(),
            output_path=args.output.absolute(),
            trace_path=args.trace_output.absolute(),
            stage_path=args.stage_path.absolute(),
            entrypoint=args.entrypoint,
            evaluation_assets=args.evaluation_assets.absolute() if args.evaluation_assets else None,
            workspace=args.workspace,
            phase_timeout_seconds=args.timeout_seconds,
            preparation_timeout_seconds=args.preparation_timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
