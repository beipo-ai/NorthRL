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

"""Run isolated Python checkers with independent, bounded stage recovery."""

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from ._runtime_context import reward_stop_requested
from .code_e2b import (
    _command_failure_details,
    _e2b_retry_reason,
    _error_message,
    _raise_if_reward_stopping,
)

if TYPE_CHECKING:
    from e2b import CommandResult

logger = logging.getLogger(__name__)

_MAX_SANDBOX_OUTPUT_BYTES = 2_000_000
_MAX_STAGE_FAILURES = 3
_ATTEMPT_TIMEOUT_S = 60
_CREATE_TIMEOUT_S = 30
_UPLOAD_TIMEOUT_S = 10
_RUN_TIMEOUT_S = 10
_TASK_CLEANUP_TIMEOUT_S = 3
_SANDBOX_CLEANUP_TIMEOUT_S = 10


class E2BCheckerError(RuntimeError):
    """An attempt cannot safely continue or supply a timeout zero."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"E2B checker {stage} failed: {message}")


@dataclass
class _CheckerAttempt:
    stage: str = "create"
    sandbox: Any = None
    deadline: asyncio.Timeout | None = None
    deadline_expired: bool = False
    command_task: "asyncio.Task[CommandResult] | None" = None
    task_cleanup_incomplete: bool = False
    result_ready: bool = False


def _observe_command(task: asyncio.Task) -> None:
    # Retrieve late failures even when bounded cleanup cannot finish waiting.
    if not task.cancelled():
        task.exception()


async def _run_command(state: _CheckerAttempt, path: str, check_output: Any) -> "CommandResult":
    async with asyncio.timeout(_RUN_TIMEOUT_S) as run_deadline:
        state.command_task = asyncio.create_task(
            state.sandbox.commands.run(
                f"python3 -I {path}",
                timeout=_RUN_TIMEOUT_S,
                request_timeout=_RUN_TIMEOUT_S,
                on_stdout=check_output,
                on_stderr=check_output,
            ),
            name="e2b-checker-command",
        )
        state.command_task.add_done_callback(_observe_command)
        try:
            return await asyncio.shield(state.command_task)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            caller_active = (
                caller is not None and caller.cancelling() == 0 and not reward_stop_requested()
            )
            timers_active = (
                not run_deadline.expired()
                and state.deadline is not None
                and not state.deadline.expired()
            )
            child_cancelled = state.command_task.done() and state.command_task.cancelled()
            if caller_active and timers_active and child_cancelled:
                raise ConnectionError(
                    "owned checker command task cancelled while caller remained active"
                ) from None
            # Keep caller cancellation and timer conversion intact.
            raise


async def _cleanup_attempt(state: _CheckerAttempt, verifier: str) -> None:
    from e2b.exceptions import SandboxNotFoundException

    try:
        if state.command_task is not None and not state.command_task.done():
            try:
                async with asyncio.timeout(_TASK_CLEANUP_TIMEOUT_S):
                    state.command_task.cancel()
                    await asyncio.wait({state.command_task})
            except Exception as error:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "E2B checker task cleanup failed: verifier=%s sandbox_id=%s error=%s",
                    verifier,
                    getattr(state.sandbox, "sandbox_id", None),
                    _error_message(error)[:320],
                )
    finally:
        if state.sandbox is not None:
            attempts = 2 if state.result_ready else 1
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    _raise_if_reward_stopping()
                try:
                    async with asyncio.timeout(_SANDBOX_CLEANUP_TIMEOUT_S):
                        await state.sandbox.kill(request_timeout=_SANDBOX_CLEANUP_TIMEOUT_S)
                    break
                except SandboxNotFoundException:
                    break
                except Exception as error:  # pylint: disable=broad-exception-caught
                    action = "return_result" if state.result_ready else "preserve_original_error"
                    if attempt < attempts:
                        action = "retry_cleanup"
                    logger.warning(
                        "E2B checker cleanup unconfirmed: verifier=%s sandbox_id=%s "
                        "cleanup_attempt=%s/%s action=%s error=%s",
                        verifier,
                        getattr(state.sandbox, "sandbox_id", None),
                        attempt,
                        attempts,
                        action,
                        _error_message(error)[:320],
                    )
        # Terminating the sandbox may also finish a previously stuck RPC task.
        state.task_cleanup_incomplete = (
            state.command_task is not None and not state.command_task.done()
        )


async def _run_checker_attempt(
    source: str, path: str, verifier: str, state: _CheckerAttempt
) -> "CommandResult":
    from e2b import AsyncSandbox, CommandExitException

    output_bytes = 0

    def check_output(chunk: str) -> None:
        nonlocal output_bytes
        output_bytes += len(chunk.encode("utf-8"))
        if output_bytes > _MAX_SANDBOX_OUTPUT_BYTES:
            raise ValueError(f"{verifier} E2B checker output exceeds 2000000 bytes")

    try:
        async with asyncio.timeout(_ATTEMPT_TIMEOUT_S) as state.deadline:
            async with asyncio.timeout(_CREATE_TIMEOUT_S):
                state.sandbox = await AsyncSandbox.create(
                    template=os.environ.get("E2B_TEMPLATE") or None,
                    timeout=_ATTEMPT_TIMEOUT_S,
                    allow_internet_access=False,
                    request_timeout=_CREATE_TIMEOUT_S,
                )
            _raise_if_reward_stopping()
            state.stage = "upload"
            async with asyncio.timeout(_UPLOAD_TIMEOUT_S):
                await state.sandbox.files.write(path, source, request_timeout=_UPLOAD_TIMEOUT_S)
            _raise_if_reward_stopping()
            state.stage = "run"
            try:
                result = await _run_command(state, path, check_output)
            except CommandExitException as error:
                result = error
            if result.exit_code != 0:
                failure = (
                    result
                    if isinstance(result, CommandExitException)
                    else CommandExitException(
                        exit_code=result.exit_code,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        error=result.error,
                    )
                )
                details = await asyncio.to_thread(_command_failure_details, failure)
                logger.warning(
                    "E2B checker command exited: verifier=%s sandbox_id=%s details=%s",
                    verifier,
                    getattr(state.sandbox, "sandbox_id", None),
                    details,
                )
                if result.exit_code in (126, 127):
                    raise RuntimeError(
                        "The E2B template must provide an executable python3"
                    ) from failure
            state.result_ready = True
            return result
    except Exception:
        # Capture the timer before independent cleanup can consume more time.
        state.deadline_expired = state.deadline is not None and state.deadline.expired()
        raise
    finally:
        await _cleanup_attempt(state, verifier)


def _log_failure(
    state: _CheckerAttempt,
    verifier: str,
    error: Exception,
    attempt: int,
    reason: str | None,
    action: str,
    delay: float,
) -> None:
    if state.deadline_expired:
        reason = "attempt_deadline_expired"
    elif state.task_cleanup_incomplete:
        reason = "task_cleanup_incomplete"
    logger.log(
        logging.ERROR if action == "raise" else logging.WARNING,
        "E2B checker failure: verifier=%s stage=%s attempt=%s reason=%s "
        "sandbox_id=%s action=%s retry_delay_s=%.3f error_type=%s error=%s",
        verifier,
        state.stage,
        attempt,
        reason,
        getattr(state.sandbox, "sandbox_id", None),
        action,
        delay,
        type(error).__name__,
        _error_message(error)[:320],
    )


async def run_e2b_checker(
    source: str,
    path: str,
    verifier: str,
    *,
    metrics: dict[str, float] | None = None,
) -> "CommandResult | None":
    """Return a command result, or None after three run-only timeout failures.

    Each stage retries twice with jitter in fresh sandboxes. Counters persist
    across attempts, while each attempt gets new timers and an output budget.
    An expired attempt deadline or unresolved owned command task never retries
    or supplies a zero. Ordinary checker exits retain their scoring semantics.
    """
    from e2b import TimeoutException

    failures = dict.fromkeys(("create", "upload", "run"), 0)
    run_timeouts = 0
    retry_stage = None
    max_attempts = 1 + len(failures) * (_MAX_STAGE_FAILURES - 1)
    for attempt in range(1, max_attempts + 1):
        _raise_if_reward_stopping()
        if retry_stage is not None and metrics is not None:
            for key in ("retry_count", f"{retry_stage}_retry_count"):
                metrics[key] = metrics.get(key, 0.0) + 1.0
        state = _CheckerAttempt()
        try:
            result = await _run_checker_attempt(source, path, verifier, state)
        except Exception as error:
            _raise_if_reward_stopping()
            reason = (
                None
                if state.deadline_expired or state.task_cleanup_incomplete
                else _e2b_retry_reason(error, state.stage)
            )
            timed_out = isinstance(error, (TimeoutError, TimeoutException, httpx.TimeoutException))
            if timed_out and metrics is not None:
                metrics["timeout_count"] = metrics.get("timeout_count", 0.0) + 1.0
            if reason is not None:
                failures[state.stage] += 1
                run_timeouts += int(state.stage == "run" and timed_out)
            exhausted = failures[state.stage] >= _MAX_STAGE_FAILURES
            timeout_zero = (
                reason is not None
                and state.stage == "run"
                and exhausted
                and run_timeouts == failures["run"]
            )
            retry = reason is not None and not exhausted
            delay = random.uniform(1.0, 3.0) if retry else 0.0
            action = "reward_zero" if timeout_zero else "raise"
            if retry:
                action = "retry"
            _log_failure(state, verifier, error, attempt, reason, action, delay)
            if state.deadline_expired or state.task_cleanup_incomplete:
                message = (
                    "attempt deadline expired"
                    if state.deadline_expired
                    else "owned command task did not stop during bounded cleanup"
                )
                raise E2BCheckerError(state.stage, message) from None
            _raise_if_reward_stopping()
            if timeout_zero:
                return None
            if not retry:
                raise
            retry_stage = state.stage
            # Cleanup already completed; cancellation during backoff propagates.
            await asyncio.sleep(delay)
            continue
        if retry_stage is not None:
            logger.info("E2B checker recovered: verifier=%s failures=%s", verifier, failures)
        return result
    raise RuntimeError("E2B checker retry budget exhausted")
