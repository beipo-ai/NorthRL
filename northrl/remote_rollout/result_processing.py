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

"""Optional process isolation for complete E2B result validation/expansion."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from ..utils.profiling import (
    measure_cpu_wall,
    result_profile_event,
    result_profiling_enabled,
)
from .contracts import RemoteRolloutResult
from .wire import run_artifact_io


def _decode_result(
    payload: bytes | bytearray, rollout_id: str, timings: dict[str, Any] | None = None
) -> RemoteRolloutResult:
    from .wire import decode_result, load_result_gzip

    try:
        with measure_cpu_wall(timings, "child_parse"):
            result = load_result_gzip(payload)
            if result.rollout_id != rollout_id:
                raise ValueError("rollout result identity does not match this execution")
        with measure_cpu_wall(timings, "child_restore"):
            return decode_result(result, packed=True)
    except ValueError as exc:
        # Pydantic's error may own the decoded token lists. Only return a small
        # exception, rather than reconstructing that rejected graph in Driver.
        raise ValueError(f"Invalid rollout result ({type(exc).__name__})") from None


def result_size_fields(payload: bytes | bytearray, result: RemoteRolloutResult) -> dict[str, int]:
    """Count expanded positions by segment length, without scanning array contents."""
    return {
        "compressed_bytes": len(payload),
        "expanded_token_positions": sum(len(segment.tokens) for segment in result.segments),
        "segments": len(result.segments),
    }


def _decode_result_profiled(
    payload: bytes | bytearray, rollout_id: str
) -> tuple[RemoteRolloutResult, dict[str, Any]]:
    timings: dict[str, Any] = {}
    result = _decode_result(payload, rollout_id, timings)
    timings.update(result_size_fields(payload, result), child_pid=os.getpid())
    return result, timings


class ResultProcessor:
    """One lazy spawn worker shared by concurrent rollouts on the same loop."""

    def __init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self.users = 0

    def _pool(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
            return self._executor

    def decode(self, payload: bytes | bytearray, rollout_id: str) -> RemoteRolloutResult:
        executor = self._pool()
        try:
            # Full packed arrays cross real IPC. This runs inside the existing
            # two-slot result-processing admission and cancellation drain.
            if not result_profiling_enabled():
                return executor.submit(_decode_result, payload, rollout_id).result()
            started = time.perf_counter()
            result, timings = executor.submit(_decode_result_profiled, payload, rollout_id).result()
            timings["parent_round_trip_wall_seconds"] = time.perf_counter() - started
            # This includes startup, serialization, queueing and scheduling as
            # well as transport. Subtracting child time does not isolate IPC.
            result_profile_event(
                "driver.result-decode", mode="process", rollout_id=rollout_id, **timings
            )
            return result
        except BrokenProcessPool:
            # A subsequent recovery read may retry the same remote artifact.
            # Retire the failed worker without retaining a permanently broken
            # pool for every other in-flight rollout in this experiment.
            with self._lock:
                if self._executor is executor:
                    self._executor = None
            executor.shutdown(wait=True, cancel_futures=True)
            raise

    def close(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)


_CURRENT: ContextVar[ResultProcessor | None] = ContextVar("northrl_result_processor", default=None)


def get_result_processor() -> ResultProcessor | None:
    return _CURRENT.get()


@asynccontextmanager
async def result_processing(mode: str):
    """Own process state for the whole rollout, including artifact recovery.

    Each caller drains its result callback before exit. Closing the last owner
    therefore joins the process without dropping downloaded results or letting
    a cancelled callback race an environment retry.
    """
    if mode not in ("thread", "process"):
        raise ValueError("rollout.execution.result_processing must be 'thread' or 'process'")
    loop = asyncio.get_running_loop()
    if mode == "thread":
        token = _CURRENT.set(None)
        try:
            yield
        finally:
            _CURRENT.reset(token)
        return
    state = getattr(loop, "_northrl_result_processor", None)
    if state is None:
        state = ResultProcessor()
        setattr(loop, "_northrl_result_processor", state)
    state.users += 1
    token = _CURRENT.set(state)
    try:
        yield
    finally:
        _CURRENT.reset(token)
        state.users -= 1
        if state.users == 0:
            delattr(loop, "_northrl_result_processor")
            await run_artifact_io(state.close)
