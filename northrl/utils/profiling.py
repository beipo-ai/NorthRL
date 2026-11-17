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

"""Small wall-clock profiling helpers for long-running rollout stages."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def profile_event(stage: str, status: str, **fields: Any) -> None:
    """Log stage timings; ``level`` optionally overrides the status-based level."""

    level = fields.pop("level", None)
    if level is None:
        if status == "failed":
            level = logging.ERROR
        elif status in {"started", "completed", "skipped"}:
            level = logging.DEBUG
        else:
            level = logging.INFO
    if not logger.isEnabledFor(level):
        return

    values = [f"stage={stage}", f"status={status}"]
    for name, value in fields.items():
        if name == "duration_seconds" and isinstance(value, (int, float)):
            values.append(f"{name}={float(value):.3f}")
        else:
            values.append(f"{name}={value!r}")
    logger.log(level, "[PROFILE] %s", " ".join(values))


@contextmanager
def profile_block(stage: str, **fields: Any) -> Iterator[None]:
    """Time a sync or async block, including failures and cancellation."""

    started_at = time.perf_counter()
    profile_event(stage, "started", **fields)
    try:
        yield
    except BaseException as exc:
        profile_event(
            stage,
            "failed",
            **fields,
            duration_seconds=time.perf_counter() - started_at,
            error_type=type(exc).__name__,
        )
        raise
    else:
        profile_event(
            stage,
            "completed",
            **fields,
            duration_seconds=time.perf_counter() - started_at,
        )


def result_profiling_enabled() -> bool:
    """Enable aggregate result diagnostics explicitly in the Driver environment."""
    return os.environ.get("NORTHRL_PROFILE_RESULTS") == "1"


def result_profile_event(stage: str, status: str = "completed", **fields: Any) -> None:
    """Optional scalar-only diagnostics must never change a rollout's outcome."""
    if result_profiling_enabled():
        try:
            profile_event(stage, status, level=logging.INFO, **fields)
        except Exception:
            pass


@contextmanager
def measure_cpu_wall(fields: dict[str, Any] | None, prefix: str) -> Iterator[None]:
    """Measure the executing thread, excluding unrelated Driver threads' CPU."""
    if fields is None:
        yield
        return
    wall, cpu = time.perf_counter(), time.thread_time()
    try:
        yield
    finally:
        fields[f"{prefix}_wall_seconds"] = time.perf_counter() - wall
        fields[f"{prefix}_cpu_seconds"] = time.thread_time() - cpu


@contextmanager
def result_profile_block(stage: str, **fields: Any) -> Iterator[None]:
    """One opt-in completion record per stage; no per-token work or logging."""
    if not result_profiling_enabled():
        yield
        return
    status = "completed"
    try:
        with measure_cpu_wall(fields, "parent"):
            yield
    except BaseException as exc:
        status = "failed"
        fields["error_type"] = type(exc).__name__
        raise
    finally:
        result_profile_event(stage, status, **fields)
