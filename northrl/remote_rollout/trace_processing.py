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

"""Validate diagnostic snapshots outside the training process's GIL."""

import asyncio
import hashlib
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from .wire import run_artifact_io


def _validate_and_save(payload: bytes, destination: Path, rollout_id: str) -> tuple[float, float]:
    # Import in the child: only compressed bytes go in and small timings come
    # back. Returning a parsed trace would recreate all its Python objects in
    # the training process and contend for its GIL again.
    from .e2b_executor import _atomic_write_bytes, _validate_rollout_id
    from .wire import load_trace_gzip

    started = time.perf_counter()
    try:
        trace = load_trace_gzip(payload)
        _validate_rollout_id(trace.rollout_id, rollout_id, "partial trace")
    except ValueError as exc:
        # Pydantic errors can retain the entire decoded input. Do not pickle
        # that object graph back into the training process on invalid input.
        raise ValueError(f"Invalid partial trace snapshot ({type(exc).__name__})") from None
    decoded = time.perf_counter()
    _atomic_write_bytes(destination, payload)
    return decoded - started, time.perf_counter() - decoded


class TraceSnapshotWriter:
    """One rollout's validated snapshot cache, used serially by its sync task."""

    def __init__(self, pool: ProcessPoolExecutor, destination: Path, rollout_id: str):
        self._pool = pool
        self._destination = destination
        self._rollout_id = rollout_id
        self._saved_digest: bytes | None = None

    def save(self, payload: bytes) -> tuple[float, float, str]:
        fingerprint = hashlib.sha256(payload).digest()
        if fingerprint == self._saved_digest:
            return 0.0, 0.0, "unchanged"
        decode_seconds, save_seconds = self._pool.submit(
            _validate_and_save, payload, self._destination, self._rollout_id
        ).result()
        # Invalid snapshots and failed writes must never advance the cache.
        self._saved_digest = fingerprint
        return decode_seconds, save_seconds, "saved"


class _SnapshotPool:
    def __init__(self):
        # Never fork a multithreaded training client with Torch/SDK state.
        self.executor = ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        )
        self.users = 0


@asynccontextmanager
async def trace_snapshot_writer(destination: Path, rollout_id: str):
    """Share two lazy child processes per loop and join them after the last sync.

    Existing background-I/O admission bounds submitted work to four snapshots.
    The caller must drain its save callback before leaving this context.
    """
    loop = asyncio.get_running_loop()
    state = getattr(loop, "_northrl_trace_snapshot_pool", None)
    if state is None:
        state = _SnapshotPool()
        setattr(loop, "_northrl_trace_snapshot_pool", state)
    state.users += 1
    try:
        yield TraceSnapshotWriter(state.executor, destination, rollout_id)
    finally:
        state.users -= 1
        if state.users == 0:
            delattr(loop, "_northrl_trace_snapshot_pool")
            # Joining processes is blocking too. Drain shutdown even when the
            # owning rollout is cancelled so retries cannot leak child workers.
            await run_artifact_io(state.executor.shutdown, wait=True)
