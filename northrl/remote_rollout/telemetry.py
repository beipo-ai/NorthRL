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

"""Publish small metric snapshots throughout the sandbox runtime and evaluation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, RLock, Thread
from typing import Any, Self

from .metrics import LIVE_METRICS_INTERVAL_SECONDS, MODEL_REQUEST_GAUGES, RolloutMetrics

logger = logging.getLogger(__name__)


class RolloutTelemetry:
    """Keep a heartbeat independent of successful model turns and debug traces."""

    def __init__(
        self,
        rollout_id: str,
        output_path: Path,
        *,
        attempt_id: str | None = None,
        phase: str = "runtime",
        completion_phase: str = "finished",
        initial_metrics: dict[str, float] | None = None,
        resume: bool = False,
    ) -> None:
        self.metrics = RolloutMetrics(rollout_id=rollout_id, attempt_id=attempt_id)
        self._rollout_id = rollout_id
        self._attempt_id = attempt_id
        self._completion_phase = completion_phase
        self._initial_metrics = dict(initial_metrics or {})
        self._path = output_path.with_name("metrics.json")
        self._interval = LIVE_METRICS_INTERVAL_SECONDS
        self._heartbeat = 0
        self._phase = phase
        self._stop = Event()
        self._write_lock = RLock()
        self._thread: Thread | None = None
        self._write_error_reported = False
        if resume:
            try:
                previous = json.loads(self._path.read_text(encoding="utf-8"))
                if (
                    isinstance(previous, dict)
                    and previous.get("rollout_id") == rollout_id
                    and previous.get("attempt_id") == attempt_id
                ):
                    heartbeat = previous.get("heartbeat")
                    if (
                        isinstance(heartbeat, int)
                        and not isinstance(heartbeat, bool)
                        and heartbeat >= 0
                    ):
                        self._heartbeat = heartbeat
            except (OSError, ValueError):
                pass

    async def __aenter__(self) -> Self:
        self._write_snapshot()
        self._thread = Thread(target=self._publish_periodically, daemon=True)
        self._thread.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.set_phase(self._completion_phase if exc_type is None else "failed")

    def set_phase(self, phase: str) -> None:
        with self._write_lock:
            self._phase = phase
            self._write_snapshot()

    def metrics_snapshot(self) -> dict[str, float]:
        """Preserve agent observations during the separate evaluation process."""
        with self._write_lock:
            snapshot = dict(self._initial_metrics)
            for name, value in self.metrics.snapshot().items():
                # A previous process or checkpoint cannot retain a live request.
                # Gauges may decrease; cumulative observations retain their maxima.
                if name in MODEL_REQUEST_GAUGES:
                    snapshot[name] = value
                else:
                    snapshot[name] = max(snapshot[name], value) if name in snapshot else value
            return snapshot

    def update_metrics(self, metrics: dict[str, float]) -> None:
        with self._write_lock:
            self._initial_metrics.update(metrics)
            self._write_snapshot()

    def _publish_periodically(self) -> None:
        # Tokenizer loading can block the runtime event loop. A separate thread
        # keeps this heartbeat independent of model and evaluator progress.
        while not self._stop.wait(self._interval):
            self._write_snapshot()

    def _write_snapshot(self) -> None:
        with self._write_lock:
            self._write_snapshot_locked()

    def _write_snapshot_locked(self) -> None:
        temporary_path: Path | None = None
        try:
            self._heartbeat += 1
            payload = {
                "rollout_id": self._rollout_id,
                "heartbeat": self._heartbeat,
                "phase": self._phase,
                "metrics": self.metrics_snapshot(),
            }
            if self._attempt_id is not None:
                payload["attempt_id"] = self._attempt_id
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # A private temporary file avoids exposing partial JSON to the driver
            # or making telemetry writable by the unprivileged agent process.
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent, delete=False
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(payload, stream, allow_nan=False)
            os.replace(temporary_path, self._path)
        except Exception:  # pylint: disable=broad-exception-caught
            if not self._write_error_reported:
                logger.warning("Could not publish rollout telemetry", exc_info=True)
                self._write_error_reported = True
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
