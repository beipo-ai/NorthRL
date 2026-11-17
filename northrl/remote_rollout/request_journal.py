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

"""Small request records kept separately from token trajectories and live snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


class RequestJournal:
    """Append one private JSONL record per finished generation call."""

    def __init__(self, path: Path, rollout_id: str, attempt_id: str | None) -> None:
        self._path = path
        self._rollout_id = rollout_id
        self._attempt_id = attempt_id
        self._lock = Lock()

    def __call__(self, record: dict[str, Any]) -> None:
        # The metrics observer catches I/O errors; telemetry must not fail inference.
        payload = {**record, "rollout_id": self._rollout_id, "attempt_id": self._attempt_id}
        line = json.dumps(payload, allow_nan=False) + "\n"
        with self._lock:
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(line)
