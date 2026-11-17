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

"""Validate host-selected E2B volume mounts without importing the E2B SDK."""

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


def validate_volume_mounts(
    value: Any, *, location: str = "rollout.execution.volume_mounts"
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must map absolute sandbox paths to volume names")
    mounts = {}
    path_error = f"{location} requires normalized absolute non-root mount paths"
    for path, volume in value.items():
        if not isinstance(path, str):
            raise ValueError(path_error)
        normalized = PurePosixPath(path)
        if not normalized.is_absolute() or path.startswith("//") or path == "/" or "\x00" in path:
            raise ValueError(path_error)
        if path != str(normalized) or ".." in normalized.parts:
            raise ValueError(path_error)
        if not isinstance(volume, str) or not volume.strip() or volume != volume.strip():
            raise ValueError(
                f"{location} requires non-empty volume names without boundary whitespace"
            )
        mounts[path] = volume
    return mounts
