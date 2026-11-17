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

"""Expose one task's pinned reference files from a read-only sandbox mount."""

import asyncio
import hashlib
import json
import os
import re
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any

from northrl.remote_rollout.contracts import RolloutEnvironmentError

_MOUNT_ROOT = Path("/mnt/data")
_SCHEMA = "jobbench-external-references-v1"
_MANIFEST_LIMIT = 8 * 1024 * 1024
_CHUNK = 4 * 1024 * 1024


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Reference path must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or ".." in path.parts or value == ".":
        raise ValueError("Reference path is not normalized")
    return value


def _directories(path: Path) -> None:
    for item in (*reversed(path.parents), path):
        if not stat.S_ISDIR(item.lstat().st_mode):
            raise ValueError("Reference directory is missing, redirects, or is not a directory")


def _regular_file(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Reference must be a regular file without additional links")
    return info


def _read_only(path: Path) -> None:
    if not os.statvfs(path).f_flag & os.ST_RDONLY:
        raise ValueError("JobBench reference volume is not mounted read-only")


def _unique_object(items: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Reference manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _select_references(root: Path, task: dict, expected_manifest_sha: str) -> Path | None:
    """Validate the selected task; payloads for other tasks are never opened."""
    path = root / "manifest.json"
    if _regular_file(path).st_size > _MANIFEST_LIMIT:
        raise ValueError("Reference manifest exceeds its size limit")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha:
        raise ValueError("Reference manifest SHA256 mismatch")
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA:
        raise ValueError("Unsupported reference manifest schema")
    tasks, files = manifest.get("tasks"), manifest.get("files")
    if not isinstance(tasks, list) or not isinstance(files, list):
        raise ValueError("Reference manifest requires task and file lists")
    key = _relative(task.get("task_key"))
    task_id = task.get("task_id")
    if not re.fullmatch(r"package-[0-9]+/[A-Za-z0-9_-]+/task[0-9]+", key):
        raise ValueError("Invalid JobBench task key")
    if task_id != key.split("/")[0]:
        raise ValueError("Reference task ID and key do not match")
    matches = [
        row
        for row in tasks
        if isinstance(row, dict) and (row.get("task_id") == task_id or row.get("task_key") == key)
    ]
    if len(matches) != 1:
        raise ValueError("Reference manifest must identify this task exactly once")
    selected = matches[0]
    if selected.get("task_id") != task_id or selected.get("task_key") != key:
        raise ValueError("Reference manifest task identity mismatch")
    count = selected.get("file_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Invalid reference file count")
    relative_directory = f"references/{key}/files_required_to_search"
    if selected.get("reference_dir") != (relative_directory if count else None):
        raise ValueError("Reference directory does not match the selected task")
    prefix = relative_directory + "/"
    records = {}
    for record in files:
        if not isinstance(record, dict):
            continue
        name = record.get("path")
        if record.get("task_key") != key and not (
            isinstance(name, str) and name.startswith(prefix)
        ):
            continue
        name = _relative(name)
        if record.get("task_key") != key or not name.startswith(prefix):
            raise ValueError("Reference file escapes its selected task")
        relative = _relative(name[len(prefix) :])
        size, digest = record.get("bytes"), record.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Invalid reference file size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid reference file digest")
        if relative in records:
            raise ValueError("Duplicate selected reference file")
        records[relative] = (size, digest)
    if len(records) != count:
        raise ValueError("Selected reference file count mismatch")
    directory = root / relative_directory
    if not count:
        if os.path.lexists(directory):
            raise ValueError("Unexpected files for a task without references")
        return None
    _directories(directory)
    _read_only(directory)
    expected_dirs = {"."}
    for relative in records:
        expected_dirs.update(str(parent) for parent in PurePosixPath(relative).parents)
    actual_dirs, actual_files = set(), set()
    pending = [directory]
    while pending:
        current = pending.pop()
        actual_dirs.add(str(current.relative_to(directory)))
        with os.scandir(current) as entries:
            for entry in entries:
                child = current / entry.name
                if entry.is_dir(follow_symlinks=False):
                    relative = str(child.relative_to(directory))
                    if relative not in expected_dirs:
                        raise ValueError("Unexpected selected reference directory")
                    pending.append(child)
                    continue
                relative = str(child.relative_to(directory))
                if relative not in records:
                    raise ValueError("Unexpected selected reference file")
                size, _ = records[relative]
                if _regular_file(child).st_size != size:
                    raise ValueError("Reference file size mismatch")
                actual_files.add(relative)
    if actual_dirs != expected_dirs or actual_files != set(records):
        raise ValueError("Selected reference directory is incomplete")
    for relative, (_, expected_digest) in records.items():
        digest = hashlib.sha256()
        with (directory / relative).open("rb") as stream:
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError("Reference file SHA256 mismatch")
    return directory


async def prepare_references(
    task: dict,
    workspace: Path,
    command: Callable[..., Awaitable[Any]],
) -> Path | None:
    """Seal the shared mount, validate this task, and return its reference path.

    Both environment variables are optional together. Their configured values
    must identify an existing release and its exact manifest. The workspace is
    unchanged: references are read directly from their verified shared path.
    """
    del workspace
    configured_root = os.environ.get("JOBBENCH_REFERENCES_ROOT", "")
    configured_sha = os.environ.get("JOBBENCH_REFERENCES_MANIFEST_SHA256", "")
    if not configured_root and not configured_sha:
        return None
    try:
        if not configured_root or not re.fullmatch(r"[0-9a-f]{64}", configured_sha):
            raise ValueError("Configure both JobBench reference root and manifest SHA256")
        root = Path(configured_root)
        if (
            not root.is_absolute()
            or str(root) != configured_root
            or ".." in root.parts
            or root == _MOUNT_ROOT
            or not root.is_relative_to(_MOUNT_ROOT)
        ):
            raise ValueError("Reference root must be a normalized directory below /mnt/data")
        _directories(root)
        if not os.path.ismount(_MOUNT_ROOT):
            raise ValueError("JobBench reference volume is not mounted at /mnt/data")
        # chmod is insufficient on the shared NFS service. Seal the entire mount
        # so the agent cannot reach a writable alias, archive, or another release.
        await command(["mount", "--bind", str(_MOUNT_ROOT), str(_MOUNT_ROOT)])
        await command(["mount", "-o", "remount,bind,ro", str(_MOUNT_ROOT)])
        _read_only(_MOUNT_ROOT)
        return await asyncio.to_thread(_select_references, root, task, configured_sha)
    except (OSError, ValueError, TypeError) as exc:
        raise RolloutEnvironmentError(f"JobBench references could not be prepared: {exc}") from exc
