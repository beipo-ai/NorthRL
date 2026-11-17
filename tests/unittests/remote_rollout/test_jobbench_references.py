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

"""Test pinned per-task references without mounting a volume or using E2B."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from northrl.remote_rollout.contracts import RolloutEnvironmentError
from northrl.rollout_entrypoints.jobbench import references


@pytest.fixture
def case(tmp_path, monkeypatch):
    mount = tmp_path / "mount"
    root = mount / "releases/v1"
    key = "package-123/accountants/task1"
    task = {"task_id": "package-123", "task_key": key}
    relative_directory = f"references/{key}/files_required_to_search"
    selected = root / relative_directory
    selected.mkdir(parents=True)
    data = b"Official reference data\n"
    file = selected / "reference.txt"
    file.write_bytes(data)
    manifest = {
        "schema_version": "jobbench-external-references-v1",
        "tasks": [{**task, "file_count": 1, "reference_dir": relative_directory}],
        "files": [
            {
                "path": relative_directory + "/reference.txt",
                "task_key": key,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }

    def save_manifest():
        raw = json.dumps(manifest).encode()
        (root / "manifest.json").write_bytes(raw)
        monkeypatch.setenv("JOBBENCH_REFERENCES_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())

    save_manifest()
    monkeypatch.setenv("JOBBENCH_REFERENCES_ROOT", str(root))
    monkeypatch.setattr(references, "_MOUNT_ROOT", mount)
    monkeypatch.setattr(references.os.path, "ismount", lambda path: path == mount)
    monkeypatch.setattr(references.os, "statvfs", lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    command = AsyncMock()
    return SimpleNamespace(
        mount=mount,
        root=root,
        selected=selected,
        file=file,
        data=data,
        task=task,
        manifest=manifest,
        save_manifest=save_manifest,
        command=command,
        workspace=tmp_path / "workspace",
    )


def prepare(case):
    return asyncio.run(references.prepare_references(case.task, case.workspace, case.command))


def test_returns_current_references_after_sealing_the_entire_volume(case):
    assert prepare(case) == case.selected
    assert case.command.await_args_list == [
        call(["mount", "--bind", str(case.mount), str(case.mount)]),
        call(["mount", "-o", "remount,bind,ro", str(case.mount)]),
    ]
    assert case.file.read_bytes() == case.data
    assert not case.workspace.exists()


def test_unconfigured_references_preserve_the_existing_runtime(case, monkeypatch):
    monkeypatch.delenv("JOBBENCH_REFERENCES_ROOT")
    monkeypatch.delenv("JOBBENCH_REFERENCES_MANIFEST_SHA256")
    assert prepare(case) is None
    case.command.assert_not_awaited()


@pytest.mark.parametrize(
    "missing", ["JOBBENCH_REFERENCES_ROOT", "JOBBENCH_REFERENCES_MANIFEST_SHA256"]
)
def test_partial_configuration_fails_closed(case, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)
    case.command.assert_not_awaited()


@pytest.mark.parametrize("root", ["relative", "/outside/release", "{mount}", "{mount}/x/../v1"])
def test_reference_root_must_be_normalized_and_below_the_mount(case, monkeypatch, root):
    monkeypatch.setenv("JOBBENCH_REFERENCES_ROOT", root.format(mount=case.mount))
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)
    case.command.assert_not_awaited()


def test_missing_release_is_an_environment_error(case, monkeypatch):
    monkeypatch.setenv("JOBBENCH_REFERENCES_ROOT", str(case.mount / "missing"))
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)


@pytest.mark.parametrize("kind", ["missing_mount", "mount_failure", "still_writable"])
def test_readonly_mount_is_required_before_exposing_references(case, monkeypatch, kind):
    if kind == "missing_mount":
        monkeypatch.setattr(references.os.path, "ismount", lambda _: False)
    elif kind == "mount_failure":
        case.command.side_effect = RolloutEnvironmentError("mount failed")
    else:
        monkeypatch.setattr(references.os, "statvfs", lambda _: SimpleNamespace(f_flag=0))
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)


def test_manifest_bytes_are_pinned_independently_of_the_file_hashes(case):
    with (case.root / "manifest.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(RolloutEnvironmentError, match="manifest SHA256"):
        prepare(case)


@pytest.mark.parametrize(
    "change",
    ["schema", "missing_task", "duplicate_task", "task_key", "task_id", "reference_dir", "count"],
)
def test_selected_task_identity_and_manifest_shape_are_checked(case, change):
    selected = case.manifest["tasks"][0]
    if change == "schema":
        case.manifest["schema_version"] = "other"
    elif change == "missing_task":
        case.manifest["tasks"] = []
    elif change == "duplicate_task":
        case.manifest["tasks"].append(dict(selected))
    elif change == "task_key":
        selected["task_key"] = "package-456/accountants/task1"
    elif change == "task_id":
        selected["task_id"] = "package-456"
    elif change == "reference_dir":
        selected["reference_dir"] = "references/another-task"
    else:
        selected["file_count"] = 2
    case.save_manifest()
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)


@pytest.mark.parametrize(
    "change", ["escape", "absolute", "duplicate", "wrong_task", "bad_digest", "bad_size"]
)
def test_selected_file_records_are_validated(case, change):
    record = case.manifest["files"][0]
    if change == "escape":
        record["path"] = str(Path(record["path"]).parent) + "/../reference.txt"
    elif change == "absolute":
        record["path"] = "/etc/passwd"
    elif change == "duplicate":
        case.manifest["files"].append(dict(record))
    elif change == "wrong_task":
        record["task_key"] = "package-456/accountants/task1"
    elif change == "bad_digest":
        record["sha256"] = "invalid"
    else:
        record["bytes"] = True
    case.save_manifest()
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "size",
        "missing",
        "extra_file",
        "extra_directory",
        "file_symlink",
        "directory_symlink",
        "hardlink",
    ],
)
def test_current_reference_tree_must_match_the_manifest(case, change):
    if change == "hash":
        case.file.write_bytes(b"x" * len(case.data))
    elif change == "size":
        case.file.write_bytes(b"x")
    elif change == "missing":
        case.file.unlink()
    elif change == "extra_file":
        (case.selected / "extra.txt").write_text("unlisted")
    elif change == "extra_directory":
        (case.selected / "extra").mkdir()
    elif change == "file_symlink":
        case.file.unlink()
        case.file.symlink_to(case.root / "manifest.json")
    elif change == "directory_symlink":
        original = case.selected.with_name("original")
        case.selected.rename(original)
        case.selected.symlink_to(original, target_is_directory=True)
    else:
        os.link(case.file, case.root / "alias")
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)


def test_zero_reference_tasks_are_supported_without_ignoring_unexpected_files(case):
    case.manifest["tasks"][0].update(file_count=0, reference_dir=None)
    case.manifest["files"] = []
    case.save_manifest()
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)
    case.file.unlink()
    case.selected.rmdir()
    assert prepare(case) is None


def test_other_tasks_payloads_are_not_required_or_read(case):
    # This unrelated row deliberately points to a nonexistent, mismatched payload.
    other_key = "package-456/accountants/task1"
    case.manifest["tasks"].append({"task_id": "package-456", "task_key": other_key})
    case.manifest["files"].append(
        {
            "task_key": other_key,
            "path": f"references/{other_key}/files_required_to_search/missing.txt",
            "bytes": 900000000,
            "sha256": "bad-unrelated-digest",
        }
    )
    case.save_manifest()
    assert prepare(case) == case.selected


def test_symlink_release_root_is_rejected_before_mounting(case, monkeypatch):
    link = case.mount / "linked-release"
    link.symlink_to(case.root, target_is_directory=True)
    monkeypatch.setenv("JOBBENCH_REFERENCES_ROOT", str(link))
    with pytest.raises(RolloutEnvironmentError):
        prepare(case)
    case.command.assert_not_awaited()
