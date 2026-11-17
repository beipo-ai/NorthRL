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

"""Tests for preparing the SWE-rebench Claude Code rollout environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from northrl.rollout_entrypoints import swerebench_claude_code_rollout as rollout


@pytest.fixture(autouse=True)
def trust_test_file_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rollout, "_ROOT_UID", os.getuid())


def test_trusted_runtime_path_keeps_toolchains_outside_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace_bin = workspace / "bin"
    workspace_bin.mkdir(parents=True)
    toolchain_bin = tmp_path / "toolchain/bin"
    toolchain_bin.mkdir(parents=True)
    claude_bin = tmp_path / "claude/bin"
    claude_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", f"{workspace_bin}:{toolchain_bin}")

    path_entries = rollout._trusted_runtime_path(workspace, claude_bin).split(os.pathsep)

    assert path_entries[0] == str(claude_bin)
    assert str(toolchain_bin) in path_entries
    assert str(workspace_bin) not in path_entries


def test_trusted_runtime_path_resolves_and_deduplicates_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trusted_bin = tmp_path / "usr/bin"
    trusted_bin.mkdir(parents=True)
    linked_bin = tmp_path / "bin"
    linked_bin.symlink_to(trusted_bin, target_is_directory=True)
    monkeypatch.setattr(rollout, "_SYSTEM_PATH", "")
    monkeypatch.setenv("PATH", f"{linked_bin}:{trusted_bin}")

    path_entries = rollout._trusted_runtime_path(workspace, linked_bin).split(os.pathsep)

    assert path_entries == [str(trusted_bin)]


@pytest.mark.parametrize("repository_name", ["wtforms", "testbed"])
def test_resolve_workspace_returns_the_checkout_without_creating_an_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_name: str,
) -> None:
    repository = tmp_path / repository_name
    git_dir = repository / ".git"
    git_dir.mkdir(parents=True)

    def glob(path: Path, pattern: str) -> list[Path]:
        assert path == Path("/")
        assert pattern == "*/.git"
        return [git_dir]

    monkeypatch.setattr(Path, "glob", glob)

    assert rollout._resolve_workspace() == repository.resolve()
    assert list(tmp_path.iterdir()) == [repository]
    assert not repository.is_symlink()


@pytest.mark.parametrize("checkout_count", [0, 2])
def test_resolve_workspace_requires_one_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkout_count: int,
) -> None:
    git_dirs = [tmp_path / f"repository-{index}" / ".git" for index in range(checkout_count)]
    for git_dir in git_dirs:
        git_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "glob", lambda _path, _pattern: git_dirs)

    with pytest.raises(RuntimeError, match="expected exactly one top-level task checkout"):
        rollout._resolve_workspace()


def test_resolve_workspace_accepts_existing_aliases_of_the_same_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "wtforms"
    (repository / ".git").mkdir(parents=True)
    alias = tmp_path / "existing-alias"
    alias.symlink_to(repository, target_is_directory=True)
    monkeypatch.setattr(Path, "glob", lambda _path, _pattern: [repository / ".git", alias / ".git"])

    assert rollout._resolve_workspace() == repository.resolve()


def test_ensure_agent_user_never_dereferences_repository_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    git_dir = workspace / ".git"
    git_dir.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    (workspace / "external").symlink_to(protected, target_is_directory=True)
    calls: list[list[str]] = []

    monkeypatch.setattr(rollout.pwd, "getpwnam", lambda _name: object())
    monkeypatch.setattr(rollout, "_run", lambda command, **_kwargs: calls.append(command))

    rollout._ensure_agent_user(workspace)

    recursive_chowns = [command for command in calls if command[:2] == ["chown", "--recursive"]]
    assert recursive_chowns
    assert all("--no-dereference" in command for command in recursive_chowns)
    assert all("--" in command for command in recursive_chowns)


def test_prepare_claude_copies_root_owned_node_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    node_root = tmp_path / "root/.nvm/versions/node/v22"
    node_bin = node_root / "bin"
    package = node_root / "lib/node_modules/@anthropic-ai/claude-code"
    node_bin.mkdir(parents=True)
    package.mkdir(parents=True)
    node = node_bin / "node"
    node.write_bytes(b"node")
    node.chmod(0o700)
    cli = package / "cli.js"
    cli.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    cli.chmod(0o600)
    vendor_binary = package / "vendor-tool"
    vendor_binary.write_bytes(b"tool")
    vendor_binary.chmod(0o700)
    source_launcher = node_bin / "claude"
    source_launcher.symlink_to("../lib/node_modules/@anthropic-ai/claude-code/cli.js")

    public_bin = tmp_path / "usr/local/bin"
    public_bin.mkdir(parents=True)
    public_claude = public_bin / "claude"
    public_claude.symlink_to(source_launcher)
    runtime_root = tmp_path / "opt/northrl-claude"

    monkeypatch.setattr(rollout, "_CLAUDE", public_claude)
    monkeypatch.setattr(rollout, "_CLAUDE_RUNTIME_ROOT", runtime_root)
    assert rollout._prepare_claude() == runtime_root / "bin"
    copied_launcher = runtime_root / "bin/claude"
    assert public_claude.resolve() == copied_launcher
    assert (runtime_root / "bin/node").stat().st_mode & 0o001
    assert (runtime_root / "package/cli.js").stat().st_mode & 0o001
    assert (runtime_root / "package/vendor-tool").stat().st_mode & 0o001
    launcher = copied_launcher.read_text(encoding="utf-8")
    assert str(runtime_root / "bin/node") in launcher
    assert str(runtime_root / "package/cli.js") in launcher


def test_prepare_claude_runs_native_binary_without_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "root/.local/share/claude/versions/1.0.0"
    package.mkdir(parents=True)
    native_cli = package / "claude.exe"
    native_cli.write_bytes(b"\x7fELF\x02\x01\x01\x00")
    native_cli.chmod(0o700)

    public_bin = tmp_path / "usr/local/bin"
    public_bin.mkdir(parents=True)
    public_claude = public_bin / "claude"
    public_claude.symlink_to(native_cli)
    runtime_root = tmp_path / "opt/northrl-claude"

    monkeypatch.setattr(rollout, "_CLAUDE", public_claude)
    monkeypatch.setattr(rollout, "_CLAUDE_RUNTIME_ROOT", runtime_root)

    assert rollout._prepare_claude() == runtime_root / "bin"
    copied_launcher = runtime_root / "bin/claude"
    copied_entrypoint = runtime_root / "package/claude.exe"
    assert public_claude.resolve() == copied_launcher
    assert copied_entrypoint.read_bytes() == b"\x7fELF\x02\x01\x01\x00"
    assert copied_entrypoint.stat().st_mode & 0o001
    assert not (runtime_root / "bin/node").exists()
    assert copied_launcher.read_text(encoding="utf-8") == (
        "#!/bin/sh\n" f'exec "{copied_entrypoint}" "$@"\n'
    )
