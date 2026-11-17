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

"""Tests for the shared, task-independent E2B Python bootstrap."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from northrl.remote_rollout import e2b_bootstrap as rollout


@pytest.fixture(autouse=True)
def trust_test_file_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rollout, "_ROOT_UID", os.getuid())


def test_ensure_runtime_bootstraps_once_with_uv_managed_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_install_root: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_python = runtime_root / "bin/python"
    runtime_marker = runtime_root / ".ready"
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    ensured_workspaces: list[Path] = []

    def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        calls.append((command, env.copy() if env is not None else None))
        if len(command) > 1 and command[1] == "venv":
            runtime_python.parent.mkdir(parents=True)
            runtime_python.touch()

    monkeypatch.setattr(rollout, "_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(rollout, "_RUNTIME_PYTHON", runtime_python)
    monkeypatch.setattr(rollout, "_RUNTIME_MARKER", runtime_marker)
    monkeypatch.setattr(
        rollout,
        "_ensure_uv",
        lambda workspace: ensured_workspaces.append(workspace) or "/bin/uv",
    )
    monkeypatch.setattr(rollout, "_run", run)
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "/untrusted/python")
    monkeypatch.setenv("NORTHGATE_API_KEY", "private")

    workspace = tmp_path / "workspace"
    assert rollout._ensure_runtime(workspace) == runtime_python
    assert runtime_marker.is_file()
    assert python_install_root.is_dir()
    assert calls[0][0] == [
        "/bin/uv",
        "venv",
        "--clear",
        "--managed-python",
        "--python",
        "3.12",
        str(runtime_root),
    ]
    assert calls[1][0] == [
        "/bin/uv",
        "--no-progress",
        "pip",
        "install",
        "--python",
        str(runtime_python),
        "--no-deps",
        "--index-url",
        rollout._PYPI_INDEX_URL,
        *rollout.SOURCE_WHEEL_REQUIREMENTS,
    ]
    assert calls[2][0][:8] == [
        "/bin/uv",
        "--no-progress",
        "pip",
        "install",
        "--python",
        str(runtime_python),
        "--only-binary",
        ":all:",
    ]
    assert calls[2][0][8:] == [
        "--index-url",
        rollout._PYPI_INDEX_URL,
        *rollout.RUNTIME_REQUIREMENTS,
    ]
    creation_environment = calls[0][1]
    assert creation_environment is not None
    assert creation_environment["UV_PYTHON_DOWNLOADS"] == "automatic"
    assert creation_environment["UV_PYTHON_INSTALL_DIR"] == str(python_install_root)
    for command, environment in calls:
        assert environment is not None
        assert "NORTHGATE_API_KEY" not in environment
        if "install" in command:
            assert environment["UV_PYTHON_DOWNLOADS"] == "never"
            assert command[command.index("--python") + 1] == str(runtime_python)
            assert command[command.index("--index-url") + 1] == rollout._PYPI_INDEX_URL
    assert calls[1][1] == calls[2][1]

    assert rollout._ensure_runtime(workspace) == runtime_python
    assert len(calls) == 3
    assert ensured_workspaces == [workspace]


@pytest.mark.parametrize("stale_runtime", ["legacy-marker", "changed-requirements"])
def test_runtime_rebuilds_stale_ready_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_install_root: Path,
    stale_runtime: str,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_python = runtime_root / "bin/python"
    runtime_marker = runtime_root / ".ready"
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> None:
        calls.append(command)
        if command[1] == "venv":
            runtime_python.parent.mkdir(parents=True, exist_ok=True)
            runtime_python.touch()

    monkeypatch.setattr(rollout, "_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(rollout, "_RUNTIME_PYTHON", runtime_python)
    monkeypatch.setattr(rollout, "_RUNTIME_MARKER", runtime_marker)
    monkeypatch.setattr(rollout, "_ensure_uv", lambda _workspace: "/trusted/uv")
    monkeypatch.setattr(rollout, "_run", run)

    workspace = tmp_path / "workspace"
    assert rollout._ensure_runtime(workspace) == runtime_python
    assert len(calls) == 3
    if stale_runtime == "legacy-marker":
        runtime_marker.write_text("")
    else:
        monkeypatch.setattr(
            rollout, "RUNTIME_REQUIREMENTS", (*rollout.RUNTIME_REQUIREMENTS, "packaging==25.0")
        )
    stale_marker = runtime_marker.read_text()

    assert rollout._ensure_runtime(workspace) == runtime_python
    assert len(calls) == 6
    assert calls[3][1] == "venv"
    assert runtime_marker.read_text() != stale_marker
    if stale_runtime == "changed-requirements":
        assert "packaging==25.0" in calls[5]

    assert rollout._ensure_runtime(workspace) == runtime_python
    assert len(calls) == 6


def test_installation_environment_removes_credentials_and_external_index_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("WEAVER_API_KEY", "weaver-secret")
    monkeypatch.setenv("NORTHGATE_API_KEY", "northgate-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-secret")
    monkeypatch.setenv("PIP_CONFIG_FILE", "/inherited/pip.conf")
    monkeypatch.setenv("UV_NO_CONFIG", "0")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "automatic")
    overrides = (
        "UV_INDEX",
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_CONFIG_FILE",
        "UV_PYTHON",
        "UV_PYTHON_INSTALL_DIR",
        "UV_PYTHON_INSTALL_MIRROR",
        "UV_PYTHON_DOWNLOADS_JSON_URL",
        "UV_PYTHON_PREFERENCE",
        "UV_MANAGED_PYTHON",
        "UV_NO_MANAGED_PYTHON",
    )
    for name in overrides:
        monkeypatch.setenv(name, "https://other.example/packages")

    environment = rollout._installation_environment()

    assert environment["PATH"] == rollout._SYSTEM_PATH
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert "WEAVER_API_KEY" not in environment
    assert "NORTHGATE_API_KEY" not in environment
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in environment
    assert "AWS_ACCESS_KEY_ID" not in environment
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["UV_NO_CONFIG"] == "1"
    assert environment["UV_PYTHON_DOWNLOADS"] == "never"
    for name in overrides:
        assert name not in environment


@pytest.fixture
def uv_install_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "managed-uv"
    monkeypatch.setattr(rollout, "_UV_INSTALL_ROOT", root)
    monkeypatch.setattr(rollout, "_UV_CANDIDATES", ())
    monkeypatch.setenv("PATH", "")
    return root


def test_resolve_uv_rejects_executable_inside_task_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
) -> None:
    workspace = tmp_path / "workspace"
    uv = workspace / ".local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.touch(mode=0o755)
    monkeypatch.setattr(rollout, "_UV_CANDIDATES", (uv,))
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("must not execute uv from the task workspace"),
    )

    with pytest.raises(FileNotFoundError, match="inside the task workspace"):
        rollout._resolve_uv(workspace)


def test_ensure_uv_reuses_existing_usable_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    uv = tmp_path / "trusted/uv"
    uv.parent.mkdir()
    uv.touch(mode=0o755)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="uv 0.12.17\n")

    monkeypatch.setattr(rollout, "_UV_CANDIDATES", (uv,))
    monkeypatch.setattr(rollout.subprocess, "run", run)
    monkeypatch.setattr(
        rollout, "_install_uv", lambda _workspace: pytest.fail("usable uv must be reused")
    )
    monkeypatch.setenv("WEAVER_API_KEY", "weaver-secret")

    assert rollout._ensure_uv(workspace) == str(uv)

    [(command, kwargs)] = calls
    assert command == [str(uv), "--version"]
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 10
    assert "WEAVER_API_KEY" not in kwargs["env"]
    assert not uv_install_root.exists()


@pytest.mark.parametrize("old_version", ["uv 0.7.13\n", "unexpected version output\n"])
def test_ensure_uv_upgrades_old_or_unrecognized_trusted_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
    old_version: str,
) -> None:
    old_uv = tmp_path / "trusted/uv"
    old_uv.parent.mkdir()
    old_uv.touch(mode=0o755)
    installed_uv = uv_install_root / "bin/uv"
    installations: list[Path] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output = old_version if command[0] == str(old_uv) else f"uv {rollout._UV_VERSION}\n"
        return subprocess.CompletedProcess(command, 0, stdout=output)

    def install(workspace: Path) -> None:
        installations.append(workspace)
        installed_uv.parent.mkdir(parents=True)
        installed_uv.touch(mode=0o755)

    monkeypatch.setattr(rollout, "_UV_CANDIDATES", (old_uv,))
    monkeypatch.setattr(rollout.subprocess, "run", run)
    monkeypatch.setattr(rollout, "_install_uv", install)
    workspace = tmp_path / "workspace"

    assert rollout._ensure_uv(workspace) == str(installed_uv)
    assert rollout._ensure_uv(workspace) == str(installed_uv)
    assert installations == [workspace]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "uv --version"),
        subprocess.TimeoutExpired("uv --version", 10),
        OSError("invalid executable"),
    ],
    ids=["nonzero-exit", "timeout", "exec-error"],
)
def test_resolve_uv_finds_path_executable_after_broken_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
    failure: Exception,
) -> None:
    workspace = tmp_path / "workspace"
    untrusted = workspace / "bin/uv"
    broken = tmp_path / "broken/uv"
    usable = tmp_path / "toolchain/bin/uv"
    for candidate in (untrusted, broken, usable):
        candidate.parent.mkdir(parents=True)
        candidate.touch(mode=0o755)
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == str(broken):
            raise failure
        assert command[0] == str(usable)
        return subprocess.CompletedProcess(command, 0, stdout="uv 0.12.17\n")

    monkeypatch.setattr(rollout, "_UV_CANDIDATES", (broken,))
    monkeypatch.setenv("PATH", os.pathsep.join([".", str(untrusted.parent), str(usable.parent)]))
    monkeypatch.setattr(rollout.subprocess, "run", run)

    assert rollout._resolve_uv(workspace) == str(usable)
    assert calls == [[str(broken), "--version"], [str(usable), "--version"]]


def test_resolve_uv_skips_executable_inside_runtime_to_be_recreated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_uv = runtime_root / "bin/uv"
    external_uv = tmp_path / "toolchain/bin/uv"
    for candidate in (runtime_uv, external_uv):
        candidate.parent.mkdir(parents=True)
        candidate.touch(mode=0o755)
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert command[0] == str(external_uv)
        return subprocess.CompletedProcess(command, 0, stdout="uv 0.12.17\n")

    monkeypatch.setattr(rollout, "_RUNTIME_ROOT", runtime_root)
    monkeypatch.setenv("PATH", os.pathsep.join([str(runtime_uv.parent), str(external_uv.parent)]))
    monkeypatch.setattr(rollout.subprocess, "run", run)

    assert rollout._resolve_uv(tmp_path / "workspace") == str(external_uv)
    assert calls == [[str(external_uv), "--version"]]


def test_ensure_uv_installs_missing_executable_once_then_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
) -> None:
    workspace = tmp_path / "workspace"
    uv = uv_install_root / "bin/uv"
    installations: list[Path] = []

    def install(requested_workspace: Path) -> None:
        installations.append(requested_workspace)
        uv.parent.mkdir(parents=True)
        uv.touch(mode=0o755)

    monkeypatch.setattr(rollout, "_install_uv", install)
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout="uv 0.12.17\n"),
    )

    assert rollout._ensure_uv(workspace) == str(uv)
    assert rollout._ensure_uv(workspace) == str(uv)
    assert installations == [workspace]


@pytest.mark.parametrize("installation_fails", [True, False], ids=["install-error", "no-binary"])
def test_ensure_uv_reports_failed_or_invalid_installation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
    installation_fails: bool,
) -> None:
    workspace = tmp_path / "workspace"
    installations: list[Path] = []

    def install(requested_workspace: Path) -> None:
        installations.append(requested_workspace)
        if installation_fails:
            raise OSError("download failed")

    monkeypatch.setattr(rollout, "_install_uv", install)
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("no executable is available"),
    )

    with pytest.raises(RuntimeError, match="uv") as failure:
        rollout._ensure_uv(workspace)

    assert failure.value.__cause__ is not None
    assert installations == [workspace]


def _mock_uv_wheel_download(
    monkeypatch: pytest.MonkeyPatch,
    *,
    machine: str = "x86_64",
    binary: bytes = b"\x7fELFuv-native-binary",
    checksum: str | None = None,
    wheel_url: str = "https://files.pythonhosted.org/packages/uv.whl",
) -> list[tuple[str, int]]:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(f"uv-{rollout._UV_VERSION}.data/scripts/uv", binary)
        archive.writestr(f"uv-{rollout._UV_VERSION}.data/scripts/uvx", b"unused uvx")
        archive.writestr("uv/__init__.py", "raise RuntimeError('must not import this')")
        archive.writestr("../../escaped", b"must not extract arbitrary archive members")
    payload = archive_bytes.getvalue()
    metadata = {
        "urls": [
            {"filename": "uv-unrelated-windows.whl"},
            {
                "filename": (
                    f"uv-{rollout._UV_VERSION}-py3-none-manylinux_2_17_{machine}."
                    f"manylinux2014_{machine}.whl"
                ),
                "url": wheel_url,
                "digests": {"sha256": checksum or hashlib.sha256(payload).hexdigest()},
            },
        ]
    }
    calls: list[tuple[str, int]] = []

    def download(url: str, *, timeout: int) -> io.BytesIO:
        calls.append((url, timeout))
        if url == f"https://pypi.org/pypi/uv/{rollout._UV_VERSION}/json":
            return io.BytesIO(json.dumps(metadata).encode())
        assert url == wheel_url
        return io.BytesIO(payload)

    monkeypatch.setattr(rollout.platform, "machine", lambda: machine)
    monkeypatch.setattr(rollout.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(rollout.sys, "platform", "linux")
    monkeypatch.setattr(rollout, "urlopen", download)
    return calls


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_install_uv_downloads_native_binary_without_pip_or_ensurepip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
    machine: str,
) -> None:
    binary = b"\x7fELFuv-native-binary"
    downloads = _mock_uv_wheel_download(monkeypatch, machine=machine, binary=binary)
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("uv bootstrap must not require pip or ensurepip"),
    )

    rollout._install_uv(tmp_path / "workspace")

    installed = uv_install_root / "bin/uv"
    assert installed.read_bytes() == binary
    assert installed.stat().st_mode & 0o777 == 0o755
    assert downloads == [
        (f"https://pypi.org/pypi/uv/{rollout._UV_VERSION}/json", 30),
        ("https://files.pythonhosted.org/packages/uv.whl", 120),
    ]
    assert list(uv_install_root.iterdir()) == [uv_install_root / "bin"]
    assert list(installed.parent.iterdir()) == [installed]
    assert not (tmp_path / "escaped").exists()


def test_install_uv_checksum_failure_preserves_existing_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
) -> None:
    installed = uv_install_root / "bin/uv"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"previous executable")
    installed.chmod(0o755)
    _mock_uv_wheel_download(monkeypatch, checksum="0" * 64)

    with pytest.raises(RuntimeError, match="SHA256 verification failed"):
        rollout._install_uv(tmp_path / "workspace")

    assert installed.read_bytes() == b"previous executable"
    assert list(uv_install_root.iterdir()) == [uv_install_root / "bin"]


@pytest.mark.parametrize(
    "wheel_url", ["http://files.pythonhosted.org/uv.whl", "https://untrusted.example/uv.whl"]
)
def test_install_uv_rejects_unofficial_download_before_fetching_wheel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
    wheel_url: str,
) -> None:
    downloads = _mock_uv_wheel_download(monkeypatch, wheel_url=wheel_url)

    with pytest.raises(RuntimeError, match="official PyPI storage"):
        rollout._install_uv(tmp_path / "workspace")

    assert len(downloads) == 1
    assert not (uv_install_root / "bin/uv").exists()


def test_install_uv_rejects_non_native_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
) -> None:
    _mock_uv_wheel_download(monkeypatch, binary=b"#!/usr/bin/python3\nimport uv")

    with pytest.raises(RuntimeError, match="does not contain a native executable"):
        rollout._install_uv(tmp_path / "workspace")

    assert not (uv_install_root / "bin/uv").exists()
    assert list(uv_install_root.iterdir()) == [uv_install_root / "bin"]


def test_install_uv_rejects_destination_inside_task_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_install_root: Path,
) -> None:
    monkeypatch.setattr(
        rollout,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("must reject task workspace before downloading"),
    )

    with pytest.raises(RuntimeError, match="workspace"):
        rollout._install_uv(tmp_path)


@pytest.fixture
def python_install_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "managed-python"
    monkeypatch.setattr(rollout, "_PYTHON_INSTALL_ROOT", root)
    return root


def test_runtime_rejects_python_installation_directory_inside_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    install_root = workspace / "managed-python"
    runtime_root = tmp_path / "runtime"
    runtime_marker = runtime_root / ".ready"
    monkeypatch.setattr(rollout, "_PYTHON_INSTALL_ROOT", install_root)
    monkeypatch.setattr(rollout, "_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(rollout, "_RUNTIME_PYTHON", runtime_root / "bin/python")
    monkeypatch.setattr(rollout, "_RUNTIME_MARKER", runtime_marker)
    monkeypatch.setattr(rollout, "_ensure_uv", lambda _workspace: "/trusted/uv")
    monkeypatch.setattr(
        rollout,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("workspace Python directory must not be used"),
    )

    with pytest.raises(RuntimeError, match="inside the task workspace"):
        rollout._ensure_runtime(workspace)

    assert not install_root.exists()
    assert not runtime_marker.exists()


@pytest.mark.parametrize("invalid_directory", ["writable", "symlink", "wrong-owner"])
def test_runtime_rejects_untrusted_python_installation_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_install_root: Path,
    invalid_directory: str,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_marker = runtime_root / ".ready"
    if invalid_directory == "symlink":
        target = tmp_path / "elsewhere"
        target.mkdir()
        python_install_root.symlink_to(target, target_is_directory=True)
    else:
        python_install_root.mkdir()
        if invalid_directory == "writable":
            python_install_root.chmod(0o777)
        else:
            monkeypatch.setattr(rollout, "_ROOT_UID", os.getuid() + 1)
    monkeypatch.setattr(rollout, "_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(rollout, "_RUNTIME_PYTHON", runtime_root / "bin/python")
    monkeypatch.setattr(rollout, "_RUNTIME_MARKER", runtime_marker)
    monkeypatch.setattr(rollout, "_ensure_uv", lambda _workspace: "/trusted/uv")
    monkeypatch.setattr(
        rollout,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("untrusted Python directory must not be used"),
    )

    with pytest.raises((RuntimeError, PermissionError), match="Python installation"):
        rollout._ensure_runtime(tmp_path / "workspace")

    assert not runtime_marker.exists()


@pytest.mark.parametrize("failure_stage", ["uv", "venv", "source-package", "dependencies"])
def test_runtime_failure_does_not_publish_ready_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_install_root: Path,
    failure_stage: str,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_python = runtime_root / "bin/python"
    runtime_marker = runtime_root / ".ready"
    calls: list[list[str]] = []

    def ensure_uv(_workspace: Path) -> str:
        if failure_stage == "uv":
            raise RuntimeError("uv download failed")
        return "/trusted/uv"

    def run(command: list[str], **_kwargs: Any) -> None:
        calls.append(command)
        stage = ["venv", "source-package", "dependencies"][len(calls) - 1]
        if stage == failure_stage:
            raise subprocess.CalledProcessError(1, command)
        if stage == "venv":
            runtime_python.parent.mkdir(parents=True)
            runtime_python.touch()

    monkeypatch.setattr(rollout, "_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(rollout, "_RUNTIME_PYTHON", runtime_python)
    monkeypatch.setattr(rollout, "_RUNTIME_MARKER", runtime_marker)
    monkeypatch.setattr(rollout, "_ensure_uv", ensure_uv)
    monkeypatch.setattr(rollout, "_run", run)

    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        rollout._ensure_runtime(tmp_path / "workspace")

    assert not runtime_marker.exists()
    assert len(calls) == ["uv", "venv", "source-package", "dependencies"].index(failure_stage)


def test_bootstrap_helper_loads_without_site_packages():
    helper = Path(rollout.__file__).resolve()
    code = (
        "import runpy, sys; "
        f"module = runpy.run_path({str(helper)!r}); "
        "assert callable(module['_ensure_runtime']); "
        "assert not any(name.startswith('northrl.swerebench') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sandbox_modules_import_without_training_or_tokenizer_dependencies():
    repository = Path(__file__).resolve().parents[3]
    code = f"""
import importlib.abc
import runpy
import sys

sys.path.insert(0, {str(repository)!r})
blocked = {{
    "weaver", "torch", "transformers", "sglang", "openai", "tokenizers", "aiohttp",
    "IPython", "numpy", "requests", "tqdm",
}}

class WithoutTrainingStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked or fullname.startswith("northrl.weaver"):
            raise ImportError("sandbox dependency removed: " + fullname)
        return None

sys.meta_path.insert(0, WithoutTrainingStack())
import northrl.remote_rollout.e2b_runtime
import northrl.rollout_entrypoints.swerebench_claude_code_rollout
runpy.run_path({str(repository / "recipe/e2b_minimal/rollout.py")!r})
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_shell_bootstrap_hands_off_to_common_helper(tmp_path):
    bootstrap = Path(rollout.__file__).with_suffix(".sh")
    shell = tmp_path / "e2b_bootstrap.sh"
    root = tmp_path / "bootstrap"
    shell.write_text(
        bootstrap.read_text().replace(
            "_BOOTSTRAP_ROOT=/opt/northrl-bootstrap", f"_BOOTSTRAP_ROOT={root}"
        )
    )
    marker = tmp_path / "helper-called"
    helper = tmp_path / "e2b_bootstrap.py"
    helper.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ready')\n")
    result = subprocess.run(["/bin/sh", str(shell)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "ready"
    assert (root / "python").is_symlink()
