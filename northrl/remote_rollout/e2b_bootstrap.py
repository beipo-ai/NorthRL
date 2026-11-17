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

"""Install the shared Python environment used by E2B task hooks.

Invoked by e2b_bootstrap.sh with its standard-library-only interpreter. This
module deliberately does not import a task, a gateway, or third-party packages.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from urllib.request import urlopen

_PYTHON_VERSION = "3.12"
_PYTHON_INSTALL_ROOT = Path("/opt/northrl-python")
_RUNTIME_ROOT = Path("/opt/northrl-runtime")
_RUNTIME_PYTHON = _RUNTIME_ROOT / "bin/python"
_RUNTIME_MARKER = _RUNTIME_ROOT / ".northrl-runtime-ready"
_ELF_MAGIC = b"\x7fELF"
_ROOT_UID = 0
_UV_VERSION = "0.12.17"
_PYPI_INDEX_URL = "https://pypi.org/simple"
_UV_INSTALL_ROOT = Path("/opt/northrl-uv")
_UV_CANDIDATES = (
    Path("/usr/local/bin/uv"),
    Path("/usr/bin/uv"),
    Path("/root/.local/bin/uv"),
    Path("/workspace/.local/bin/uv"),
)
_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RUNTIME_REQUIREMENTS = (
    "fastapi==0.116.1",
    "httpx==0.28.1",
    "omegaconf==2.3.0",
    "proxy.py==2.4.10",
    "pydantic==2.11.7",
    "uvicorn==0.35.0",
)
SOURCE_WHEEL_REQUIREMENTS = ("antlr4-python3-runtime==4.9.3",)
_INSTALL_SECRET_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)
_INSTALL_SECRET_ENV_SUFFIXES = (
    "_API_KEY",
    "_AUTH_TOKEN",
    "_CREDENTIALS",
    "_OAUTH_TOKEN",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_SESSION_TOKEN",
    "_TOKEN",
)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    command_environment = os.environ.copy() if env is None else env.copy()
    command_environment["PATH"] = _SYSTEM_PATH
    subprocess.run(command, check=True, env=command_environment)


def _installation_environment() -> dict[str, str]:
    """Return the ambient environment without credentials needed only by the rollout."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() not in _INSTALL_SECRET_ENV_NAMES
        and not name.upper().endswith(_INSTALL_SECRET_ENV_SUFFIXES)
    }
    environment["PATH"] = _SYSTEM_PATH
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["UV_NO_CONFIG"] = "1"
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    for name in (
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
    ):
        environment.pop(name, None)
    return environment


def _require_private_path(path: Path, *, description: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink: {path}")
    metadata = path.stat()
    if metadata.st_uid != _ROOT_UID:
        raise PermissionError(f"{description} must be owned by root: {path}")
    if metadata.st_mode & 0o022:
        raise PermissionError(f"{description} must not be group/world-writable: {path}")


def _resolve_uv(workspace: Path | None) -> str:
    workspace = workspace.resolve() if workspace is not None else None
    candidates = [
        _UV_INSTALL_ROOT / "bin/uv",
        *_UV_CANDIDATES,
        *(
            Path(directory) / "uv"
            for directory in os.getenv("PATH", "").split(os.pathsep)
            if directory and Path(directory).is_absolute()
        ),
    ]
    environment = _installation_environment()
    rejected: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if workspace is not None and resolved.is_relative_to(workspace):
                rejected.append(f"{candidate} is inside the task workspace")
                continue
            if resolved.is_relative_to(_RUNTIME_ROOT.resolve()):
                rejected.append(f"{candidate} is inside the runtime that will be recreated")
                continue
            _require_private_path(candidate, description="uv executable")
            _require_private_path(candidate.parent.resolve(), description="uv executable directory")
            if not os.access(candidate, os.X_OK):
                rejected.append(f"{candidate} is not executable")
                continue
            version_result = subprocess.run(
                [str(resolved), "--version"],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = re.match(r"uv (\d+)\.(\d+)\.(\d+)(?:\s|$)", version_result.stdout)
            minimum = tuple(int(part) for part in _UV_VERSION.split("."))
            if version is None or tuple(int(part) for part in version.groups()) < minimum:
                rejected.append(
                    f"{candidate} requires uv >= {_UV_VERSION}, got {version_result.stdout.strip()!r}"
                )
                continue
            return str(resolved)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            rejected.append(f"{candidate}: {exc}")
    detail = f": {'; '.join(rejected)}" if rejected else ""
    raise FileNotFoundError(f"E2B template does not provide a usable trusted uv{detail}")


def _install_uv(workspace: Path | None) -> None:
    """Install the official PyPI wheel's native executable using only the stdlib."""

    if workspace is not None and _UV_INSTALL_ROOT.resolve().is_relative_to(workspace.resolve()):
        raise RuntimeError("uv installation directory must not be inside the task workspace")
    _require_private_path(_UV_INSTALL_ROOT.parent, description="uv installation parent")
    _UV_INSTALL_ROOT.mkdir(mode=0o755, exist_ok=True)
    _require_private_path(_UV_INSTALL_ROOT, description="uv installation directory")
    install_bin = _UV_INSTALL_ROOT / "bin"
    install_bin.mkdir(mode=0o755, exist_ok=True)
    _require_private_path(install_bin, description="uv installation bin directory")
    destination = install_bin / "uv"
    if destination.exists() or destination.is_symlink():
        _require_private_path(destination, description="uv installation destination")

    machine = platform.machine()
    if sys.platform != "linux" or machine not in ("x86_64", "aarch64"):
        raise RuntimeError(f"unsupported uv bootstrap platform: {sys.platform}/{machine}")
    if platform.libc_ver()[0] != "glibc":
        raise RuntimeError("uv bootstrap requires a glibc-based Linux template")
    wheel_suffix = f"-py3-none-manylinux_2_17_{machine}.manylinux2014_{machine}.whl"
    with urlopen(f"https://pypi.org/pypi/uv/{_UV_VERSION}/json", timeout=30) as response:
        metadata = json.load(response)
    wheels = [item for item in metadata["urls"] if item["filename"].endswith(wheel_suffix)]
    if len(wheels) != 1:
        raise RuntimeError(f"official uv {_UV_VERSION} wheel not found for {machine}")
    wheel = wheels[0]
    location = urlparse(wheel["url"])
    if location.scheme != "https" or location.hostname != "files.pythonhosted.org":
        raise RuntimeError("uv wheel must be downloaded from official PyPI storage")
    with TemporaryDirectory(prefix="installer-", dir=_UV_INSTALL_ROOT) as temporary_directory:
        archive_path = Path(temporary_directory) / "uv.whl"
        with urlopen(wheel["url"], timeout=120) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != wheel["digests"]["sha256"]:
            raise RuntimeError("official uv wheel SHA256 verification failed")
        binary = Path(temporary_directory) / "uv"
        with zipfile.ZipFile(archive_path) as archive:
            with (
                archive.open(f"uv-{_UV_VERSION}.data/scripts/uv") as source,
                binary.open("wb") as output,
            ):
                shutil.copyfileobj(source, output)
        if not _is_elf(binary):
            raise RuntimeError("official uv wheel does not contain a native executable")
        binary.chmod(0o755)
        os.replace(binary, destination)


def _ensure_uv(workspace: Path | None) -> str:
    try:
        return _resolve_uv(workspace)
    except FileNotFoundError as exc:
        print(
            f"[bootstrap] {exc}; installing uv {_UV_VERSION} into {_UV_INSTALL_ROOT}",
            file=sys.stderr,
            flush=True,
        )

    try:
        _install_uv(workspace)
        return _resolve_uv(workspace)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Failed to install a usable uv {_UV_VERSION} at {_UV_INSTALL_ROOT}: {exc}"
        ) from exc


def _ensure_runtime(workspace: Path | None = None) -> Path:
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "python": _PYTHON_VERSION,
                "requirements": RUNTIME_REQUIREMENTS,
                "source_requirements": SOURCE_WHEEL_REQUIREMENTS,
            }
        ).encode()
    ).hexdigest()
    if (
        _RUNTIME_MARKER.is_file()
        and _RUNTIME_PYTHON.is_file()
        and _RUNTIME_MARKER.read_text(encoding="utf-8") == fingerprint
    ):
        return _RUNTIME_PYTHON

    uv = _ensure_uv(workspace)
    if workspace is not None and _PYTHON_INSTALL_ROOT.resolve().is_relative_to(workspace.resolve()):
        raise RuntimeError("Python installation directory must not be inside the task workspace")
    _require_private_path(_PYTHON_INSTALL_ROOT.parent, description="Python installation parent")
    _PYTHON_INSTALL_ROOT.mkdir(mode=0o755, exist_ok=True)
    _require_private_path(_PYTHON_INSTALL_ROOT, description="Python installation directory")

    install_environment = _installation_environment()
    python_environment = {
        **install_environment,
        "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PYTHON_INSTALL_DIR": str(_PYTHON_INSTALL_ROOT),
    }
    _run(
        [
            uv,
            "venv",
            "--clear",
            "--managed-python",
            "--python",
            _PYTHON_VERSION,
            str(_RUNTIME_ROOT),
        ],
        env=python_environment,
    )
    # OmegaConf depends on antlr4-python3-runtime==4.9.*, for which PyPI has
    # no wheel. Install that one pinned source release before keeping the
    # rest of the runtime binary-only.
    _run(
        [
            uv,
            "--no-progress",
            "pip",
            "install",
            "--python",
            str(_RUNTIME_PYTHON),
            "--no-deps",
            "--index-url",
            _PYPI_INDEX_URL,
            *SOURCE_WHEEL_REQUIREMENTS,
        ],
        env=install_environment,
    )
    _run(
        [
            uv,
            "--no-progress",
            "pip",
            "install",
            "--python",
            str(_RUNTIME_PYTHON),
            "--only-binary",
            ":all:",
            "--index-url",
            _PYPI_INDEX_URL,
            *RUNTIME_REQUIREMENTS,
        ],
        env=install_environment,
    )
    _RUNTIME_MARKER.touch(mode=0o600)
    _RUNTIME_MARKER.write_text(fingerprint, encoding="utf-8")
    return _RUNTIME_PYTHON


def _is_elf(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(len(_ELF_MAGIC)) == _ELF_MAGIC


if __name__ == "__main__":
    print(f"[bootstrap] NorthRL environment ready: {_ensure_runtime()}", flush=True)
