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

import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
EXCLUDED_FILES = {
    "scripts/__init__.py",
    "scripts/run.py",
    "scripts/private_runtime.sh",
    "cli/__init__.py",
}


def _assert_runtime_package(filenames):
    assert {"northrl/__init__.py", "northrl/main.py"} <= filenames
    assert not any(filename.startswith(("scripts/", "cli/")) for filename in filenames)


def _build(project, *formats):
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", *formats],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_wheel_and_sdist_include_only_runtime(tmp_path):
    for module in ("build", "setuptools", "wheel"):
        pytest.importorskip(module, reason=f"Packaging check requires optional tool {module}")
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPOSITORY / filename, project / filename)
    for filename in EXCLUDED_FILES:
        destination = project / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("excluded test sentinel\n", encoding="utf-8")
    shutil.copytree(
        REPOSITORY / "northrl",
        project / "northrl",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    configuration = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    resources = {
        resource.relative_to(REPOSITORY).as_posix()
        for pattern in configuration["tool"]["setuptools"]["package-data"]["northrl"]
        for resource in (REPOSITORY / "northrl").glob(pattern)
        if resource.is_file()
    }
    assert resources
    _build(project, "--wheel", "--sdist")

    wheel_path = next((project / "dist").glob("*.whl"))
    with ZipFile(wheel_path) as wheel:
        wheel_files = set(wheel.namelist())
        _assert_runtime_package(wheel_files)
        assert resources <= wheel_files
        for filename in wheel_files:
            if filename.endswith(".dist-info/entry_points.txt"):
                assert "northrl =" not in wheel.read(filename).decode()
        installed = tmp_path / "installed"
        wheel.extractall(installed)

    result = subprocess.run(
        [sys.executable, "-m", "northrl.main", "--version"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(installed)},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Hydra " in result.stdout
    sdist_path = next((project / "dist").glob("*.tar.gz"))
    with tarfile.open(sdist_path) as sdist:
        sdist_files = {
            member.name.split("/", 1)[1]
            for member in sdist.getmembers()
            if member.isfile() and "/" in member.name
        }
        _assert_runtime_package(sdist_files)
        assert resources <= sdist_files
        extracted = tmp_path / "sdist"
        sdist.extractall(extracted, filter="data")
    rebuilt_project = next(extracted.iterdir())
    _build(rebuilt_project, "--wheel")
    with ZipFile(next((rebuilt_project / "dist").glob("*.whl"))) as rebuilt_wheel:
        _assert_runtime_package(set(rebuilt_wheel.namelist()))
        assert resources <= set(rebuilt_wheel.namelist())


def test_docker_installs_local_runtime():
    path = REPOSITORY / "docker/Dockerfile"
    contents = path.read_text(encoding="utf-8")
    sources = [
        source
        for line in contents.splitlines()
        if line.startswith("COPY ") and "--from=" not in line
        for source in line.split()[1:-1]
    ]
    assert "northrl/" in sources
    assert not any(source.startswith(("scripts/", "cli/")) for source in sources)
    assert "python -m northrl.main --version" in contents
