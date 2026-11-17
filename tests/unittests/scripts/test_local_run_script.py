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

"""Run the local shell entry without starting training or contacting services."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]


@pytest.fixture
def local_run(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "run.sh"
    shutil.copy2(REPOSITORY / "run.sh", script)
    caller = tmp_path / "caller"
    caller.mkdir()
    recipe_directory = caller / "recipes with spaces ' $HOME ;"
    recipe_directory.mkdir()
    recipe = recipe_directory / "train.yaml"
    recipe.write_text("experiment: {}\n", encoding="utf-8")
    capture = tmp_path / "capture.json"
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "child = subprocess.check_output([sys.executable, '-c',\n"
        '    \'import os; print(os.environ.get("TEST_VALUE", "missing"))\'],\n'
        "    text=True).strip()\n"
        "Path(os.environ['TEST_CAPTURE']).write_text(json.dumps({\n"
        "    'argv': sys.argv[1:], 'config': os.environ.get('TRAIN_CONFIG'),\n"
        "    'value': os.environ.get('TEST_VALUE'), 'child': child,\n"
        "    'python': sys.argv[0], 'cwd': os.getcwd(),\n"
        "    'repository': os.environ.get('NORTHRL_PATH'),\n"
        "}))\n"
        "sys.exit(int(os.environ.get('TEST_EXIT', '0')))\n",
        encoding="utf-8",
    )
    python.chmod(0o700)
    env = os.environ.copy()
    for name in ("NORTHRL_ENV_FILE", "NORTHRL_PATH", "TEST_VALUE", "TEST_EXIT", "PYTHON"):
        env.pop(name, None)
    env.update(
        PATH=f"{python.parent}{os.pathsep}{env.get('PATH', '')}",
        TRAIN_CONFIG="old-config.yaml",
        TEST_CAPTURE=str(capture),
    )

    def run(*arguments, extra_env=None):
        return subprocess.run(
            ["bash", str(script), *map(str, arguments)],
            cwd=caller,
            env={**env, **(extra_env or {})},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    return run, project, recipe, capture, python


def test_shell_environment_does_not_load_nearby_files(local_run):
    run, project, recipe, capture, python = local_run
    for directory in (project, recipe.parent, recipe.parent.parent):
        for name in (".env", "env.sh"):
            (directory / name).write_text("exit 29\n", encoding="utf-8")
    result = run(recipe, extra_env={"TEST_VALUE": "parent"})
    assert result.returncode == 0, result.stderr
    data = json.loads(capture.read_text())
    assert data["value"] == data["child"] == "parent"
    assert data["python"] == str(python)


@pytest.mark.parametrize("selection", ["option", "equals", "alias"])
def test_explicit_env_file_selects_variables_and_python(local_run, selection):
    run, project, recipe, capture, python = local_run
    (project / ".env").write_text("exit 29\n", encoding="utf-8")
    chosen_python = project / "custom python"
    shutil.copy2(python, chosen_python)
    caller = recipe.parent.parent
    env_file = caller / "private env.sh"
    env_file.write_text(
        'declare -x TEST_VALUE="explicit"\n'
        f"PYTHON={shlex.quote(str(chosen_python))}\n"
        "NORTHRL_PATH=/chosen/framework\n"
        "TRAIN_CONFIG=wrong.yaml\n",
        encoding="utf-8",
    )
    extra_env = {"TEST_VALUE": "parent", "NORTHRL_PATH": "/inherited/framework"}
    if selection == "alias":
        arguments = [recipe]
        extra_env["NORTHRL_ENV_FILE"] = env_file.name
    else:
        arguments = (
            ["--env-file", env_file.name, recipe]
            if selection == "option"
            else [f"--env-file={env_file.name}", recipe]
        )
        extra_env["NORTHRL_ENV_FILE"] = str(caller / "must-not-be-loaded.env")
    result = run(*arguments, extra_env=extra_env)
    assert result.returncode == 0, result.stderr
    data = json.loads(capture.read_text())
    assert data["value"] == data["child"] == "explicit"
    assert data["python"] == str(chosen_python)
    assert data["repository"] == "/chosen/framework"
    assert data["config"] == str(recipe)


@pytest.mark.parametrize("unreadable", [False, True])
def test_bad_explicit_environment_file_stops_before_python(local_run, unreadable):
    if unreadable and os.geteuid() == 0:
        pytest.skip("Root can read files without permission bits")
    run, project, recipe, capture, _ = local_run
    env_file = project / "private.env"
    if unreadable:
        env_file.write_text("TEST_VALUE=private\n", encoding="utf-8")
        env_file.chmod(0)
    try:
        result = run("--env-file", env_file, recipe)
    finally:
        if unreadable:
            env_file.chmod(0o600)
    assert result.returncode != 0
    assert not capture.exists()


def test_failed_environment_setup_stops_before_python(local_run):
    run, project, recipe, capture, _ = local_run
    env_file = project / "private.env"
    env_file.write_text("false\nTEST_VALUE=should-not-run\n", encoding="utf-8")
    result = run("--env-file", env_file, recipe)
    assert result.returncode != 0
    assert not capture.exists()


def test_paths_use_caller_directory_and_overrides_pass_unchanged(local_run):
    run, project, recipe, capture, _ = local_run
    caller = recipe.parent.parent
    env_file = caller / "private.env"
    env_file.write_text(
        f"cd {shlex.quote(str(project))}\nset -- replaced arguments\nTRAIN_CONFIG=wrong.yaml\n",
        encoding="utf-8",
    )
    overrides = [
        "training.epochs=2",
        "experiment.name='a b; $(touch injected)'",
        "--cfg=job",
        "--env-file=passed-to-hydra",
    ]
    result = run("--env-file", env_file.name, recipe.relative_to(caller), *overrides)
    assert result.returncode == 0, result.stderr
    data = json.loads(capture.read_text())
    assert data["config"] == str(recipe)
    assert data["cwd"] == str(caller)
    assert data["argv"][:2] == ["-m", "northrl.main"]
    assert data["argv"][-len(overrides) :] == overrides
    assert not (caller / "injected").exists()


@pytest.mark.parametrize("override_python", [False, True])
def test_active_python_or_explicit_override_is_used(local_run, override_python):
    run, project, recipe, capture, python = local_run
    virtualenv_python = project / ".venv/bin/python"
    virtualenv_python.parent.mkdir(parents=True)
    shutil.copy2(python, virtualenv_python)
    chosen_python = project / "selected python"
    shutil.copy2(python, chosen_python)
    result = run(recipe, extra_env={"PYTHON": str(chosen_python)} if override_python else {})
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text())["python"] == str(
        chosen_python if override_python else python
    )


@pytest.mark.parametrize("value", [None, "", "/inherited/framework"])
def test_framework_path_defaults_to_script_root_or_preserves_shell_value(local_run, value):
    run, project, recipe, capture, _ = local_run
    result = run(recipe, extra_env={} if value is None else {"NORTHRL_PATH": value})
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text())["repository"] == (value or str(project))


def test_empty_framework_path_in_env_file_uses_script_root(local_run):
    run, project, recipe, capture, _ = local_run
    env_file = project / "private.env"
    env_file.write_text("NORTHRL_PATH=\n", encoding="utf-8")
    result = run("--env-file", env_file, recipe, extra_env={"NORTHRL_PATH": "/inherited/framework"})
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text())["repository"] == str(project)


def test_environment_file_can_derive_paths_from_framework_root(local_run):
    run, project, recipe, capture, _ = local_run
    env_file = project / "private.env"
    env_file.write_text('TEST_VALUE="$NORTHRL_PATH/task-assets"\n', encoding="utf-8")
    result = run("--env-file", env_file, recipe)
    assert result.returncode == 0, result.stderr
    data = json.loads(capture.read_text())
    assert data["value"] == data["child"] == str(project / "task-assets")


def test_training_exit_code_is_preserved(local_run):
    run, _, recipe, capture, _ = local_run
    result = run(recipe, extra_env={"TEST_EXIT": "23"})
    assert result.returncode == 23, result.stderr
    assert capture.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--env-file"],
        ["--env-file", ""],
        ["--env-file="],
        ["--env-file", "--unknown"],
        ["--env-file", "private.env"],
        ["--unknown", "{recipe}"],
        ["--env-file", "private.env", "--env-file", "private.env", "{recipe}"],
        ["missing.yaml"],
    ],
)
def test_invalid_arguments_stop_before_loading_environment(local_run, arguments):
    run, _, recipe, capture, _ = local_run
    caller = recipe.parent.parent
    marker = caller / "env-was-loaded"
    env_file = caller / "private.env"
    env_file.write_text(f"touch {shlex.quote(str(marker))}\n", encoding="utf-8")
    result = run(
        *(argument.replace("{recipe}", str(recipe)) for argument in arguments),
        extra_env={"NORTHRL_ENV_FILE": str(env_file)},
    )
    assert result.returncode != 0
    assert not capture.exists()
    assert not marker.exists()


def test_real_hydra_resolves_recipe_after_environment_setup(tmp_path):
    env_file = tmp_path / "sft.env"
    env_file.write_text(
        f"PYTHON={shlex.quote(sys.executable)}\n"
        f"PYTHONPATH={shlex.quote(str(REPOSITORY))}\n"
        "SFT_DATA_PATH=/tmp/local-run-example.jsonl\n"
        "EXPERIMENT_PATH=/tmp/local-run-example\n"
        "SFT_BATCH_SIZE=7\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            str(REPOSITORY / "run.sh"),
            "--env-file",
            str(env_file),
            str(REPOSITORY / "recipe/sft_quickstart/weaver.yaml"),
            "--cfg=job",
            "--resolve",
            "training.epochs=3",
        ],
        cwd=tmp_path,
        env={**os.environ, "NORTHRL_ENV_FILE": "/must-not-load.env"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    from omegaconf import OmegaConf

    config = OmegaConf.create(result.stdout)
    assert config.data.train.files == ["/tmp/local-run-example.jsonl"]
    assert config.data.train.batch_size == 7
    assert config.training.save.dir == "/tmp/local-run-example/ckpt"
    assert config.training.epochs == 3


@pytest.mark.parametrize("use_env_file", [False, True])
def test_real_e2b_recipe_resolves_without_explicit_framework_path(tmp_path, use_env_file):
    """Catch the missing NORTHRL_PATH failure with real Hydra, without remote calls."""
    settings = {
        "PYTHON": sys.executable,
        "PYTHONPATH": str(REPOSITORY),
        "NORTHRL_TMAX_MODEL": "example/model",
        "NORTHRL_TMAX_TOKENIZER": "/example/tokenizer",
        "NORTHRL_TMAX_TRAIN_DATA": "/example/train.jsonl",
        "NORTHRL_TMAX_TASK_FILES_ROOT": "/example/evaluation",
        "NORTHRL_TMAX_CC_VOLUME": "example-volume",
        "EXPERIMENT_PATH": "/example/run",
    }
    env = os.environ.copy()
    for name in ("NORTHRL_ENV_FILE", "NORTHRL_PATH"):
        env.pop(name, None)
    arguments = ["bash", str(REPOSITORY / "run.sh")]
    if use_env_file:
        env_file = tmp_path / "tmax.env"
        env_file.write_text(
            "".join(f"{name}={shlex.quote(value)}\n" for name, value in settings.items()),
            encoding="utf-8",
        )
        arguments.extend(["--env-file", str(env_file)])
    else:
        env.update(settings)
    result = subprocess.run(
        [
            *arguments,
            str(REPOSITORY / "recipe/tmax/train.yaml"),
            "--cfg=job",
            "--resolve",
            "training.total_steps=1",
            "runtime.logger.backends=[console]",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    from omegaconf import OmegaConf

    config = OmegaConf.create(result.stdout)
    assert config.rollout.execution.entrypoint.target == str(
        REPOSITORY / "northrl/rollout_entrypoints/tmax_claude_code_rollout.py"
    )
    assert config.rollout.execution.entrypoint.task_files_root == "/example/evaluation"
    assert config.rollout.execution.volume_mounts["/mnt/ccbin"] == "example-volume"
    assert config.training.total_steps == 1
