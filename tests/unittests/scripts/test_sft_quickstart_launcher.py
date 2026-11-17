"""Exercise the shell launcher without contacting Weaver or allocating GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "recipe/sft_quickstart/run.sh"


@pytest.fixture
def launch_env(tmp_path):
    python = tmp_path / "python-shim"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-':\n"
        "    sys.stdin.read()\n"
        "    print('preflight')\n"
        "else:\n"
        "    Path(os.environ['CAPTURE']).write_text(json.dumps({\n"
        "        'argv': sys.argv[1:],\n"
        "        'data': os.environ['SFT_DATA_PATH'],\n"
        "        'key': os.environ['WEAVER_API_KEY'],\n"
        "    }))\n"
        "    print('driver output')\n"
        "    sys.exit(int(os.environ.get('DRIVER_EXIT', '0')))\n"
    )
    python.chmod(0o700)
    run = tmp_path / "run with spaces"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "WEAVER_BASE_URL=https://example.invalid\n"
        "WEAVER_API_KEY=local-test-key\n"
        f"SFT_DATA_PATH='{tmp_path}/data with spaces.jsonl'\n"
        f"PYTHON='{python}'\n"
        f"EXPERIMENT_PATH='{run}'\n"
    )
    env = os.environ.copy()
    env.update(NORTHRL_ENV_FILE=str(env_file), CAPTURE=str(tmp_path / "capture.json"))
    return env, run


@pytest.mark.parametrize("driver_exit", [0, 23])
def test_launcher_preserves_driver_exit_and_env(launch_env, driver_exit):
    env, run = launch_env
    env["DRIVER_EXIT"] = str(driver_exit)
    result = subprocess.run(
        ["bash", str(SCRIPT), "training.epochs=2"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == driver_exit, result.stderr
    assert (run / "exit_code").read_text().strip() == str(driver_exit)
    assert "driver output" in (run / "northrl.log").read_text()
    assert (run / "recipe.yaml").is_file()
    capture = json.loads(Path(env["CAPTURE"]).read_text())
    assert capture["argv"][-1] == "training.epochs=2"
    assert capture["data"].endswith("data with spaces.jsonl")
    assert capture["key"] == "local-test-key"


def test_check_does_not_start_driver_or_create_run_directory(launch_env):
    env, run = launch_env
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"], env=env, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["CAPTURE"]).exists()
    assert not run.exists()


def test_existing_run_is_not_overwritten(launch_env):
    env, run = launch_env
    run.mkdir()
    log = run / "northrl.log"
    log.write_text("earlier run")
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, check=False)
    assert result.returncode != 0
    assert log.read_text() == "earlier run"
    assert not Path(env["CAPTURE"]).exists()


@pytest.mark.parametrize(
    ("row_count", "row", "expected_success"),
    [
        (8, {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}, True),
        (3, {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}, False),
        (8, {"input_ids": [1, 2, 3]}, False),
        (8, {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]}, False),
    ],
)
def test_real_preflight_validates_data_before_training(
    launch_env, tmp_path, row_count, row, expected_success
):
    env, run = launch_env
    data = tmp_path / "train.jsonl"
    data.write_text((json.dumps(row) + "\n") * row_count)
    with Path(env["NORTHRL_ENV_FILE"]).open("a") as stream:
        stream.write(
            f"PYTHON='{sys.executable}'\n"
            f"SFT_DATA_PATH='{os.path.relpath(data, SCRIPT.parents[2])}'\n"
            "SFT_BATCH_SIZE=4\n"
            "SFT_MAX_LENGTH=4096\n"
        )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == expected_success, result.stderr
    if expected_success:
        assert "2 optimizer windows per epoch" in result.stdout
    assert not run.exists()
    assert not Path(env["CAPTURE"]).exists()
