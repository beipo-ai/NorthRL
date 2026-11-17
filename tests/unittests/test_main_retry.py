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

import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import northrl.main as main_module


@pytest.fixture
def main_cli(tmp_path):
    """Run the actual local entry point without service credentials or run config."""
    root = Path(main_module.__file__).resolve().parents[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("WEAVER_", "NORTHGATE_", "E2B_", "SFT_"))
    }
    env.pop("TRAIN_CONFIG", None)
    env.update(
        PYTHONPATH=str(root),
        EXPERIMENT_PATH=str(tmp_path / "run"),
        SFT_DATA_PATH=str(root / "recipe/sft_quickstart/train.example.jsonl"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "northrl.main", *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    return run


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--help", "powered by Hydra"), ("--hydra-help", "Hydra")],
)
def test_main_help_without_recipe_or_services(main_cli, tmp_path, flag, expected):
    result = main_cli(flag)

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert not (tmp_path / "run").exists()


def test_main_composes_recipe_without_starting_training(main_cli, tmp_path):
    recipe_dir = Path(main_module.__file__).resolve().parents[1] / "recipe/sft_quickstart"
    result = main_cli(
        f"--config-path={recipe_dir}",
        "--config-name=weaver",
        "--cfg=job",
        "--resolve",
        "training.epochs=3",
    )

    assert result.returncode == 0, result.stderr
    config = OmegaConf.create(result.stdout)
    assert config.algorithm.type == "sft"
    assert config.training.epochs == 3
    assert config.data.train.files == [str(recipe_dir / "train.example.jsonl")]
    assert config.training.save.dir == str(tmp_path / "run/ckpt")
    assert not (tmp_path / "run").exists()


def test_main_without_recipe_explains_local_configuration(main_cli, tmp_path):
    result = main_cli()

    assert result.returncode != 0
    assert "Set TRAIN_CONFIG to the recipe file" in result.stderr
    assert "--config-path=<recipe-directory> --config-name=<recipe-name>" in result.stderr
    assert not (tmp_path / "run").exists()


def _config(*, max_retries: int = 1):
    return OmegaConf.create(
        {
            "experiment": {"project": "test-project", "name": "test-run"},
            "training": {
                "resume": {
                    "mode": "from_path",
                    "path": "weaver://state/initial",
                    "max_retries": max_retries,
                }
            },
            "_config_file_path": "/tmp/test.yaml",
        }
    )


@pytest.mark.parametrize("resume_mode", ["from_path", "auto", "disable"])
def test_main_task_rebuilds_controller_and_auto_resumes_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    resume_mode: str,
) -> None:
    first = MagicMock()
    first.run.side_effect = RuntimeError("worker died")
    second = MagicMock()
    second.run.side_effect = RuntimeError("worker died again")
    third = MagicMock()
    controller_factory = MagicMock(side_effect=[first, second, third])
    config = _config(max_retries=2)
    config.training.resume.mode = resume_mode

    monkeypatch.setattr(main_module, "NorthRLController", controller_factory)
    monkeypatch.setattr(main_module, "dump_resolved_config", MagicMock())

    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        main_module.main_task(config)

    assert controller_factory.call_count == 3
    first.run.assert_called_once_with()
    second.run.assert_called_once_with()
    third.run.assert_called_once_with()
    expected_path = "" if resume_mode == "disable" else "weaver://state/initial"
    for call in controller_factory.call_args_list[1:]:
        retry_config = call.args[0]
        assert retry_config is not config
        assert retry_config.training.resume.mode == "auto"
        assert retry_config.training.resume.path == expected_path
        assert OmegaConf.is_struct(retry_config)
    # The requested configuration remains the reproducible source config.
    assert config.training.resume.mode == resume_mode
    assert config.training.resume.path == "weaver://state/initial"
    assert "Starting NorthRL retry 1/2 with training.resume.mode=auto" in caplog.messages
    assert "Starting NorthRL retry 2/2 with training.resume.mode=auto" in caplog.messages


def test_main_task_raises_after_retry_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = MagicMock()
    first.run.side_effect = RuntimeError("first failure")
    second = MagicMock()
    second.run.side_effect = RuntimeError("second failure")
    controller_factory = MagicMock(side_effect=[first, second])

    monkeypatch.setattr(main_module, "NorthRLController", controller_factory)
    monkeypatch.setattr(main_module, "dump_resolved_config", MagicMock())

    with pytest.raises(RuntimeError, match="second failure"):
        main_module.main_task(_config())

    assert controller_factory.call_count == 2


def test_main_task_can_disable_automatic_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = MagicMock()
    controller.run.side_effect = RuntimeError("no retry")
    controller_factory = MagicMock(return_value=controller)

    monkeypatch.setattr(main_module, "NorthRLController", controller_factory)
    monkeypatch.setattr(main_module, "dump_resolved_config", MagicMock())

    with pytest.raises(RuntimeError, match="no retry"):
        main_module.main_task(_config(max_retries=0))

    controller_factory.assert_called_once()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(143)])
def test_main_task_does_not_retry_external_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    controller = MagicMock()
    controller.run.side_effect = interrupt
    controller_factory = MagicMock(return_value=controller)

    monkeypatch.setattr(main_module, "NorthRLController", controller_factory)
    monkeypatch.setattr(main_module, "dump_resolved_config", MagicMock())

    with pytest.raises(type(interrupt)):
        main_module.main_task(_config())

    controller_factory.assert_called_once()
