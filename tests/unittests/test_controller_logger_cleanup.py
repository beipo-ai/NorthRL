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

"""Exercise logger shutdown through the real controller and retry boundaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import northrl.main as main_module
from northrl.controller import NorthRLController


def _controller(events, *, attempt=0, run_error=None, stop_error=None):
    controller = NorthRLController.__new__(NorthRLController)
    experiment_logger = MagicMock()
    experiment_logger.close.side_effect = lambda *, exit_code: events.append(
        (attempt, "close", exit_code)
    )
    controller.activity_tracker = SimpleNamespace(experiment_logger=experiment_logger)

    def run_training():
        events.append((attempt, "train"))
        if run_error is not None:
            raise run_error

    def stop(**_kwargs):
        # Workers may emit their final metrics while draining in _stop().
        experiment_logger.close.assert_not_called()
        experiment_logger.log({"final_metric": 1.0}, step=1)
        events.append((attempt, "stop"))
        if stop_error is not None:
            raise stop_error

    controller._run_training_process = MagicMock(side_effect=run_training)
    controller._stop = MagicMock(side_effect=stop)
    return controller


@pytest.mark.parametrize(
    ("run_error", "stop_error", "exit_code"),
    [
        pytest.param(None, None, 0, id="success"),
        pytest.param(RuntimeError("training failed"), None, 1, id="training-failure"),
        pytest.param(KeyboardInterrupt(), None, 1, id="training-interrupt"),
        pytest.param(SystemExit(143), None, 1, id="training-sigterm"),
        pytest.param(None, RuntimeError("cleanup failed"), 1, id="cleanup-failure"),
        pytest.param(None, KeyboardInterrupt(), 1, id="cleanup-interrupt"),
        pytest.param(
            ValueError("training failed"),
            RuntimeError("cleanup failed"),
            1,
            id="preserve-training-failure",
        ),
        pytest.param(
            KeyboardInterrupt(),
            RuntimeError("cleanup failed"),
            1,
            id="preserve-training-interrupt",
        ),
    ],
)
def test_run_closes_logger_after_stop_with_attempt_exit_code(run_error, stop_error, exit_code):
    events = []
    controller = _controller(events, run_error=run_error, stop_error=stop_error)
    expected_error = run_error if run_error is not None else stop_error

    if expected_error is None:
        controller.run()
    else:
        with pytest.raises(type(expected_error)) as exc_info:
            controller.run()
        assert exc_info.value is expected_error

    assert events == [(0, "train"), (0, "stop"), (0, "close", exit_code)]
    controller.activity_tracker.experiment_logger.close.assert_called_once_with(exit_code=exit_code)
    if isinstance(run_error, (KeyboardInterrupt, SystemExit)):
        controller._stop.assert_called_once_with(close_weaver_first=True)
    else:
        controller._stop.assert_called_once_with()


def test_main_retry_finishes_failed_logger_before_creating_next_controller(monkeypatch):
    events = []
    controllers = []
    config = OmegaConf.create(
        {
            "experiment": {"project": "test-project", "name": "test-run"},
            "training": {"resume": {"mode": "disable", "path": "", "max_retries": 1}},
        }
    )

    def controller_factory(_config):
        attempt = len(controllers)
        if attempt:
            controllers[-1].activity_tracker.experiment_logger.close.assert_called_once_with(
                exit_code=1
            )
        events.append((attempt, "init"))
        controller = _controller(
            events,
            attempt=attempt,
            run_error=RuntimeError("retry this attempt") if attempt == 0 else None,
        )
        controllers.append(controller)
        return controller

    monkeypatch.setattr(main_module, "NorthRLController", controller_factory)
    monkeypatch.setattr(main_module, "dump_resolved_config", MagicMock())

    main_module.main_task(config)

    assert events == [
        (0, "init"),
        (0, "train"),
        (0, "stop"),
        (0, "close", 1),
        (1, "init"),
        (1, "train"),
        (1, "stop"),
        (1, "close", 0),
    ]
