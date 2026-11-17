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

"""Regression coverage for W&B run ownership across controller retries."""

import gc
import json
import logging
import sys
import weakref
from threading import Event, Thread
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import northrl.utils.tracking as tracking_module
from northrl.utils.tracking import Tracking


class _RunConfig(dict):
    def update(self, values, **_kwargs):
        super().update(values)


class _WandbRun:
    def __init__(self, wandb):
        self.config = _RunConfig()
        self._wandb = wandb
        self.finished = False
        if not wandb.disabled:
            self._is_finished = False
        self.finish_calls = []
        self.logged = []
        self.log_calls = []
        self.history = []
        self.dropped = []
        self._step = 0
        self._partial_history = {}
        self.metric_definitions = []
        self.finish_event = Event()

    @property
    def step(self):
        return self._step

    def define_metric(self, name, **kwargs):
        self.metric_definitions.append((name, kwargs))

    def get_url(self):
        return "https://wandb.test/run"

    def log(self, data, step=None, commit=None):
        if self.finished:
            raise RuntimeError("Cannot log to a finished run")
        self.logged.append((dict(data), step))
        self.log_calls.append({"data": dict(data), "step": step, "commit": commit})
        # Mirror W&B's partial history behavior: same-step data merges until
        # commit, a later step flushes the prior row, and past steps are dropped.
        if step is not None:
            if step < self._step:
                self.dropped.append((dict(data), step))
                return
            if step > self._step:
                self._flush_partial_history()
                self._step = step
        self._partial_history.update(data)
        if commit or (step is None and commit is None):
            self._flush_partial_history()

    def _flush_partial_history(self):
        if self._partial_history:
            self.history.append({**self._partial_history, "_step": self._step})
            self._partial_history = {}
            self._step += 1

    def finish(self, exit_code=0):
        self._flush_partial_history()
        self.finish_calls.append(exit_code)
        self.finished = True
        if not self._wandb.disabled:
            self._is_finished = True
        # W&B 0.19.8 clears global run state even if this run was already
        # finished. Calling finish twice on an old run can break a retry.
        self._wandb.run = None
        self.finish_event.set()


class _Wandb(ModuleType):
    def __init__(self, *, disabled=False):
        super().__init__("wandb")
        self.disabled = disabled
        self.run = None
        self.runs = []
        self.init_calls = []
        self.login = MagicMock()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.run is not None:
            if not kwargs.get("reinit"):
                return self.run
            self.run.finish()
        self.run = _WandbRun(self)
        self.runs.append(self.run)
        return self.run

    def log(self, *, data, step=None, commit=None):
        if self.run is None:
            raise RuntimeError("You must call wandb.init() before wandb.log()")
        self.run.log(data=data, step=step, commit=commit)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


@pytest.fixture
def wandb(monkeypatch, request):
    fake = _Wandb(disabled=getattr(request, "param", False))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_HOST", "https://wandb.test")
    monkeypatch.setenv("WANDB_KEY", "test-key")
    monkeypatch.setattr(tracking_module, "capture_git_change", lambda: {})
    monkeypatch.setattr(tracking_module, "configure_wandb_workspaces", MagicMock())
    return fake


def _tracking():
    return Tracking("test-project", "test-run", default_backend="wandb", config={})


def test_workspace_ordering_receives_owned_run_after_metric_definitions(wandb, monkeypatch):
    def configure(run):
        assert run is wandb.run
        assert run.metric_definitions
        assert not run.log_calls

    configure_mock = MagicMock(side_effect=configure)
    monkeypatch.setattr(tracking_module, "configure_wandb_workspaces", configure_mock)
    tracker = _tracking()
    configure_mock.assert_called_once_with(wandb.run)
    tracker.close()


@pytest.mark.parametrize("wandb", [False, True], indirect=True, ids=["online", "disabled"])
def test_old_cyclic_logger_collection_does_not_finish_retry_run(wandb):
    # An ActivityTracker can keep the old logger alive through a reference
    # cycle until garbage collection happens in a later retry.
    old = _tracking()
    old.cycle = old
    old_ref = weakref.ref(old)
    old_run = wandb.run

    current = _tracking()
    current_run = wandb.run
    assert current_run is not old_run
    assert current.logger["wandb"] is current_run
    assert old_run.finish_calls == [0]  # reinit=True closed the old run.

    del old
    gc.collect()

    assert old_ref() is None
    assert old_run.finish_calls == [0]
    assert wandb.run is current_run
    current.log({"loss": 0.5}, step=1)
    wandb.log(data={"loss": 0.25}, step=2)
    assert current_run.logged == [
        ({"loss": 0.5, "training/train_step": 1}, 1),
        ({"loss": 0.25}, 2),
    ]
    current.close()


@pytest.mark.parametrize("exit_code", [0, 1])
def test_close_is_idempotent_and_preserves_first_exit_code(wandb, exit_code):
    old = _tracking()
    old_run = wandb.run
    old.close(exit_code=exit_code)
    assert old_run.finish_calls == [exit_code]

    current = _tracking()
    current_run = wandb.run
    old.close(exit_code=1 - exit_code)
    del old
    gc.collect()

    assert old_run.finish_calls == [exit_code]
    assert wandb.run is current_run
    wandb.log(data={"loss": 0.5}, step=1)
    current.close()


def test_close_failure_warns_continues_and_is_not_retried_by_destructor(wandb, caplog):
    tracker = _tracking()
    run = wandb.run
    run.finish = MagicMock(side_effect=RuntimeError("finish failed"))
    other_backend = SimpleNamespace(finish=MagicMock())
    tracker.logger["swanlab"] = other_backend

    with caplog.at_level(logging.WARNING, logger=tracking_module.__name__):
        tracker.close(exit_code=1)

    run.finish.assert_called_once_with(exit_code=1)
    other_backend.finish.assert_called_once_with()
    assert any("wandb" in record.message for record in caplog.records)

    tracker.close()
    # The logged exception retains traceback references, so exercise the
    # destructor explicitly instead of relying on collection in this test.
    tracker.__del__()

    run.finish.assert_called_once_with(exit_code=1)
    other_backend.finish.assert_called_once_with()


def test_close_swanlab_uses_its_argument_free_finish_api(monkeypatch):
    swanlab = ModuleType("swanlab")
    swanlab.init = MagicMock()
    swanlab.login = MagicMock()
    swanlab.finish = MagicMock(spec=lambda: None)
    monkeypatch.setitem(sys.modules, "swanlab", swanlab)
    tracker = Tracking("test-project", "test-run", default_backend="swanlab", config={})

    tracker.close(exit_code=1)
    tracker.close()

    swanlab.finish.assert_called_once_with()


def test_destructor_handles_partially_initialized_tracker():
    tracker = Tracking.__new__(Tracking)
    tracker.__del__()


@pytest.mark.parametrize("wandb", [False, True], indirect=True, ids=["online", "disabled"])
def test_live_snapshots_are_logged_without_writing_metrics(wandb, caplog):
    tracker = _tracking()
    run = wandb.run
    backends = [SimpleNamespace(log=MagicMock(), finish=MagicMock()) for _ in range(3)]
    tracker.logger.update(zip(["console", "mlflow", "swanlab"], backends))
    original = {
        "live/tool/errors": 1,
        "live/inference/latency_seconds": 0.125,
        "live/trace/generated_tokens": 12,
        "live/e2b/active_sandboxes": 3,
    }
    snapshot = dict(original)
    metric_definitions = list(run.metric_definitions)

    with caplog.at_level(logging.INFO, logger=tracking_module.__name__):
        tracker.log_live(original, elapsed_seconds=10)
        tracker.log_live(original, elapsed_seconds=20)

    assert original == snapshot
    assert run.logged == []
    assert run.metric_definitions == metric_definitions
    assert not any(name.startswith("live/") for name, _ in metric_definitions)
    for backend in backends:
        backend.log.assert_not_called()
    assert len(caplog.records) == 2
    for record, elapsed in zip(caplog.records, [10, 20]):
        assert record.name == tracking_module.__name__
        assert record.levelno == logging.INFO
        assert record.message.startswith("[rollout-monitor]")
        assert f"elapsed_seconds={elapsed}" in record.message
        encoded_snapshot = record.message[record.message.index("{") :]
        assert json.loads(encoded_snapshot) == snapshot
        assert list(json.loads(encoded_snapshot)) == sorted(snapshot)
    tracker.close()


def test_live_snapshots_are_logged_without_any_metric_backend(caplog):
    tracker = Tracking("test-project", "test-run", default_backend=[], config={})

    with caplog.at_level(logging.INFO, logger=tracking_module.__name__):
        tracker.log_live({"live/e2b/active_sandboxes": 3}, elapsed_seconds=10)

    assert tracker.logger == {}
    assert len(caplog.records) == 1
    assert caplog.records[0].message.startswith("[rollout-monitor]")
    assert '"live/e2b/active_sandboxes": 3' in caplog.records[0].message
    tracker.close()


def test_training_metrics_keep_round_axis_and_live_logs_do_not_add_history_rows(wandb):
    tracker = _tracking()
    run = wandb.run
    original = {"loss": 0.5}
    tracker.log_live({"live/e2b/active_sandboxes": 3}, elapsed_seconds=10)
    tracker.log(original, step=1)
    tracker.log_live({"live/e2b/active_sandboxes": 0}, elapsed_seconds=30)
    tracker.log({"loss": 0.25}, step=2)

    assert original == {"loss": 0.5}
    assert run.logged == [
        ({"loss": 0.5, "training/train_step": 1}, 1),
        ({"loss": 0.25, "training/train_step": 2}, 2),
    ]
    assert (
        "training/train_step",
        {
            "step_metric": "training/train_step",
            "step_sync": False,
            "hidden": True,
            "summary": "last",
        },
    ) in run.metric_definitions
    assert (
        "*",
        {
            "step_metric": "training/train_step",
            "step_sync": False,
            "hidden": False,
            "summary": "none",
        },
    ) in run.metric_definitions
    tracker.close()


def test_algorithm_metrics_reach_wandb_once_on_the_completed_training_step(wandb):
    tracker = _tracking()
    run = wandb.run
    raw = {
        "trajectory/reward_mean": 0.75,
        "trajectory/advantage_mean": -0.5,
        "trajectory/advantage_std": None,
        "group/reward_std_max": 0.25,
        "kl/actor_ref_mean": 0.02,
        "kl/actor_ref_loss": float("nan"),
        "weaver/entropy_coeff": 0.0,
        "weaver/ppo_clip/clipfrac": 0.1,
        "weaver/is:ratio": 1.1,
        "sampling/logprob_mean": -2.0,
        "mis/token_rejection_fraction": 0.0,
        "opd/teacher_kl": 0.03,
        "weaver/scale": 99.0,
        "weaver/loss": 0.8,
        "batch/trajectory_count": 4,
    }
    expected_algorithm = {
        "01_algorithm/01_trajectory/reward_mean": 0.75,
        "01_algorithm/09_trajectory/advantage_mean": -0.5,
        "01_algorithm/17_group/reward_std_max": 0.25,
        "01_algorithm/18_kl/actor_ref_mean": 0.02,
        "01_algorithm/26_weaver/entropy_coeff": 0.0,
        "01_algorithm/22_weaver/ppo_clip/clipfrac": 0.1,
        "01_algorithm/32_weaver/is:ratio": 1.1,
        "01_algorithm/28_sampling/logprob_mean": -2.0,
        "12_mis/token_rejection_fraction": 0.0,
        "01_algorithm/54_opd/teacher_kl": 0.03,
    }

    tracker.log(raw, step=7)
    assert run.logged == [
        (
            {
                **expected_algorithm,
                "13_weaver/loss": 0.8,
                "15_batch/trajectory_count": 4,
                "training/train_step": 7,
            },
            7,
        )
    ]
    assert "trajectory/reward_mean" in raw
    assert "algorithm/trajectory/reward_mean" not in raw
    assert run.history == []

    # Producers may already have normalized a payload before Tracking sees it.
    tracker.log({"algorithm/kl/actor_ref_mean": 0.02}, step=7)
    tracker.log({"timing_weights/weight_sync_seconds": 0.4}, step=7)
    assert run.history == []
    assert all(call["commit"] is False for call in run.log_calls)
    tracker.log({"val/reward": 0.6}, step=7, commit=True)
    tracker.flush(7)

    assert run.history == [
        {
            **expected_algorithm,
            "13_weaver/loss": 0.8,
            "15_batch/trajectory_count": 4,
            "09_timing_weights/weight_sync_seconds": 0.4,
            "02_val/reward": 0.6,
            "training/train_step": 7,
            "_step": 7,
        }
    ]
    assert len(run.log_calls) == 4
    assert run.dropped == []
    definitions = dict(run.metric_definitions)
    assert not any(key.startswith(("algorithm/", "mis/")) for key in definitions)
    for key in expected_algorithm:
        assert definitions[key]["step_metric"] == "training/train_step"
        assert definitions[key]["hidden"] is False
    assert definitions["01_algorithm/01_trajectory/reward_mean"]["summary"] == "last"
    tracker.close()


def test_duration_metrics_reach_wandb_once_in_stage_sections_on_the_consuming_step(wandb):
    tracker = _tracking()
    run = wandb.run
    raw = {
        "trajectory/duration_seconds_mean": 12.5,
        "group/duration_seconds_mean": 20.0,
        "timing/group/duration_seconds_mean": 21.0,
        "timing_rollout/group_mean_seconds": 22.0,
        "group/duration_seconds_observed_count": 2,
        "model_request/duration_seconds_max": 9.0,
        "reward_fn/custom_reward/verifier/duration_seconds_mean": 3.0,
        "val/reward_fn/custom_reward/group/duration_seconds_mean": 24.0,
        "val/timing/verifier/duration_seconds_observed_count": 3,
        "task/duration_seconds_mean": 99.0,
        "batch/trajectory_count": 4,
        "verifier/skipped_length_count": 1,
        "timing/batch_rollout_span_seconds": 45.0,
        "timing/batch_training_seconds": 90.0,
        "timing_train/batch_training_seconds": 12.0,
        "timing_train/forward_backward_wall_seconds": 5.0,
        "timing_train/forward_backward_compute_seconds": 4.0,
        "timing_train/optimizer_step_wall_seconds": 1.0,
        "timing_train/optimizer_step_compute_seconds": 0.5,
        "timing/trainer_wait_for_batch_seconds": 0.0,
        "timing_weights/weight_sync_seconds": 0.4,
        "timing/weight_export_seconds": 0.2,
        "timing/recompute_logprobs": 1.0,
        "timing/batch_completion_interval_seconds": 60.0,
        "model_request/count": 4,
    }
    snapshot = dict(raw)
    expected = {
        "06_timing_rollout/trajectory_mean_seconds": 12.5,
        "06_timing_rollout/group_mean_seconds": 22.0,
        "06_timing_rollout/model_request_max_seconds": 9.0,
        "07_timing_reward/custom_reward_verifier_mean_seconds": 3.0,
        "10_timing_val/custom_reward_group_mean_seconds": 24.0,
        "15_batch/trajectory_count": 4,
        "19_verifier/skipped_length_count": 1,
        "08_timing_train/batch_training_seconds": 12.0,
        "08_timing_train/forward_backward_wall_seconds": 5.0,
        "08_timing_train/forward_backward_compute_seconds": 4.0,
        "08_timing_train/optimizer_step_wall_seconds": 1.0,
        "08_timing_train/optimizer_step_compute_seconds": 0.5,
        "05_timing/trainer_wait_for_batch_seconds": 0.0,
        "09_timing_weights/weight_sync_seconds": 0.4,
        "09_timing_weights/weight_export_seconds": 0.2,
        "08_timing_train/recompute_logprobs_seconds": 1.0,
        "05_timing/batch_completion_interval_seconds": 60.0,
        "22_model_request/count": 4,
        "training/train_step": 7,
    }

    tracker.log(raw, step=7, commit=True)

    assert raw == snapshot
    assert run.logged == [(expected, 7)]
    assert run.history == [{**expected, "_step": 7}]
    assert run.dropped == []
    tracker.close()


def test_training_sync_and_validation_share_native_training_step_without_losing_metrics(wandb):
    tracker = _tracking()
    run = wandb.run
    tracker.log({"val/reward": 0.1}, step=0, commit=True)
    expected = [{"02_val/reward": 0.1, "training/train_step": 0, "_step": 0}]
    assert run.history == expected

    for step in range(1, 4):
        tracker.log({"loss": 1 / step}, step=step)
        tracker.log_live({"live/e2b/active_sandboxes": 2}, elapsed_seconds=step * 10)
        tracker.log({"timing_weights/weight_sync_seconds": step + 0.2}, step=step)
        assert run.history == expected
        row = {
            "loss": 1 / step,
            "09_timing_weights/weight_sync_seconds": step + 0.2,
            "training/train_step": step,
            "_step": step,
        }
        if step == 2:
            tracker.log({"val/reward": 0.2}, step=step, commit=True)
            row["02_val/reward"] = 0.2
        else:
            tracker.flush(step)
        expected.append(row)
        assert run.history == expected

    tracker.close()
    assert run.history == expected
    assert run.dropped == []
    assert all(call["step"] is not None for call in run.log_calls)
    assert [row["_step"] for row in run.history] == [0, 1, 2, 3]


def test_later_training_step_commits_previous_row_and_finish_preserves_last_row(wandb):
    tracker = _tracking()
    run = wandb.run
    tracker.log({"loss": 0.5}, step=1)
    tracker.log({"timing_weights/weight_sync_seconds": 1.2}, step=1)
    assert run.history == []

    tracker.log({"loss": 0.25}, step=2)
    assert run.history == [
        {
            "loss": 0.5,
            "09_timing_weights/weight_sync_seconds": 1.2,
            "training/train_step": 1,
            "_step": 1,
        }
    ]
    assert all(call["commit"] is False for call in run.log_calls)
    tracker.close()

    assert run.history[-1] == {"loss": 0.25, "training/train_step": 2, "_step": 2}
    assert len(run.history) == 2
    assert run.dropped == []


def test_flush_is_idempotent_and_never_commits_a_different_pending_step(wandb):
    tracker = _tracking()
    run = wandb.run
    tracker.flush(0)
    assert run.log_calls == []

    tracker.log({"loss": 0.5}, step=58)
    tracker.flush(57)
    tracker.flush(59)
    assert len(run.log_calls) == 1
    assert run.history == []
    tracker.log({"timing_weights/weight_sync_seconds": 1.2}, step=58)
    tracker.flush(58)
    assert run.log_calls[-1] == {"data": {}, "step": 58, "commit": True}
    assert run.history == [
        {
            "loss": 0.5,
            "09_timing_weights/weight_sync_seconds": 1.2,
            "training/train_step": 58,
            "_step": 58,
        }
    ]
    tracker.flush(58)
    assert len(run.log_calls) == 3

    tracker.log({"loss": 0.25}, step=59)
    tracker.flush(58)
    assert len(run.log_calls) == 4
    assert len(run.history) == 1
    tracker.log({"timing_weights/weight_sync_seconds": 1.5}, step=59)
    tracker.flush(59)
    tracker.flush(59)
    tracker.close()

    assert run.history[-1] == {
        "loss": 0.25,
        "09_timing_weights/weight_sync_seconds": 1.5,
        "training/train_step": 59,
        "_step": 59,
    }
    assert len(run.history) == 2
    assert len(run.log_calls) == 6
    assert run.dropped == []


def test_late_metrics_do_not_reopen_or_overwrite_committed_training_rows(wandb, caplog):
    tracker = _tracking()
    run = wandb.run
    tracker.log({"loss": 0.5}, step=1)
    tracker.log({"loss": 0.25}, step=2)

    with caplog.at_level(logging.WARNING, logger=tracking_module.__name__):
        tracker.log({"loss": 999}, step=1)
        tracker.flush(2)
        tracker.log({"loss": 999}, step=2)

    tracker.log({"loss": 0.1}, step=3)
    tracker.flush(3)
    tracker.close()

    assert run.history == [
        {"loss": 0.5, "training/train_step": 1, "_step": 1},
        {"loss": 0.25, "training/train_step": 2, "_step": 2},
        {"loss": 0.1, "training/train_step": 3, "_step": 3},
    ]
    assert len(run.log_calls) == 5
    assert run.dropped == []
    assert len(caplog.records) == 2
    assert all("Ignoring late W&B metrics" in record.message for record in caplog.records)


def test_validation_commit_makes_followup_flush_a_noop(wandb):
    tracker = _tracking()
    run = wandb.run
    tracker.log({"val/reward": 0.1}, step=0, commit=True)
    tracker.flush(0)
    tracker.log({"loss": 0.5}, step=1)
    tracker.log({"timing_weights/weight_sync_seconds": 1.2}, step=1)
    tracker.log({"val/reward": 0.2}, step=1, commit=True)
    tracker.flush(1)
    tracker.flush(0)

    assert len(run.log_calls) == 4
    assert run.log_calls[-1]["commit"] is True
    assert run.history == [
        {"02_val/reward": 0.1, "training/train_step": 0, "_step": 0},
        {
            "loss": 0.5,
            "09_timing_weights/weight_sync_seconds": 1.2,
            "02_val/reward": 0.2,
            "training/train_step": 1,
            "_step": 1,
        },
    ]
    assert run.dropped == []
    tracker.close()


@pytest.mark.parametrize("exit_code", [0, 1])
def test_resume_and_finish_preserve_pending_last_training_row(wandb, exit_code):
    tracker = _tracking()
    run = wandb.run
    tracker.log({"loss": 0.5}, step=58)
    tracker.log({"timing_weights/weight_sync_seconds": 1.2}, step=58)
    assert run.history == []
    tracker.close(exit_code=exit_code)
    tracker.flush(58)

    assert run.history == [
        {
            "loss": 0.5,
            "09_timing_weights/weight_sync_seconds": 1.2,
            "training/train_step": 58,
            "_step": 58,
        }
    ]
    assert run.finish_calls == [exit_code]
    assert run.dropped == []


def test_commit_and_flush_leave_non_wandb_backends_unchanged(wandb):
    tracker = _tracking()
    run = wandb.run
    console = SimpleNamespace(log=MagicMock())
    mlflow = SimpleNamespace(log=MagicMock())
    tracker.logger.update(console=console, mlflow=mlflow)
    original = {"loss": 0.5}
    tracker.log(original, step=9, backend=["console", "mlflow"], commit=True)
    tracker.flush(9)

    assert original == {"loss": 0.5}
    console.log.assert_called_once_with(data=original, step=9)
    mlflow.log.assert_called_once_with(data=original, step=9)
    assert run.log_calls == []
    tracker.close()


@pytest.mark.parametrize("backend_order", [("wandb", "console"), ("console", "wandb")])
def test_numbered_wandb_names_do_not_leak_to_console_or_change_step_commits(wandb, backend_order):
    tracker = _tracking()
    run = wandb.run
    console = SimpleNamespace(log=MagicMock())
    backends = {"wandb": run, "console": console}
    tracker.logger = {name: backends[name] for name in backend_order}
    raw = {
        "trajectory/reward_mean": 0.6,
        "algorithm/trajectory/reward_mean": 0.7,
        "mis/token_rejection_fraction": 0.25,
        "batch/trajectory_count": 8,
        "reward_fn/org/code_v2/reward_mean": 0.8,
        "reward_fn/org/code_v2/reward_observed_count": 7,
        "val/org/math/reward_mean": 0.9,
        "val/org/math/custom_observed_count": 0,
        "val/org/math/complete_group_ratio": 0.5,
        "timing_rollout/org/code_v2_verifier_mean_seconds": 4.0,
        "timing_val/org/code_v2_verifier_mean_seconds": 5.0,
        "custom/quality": 0.2,
        "live/tool/errors": 99,
        "weaver/scale": 99,
        "algorithm/trajectory/advantage_std": None,
        "custom/nonfinite": float("nan"),
    }
    snapshot = dict(raw)
    canonical = {
        "algorithm/trajectory/reward_mean": 0.7,
        "algorithm/mis/token_rejection_fraction": 0.25,
        "batch/trajectory_count": 8,
        "reward_fn/org/code_v2/reward_mean": 0.8,
        "reward_fn/org/code_v2/reward_observed_count": 7,
        "val/org/math/reward_mean": 0.9,
        "val/org/math/custom_observed_count": 0,
        "val/org/math/complete_group_ratio": 0.5,
        "timing_rollout/org/code_v2_verifier_mean_seconds": 4.0,
        "timing_val/org/code_v2_verifier_mean_seconds": 5.0,
        "custom/quality": 0.2,
    }
    numbered = {
        "01_algorithm/01_trajectory/reward_mean": 0.7,
        "12_mis/token_rejection_fraction": 0.25,
        "15_batch/trajectory_count": 8,
        "03_reward_fn/01_reward_mean/org/code_v2": 0.8,
        "16_count_reward/org/code_v2/reward_observed_count": 7,
        "02_val/org/math/reward_mean": 0.9,
        "17_count_val/org/math/custom_observed_count": 0,
        "18_status_val/org/math/complete_group_ratio": 0.5,
        "07_timing_reward/org/code_v2_verifier_mean_seconds": 4.0,
        "10_timing_val/org/code_v2_verifier_mean_seconds": 5.0,
        "custom/quality": 0.2,
    }

    tracker.log(raw, step=11)

    assert raw == snapshot
    console.log.assert_called_once_with(data=canonical, step=11)
    assert run.log_calls == [
        {"data": {**numbered, "training/train_step": 11}, "step": 11, "commit": False}
    ]
    assert run.history == []
    tracker.log({"timing_weights/weight_sync_seconds": 0.4}, step=11, commit=True)
    tracker.flush(11)

    assert console.log.call_args_list[-1].kwargs == {
        "data": {"timing_weights/weight_sync_seconds": 0.4},
        "step": 11,
    }
    assert console.log.call_count == 2
    assert len(run.log_calls) == 2
    assert run.log_calls[-1] == {
        "data": {"09_timing_weights/weight_sync_seconds": 0.4, "training/train_step": 11},
        "step": 11,
        "commit": True,
    }
    assert run.history == [
        {
            **numbered,
            "09_timing_weights/weight_sync_seconds": 0.4,
            "training/train_step": 11,
            "_step": 11,
        }
    ]
    definitions = dict(run.metric_definitions)
    assert definitions["training/train_step"]["hidden"] is True
    assert "14_training/train_step" not in definitions
    assert definitions["15_batch/trajectory_count"]["step_metric"] == "training/train_step"
    assert run.dropped == []
    tracker.close()


def test_close_disables_training_metrics_and_live_logs(wandb, caplog):
    tracker = _tracking()
    run = wandb.run
    other = SimpleNamespace(log=MagicMock())
    tracker.logger["console"] = other
    with caplog.at_level(logging.INFO, logger=tracking_module.__name__):
        tracker.log({"loss": 0.5}, step=4)
        tracker.log_live({"live/e2b/active_sandboxes": 1}, elapsed_seconds=10)
        tracker.close()
        tracker.log({"loss": 0}, step=5)
        tracker.log_live({"live/e2b/active_sandboxes": 0}, elapsed_seconds=20)

    other.log.assert_called_once_with(data={"loss": 0.5}, step=4)
    assert run.logged == [({"loss": 0.5, "training/train_step": 4}, 4)]
    assert len(caplog.records) == 1
    assert '"live/e2b/active_sandboxes": 1' in caplog.records[0].message


@pytest.mark.parametrize("enable_wandb", [False, True])
def test_live_logging_does_not_wait_for_another_backend(wandb, enable_wandb, caplog):
    tracker = _tracking() if enable_wandb else Tracking("test", "test", [], {})
    started, release, live_returned = Event(), Event(), Event()

    def blocked_log(**_kwargs):
        started.set()
        assert release.wait(5)

    tracker.logger["mlflow"] = SimpleNamespace(log=blocked_log)
    writer = Thread(target=lambda: tracker.log({"loss": 1}, 1, backend=["mlflow"]), daemon=True)
    writer.start()
    try:
        assert started.wait(2)
        with caplog.at_level(logging.INFO, logger=tracking_module.__name__):
            caller = Thread(
                target=lambda: (
                    tracker.log_live({"live/e2b/completed_rollouts": 1}, 1),
                    live_returned.set(),
                ),
                daemon=True,
            )
            caller.start()
            assert live_returned.wait(1)
            caller.join(2)
        assert len(caplog.records) == 1
        assert '"live/e2b/completed_rollouts": 1' in caplog.records[0].message
        if enable_wandb:
            assert wandb.run.logged == []
    finally:
        release.set()
        writer.join(2)
        tracker.close()


def test_close_defers_finish_until_inflight_training_write_ends_without_blocking(wandb):
    tracker = _tracking()
    run = wandb.run
    original_log = run.log
    started, release, closed = Event(), Event(), Event()

    def blocked_log(**kwargs):
        started.set()
        assert release.wait(5)
        original_log(**kwargs)

    run.log = blocked_log
    writer = Thread(target=lambda: tracker.log({"loss": 1}, 1), daemon=True)
    writer.start()
    try:
        assert started.wait(2)
        closer = Thread(target=lambda: (tracker.close(exit_code=1), closed.set()), daemon=True)
        closer.start()
        assert closed.wait(1)
        assert not run.finished
        tracker.log({"loss": 0}, 3)
        release.set()
        writer.join(2)
        closer.join(2)
        assert run.finish_event.wait(2)
        tracker.close(exit_code=0)
        assert run.finish_calls == [1]
        assert run.logged == [({"loss": 1, "training/train_step": 1}, 1)]
        assert run.history == [{"loss": 1, "training/train_step": 1, "_step": 1}]
        assert run.dropped == []
    finally:
        release.set()
        writer.join(2)
        tracker.close()


def test_close_waits_for_swanlab_write_without_blocking_controller_or_logging_after_finish():
    tracker = Tracking("test", "test", [], {})
    started, release, finished = Event(), Event(), Event()
    events = []

    def blocked_log(**_kwargs):
        events.append("log-start")
        started.set()
        assert release.wait(5)
        assert not finished.is_set()
        events.append("log-end")

    def finish():
        events.append("finish")
        finished.set()

    tracker.logger["swanlab"] = SimpleNamespace(log=blocked_log, finish=finish)
    writer = Thread(target=lambda: tracker.log({"loss": 1}, 1), daemon=True)
    writer.start()
    try:
        assert started.wait(2)
        tracker.close()
        assert not finished.is_set()
        tracker.log({"loss": 2}, 2)
        release.set()
        writer.join(2)
        assert finished.wait(2)
        assert events == ["log-start", "log-end", "finish"]
    finally:
        release.set()
        tracker.close()


def test_explicit_missing_numbered_metric_is_not_refilled_from_an_old_spelling(wandb):
    tracker = _tracking()
    run = wandb.run
    raw = {
        "trajectory/reward_mean": 0.5,
        "01_algorithm/01_trajectory/reward_mean": None,
        "training/train_step": 999,
    }

    tracker.log(raw, step=7, commit=True)

    assert run.history == [{"training/train_step": 7, "_step": 7}]
    assert raw["trajectory/reward_mean"] == 0.5
    assert raw["01_algorithm/01_trajectory/reward_mean"] is None
    assert raw["training/train_step"] == 999
    tracker.close()


def _weaver_observations(value=1.5, *, attempt=5, model="actor", epoch="run-a"):
    from weaver.types.metrics import MetricObservation, MetricObservations

    return MetricObservations(
        model,
        attempt,
        epoch,
        "trainer_lifetime",
        (
            MetricObservation("entropy/top10_mean", "ok", value, {}),
            MetricObservation("grad/norm/total", "ok", 2.5, {}),
            MetricObservation("future/diagnostic", "ok", 7.0, {"layer": "decoder/2"}),
            MetricObservation("optim/skipped_step", "ok", False, {}),
        ),
    )


def test_weaver_rows_preserve_every_window_and_do_not_interfere_with_rl(wandb):
    tracker = _tracking()
    run = wandb.run
    tracker.log({"loss": 0.75}, step=1)
    tracker.queue_weaver_metrics("fb-1", _weaver_observations(1.0, attempt=1))
    tracker.queue_weaver_metrics("opt-1", _weaver_observations(2.0, attempt=1))
    tracker.queue_weaver_metrics("opt-2", _weaver_observations(4.0, attempt=2))
    assert run.history == []  # No SDK callback may commit a partial NorthRL round.
    tracker.log({"reward": 3.0}, step=1)
    tracker.flush(1)
    tracker.log({"loss": 0.25}, step=2)
    tracker.flush(2)
    assert not run.dropped
    assert run.history[0] == {"loss": 0.75, "reward": 3.0, "training/train_step": 1, "_step": 1}
    assert run.history[-1]["loss"] == 0.25
    assert run.history[-1]["training/train_step"] == 2
    rows = run.history[1:-1]
    assert [row["weaver_metrics_v1/entropy/top10_mean"] for row in rows] == [1.0, 2.0, 4.0]
    assert [row["weaver_metrics_v1/step"] for row in rows] == [1, 1, 2]
    assert all(row["weaver_metrics_v1/future/diagnostic/decoder/2"] == 7.0 for row in rows)
    assert all(type(row["weaver_metrics_v1/optim/skipped_step"]) is int for row in rows)
    assert all("training/train_step" not in row for row in rows)
    assert (
        "weaver_metrics_v1/entropy/top10_mean",
        {
            "step_metric": "weaver_metrics_v1/step",
            "step_sync": False,
        },
    ) in run.metric_definitions
    assert len(wandb.init_calls) == 1
    # Late RL metrics still obey the existing guard despite extra W&B history rows.
    tracker.log({"loss": 99}, step=1)
    assert run.history[-1]["loss"] == 0.25
    tracker.close()
    assert run.finish_calls == [0]


@pytest.mark.parametrize("local_enabled", [False, True])
def test_sdk_completion_bridges_once_with_independent_local_storage(wandb, tmp_path, local_enabled):
    import asyncio

    from weaver import AsyncOperationHandle, AsyncServiceClient, MetricsStoreConfig

    tracker = _tracking()
    client = AsyncServiceClient(
        metrics_store=MetricsStoreConfig(enabled=local_enabled, path=tmp_path),
        metric_observer=tracker.queue_weaver_metrics,
        wandb_link=None,
    )
    raw = {
        "result": {
            "metrics": {"loss": 0.5},
            "metric_observations": {
                "version": 1,
                "model_id": "actor",
                "attempt": 5,
                "epoch": "run-a",
                "counter_scope": "trainer_lifetime",
                "points": [
                    {"name": "entropy/top10_mean", "status": "ok", "value": 1.25, "labels": {}},
                ],
            },
        }
    }
    handle = AsyncOperationHandle(
        SimpleNamespace(metric_sink=client._metric_persistence),
        "operation-1",
        _cached={"status": "done", "response": raw},
    )
    assert asyncio.run(handle.result()) == raw
    assert asyncio.run(handle.result()) == raw
    tracker.log({"loss": 0.5}, step=1, commit=True)
    rows = [r for r in wandb.run.history if "weaver_metrics_v1/entropy/top10_mean" in r]
    assert len(rows) == 1
    assert rows[0]["weaver_metrics_v1/entropy/top10_mean"] == 1.25
    files = list(tmp_path.rglob("observations.jsonl"))
    assert len(files) == int(local_enabled)
    if files:
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(records) == 1 and records[0]["step"] == 5
        assert records[0]["trainer_run_id"] == "run-a"
    client._metric_persistence.close()
    assert not wandb.run.finished  # SDK never takes ownership of the caller's run.
    tracker.close()


def test_weaver_forward_restart_and_shutdown_are_not_dropped(wandb):
    tracker = _tracking()
    tracker.queue_weaver_metrics("forward", _weaver_observations(attempt=None))
    tracker.queue_weaver_metrics("restart", _weaver_observations(epoch="run-b", attempt=1))
    tracker.queue_weaver_metrics("reference", _weaver_observations(model="reference", attempt=None))
    tracker.log({"loss": 0.5}, step=1)
    tracker.close()
    run = wandb.runs[0]
    assert run.history[0]["loss"] == 0.5
    assert run.history[1]["weaver_metrics_v1/forward/operation_index"] == 1
    assert run.history[2]["weaver_metrics_v1/restart_2/step"] == 1
    assert run.history[3]["weaver_metrics_v1/model_2/forward/operation_index"] == 2
    assert not run.dropped


def test_no_wandb_does_not_buffer_weaver_metrics():
    tracker = Tracking("p", "r", default_backend=[], config={})
    tracker.queue_weaver_metrics("op", _weaver_observations())
    assert not tracker._weaver_metrics
    tracker.close()


def test_weaver_rows_do_not_merge_successive_unflushed_rl_rounds(wandb):
    tracker = _tracking()
    for index in range(3):
        tracker.queue_weaver_metrics(str(index), _weaver_observations(attempt=index + 1))
    tracker.log({"loss": 1}, step=1, commit=True)
    tracker.log({"loss": 2, "only_round_two": 42}, step=2)
    tracker.log({"loss": 3}, step=3)  # Existing implicit-commit behavior.
    tracker.flush(3)
    rows = [row for row in wandb.run.history if "training/train_step" in row]
    assert [row["loss"] for row in rows] == [1, 2, 3]
    assert [row["training/train_step"] for row in rows] == [1, 2, 3]
    assert "only_round_two" not in rows[2]
    assert not wandb.run.dropped
    tracker.close()
