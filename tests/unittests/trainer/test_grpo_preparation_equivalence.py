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

"""GRPO preparation preserves batching, final-mask counts and failure boundaries."""

import math
from array import array
from unittest.mock import patch

import pytest

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import KLConfig
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyActivityTracker,
    _DummyDataDumper,
    _DummyRemoteApiTrainer,
    _FakeWeaverHolder,
    _FakeWeightSyncController,
    _make_config,
    _make_trajectory,
)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
def test_32_by_16_batch_retains_recompute_normalization_and_single_update(
    monkeypatch, packed, temperature
):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    config = _make_config(batch_size=None, mini_batch_size=512)
    config.algorithm.kl = KLConfig().model_dump()
    config.models = {"reference": None}
    config.algorithm.sampling_logprob_source = "recomputed"
    config.algorithm.loss.mode = "ppo_clip"
    config.rollout = {"sampling": {"temperature": temperature}}
    trainer = RemoteApiGrpoTrainer(config)
    trainer._activity_tracker = _DummyActivityTracker()
    trainer._weight_sync_controller = _FakeWeightSyncController()
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    def unexpected_recount(datum):
        raise AssertionError("Reuse main's final-mask counts instead of rescanning the datum")

    monkeypatch.setattr(trainer_module, "datum_response_token_count", unexpected_recount)
    trajectories = []
    for group in range(32):
        for run in range(16):
            length = 3 + run % 4
            trajectories.append(
                Trajectory(
                    tokens=[1, 2] + [3] * length,
                    loss_mask=[0, 0] + [1] * length,
                    reward=float(run % 2),
                    extra_fields={
                        "group_id": f"q{group}",
                        "run_id": run,
                        "expected_rollouts": 16,
                        "rollout_id": f"q{group}-r{run}",
                        "model_tag": "actor",
                        "old_log_probs": [0.0, 0.0] + [-0.5] * length,
                    },
                )
            )
    if packed:
        for trajectory in trajectories:
            trajectory.tokens = array("q", trajectory.tokens)
            trajectory.loss_mask = array("b", trajectory.loss_mask)
            trajectory["old_log_probs"] = array("d", trajectory["old_log_probs"])
    with patch.object(trainer_module.logger, "info", wraps=trainer_module.logger.info) as log_info:
        metrics, _, _ = run_coro(trainer.train(trajectories))
    preparation_logs = [
        call.args[0] % call.args[1:]
        for call in log_info.call_args_list
        if call.args[0].startswith("[training_preparation]")
    ]
    assert len(preparation_logs) == 3
    for message, stage in zip(preparation_logs, ("conversion", "metrics", "normalization")):
        assert f"stage={stage} " in message
        assert float(message.split("wall_seconds=")[1]) >= 0
    assert len(holder.compute_logprobs_calls) == 1
    assert len(holder.builtin_calls) == 1
    assert len(holder.optim_calls) == 1
    assert holder.compute_logprobs_calls[0]["loss_fn_config"] == {"temperature": temperature}
    assert holder.builtin_calls[0]["loss_fn_config"]["temperature"] == temperature
    assert holder.builtin_calls[0]["loss_fn_config"]["loss_agg_mode"] == "sum"
    datums = holder.builtin_calls[0]["datums_data"]
    assert len(datums) == 512
    assert metrics["batch/trainable_group_count"] == 32
    assert metrics["batch/trainable_trajectory_count"] == 512
    assert metrics["batch/trainable_response_tokens_total"] == 32 * (3 + 4 + 5 + 6) * 4
    advantage = -0.5 / (math.sqrt(4 / 15) + 1e-6)
    assert list(datums[0]["loss_fn_inputs"]["loss_mask"]) == [0.0, 1.0, 1.0, 1.0]
    assert datums[0]["loss_fn_inputs"]["advantages"] == pytest.approx(
        [0.0] + [advantage / (512 * 3)] * 3
    )
    if packed:
        assert all(isinstance(trajectory["old_log_probs"], array) for trajectory in trajectories)
        assert all(isinstance(datum["input_tokens"], array) for datum in datums)
        assert all(isinstance(datum["loss_fn_inputs"]["advantages"], array) for datum in datums)


def test_conversion_error_propagates_without_submitting_an_update(monkeypatch):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    error = ValueError("original converter failure")

    def fail(trajectories):
        raise error

    monkeypatch.setattr(trainer, "_convert_to_datums", fail)
    with pytest.raises(ValueError) as caught:
        run_coro(trainer.train([_make_trajectory()]))
    assert caught.value is error
    assert not holder.builtin_calls
    assert not holder.optim_calls
