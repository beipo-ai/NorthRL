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

"""Actor temperature reaches real SDK payload builders without remote operations."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from weaver._payloads import forward_backward_payload, forward_payload
from weaver.async_training_client import AsyncTrainingClient

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import KLConfig
from northrl.weaver.weaver_service_holder import WeaverServiceHolder
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyDataDumper,
    _DummyRemoteApiTrainer,
    _make_config,
)


class _Operation:
    def __init__(self, result):
        self._result = result

    async def result(self):
        return self._result


class _RecordingSDKClient:
    """Replace remote execution only; serialize real Datums and execute SDK custom loss."""

    forward_backward_custom = AsyncTrainingClient.forward_backward_custom

    def __init__(self, fail_forward_at=None):
        self.requests = []
        self.datums = []
        self.forward_count = 0
        self.optim_count = 0
        self.fail_forward_at = fail_forward_at

    async def forward(self, data, loss_fn, *, loss_fn_config=None, wait):
        self.forward_count += 1
        payload = forward_payload(
            model_id="offline-temperature-test",
            seq_id=len(self.requests),
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            request_metadata=None,
        )
        self.requests.append(("forward", payload["forward_input"]))
        self.datums.append(data)
        if self.forward_count == self.fail_forward_at:
            raise RuntimeError("injected read-only forward failure")
        uri = f"weaver://offline-temperature-test/router-replay/record-{self.forward_count}"
        result = {
            "result": {
                "loss_fn_outputs": [
                    {"logprobs": [-0.25] * len(datum.loss_fn_inputs["target_tokens"])}
                    for datum in data
                ],
                "metrics": {
                    "router_replay_recorded_indices": {
                        "schema": "weaver.router_replay.index_set.v1",
                        "transport": "gpfs_safetensors",
                        "uri": uri,
                        "manifest_uri": f"{uri}/manifest.json",
                    }
                },
            }
        }
        return result if wait else _Operation(result)

    async def forward_backward(self, data, loss_fn, *, loss_fn_config=None, wait):
        payload = forward_backward_payload(
            model_id="offline-temperature-test",
            seq_id=len(self.requests),
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            request_metadata=None,
        )
        self.requests.append(("backward", payload["forward_backward_input"]))
        self.datums.append(data)
        result = {"result": {"metrics": {"loss": 0.1}}}
        return result if wait else _Operation(result)

    async def optim_step(self, params, *, wait):
        self.optim_count += 1
        result = {"result": {"metrics": {}}}
        return result if wait else _Operation(result)


def _trainer(monkeypatch, temperature, *, mode=None, ahead=0, custom=False, mini_batch_size=2):
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda datums: None)
    config = _make_config(batch_size=None, mini_batch_size=mini_batch_size)
    config.algorithm.loss.mode = "ppo_clip"
    config.algorithm.sampling_logprob_source = "recomputed"
    config.algorithm.loss.params.clip.ratio_low = 0.2
    config.algorithm.loss.params.clip.ratio_high = 0.2
    config.algorithm.kl = KLConfig().model_dump()
    config.algorithm.router_replay = {"enabled": mode is not None, "mode": mode or "R3"}
    config.algorithm.optim.submit_ahead = ahead
    if custom:
        config.algorithm.loss.mode = "custom"
        config.algorithm.loss.params.name = "direct_logprob"
        config.algorithm.loss.params.base_loss_fn = "ppo_clip"
    config.models = {"reference": None}
    config.backend = {"type": "weaver"}
    config.rollout = {"sampling": {"temperature": temperature}}
    # Deliberately disagree with training; this must never affect the actor's temperature.
    config.validate = {"sampling": {"temperature": 0.7 if temperature == 0 else 0.0}}
    trainer = RemoteApiGrpoTrainer(config)
    trainer._need_sampling = False
    return trainer


def _holder(*, fail_forward_at=None):
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._closed = False
    holder._ensure_provisioned = AsyncMock()
    holder._training_client = _RecordingSDKClient(fail_forward_at)
    return holder


def _trajectories(*, masks=False):
    trajectories = []
    for index in range(4):
        extra = {
            "group_id": f"group-{index // 2}",
            "run_id": index % 2,
            "expected_rollouts": 2,
            "model_tag": "actor",
            "old_log_probs": [0.0, -0.5, -0.5],
            "moe_topk_indices_ref": {"relative_path": f"rollout-{index}.safetensors"},
        }
        if masks:
            extra["sampling_mask"] = [[20, 21], [30, 31]]
        trajectories.append(
            Trajectory(
                tokens=[10 + index, 20, 30],
                loss_mask=[0, 1, 1],
                reward=float(index % 2),
                extra_fields=extra,
            )
        )
    return trajectories


@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
@pytest.mark.parametrize("mode", [None, "R2", "R3"])
@pytest.mark.parametrize("ahead", [0, 1])
@pytest.mark.parametrize("masks", [False, True])
def test_grpo_round_sends_temperature_to_forward_and_backward_payloads(
    monkeypatch, temperature, mode, ahead, masks
):
    trainer = _trainer(monkeypatch, temperature, mode=mode, ahead=ahead)
    holder = _holder()
    trainer.set_service_holder(holder)
    trajectories = _trajectories(masks=masks)
    original_tokens = [list(trajectory.tokens) for trajectory in trajectories]

    metrics, _, _ = run_coro(trainer.train(trajectories))

    client = holder._training_client
    assert [stage for stage, _ in client.requests] == ["forward", "forward", "backward", "backward"]
    assert client.optim_count == 1
    assert metrics["training/optimizer_step_count"] == 1
    assert [trajectory.tokens for trajectory in trajectories] == original_tokens
    for stage, request in client.requests:
        assert request["loss_fn_config"]["temperature"] == temperature
        if stage == "forward":
            assert request["loss_fn"] == "forward_logprob"
            assert request["loss_fn_config"] == {"temperature": temperature}
        else:
            assert request["loss_fn"] == "ppo_clip"
            assert request["loss_fn_config"] == {
                "temperature": temperature,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "entropy_coeff": 0.0,
                "enable_debug_dump": False,
                "loss_agg_mode": "sum",
            }
    for forward_chunk, backward_chunk in zip(client.datums[:2], client.datums[2:], strict=True):
        for forward, backward in zip(forward_chunk, backward_chunk, strict=True):
            assert forward.loss_fn_inputs["target_tokens"].tolist() == [20, 30]
            assert backward.loss_fn_inputs["logprobs"].tolist() == [-0.25, -0.25]
            if masks:
                assert forward.loss_fn_inputs["sampling_mask"].tolist() == [[20, 21], [30, 31]]
                assert backward.loss_fn_inputs["sampling_mask"].tolist() == [[20, 21], [30, 31]]
            else:
                assert "sampling_mask" not in forward.loss_fn_inputs
                assert "sampling_mask" not in backward.loss_fn_inputs
            if mode == "R3":
                assert backward.metadata == forward.metadata
                assert forward.metadata["router_replay"]["action"] == "REPLAY"
            elif mode == "R2":
                assert forward.metadata["router_replay"]["action"] == "RECORD"
                assert backward.metadata["router_replay"]["action"] == "REPLAY"


@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
@pytest.mark.parametrize("mode", [None, "R2", "R3"])
def test_recompute_retry_retains_temperature_and_only_retries_failed_chunk(
    monkeypatch, temperature, mode
):
    trainer = _trainer(monkeypatch, temperature, mode=mode)
    holder = _holder(fail_forward_at=2)
    trainer.set_service_holder(holder)
    run_coro(trainer._recompute_logprobs(_trajectories(), {}))
    client = holder._training_client
    assert len(client.requests) == 3
    assert client.requests[1][1] == client.requests[2][1]
    assert client.requests[0][1]["data"] != client.requests[1][1]["data"]
    assert all(
        request["loss_fn_config"] == {"temperature": temperature} for _, request in client.requests
    )
    assert client.optim_count == 0


@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
def test_diagnostic_forward_uses_temperature_without_replacing_denominators(
    monkeypatch, temperature
):
    trainer = _trainer(monkeypatch, temperature)
    trainer._mis_config.get_mismatch_metrics = True
    holder = _holder()
    trainer.set_service_holder(holder)
    trajectories = _trajectories()
    before = deepcopy(trajectories)
    metrics = {}
    run_coro(trainer._compute_inference_mismatch_metrics(trajectories, metrics))
    assert trajectories == before
    assert metrics["mis/metrics_available"] == 1
    assert len(holder._training_client.requests) == 2
    for _, request in holder._training_client.requests:
        assert request["loss_fn_config"] == {"temperature": temperature}


@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
@pytest.mark.parametrize("mini_batch_size", [2, 4])
def test_custom_loss_forward_and_surrogate_backward_share_temperature(
    monkeypatch, temperature, mini_batch_size
):
    trainer = _trainer(monkeypatch, temperature, custom=True, mini_batch_size=mini_batch_size)
    holder = _holder()
    trainer.set_service_holder(holder)
    run_coro(trainer.train(_trajectories()))
    client = holder._training_client
    chunks = 4 // mini_batch_size
    assert [stage for stage, _ in client.requests] == ["forward"] * (chunks * 2) + [
        "backward"
    ] * chunks
    for stage, request in client.requests:
        assert request["loss_fn_config"] == {"temperature": temperature}
        assert request["loss_fn"] == ("forward_logprob" if stage == "forward" else "surrogate")
    assert client.optim_count == 1


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_temperature_is_rejected_before_remote_work(monkeypatch, temperature):
    with pytest.raises(ValueError, match="temperature must be finite and >= 0"):
        _trainer(monkeypatch, temperature)


def test_temperature_config_is_fresh_and_does_not_mutate_other_loss_fields(monkeypatch):
    trainer = _trainer(monkeypatch, 0.0)
    first = trainer._build_forward_loss_fn_config()
    first["temperature"] = 99.0
    first["clip_ratio_low"] = 42.0
    assert trainer._build_forward_loss_fn_config() == {"temperature": 0.0}
    assert trainer._build_loss_fn_config()["temperature"] == 0.0
    assert trainer._build_loss_fn_config()["clip_ratio_low"] == 0.2
    trainer._kl_enabled = True
    trainer._kl_coeff = 0.1
    trainer._kl_estimator = "k3"
    assert trainer._build_loss_fn_config()["kl_coeff"] == 0.1
    assert trainer._build_loss_fn_config()["kl_estimator"] == "k3"


@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
def test_temperature_preserves_sampling_mask_refs_r3_metadata_and_caller_config(temperature):
    holder = _holder()
    mask_ref = {"sampling_mask_ref": {"uri": "weaver://offline/mask", "future_field": 1}}
    routing = {"router_replay": {"mode": "R3", "value_ref": {"uri": "weaver://offline/routing"}}}
    config = {"temperature": temperature}
    before = deepcopy((mask_ref, routing, config))
    result = run_coro(
        holder.compute_target_logprobs_batch(
            [[10, 20, 30]], [mask_ref], [routing], loss_fn_config=config
        )
    )
    assert result == [[-0.25, -0.25]]
    datum = holder._training_client.datums[0][0]
    assert datum.metadata == {**mask_ref, **routing}
    assert "sampling_mask" not in datum.loss_fn_inputs
    assert holder._training_client.requests[0][1]["loss_fn_config"] == config
    assert (mask_ref, routing, config) == before


def test_omitted_temperature_uses_schema_default_without_validation_leakage(monkeypatch):
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    config = _make_config()
    config.algorithm.kl = KLConfig().model_dump()
    config.models = {"reference": None}
    config.validate = {"sampling": {"temperature": 0.7}}
    trainer = RemoteApiGrpoTrainer(config)
    assert trainer._build_forward_loss_fn_config() == {"temperature": 1.0}
    assert trainer._build_loss_fn_config()["temperature"] == 1.0


def test_other_trainer_keeps_no_forward_override_and_original_backward_config(monkeypatch):
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    config = _make_config()
    config.rollout = {"sampling": {"temperature": 0.7}}
    trainer = _DummyRemoteApiTrainer(config)
    assert trainer._build_forward_loss_fn_config() is None
    assert "temperature" not in trainer._build_loss_fn_config()
