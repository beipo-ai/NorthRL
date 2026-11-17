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

"""Tests for forwarding holder version failures through the remote API client."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.exceptions import WeightVersionMismatchError
from northrl.inference_service_client.remote_api_inference_service_client import (
    RemoteApiInferenceServiceClient,
)
from northrl.utils.async_utils import run_coro
from northrl.weight_sync.weight_sync_controller import WeightSyncController


class _RemoteClient(RemoteApiInferenceServiceClient):
    def set_service_holder(self, service_holder) -> None:
        self._service_holder = service_holder


_REQUESTS = [
    ("completion", "hello", "sample_from_prompt"),
    ("generate", [{"role": "user", "content": "hello"}], "sample_from_messages"),
    ("generate_with_token", [1], "sample_from_token_ids"),
]


@pytest.mark.parametrize(("method_name", "argument", "holder_method"), _REQUESTS)
@pytest.mark.parametrize("async_holder", [False, True])
@pytest.mark.parametrize("with_controller", [False, True])
def test_holder_mismatch_propagates_without_controller_changes(
    rollout_worker_config, method_name, argument, holder_method, async_holder, with_controller
):
    client = _RemoteClient(rollout_worker_config)
    error = WeightVersionMismatchError("Weaver version mismatch: expected='v1', actual='v0'")
    sample = (AsyncMock if async_holder else Mock)(side_effect=error)
    client.set_service_holder(SimpleNamespace(**{holder_method: sample}))
    controller = Mock(spec=[])
    if with_controller:
        client.set_weight_sync_controller(controller)

    with pytest.raises(WeightVersionMismatchError) as raised:
        run_coro(getattr(client, method_name)(argument))

    assert raised.value is error
    sample.assert_called_once()
    if async_holder:
        sample.assert_awaited_once()
    assert controller.method_calls == []


@pytest.mark.parametrize(("method_name", "argument", "holder_method"), _REQUESTS)
@pytest.mark.parametrize("response_version", [None, "northrl_actor_v000001"])
def test_client_preserves_only_the_holder_response_version(
    rollout_worker_config, method_name, argument, holder_method, response_version
):
    client = _RemoteClient(rollout_worker_config)
    result = {
        "response": "ok",
        "prompt_tokens": [1],
        "response_tokens": [2],
        "response_logprobs": [-0.1],
        "finish_reason": "stop",
    }
    if response_version is not None:
        result["weight_version"] = response_version
    sample = AsyncMock(return_value=result)
    # No version getter: the wrapper uses the result of this request alone.
    client.set_service_holder(SimpleNamespace(**{holder_method: sample}))
    controller = Mock(spec=[])
    client.set_weight_sync_controller(controller)

    response = run_coro(getattr(client, method_name)(argument))

    if response_version is None:
        assert "weight_version" not in response
    else:
        assert response["weight_version"] == response_version
    assert controller.method_calls == []
    sample.assert_awaited_once()


@pytest.mark.parametrize(("method_name", "argument", "holder_method"), _REQUESTS)
@pytest.mark.parametrize("async_holder", [False, True])
def test_other_holder_failures_propagate_without_controller_changes(
    rollout_worker_config, method_name, argument, holder_method, async_holder
):
    client = _RemoteClient(rollout_worker_config)
    error = RuntimeError("Weaver service unavailable")
    sample = (AsyncMock if async_holder else Mock)(side_effect=error)
    client.set_service_holder(SimpleNamespace(**{holder_method: sample}))
    controller = Mock(spec=[])
    client.set_weight_sync_controller(controller)

    with pytest.raises(RuntimeError) as raised:
        run_coro(getattr(client, method_name)(argument))

    assert raised.value is error
    assert controller.method_calls == []
    sample.assert_called_once()
    if async_holder:
        sample.assert_awaited_once()


@pytest.mark.parametrize(("method_name", "argument", "holder_method"), _REQUESTS)
@pytest.mark.parametrize("freeze_inflight", [False, True])
def test_fully_async_requests_during_weight_sync(
    rollout_worker_config, method_name, argument, holder_method, freeze_inflight, monkeypatch
):
    rollout_worker_config.training.weight_sync.mode = "fully-async"
    rollout_worker_config.training.weight_sync.freeze_inflight = freeze_inflight
    rollout_worker_config.validate.schedule.every_n_steps = 0
    controller = WeightSyncController(rollout_worker_config)
    service = controller._rollout_services["actor"]
    service.state = "syncing"
    client = _RemoteClient(rollout_worker_config)
    client.set_weight_sync_controller(controller)
    sample = AsyncMock(
        return_value={
            "response": "ok",
            "prompt_tokens": [1],
            "response_tokens": [2],
            "response_logprobs": [-0.1],
            "finish_reason": "stop",
        }
    )
    client.set_service_holder(SimpleNamespace(**{holder_method: sample}))

    async def request_during_sync():
        blocked = asyncio.Event()
        original_check = controller.check_rollout_service_status
        observed_statuses = []

        def check_status(model_tag):
            status = original_check(model_tag)
            observed_statuses.append(status)
            if status == "block":
                blocked.set()
            return status

        monkeypatch.setattr(controller, "check_rollout_service_status", check_status)
        request = asyncio.create_task(getattr(client, method_name)(argument))
        try:
            if freeze_inflight:
                await asyncio.wait_for(blocked.wait(), timeout=2)
                sample.assert_not_awaited()
                service.state = "running"
            await asyncio.wait_for(request, timeout=2)
            if freeze_inflight:
                assert observed_statuses[0] == "block"
                assert observed_statuses[-1] == "continue"
            else:
                assert service.state == "syncing"
                assert observed_statuses == []
            sample.assert_awaited_once()
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)

    run_coro(request_during_sync())
