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

"""Weight publication counters without controller version labels."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from northrl.utils.async_utils import run_coro
from northrl.weaver.weaver_service_holder import WeaverServiceHolder
from northrl.weight_sync.weight_sync_controller import WeightSyncController


def _make_controller(*, mode="sync", threshold=0):
    return WeightSyncController(
        OmegaConf.create(
            {
                "training": {"weight_sync": {"mode": mode, "staleness_threshold": threshold}},
                "backend": {"type": "weaver"},
                "validate": {"schedule": {"every_n_steps": 0}},
            }
        )
    )


@pytest.fixture
def make_holder():
    holders = []

    def make():
        holder = WeaverServiceHolder(base_model="actor", sampler_ttl_seconds=None)
        holders.append(holder)
        holder._ensure_provisioned = AsyncMock()
        holder._training_client = SimpleNamespace(
            save_weights_for_sampler=AsyncMock(return_value="weaver://resumed-weights"),
            model_id="training-model",
        )
        holder._service_client = SimpleNamespace(create_sampling_client=AsyncMock())
        holder._active_sampling_snapshot = ("weaver://initial", "default")
        holder._sampling_client = object()
        return holder

    yield make

    for holder in holders:
        holder._closed = True  # No destructor cleanup for these in-memory test clients.


def test_resume_publishes_export_name_independent_of_process_counter(make_holder):
    controller = _make_controller()
    holder = make_holder()
    controller.set_weaver_service_holder(holder)
    service = controller._rollout_services["actor"]

    sampling_path = run_coro(holder.save_weights_for_sampler("northrl_actor_v000100"))
    assert run_coro(holder.get_current_sampling_version()) == "default"
    holder._service_client.create_sampling_client.assert_not_awaited()
    controller.sync_weight_to_rollout_service("actor", sampling_path, "northrl_actor_v000100")

    assert run_coro(holder.get_current_sampling_snapshot()) == (
        "weaver://resumed-weights",
        "northrl_actor_v000100",
    )
    assert service.train_model_version == service.rollout_model_version == 0


def test_sync_records_counter_captured_before_backend_call():
    controller = _make_controller()
    service = controller._rollout_services["actor"]
    service.train_model_version = 2

    async def update(sampling_path, weight_version):
        service.train_model_version = 4

    holder = SimpleNamespace(update_sampling_client=AsyncMock(side_effect=update))
    controller.set_weaver_service_holder(holder)
    controller.sync_weight_to_rollout_service(
        "actor", "weaver://weights/2", "northrl_actor_v000002"
    )

    holder.update_sampling_client.assert_awaited_once_with(
        "weaver://weights/2", "northrl_actor_v000002"
    )
    assert service.rollout_model_version == 2
    assert service.train_model_version == 4


def test_failed_sync_keeps_published_counter():
    controller = _make_controller()
    holder = SimpleNamespace(
        update_sampling_client=AsyncMock(side_effect=RuntimeError("sampler unavailable"))
    )
    controller.set_weaver_service_holder(holder)
    service = controller._rollout_services["actor"]
    service.train_model_version = 2
    service.rollout_model_version = 1

    with pytest.raises(RuntimeError, match="sampler unavailable"):
        controller.sync_weight_to_rollout_service(
            "actor", "weaver://weights/2", "northrl_actor_v000002"
        )

    assert controller.get_rollout_model_version("actor") == 1


def test_batch_async_publishes_only_the_export_supplied_when_sync_is_due(make_holder):
    controller = _make_controller(mode="batch-async", threshold=1)
    holder = make_holder()
    holder._training_client.save_weights_for_sampler.side_effect = [
        "weaver://weights/1",
        "weaver://weights/2",
    ]
    controller.set_weaver_service_holder(holder)
    dataloader, pool = Mock(), Mock()
    controller.set_module_references(dataloader, pool)

    controller.trajectory_pool_notify_batch_ready("actor")
    first_path = run_coro(holder.save_weights_for_sampler("northrl_actor_v000001"))
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=first_path, weight_version="northrl_actor_v000001"
    )
    assert run_coro(holder.get_current_sampling_version()) == "default"
    holder._service_client.create_sampling_client.assert_not_awaited()
    assert controller._rollout_services["actor"].train_model_version == 1
    assert controller.get_rollout_model_version("actor") == 0

    controller.trajectory_pool_notify_batch_ready("actor")
    assert controller.check_rollout_service_status("actor") == "block"
    second_path = run_coro(holder.save_weights_for_sampler("northrl_actor_v000002"))
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=second_path, weight_version="northrl_actor_v000002"
    )

    assert run_coro(holder.get_current_sampling_snapshot()) == (
        "weaver://weights/2",
        "northrl_actor_v000002",
    )
    assert controller.get_rollout_model_version("actor") == 2
    holder._service_client.create_sampling_client.assert_awaited_once_with(
        model_path="weaver://weights/2", base_model="actor", model_id="training-model"
    )
    assert dataloader.release_next_batch.call_count == 2


@pytest.mark.parametrize(
    ("sampling_path", "weight_version"),
    [(None, None), (None, "northrl_actor_v000001"), ("weaver://weights/1", None)],
)
def test_sync_requires_both_export_arguments(sampling_path, weight_version):
    controller = _make_controller()
    holder = SimpleNamespace(update_sampling_client=AsyncMock())
    controller.set_weaver_service_holder(holder)
    dataloader, pool = Mock(), Mock()
    controller.set_module_references(dataloader, pool)
    controller.trajectory_pool_notify_batch_ready("actor")

    with pytest.raises(ValueError, match="Sampling path and weight version are required"):
        controller.train_worker_notify_weight_update(
            "trainer", "actor", sampling_path=sampling_path, weight_version=weight_version
        )

    holder.update_sampling_client.assert_not_awaited()
    assert controller.get_rollout_model_version("actor") == 0
    dataloader.release_next_batch.assert_not_called()
