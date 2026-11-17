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

"""Fully async batch callbacks preserve an in-progress sampler update."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from .test_weight_sync_controller import _make_fully_async_weaver_controller


@pytest.mark.parametrize("pause_mode", [None, "retract", "in_place"])
def test_fully_async_batch_ready_does_not_clear_in_progress_sync(pause_mode):
    controller, holder, dataloader = _make_fully_async_weaver_controller(pause_mode=pause_mode)
    service = controller._rollout_services["actor"]  # pylint: disable=protected-access
    sync_entered = threading.Event()
    finish_sync = threading.Event()

    pause_kwargs = {} if pause_mode is None else {"pause_mode": pause_mode}

    async def _update(sampling_path, weight_version, **kwargs):
        assert (sampling_path, weight_version) == ("/samplers/1", "export-1")
        assert kwargs == pause_kwargs
        sync_entered.set()
        assert await asyncio.to_thread(finish_sync.wait, 5), "sync was not released"

    holder.update_sampling_client.side_effect = _update
    controller.trajectory_pool_notify_batch_ready("actor")
    with ThreadPoolExecutor(max_workers=1) as executor:
        sync = executor.submit(
            controller.train_worker_notify_weight_update,
            "trainer",
            "actor",
            sampling_path="/samplers/1",
            weight_version="export-1",
        )
        try:
            assert sync_entered.wait(timeout=5)
            assert service.state == "syncing"
            assert service.train_model_version == 1
            assert service.rollout_model_version == 0

            # Exercise overlap directly: claiming another ready batch must
            # not mark an unfinished sampler update as complete.
            controller.trajectory_pool_notify_batch_ready("actor")
            assert service.state == "syncing"
            assert controller.check_rollout_service_status("actor") == "block"
            assert service.rollout_model_version == 0
            assert dataloader.release_count == 0
        finally:
            finish_sync.set()
        sync.result(timeout=5)

    holder.update_sampling_client.assert_awaited_once_with(
        "/samplers/1", "export-1", **pause_kwargs
    )
    assert service.state == "running"
    assert service.train_model_version == service.rollout_model_version == 1
    assert controller.check_rollout_service_status("actor") == "continue"
    assert dataloader.release_count == 0
