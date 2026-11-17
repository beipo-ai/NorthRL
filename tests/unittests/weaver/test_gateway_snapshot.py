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

"""Host-owned sampler publication is shared and captured with its weight version."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


def _published(model="weaver-sampler-1"):
    return {
        "Status": "ready",
        "gateway": {
            "state": "published",
            "base_url": "https://northgate.example/",
            "model": model,
        },
    }


@pytest.fixture
def make_holder(monkeypatch):
    instances = []

    def make(**kwargs):
        instance = WeaverServiceHolder(base_model="actor", **kwargs)
        instance._ensure_provisioned = AsyncMock()
        instance._training_client = SimpleNamespace(model_id="actor-id")
        instance._sampling_client = SimpleNamespace(
            sampling_session_id="sampler-1",
            pause_generation=AsyncMock(),
            continue_generation=AsyncMock(),
        )
        instance._active_sampling_snapshot = ("weaver://actor/checkpoints/step-1", "step-1")
        instance._service_client = SimpleNamespace(
            http=SimpleNamespace(get=AsyncMock(return_value=_published()), delete=AsyncMock()),
            create_sampling_client=AsyncMock(),
            aclose=AsyncMock(),
        )
        instances.append(instance)
        return instance

    monkeypatch.setattr(holder_module, "GATEWAY_PUBLICATION_POLL_SECONDS", 0)
    yield make
    for instance in instances:
        instance._closed = True


@pytest.fixture
def holder(make_holder):
    return make_holder()


def run(coro):
    async def bounded():
        return await asyncio.wait_for(coro, 5)

    return asyncio.run(bounded())


def test_rollout_fanout_shares_publication_and_session_cache(holder):
    async def scenario():
        holder._service_client.http.get.side_effect = [
            {"Status": "ready", "gateway": {"state": "pending"}},
            _published(),
        ]
        snapshots = await asyncio.gather(
            *(holder.get_current_gateway_snapshot() for _ in range(20))
        )
        assert all(
            snapshot
            == {
                "base_url": "https://northgate.example",
                "model": "weaver-sampler-1",
                "sampling_session_id": "sampler-1",
                "weight_version": "step-1",
            }
            for snapshot in snapshots
        )
        assert await holder.get_current_gateway_snapshot() == snapshots[0]
        assert holder._service_client.http.get.await_count == 2
        holder._service_client.http.get.assert_awaited_with("/api/v1/sampling-sessions/sampler-1")
        holder._service_client.create_sampling_client.assert_not_awaited()
        snapshots[0]["model"] = "changed-by-caller"
        assert (await holder.get_current_gateway_snapshot())["model"] == "weaver-sampler-1"

    asyncio.run(scenario())


def test_repeated_rollouts_reuse_publication_after_long_idle(holder, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(holder_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def scenario():
        first = await holder.get_current_gateway_snapshot()
        clock[0] += 86400
        snapshots = await asyncio.gather(
            *(holder.get_current_gateway_snapshot() for _ in range(20))
        )
        assert all(snapshot == first for snapshot in snapshots)
        holder._service_client.http.get.assert_awaited_once()

    run(scenario())


def test_gateway_and_weight_version_are_captured_before_concurrent_swap(holder):
    async def scenario():
        queried, release = asyncio.Event(), asyncio.Event()

        async def get(path):
            assert path == "/api/v1/sampling-sessions/sampler-1"
            queried.set()
            await release.wait()
            return _published()

        holder._service_client.http.get.side_effect = get
        pending = asyncio.create_task(holder.get_current_gateway_snapshot())
        await queried.wait()
        holder._sampling_client = SimpleNamespace(sampling_session_id="sampler-2")
        holder._active_sampling_snapshot = ("weaver://actor/checkpoints/step-2", "step-2")
        release.set()
        old = await pending
        assert old["sampling_session_id"] == "sampler-1"
        assert old["weight_version"] == "step-1"
        holder._service_client.http.get.side_effect = None
        holder._service_client.http.get.return_value = _published("weaver-sampler-2")
        new = await holder.get_current_gateway_snapshot()
        assert new["sampling_session_id"] == "sampler-2"
        assert new["weight_version"] == "step-2"
        assert new["model"] == "weaver-sampler-2"

    asyncio.run(scenario())


def test_cancelled_rollout_does_not_cancel_shared_publication(holder):
    async def scenario():
        queried, release = asyncio.Event(), asyncio.Event()

        async def get(path):
            queried.set()
            await release.wait()
            return _published()

        holder._service_client.http.get.side_effect = get
        cancelled = asyncio.create_task(holder.get_current_gateway_snapshot())
        surviving = asyncio.create_task(holder.get_current_gateway_snapshot())
        await queried.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        assert (await surviving)["model"] == "weaver-sampler-1"
        assert holder._service_client.http.get.await_count == 1

    asyncio.run(scenario())


def test_publication_timeout_is_bounded_and_can_retry_same_sampler(holder):
    async def scenario():
        holder._service_client.http.get.return_value = {"Status": "ready"}
        with pytest.raises(TimeoutError):
            await holder.get_current_gateway_snapshot(timeout_seconds=0.01)
        await asyncio.sleep(0.01)
        holder._service_client.http.get.return_value = _published()
        assert (await holder.get_current_gateway_snapshot())["sampling_session_id"] == "sampler-1"
        holder._service_client.create_sampling_client.assert_not_awaited()

    asyncio.run(scenario())


def test_explicit_refresh_rechecks_gateway_retirement_and_invalidates_cache(holder):
    async def scenario():
        await holder.get_current_gateway_snapshot()
        holder._service_client.http.get.return_value = {"gateway": {"state": "retired"}}
        with pytest.raises(RuntimeError, match=r"call sample\(\) or compute_logprobs\(\)"):
            await holder.refresh_gateway_snapshot()
        with pytest.raises(RuntimeError, match=r"call sample\(\) or compute_logprobs\(\)"):
            await holder.get_current_gateway_snapshot()
        assert holder._service_client.http.get.await_count == 3

    run(scenario())


def test_explicit_refresh_replaces_cached_route_after_republication(holder):
    async def scenario():
        first = await holder.get_current_gateway_snapshot()
        holder._service_client.http.get.return_value = _published("republished-sampler-1")
        refreshed = await holder.refresh_gateway_snapshot()
        assert refreshed == {**first, "model": "republished-sampler-1"}
        assert await holder.get_current_gateway_snapshot() == refreshed
        assert holder._service_client.http.get.await_count == 2

    run(scenario())


@pytest.mark.parametrize("pause_mode", [None, "retract", "in_place"])
def test_update_keeps_previous_snapshot_until_candidate_is_published(make_holder, pause_mode):
    async def scenario():
        instance = make_holder(training_mode="full_ft", gateway_publication_timeout_seconds=1)
        previous = await instance.get_current_gateway_snapshot()
        previous_client = instance._sampling_client
        candidate = SimpleNamespace(
            sampling_session_id="sampler-2", continue_generation=AsyncMock()
        )
        instance._service_client.create_sampling_client.return_value = candidate
        queried, release = asyncio.Event(), asyncio.Event()

        async def get(path):
            assert path == "/api/v1/sampling-sessions/sampler-2"
            queried.set()
            await release.wait()
            return _published("weaver-sampler-2")

        instance._service_client.http.get.side_effect = get
        update = asyncio.create_task(
            instance.update_sampling_client(
                "weaver://actor/checkpoints/step-2", "step-2", pause_mode=pause_mode
            )
        )
        await queried.wait()
        try:
            assert not update.done()
            assert instance._sampling_client is previous_client
            assert await instance.get_current_gateway_snapshot() == previous
        finally:
            release.set()
            await update

        assert instance._sampling_client is candidate
        current = await instance.get_current_gateway_snapshot()
        assert current == {
            **previous,
            "sampling_session_id": "sampler-2",
            "weight_version": "step-2",
            "model": "weaver-sampler-2",
        }
        snapshots = await asyncio.gather(
            *(instance.get_current_gateway_snapshot() for _ in range(20))
        )
        assert all(snapshot == current for snapshot in snapshots)
        assert instance._service_client.http.get.await_count == 2

    run(scenario())


@pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
def test_gateway_snapshot_keeps_session_and_version_paired_during_resume(make_holder, pause_mode):
    async def scenario():
        instance = make_holder(training_mode="full_ft", gateway_publication_timeout_seconds=1)
        await instance.get_current_gateway_snapshot()
        resuming, release = asyncio.Event(), asyncio.Event()

        async def resume():
            resuming.set()
            await release.wait()

        candidate = SimpleNamespace(
            sampling_session_id="sampler-2", continue_generation=AsyncMock(side_effect=resume)
        )
        instance._service_client.create_sampling_client.return_value = candidate
        instance._service_client.http.get.return_value = _published("weaver-sampler-2")
        update = asyncio.create_task(
            instance.update_sampling_client(
                "weaver://actor/checkpoints/step-2", "step-2", pause_mode=pause_mode
            )
        )
        await resuming.wait()
        snapshot = asyncio.create_task(instance.get_current_gateway_snapshot())
        try:
            await asyncio.sleep(0)
        finally:
            release.set()
            await update
        current = await snapshot
        assert (current["sampling_session_id"], current["weight_version"]) in {
            ("sampler-1", "step-1"),
            ("sampler-2", "step-2"),
        }

    run(scenario())


@pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
def test_candidate_resume_failure_preserves_previous_gateway_snapshot(make_holder, pause_mode):
    async def scenario():
        instance = make_holder(training_mode="full_ft", gateway_publication_timeout_seconds=1)
        previous = await instance.get_current_gateway_snapshot()
        previous_client = instance._sampling_client
        failure = RuntimeError("candidate resume failed")
        candidate = SimpleNamespace(
            sampling_session_id="sampler-2",
            continue_generation=AsyncMock(side_effect=failure),
        )
        instance._service_client.create_sampling_client.return_value = candidate
        instance._service_client.http.get.return_value = _published("weaver-sampler-2")
        with pytest.raises(RuntimeError) as caught:
            await instance.update_sampling_client(
                "weaver://actor/checkpoints/step-2", "step-2", pause_mode=pause_mode
            )
        assert caught.value is failure
        assert instance._sampling_client is previous_client
        assert await instance.get_current_gateway_snapshot() == previous
        assert await instance.get_current_sampling_snapshot() == (
            "weaver://actor/checkpoints/step-1",
            "step-1",
        )
        previous_client.continue_generation.assert_awaited_once()
        instance._service_client.http.delete.assert_awaited_once_with(
            "/api/v1/sampling-sessions/sampler-2"
        )
        assert "sampler-2" not in instance._gateway_publication_cache
        assert "sampler-2" not in instance._gateway_publication_tasks

    run(scenario())


@pytest.mark.parametrize("pause_mode", [None, "retract", "in_place"])
def test_candidate_publication_timeout_preserves_client_and_weight_version(make_holder, pause_mode):
    async def scenario():
        instance = make_holder(training_mode="full_ft", gateway_publication_timeout_seconds=0.01)
        previous = await instance.get_current_gateway_snapshot()
        previous_client = instance._sampling_client
        candidate = SimpleNamespace(
            sampling_session_id="sampler-2", continue_generation=AsyncMock()
        )
        instance._service_client.create_sampling_client.return_value = candidate

        async def never_published(path):
            assert path == "/api/v1/sampling-sessions/sampler-2"
            await asyncio.Event().wait()

        instance._service_client.http.get.side_effect = never_published
        with pytest.raises(TimeoutError, match="sampler-2"):
            await instance.update_sampling_client(
                "weaver://actor/checkpoints/step-2", "step-2", pause_mode=pause_mode
            )
        assert instance._sampling_client is previous_client
        assert await instance.get_current_sampling_snapshot() == (
            "weaver://actor/checkpoints/step-1",
            "step-1",
        )
        assert await instance.get_current_gateway_snapshot() == previous
        if pause_mode is not None:
            previous_client.continue_generation.assert_awaited_once()
        candidate.continue_generation.assert_not_awaited()
        instance._service_client.http.delete.assert_awaited_once_with(
            "/api/v1/sampling-sessions/sampler-2"
        )
        assert "sampler-2" not in instance._gateway_publication_cache
        assert "sampler-2" not in instance._gateway_publication_tasks

    run(scenario())


@pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
def test_cancel_during_previous_sampler_recovery_keeps_candidate_failure_as_cause(
    make_holder, pause_mode
):
    async def scenario():
        instance = make_holder(training_mode="full_ft", gateway_publication_timeout_seconds=1)
        previous = await instance.get_current_gateway_snapshot()
        previous_client = instance._sampling_client
        failure = RuntimeError("candidate resume failed")
        candidate = SimpleNamespace(
            sampling_session_id="sampler-2",
            continue_generation=AsyncMock(side_effect=failure),
        )
        instance._service_client.create_sampling_client.return_value = candidate
        instance._service_client.http.get.return_value = _published("weaver-sampler-2")
        recovering, release = asyncio.Event(), asyncio.Event()

        async def recover():
            recovering.set()
            await release.wait()

        previous_client.continue_generation.side_effect = recover
        update = asyncio.create_task(
            instance.update_sampling_client(
                "weaver://actor/checkpoints/step-2", "step-2", pause_mode=pause_mode
            )
        )
        await recovering.wait()
        update.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await update
        assert update.cancelled()
        assert caught.value.__cause__ is failure
        assert instance._sampling_client is previous_client
        assert await instance.get_current_gateway_snapshot() == previous
        assert await instance.get_current_sampling_snapshot() == (
            "weaver://actor/checkpoints/step-1",
            "step-1",
        )
        previous_client.continue_generation.assert_awaited_once()
        instance._service_client.http.delete.assert_awaited_once_with(
            "/api/v1/sampling-sessions/sampler-2"
        )
        assert "sampler-2" not in instance._gateway_publication_cache
        assert not instance._gateway_publication_tasks

    run(scenario())


@pytest.mark.parametrize("pause_mode", [None, "retract", "in_place"])
def test_cancelled_candidate_publication_preserves_previous_sampler_and_cleans_up(
    make_holder, pause_mode
):
    async def scenario():
        instance = make_holder(training_mode="full_ft", gateway_publication_timeout_seconds=1)
        previous = await instance.get_current_gateway_snapshot()
        previous_client = instance._sampling_client
        candidate = SimpleNamespace(
            sampling_session_id="sampler-2", continue_generation=AsyncMock()
        )
        instance._service_client.create_sampling_client.return_value = candidate
        queried = asyncio.Event()

        async def never_published(path):
            assert path == "/api/v1/sampling-sessions/sampler-2"
            queried.set()
            await asyncio.Event().wait()

        instance._service_client.http.get.side_effect = never_published
        update = asyncio.create_task(
            instance.update_sampling_client(
                "weaver://actor/checkpoints/step-2", "step-2", pause_mode=pause_mode
            )
        )
        await queried.wait()
        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update
        assert instance._sampling_client is previous_client
        assert await instance.get_current_gateway_snapshot() == previous
        instance._service_client.http.delete.assert_awaited_once_with(
            "/api/v1/sampling-sessions/sampler-2"
        )
        assert not instance._gateway_publication_tasks
        assert "sampler-2" not in instance._gateway_publication_cache
        if pause_mode is not None:
            previous_client.continue_generation.assert_awaited_once()
        candidate.continue_generation.assert_not_awaited()

    run(scenario())


@pytest.mark.parametrize("publication_timeout", [None, 1])
def test_initial_sampler_warmup_probes_before_optional_gateway_wait(
    make_holder, publication_timeout
):
    async def scenario():
        instance = make_holder(gateway_publication_timeout_seconds=publication_timeout)
        instance._sampling_client = None
        instance._active_sampling_snapshot = None
        candidate = SimpleNamespace(sampling_session_id="sampler-2")
        instance._service_client.create_sampling_client.return_value = candidate
        await instance.update_sampling_client("weaver://actor/checkpoints/step-2", "step-2")
        instance._service_client.http.get.assert_not_awaited()
        events = []

        async def probe(**kwargs):
            events.append("probe")

        async def get(path):
            assert path == "/api/v1/sampling-sessions/sampler-2"
            events.append("gateway")
            return _published("weaver-sampler-2")

        instance.sample_from_token_ids = AsyncMock(side_effect=probe)
        instance._service_client.http.get.side_effect = get
        await instance.warmup_sampling()
        assert events == (["probe"] if publication_timeout is None else ["probe", "gateway"])
        if publication_timeout is not None:
            assert (await instance.get_current_gateway_snapshot())["weight_version"] == "step-2"
            instance._service_client.http.get.assert_awaited_once()

    run(scenario())


def test_warmup_reuses_prepared_gateway_and_recovery_explicitly_refreshes(make_holder):
    async def scenario():
        instance = make_holder(gateway_publication_timeout_seconds=1)
        previous = await instance.get_current_gateway_snapshot()
        events = []

        async def probe(**kwargs):
            events.append("probe")

        async def get(path):
            assert path == "/api/v1/sampling-sessions/sampler-1"
            events.append("gateway")
            return _published("republished-sampler-1")

        instance.sample_from_token_ids = AsyncMock(side_effect=probe)
        instance._service_client.http.get.side_effect = get
        await instance.warmup_sampling()
        assert events == ["probe"]
        assert await instance.get_current_gateway_snapshot() == previous
        instance._service_client.http.get.assert_awaited_once()

        events.clear()
        await instance.warmup_sampling(refresh_gateway=True)
        assert events == ["probe", "gateway"]
        assert (await instance.get_current_gateway_snapshot())["model"] == "republished-sampler-1"
        assert instance._service_client.http.get.await_count == 2

    run(scenario())


def test_non_e2b_sampler_update_does_not_wait_for_gateway(holder):
    async def scenario():
        candidate = SimpleNamespace(sampling_session_id="sampler-2")
        holder._service_client.create_sampling_client.return_value = candidate
        await holder.update_sampling_client("weaver://actor/checkpoints/step-2", "step-2")
        assert holder._sampling_client is candidate
        holder._service_client.http.get.assert_not_awaited()

    run(scenario())


@pytest.mark.parametrize("state", ["deleted", "expired"])
def test_unavailable_sampler_is_not_published(holder, state):
    holder._service_client.http.get.return_value = {"Status": state}
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(holder.get_current_gateway_snapshot())


def test_close_cancels_outstanding_publication(holder):
    async def scenario():
        queried = asyncio.Event()

        async def get(path):
            queried.set()
            await asyncio.Event().wait()

        holder._service_client.http.get.side_effect = get
        pending = asyncio.create_task(holder.get_current_gateway_snapshot())
        await queried.wait()
        await holder.close()
        with pytest.raises(asyncio.CancelledError):
            await pending
        holder._service_client.aclose.assert_awaited_once()
        assert not holder._gateway_publication_tasks

    asyncio.run(scenario())
