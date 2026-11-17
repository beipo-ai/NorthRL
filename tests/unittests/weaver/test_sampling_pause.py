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

"""Engine pause transactions must not wait for the generations they suspend."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


class SamplingClient:
    def __init__(self, version, completed, events):
        self.version = version
        self.completed = completed
        self.events = events
        self.submitted = asyncio.Event()
        self.acknowledged = asyncio.Event()
        self.acknowledged.set()
        self.pause_generation = AsyncMock(side_effect=self.pause)
        self.continue_generation = AsyncMock(side_effect=self.resume)

    async def pause(self, *, mode):
        self.events.append(("pause", self.version, mode))

    async def resume(self):
        self.events.append(("continue", self.version))
        self.completed.set()

    async def sample(self, *, wait, **kwargs):
        assert wait is False
        self.events.append(("sample", self.version))
        self.submitted.set()
        await self.acknowledged.wait()
        return SimpleNamespace(operation_id=f"sample-{self.version}", result=self.result)

    async def result(self):
        await self.completed.wait()
        return {
            "result": {
                "sequences": [
                    {
                        "tokens": [3],
                        "text": "ok",
                        "logprobs": [-0.1],
                        "old_logprobs": [-0.2],
                        "sampling_masks": [[3]],
                        "moe_topk_indices_ref": {"uri": "/routing.safetensors"},
                        "stop_reason": "stop",
                        "weight_version": self.version,
                    }
                ]
            }
        }


@pytest.fixture
def setup_holder():
    holders = []

    def make(capacity=1):
        completed, events = asyncio.Event(), []
        old = SamplingClient("old", completed, events)
        new = SamplingClient("new", completed, events)
        holder = WeaverServiceHolder(
            base_model="actor",
            training_mode="full_ft",
            max_concurrent_requests=capacity,
            allow_weight_version_mismatch=True,
        )
        holder._ensure_provisioned = AsyncMock()
        holder._training_client = SimpleNamespace(model_id="actor-id")
        holder._sampling_client = old
        holder._active_sampling_snapshot = ("/old", "old")
        holder._tokenizer = SimpleNamespace(
            encode=lambda *args, **kwargs: [1, 2], decode=lambda *args, **kwargs: "ok"
        )
        holder.apply_chat_template = AsyncMock(return_value=[1, 2])

        async def create(**kwargs):
            assert kwargs == {"model_path": "/new", "base_model": "actor", "model_id": "actor-id"}
            events.append(("create",))
            return new

        holder._service_client = SimpleNamespace(
            create_sampling_client=AsyncMock(side_effect=create)
        )
        holders.append(holder)
        return holder, old, new, completed, events

    yield make
    for holder in holders:
        holder._closed = True


def run(coro):
    async def bounded():
        return await asyncio.wait_for(coro, 5)

    return asyncio.run(bounded())


@pytest.mark.parametrize("mode", ["retract", "in_place"])
@pytest.mark.parametrize(
    "entrypoint,kwargs",
    [
        ("sample_from_prompt", {"prompt": "hello"}),
        ("sample_from_messages", {"messages": [{"role": "user", "content": "hello"}]}),
        ("sample_from_token_ids", {"input_ids": [1, 2]}),
    ],
)
def test_pending_generation_survives_swap(setup_holder, mode, entrypoint, kwargs):
    async def scenario():
        holder, old, new, _, events = setup_holder()
        sample = getattr(holder, entrypoint)
        first = asyncio.create_task(sample(max_tokens=8, **kwargs))
        await old.submitted.wait()
        assert not first.done()
        await holder.update_sampling_client("/new", "new", pause_mode=mode)
        results = [await first, await sample(max_tokens=8, **kwargs)]
        assert events == [
            ("sample", "old"),
            ("pause", "old", mode),
            ("create",),
            ("continue", "new"),
            ("sample", "new"),
        ]
        assert [item["weight_version"] for item in results] == ["old", "new"]
        for item in results:
            assert item["response"] == "ok"
            assert item["response_tokens"] == [3]
            assert item["response_logprobs"] == [-0.1]
            assert item["response_old_logprobs"] == [-0.2]
            assert item["response_sampling_masks"] == [[3]]
            assert item["response_moe_topk_indices_ref"] == {"uri": "/routing.safetensors"}
        old.continue_generation.assert_not_awaited()
        new.continue_generation.assert_awaited_once()
        assert await holder.get_current_sampling_snapshot() == ("/new", "new")

    run(scenario())


@pytest.mark.parametrize("mode", ["retract", "in_place"])
def test_update_does_not_wait_for_submission_acknowledgement(setup_holder, mode):
    async def scenario():
        holder, old, _, _, events = setup_holder()
        old.acknowledged.clear()
        pending = asyncio.create_task(holder.sample_from_prompt("hello", max_tokens=8))
        await old.submitted.wait()
        await holder.update_sampling_client("/new", "new", pause_mode=mode)
        assert ("continue", "new") in events
        assert not pending.done()
        old.acknowledged.set()
        await pending

    run(scenario())


@pytest.mark.parametrize("mode", ["retract", "in_place"])
@pytest.mark.parametrize("stage", ["pause", "create", "continue"])
def test_sampling_can_overlap_engine_control(setup_holder, mode, stage):
    async def scenario():
        holder, old, new, _, _ = setup_holder(capacity=None)
        entered, release = asyncio.Event(), asyncio.Event()
        control = {
            "pause": old.pause_generation,
            "create": holder._service_client.create_sampling_client,
            "continue": new.continue_generation,
        }[stage]
        original = control.side_effect

        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            return await original(**kwargs)

        control.side_effect = blocked
        update = asyncio.create_task(holder.update_sampling_client("/new", "new", pause_mode=mode))
        await entered.wait()
        current = new if stage == "continue" else old
        pending = asyncio.create_task(holder.sample_from_prompt("hello", max_tokens=8))
        # The SDK receives this request while the control operation is still
        # pending. Queueing and scheduling are the backend's responsibility.
        await current.submitted.wait()
        assert not update.done()
        release.set()
        await update
        assert (await pending)["weight_version"] == current.version

    run(scenario())


@pytest.mark.parametrize("mode", ["retract", "in_place"])
@pytest.mark.parametrize("stage", ["pause", "create", "continue"])
def test_failed_transaction_recovers_and_keeps_published_version(setup_holder, mode, stage):
    async def scenario():
        holder, old, new, _, _ = setup_holder()
        failure = RuntimeError(stage)
        target = {
            "pause": old.pause_generation,
            "create": holder._service_client.create_sampling_client,
            "continue": new.continue_generation,
        }[stage]
        target.side_effect = failure
        with pytest.raises(RuntimeError) as caught:
            await holder.update_sampling_client("/new", "new", pause_mode=mode)
        assert caught.value is failure
        current = new if stage == "continue" else old
        current.continue_generation.assert_awaited_once()
        assert holder._sampling_client is current
        assert await holder.get_current_sampling_version() == "old"
        assert not old.submitted.is_set() and not new.submitted.is_set()

    run(scenario())


def test_sync_failure_is_not_masked_by_recovery_failure(setup_holder):
    async def scenario():
        holder, old, _, _, _ = setup_holder()
        failure = RuntimeError("create failed")
        holder._service_client.create_sampling_client.side_effect = failure
        old.continue_generation.side_effect = RuntimeError("continue failed")
        with pytest.raises(RuntimeError) as caught:
            await holder.update_sampling_client("/new", "new", pause_mode="retract")
        assert caught.value is failure

    run(scenario())


@pytest.mark.parametrize("mode", ["retract", "in_place"])
@pytest.mark.parametrize("stage", ["pause", "create", "continue"])
def test_cancelled_transaction_settles_control_and_recovers(setup_holder, mode, stage):
    async def scenario():
        holder, old, new, _, events = setup_holder()
        entered, release, recovered = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            events.append(("settled", stage))
            return new

        target = {
            "pause": old.pause_generation,
            "create": holder._service_client.create_sampling_client,
            "continue": new.continue_generation,
        }[stage]
        target.side_effect = blocked
        if stage != "continue":

            async def resume():
                recovered.set()

            old.continue_generation.side_effect = resume
        update = asyncio.create_task(holder.update_sampling_client("/new", "new", pause_mode=mode))
        await entered.wait()
        update.cancel()
        await asyncio.sleep(0)
        if stage != "create":
            # Control RPCs must settle even after repeated cancellation, so a
            # delayed pause cannot arrive after its matching continue.
            update.cancel()
            await asyncio.sleep(0)
            assert not update.done()
            assert not recovered.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await update
        if stage == "continue":
            new.continue_generation.assert_awaited_once()
            assert ("settled", "continue") in events
        else:
            assert recovered.is_set()
            old.continue_generation.assert_awaited_once()
        assert await holder.get_current_sampling_version() == "old"

    run(scenario())


def test_continue_timeout_does_not_publish_success(setup_holder, monkeypatch):
    async def scenario():
        holder, _, new, _, _ = setup_holder()

        async def blocked():
            await asyncio.Event().wait()

        new.continue_generation.side_effect = blocked
        monkeypatch.setattr(holder_module, "SAMPLING_CONTROL_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(holder_module, "SAMPLING_CONTROL_RETRY_BACKOFF_SECONDS", 0)
        with pytest.raises(TimeoutError):
            await holder.update_sampling_client("/new", "new", pause_mode="retract")
        assert new.continue_generation.await_count == 3
        assert await holder.get_current_sampling_version() == "old"

    run(scenario())


def test_continue_timeout_retries_then_publishes_success(setup_holder, monkeypatch):
    async def scenario():
        holder, _, new, _, _ = setup_holder()
        attempts = 0

        async def continue_generation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                await asyncio.Event().wait()

        new.continue_generation.side_effect = continue_generation
        monkeypatch.setattr(holder_module, "SAMPLING_CONTROL_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(holder_module, "SAMPLING_CONTROL_RETRY_BACKOFF_SECONDS", 0)
        await holder.update_sampling_client("/new", "new", pause_mode="retract")
        assert attempts == 3
        assert await holder.get_current_sampling_version() == "new"

    run(scenario())


def test_continue_timeout_uses_exponential_backoff(setup_holder, monkeypatch):
    async def scenario():
        holder, _, new, _, _ = setup_holder()
        new.continue_generation.side_effect = [TimeoutError(), TimeoutError(), None]
        sleep = AsyncMock()
        monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
        await holder.update_sampling_client("/new", "new", pause_mode="retract")
        assert [call.args for call in sleep.await_args_list] == [(1.0,), (2.0,)]
        assert new.continue_generation.await_count == 3
        assert await holder.get_current_sampling_version() == "new"

    run(scenario())


def test_pause_timeout_is_not_retried_and_recovers(setup_holder, monkeypatch):
    async def scenario():
        holder, old, _, _, _ = setup_holder()

        async def blocked(**kwargs):
            await asyncio.Event().wait()

        old.pause_generation.side_effect = blocked
        monkeypatch.setattr(holder_module, "SAMPLING_CONTROL_TIMEOUT_SECONDS", 0.01)
        with pytest.raises(TimeoutError):
            await holder.update_sampling_client("/new", "new", pause_mode="retract")
        old.pause_generation.assert_awaited_once_with(mode="retract")
        old.continue_generation.assert_awaited_once()
        holder._service_client.create_sampling_client.assert_not_awaited()
        assert await holder.get_current_sampling_version() == "old"

    run(scenario())


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("retry_fails", [False, True])
def test_cancellation_during_continue_backoff_settles_retries(
    setup_holder, monkeypatch, recovery, retry_fails
):
    async def scenario():
        holder, old, new, _, _ = setup_holder()
        failure = RuntimeError("create failed")
        if recovery:
            holder._service_client.create_sampling_client.side_effect = failure
        current = old if recovery else new
        current.continue_generation.side_effect = [
            TimeoutError(),
            RuntimeError("continue failed") if retry_fails else None,
        ]
        entered, release = asyncio.Event(), asyncio.Event()
        original_sleep = asyncio.sleep

        async def backoff(delay):
            entered.set()
            await release.wait()

        monkeypatch.setattr(holder_module.asyncio, "sleep", backoff)
        update = asyncio.create_task(
            holder.update_sampling_client("/new", "new", pause_mode="retract")
        )
        await entered.wait()
        for _ in range(2):
            update.cancel()
            await original_sleep(0)
            assert not update.done()
            current.continue_generation.assert_awaited_once()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await update
        if recovery:
            assert caught.value.__cause__ is failure
        assert current.continue_generation.await_count == 2
        assert await holder.get_current_sampling_version() == "old"

    run(scenario())


@pytest.mark.parametrize("mode", [None, "retract", "in_place"])
def test_initial_client_creation_does_not_pause(setup_holder, mode):
    async def scenario():
        holder, old, new, _, _ = setup_holder()
        holder._sampling_client = None
        await holder.update_sampling_client("/new", "new", pause_mode=mode)
        old.pause_generation.assert_not_awaited()
        new.continue_generation.assert_not_awaited()
        assert await holder.get_current_sampling_version() == "new"

    run(scenario())


def test_disabled_pause_preserves_existing_swap(setup_holder):
    async def scenario():
        holder, old, new, _, _ = setup_holder()
        await holder.update_sampling_client("/new", "new")
        old.pause_generation.assert_not_awaited()
        new.continue_generation.assert_not_awaited()
        assert holder._sampling_client is new

    run(scenario())


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_cancellation_during_failure_recovery_is_propagated(setup_holder, recovery_fails):
    async def scenario():
        holder, old, _, _, _ = setup_holder()
        failure = RuntimeError("create failed")
        holder._service_client.create_sampling_client.side_effect = failure
        entered, release = asyncio.Event(), asyncio.Event()

        async def resume():
            entered.set()
            await release.wait()
            if recovery_fails:
                raise RuntimeError("continue failed")

        old.continue_generation.side_effect = resume
        update = asyncio.create_task(
            holder.update_sampling_client("/new", "new", pause_mode="retract")
        )
        await entered.wait()
        update.cancel()
        await asyncio.sleep(0)
        assert not update.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await update
        assert caught.value.__cause__ is failure
        assert update.cancelled()

    run(scenario())
