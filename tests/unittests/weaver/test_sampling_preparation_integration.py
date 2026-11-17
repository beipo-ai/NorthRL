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

"""Exercise the holder's actual preparation boundaries with blocking work."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northrl.utils.async_utils import run_coro, submit_coro
from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


def _holder(tokenizer):
    holder = WeaverServiceHolder(base_model="test", max_concurrent_requests=2)
    holder._training_client = SimpleNamespace()
    holder._sampling_client = SimpleNamespace(sample=AsyncMock())
    holder._tokenizer = tokenizer
    holder._active_sampling_snapshot = ("weaver://weights/version-1", "version-1")
    holder._sample_with_context_limit = AsyncMock(
        return_value={
            "tokens": [3],
            "logprobs": [-0.1],
            "text": "ok",
            "stop_reason": "stop",
            "weight_version": "version-1",
        }
    )
    return holder


@pytest.mark.parametrize("path", ["messages", "prompt", "token_ids", "decode"])
def test_slow_sampling_preparation_leaves_sdk_loop_responsive(monkeypatch, path):
    entered, release = threading.Event(), threading.Event()
    worker_threads = []

    def block():
        worker_threads.append(threading.current_thread().name)
        entered.set()
        assert release.wait(3), "preparation worker was not released"

    def render(*args, **kwargs):
        if path == "messages":
            block()
        return [1, 2]

    def encode(*args, **kwargs):
        if path == "prompt":
            block()
        return [1, 2]

    def decode(*args, **kwargs):
        if path == "decode":
            block()
        return "decoded"

    holder = _holder(SimpleNamespace(apply_chat_template=render, encode=encode, decode=decode))
    if path == "token_ids":
        original = holder_module.types.ModelInput.from_ints

        def make_input(*args, **kwargs):
            block()
            return original(*args, **kwargs)

        monkeypatch.setattr(holder_module.types.ModelInput, "from_ints", make_input)
    if path == "decode":
        sequence = holder._sample_with_context_limit.return_value
        sequence["text"] = None
        del holder._sample_with_context_limit
        holder._sample_with_limit = AsyncMock(return_value={"result": {"sequences": [sequence]}})
    if path == "messages":
        coro = holder.sample_from_messages([{"role": "user", "content": "hello"}], 16)
    elif path == "prompt":
        coro = holder.sample_from_prompt("hello", 16)
    else:
        coro = holder.sample_from_token_ids([1, 2], 16)
    result = submit_coro(coro)
    try:
        assert entered.wait(3), "slow preparation never started"
        probe = submit_coro(asyncio.sleep(0, result="responsive"))
        assert probe.result(1) == "responsive"
    finally:
        release.set()
    assert result.result(3)["response"] == ("decoded" if path == "decode" else "ok")
    assert worker_threads and "northrl-async-loop" not in worker_threads
    run_coro(holder.close())


def test_holder_serializes_chat_template_and_prompt_tokenizer_access():
    entered, release, attempting, encoding = (threading.Event() for _ in range(4))

    def render(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return [1, 2]

    def encode(*args, **kwargs):
        encoding.set()
        return [1, 2]

    holder = _holder(SimpleNamespace(apply_chat_template=render, encode=encode))

    async def encode_prompt():
        attempting.set()
        return await holder.sample_from_prompt("hello", 16)

    first = submit_coro(holder.apply_chat_template([{"role": "user", "content": "hello"}]))
    second = None
    try:
        assert entered.wait(3)
        second = submit_coro(encode_prompt())
        assert attempting.wait(3)
        run_coro(asyncio.sleep(0))
        assert not encoding.is_set(), "shared tokenizer accessed while rendering"
    finally:
        release.set()
    assert first.result(3) == [1, 2]
    assert second is not None and second.result(3)["response"] == "ok"
    assert encoding.is_set()
    run_coro(holder.close())


def test_holder_close_drains_preparation_and_prevents_submission():
    entered, release, closing = (threading.Event() for _ in range(3))

    def encode(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return [1, 2]

    holder = _holder(SimpleNamespace(encode=encode))
    sample = submit_coro(holder.sample_from_prompt("hello", 16))

    async def close():
        closing.set()
        await holder.close()

    closed = None
    try:
        assert entered.wait(3)
        closed = submit_coro(close())
        assert closing.wait(3)
        run_coro(asyncio.sleep(0))
        assert not closed.done(), "close returned before preparation drained"
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="closed"):
        sample.result(3)
    assert closed is not None
    closed.result(3)
    holder._sample_with_context_limit.assert_not_awaited()


@pytest.mark.parametrize("first_failure", ["error", "cancel"])
def test_training_provisioning_retries_tokenizer_after_interrupted_load(first_failure):
    entered, release = threading.Event(), threading.Event()
    tokenizer = SimpleNamespace()
    error = ValueError("tokenizer unavailable")
    calls = []

    def load():
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3), "initial tokenizer load was not released"
            if first_failure == "error":
                raise error
        return tokenizer

    training_client = SimpleNamespace(model_id="model-1", get_tokenizer=load)
    service = SimpleNamespace(
        create_model=AsyncMock(return_value=training_client), aclose=AsyncMock()
    )
    holder = WeaverServiceHolder(base_model="test", need_sampling=False)
    holder._service_client = service

    async def exercise():
        initial = asyncio.create_task(holder._ensure_provisioned())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            if first_failure == "cancel":
                initial.cancel()
                await asyncio.sleep(0)
                assert not initial.done(), "initialization detached the tokenizer load"
            assert await asyncio.sleep(0, result="SDK responsive") == "SDK responsive"
        finally:
            release.set()
        expected_error = asyncio.CancelledError if first_failure == "cancel" else ValueError
        with pytest.raises(expected_error) as raised:
            await initial
        if first_failure == "error":
            assert raised.value is error
        assert holder._training_client is training_client
        assert holder._tokenizer is None

        await holder._ensure_provisioned()
        assert holder._tokenizer is tokenizer
        service.create_model.assert_awaited_once()
        assert len(calls) == 2
        assert "northrl-async-loop" not in calls
        await holder.close()

    run_coro(exercise())


def test_cancelled_and_concurrent_holder_close_wait_for_full_client_cleanup():
    entered, release = threading.Event(), threading.Event()

    def encode(*args, **kwargs):
        entered.set()
        assert release.wait(3), "encoding was not released"
        return [1, 2]

    holder = _holder(SimpleNamespace(encode=encode))

    async def exercise():
        client_closing, client_release, client_closed = (asyncio.Event() for _ in range(3))
        publication_queried, publication_cancelling, publication_release, publication_closed = (
            asyncio.Event() for _ in range(4)
        )

        async def get_publication(path):
            publication_queried.set()
            try:
                await asyncio.Event().wait()
            finally:
                publication_cancelling.set()
                await publication_release.wait()
                publication_closed.set()

        async def close_client():
            assert publication_closed.is_set(), "client closed before publication settled"
            assert not holder._gateway_publication_tasks
            client_closing.set()
            await client_release.wait()
            client_closed.set()

        service = holder._service_client = SimpleNamespace(
            http=SimpleNamespace(get=AsyncMock(side_effect=get_publication)),
            aclose=AsyncMock(side_effect=close_client),
        )
        holder._sampling_client.sampling_session_id = "sampler-1"
        publication = asyncio.create_task(holder.get_current_gateway_snapshot())
        await asyncio.wait_for(publication_queried.wait(), 3)
        sample = asyncio.create_task(holder.sample_from_prompt("hello", 16))
        assert await asyncio.to_thread(entered.wait, 3)
        closing = asyncio.create_task(holder.close())
        await asyncio.sleep(0)
        second_close = asyncio.create_task(holder.close())
        try:
            for _ in range(3):
                closing.cancel()
                await asyncio.sleep(0)
                assert not closing.done(), "cancelled close returned before preparation stopped"
                assert not second_close.done(), "concurrent close skipped pending cleanup"
            service.aclose.assert_not_awaited()
            release.set()
            await asyncio.wait_for(publication_cancelling.wait(), 3)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done(), "cancelled close abandoned publication cleanup"
            assert not second_close.done()
            service.aclose.assert_not_awaited()
            publication_release.set()
            await asyncio.wait_for(client_closing.wait(), 3)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done(), "cancelled close abandoned the SDK client"
            assert not second_close.done()
            assert not client_closed.is_set()
        finally:
            release.set()
            publication_release.set()
            client_release.set()

        with pytest.raises(RuntimeError, match="closed"):
            await sample
        with pytest.raises(asyncio.CancelledError):
            await publication
        with pytest.raises(asyncio.CancelledError):
            await closing
        await second_close
        assert client_closed.is_set(), "cancellation propagated before the client closed"
        service.aclose.assert_awaited_once_with()
        holder._sample_with_context_limit.assert_not_awaited()
        await holder.close()
        service.aclose.assert_awaited_once_with()

    run_coro(exercise())
