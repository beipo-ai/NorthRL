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

import asyncio
import threading
from functools import partial
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.rewards import REWARD_FNS, evaluate_reward
from northrl.rewards._runtime_context import reward_runtime_context, reward_stop_requested


@pytest.mark.parametrize("name", list(REWARD_FNS))
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_length_finished_response_skips_every_verifier(name, is_async):
    verifier = AsyncMock() if is_async else Mock()
    verifier.side_effect = AssertionError("Length-limited responses must not reach a verifier")
    task = {"reward_fn": name}
    result = {"response": "", "finish_reason": "length"}

    assert asyncio.run(evaluate_reward(task, result, {name: verifier})) == {
        "reward": 0.0,
        "correct": False,
        "reason": "finish_reason_length",
    }
    verifier.assert_not_called()
    assert task == {"reward_fn": name}
    assert result == {"response": "", "finish_reason": "length"}


async def async_reward(task, result):
    await asyncio.sleep(0)
    return {
        "reward": task["label"]["score"],
        "correct": True,
        "metrics": {"response_length": len(result["response"])},
        "reason": task["metadata"]["reason"],
    }


def test_short_name_selects_function_and_passes_original_inputs():
    task = {"reward_fn": "unit", "metadata": {"style": "other", "rm_type": "other"}}
    result = {"response": "answer"}

    def unit_reward(received_task, received_result):
        assert received_task is task
        assert received_result is result
        return 0.75

    def other_reward(_task, _result):
        pytest.fail("Selected a reward from metadata instead of reward_fn")

    assert asyncio.run(
        evaluate_reward(task, result, {"unit": unit_reward, "other": other_reward})
    ) == {"reward": 0.75}


def test_async_reward_preserves_score_details():
    task = {
        "reward_fn": "async",
        "label": {"score": 0.25},
        "metadata": {"reason": "test"},
    }
    assert asyncio.run(evaluate_reward(task, {"response": "answer"}, {"async": async_reward})) == {
        "reward": 0.25,
        "correct": True,
        "metrics": {"response_length": 6.0},
        "reason": "test",
    }


def test_blocked_sync_reward_leaves_event_loop_available_and_preserves_stop_context():
    release = threading.Event()
    stop = threading.Event()
    scorer_threads = []

    async def exercise():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        responsive = asyncio.Event()

        def blocked_reward(_task, _result):
            scorer_threads.append(threading.get_ident())
            assert not reward_stop_requested()
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "The event loop did not release the synchronous scorer"
            assert reward_stop_requested()
            return 1

        async def unrelated_network_work():
            await started.wait()
            stop.set()
            responsive.set()
            release.set()

        # The safety release prevents a regression from hanging asyncio.run().
        safety_release = threading.Timer(5, release.set)
        safety_release.start()
        try:
            network_task = asyncio.create_task(unrelated_network_work())
            with reward_runtime_context(stop):
                reward = await evaluate_reward({"reward_fn": "sync"}, {}, {"sync": blocked_reward})
            await asyncio.wait_for(network_task, timeout=1)
            assert responsive.is_set()
            assert reward == {"reward": 1.0}
            assert len(scorer_threads) == 1
            assert scorer_threads[0] != loop_thread
            assert not reward_stop_requested()
        finally:
            release.set()
            safety_release.cancel()
            safety_release.join()

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["function", "partial", "callable"])
def test_async_reward_runs_on_callers_loop(kind):
    async def exercise():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()

        async def reward(_task, _result):
            assert asyncio.get_running_loop() is loop
            assert threading.get_ident() == loop_thread
            return 1

        class AsyncReward:
            async def __call__(self, task, result):
                return await reward(task, result)

        function = {"function": reward, "partial": partial(reward), "callable": AsyncReward()}[kind]
        assert await evaluate_reward({"reward_fn": "async"}, {}, {"async": function}) == {
            "reward": 1.0
        }

    asyncio.run(exercise())


def test_cancelled_sync_reward_keeps_live_stop_context_without_retrying():
    release = threading.Event()
    stop = threading.Event()
    observed_stop = threading.Event()
    calls = []

    async def exercise():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def blocked_reward(_task, _result):
            calls.append(1)
            loop.call_soon_threadsafe(started.set)
            if release.wait(5) and reward_stop_requested():
                observed_stop.set()
            return 1

        with reward_runtime_context(stop):
            reward_task = asyncio.create_task(
                evaluate_reward({"reward_fn": "sync"}, {}, {"sync": blocked_reward})
            )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            stop.set()
            reward_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(reward_task, timeout=1)
        finally:
            release.set()

    asyncio.run(exercise())
    assert observed_stop.is_set()
    assert calls == [1]


@pytest.mark.parametrize("fields", [{}, {"reward_fn": "unknown"}])
def test_missing_and_unknown_names_raise_key_error_without_fallback(fields):
    task = {**fields, "metadata": {"style": "async", "rm_type": "async"}}
    with pytest.raises(KeyError):
        asyncio.run(evaluate_reward(task, {}, {"async": async_reward}))


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        "1",
        None,
        {},
        {"reward": 1, "correct": "false"},
        {"reward": 1, "metrics": []},
        {"reward": 1, "metrics": {1: 0.5}},
        {"reward": 1, "metrics": {"score": float("nan")}},
        {"reward": 1, "reason": 5},
    ],
)
def test_invalid_scores_are_not_silently_used(value):
    def reward_fn(_task, _result):
        return value

    with pytest.raises(ValueError):
        asyncio.run(evaluate_reward({"reward_fn": "result"}, {}, {"result": reward_fn}))


def test_function_failure_propagates_unchanged():
    failure = ValueError("checker unavailable")

    def failing_reward(_task, _result):
        raise failure

    with pytest.raises(ValueError) as error:
        asyncio.run(evaluate_reward({"reward_fn": "broken"}, {}, {"broken": failing_reward}))
    assert error.value is failure


@pytest.mark.parametrize("kind", ["function", "callable"])
def test_sync_wrapper_returning_awaitable_is_supported(kind):
    async def exercise():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()

        async def inner_reward(task, result):
            assert asyncio.get_running_loop() is loop
            assert threading.get_ident() == loop_thread
            return await async_reward(task, result)

        def wrapped_async_reward(task, result):
            assert threading.get_ident() != loop_thread
            return inner_reward(task, result)

        class WrappedReward:
            def __call__(self, task, result):
                return wrapped_async_reward(task, result)

        function = wrapped_async_reward if kind == "function" else WrappedReward()
        result = await evaluate_reward(
            {"reward_fn": "wrapped", "label": {"score": 1}, "metadata": {"reason": "ok"}},
            {"response": "answer"},
            {"wrapped": function},
        )
        assert result["reward"] == 1.0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "fields, expected",
    [({"reward_fn": "sample"}, 0.25), ({}, 0.75), ({"reward_fn": None}, 0.75)],
)
def test_sample_reward_overrides_configured_default(fields, expected):
    task = {**fields, "metadata": {"style": "ignored", "rm_type": "ignored"}}
    functions = {"sample": lambda task, result: 0.25, "configured": lambda task, result: 0.75}
    reward = asyncio.run(evaluate_reward(task, {}, functions, default_reward_fn="configured"))
    assert reward == {"reward": expected}
    assert task == {**fields, "metadata": {"style": "ignored", "rm_type": "ignored"}}


@pytest.mark.parametrize("name", ["unknown", ""])
def test_explicit_sample_name_never_falls_back_to_config(name):
    def configured_reward(task, result):
        pytest.fail("An explicit sample name must not fall back to the configured reward")

    with pytest.raises(KeyError) as error:
        asyncio.run(
            evaluate_reward(
                {"reward_fn": name},
                {},
                {"configured": configured_reward},
                default_reward_fn="configured",
            )
        )
    assert error.value.args == (name,)
