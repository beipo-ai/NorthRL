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

"""Historical tool inputs retain sampled key order without changing their values."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from tests.unittests.remote_rollout.test_northgate_relay import (
    _Chunks,
    _events,
    _harness,
    _path,
    _payload,
    _result,
    _sse,
)


@pytest.fixture(autouse=True)
def _gateway_key(monkeypatch):
    monkeypatch.setenv("NORTHGATE_API_KEY", "test-ng-key")


def _arguments():
    return {
        "command": "true",
        "dangerouslyDisableSandbox": False,
        "run_in_background": False,
        "timeout": 1000,
        "options": {"z": 1, "a": {"right": 2, "left": 3}},
        "items": [0, {"beta": "two", "alpha": "one"}, [], {}, [{"last": 4, "first": 5}]],
    }


def _reordered(value):
    if isinstance(value, dict):
        return {key: _reordered(value[key]) for key in reversed(value)}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tool_result(arguments):
    result = _result("anthropic")
    result["content"][1].update(name="Bash", input=deepcopy(arguments))
    return result


def _done_result():
    result = _result("anthropic")
    result["content"] = [{"type": "text", "text": "done"}]
    result["stop_reason"] = "end_turn"
    return result


def _tool_events(result):
    events = _events("anthropic")
    events[5]["content_block"].update(
        id=result["content"][1]["id"], name=result["content"][1]["name"]
    )
    events[6]["delta"]["partial_json"] = _encoded(result["content"][1]["input"])
    return events


def _response(result, stream=False):
    if stream:
        return httpx.Response(
            200,
            stream=_Chunks(_sse(_tool_events(result))),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(200, json=result)


def _history(arguments, *, tool_id="tool-1", name="Bash"):
    assistant = {"role": "assistant", "content": deepcopy(_tool_result(arguments)["content"])}
    assistant["content"][1].update(id=tool_id, name=name)
    return [
        {"role": "user", "content": "question"},
        assistant,
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "done"}],
        },
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_successful_tool_input_order_is_restored_in_next_http_request(stream):
    async def scenario():
        original = _arguments()
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            return (
                _response(_tool_result(original), stream)
                if len(captured) == 1
                else _response(_done_result())
            )

        async with _harness(upstream) as (client, recorder, _, _, _):
            await client.post(_path("anthropic"), json=_payload(stream=stream))
            assert _encoded(recorder.to_messages()[-1]["content"][1]["input"]) == _encoded(original)
            history = _history(_reordered(original))
            response = await client.post(_path("anthropic"), json=_payload(messages=history))
            assert response.status_code == 200
            expected = deepcopy(history)
            expected[1]["content"][1]["input"] = original
            assert _encoded(captured[1]["messages"]) == _encoded(expected)
            assert _encoded(history[1]["content"][1]["input"]) == _encoded(_reordered(original))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "bool_to_int",
        "int_to_float",
        "nested_int_to_bool",
        "array_order",
        "array_type",
        "added_field",
        "removed_field",
        "unknown_id",
        "different_name",
    ],
)
def test_changed_or_unknown_historical_inputs_are_forwarded_unchanged(change):
    async def scenario():
        original = _arguments()
        current = _reordered(original)
        if change == "value":
            current["command"] = "printf changed"
        elif change == "bool_to_int":
            current["run_in_background"] = 0
        elif change == "int_to_float":
            current["timeout"] = 1000.0
        elif change == "nested_int_to_bool":
            current["options"]["z"] = True
        elif change == "array_order":
            current["items"].reverse()
        elif change == "array_type":
            current["items"] = {}
        elif change == "added_field":
            current["new"] = None
        elif change == "removed_field":
            current.pop("timeout")
        history = _history(
            current,
            tool_id="unknown" if change == "unknown_id" else "tool-1",
            name="OtherTool" if change == "different_name" else "Bash",
        )
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            return _response(_tool_result(original) if len(captured) == 1 else _done_result())

        async with _harness(upstream) as (client, _, _, _, _):
            await client.post(_path("anthropic"), json=_payload())
            await client.post(_path("anthropic"), json=_payload(messages=history))
            assert _encoded(captured[1]["messages"]) == _encoded(history)

    asyncio.run(scenario())


@pytest.mark.parametrize("duplicate", ["identical", "different_order", "different_values"])
def test_duplicate_tool_ids_only_restore_unambiguous_original_order(duplicate):
    async def scenario():
        original = _arguments()
        second = deepcopy(original)
        if duplicate == "different_order":
            second = _reordered(second)
        elif duplicate == "different_values":
            second["command"] = "printf another"
        results = [_tool_result(original), _tool_result(second), _tool_result(original)]
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            return _response(results[len(captured) - 1] if len(captured) <= 3 else _done_result())

        async with _harness(upstream) as (client, _, _, _, _):
            for index in range(3):
                await client.post(
                    _path("anthropic"),
                    json=_payload(messages=[{"role": "user", "content": f"branch {index}"}]),
                )
            await client.post(
                _path("anthropic"), json=_payload(messages=_history(_reordered(original)))
            )
            expected = original if duplicate == "identical" else _reordered(original)
            assert _encoded(captured[-1]["messages"][1]["content"][1]["input"]) == _encoded(
                expected
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "stream,failure",
    [
        (False, "metadata"),
        (True, "metadata"),
        (True, "truncated"),
        (True, "sse_error"),
        (False, "recorder"),
        (True, "recorder"),
    ],
)
def test_failed_responses_do_not_supply_historical_input_order(stream, failure, monkeypatch):
    async def scenario():
        original = _arguments()
        result = _tool_result(original)
        events = _tool_events(result)
        if failure == "metadata":
            result.pop("token_data")
            events[8].pop("token_data")
        elif failure == "truncated":
            events.pop()
        elif failure == "sse_error":
            events[-1] = {"type": "error", "error": {"type": "api_error", "message": "failed"}}
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            if len(captured) != 1:
                return _response(_done_result())
            if stream:
                return httpx.Response(
                    200,
                    stream=_Chunks(_sse(events)),
                    headers={"content-type": "text/event-stream"},
                )
            return _response(result)

        async with _harness(upstream) as (client, recorder, _, _, _):
            with monkeypatch.context() as patch:
                if failure == "recorder":

                    def reject(*args, **kwargs):
                        raise ValueError("recording failed")

                    patch.setattr(recorder, "record_inference_result", reject)
                with pytest.raises(Exception):
                    await client.post(_path("anthropic"), json=_payload(stream=stream))
            assert recorder.segment_count == 0
            history = _history(_reordered(original))
            await client.post(_path("anthropic"), json=_payload(messages=history))
            assert _encoded(captured[1]["messages"]) == _encoded(history)

    asyncio.run(scenario())


def test_openai_tool_argument_strings_are_forwarded_unchanged():
    async def scenario():
        original = _arguments()
        arguments = _encoded(_reordered(original))
        history = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tool-1", "content": "done"},
        ]
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            return (
                _response(_tool_result(original))
                if len(captured) == 1
                else httpx.Response(200, json=_result("openai"))
            )

        async with _harness(upstream) as (client, _, _, _, _):
            await client.post(_path("anthropic"), json=_payload())
            await client.post(_path("openai"), json=_payload(messages=history))
            assert _encoded(captured[1]["messages"]) == _encoded(history)

    asyncio.run(scenario())
