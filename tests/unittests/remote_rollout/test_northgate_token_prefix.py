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

"""Native chat prefix reuse with a mock gateway, not the real NorthGate renderer."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from northrl.remote_rollout.northgate_relay import _UpstreamFailure
from tests.unittests.remote_rollout.test_northgate_relay import (
    _Chunks,
    _config,
    _events,
    _harness,
    _path,
    _payload,
    _result,
    _sse,
    _tokens,
    _tools,
)


@pytest.fixture(autouse=True)
def _gateway_key(monkeypatch):
    monkeypatch.setenv("NORTHGATE_API_KEY", "test-ng-key")


def _enabled_config():
    config = _config()
    config.rollout.execution = {"token_prefix_reuse": True}
    return config


def _assistant(api="openai"):
    if api == "anthropic":
        return {"role": "assistant", "content": _result(api)["content"]}
    return {
        "role": "assistant",
        "content": "答案",
        "tool_calls": [
            {
                "id": "tool-1",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command":"true"}'},
            }
        ],
    }


def _continuation(payload, api="openai"):
    following = deepcopy(payload)
    result = {"role": "tool", "tool_call_id": "tool-1", "content": "done"}
    if api == "anthropic":
        result = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "done"}],
        }
    following["messages"] += [_assistant(api), result]
    return following


def _response(api="openai", *, stream=False, token_data=None, payload=None, events=None):
    if token_data is None:
        token_data = _tokens()
        if payload and "history_tokens" in payload:
            token_data["prompt_tokens"] = [*payload["history_tokens"], 5, 6]
    if stream:
        events = deepcopy(events) if events is not None else _events(api)
        for event in events:
            if isinstance(event, dict) and "token_data" in event:
                event["token_data"] = token_data
        return httpx.Response(
            200, stream=_Chunks(_sse(events)), headers={"content-type": "text/event-stream"}
        )
    result = _result(api)
    result["token_data"] = token_data
    if api == "openai":
        result["choices"][0].update(message=_assistant(), finish_reason="tool_calls")
    return httpx.Response(200, json=result)


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_mock_gateway_reuses_actual_tokens_and_records_continuation(api, stream):
    async def scenario():
        captured = []
        initial = _payload(stream=stream, tools=_tools(api))
        initial["history_tokens"] = {"malformed": [999]}
        initial["token_prefix"] = {"tokens": [999], "message_count": 999}
        config = _enabled_config()
        config.algorithm.sampling_logprob_source = "inference"

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            assert payload["token_prefix"] == initial["token_prefix"]
            assert "message_count" not in payload
            if len(captured) == 1:
                assert "history_tokens" not in payload
                return _response(
                    api,
                    stream=stream,
                    token_data={
                        "prompt_tokens": [10, 11],
                        "response_tokens": [20, 21],
                        "response_logprobs": [-0.1, -0.2],
                        "response_old_logprobs": [-0.3, -0.4],
                    },
                )
            if len(captured) == 3:
                assert payload["history_tokens"] == [10, 11, 20, 21, 30, 31, 40]
                return _response(
                    api,
                    stream=stream,
                    token_data={
                        "prompt_tokens": [*payload["history_tokens"], 50],
                        "response_tokens": [60],
                        "response_logprobs": [-0.7],
                        "response_old_logprobs": [-0.8],
                    },
                )
            assert payload["history_tokens"] == [10, 11, 20, 21]
            # Mock the server's splice; this does not exercise its real renderer.
            rendered, message_boundary = [10, 11, 25, 26, 21, 30, 31], 5
            return _response(
                api,
                stream=stream,
                token_data={
                    "prompt_tokens": payload["history_tokens"] + rendered[message_boundary:],
                    "response_tokens": [40],
                    "response_logprobs": [-0.5],
                    "response_old_logprobs": [-0.6],
                },
            )

        async with _harness(upstream, config) as (client, recorder, *_):
            assert (await client.post(_path(api), json=initial)).status_code == 200
            following = _continuation(initial, api)
            following["history_tokens"] = [999]
            assert (await client.post(_path(api), json=following)).status_code == 200
            assert captured[1]["messages"] == following["messages"]
            expected_tools = deepcopy(initial["tools"])
            if api == "anthropic":
                expected_tools[0].pop("cache_control")
            assert captured[1]["tools"] == expected_tools
            third = _continuation(following, api)
            third["history_tokens"] = None
            assert (await client.post(_path(api), json=third)).status_code == 200
            assert captured[2]["messages"] == third["messages"]
            [segment] = recorder.to_remote_segments()
            assert segment.tokens == [10, 11, 20, 21, 30, 31, 40, 50, 60]
            assert segment.loss_mask == [0, 0, 1, 1, 0, 0, 1, 0, 1]
            assert segment.log_probs == [0, 0, -0.1, -0.2, 0, 0, -0.5, 0, -0.7]
            assert segment.old_log_probs == [0, 0, -0.3, -0.4, 0, 0, -0.6, 0, -0.8]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("enabled", [None, False], ids=["default", "explicit-false"])
@pytest.mark.parametrize("history_tokens", [[999], {"malformed": [999]}, [], None])
def test_disabled_feature_strips_inbound_history_tokens(api, enabled, history_tokens):
    async def scenario():
        captured = []
        config = _config()
        if enabled is not None:
            config.rollout.execution = {"token_prefix_reuse": enabled}

        def upstream(request):
            captured.append(json.loads(request.content))
            return _response(api)

        initial = _payload(
            history_tokens=history_tokens, token_prefix={"tokens": [999], "message_count": 1}
        )
        async with _harness(upstream, config) as (client, _, _, _, relay):
            await client.post(_path(api), json=initial)
            await client.post(_path(api), json=_continuation(initial, api))
            assert len(captured) == 2
            assert all(
                "history_tokens" not in payload
                and payload["token_prefix"] == initial["token_prefix"]
                for payload in captured
            )
            assert relay._token_prefix_cache is None

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_rejected_history_tokens_does_not_retry_or_change_recorded_history(api, stream):
    async def scenario():
        captured = []

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            if len(captured) == 2:
                assert payload["history_tokens"] == [1, 2, 3, 99]
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "code": "invalid_request",
                            "message": "history_tokens contains more terminators than messages",
                        }
                    },
                )
            return _response(api, stream=stream, payload=payload)

        initial = _payload(stream=stream)
        async with _harness(upstream, _enabled_config()) as (client, recorder, _, calls, _):
            await client.post(_path(api), json=initial)
            messages, segments = recorder.to_messages(), recorder.to_remote_segments()
            following = _continuation(initial, api)
            with pytest.raises(_UpstreamFailure, match="HTTP 400") as failure:
                await client.post(_path(api), json=following)
            assert len(captured) == 2
            assert recorder.to_messages() == messages
            [recorded] = recorder.to_remote_segments()
            # Inbound tool-result metrics may advance, but no training data may change.
            for field in ("tokens", "loss_mask", "log_probs", "old_log_probs", "sampling_mask"):
                assert getattr(recorded, field) == getattr(segments[0], field)
            assert calls[-1][1] is failure.value
            # An explicit new attempt must still use the previously successful history.
            assert (await client.post(_path(api), json=following)).status_code == 200
            assert captured[2] == captured[1]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_ignored_prefix_errors_before_recording(api, stream):
    async def scenario():
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            return _response(
                api, stream=stream, payload=captured[-1] if len(captured) > 2 else None
            )

        initial = _payload(stream=stream)
        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            await client.post(_path(api), json=initial)
            before = recorder.to_messages()
            with pytest.raises(Exception, match="did not preserve the requested history_tokens"):
                await client.post(_path(api), json=_continuation(initial, api))
            assert recorder.to_messages() == before
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]
            await client.post(_path(api), json=_continuation(initial, api))
            assert captured[2]["history_tokens"] == captured[1]["history_tokens"]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("failure", ["http", "missing-terminal", "length", "recorder"])
def test_incomplete_or_failed_response_does_not_seed_cache(api, failure, monkeypatch):
    async def scenario():
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            if len(captured) > 1:
                return _response(api, stream=True, payload=captured[-1])
            if failure == "http":
                return httpx.Response(400, json={"error": {"message": "invalid request"}})
            events = _events(api)
            if failure == "missing-terminal":
                events.pop()
            elif failure == "length":
                if api == "openai":
                    events[-3]["choices"][0]["finish_reason"] = "length"
                else:
                    events[-2]["delta"]["stop_reason"] = "max_tokens"
            return _response(api, stream=True, events=events)

        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            initial = _payload(stream=True)
            with monkeypatch.context() as patch:
                if failure == "recorder":

                    def reject(*args, **kwargs):
                        raise ValueError("recording failed")

                    patch.setattr(recorder, "record_inference_result", reject)
                if failure == "length":
                    assert (await client.post(_path(api), json=initial)).status_code == 200
                else:
                    with pytest.raises(Exception):
                        await client.post(_path(api), json=initial)
            await client.post(_path(api), json=_continuation(initial, api))
            assert "history_tokens" not in captured[1]

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "change", ["input-order", "input-value", "thinking", "signature", "redacted"]
)
def test_anthropic_history_preserves_native_content_and_tool_inputs(stream, change):
    async def scenario():
        captured = []
        assistant = _assistant("anthropic")
        assistant["content"][1]["input"]["timeout"] = 30
        assistant["content"].append({"type": "redacted_thinking", "data": "opaque-thinking"})
        events = _events("anthropic")
        events[6]["delta"]["partial_json"] = json.dumps(assistant["content"][1]["input"])
        events[-2:-2] = [
            {"type": "content_block_start", "index": 2, "content_block": assistant["content"][2]},
            {"type": "content_block_stop", "index": 2},
        ]

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            if len(captured) > 1:
                return _response("anthropic", stream=stream, payload=payload)
            if stream:
                return _response("anthropic", stream=True, events=events)
            result = _result("anthropic")
            result["content"] = assistant["content"]
            return httpx.Response(200, json=result)

        initial = _payload(stream=stream, tools=_tools("anthropic"))
        initial["history_tokens"] = [999]
        initial["token_prefix"] = {"tokens": [999], "message_count": 999}
        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            await client.post(_path("anthropic"), json=initial)
            assert recorder.to_messages()[-1] == assistant
            following = _continuation(initial, "anthropic")
            following["messages"][1] = deepcopy(assistant)
            content = following["messages"][1]["content"]
            if change == "input-order":
                content[1]["input"] = {"timeout": 30, "command": "true"}
            elif change == "input-value":
                content[1]["input"]["command"] = "false"
            elif change == "redacted":
                content[2]["data"] = "different-thinking"
            else:
                content[0][change] += " changed"
            await client.post(_path("anthropic"), json=following)
            assert captured[1]["token_prefix"] == initial["token_prefix"]
            assert ("history_tokens" in captured[1]) is (change == "input-order")
            if change == "input-order":
                assert captured[1]["history_tokens"] == [1, 2, 3, 99]
                assert list(captured[1]["messages"][1]["content"][1]["input"]) == [
                    "command",
                    "timeout",
                ]
            else:
                assert captured[1]["messages"] == following["messages"]

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
@pytest.mark.parametrize(
    "stop_options, reusable",
    [
        ({"stop": "custom terminator"}, False),
        ({"sampling_params": {"stop": ["custom terminator"]}}, False),
        ({"sampling_params": {"stop_token_ids": [123]}}, False),
        ({"stop": []}, True),
        ({"stop": None}, True),
    ],
)
def test_openai_custom_stops_do_not_seed_history(stream, finish_reason, stop_options, reusable):
    async def scenario():
        captured = []

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            if len(captured) > 1:
                return _response(stream=stream, payload=payload)
            if stream:
                events = _events("openai")
                events[-3]["choices"][0]["finish_reason"] = finish_reason
                if finish_reason == "stop":
                    for event in events:
                        if isinstance(event, dict):
                            for choice in event["choices"]:
                                choice["delta"].pop("tool_calls", None)
                return _response(stream=True, events=events)
            result = _result()
            if finish_reason == "tool_calls":
                result["choices"][0].update(message=_assistant(), finish_reason=finish_reason)
            return httpx.Response(200, json=result)

        initial = _payload(stream=stream, **stop_options)
        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            assert (await client.post(_path("openai"), json=initial)).status_code == 200
            following = {
                **initial,
                "messages": [*recorder.to_messages(), {"role": "user", "content": "continue"}],
            }
            assert (await client.post(_path("openai"), json=following)).status_code == 200
            assert ("history_tokens" in captured[1]) is reusable

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "stop_reason", ["end_turn", "tool_use", "max_tokens", "pause_turn", "future"]
)
def test_anthropic_only_complete_supported_stop_reasons_seed_cache(stream, stop_reason):
    async def scenario():
        captured = []

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            if len(captured) > 1:
                return _response("anthropic", stream=stream, payload=payload)
            result = _result("anthropic")
            result["stop_reason"] = stop_reason
            events = _events("anthropic")
            events[-2]["delta"]["stop_reason"] = stop_reason
            if stop_reason != "tool_use":
                result["content"] = [{"type": "text", "text": "answer"}]
                events[1:-2] = [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "answer"},
                    },
                    {"type": "content_block_stop", "index": 0},
                ]
            if stream:
                return _response("anthropic", stream=True, events=events)
            return httpx.Response(200, json=result)

        initial = _payload(stream=stream)
        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            assert (await client.post(_path("anthropic"), json=initial)).status_code == 200
            following = {
                **initial,
                "messages": [*recorder.to_messages(), {"role": "user", "content": "continue"}],
            }
            await client.post(_path("anthropic"), json=following)
            assert ("history_tokens" in captured[1]) is (stop_reason in {"end_turn", "tool_use"})

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", ["system", "tools", "thinking", "anthropic-version", "anthropic-beta", "request-id"]
)
def test_anthropic_context_and_protocol_headers_partition_histories(change):
    async def scenario():
        captured = []

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            return _response("anthropic", payload=payload)

        initial = _payload(
            system="initial system",
            tools=_tools("anthropic"),
            thinking={"type": "enabled", "budget_tokens": 8},
            history_tokens=[999],
            token_prefix={"tokens": [999], "message_count": 999},
        )
        headers = {"anthropic-version": "2023-06-01", "anthropic-beta": "thinking-beta"}
        async with _harness(upstream, _enabled_config()) as (client, *_):
            await client.post(_path("anthropic"), json=initial, headers=headers)
            following = _continuation(initial, "anthropic")
            changed_headers = dict(headers)
            if change == "system":
                following["system"] = "changed system"
            elif change == "tools":
                following["tools"][0]["description"] = "changed tool description"
            elif change == "thinking":
                following["thinking"]["budget_tokens"] += 1
            else:
                changed_headers[change] = "changed-value"
            await client.post(_path("anthropic"), json=following, headers=changed_headers)
            assert captured[1]["token_prefix"] == initial["token_prefix"]
            assert ("history_tokens" in captured[1]) is (change == "request-id")
            await client.post(
                _path("anthropic"), json=_continuation(initial, "anthropic"), headers=headers
            )
            assert captured[2]["history_tokens"] == [1, 2, 3, 99]

    asyncio.run(scenario())


@pytest.mark.parametrize("seed_api", ["openai", "anthropic"])
def test_identical_native_histories_do_not_cross_api_protocols(seed_api):
    async def scenario():
        captured = []

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            if len(captured) == 1:
                result = _result(seed_api)
                if seed_api == "openai":
                    result["choices"][0]["message"] = _assistant("anthropic")
                return httpx.Response(200, json=result)
            receiving_api = "anthropic" if seed_api == "openai" else "openai"
            return _response(receiving_api, payload=payload)

        receiving_api = "anthropic" if seed_api == "openai" else "openai"
        initial = _payload()
        async with _harness(upstream, _enabled_config()) as (client, *_):
            await client.post(_path(seed_api), json=initial)
            await client.post(_path(receiving_api), json=_continuation(initial, "anthropic"))
            assert "history_tokens" not in captured[1]

    asyncio.run(scenario())


@pytest.mark.parametrize("unsupported", ["delta", "block", "event", "unfinished-block"])
def test_anthropic_unsupported_sse_forwards_without_seeding_cache(unsupported):
    async def scenario():
        captured = []
        events = _events("anthropic")
        if unsupported == "delta":
            events[1]["content_block"] = {"type": "text", "text": ""}
            events[2]["delta"] = {"type": "text_delta", "text": "answer"}
            events[3]["delta"] = {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": "source",
                    "document_index": 0,
                    "document_title": "document",
                    "start_char_index": 0,
                    "end_char_index": 6,
                },
            }
        elif unsupported == "block":
            events[1]["content_block"]["type"] = "vendor_block"
        elif unsupported == "event":
            events.insert(1, {"type": "vendor_event", "vendor_content": "not reconstructed"})
        else:
            del events[4]  # The thinking block was never closed.
        body = _sse(events)

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            return _response("anthropic", stream=len(captured) == 1, events=events, payload=payload)

        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            response = await client.post(_path("anthropic"), json=_payload(stream=True))
            assert response.content == body
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]
            await client.post(
                _path("anthropic"),
                json=_payload(
                    messages=[*recorder.to_messages(), {"role": "user", "content": "continue"}]
                ),
            )
            assert "history_tokens" not in captured[1]

    asyncio.run(scenario())


def test_anthropic_stream_with_ping_and_separate_token_metadata_remains_reusable():
    async def scenario():
        captured = []
        events = _events("anthropic")
        metadata = {name: events[-2].pop(name) for name in ("token_data", "weight_version")}
        events.insert(-1, metadata)
        events.insert(3, {"type": "ping"})

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            return _response("anthropic", stream=True, events=events, payload=payload)

        initial = _payload(stream=True)
        async with _harness(upstream, _enabled_config()) as (client, *_):
            response = await client.post(_path("anthropic"), json=initial)
            assert response.content == _sse(events)
            await client.post(_path("anthropic"), json=_continuation(initial, "anthropic"))
            assert captured[1]["history_tokens"] == [1, 2, 3, 99]

    asyncio.run(scenario())


def test_anthropic_split_content_deltas_reconstruct_reusable_assistant():
    async def scenario():
        captured = []
        events = _events("anthropic")
        events[6:7] = [
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": fragment},
            }
            for fragment in ('{"command":', '"tr', 'ue"}')
        ]
        events[2:4] = [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": f"{field}_delta", field: fragment},
            }
            for field, fragment in (
                ("thinking", "tho"),
                ("thinking", "ught"),
                ("signature", "native-"),
                ("signature", "signature"),
            )
        ]

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            return _response("anthropic", stream=True, events=events, payload=payload)

        initial = _payload(stream=True)
        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            response = await client.post(_path("anthropic"), json=initial)
            assert response.content == _sse(events)
            assert recorder.to_messages()[-1] == _assistant("anthropic")
            await client.post(_path("anthropic"), json=_continuation(initial, "anthropic"))
            assert captured[1]["history_tokens"] == [1, 2, 3, 99]

    asyncio.run(scenario())


@pytest.mark.parametrize("content_tail", [None, "done"])
def test_sse_preserves_null_content_when_matching(content_tail):
    async def scenario():
        captured = []
        events = _events("openai")
        events[0]["choices"][0]["delta"]["content"] = None
        events[1]["choices"][0]["delta"]["content"] = content_tail

        def upstream(request):
            captured.append(json.loads(request.content))
            return _response(stream=True, events=events, payload=captured[-1])

        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            await client.post(_path("openai"), json=_payload(stream=True))
            assert recorder.to_messages()[-1]["content"] == content_tail
            await client.post(
                _path("openai"),
                json=_payload(
                    stream=True,
                    messages=[
                        *recorder.to_messages(),
                        {"role": "tool", "tool_call_id": "tool-1", "content": "done"},
                    ],
                ),
            )
            assert captured[1]["history_tokens"] == [1, 2, 3, 99]

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["message", "tool", "function"])
def test_unknown_sse_fields_forward_normally_without_caching(field):
    async def scenario():
        captured = []
        events = _events("openai")
        delta = events[1]["choices"][0]["delta"]
        target = delta if field == "message" else delta["tool_calls"][0]
        if field == "function":
            target = target["function"]
        target["vendor_field"] = {"unknown_fragment": True}
        body = _sse(events)

        def upstream(request):
            captured.append(json.loads(request.content))
            return _response(stream=len(captured) == 1, events=events, payload=captured[-1])

        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            response = await client.post(_path("openai"), json=_payload(stream=True))
            assert response.content == body
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]
            await client.post(
                _path("openai"),
                json=_payload(
                    messages=[
                        *recorder.to_messages(),
                        {"role": "tool", "tool_call_id": "tool-1", "content": "done"},
                    ]
                ),
            )
            assert "history_tokens" not in captured[1]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_cancelled_stream_does_not_seed_cache(api):
    async def scenario():
        entered, captured = asyncio.Event(), []

        class PendingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _sse(_events(api)[:-1])
                entered.set()
                await asyncio.Event().wait()

        def upstream(request):
            captured.append(json.loads(request.content))
            if len(captured) == 1:
                return httpx.Response(
                    200, stream=PendingStream(), headers={"content-type": "text/event-stream"}
                )
            return _response(api, stream=True, payload=captured[-1])

        async with _harness(upstream, _enabled_config()) as (client, recorder, *_):
            initial = _payload(stream=True)
            task = asyncio.create_task(client.post(_path(api), json=initial))
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert recorder.segment_count == 0
            await client.post(_path(api), json=_continuation(initial, api))
            assert "history_tokens" not in captured[1]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_retry_keeps_selected_prefix_when_another_completion_makes_cache_ambiguous(
    api, monkeypatch
):
    async def scenario():
        entered_backoff, allow_retry = asyncio.Event(), asyncio.Event()
        captured, continuation_attempts, seed_count = [], 0, 0
        original_sleep = asyncio.sleep

        async def backoff(delay):
            if delay == 0:
                await original_sleep(0)
                return
            entered_backoff.set()
            await allow_retry.wait()

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)

        def upstream(request):
            nonlocal continuation_attempts, seed_count
            payload = json.loads(request.content)
            captured.append((request.content, payload))
            if len(payload["messages"]) == 1:
                seed_count += 1
                token_data = _tokens()
                if seed_count > 1:
                    token_data.update(prompt_tokens=[7, 8], response_tokens=[9, 99])
                return _response(api, token_data=token_data)
            continuation_attempts += 1
            if continuation_attempts == 1:
                return httpx.Response(503, json={"error": {"message": "try again"}})
            return _response(api, payload=payload)

        initial = _payload()
        following = _continuation(initial, api)
        async with _harness(upstream, _enabled_config()) as (client, *_):
            await client.post(_path(api), json=initial)
            task = asyncio.create_task(client.post(_path(api), json=following))
            try:
                await asyncio.wait_for(entered_backoff.wait(), timeout=2)
                await client.post(_path(api), json=initial)
            finally:
                allow_retry.set()
            assert (await task).status_code == 200
            assert captured[1][0] == captured[3][0]
            assert captured[3][1]["history_tokens"] == [1, 2, 3, 99]
            await client.post(_path(api), json=following)
            assert "history_tokens" not in captured[4][1]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_close_clears_prefix_cache(api):
    async def scenario():
        async with _harness(lambda request: _response(api), _enabled_config()) as (
            client,
            _,
            _,
            _,
            relay,
        ):
            await client.post(_path(api), json=_payload())
            assert relay._token_prefix_cache is not None
            await relay.close()
            assert relay._token_prefix_cache is None
            await relay.close()

    asyncio.run(scenario())
