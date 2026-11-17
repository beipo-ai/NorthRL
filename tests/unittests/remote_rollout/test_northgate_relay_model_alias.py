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

"""Client model aliases survive replies without changing captured training data."""

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
)


@pytest.fixture(autouse=True)
def _gateway_key(monkeypatch):
    monkeypatch.setenv("NORTHGATE_API_KEY", "test-ng-key")


MODEL_PAIRS = [
    ("northrl", "weaver-sampler"),
    ("northrl", "northrl"),
    ("weaver-sampler", "weaver-sampler"),
]


def _capture_recordings(recorder, monkeypatch):
    captured = []
    original = recorder.record_inference_result

    def record(result, **kwargs):
        captured.append(deepcopy(result))
        return original(result, **kwargs)

    monkeypatch.setattr(recorder, "record_inference_result", record)
    return captured


@pytest.mark.parametrize("api", ["anthropic", "openai"])
@pytest.mark.parametrize("client_model,upstream_model", MODEL_PAIRS)
def test_json_replies_echo_requested_model_and_record_original(
    api, client_model, upstream_model, monkeypatch
):
    async def scenario():
        result = _result(api)
        result["model"] = upstream_model
        result["vendor_field"]["model"] = "nested-model"
        if api == "anthropic":
            result["content"][1]["input"]["model"] = "tool-model"
        original_body = json.dumps(result, ensure_ascii=False, indent=2).encode()

        def upstream(request):
            assert json.loads(request.content)["model"] == "weaver-sampler"
            return httpx.Response(
                200, content=original_body, headers={"content-type": "application/json"}
            )

        async with _harness(upstream) as (client, recorder, _, _, _):
            captured = _capture_recordings(recorder, monkeypatch)
            response = await client.post(_path(api), json=_payload(model=client_model))
            assert response.json() == {**result, "model": client_model}
            assert int(response.headers["content-length"]) == len(response.content)
            if client_model == upstream_model:
                assert response.content == original_body
            assert captured[0]["model"] == upstream_model
            assert captured[0]["token_data"] == result["token_data"]
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]

    asyncio.run(scenario())


def _framed_event(event, index):
    payload = event if isinstance(event, str) else json.dumps(event, ensure_ascii=False, indent=2)
    event_type = event.get("type", "message") if isinstance(event, dict) else "message"
    lines = [f": frame {index}", f"event: {event_type}", f"id: {index}", "retry: 1000"]
    lines.extend(f"data: {line}" for line in payload.splitlines())
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def _frame_payload(frame):
    parts = [
        line[5:].lstrip(" ") for line in frame.decode().splitlines() if line.startswith("data:")
    ]
    payload = "\n".join(parts)
    if not payload or payload == "[DONE]":
        return payload
    return json.loads(payload)


@pytest.mark.parametrize("api", ["anthropic", "openai"])
@pytest.mark.parametrize("client_model,upstream_model", MODEL_PAIRS)
def test_fragmented_stream_echoes_model_and_preserves_other_frames(
    api, client_model, upstream_model, monkeypatch
):
    async def scenario():
        events = _events(api)
        if api == "anthropic":
            events[0]["message"]["model"] = upstream_model
            events[6]["delta"]["partial_json"] = '{"command":"true","model":"tool-model"}'
        else:
            for event in events:
                if isinstance(event, dict):
                    event["model"] = upstream_model
                    event["vendor_field"] = {"model": "nested-model"}
            events[2]["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ] = '"true","model":"tool-model"}'
        frames = [b": keepalive\r\n\r\n"]
        frames.extend(_framed_event(event, index) for index, event in enumerate(events))
        body = b"".join(frames)

        def upstream(request):
            assert json.loads(request.content)["model"] == "weaver-sampler"
            return httpx.Response(
                200, stream=_Chunks(body), headers={"content-type": "text/event-stream"}
            )

        async with _harness(upstream) as (client, recorder, _, _, _):
            captured = _capture_recordings(recorder, monkeypatch)
            response = await client.post(_path(api), json=_payload(model=client_model, stream=True))
            received = response.content.split(b"\r\n\r\n")
            assert received.pop() == b""
            assert len(received) == len(frames)
            for original, actual in zip(frames, received, strict=True):
                expected = _frame_payload(original)
                rewritten = False
                if isinstance(expected, dict):
                    if api == "anthropic" and expected.get("type") == "message_start":
                        rewritten = expected["message"]["model"] != client_model
                        expected["message"]["model"] = client_model
                    elif api == "openai" and "model" in expected:
                        rewritten = expected["model"] != client_model
                        expected["model"] = client_model
                assert _frame_payload(actual) == expected
                if not rewritten:
                    assert actual + b"\r\n\r\n" == original
                assert [line for line in actual.splitlines() if not line.startswith(b"data:")] == [
                    line
                    for line in original.rstrip(b"\r\n").splitlines()
                    if not line.startswith(b"data:")
                ]
            assert captured[0]["token_data"] == _result(api)["token_data"]
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]
            assistant = recorder.to_messages()[-1]
            if api == "anthropic":
                assert captured[0]["model"] == upstream_model
                assert assistant["content"][0] == {
                    "type": "thinking",
                    "thinking": "thought",
                    "signature": "native-signature",
                }
                assert assistant["content"][1]["input"]["model"] == "tool-model"
            else:
                arguments = assistant["tool_calls"][0]["function"]["arguments"]
                assert json.loads(arguments)["model"] == "tool-model"

    asyncio.run(scenario())
