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

"""Native HTTP forwarding, exact token capture, and stream failure boundaries."""

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from omegaconf import OmegaConf

from northrl.exceptions import ContextLengthExceededError
from northrl.remote_rollout.metrics import RolloutMetrics
from northrl.remote_rollout.northgate_relay import NorthGateRelay, _UpstreamFailure
from northrl.remote_rollout.trajectory_recorder import TrajectoryRecorder


def _config():
    return OmegaConf.create(
        {
            "rollout": {
                "sampling": {"temperature": 0.8, "top_p": 1, "top_k": -1, "max_tokens": 32},
                "keep_sampling_mask": False,
            },
            "validate": {
                "sampling": {"temperature": 0, "top_p": 0.9, "top_k": 7, "max_tokens": 10}
            },
            "data": {"max_sequence_length": 128, "max_prompt_length": 64},
            "algorithm": {
                "sampling_logprob_source": "recomputed",
                "loss": {"mode": "importance_sampling"},
            },
        }
    )


def _tokens():
    return {
        "prompt_tokens": [1, 2],
        "response_tokens": [3, 99],
        "response_logprobs": [-0.1, -0.2],
        "response_old_logprobs": None,
    }


def _result(api="openai"):
    result = {
        "token_data": _tokens(),
        "weight_version": {"revision": 42},
        "vendor_field": {"kept": True},
    }
    if api == "openai":
        result.update(
            {
                "id": "upstream-id",
                "choices": [
                    {"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            }
        )
    else:
        result.update(
            {
                "id": "native-id",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "thought", "signature": "native-signature"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "shell",
                        "input": {"command": "true"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 2, "output_tokens": 2},
            }
        )
    return result


def _path(api):
    return "/v1/messages" if api == "anthropic" else "/v1/chat/completions"


def _payload(**kwargs):
    return {"model": "northrl", "messages": [{"role": "user", "content": "question"}], **kwargs}


def _tools(api):
    definition = {
        "name": "shell",
        "description": "Execute shell commands.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    if api == "anthropic":
        return [{**definition, "cache_control": {"type": "ephemeral"}}]
    definition["parameters"] = definition.pop("input_schema")
    return [{"type": "function", "function": {**definition, "strict": True}}]


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("NORTHGATE_API_KEY", "test-ng-key")


@asynccontextmanager
async def _harness(handler, config=None, **kwargs):
    recorder, metrics, calls = TrajectoryRecorder(), RolloutMetrics(), []

    async def strict_upstream(request):
        options = json.loads(request.content).get("weaver_options", {})
        forbidden = {"num_samples", "sampling_params"} & options.keys()
        if forbidden:
            return httpx.Response(
                400, json={"error": {"message": f"Unsupported weaver_options: {sorted(forbidden)}"}}
            )
        response = handler(request)
        return await response if inspect.isawaitable(response) else response

    relay = NorthGateRelay(
        config or _config(),
        {
            "base_url": "https://northgate.test",
            "model": "weaver-sampler",
            "sampling_session_id": "sampler",
            "weight_version": "trainer-step-1",
        },
        recorder=recorder,
        metrics=metrics,
        on_service_call=lambda operation, error: calls.append((operation, error)),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(strict_upstream)),
        **kwargs,
    )
    app = FastAPI()
    app.include_router(relay.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://local"
    ) as client:
        try:
            yield client, recorder, metrics, calls, relay
        finally:
            await relay.close()


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("effort", [None, "high", "none"])
@pytest.mark.parametrize("agent_effort", [None, "low"])
@pytest.mark.parametrize("is_validation", [False, True])
def test_reasoning_effort_forwarding(api, effort, agent_effort, is_validation):
    async def scenario():
        config = _config()
        config.rollout.sampling.reasoning_effort = "medium"
        config.validate.sampling.reasoning_effort = "medium"
        sampling = config.validate.sampling if is_validation else config.rollout.sampling
        sampling.reasoning_effort = effort
        payload = _payload(weaver_options={"sampling_params": {"seed": 7}})
        if api == "anthropic":
            payload["output_config"] = {"format": {"type": "json_schema"}}
            if agent_effort is not None:
                payload["output_config"]["effort"] = agent_effort
        elif agent_effort is not None:
            payload["reasoning_effort"] = agent_effort
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_result(api))

        async with _harness(upstream, config, is_validation=is_validation) as (client, *_):
            response = await client.post(_path(api), json=payload)
        assert response.status_code == 200
        actual = captured[0]
        expected = effort if effort is not None else agent_effort
        if api == "anthropic":
            assert actual["output_config"] == {
                "format": {"type": "json_schema"},
                **({"effort": expected} if expected is not None else {}),
            }
            assert "reasoning_effort" not in actual
        else:
            assert actual.get("reasoning_effort") == expected
            assert ("reasoning_effort" in actual) == (expected is not None)
        assert "reasoning_effort" not in actual["sampling_params"]
        assert "sampling_params" not in actual["weaver_options"]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_native_json_and_all_request_fields_are_forwarded(api):
    async def scenario():
        response_data = _result(api)
        captured = []

        def upstream(request):
            captured.append(request)
            return httpx.Response(
                200, json=response_data, headers={"x-northgate-request-id": "req-1"}
            )

        payload = _payload(
            max_tokens=999,
            max_completion_tokens=999,
            temperature=0.1,
            tool_choice={"type": "tool", "name": "shell"},
            tools=_tools(api),
            thinking={"type": "enabled", "budget_tokens": 100},
            vendor_request=[1, 2],
            sampling_params={"seed": 8, "max_tokens": 999},
            weaver_options={"custom": {"ok": True}, "return_old_logprob": True},
        )
        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            response = await client.post(
                _path(api),
                json=payload,
                headers={
                    "anthropic-version": "2023-06-01",
                    "authorization": "Bearer local-placeholder",
                    "x-api-key": "local-placeholder",
                },
            )
            assert response.status_code == 200
            assert response.json() == response_data
            assert response.headers["x-northgate-request-id"] == "req-1"
            actual = json.loads(captured[0].content)
            assert actual["model"] == "weaver-sampler"
            assert actual["max_tokens"] == actual["sampling_params"]["max_tokens"] == 32
            assert "max_completion_tokens" not in actual
            assert actual["temperature"] == 0.8
            for field in ("tool_choice", "thinking", "vendor_request"):
                assert actual[field] == payload[field]
            expected_tools = deepcopy(payload["tools"])
            if api == "anthropic":
                expected_tools[0].pop("cache_control")
            assert actual["tools"] == expected_tools
            assert actual["weaver_options"]["custom"] == {"ok": True}
            assert actual["weaver_options"]["return_old_logprob"] is False
            assert "num_samples" not in actual["weaver_options"]
            assert captured[0].headers["anthropic-version"] == "2023-06-01"
            if api == "openai":
                assert captured[0].headers["authorization"] == "Bearer test-ng-key"
                assert "x-api-key" not in captured[0].headers
            else:
                assert captured[0].headers["x-api-key"] == "test-ng-key"
                assert "authorization" not in captured[0].headers
            [segment] = recorder.to_remote_segments()
            assert segment.tokens == [1, 2, 3, 99]
            assert segment.loss_mask == [0, 0, 1, 1]
            assert segment.old_log_probs == [0.0, 0.0, -0.1, -0.2]
            assert recorder.message_format == api
            assert recorder.tools == expected_tools
            assert metrics.snapshot()["model_requests_success"] == 1
            assert calls[-1][1] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("reuse_prefix", [False, True])
def test_cache_control_normalization_preserves_application_data_and_raw_payload(api, reuse_prefix):
    async def scenario():
        marker = {"type": "ephemeral"}
        application_value = {"application": "must survive"}
        payload = _payload(
            cache_control=marker,
            system=[{"type": "text", "text": "system", "cache_control": marker}],
            messages=[
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "previous-tool",
                            "name": "shell",
                            "input": {"command": "true", "cache_control": application_value},
                            "cache_control": marker,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "previous-tool",
                            "content": [
                                {"type": "text", "text": "done", "cache_control": application_value}
                            ],
                            "cache_control": marker,
                        }
                    ],
                },
            ],
            tools=[
                {
                    "name": "shell",
                    "cache_control": marker,
                    "input_schema": {
                        "type": "object",
                        "properties": {"cache_control": {"type": "string"}},
                        "cache_control": application_value,
                    },
                }
            ],
            vendor_request={"cache_control": application_value},
        )
        original = deepcopy(payload)
        expected = deepcopy(payload)
        if api == "anthropic":
            del expected["cache_control"]
            del expected["system"][0]["cache_control"]
            del expected["messages"][1]["content"][0]["cache_control"]
            del expected["messages"][2]["content"][0]["cache_control"]
            del expected["tools"][0]["cache_control"]
        config = _config()
        config.rollout.execution = {"token_prefix_reuse": reuse_prefix}
        captured = []

        def upstream(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_result(api))

        async with _harness(upstream, config) as (client, _, _, _, relay):
            # Call with the original object too: HTTP serialization would hide mutations.
            normalized = relay._payload(payload, api)  # pylint: disable=protected-access
            assert payload == original
            response = await client.post(_path(api), json=payload)
            assert response.status_code == 200
            assert captured == [normalized]
            for field in ("system", "messages", "tools", "vendor_request"):
                assert normalized[field] == expected[field]
            assert ("cache_control" in normalized) is (api == "openai")
            if api == "openai":
                assert normalized["cache_control"] == marker

    asyncio.run(scenario())


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        for offset in range(0, len(self.body), 7):
            yield self.body[offset : offset + 7]


def _sse(events):
    return b"".join(
        (
            "data: "
            + (event if isinstance(event, str) else json.dumps(event, ensure_ascii=False))
            + "\r\n\r\n"
        ).encode()
        for event in events
    )


def _events(api):
    metadata = {"token_data": _tokens(), "weight_version": 42}
    if api == "openai":
        return [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "答"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "案",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tool-1",
                                    "type": "function",
                                    "function": {"name": "shell", "arguments": '{"command":'},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '"true"}'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [], **metadata},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
            "[DONE]",
        ]
    return [
        {
            "type": "message_start",
            "message": {
                "id": "native-stream",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 2, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "thought"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "native-signature"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tool-1", "name": "shell", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"command":"true"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 2},
            **metadata,
        },
        {"type": "message_stop"},
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "history_change", [False, True], ids=["unchanged", "business-field-changed"]
)
def test_anthropic_moving_cache_markers_reuse_tokens_without_ignoring_content_changes(
    stream, history_change
):
    async def scenario():
        config = _config()
        config.rollout.execution = {"token_prefix_reuse": True}
        payload = _payload(
            stream=stream,
            cache_control={"type": "ephemeral"},
            system=[{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}],
            tools=_tools("anthropic"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "question", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ],
        )
        captured: list[dict[str, Any]] = []
        assistants = []
        completed_tokens: list[list[int]] = []

        def upstream(request):
            actual = json.loads(request.content)
            turn = len(captured)
            captured.append(actual)
            if turn and not (turn == 2 and history_change):
                assert actual["history_tokens"] == completed_tokens[-1]
            else:
                assert "history_tokens" not in actual
            # Model the gateway's token splice, without exercising its renderer.
            token_data = _tokens()
            token_data["prompt_tokens"] = actual.get("history_tokens", []) + [
                10 * turn + 1,
                10 * turn + 2,
            ]
            token_data["response_tokens"] = [10 * turn + 3, 99]
            completed_tokens.append(token_data["prompt_tokens"] + token_data["response_tokens"])
            result = _result("anthropic")
            result["token_data"] = token_data
            tool = result["content"][1]
            tool["id"] = f"tool-{turn + 1}"
            tool["input"]["cache_control"] = "application value"
            assistants.append({"role": "assistant", "content": result["content"]})
            if not stream:
                return httpx.Response(200, json=result)
            events = _events("anthropic")
            events[5]["content_block"]["id"] = tool["id"]
            events[6]["delta"]["partial_json"] = json.dumps(tool["input"])
            events[-2]["token_data"] = token_data
            return httpx.Response(
                200, stream=_Chunks(_sse(events)), headers={"content-type": "text/event-stream"}
            )

        async with _harness(upstream, config) as (client, recorder, *_):
            for turn in range(3):
                if turn == 2 and history_change:
                    payload["messages"][1]["content"][1]["input"]["cache_control"] = "changed"
                response = await client.post(_path("anthropic"), json=payload)
                assert response.status_code == 200
                if turn < 2:
                    # Claude moves the marker from the old user block to the new result.
                    payload["messages"][-1]["content"][-1].pop("cache_control")
                    payload["messages"] += [
                        deepcopy(assistants[-1]),
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": f"tool-{turn + 1}",
                                    "content": "done",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ]
                    ttl = ("1h", "5m")[turn]
                    payload["cache_control"]["ttl"] = ttl
                    payload["system"][0]["cache_control"]["ttl"] = ttl
                    payload["tools"][0]["cache_control"]["ttl"] = ttl
            if not history_change:
                [segment] = recorder.to_remote_segments()
                assert segment.tokens == [1, 2, 3, 99, 11, 12, 13, 99, 21, 22, 23, 99]
                assert segment.loss_mask == [0, 0, 1, 1] * 3
            else:
                assert recorder.segment_count == 2
                tool_input = captured[-1]["messages"][1]["content"][1]["input"]
                assert tool_input["cache_control"] == "changed"

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_current_northgate_contract_normalizes_legacy_options(api, stream, legacy):
    async def scenario():
        captured = []
        body = _sse(_events(api))

        def upstream(request):
            captured.append(json.loads(request.content))
            if stream:
                return httpx.Response(
                    200, stream=_Chunks(body), headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(200, json=_result(api))

        payload = _payload(
            stream=stream,
            n=1,
            sampling_params={"seed": 7, "min_p": 0.1, "temperature": 999, "max_new_tokens": 999},
        )
        if legacy:
            payload["weaver_options"] = {
                "num_samples": 5,
                "sampling_params": {"seed": 13, "temperature": 999, "max_new_tokens": 999},
                "return_old_logprob": True,
                "custom": {"ok": True},
            }
        async with _harness(upstream) as (client, recorder, _, _, _):
            response = await client.post(_path(api), json=payload)
            assert response.status_code == 200
            assert recorder.segment_count == 1
        if stream:
            assert response.content == body
        else:
            assert response.json() == _result(api)
        [actual] = captured
        assert actual["n"] == 1
        assert actual["sampling_params"] == {
            "seed": 13 if legacy else 7,
            **({} if legacy else {"min_p": 0.1}),
            "temperature": 0.8,
            "top_p": 1,
            "top_k": -1,
            "max_tokens": 32,
        }
        assert actual["weaver_options"] == {
            **({"custom": {"ok": True}} if legacy else {}),
            "return_old_logprob": False,
            "return_sampling_mask": False,
            "return_moe_topk_indices": False,
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "override",
    [
        {"sampling_params": []},
        {"weaver_options": []},
        {"weaver_options": {"sampling_params": []}},
        {"weaver_options": {"sampling_params": None}},
        {"n": 2},
        {"n": True},
    ],
)
def test_invalid_sampling_options_and_multiple_completions_are_rejected_locally(api, override):
    async def scenario():
        async with _harness(lambda request: pytest.fail("no upstream request expected")) as (
            client,
            recorder,
            metrics,
            calls,
            _,
        ):
            response = await client.post(_path(api), json=_payload(**override))
            assert response.status_code == 400
            assert recorder.segment_count == 0
            assert metrics.snapshot()["model_requests_started"] == 0
            assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_stream_is_forwarded_byte_for_byte_and_records_native_tools(api):
    async def scenario():
        body = _sse(_events(api))

        def upstream(request):
            return httpx.Response(
                200, stream=_Chunks(body), headers={"content-type": "text/event-stream"}
            )

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            response = await client.post(_path(api), json=_payload(stream=True, tools=_tools(api)))
            assert response.content == body
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]
            expected_tools = _tools(api)
            if api == "anthropic":
                expected_tools[0].pop("cache_control")
            assert recorder.tools == expected_tools
            assistant = recorder.to_messages()[-1]
            if api == "openai":
                assert assistant["content"] == "答案"
                assert assistant["tool_calls"][0]["function"]["arguments"] == '{"command":"true"}'
            else:
                assert assistant["content"][0] == {
                    "type": "thinking",
                    "thinking": "thought",
                    "signature": "native-signature",
                }
                assert assistant["content"][1]["input"] == {"command": "true"}
            assert metrics.snapshot()["tool_calls_emitted"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0
            assert calls[-1][1] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "diagnostics,expected",
    [
        pytest.param(
            {
                "checked": True,
                "malformed": True,
                "detected": 2,
                "invalid": 1,
                "reasons": ["undeclared_tool"],
            },
            {
                "tool_calls_detected": 2,
                "tool_calls_invalid": 1,
                "tool_format_error_turns": 1,
                "tool_format_unknown_turns": 0,
            },
            id="raw-error-after-native-recovery",
        ),
        pytest.param(
            {
                "checked": True,
                "malformed": False,
                "detected": 1,
                "invalid": 0,
                "reasons": [],
            },
            {
                "tool_calls_detected": 1,
                "tool_calls_invalid": 0,
                "tool_format_error_turns": 0,
                "tool_format_unknown_turns": 0,
            },
            id="checked-clean",
        ),
        pytest.param(
            {
                "checked": True,
                "malformed": True,
                "detected": None,
                "invalid": None,
                "reasons": ["incomplete_tool_call"],
            },
            {"tool_format_error_turns": 1, "tool_format_unknown_turns": 0},
            id="malformed-with-unknown-counts",
        ),
        pytest.param(
            {
                "checked": False,
                "malformed": False,
                "detected": None,
                "invalid": None,
                "reasons": [],
            },
            {"tool_format_error_turns": 0, "tool_format_unknown_turns": 1},
            id="unchecked",
        ),
        pytest.param(
            None,
            {"tool_format_error_turns": 0, "tool_format_unknown_turns": 1},
            id="legacy-without-diagnostics",
        ),
    ],
)
def test_native_tool_diagnostics_reach_metrics_without_changing_forwarded_output(
    api, stream, diagnostics, expected
):
    async def scenario():
        response_data = _result(api)
        events = _events(api)
        if diagnostics is not None:
            response_data["tool_call_diagnostics"] = diagnostics
            [metadata] = [
                event for event in events if isinstance(event, dict) and "token_data" in event
            ]
            # These are the actual NorthGate metadata locations, not assistant deltas.
            if api == "openai":
                assert metadata["choices"] == []
            else:
                assert metadata["type"] == "message_delta"
            metadata["tool_call_diagnostics"] = diagnostics
        body = _sse(events)

        def upstream(_request):
            if stream:
                return httpx.Response(
                    200, stream=_Chunks(body), headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(200, json=response_data)

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            response = await client.post(
                _path(api), json=_payload(stream=stream, tools=_tools(api))
            )
            assert response.status_code == 200
            if stream:
                assert response.content == body
            else:
                assert response.json() == response_data
            [segment] = recorder.to_remote_segments()
            assert segment.tokens == [1, 2, 3, 99]
            assert segment.loss_mask == [0, 0, 1, 1]
            assert segment.log_probs == [0.0, 0.0, -0.1, -0.2]
            assert segment.old_log_probs == [0.0, 0.0, -0.1, -0.2]
            [annotation] = segment.turn_annotations
            assert annotation == {
                "version": 1,
                "turn_id": "turn-0",
                "response_span": [2, 4],
                "diagnostics": diagnostics,
                "finish_reason": (
                    "tool_use" if api == "anthropic" else "tool_calls" if stream else "stop"
                ),
                "truncated": False,
                "tools_declared": True,
                "source": "northgate",
            }
            expected_tools = _tools(api)
            if api == "anthropic":
                expected_tools[0].pop("cache_control")
            assert recorder.tools == expected_tools
            snapshot = metrics.snapshot()
            for name, value in expected.items():
                assert snapshot[name] == value, name
            # Raw syntax can contain calls filtered out of the native result.
            assert snapshot["tool_calls_emitted"] == int(stream or api == "anthropic")
            assert snapshot["model_turns"] == 1
            assert snapshot["model_requests_success"] == 1
            assert calls[-1][1] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "api,finish_reason,custom_stop,expected_truncated",
    [
        ("openai", "stop", False, False),
        ("openai", "length", False, True),
        ("openai", "stop", True, True),
        ("anthropic", "end_turn", False, False),
        ("anthropic", "max_tokens", False, True),
        ("anthropic", "stop_sequence", False, True),
        ("anthropic", "end_turn", True, True),
    ],
)
def test_turn_annotations_record_final_stop_provenance(
    api, stream, finish_reason, custom_stop, expected_truncated
):
    async def scenario():
        response_data = _result(api)
        events = _events(api)
        if api == "openai":
            response_data["choices"][0]["finish_reason"] = finish_reason
            for event in events:
                for choice in event.get("choices", []) if isinstance(event, dict) else []:
                    if choice.get("finish_reason") is not None:
                        choice["finish_reason"] = finish_reason
        else:
            response_data["stop_reason"] = finish_reason
            for event in events:
                if event.get("type") == "message_delta":
                    event["delta"]["stop_reason"] = finish_reason

        def upstream(_request):
            if stream:
                return httpx.Response(
                    200,
                    stream=_Chunks(_sse(events)),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=response_data)

        async with _harness(upstream) as (client, recorder, _, _, _):
            payload = _payload(stream=stream)
            if custom_stop:
                payload["stop" if api == "openai" else "stop_sequences"] = ["<cut>"]
            response = await client.post(_path(api), json=payload)
            assert response.status_code == 200
            [annotation] = recorder.to_remote_segments()[0].turn_annotations
            assert annotation["finish_reason"] == finish_reason
            assert annotation["truncated"] is expected_truncated
            assert annotation["diagnostics"] is None
            assert annotation["tools_declared"] is False

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_tool_diagnostics_preserve_token_prefix_reuse_and_native_history(api, stream):
    async def scenario():
        config = _config()
        config.rollout.execution = {"token_prefix_reuse": True}
        diagnostics = {
            "checked": True,
            "malformed": True,
            "detected": 2,
            "invalid": 1,
            "reasons": ["invalid_arguments_json"],
        }
        captured = []

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            tokens = _tokens()
            if len(captured) == 2:
                assert payload["history_tokens"] == [1, 2, 3, 99]
                tokens["prompt_tokens"] = [*payload["history_tokens"], 5]
            if stream:
                events = _events(api)
                for event in events:
                    if isinstance(event, dict) and "token_data" in event:
                        event["token_data"] = tokens
                        event["tool_call_diagnostics"] = diagnostics
                return httpx.Response(
                    200, stream=_Chunks(_sse(events)), headers={"content-type": "text/event-stream"}
                )
            result = _result(api)
            result["token_data"] = tokens
            result["tool_call_diagnostics"] = diagnostics
            return httpx.Response(200, json=result)

        async with _harness(upstream, config) as (client, recorder, metrics, _, _):
            initial = _payload(stream=stream, tools=_tools(api))
            assert (await client.post(_path(api), json=initial)).status_code == 200
            messages = recorder.to_messages()
            assert all("tool_call_diagnostics" not in message for message in messages)
            if api == "anthropic":
                continuation = {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool-1", "content": "done"}
                    ],
                }
            elif stream:
                continuation = {"role": "tool", "tool_call_id": "tool-1", "content": "done"}
            else:
                continuation = {"role": "user", "content": "continue"}
            following = {**initial, "messages": [*messages, continuation]}
            assert (await client.post(_path(api), json=following)).status_code == 200
            assert captured[1]["messages"] == following["messages"]
            [segment] = recorder.to_remote_segments()
            assert segment.tokens == [1, 2, 3, 99, 5, 3, 99]
            assert segment.loss_mask == [0, 0, 1, 1, 0, 1, 1]
            assert segment.log_probs == [0.0, 0.0, -0.1, -0.2, 0.0, -0.1, -0.2]
            assert segment.old_log_probs == segment.log_probs
            snapshot = metrics.snapshot()
            assert snapshot["tool_calls_detected"] == 4
            assert snapshot["tool_calls_invalid"] == 2
            assert snapshot["tool_format_error_turns"] == 2
            assert snapshot["tool_format_unknown_turns"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [429, 500, 503, 599])
def test_http_retries_back_off_until_success_and_close_responses(api, stream, status, monkeypatch):
    async def scenario():
        failures = [status, status]
        requests, responses, delays = [], [], []
        original_sleep = asyncio.sleep
        body = _sse(_events(api)) if stream else json.dumps(_result(api)).encode()

        async def backoff(delay):
            if delay == 0:
                await original_sleep(0)
                return
            assert all(response.is_closed for response in responses)
            delays.append(delay)
            await original_sleep(0)

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: high
        )

        def upstream(request):
            requests.append(request)
            index = len(requests) - 1
            failed = index < len(failures)
            response = httpx.Response(
                failures[index] if failed else 200,
                stream=_Chunks(
                    b'{"error":{"code":"sampler_not_ready","message":"temporarily unavailable"}}'
                    if failed
                    else body
                ),
                headers={
                    "content-type": (
                        "text/event-stream" if stream and not failed else "application/json"
                    ),
                    "x-northgate-request-id": "recovered",
                },
            )
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            response = await client.post(_path(api), json=_payload(stream=stream))
            assert response.status_code == 200
            assert response.content == body
            assert response.headers["x-northgate-request-id"] == "recovered"
            assert len(requests) == len(failures) + 1
            assert delays == [1, 2]
            assert all(request.content == requests[0].content for request in requests)
            assert all(response.is_closed for response in responses)
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]
            snapshot = metrics.snapshot()
            assert snapshot["model_requests_started"] == 1
            assert snapshot["model_requests_completed"] == 1
            assert snapshot["model_requests_success"] == 1
            assert snapshot["model_requests_error"] == 0
            assert snapshot["model_requests_timeout"] == 0
            assert snapshot["model_request_inflight"] == 0
            assert len(calls) == 1 and calls[0][1] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 408])
def test_other_http_errors_raise_once_and_close_response(api, stream, status, monkeypatch):
    async def scenario():
        requests, responses = [], []

        async def unexpected_backoff(delay):
            pytest.fail(f"HTTP {status} must not retry (delay={delay})")

        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.asyncio.sleep", unexpected_backoff
        )

        def upstream(request):
            requests.append(request)
            response = httpx.Response(
                status,
                stream=_Chunks(b'{"error":{"message":"request rejected"}}'),
                headers={"content-type": "application/json"},
            )
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            with pytest.raises(_UpstreamFailure, match=f"HTTP {status}") as failure:
                await client.post(_path(api), json=_payload(stream=stream))
            assert len(requests) == 1
            assert responses[0].is_closed
            assert recorder.segment_count == 0
            assert len(calls) == 1 and calls[0][1] is failure.value
            assert metrics.snapshot()["model_requests_completed"] == 1
            assert metrics.snapshot()["model_requests_error"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "error_type",
    [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.WriteError],
)
def test_transport_retries_stop_at_attempt_limit_and_raise_last_error(
    api, stream, error_type, monkeypatch
):
    async def scenario():
        requests = []
        error = error_type("upstream transport failed")

        delays = []
        original_sleep = asyncio.sleep

        async def backoff(delay):
            delays.append(delay)
            await original_sleep(0)

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: high
        )

        def upstream(request):
            requests.append(request)
            raise error

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            with pytest.raises(error_type) as failure:
                await client.post(_path(api), json=_payload(stream=stream))
            assert failure.value is error
            assert len(requests) == 3
            assert delays == [1, 2]
            assert recorder.segment_count == 0
            assert len(calls) == 1 and calls[0][1] is error
            snapshot = metrics.snapshot()
            status = "timeout" if error_type is httpx.ReadTimeout else "error"
            assert snapshot[f"model_requests_{status}"] == 1
            assert snapshot["model_requests_completed"] == 1
            assert snapshot["model_request_inflight"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("status", [429, 503])
def test_retry_backoff_can_be_cancelled_and_closes_response(api, status, monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        requests, responses = [], []
        original_sleep = asyncio.sleep

        async def backoff(delay):
            if delay == 0:
                await original_sleep(0)
                return
            assert delay == 1
            assert responses[0].is_closed
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: high
        )

        def upstream(request):
            requests.append(request)
            response = httpx.Response(
                status,
                stream=_Chunks(b'{"error":{"message":"unavailable"}}'),
                headers={"content-type": "application/json"},
            )
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            task = asyncio.create_task(client.post(_path(api), json=_payload()))
            await entered.wait()
            assert metrics.snapshot()["model_request_inflight"] == 1
            assert calls == []
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(requests) == 1
            assert responses[0].is_closed
            assert recorder.segment_count == 0
            assert metrics.snapshot()["model_requests_cancelled"] == 1
            assert metrics.snapshot()["model_requests_completed"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0
            assert len(calls) == 1
            assert f"while retrying HTTP {status}" in str(calls[0][1])
            assert isinstance(calls[0][1].__cause__, asyncio.CancelledError)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("recover", [False, True])
def test_http_retries_have_a_finite_attempt_limit(status, recover, monkeypatch):
    async def scenario():
        requests, responses, delays = [], [], []
        original_sleep = asyncio.sleep

        async def backoff(delay):
            delays.append(delay)
            await original_sleep(0)

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: high
        )

        def upstream(request):
            requests.append(request)
            success = recover and len(requests) == 3
            response = httpx.Response(
                200 if success else status,
                json=_result() if success else {"error": {"message": f"failure {len(requests)}"}},
            )
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            if recover:
                assert (await client.post(_path("openai"), json=_payload())).status_code == 200
                assert recorder.segment_count == 1
                assert len(calls) == 1 and calls[0][1] is None
            else:
                with pytest.raises(_UpstreamFailure, match="failure 3") as failure:
                    await client.post(_path("openai"), json=_payload())
                assert recorder.segment_count == 0
                assert len(calls) == 1 and calls[0][1] is failure.value
            assert len(requests) == 3
            assert delays == [1, 2]
            assert all(response.is_closed for response in responses)
            assert metrics.snapshot()["model_requests_started"] == 1
            assert metrics.snapshot()["model_requests_completed"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["seconds", "http_date", "invalid", "past"])
def test_retry_after_is_respected_with_jitter_as_a_minimum(kind, monkeypatch):
    async def scenario():
        requests, delays = [], []
        original_sleep = asyncio.sleep
        retry_after = {
            "seconds": "5",
            "http_date": format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30)),
            "invalid": "not-a-date",
            "past": "Wed, 21 Oct 2015 07:28:00 GMT",
        }[kind]

        async def backoff(delay):
            delays.append(delay)
            await original_sleep(0)

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: low
        )

        def upstream(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(
                    429,
                    json={"error": {"message": "rate limited"}},
                    headers={"Retry-After": retry_after},
                )
            return httpx.Response(200, json=_result())

        async with _harness(upstream) as (client, _, _, calls, _):
            assert (await client.post(_path("openai"), json=_payload())).status_code == 200
            assert len(requests) == 2
            assert len(delays) == 1
            if kind == "http_date":
                assert 28 <= delays[0] <= 30
            else:
                assert delays == [5 if kind == "seconds" else 0.5]
            assert len(calls) == 1 and calls[0][1] is None

    asyncio.run(scenario())


def test_retry_after_longer_than_remaining_budget_does_not_retry_early(monkeypatch):
    monkeypatch.setattr("northrl.remote_rollout.northgate_relay._RETRY_WINDOW_SECONDS", 1)

    async def scenario():
        requests = []

        async def unexpected_backoff(delay):
            pytest.fail(f"Cannot honor Retry-After within the remaining budget: {delay}")

        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.asyncio.sleep", unexpected_backoff
        )

        def upstream(request):
            requests.append(request)
            return httpx.Response(
                429, json={"error": {"message": "rate limited"}}, headers={"Retry-After": "600"}
            )

        async with _harness(upstream) as (
            client,
            recorder,
            metrics,
            calls,
            _,
        ):
            with pytest.raises(TimeoutError) as failure:
                await client.post(_path("openai"), json=_payload())
            assert len(requests) == 1
            assert recorder.segment_count == 0
            assert len(calls) == 1 and calls[0][1] is failure.value
            assert metrics.snapshot()["model_requests_timeout"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["error_body", "backoff"])
def test_total_retry_budget_interrupts_blocked_operations(phase, monkeypatch):
    monkeypatch.setattr("northrl.remote_rollout.northgate_relay._RETRY_WINDOW_SECONDS", 0.02)

    async def scenario():
        requests, responses = [], []
        original_sleep = asyncio.sleep

        class BlockingBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.Event().wait()
                yield b"unreachable"

        async def backoff(delay):
            if phase == "backoff":
                await asyncio.Event().wait()
            else:
                await original_sleep(0)

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: 0.001
        )

        async def upstream(request):
            requests.append(request)
            response = httpx.Response(
                503,
                stream=BlockingBody() if phase == "error_body" else _Chunks(b"unavailable"),
            )
            responses.append(response)
            return response

        async with _harness(upstream) as (
            client,
            recorder,
            metrics,
            calls,
            _,
        ):
            with pytest.raises(TimeoutError) as failure:
                await asyncio.wait_for(client.post(_path("openai"), json=_payload()), timeout=0.5)
            assert len(requests) == 1
            assert all(response.is_closed for response in responses)
            assert recorder.segment_count == 0
            # An outer test timeout would cancel the request instead of recording a timeout.
            assert metrics.snapshot()["model_requests_timeout"] == 1
            assert metrics.snapshot()["model_requests_cancelled"] == 0
            assert metrics.snapshot()["model_request_inflight"] == 0
            assert len(calls) == 1 and calls[0][1] is failure.value

    asyncio.run(scenario())


@pytest.mark.parametrize("budget,connect_limit", [(300, 120), (2, 2)])
def test_each_attempt_connection_timeout_is_capped_by_remaining_budget(
    budget, connect_limit, monkeypatch
):
    monkeypatch.setattr("northrl.remote_rollout.northgate_relay._RETRY_WINDOW_SECONDS", budget)

    async def scenario():
        captured = []

        def upstream(request):
            captured.append(request)
            return httpx.Response(200, json=_result())

        async with _harness(upstream) as (
            client,
            _,
            _,
            _,
            relay,
        ):
            relay._http_client.timeout = httpx.Timeout(None, connect=120)
            await client.post(_path("openai"), json=_payload())
            [request] = captured
            assert 0 < request.extensions["timeout"]["connect"] <= connect_limit
            assert request.extensions["timeout"]["read"] is None

    asyncio.run(scenario())


def test_retry_budget_is_shared_across_attempts(monkeypatch):
    monkeypatch.setattr("northrl.remote_rollout.northgate_relay._RETRY_WINDOW_SECONDS", 0.04)

    async def scenario():
        requests = []
        original_sleep = asyncio.sleep
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: 0.001
        )

        async def upstream(request):
            requests.append(request)
            await original_sleep(0.025)
            return httpx.Response(503, json={"error": {"message": "unavailable"}})

        async with _harness(upstream) as (
            client,
            _,
            metrics,
            calls,
            _,
        ):
            with pytest.raises(TimeoutError) as failure:
                await asyncio.wait_for(client.post(_path("openai"), json=_payload()), timeout=0.5)
            assert len(requests) == 2
            assert requests[1].extensions["timeout"]["connect"] < 0.025
            assert metrics.snapshot()["model_requests_timeout"] == 1
            assert len(calls) == 1 and calls[0][1] is failure.value

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("phase", ["headers", "body"])
def test_normal_generation_can_outlast_the_retry_window(stream, phase, monkeypatch):
    monkeypatch.setattr("northrl.remote_rollout.northgate_relay._RETRY_WINDOW_SECONDS", 0.02)

    async def scenario():
        class SlowBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                if phase == "body":
                    await asyncio.sleep(0.04)
                yield _sse(_events("openai")) if stream else json.dumps(_result()).encode()

        async def upstream(request):
            if phase == "headers":
                await asyncio.sleep(0.04)
            return httpx.Response(
                200,
                stream=SlowBody(),
                headers={"content-type": "text/event-stream" if stream else "application/json"},
            )

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            response = await client.post(_path("openai"), json=_payload(stream=stream))
            assert response.status_code == 200
            assert recorder.segment_count == 1
            assert metrics.snapshot()["model_requests_success"] == 1
            assert len(calls) == 1 and calls[0][1] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("error_type", [httpx.UnsupportedProtocol, httpx.LocalProtocolError])
def test_nontransient_transport_errors_are_not_retried(error_type, monkeypatch):
    async def scenario():
        requests = []
        error = error_type("invalid request")

        async def unexpected_backoff(delay):
            pytest.fail(f"A local request error must not retry: {delay}")

        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.asyncio.sleep", unexpected_backoff
        )

        def upstream(request):
            requests.append(request)
            raise error

        async with _harness(upstream) as (client, _, _, calls, _):
            with pytest.raises(error_type) as failure:
                await client.post(_path("openai"), json=_payload())
            assert len(requests) == 1
            assert failure.value is error
            assert len(calls) == 1 and calls[0][1] is error

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [200, 404, 429, 503])
@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.ReadTimeout])
def test_failed_body_retries_only_a_confirmed_retryable_http_status(
    status, error_type, monkeypatch
):
    async def scenario():
        requests, responses, delays = [], [], []
        error = error_type("body interrupted after response headers")
        retryable = status in (429, 503)
        original_sleep = asyncio.sleep

        class FailingBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"id":"upstream-id",'
                raise error

        async def backoff(delay):
            assert retryable, "Only a known retryable HTTP failure may be replayed"
            assert all(response.is_closed for response in responses)
            delays.append(delay)
            await original_sleep(0)

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: high
        )

        def upstream(request):
            requests.append(request)
            response = httpx.Response(status, stream=FailingBody())
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            with pytest.raises(error_type) as failure:
                await client.post(_path("openai"), json=_payload())
            assert len(requests) == (3 if retryable else 1)
            assert delays == ([1, 2] if retryable else [])
            assert all(response.is_closed for response in responses)
            assert recorder.segment_count == 0
            assert failure.value is error
            assert len(calls) == 1 and calls[0][1] is error
            assert metrics.snapshot()["model_requests_completed"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("retry_after", [None, "5"])
def test_interrupted_retryable_error_body_can_recover_and_respects_retry_after(
    status, retry_after, monkeypatch
):
    async def scenario():
        requests, responses, delays = [], [], []
        original_sleep = asyncio.sleep

        class FailingBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"error":'
                raise httpx.ReadError("error body interrupted")

        async def backoff(delay):
            assert all(response.is_closed for response in responses)
            delays.append(delay)
            await original_sleep(0)

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)
        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.random.uniform", lambda low, high: high
        )

        def upstream(request):
            requests.append(request)
            if len(requests) == 1:
                response = httpx.Response(
                    status,
                    stream=FailingBody(),
                    headers={"Retry-After": retry_after} if retry_after is not None else {},
                )
            else:
                response = httpx.Response(200, json=_result())
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            assert (await client.post(_path("openai"), json=_payload())).status_code == 200
            assert len(requests) == 2
            assert delays == [5 if retry_after is not None else 1]
            assert all(response.is_closed for response in responses)
            assert recorder.segment_count == 1
            assert len(calls) == 1 and calls[0][1] is None
            assert metrics.snapshot()["model_requests_completed"] == 1
            assert metrics.snapshot()["model_requests_success"] == 1

    asyncio.run(scenario())


def test_matching_request_recovers_previous_failure_only_after_final_retry_success(monkeypatch):
    async def scenario():
        retry_started, allow_retry = asyncio.Event(), asyncio.Event()
        requests = []

        async def backoff(delay):
            retry_started.set()
            await allow_retry.wait()

        monkeypatch.setattr("northrl.remote_rollout.northgate_relay.asyncio.sleep", backoff)

        def upstream(request):
            requests.append(request)
            if len(requests) <= 2:
                return httpx.Response(
                    400 if len(requests) == 1 else 503, json={"error": {"message": "failure"}}
                )
            return httpx.Response(200, json=_result())

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            with pytest.raises(_UpstreamFailure):
                await client.post(_path("openai"), json=_payload())
            failed_operation = calls[0][0]
            task = asyncio.create_task(client.post(_path("openai"), json=_payload()))
            await asyncio.wait_for(retry_started.wait(), timeout=0.5)
            assert len(calls) == 1
            assert metrics.snapshot()["model_requests_completed"] == 1
            assert (failed_operation, None) not in calls
            allow_retry.set()
            assert (await task).status_code == 200
            assert len(requests) == 3
            assert recorder.segment_count == 1
            assert len(calls) == 3
            assert calls[1] == (failed_operation, None)
            assert calls[2][1] is None
            assert metrics.snapshot()["model_requests_started"] == 2
            assert metrics.snapshot()["model_requests_completed"] == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_invalid_json_raises_once_and_closes_response(api):
    async def scenario():
        requests, responses = [], []

        def upstream(request):
            requests.append(request)
            response = httpx.Response(
                200, stream=_Chunks(b"{invalid json"), headers={"content-type": "application/json"}
            )
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            with pytest.raises(json.JSONDecodeError) as failure:
                await client.post(_path(api), json=_payload())
            assert len(requests) == 1
            assert responses[0].is_closed
            assert recorder.segment_count == 0
            assert calls[-1][1] is failure.value
            assert metrics.snapshot()["model_requests_error"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("failure", ["event", "context", "malformed", "transport"])
def test_stream_failures_raise_without_replaying_and_close_response(api, failure, monkeypatch):
    async def scenario():
        requests, responses = [], []
        original_sleep = asyncio.sleep

        async def unexpected_backoff(delay):
            if delay == 0:
                await original_sleep(0)
                return
            pytest.fail(f"An HTTP 200 stream must not retry (delay={delay})")

        monkeypatch.setattr(
            "northrl.remote_rollout.northgate_relay.asyncio.sleep", unexpected_backoff
        )
        if failure == "context":
            event = {"error": {"code": "context_length_exceeded", "message": "context exceeded"}}
            expected = ContextLengthExceededError
        else:
            event = {"error": {"code": "server_error", "message": "unavailable", "status": 503}}
            expected = _UpstreamFailure
        ending = b"data: {invalid json\n\n" if failure == "malformed" else _sse([event])
        if failure == "malformed":
            expected = json.JSONDecodeError
        elif failure == "transport":
            expected = httpx.ReadError

        class FailingStream(_Chunks):
            async def __aiter__(self):
                yield _sse(_events(api)[:1])
                if failure == "transport":
                    raise httpx.ReadError("stream interrupted")
                yield ending

        def upstream(request):
            requests.append(request)
            response = httpx.Response(
                200, stream=FailingStream(b""), headers={"content-type": "text/event-stream"}
            )
            responses.append(response)
            return response

        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            with pytest.raises(Exception):
                await client.post(_path(api), json=_payload(stream=True))
            assert len(requests) == 1
            assert responses[0].is_closed
            assert recorder.segment_count == 0
            assert len(calls) == 1 and isinstance(calls[0][1], expected)
            assert metrics.snapshot()["model_requests_error"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "override",
    [
        {"token_data": None},
        {"token_data": {**_tokens(), "prompt_tokens": None}},
        {"token_data": {**_tokens(), "response_tokens": "invalid"}},
        {"token_data": {**_tokens(), "response_tokens": [3]}},
        {"token_data": {**_tokens(), "response_logprobs": None}},
        {"token_data": {**_tokens(), "response_logprobs": [float("nan"), -0.2]}},
        {"token_data": {**_tokens(), "response_old_logprobs": [-0.3]}},
    ],
)
def test_invalid_training_metadata_is_a_service_failure(override):
    async def scenario():
        async with _harness(
            lambda request: httpx.Response(
                200,
                content=json.dumps({**_result(), **override}).encode(),
                headers={"content-type": "application/json"},
            )
        ) as (client, recorder, metrics, calls, _):
            with pytest.raises(ValueError) as failure:
                await client.post(_path("openai"), json=_payload())
            assert recorder.segment_count == 0
            assert calls[-1][1] is failure.value
            assert metrics.snapshot()["model_requests_error"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["prompt_tokens", "response_tokens"])
def test_negative_token_ids_are_rejected_before_recording(field):
    async def scenario():
        result = _result()
        result["token_data"][field][0] = -1
        async with _harness(lambda request: httpx.Response(200, json=result)) as (
            client,
            recorder,
            *_,
        ):
            with pytest.raises(ValueError, match=f"{field} must contain nonnegative integer"):
                await client.post(_path("openai"), json=_payload())
            assert recorder.segment_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("bad_token", [True, 1.0], ids=["bool", "float"])
def test_type_coercible_prompt_prefix_is_rejected_without_changing_cache(bad_token):
    async def scenario():
        config = _config()
        config.rollout.execution = {"token_prefix_reuse": True}
        captured = []

        def upstream(request):
            payload = json.loads(request.content)
            captured.append(payload)
            result = _result()
            if "history_tokens" in payload:
                prefix = payload["history_tokens"]
                result["token_data"]["prompt_tokens"] = [*prefix, 5]
                if len(captured) == 2:
                    result["token_data"]["prompt_tokens"][0] = bad_token
                    # List equality alone accepts True == 1 and 1.0 == 1.
                    assert result["token_data"]["prompt_tokens"][: len(prefix)] == prefix
            return httpx.Response(200, json=result)

        async with _harness(upstream, config) as (client, recorder, *_):
            await client.post(_path("openai"), json=_payload())
            messages = recorder.to_messages()
            following = _payload(messages=[*messages, {"role": "user", "content": "continue"}])
            with pytest.raises(ValueError, match="prompt_tokens must contain nonnegative integer"):
                await client.post(_path("openai"), json=following)
            assert recorder.to_messages() == messages
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99]
            await client.post(_path("openai"), json=following)
            assert captured[2]["history_tokens"] == captured[1]["history_tokens"]
            assert recorder.to_remote_segments()[0].tokens == [1, 2, 3, 99, 5, 3, 99]

    asyncio.run(scenario())


def test_invalid_training_metadata_preserves_previously_recorded_turn():
    async def scenario():
        requests = 0

        def upstream(request):
            nonlocal requests
            requests += 1
            result = _result()
            if requests == 2:
                result["token_data"]["response_logprobs"] = [-0.1]
                result["choices"][0]["message"]["content"] = "invalid new turn"
            return httpx.Response(200, json=result)

        async with _harness(upstream) as (client, recorder, _, calls, _):
            assert (await client.post(_path("openai"), json=_payload())).status_code == 200
            segments, messages = recorder.to_remote_segments(), recorder.to_messages()
            with pytest.raises(ValueError):
                await client.post(
                    _path("openai"),
                    json=_payload(messages=[{"role": "user", "content": "new request"}]),
                )
            assert recorder.to_remote_segments() == segments
            assert recorder.to_messages() == messages
            assert calls[-1][1] is not None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "source,old",
    [("recomputed", None), ("inference", None), ("inference", [-0.3, -0.4])],
)
def test_behavior_logprobs_are_required_only_when_training_requests_them(source, old):
    async def scenario():
        config = _config()
        config.algorithm.sampling_logprob_source = source
        result = _result()
        result["token_data"]["response_old_logprobs"] = old
        async with _harness(lambda request: httpx.Response(200, json=result), config) as (
            client,
            recorder,
            _,
            _,
            _,
        ):
            if source == "inference" and old is None:
                with pytest.raises(ValueError, match="response_old_logprobs"):
                    await client.post(_path("openai"), json=_payload())
            else:
                response = await client.post(_path("openai"), json=_payload())
                assert response.status_code == 200
            if old is not None:
                assert recorder.to_remote_segments()[0].old_log_probs[-2:] == old

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("missing", ["terminal", "metadata"])
def test_incomplete_stream_never_becomes_a_training_segment(api, missing):
    async def scenario():
        events = _events(api)
        if missing == "terminal":
            events = events[:-1]
        else:
            events = (
                [
                    event
                    for event in events
                    if not isinstance(event, dict) or "token_data" not in event
                ]
                if api == "openai"
                else [
                    {
                        key: value
                        for key, value in event.items()
                        if key not in {"token_data", "weight_version"}
                    }
                    for event in events
                ]
            )
        async with _harness(
            lambda request: httpx.Response(
                200, stream=_Chunks(_sse(events)), headers={"content-type": "text/event-stream"}
            )
        ) as (client, recorder, metrics, calls, _):
            with pytest.raises(Exception):
                await client.post(_path(api), json=_payload(stream=True))
            assert recorder.segment_count == 0
            assert metrics.snapshot()["model_requests_error"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0
            assert calls[-1][1] is not None

    asyncio.run(scenario())


def test_only_a_later_matching_success_recovers_a_failed_request():
    async def scenario():
        requests = 0

        def upstream(request):
            nonlocal requests
            requests += 1
            return (
                httpx.Response(400, json={"error": {"message": "invalid request"}})
                if requests == 1
                else httpx.Response(200, json=_result())
            )

        async with _harness(upstream) as (client, _, _, calls, _):
            with pytest.raises(_UpstreamFailure, match="HTTP 400"):
                await client.post(_path("openai"), json=_payload())
            failed_operation = calls[0][0]
            await client.post(
                _path("openai"), json=_payload(messages=[{"role": "user", "content": "other"}])
            )
            assert (failed_operation, None) not in calls
            await client.post(_path("openai"), json=_payload())
            assert (failed_operation, None) in calls

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_context_errors_raise_and_signal_context_stop(api):
    async def scenario():
        error = {
            "error": {
                "type": "invalid_request_error",
                "message": "sequence length exceeds model max_seq_len: prompt 100 + max_new_tokens 32 = 132 > 128",
            }
        }
        async with _harness(lambda request: httpx.Response(400, json=error)) as (
            client,
            recorder,
            _,
            calls,
            _,
        ):
            with pytest.raises(ContextLengthExceededError) as failure:
                await client.post(_path(api), json=_payload())
            assert recorder.segment_count == 0
            observed = calls[-1][1]
            assert observed is failure.value
            assert isinstance(observed, ContextLengthExceededError)
            assert observed.length_details == {
                "prompt_tokens": 100,
                "requested_max_tokens": 32,
                "max_sequence_length": 128,
            }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "stream,budget,reported,expected",
    [
        (False, {"max_tokens": 51}, None, 51),
        (False, {"max_completion_tokens": 52}, None, 52),
        (False, {}, None, None),
        (True, {"sampling_params": {"max_new_tokens": 53}}, None, 53),
        (True, {"weaver_options": {"sampling_params": {"max_new_tokens": 54}}}, None, 54),
        (True, {}, None, None),
        (
            False,
            {
                "max_tokens": 51,
                "max_completion_tokens": 52,
                "sampling_params": {"max_new_tokens": 53},
                "weaver_options": {"sampling_params": {"max_tokens": 54, "max_new_tokens": 55}},
            },
            None,
            55,
        ),
        (False, {"max_tokens": 51, "weaver_options": {"sampling_params": {}}}, None, 51),
        (True, {"max_tokens": 51}, 70, 70),
    ],
)
def test_context_errors_without_recipe_budget_use_request_or_unknown(
    stream, budget, reported, expected
):
    async def scenario():
        config = _config()
        if stream:
            config.rollout.sampling.max_tokens = None
        else:
            del config.rollout.sampling["max_tokens"]
        error = {"error": {"code": "context_length_exceeded", "message": "context exceeded"}}
        if reported is not None:
            error["error"]["details"] = {"max_new_tokens": reported}

        def upstream(request):
            if stream:
                return httpx.Response(
                    200,
                    stream=_Chunks(_sse([error])),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(400, json=error)

        async with _harness(upstream, config) as (client, recorder, metrics, calls, _):
            with pytest.raises(Exception):
                await client.post(_path("openai"), json=_payload(stream=stream, **budget))
            assert recorder.segment_count == 0
            observed = calls[-1][1]
            assert isinstance(observed, ContextLengthExceededError)
            assert observed.length_details == (
                {} if expected is None else {"requested_max_tokens": expected}
            )
            assert metrics.snapshot()["model_requests_error"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0

    asyncio.run(scenario())


def test_validation_uses_validation_sampling_and_unrelated_models_are_rejected():
    async def scenario():
        requests = []
        async with _harness(
            lambda request: (
                requests.append(json.loads(request.content)) or httpx.Response(200, json=_result())
            ),
            is_validation=True,
        ) as (client, _, metrics, _, _):
            assert (
                await client.post(_path("openai"), json=_payload(model="wrong"))
            ).status_code == 400
            assert requests == []
            await client.post(_path("openai"), json=_payload())
            assert requests[0]["temperature"] == 0
            assert requests[0]["max_tokens"] == 10
            assert metrics.snapshot()["model_requests_started"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["transport_timeout", "cancelled"])
def test_request_timing_covers_timeouts_and_cancellation(failure, monkeypatch):
    monkeypatch.setattr("northrl.remote_rollout.northgate_relay._RETRY_MAX_ATTEMPTS", 1)

    async def scenario():
        entered = asyncio.Event()

        async def upstream(request):
            if failure == "transport_timeout":
                raise httpx.ReadTimeout("upstream timeout")
            entered.set()
            await asyncio.Event().wait()

        async with _harness(upstream) as (
            client,
            recorder,
            metrics,
            calls,
            _,
        ):
            task = asyncio.create_task(client.post(_path("openai"), json=_payload()))
            if failure == "cancelled":
                await entered.wait()
                assert metrics.snapshot()["model_request_inflight"] == 1
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                status = "cancelled"
            else:
                with pytest.raises(httpx.ReadTimeout):
                    await task
                status = "timeout"
            assert metrics.snapshot()[f"model_requests_{status}"] == 1
            assert metrics.snapshot()["model_request_inflight"] == 0
            assert metrics.snapshot()["model_requests_completed"] == 1
            assert recorder.segment_count == 0
            if failure == "cancelled":
                # The client gave up on a healthy call; the model service did not fail.
                assert calls == []
            else:
                assert calls[-1][1] is not None

    asyncio.run(scenario())


def test_cancelled_stream_drops_pending_metadata_and_clears_request_timing():
    async def scenario():
        entered = asyncio.Event()

        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _sse(_events("openai")[:-1])
                entered.set()
                await asyncio.Event().wait()

        async with _harness(
            lambda request: httpx.Response(
                200, stream=BlockingStream(), headers={"content-type": "text/event-stream"}
            )
        ) as (client, recorder, metrics, calls, _):
            task = asyncio.create_task(client.post(_path("openai"), json=_payload(stream=True)))
            await entered.wait()
            assert recorder.segment_count == 0
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert recorder.segment_count == 0
            assert metrics.snapshot()["model_request_inflight"] == 0
            assert metrics.snapshot()["model_requests_cancelled"] == 1
            assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_client_cancellation_after_a_silent_upstream_is_a_service_failure(stream, monkeypatch):
    async def scenario():
        entered = asyncio.Event()

        class SilentStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b""

        async def upstream(request):
            if stream:
                return httpx.Response(
                    200, stream=SilentStream(), headers={"content-type": "text/event-stream"}
                )
            entered.set()
            await asyncio.Event().wait()

        # Every wait counts as a stall, as when a client gives up on a hung upstream.
        monkeypatch.setattr("northrl.remote_rollout.northgate_relay._UPSTREAM_STALL_SECONDS", 0.0)
        async with _harness(upstream) as (client, recorder, metrics, calls, _):
            task = asyncio.create_task(
                client.post(_path("anthropic"), json=_payload(stream=stream))
            )
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert recorder.segment_count == 0
            assert metrics.snapshot()["model_requests_cancelled"] == 1
            assert len(calls) == 1
            assert "without upstream data" in str(calls[0][1])
            assert isinstance(calls[0][1].__cause__, asyncio.CancelledError)

    asyncio.run(scenario())


def test_a_success_already_in_flight_cannot_recover_a_later_failure():
    async def scenario():
        first_started, second_started, release_success = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        count = 0

        async def upstream(request):
            nonlocal count
            count += 1
            if count == 1:
                first_started.set()
                await second_started.wait()
                return httpx.Response(400, json={"error": {"message": "invalid request"}})
            second_started.set()
            await release_success.wait()
            return httpx.Response(200, json=_result())

        async with _harness(upstream) as (client, _, _, calls, _):
            first = asyncio.create_task(client.post(_path("openai"), json=_payload()))
            await first_started.wait()
            second = asyncio.create_task(client.post(_path("openai"), json=_payload()))
            with pytest.raises(_UpstreamFailure, match="HTTP 400"):
                await first
            failed_operation = calls[0][0]
            release_success.set()
            assert (await second).status_code == 200
            assert (failed_operation, None) not in calls

    asyncio.run(scenario())


@pytest.mark.parametrize("reference", [False, True])
def test_requested_masks_are_captured_without_sdk_imports(reference):
    async def scenario():
        config = _config()
        config.rollout.keep_sampling_mask = True
        result = _result()
        if reference:
            result["token_data"]["response_sampling_mask_ref"] = {
                "uri": "opaque://masks",
                "token_count": 2,
            }
        else:
            result["token_data"]["response_sampling_masks"] = [[3, 9], [99]]
        async with _harness(lambda request: httpx.Response(200, json=result), config) as (
            client,
            recorder,
            _,
            _,
            _,
        ):
            assert (await client.post(_path("openai"), json=_payload())).status_code == 200
            mask = recorder.to_remote_segments()[0].sampling_mask
            if reference:
                assert mask == {"sampling_mask_ref": {"uri": "opaque://masks", "token_count": 2}}
            else:
                assert mask == [[3, 9], [99]]

    asyncio.run(scenario())


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_requested_missing_masks_are_a_service_failure(api):
    async def scenario():
        config = _config()
        config.rollout.keep_sampling_mask = True
        async with _harness(lambda request: httpx.Response(200, json=_result(api)), config) as (
            client,
            recorder,
            metrics,
            calls,
            _,
        ):
            with pytest.raises(ValueError, match="sampling masks") as failure:
                await client.post(_path(api), json=_payload())
            assert recorder.segment_count == 0
            assert recorder.to_messages() == []
            assert metrics.snapshot()["model_requests_error"] == 1
            assert calls[-1][1] is failure.value

    asyncio.run(scenario())


def test_sampling_defaults_include_native_parameters_without_clipping():
    async def scenario():
        config = _config()
        config.rollout.sampling.min_p = 0.01
        config.rollout.sampling.repetition_penalty = 1.1
        config.rollout.sampling.max_tokens = 128
        captured = []
        async with _harness(
            lambda request: (
                captured.append(json.loads(request.content)) or httpx.Response(200, json=_result())
            ),
            config,
        ) as (client, _, _, _, _):
            await client.post(
                _path("openai"),
                json=_payload(
                    weaver_options={"sampling_params": {"temperature": 999, "seed": 123}}
                ),
            )
            params = captured[0]["sampling_params"]
            assert params["min_p"] == 0.01
            assert params["repetition_penalty"] == 1.1
            assert params["max_tokens"] == 128
            assert params["temperature"] == 0.8
            assert params["seed"] == 123
            assert "sampling_params" not in captured[0]["weaver_options"]

    asyncio.run(scenario())


@pytest.mark.parametrize("missing", [False, True], ids=["null", "missing"])
@pytest.mark.parametrize("nested", [False, True])
def test_unconfigured_token_budget_preserves_agent_aliases_and_does_not_invent_one(missing, nested):
    async def scenario():
        config = _config()
        if missing:
            del config.rollout.sampling["max_tokens"]
        else:
            config.rollout.sampling.max_tokens = None
        captured = []
        async with _harness(
            lambda request: (
                captured.append(json.loads(request.content)) or httpx.Response(200, json=_result())
            ),
            config,
        ) as (client, _, _, _, _):
            budget = {
                "max_tokens": 81,
                "max_completion_tokens": 82,
                "sampling_params": {"max_tokens": 83, "max_new_tokens": 84},
            }
            native = budget["sampling_params"]
            if nested:
                native = {"max_tokens": 85, "max_new_tokens": 86}
                budget["weaver_options"] = {"sampling_params": native}
            assert (await client.post(_path("openai"), json=_payload(**budget))).status_code == 200
            actual = captured[0]
            assert actual["max_tokens"] == 81
            assert actual["max_completion_tokens"] == 82
            for key, value in native.items():
                assert actual["sampling_params"][key] == value
            assert "sampling_params" not in actual["weaver_options"]
            assert (await client.post(_path("openai"), json=_payload())).status_code == 200
            for params in (
                captured[1],
                captured[1]["sampling_params"],
            ):
                assert not {"max_tokens", "max_completion_tokens", "max_new_tokens"} & params.keys()

    asyncio.run(scenario())


@pytest.mark.parametrize("nested", [False, True])
def test_native_max_new_tokens_alias_cannot_override_recipe(nested):
    async def scenario():
        captured = []
        async with _harness(
            lambda request: (
                captured.append(json.loads(request.content)) or httpx.Response(200, json=_result())
            )
        ) as (client, _, _, _, _):
            params = {"max_tokens": 999, "max_new_tokens": 999}
            override = (
                {"weaver_options": {"sampling_params": params}}
                if nested
                else {"sampling_params": params}
            )
            assert (
                await client.post(_path("openai"), json=_payload(**override))
            ).status_code == 200
            outgoing = captured[0]["sampling_params"]
            assert outgoing["max_tokens"] == 32
            assert "max_new_tokens" not in outgoing
            assert "sampling_params" not in captured[0]["weaver_options"]

    asyncio.run(scenario())


def test_count_tokens_is_not_emulated_and_unconfigured_features_fail_explicitly():
    async def scenario():
        async with _harness(lambda request: pytest.fail("no request expected")) as (
            client,
            _,
            metrics,
            _,
            _,
        ):
            assert (
                await client.post("/v1/messages/count_tokens", json=_payload())
            ).status_code == 404
            assert metrics.snapshot()["model_requests_started"] == 0
        config = _config()
        config.algorithm.router_replay = {"enabled": True, "mode": "R3"}
        with pytest.raises(ValueError, match="R3"):
            async with _harness(lambda request: None, config):
                pass
        config.algorithm.router_replay.mode = "R2"
        async with _harness(lambda request: httpx.Response(200, json=_result()), config) as (
            client,
            _,
            _,
            _,
            _,
        ):
            assert (await client.post(_path("openai"), json=_payload())).status_code == 200

    asyncio.run(scenario())
