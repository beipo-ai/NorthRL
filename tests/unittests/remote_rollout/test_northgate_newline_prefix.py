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

"""Claude Code history adaptation through a mock gateway, including SSE chunks."""

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
    _tools,
)
from tests.unittests.remote_rollout.test_northgate_token_prefix import _enabled_config


@pytest.fixture(autouse=True)
def _gateway_key(monkeypatch):
    monkeypatch.setenv("NORTHGATE_API_KEY", "test-ng-key")


def _assistant(turn, separator):
    blocks = deepcopy(_result("anthropic")["content"])
    blocks[0]["thinking"] = f"Inspect turn {turn}.\n\n"
    blocks[1]["id"] = f"tool-{turn + 1}"
    blocks[1]["input"]["command"] = f"echo {turn}\n\npwd"
    blocks.insert(1, {"type": "text", "text": separator})
    return {"role": "assistant", "content": blocks}


def _response(assistant, token_data, *, stream):
    if not stream:
        result = _result("anthropic")
        result["content"] = deepcopy(assistant["content"])
        result["token_data"] = deepcopy(token_data)
        response = httpx.Response(200, json=result)
        return response, response.content

    thinking, text, call = assistant["content"]
    events = _events("anthropic")
    events[2]["delta"]["thinking"] = thinking["thinking"]
    for event in events[5:8]:
        event["index"] = 2
    events[5]["content_block"]["id"] = call["id"]
    events[6]["delta"]["partial_json"] = json.dumps(call["input"])
    events[-2]["token_data"] = deepcopy(token_data)
    events[5:5] = [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        *[
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": character},
            }
            for character in text["text"]
        ],
        {"type": "content_block_stop", "index": 1},
    ]
    body = _sse(events)
    return (
        httpx.Response(200, stream=_Chunks(body), headers={"content-type": "text/event-stream"}),
        body,
    )


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("drop_separator", [True, False], ids=["claude-history", "native-history"])
@pytest.mark.parametrize(
    "streams",
    [(False, False, False), (True, True, True), (True, False, True), (False, True, False)],
    ids=["json", "sse", "sse-json-sse", "json-sse-json"],
)
def test_newline_history_reuses_original_tokens_across_three_turns(
    separator, drop_separator, streams
):
    async def scenario():
        config = _enabled_config()
        config.algorithm.sampling_logprob_source = "inference"
        config.rollout.keep_sampling_mask = True
        captured = []
        upstream_bodies = []
        assistants = [_assistant(turn, separator) for turn in range(3)]
        training = [
            {
                "prompt_tokens": [10, 11],
                "response_tokens": [20, 21],
                "response_logprobs": [-0.1, -0.2],
                "response_old_logprobs": [-0.3, -0.4],
                "response_sampling_masks": [[20, 22], [21]],
            },
            {
                "prompt_tokens": [10, 11, 20, 21, 30, 31],
                "response_tokens": [40],
                "response_logprobs": [-0.5],
                "response_old_logprobs": [-0.6],
                "response_sampling_masks": [[40, 41]],
            },
            {
                "prompt_tokens": [10, 11, 20, 21, 30, 31, 40, 50],
                "response_tokens": [60],
                "response_logprobs": [-0.7],
                "response_old_logprobs": [-0.8],
                "response_sampling_masks": [[60, 61]],
            },
        ]

        def upstream(request):
            turn = len(captured)
            payload = json.loads(request.content)
            captured.append(payload)
            if turn == 0:
                assert "history_tokens" not in payload
            else:
                previous = training[turn - 1]
                assert payload["history_tokens"] == (
                    previous["prompt_tokens"] + previous["response_tokens"]
                )
            # The mock checks the splice contract, not NorthGate's real renderer.
            response, body = _response(assistants[turn], training[turn], stream=streams[turn])
            upstream_bodies.append(body)
            return response

        payload = _payload(tools=_tools("anthropic"))
        async with _harness(upstream, config) as (client, recorder, *_):
            for turn, stream in enumerate(streams):
                payload["stream"] = stream
                before = deepcopy(payload)
                response = await client.post(_path("anthropic"), json=payload)
                assert response.status_code == 200
                # Neither JSON content nor the SSE block/deltas are rewritten.
                assert response.content == upstream_bodies[turn]
                assert captured[turn]["messages"] == before["messages"]
                assert payload == before
                assert recorder.to_messages() == [*before["messages"], assistants[turn]]

                historical_assistant = deepcopy(assistants[turn])
                if drop_separator:
                    # Simulate Claude Code dropping the complete text block.
                    historical_assistant["content"].pop(1)
                payload["messages"] += [
                    historical_assistant,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"tool-{turn + 1}",
                                "content": "done\n\n",
                            }
                        ],
                    },
                ]

            [segment] = recorder.to_remote_segments()
            assert segment.tokens == [10, 11, 20, 21, 30, 31, 40, 50, 60]
            assert segment.loss_mask == [0, 0, 1, 1, 0, 0, 1, 0, 1]
            assert segment.log_probs == [0, 0, -0.1, -0.2, 0, 0, -0.5, 0, -0.7]
            assert segment.old_log_probs == [0, 0, -0.3, -0.4, 0, 0, -0.6, 0, -0.8]
            assert segment.sampling_mask == [[20, 22], [21], [40, 41], [60, 61]]

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_newline_equivalent_histories_with_different_tokens_are_not_reused(stream):
    async def scenario():
        captured = []
        assistants = [_assistant(0, "\n\n"), _assistant(0, "\r\n")]

        def upstream(request):
            turn = len(captured)
            payload = json.loads(request.content)
            captured.append(payload)
            # Two completed branches become the same cache key, but their
            # generated tokens differ. Dropped text cannot identify the branch.
            assert "history_tokens" not in payload
            token_data = {
                "prompt_tokens": [10, 11],
                "response_tokens": [20, 21 + turn],
                "response_logprobs": [-0.1, -0.2],
            }
            response, _ = _response(assistants[min(turn, 1)], token_data, stream=stream)
            return response

        initial = _payload(stream=stream, tools=_tools("anthropic"))
        async with _harness(upstream, _enabled_config()) as (client, *_):
            for _ in range(2):
                assert (await client.post(_path("anthropic"), json=initial)).status_code == 200
            history = deepcopy(assistants[0])
            history["content"].pop(1)
            following = deepcopy(initial)
            following["messages"] += [
                history,
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool-1", "content": "done"}
                    ],
                },
            ]
            assert (await client.post(_path("anthropic"), json=following)).status_code == 200
            assert captured[-1]["messages"] == following["messages"]

    asyncio.run(scenario())
