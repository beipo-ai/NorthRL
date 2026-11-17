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

from __future__ import annotations

import asyncio
import json
import types as pytypes
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf
from transformers import BatchEncoding
from weaver import WeaverAPIError

from northrl.exceptions import WeightVersionMismatchError
from northrl.rollout_worker.single_turn_rollout_worker import SingleTurnRolloutWorker
from northrl.utils.async_utils import run_coro
from northrl.weaver import WeaverInferenceServiceClient
from northrl.weaver.weaver_service_holder import WeaverServiceHolder
from northrl.weight_sync.versioning import DEFAULT_WEIGHT_VERSION


def _config(
    parser_config: dict[str, Any] | None = None,
    config_file_path: str | None = None,
):
    config = OmegaConf.create(
        {
            "training": {
                "weight_sync": {
                    "mode": "sync",
                    "staleness_threshold": 0,
                    "freeze_inflight": False,
                    "sampler_ttl_s": 86400,
                },
            },
            "backend": {"type": "weaver"},
            "rollout": {
                "parsers": parser_config if parser_config is not None else {"type": "builtin"},
                "sampling": {
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "top_k": 32,
                    "max_tokens": 128,
                },
            },
            "data": {"max_sequence_length": 2048, "max_prompt_length": 1024},
            "models": {"actor": {"name": "Qwen/Qwen3-8B"}},
        }
    )
    if config_file_path is not None:
        config._config_file_path = config_file_path  # pylint: disable=protected-access
    return config


def _write_response_parser(tmp_path: Path, source: str) -> Path:
    parser_path = tmp_path / "custom_response_parser.py"
    parser_path.write_text(source, encoding="utf-8")
    return parser_path


def _sample_result(response: str = "ok") -> dict[str, Any]:
    return {
        "response": response,
        "prompt_tokens": [1, 2],
        "response_tokens": [3],
        "response_logprobs": [-0.1],
        "response_old_logprobs": [-0.1],
        "finish_reason": "stop",
    }


class FakeServiceHolder:
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def sample_from_prompt(self, **kwargs):
        self.calls.append(("prompt", kwargs))
        return _sample_result(self.response)

    def sample_from_messages(self, **kwargs):
        self.calls.append(("messages", kwargs))
        return _sample_result(self.response)

    def sample_from_token_ids(self, **kwargs):
        self.calls.append(("token_ids", kwargs))
        return _sample_result(self.response)


def test_weaver_inference_client_forwards_configured_sampling_params():
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(_config(), holder)  # type: ignore[arg-type]

    run_coro(client.completion("hello"))
    result = run_coro(client.generate([{"role": "user", "content": "hello"}]))

    message = result["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "ok"}

    assert holder.calls[0][0] == "prompt"
    assert holder.calls[0][1]["temperature"] == 0.4
    assert holder.calls[0][1]["top_p"] == 0.8
    assert holder.calls[0][1]["top_k"] == 32
    assert holder.calls[0][1]["num_samples"] == 1

    assert holder.calls[1][0] == "messages"
    assert holder.calls[1][1]["temperature"] == 0.4
    assert holder.calls[1][1]["top_p"] == 0.8
    assert holder.calls[1][1]["top_k"] == 32
    assert holder.calls[1][1]["num_samples"] == 1


@pytest.mark.parametrize("operation", ["completion", "generate", "generate_with_token"])
@pytest.mark.parametrize("max_tokens,expected", [(64000, 64000), (None, 1024)])
def test_validation_sampling_and_training_restore(operation, max_tokens, expected):
    config = _config()
    config.validate = {
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.0,
            "max_tokens": max_tokens,
        }
    }
    config.algorithm = {"mis": {"enabled": True}}
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(config, holder)

    def generate():
        if operation == "completion":
            return client.completion("hello")
        if operation == "generate":
            return client.generate([{"role": "user", "content": "hello"}])
        return client.generate_with_token(
            [1, 2],
            sampling_params={
                "max_new_tokens": 128,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
        )

    client.set_validation_mode(True)
    run_coro(generate())
    params = holder.calls[-1][1]
    assert (params["max_tokens"], params["temperature"], params["top_p"], params["top_k"]) == (
        expected,
        0.7,
        0.95,
        40,
    )
    client.set_validation_mode(False)
    config.algorithm.mis.enabled = False
    run_coro(generate())
    assert holder.calls[-1][1]["max_tokens"] == 128
    assert holder.calls[-1][1]["temperature"] == (
        1.0 if operation == "generate_with_token" else 0.4
    )


def test_weaver_generate_with_token_allows_sampling_param_overrides():
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(_config(), holder)  # type: ignore[arg-type]

    run_coro(
        client.generate_with_token(
            [1, 2],
            sampling_params={
                "max_new_tokens": 16,
                "temperature": 0.2,
                "top_p": 0.6,
                "top_k": 8,
            },
        )
    )

    call = holder.calls[0][1]
    assert call["max_tokens"] == 16
    assert call["temperature"] == 0.2
    assert call["top_p"] == 0.6
    assert call["top_k"] == 8
    assert call["num_samples"] == 1


@pytest.mark.parametrize(
    "overrides",
    [{"top_p": 0.9}, {"top_p": 1.1}, {"top_k": 8}, {"min_p": 0.1}, {"min_p": -0.1}],
)
@pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
def test_mis_token_request_rejects_sampling_filter_overrides_before_sampling(overrides, feature):
    config = _config()
    config.algorithm = {"mis": {feature: True}}
    config.rollout.sampling.update({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0})
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(config, holder)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="recompute and training sampling_mask"):
        run_coro(client.generate_with_token([1, 2], sampling_params=overrides))

    assert holder.calls == []


@pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
def test_mis_token_request_accepts_unfiltered_sampling_overrides(feature):
    config = _config()
    config.algorithm = {"mis": {feature: True}}
    config.rollout.sampling.update({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0})
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(config, holder)  # type: ignore[arg-type]

    run_coro(client.generate_with_token([1, 2], sampling_params={"temperature": 1.0, "top_k": 0}))

    assert holder.calls[0][1]["temperature"] == 1.0
    assert holder.calls[0][1]["top_p"] == 1.0
    assert holder.calls[0][1]["top_k"] == 0


@pytest.mark.parametrize("overrides", [{"temperature": 0.7}, {"repetition_penalty": 1.2}])
@pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
def test_mis_token_request_rejects_unverified_probability_transforms(overrides, feature):
    config = _config()
    config.algorithm = {"mis": {feature: True}}
    config.rollout.sampling.update({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0})
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(config, holder)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sampler old_logprobs semantics"):
        run_coro(client.generate_with_token([1, 2], sampling_params=overrides))

    assert holder.calls == []


def test_disabled_mis_token_request_keeps_sampling_filters():
    config = _config()
    config.algorithm = {"mis": {"enabled": False, "get_mismatch_metrics": False}}
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(config, holder)  # type: ignore[arg-type]

    run_coro(client.generate_with_token([1, 2], sampling_params={"top_p": 0.6, "top_k": 8}))

    assert holder.calls[0][1]["top_p"] == 0.6
    assert holder.calls[0][1]["top_k"] == 8


def test_remote_completion_keeps_raw_response():
    raw_response = (
        "<think>private reasoning</think>visible text\n"
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    )
    holder = FakeServiceHolder(raw_response)
    client = WeaverInferenceServiceClient(_config(), holder)  # type: ignore[arg-type]

    result = run_coro(client.completion("hello"))

    assert result["choices"][0]["text"] == raw_response
    assert holder.calls[0][0] == "prompt"


@pytest.mark.parametrize("entrypoint", ["generate", "generate_with_token"])
def test_remote_client_parses_raw_holder_response(entrypoint):
    raw_response = (
        "<think>private reasoning</think>visible text\n"
        '<tool_call>{"name":"lookup","arguments":{"q":"northrl"}}</tool_call>\n'
        '<tool_call>{"name":"count","arguments":{}}</tool_call>'
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("lookup", "count")
    ]
    holder = FakeServiceHolder(raw_response)
    client = WeaverInferenceServiceClient(_config(), holder)  # type: ignore[arg-type]

    if entrypoint == "generate":
        result = run_coro(
            client.generate(
                [{"role": "user", "content": "hello"}],
                tools=tools,
            )
        )
    else:
        result = run_coro(client.generate_with_token([1, 2], tools=tools))

    choice = result["choices"][0]
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "visible text"
    assert message["reasoning_content"] == "private reasoning"
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["lookup", "count"]
    assert all(call["id"] for call in message["tool_calls"])
    assert all(call["type"] == "function" for call in message["tool_calls"])
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"q": "northrl"}
    assert json.loads(message["tool_calls"][1]["function"]["arguments"]) == {}
    assert choice["finish_reason"] == "stop"
    assert result["northrl_train"]["response_tokens"] == [3]
    assert result["northrl_train"]["response_logprobs"] == [-0.1]


@pytest.mark.parametrize("entrypoint", ["generate", "generate_with_token"])
def test_remote_client_uses_recipe_response_parser(entrypoint, tmp_path: Path):
    _write_response_parser(
        tmp_path,
        """
import json


def parse_response(*, response, tools):
    tool_name = tools[0]["function"]["name"]
    tools.clear()
    return {
        "content": f"custom|{response}|{tool_name}",
        "reasoning_content": "custom reasoning",
        "tool_calls": [
            {
                "id": "call-custom",
                "type": "function",
                "function": {
                    "name": "custom_lookup",
                    "arguments": json.dumps({"raw": response}),
                },
            }
        ],
    }
""",
    )
    raw_response = (
        "<think>private reasoning</think>visible text\n"
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    config = _config(
        parser_config={
            "type": "custom",
            "module_path": "custom_response_parser.py",
        },
        config_file_path=str(tmp_path / "weaver.yaml"),
    )
    holder = FakeServiceHolder(raw_response)
    client = WeaverInferenceServiceClient(config, holder)  # type: ignore[arg-type]

    def invoke():
        if entrypoint == "generate":
            return run_coro(
                client.generate(
                    [{"role": "user", "content": "hello"}],
                    tools=tools,
                )
            )
        return run_coro(client.generate_with_token([1, 2], tools=tools))

    for result in (invoke(), invoke()):
        message = result["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert message["content"] == f"custom|{raw_response}|lookup"
        assert message["reasoning_content"] == "custom reasoning"
        assert len(message["tool_calls"]) == 1
        tool_call = message["tool_calls"][0]
        assert tool_call["id"] == "call-custom"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "custom_lookup"
        assert json.loads(tool_call["function"]["arguments"]) == {"raw": raw_response}

    assert tools[0]["function"]["name"] == "lookup"


def test_warmup_sampling_uses_ten_token_probe() -> None:
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder.sample_from_token_ids = AsyncMock(return_value={})

    run_coro(holder.warmup_sampling())

    holder.sample_from_token_ids.assert_awaited_once_with(
        input_ids=[0],
        max_tokens=10,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
    )


class FakeTokenizer:
    def encode(self, prompt, add_special_tokens=False):  # pylint: disable=unused-argument
        return [1, 2]

    def decode(self, tokens, skip_special_tokens=True):  # pylint: disable=unused-argument
        return "ok"

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=True,
        tokenize=False,
        tools=None,
    ):  # pylint: disable=unused-argument
        return [1, 2] if tokenize else "prompt"


@pytest.mark.parametrize(
    ("tokenize", "template_output", "expected"),
    [
        pytest.param(True, [1, 2, 3], [1, 2, 3], id="list"),
        pytest.param(
            True,
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            [1, 2, 3],
            id="dict",
        ),
        pytest.param(
            True,
            BatchEncoding({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}),
            [1, 2, 3],
            id="batch-encoding",
        ),
        pytest.param(False, "rendered prompt", "rendered prompt", id="text"),
    ],
)
def test_apply_chat_template_preserves_service_return_types(tokenize, template_output, expected):
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = Mock()  # pylint: disable=protected-access
    holder._tokenizer.apply_chat_template.return_value = template_output
    holder._ensure_provisioned = AsyncMock()  # pylint: disable=protected-access

    result = run_coro(
        holder.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=tokenize)
    )

    assert result == expected
    assert isinstance(result, list if tokenize else str)


@pytest.mark.parametrize("token_count", [3, 4], ids=["at-limit", "over-limit"])
def test_single_turn_with_batch_encoding_uses_token_count(token_count):
    config = _config()
    config.data.max_prompt_length = 3
    config.rollout.reward_fn = "math_answer_only"
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = Mock()  # pylint: disable=protected-access
    holder._tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
        BatchEncoding({"input_ids": list(range(token_count)), "attention_mask": [1] * token_count})
        if kwargs["tokenize"]
        else "rendered prompt"
    )
    holder._ensure_provisioned = AsyncMock()  # pylint: disable=protected-access
    holder.sample_from_messages = AsyncMock(return_value=_sample_result(r"\boxed{42}"))
    client = WeaverInferenceServiceClient(config, holder)
    task = {
        "prompt": [{"role": "user", "content": "What is 40 + 2?"}],
        "reward_model": {"ground_truth": "42"},
    }

    worker = SingleTurnRolloutWorker(config)
    worker._inference_client = client
    worker._save_rollout_trace = Mock()
    worker._trajectory_pool = Mock()
    worker._trajectory_pool.put_trajectory.return_value = "success"

    result = run_coro(worker.rollout(task))

    if token_count > config.data.max_prompt_length:
        assert result is None
        holder.sample_from_messages.assert_not_awaited()
    else:
        assert result == "success"
        trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
        trace = worker._save_rollout_trace.call_args.args[2]
        assert trace["response"] == r"\boxed{42}"
        assert trajectory.reward == 1.0
        assert trace["prompt_str"] == "rendered prompt"
        holder.sample_from_messages.assert_awaited_once()


class FakeSamplingClient:
    def __init__(self, response: str = "ok", stop_reason: str = "stop") -> None:
        self.response = response
        self.stop_reason = stop_reason
        self.payload: dict[str, Any] | None = None
        self.num_samples: int | None = None
        self.wait: bool | None = None

    async def sample(self, **kwargs):
        self.payload = kwargs["sampling_params"].to_payload()
        self.num_samples = kwargs["num_samples"]
        self.wait = kwargs["wait"]
        sequence = {
            "tokens": [3],
            "logprobs": [-0.1],
            "text": self.response,
            "stop_reason": self.stop_reason,
            "weight_version": "default",
        }
        return {"sequences": [sequence]}


class R3SamplingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def sample(
        self,
        *,
        prompt,
        num_samples,
        sampling_params,
        include_prompt_logprobs,
        return_sampling_mask,
        return_old_logprob,
        return_moe_topk_indices,
        wait=False,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "num_samples": num_samples,
                "sampling_params": sampling_params,
                "include_prompt_logprobs": include_prompt_logprobs,
                "return_sampling_mask": return_sampling_mask,
                "return_old_logprob": return_old_logprob,
                "return_moe_topk_indices": return_moe_topk_indices,
                "wait": wait,
            }
        )
        return {
            "sequences": [
                {
                    "tokens": [3],
                    "logprobs": [-0.1],
                    "text": "ok",
                    "stop_reason": "stop",
                    "weight_version": "default",
                    "moe_topk_indices_ref": {
                        "storage": "gpfs",
                        "relative_path": "router-replay/sample-0.safetensors",
                    },
                }
            ]
        }


@pytest.mark.parametrize(
    ("prompt_tokens", "error_type", "match"),
    [
        ((1, 2), TypeError, r"actual_type=tuple, actual_value=\(1, 2\)"),
        ([], ValueError, "must return a non-empty list"),
        ([1, "2", True], TypeError, "invalid_count=2"),
    ],
)
def test_sample_from_messages_validates_tokenized_prompt(prompt_tokens, error_type, match) -> None:
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = object()  # pylint: disable=protected-access
    holder._ensure_provisioned = AsyncMock()  # pylint: disable=protected-access
    holder.apply_chat_template = AsyncMock(return_value=prompt_tokens)

    with pytest.raises(error_type, match=match):
        run_coro(
            holder.sample_from_messages(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
            )
        )


def test_initial_sampler_uses_default_weight_version() -> None:
    holder = WeaverServiceHolder(
        base_model="test-model",
        need_sampling=False,
        sampler_ttl_seconds=123,
    )
    sampling_client = object()
    training_client = pytypes.SimpleNamespace(
        model_id="test-model-id",
        save_weights_for_sampler=AsyncMock(return_value="weaver://weights/default"),
    )
    service_client = pytypes.SimpleNamespace(
        create_sampling_client=AsyncMock(return_value=sampling_client),
        aclose=AsyncMock(),
    )
    holder._training_client = training_client  # pylint: disable=protected-access
    holder._service_client = service_client  # pylint: disable=protected-access
    holder._tokenizer = FakeTokenizer()  # pylint: disable=protected-access

    run_coro(holder._ensure_provisioned(require_sampling=True))  # pylint: disable=protected-access

    training_client.save_weights_for_sampler.assert_awaited_once_with(
        name=DEFAULT_WEIGHT_VERSION,
        ttl_seconds=123,
    )
    service_client.create_sampling_client.assert_awaited_once_with(
        model_path="weaver://weights/default",
        base_model="test-model",
        model_id="test-model-id",
    )
    assert holder._sampling_client is sampling_client  # pylint: disable=protected-access
    assert run_coro(holder.get_current_sampling_version()) == DEFAULT_WEIGHT_VERSION
    run_coro(holder.close())


@pytest.mark.parametrize("entrypoint", ["messages", "prompt", "token_ids"])
def test_weaver_holder_returns_raw_response(entrypoint):
    raw_response = (
        "<think>private reasoning</think>visible text\n"
        '<tool_call>{"name":"lookup","arguments":{"q":"northrl"}}</tool_call>'
    )
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = FakeTokenizer()  # pylint: disable=protected-access
    holder._sampling_client = FakeSamplingClient(  # pylint: disable=protected-access
        raw_response,
        stop_reason="backend-stop",
    )
    holder._active_sampling_snapshot = (  # pylint: disable=protected-access
        "weaver://weights/default",
        "default",
    )
    holder._return_sampling_mask = False  # pylint: disable=protected-access
    holder._return_old_logprob = False  # pylint: disable=protected-access
    holder._max_concurrent_requests = None  # pylint: disable=protected-access
    holder._sampling_semaphore = None  # pylint: disable=protected-access
    holder._sampling_semaphore_loop = None  # pylint: disable=protected-access

    async def _noop_ensure(self, require_sampling=False):
        del self, require_sampling

    holder._ensure_provisioned = pytypes.MethodType(  # pylint: disable=protected-access
        _noop_ensure, holder
    )

    if entrypoint == "messages":
        result = run_coro(
            holder.sample_from_messages(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
            )
        )
    elif entrypoint == "prompt":
        result = run_coro(holder.sample_from_prompt(prompt="hello", max_tokens=16))
    else:
        result = run_coro(holder.sample_from_token_ids(input_ids=[1, 2], max_tokens=16))

    assert result["response"] == raw_response
    assert result["response_tokens"] == [3]
    assert result["response_logprobs"] == [-0.1]
    assert result["finish_reason"] == "backend-stop"
    assert result["weight_version"] == "default"
    assert {
        "tool_string",
        "reasoning_string",
        "tool_calls",
        "is_valid",
    }.isdisjoint(result)


@pytest.mark.parametrize("entrypoint", ["messages", "prompt", "token_ids"])
def test_weaver_holder_requests_and_propagates_r3_router_ref(entrypoint):
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = FakeTokenizer()  # pylint: disable=protected-access
    holder._sampling_client = R3SamplingClient()  # pylint: disable=protected-access
    holder._active_sampling_snapshot = (  # pylint: disable=protected-access
        "weaver://weights/default",
        "default",
    )
    holder._return_sampling_mask = False  # pylint: disable=protected-access
    holder._return_old_logprob = False  # pylint: disable=protected-access
    holder._return_moe_topk_indices = True  # pylint: disable=protected-access
    holder._max_concurrent_requests = None  # pylint: disable=protected-access
    holder._sampling_semaphore = None  # pylint: disable=protected-access
    holder._sampling_semaphore_loop = None  # pylint: disable=protected-access

    async def _noop_ensure(self, require_sampling=False):
        del self, require_sampling

    holder._ensure_provisioned = pytypes.MethodType(  # pylint: disable=protected-access
        _noop_ensure, holder
    )

    if entrypoint == "messages":
        result = run_coro(
            holder.sample_from_messages(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
            )
        )
    elif entrypoint == "prompt":
        result = run_coro(holder.sample_from_prompt(prompt="hello", max_tokens=16))
    else:
        result = run_coro(holder.sample_from_token_ids(input_ids=[1, 2], max_tokens=16))

    sampling_client = holder._sampling_client  # pylint: disable=protected-access
    assert len(sampling_client.calls) == 1
    assert sampling_client.calls[0]["return_moe_topk_indices"] is True
    assert sampling_client.calls[0]["wait"] is False
    assert result["response_moe_topk_indices_ref"] == {
        "storage": "gpfs",
        "relative_path": "router-replay/sample-0.safetensors",
    }


def test_weaver_service_holder_writes_sampling_params_to_sdk_payload():
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = FakeTokenizer()  # pylint: disable=protected-access
    holder._sampling_client = FakeSamplingClient()  # pylint: disable=protected-access
    holder._active_sampling_snapshot = (  # pylint: disable=protected-access
        "weaver://weights/default",
        "default",
    )
    holder._return_sampling_mask = False  # pylint: disable=protected-access
    holder._return_old_logprob = False  # pylint: disable=protected-access
    holder._max_concurrent_requests = None  # pylint: disable=protected-access
    holder._sampling_semaphore = None  # pylint: disable=protected-access
    holder._sampling_semaphore_loop = None  # pylint: disable=protected-access

    async def _noop_ensure(self, require_sampling=False):
        del require_sampling

    holder._ensure_provisioned = pytypes.MethodType(  # pylint: disable=protected-access
        _noop_ensure, holder
    )

    result = run_coro(
        holder.sample_from_prompt(
            prompt="hello",
            max_tokens=16,
            temperature=0.2,
            top_p=0.6,
            top_k=8,
        )
    )

    payload = holder._sampling_client.payload  # pylint: disable=protected-access
    assert payload == {
        "temperature": 0.2,
        "top_p": 0.6,
        "top_k": 8,
        "max_tokens": 16,
    }
    assert holder._sampling_client.num_samples == 1  # pylint: disable=protected-access
    assert holder._sampling_client.wait is False  # pylint: disable=protected-access
    assert result["weight_version"] == "default"


class SlowSamplingClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def sample(self, **kwargs):
        del kwargs
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return {
                "sequences": [
                    {
                        "tokens": [3],
                        "logprobs": [-0.1],
                        "text": "ok",
                        "stop_reason": "stop",
                        "weight_version": "default",
                    }
                ]
            }
        finally:
            self.active -= 1


class RateLimitedSamplingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def sample(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.calls == 1:
            raise WeaverAPIError(
                429,
                "rate_limited",
                "request rate limit exceeded",
                True,
                retry_after="2",
            )
        return {
            "sequences": [
                {
                    "tokens": [3],
                    "logprobs": [-0.1],
                    "text": "ok",
                    "stop_reason": "stop",
                    "weight_version": "default",
                }
            ]
        }


def test_weaver_service_holder_retries_rate_limited_sample_submission(monkeypatch):
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._max_concurrent_requests = None  # pylint: disable=protected-access
    client = RateLimitedSamplingClient()
    sleep = AsyncMock()
    monkeypatch.setattr("northrl.weaver.weaver_service_holder.asyncio.sleep", sleep)

    result = run_coro(
        holder._sample_with_retries(client, prompt="hello")  # pylint: disable=protected-access
    )

    assert result["text"] == "ok"
    assert client.calls == 2
    sleep.assert_awaited_once_with(2.0)


def test_weaver_service_holder_limits_concurrent_sampling_requests():
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = FakeTokenizer()  # pylint: disable=protected-access
    holder._sampling_client = SlowSamplingClient()  # pylint: disable=protected-access
    holder._active_sampling_snapshot = (  # pylint: disable=protected-access
        "weaver://weights/default",
        "default",
    )
    holder._return_sampling_mask = False  # pylint: disable=protected-access
    holder._return_old_logprob = False  # pylint: disable=protected-access
    holder._max_concurrent_requests = 2  # pylint: disable=protected-access
    holder._sampling_semaphore = None  # pylint: disable=protected-access
    holder._sampling_semaphore_loop = None  # pylint: disable=protected-access

    async def _noop_ensure(self, require_sampling=False):
        del require_sampling

    holder._ensure_provisioned = pytypes.MethodType(  # pylint: disable=protected-access
        _noop_ensure, holder
    )

    async def _run_samples():
        await asyncio.gather(
            *(holder.sample_from_prompt(prompt=f"hello {idx}", max_tokens=16) for idx in range(5))
        )

    run_coro(_run_samples())

    sampling_client = holder._sampling_client  # pylint: disable=protected-access
    assert sampling_client.max_active == 2


def _contract_sequence(**overrides: Any) -> dict[str, Any]:
    sequence: dict[str, Any] = {
        "tokens": [3, 4],
        "logprobs": [-0.1, -0.2],
        "old_logprobs": [-0.3, -0.4],
        "sampling_masks": [[3, 5], [4, 6]],
        "text": "raw response",
        "stop_reason": "backend-stop",
        "weight_version": "version-1",
    }
    sequence.update(overrides)
    return sequence


def _contract_holder() -> WeaverServiceHolder:
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._return_old_logprob = True  # pylint: disable=protected-access
    holder._return_sampling_mask = True  # pylint: disable=protected-access
    return holder


def test_weaver_sampling_result_preserves_backend_sequence_contract():
    holder = _contract_holder()

    result = holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
        _contract_sequence(),
        [1, 2],
        "version-1",
    )

    assert result == {
        "response": "raw response",
        "prompt_tokens": [1, 2],
        "response_tokens": [3, 4],
        "response_logprobs": [-0.1, -0.2],
        "response_old_logprobs": [-0.3, -0.4],
        "response_sampling_masks": [[3, 5], [4, 6]],
        "finish_reason": "backend-stop",
        "weight_version": "version-1",
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        pytest.param("logprobs", [-0.1], "token/logprob", id="logprobs"),
        pytest.param("old_logprobs", [-0.3], "token/old_logprob", id="old-logprobs"),
        pytest.param("sampling_masks", [[3, 5]], "token/sampling-mask", id="sampling-masks"),
    ],
)
def test_weaver_sampling_result_rejects_misaligned_token_metadata(field, value, error):
    holder = _contract_holder()

    with pytest.raises(ValueError, match=error):
        holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
            _contract_sequence(**{field: value}),
            [1, 2],
            "version-1",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        pytest.param("tokens", (3, 4), "tokens must be a list", id="tokens-container"),
        pytest.param("tokens", [3, "4"], "tokens must contain only integers", id="tokens"),
        pytest.param("logprobs", (-0.1, -0.2), "logprobs must be a list", id="logprobs"),
        pytest.param(
            "logprobs",
            [-0.1, None],
            "invalid_count=1",
            id="logprobs-element",
        ),
        pytest.param(
            "old_logprobs",
            [-0.3, None],
            "old_logprobs must contain only floats",
            id="old-logprobs",
        ),
        pytest.param(
            "sampling_masks",
            [3, 4],
            "sampling_masks must contain only token-ID lists",
            id="sampling-masks",
        ),
        pytest.param("text", None, "text must be a string", id="text"),
        pytest.param(
            "stop_reason",
            None,
            "stop_reason must be a non-empty string",
            id="stop-reason",
        ),
    ],
)
def test_weaver_sampling_result_rejects_invalid_sequence_field_types(field, value, error):
    holder = _contract_holder()

    with pytest.raises(TypeError, match=error):
        holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
            _contract_sequence(**{field: value}),
            [1, 2],
            "version-1",
        )


@pytest.mark.parametrize(
    ("field", "error"),
    [
        pytest.param("old_logprobs", "old_logprobs must be a list", id="old-logprobs"),
        pytest.param("sampling_masks", "sampling_masks must be a list", id="sampling-masks"),
    ],
)
def test_weaver_sampling_result_requires_requested_metadata(field, error):
    holder = _contract_holder()
    sequence = _contract_sequence()
    sequence.pop(field)

    with pytest.raises(TypeError, match=error):
        holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
            sequence,
            [1, 2],
            "version-1",
        )


def test_weaver_sampling_result_accepts_empty_aborted_sequence():
    holder = _contract_holder()

    result = holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
        _contract_sequence(
            tokens=[],
            logprobs=[],
            old_logprobs=[],
            sampling_masks=[],
            text="",
            stop_reason="abort",
        ),
        [1, 2],
        "version-1",
    )

    assert result["response"] == ""
    assert result["response_tokens"] == []
    assert result["finish_reason"] == "abort"


def test_weaver_sampling_result_requires_sequence_weight_version():
    holder = _contract_holder()
    sequence = _contract_sequence()
    sequence.pop("weight_version")

    with pytest.raises(ValueError, match="sequence is missing weight_version"):
        holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
            sequence,
            [1, 2],
            "version-1",
        )


def test_weaver_sampling_result_rejects_empty_sequence_weight_version():
    holder = _contract_holder()

    with pytest.raises(ValueError, match="weight_version must be non-empty"):
        holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
            _contract_sequence(weight_version=""),
            [1, 2],
            "version-1",
        )


@pytest.mark.parametrize("expected_version", [None, "42"])
def test_weaver_sampling_result_returns_opaque_sequence_weight_version(expected_version):
    holder = _contract_holder()

    result = holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
        _contract_sequence(weight_version=42),
        [1, 2],
        expected_version,
    )

    assert result["weight_version"] == "42"


def test_weaver_sampling_result_rejects_wrong_request_weight_version():
    holder = _contract_holder()

    with pytest.raises(
        WeightVersionMismatchError, match="expected='version-1'.*observed='version-2'"
    ):
        holder._weaver_sequence_to_sampling_result(  # pylint: disable=protected-access
            _contract_sequence(weight_version="version-2"),
            [1, 2],
            "version-1",
        )
