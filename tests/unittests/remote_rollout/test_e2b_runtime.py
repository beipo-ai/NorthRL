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

"""Tests for the model runtime that runs inside E2B."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from omegaconf import OmegaConf

from northrl.remote_rollout.contracts import (
    RemoteRolloutTrace,
    RolloutEnvironmentError,
    RolloutInput,
)
from northrl.remote_rollout.e2b_runtime import E2BRolloutRuntime
from northrl.remote_rollout.northgate_relay import NorthGateRelay
from northrl.remote_rollout.wire import (
    decode_agent_stage,
    decode_result,
    decode_trace,
    dump_artifact,
    encode_agent_stage,
    encode_trace,
    load_agent_stage_gzip,
    load_result_gzip,
    load_trace_gzip,
)


def _load_dense_trace(payload):
    return decode_trace(load_trace_gzip(payload))


def _load_dense_result(payload):
    return decode_result(load_result_gzip(payload))


def _load_dense_agent_stage(payload):
    return decode_agent_stage(load_agent_stage_gzip(payload))


def _read_artifact_json(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def _write_artifact_json(path, payload):
    path.write_bytes(gzip.compress(json.dumps(payload).encode(), compresslevel=1, mtime=0))


def _state_path(output_path):
    return Path(f"{str(output_path).removesuffix('.gz')}.state.json")


def _gateway():
    return {
        "base_url": "https://northgate.example",
        "model": "weaver-sampler-1",
        "sampling_session_id": "sampler-1",
        "weight_version": "actor-step-7",
    }


def _tools(*, cache_control=True):
    return [
        {
            "name": "Read",
            "description": "Read a file from the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "Absolute path"}},
                "required": ["file_path"],
            },
            **({"cache_control": {"type": "ephemeral"}} if cache_control else {}),
        }
    ]


class _FakeRelay(NorthGateRelay):
    instances: list["_FakeRelay"] = []

    def __init__(self, config, gateway, **kwargs):
        self.closed = False
        self.kwargs = dict(gateway)
        fake_client = _FakeInferenceClient(config, self, kwargs["on_service_call"])

        async def upstream(request):
            from northrl.exceptions import ContextLengthExceededError

            body = json.loads(request.content)
            try:
                result = await fake_client.generate(body["messages"])
            except ContextLengthExceededError as exc:
                message = (
                    "sequence length exceeds model max_seq_len: "
                    f"prompt {exc.prompt_tokens} + max_new_tokens {exc.max_tokens} = "
                    f"{(exc.prompt_tokens or 0) + (exc.max_tokens or 0)} > {exc.max_sequence_length}"
                )
                if exc.prompt_tokens is None:
                    message = "request exceeds context length"
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded",
                            "message": message,
                        }
                    },
                )
            token_data = result.pop("northrl_train")
            token_data["response_sampling_masks"] = result.pop("sampling_mask", None)
            if request.url.path == "/v1/messages":
                message = result["choices"][0]["message"]
                content = [
                    {
                        "type": "thinking",
                        "thinking": message.get("reasoning_content", ""),
                        "signature": "native-ng-signature",
                    },
                    {"type": "text", "text": message.get("content", "")},
                ]
                result = {
                    "id": result["id"],
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": content,
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": len(token_data["prompt_tokens"]),
                        "output_tokens": len(token_data["response_tokens"]),
                    },
                }
            result["token_data"] = token_data
            return httpx.Response(200, json=result)

        super().__init__(
            config,
            gateway,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
            **kwargs,
        )
        self.__class__.instances.append(self)

    async def close(self):
        self.closed = True
        await super().close()


class _FakeInferenceClient:
    def set_validation_mode(self, enabled: bool) -> None:
        self._is_validation = enabled

    def __init__(
        self,
        config: Any,
        holder: _FakeRelay,
        on_service_call: Callable[[str, Exception | None], None],
    ) -> None:
        del config, holder
        self.call_count = 0
        self.on_service_call = on_service_call

    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del messages, kwargs
        turns = [
            ([1, 2], [3], [-0.1], [-0.2], [[3, 8]]),
            ([1, 2, 3, 4], [5], [-0.3], [-0.4], [[5]]),
        ]
        prompt, response, log_probs, old_log_probs, sampling_mask = turns[self.call_count]
        self.call_count += 1
        self.on_service_call("sample_from_messages", None)
        return {
            "id": f"chatcmpl_{self.call_count}",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                        "reasoning_content": f"private-{self.call_count}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(response),
                "total_tokens": len(prompt) + len(response),
            },
            "northrl_train": {
                "prompt_tokens": prompt,
                "response_tokens": response,
                "response_logprobs": log_probs,
                "response_old_logprobs": old_log_probs,
            },
            "sampling_mask": sampling_mask,
        }


class _FakeUvicornConfig:
    instances: list["_FakeUvicornConfig"] = []

    def __init__(self, app: Any, **kwargs: Any) -> None:
        self.app = app
        self.kwargs = kwargs
        self.__class__.instances.append(self)


class _FakeUvicornServer:
    instances: list["_FakeUvicornServer"] = []

    def __init__(self, config: _FakeUvicornConfig) -> None:
        self.config = config
        self.started = False
        self.should_exit = False
        self.__class__.instances.append(self)

    async def serve(self) -> None:
        self.started = True
        while not self.should_exit:
            await asyncio.sleep(0.001)


def _config():
    return OmegaConf.create(
        {
            "backend": {
                "type": "weaver",
                "base_url": "https://weaver.example",
                "api_key": "config-key",
            },
            "models": {
                "actor": {
                    "name": "actor-model",
                    "tokenizer": "actor-tokenizer",
                }
            },
            "rollout": {
                "num_workers": 4,
                "max_concurrent_requests": 2,
                "sampling": {
                    "temperature": 1.0,
                    "max_tokens": 32,
                    "top_p": 0.9,
                    "top_k": -1,
                    "min_p": 0.0,
                },
                "parsers": {"type": "builtin", "reasoning": None, "tool": None},
            },
            "training": {"weight_sync": {"freeze_inflight": False}},
            "algorithm": {
                "sampling_logprob_source": "inference",
                "loss": {"mode": "importance_sampling"},
            },
            "data": {"max_sequence_length": 262144},
        }
    )


def _short_execution_budgets(monkeypatch, **overrides):
    """Use short wall-clock budgets without relaxing production integer validation."""
    from types import SimpleNamespace

    from northrl.remote_rollout import e2b_runtime

    values = {
        "preparation_seconds": 2.0,
        "agent_seconds": 0.03,
        "finalize_seconds": 1.0,
        "evaluation_seconds": 2.0,
        "recovery_seconds": 2.0,
        "io_seconds": 1.0,
        **overrides,
    }
    monkeypatch.setattr(
        e2b_runtime.ExecutionBudgets,
        "from_config",
        lambda _config: SimpleNamespace(**values),
    )


@pytest.fixture(autouse=True)
def _fake_runtime_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from northrl.remote_rollout import e2b_runtime

    _FakeRelay.instances.clear()
    _FakeUvicornConfig.instances.clear()
    _FakeUvicornServer.instances.clear()
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)
    monkeypatch.setenv("NORTHGATE_API_KEY", "ng-test-key")
    monkeypatch.setattr(e2b_runtime, "NorthGateRelay", _FakeRelay)
    monkeypatch.setattr(e2b_runtime.uvicorn, "Config", _FakeUvicornConfig)
    monkeypatch.setattr(e2b_runtime.uvicorn, "Server", _FakeUvicornServer)


def test_runtime_serves_both_model_apis_and_returns_recorded_trajectory(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json.gz"

    async def scenario() -> None:
        input_data = RolloutInput(
            rollout_id="rollout-1",
            task={"instruction": "fix it"},
            metadata={
                "task_id": "task-1",
                "group_id": "group-1",
                "run_id": 2,
                "weight_version": "actor-step-7",
            },
        )
        captured_environment: dict[str, str] = {}

        async def run_harness(
            actual_input_data: RolloutInput,
            model_environment: dict[str, str],
        ) -> tuple[float, dict[str, float]]:
            assert actual_input_data is input_data
            assert _FakeUvicornServer.instances[-1].started
            captured_environment.update(model_environment)
            app = _FakeUvicornConfig.instances[-1].app
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://runtime",
            ) as http:
                openai_response = await http.post(
                    "/v1/chat/completions",
                    json={
                        "model": "northrl",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                anthropic_response = await http.post(
                    "/v1/messages",
                    json={
                        "model": "northrl",
                        "max_tokens": 32,
                        "tools": _tools(),
                        "messages": [{"role": "user", "content": "continue"}],
                    },
                )
            assert openai_response.status_code == 200
            assert anthropic_response.status_code == 200
            assert "reasoning_content" not in anthropic_response.text
            assert anthropic_response.json()["content"][0] == {
                "type": "thinking",
                "thinking": "private-2",
                "signature": "native-ng-signature",
            }
            return 1.0, {"tests_passed": 1.0}

        config = _config()
        config.rollout.keep_sampling_mask = True
        runtime = E2BRolloutRuntime(
            config,
            gateway=_gateway(),
            port=18080,
            trace_path=trace_path,
        )
        result = await runtime.run(input_data, run_harness)

        snapshot_json = _read_artifact_json(trace_path)
        assert "segments" in snapshot_json
        assert "trajectories" not in snapshot_json
        snapshot = _load_dense_trace(trace_path.read_bytes())
        assert snapshot.rollout_id == "rollout-1"
        assert snapshot.metadata == input_data.metadata
        assert snapshot.tools == _tools(cache_control=False)
        assert snapshot.messages[-1]["role"] == "assistant"
        assert snapshot.segments[-1].tokens == [1, 2, 3, 4, 5]
        assert "segments" in result.model_dump()
        assert "trajectories" not in result.model_dump()
        assert result.rollout_id == "rollout-1"
        assert result.reward == 1.0
        assert result.metrics["tests_passed"] == 1.0
        assert result.metrics["model_turns"] == 2.0
        assert result.metrics["generated_tokens"] == 2.0
        assert result.metrics["max_context_tokens"] == 5.0
        assert result.metrics["trace_segments"] == 1.0
        assert result.message_format == "anthropic"
        assert result.message_system is None
        assert result.tools == _tools(cache_control=False)
        assert result.messages == [
            {"role": "user", "content": "continue"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private-2",
                        "signature": "native-ng-signature",
                    },
                    {"type": "text", "text": "ok"},
                ],
            },
        ]
        [trajectory] = result.segments
        assert trajectory.tokens == [1, 2, 3, 4, 5]
        assert trajectory.loss_mask == [0, 0, 1, 0, 1]
        assert trajectory.log_probs == [0.0, 0.0, -0.1, 0.0, -0.3]
        assert trajectory.old_log_probs == [0.0, 0.0, -0.2, 0.0, -0.4]
        assert trajectory.sampling_mask == [[3, 8], [5]]
        assert captured_environment == {
            "OPENAI_BASE_URL": "http://127.0.0.1:18080/v1",
            "OPENAI_API_KEY": "northrl",
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:18080",
            "ANTHROPIC_API_KEY": "northrl",
        }

        assert _FakeRelay.instances[-1].kwargs == _gateway()
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit

    asyncio.run(scenario())


def test_runtime_cleans_up_when_harness_fails() -> None:
    async def scenario() -> None:
        async def fail_harness(
            input_data: RolloutInput,
            model_environment: dict[str, str],
        ) -> tuple[float, dict[str, float]]:
            del input_data, model_environment
            raise RuntimeError("harness failed")

        runtime = E2BRolloutRuntime(
            _config(),
            gateway=_gateway(),
        )
        input_data = RolloutInput(rollout_id="rollout-1", task={})
        with pytest.raises(RuntimeError, match="harness failed"):
            await runtime.run(input_data, fail_harness)

        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit

    asyncio.run(scenario())


@pytest.mark.parametrize("harness_fails", [True, False])
def test_runtime_reports_unrecovered_service_failure(
    monkeypatch: pytest.MonkeyPatch, harness_fails: bool
) -> None:
    from northrl.remote_rollout import e2b_runtime

    class FailingClient(_FakeInferenceClient):
        async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            del messages, kwargs
            error = ConnectionError("Weaver unavailable")
            self.on_service_call("sample_from_messages", error)
            raise error

    monkeypatch.setattr(sys.modules[__name__], "_FakeInferenceClient", FailingClient)

    async def scenario() -> None:
        async def run_harness(
            input_data: RolloutInput, model_environment: dict[str, str]
        ) -> tuple[float, dict[str, float]]:
            del input_data, model_environment
            app = _FakeUvicornConfig.instances[-1].app
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as http:
                with pytest.raises(ConnectionError, match="Weaver unavailable"):
                    await http.post(
                        "/v1/messages",
                        json={
                            "model": "northrl",
                            "messages": [{"role": "user", "content": "fix it"}],
                        },
                    )
            if harness_fails:
                raise RuntimeError("Claude exited after retries")
            # Even a zero exit cannot produce a successful rollout without any trajectory.
            return 0.0, {}

        runtime = E2BRolloutRuntime(_config(), gateway=_gateway())
        with pytest.raises(RolloutEnvironmentError, match="Weaver unavailable"):
            await runtime.run(RolloutInput(rollout_id="rollout-1", task={}), run_harness)
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit

    asyncio.run(scenario())


@pytest.mark.parametrize("recover", [True, False])
@pytest.mark.parametrize("harness_fails", [True, False])
def test_runtime_requires_service_recovery_even_when_harness_exits_successfully(
    monkeypatch: pytest.MonkeyPatch, recover: bool, harness_fails: bool
) -> None:
    from northrl.remote_rollout import e2b_runtime

    class IntermittentClient(_FakeInferenceClient):
        async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            # A completed call must not clear a different failed request.
            self.on_service_call("earlier request", ConnectionError("gateway unavailable"))
            result = await super().generate(messages, **kwargs)
            if recover:
                self.on_service_call("earlier request", None)
            return result

    monkeypatch.setattr(sys.modules[__name__], "_FakeInferenceClient", IntermittentClient)

    async def scenario() -> None:
        async def run_harness(
            input_data: RolloutInput, model_environment: dict[str, str]
        ) -> tuple[float, dict[str, float]]:
            del input_data, model_environment
            app = _FakeUvicornConfig.instances[-1].app
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as http:
                response = await http.post(
                    "/v1/chat/completions",
                    json={"model": "northrl", "messages": [{"role": "user", "content": "fix it"}]},
                )
            assert response.status_code == 200
            if harness_fails:
                raise RuntimeError("agent failed")
            return 0.0, {}

        runtime = E2BRolloutRuntime(_config(), gateway=_gateway())
        input_data = RolloutInput(rollout_id="rollout-1", task={})
        if harness_fails or not recover:
            expected = RuntimeError if recover else RolloutEnvironmentError
            with pytest.raises(expected) as caught:
                await runtime.run(input_data, run_harness)
            assert isinstance(caught.value, RolloutEnvironmentError) is not recover
        else:
            result = await runtime.run(input_data, run_harness)
            assert result.reward == 0.0
            assert result.segments
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit

    asyncio.run(scenario())


def test_load_input_preserves_complete_task_and_makes_payload_private(tmp_path: Path) -> None:
    from northrl.remote_rollout.e2b_runtime import _load_input

    task = {
        "instruction": "fix it",
        "interface": "Required API",
        "patch": "reference solution",
        "test_patch": "hidden tests",
        "custom": {"options": [1, None, True]},
    }
    payload = {
        "input_data": {"rollout_id": "rollout-1", "task": task},
        "config": {"custom": "configuration"},
        "gateway": _gateway(),
        "attempt_id": "attempt-1",
    }
    path = tmp_path / "input.json"
    contents = json.dumps(payload)
    path.write_text(contents)
    path.chmod(0o644)

    input_data, config, gateway, attempt_id = _load_input(path)

    assert input_data.task == task
    assert config == payload["config"]
    assert gateway == payload["gateway"]
    assert attempt_id == payload["attempt_id"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_text() == contents


@pytest.mark.parametrize("symlink_parent", [False, True])
def test_load_input_rejects_symlink_without_reading_or_changing_target(
    tmp_path: Path, symlink_parent: bool
) -> None:
    from northrl.remote_rollout.e2b_runtime import _load_input

    directory = tmp_path / "target"
    directory.mkdir()
    target = directory / "input.json"
    target.write_text("not JSON")
    target.chmod(0o644)
    path = tmp_path / "link"
    if symlink_parent:
        path.symlink_to(directory, target_is_directory=True)
        path /= "input.json"
    else:
        path.symlink_to(target)

    with pytest.raises(RuntimeError, match="rollout input must"):
        _load_input(path)

    assert target.stat().st_mode & 0o777 == 0o644
    assert target.read_text() == "not JSON"


def test_load_input_rejects_foreign_owner_without_reading_or_changing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from northrl.remote_rollout import e2b_runtime

    path = tmp_path / "input.json"
    path.write_text("not JSON")
    path.chmod(0o644)
    foreign_uid = path.stat().st_uid + 1
    monkeypatch.setattr(e2b_runtime.os, "geteuid", lambda: foreign_uid)

    with pytest.raises(RuntimeError, match="rollout input must be owned by the runtime"):
        e2b_runtime._load_input(path)

    assert path.stat().st_mode & 0o777 == 0o644
    assert path.read_text() == "not JSON"


def _phase_setup(tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType

    from northrl.remote_rollout.contracts import AgentResult, EvaluationResult

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assets = tmp_path / "private-evaluation"
    artifacts = tmp_path / "platform"
    artifacts.mkdir()
    input_path = artifacts / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "input_data": {
                    "rollout_id": "rollout-1",
                    "task": {
                        "instruction": "fix it",
                        "task_id": "task-1",
                        "workspace": str(workspace),
                    },
                },
                "config": OmegaConf.to_container(_config()),
                "gateway": _gateway(),
                "attempt_id": "attempt-1",
            }
        )
    )
    events = []
    adapter = ModuleType("test_e2b_stage_adapter")

    async def prepare(input_data, task_workspace):
        assert input_path.stat().st_mode & 0o777 == 0o600
        assert set(input_data.model_dump()) == {"rollout_id", "task", "metadata"}
        assert input_data.rollout_id == "rollout-1"
        assert input_data.task["instruction"] == "fix it"
        assert input_data.task["task_id"] == "task-1"
        assert task_workspace == workspace
        assert assets.is_dir() and not list(assets.iterdir())
        assert assets.stat().st_mode & 0o777 == 0o700
        events.append("prepare")

    async def run_agent(input_data, task_workspace, model_environment):
        assert input_path.stat().st_mode & 0o777 == 0o600
        assert input_data.rollout_id == "rollout-1"
        assert input_data.task["instruction"] == "fix it"
        assert input_data.task["task_id"] == "task-1"
        assert assets.is_dir() and not list(assets.iterdir())
        assert model_environment["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
        assert _FakeUvicornServer.instances[-1].started
        events.append("agent")
        (task_workspace / "solution.txt").write_text("agent answer")
        app = _FakeUvicornConfig.instances[-1].app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as http:
            response = await http.post(
                "/v1/messages",
                json={
                    "model": "northrl",
                    "tools": _tools(),
                    "messages": [{"role": "user", "content": "fix it"}],
                },
            )
            assert response.status_code == 200
        return AgentResult(final_answer="done", artifacts={"answer": "solution.txt"})

    async def evaluate(input_data, task_workspace, agent_result, evaluation_assets):
        assert input_data.rollout_id == "rollout-1"
        assert input_data.task["instruction"] == "fix it"
        assert input_data.task["task_id"] == "task-1"
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit
        assert (task_workspace / agent_result.artifacts["answer"]).read_text() == "agent answer"
        assert (evaluation_assets / "test.sh").read_text() == "hidden evaluation"
        events.append("evaluate")
        return EvaluationResult(reward=0.5, metrics={"task_score": 0.5})

    adapter.prepare = prepare
    adapter.run_agent = run_agent
    adapter.evaluate = evaluate
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    script = tmp_path / "adapter.py"
    script.write_text("from test_e2b_stage_adapter import *\n")
    return (
        adapter,
        events,
        {
            "input_path": input_path,
            "output_path": artifacts / "result.json.gz",
            "trace_path": artifacts / "trace.json.gz",
            "stage_path": artifacts / "agent-stage.json.gz",
            "entrypoint": str(script),
            "evaluation_assets": assets,
        },
    )


@pytest.mark.parametrize("has_prepare", [False, True])
@pytest.mark.parametrize(
    "policy, enabled",
    [
        (None, False),
        ({"deny_domains": []}, False),
        ({"deny_domains": ["github.com"]}, True),
        ({"mode": "local_only"}, True),
        ({"mode": "local_only", "deny_domains": []}, True),
    ],
)
def test_network_policy_requires_adapter_support_only_when_enabled(
    tmp_path, monkeypatch, has_prepare, policy, enabled
):
    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    if not has_prepare:
        del adapter.prepare
    payload = json.loads(options["input_path"].read_text())
    if policy is not None:
        payload["config"]["rollout"].setdefault("execution", {})["network_policy"] = policy
    options["input_path"].write_text(json.dumps(payload))

    if enabled:
        with pytest.raises(RolloutEnvironmentError, match="network policy"):
            asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
        assert "agent" not in events
        assert not options["stage_path"].exists()
        assert _load_dense_result(options["output_path"].read_bytes()).status == "env_error"
    else:
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
        assert "agent" in events
        assert options["stage_path"].exists()


def test_runtime_passes_trusted_network_policy_to_all_hooks(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from contextvars import ContextVar

    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    payload = json.loads(options["input_path"].read_text())
    expected = {"deny_domains": ["github.com", "example.org"]}
    payload["config"]["rollout"].setdefault("execution", {})["network_policy"] = expected
    payload["input_data"]["metadata"] = {"task_id": "trusted-task"}
    # Dataset content must not override the host-selected policy.
    payload["input_data"]["task"]["network_policy"] = {"deny_domains": []}
    options["input_path"].write_text(json.dumps(payload))
    current = ContextVar("test_network_policy", default=None)
    observed = []

    @contextmanager
    def policy_context(policy, *, task_id):
        token = current.set((policy, task_id))
        try:
            yield
        finally:
            current.reset(token)

    def prepared():
        assert current.get() == (expected, "trusted-task")
        observed.append("prepared")

    def observe(name, hook):
        async def call(*args):
            assert current.get() == (expected, "trusted-task")
            observed.append(name)
            return await hook(*args)

        return call

    monkeypatch.setattr(e2b_runtime, "network_policy_context", policy_context)
    monkeypatch.setattr(e2b_runtime, "assert_network_policy_prepared", prepared)
    for name in ("prepare", "run_agent", "evaluate"):
        setattr(adapter, name, observe(name, getattr(adapter, name)))
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    assert current.get() is None
    (options["evaluation_assets"] / "test.sh").write_text("hidden evaluation")
    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    assert current.get() is None
    assert observed == ["prepare", "prepared", "run_agent", "evaluate"]
    assert events == ["prepare", "agent", "evaluate"]


def test_two_phases_commit_only_after_shutdown_and_evaluate_without_restarting(
    tmp_path, monkeypatch
):
    import json

    from northrl.remote_rollout import e2b_runtime

    _adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    payload = json.loads(options["input_path"].read_text())
    metadata = {
        "task_id": "task-1",
        "group_id": "group-1",
        "run_id": 2,
        "weight_version": "v1",
    }
    payload["input_data"]["metadata"] = metadata
    options["input_path"].write_text(json.dumps(payload))
    write_json = e2b_runtime.write_artifact

    def observe_stage_commit(path, payload):
        if path == options["stage_path"]:
            assert _FakeRelay.instances[-1].closed
            assert _FakeUvicornServer.instances[-1].should_exit
            assert not options["output_path"].exists()
            events.append("stage-commit")
        write_json(path, payload)

    monkeypatch.setattr(e2b_runtime, "write_artifact", observe_stage_commit)
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    assert events == ["prepare", "agent", "stage-commit"]
    assert not options["output_path"].exists()
    stage_bytes = options["stage_path"].read_bytes()
    trace_bytes = options["trace_path"].read_bytes()
    assert json.loads(gzip.decompress(stage_bytes))["schema_version"] == 2
    assert json.loads(gzip.decompress(trace_bytes))["schema_version"] == 2
    stage = _load_dense_agent_stage(stage_bytes)
    assert stage.attempt_id == "attempt-1"
    assert stage.agent_result.final_answer == "done"
    assert stage.trace == _load_dense_trace(trace_bytes)
    assert stage.trace.metadata == metadata
    assert stage.trace.tools == _tools(cache_control=False)
    metrics_path = options["output_path"].with_name("metrics.json")
    before = json.loads(metrics_path.read_text())
    assert before["phase"] == "agent_completed"
    journal_path = options["output_path"].with_name("requests.jsonl")
    journal_bytes = journal_path.read_bytes()
    records = [json.loads(line) for line in journal_bytes.splitlines()]
    assert len(records) == 1
    assert records[0]["attempt_id"] == "attempt-1"
    assert records[0]["rollout_id"] == payload["input_data"]["rollout_id"]
    assert records[0]["status"] == "success"
    assert records[0]["duration_seconds"] >= 0
    assert records[0]["completion_tokens"] == 1
    assert "messages" not in records[0]
    assert journal_path.stat().st_mode & 0o777 == 0o600
    (options["evaluation_assets"] / "test.sh").write_text("hidden evaluation")
    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    assert events == ["prepare", "agent", "stage-commit", "evaluate"]
    assert len(_FakeRelay.instances) == 1
    assert options["stage_path"].read_bytes() == stage_bytes
    assert options["trace_path"].read_bytes() == trace_bytes
    assert journal_path.read_bytes() == journal_bytes
    assert _read_artifact_json(options["output_path"])["schema_version"] == 2
    result = _load_dense_result(options["output_path"].read_bytes())
    assert result.reward == 0.5
    assert result.segments == stage.trace.segments
    assert result.tools == _tools(cache_control=False)
    assert result.metrics["model_turns"] == 1.0
    final = json.loads(metrics_path.read_text())
    assert final["phase"] == "finished"
    assert final["heartbeat"] > before["heartbeat"]
    assert final["metrics"]["model_turns"] == 1.0
    assert final["metrics"]["task_score"] == 0.5
    assert options["stage_path"].stat().st_mode & 0o777 == 0o600
    assert options["output_path"].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure_phase", ["prepare", "service", "cleanup", "evaluate"])
def test_environment_failures_preserve_diagnostic_trace_without_training_data(
    tmp_path, monkeypatch, failure_phase
):
    import json

    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    run_agent = adapter.run_agent

    async def fail(*_args):
        raise RuntimeError(f"{failure_phase} failed")

    async def agent_failure(*args):
        await run_agent(*args)
        raise RolloutEnvironmentError("model service unavailable")

    if failure_phase == "prepare":
        adapter.prepare = fail
    elif failure_phase == "service":
        adapter.run_agent = agent_failure
    elif failure_phase == "cleanup":
        monkeypatch.setattr(_FakeRelay, "close", fail)
    elif failure_phase == "evaluate":
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
        adapter.evaluate = fail
    with pytest.raises(RuntimeError):
        asyncio.run(
            e2b_runtime._run_phase(
                phase="evaluate" if failure_phase == "evaluate" else "agent", **options
            )
        )
    result = _load_dense_result(options["output_path"].read_bytes())
    assert result.status == "env_error"
    assert result.segments == []
    assert result.tools is None
    assert result.reward == 0.0
    assert options["trace_path"].exists() is (failure_phase != "prepare")
    if failure_phase != "prepare":
        trace = _load_dense_trace(options["trace_path"].read_bytes())
        assert trace.tools == _tools(cache_control=False)
    assert options["stage_path"].exists() is (failure_phase == "evaluate")
    state = json.loads(_state_path(options["output_path"]).read_text())
    assert state["attempt_id"] == "attempt-1"
    assert state["status"] == result.status


@pytest.mark.parametrize("agent_outcome", ["success", "exit1", "exit124", "raises", "timeout"])
@pytest.mark.parametrize("record_turn", [True, False])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_agent_exit_status_does_not_replace_workspace_evaluation(
    tmp_path, monkeypatch, agent_outcome, record_turn, reward
):
    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.contracts import AgentResult, EvaluationResult

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    original_agent = adapter.run_agent
    expected_exit_code = {
        "success": 0,
        "exit1": 1,
        "exit124": 124,
        "raises": 1,
        "timeout": 124,
    }[agent_outcome]

    async def run_agent(input_data, workspace, model_environment):
        try:
            if record_turn:
                result = await original_agent(input_data, workspace, model_environment)
            else:
                events.append("agent")
                (workspace / "solution.txt").write_text("agent answer")
                result = AgentResult(artifacts={"answer": "solution.txt"})
            if agent_outcome == "raises":
                raise RuntimeError("Claude exited without a successful final event")
            if agent_outcome == "timeout":
                raise TimeoutError("Claude exceeded its execution limit")
            return result.model_copy(update={"exit_code": expected_exit_code})
        finally:
            await asyncio.sleep(0)
            events.append("agent-cleaned")

    async def evaluate(input_data, workspace, agent_result, evaluation_assets):
        assert input_data.rollout_id == "rollout-1"
        assert events[-1] == "agent-cleaned"
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit
        assert agent_result.exit_code == expected_exit_code
        assert (workspace / "solution.txt").read_text() == "agent answer"
        assert (evaluation_assets / "test.sh").read_text() == "hidden evaluation"
        events.append("evaluate")
        return EvaluationResult(
            reward=reward,
            status="success" if reward else "task_failed",
            metrics={"task_score": reward},
        )

    adapter.run_agent = run_agent
    adapter.evaluate = evaluate
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
    assert stage.agent_result.exit_code == expected_exit_code
    assert (stage.trace is not None) is record_turn
    stage_metrics = stage.trace.metrics if record_turn else stage.metrics
    assert stage_metrics["agent_exit_code"] == float(expected_exit_code)
    assert stage.termination_reason is None
    assert events == ["prepare", "agent", "agent-cleaned"]
    assert not options["output_path"].exists()

    (options["evaluation_assets"] / "test.sh").write_text("hidden evaluation")
    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    result = _load_dense_result(options["output_path"].read_bytes())
    assert events == ["prepare", "agent", "agent-cleaned", "evaluate"]
    assert len(_FakeRelay.instances) == 1
    assert result.reward == reward
    assert result.status == ("success" if reward else "task_failed")
    assert result.segments == (stage.trace.segments if record_turn else [])
    assert result.metrics["task_score"] == reward
    assert result.metrics["agent_exit_code"] == float(expected_exit_code)


@pytest.mark.parametrize(
    "field,value", [("rollout_id", "other-run"), ("attempt_id", "old-attempt")]
)
def test_evaluate_rejects_wrong_stage_before_calling_task(tmp_path, monkeypatch, field, value):
    import json

    from northrl.remote_rollout import e2b_runtime

    _adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    stage = _read_artifact_json(options["stage_path"])
    stage[field] = value
    if field == "rollout_id":
        stage["trace"][field] = value
    _write_artifact_json(options["stage_path"], stage)
    with pytest.raises(ValueError, match="identity"):
        asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    assert events == ["prepare", "agent"]
    assert _read_artifact_json(options["output_path"])["status"] == "env_error"


@pytest.mark.parametrize("preexisting", ["file", "symlink"])
def test_agent_refuses_preloaded_hidden_assets_before_loading_task(
    tmp_path, monkeypatch, preexisting
):
    from northrl.remote_rollout import e2b_runtime

    _adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    assets = options["evaluation_assets"]
    if preexisting == "file":
        assets.mkdir()
        (assets / "test.sh").write_text("preloaded secret")
    else:
        target = tmp_path / "preloaded"
        target.mkdir()
        (target / "test.sh").write_text("preloaded secret")
        assets.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        e2b_runtime, "_load_adapter", lambda *_: pytest.fail("task loaded too soon")
    )
    with pytest.raises(RuntimeError):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    assert (assets / "test.sh").read_text() == "preloaded secret"
    assert not events
    assert not _FakeRelay.instances


def test_script_adapter_is_imported_without_executing_main(tmp_path):
    from northrl.remote_rollout import e2b_runtime

    path = tmp_path / "adapter.py"
    path.write_text(
        "async def run_agent(*args): pass\n"
        "async def evaluate(*args): pass\n"
        "if __name__ == '__main__':\n    raise AssertionError('main ran')\n"
    )
    adapter = e2b_runtime._load_adapter(str(path))
    assert adapter.__name__ != "__main__"
    assert callable(adapter.run_agent)


@pytest.mark.parametrize(
    "workspace", [None, "relative/path", "/nonexistent-northrl-test-workspace"]
)
def test_workspace_must_be_explicit_absolute_existing_directory(workspace):
    from types import ModuleType

    from northrl.remote_rollout import e2b_runtime

    input_data = RolloutInput(rollout_id="run", task={"workspace": workspace})
    with pytest.raises((ValueError, FileNotFoundError)):
        asyncio.run(e2b_runtime._resolve_workspace(ModuleType("adapter"), input_data, None))


@pytest.mark.parametrize("invalid", ["sync_hook", "wrong_type", "invalid_model"])
def test_agent_hook_contract_errors_are_not_training_failures(tmp_path, monkeypatch, invalid):
    import json

    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.contracts import AgentResult

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    agent = adapter.run_agent
    if invalid == "sync_hook":
        adapter.run_agent = lambda *_: None
    else:

        async def bad_result(*args):
            await agent(*args)
            if invalid == "wrong_type":
                return {"exit_code": 0}
            return AgentResult.model_construct(exit_code="invalid")

        adapter.run_agent = bad_result
    with pytest.raises((RuntimeError, TypeError)):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    result = _read_artifact_json(options["output_path"])
    assert result["status"] == "env_error"
    assert result["segments"] == []
    assert not options["stage_path"].exists()


def test_agent_none_output_is_normalized_and_evaluator_cannot_rewrite_model_metrics(
    tmp_path, monkeypatch
):
    import json

    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.contracts import AgentResult, EvaluationResult

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    agent = adapter.run_agent

    async def no_agent_result(*args):
        await agent(*args)

    async def evaluate(*args):
        return EvaluationResult(reward=0.0, status="task_failed", metrics={"model_turns": 0.0})

    adapter.run_agent = no_agent_result
    adapter.evaluate = evaluate
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
    assert stage.agent_result == AgentResult()
    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    result = _read_artifact_json(options["output_path"])
    assert result["status"] == "task_failed"
    assert result["segments"]
    assert result["metrics"]["model_turns"] == 1.0


def test_evaluation_cli_runs_in_a_fresh_process_without_model_credentials(tmp_path):
    import json
    import os
    import subprocess
    import sys

    from northrl.remote_rollout.contracts import AgentResult, AgentStageResult, RemoteSegment

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assets = tmp_path / "evaluation"
    assets.mkdir()
    (assets / "test.sh").write_text("secret tests")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    trace = RemoteRolloutTrace(
        rollout_id="cli-rollout",
        tools=_tools(),
        segments=[
            RemoteSegment(
                tokens=[1, 2], loss_mask=[0, 1], log_probs=[0.0, -0.1], old_log_probs=[0.0, -0.2]
            )
        ],
        metrics={"model_turns": 1.0},
    )
    stage = AgentStageResult(
        rollout_id="cli-rollout",
        attempt_id="cli-attempt",
        workspace=str(workspace),
        agent_result=AgentResult(final_answer="answer"),
        trace=trace,
    )
    stage_path = artifacts / "stage.json.gz"
    trace_path = artifacts / "trace.json.gz"
    input_path = artifacts / "input.json"
    output_path = artifacts / "result.json.gz"
    stage_path.write_bytes(dump_artifact(encode_agent_stage(stage)))
    trace_path.write_bytes(dump_artifact(encode_trace(trace)))
    input_path.write_text(
        json.dumps(
            {
                "input_data": {"rollout_id": "cli-rollout", "task": {}},
                "config": {},
                "gateway": _gateway(),
                "attempt_id": "cli-attempt",
            }
        )
    )
    script = tmp_path / "adapter.py"
    script.write_text(
        "from northrl.remote_rollout.contracts import EvaluationResult\n"
        "async def prepare(*args): raise AssertionError('prepare restarted')\n"
        "async def run_agent(*args): raise AssertionError('agent restarted')\n"
        "async def evaluate(input_data, workspace, agent_result, evaluation_assets):\n"
        "    assert agent_result.final_answer == 'answer'\n"
        "    assert (evaluation_assets / 'test.sh').read_text() == 'secret tests'\n"
        "    return EvaluationResult(reward=0.75)\n"
        "if __name__ == '__main__': raise AssertionError('main executed')\n"
    )
    environment = os.environ.copy()
    environment.pop("WEAVER_API_KEY", None)
    environment.pop("WEAVER_BASE_URL", None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    command = [
        sys.executable,
        "-m",
        "northrl.remote_rollout.e2b_runtime",
        "--phase",
        "evaluate",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--trace-output",
        str(trace_path),
        "--stage-path",
        str(stage_path),
        "--evaluation-assets",
        str(assets),
        "--entrypoint",
        str(script),
    ]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = _load_dense_result(output_path.read_bytes())
    assert result.reward == 0.75
    assert result.tools == _tools()
    assert result.metrics["model_turns"] == 1.0
    assert _load_dense_agent_stage(stage_path.read_bytes()) == stage
    assert _load_dense_trace(trace_path.read_bytes()) == trace


def test_agent_phase_cancellation_closes_runtime_without_committing_stage(tmp_path, monkeypatch):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    agent = adapter.run_agent

    async def scenario():
        ready = asyncio.Event()

        async def wait_after_recording(*args):
            await agent(*args)
            ready.set()
            await asyncio.Event().wait()

        adapter.run_agent = wait_after_recording
        task = asyncio.create_task(e2b_runtime._run_phase(phase="agent", **options))
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit
        assert options["trace_path"].is_file()
        assert not options["stage_path"].exists()
        assert not options["output_path"].exists()

    asyncio.run(scenario())


def test_runtime_cleanup_failure_takes_precedence_over_agent_failure(tmp_path, monkeypatch):
    import json

    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    agent = adapter.run_agent

    async def agent_failure(*args):
        await agent(*args)
        raise RuntimeError("agent failed")

    async def cleanup_failure(_self):
        raise RuntimeError("relay close failed")

    adapter.run_agent = agent_failure
    monkeypatch.setattr(_FakeRelay, "close", cleanup_failure)
    with pytest.raises(RolloutEnvironmentError, match="relay close failed"):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    result = _read_artifact_json(options["output_path"])
    assert result["status"] == "env_error"
    assert result["segments"] == []
    assert options["trace_path"].is_file()
    assert not options["stage_path"].exists()


@pytest.mark.parametrize("hook_name", ["run_agent", "evaluate", "prepare"])
def test_invalid_async_hook_signature_is_rejected_before_agent_execution(
    tmp_path, monkeypatch, hook_name
):
    import json

    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)

    async def wrong_signature(input_data):
        pytest.fail("incompatible hook must never execute")

    setattr(adapter, hook_name, wrong_signature)
    with pytest.raises(TypeError, match=f"{hook_name} must accept"):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    result = _read_artifact_json(options["output_path"])
    assert result["status"] == "env_error"
    assert result["segments"] == []
    assert not events
    assert not _FakeRelay.instances
    assert not options["stage_path"].exists()


@pytest.mark.parametrize("known_counts", [True, False])
@pytest.mark.parametrize("record_turn", [True, False])
@pytest.mark.parametrize("agent_exit", ["raises", "nonzero", "wait"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_context_stop_cleans_up_commits_stage_and_preserves_evaluation_reward(
    tmp_path, monkeypatch, record_turn, agent_exit, reward, known_counts
):
    import json

    from northrl.exceptions import ContextLengthExceededError
    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.contracts import AgentResult, EvaluationResult

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    original_agent = adapter.run_agent

    class ContextLimitedClient(_FakeInferenceClient):
        async def generate(self, messages, **kwargs):
            if record_turn and self.call_count == 0:
                return await super().generate(messages, **kwargs)
            error = (
                ContextLengthExceededError(262144, 8192, 262144)
                if known_counts
                else ContextLengthExceededError()
            )
            raise error

    monkeypatch.setattr(sys.modules[__name__], "_FakeInferenceClient", ContextLimitedClient)

    async def stopped_agent(input_data, workspace, model_environment):
        if record_turn:
            await original_agent(input_data, workspace, model_environment)
        else:
            (workspace / "solution.txt").write_text("agent answer")
        try:
            app = _FakeUvicornConfig.instances[-1].app
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as http:
                with pytest.raises(ContextLengthExceededError) as caught:
                    await http.post(
                        "/v1/messages",
                        json={
                            "model": "northrl",
                            "messages": [{"role": "user", "content": "next"}],
                        },
                    )
            assert caught.value.length_details == (
                {
                    "prompt_tokens": 262144,
                    "requested_max_tokens": 8192,
                    "max_sequence_length": 262144,
                }
                if known_counts
                else {"requested_max_tokens": 32}
            )
            if agent_exit == "raises":
                raise RuntimeError("agent exited after context limit")
            if agent_exit == "nonzero":
                return AgentResult(exit_code=1)
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            events.append("agent-cleaned")

    async def evaluate(input_data, workspace, agent_result, evaluation_assets):
        assert events[-1] == "agent-cleaned"
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit
        assert (workspace / "solution.txt").read_text() == "agent answer"
        assert (evaluation_assets / "test.sh").read_text() == "hidden evaluation"
        events.append("evaluate-length-stop")
        return EvaluationResult(reward=reward, metrics={"swebench_resolved": reward})

    adapter.run_agent = stopped_agent
    adapter.evaluate = evaluate

    async def agent_phase():
        await asyncio.wait_for(e2b_runtime._run_phase(phase="agent", **options), timeout=2)

    asyncio.run(agent_phase())
    stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
    assert stage.termination_reason == "context_length_exceeded"
    assert (stage.trace is not None) == record_turn
    assert not options["output_path"].exists()
    assert _FakeRelay.instances[-1].kwargs == _gateway()
    assert json.loads(_state_path(options["output_path"]).read_text())["phase"] == "agent_completed"

    (options["evaluation_assets"] / "test.sh").write_text("hidden evaluation")
    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    result = _load_dense_result(options["output_path"].read_bytes())
    assert result.status == "success"
    assert result.reward == reward
    assert result.metrics["swebench_resolved"] == reward
    assert result.termination_reason == "context_length_exceeded"
    assert result.length_details == (
        {
            "prompt_tokens": 262144,
            "requested_max_tokens": 8192,
            "max_sequence_length": 262144,
        }
        if known_counts
        else {"requested_max_tokens": 32}
    )
    assert bool(result.segments) == record_turn
    assert result.metrics["model_turns"] == float(record_turn)
    assert result.error_type is None
    assert len(_FakeRelay.instances) == 1
    assert events.count("agent-cleaned") == 1


def test_context_stop_cancels_retry_backoff_without_discarding_recorded_segments(
    tmp_path, monkeypatch
):
    from northrl.exceptions import ContextLengthExceededError
    from northrl.remote_rollout import e2b_runtime

    async def scenario():
        backoff_ready = asyncio.Event()
        retry_cancelled = asyncio.Event()
        clients = []
        retry_responses = []

        class RetryBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"error":{"message":"temporarily unavailable"}}'

            async def aclose(self):
                backoff_ready.set()

        def upstream(request):
            content = json.loads(request.content)["messages"][-1]["content"]
            if content == "retry":
                # Keep the retry asleep until context-stop cleanup cancels it.
                response = httpx.Response(503, stream=RetryBody(), headers={"Retry-After": "30"})
                retry_responses.append(response)
                return response
            if content == "limit":
                return httpx.Response(
                    400,
                    json={"error": {"code": "context_length_exceeded"}},
                )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "token_data": {
                        "prompt_tokens": [1, 2],
                        "response_tokens": [3],
                        "response_logprobs": [-0.1],
                        "response_old_logprobs": [-0.2],
                    },
                },
            )

        def relay(config, gateway, **kwargs):
            client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            clients.append(client)
            return NorthGateRelay(config, gateway, http_client=client, **kwargs)

        monkeypatch.setattr(e2b_runtime, "NorthGateRelay", relay)

        async def agent(_input, _env):
            app = _FakeUvicornConfig.instances[-1].app
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as http:

                async def request(content):
                    return await http.post(
                        "/v1/chat/completions",
                        json={
                            "model": "northrl",
                            "messages": [{"role": "user", "content": content}],
                        },
                    )

                async def retry_request():
                    try:
                        await request("retry")
                    except asyncio.CancelledError:
                        retry_cancelled.set()
                        raise

                assert (await request("warmup")).status_code == 200
                pending = asyncio.create_task(retry_request())
                try:
                    await backoff_ready.wait()
                    assert not pending.done()
                    with pytest.raises(ContextLengthExceededError):
                        await request("limit")
                    await asyncio.Event().wait()
                finally:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

        result = await asyncio.wait_for(
            E2BRolloutRuntime(
                _config(), gateway=_gateway(), trace_path=tmp_path / "trace.json.gz"
            ).run(RolloutInput(rollout_id="rollout-1", task={}), agent),
            timeout=5,  # Must finish well before the 30-second Retry-After delay.
        )
        assert result.status == "success"
        assert result.termination_reason == "context_length_exceeded"
        assert result.segments[0].tokens == [1, 2, 3]
        assert result.metrics["model_requests_cancelled"] == 1
        assert result.metrics["model_request_inflight"] == 0
        assert retry_cancelled.is_set()
        assert len(retry_responses) == 1
        assert retry_responses[0].is_closed
        assert clients[0].is_closed
        assert _FakeUvicornServer.instances[-1].should_exit

    asyncio.run(scenario())


def test_evaluation_failure_preserves_context_stop_reason(tmp_path, monkeypatch):
    import json

    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    stage = _read_artifact_json(options["stage_path"])
    stage["termination_reason"] = "context_length_exceeded"
    stage["length_details"] = {"prompt_tokens": 262144, "max_sequence_length": 262144}
    _write_artifact_json(options["stage_path"], stage)

    async def unavailable_evaluator(*args):
        raise ConnectionError("evaluator unavailable")

    adapter.evaluate = unavailable_evaluator
    with pytest.raises(ConnectionError, match="evaluator unavailable"):
        asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    result = _read_artifact_json(options["output_path"])
    state = json.loads(_state_path(options["output_path"]).read_text())
    assert result["status"] == "env_error"
    for artifact in (result, state):
        assert artifact["termination_reason"] == "context_length_exceeded"
        assert artifact["length_details"] == stage["length_details"]


@pytest.mark.parametrize("wait_for_cancellation", [False, True])
def test_context_stop_does_not_hide_agent_cleanup_failure(
    tmp_path, monkeypatch, wait_for_cancellation
):
    import json

    from northrl.exceptions import ContextLengthExceededError
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    original_agent = adapter.run_agent

    async def unsafe_agent(*args):
        await original_agent(*args)
        try:
            error = ContextLengthExceededError(262144, 8192, 262144)
            observer = client_instances[-1].on_service_call
            observer("sample_from_messages", error)
            if wait_for_cancellation:
                await asyncio.Event().wait()
        finally:
            raise RolloutEnvironmentError("agent process cleanup failed")

    client_instances = []

    class ObservedClient(_FakeInferenceClient):
        def __init__(self, *args):
            super().__init__(*args)
            client_instances.append(self)

    monkeypatch.setattr(sys.modules[__name__], "_FakeInferenceClient", ObservedClient)
    adapter.run_agent = unsafe_agent
    with pytest.raises(Exception, match="agent process cleanup failed"):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    assert not options["stage_path"].exists()
    result = _read_artifact_json(options["output_path"])
    assert result["status"] == "env_error"
    assert _FakeRelay.instances[-1].closed


@pytest.mark.parametrize("environment_failure", [True, False])
@pytest.mark.parametrize("nested", [True, False])
def test_task_group_failure_preserves_environment_classification(
    tmp_path, monkeypatch, environment_failure, nested
):
    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.contracts import EvaluationResult

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    original_agent = adapter.run_agent

    async def grouped_agent_failure(*args):
        await original_agent(*args)  # A prior successful turn must not hide the failure.

        async def fail():
            if environment_failure:
                raise RolloutEnvironmentError("task service unavailable")
            raise RuntimeError("agent failed to solve the task")

        async def child():
            if nested:
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(fail())
            else:
                await fail()

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(child())

    adapter.run_agent = grouped_agent_failure
    if environment_failure:
        with pytest.raises(Exception):
            asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
        result = _read_artifact_json(options["output_path"])
        assert result["status"] == "env_error"
        assert result["segments"] == []
        assert not options["stage_path"].exists()
    else:
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
        stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
        assert stage.agent_result.exit_code == 1
        assert stage.trace.segments

        async def evaluate(_input_data, workspace, agent_result, _assets):
            assert (workspace / "solution.txt").read_text() == "agent answer"
            assert agent_result.exit_code == 1
            return EvaluationResult(reward=1.0)

        adapter.evaluate = evaluate
        asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
        result = _read_artifact_json(options["output_path"])
        assert result["status"] == "success"
        assert result["reward"] == 1.0
        assert len(result["segments"]) == 1
    assert _FakeRelay.instances[-1].closed
    assert _FakeUvicornServer.instances[-1].should_exit


def test_failure_status_checks_mixed_exception_groups_and_causal_cycles():
    from northrl.remote_rollout.e2b_runtime import _failure_status

    class TaskFailure(RuntimeError):
        rollout_status = "task_failed"

    environment_failure = RolloutEnvironmentError("task service unavailable")
    group = BaseExceptionGroup(
        "mixed failures",
        [asyncio.CancelledError(), ExceptionGroup("nested", [environment_failure])],
    )
    error = TaskFailure("agent failed")
    error.__cause__ = group
    environment_failure.__context__ = error
    assert _failure_status(error) == "env_error"


@pytest.mark.parametrize(
    "case",
    [
        "concurrent_same_payload",
        "concurrent_different_payload",
        "serial_retry",
        "serial_different_payload",
        "retry_with_new_concurrent_failure",
    ],
)
def test_real_service_observer_scopes_recovery_to_prior_matching_calls(tmp_path, monkeypatch, case):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    started: dict[int, asyncio.Event] = {}
    finish: dict[int, asyncio.Event] = {}
    calls: list[str] = []

    async def service_call(self, messages, **kwargs):
        del self, kwargs
        content = messages[-1]["content"]
        if content != "warmup":
            calls.append(content)
            index = len(calls)
            started[index].set()
            await finish[index].wait()
            if index == 1:
                raise ConnectionError("first provider request failed")
            if index == 3:
                raise ConnectionError("new concurrent provider request failed")
        return {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "northrl_train": {
                "prompt_tokens": [1, 2],
                "response_tokens": [3],
                "response_logprobs": [-0.1],
                "response_old_logprobs": [-0.2],
            },
        }

    monkeypatch.setattr(_FakeInferenceClient, "generate", service_call)

    async def agent(*_args):
        for index in (1, 2, 3):
            started[index] = asyncio.Event()
            finish[index] = asyncio.Event()
        app = _FakeUvicornConfig.instances[-1].app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as http:

            async def request(content):
                return await http.post(
                    "/v1/chat/completions",
                    json={
                        "model": "northrl",
                        "messages": [{"role": "user", "content": content}],
                    },
                )

            # Seed a real trace so env_error must discard otherwise usable segments.
            assert (await request("warmup")).status_code == 200
            first = asyncio.create_task(request("request-a"))
            pending = [first]
            try:
                await started[1].wait()
                if case.startswith("concurrent_"):
                    content = "request-a" if case == "concurrent_same_payload" else "request-b"
                    second = asyncio.create_task(request(content))
                    pending.append(second)
                    await started[2].wait()
                    finish[1].set()
                    with pytest.raises(ConnectionError, match="first provider request failed"):
                        await first
                    finish[2].set()
                    assert (await second).status_code == 200
                else:
                    finish[1].set()
                    with pytest.raises(ConnectionError, match="first provider request failed"):
                        await first
                    content = "request-b" if case == "serial_different_payload" else "request-a"
                    second = asyncio.create_task(request(content))
                    pending.append(second)
                    await started[2].wait()
                    if case == "retry_with_new_concurrent_failure":
                        # This failure begins after the retry starts. The retry can
                        # recover call 1, but must leave call 3's identical payload alone.
                        third = asyncio.create_task(request("request-a"))
                        pending.append(third)
                        await started[3].wait()
                        finish[3].set()
                        with pytest.raises(
                            ConnectionError, match="new concurrent provider request failed"
                        ):
                            await third
                    finish[2].set()
                    assert (await second).status_code == 200
            finally:
                for task in pending:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        raise RuntimeError("agent failed after provider calls")

    adapter.run_agent = agent

    async def agent_phase():
        await asyncio.wait_for(e2b_runtime._run_phase(phase="agent", **options), timeout=3)

    unrecovered = case != "serial_retry"
    if unrecovered:
        with pytest.raises(RolloutEnvironmentError):
            asyncio.run(agent_phase())
        result = _load_dense_result(options["output_path"].read_bytes())
        assert result.status == "env_error"
        assert result.segments == []
        assert result.reward == 0.0
        assert not options["stage_path"].exists()
    else:
        from northrl.remote_rollout.contracts import EvaluationResult

        asyncio.run(agent_phase())
        stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
        assert stage.agent_result.exit_code == 1

        async def evaluate(_input_data, _workspace, agent_result, _assets):
            assert agent_result.exit_code == 1
            return EvaluationResult(reward=1.0)

        adapter.evaluate = evaluate
        asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
        result = _load_dense_result(options["output_path"].read_bytes())
        assert result.status == "success"
        assert result.segments
        assert result.reward == 1.0
    trace = _load_dense_trace(options["trace_path"].read_bytes())
    assert trace.segments
    if unrecovered:
        expected_failure = (
            "new concurrent provider request failed"
            if case == "retry_with_new_concurrent_failure"
            else "first provider request failed"
        )
        assert expected_failure in result.error_message
    assert _FakeRelay.instances[-1].closed
    assert _FakeUvicornServer.instances[-1].should_exit


@pytest.mark.parametrize("failure_before_telemetry", [False, True])
def test_evaluation_failure_preserves_metrics_without_trace(
    tmp_path, monkeypatch, failure_before_telemetry
):
    import json

    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    stage = _read_artifact_json(options["stage_path"])
    stage["trace"] = None
    stage["termination_reason"] = "context_length_exceeded"
    stage["metrics"] = {"model_turns": 0.0, "tool_results_observed": 2.0}
    _write_artifact_json(options["stage_path"], stage)
    options["trace_path"].unlink()

    async def unavailable_evaluator(*_args):
        raise ConnectionError("evaluator unavailable")

    adapter.evaluate = unavailable_evaluator
    if failure_before_telemetry:
        options["evaluation_assets"] = tmp_path / "absent-assets"
    with pytest.raises(ValueError if failure_before_telemetry else ConnectionError):
        asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    result = _load_dense_result(options["output_path"].read_bytes())
    assert result.status == "env_error"
    assert result.segments == []
    assert result.metrics["model_turns"] == 0.0
    assert result.metrics["tool_results_observed"] == 2.0
    assert result.termination_reason == "context_length_exceeded"


@pytest.mark.parametrize("record_turn", [False, True])
def test_invalid_gateway_training_data_keeps_diagnostics_but_discards_training(
    tmp_path, monkeypatch, record_turn
):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    original_agent = adapter.run_agent
    original_generate = _FakeInferenceClient.generate

    async def invalid_metadata(self, messages, **kwargs):
        result = await original_generate(self, messages, **kwargs)
        if not record_turn or self.call_count > 1:
            result["northrl_train"]["response_logprobs"] = []
        return result

    async def failed_agent(*args):
        if record_turn:
            await original_agent(*args)
        app = _FakeUvicornConfig.instances[-1].app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as http:
            with pytest.raises(ValueError, match="must have equal length"):
                await http.post(
                    "/v1/messages",
                    json={"model": "northrl", "messages": [{"role": "user", "content": "next"}]},
                )
        raise RuntimeError("agent exited after invalid gateway data")

    monkeypatch.setattr(_FakeInferenceClient, "generate", invalid_metadata)
    adapter.run_agent = failed_agent
    with pytest.raises(RolloutEnvironmentError):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    result = _load_dense_result(options["output_path"].read_bytes())
    assert result.status == "env_error"
    assert result.segments == []
    assert result.metrics["model_requests_completed"] == 1 + record_turn
    assert result.metrics["model_requests_error"] == 1
    assert result.metrics["model_request_inflight"] == 0
    assert options["trace_path"].exists() is record_turn
    if record_turn:
        trace = _load_dense_trace(options["trace_path"].read_bytes())
        assert trace.segments
        assert trace.metrics["model_requests_error"] == 1


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_partial_native_stream_cannot_commit_when_agent_swallows_error(tmp_path, monkeypatch, api):
    from northrl.remote_rollout import e2b_runtime

    def upstream(request):
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "token_data": {
                        "prompt_tokens": [1, 2],
                        "response_tokens": [3],
                        "response_logprobs": [-0.1],
                        "response_old_logprobs": [-0.2],
                    },
                },
            )
        if api == "openai":
            event = {"choices": [{"index": 0, "delta": {"content": "partial"}}]}
        else:
            event = {"type": "message_start", "message": {"role": "assistant", "content": []}}
        # Native content arrived, but the training metadata and terminal event did not.
        return httpx.Response(
            200,
            content=f"data: {json.dumps(event)}\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    def relay(config, gateway, **kwargs):
        return NorthGateRelay(
            config,
            gateway,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
            **kwargs,
        )

    monkeypatch.setattr(e2b_runtime, "NorthGateRelay", relay)

    async def agent(_input, _env):
        app = _FakeUvicornConfig.instances[-1].app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as http:
            payload = {"model": "northrl", "messages": [{"role": "user", "content": "next"}]}
            assert (await http.post("/v1/chat/completions", json=payload)).status_code == 200
            path = "/v1/messages" if api == "anthropic" else "/v1/chat/completions"
            with pytest.raises(Exception, match="before its terminal event"):
                await http.post(path, json={**payload, "stream": True})
        return 1.0, {}

    trace_path = tmp_path / "trace.json.gz"
    with pytest.raises(RolloutEnvironmentError, match="before its terminal event"):
        asyncio.run(
            E2BRolloutRuntime(_config(), gateway=_gateway(), trace_path=trace_path).run(
                RolloutInput(rollout_id="rollout-1", task={}), agent
            )
        )
    trace = _load_dense_trace(trace_path.read_bytes())
    assert trace.segments[-1].tokens == [1, 2, 3]
    assert trace.metrics["model_requests_error"] == 1


def test_failure_metrics_error_does_not_replace_environment_failure(tmp_path, monkeypatch):
    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.telemetry import RolloutTelemetry

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    original_agent = adapter.run_agent
    original_snapshot = RolloutTelemetry.metrics_snapshot

    def snapshot(telemetry):
        if telemetry._phase == "failed":
            raise RuntimeError("metrics snapshot unavailable")
        return original_snapshot(telemetry)

    async def failed_agent(*args):
        await original_agent(*args)
        raise RolloutEnvironmentError("original environment failure")

    monkeypatch.setattr(RolloutTelemetry, "metrics_snapshot", snapshot)
    adapter.run_agent = failed_agent
    with pytest.raises(RolloutEnvironmentError, match="original environment failure"):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    result = _load_dense_result(options["output_path"].read_bytes())
    assert result.status == "env_error"
    assert result.error_message == "original environment failure"
    assert result.metrics["model_turns"] == 1.0


@pytest.mark.parametrize(
    ("log_level", "show_info", "show_profile"),
    [
        (None, True, False),
        ("debug", True, True),
        ("invalid", True, False),
        ("WARNING", False, False),
    ],
)
def test_runtime_cli_honors_log_level_before_running_phase(
    monkeypatch, caplog, tmp_path, log_level, show_info, show_profile
):
    from northrl.remote_rollout import e2b_runtime
    from northrl.utils.profiling import profile_event

    if log_level is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", log_level)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2b_runtime",
            "--phase",
            "agent",
            "--input",
            str(tmp_path / "input.json"),
            "--output",
            str(tmp_path / "result.json.gz"),
            "--trace-output",
            str(tmp_path / "trace.json.gz"),
            "--stage-path",
            str(tmp_path / "stage.json.gz"),
            "--evaluation-assets",
            str(tmp_path / "evaluation"),
            "--entrypoint",
            "test-adapter",
        ],
    )
    phases = []

    async def run_phase(**kwargs):
        phases.append(kwargs["phase"])
        e2b_runtime.logger.info("runtime test lifecycle")
        profile_event("runtime.test", "completed", duration_seconds=1.0)

    monkeypatch.setattr(e2b_runtime, "_run_phase", run_phase)
    # Existing handlers must not prevent the CLI from applying LOG_LEVEL.
    # The context also restores the process-wide level after main changes it.
    with caplog.at_level(logging.DEBUG):
        logging.getLogger().setLevel(logging.INFO)
        e2b_runtime.main()

    assert phases == ["agent"]
    assert ("runtime test lifecycle" in caplog.messages) is show_info
    assert (
        "[PROFILE] stage=runtime.test status=completed duration_seconds=1.000" in caplog.messages
    ) is show_profile


@pytest.mark.parametrize("cancel_during_shutdown", [False, True])
@pytest.mark.parametrize("context_stopped", [False, True])
def test_requests_finishing_during_shutdown_survive_evaluation_checkpoint(
    tmp_path, monkeypatch, cancel_during_shutdown, context_stopped
):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()
    pending = {}
    original_generate = _FakeInferenceClient.generate
    original_agent = adapter.run_agent
    original_serve = _FakeUvicornServer.serve

    async def delayed_generate(self, *args, **kwargs):
        if self.call_count == 1:
            entered.set()
            await release.wait()
        return await original_generate(self, *args, **kwargs)

    async def pending_request():
        app = _FakeUvicornConfig.instances[-1].app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as http:
            response = await http.post(
                "/v1/messages",
                json={"model": "northrl", "messages": [{"role": "user", "content": "continue"}]},
            )
            assert response.status_code == 200

    async def run_agent(*args):
        result = await original_agent(*args)
        # A disconnected agent may leave a gateway request running while it exits.
        pending["task"] = asyncio.create_task(pending_request())
        await entered.wait()
        if context_stopped:
            from northrl.exceptions import ContextLengthExceededError

            _FakeRelay.instances[-1]._on_service_call("limit", ContextLengthExceededError())
        return result

    async def serve(self):
        await original_serve(self)
        # Drain or cancel that request only after runtime.run starts gateway shutdown.
        if cancel_during_shutdown:
            pending["task"].cancel()
        else:
            release.set()
        outcomes = await asyncio.gather(pending["task"], return_exceptions=True)
        if cancel_during_shutdown:
            assert isinstance(outcomes[0], asyncio.CancelledError)
        else:
            assert outcomes == [None]

    monkeypatch.setattr(_FakeInferenceClient, "generate", delayed_generate)
    monkeypatch.setattr(_FakeUvicornServer, "serve", serve)
    adapter.run_agent = run_agent
    # Cancelling a healthy request the agent abandoned is not a model-service
    # failure, with or without a context stop; the agent stage still completes.
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    metrics_path = options["output_path"].with_name("metrics.json")
    agent_metrics = json.loads(metrics_path.read_text())["metrics"]
    stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
    assert stage.trace is not None
    assert stage.termination_reason == ("context_length_exceeded" if context_stopped else None)
    assert stage.trace.segments[-1].tokens == (
        [1, 2, 3] if cancel_during_shutdown else [1, 2, 3, 4, 5]
    )
    assert agent_metrics["model_requests_completed"] == 2
    assert stage.trace.metrics["model_requests_completed"] == 2
    assert stage.trace.metrics["model_request_inflight"] == 0
    assert stage.trace.metrics["model_requests_cancelled"] == int(cancel_during_shutdown)
    assert stage.trace.metrics["model_requests_success"] == 2 - int(cancel_during_shutdown)
    assert (
        stage.trace.metrics["model_request_seconds_sum"]
        == agent_metrics["model_request_seconds_sum"]
    )
    assert stage.trace.metrics["model_request_seconds_sum"] > 0
    (options["evaluation_assets"] / "test.sh").write_text("hidden evaluation")
    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    evaluation_metrics = json.loads(metrics_path.read_text())["metrics"]
    for name in (
        "model_requests_started",
        "model_requests_completed",
        "model_requests_success",
        "model_requests_cancelled",
        "model_request_seconds_sum",
        "model_request_seconds_bucket_le_inf",
        "model_request_inflight",
        "model_request_oldest_inflight_seconds",
    ):
        assert evaluation_metrics[name] == agent_metrics[name]


@pytest.mark.parametrize("record_turn", [False, True])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_agent_budget_stops_agent_cleans_up_and_evaluates_workspace(
    tmp_path, monkeypatch, record_turn, reward
):
    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.contracts import EvaluationResult

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch)
    original_agent = adapter.run_agent

    async def agent(*args):
        if record_turn:
            await original_agent(*args)
        else:
            (args[1] / "solution.txt").write_text("agent answer")
            events.append("agent")
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            events.append("agent-cleaned")

    async def evaluate(_input, workspace, agent_result, _assets):
        assert events[-1] == "agent-cleaned"
        assert agent_result.exit_code == 124
        assert (workspace / "solution.txt").read_text() == "agent answer"
        assert _FakeRelay.instances[-1].closed
        events.append("evaluate")
        return EvaluationResult(reward=reward, metrics={"swebench_resolved": reward})

    adapter.run_agent = agent
    adapter.evaluate = evaluate
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))

    stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
    assert stage.agent_result.exit_code == 124
    assert stage.termination_reason is None
    assert (stage.trace is not None) is record_turn
    metrics = stage.trace.metrics if record_turn else stage.metrics
    assert metrics["agent_exit_code"] == 124.0
    assert metrics["e2b_timed_out"] == 1.0
    assert metrics["agent_execution_seconds"] > 0
    assert metrics["agent_finalize_seconds"] >= 0
    assert events == ["prepare", "agent", "agent-cleaned"]
    assert not options["output_path"].exists()

    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    result = _load_dense_result(options["output_path"].read_bytes())
    assert result.status == "success"
    assert result.reward == reward
    assert result.metrics["swebench_resolved"] == reward
    assert result.metrics["e2b_timed_out"] == 1.0
    assert events == ["prepare", "agent", "agent-cleaned", "evaluate"]


def test_preparation_does_not_consume_agent_budget(tmp_path, monkeypatch):
    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch, agent_seconds=0.1)
    original_prepare = adapter.prepare
    original_agent = adapter.run_agent

    async def prepare(*args):
        await original_prepare(*args)
        await asyncio.sleep(0.15)

    async def agent(*args):
        await asyncio.sleep(0.02)
        return await original_agent(*args)

    adapter.prepare = prepare
    adapter.run_agent = agent
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    stage = _load_dense_agent_stage(options["stage_path"].read_bytes())
    assert stage.agent_result.exit_code == 0
    assert "e2b_timed_out" not in stage.trace.metrics
    assert events == ["prepare", "agent"]


@pytest.mark.parametrize("phase", ["resolve", "prepare", "server"])
def test_preparation_budget_expires_before_agent_execution(tmp_path, monkeypatch, phase):
    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch, preparation_seconds=0.02)

    async def hang(*_args):
        await asyncio.Event().wait()

    if phase == "resolve":
        adapter.resolve_workspace = hang
    elif phase == "prepare":
        adapter.prepare = hang
    else:

        async def never_start(server):
            while not server.should_exit:
                await asyncio.sleep(0.001)

        monkeypatch.setattr(_FakeUvicornServer, "serve", never_start)
    with pytest.raises(TimeoutError):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    assert "agent" not in events
    assert not options["stage_path"].exists()
    assert _read_artifact_json(options["output_path"])["status"] == "env_error"


@pytest.mark.parametrize("cleanup_failure", ["agent", "relay", "deadline"])
def test_agent_timeout_cannot_commit_an_unsafe_cleanup(tmp_path, monkeypatch, cleanup_failure):
    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch, finalize_seconds=0.03)
    original_agent = adapter.run_agent

    async def agent(*args):
        await original_agent(*args)
        try:
            await asyncio.Event().wait()
        finally:
            events.append("cleanup-started")
            if cleanup_failure == "agent":
                raise RuntimeError("agent cleanup did not succeed")
            if cleanup_failure == "deadline":
                await asyncio.Event().wait()

    async def fail_close(_relay):
        raise RuntimeError("relay cleanup did not succeed")

    adapter.run_agent = agent
    if cleanup_failure == "relay":
        monkeypatch.setattr(_FakeRelay, "close", fail_close)
    with pytest.raises(RolloutEnvironmentError):
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    assert not options["stage_path"].exists()
    assert "cleanup-started" in events
    assert _read_artifact_json(options["output_path"])["status"] == "env_error"


@pytest.mark.parametrize("real_service_error", [False, True])
def test_agent_budget_ignores_its_cancellation_errors_but_preserves_service_failures(
    tmp_path, monkeypatch, real_service_error
):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch)
    original_agent = adapter.run_agent

    async def agent(*args):
        await original_agent(*args)
        relay = _FakeRelay.instances[-1]
        if real_service_error:
            relay._on_service_call("request", RuntimeError("genuine model service failure"))
        try:
            await asyncio.Event().wait()
        finally:
            cancelled_error = RuntimeError("request cancelled during shutdown")
            cancelled_error.__cause__ = asyncio.CancelledError()
            relay._on_service_call("request", cancelled_error)

    adapter.run_agent = agent
    if real_service_error:
        with pytest.raises(RolloutEnvironmentError, match="genuine model service failure"):
            asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
        assert not options["stage_path"].exists()
    else:
        asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
        assert options["stage_path"].exists()


@pytest.mark.parametrize("timeout_override", [None, 0.02, 1.0])
def test_evaluation_phase_deadline_cancels_hook_and_reports_environment_error(
    tmp_path, monkeypatch, timeout_override
):
    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    _short_execution_budgets(monkeypatch, evaluation_seconds=0.02)
    stage_bytes = options["stage_path"].read_bytes()

    async def evaluate(*_args):
        try:
            await asyncio.Event().wait()
        finally:
            events.append("evaluation-cleaned")

    adapter.evaluate = evaluate

    async def evaluate_phase():
        await asyncio.wait_for(
            e2b_runtime._run_phase(
                phase="evaluate", phase_timeout_seconds=timeout_override, **options
            ),
            timeout=0.5,
        )

    with pytest.raises(TimeoutError):
        asyncio.run(evaluate_phase())
    assert events[-1] == "evaluation-cleaned"
    result = _read_artifact_json(options["output_path"])
    assert result["status"] == "env_error"
    assert result["metrics"]["e2b_timed_out"] == 1.0
    assert result["segments"] == []
    assert options["stage_path"].read_bytes() == stage_bytes


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
@pytest.mark.parametrize(
    ("phase", "keyword"),
    [("evaluate", "phase_timeout_seconds"), ("agent", "preparation_timeout_seconds")],
)
def test_phase_timeout_override_requires_positive_finite_seconds(
    tmp_path, monkeypatch, seconds, phase, keyword
):
    from northrl.remote_rollout import e2b_runtime

    _adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="positive and finite"):
        asyncio.run(e2b_runtime._run_phase(phase=phase, **{keyword: seconds}, **options))
    assert events == []


@pytest.mark.parametrize(
    ("phase", "keyword"),
    [("agent", "phase_timeout_seconds"), ("evaluate", "preparation_timeout_seconds")],
)
def test_deadline_override_cannot_target_the_wrong_phase(tmp_path, monkeypatch, phase, keyword):
    from northrl.remote_rollout import e2b_runtime

    _adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="only for"):
        asyncio.run(e2b_runtime._run_phase(phase=phase, **{keyword: 1.0}, **options))
    assert events == []


def test_preparation_override_uses_time_remaining_after_host_input_upload(tmp_path, monkeypatch):
    from northrl.remote_rollout import e2b_runtime

    adapter, events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch, preparation_seconds=2.0)

    async def prepare(*_args):
        await asyncio.Event().wait()

    adapter.prepare = prepare

    async def agent_phase():
        await asyncio.wait_for(
            e2b_runtime._run_phase(phase="agent", preparation_timeout_seconds=0.02, **options),
            timeout=0.5,
        )

    with pytest.raises(TimeoutError):
        asyncio.run(agent_phase())
    # The internal preparation deadline reports a trusted environment failure;
    # an external wait_for cancellation would not publish this result at all.
    assert _read_artifact_json(options["output_path"])["status"] == "env_error"
    assert not options["stage_path"].exists()
    assert "agent" not in events


def test_repeated_external_cancellation_drains_agent_without_committing_stage(
    tmp_path, monkeypatch
):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch, agent_seconds=2.0)
    original_agent = adapter.run_agent

    async def scenario():
        ready = asyncio.Event()
        cleaning = asyncio.Event()
        release = asyncio.Event()

        async def agent(*args):
            await original_agent(*args)
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()

        adapter.run_agent = agent
        task = asyncio.create_task(e2b_runtime._run_phase(phase="agent", **options))
        await ready.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _FakeRelay.instances[-1].closed
        assert _FakeUvicornServer.instances[-1].should_exit
        assert not options["stage_path"].exists()
        assert not options["output_path"].exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup_stage", ["agent", "server", "relay"])
@pytest.mark.parametrize("cleanup_failure", ["error", "timeout"])
def test_external_cancellation_survives_cleanup_failure(
    tmp_path, monkeypatch, cleanup_stage, cleanup_failure
):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    _short_execution_budgets(monkeypatch, agent_seconds=2.0, finalize_seconds=0.03)
    original_agent = adapter.run_agent
    original_serve = _FakeUvicornServer.serve

    async def fail_cleanup():
        if cleanup_failure == "error":
            raise RuntimeError("cleanup failed after external cancellation")
        await asyncio.Event().wait()

    async def serve(server):
        await original_serve(server)
        await fail_cleanup()

    async def close(_relay):
        await fail_cleanup()

    if cleanup_stage == "server":
        monkeypatch.setattr(_FakeUvicornServer, "serve", serve)
    elif cleanup_stage == "relay":
        monkeypatch.setattr(_FakeRelay, "close", close)

    async def scenario():
        ready = asyncio.Event()

        async def agent(*args):
            await original_agent(*args)
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                if cleanup_stage == "agent":
                    await fail_cleanup()

        adapter.run_agent = agent
        task = asyncio.create_task(e2b_runtime._run_phase(phase="agent", **options))
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        assert not options["stage_path"].exists()
        assert not options["output_path"].exists()
        # A partial trace is still useful, but cannot certify task completion.
        assert options["trace_path"].exists()

    asyncio.run(scenario())


def test_trace_callback_uses_compact_recorder_without_dense_history_copy(tmp_path, monkeypatch):
    from northrl.remote_rollout import TrajectoryRecorder

    recorder = TrajectoryRecorder()
    recorder.record_model_turn(
        prompt_tokens=[10, 11],
        response_tokens=[20],
        response_log_probs=[-0.1],
        response_old_log_probs=[-0.1],
    )

    def fail_dense_export():
        raise AssertionError("trace callbacks must not copy the dense history")

    monkeypatch.setattr(recorder, "to_remote_segments", fail_dense_export)
    path = tmp_path / "trace.json.gz"
    runtime = E2BRolloutRuntime(_config(), gateway=_gateway(), trace_path=path)
    runtime._trace_writer(RolloutInput(rollout_id="compact-trace", task={}))(recorder)
    asyncio.run(runtime._trace_sink.aclose())
    payload = _read_artifact_json(path)
    assert payload["schema_version"] == 2
    assert payload["segments"][0]["response_spans"] == [[2, 3]]
    assert "loss_mask" not in payload["segments"][0]
    assert "old_log_probs" not in payload["segments"][0]
    assert gzip.decompress(path.read_bytes()).decode() == json.dumps(payload, separators=(",", ":"))
    assert _load_dense_trace(path.read_bytes()).segments[0].tokens == [10, 11, 20]


@pytest.mark.parametrize("version", [None, 1])
def test_evaluation_rejects_non_v2_stage_and_writes_v2_failure(tmp_path, monkeypatch, version):
    from northrl.remote_rollout import e2b_runtime

    _adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    payload = _read_artifact_json(options["stage_path"])
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version
    _write_artifact_json(options["stage_path"], payload)
    with pytest.raises(ValueError):
        asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    failure_payload = _read_artifact_json(options["output_path"])
    assert failure_payload["schema_version"] == 2
    failure = _load_dense_result(options["output_path"].read_bytes())
    assert failure.status == "env_error"
    assert failure.segments == []


def test_agent_commit_waits_for_background_trace_and_cannot_be_overwritten(tmp_path, monkeypatch):
    from northrl.remote_rollout import e2b_runtime

    _adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    write = e2b_runtime.write_artifact
    first_trace_started = threading.Event()
    release_first_trace = threading.Event()
    writes = []
    loop_thread = threading.get_ident()

    def delayed_write(path, model):
        assert threading.get_ident() != loop_thread
        if path == options["trace_path"] and not first_trace_started.is_set():
            first_trace_started.set()
            assert release_first_trace.wait(3)
        write(path, model)
        writes.append(path)

    monkeypatch.setattr(e2b_runtime, "write_artifact", delayed_write)

    async def scenario():
        task = asyncio.create_task(e2b_runtime._run_phase(phase="agent", **options))
        try:
            async with asyncio.timeout(3):
                while not first_trace_started.is_set():
                    await asyncio.sleep(0.001)
            await asyncio.sleep(0.02)
            assert not task.done()
            assert not options["stage_path"].exists()
        finally:
            release_first_trace.set()
        await task
        stage = load_agent_stage_gzip(options["stage_path"].read_bytes())
        trace = load_trace_gzip(options["trace_path"].read_bytes())
        assert trace == stage.trace
        assert "agent_finalize_seconds" in trace.metrics
        assert writes[-2:] == [options["trace_path"], options["stage_path"]]
        assert not any(thread.name == "northrl-artifact-writer" for thread in threading.enumerate())

    asyncio.run(scenario())


def test_cancelled_runtime_drains_slow_trace_without_committing_stage(tmp_path, monkeypatch):
    from northrl.remote_rollout import e2b_runtime

    adapter, _events, options = _phase_setup(tmp_path, monkeypatch)
    agent = adapter.run_agent
    write = e2b_runtime.write_artifact
    started, release = threading.Event(), threading.Event()

    def delayed_write(path, model):
        if path == options["trace_path"]:
            started.set()
            assert release.wait(3)
        write(path, model)

    monkeypatch.setattr(e2b_runtime, "write_artifact", delayed_write)

    async def scenario():
        ready = asyncio.Event()

        async def wait_after_turn(*args):
            await agent(*args)
            ready.set()
            await asyncio.Event().wait()

        adapter.run_agent = wait_after_turn
        task = asyncio.create_task(e2b_runtime._run_phase(phase="agent", **options))
        await ready.wait()
        task.cancel()
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            await asyncio.sleep(0.02)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert load_trace_gzip(options["trace_path"].read_bytes()).segments
        assert not options["stage_path"].exists()
        assert not options["output_path"].exists()
        assert not any(thread.name == "northrl-artifact-writer" for thread in threading.enumerate())

    asyncio.run(scenario())
