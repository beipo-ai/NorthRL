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
import logging
from unittest.mock import AsyncMock, Mock, call, create_autospec

import pytest

from northrl.exceptions import WeightVersionMismatchError
from northrl.utils.async_utils import flush_sdk_logs, run_coro
from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


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


class FakeSamplingClient:
    def __init__(self, weight_version: str = "default") -> None:
        self.weight_version = weight_version

    async def sample(self, **kwargs):
        del kwargs
        return {
            "sequences": [
                {
                    "tokens": [3],
                    "logprobs": [-0.1],
                    "text": "ok",
                    "weight_version": self.weight_version,
                    "stop_reason": "stop",
                }
            ]
        }


def _fake_service(sampling_client):
    service = Mock()
    service.connect = AsyncMock()
    service.create_model = AsyncMock()
    service.create_sampling_client = AsyncMock(return_value=sampling_client)
    service.aclose = AsyncMock()
    return service


def _holder_for_close(service_client):
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._closed = False
    holder._service_client = service_client
    holder._training_client = object()
    holder._sampling_client = object()
    return holder


def test_holder_close_emits_lifecycle_logs(caplog):
    service = _fake_service(FakeSamplingClient())
    holder = _holder_for_close(service)

    with caplog.at_level(logging.DEBUG, logger="northrl"):
        run_coro(holder.close())
        run_coro(holder.close())
        flush_sdk_logs()

    assert any(
        message.startswith("Closing Weaver async service client") for message in caplog.messages
    )
    assert "Weaver async service client close completed" in caplog.messages
    assert "WeaverServiceHolder close skipped: holder is already closed" in caplog.messages
    profile_output = caplog.text
    assert "stage=weaver.service-holder-close status=started" in profile_output
    assert "stage=weaver.service-holder-close status=completed" in profile_output
    assert (
        "stage=weaver.service-holder-close status=skipped reason='already_closed'" in profile_output
    )
    service.aclose.assert_awaited_once_with()


def test_holder_close_logs_service_client_failure(caplog):
    service = _fake_service(FakeSamplingClient())
    service.aclose.side_effect = RuntimeError("test close failure")
    holder = _holder_for_close(service)

    with caplog.at_level(logging.DEBUG, logger="northrl"):
        run_coro(holder.close())
        flush_sdk_logs()

    assert holder._closed is True
    failure_message = "Weaver async service client close failed; holder remains marked closed"
    assert failure_message in caplog.messages
    assert any(
        record.exc_info is not None
        for record in caplog.records
        if record.getMessage() == failure_message
    )
    assert "stage=weaver.service-holder-close status=failed" in caplog.text
    service.aclose.assert_awaited_once_with()


def test_holder_close_logs_when_service_client_is_missing(caplog):
    holder = _holder_for_close(None)

    with caplog.at_level(logging.DEBUG, logger="northrl"):
        run_coro(holder.close())
        flush_sdk_logs()

    assert (
        "WeaverServiceHolder close skipped: no async service client was provisioned"
        in caplog.messages
    )
    assert (
        "stage=weaver.service-holder-close status=skipped reason='no_service_client'" in caplog.text
    )


def test_holder_initializes_inference_only_sampling_client(
    monkeypatch,
    caplog,
):
    tokenizer = FakeTokenizer()
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", tokenizer_loader)
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")

    sampling_client = FakeSamplingClient(weight_version="actor-step-7")
    service = _fake_service(sampling_client)
    training_client = Mock()
    training_client.save_weights_for_sampler = AsyncMock()
    service.create_model.return_value = training_client
    service_factory = create_autospec(holder_module.AsyncServiceClient, return_value=service)
    monkeypatch.setattr(holder_module, "AsyncServiceClient", service_factory)

    holder = WeaverServiceHolder(
        base_model="actor-model",
        base_url="weaver.example",
        tokenizer_path="/models/tokenizer",
        model_path="weaver://weights/actor-step-7",
        performance_tier="flash",
    )

    assert run_coro(holder.get_current_sampling_version()) is None
    service.connect.assert_not_awaited()

    result = run_coro(
        holder.sample_from_messages(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
        )
    )
    sampling_version = run_coro(holder.get_current_sampling_version())
    run_coro(holder.close())
    run_coro(holder.close())
    flush_sdk_logs()

    assert result["response"] == "ok"
    assert result["prompt_tokens"] == [1, 2]
    assert result["weight_version"] == "actor-step-7"
    assert sampling_version is None
    assert "different weight version" not in caplog.text
    service_factory.assert_called_once_with(
        base_url="https://weaver.example",
        api_key="test-key",
        tensor_transport="http-binary",
        tensor_compression="zstd",
        metrics_store=holder_module.MetricsStoreConfig(),
        wandb_link=None,
        metric_observer=None,
    )
    service.connect.assert_awaited_once_with()
    service.create_model.assert_not_awaited()
    training_client.save_weights_for_sampler.assert_not_awaited()
    service.create_sampling_client.assert_awaited_once_with(
        model_path="weaver://weights/actor-step-7",
        base_model="actor-model",
        tokenizer_path="/models/tokenizer",
    )
    tokenizer_loader.assert_called_once_with(
        "/models/tokenizer",
        trust_remote_code=True,
    )
    service.aclose.assert_awaited_once_with()


def test_holder_forwards_scheduling_policy_to_compatible_sdk(monkeypatch):
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", Mock(return_value=FakeTokenizer())
    )
    service = _fake_service(FakeSamplingClient())

    def compatible_service_client(
        *,
        base_url,
        api_key,
        tensor_transport,
        tensor_compression,
        scheduling_policy,
        metrics_store,
        wandb_link,
        metric_observer,
    ):
        del base_url, api_key, tensor_transport, tensor_compression, scheduling_policy
        del metrics_store, wandb_link, metric_observer
        return service

    service_factory = create_autospec(
        compatible_service_client, side_effect=compatible_service_client
    )
    monkeypatch.setattr(holder_module, "AsyncServiceClient", service_factory)
    policy = {"mode": "fixed", "clusters": ["cluster-a"]}
    holder = WeaverServiceHolder(
        base_model="actor-model",
        base_url="weaver.example",
        scheduling_policy=policy,
        sampling_only=True,
    )
    policy["clusters"].append("cluster-b")

    try:
        run_coro(holder._ensure_provisioned())  # pylint: disable=protected-access
        service_factory.assert_called_once_with(
            base_url="https://weaver.example",
            api_key="test-key",
            tensor_transport="http-binary",
            tensor_compression="zstd",
            metrics_store=holder_module.MetricsStoreConfig(),
            wandb_link=None,
            metric_observer=None,
            scheduling_policy={"mode": "fixed", "clusters": ["cluster-a"]},
        )
        service.connect.assert_awaited_once_with()
    finally:
        run_coro(holder.close())


def test_holder_rejects_configured_scheduling_policy_with_unsupported_sdk(monkeypatch):
    service = _fake_service(FakeSamplingClient())

    def legacy_service_client(
        *,
        base_url,
        api_key,
        tensor_transport,
        tensor_compression,
        metrics_store,
        wandb_link,
        metric_observer,
    ):
        del base_url, api_key, tensor_transport, tensor_compression
        del metrics_store, wandb_link, metric_observer
        return service

    service_factory = create_autospec(legacy_service_client, side_effect=legacy_service_client)
    monkeypatch.setattr(holder_module, "AsyncServiceClient", service_factory)
    holder = WeaverServiceHolder(
        base_model="actor-model",
        sampling_only=True,
        scheduling_policy={"mode": "fixed", "clusters": ["cluster-a"]},
    )

    try:
        with pytest.raises(TypeError, match="scheduling_policy"):
            run_coro(holder._ensure_provisioned())  # pylint: disable=protected-access
        service.connect.assert_not_awaited()
    finally:
        run_coro(holder.close())


@pytest.mark.parametrize("provisioning", ["inference", "training-local", "training-client"])
@pytest.mark.parametrize("chat_template", [None, "custom {{ messages[0]['content'] }}"])
def test_holder_renders_configured_chat_template_after_lazy_provisioning(
    monkeypatch, provisioning, chat_template
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "default": 1, "custom": 2, "hello": 3}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="default {{ messages[0]['content'] }}",
    )
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", tokenizer_loader)
    service = _fake_service(FakeSamplingClient())
    training_client = Mock(model_id="model-id")
    training_client.get_tokenizer.return_value = tokenizer
    service.create_model.return_value = training_client
    monkeypatch.setattr(holder_module, "AsyncServiceClient", Mock(return_value=service))

    holder = WeaverServiceHolder(
        base_model="actor-model",
        tokenizer_path=None if provisioning == "training-client" else "/models/tokenizer",
        model_path="weaver://weights/actor" if provisioning == "inference" else None,
        need_sampling=False,
        chat_template=chat_template,
    )
    assert holder._tokenizer is None
    service.connect.assert_not_awaited()

    async def scenario():
        try:
            messages = [{"role": "user", "content": "hello"}]
            expected_prefix = "custom" if chat_template is not None else "default"
            assert await holder.apply_chat_template(messages) == f"{expected_prefix} hello"
            assert await holder.apply_chat_template(messages, tokenize=True) == [
                2 if chat_template is not None else 1,
                3,
            ]
        finally:
            await holder.close()

    asyncio.run(scenario())
    if provisioning == "training-client":
        training_client.get_tokenizer.assert_called_once_with()
        tokenizer_loader.assert_not_called()
    else:
        tokenizer_loader.assert_called_once_with("/models/tokenizer", trust_remote_code=True)
    if provisioning == "inference":
        service.create_sampling_client.assert_awaited_once_with(
            model_path="weaver://weights/actor",
            base_model="actor-model",
            tokenizer_path="/models/tokenizer",
        )


def test_training_model_pins_max_sequence_length(monkeypatch) -> None:
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")
    service = _fake_service(FakeSamplingClient())
    training_client = Mock()
    training_client.get_tokenizer.return_value = FakeTokenizer()
    service.create_model.return_value = training_client
    monkeypatch.setattr(holder_module, "AsyncServiceClient", Mock(return_value=service))
    holder = WeaverServiceHolder(
        base_model="actor-model",
        base_url="weaver.example",
        need_sampling=False,
        training_max_sequence_length=8192,
    )

    run_coro(holder._ensure_provisioned())  # pylint: disable=protected-access

    service.create_model.assert_awaited_once_with(
        base_model="actor-model",
        training_mode="lora",
        lora_config={"rank": 32},
        training_max_sequence_length=8192,
    )
    run_coro(holder.close())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_path": ""}, "model_path must not be empty"),
        (
            {"model_path": "weaver://weights/actor"},
            "tokenizer_path is required when loading a sampling model path",
        ),
    ],
)
def test_inference_only_target_must_be_complete(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WeaverServiceHolder(**kwargs)


@pytest.mark.parametrize(
    "holder_kwargs,create_model_kwargs",
    [
        ({}, {}),
        ({"performance_tier": None}, {}),
        ({"performance_tier": "normal"}, {"performance_tier": "normal"}),
        ({"performance_tier": "fast"}, {"performance_tier": "fast"}),
        ({"performance_tier": "flash"}, {"performance_tier": "flash"}),
    ],
    ids=["default", "null", "normal", "fast", "flash"],
)
def test_holder_provisions_training_and_sampling(monkeypatch, holder_kwargs, create_model_kwargs):
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")

    tokenizer = FakeTokenizer()
    training_client = Mock()
    training_client.model_id = "model-id"
    training_client.get_tokenizer.return_value = tokenizer
    training_client.save_weights_for_sampler = AsyncMock(return_value="weaver://initial")
    sampling_client = FakeSamplingClient()
    service = _fake_service(sampling_client)
    service.create_model.return_value = training_client
    service_factory = create_autospec(holder_module.AsyncServiceClient, return_value=service)
    monkeypatch.setattr(holder_module, "AsyncServiceClient", service_factory)

    holder = WeaverServiceHolder(
        base_model="actor-model", base_url="https://weaver.example", **holder_kwargs
    )
    assert run_coro(holder.get_current_sampling_version()) == "default"
    service.connect.assert_not_awaited()

    result = run_coro(holder.sample_from_prompt(prompt="hello", max_tokens=16))
    current_sampling_version = run_coro(holder.get_current_sampling_version())
    initial_snapshot = run_coro(holder.get_current_sampling_snapshot())
    training_client.save_weights_for_sampler.return_value = "weaver://next"
    sampling_path = run_coro(holder.save_weights_for_sampler("actor-step-1"))
    assert sampling_path == "weaver://next"
    snapshot_after_export = run_coro(holder.get_current_sampling_snapshot())
    assert run_coro(holder.get_current_sampling_version()) == "default"
    service.create_sampling_client.return_value = FakeSamplingClient(weight_version="actor-step-1")
    run_coro(holder.update_sampling_client(sampling_path, "actor-step-1"))
    updated_snapshot = run_coro(holder.get_current_sampling_snapshot())
    assert run_coro(holder.get_current_sampling_version()) == "actor-step-1"
    run_coro(holder.close())

    assert result["response"] == "ok"
    assert current_sampling_version == "default"
    assert initial_snapshot == ("weaver://initial", "default")
    assert snapshot_after_export == initial_snapshot
    assert updated_snapshot == ("weaver://next", "actor-step-1")
    service_factory.assert_called_once_with(
        base_url="https://weaver.example",
        api_key="test-key",
        tensor_transport="http-binary",
        tensor_compression="zstd",
        metrics_store=holder_module.MetricsStoreConfig(),
        wandb_link=None,
        metric_observer=None,
    )
    service.create_model.assert_awaited_once_with(
        base_model="actor-model",
        training_mode="lora",
        lora_config={"rank": 32},
        **create_model_kwargs,
    )
    assert training_client.save_weights_for_sampler.await_args_list == [
        call(name="default", ttl_seconds=86400),
        call(name="actor-step-1", ttl_seconds=86400),
    ]
    assert service.create_sampling_client.await_args_list == [
        call(model_path="weaver://initial", base_model="actor-model", model_id="model-id"),
        call(model_path="weaver://next", base_model="actor-model", model_id="model-id"),
    ]


def _initialized_holder(*, allow_weight_version_mismatch=False):
    holder = WeaverServiceHolder(
        base_model="actor-model",
        allow_weight_version_mismatch=allow_weight_version_mismatch,
    )
    sampling_client = FakeSamplingClient(weight_version="actor-step-1")
    service = _fake_service(sampling_client)
    training_client = Mock(model_id="model-id")
    training_client.save_weights_for_sampler = AsyncMock(
        side_effect=lambda name, ttl_seconds: f"weaver://{name}"
    )
    holder._service_client = service
    holder._training_client = training_client
    holder._sampling_client = sampling_client
    holder._tokenizer = FakeTokenizer()
    holder._active_sampling_snapshot = ("weaver://actor-step-1", "actor-step-1")
    return holder, service, training_client


def test_failed_sampling_client_update_preserves_active_and_retries_same_export():
    holder, service, training_client = _initialized_holder()
    previous_client = holder._sampling_client
    sampling_path = run_coro(holder.save_weights_for_sampler("actor-step-2"))
    service.create_sampling_client.side_effect = RuntimeError("test sampler creation failure")

    with pytest.raises(RuntimeError, match="test sampler creation failure"):
        run_coro(holder.update_sampling_client(sampling_path, "actor-step-2"))

    assert holder._sampling_client is previous_client
    assert run_coro(holder.get_current_sampling_snapshot()) == (
        "weaver://actor-step-1",
        "actor-step-1",
    )

    new_sampling_client = FakeSamplingClient(weight_version="actor-step-2")
    service.create_sampling_client.side_effect = None
    service.create_sampling_client.return_value = new_sampling_client
    run_coro(holder.update_sampling_client(sampling_path, "actor-step-2"))

    assert holder._sampling_client is new_sampling_client
    assert run_coro(holder.get_current_sampling_snapshot()) == (sampling_path, "actor-step-2")
    assert run_coro(holder.get_current_sampling_version()) == "actor-step-2"
    training_client.save_weights_for_sampler.assert_awaited_once_with(
        name="actor-step-2", ttl_seconds=86400
    )
    assert service.create_sampling_client.await_args_list == [
        call(model_path=sampling_path, base_model="actor-model", model_id="model-id"),
        call(model_path=sampling_path, base_model="actor-model", model_id="model-id"),
    ]
    run_coro(holder.close())


def test_failed_export_preserves_active_and_caller_can_use_previous_export():
    holder, service, training_client = _initialized_holder()
    previous_client = holder._sampling_client
    sampling_path = run_coro(holder.save_weights_for_sampler("actor-step-2"))
    training_client.save_weights_for_sampler.side_effect = RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        run_coro(holder.save_weights_for_sampler("actor-step-3"))

    assert holder._sampling_client is previous_client
    assert run_coro(holder.get_current_sampling_snapshot()) == (
        "weaver://actor-step-1",
        "actor-step-1",
    )
    service.create_sampling_client.assert_not_awaited()
    service.create_sampling_client.return_value = FakeSamplingClient("actor-step-2")
    run_coro(holder.update_sampling_client(sampling_path, "actor-step-2"))
    service.create_sampling_client.assert_awaited_once_with(
        model_path="weaver://actor-step-2", base_model="actor-model", model_id="model-id"
    )
    assert run_coro(holder.get_current_sampling_version()) == "actor-step-2"
    run_coro(holder.close())


def test_sampling_client_update_uses_explicit_weights_without_previous_export():
    holder, service, training_client = _initialized_holder()
    holder._sampling_client = None
    holder._active_sampling_snapshot = None
    service.create_sampling_client.return_value = FakeSamplingClient("actor-step-100")

    run_coro(holder.update_sampling_client("weaver://restored", "actor-step-100"))

    service.create_sampling_client.assert_awaited_once_with(
        model_path="weaver://restored", base_model="actor-model", model_id="model-id"
    )
    training_client.save_weights_for_sampler.assert_not_awaited()
    assert run_coro(holder.get_current_sampling_snapshot()) == (
        "weaver://restored",
        "actor-step-100",
    )
    run_coro(holder.close())


@pytest.mark.parametrize("export_timing", ["before-update", "during-update"])
def test_sampling_client_update_uses_explicit_weights_despite_another_export(export_timing):
    async def scenario():
        holder, service, _ = _initialized_holder()
        sampling_path = await holder.save_weights_for_sampler("actor-step-2")
        next_sampling_path = None
        if export_timing == "before-update":
            next_sampling_path = await holder.save_weights_for_sampler("actor-step-3")
        started = asyncio.Event()
        release = asyncio.Event()

        async def create_sampling_client(*, model_path, **kwargs):
            del kwargs
            started.set()
            await release.wait()
            return FakeSamplingClient(weight_version=model_path.removeprefix("weaver://"))

        service.create_sampling_client.side_effect = create_sampling_client
        update = asyncio.create_task(holder.update_sampling_client(sampling_path, "actor-step-2"))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            if export_timing == "during-update":
                next_sampling_path = await holder.save_weights_for_sampler("actor-step-3")
            assert await holder.get_current_sampling_version() == "actor-step-1"
            release.set()
            await asyncio.wait_for(update, timeout=5)

            assert await holder.get_current_sampling_snapshot() == (sampling_path, "actor-step-2")
            assert next_sampling_path is not None
            await holder.update_sampling_client(next_sampling_path, "actor-step-3")
            assert await holder.get_current_sampling_snapshot() == (
                next_sampling_path,
                "actor-step-3",
            )
            assert service.create_sampling_client.await_args_list == [
                call(model_path=sampling_path, base_model="actor-model", model_id="model-id"),
                call(model_path=next_sampling_path, base_model="actor-model", model_id="model-id"),
            ]
        finally:
            release.set()
            await asyncio.wait_for(update, timeout=5)
            await holder.close()

    run_coro(scenario())


def test_initial_sampling_retry_exports_weights_again():
    holder, service, training_client = _initialized_holder()
    holder._sampling_client = None
    holder._active_sampling_snapshot = None
    training_client.save_weights_for_sampler.side_effect = [
        "weaver://default-first",
        "weaver://default-second",
    ]
    service.create_sampling_client.side_effect = RuntimeError("first creation failed")

    with pytest.raises(RuntimeError, match="first creation failed"):
        run_coro(holder._ensure_provisioned(require_sampling=True))

    assert holder._sampling_client is None
    assert holder._active_sampling_snapshot is None
    service.create_sampling_client.side_effect = None
    service.create_sampling_client.return_value = FakeSamplingClient("default")
    run_coro(holder._ensure_provisioned(require_sampling=True))

    assert run_coro(holder.get_current_sampling_snapshot()) == (
        "weaver://default-second",
        "default",
    )
    assert training_client.save_weights_for_sampler.await_args_list == [
        call(name="default", ttl_seconds=86400),
        call(name="default", ttl_seconds=86400),
    ]
    assert service.create_sampling_client.await_args_list == [
        call(model_path="weaver://default-first", base_model="actor-model", model_id="model-id"),
        call(model_path="weaver://default-second", base_model="actor-model", model_id="model-id"),
    ]
    run_coro(holder.close())


def test_inflight_sampling_uses_its_original_version_after_client_update():
    async def scenario():
        holder, service, _ = _initialized_holder()
        previous_client = holder._sampling_client
        started = asyncio.Event()
        release = asyncio.Event()
        original_sample = previous_client.sample

        async def sample(**kwargs):
            started.set()
            await release.wait()
            return await original_sample(**kwargs)

        previous_client.sample = sample
        request = asyncio.create_task(holder.sample_from_prompt("hello", max_tokens=16))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            sampling_path = await holder.save_weights_for_sampler("actor-step-2")
            service.create_sampling_client.return_value = FakeSamplingClient("actor-step-2")
            await holder.update_sampling_client(sampling_path, "actor-step-2")
            assert await holder.get_current_sampling_version() == "actor-step-2"
            release.set()
            result = await asyncio.wait_for(request, timeout=5)
            assert result["weight_version"] == "actor-step-1"
        finally:
            release.set()
            await asyncio.wait_for(request, timeout=5)
            await holder.close()

    run_coro(scenario())


def test_warmup_rejects_wrong_sampler_weight_version():
    holder, _, _ = _initialized_holder()
    holder._sampling_client.weight_version = "wrong-version"

    with pytest.raises(WeightVersionMismatchError, match="expected='actor-step-1'.*wrong-version"):
        run_coro(holder.warmup_sampling())

    run_coro(holder.close())


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("sample_from_messages", {"messages": [{"role": "user", "content": "hello"}]}),
        ("sample_from_prompt", {"prompt": "hello"}),
        ("sample_from_token_ids", {"input_ids": [1, 2]}),
    ],
)
def test_allowed_sampling_version_mismatch_warns_and_preserves_observed_version(
    method_name, kwargs, caplog
):
    holder, _, _ = _initialized_holder(allow_weight_version_mismatch=True)
    holder._sampling_client.weight_version = "actor-step-2"

    try:
        with caplog.at_level(logging.WARNING, logger=holder_module.__name__):
            result = run_coro(getattr(holder, method_name)(max_tokens=16, **kwargs))
            flush_sdk_logs()

        assert result["weight_version"] == "actor-step-2"
        assert result["response"] == "ok"
        assert run_coro(holder.get_current_sampling_version()) == "actor-step-1"
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "expected='actor-step-1'" in warnings[0].getMessage()
        assert "observed='actor-step-2'" in warnings[0].getMessage()
        assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    finally:
        run_coro(holder.close())


@pytest.mark.parametrize(
    ("version_fields", "message"),
    [
        ({}, "Weaver sequence is missing weight_version"),
        ({"weight_version": ""}, "Weaver sequence weight_version must be non-empty"),
    ],
)
def test_allowed_sampling_version_mismatch_still_requires_valid_version(version_fields, message):
    holder, _, _ = _initialized_holder(allow_weight_version_mismatch=True)
    sample_result = run_coro(holder._sampling_client.sample())
    sequence = sample_result["sequences"][0]
    sequence.pop("weight_version")
    sequence.update(version_fields)
    holder._sampling_client.sample = AsyncMock(return_value=sample_result)

    try:
        with pytest.raises(ValueError, match=message):
            run_coro(holder.sample_from_prompt("hello", max_tokens=16))
    finally:
        run_coro(holder.close())


@pytest.mark.parametrize("enabled", [True, False])
def test_holder_passes_independent_storage_and_observer(monkeypatch, tmp_path, enabled):
    service = _fake_service(FakeSamplingClient())
    service_factory = create_autospec(holder_module.AsyncServiceClient, return_value=service)
    monkeypatch.setattr(holder_module, "AsyncServiceClient", service_factory)
    observer = Mock()
    holder = WeaverServiceHolder(
        base_model="actor-model",
        sampling_only=True,
        metrics_store={"enabled": enabled, "path": str(tmp_path)},
        metric_observer=observer,
    )
    holder._tokenizer = FakeTokenizer()
    try:
        run_coro(holder._ensure_provisioned())
        kwargs = service_factory.call_args.kwargs
        assert kwargs["metrics_store"] == holder_module.MetricsStoreConfig(enabled, str(tmp_path))
        assert kwargs["metric_observer"] is observer
        assert kwargs["wandb_link"] is None
    finally:
        run_coro(holder.close())
