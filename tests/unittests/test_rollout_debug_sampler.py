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
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


@pytest.fixture
def sampling_setup(monkeypatch):
    tokenizer = Mock()
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", tokenizer_loader)
    sampler = Mock()
    sampler.sample = AsyncMock(
        return_value={
            "sequences": [
                {
                    "tokens": [51],
                    "logprobs": [-0.4],
                    "text": "answer",
                    "weight_version": "server-version",
                    "stop_reason": "stop",
                }
            ]
        }
    )
    service = Mock(
        connect=AsyncMock(),
        aclose=AsyncMock(),
        create_model=AsyncMock(),
        create_sampling_client=AsyncMock(return_value=sampler),
    )
    monkeypatch.setattr(holder_module, "AsyncServiceClient", Mock(return_value=service))
    return tokenizer, tokenizer_loader, service


@pytest.mark.parametrize("tokenizer_path", [None, "/explicit/tokenizer"])
def test_sampling_only_base_model_never_provisions_training(sampling_setup, tokenizer_path):
    tokenizer, tokenizer_loader, service = sampling_setup
    holder = WeaverServiceHolder(
        base_model="base-model", tokenizer_path=tokenizer_path, sampling_only=True
    )

    async def scenario():
        assert await holder.get_current_sampling_version() is None
        assert await holder.get_tokenizer() is tokenizer
        result = await holder.sample_from_token_ids([11, 12], max_tokens=10)
        assert result["weight_version"] == "server-version"
        assert await holder.get_current_sampling_version() is None
        with pytest.raises(RuntimeError, match="no model path and version"):
            await holder.get_current_sampling_snapshot()
        await holder.close()
        await holder.close()
        with pytest.raises(RuntimeError, match="closed"):
            await holder.get_tokenizer()

    asyncio.run(scenario())
    assert holder._training_client is None
    service.create_model.assert_not_awaited()
    service.create_sampling_client.assert_awaited_once_with(
        base_model="base-model", tokenizer_path=tokenizer_path or "base-model"
    )
    tokenizer_loader.assert_called_once_with(tokenizer_path or "base-model", trust_remote_code=True)
    service.aclose.assert_awaited_once_with()


@pytest.mark.parametrize("sampling_only", [False, True])
def test_checkpoint_keeps_explicit_model_and_tokenizer(sampling_setup, sampling_only):
    _, tokenizer_loader, service = sampling_setup
    holder = WeaverServiceHolder(
        base_model="base-model",
        tokenizer_path="/checkpoint/tokenizer",
        model_path="weaver://actor-step-5",
        sampling_only=sampling_only,
    )

    async def scenario():
        result = await holder.sample_from_token_ids([1], max_tokens=10)
        assert result["weight_version"] == "server-version"
        assert await holder.get_current_sampling_version() is None
        await holder.close()

    asyncio.run(scenario())
    service.create_model.assert_not_awaited()
    service.create_sampling_client.assert_awaited_once_with(
        base_model="base-model",
        model_path="weaver://actor-step-5",
        tokenizer_path="/checkpoint/tokenizer",
    )
    tokenizer_loader.assert_called_once_with("/checkpoint/tokenizer", trust_remote_code=True)
    service.aclose.assert_awaited_once_with()


def test_sampling_only_requires_base_model_or_complete_checkpoint():
    with pytest.raises(ValueError, match="base_model is required"):
        WeaverServiceHolder(sampling_only=True)
    with pytest.raises(ValueError, match="tokenizer_path is required"):
        WeaverServiceHolder(sampling_only=True, model_path="weaver://actor-step-5")
    with pytest.raises(ValueError, match="model_path must not be empty"):
        WeaverServiceHolder(sampling_only=True, base_model="base-model", model_path="")
