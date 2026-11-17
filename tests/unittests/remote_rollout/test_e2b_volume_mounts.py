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

"""Host-selected volumes survive creation retries and stay isolated by profile."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from northrl.remote_rollout import e2b_executor
from northrl.utils.e2b_volume_mounts import validate_volume_mounts


@pytest.mark.parametrize("value", [None, {}, OmegaConf.create({})])
def test_empty_mounts_preserve_the_existing_sdk_call(value):
    create = AsyncMock(return_value="sandbox")
    result = asyncio.run(
        e2b_executor._create_sandbox_with_backoff(
            create,
            template="template",
            sandbox_timeout=3600,
            rollout_id="test",
            retry_budget=e2b_executor._SandboxCreationRetryBudget(10),
            volume_mounts=value,
        )
    )
    assert result == "sandbox"
    create.assert_awaited_once_with(template="template", timeout=3600)


def test_mounts_survive_placement_retry_and_sdk_argument_mutation(monkeypatch):
    class SandboxException(Exception):
        pass

    calls = []

    async def create(**kwargs):
        calls.append({**kwargs, "volume_mounts": dict(kwargs["volume_mounts"])})
        if len(calls) == 1:
            kwargs["volume_mounts"].clear()
            raise SandboxException("500: placement failed")
        return "sandbox"

    monkeypatch.setattr(e2b_executor.asyncio, "sleep", AsyncMock())
    mounts = OmegaConf.create({"/mnt/data": "public-data", "/mnt/tools": "shared-tools"})
    expected = OmegaConf.to_container(mounts)
    result = asyncio.run(
        e2b_executor._create_sandbox_with_backoff(
            create,
            template="template",
            sandbox_timeout=3600,
            rollout_id="test",
            retry_budget=e2b_executor._SandboxCreationRetryBudget(10),
            volume_mounts=mounts,
        )
    )
    assert result == "sandbox"
    assert calls == [{"template": "template", "timeout": 3600, "volume_mounts": expected}] * 2
    assert OmegaConf.to_container(mounts) == expected


@pytest.mark.parametrize(
    "mounts",
    [
        [],
        "volume",
        {1: "volume"},
        {"relative": "volume"},
        {"/": "volume"},
        {"/mnt/../data": "volume"},
        {"/mnt//data": "volume"},
        {"//mnt/data": "volume"},
        {"/mnt/data/": "volume"},
        {"/mnt/./data": "volume"},
        {"/mnt/\x00data": "volume"},
        {"/mnt/data": ""},
        {"/mnt/data": " volume"},
        {"/mnt/data": "volume "},
        {"/mnt/data": {"name": "volume"}},
    ],
)
def test_bad_mounts_fail_before_remote_creation(mounts):
    create = AsyncMock()
    with pytest.raises(ValueError, match="volume_mounts"):
        asyncio.run(
            e2b_executor._create_sandbox_with_backoff(
                create,
                template="template",
                sandbox_timeout=3600,
                rollout_id="test",
                retry_budget=e2b_executor._SandboxCreationRetryBudget(10),
                volume_mounts=mounts,
            )
        )
    create.assert_not_called()


def test_validation_returns_an_independent_plain_dictionary():
    original = {"/mnt/data": "public-data"}
    checked = validate_volume_mounts(original)
    checked["/mnt/data"] = "changed"
    assert original == {"/mnt/data": "public-data"}
