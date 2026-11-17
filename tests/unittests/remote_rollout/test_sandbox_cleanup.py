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

"""Cleanup confirmation is bounded and never guesses absence from network errors."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.remote_rollout import sandbox_cleanup


class _NotFound(Exception):
    pass


@pytest.fixture(autouse=True)
def cleanup_policy(monkeypatch):
    exceptions = ModuleType("e2b.exceptions")
    exceptions.SandboxNotFoundException = _NotFound
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exceptions)
    monkeypatch.setattr(sandbox_cleanup, "_CLEANUP_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(sandbox_cleanup, "_CLEANUP_RPC_SECONDS", 0.01)
    monkeypatch.setattr(sandbox_cleanup, "_CLEANUP_DEADLINE_SECONDS", 0.1)


@pytest.mark.parametrize(
    "probes,kills,expected_kills,confirmed",
    [
        ([_NotFound("gone")], [], 0, True),
        ([object()], [True], 1, True),
        ([object()], [False], 1, True),
        ([TimeoutError("unreachable")], [True], 1, True),
        ([object(), _NotFound("first retry succeeded remotely")], [TimeoutError()], 1, True),
        (
            [object(), object(), _NotFound("last retry succeeded remotely")],
            [TimeoutError(), TimeoutError()],
            2,
            True,
        ),
        ([object(), object(), object()], [TimeoutError(), TimeoutError()], 2, False),
        ([ConnectionError("404 from proxy")] * 3, [TimeoutError(), TimeoutError()], 2, False),
    ],
)
def test_confirms_only_successful_delete_or_control_plane_not_found(
    probes, kills, expected_kills, confirmed, caplog
):
    sandbox = SimpleNamespace(
        get_info=AsyncMock(side_effect=probes), kill=AsyncMock(side_effect=kills)
    )
    callback = Mock()
    asyncio.run(sandbox_cleanup._reconcile(sandbox, "original-id", callback))
    assert sandbox.kill.await_count == expected_kills
    assert callback.call_count == int(confirmed)
    assert ("closure unconfirmed" in caplog.text) is not confirmed
    if not confirmed:
        assert "sandbox_id=original-id" in caplog.text


def test_hung_status_and_delete_requests_exhaust_finite_retries():
    async def hang(**_kwargs):
        await asyncio.Event().wait()

    sandbox = SimpleNamespace(
        get_info=AsyncMock(side_effect=hang), kill=AsyncMock(side_effect=hang)
    )
    callback = Mock()
    asyncio.run(asyncio.wait_for(sandbox_cleanup._reconcile(sandbox, "hung-id", callback), 0.3))
    assert sandbox.get_info.await_count == 3
    assert sandbox.kill.await_count == 2
    callback.assert_not_called()


def test_total_cleanup_deadline_bounds_long_requests(monkeypatch, caplog):
    async def hang(**_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(sandbox_cleanup, "_CLEANUP_RPC_SECONDS", 10)
    monkeypatch.setattr(sandbox_cleanup, "_CLEANUP_DEADLINE_SECONDS", 0.01)
    sandbox = SimpleNamespace(get_info=AsyncMock(side_effect=hang), kill=AsyncMock())
    callback = Mock()
    asyncio.run(asyncio.wait_for(sandbox_cleanup._reconcile(sandbox, "deadline-id", callback), 0.3))
    sandbox.kill.assert_not_awaited()
    callback.assert_not_called()
    assert "sandbox_id=deadline-id closure unconfirmed" in caplog.text


def test_background_task_is_retained_until_confirmation():
    async def run():
        release = asyncio.Event()

        async def probe(**_kwargs):
            await release.wait()
            raise _NotFound("gone")

        callback = Mock()
        sandbox = SimpleNamespace(get_info=AsyncMock(side_effect=probe), kill=AsyncMock())
        sandbox_cleanup.schedule_sandbox_cleanup(
            sandbox, sandbox_id="retained-id", on_confirmed=callback
        )
        (task,) = sandbox_cleanup._pending_cleanups
        callback.assert_not_called()
        release.set()
        await task
        await asyncio.sleep(0)
        assert task not in sandbox_cleanup._pending_cleanups
        callback.assert_called_once_with()
        sandbox.kill.assert_not_awaited()

    asyncio.run(run())


def test_worker_shutdown_keeps_unconfirmed_id_in_log(caplog):
    async def run():
        started = asyncio.Event()

        async def probe(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        sandbox = SimpleNamespace(get_info=AsyncMock(side_effect=probe), kill=AsyncMock())
        callback = Mock()
        sandbox_cleanup.schedule_sandbox_cleanup(
            sandbox, sandbox_id="shutdown-id", on_confirmed=callback
        )
        await started.wait()
        return callback

    callback = asyncio.run(run())
    callback.assert_not_called()
    assert "sandbox_id=shutdown-id closure unconfirmed" in caplog.text
    assert not sandbox_cleanup._pending_cleanups
