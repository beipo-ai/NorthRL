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

"""Bounded background confirmation of failed E2B sandbox termination."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_CLEANUP_RETRIES = 2
_CLEANUP_RPC_SECONDS = 10.0
_CLEANUP_DEADLINE_SECONDS = 60.0
_CLEANUP_RETRY_DELAY_SECONDS = 1.0
# Strong references keep cleanup alive after the owning rollout returns.
_pending_cleanups: set[asyncio.Task[None]] = set()


async def _confirm_absent(sandbox: Any, sandbox_id: str) -> bool:
    # Use the control plane: envd health failures do not prove termination.
    from e2b.exceptions import SandboxNotFoundException  # pylint: disable=import-outside-toplevel

    try:
        await asyncio.wait_for(
            sandbox.get_info(request_timeout=_CLEANUP_RPC_SECONDS), _CLEANUP_RPC_SECONDS
        )
    except SandboxNotFoundException:
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("E2B cleanup sandbox_id=%s status unknown: %s", sandbox_id, exc)
    return False


async def _retry_cleanup(sandbox: Any, sandbox_id: str) -> bool:
    for attempt in range(1, _CLEANUP_RETRIES + 1):
        await asyncio.sleep(_CLEANUP_RETRY_DELAY_SECONDS * attempt)
        if await _confirm_absent(sandbox, sandbox_id):
            return True
        try:
            # kill is idempotent: False also confirms the sandbox was absent.
            await asyncio.wait_for(sandbox.kill(), _CLEANUP_RPC_SECONDS)
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "E2B cleanup sandbox_id=%s retry=%s failed: %s", sandbox_id, attempt, exc
            )
    # A successful DELETE may lose its response. Check after the final retry.
    return await _confirm_absent(sandbox, sandbox_id)


async def _reconcile(sandbox: Any, sandbox_id: str, on_confirmed: Callable[[], None]) -> None:
    confirmed = False
    try:
        confirmed = await asyncio.wait_for(
            _retry_cleanup(sandbox, sandbox_id), _CLEANUP_DEADLINE_SECONDS
        )
        if confirmed:
            logger.info("E2B cleanup sandbox_id=%s closure confirmed", sandbox_id)
            on_confirmed()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("E2B cleanup sandbox_id=%s reconciliation failed: %s", sandbox_id, exc)
    finally:
        if not confirmed:
            logger.warning(
                "E2B cleanup sandbox_id=%s closure unconfirmed; retain for investigation",
                sandbox_id,
            )


def schedule_sandbox_cleanup(
    sandbox: Any, *, sandbox_id: str, on_confirmed: Callable[[], None]
) -> None:
    """Retain the original sandbox and reconcile without delaying its result."""
    logger.warning("E2B cleanup sandbox_id=%s queued for background confirmation", sandbox_id)
    task = asyncio.create_task(
        _reconcile(sandbox, sandbox_id, on_confirmed), name=f"e2b-cleanup-{sandbox_id}"
    )
    _pending_cleanups.add(task)
    task.add_done_callback(_pending_cleanups.discard)
