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

"""Bound synchronous sampling work without occupying the shared SDK loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")


async def settle_task(task: asyncio.Task[_T]) -> _T:
    """Drain work before forwarding cancellation, including repeated cancels."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException as exc:
        if cancelled:
            raise asyncio.CancelledError() from exc
        raise
    if cancelled:
        raise asyncio.CancelledError()
    return result


class SamplingPreparationWorker:
    """Serialize one holder's tokenizer and local request/response factories.

    Call ``run`` from the holder's SDK event loop for *every* access to its
    tokenizer, including loading, rendering, encoding, and response decoding.
    Each callback runs off the loop. It may only read caller-owned inputs and
    build local values; client submission and holder state updates belong on
    the loop after ``run`` returns. Callers must keep inputs unchanged until
    it settles, including after cancelling their request.

    Waiting callers consume no executor threads. An admitted callback keeps
    the single slot until it actually finishes, even if its caller is cancelled
    repeatedly. Thus cancellation cannot accumulate detached background work
    or allow concurrent accesses to a shared tokenizer. ``close`` rejects new
    work immediately and drains the admitted callback before returning.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._closed = False

    async def run(self, callback: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Run a synchronous callback in the holder's single preparation slot."""
        if self._closed:
            raise RuntimeError("Sampling preparation worker is closed")
        async with self._lock:
            if self._closed:
                raise RuntimeError("Sampling preparation worker is closed")
            task = asyncio.create_task(asyncio.to_thread(callback, *args, **kwargs))
            result = await settle_task(task)
            if self._closed:
                # Closing a holder during preparation must not submit the result.
                raise RuntimeError("Sampling preparation worker is closed")
            return result

    async def close(self) -> None:
        """Reject queued/new work and await any admitted callback's completion."""
        self._closed = True

        async def wait_until_idle() -> None:
            async with self._lock:
                pass

        await settle_task(asyncio.create_task(wait_until_idle()))
