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

"""Bound remote requests without changing their submission or result order."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any, TypeVar

Item = TypeVar("Item")
Handle = TypeVar("Handle")
Result = TypeVar("Result")


async def run_request_pipeline(
    items: Iterable[Item],
    submit: Callable[[Item], Awaitable[Handle]],
    collect: Callable[[Item, Handle], Coroutine[Any, Any, Result]],
    *,
    submit_ahead: int,
) -> list[Result]:
    """Keep at most 1 + submit_ahead requests outstanding; drain on failure.

    Submission stays serial. Cancellation stops further admission, but does not
    abandon accepted operations or let their caller proceed to an optimizer.
    """
    if submit_ahead < 0:
        raise ValueError("submit_ahead must be non-negative")
    stopping = False

    async def run() -> list[Result]:
        pending: deque[asyncio.Task[Result]] = deque()
        results = []
        try:
            for item in items:
                if stopping:
                    break
                for task in pending:
                    if task.done():
                        task.result()  # Stop admission on an already-observed failure.
                handle = await submit(item)
                pending.append(asyncio.create_task(collect(item, handle)))
                if len(pending) > submit_ahead:
                    results.append(await pending[0])
                    pending.popleft()
            while pending:
                results.append(await pending[0])
                pending.popleft()
            return results
        finally:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    worker = asyncio.create_task(run())
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        stopping = True
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise
