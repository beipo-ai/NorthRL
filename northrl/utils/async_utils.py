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

"""Utilities for driving asyncio-native clients from synchronous NorthRL code."""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import nullcontext
from typing import Any, TypeVar

from northrl.utils.async_logging import AsyncLogDispatcher

_T = TypeVar("_T")

_ASYNC_LOOP_THREAD_NAME = "northrl-async-loop"
_ROLLOUT_LOOP_THREAD_NAME = "northrl-rollout-loop"
_TRAINER_LOOP_THREAD_NAME = "northrl-trainer-loop"


class _AsyncLoopThread:
    """Owns one background event loop."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._log_dispatcher: AsyncLogDispatcher | None = None

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            dispatcher = (
                AsyncLogDispatcher()
                if self._name in (_ASYNC_LOOP_THREAD_NAME, _ROLLOUT_LOOP_THREAD_NAME)
                else None
            )
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop, dispatcher),
                name=self._name,
                daemon=True,
            )
            try:
                thread.start()
            except BaseException:
                loop.close()
                if dispatcher is not None:
                    dispatcher.close()
                raise
            self._loop = loop
            self._thread = thread
            self._log_dispatcher = dispatcher
            return loop

    @staticmethod
    def _run_loop(
        loop: asyncio.AbstractEventLoop, dispatcher: AsyncLogDispatcher | None = None
    ) -> None:
        asyncio.set_event_loop(loop)
        with dispatcher.activate() if dispatcher is not None else nullcontext():
            try:
                loop.run_forever()
            finally:
                try:
                    # pylint: disable-next=protected-access
                    asyncio.runners._cancel_all_tasks(loop)  # type: ignore[attr-defined]
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    try:
                        # Cancelling a to_thread task does not stop its running
                        # function. Join those workers before closing their loop
                        # or draining the loop's log dispatcher.
                        loop.run_until_complete(loop.shutdown_default_executor())
                    finally:
                        loop.close()

    def is_current_thread(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def submit(self, coro: Coroutine[Any, Any, _T]) -> Future[_T]:
        loop = self._ensure_started()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        if self.is_current_thread():
            coro.close()
            raise RuntimeError(
                f"run_coro() cannot be called from the {self._name} thread; "
                "await the coroutine directly instead."
            )
        return self.submit(coro).result()

    async def await_result(self, coro: Coroutine[Any, Any, _T]) -> _T:
        if self.is_current_thread():
            return await coro
        return await asyncio.wrap_future(self.submit(coro))

    def flush_logs(self) -> None:
        with self._lock:
            dispatcher = self._log_dispatcher
        if dispatcher is not None:
            dispatcher.flush()

    def shutdown(self) -> None:
        with self._lock:
            loop, thread, dispatcher = self._loop, self._thread, self._log_dispatcher
            self._loop, self._thread, self._log_dispatcher = None, None, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join()
        if dispatcher is not None:
            dispatcher.close()


_ASYNC_BRIDGE = _AsyncLoopThread(_ASYNC_LOOP_THREAD_NAME)
_ROLLOUT_BRIDGE = _AsyncLoopThread(_ROLLOUT_LOOP_THREAD_NAME)
_TRAINER_BRIDGE = _AsyncLoopThread(_TRAINER_LOOP_THREAD_NAME)


def submit_coro(coro: Coroutine[Any, Any, _T]) -> Future[_T]:
    """Submit a coroutine to NorthRL's process-wide async SDK loop."""
    return _ASYNC_BRIDGE.submit(coro)


def submit_rollout_coro(coro: Coroutine[Any, Any, _T]) -> Future[_T]:
    """Submit a rollout coroutine to NorthRL's dedicated rollout loop."""
    return _ROLLOUT_BRIDGE.submit(coro)


def submit_trainer_coro(coro: Coroutine[Any, Any, _T]) -> Future[_T]:
    """Submit a trainer coroutine to NorthRL's dedicated trainer loop."""
    return _TRAINER_BRIDGE.submit(coro)


def run_coro(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine on NorthRL's process-wide async SDK loop."""
    return _ASYNC_BRIDGE.run(coro)


async def await_coro(coro: Coroutine[Any, Any, _T]) -> _T:
    """Await a coroutine on NorthRL's async SDK loop from another event loop."""
    return await _ASYNC_BRIDGE.await_result(coro)


def flush_sdk_logs() -> None:
    """Wait for SDK log delivery from a non-SDK thread, e.g. before assertions."""
    _ASYNC_BRIDGE.flush_logs()


def shutdown_event_loop() -> None:
    """Stop NorthRL background event loops. Intended for tests."""
    _ROLLOUT_BRIDGE.shutdown()
    _TRAINER_BRIDGE.shutdown()
    _ASYNC_BRIDGE.shutdown()


# Registered after logging imports so loop records drain before logging shuts down.
atexit.register(shutdown_event_loop)
