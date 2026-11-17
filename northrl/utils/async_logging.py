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

"""Deliver SDK-loop log records without doing handler I/O on the network loop."""

from __future__ import annotations

import logging
import queue
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

_HandlerCall = Callable[[logging.Logger, logging.LogRecord], None]
_LogItem = tuple[_HandlerCall, logging.Logger, logging.LogRecord]
_LOCAL = threading.local()


class _HookState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0
        self.installed: _HandlerCall | None = None
        self.previous: _HandlerCall | None = None


_HOOK_STATE = _HookState()


def _install_hook() -> None:
    with _HOOK_STATE.lock:
        if _HOOK_STATE.users == 0:
            previous = logging.Logger.callHandlers

            def call_handlers(logger: logging.Logger, record: logging.LogRecord) -> None:
                dispatcher = getattr(_LOCAL, "dispatcher", None)
                if dispatcher is None:
                    previous(logger, record)
                else:
                    dispatcher.enqueue(previous, logger, record)

            # Capture the prior function in this closure. An external hook can
            # wrap ours during a run; after shutdown it must still pass through
            # safely, including if the SDK loop later restarts.
            _HOOK_STATE.previous = previous
            _HOOK_STATE.installed = call_handlers
            setattr(logging.Logger, "callHandlers", call_handlers)
        _HOOK_STATE.users += 1


def _remove_hook() -> None:
    with _HOOK_STATE.lock:
        _HOOK_STATE.users -= 1
        if _HOOK_STATE.users == 0 and logging.Logger.callHandlers is _HOOK_STATE.installed:
            assert _HOOK_STATE.previous is not None
            setattr(logging.Logger, "callHandlers", _HOOK_STATE.previous)


class AsyncLogDispatcher:
    """One bounded FIFO for records produced by a designated SDK thread.

    A temporary process-wide ``Logger.callHandlers`` hook is necessary to
    cover third-party SDK loggers and handlers attached after loop startup.
    Handler objects, filters, levels, propagation, and log-record metadata are
    preserved. Only threads inside ``activate()`` enqueue; all other producers
    retain ordinary synchronous logging, including pytest's capture handlers.
    Formatting and exception rendering happen on the worker, not on enqueue.
    When a slow handler fills the queue, new records are dropped and the worker
    reports the number dropped rather than blocking the network event loop.
    """

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("Log queue capacity must be positive")
        self._queue: queue.Queue[_LogItem | None] = queue.Queue(maxsize=capacity)
        self._dropped = 0
        self._state_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="northrl-sdk-logs", daemon=True)
        _install_hook()
        try:
            self._thread.start()
        except BaseException:
            _remove_hook()
            raise

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Route logging on the current thread through this dispatcher."""
        previous = getattr(_LOCAL, "dispatcher", None)
        _LOCAL.dispatcher = self
        try:
            yield
        finally:
            _LOCAL.dispatcher = previous

    def enqueue(
        self, callback: _HandlerCall, logger: logging.Logger, record: logging.LogRecord
    ) -> None:
        """Enqueue the original record, without formatting or waiting for I/O."""
        with self._state_lock:
            if self._closed:
                return
            try:
                self._queue.put_nowait((callback, logger, record))
            except queue.Full:
                self._dropped += 1

    def _report_drops(self) -> None:
        with self._state_lock:
            dropped, self._dropped = self._dropped, 0
        if dropped:
            logging.getLogger(__name__).warning(
                "SDK log queue full: dropped %s records to keep the network loop responsive",
                dropped,
            )

    def _deliver(self, item: _LogItem) -> None:
        callback, logger, record = item
        try:
            callback(logger, record)
        except Exception:  # pylint: disable=broad-exception-caught
            # A broken custom handler must not terminate the worker.
            # Avoid routing this diagnostic back through that handler.
            traceback.print_exc()
        try:
            self._report_drops()
        except Exception:  # pylint: disable=broad-exception-caught
            traceback.print_exc()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._deliver(item)
            finally:
                self._queue.task_done()
                # Tracebacks can retain a large rollout; release each record
                # before waiting for more work on an otherwise idle worker.
                del item

    def flush(self) -> None:
        """Wait outside the SDK thread until all accepted records are delivered."""
        if (
            getattr(_LOCAL, "dispatcher", None) is self
            or threading.current_thread() is self._thread
        ):
            raise RuntimeError(
                "Cannot wait for log delivery from the SDK loop or log worker thread"
            )
        self._queue.join()

    def close(self) -> None:
        """Drain accepted records and restore the logging hook after loop shutdown."""
        if (
            getattr(_LOCAL, "dispatcher", None) is self
            or threading.current_thread() is self._thread
        ):
            raise RuntimeError(
                "Cannot close the log dispatcher from the SDK loop or log worker thread"
            )
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()
        _remove_hook()
