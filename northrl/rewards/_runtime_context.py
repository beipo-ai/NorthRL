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

"""Task-local worker stop signals, without changing the reward/data contracts."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

_stop_event: ContextVar[Event | None] = ContextVar("reward_stop_event", default=None)


@contextmanager
def reward_runtime_context(stop_event: Event) -> Iterator[None]:
    """Bind the live event, not a snapshot, and restore the caller's context."""
    token = _stop_event.set(stop_event)
    try:
        yield
    finally:
        _stop_event.reset(token)


def reward_stop_requested() -> bool:
    """A worker can request graceful stop before cancelling its asyncio task."""
    event = _stop_event.get()
    return event is not None and event.is_set()
