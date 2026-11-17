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

"""Readiness notifications let consumers resume after an ordered batch opens."""

from omegaconf import OmegaConf

from northrl.data_loader import BaseDataLoader, SequentialDataLoader


class _OrderedLoader(SequentialDataLoader):
    def __init__(self):
        super().__init__(OmegaConf.create({"batch_size": 2, "keep_batch_order": True}))
        self._cursor = 0

    def _fetch_batch_data(self):
        end = min(self._cursor + self._batch_size, 4)
        items = [{"id": index} for index in range(self._cursor, end)]
        self._cursor = end
        return items

    def _reset_iterator(self):
        self._cursor = 0

    def is_finished(self):
        return self._cursor == 4 and not self.can_return_item()


def _read_from_callback(loader):
    # A regression invoking callbacks under the non-reentrant loader lock must
    # fail an assertion instead of hanging in can_return_item/get_next_item.
    if not loader._lock.acquire(blocking=False):
        return False, None, None
    loader._lock.release()
    return True, loader.can_return_item(), loader.get_next_item()


def test_release_notifies_after_next_batch_is_available_outside_loader_lock():
    loader = _OrderedLoader()
    assert [loader.get_next_item()["id"] for _ in range(2)] == [0, 1]
    assert loader.get_next_item() is None
    assert not loader.can_return_item()
    observed = []
    unsubscribe = loader.subscribe_ready(lambda: observed.append(_read_from_callback(loader)))

    try:
        loader.release_next_batch()

        assert len(observed) == 1
        lock_available, ready, item = observed[0]
        assert lock_available and ready
        assert item["id"] == 2
        assert loader.get_next_item()["id"] == 3
    finally:
        unsubscribe()


def test_reset_notifies_after_rewinding_outside_loader_lock():
    loader = _OrderedLoader()
    assert [loader.get_next_item()["id"] for _ in range(2)] == [0, 1]
    loader.release_next_batch()
    assert loader.get_next_item()["id"] == 2
    observed = []
    unsubscribe = loader.subscribe_ready(lambda: observed.append(_read_from_callback(loader)))

    try:
        loader.reset()

        assert len(observed) == 1
        lock_available, _, item = observed[0]
        assert lock_available
        assert item["id"] == 0
        assert loader.get_next_item()["id"] == 1
    finally:
        unsubscribe()


def test_unsubscribe_removes_only_that_ready_listener():
    loader = _OrderedLoader()
    assert [loader.get_next_item()["id"] for _ in range(2)] == [0, 1]
    removed_calls = []
    active_calls = []
    unsubscribe = loader.subscribe_ready(lambda: removed_calls.append(True))
    unsubscribe_active = loader.subscribe_ready(lambda: active_calls.append(True))
    unsubscribe()

    try:
        loader.release_next_batch()
        loader.reset()

        assert removed_calls == []
        assert active_calls == [True, True]
    finally:
        unsubscribe_active()

    loader.release_next_batch()
    assert active_calls == [True, True]


class _LegacyLoader(BaseDataLoader):
    """Existing custom loaders do not need to implement readiness notifications."""

    def get_next_item(self):
        return None

    def can_return_item(self):
        return False

    def is_finished(self):
        return True

    def release_next_batch(self):
        pass

    def reset(self):
        pass

    def skip_batches(self, num_batches):
        pass


def test_base_loader_without_notification_support_retains_polling_fallback():
    loader = _LegacyLoader(OmegaConf.create({}))
    calls = []

    assert loader.subscribe_ready(lambda: calls.append(True)) is None
    loader.release_next_batch()
    loader.reset()
    assert calls == []
