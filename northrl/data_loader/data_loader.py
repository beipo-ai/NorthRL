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

"""
DataLoader - Abstract base class for data loading
"""

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, override

from omegaconf import DictConfig

from ..base_module import NorthRLModule
from ..northrl_types import Trajectory
from ..utils.rollout_progress import ContinuousRolloutProgress, RolloutProgress

logger = logging.getLogger(__name__)


class BaseDataLoader(NorthRLModule, ABC):
    """
    Abstract base class for data loading
    """

    def __init__(
        self, config: DictConfig, is_validate: bool = False
    ):  # pylint: disable=unused-argument
        """
        Read file path from config.file, call _read_file to read file

        Args:
            config: Configuration file
        """
        super().__init__()
        self._is_validate: bool = is_validate

    @abstractmethod
    def get_next_item(self) -> dict[str, Any] | None:
        """
        Get the next item in sequence (mock implementation of sequential access)

        Returns:
            dict[str, Any]: Next data item or None if no more items
        """

    @abstractmethod
    def can_return_item(self) -> bool:
        """
        Check if the data loader can return an item
        """

    @abstractmethod
    def is_finished(self) -> bool:
        """
        Check if the data loader is empty
        """

    @abstractmethod
    def release_next_batch(self) -> None:
        """
        Release the next ordered batch after the current batch is processed.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the data loader to the beginning.

        This allows the data loader to be reused, particularly useful for
        validation data loaders that need to iterate over the same dataset
        multiple times.
        """

    @abstractmethod
    def skip_batches(self, num_batches: int) -> None:
        """
        Skip the first num_batches batches from the dataloader.
        """

    def close(self) -> None:
        """Release optional background resources owned by the data loader."""

    def subscribe_ready(
        self, callback: Callable[[], None]  # pylint: disable=unused-argument
    ) -> Callable[[], None] | None:
        """Subscribe to new work; return an unsubscribe function, or None to poll.

        Notifications must follow every change that can make an empty loader
        return work. Callbacks may run on the thread publishing that change.
        """
        return None

    def record_rollout_result(self, trajectory: Trajectory) -> None:
        """Observe a complete result accepted by the pool, for progress reporting only."""

    def get_rollout_progress(
        self, *, grouped: bool = True  # pylint: disable=unused-argument
    ) -> dict[str, int] | None:
        """Return current ordered-round progress, if this loader supports it."""
        return None

    def get_continuous_rollout_progress(self) -> dict[str, int] | None:
        """Return cross-batch sampling progress, if this loader supports it."""
        return None

    async def prepare(self) -> None:
        """Resolve optional external source metadata before resume/iteration."""

    @staticmethod
    def repeat_item(item: dict[str, Any], n: int) -> list[dict[str, Any]]:
        """
        Create one sampling group with n independently identified rollout requests

        Args:
            item: Original data item to repeat
            n: Number of times to repeat the item

        Returns:
            list[dict[str, Any]]: List of n repeated items, each with group_id and run_id
        """
        # A task can be sampled again while an earlier group is still running.
        # Keep task_id stable, but never reuse execution identities across draws.
        group_id = uuid.uuid4().hex
        repeated_items = []
        for run_id in range(n):
            item_copy = item.copy()
            item_copy.update(
                group_id=group_id,
                run_id=run_id,
                rollout_id=uuid.uuid4().hex,
                expected_rollouts=n,
            )
            repeated_items.append(item_copy)
        return repeated_items


class SequentialDataLoader(BaseDataLoader, ABC):
    """
    Abstract base class for sequential data loaders that maintain insertion order

    This class provides the sequential iteration logic using __getitem__ while leaving
    the actual storage mechanism to concrete implementations. It maintains a current
    index for sequential access and implements get_next_item() using the abstract
    __getitem__ method.

    Concrete implementations must provide:
    - __len__, __getitem__, is_finished
    """

    def __init__(self, config: DictConfig, is_validate: bool = False):
        super().__init__(config, is_validate)

        self._lock: threading.Lock = threading.Lock()
        self._ready_callbacks: set[Callable[[], None]] = set()
        self._ready_callbacks_lock = threading.Lock()
        self._data_index: int = 0  # index of next loaded data, also the number of loaded data

        # keep_batch_order gates batch N+1 until the current consumer explicitly
        # releases it after processing batch N.
        self._batch_size: int = config.batch_size
        self._data_buffer: list[dict[str, Any]] = (
            []
        )  # buffer of items to return. If no buffered data is available, the data loader will fetch a new batch from the underlying storage.
        self._buffer_index: int = 0  # index of next item to return from _data_buffer

        self._rollout_progress: RolloutProgress | None = None

        # Batch order control
        self._keep_batch_order: bool = config.get("keep_batch_order", False)

        if self._is_validate:
            self._keep_batch_order = False
        self._continuous_rollout_progress = (
            ContinuousRolloutProgress()
            if not self._keep_batch_order and not self._is_validate
            else None
        )

        # Rollout repeat configuration. Production always derives this from
        # data.train.rollout_repeat_n; the .get() fallback keeps the
        # MockDataLoader-based unit tests from needing schema-shaped configs.
        self._rollout_repeat_n: int = config.get("rollout_repeat_n", 1)
        # Validation uses its own repeat count (the k for grouped pass@k);
        # rollout_repeat_n never applies to validation.
        self._validation_repeat_n: int = config.get("validation_repeat_n", 1)

        logger.info(
            f"SequentialDataLoader initialized - is_validate: {self._is_validate}, batch_size: {self._batch_size}, "
            f"keep_batch_order: {self._keep_batch_order}, rollout_repeat_n: {self._rollout_repeat_n}, "
            f"validation_repeat_n: {self._validation_repeat_n}"
        )

    @override
    def can_return_item(self) -> bool:
        """
        Check if the data loader can return an item
        """
        with self._lock:
            return self._buffer_index < len(self._data_buffer)

    @override
    def get_next_item(self) -> dict[str, Any] | None:
        """
        Get the next item from the data buffer

        If buffer is empty or exhausted, fetches a new batch using _fetch_one_batch().
        Handles batch order control if keep_batch_order is enabled.

        Returns:
            dict[str, Any]: Next data item or None if no more items available
        """
        with self._lock:
            # Check if we need to fetch a new batch
            if self._buffer_index >= len(self._data_buffer):
                # Buffer is exhausted, try to fetch a new batch.
                # Ordered loaders wait for the current consumer to release the
                # next batch. The initial batch can be fetched immediately.
                if not self._keep_batch_order or self._data_index == 0:
                    fetch_success = self._try_fetch_batch()
                    if not fetch_success:
                        return None
                else:
                    return None

            # Return item from buffer
            item = self._data_buffer[self._buffer_index]
            self._buffer_index += 1
            self._data_index += 1
            if self._rollout_progress is not None:
                self._rollout_progress.claim(item["rollout_id"])
            if self._continuous_rollout_progress is not None:
                self._continuous_rollout_progress.claim(item["rollout_id"])
            return item

    @override
    def release_next_batch(self) -> None:
        """Release the next ordered batch after the current batch is processed."""
        with self._lock:
            if self._keep_batch_order and self._buffer_index >= len(self._data_buffer):
                self._try_fetch_batch()
        self._notify_ready()

    @override
    def subscribe_ready(self, callback: Callable[[], None]) -> Callable[[], None]:
        with self._ready_callbacks_lock:
            self._ready_callbacks.add(callback)

        def unsubscribe() -> None:
            with self._ready_callbacks_lock:
                self._ready_callbacks.discard(callback)

        return unsubscribe

    def _notify_ready(self) -> None:
        # Publish state under _lock first, then invoke callbacks outside it.
        with self._ready_callbacks_lock:
            callbacks = tuple(self._ready_callbacks)
        for callback in callbacks:
            callback()

    def _try_fetch_batch(self) -> bool:
        """
        Try to fetch a new batch, respecting batch order control

        Returns:
            bool: True if batch was fetched successfully, False if no more data
        """

        # Fetch new batch from implementation
        new_batch = self._fetch_batch_data()
        if not new_batch:  # Empty batch means no more data
            return False

        # Expand each prompt by its repeat count. Training uses rollout_repeat_n
        # (always, even at n==1, for explicit execution identities); validation uses
        # validation_repeat_n and only expands when k>1 (grouped pass@k).
        repeat_n = self._validation_repeat_n if self._is_validate else self._rollout_repeat_n
        if (not self._is_validate) or repeat_n > 1:
            base_index_start = self._data_index
            repeated_batch = []
            for j, item in enumerate(new_batch):
                # Ensure stable task_id exists (do not override if user already provided one)
                if "task_id" not in item:
                    item = item.copy()
                    # Prefer a dataset-provided stable index if present (e.g. TorchDataLoader adds `index`).
                    # Fall back to a sequential counter for this run.
                    item["task_id"] = item.get("index", base_index_start + j)

                repeated_items = BaseDataLoader.repeat_item(item, repeat_n)
                repeated_batch.extend(repeated_items)
            new_batch = repeated_batch
            logger.info(
                f"SequentialDataLoader: Applied repeat_n={repeat_n} "
                f"(is_validate={self._is_validate}), expanded batch to {len(new_batch)} items"
            )

        # Replace buffer with new batch
        self._data_buffer = new_batch
        self._buffer_index = 0
        self._rollout_progress = (
            RolloutProgress(new_batch) if self._keep_batch_order and not self._is_validate else None
        )
        if self._continuous_rollout_progress is not None:
            self._continuous_rollout_progress.set_queued(len(new_batch))

        logger.info(f"SequentialDataLoader: Fetched new batch of {len(new_batch)} items")
        return True

    def record_rollout_result(self, trajectory: Trajectory) -> None:
        """Observe an accepted result without using worker activity as completion."""
        # The continuous tracker outlives individual input buffers and has its own
        # lock. Never wait for an asynchronous fetch while accepting a result.
        continuous_progress = self._continuous_rollout_progress
        if continuous_progress is not None:
            continuous_progress.finish(trajectory)
            return
        if not self._keep_batch_order:
            return
        with self._lock:
            if self._rollout_progress is not None:
                self._rollout_progress.finish(trajectory)

    def get_rollout_progress(self, *, grouped: bool = True) -> dict[str, int] | None:
        if not self._keep_batch_order:
            return None
        with self._lock:
            if self._rollout_progress is None:
                return None
            return self._rollout_progress.snapshot(grouped=grouped)

    def get_continuous_rollout_progress(self) -> dict[str, int] | None:
        continuous_progress = self._continuous_rollout_progress
        return continuous_progress.snapshot() if continuous_progress is not None else None

    def get_progress_info(self) -> dict[str, Any]:
        """
        Get progress information for the current batch.

        Returns:
            dict with keys:
                - batch_size: Total size of current batch
                - batch_remaining: Number of items remaining in current batch
                - batch_processed: Number of items processed from current batch
                - total_processed: Total number of items processed so far
        """
        with self._lock:
            batch_size = len(self._data_buffer)
            batch_processed = self._buffer_index
            batch_remaining = batch_size - batch_processed

            return {
                "batch_size": batch_size,
                "batch_remaining": batch_remaining,
                "batch_processed": batch_processed,
                "total_processed": self._data_index,
            }

    @abstractmethod
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        """
        Fetch one batch of items from the underlying storage

        This method should return up to batch_size items. Concrete implementations
        should handle their specific data source logic here.

        Returns:
            list[dict[str, Any]]: List of data items (up to batch_size)
        """

    @abstractmethod
    def _reset_iterator(self) -> None:
        """
        Reset the underlying iterator/data source to the beginning.

        Concrete implementations should override this to handle their specific
        iterator reset logic (e.g., creating a new DataLoader iterator).
        """

    @override
    def reset(self) -> None:
        """
        Reset the dataloader to the beginning.

        This allows the data loader to be reused across multiple iterations,
        particularly useful for validation data loaders.

        The method:
        - Resets the underlying iterator via _reset_iterator()
        - Clears the buffer and resets buffer/data indices
        """
        with self._lock:
            # Reset the underlying iterator (implementation-specific)
            self._reset_iterator()

            # Clear the buffer and reset indices
            self._data_buffer = []
            self._buffer_index = 0
            self._rollout_progress = None
            if self._continuous_rollout_progress is not None:
                self._continuous_rollout_progress = ContinuousRolloutProgress()
            self._data_index = 0

            logger.info(
                f"SequentialDataLoader: Reset dataloader to beginning "
                f"(is_validate={self._is_validate})"
            )
        self._notify_ready()

    def skip_batches(self, num_batches: int) -> None:
        """
        Skip the first num_batches batches from the dataloader.

        This is used for resuming training from a checkpoint. When resuming,
        we need to skip the batches that have already been consumed in the
        previous training run.

        Note: This method skips underlying data batches (before rollout_repeat_n
        expansion). If rollout_repeat_n > 1, each underlying batch will be expanded
        into multiple repeated items, but we skip at the batch level.

        Args:
            num_batches: Number of batches to skip

        Raises:
            RuntimeError: If dataloader has already started iteration
        """
        with self._lock:
            if self._data_index > 0:
                raise RuntimeError(
                    f"Cannot skip batches after dataloader has started iteration. "
                    f"Current data_index: {self._data_index}"
                )

            if num_batches <= 0:
                logger.info("No batches to skip (num_batches <= 0)")
                return

            logger.info(
                f"SequentialDataLoader: Skipping {num_batches} batches for resume "
                f"(rollout_repeat_n={self._rollout_repeat_n})"
            )

            skipped_count = 0
            for i in range(num_batches):
                # Fetch and discard the batch
                success = self._try_fetch_batch()
                if not success:
                    logger.warning(
                        f"SequentialDataLoader: Reached end of data after skipping "
                        f"{skipped_count}/{num_batches} batches. Dataset may be smaller "
                        f"than expected or checkpoint step is invalid."
                    )
                    break

                skipped_count += 1

                # Log progress for large skips
                if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                    logger.info(f"Skipped {i + 1}/{num_batches} batches...")

            # After skipping, clear the buffer so the next get_next_item() starts fresh
            # Update _data_index to reflect the skipped items
            total_items_skipped = self._data_index
            self._data_buffer = []
            self._buffer_index = 0
            self._rollout_progress = None
            if self._continuous_rollout_progress is not None:
                self._continuous_rollout_progress = ContinuousRolloutProgress()

            logger.info(
                f"SequentialDataLoader: Successfully skipped {skipped_count} batches "
                f"({total_items_skipped} items after rollout_repeat_n expansion)"
            )
