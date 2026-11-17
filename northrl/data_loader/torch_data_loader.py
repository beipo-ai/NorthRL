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
TorchDataLoader - A real data loader that uses PyTorch's DataLoader
"""

import copy
import json
import logging
from typing import Any, BinaryIO, Iterator, List, Union, override

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, Dataset

from .data_loader import SequentialDataLoader

logger = logging.getLogger(__name__)


class FileDataset(Dataset):
    """
    PyTorch Dataset for loading data from parquet, CSV, or JSONL files.

    JSONL records stay on disk: only byte offsets are indexed, and each record
    is parsed when requested. Prompt filtering parses one record at a time
    while building the index. JSON types and absent fields are preserved.
    CSV and parquet inputs retain eager loading.

    Args:
        data_files: Single file path or list of paths (supports .parquet, .csv, and .jsonl)
        filter_prompts: Whether to filter prompts based on length
        max_prompt_length: Maximum allowed prompt length (in tokens)
        tokenizer: Tokenizer for filtering prompts
        prompt_key: Key in the data dict that contains the prompt
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        filter_prompts: bool = False,
        max_prompt_length: int = 1024,
        tokenizer=None,
        prompt_key: str = "prompt",
    ) -> None:
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume

        self.filter_prompts = filter_prompts
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        self._records: list[tuple[int, int]] | None = None
        self._dataframes: dict[int, pd.DataFrame] = {}
        self._jsonl_handles: dict[int, BinaryIO] = {}

        self._read_files_and_filter()
        self.serialize_dataset = False

    def _read_files_and_filter(self) -> None:
        """Read parquet, CSV, or JSONL files and optionally filter based on prompt length"""
        self.close()
        self._records = None
        self._dataframes = {}
        self.__dict__.pop("dataframe", None)
        if any(data_file.lower().endswith(".jsonl") for data_file in self.data_files):
            self._index_files_and_filter()
            return

        dataframes = []
        for data_file in self.data_files:
            # read files based on extension
            file_ext = data_file.lower().split(".")[-1]
            if file_ext == "parquet":
                dataframe = pd.read_parquet(data_file)
            elif file_ext == "csv":
                # Explicitly specify UTF-8 encoding to handle Chinese characters correctly
                dataframe = pd.read_csv(data_file, encoding="utf-8")
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}. Only 'parquet', 'csv', and 'jsonl' are supported."
                )
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        logger.info(
            f"FileDataset: Loaded {len(self.dataframe)} items from {len(self.data_files)} files"
        )

        # filter out too long prompts
        if self.filter_prompts:
            logger.info(
                f"FileDataset: Filtering prompts with max_prompt_length={self.max_prompt_length}"
            )
            assert self.tokenizer is not None, "tokenizer is required when filter_prompts is True"
            original_len = len(self.dataframe)
            self.dataframe = self.dataframe[self.dataframe.apply(self._check_prompt_length, axis=1)]
            self.dataframe = self.dataframe.reset_index(drop=True)
            logger.info(
                f"FileDataset: Filtered dataset from {original_len} to {len(self.dataframe)} items"
            )

    def _check_prompt_length(self, doc) -> bool:
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                doc[self.prompt_key],
                add_generation_prompt=True,
                tokenize=False,
                add_special_tokens=True,
            )
            tokenized_prompt = self.tokenizer(
                formatted_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = tokenized_prompt["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids])
            assert input_ids.ndim == 2
            return input_ids.shape[-1] <= self.max_prompt_length
        except Exception as e:
            logger.warning(f"Failed to process prompt: {e}")
            return False

    @staticmethod
    def _parse_jsonl_record(raw: bytes, path: str, offset: int) -> dict[str, Any]:
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSONL record at {path}, byte offset {offset}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record must be an object at {path}, byte offset {offset}")
        return record

    @staticmethod
    def _iter_jsonl_offsets(handle: BinaryIO) -> Iterator[int]:
        """Find nonblank records without buffering even a very large JSONL line."""
        chunk_offset = 0
        record_offset = 0
        has_content = False
        while chunk := handle.read(1024 * 1024):
            start = 0
            while start < len(chunk):
                newline = chunk.find(b"\n", start)
                end = len(chunk) if newline < 0 else newline + 1
                if not has_content and chunk[start:end].strip():
                    has_content = True
                if newline >= 0:
                    if has_content:
                        yield record_offset
                    record_offset = chunk_offset + end
                    has_content = False
                start = end
            chunk_offset += len(chunk)
        if has_content:
            yield record_offset

    def _index_files_and_filter(self) -> None:
        """Keep JSONL offsets and eager non-JSONL rows in configured file order."""
        self._records = []
        self._dataframes = {}
        if self.filter_prompts:
            assert self.tokenizer is not None, "tokenizer is required when filter_prompts is True"
        original_len = 0
        for file_index, data_file in enumerate(self.data_files):
            file_ext = data_file.lower().split(".")[-1]
            if file_ext == "jsonl":
                logger.info("FileDataset: Indexing JSONL file %s", data_file)
                with open(data_file, "rb") as handle:
                    offsets = list(self._iter_jsonl_offsets(handle))
                    original_len += len(offsets)
                    for offset in offsets:
                        if self.filter_prompts:
                            handle.seek(offset)
                            record = self._parse_jsonl_record(handle.readline(), data_file, offset)
                            if not self._check_prompt_length(record):
                                continue
                        self._records.append((file_index, offset))
                logger.info("FileDataset: Finished indexing JSONL file %s", data_file)
            else:
                if file_ext == "parquet":
                    dataframe = pd.read_parquet(data_file)
                elif file_ext == "csv":
                    dataframe = pd.read_csv(data_file, encoding="utf-8")
                else:
                    raise ValueError(
                        f"Unsupported file format: {file_ext}. "
                        "Only 'parquet', 'csv', and 'jsonl' are supported."
                    )
                original_len += len(dataframe)
                if self.filter_prompts:
                    dataframe = dataframe[dataframe.apply(self._check_prompt_length, axis=1)]
                self._dataframes[file_index] = dataframe.reset_index(drop=True)
                self._records.extend((file_index, row) for row in range(len(dataframe)))
        logger.info(
            "FileDataset: Indexed %d items from %d files (JSONL records loaded on demand)",
            len(self._records),
            len(self.data_files),
        )
        if self.filter_prompts:
            logger.info(
                "FileDataset: Filtered dataset from %d to %d items",
                original_len,
                len(self._records),
            )

    def close(self) -> None:
        """Release cached JSONL handles; subsequent reads reopen them lazily."""
        for handle in getattr(self, "_jsonl_handles", {}).values():
            handle.close()
        self._jsonl_handles = {}

    def resume_dataset_state(self) -> None:
        """Resume dataset state by re-reading files"""
        self.serialize_dataset = not hasattr(self, "original_data_files")
        self._read_files_and_filter()

    def __len__(self) -> int:
        if self._records is not None:
            return len(self._records)
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Get item from dataset as a dictionary"""
        if self._records is None:
            row_dict = self.dataframe.iloc[item].to_dict()
        else:
            file_index, locator = self._records[item]
            data_file = self.data_files[file_index]
            if data_file.lower().endswith(".jsonl"):
                handle = self._jsonl_handles.get(file_index)
                if handle is None:
                    # Keep readers for random access; close() owns their lifetime.
                    handle = open(data_file, "rb")  # pylint: disable=consider-using-with
                    self._jsonl_handles[file_index] = handle
                handle.seek(locator)
                row_dict = self._parse_jsonl_record(handle.readline(), data_file, locator)
            else:
                row_dict = self._dataframes[file_index].iloc[locator].to_dict()

        # Pandas fills missing cells when samples have different reward_fn fields.
        reward_fn = row_dict.get("reward_fn")
        if pd.api.types.is_scalar(reward_fn) and pd.isna(reward_fn):
            row_dict.pop("reward_fn", None)

        # Mixed samples introduce missing answer and optional metadata cells.
        for field in ("label", "ground_truth", "reward_model", "metadata"):
            if field in row_dict:
                value = row_dict[field]
                if pd.api.types.is_scalar(value) and pd.isna(value):
                    row_dict[field] = None

        # add index for each prompt if available
        extra_info = row_dict.get("extra_info")
        index = extra_info.get("index", item) if isinstance(extra_info, dict) else item
        row_dict["index"] = index

        return row_dict

    def __getstate__(self) -> dict[str, Any]:
        """Retain lightweight indices, excluding open handles and eager frames."""
        state = self.__dict__.copy()
        state["_jsonl_handles"] = {}
        if not self.serialize_dataset:
            state.pop("dataframe", None)
            state["_dataframes"] = {}
        return state


class TorchDataLoader(SequentialDataLoader):
    """
    A real data loader that uses PyTorch's DataLoader internally.

    This class wraps PyTorch's DataLoader and adapts it to the SequentialDataLoader interface.
    It reads data from parquet, CSV, or JSONL files and provides batched iteration.

    Config parameters (sub-config derived by the controller from new schema):
        - files: List of file paths to load (supports .parquet, .csv, .jsonl)
        - batch_size: Batch size for loading data
        - filter_prompts: Whether to filter prompts by length
        - max_prompt_length: Maximum prompt length in tokens
        - tokenizer: HF tokenizer name_or_path (required when filter_prompts=true)
        - prompt_key: Key for prompt in data dict
        - shuffle: Whether to shuffle data
        - seed: Base random seed. Cycling epoch N shuffles with seed+N.
        - drop_last: Whether to drop last incomplete batch
        - cycle: Internal controller-derived flag. Step-based training cycles
          across shuffled epochs; validation remains finite.
    """

    def __init__(self, config: DictConfig, is_validate: bool = False) -> None:
        super().__init__(config, is_validate)
        self._config = config

        # Get data loading parameters (controller derives these from
        # data.train.* / data.* / models.actor.tokenizer; see
        # Controller._derive_train_dataloader_config /
        # _derive_validate_dataloader_config).
        self._data_files = list(config.files)
        if not self._data_files:
            raise ValueError("files must be specified in dataloader config")

        self._filter_prompts = config.filter_prompts
        self._max_prompt_length = config.max_prompt_length
        self._prompt_key = config.prompt_key
        self._shuffle = config.shuffle
        self._drop_last = config.drop_last
        self._cycle = bool(config.get("cycle", False)) and not self._is_validate
        self._seed = int(config.get("seed", 42))
        self._epoch_id = 0
        self._pending_items: list[dict[str, Any]] = []

        self._is_sft = bool(config.get("is_sft", False))
        if self._is_sft:
            raise ValueError(
                "SFT is supported only through SFTStreamingDataLoader; "
                "NorthRL does not perform local SFT packing"
            )

        # Initialize tokenizer if needed for filtering. Schema validator already
        # enforces models.actor.tokenizer != null when data.train.filter_prompts=true.
        self._tokenizer = None
        self._tokenizer_path = config.get("tokenizer")
        if self._filter_prompts:
            self._ensure_tokenizer()

        # Create dataset
        self._dataset = FileDataset(
            data_files=self._data_files,
            filter_prompts=self._filter_prompts,
            max_prompt_length=self._max_prompt_length,
            tokenizer=self._tokenizer,
            prompt_key=self._prompt_key,
        )
        if self._cycle and len(self._dataset) == 0:
            raise ValueError("Cannot cycle an empty training dataset")

        # Create a dedicated random generator for this dataloader to ensure deterministic,
        # independent shuffling that doesn't interfere with other dataloaders

        self._generator = None
        if self._shuffle:
            self._generator = torch.Generator()
            self._generator.manual_seed(self._seed + self._epoch_id)
            logger.info(
                f"TorchDataLoader: Created dedicated random generator with seed={self._seed} "
                f"(is_validate={is_validate})"
            )

        # In cycle mode the epoch tail must remain visible so it can be joined
        # with the beginning of the next epoch into one full training batch.
        # Finite loaders keep the configured drop_last behavior.
        self._dataloader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last and not self._cycle,
            collate_fn=self._collate_fn,
            generator=self._generator,  # Use dedicated generator for isolated random state
        )

        # Iterator for sequential access
        self._dataloader_iter = self._new_epoch_iterator()
        self._is_exhausted = False

        logger.info(
            f"TorchDataLoader initialized - "
            f"is_validate: {self._is_validate}, "
            f"files: {len(self._data_files)}, "
            f"dataset size: {len(self._dataset)}, "
            f"batch_size: {self._batch_size}, "
            f"shuffle: {self._shuffle}, "
            f"cycle: {self._cycle}, "
            f"epoch_id: {self._epoch_id}, "
            f"filter_prompts: {self._filter_prompts}"
        )

    @staticmethod
    def _collate_fn(batch: List[dict]) -> List[dict]:
        """
        Collate function that simply returns the batch as a list of dicts.
        This avoids PyTorch's default collation which tries to stack tensors.

        Args:
            batch: List of data items from dataset

        Returns:
            List of data items unchanged
        """
        return batch

    @override
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        """
        Fetch one batch of items from the PyTorch DataLoader.

        Returns:
            list[dict[str, Any]]: List of data items (up to batch_size)
        """
        if not self._cycle:
            if self._is_exhausted:
                return []

            try:
                batch = next(self._dataloader_iter)
            except StopIteration:
                self._is_exhausted = True
                logger.info("TorchDataLoader: All data has been consumed")
                return []

            # A short batch from this map-style loader is the final batch.
            # Ordered consumers must see exhaustion after claiming its items,
            # without needing to release another batch just to discover EOF.
            if len(batch) < self._batch_size:
                self._is_exhausted = True
            return batch

        # A step-based training source is conceptually infinite. Preserve the
        # complete shuffled order of each epoch, and fill an epoch-tail batch
        # from the beginning of the next reshuffled epoch (Slime semantics).
        batch = self._pending_items
        self._pending_items = []
        while len(batch) < self._batch_size:
            try:
                chunk = next(self._dataloader_iter)
            except StopIteration:
                self._advance_epoch()
                continue

            needed = self._batch_size - len(batch)
            batch.extend(chunk[:needed])
            if len(chunk) > needed:
                self._pending_items.extend(chunk[needed:])

        return batch

    def _new_epoch_iterator(self):
        """Create an iterator whose shuffle is deterministic for the current epoch."""
        if self._generator is not None:
            self._generator.manual_seed(self._seed + self._epoch_id)
        return iter(self._dataloader)

    def _advance_epoch(self) -> None:
        """Advance a cycling training source and reshuffle the complete dataset."""
        self._epoch_id += 1
        self._dataloader_iter = self._new_epoch_iterator()
        logger.info(
            "TorchDataLoader: Starting epoch_id=%d with shuffle_seed=%d",
            self._epoch_id,
            self._seed + self._epoch_id,
        )

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if not self._tokenizer_path:
            raise ValueError("messages-format SFT data requires models.actor.tokenizer")

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        logger.info("TorchDataLoader: Loaded tokenizer from %s", self._tokenizer_path)
        return self._tokenizer

    @override
    def is_finished(self) -> bool:
        """
        Return True when all data has been fetched and buffer is empty.

        Cycling training loaders remain unfinished until Controller reaches
        ``training.total_steps``. Finite loaders check that:
        1. Main DataLoader is exhausted
        2. Current buffer is fully consumed

        Returns:
            True if all data has been fetched and buffer is empty, False otherwise
        """
        with self._lock:
            return self._is_exhausted and self._buffer_index >= len(self._data_buffer)

    def get_dataset_size(self) -> int:
        """
        Get the total size of the underlying dataset.

        Returns:
            Total number of items in the dataset
        """
        return len(self._dataset)

    @override
    def close(self) -> None:
        self._dataset.close()

    @override
    def _reset_iterator(self) -> None:
        """
        Reset the PyTorch DataLoader iterator to the beginning.

        Restores epoch 0, clears cross-epoch pending items, creates a new
        deterministic iterator, and resets the exhausted flag.
        """
        self._epoch_id = 0
        self._pending_items = []
        self._dataloader_iter = self._new_epoch_iterator()
        self._is_exhausted = False
