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
Multi-dataset validation dataloader that wraps multiple BaseDataLoader instances.

This loader iterates over a list of validation datasets sequentially, returning
items tagged with their dataset names so downstream components can aggregate
metrics separately per dataset.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

from omegaconf import DictConfig

from .data_loader import BaseDataLoader
from .torch_data_loader import TorchDataLoader

logger = logging.getLogger(__name__)


class MultiDatasetValidationDataLoader(BaseDataLoader):
    """Wrap multiple validation datasets and expose them as a single dataloader.

    New schema: ``validate.data.datasets`` is a dict {name: {files, batch_size}};
    Controller projects each entry into a full TorchDataLoader sub-config
    (with prompt_key/max_prompt_length/tokenizer inherited and
    shuffle/drop_last/filter_prompts pinned to false). See
    Controller._derive_validate_dataloader_config.
    """

    def __init__(self, config: DictConfig, is_validate: bool = True) -> None:
        super().__init__(config, is_validate)
        datasets_cfg = config.get("datasets") or {}
        self._datasets: List[Dict[str, Any]] = []
        if not datasets_cfg:
            logger.info(
                "MultiDatasetValidationDataLoader: no datasets configured; "
                "validation is disabled"
            )
        for name, sub_cfg in datasets_cfg.items():
            loader = TorchDataLoader(sub_cfg, is_validate=True)
            self._datasets.append({"name": name, "loader": loader})
            logger.info("MultiDatasetValidationDataLoader: registered dataset '%s'", name)

        self._lock = threading.Lock()
        self._current_index = 0

    def set_activity_tracker(self, tracker):  # type: ignore[override]
        super().set_activity_tracker(tracker)
        for dataset in self._datasets:
            dataset["loader"].set_activity_tracker(tracker)

    def set_module_name(self, module_name: str):  # type: ignore[override]
        super().set_module_name(module_name)
        for dataset in self._datasets:
            dataset_name = dataset["name"]
            dataset["loader"].set_module_name(f"{module_name}/{dataset_name}")

    def get_next_item(self) -> dict[str, Any] | None:  # type: ignore[override]
        with self._lock:
            if not self._datasets:
                return None

            checked = 0
            total = len(self._datasets)
            while checked < total:
                dataset_entry = self._datasets[self._current_index]
                loader = dataset_entry["loader"]
                item = loader.get_next_item()
                if item is not None:
                    item["dataset_name"] = dataset_entry["name"]
                    item["is_val"] = True
                    return item

                if loader.is_finished():
                    checked += 1
                    self._current_index = (self._current_index + 1) % total
                    continue

                return None

            return None

    def can_return_item(self) -> bool:  # type: ignore[override]
        with self._lock:
            return any(dataset["loader"].can_return_item() for dataset in self._datasets)

    def is_finished(self) -> bool:  # type: ignore[override]
        return all(dataset["loader"].is_finished() for dataset in self._datasets)

    def release_next_batch(self) -> None:  # type: ignore[override]
        for dataset in self._datasets:
            dataset["loader"].release_next_batch()

    def reset(self) -> None:  # type: ignore[override]
        with self._lock:
            for dataset in self._datasets:
                dataset["loader"].reset()
            self._current_index = 0

    def skip_batches(self, num_batches: int) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "MultiDatasetValidationDataLoader does not support skip_batches(). "
            "Resume should be applied to the training dataloader."
        )
