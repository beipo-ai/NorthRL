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

"""Weighted prompt sampling from independent local RL task sources."""

import random
from dataclasses import dataclass, field
from itertools import accumulate
from math import isfinite
from typing import Any, override

import pandas as pd
from omegaconf import DictConfig

from .data_loader import SequentialDataLoader
from .torch_data_loader import FileDataset


@dataclass
class _Source:
    name: str
    dataset: FileDataset
    ratio: float
    rollout_profile: str | None
    rng: random.Random = field(default_factory=random.Random)
    order: list[int] = field(default_factory=list)
    cursor: int = 0


class MixtureDataLoader(SequentialDataLoader):
    """Choose a source by weight, then consume its own shuffled prompt stream.

    Source probabilities depend only on ``ratio``, independently of dataset
    sizes. Each source cycles through a full permutation before reshuffling.
    ``shuffle=false`` disables only these within-source permutations; source
    selection remains random. Separate RNGs make source choices independent
    of shuffling and permit deterministic reset/``skip_batches`` replay.

    This training-only loader returns raw inputs to SequentialDataLoader,
    which creates complete rollout groups. It preserves the original task_id
    because E2B task assets and evaluators use that identity. ``data_source``
    identifies the configured source without changing task paths.

    Controller-derived ``cycle=true`` produces full batches indefinitely.
    A finite loader draws the sum of filtered source sizes and honors
    ``drop_last`` for a short final batch.
    This finite budget does not promise full dataset coverage or fixed quotas.
    """

    def __init__(self, config: DictConfig, is_validate: bool = False) -> None:
        if is_validate or config.get("is_sft", False):
            raise ValueError("MixtureDataLoader supports RL training only")
        super().__init__(config, is_validate)
        self._seed = int(config.get("seed", 42))
        self._shuffle = bool(config.get("shuffle", True))
        self._cycle = bool(config.get("cycle", False))
        self._drop_last = bool(config.get("drop_last", False))
        self._is_exhausted: bool = False
        self._sources: list[_Source] = []
        self._source_rng = random.Random(self._seed)

        sources_config = config.get("sources")
        if not sources_config:
            raise ValueError("MixtureDataLoader requires nonempty sources")
        tokenizer = None
        if config.get("filter_prompts", False):
            if not config.get("tokenizer"):
                raise ValueError("Prompt filtering requires a tokenizer")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        names: set[str] = set()
        try:
            for source_config in sources_config:
                if source_config.get("type") != "local":
                    raise ValueError("MixtureDataLoader supports local sources only")
                name = source_config.get("name")
                if not isinstance(name, str) or not name or name != name.strip() or name in names:
                    raise ValueError(
                        "Mixture source names must be nonempty and unique without boundary whitespace"
                    )
                names.add(name)
                profile = source_config.get("rollout_profile")
                if profile is not None and (
                    not isinstance(profile, str) or not profile or profile != profile.strip()
                ):
                    raise ValueError(
                        "Source rollout_profile must be nonblank without boundary whitespace"
                    )
                ratio = float(source_config.get("ratio", 0))
                if not isfinite(ratio) or ratio <= 0:
                    raise ValueError(f"Source {name!r} ratio must be finite and positive")
                files = source_config.get("files")
                if not files:
                    raise ValueError(f"Source {name!r} requires files")
                dataset = FileDataset(
                    data_files=list(files),
                    filter_prompts=bool(config.get("filter_prompts", False)),
                    max_prompt_length=config.get("max_prompt_length", 1024),
                    tokenizer=tokenizer,
                    prompt_key=config.get("prompt_key", "prompt"),
                )
                self._sources.append(_Source(name, dataset, ratio, profile))
                if len(dataset) == 0:
                    raise ValueError(f"Mixture source {name!r} has no usable samples")

            self._dataset_size = sum(len(source.dataset) for source in self._sources)
            self._sample_limit = self._dataset_size
            # Scaling avoids overflow when otherwise valid weights have a large sum.
            max_ratio = max(source.ratio for source in self._sources)
            self._cumulative_weights = list(
                accumulate(source.ratio / max_ratio for source in self._sources)
            )
            self._reset_iterator()
        except Exception:
            self.close()
            raise

    @override
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        if self._is_exhausted:
            return []
        count = self._batch_size
        if not self._cycle:
            count = min(count, self._sample_limit - self._drawn_samples)
            if count == 0 or (count < self._batch_size and self._drop_last):
                self._is_exhausted = True
                return []

        batch = []
        for _ in range(count):
            source = self._source_rng.choices(
                self._sources, cum_weights=self._cumulative_weights, k=1
            )[0]
            if source.cursor == len(source.order):
                source.cursor = 0
                source.order = list(range(len(source.dataset)))
                if self._shuffle:
                    source.rng.shuffle(source.order)
            item = source.dataset[source.order[source.cursor]]
            source.cursor += 1
            item["data_source"] = source.name
            if source.rollout_profile is not None:
                row_profile = item.get("rollout_profile")
                if pd.api.types.is_scalar(row_profile) and pd.isna(row_profile):
                    row_profile = None
                if row_profile is not None and (
                    not isinstance(row_profile, str) or row_profile != source.rollout_profile
                ):
                    raise ValueError(
                        f"Source {source.name!r} rollout_profile {source.rollout_profile!r} "
                        f"conflicts with row rollout_profile {row_profile!r}"
                    )
                item["rollout_profile"] = source.rollout_profile
            batch.append(item)

        self._drawn_samples += count
        if not self._cycle and self._drawn_samples == self._sample_limit:
            self._is_exhausted = True
        return batch

    @override
    def _reset_iterator(self) -> None:
        self._source_rng.seed(self._seed)
        self._drawn_samples = 0
        self._is_exhausted = False
        for source in self._sources:
            # Preserve the historical seed domain so resumed runs keep their sample order.
            source.rng.seed(f"nexrl-mixture-source-v1:{self._seed}:{source.name}")
            source.order = list(range(len(source.dataset)))
            if self._shuffle:
                source.rng.shuffle(source.order)
            source.cursor = 0

    @override
    def is_finished(self) -> bool:
        with self._lock:
            return self._is_exhausted and self._buffer_index >= len(self._data_buffer)

    def get_dataset_size(self) -> int:
        """Return physical filtered sample count, independently of draw budget."""
        return self._dataset_size

    @override
    def close(self) -> None:
        for source in self._sources:
            source.dataset.close()
