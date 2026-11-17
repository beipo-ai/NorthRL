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

"""Streaming, token-budgeted SFT data loading for Weaver training."""

from __future__ import annotations

import json
import logging
import random
import threading
from asyncio import get_running_loop
from asyncio import sleep as async_sleep
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, override

from omegaconf import DictConfig

try:
    from weaver import WeaverAPIError
except ImportError:  # pragma: no cover - reported by the holder with a clearer error
    WeaverAPIError = None  # type: ignore[assignment,misc]

from northrl.utils.async_utils import run_coro
from northrl.utils.path_utils import resolve_path_from_config

from .data_loader import SequentialDataLoader
from .sft_data import load_render_chat, prepare_sft_item
from .sft_mixing import (
    MIX_PLAN_VERSION,
    MIX_SHUFFLE_ALGORITHM,
    MixSource,
    VirtualSample,
    allocate_target_counts,
    build_epoch_order,
    mixing_plan_fingerprint,
    occurrence_datum_id,
)

logger = logging.getLogger(__name__)

_CURSOR_VERSION = 1
_MIX_CURSOR_VERSION = 2
_SHUFFLE_ALGORITHM = "python-mt19937-fisher-yates-v1"
_MANAGED_LENGTH_LOOKAHEAD = 256
_MANAGED_LENGTH_CACHE_SIZE = 4096
_RETRYABLE_MANAGED_VARIANT_CODES = frozenset(
    {"managed_dataset_variant_not_ready", "managed_datasets_unavailable"}
)


@dataclass(frozen=True)
class _SourceRecord:
    file_index: int
    record_index: int
    locator: int


@dataclass(frozen=True)
class _PreparedBatch:
    items: list[dict[str, Any]]
    input_token_count: int
    start_cursor: dict[str, Any]
    end_cursor: dict[str, Any]
    batch_id: str


@dataclass(frozen=True)
class _PreparedItem:
    item: dict[str, Any]
    start_cursor: dict[str, Any]
    end_cursor: dict[str, Any]


@dataclass
class _MixedSource:
    kind: str
    ratio: float
    local_source: _IndexedSFTSource | None = None
    dataset: str | None = None
    version: str | None = None
    size: int | None = None
    identity: dict[str, Any] | None = None
    recommended_ratio: float | None = None
    content_visibility: str | None = None


class _IndexedSFTSource:
    """Lightweight random-access index over JSONL records."""

    def __init__(self, files: list[str], config_file_path: str | None) -> None:
        if not files:
            raise ValueError("SFT data.train.files must not be empty")

        self._paths = []
        for configured_path in files:
            expanded_path = str(Path(configured_path).expanduser())
            resolved_path = resolve_path_from_config(expanded_path, config_file_path)
            assert resolved_path is not None
            self._paths.append(Path(resolved_path).resolve())
        if len(set(self._paths)) != len(self._paths):
            raise ValueError("SFT data.train.files must not contain duplicate paths")

        self._records: list[_SourceRecord] = []
        self._jsonl_handles: dict[int, Any] = {}

        for file_index, path in enumerate(self._paths):
            if not path.is_file():
                raise ValueError(f"SFT data file does not exist: {path}")
            suffix = path.suffix.lower()
            if suffix != ".jsonl":
                raise ValueError(f"SFT streaming supports only JSONL files; got {path}")

            self._index_jsonl(file_index, path)

        if not self._records:
            raise ValueError("SFT dataset contains no records")

    def _index_jsonl(self, file_index: int, path: Path) -> None:
        record_index = 0
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                self._records.append(_SourceRecord(file_index, record_index, offset))
                record_index += 1

    def __len__(self) -> int:
        return len(self._records)

    def read(self, logical_index: int) -> tuple[dict[str, Any], str]:
        record = self._records[logical_index]
        path = self._paths[record.file_index]
        if path.suffix.lower() == ".jsonl":
            handle = self._jsonl_handles.get(record.file_index)
            if handle is None:
                handle = path.open("rb")
                self._jsonl_handles[record.file_index] = handle
            handle.seek(record.locator)
            raw = handle.readline()
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{record.locator}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"SFT record must be an object at {path}:{record.locator}")
            return value, f"{record.file_index}:jsonl:{record.record_index}"
        raise AssertionError(f"Indexed non-JSONL SFT source: {path}")

    def close(self) -> None:
        for handle in self._jsonl_handles.values():
            handle.close()
        self._jsonl_handles.clear()


class SFTStreamingDataLoader(SequentialDataLoader):
    """Stream independent SFT samples into bounded token-budget batches.

    NorthRL deliberately does not pack samples. Each returned item remains an
    independent Datum boundary; Weaver performs the actual packing.
    """

    def __init__(self, config: DictConfig, is_validate: bool = False) -> None:
        if is_validate:
            raise ValueError("SFTStreamingDataLoader is training-only")
        super().__init__(config, is_validate)
        self._config = config
        config_file_path = config.get("config_file_path")
        configured_sources = config.get("sources")
        self._mixed_mode = configured_sources is not None
        self._source: _IndexedSFTSource | None = None
        self._mixed_sources: list[_MixedSource] = []
        if self._mixed_mode:
            for source_config in configured_sources:
                source_kind = str(source_config.type)
                if source_kind == "local":
                    local_source = _IndexedSFTSource(list(source_config.files), config_file_path)
                    self._mixed_sources.append(
                        _MixedSource(
                            kind="local",
                            ratio=float(source_config.ratio),
                            local_source=local_source,
                            size=len(local_source),
                            identity={
                                "type": "local",
                                "configured_ratio": float(source_config.ratio),
                            },
                        )
                    )
                elif source_kind == "managed":
                    self._mixed_sources.append(
                        _MixedSource(
                            kind="managed",
                            ratio=float(source_config.ratio),
                            dataset=str(source_config.dataset),
                            version=str(source_config.version),
                        )
                    )
                else:  # Config validation should make this unreachable.
                    raise ValueError(f"Unsupported SFT source type: {source_kind!r}")
        else:
            files = config.get("files")
            self._source = _IndexedSFTSource(list(files or []), config_file_path)
        self._seed = int(config.get("seed", 42))
        self._shuffle = bool(config.get("shuffle", True))
        self._max_sequence_length = int(config.max_sequence_length)
        self._token_budget = self._max_sequence_length * int(config.batch_size)
        window_samples = config.get("window_samples")
        self._window_samples = int(window_samples) if window_samples is not None else None
        self._managed_variant_ready_timeout = float(
            config.get("managed_variant_ready_timeout_seconds", 300.0)
        )
        self._managed_variant_poll_initial = float(
            config.get("managed_variant_poll_initial_seconds", 1.0)
        )
        self._managed_variant_poll_max = float(config.get("managed_variant_poll_max_seconds", 10.0))
        configured_epochs = config.get("epochs")
        self._max_epochs = int(configured_epochs) if configured_epochs is not None else None
        self._render_chat = None
        render_reference = config.get("render_chat")
        if render_reference:
            config_file_path = config.get("config_file_path")
            self._render_chat = load_render_chat(render_reference, config_file_path)
        self._tokenizer = None
        self._tokenizer_path = self._resolve_tokenizer_reference(
            config.get("tokenizer"), config_file_path
        )

        self._epoch = 0
        self._position = 0
        self._mix_plan_sources: list[MixSource] = []
        self._mix_plan_fingerprint: str | None = None
        self._order: list[int] | list[VirtualSample] = (
            [] if self._mixed_mode else self._make_epoch_order(self._epoch)
        )
        self._carry: _PreparedItem | None = None
        self._committed_cursor = (
            {} if self._mixed_mode else self._cursor(self._epoch, self._position)
        )
        self._active_batch: _PreparedBatch | None = None
        self._prepared_future: Future[_PreparedBatch | None] | None = None
        self._source_exhausted = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sft-tokenize")
        self._prepare_lock = threading.Lock()
        self._closed = False
        self._batch_serial = 0
        self._service_holder: Any | None = None
        self._managed_length_cache: OrderedDict[tuple[str, str, int], int] = OrderedDict()
        self._optimizer_window_count: int | None = None
        self._prepared = not self._mixed_mode

        if self._mixed_mode:
            logger.info(
                "Mixed SFTStreamingDataLoader initialized: sources=%d total_samples=%d "
                "token_budget=%d max_sequence_length=%d",
                len(self._mixed_sources),
                int(config.total_samples),
                self._token_budget,
                self._max_sequence_length,
            )
        else:
            assert self._source is not None
            logger.info(
                "SFTStreamingDataLoader initialized: records=%d token_budget=%d "
                "max_sequence_length=%d shuffle=%s",
                len(self._source),
                self._token_budget,
                self._max_sequence_length,
                self._shuffle,
            )

    @staticmethod
    def _resolve_tokenizer_reference(
        reference: str | None, config_file_path: str | None
    ) -> str | None:
        """Resolve an existing local tokenizer while preserving remote model IDs."""

        if reference is None:
            return None
        expanded = Path(reference).expanduser()
        if expanded.is_absolute() or expanded.exists():
            return str(expanded.resolve())
        if config_file_path is not None:
            recipe_candidate = Path(config_file_path).expanduser().resolve().parent / expanded
            if recipe_candidate.exists():
                return str(recipe_candidate.resolve())
        return reference

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if not self._tokenizer_path:
            raise ValueError("messages-format SFT data requires models.actor.tokenizer")
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        return self._tokenizer

    def set_service_holder(self, service_holder: Any) -> None:
        """Attach the shared Weaver holder used for managed-only metadata calls."""

        self._service_holder = service_holder

    async def prepare(self) -> None:
        """Authorize managed sources and freeze the resumable virtual plan."""

        if self._prepared:
            return
        if not self._mixed_mode:
            self._prepared = True
            return

        for source in self._mixed_sources:
            if source.kind != "managed":
                continue
            if self._service_holder is None:
                raise RuntimeError("Managed SFT sources require a Weaver service holder")
            assert source.dataset is not None and source.version is not None
            info = await self._service_holder.get_managed_dataset_info(
                name=source.dataset,
                version=source.version,
            )
            returned_name = self._metadata_value(info, "name")
            returned_version = self._metadata_value(info, "version")
            if returned_name != source.dataset or returned_version != source.version:
                raise ValueError(
                    "Weaver managed dataset catalog returned a mismatched identity: "
                    f"got=({returned_name!r}, {returned_version!r}) "
                    f"expected=({source.dataset!r}, {source.version!r})"
                )
            status = self._metadata_value(info, "status")
            status = getattr(status, "value", status)
            if status != "published":
                raise ValueError(
                    f"Managed dataset {source.dataset!r} version {source.version!r} "
                    "is not published"
                )
            sample_count = self._metadata_value(info, "sample_count")
            if (
                not isinstance(sample_count, int)
                or isinstance(sample_count, bool)
                or sample_count <= 0
            ):
                raise ValueError(
                    f"Managed dataset {source.dataset!r} version {source.version!r} "
                    f"has invalid sample_count={sample_count!r}"
                )
            recommended_ratio = self._metadata_value(info, "recommended_ratio")
            if not isinstance(recommended_ratio, (int, float)) or isinstance(
                recommended_ratio, bool
            ):
                recommended_ratio = None
            source.recommended_ratio = recommended_ratio
            content_visibility = self._metadata_value(info, "content_visibility")
            content_visibility = getattr(content_visibility, "value", content_visibility)
            if content_visibility not in {"protected", "public"}:
                raise RuntimeError(
                    f"Managed dataset {source.dataset!r} version {source.version!r} "
                    "has no valid content_visibility; upgrade Weaver server and SDK together"
                )
            source.content_visibility = content_visibility
            source.size = sample_count
            source.identity = {
                "type": "managed",
                "dataset": source.dataset,
                "version": source.version,
                "sample_count": sample_count,
                "configured_ratio": source.ratio,
            }

        # Resolve one model-bound sample per managed source before freezing the
        # plan. This is an opaque compatibility check for the pinned tokenizer,
        # chat template, and max-length policy; no managed tokens enter NorthRL.
        compatibility_refs = [
            {
                "dataset": source.dataset,
                "version": source.version,
                "sample_idx": 0,
            }
            for source in self._mixed_sources
            if source.kind == "managed"
        ]
        if compatibility_refs:
            assert self._service_holder is not None
            resolved = await self._resolve_managed_compatibility_lengths(compatibility_refs)
            self._cache_managed_length_results(resolved)

        ratios = [source.ratio for source in self._mixed_sources]
        total_samples = int(self._config.total_samples)
        targets = allocate_target_counts(total_samples, ratios)
        self._mix_plan_sources = []
        for source, target_count in zip(self._mixed_sources, targets):
            assert source.size is not None and source.identity is not None
            self._mix_plan_sources.append(
                MixSource(
                    identity=dict(source.identity),
                    size=source.size,
                    target_count=target_count,
                )
            )

        source_plan_fingerprint = mixing_plan_fingerprint(
            self._mix_plan_sources,
            seed=self._seed,
            total_samples=total_samples,
        )
        self._mix_plan_fingerprint = source_plan_fingerprint
        self._order = self._make_epoch_order(0)
        self._committed_cursor = self._cursor(0, 0)
        self._prepared = True

        logger.info(
            "Prepared deterministic mixed SFT plan: total_samples=%d algorithm=%s sources=%s",
            total_samples,
            MIX_SHUFFLE_ALGORITHM,
            [
                {
                    **source.identity,
                    "recommended_ratio": self._mixed_sources[index].recommended_ratio,
                    "planned_count": (
                        total_samples * self._mixed_sources[index].ratio / sum(ratios)
                    ),
                    "realized_count": source.target_count,
                }
                for index, source in enumerate(self._mix_plan_sources)
            ],
        )

    async def count_optimizer_windows(self) -> int:
        """Materialize the finite epoch plan once and return its exact window count."""

        if not self._prepared:
            raise RuntimeError("SFT dataloader must be prepared before counting optimizer windows")
        if self._max_epochs is None:
            raise RuntimeError(
                "A finite training.epochs value is required to count optimizer windows"
            )
        if self._prepared_future is not None or self._active_batch is not None:
            raise RuntimeError("Optimizer windows must be counted before SFT iteration starts")
        if self._optimizer_window_count is not None:
            return self._optimizer_window_count

        loop = get_running_loop()
        count = await loop.run_in_executor(self._executor, self._count_optimizer_windows_sync)
        if count <= 0:
            raise RuntimeError("Finite SFT epoch plan produced no optimizer windows")
        self._optimizer_window_count = count
        logger.info(
            "Counted exact SFT optimizer-window horizon: epochs=%d windows=%d token_budget=%d",
            self._max_epochs,
            count,
            self._token_budget,
        )
        return count

    def _count_optimizer_windows_sync(self) -> int:
        """Use the runtime materializer so planning and training boundaries are identical."""

        count = 0
        try:
            while self._prepare_next_batch() is not None:
                count += 1
            return count
        finally:
            self._reset_iterator()
            self._batch_serial = 0

    async def _resolve_managed_compatibility_lengths(
        self, refs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Wait boundedly for asynchronously-built model-bound variants.

        The SDK performs one HTTP attempt for each call. NorthRL retries only the
        two explicit, retryable readiness codes; authorization, identity,
        protocol, and all other failures remain fail-fast.
        """

        assert self._service_holder is not None
        deadline = monotonic() + self._managed_variant_ready_timeout
        delay = self._managed_variant_poll_initial
        attempt = 1
        safe_sources = [{"dataset": ref["dataset"], "version": ref["version"]} for ref in refs]

        while True:
            try:
                return await self._service_holder.resolve_managed_sample_ref_lengths(refs)
            except Exception as exc:
                is_retryable_readiness = (
                    WeaverAPIError is not None
                    and isinstance(exc, WeaverAPIError)
                    and getattr(exc, "code", None) in _RETRYABLE_MANAGED_VARIANT_CODES
                    and getattr(exc, "retryable", None) is True
                )
                if not is_retryable_readiness:
                    raise

                remaining = deadline - monotonic()
                if remaining <= 0:
                    code = getattr(exc, "code", "managed_dataset_variant_not_ready")
                    raise TimeoutError(
                        "Timed out waiting for Weaver managed dataset variants: "
                        f"sources={safe_sources!r} status={code!r}"
                    ) from exc

                sleep_seconds = min(delay, self._managed_variant_poll_max, remaining)
                logger.warning(
                    "Waiting for Weaver managed dataset variants: sources=%s status=%s "
                    "attempt=%d retry_in_seconds=%.3f",
                    safe_sources,
                    getattr(exc, "code", "managed_dataset_variant_not_ready"),
                    attempt,
                    sleep_seconds,
                )
                await async_sleep(sleep_seconds)
                delay = min(delay * 2, self._managed_variant_poll_max)
                attempt += 1

    @staticmethod
    def _metadata_value(info: Any, field_name: str) -> Any:
        if isinstance(info, dict):
            return info.get(field_name)
        return getattr(info, field_name, None)

    def _make_epoch_order(self, epoch: int) -> list[int] | list[VirtualSample]:
        if self._mixed_mode:
            if not self._mix_plan_sources:
                raise RuntimeError("Mixed SFT dataloader must be prepared before iteration")
            return build_epoch_order(self._mix_plan_sources, seed=self._seed, epoch=epoch)

        assert self._source is not None
        order = list(range(len(self._source)))
        if self._shuffle:
            random.Random(self._seed + epoch).shuffle(order)
        return order

    def _cursor(self, epoch: int, position: int) -> dict[str, Any]:
        if self._mixed_mode:
            if self._mix_plan_fingerprint is None:
                raise RuntimeError("Mixed SFT dataloader must be prepared before creating a cursor")
            return {
                "version": _MIX_CURSOR_VERSION,
                "mix_plan_version": MIX_PLAN_VERSION,
                "mix_plan_fingerprint": self._mix_plan_fingerprint,
                "shuffle_algorithm": MIX_SHUFFLE_ALGORITHM,
                "seed": self._seed,
                "total_samples": int(self._config.total_samples),
                "sources": [
                    {
                        **source.identity,
                        "size": source.size,
                        "target_count": source.target_count,
                    }
                    for source in self._mix_plan_sources
                ],
                "epoch": epoch,
                "position_in_epoch": position,
            }

        assert self._source is not None
        return {
            "version": _CURSOR_VERSION,
            "shuffle_algorithm": _SHUFFLE_ALGORITHM,
            "seed": self._seed,
            "epoch": epoch,
            "position_in_epoch": position,
        }

    def _advance_source(self) -> _PreparedItem | None:
        if self._mixed_mode and not self._prepared:
            raise RuntimeError("Mixed SFT dataloader must be prepared before iteration")
        if self._position >= len(self._order):
            return None
        plan_position = self._position
        start_cursor = self._cursor(self._epoch, plan_position)
        selected = self._order[plan_position]
        self._position += 1

        if self._mixed_mode:
            assert isinstance(selected, VirtualSample)
            source = self._mixed_sources[selected.source_ordinal]
            if source.kind == "managed":
                assert source.dataset is not None and source.version is not None
                sample_ref = {
                    "dataset": source.dataset,
                    "version": source.version,
                    "sample_idx": selected.sample_idx,
                }
                prepared = {
                    "sample_ref": sample_ref,
                    "input_token_count": self._resolve_managed_input_token_count(
                        selected,
                    ),
                    "datum_id": occurrence_datum_id(
                        self._require_mix_plan_fingerprint(),
                        epoch=self._epoch,
                        position=plan_position,
                    ),
                    # Internal policy carrier. It is kept next to (not inside)
                    # SampleRef and is consumed before the SDK wire boundary.
                    "content_visibility": source.content_visibility,
                    "index": (f"managed:{source.dataset}:{source.version}:{selected.sample_idx}"),
                }
                return _PreparedItem(
                    item=prepared,
                    start_cursor=start_cursor,
                    end_cursor=self._cursor(self._epoch, self._position),
                )

            assert source.local_source is not None
            raw, source_id = source.local_source.read(selected.sample_idx)
            source_id = f"{selected.source_ordinal}:{source_id}"
        else:
            assert self._source is not None and isinstance(selected, int)
            raw, source_id = self._source.read(selected)

        raw.setdefault("index", source_id)
        try:
            tokenizer = self._tokenizer
            if raw.get("input_ids") is None or self._render_chat is not None:
                tokenizer = self._ensure_tokenizer()
            prepared = prepare_sft_item(
                raw,
                tokenizer=tokenizer,
                render_chat=self._render_chat,
                max_sequence_length=self._max_sequence_length,
            )
        except Exception as exc:
            raise ValueError(f"Invalid SFT item at source {source_id}") from exc
        # These are internal managed-carrier fields, not user JSONL metadata.
        # Local rows cannot opt into SampleRef semantics or inject attached
        # managed loss inputs by choosing reserved field names.
        prepared.pop("sample_ref", None)
        prepared.pop("loss_fn_inputs", None)
        prepared.pop("input_token_count", None)
        prepared.pop("content_visibility", None)
        if self._mixed_mode:
            prepared_input_ids = prepared.get("input_ids")
            if not isinstance(prepared_input_ids, list):
                raise ValueError("Prepared local SFT item is missing input_ids")
            prepared["input_token_count"] = len(prepared_input_ids) - 1
            prepared["datum_id"] = occurrence_datum_id(
                self._require_mix_plan_fingerprint(),
                epoch=self._epoch,
                position=plan_position,
            )
        else:
            # Legacy local-only rows historically treated datum_id as ordinary
            # unused metadata. Keep them on the legacy SDK wire shape.
            prepared.pop("datum_id", None)
        return _PreparedItem(
            item=prepared,
            start_cursor=start_cursor,
            end_cursor=self._cursor(self._epoch, self._position),
        )

    def _require_mix_plan_fingerprint(self) -> str:
        if self._mix_plan_fingerprint is None:
            raise RuntimeError("Mixed SFT plan fingerprint is unavailable before prepare")
        return self._mix_plan_fingerprint

    def _resolve_managed_input_token_count(self, selected: VirtualSample) -> int:
        source = self._mixed_sources[selected.source_ordinal]
        assert source.dataset is not None and source.version is not None
        key = (source.dataset, source.version, selected.sample_idx)
        cached = self._managed_length_cache.get(key)
        if cached is not None:
            self._managed_length_cache.move_to_end(key)
            return cached
        if self._service_holder is None:
            raise RuntimeError("Managed SFT sources require a Weaver service holder")

        candidates: list[VirtualSample] = [selected]
        for candidate in self._order[
            self._position : self._position + _MANAGED_LENGTH_LOOKAHEAD - 1
        ]:
            assert isinstance(candidate, VirtualSample)
            if self._mixed_sources[candidate.source_ordinal].kind == "managed":
                candidates.append(candidate)

        refs: list[dict[str, Any]] = []
        requested_keys: set[tuple[str, str, int]] = set()
        for candidate in candidates:
            candidate_source = self._mixed_sources[candidate.source_ordinal]
            assert candidate_source.dataset is not None and candidate_source.version is not None
            candidate_key = (
                candidate_source.dataset,
                candidate_source.version,
                candidate.sample_idx,
            )
            if candidate_key in self._managed_length_cache or candidate_key in requested_keys:
                continue
            requested_keys.add(candidate_key)
            refs.append(
                {
                    "dataset": candidate_key[0],
                    "version": candidate_key[1],
                    "sample_idx": candidate_key[2],
                }
            )

        resolved = run_coro(self._service_holder.resolve_managed_sample_ref_lengths(refs))
        self._cache_managed_length_results(resolved)

        if key not in self._managed_length_cache:
            raise ValueError(f"Weaver managed length response omitted requested ref {key!r}")
        return self._managed_length_cache[key]

    def _cache_managed_length_results(self, resolved: list[dict[str, Any]]) -> None:
        """Validate safe length metadata and place it in the bounded LRU."""

        for result in resolved:
            sample_ref = result["sample_ref"]
            result_key = (
                sample_ref["dataset"],
                sample_ref["version"],
                sample_ref["sample_idx"],
            )
            count = result["input_token_count"]
            if count >= self._max_sequence_length:
                raise ValueError(
                    "Managed input_token_count must describe shifted input positions after "
                    "the model-bound max-length policy: "
                    f"got={count} max_allowed={self._max_sequence_length - 1}"
                )
            self._managed_length_cache[result_key] = count
            self._managed_length_cache.move_to_end(result_key)
            while len(self._managed_length_cache) > _MANAGED_LENGTH_CACHE_SIZE:
                self._managed_length_cache.popitem(last=False)

    def _prepare_next_batch(self) -> _PreparedBatch | None:
        with self._prepare_lock:
            while True:
                start_cursor = (
                    self._carry.start_cursor
                    if self._carry is not None
                    else self._cursor(self._epoch, self._position)
                )
                items: list[dict[str, Any]] = []
                input_tokens = 0
                end_cursor = start_cursor

                if self._carry is not None:
                    carried = self._carry
                    self._carry = None
                    items.append(carried.item)
                    input_tokens += self._item_input_token_count(carried.item)
                    end_cursor = carried.end_cursor

                emit_batch = self._window_full(items, input_tokens)
                while not emit_batch:
                    next_item = self._advance_source()
                    if next_item is None:
                        # A mixed plan has an explicit sample count. Preserve its
                        # final non-empty group even when variable sample lengths
                        # leave token-budget headroom. It is an optimizer-window
                        # tail and is never filled from the next epoch.
                        if self._mixed_mode and items:
                            emit_batch = True
                            break

                        # Legacy local-only token-budget tails are intentionally
                        # dropped and never filled from the next epoch.
                        dropped = len(items)
                        dropped_tokens = input_tokens
                        if dropped:
                            logger.info(
                                "Dropping EOF SFT batch: samples=%d input_tokens=%d budget=%d",
                                dropped,
                                dropped_tokens,
                                self._token_budget,
                            )
                        # Starting an epoch at zero and reaching EOF without an
                        # exact/overflow boundary would drop that epoch forever.
                        if start_cursor["position_in_epoch"] == 0:
                            raise RuntimeError(
                                "SFT epoch cannot produce a full token-budget batch; "
                                f"epoch input tokens are below budget={self._token_budget}"
                            )
                        self._epoch += 1
                        self._position = 0
                        self._carry = None
                        if self._max_epochs is not None and self._epoch >= self._max_epochs:
                            self._source_exhausted = True
                            return None
                        self._order = self._make_epoch_order(self._epoch)

                        items = []
                        break

                    candidate_tokens = self._item_input_token_count(next_item.item)
                    if items and input_tokens + candidate_tokens > self._token_budget:
                        self._carry = next_item
                        emit_batch = True
                        break
                    items.append(next_item.item)
                    input_tokens += candidate_tokens
                    end_cursor = next_item.end_cursor
                    if self._window_full(items, input_tokens):
                        emit_batch = True

                if not emit_batch:
                    continue

                self._batch_serial += 1
                batch_id = f"sft-{start_cursor['epoch']}-{self._batch_serial}"
                for item in items:
                    item["sft_batch_id"] = batch_id
                return _PreparedBatch(
                    items=items,
                    input_token_count=input_tokens,
                    start_cursor=start_cursor,
                    end_cursor=end_cursor,
                    batch_id=batch_id,
                )

    def _window_full(self, items: list[dict[str, Any]], input_tokens: int) -> bool:
        # A sample cap fixes the per-step sample count; the token budget still bounds it.
        if self._window_samples is not None and len(items) >= self._window_samples:
            return True
        return input_tokens == self._token_budget

    @staticmethod
    def _item_input_token_count(item: dict[str, Any]) -> int:
        count = item.get("input_token_count")
        if count is None:
            count = len(item["input_ids"]) - 1
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"SFT item has invalid input_token_count={count!r}")
        return count

    def _start_prefetch(self) -> None:
        if not self._prepared:
            raise RuntimeError("SFT dataloader must be prepared before iteration")
        if self._prepared_future is None and not self._source_exhausted:
            self._prepared_future = self._executor.submit(self._prepare_next_batch)

    @override
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        if self._closed:
            return []
        self._start_prefetch()
        if self._prepared_future is None:
            return []
        prepared = self._prepared_future.result()
        self._prepared_future = None
        if prepared is None:
            return []
        self._active_batch = prepared
        # Exactly one lookahead: start N+1 only after N has been consumed from
        # the future and exposed to rollout workers.
        self._start_prefetch()
        logger.info(
            "Prepared SFT batch %s: samples=%d input_tokens=%d/%d",
            prepared.batch_id,
            len(prepared.items),
            prepared.input_token_count,
            self._token_budget,
        )
        return prepared.items

    def commit_batch(self, batch_id: str) -> None:
        with self._lock:
            if self._active_batch is None or self._active_batch.batch_id != batch_id:
                raise RuntimeError(
                    f"Cannot commit SFT batch {batch_id!r}; active batch is "
                    f"{getattr(self._active_batch, 'batch_id', None)!r}"
                )
            self._committed_cursor = dict(self._active_batch.end_cursor)
            self._active_batch = None

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            if not self._prepared:
                raise RuntimeError("SFT dataloader must be prepared before checkpointing")
            return dict(self._committed_cursor)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            if self._prepared_future is not None or self._active_batch is not None:
                raise RuntimeError("Cannot restore SFT dataloader after iteration has started")
            expected = self._cursor(0, 0)
            identity_keys = [
                "version",
                "shuffle_algorithm",
                "seed",
            ]
            if self._mixed_mode:
                identity_keys.extend(
                    [
                        "mix_plan_version",
                        "mix_plan_fingerprint",
                        "total_samples",
                        "sources",
                    ]
                )
            for key in identity_keys:
                if state.get(key) != expected[key]:
                    raise ValueError(
                        f"SFT dataloader resume state mismatch for {key}: "
                        f"{state.get(key)!r} != {expected[key]!r}"
                    )
            epoch = int(state["epoch"])
            position = int(state["position_in_epoch"])
            if epoch < 0 or not 0 <= position <= len(self._order):
                raise ValueError("Invalid SFT dataloader resume cursor")
            self._epoch = epoch
            self._position = position
            self._order = self._make_epoch_order(epoch)
            self._carry = None
            self._committed_cursor = dict(state)
            self._source_exhausted = self._max_epochs is not None and epoch >= self._max_epochs

    @override
    def skip_batches(self, num_batches: int) -> None:
        raise RuntimeError("SFT resume requires an exact dataloader cursor, not skip_batches")

    @override
    def _reset_iterator(self) -> None:
        if not self._prepared:
            raise RuntimeError("SFT dataloader must be prepared before reset")
        self._epoch = 0
        self._position = 0
        self._order = self._make_epoch_order(0)
        self._carry = None
        self._committed_cursor = self._cursor(0, 0)
        self._active_batch = None
        self._source_exhausted = False
        self._managed_length_cache.clear()

    @override
    def is_finished(self) -> bool:
        with self._lock:
            return (
                self._source_exhausted
                and self._prepared_future is None
                and self._active_batch is None
                and self._buffer_index >= len(self._data_buffer)
            )

    def close(self) -> None:
        self._closed = True
        if self._prepared_future is not None:
            self._prepared_future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._source is not None:
            self._source.close()
        for source in self._mixed_sources:
            if source.local_source is not None:
                source.local_source.close()
