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

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import DictConfig, OmegaConf
from weaver import WeaverAPIError

import northrl.data_loader.sft_streaming_data_loader as sft_loader_module
from northrl.data_loader.sft_streaming_data_loader import SFTStreamingDataLoader
from northrl.utils.async_utils import run_coro


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(marker: int, length: int = 5) -> dict:
    return {
        "input_ids": [marker, *range(marker + 1, marker + length)],
        "loss_mask": [0, *([1] * (length - 1))],
    }


def _config(
    path: Path,
    *,
    max_sequence_length: int = 5,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 7,
    epochs: int | None = None,
    window_samples: int | None = None,
) -> DictConfig:
    return OmegaConf.create(
        {
            "files": [str(path)],
            "batch_size": batch_size,
            "rollout_repeat_n": 1,
            "prompt_key": "prompt",
            "filter_prompts": False,
            "shuffle": shuffle,
            "drop_last": True,
            "max_prompt_length": max_sequence_length,
            "max_sequence_length": max_sequence_length,
            "tokenizer": None,
            "is_sft": True,
            "render_chat": None,
            "config_file_path": None,
            "seed": seed,
            "keep_batch_order": True,
            "epochs": epochs,
            "window_samples": window_samples,
        }
    )


def _mixed_config(path: Path, *, epochs: int | None = None) -> DictConfig:
    config = _config(path, max_sequence_length=5, batch_size=2, seed=19, epochs=epochs)
    config.files = None
    config.sources = [
        {"type": "local", "files": [str(path)], "ratio": 1.0},
        {
            "type": "managed",
            "dataset": "hq-math",
            "version": "2026-08",
            "ratio": 1.0,
        },
    ]
    config.total_samples = 8
    return config


class _ManagedDatasetHolder:
    def __init__(
        self,
        *,
        input_token_count: int = 4,
        recommended_ratio: float = 0.2,
        content_visibility: str | None = "protected",
    ) -> None:
        self.input_token_count = input_token_count
        self.recommended_ratio = recommended_ratio
        self.content_visibility = content_visibility
        self.catalog_requests: list[tuple[str, str]] = []
        self.length_requests: list[list[dict]] = []

    async def get_managed_dataset_info(self, *, name: str, version: str):
        self.catalog_requests.append((name, version))
        return SimpleNamespace(
            name=name,
            version=version,
            status="published",
            sample_count=4,
            recommended_ratio=self.recommended_ratio,
            content_visibility=self.content_visibility,
        )

    async def resolve_managed_sample_ref_lengths(self, refs: list[dict]):
        self.length_requests.append(refs)
        return [
            {
                "sample_ref": dict(ref),
                "input_token_count": self.input_token_count,
            }
            for ref in refs
        ]


class _ReadinessHolder(_ManagedDatasetHolder):
    def __init__(self, failures: list[Exception]) -> None:
        super().__init__()
        self.failures = list(failures)

    async def resolve_managed_sample_ref_lengths(self, refs: list[dict]):
        self.length_requests.append(refs)
        if self.failures:
            raise self.failures.pop(0)
        return [
            {
                "sample_ref": dict(ref),
                "input_token_count": self.input_token_count,
            }
            for ref in refs
        ]


def _weaver_error(
    *,
    code: str,
    retryable: bool,
    status_code: int = 409,
    message: str = "managed variant status",
) -> WeaverAPIError:
    return WeaverAPIError(
        status_code=status_code,
        code=code,
        message=message,
        retryable=retryable,
    )


def _consume_active_batch(loader: SFTStreamingDataLoader) -> list[dict]:
    items = []
    while True:
        item = loader.get_next_item()
        if item is None:
            return items
        items.append(item)


def _consume_finite_plan(loader: SFTStreamingDataLoader) -> list[list[dict]]:
    batches = []
    while True:
        batch = _consume_active_batch(loader)
        if not batch:
            return batches
        batches.append(batch)
        loader.commit_batch(batch[0]["sft_batch_id"])
        loader.release_next_batch()


def test_streams_independent_datums_to_token_budget_and_commits_cursor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40, 50, 60)])
    loader = SFTStreamingDataLoader(_config(path, max_sequence_length=5, batch_size=2))
    try:
        batch = _consume_active_batch(loader)

        assert [item["input_ids"][0] for item in batch] == [10, 20]
        assert len({item["sft_batch_id"] for item in batch}) == 1
        # The producer has already tokenized the overflow/carry item and starts
        # preparing N+1, but checkpoint state remains at the committed cursor.
        assert loader._prepared_future is not None
        assert loader.state_dict()["position_in_epoch"] == 0

        loader.commit_batch(batch[0]["sft_batch_id"])
        assert loader.state_dict()["position_in_epoch"] == 2
    finally:
        loader.close()


def test_local_rows_cannot_inject_managed_carrier_fields(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    records = []
    for marker in (10, 20, 30):
        record = _record(marker)
        record.update(
            {
                "sample_ref": {
                    "dataset": "untrusted",
                    "version": "v1",
                    "sample_idx": 0,
                },
                "datum_id": "untrusted-id",
                "input_token_count": 999,
                "content_visibility": "public",
                "loss_fn_inputs": {"old_logprobs": [0.0]},
            }
        )
        records.append(record)
    _write_jsonl(path, records)
    loader = SFTStreamingDataLoader(_config(path, max_sequence_length=5, batch_size=2))
    try:
        batch = _consume_active_batch(loader)
        assert len(batch) == 2
        assert all("sample_ref" not in item for item in batch)
        assert all("datum_id" not in item for item in batch)
        assert all("input_token_count" not in item for item in batch)
        assert all("content_visibility" not in item for item in batch)
        assert all("loss_fn_inputs" not in item for item in batch)
    finally:
        loader.close()


def test_mixed_loader_keeps_managed_refs_opaque_and_whole(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    holder = _ManagedDatasetHolder()
    loader = SFTStreamingDataLoader(_mixed_config(path, epochs=1))
    loader.set_service_holder(holder)
    run_coro(loader.prepare())

    try:
        all_items = []
        for _ in range(4):
            batch = _consume_active_batch(loader)
            assert len(batch) == 2
            assert sum(item["input_token_count"] for item in batch) == 8
            all_items.extend(batch)
            loader.commit_batch(batch[0]["sft_batch_id"])
            loader.release_next_batch()

        managed = [item for item in all_items if "sample_ref" in item]
        local = [item for item in all_items if "input_ids" in item]
        assert len(managed) == 4
        assert len(local) == 4
        assert len({item["datum_id"] for item in all_items}) == 8
        assert all("input_ids" not in item and "loss_mask" not in item for item in managed)
        assert all(item["content_visibility"] == "protected" for item in managed)
        assert all("content_visibility" not in item for item in local)
        assert all(item["input_token_count"] == 4 for item in all_items)
        assert holder.catalog_requests == [("hq-math", "2026-08")]
        assert holder.length_requests
        assert all(len(request) <= 4 for request in holder.length_requests)
    finally:
        loader.close()


def test_managed_only_loader_never_constructs_token_fields(tmp_path: Path) -> None:
    path = tmp_path / "unused-local.jsonl"
    _write_jsonl(path, [_record(10)])
    config = _mixed_config(path, epochs=1)
    config.sources = [
        {
            "type": "managed",
            "dataset": "hq-math",
            "version": "2026-08",
            "ratio": 1.0,
        }
    ]
    config.total_samples = 4
    loader = SFTStreamingDataLoader(config)
    loader.set_service_holder(_ManagedDatasetHolder())
    run_coro(loader.prepare())

    try:
        items = []
        for _ in range(2):
            batch = _consume_active_batch(loader)
            items.extend(batch)
            loader.commit_batch(batch[0]["sft_batch_id"])
            loader.release_next_batch()

        assert len(items) == 4
        assert all(
            set(item) >= {"sample_ref", "input_token_count", "datum_id", "content_visibility"}
            for item in items
        )
        assert all(item["content_visibility"] == "protected" for item in items)
        assert all("input_ids" not in item and "loss_mask" not in item for item in items)
    finally:
        loader.close()


def test_mixed_loader_emits_optimizer_window_tail_smaller_than_batch_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unused-local.jsonl"
    _write_jsonl(path, [_record(10)])
    config = _mixed_config(path, epochs=1)
    config.sources = [
        {
            "type": "managed",
            "dataset": "hq-math",
            "version": "2026-08",
            "ratio": 1.0,
        }
    ]
    config.batch_size = 5
    config.total_samples = 7
    loader = SFTStreamingDataLoader(config)
    loader.set_service_holder(_ManagedDatasetHolder())
    run_coro(loader.prepare())

    try:
        assert run_coro(loader.count_optimizer_windows()) == 2
        items = []
        batch_sizes = []
        for _ in range(2):
            batch = _consume_active_batch(loader)
            assert batch[0]["sft_batch_id"] == f"sft-0-{len(batch_sizes) + 1}"
            batch_sizes.append(len(batch))
            items.extend(batch)
            loader.commit_batch(batch[0]["sft_batch_id"])
            loader.release_next_batch()

        assert batch_sizes == [6, 1]
        assert len(items) == 7
        assert _consume_active_batch(loader) == []
        assert loader.is_finished()
    finally:
        loader.close()


def test_count_optimizer_windows_preserves_unplanned_batches_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "unused-local.jsonl"
    _write_jsonl(path, [_record(10)])
    config = _mixed_config(path, epochs=1)
    config.sources = [
        {
            "type": "managed",
            "dataset": "hq-math",
            "version": "2026-08",
            "ratio": 1.0,
        }
    ]
    config.batch_size = 3
    config.total_samples = 7

    class _VariableLengthHolder(_ManagedDatasetHolder):
        async def resolve_managed_sample_ref_lengths(self, refs: list[dict]):
            requested = [dict(ref) for ref in refs]
            self.length_requests.append(requested)
            lengths = {0: 1, 1: 4, 2: 2, 3: 3}
            return [
                {
                    "sample_ref": dict(ref),
                    "input_token_count": lengths[ref["sample_idx"]],
                }
                for ref in refs
            ]

    planned = SFTStreamingDataLoader(config)
    baseline = SFTStreamingDataLoader(config)
    planned.set_service_holder(_VariableLengthHolder())
    baseline.set_service_holder(_VariableLengthHolder())
    run_coro(planned.prepare())
    run_coro(baseline.prepare())
    try:
        horizon = run_coro(planned.count_optimizer_windows())
        planned_batches = _consume_finite_plan(planned)
        baseline_batches = _consume_finite_plan(baseline)

        assert len(planned_batches) == horizon
        assert planned_batches == baseline_batches
        assert [batch[0]["sft_batch_id"] for batch in planned_batches] == [
            f"sft-0-{serial}" for serial in range(1, horizon + 1)
        ]
    finally:
        planned.close()
        baseline.close()


def test_count_optimizer_windows_scans_managed_lengths_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unused-local.jsonl"
    _write_jsonl(path, [_record(10)])
    config = _mixed_config(path, epochs=1)
    config.sources = [
        {
            "type": "managed",
            "dataset": "hq-math",
            "version": "2026-08",
            "ratio": 1.0,
        }
    ]
    config.batch_size = 2
    config.total_samples = 10

    class _TenSampleHolder(_ManagedDatasetHolder):
        async def get_managed_dataset_info(self, *, name: str, version: str):
            info = await super().get_managed_dataset_info(name=name, version=version)
            info.sample_count = 10
            return info

        async def resolve_managed_sample_ref_lengths(self, refs: list[dict]):
            requested = [dict(ref) for ref in refs]
            self.length_requests.append(requested)
            return [
                {"sample_ref": dict(ref), "input_token_count": ref["sample_idx"] % 4 + 1}
                for ref in refs
            ]

    monkeypatch.setattr(sft_loader_module, "_MANAGED_LENGTH_LOOKAHEAD", 3)
    holder = _TenSampleHolder()
    loader = SFTStreamingDataLoader(config)
    loader.set_service_holder(holder)
    run_coro(loader.prepare())
    try:
        horizon = run_coro(loader.count_optimizer_windows())

        scan_requests = holder.length_requests[1:]
        assert horizon > 0
        assert len(scan_requests) > 1
        assert all(1 <= len(request) <= 3 for request in scan_requests)
        requested_indices = [ref["sample_idx"] for request in scan_requests for ref in request]
        assert set(requested_indices) == set(range(1, 10))
        assert len(requested_indices) == len(set(requested_indices))
    finally:
        loader.close()


def test_mixed_resume_replays_occurrence_ids_from_committed_cursor(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    config = _mixed_config(path)

    first = SFTStreamingDataLoader(config)
    first.set_service_holder(_ManagedDatasetHolder())
    run_coro(first.prepare())
    first_batch = _consume_active_batch(first)
    first.commit_batch(first_batch[0]["sft_batch_id"])
    state = first.state_dict()
    assert state["version"] == 2
    assert "model_data_revision" not in state
    assert state["mix_plan_fingerprint"]
    first.release_next_batch()
    expected = _consume_active_batch(first)
    first.close()

    resumed = SFTStreamingDataLoader(config)
    resumed.set_service_holder(_ManagedDatasetHolder())
    run_coro(resumed.prepare())
    try:
        resumed.load_state_dict(state)
        actual = _consume_active_batch(resumed)
        assert [item["datum_id"] for item in actual] == [item["datum_id"] for item in expected]
        assert [item.get("sample_ref") for item in actual] == [
            item.get("sample_ref") for item in expected
        ]
        assert [item.get("content_visibility") for item in actual] == [
            item.get("content_visibility") for item in expected
        ]
    finally:
        resumed.close()


def test_mixed_resume_rebuilds_later_epoch_carry_and_managed_length_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    config = _mixed_config(path)
    config.total_samples = 16

    class _VariableLengthHolder(_ManagedDatasetHolder):
        async def resolve_managed_sample_ref_lengths(self, refs: list[dict]):
            requested = [dict(ref) for ref in refs]
            self.length_requests.append(requested)
            lengths = {0: 1, 1: 2, 2: 3, 3: 4}
            return [
                {
                    "sample_ref": dict(ref),
                    "input_token_count": lengths[ref["sample_idx"]],
                }
                for ref in refs
            ]

    first = SFTStreamingDataLoader(config)
    first.set_service_holder(_VariableLengthHolder())
    run_coro(first.prepare())
    original_start_prefetch = first._start_prefetch

    def _pause_later_epoch_lookahead_with_carry() -> None:
        if first._epoch > 0 and first._carry is not None:
            return
        original_start_prefetch()

    first._start_prefetch = _pause_later_epoch_lookahead_with_carry
    try:
        while True:
            batch = _consume_active_batch(first)
            first.commit_batch(batch[0]["sft_batch_id"])
            state = first.state_dict()
            if state["epoch"] > 0:
                break
            first.release_next_batch()

        assert state["position_in_epoch"] == 2
        assert first._carry is not None
        assert first._carry.start_cursor["epoch"] == state["epoch"]
        assert first._carry.start_cursor["position_in_epoch"] == state["position_in_epoch"]

        first._start_prefetch = original_start_prefetch
        first.release_next_batch()
        expected = _consume_active_batch(first)
    finally:
        first.close()

    fresh_holder = _VariableLengthHolder()
    resumed = SFTStreamingDataLoader(config)
    resumed.set_service_holder(fresh_holder)
    run_coro(resumed.prepare())
    resumed.load_state_dict(state)
    original_resumed_start_prefetch = resumed._start_prefetch

    def _prepare_only_resumed_batch() -> None:
        if resumed._active_batch is None:
            original_resumed_start_prefetch()

    resumed._start_prefetch = _prepare_only_resumed_batch
    try:
        actual = _consume_active_batch(resumed)
        assert [
            (item["datum_id"], item.get("sample_ref"), item["input_token_count"]) for item in actual
        ] == [
            (item["datum_id"], item.get("sample_ref"), item["input_token_count"])
            for item in expected
        ]
        assert fresh_holder.length_requests[0] == [
            {"dataset": "hq-math", "version": "2026-08", "sample_idx": 0}
        ]
        assert len(fresh_holder.length_requests) == 2
        lookahead_indices = [ref["sample_idx"] for ref in fresh_holder.length_requests[1]]
        assert set(lookahead_indices) == {1, 2, 3}
        assert len(lookahead_indices) == len(set(lookahead_indices))
    finally:
        resumed.close()


def test_mixed_resume_does_not_bind_managed_content_visibility(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    config = _mixed_config(path)

    first = SFTStreamingDataLoader(config)
    first.set_service_holder(_ManagedDatasetHolder(content_visibility="protected"))
    run_coro(first.prepare())
    state = first.state_dict()
    assert "content_visibility" not in state["sources"][1]
    first.close()

    resumed = SFTStreamingDataLoader(config)
    resumed.set_service_holder(_ManagedDatasetHolder(content_visibility="public"))
    run_coro(resumed.prepare())
    try:
        resumed.load_state_dict(state)
    finally:
        resumed.close()


def test_mixed_loader_fails_closed_without_content_visibility(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    loader = SFTStreamingDataLoader(_mixed_config(path))
    loader.set_service_holder(_ManagedDatasetHolder(content_visibility=None))
    try:
        with pytest.raises(RuntimeError, match="content_visibility"):
            run_coro(loader.prepare())
    finally:
        loader.close()


def test_local_only_mixed_loader_retains_existing_cursor_schema(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_jsonl(first_path, [_record(marker) for marker in (10, 20)])
    _write_jsonl(second_path, [_record(marker) for marker in (30, 40)])
    config = _mixed_config(first_path)
    config.sources = [
        {"type": "local", "files": [str(first_path)], "ratio": 1.0},
        {"type": "local", "files": [str(second_path)], "ratio": 1.0},
    ]

    loader = SFTStreamingDataLoader(config)
    run_coro(loader.prepare())
    try:
        state = loader.state_dict()
        assert state["version"] == 2
        assert "model_data_revision" not in state
    finally:
        loader.close()


def test_recommended_ratio_change_does_not_invalidate_resume_plan(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    config = _mixed_config(path)

    first = SFTStreamingDataLoader(config)
    first.set_service_holder(_ManagedDatasetHolder(recommended_ratio=0.2))
    run_coro(first.prepare())
    state = first.state_dict()
    assert all("recommended_ratio" not in source for source in state["sources"])
    first.close()

    resumed = SFTStreamingDataLoader(config)
    resumed.set_service_holder(_ManagedDatasetHolder(recommended_ratio=0.35))
    run_coro(resumed.prepare())
    try:
        assert resumed.state_dict()["mix_plan_fingerprint"] == state["mix_plan_fingerprint"]
        resumed.load_state_dict(state)
    finally:
        resumed.close()


def test_mixed_loader_rejects_length_outside_model_bound_limit(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    loader = SFTStreamingDataLoader(_mixed_config(path))
    loader.set_service_holder(_ManagedDatasetHolder(input_token_count=5))
    try:
        with pytest.raises(ValueError, match="max_allowed=4"):
            run_coro(loader.prepare())
    finally:
        loader.close()


def test_mixed_loader_waits_for_retryable_variant_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    config = _mixed_config(path)
    config.managed_variant_ready_timeout_seconds = 30.0
    config.managed_variant_poll_initial_seconds = 0.25
    config.managed_variant_poll_max_seconds = 1.0
    holder = _ReadinessHolder(
        [
            _weaver_error(
                code="managed_dataset_variant_not_ready",
                retryable=True,
                message="must-not-appear-in-readiness-logs",
            ),
            _weaver_error(code="managed_datasets_unavailable", retryable=True, status_code=503),
        ]
    )
    sleeps: list[float] = []

    async def _record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(sft_loader_module, "async_sleep", _record_sleep)
    loader = SFTStreamingDataLoader(config)
    loader.set_service_holder(holder)
    try:
        run_coro(loader.prepare())

        assert len(holder.length_requests) == 3
        assert sleeps == [0.25, 0.5]
        assert "must-not-appear-in-readiness-logs" not in caplog.text
        assert "hq-math" in caplog.text
        assert "2026-08" in caplog.text
    finally:
        loader.close()


@pytest.mark.parametrize(
    ("code", "retryable", "status_code"),
    [
        ("managed_dataset_variant_not_ready", False, 409),
        ("forbidden", True, 403),
        ("not_found", True, 404),
        ("protocol_error", True, 500),
    ],
)
def test_mixed_loader_does_not_retry_non_whitelisted_variant_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    retryable: bool,
    status_code: int,
) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    error = _weaver_error(code=code, retryable=retryable, status_code=status_code)
    holder = _ReadinessHolder([error])

    async def _unexpected_sleep(delay: float) -> None:
        raise AssertionError(f"unexpected readiness sleep: {delay}")

    monkeypatch.setattr(sft_loader_module, "async_sleep", _unexpected_sleep)
    loader = SFTStreamingDataLoader(_mixed_config(path))
    loader.set_service_holder(holder)
    try:
        with pytest.raises(WeaverAPIError) as exc_info:
            run_coro(loader.prepare())

        assert exc_info.value is error
        assert len(holder.length_requests) == 1
    finally:
        loader.close()


def test_mixed_loader_bounds_variant_readiness_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])
    holder = _ReadinessHolder(
        [_weaver_error(code="managed_dataset_variant_not_ready", retryable=True)]
    )
    monotonic_values = iter((100.0, 101.0))
    monkeypatch.setattr(sft_loader_module, "monotonic", lambda: next(monotonic_values))
    config = _mixed_config(path)
    config.managed_variant_ready_timeout_seconds = 0.5
    loader = SFTStreamingDataLoader(config)
    loader.set_service_holder(holder)
    try:
        with pytest.raises(TimeoutError, match="managed_dataset_variant_not_ready"):
            run_coro(loader.prepare())

        assert len(holder.length_requests) == 1
    finally:
        loader.close()


def test_mixed_loader_rejects_unpublished_catalog_version(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])

    class _UnpublishedHolder(_ManagedDatasetHolder):
        async def get_managed_dataset_info(self, *, name: str, version: str):
            info = await super().get_managed_dataset_info(name=name, version=version)
            info.status = "processing"
            return info

    loader = SFTStreamingDataLoader(_mixed_config(path))
    loader.set_service_holder(_UnpublishedHolder())
    try:
        with pytest.raises(ValueError, match="is not published"):
            run_coro(loader.prepare())
    finally:
        loader.close()


def test_mixed_loader_propagates_inaccessible_catalog_version(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40)])

    class _InaccessibleHolder(_ManagedDatasetHolder):
        async def get_managed_dataset_info(self, *, name: str, version: str):
            del name, version
            raise PermissionError("managed dataset version is unavailable")

    loader = SFTStreamingDataLoader(_mixed_config(path))
    loader.set_service_holder(_InaccessibleHolder())
    try:
        with pytest.raises(PermissionError, match="unavailable"):
            run_coro(loader.prepare())
    finally:
        loader.close()


def test_resume_replays_from_committed_cursor_not_prefetch_cursor(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40, 50, 60)])

    first_loader = SFTStreamingDataLoader(_config(path, max_sequence_length=5, batch_size=2))
    first_batch = _consume_active_batch(first_loader)
    first_loader.commit_batch(first_batch[0]["sft_batch_id"])
    state = first_loader.state_dict()
    first_loader.close()

    resumed = SFTStreamingDataLoader(_config(path, max_sequence_length=5, batch_size=2))
    try:
        resumed.load_state_dict(state)
        resumed_batch = _consume_active_batch(resumed)
        assert [item["input_ids"][0] for item in resumed_batch] == [30, 40]
    finally:
        resumed.close()


def test_eof_partial_batch_is_dropped_without_cross_epoch_fill(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    loader = SFTStreamingDataLoader(_config(path))
    try:
        first = _consume_active_batch(loader)
        loader.commit_batch(first[0]["sft_batch_id"])
        loader.release_next_batch()

        second = _consume_active_batch(loader)
        loader.commit_batch(second[0]["sft_batch_id"])
        loader.release_next_batch()

        after_eof = _consume_active_batch(loader)
        # Record 30 was the EOF partial and was dropped. A new epoch starts at
        # record 10 rather than combining record 30 with the next epoch.
        assert [item["input_ids"][0] for item in after_eof] == [10]
    finally:
        loader.close()


def test_epoch_limit_stops_after_final_epoch_is_committed(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    loader = SFTStreamingDataLoader(_config(path, epochs=2))
    try:
        seen = []
        for batch_index in range(4):
            batch = _consume_active_batch(loader)
            assert batch
            if batch_index == 3:
                terminal_future = loader._prepared_future
                assert terminal_future is not None
                assert terminal_future.result(timeout=5) is None
            seen.append(batch[0]["input_ids"][0])
            assert not loader.is_finished()
            loader.commit_batch(batch[0]["sft_batch_id"])
            assert not loader.is_finished()
            loader.release_next_batch()

        assert seen == [10, 20, 10, 20]
        assert loader.is_finished()
        assert loader.get_next_item() is None
    finally:
        loader.close()


def test_step_mode_keeps_repeating_epochs(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    loader = SFTStreamingDataLoader(_config(path))
    try:
        for _ in range(4):
            batch = _consume_active_batch(loader)
            loader.commit_batch(batch[0]["sft_batch_id"])
            loader.release_next_batch()

        assert not loader.is_finished()
    finally:
        loader.close()


def test_resume_after_final_commit_drops_tail_and_finishes(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    config = _config(path, epochs=1)
    loader = SFTStreamingDataLoader(config)
    try:
        first = _consume_active_batch(loader)
        loader.commit_batch(first[0]["sft_batch_id"])
        loader.release_next_batch()
        final = _consume_active_batch(loader)
        loader.commit_batch(final[0]["sft_batch_id"])
        state = loader.state_dict()

        assert state["epoch"] == 0
        assert state["position_in_epoch"] == 2
    finally:
        loader.close()
    resumed = SFTStreamingDataLoader(config)

    try:
        resumed.load_state_dict(state)
        assert resumed.get_next_item() is None
        assert resumed.is_finished()
    finally:
        resumed.close()


def test_too_short_epoch_fails_instead_of_spinning(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(10)])
    loader = SFTStreamingDataLoader(_config(path))
    try:
        with pytest.raises(RuntimeError, match="cannot produce a full token-budget batch"):
            loader.get_next_item()
    finally:
        loader.close()


def test_parquet_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    path.write_bytes(b"not-a-supported-sft-source")

    with pytest.raises(ValueError, match="supports only JSONL"):
        SFTStreamingDataLoader(_config(path))


def test_streaming_loader_renders_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": f"question {marker}"},
                    {"role": "assistant", "content": f"answer {marker}"},
                ]
            }
            for marker in range(3)
        ],
    )

    class _Tokenizer:
        bos_token_id = 10

        def apply_chat_template(self, messages, **kwargs):
            assert set(kwargs) == {"add_generation_prompt", "tokenize"}
            assert kwargs["tokenize"] is False
            if kwargs["add_generation_prompt"]:
                assert [message["role"] for message in messages] == ["user"]
                return "abc"
            assert [message["role"] for message in messages] == ["user", "assistant"]
            return "abcdef"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"abc": [10, 11, 12], "def": [20, 21, 22]}[text]

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda path: _Tokenizer(),
    )
    config = _config(path, max_sequence_length=6)
    config.tokenizer = "fake-tokenizer"
    loader = SFTStreamingDataLoader(config)
    try:
        batch = _consume_active_batch(loader)
        assert batch[0]["input_ids"] == [10, 11, 12, 20, 21, 22]
        assert batch[0]["loss_mask"] == [0, 0, 0, 1, 1, 1]
        assert "messages" not in batch[0]
    finally:
        loader.close()


def test_streaming_loader_passes_full_row_to_custom_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [{"prompt": "question", "answer": "answer", "source": str(index)} for index in range(3)],
    )
    module_path = tmp_path / "custom_sft.py"
    module_path.write_text(
        "def render_chat(row, tokenizer):\n"
        "    assert row['prompt'] == 'question'\n"
        "    assert row['answer'] == 'answer'\n"
        "    assert row['source'] in {'0', '1', '2'}\n"
        "    assert tokenizer is not None\n"
        "    return [1, 2, 3, 4, 5, 6], [0, 1, 1, 1, 1, 1]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda path: object(),
    )
    config = _config(path, max_sequence_length=6)
    config.tokenizer = "fake-tokenizer"
    config.render_chat = f"{module_path}:render_chat"
    loader = SFTStreamingDataLoader(config)
    try:
        batch = _consume_active_batch(loader)
        assert batch[0]["input_ids"] == [1, 2, 3, 4, 5, 6]
        assert batch[0]["loss_mask"] == [0, 1, 1, 1, 1, 1]
        assert batch[0]["source"] in {"0", "1", "2"}
    finally:
        loader.close()


def test_epoch_shuffle_is_deterministic_and_changes_by_epoch(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in range(10, 110, 10)])
    first = SFTStreamingDataLoader(_config(path, shuffle=True, seed=123))
    second = SFTStreamingDataLoader(_config(path, shuffle=True, seed=123))
    try:
        assert first._make_epoch_order(0) == second._make_epoch_order(0)
        assert first._make_epoch_order(0) != first._make_epoch_order(1)
        assert sorted(first._make_epoch_order(0)) == list(range(10))
    finally:
        first.close()
        second.close()


def test_resume_state_has_no_content_or_config_hashes(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    loader = SFTStreamingDataLoader(_config(path))
    try:
        state = loader.state_dict()
        assert "dataset_fingerprint" not in state
        assert "pipeline_fingerprint" not in state
    finally:
        loader.close()


def test_relative_data_path_is_resolved_from_recipe(tmp_path: Path) -> None:
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    path = recipe_dir / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    recipe_path = recipe_dir / "train.yaml"
    recipe_path.write_text("# test recipe\n", encoding="utf-8")
    config = _config(path)
    config.files = ["train.jsonl"]
    config.config_file_path = str(recipe_path)

    loader = SFTStreamingDataLoader(config)
    try:
        batch = _consume_active_batch(loader)
        assert batch[0]["input_ids"][0] == 10
    finally:
        loader.close()


def test_resume_does_not_hash_jsonl_content(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    loader = SFTStreamingDataLoader(_config(path))
    state = loader.state_dict()
    loader.close()

    _write_jsonl(path, [_record(marker) for marker in (11, 21, 31)])
    resumed = SFTStreamingDataLoader(_config(path))
    try:
        resumed.load_state_dict(state)
    finally:
        resumed.close()


def test_resume_does_not_hash_local_tokenizer_assets(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_asset = tokenizer_dir / "tokenizer.json"
    tokenizer_asset.write_text('{"version": 1}\n', encoding="utf-8")
    recipe_path = tmp_path / "train.yaml"
    recipe_path.write_text("# test recipe\n", encoding="utf-8")
    config = _config(path)
    config.tokenizer = "tokenizer"
    config.config_file_path = str(recipe_path)

    loader = SFTStreamingDataLoader(config)
    assert loader._tokenizer_path == str(tokenizer_dir)
    state = loader.state_dict()
    loader.close()

    tokenizer_asset.write_text('{"version": 2}\n', encoding="utf-8")
    resumed = SFTStreamingDataLoader(config)
    try:
        resumed.load_state_dict(state)
    finally:
        resumed.close()


def test_resume_inside_shuffled_later_epoch_replays_exact_next_batch(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    config = _config(path, shuffle=True, seed=123)
    loader = SFTStreamingDataLoader(config)
    try:
        for _ in range(3):
            batch = _consume_active_batch(loader)
            loader.commit_batch(batch[0]["sft_batch_id"])
            loader.release_next_batch()
        state = loader.state_dict()
        assert state["epoch"] == 1
        expected = _consume_active_batch(loader)
    finally:
        loader.close()

    resumed = SFTStreamingDataLoader(config)
    try:
        resumed.load_state_dict(state)
        actual = _consume_active_batch(resumed)
        assert [item["input_ids"] for item in actual] == [item["input_ids"] for item in expected]
    finally:
        resumed.close()


def test_resume_does_not_hash_custom_renderer(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30)])
    module_path = tmp_path / "custom_sft.py"
    module_path.write_text(
        "def render_chat(row, tokenizer):\n" "    return [1, 2], [0, 1]\n",
        encoding="utf-8",
    )
    config = _config(path)
    config.tokenizer = "fake-tokenizer"
    config.render_chat = f"{module_path}:render_chat"
    loader = SFTStreamingDataLoader(config)
    state = loader.state_dict()
    loader.close()

    module_path.write_text(
        "def render_chat(row, tokenizer):\n" "    return [1, 2, 3], [0, 1, 1]\n",
        encoding="utf-8",
    )
    resumed = SFTStreamingDataLoader(config)
    try:
        resumed.load_state_dict(state)
    finally:
        resumed.close()


def test_starts_exactly_one_lookahead_preparation(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in range(10, 110, 10)])
    loader = SFTStreamingDataLoader(_config(path, max_sequence_length=5, batch_size=2))
    second_started = threading.Event()
    release_second = threading.Event()
    prepare_calls = 0
    original_prepare = loader._prepare_next_batch

    def prepare_with_blocked_lookahead():
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 2:
            second_started.set()
            assert release_second.wait(timeout=5)
        return original_prepare()

    loader._prepare_next_batch = prepare_with_blocked_lookahead
    try:
        batch = _consume_active_batch(loader)

        assert batch
        assert second_started.wait(timeout=5)
        assert prepare_calls == 2
        assert loader._prepared_future is not None
        assert not loader._prepared_future.done()
    finally:
        release_second.set()
        loader.close()


def test_window_samples_fixes_samples_per_window_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    lengths = ((10, 3), (20, 9), (30, 2), (40, 5), (50, 4), (60, 6), (70, 2))
    _write_jsonl(path, [_record(marker, length) for marker, length in lengths])
    loader = SFTStreamingDataLoader(
        _config(path, max_sequence_length=9, batch_size=100, epochs=1, window_samples=3)
    )
    try:
        assert run_coro(loader.count_optimizer_windows()) == 2
        batches = _consume_finite_plan(loader)
    finally:
        loader.close()

    # Variable lengths fit the budget many times over; the sample cap alone closes
    # each window, rows stay in file order, and the 1-sample EOF tail is dropped.
    assert [[item["input_ids"][0] for item in batch] for batch in batches] == [
        [10, 20, 30],
        [40, 50, 60],
    ]


def test_token_budget_still_closes_a_window_before_the_sample_cap(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(marker) for marker in (10, 20, 30, 40, 50)])
    loader = SFTStreamingDataLoader(
        _config(path, max_sequence_length=5, batch_size=2, epochs=1, window_samples=3)
    )
    try:
        batches = _consume_finite_plan(loader)
    finally:
        loader.close()

    # Each record has 4 shifted positions against a budget of 10: the budget closes
    # windows at 2 samples, below the cap of 3, and the EOF tail [50] is dropped.
    assert [[item["input_ids"][0] for item in batch] for batch in batches] == [[10, 20], [30, 40]]
