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

"""Regression coverage for lazy, map-style JSONL training datasets."""

import json
import pickle
from builtins import open as builtin_open
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from northrl.data_loader import torch_data_loader as loader_module
from northrl.data_loader.torch_data_loader import FileDataset, TorchDataLoader


def _write_jsonl(path: Path, records: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return str(path)


def _config(path: str, **overrides):
    config = {
        "files": [path],
        "batch_size": 3,
        "rollout_repeat_n": 2,
        "prompt_key": "prompt",
        "filter_prompts": False,
        "max_prompt_length": 1024,
        "shuffle": False,
        "drop_last": False,
        "cycle": False,
        "seed": 23,
        "keep_batch_order": False,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def test_jsonl_construction_does_not_parse_or_materialize_records(tmp_path, monkeypatch):
    path = _write_jsonl(
        tmp_path / "large.jsonl",
        [{"prompt": "first", "label": "x" * 65536}, {"prompt": "second", "label": "42"}],
    )

    def unexpected_read(*args, **kwargs):
        pytest.fail("Unfiltered JSONL construction must index bytes without parsing records")

    with monkeypatch.context() as patch:
        patch.setattr(pd, "read_json", unexpected_read)
        patch.setattr(loader_module.json, "loads", unexpected_read)
        dataset = FileDataset(path)
        assert len(dataset) == 2
        # The checkpoint must not retain the large answer payload either.
        assert len(pickle.dumps(dataset)) < 65536

    assert dataset[1]["label"] == "42"
    assert len(dataset[0]["label"]) == 65536
    dataset.close()


def test_jsonl_random_access_handles_utf8_blank_lines_and_multiple_files(tmp_path):
    first_path = tmp_path / "first.jsonl"
    first_path.write_text(
        '\n  \r\n{"prompt":"中文问题","extra_info":{"index":"task-甲"}}\r\n'
        '\n{"prompt":"second"}',
        encoding="utf-8",
    )
    second_path = _write_jsonl(tmp_path / "second.jsonl", [{"prompt": "第三条"}])
    dataset = FileDataset([str(first_path), second_path])

    assert len(dataset) == 3
    assert [dataset[i]["prompt"] for i in (2, 0, 1)] == ["第三条", "中文问题", "second"]
    assert [dataset[i]["index"] for i in range(3)] == ["task-甲", 1, 2]
    dataset.close()


def test_jsonl_large_records_and_whitespace_lines_cross_scanner_chunks(tmp_path):
    path = tmp_path / "large-lines.jsonl"
    large_label = "答" * 800000
    first_record = json.dumps({"prompt": "first", "label": large_label}, ensure_ascii=False)
    middle_record = json.dumps({"prompt": "middle", "extra_info": {"index": "middle-id"}})
    path.write_bytes(
        first_record.encode("utf-8")
        + b"\r\n"
        + b" \t\r" * 800000
        + b"\n"
        + middle_record.encode("utf-8")
        + b"\r\n\r\n"
        + b'{"prompt":"last"}'  # Final record deliberately has no trailing newline.
    )
    dataset = FileDataset(str(path))

    assert len(dataset) == 3
    assert [dataset[i]["prompt"] for i in (2, 0, 1)] == ["last", "first", "middle"]
    assert dataset[0]["label"] == large_label
    assert [dataset[i]["index"] for i in range(3)] == [0, "middle-id", 2]
    dataset.close()


def test_jsonl_index_uses_bounded_reads_instead_of_loading_whole_lines(tmp_path, monkeypatch):
    path = tmp_path / "bounded.jsonl"
    large_record = json.dumps({"prompt": "large", "label": "x" * (4 * 1024 * 1024)})
    path.write_text(large_record + '\n{"prompt":"small"}\n', encoding="utf-8")
    read_sizes = []

    class IndexReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def read(self, size=-1):
            assert 0 < size < len(large_record) // 2
            read_sizes.append(size)
            return self.handle.read(size)

        def readline(self, *args):
            pytest.fail("Unfiltered startup must not buffer a complete JSONL line")

    def open_index_file(file, mode, *args, **kwargs):
        assert file == str(path)
        assert mode == "rb"
        return IndexReader(builtin_open(file, mode, *args, **kwargs))

    with monkeypatch.context() as patch:
        patch.setattr(loader_module, "open", open_index_file, raising=False)
        dataset = FileDataset(str(path))
        assert len(dataset) == 2
    assert len(read_sizes) > 2
    assert dataset[1]["prompt"] == "small"
    dataset.close()


def test_jsonl_normalizes_sparse_answers_and_optional_metadata(tmp_path):
    path = _write_jsonl(
        tmp_path / "mixed.jsonl",
        [
            {"prompt": "scalar", "label": "42", "reward_fn": "custom"},
            {"prompt": "reference", "ground_truth": "43", "reward_fn": None},
            {
                "prompt": "nested",
                "reward_model": {"ground_truth": "44"},
                "metadata": {"language": "中文"},
            },
            {
                "prompt": "empty",
                "reward_fn": float("nan"),
                "label": float("nan"),
                "ground_truth": None,
                "reward_model": None,
                "metadata": None,
                "extra_info": None,
            },
        ],
    )
    dataset = FileDataset(path)

    assert dataset[0]["label"] == "42"
    assert dataset[0]["reward_fn"] == "custom"
    assert "ground_truth" not in dataset[0]
    assert "reward_model" not in dataset[0]
    assert "label" not in dataset[1]
    assert dataset[1]["ground_truth"] == "43"
    assert "reward_model" not in dataset[1]
    assert "reward_fn" not in dataset[1]
    assert dataset[2]["reward_model"] == {"ground_truth": "44"}
    assert dataset[2]["metadata"] == {"language": "中文"}
    assert dataset[3]["label"] is None
    assert dataset[3]["ground_truth"] is None
    assert dataset[3]["reward_model"] is None
    assert dataset[3]["metadata"] is None
    assert dataset[3]["index"] == 3
    assert "reward_fn" not in dataset[3]
    dataset.close()


def test_jsonl_prompt_filter_keeps_filtered_order_and_explicit_indices(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, prompt, **kwargs):
            if prompt == "invalid":
                raise ValueError("invalid prompt")
            return prompt

        def __call__(self, prompt, **kwargs):
            return {"input_ids": list(range(len(prompt)))}

    path = _write_jsonl(
        tmp_path / "filtered.jsonl",
        [
            {"prompt": "ok", "extra_info": {"index": 41}},
            {"prompt": "too long"},
            {"prompt": "invalid"},
            {"prompt": "yes"},
        ],
    )
    dataset = FileDataset(path, filter_prompts=True, max_prompt_length=3, tokenizer=Tokenizer())

    assert len(dataset) == 2
    assert [dataset[i]["prompt"] for i in range(2)] == ["ok", "yes"]
    assert [dataset[i]["index"] for i in range(2)] == [41, 1]
    dataset.resume_dataset_state()
    assert [dataset[i]["index"] for i in range(2)] == [41, 1]
    dataset.close()


def test_invalid_json_is_reported_only_when_requested_with_file_and_byte_offset(tmp_path):
    path = tmp_path / "invalid.jsonl"
    valid_line = '{"prompt":"中文"}\n'.encode("utf-8")
    path.write_bytes(valid_line + b'{"prompt": invalid}\n')
    dataset = FileDataset(str(path))

    assert len(dataset) == 2
    assert dataset[0]["prompt"] == "中文"
    with pytest.raises(ValueError) as error:
        dataset[1]
    assert str(path) in str(error.value)
    assert str(len(valid_line)) in str(error.value)
    dataset.close()


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_jsonl_interleaves_with_eager_formats_in_file_order(tmp_path, monkeypatch, file_format):
    first_path = _write_jsonl(tmp_path / "first.jsonl", [{"prompt": "json-first"}])
    eager_path = tmp_path / f"eager.{file_format}"
    dataframe = pd.DataFrame([{"prompt": "eager", "label": "43", "reward_fn": None}])
    if file_format == "csv":
        dataframe.to_csv(eager_path, index=False)
    else:
        # Check dispatch without adding a parquet engine to the test's dependencies.
        def read_parquet(path):
            assert path == str(eager_path)
            return dataframe

        monkeypatch.setattr(pd, "read_parquet", read_parquet)
    last_path = _write_jsonl(
        tmp_path / "last.jsonl", [{"prompt": "json-last", "ground_truth": "44"}]
    )
    dataset = FileDataset([first_path, str(eager_path), last_path])

    assert len(dataset) == 3
    assert [dataset[i]["prompt"] for i in range(3)] == ["json-first", "eager", "json-last"]
    assert [dataset[i]["index"] for i in range(3)] == [0, 1, 2]
    assert dataset[2]["ground_truth"] == "44"
    assert "reward_fn" not in dataset[1]
    dataset.close()


def test_jsonl_pickle_resume_omits_open_handles_and_record_payloads(tmp_path, monkeypatch):
    path = _write_jsonl(tmp_path / "checkpoint.jsonl", [{"prompt": "test", "label": "x" * 65536}])
    dataset = FileDataset(path)
    assert len(dataset[0]["label"]) == 65536  # Open a reader before serializing.
    payload = pickle.dumps(dataset)
    assert len(payload) < 65536
    restored = pickle.loads(payload)
    assert restored[0]["label"] == dataset[0]["label"]

    def unexpected_parse(*args, **kwargs):
        pytest.fail("Resume must rebuild the unfiltered index without parsing records")

    with monkeypatch.context() as patch:
        patch.setattr(loader_module.json, "loads", unexpected_parse)
        restored.resume_dataset_state()
        assert len(restored) == 1

    assert restored[0]["label"] == dataset[0]["label"]
    restored.close()
    dataset.close()


def test_jsonl_close_is_idempotent_and_reopened_reads_are_fresh(tmp_path):
    path = _write_jsonl(
        tmp_path / "fresh.jsonl", [{"prompt": "test", "metadata": {"source": "original"}}]
    )
    dataset = FileDataset(path)
    first = dataset[0]
    first["metadata"]["source"] = "changed"
    dataset.close()
    dataset.close()

    assert dataset[0]["metadata"] == {"source": "original"}
    dataset.close()


def test_resume_replaces_state_when_switching_between_jsonl_and_csv(tmp_path):
    jsonl_path = _write_jsonl(tmp_path / "indexed.jsonl", [{"prompt": "jsonl"}])
    csv_path = tmp_path / "eager.csv"
    pd.DataFrame([{"prompt": "csv-first"}, {"prompt": "csv-second"}]).to_csv(csv_path, index=False)
    dataset = FileDataset(jsonl_path)
    assert dataset[0]["prompt"] == "jsonl"
    opened_handles = list(dataset._jsonl_handles.values())

    dataset.data_files = [str(csv_path)]
    dataset.resume_dataset_state()
    assert all(handle.closed for handle in opened_handles)
    assert len(dataset) == 2
    assert [dataset[i]["prompt"] for i in range(2)] == ["csv-first", "csv-second"]
    assert hasattr(dataset, "dataframe")

    dataset.data_files = [jsonl_path]
    dataset.resume_dataset_state()
    assert len(dataset) == 1
    assert dataset[0]["prompt"] == "jsonl"
    assert not hasattr(dataset, "dataframe")
    dataset.close()


@pytest.mark.parametrize("file_format", ["jsonl", "csv"])
def test_resume_accepts_legacy_dataset_state_without_index_or_handle_attributes(
    tmp_path, file_format
):
    path = tmp_path / f"legacy.{file_format}"
    records = [{"prompt": "legacy-first"}, {"prompt": "legacy-second"}]
    if file_format == "jsonl":
        _write_jsonl(path, records)
    else:
        pd.DataFrame(records).to_csv(path, index=False)

    # Old checkpoints contained this metadata and omitted the eager dataframe.
    dataset = FileDataset.__new__(FileDataset)
    dataset.__dict__.update(
        data_files=[str(path)],
        original_data_files=[str(path)],
        filter_prompts=False,
        prompt_key="prompt",
        max_prompt_length=1024,
        tokenizer=None,
        serialize_dataset=False,
    )
    dataset.resume_dataset_state()

    assert len(dataset) == 2
    assert [dataset[i]["prompt"] for i in range(2)] == ["legacy-first", "legacy-second"]
    assert [dataset[i]["index"] for i in range(2)] == [0, 1]
    assert not dataset.serialize_dataset
    dataset.close()


def test_jsonl_shuffle_cycle_repeat_and_resume_preserve_the_same_stream(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", [{"prompt": f"task-{i}"} for i in range(5)])
    config = _config(path, shuffle=True, cycle=True, drop_last=True)
    reference = TorchDataLoader(config)
    items = [reference.get_next_item() for _ in range(24)]
    assert all(item is not None for item in items)

    for first, second in zip(items[::2], items[1::2]):
        assert first["index"] == second["index"]
        assert first["task_id"] == second["task_id"] == first["index"]
        assert first["group_id"] == second["group_id"]
        assert first["rollout_id"] != second["rollout_id"]
        assert [first["run_id"], second["run_id"]] == [0, 1]
    base_indices = [item["index"] for item in items[::2]]
    assert sorted(base_indices[:5]) == list(range(5))
    assert sorted(base_indices[5:10]) == list(range(5))

    resumed = TorchDataLoader(config)
    resumed.skip_batches(3)  # Nine prompts, including a boundary between shuffled epochs.
    actual = [resumed.get_next_item()["index"] for _ in range(6)]
    assert actual == [item["index"] for item in items[18:24]]
    reference.reset()
    assert [reference.get_next_item()["index"] for _ in range(10)] == [
        item["index"] for item in items[:10]
    ]
    reference.close()
    resumed.close()


def test_jsonl_validation_keeps_finite_tail_without_training_repeats(tmp_path):
    path = _write_jsonl(tmp_path / "validate.jsonl", [{"prompt": f"task-{i}"} for i in range(5)])
    loader = TorchDataLoader(_config(path, cycle=True), is_validate=True)
    items = [loader.get_next_item() for _ in range(5)]

    assert [item["index"] for item in items] == list(range(5))
    assert all("group_id" not in item for item in items)
    assert loader.is_finished()
    assert loader.get_next_item() is None
    loader.close()


@pytest.mark.parametrize("record", [[], "text"])
def test_jsonl_rejects_non_object_records_on_access(tmp_path, record):
    path = tmp_path / "non-object.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = FileDataset(str(path))

    with pytest.raises(ValueError) as error:
        dataset[0]
    assert str(path) in str(error.value)
    dataset.close()
