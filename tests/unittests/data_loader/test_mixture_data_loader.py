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

"""Weighted source selection, group identity, and resumable prompt streams."""

import json
from collections import Counter

import pandas as pd
import pytest
from omegaconf import OmegaConf

from northrl.data_loader import MixtureDataLoader


def _source(tmp_path, name, count, ratio=1, rows=None, **overrides):
    path = tmp_path / f"{name}.jsonl"
    if rows is None:
        rows = [{"task_id": f"task-{index}", "instruction": name} for index in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"type": "local", "name": name, "files": [str(path)], "ratio": ratio, **overrides}


def _config(sources, **overrides):
    return OmegaConf.create(
        {
            "sources": sources,
            "batch_size": 4,
            "rollout_repeat_n": 1,
            "prompt_key": "instruction",
            "filter_prompts": False,
            "shuffle": True,
            "drop_last": False,
            "max_prompt_length": 1024,
            "tokenizer": None,
            "seed": 42,
            "cycle": True,
            "keep_batch_order": False,
            **overrides,
        }
    )


def _take(loader, count):
    items = []
    for _ in range(count):
        item = loader.get_next_item()
        if item is None and not loader.is_finished():
            loader.release_next_batch()
            item = loader.get_next_item()
        assert item is not None
        items.append(item)
    return items


def _identity(items):
    return [(item["data_source"], item["task_id"], item["index"]) for item in items]


def test_source_weights_are_independent_of_dataset_sizes_and_shuffle(tmp_path):
    sources = [_source(tmp_path, "small", 1, 1), _source(tmp_path, "large", 99, 3)]
    loader = MixtureDataLoader(_config(sources, batch_size=100))
    items = _take(loader, 8000)
    counts = Counter(item["data_source"] for item in items)

    assert abs(counts["large"] / len(items) - 0.75) < 0.02
    unshuffled = MixtureDataLoader(_config(sources, batch_size=100, shuffle=False))
    assert [item["data_source"] for item in _take(unshuffled, 8000)] == [
        item["data_source"] for item in items
    ]


def test_repeated_draws_preserve_task_metadata_and_create_new_groups(tmp_path):
    row = {
        "task_id": "original-task-directory",
        "instruction": "Fix the bug",
        "task_template_id": "template-123",
        "metadata": {"tests": ["test-a", "test-b"], "enabled": True},
        "answer": None,
        "rollout_profile": None,
    }
    source = _source(tmp_path, "swe", 1, rows=[row], rollout_profile="swe-agent")
    loader = MixtureDataLoader(_config([source], batch_size=2, rollout_repeat_n=3))
    items = _take(loader, 6)

    assert len({item["rollout_id"] for item in items}) == 6
    assert len({item["group_id"] for item in items[:3]}) == 1
    assert len({item["group_id"] for item in items[3:]}) == 1
    assert items[0]["group_id"] != items[3]["group_id"]
    for group in (items[:3], items[3:]):
        assert [item["run_id"] for item in group] == [0, 1, 2]
        for item in group:
            assert item["expected_rollouts"] == 3
            assert item["task_id"] == row["task_id"]
            assert item["task_template_id"] == row["task_template_id"]
            assert item["metadata"] == row["metadata"]
            assert item["answer"] is None
            assert item["data_source"] == "swe"
            assert item["rollout_profile"] == "swe-agent"


@pytest.mark.parametrize("shuffle", [False, True])
def test_reset_and_skip_replay_source_and_prompt_choices(tmp_path, shuffle):
    sources = [_source(tmp_path, "code", 3, 2), _source(tmp_path, "swe", 7, 1)]
    config = _config(sources, batch_size=4, rollout_repeat_n=3, shuffle=shuffle)
    baseline = MixtureDataLoader(config)
    items = _take(baseline, 84)
    baseline.reset()
    replay = _take(baseline, 84)

    assert _identity(replay) == _identity(items)
    assert {item["rollout_id"] for item in items}.isdisjoint(item["rollout_id"] for item in replay)

    config.keep_batch_order = True
    resumed = MixtureDataLoader(config)
    resumed.skip_batches(5)
    assert _identity(_take(resumed, 24)) == _identity(items[60:])
    with pytest.raises(RuntimeError, match="has started iteration"):
        resumed.skip_batches(1)


@pytest.mark.parametrize("shuffle", [False, True])
def test_each_source_consumes_complete_permutations_before_reuse(tmp_path, shuffle):
    source = _source(tmp_path, "code", 5)
    loader = MixtureDataLoader(_config([source], shuffle=shuffle, batch_size=3))
    items = _take(loader, 15)

    for epoch in range(3):
        assert {item["task_id"] for item in items[epoch * 5 : (epoch + 1) * 5]} == {
            f"task-{index}" for index in range(5)
        }
    if not shuffle:
        assert [item["task_id"] for item in items] == [f"task-{index}" for index in range(5)] * 3
    else:
        # Keep the pre-rename ordering so skipping a checkpoint cursor is stable.
        assert [item["task_id"] for item in items[:5]] == [
            "task-4",
            "task-1",
            "task-0",
            "task-3",
            "task-2",
        ]
    assert not loader.is_finished()


@pytest.mark.parametrize("size", [5, 6])
@pytest.mark.parametrize("drop_last", [False, True])
def test_finite_loader_preserves_final_buffer_and_honors_drop_last(tmp_path, size, drop_last):
    source = _source(tmp_path, "code", size)
    loader = MixtureDataLoader(
        _config(
            [source],
            batch_size=2,
            rollout_repeat_n=2,
            cycle=False,
            shuffle=False,
            drop_last=drop_last,
            keep_batch_order=True,
        )
    )
    expected = (size // 2 * 2 if drop_last else size) * 2
    items = _take(loader, expected - 1)
    assert not loader.is_finished()
    items += _take(loader, 1)

    assert [item["task_id"] for item in items] == [
        f"task-{index}" for index in range(expected // 2) for _ in range(2)
    ]
    if drop_last and size % 2:
        loader.release_next_batch()
    assert loader.is_finished()
    assert loader.get_next_item() is None
    assert loader.get_dataset_size() == size
    loader.reset()
    assert not loader.is_finished()
    assert _identity(_take(loader, expected)) == _identity(items)


def test_cycle_fills_batches_across_source_epochs_even_with_drop_last(tmp_path):
    source = _source(tmp_path, "code", 2)
    loader = MixtureDataLoader(
        _config([source], batch_size=5, shuffle=False, drop_last=True, keep_batch_order=True)
    )

    assert [item["task_id"] for item in _take(loader, 15)] == [
        f"task-{index % 2}" for index in range(15)
    ]
    assert not loader.is_finished()


@pytest.mark.parametrize("row_profile", ["code", "swe"])
def test_source_profile_accepts_match_and_rejects_conflicting_row(tmp_path, row_profile):
    source = _source(
        tmp_path,
        "code",
        1,
        rows=[{"task_id": "task-a", "rollout_profile": row_profile}],
        rollout_profile="code",
    )
    loader = MixtureDataLoader(_config([source]))

    if row_profile == "code":
        assert _take(loader, 1)[0]["rollout_profile"] == "code"
    else:
        with pytest.raises(ValueError, match="conflicts with row rollout_profile"):
            loader.get_next_item()


def test_row_profile_is_retained_without_source_override_and_handles_close(tmp_path):
    source = _source(tmp_path, "code", 1, rows=[{"task_id": "task-a", "rollout_profile": "custom"}])
    loader = MixtureDataLoader(_config([source]))
    assert _take(loader, 1)[0]["rollout_profile"] == "custom"
    handles = list(loader._sources[0].dataset._jsonl_handles.values())
    assert handles and all(not handle.closed for handle in handles)
    loader.close()
    assert all(handle.closed for handle in handles)


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_source_profile_fills_missing_pandas_profile_cells(tmp_path, extension):
    path = tmp_path / f"code.{extension}"
    dataframe = pd.DataFrame({"task_id": ["original-task"], "rollout_profile": [float("nan")]})
    if extension == "parquet":
        dataframe.to_parquet(path)
    else:
        dataframe.to_csv(path, index=False)
    source = {
        "type": "local",
        "name": "code",
        "files": [str(path)],
        "ratio": 1,
        "rollout_profile": "code-agent",
    }
    loader = MixtureDataLoader(_config([source]))
    item = _take(loader, 1)[0]
    assert item["task_id"] == "original-task"
    assert item["rollout_profile"] == "code-agent"


@pytest.mark.parametrize("cycle", [False, True])
def test_empty_source_fails_fast_instead_of_spinning(tmp_path, cycle):
    source = _source(tmp_path, "empty", 0)
    with pytest.raises(ValueError, match="no usable samples"):
        MixtureDataLoader(_config([source], cycle=cycle))


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_source_ratio_fails_fast(tmp_path, ratio):
    source = _source(tmp_path, "code", 1, ratio)
    with pytest.raises(ValueError, match="ratio must be finite and positive"):
        MixtureDataLoader(_config([source]))


def test_large_finite_weights_do_not_overflow_the_categorical_sampler(tmp_path):
    sources = [_source(tmp_path, "code", 1, 1e308), _source(tmp_path, "swe", 1, 1e308)]
    loader = MixtureDataLoader(_config(sources, batch_size=100))
    counts = Counter(item["data_source"] for item in _take(loader, 1000))
    assert abs(counts["code"] - 500) < 100


def test_duplicate_names_and_nonlocal_sources_are_rejected(tmp_path):
    source = _source(tmp_path, "code", 1)
    with pytest.raises(ValueError, match="nonempty and unique"):
        MixtureDataLoader(_config([source, source]))
    source["type"] = "managed"
    with pytest.raises(ValueError, match="local sources only"):
        MixtureDataLoader(_config([source]))


@pytest.mark.parametrize("field", ["name", "rollout_profile"])
def test_source_and_profile_names_reject_boundary_whitespace(tmp_path, field):
    source = _source(tmp_path, "code", 1)
    source[field] = " code"
    with pytest.raises(ValueError, match="boundary whitespace"):
        MixtureDataLoader(_config([source]))


def test_validation_does_not_use_random_training_mixture(tmp_path):
    source = _source(tmp_path, "code", 1)
    with pytest.raises(ValueError, match="RL training only"):
        MixtureDataLoader(_config([source]), is_validate=True)
