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

"""Tests for Weaver resume-from-checkpoint wiring (issue #244)."""

from __future__ import annotations

import asyncio
import types
from contextlib import nullcontext
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from northrl.controller import NorthRLController
from northrl.main import _make_retry_config
from northrl.trainer.base_trainer import BaseTrainer
from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.utils.async_utils import run_coro


class _CheckpointRemoteApiTrainer(RemoteApiTrainer):
    async def _prepare_trajectories(self, trajectories, metrics):
        del metrics
        return trajectories


class _FakeWeaverHolder:
    def __init__(self, sampling_path: str = "weaver://state/abc/resumed") -> None:
        self.loaded_path: str | None = None
        self.sampler_save_calls: list[str] = []
        self._sampler_path = sampling_path

    async def load_state_with_optimizer(self, path: str) -> None:
        self.loaded_path = path

    async def save_weights_for_sampler(self, name: str) -> str:
        self.sampler_save_calls.append(name)
        return self._sampler_path


class _FakeTrainer:
    def __init__(self) -> None:
        self.train_step: int | None = None

    def set_train_step(self, step: int) -> None:
        self.train_step = step


class _FakeDataLoader:
    def __init__(self) -> None:
        self.skipped: int | None = None
        self.loaded_states: list[dict[str, Any]] = []

    def skip_batches(self, n: int) -> None:
        self.skipped = n

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded_states.append(state)


class _FakeWeightSyncController:
    def __init__(self) -> None:
        self.synced: list[Any] = []

    def sync_weight_to_rollout_service(
        self, model_tag: Any, sampling_path: str, weight_version: str
    ) -> None:
        self.synced.append((model_tag, sampling_path, weight_version))


class _FakeActivityTracker:
    def __init__(self) -> None:
        self.training_steps: list[int] = []

    def set_training_step(self, step: int) -> None:
        self.training_steps.append(step)


def _make_config(
    mode: str,
    resume_path: str = "",
    checkpoint_path: str = "",
    algorithm_type: str = "grpo",
) -> Any:
    config: dict[str, Any] = {
        "training": {
            "resume": {
                "mode": mode,
                "path": resume_path,
                "skip_dataloader": True,
            },
            "save": {
                "dir": checkpoint_path,
            },
        },
        "algorithm": {"type": algorithm_type},
        "rollout": {},
    }
    if algorithm_type != "sft":
        config["rollout"].update({"type": "default"})
    return OmegaConf.create(config)


def _make_controller(config) -> NorthRLController:
    """Construct a controller with the bare attributes the resume helper touches."""
    controller = NorthRLController.__new__(NorthRLController)
    controller._config = config  # type: ignore[attr-defined]
    controller.use_tinker = False  # type: ignore[attr-defined]
    controller.use_weaver = True  # type: ignore[attr-defined]
    controller._weaver_service_holder = _FakeWeaverHolder()  # type: ignore[attr-defined]
    controller.trainer = _FakeTrainer()  # type: ignore[assignment]
    controller.dataloader = _FakeDataLoader()  # type: ignore[assignment]
    controller.weight_sync_controller = _FakeWeightSyncController()  # type: ignore[assignment]
    controller.activity_tracker = _FakeActivityTracker()  # type: ignore[assignment]
    return controller


def test_extract_global_step_from_path():
    extract = NorthRLController._extract_global_step_from_path
    assert extract("weaver://state/abc/global_step_500") == 500
    assert extract("/tmp/run/global_step_42/extra") == 42
    assert extract("no-step-here") == 0
    assert extract("") == 0


def test_load_initial_checkpoint_weaver_disabled_is_noop():
    controller = _make_controller(_make_config("disable"))
    controller._load_initial_checkpoint_weaver()
    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path is None
    assert holder.sampler_save_calls == []
    assert controller.weight_sync_controller.synced == []


def test_load_initial_checkpoint_weaver_auto_with_no_index_is_noop(tmp_path):
    config = _make_config("auto", checkpoint_path=str(tmp_path))
    controller = _make_controller(config)
    controller._load_initial_checkpoint_weaver()
    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path is None
    assert controller.weight_sync_controller.synced == []


@pytest.mark.parametrize("fallback_path", ["", "weaver://run/global_step_000005"])
def test_load_initial_checkpoint_weaver_auto_picks_latest_index_entry(tmp_path, fallback_path):
    import json

    index_path = tmp_path / "weaver_checkpoints.jsonl"
    index_path.write_text(
        json.dumps({"step": 10, "path": "weaver://run/global_step_000010"})
        + "\n"
        + json.dumps(
            {
                "step": 30,
                "path": "weaver://run/global_step_000030",
            }
        )
        + "\n"
        + json.dumps({"step": 20, "path": "weaver://run/global_step_000020"})
        + "\n"
    )

    config = _make_config("auto", resume_path=fallback_path, checkpoint_path=str(tmp_path))
    controller = _make_controller(config)
    controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path == "weaver://run/global_step_000030"
    assert controller.trainer.train_step == 30  # type: ignore[attr-defined]
    assert controller.dataloader.skipped == 30  # type: ignore[attr-defined]
    assert holder.sampler_save_calls == ["northrl_actor_v000030"]
    assert controller.weight_sync_controller.synced == [
        ("actor", "weaver://state/abc/resumed", "northrl_actor_v000030")
    ]
    assert controller.activity_tracker.training_steps == [30]  # type: ignore[attr-defined]


@pytest.mark.parametrize("index_contents", [None, "", 'not-json\n{"step": 1001}\n'])
def test_retry_preserves_explicit_checkpoint_when_no_valid_index_exists(tmp_path, index_contents):
    if index_contents is not None:
        (tmp_path / "weaver_checkpoints.jsonl").write_text(index_contents)
    resume_path = "weaver://original/checkpoints/global_step_001000"
    config = _make_config("from_path", resume_path=resume_path, checkpoint_path=str(tmp_path))
    controller = _make_controller(_make_retry_config(config))

    controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder
    assert holder.loaded_path == resume_path
    assert controller.trainer.train_step == 1000
    assert controller.activity_tracker.training_steps == [1000]
    assert controller.dataloader.skipped == 1000
    assert holder.sampler_save_calls == ["northrl_actor_v001000"]
    assert controller.weight_sync_controller.synced == [
        ("actor", "weaver://state/abc/resumed", "northrl_actor_v001000")
    ]


def test_retry_does_not_activate_disabled_resume_path(tmp_path):
    config = _make_config(
        "disable",
        resume_path="weaver://unused/checkpoints/global_step_001000",
        checkpoint_path=str(tmp_path),
    )
    controller = _make_controller(_make_retry_config(config))

    controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder
    assert holder.loaded_path is None
    assert controller.trainer.train_step is None
    assert controller.dataloader.skipped is None
    assert holder.sampler_save_calls == []
    assert controller.weight_sync_controller.synced == []


def test_auto_resume_does_not_fall_back_after_checkpoint_load_failure(tmp_path, monkeypatch):
    import json

    latest_path = "weaver://run/global_step_001010"
    (tmp_path / "weaver_checkpoints.jsonl").write_text(
        json.dumps({"step": 1010, "path": latest_path}) + "\n"
    )
    config = _make_config(
        "auto",
        resume_path="weaver://original/global_step_001000",
        checkpoint_path=str(tmp_path),
    )
    controller = _make_controller(config)
    holder = controller._weaver_service_holder
    load_state = AsyncMock(side_effect=RuntimeError("checkpoint load failed"))
    monkeypatch.setattr(holder, "load_state_with_optimizer", load_state)

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        controller._load_initial_checkpoint_weaver()

    load_state.assert_awaited_once_with(latest_path)
    assert controller.trainer.train_step is None
    assert controller.dataloader.skipped is None
    assert holder.sampler_save_calls == []
    assert controller.weight_sync_controller.synced == []


def test_find_latest_weaver_checkpoint_skips_malformed_lines(tmp_path):
    import json

    index_path = tmp_path / "weaver_checkpoints.jsonl"
    index_path.write_text(
        "not-json-at-all\n"
        + json.dumps({"step": 5, "path": "weaver://run/global_step_000005"})
        + "\n"
        + json.dumps({"step": 7})  # missing path
        + "\n"
        + "\n"  # blank line
        + json.dumps({"step": 12, "path": "weaver://run/global_step_000012"})
        + "\n"
    )
    config = _make_config("auto", checkpoint_path=str(tmp_path))
    controller = _make_controller(config)
    assert controller._find_latest_weaver_checkpoint() == "weaver://run/global_step_000012"


def test_load_initial_checkpoint_weaver_requires_resume_path():
    controller = _make_controller(_make_config("from_path", resume_path=""))
    with pytest.raises(ValueError, match="resume_path"):
        controller._load_initial_checkpoint_weaver()


def test_load_initial_checkpoint_weaver_from_path_uses_matching_index_metadata(tmp_path):
    import json

    requested_path = "weaver://state/abc/custom-checkpoint-name"
    index_path = tmp_path / "weaver_checkpoints.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "step": 10,
                "path": requested_path,
            }
        )
        + "\n"
    )
    config = _make_config(
        "from_path",
        resume_path=requested_path,
        checkpoint_path=str(tmp_path),
    )
    controller = _make_controller(config)

    controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path == requested_path
    assert holder.sampler_save_calls == ["northrl_actor_v000010"]
    assert controller.trainer.train_step == 10  # type: ignore[attr-defined]
    assert controller.dataloader.skipped == 10  # type: ignore[attr-defined]
    assert controller.weight_sync_controller.synced == [
        ("actor", "weaver://state/abc/resumed", "northrl_actor_v000010")
    ]
    assert controller.activity_tracker.training_steps == [10]  # type: ignore[attr-defined]


def test_load_initial_checkpoint_weaver_from_path_drives_full_resume():
    config = _make_config("from_path", resume_path="weaver://state/abc/global_step_000100")
    controller = _make_controller(config)

    controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path == "weaver://state/abc/global_step_000100"
    assert holder.sampler_save_calls == ["northrl_actor_v000100"]

    assert controller.trainer.train_step == 100  # type: ignore[attr-defined]
    assert controller.dataloader.skipped == 100  # type: ignore[attr-defined]
    assert controller.weight_sync_controller.synced == [
        ("actor", "weaver://state/abc/resumed", "northrl_actor_v000100")
    ]
    assert controller.activity_tracker.training_steps == [100]  # type: ignore[attr-defined]


def _assert_sft_resume_did_not_touch_sampler(controller):
    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.sampler_save_calls == []
    assert controller.weight_sync_controller.synced == []


def test_load_initial_checkpoint_weaver_sft_auto_restores_exact_index_entry(tmp_path):
    import json

    old_cursor = {"version": 1, "committed_cursor": {"epoch": 1, "position": 4}}
    latest_cursor = {"version": 1, "committed_cursor": {"epoch": 3, "position": 9}}
    index_path = tmp_path / "weaver_checkpoints.jsonl"
    index_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 10,
                        "path": "weaver://run/global_step_000010",
                        "dataloader_state": old_cursor,
                    }
                ),
                json.dumps(
                    {
                        "step": 30,
                        "path": "weaver://run/global_step_000030",
                        "dataloader_state": latest_cursor,
                    }
                ),
                json.dumps(
                    {
                        "step": 20,
                        "path": "weaver://run/global_step_000020",
                        "dataloader_state": {
                            "version": 1,
                            "committed_cursor": {"epoch": 2, "position": 6},
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    controller = _make_controller(
        _make_config("auto", checkpoint_path=str(tmp_path), algorithm_type="sft")
    )

    controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path == "weaver://run/global_step_000030"
    assert controller.dataloader.loaded_states == [latest_cursor]
    assert controller.dataloader.skipped is None
    assert controller.trainer.train_step == 30
    assert controller.activity_tracker.training_steps == [30]  # type: ignore[attr-defined]
    _assert_sft_resume_did_not_touch_sampler(controller)


def test_load_initial_checkpoint_weaver_sft_from_path_uses_matching_entry(tmp_path):
    import json

    requested_path = "weaver://run/checkpoint-requested-by-user"
    requested_cursor = {
        "version": 1,
        "committed_cursor": {"epoch": 4, "position": 11},
    }
    index_path = tmp_path / "weaver_checkpoints.jsonl"
    index_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 18,
                        "path": requested_path,
                        "dataloader_state": requested_cursor,
                    }
                ),
                json.dumps(
                    {
                        "step": 99,
                        "path": "weaver://run/newer-but-not-requested",
                        "dataloader_state": {
                            "version": 1,
                            "committed_cursor": {"epoch": 9, "position": 99},
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    controller = _make_controller(
        _make_config(
            "from_path",
            resume_path=requested_path,
            checkpoint_path=str(tmp_path),
            algorithm_type="sft",
        )
    )

    controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path == requested_path
    assert controller.dataloader.loaded_states == [requested_cursor]
    assert controller.dataloader.skipped is None
    assert controller.trainer.train_step == 18
    assert controller.activity_tracker.training_steps == [18]  # type: ignore[attr-defined]
    _assert_sft_resume_did_not_touch_sampler(controller)


def test_load_initial_checkpoint_weaver_sft_requires_dataloader_state(tmp_path):
    import json

    index_path = tmp_path / "weaver_checkpoints.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "step": 10,
                "path": "weaver://run/global_step_000010",
            }
        )
        + "\n"
    )
    controller = _make_controller(
        _make_config("auto", checkpoint_path=str(tmp_path), algorithm_type="sft")
    )

    with pytest.raises(ValueError, match="dataloader_state"):
        controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path is None
    assert controller.dataloader.loaded_states == []
    assert controller.trainer.train_step is None
    _assert_sft_resume_did_not_touch_sampler(controller)


def test_sft_retry_fallback_requires_matching_dataloader_state(tmp_path):
    config = _make_config(
        "from_path",
        resume_path="weaver://original/global_step_001000",
        checkpoint_path=str(tmp_path),
        algorithm_type="sft",
    )
    controller = _make_controller(_make_retry_config(config))

    with pytest.raises(ValueError, match="matching Weaver checkpoint index entry"):
        controller._load_initial_checkpoint_weaver()

    assert controller._weaver_service_holder.loaded_path is None
    assert controller.dataloader.loaded_states == []
    assert controller.trainer.train_step is None
    _assert_sft_resume_did_not_touch_sampler(controller)


def test_load_initial_checkpoint_weaver_sft_from_path_requires_matching_entry(tmp_path):
    import json

    index_path = tmp_path / "weaver_checkpoints.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "step": 10,
                "path": "weaver://run/a-different-checkpoint",
                "dataloader_state": {"version": 1},
            }
        )
        + "\n"
    )
    controller = _make_controller(
        _make_config(
            "from_path",
            resume_path="weaver://run/missing-checkpoint",
            checkpoint_path=str(tmp_path),
            algorithm_type="sft",
        )
    )

    with pytest.raises(ValueError, match="matching Weaver checkpoint index entry"):
        controller._load_initial_checkpoint_weaver()

    holder = controller._weaver_service_holder  # type: ignore[attr-defined]
    assert holder.loaded_path is None
    assert controller.dataloader.loaded_states == []
    assert controller.trainer.train_step is None
    _assert_sft_resume_did_not_touch_sampler(controller)


def test_weaver_holder_save_weight_and_optimizer_requests_full_checkpoint():
    from northrl.weaver.weaver_service_holder import WeaverServiceHolder

    calls: list[dict[str, Any]] = []

    class _TrainingClient:
        async def save_state(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return types.SimpleNamespace(path="weaver://state/abc/global_step_000100")

    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._closed = False  # type: ignore[attr-defined]
    holder._training_client = _TrainingClient()  # type: ignore[attr-defined]
    holder._tokenizer = object()  # type: ignore[attr-defined]

    try:
        path = run_coro(holder.save_weight_and_optimizer(name="global_step_000100"))
    finally:
        holder._closed = True  # type: ignore[attr-defined]

    assert path == "weaver://state/abc/global_step_000100"
    assert calls == [
        {
            "name": "global_step_000100",
            "checkpoint_type": "weight_and_optimizer",
        }
    ]


def test_remote_api_trainer_save_full_checkpoint_invokes_full_save(tmp_path):
    """Periodic saving uses the full-state holder API and updates the index."""
    import json

    holder_calls: list[str] = []

    class _Holder:
        async def save_weight_and_optimizer(self, name: str) -> str:
            holder_calls.append(name)
            return f"weaver://state/abc/{name}"

    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    trainer._service_holder = _Holder()  # type: ignore[attr-defined]
    trainer._config = OmegaConf.create({"training": {"save": {"dir": str(tmp_path)}}})  # type: ignore[attr-defined]

    path = run_coro(trainer._save_full_checkpoint(step=100))

    assert holder_calls == ["global_step_000100"]
    assert path == "weaver://state/abc/global_step_000100"

    # The auto-resume index file should now contain one entry pointing at the saved path.
    index_file = tmp_path / "weaver_checkpoints.jsonl"
    assert index_file.exists()
    entries = [json.loads(line) for line in index_file.read_text().splitlines() if line]
    assert entries == [
        {
            "step": 100,
            "path": "weaver://state/abc/global_step_000100",
            "ts": entries[0]["ts"],  # timestamp varies
        }
    ]


def test_remote_api_trainer_saves_final_checkpoint_once():
    holder_calls: list[str] = []

    class _Holder:
        async def save_weight_and_optimizer(self, name: str) -> str:
            holder_calls.append(name)
            return f"weaver://state/abc/{name}"

    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    trainer._service_holder = _Holder()  # type: ignore[attr-defined]
    trainer._config = OmegaConf.create(  # type: ignore[attr-defined]
        {"training": {"save": {"dir": ""}}}
    )
    trainer._train_step = 7  # type: ignore[attr-defined]
    trainer._last_full_checkpoint_step = None  # type: ignore[attr-defined]

    run_coro(trainer._on_training_complete())
    run_coro(trainer._on_training_complete())

    assert holder_calls == ["global_step_000007"]
    assert trainer._last_full_checkpoint_step == 7  # type: ignore[attr-defined]


def test_remote_api_trainer_skips_final_checkpoint_when_disabled():
    save = AsyncMock(return_value="weaver://state/abc/global_step_000007")
    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    trainer._service_holder = types.SimpleNamespace(save_weight_and_optimizer=save)  # type: ignore[attr-defined]
    trainer._config = OmegaConf.create(  # type: ignore[attr-defined]
        {"training": {"save": {"dir": "", "final": False}}}
    )
    trainer._train_step = 7  # type: ignore[attr-defined]
    trainer._last_full_checkpoint_step = None  # type: ignore[attr-defined]

    run_coro(trainer._on_training_complete())

    save.assert_not_awaited()


def _make_checkpoint_loop_trainer(tmp_path):
    config = OmegaConf.create(
        {
            "algorithm": {"type": "grpo"},
            "training": {
                "total_steps": 150,
                "save": {"every_n_steps": 20, "dir": str(tmp_path)},
            },
        }
    )
    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    BaseTrainer.__init__(trainer, config)
    trainer._train_step = 3
    trainer._last_full_checkpoint_step = None
    trainer.train = AsyncMock(return_value=({}, "weaver://sampler/4", "northrl_actor_v000004"))
    trainer._service_holder = types.SimpleNamespace(
        save_weight_and_optimizer=AsyncMock(return_value="weaver://state/global_step_000004")
    )
    trainer.set_activity_tracker(
        types.SimpleNamespace(
            track=lambda *_args: nullcontext(),
            experiment_logger_post=lambda **_kwargs: None,
            set_training_step=lambda _step: None,
        )
    )
    sync_error = TimeoutError("sampling pause timed out")
    trainer.set_module_references(
        types.SimpleNamespace(get_trajectories=lambda: [], get_discarded_metrics=dict),
        types.SimpleNamespace(
            is_waiting_for_validation=lambda: False,
            train_worker_notify_weight_update=Mock(side_effect=sync_error),
        ),
    )
    return trainer, sync_error


@pytest.mark.parametrize("already_saved", [False, True])
def test_weight_sync_failure_saves_completed_step_once(tmp_path, already_saved):
    import json

    trainer, sync_error = _make_checkpoint_loop_trainer(tmp_path)
    if already_saved:
        # A periodic checkpoint may already have saved this round before synchronization.
        run_coro(trainer._save_full_checkpoint(4))

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(trainer._main_loop())

    assert caught.value is sync_error
    trainer.train.assert_awaited_once_with([])
    trainer._service_holder.save_weight_and_optimizer.assert_awaited_once_with(
        name="global_step_000004"
    )
    assert trainer.get_train_step() == 3
    assert trainer._last_full_checkpoint_step == 4
    entries = [
        json.loads(line)
        for line in (tmp_path / "weaver_checkpoints.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["step"] == 4
    assert entries[0]["path"] == "weaver://state/global_step_000004"


@pytest.mark.parametrize("failure_source", ["save_api", "hook"])
def test_weight_sync_failure_preserves_original_error_when_checkpoint_fails(
    tmp_path, failure_source, monkeypatch, caplog
):
    trainer, sync_error = _make_checkpoint_loop_trainer(tmp_path)
    save = trainer._service_holder.save_weight_and_optimizer
    checkpoint_error = RuntimeError("checkpoint unavailable")
    if failure_source == "save_api":
        save.side_effect = checkpoint_error
    else:
        hook = AsyncMock(side_effect=checkpoint_error)
        monkeypatch.setattr(trainer, "_on_weight_sync_failed", hook)

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(trainer._main_loop())

    assert caught.value is sync_error
    assert trainer.get_train_step() == 3
    assert trainer._last_full_checkpoint_step is None
    assert not (tmp_path / "weaver_checkpoints.jsonl").exists()
    assert "checkpoint unavailable" in caplog.text
    if failure_source == "save_api":
        save.assert_awaited_once_with(name="global_step_000004")
    else:
        hook.assert_awaited_once_with(4)
        save.assert_not_awaited()


def test_training_failure_does_not_save_a_checkpoint(tmp_path):
    trainer, _ = _make_checkpoint_loop_trainer(tmp_path)
    train_error = RuntimeError("optimizer did not complete")
    trainer.train.side_effect = train_error

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(trainer._main_loop())

    assert caught.value is train_error
    trainer._weight_sync_controller.train_worker_notify_weight_update.assert_not_called()
    trainer._service_holder.save_weight_and_optimizer.assert_not_awaited()
    assert trainer.get_train_step() == 3
    assert not (tmp_path / "weaver_checkpoints.jsonl").exists()


def test_resumed_completed_step_selects_the_next_cosine_learning_rate():
    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    trainer._learning_rate = 2.0e-5  # type: ignore[attr-defined]
    trainer._lr_schedule = OmegaConf.create(  # type: ignore[attr-defined]
        {"style": "cosine", "min_lr": 2.0e-6, "warmup_fraction": 0.03}
    )
    trainer._lr_schedule_total_steps = 85  # type: ignore[attr-defined]

    trainer.set_train_step(2)

    assert trainer.get_train_step() == 2
    assert trainer._effective_learning_rate() == pytest.approx(1.568627450980392e-5)


def test_remote_checkpoint_index_binds_committed_sft_cursor(tmp_path):
    import json

    cursor = {
        "version": 1,
        "epoch": 3,
        "position_in_epoch": 17,
        "shuffle_algorithm": "algorithm",
        "seed": 42,
    }

    class _DataLoader:
        @staticmethod
        def state_dict():
            return cursor

    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    trainer._config = OmegaConf.create(  # type: ignore[attr-defined]
        {"training": {"save": {"dir": str(tmp_path)}}}
    )
    trainer._dataloader = _DataLoader()  # type: ignore[attr-defined]

    trainer._append_remote_checkpoint_index(18, "weaver://state/step-18")

    entry = json.loads((tmp_path / "weaver_checkpoints.jsonl").read_text())
    assert entry["step"] == 18
    assert entry["path"] == "weaver://state/step-18"
    assert entry["dataloader_state"] == cursor
    assert entry["schema_version"] == 1


def test_remote_api_trainer_save_full_checkpoint_handles_missing_checkpoint_path(tmp_path):
    """When trainer.checkpoint_path is empty the index update is skipped, not fatal."""

    class _Holder:
        async def save_weight_and_optimizer(self, name: str) -> str:
            return f"weaver://state/abc/{name}"

    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    trainer._service_holder = _Holder()  # type: ignore[attr-defined]
    trainer._config = OmegaConf.create({"training": {"save": {"dir": ""}}})  # type: ignore[attr-defined]

    # Should not raise even though no directory was provided.
    path = run_coro(trainer._save_full_checkpoint(step=42))
    assert path == "weaver://state/abc/global_step_000042"


def test_remote_api_trainer_save_full_checkpoint_returns_none_when_save_fails():
    class _Holder:
        async def save_weight_and_optimizer(self, name: str) -> str:
            raise RuntimeError(f"failed to save {name}")

    trainer = _CheckpointRemoteApiTrainer.__new__(_CheckpointRemoteApiTrainer)
    trainer._service_holder = _Holder()  # type: ignore[attr-defined]

    assert run_coro(trainer._save_full_checkpoint(step=100)) is None
