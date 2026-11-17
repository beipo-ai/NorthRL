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

"""Focused tests for async GRPO/OPD auxiliary Weaver clients."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

from northrl.algorithm.advantage import build_estimator
from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.trainer.remote_api_opd_trainer import RemoteApiOpdTrainer
from northrl.utils.async_utils import submit_trainer_coro
from northrl.utils.config_schema import AlgorithmConfig
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


def _run_on_trainer_loop(coro):
    return submit_trainer_coro(coro).result(timeout=5)


class _FakeRefClient:
    model_id = "ref-client"

    def __init__(self) -> None:
        self.load_calls: list[tuple[str, str]] = []

    async def load_state(self, path: str) -> None:
        self.load_calls.append((path, threading.current_thread().name))


class _FakeRefHolder(WeaverServiceHolder):
    def __init__(self) -> None:
        # Do not provision a real Weaver service; only the fields read by
        # RemoteApiGrpoTrainer.initialize_workers are needed.
        self._closed = True
        self._base_model = "actor-base"
        self._lora_rank = 16
        self.client = _FakeRefClient()
        self.create_calls: list[tuple[dict[str, Any], str]] = []

    async def create_training_client(self, **kwargs: Any) -> Any:
        self.create_calls.append((kwargs, threading.current_thread().name))
        return self.client


class _FakeTeacherHolder:
    def __init__(self) -> None:
        self.create_calls: list[tuple[dict[str, Any], str]] = []

    async def create_sampling_client(self, **kwargs: Any) -> Any:
        self.create_calls.append((kwargs, threading.current_thread().name))
        return SimpleNamespace(client_index=len(self.create_calls))


class _FakeInferenceTeacher:
    def __init__(self) -> None:
        self.call_threads: list[str] = []
        self.completion_order: list[int] = []
        self._release_middle = asyncio.Event()
        self._release_slow = asyncio.Event()

    async def compute_logprobs(self, *, prompt) -> list[float]:
        token = prompt.to_ints()[0]
        self.call_threads.append(threading.current_thread().name)

        # Enforce completion order 20 -> 30 -> 10 without timing assumptions.
        if token == 10:
            await self._release_slow.wait()
        elif token == 20:
            self._release_middle.set()
            self.completion_order.append(token)
            raise RuntimeError("one teacher request failed")
        elif token == 30:
            await self._release_middle.wait()
            self.completion_order.append(token)
            self._release_slow.set()

        if token == 10:
            self.completion_order.append(token)
        return [float(token), float(token) + 0.5]


def _trajectory(first_token: int) -> Trajectory:
    return Trajectory(
        tokens=[first_token, first_token + 1],
        loss_mask=[0, 1],
        reward=0.0,
        is_val=False,
    )


def test_grpo_initialize_workers_creates_and_loads_ref_client_on_sdk_loop():
    trainer = object.__new__(RemoteApiGrpoTrainer)
    trainer._kl_enabled = True
    trainer._ref_training_client = None
    trainer._ref_model_cfg = SimpleNamespace(name=None, checkpoint="weaver://ref/checkpoint")
    holder = _FakeRefHolder()
    trainer.set_service_holder(holder)

    _run_on_trainer_loop(trainer.initialize_workers())
    _run_on_trainer_loop(trainer.initialize_workers())

    assert holder.create_calls == [
        (
            {
                "base_model": "actor-base",
                "training_mode": "lora",
                "lora_config": {"rank": 16},
            },
            "northrl-async-loop",
        )
    ]
    assert holder.client.load_calls == [("weaver://ref/checkpoint", "northrl-async-loop")]
    assert trainer._ref_training_client is holder.client


def test_opd_initialize_workers_creates_ordered_teacher_clients_on_sdk_loop():
    trainer = object.__new__(RemoteApiOpdTrainer)
    trainer._teacher_logprob_source = "inference_service"
    trainer._teacher_configs = [
        SimpleNamespace(name="teacher-a", checkpoint=None),
        SimpleNamespace(name="teacher-b", checkpoint="weaver://teacher-b/checkpoint"),
    ]
    trainer._teacher_clients = []
    holder = _FakeTeacherHolder()
    trainer.set_service_holder(holder)

    _run_on_trainer_loop(trainer.initialize_workers())
    _run_on_trainer_loop(trainer.initialize_workers())

    assert holder.create_calls == [
        ({"base_model": "teacher-a"}, "northrl-async-loop"),
        (
            {
                "base_model": "teacher-b",
                "model_path": "weaver://teacher-b/checkpoint",
            },
            "northrl-async-loop",
        ),
    ]
    assert [client.client_index for client in trainer._teacher_clients] == [1, 2]


def test_opd_inference_fanout_preserves_order_and_isolates_single_failure():
    trainer = object.__new__(RemoteApiOpdTrainer)
    trainer._teacher_service_timeout = 5.0
    teacher = _FakeInferenceTeacher()
    trajectories = [_trajectory(10), _trajectory(20), _trajectory(30)]

    results = _run_on_trainer_loop(
        trainer._teacher_logprobs_from_inference_service(teacher, trajectories)
    )

    assert teacher.completion_order == [20, 30, 10]
    assert teacher.call_threads == ["northrl-async-loop"] * 3
    assert results == [[10.0, 10.5], [], [30.0, 30.5]]


def test_reference_kl_attaches_logprobs_to_plain_and_nested_sequences():
    class RefClient:
        def __init__(self):
            self.batch_sizes = []

        async def forward(self, datums, **kwargs):
            self.batch_sizes.append(len(datums))
            return {"result": {"loss_fn_outputs": [{"logprobs": [-0.3]} for _ in datums]}}

    trainer = object.__new__(RemoteApiGrpoTrainer)
    trainer._ref_training_client = RefClient()
    trainer._ref_logprobs_batch_size = 2
    plain = _trajectory(10)
    nested_sequences = [
        {"tokens": [20, 21], "loss_mask": [0, 1]},
        {"tokens": [30, 31], "loss_mask": [0, 1]},
    ]
    nested = Trajectory(segments=nested_sequences, reward=1.0)
    metrics = {}

    _run_on_trainer_loop(trainer._compute_and_attach_ref_logprobs([plain, nested], metrics))

    assert trainer._ref_training_client.batch_sizes == [2, 1]
    assert plain["ref_logprobs"] == [0.0, -0.3]
    assert [sequence["ref_logprobs"] for sequence in nested_sequences] == [[0.0, -0.3]] * 2
    assert plain.segments is None
    assert nested.segments is nested_sequences
    assert "ref_logprobs" not in nested.extra_fields
    assert "kl/ref_logprobs_attached" not in metrics
    assert metrics["timing/ref_logprobs"] >= 0


def test_opd_preparation_returns_parents_and_updates_nested_token_advantages(monkeypatch):
    import pytest

    from northrl.utils.finetune_service_utils import convert_trajectories_to_datums

    trainer = object.__new__(RemoteApiOpdTrainer)
    trainer._algorithm_config = AlgorithmConfig(
        type="opd",
        optim={"lr": 1e-6},
        advantage={"estimator": {"mode": "opd", "params": {"kl_penalty_coef": 0.5}}},
    )
    trainer._advantage_estimator = build_estimator(trainer._algorithm_config.advantage.estimator)
    trainer._kl_penalty_coef = 0.5
    plain = _trajectory(10)
    plain["old_log_probs"] = [0.0, -2.0]
    nested_sequences = [
        {"tokens": [20, 21], "loss_mask": [0, 1], "old_log_probs": [0.0, -3.0]},
        {"tokens": [30, 31], "loss_mask": [0, 1], "old_log_probs": [0.0, -4.0]},
    ]
    nested = Trajectory(segments=nested_sequences, reward=1.0)
    parents = [plain, nested]

    async def teacher_logprobs(sequences):
        assert sequences[0] is plain
        assert all(sequences[i + 1] is nested_sequences[i] for i in range(2))
        return [[0.0, -1.0]] * 3

    monkeypatch.setattr(trainer, "_compute_teacher_logprobs_parallel", teacher_logprobs)
    metrics = {}
    prepared = _run_on_trainer_loop(trainer._prepare_trajectories(parents, metrics))
    inputs = _run_on_trainer_loop(trainer._prepare_advantage_inputs(prepared, metrics))
    prepared = trainer._advantage_estimator.estimate(prepared, inputs, metrics)
    datums = convert_trajectories_to_datums(prepared)

    assert prepared is parents
    assert plain.segments is None
    assert "token_advantages" not in nested.extra_fields
    assert [datum["loss_fn_inputs"]["advantages"][0] for datum in datums] == pytest.approx(
        [0.5, 1.0, 1.5]
    )
