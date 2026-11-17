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

"""Exercise complete train rounds across the shared advantage stage boundary."""

import math

import pytest

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.trainer.remote_api_opd_trainer import RemoteApiOpdTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.trajectory_sequences import iter_training_sequences
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyActivityTracker,
    _DummyDataDumper,
    _FakeWeaverHolder,
    _FakeWeightSyncController,
    _make_config,
)


class _PreparedDumper(_DummyDataDumper):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.prepared = None

    def should_dump(self, data_type, step):
        return data_type == "prepared_trajectories"

    def dump_prepared_trajectories(self, *, step, trajectories):
        self.events.append("dump")
        self.prepared = trajectories


def _trainer(
    monkeypatch, events, *, flow="grpo", batch_size=None, mini_batch_size=None, coef=0.5, mis=True
):
    dumper = _PreparedDumper(events)
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *args, **kwargs: dumper)
    config = _make_config(batch_size=batch_size, mini_batch_size=mini_batch_size)
    config.algorithm.type = flow
    config.algorithm.sampling_logprob_source = "recomputed"
    config.algorithm.loss.reduction.mode = "token_mean"
    config.models = {"reference": None, "teachers": []}
    if flow == "grpo":
        config.algorithm.mis = {"enabled": mis, "tis_mode": "mask", "tis_upper_bound": 2.0}
        trainer_type = RemoteApiGrpoTrainer
    else:
        config.algorithm.advantage = {
            "estimator": {"mode": "opd", "params": {"kl_penalty_coef": coef}}
        }
        trainer_type = RemoteApiOpdTrainer
    trainer = trainer_type(config)
    trainer._activity_tracker = _DummyActivityTracker()
    trainer._weight_sync_controller = _FakeWeightSyncController()
    trainer.set_service_holder(_FakeWeaverHolder())
    return trainer, dumper


def _parents():
    parents = []
    for index, (reward, segment_count) in enumerate(((0.0, 1), (2.0, 2), (3.0, 1), (7.0, 0))):
        group, run = divmod(index, 2)
        parents.append(
            Trajectory(
                segments=[
                    {
                        "tokens": [10 + index, 20 + segment],
                        "loss_mask": [0, 1],
                        "old_log_probs": [0.0, -0.2],
                    }
                    for segment in range(segment_count)
                ],
                reward=reward,
                extra_fields={
                    "group_id": str(group),
                    "run_id": run,
                    "rollout_id": str(index),
                    "expected_rollouts": 2,
                },
            )
        )
    return parents


def _record_stage_calls(monkeypatch, trainer, events, check_estimate):
    original_recompute = trainer._recompute_logprobs
    original_estimator = trainer._advantage_estimator
    original_convert = trainer._convert_to_datums

    async def recompute(trajectories, metrics):
        events.append("recompute")
        return await original_recompute(trajectories, metrics)

    class RecordingEstimator:
        def estimate(self, trajectories, inputs, metrics, *, logger=None):
            events.append("estimate")
            check_estimate(trajectories, inputs)
            return original_estimator.estimate(trajectories, inputs, metrics, logger=logger)

    class RecordingPostprocessor:
        def process(self, trajectories, metrics, *, logger=None):
            events.append("postprocess")
            return trajectories

    def convert(trajectories):
        events.append("convert")
        return original_convert(trajectories)

    monkeypatch.setattr(trainer, "_recompute_logprobs", recompute)
    trainer._advantage_estimator = RecordingEstimator()
    trainer._advantage_postprocessor = RecordingPostprocessor()
    monkeypatch.setattr(trainer, "_convert_to_datums", convert)


@pytest.mark.parametrize("batch_size,mini_batch_size", [(None, None), (1, 1), (3, 2), (8, 1)])
def test_grpo_estimates_once_per_complete_step_before_dump_mis_and_splitting(
    monkeypatch, batch_size, mini_batch_size
):
    events = []
    trainer, dumper = _trainer(
        monkeypatch, events, batch_size=batch_size, mini_batch_size=mini_batch_size
    )
    parents = _parents()
    sequences = list(iter_training_sequences(parents))
    buffers = [(sequence["tokens"], sequence["loss_mask"]) for sequence in sequences]

    def check_estimate(prepared, inputs):
        assert all(actual is expected for actual, expected in zip(prepared, parents, strict=True))
        assert inputs == {}
        assert all("advantage" not in parent for parent in prepared)
        assert all(sequence["old_log_probs"][1] == -0.1 for sequence in sequences)
        assert all("correction_mask" not in sequence for sequence in sequences)

    _record_stage_calls(monkeypatch, trainer, events, check_estimate)
    original_mis = trainer._compute_mis

    def mis(prepared, metrics):
        events.append("mis")
        assert prepared is dumper.prepared
        assert all("advantage" in parent for parent in prepared)
        original_mis(prepared, metrics)

    monkeypatch.setattr(trainer, "_compute_mis", mis)
    metrics, _, _ = run_coro(trainer.train(list(reversed(parents))))

    assert events == ["recompute", "estimate", "postprocess", "dump", "mis", "convert"]
    expected = [
        -1 / (math.sqrt(2) + 1e-6),
        1 / (math.sqrt(2) + 1e-6),
        -2 / (math.sqrt(8) + 1e-6),
        2 / (math.sqrt(8) + 1e-6),
    ]
    assert [float(parent["advantage"]) for parent in parents] == pytest.approx(expected)
    # Rollout logprob diagnostics retain their original pre-recompute source.
    assert metrics["algorithm/sampling/logprob_mean"] == pytest.approx(-0.2)
    assert metrics["training/optimizer_step_count"] == (
        1 if batch_size is None else math.ceil(4 / batch_size)
    )
    for sequence, (tokens, mask) in zip(sequences, buffers, strict=True):
        assert sequence["tokens"] is tokens
        assert sequence["loss_mask"] is mask
    assert dumper.prepared[-1] is parents[-1]
    assert parents[-1].segments == []


@pytest.mark.parametrize("coef", [0.0, 0.5])
@pytest.mark.parametrize("mini_batch_size", [None, 1])
def test_opd_preserves_rollout_scores_before_recompute_through_full_train(
    monkeypatch, coef, mini_batch_size
):
    events = []
    trainer, dumper = _trainer(
        monkeypatch, events, flow="opd", coef=coef, mini_batch_size=mini_batch_size
    )
    parents = _parents()[:2]
    sequences = list(iter_training_sequences(parents))
    rollout_scores = [sequence["old_log_probs"] for sequence in sequences]
    buffers = [(sequence["tokens"], sequence["loss_mask"]) for sequence in sequences]
    teacher_scores = [[0.0, -2.0], [0.0, -3.0], [0.0, -4.0]]

    async def teacher(prepared_sequences):
        events.append("teacher")
        assert coef != 0
        assert all(
            actual is expected
            for actual, expected in zip(prepared_sequences, sequences, strict=True)
        )
        return teacher_scores

    monkeypatch.setattr(trainer, "_compute_teacher_logprobs_parallel", teacher)

    def check_estimate(prepared, inputs):
        assert all(actual is expected for actual, expected in zip(prepared, parents, strict=True))
        assert all(sequence["old_log_probs"][1] == -0.1 for sequence in sequences)
        if coef:
            assert inputs["teacher_logprobs"] is teacher_scores
            assert all(
                actual is expected
                for actual, expected in zip(inputs["student_logprobs"], rollout_scores, strict=True)
            )
            assert all(
                actual is expected
                for actual, expected in zip(inputs["sequences"], sequences, strict=True)
            )
        else:
            assert inputs == {}

    _record_stage_calls(monkeypatch, trainer, events, check_estimate)
    metrics, _, _ = run_coro(trainer.train(parents))

    assert events == (["teacher"] if coef else []) + [
        "recompute",
        "estimate",
        "postprocess",
        "dump",
        "convert",
    ]
    assert dumper.prepared is parents
    assert [sequence["token_advantages"][0] for sequence in sequences] == pytest.approx(
        [-coef * 1.8, -coef * 2.8, -coef * 3.8]
    )
    for sequence, old_scores, (tokens, mask) in zip(
        sequences, rollout_scores, buffers, strict=True
    ):
        assert sequence["tokens"] is tokens
        assert sequence["loss_mask"] is mask
        assert old_scores == [0.0, -0.2]
        assert sequence["old_log_probs"] is not old_scores
    if coef:
        assert metrics["algorithm/opd/teacher_kl"] == pytest.approx(2.8)
        datums = [
            datum for call in trainer._service_holder.builtin_calls for datum in call["datums_data"]
        ]
        assert [datum["loss_fn_inputs"]["logprobs"] for datum in datums] == [[-0.1]] * 3
    else:
        assert "algorithm/opd/teacher_kl" not in metrics


def test_incomplete_grpo_group_fails_before_any_service_io_or_estimation(monkeypatch):
    events = []
    trainer, _ = _trainer(monkeypatch, events)

    def unexpected(*args, **kwargs):
        raise AssertionError("An invalid complete group must fail before I/O or estimation")

    monkeypatch.setattr(trainer, "_prepare_advantage_inputs", unexpected)

    class UnexpectedEstimator:
        estimate = staticmethod(unexpected)

    trainer._advantage_estimator = UnexpectedEstimator()
    with pytest.raises(ValueError, match="Incomplete trajectory group"):
        run_coro(trainer.train(_parents()[:1]))
    assert events == []


@pytest.mark.parametrize("batch_size,mini_batch_size", [(None, None), (1, 1), (2, 1), (4, 2)])
def test_postprocessed_token_advantages_reach_service_after_full_step_processing(
    monkeypatch, batch_size, mini_batch_size
):
    trainer, dumper = _trainer(
        monkeypatch, [], batch_size=batch_size, mini_batch_size=mini_batch_size, mis=False
    )
    # Isolate the postprocessor payload from the separate loss reduction.
    trainer._algorithm_config.loss.reduction.mode = "sum"
    parents = _parents()
    processed = []

    class TokenPostprocessor:
        def process(self, trajectories, metrics, *, logger=None):
            assert len(trajectories) == 4
            assert not processed  # Exactly once, including the empty parent.
            assert all("advantage" in parent for parent in trajectories)
            processed.extend(trajectories)
            for index, sequence in enumerate(iter_training_sequences(processed)):
                sequence["token_advantages"] = [float(len(processed) + index)]
            metrics["postprocessing/parent_count"] = len(processed)
            return processed

    trainer._advantage_postprocessor = TokenPostprocessor()
    metrics, _, _ = run_coro(trainer.train(parents))

    assert dumper.prepared is processed
    assert metrics["postprocessing/parent_count"] == 4
    datums = [
        datum for call in trainer._service_holder.builtin_calls for datum in call["datums_data"]
    ]
    # Optimizer batching may shuffle Datums, but must retain every token value.
    assert sorted(list(datum["loss_fn_inputs"]["advantages"]) for datum in datums) == [
        [4.0],
        [5.0],
        [6.0],
        [7.0],
    ]


def test_disabled_postprocessor_preserves_estimator_output_in_service_payload(monkeypatch):
    trainer, _ = _trainer(monkeypatch, [], mis=False)
    trainer._algorithm_config.loss.reduction.mode = "sum"
    assert trainer._advantage_postprocessor is None
    parents = _parents()

    run_coro(trainer.train(parents))

    expected = [float(parent["advantage"]) for parent in parents for _ in parent.segments]
    datums = trainer._service_holder.builtin_calls[0]["datums_data"]
    assert [datum["loss_fn_inputs"]["advantages"][0] for datum in datums] == pytest.approx(expected)
    assert all("token_advantages" not in sequence for sequence in iter_training_sequences(parents))
