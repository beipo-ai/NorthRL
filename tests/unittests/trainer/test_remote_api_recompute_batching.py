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

"""Bound recompute requests while preserving whole-round training semantics."""

import asyncio
import math
from copy import deepcopy

import pytest

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.utils.async_utils import run_coro
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyDataDumper,
    _make_config,
)
from tests.unittests.trainer.test_remote_api_trainer_mis import (
    _Holder,
    _recomputed,
    _Trainer,
    _trajectory,
)


class _BatchHolder(_Holder):
    def __init__(self, results):
        super().__init__(results)
        self.failures = {}
        self.forward_count = 0

    async def compute_target_logprobs_batch(self, tokens_batch, sampling_masks=None):
        self.forward_count += 1
        self.events.append(("recompute", deepcopy(tokens_batch), deepcopy(sampling_masks)))
        if self.forward_count in self.failures:
            result = self.failures[self.forward_count]
            if isinstance(result, Exception):
                raise result
            return result
        return [deepcopy(self.results[tokens[0] - 10]) for tokens in tokens_batch]


def _make_trainer(monkeypatch, results, *, mini_batch_size=2, batch_size=None, **mis):
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda datums: None)
    config = _make_config(batch_size=batch_size, mini_batch_size=mini_batch_size)
    config.algorithm.sampling_logprob_source = "recomputed"
    config.algorithm.loss.reduction.mode = "token_mean"
    config.algorithm.mis = mis
    trainer = _Trainer(config)
    trainer._need_sampling = False
    trainer.completed_rounds = 0
    holder = _BatchHolder(results)
    trainer.set_service_holder(holder)
    return trainer, holder


def test_recompute_bounds_requests_and_preserves_nested_sequence_masks(monkeypatch):
    results = [[-index - 0.1, -index - 0.2] for index in range(5)]
    trainer, holder = _make_trainer(monkeypatch, results)
    sequences = [_trajectory(index) for index in range(5)]
    for index, sequence in enumerate(sequences):
        sequence.loss_mask = [0, 0, 1]
        sequence["sampling_mask"] = [[30, 100 + index]]
    segments = [
        {
            "tokens": sequence.tokens,
            "loss_mask": sequence.loss_mask,
            **sequence.extra_fields,
        }
        for sequence in sequences[1:4]
    ]
    trajectories = [
        sequences[0],
        Trajectory(segments=segments, reward=1.0, extra_fields={"advantage": 1.0}),
        sequences[4],
    ]

    returned = run_coro(trainer._recompute_logprobs(trajectories, {}))

    assert returned is trajectories
    assert [len(event[1]) for event in holder.events] == [2, 2, 1]
    assert [tokens for event in holder.events for tokens in event[1]] == [
        sequence.tokens for sequence in sequences
    ]
    assert [mask for event in holder.events for mask in event[2]] == [
        [[20], [30, 100 + index]] for index in range(5)
    ]
    flattened = [trajectories[0], *segments, trajectories[2]]
    assert [sequence["old_log_probs"] for sequence in flattened] == [
        [0.0, *values] for values in results
    ]


@pytest.mark.parametrize("mini_batch_size", [None, 5, 10])
def test_recompute_unset_or_large_mini_batch_sends_one_request(monkeypatch, mini_batch_size):
    results = [[-1.0, -2.0]] * 5
    trainer, holder = _make_trainer(monkeypatch, results, mini_batch_size=mini_batch_size)
    trajectories = [_trajectory(index) for index in range(5)]

    run_coro(trainer._recompute_logprobs(trajectories, {}))

    assert holder.events == [
        ("recompute", [trajectory.tokens for trajectory in trajectories], None)
    ]
    assert [trajectory["old_log_probs"] for trajectory in trajectories] == [[0.0, -1.0, -2.0]] * 5


def test_recompute_empty_batch_does_not_send_request(monkeypatch):
    trainer, holder = _make_trainer(monkeypatch, [])
    metrics = {}

    assert run_coro(trainer._recompute_logprobs([], metrics)) == []
    assert holder.events == []
    assert metrics["timing/recompute_logprobs"] == 0.0


def test_diagnostic_forward_batches_without_changing_training_denominators(monkeypatch):
    ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    trainer, holder = _make_trainer(monkeypatch, _recomputed(ratios), get_mismatch_metrics=True)
    trainer._sampling_logprob_source = "inference"
    trajectories = [_trajectory(index) for index in range(5)]
    before = deepcopy(trajectories)
    metrics = {}

    run_coro(trainer._compute_inference_mismatch_metrics(trajectories, metrics))

    assert [len(event[1]) for event in holder.events] == [2, 2, 1]
    assert [tokens for event in holder.events for tokens in event[1]] == [
        trajectory.tokens for trajectory in trajectories
    ]
    assert trajectories == before
    assert metrics["mis/metrics_available"] == 1
    assert metrics["mis/mismatch_tokens"] == 10
    assert metrics["mis/kl"] == pytest.approx(-sum(map(math.log, ratios)) / len(ratios))


def test_all_recompute_chunks_precede_updates_and_keep_global_mis_normalization(monkeypatch):
    results = _recomputed([1.0, 2.0, 3.0, 4.0, 5.0])
    trainer, holder = _make_trainer(
        monkeypatch,
        results,
        batch_size=3,
        enabled=True,
        tis_upper_bound=10.0,
        tis_batch_normalize=True,
    )
    trajectories = [_trajectory(index, advantage=-2.0) for index in range(5)]

    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == [
        "recompute",
        "recompute",
        "recompute",
        "forward_backward",
        "forward_backward",
        "update",
        "forward_backward",
        "forward_backward",
        "update",
    ]
    assert [len(event[1]) for event in holder.events[:3]] == [2, 2, 1]
    assert [len(call["datums_data"]) for call in holder.calls] == [2, 1, 2, 1]
    assert metrics["algorithm/mis/batch_norm_factor"] == pytest.approx(3.0)
    assert metrics["algorithm/mis/optimizer_steps"] == 2
    datums = [datum for call in holder.calls for datum in call["datums_data"]]
    # The final optimizer batch is padded; recompute must never pad its requests.
    assert [datum["input_tokens"][0] for datum in datums] == [10, 11, 12, 13, 14, 13]
    for datum in datums:
        index = datum["input_tokens"][0] - 10
        assert datum["loss_fn_inputs"]["logprobs"] == results[index]
        # Full-round MIS mean = 3; optimizer-batch token denominator = 6.
        assert datum["loss_fn_inputs"]["advantages"] == pytest.approx([-(index + 1) / 9.0] * 2)
    assert all(trajectory["rollout_log_probs"] == [0.0, -4.0, -4.0] for trajectory in trajectories)


@pytest.mark.parametrize("submit_ahead", [0, 1])
@pytest.mark.parametrize("failure", [RuntimeError("later forward failed"), [[-1.0, -2.0]]])
def test_later_chunk_failure_retries_without_partial_logprob_replacement(
    monkeypatch, caplog, failure, submit_ahead
):
    trainer, holder = _make_trainer(monkeypatch, [[-1.0, -2.0]] * 4, enabled=True)
    trainer._submit_ahead = submit_ahead
    holder.failures = {call: failure for call in (2, 3, 4, 5)}
    trajectories = [_trajectory(index) for index in range(4)]
    published = []
    monkeypatch.setattr(trainer, "_after_recompute_logprobs", lambda *args: published.append(args))

    with pytest.raises((RuntimeError, ValueError)):
        # Keep caller-side warnings synchronous instead of using SDK log dispatch.
        asyncio.run(trainer.train(trajectories))

    assert [event[1][0][0] for event in holder.events] == [10, 12, 12, 12, 12]
    assert all(event[0] == "recompute" for event in holder.events)
    assert holder.calls == []
    assert all(trajectory["old_log_probs"] == [0.0, -4.0, -4.0] for trajectory in trajectories)
    assert published == []
    assert trainer.completed_rounds == 0
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 3
    for attempt, message in enumerate(warnings, start=1):
        assert "chunk_start=2" in message
        assert "chunk_size=2" in message
        assert f"attempt={attempt}/4" in message


@pytest.mark.parametrize("submit_ahead", [0, 1])
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("temporary forward failure"),
        [[-1.0], [-2.0]],  # Correct count, but invalid target alignment.
        [[float("nan"), -1.0], [-2.0, -3.0]],  # MIS rejects non-finite effective targets.
    ],
    ids=["rpc", "alignment", "mis-nonfinite"],
)
def test_later_chunk_transient_failure_retries_only_failed_chunk(
    monkeypatch, failure, submit_ahead
):
    results = [[-index - 0.1, -index - 0.2] for index in range(5)]
    trainer, holder = _make_trainer(monkeypatch, results, enabled=True)
    trainer._submit_ahead = submit_ahead
    holder.failures = {2: failure}
    trajectories = [_trajectory(index) for index in range(5)]

    run_coro(trainer._recompute_logprobs(trajectories, {}))

    assert [event[1][0][0] for event in holder.events] == [10, 12, 12, 14]
    assert [trajectory["old_log_probs"] for trajectory in trajectories] == [
        [0.0, *values] for values in results
    ]
