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

"""Exercise correction across recompute, real Datum conversion, and optimizer batches."""

import math
from array import array
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums


class _Trainer(RemoteApiTrainer):
    async def _prepare_trajectories(self, trajectories, metrics):
        return trajectories

    def _convert_to_datums(self, trajectories):
        self._last_datum_sources = []
        return convert_trajectories_to_datums(
            trajectories,
            drop_zero_advantage_datums=False,
            datum_sources=self._last_datum_sources,
        )

    def _on_optimizer_step_succeeded(self, trajectories):
        self.completed_rounds += 1


class _Holder:
    def __init__(self, results):
        self.results = results
        self.events = []
        self.calls = []

    async def compute_target_logprobs_batch(self, tokens_batch, sampling_masks=None):
        self.events.append(("recompute", deepcopy(tokens_batch), sampling_masks))
        if isinstance(self.results, Exception):
            raise self.results
        return deepcopy(self.results)

    async def forward_backward(self, **kwargs):
        self.events.append(("forward_backward", len(kwargs["datums_data"])))
        self.calls.append(deepcopy(kwargs))
        return {"loss": float(len(self.calls)), "metrics": {"loss": float(len(self.calls))}}

    async def optim_step(self, **kwargs):
        self.events.append(("update", None))
        return {"status": "success", "metrics": {}}


def _make_trainer(monkeypatch, results, *, batch_size=None, source="recomputed", **mis):
    monkeypatch.setattr(
        trainer_module,
        "get_data_dumper",
        lambda *args, **kwargs: SimpleNamespace(enabled=False, should_dump=lambda *args: False),
    )
    # Make the retained samples' optimizer batch membership deterministic.
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda datums: None)
    config = OmegaConf.create(
        {
            "training": {"total_steps": 4, "save": {"dir": "", "every_n_steps": 0}},
            "algorithm": {
                "type": "grpo",
                "entropy_coeff": 0.0,
                "sampling_logprob_source": source,
                "mis": {"enabled": True, **mis},
                "optim": {
                    "lr": 2e-6,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "weight_decay": 0.01,
                    "eps": 1e-8,
                    "grad_clip_norm": 1.0,
                    "batch_size": batch_size,
                },
                "loss": {
                    "mode": "importance_sampling",
                    "params": {
                        "clip": {
                            "truncation_c": None,
                            "ratio": None,
                            "ratio_low": None,
                            "ratio_high": None,
                            "ratio_c": None,
                        }
                    },
                    "reduction": {"mode": "token_mean"},
                },
            },
            "runtime": {"debug": {"enable_data_dump": False}},
        }
    )
    trainer = _Trainer(config)
    trainer._need_sampling = False
    trainer.completed_rounds = 0
    holder = _Holder(results)
    trainer.set_service_holder(holder)
    return trainer, holder


def _trajectory(index=0, advantage=1.0):
    return Trajectory(
        tokens=[10 + index, 20, 30],
        loss_mask=[0, 1, 1],
        reward=1.0,
        is_val=False,
        extra_fields={"old_log_probs": [0.0, -4.0, -4.0], "advantage": advantage},
    )


def _recomputed(ratios):
    return [[-4.0 + math.log(ratio)] * 2 for ratio in ratios]


def test_mis_recomputes_once_before_all_updates_and_keeps_denominators(monkeypatch):
    results = _recomputed([1.0, 2.0, 3.0])
    trainer, holder = _make_trainer(
        monkeypatch, results, batch_size=1, tis_upper_bound=10.0, tis_batch_normalize=True
    )
    trajectories = [_trajectory(i, advantage=-2.0) for i in range(3)]
    metrics, sampling_path, version = run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == [
        "recompute",
        "forward_backward",
        "update",
        "forward_backward",
        "update",
        "forward_backward",
        "update",
    ]
    assert len(holder.events[0][1]) == 3
    assert metrics["algorithm/mis/batch_norm_factor"] == pytest.approx(2.0)
    assert metrics["algorithm/mis/optimizer_steps"] == 3
    assert trainer.completed_rounds == 1
    assert (sampling_path, version) == (None, None)
    for i, call in enumerate(holder.calls):
        inputs = call["datums_data"][0]["loss_fn_inputs"]
        assert inputs["logprobs"] == results[i]
        # MIS weight normalization is followed by the optimizer batch's token mean.
        assert inputs["advantages"] == pytest.approx([-(i + 1.0) / 2.0] * 2)
        assert call["loss_fn_config"]["loss_agg_mode"] == "sum"
        assert inputs["loss_mask"] == [1.0, 1.0]
        assert set(inputs) == {"target_tokens", "logprobs", "advantages", "loss_mask"}
        assert trajectories[i]["rollout_log_probs"] == [0.0, -4.0, -4.0]
        assert trajectories[i].loss_mask == [0, 1, 1]


@pytest.mark.parametrize("reduction", ["token_mean", "datum_mean"])
def test_mis_filters_rejected_datums_before_forming_batches(monkeypatch, reduction):
    trainer, holder = _make_trainer(
        monkeypatch, _recomputed([4.0, 1.0, 0.25, 1.5]), batch_size=2, tis_mode="mask"
    )
    trainer._algorithm_config.loss.reduction.mode = reduction
    metrics, _, _ = run_coro(trainer.train([_trajectory(i) for i in range(4)]))
    assert len(holder.calls) == 1
    call = holder.calls[0]
    datums = call["datums_data"]
    assert [datum["input_tokens"][0] for datum in datums] == [11, 13]
    assert trainer._last_datum_sources == [(1, 0), (3, 0)]
    for datum, advantage in zip(datums, [0.25, 0.375], strict=True):
        assert datum["loss_fn_inputs"]["loss_mask"] == [1.0, 1.0]
        # Both retained samples now share one optimizer batch, with four tokens.
        assert datum["loss_fn_inputs"]["advantages"] == pytest.approx([advantage] * 2)
    assert call["loss_fn_config"]["loss_agg_mode"] == "sum"
    assert metrics["algorithm/mis/skipped_batches"] == 0
    assert metrics["algorithm/mis/optimizer_steps"] == 1
    assert metrics["algorithm/mis/token_rejection_fraction"] == 0.5


def test_mis_only_batches_retained_datums(monkeypatch):
    trainer, holder = _make_trainer(
        monkeypatch, _recomputed([4.0, 1.0, 0.25, 1.5]), batch_size=1, tis_mode="mask"
    )
    metrics, _, _ = run_coro(trainer.train([_trajectory(i) for i in range(4)]))
    assert [call["datums_data"][0]["input_tokens"][0] for call in holder.calls] == [11, 13]
    assert trainer._last_datum_sources == [(1, 0), (3, 0)]
    assert all(
        call["datums_data"][0]["loss_fn_inputs"]["loss_mask"] == [1.0, 1.0] for call in holder.calls
    )
    assert metrics["algorithm/mis/skipped_batches"] == 0
    assert metrics["algorithm/mis/optimizer_steps"] == 2
    assert metrics["training/skipped_flag"] == 0


def test_mis_all_rejected_skips_update_and_success_hook(monkeypatch):
    trainer, holder = _make_trainer(monkeypatch, _recomputed([4.0]), tis_mode="mask")
    metrics, path, version = run_coro(trainer.train([_trajectory()]))
    assert holder.calls == []
    assert trainer.completed_rounds == 0
    assert trainer._last_datum_sources == []
    assert metrics["training/optimizer_step_count"] == 0
    assert metrics["training/forward_backward_call_count"] == 0
    assert metrics["training/skipped_flag"] == 1
    assert (path, version) == (None, None)


def test_mis_zero_advantage_keeps_entropy_and_kl_targets(monkeypatch):
    trainer, holder = _make_trainer(monkeypatch, _recomputed([1.0]))
    trajectory = _trajectory(advantage=0.0)
    trajectory["ref_logprobs"] = [0.0, -2.0, -3.0]
    run_coro(trainer.train([trajectory]))
    inputs = holder.calls[0]["datums_data"][0]["loss_fn_inputs"]
    assert inputs["advantages"] == [0.0, 0.0]
    assert inputs["loss_mask"] == [1.0, 1.0]
    assert inputs["ref_logprobs"] == [-2.0, -3.0]


@pytest.mark.parametrize(
    "results",
    [
        RuntimeError("recompute failed"),
        [],
        [[-1.0, -1.0]],
        [[-1.0, -1.0], None],
        [[-1.0, -1.0], [-1.0]],
        [[-1.0, -1.0], [float("nan"), -1.0]],
        [[-1.0, -1.0], [-1.0, float("inf")]],
    ],
)
def test_mis_invalid_recompute_aborts_without_partial_replacement(monkeypatch, results):
    trainer, holder = _make_trainer(monkeypatch, results)
    trajectories = [_trajectory(i) for i in range(2)]
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        run_coro(trainer.train(trajectories))
    assert [event[0] for event in holder.events] == ["recompute"] * 4
    assert holder.calls == []
    assert trainer.completed_rounds == 0
    assert all(t["old_log_probs"] == [0.0, -4.0, -4.0] for t in trajectories)


@pytest.mark.parametrize("invalid", ["missing", "placeholder", "nonfinite"])
def test_mis_rejects_invalid_rollout_before_forward(monkeypatch, invalid):
    trainer, holder = _make_trainer(monkeypatch, _recomputed([1.0]))
    trajectory = _trajectory()
    if invalid == "missing":
        trajectory.extra_fields.pop("old_log_probs")
    elif invalid == "placeholder":
        trajectory["rollout_logprobs_valid"] = False
    else:
        trajectory["old_log_probs"][1] = float("nan")
    with pytest.raises(ValueError):
        run_coro(trainer.train([trajectory]))
    assert holder.events == []


def test_mis_retry_preserves_original_rollout_snapshot(monkeypatch):
    trainer, holder = _make_trainer(monkeypatch, _recomputed([2.0]))
    trajectory = _trajectory()
    run_coro(trainer._recompute_logprobs([trajectory], {}))
    holder.results = _recomputed([3.0])
    run_coro(trainer._recompute_logprobs([trajectory], {}))
    assert trajectory["rollout_log_probs"] == [0.0, -4.0, -4.0]
    assert trajectory["old_log_probs"][1:] == holder.results[0]


def test_mis_disabled_still_aborts_after_failed_training_recompute(monkeypatch):
    trainer, holder = _make_trainer(monkeypatch, RuntimeError("unavailable"), enabled=False)
    trajectory = _trajectory()
    with pytest.raises(RuntimeError):
        run_coro(trainer.train([trajectory]))
    assert [event[0] for event in holder.events] == ["recompute"] * 4
    assert holder.calls == []
    assert trajectory["old_log_probs"] == [0.0, -4.0, -4.0]
    assert "rollout_log_probs" not in trajectory


def test_mis_rejects_filtered_trajectories_before_forward(monkeypatch):
    trainer, holder = _make_trainer(monkeypatch, _recomputed([1.0]))
    trajectory = _trajectory()
    trajectory["sampling_mask"] = [[20], [30]]
    with pytest.raises(ValueError, match="sampling_mask"):
        run_coro(trainer.train([trajectory]))
    assert holder.events == []


def test_mismatch_only_reuses_recompute_and_preserves_training_datums(monkeypatch):
    results = _recomputed([4.0, 0.25])
    baseline, baseline_holder = _make_trainer(monkeypatch, results, batch_size=1, enabled=False)
    run_coro(baseline.train([_trajectory(i) for i in range(2)]))

    trainer, holder = _make_trainer(
        monkeypatch,
        results,
        batch_size=1,
        enabled=False,
        get_mismatch_metrics=True,
        tis_mode="mask",
        use_rs=True,
    )
    trajectories = [_trajectory(i) for i in range(2)]
    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == [
        "recompute",
        "forward_backward",
        "update",
        "forward_backward",
        "update",
    ]
    assert holder.calls == baseline_holder.calls
    assert metrics["algorithm/mis/metrics_available"] == 1
    assert metrics["algorithm/mis/mismatch_tokens"] == 4
    assert metrics["algorithm/mis/k3_kl"] > 0
    assert "algorithm/mis/optimizer_steps" not in metrics
    assert all("correction_weights" not in trajectory for trajectory in trajectories)
    assert all("correction_mask" not in trajectory for trajectory in trajectories)


def test_mismatch_only_inference_keeps_rollout_denominator_and_trajectories(monkeypatch):
    results = _recomputed([4.0])
    baseline, baseline_holder = _make_trainer(
        monkeypatch, results, source="inference", enabled=False
    )
    run_coro(baseline.train([_trajectory(advantage=-2.0)]))

    trainer, holder = _make_trainer(
        monkeypatch,
        results,
        enabled=False,
        get_mismatch_metrics=True,
        source="inference",
    )
    trajectories = [_trajectory(advantage=-2.0)]
    before = deepcopy(trajectories)
    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == ["recompute", "forward_backward", "update"]
    assert holder.calls == baseline_holder.calls
    assert trajectories == before
    assert holder.calls[0]["datums_data"][0]["loss_fn_inputs"]["logprobs"] == [-4.0, -4.0]
    assert metrics["algorithm/mis/metrics_available"] == 1
    assert metrics["algorithm/mis/kl"] == pytest.approx(-math.log(4.0))
    assert "recompute_logprobs/replaced" not in metrics


@pytest.mark.parametrize("source", ["inference", "recomputed"])
@pytest.mark.parametrize("results", [RuntimeError("unavailable"), [], [None], [[-1.0]]])
def test_mismatch_only_failed_forward_preserves_existing_training_behavior(
    monkeypatch, source, results
):
    baseline, baseline_holder = _make_trainer(monkeypatch, results, source=source, enabled=False)

    trainer, holder = _make_trainer(
        monkeypatch,
        results,
        enabled=False,
        get_mismatch_metrics=True,
        source=source,
    )
    if source == "recomputed":
        for current, current_holder in [(baseline, baseline_holder), (trainer, holder)]:
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                run_coro(current.train([_trajectory()]))
            assert [event[0] for event in current_holder.events] == ["recompute"] * 4
            assert current_holder.calls == []
            assert current.completed_rounds == 0
        return

    run_coro(baseline.train([_trajectory()]))
    metrics, _, _ = run_coro(trainer.train([_trajectory()]))

    assert holder.calls == baseline_holder.calls
    assert metrics["algorithm/mis/metrics_available"] == 0
    assert "algorithm/mis/kl" not in metrics
    assert trainer.completed_rounds == baseline.completed_rounds == 1


def test_mismatch_only_invalid_rollout_does_not_break_valid_training_recompute(monkeypatch):
    trainer, holder = _make_trainer(
        monkeypatch,
        _recomputed([2.0]),
        enabled=False,
        get_mismatch_metrics=True,
    )
    trajectory = _trajectory()
    trajectory.extra_fields.pop("old_log_probs")
    metrics, _, _ = run_coro(trainer.train([trajectory]))

    assert [event[0] for event in holder.events] == ["recompute", "forward_backward", "update"]
    assert metrics["algorithm/mis/metrics_available"] == 0
    assert trajectory["old_log_probs"][1:] == _recomputed([2.0])[0]
    assert "algorithm/mis/kl" not in metrics


def test_mismatch_only_partial_recompute_aborts_training(monkeypatch):
    trainer, holder = _make_trainer(
        monkeypatch,
        [[-4.0, -4.0], None],
        enabled=False,
        get_mismatch_metrics=True,
    )
    with pytest.raises((ValueError, TypeError)):
        run_coro(trainer.train([_trajectory(i) for i in range(2)]))

    assert [event[0] for event in holder.events] == ["recompute"] * 4
    assert holder.calls == []
    assert trainer.completed_rounds == 0


def test_mismatch_only_omits_empty_masks_before_training(monkeypatch):
    trainer, holder = _make_trainer(
        monkeypatch,
        _recomputed([1.0]),
        enabled=False,
        get_mismatch_metrics=True,
    )
    trajectory = _trajectory()
    trajectory.loss_mask = [0, 0, 0]
    metrics, _, _ = run_coro(trainer.train([trajectory]))

    assert holder.calls == []
    assert trainer._last_datum_sources == []
    assert metrics["training/skipped_flag"] == 1
    assert metrics["training/optimizer_step_count"] == 0
    assert metrics["mis/metrics_available"] == 0
    assert "algorithm/mis/skipped_batches" not in metrics


def test_mismatch_metrics_report_original_tokens_before_correction_rejection(monkeypatch):
    trainer, holder = _make_trainer(monkeypatch, _recomputed([4.0]), tis_mode="mask")
    metrics, _, _ = run_coro(trainer.train([_trajectory()]))

    assert holder.calls == []
    assert metrics["mis/metrics_available"] == 1
    assert metrics["mis/mismatch_tokens"] == 2
    assert metrics["mis/kept_tokens"] == 0
    assert metrics["mis/kl"] == pytest.approx(-math.log(4.0))


@pytest.mark.parametrize("correction_enabled", [False, True])
def test_mismatch_metric_computation_failure_is_isolated_from_training(
    monkeypatch, correction_enabled
):
    def fail_metrics(**kwargs):
        raise RuntimeError("metric calculation failed")

    monkeypatch.setattr(trainer_module, "compute_mismatch_metrics", fail_metrics)
    trainer, holder = _make_trainer(
        monkeypatch,
        _recomputed([1.5]),
        enabled=correction_enabled,
        get_mismatch_metrics=True,
    )
    metrics, _, _ = run_coro(trainer.train([_trajectory()]))

    assert metrics["algorithm/mis/metrics_available"] == 0
    assert len(holder.calls) == 1
    inputs = holder.calls[0]["datums_data"][0]["loss_fn_inputs"]
    assert inputs["advantages"] == pytest.approx([0.75 if correction_enabled else 0.5] * 2)
    assert inputs["loss_mask"] == [1.0, 1.0]


def test_mis_retry_uses_final_recompute_and_original_rollout_snapshot(monkeypatch):
    final = _recomputed([2.0])
    trainer, holder = _make_trainer(monkeypatch, final)
    holder.compute_target_logprobs_batch = AsyncMock(side_effect=[[[-1.0]]] * 3 + [final])
    trajectory = _trajectory()

    run_coro(trainer.train([trajectory]))

    assert holder.compute_target_logprobs_batch.await_count == 4
    assert trajectory["rollout_log_probs"] == [0.0, -4.0, -4.0]
    assert trajectory["old_log_probs"] == [0.0, *final[0]]
    assert holder.calls[0]["datums_data"][0]["loss_fn_inputs"]["advantages"] == pytest.approx(
        [1, 1]
    )


def test_nested_sequences_keep_parent_outcomes_through_recompute_mis_and_training(monkeypatch):
    results = _recomputed([1.0, 2.0, 0.5])
    trainer, holder = _make_trainer(monkeypatch, results, batch_size=3, tis_upper_bound=10.0)
    plain = _trajectory(0)
    nested_sequences = [
        {"tokens": [11 + i, 20, 30], "loss_mask": [0, 1, 1], "old_log_probs": [0.0, -4.0, -4.0]}
        for i in range(2)
    ]
    nested = Trajectory(
        segments=nested_sequences,
        reward=0.0,
        extra_fields={"advantage": -1.0, "rollout_id": "nested"},
    )
    trajectories = [plain, nested]
    converted = []
    original_convert = trainer._convert_to_datums

    def convert(parents):
        assert parents is trajectories
        converted.extend(parents)
        return original_convert(parents)

    monkeypatch.setattr(trainer, "_convert_to_datums", convert)
    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert converted == trajectories
    assert plain.segments is None
    assert nested.segments is nested_sequences
    assert holder.events[0] == ("recompute", [[10, 20, 30], [11, 20, 30], [12, 20, 30]], None)
    assert [event[0] for event in holder.events] == ["recompute", "forward_backward", "update"]
    for i, sequence in enumerate([plain, *nested_sequences]):
        assert sequence["old_log_probs"] == [0.0, *results[i]]
        assert sequence["rollout_log_probs"] == [0.0, -4.0, -4.0]
        weight = [1.0, 2.0, 0.5][i]
        assert sequence["correction_weights"] == pytest.approx([0.0, weight, weight])
    assert all("reward" not in sequence for sequence in nested_sequences)
    assert nested.reward == 0.0
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["batch/segment_count"] == metrics["batch/trainable_segment_count"] == 3
    assert "datum/input_sequences" not in metrics
    assert metrics["algorithm/trajectory/reward_mean"] == 0.5
    expected = [1.0 / 6, -2.0 / 6, -0.5 / 6]
    assert len(holder.calls) == 1
    for datum, advantage in zip(holder.calls[0]["datums_data"], expected, strict=True):
        assert datum["loss_fn_inputs"]["advantages"] == pytest.approx([advantage] * 2)


@pytest.mark.parametrize("mis_enabled", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.25])
@pytest.mark.parametrize("mask", [0, 1])
def test_compact_recompute_finite_checks_match_legacy(monkeypatch, mis_enabled, value, mask):
    results = [array("d", [value, -0.5])]
    trainer, _ = _make_trainer(monkeypatch, results, enabled=mis_enabled)
    trajectory = _trajectory()
    trajectory["tokens"] = array("q", trajectory["tokens"])
    trajectory["loss_mask"] = array("b", [0, mask, 1])
    if mis_enabled and mask and not math.isfinite(value):
        with pytest.raises(ValueError, match="non-finite"):
            run_coro(trainer._recompute_logprobs([trajectory], {}))
    else:
        run_coro(trainer._recompute_logprobs([trajectory], {}))
        actual = trajectory["old_log_probs"]
        assert isinstance(actual, array) and actual.typecode == "d"
        assert actual.tobytes() == array("d", [0.0, value, -0.5]).tobytes()


def test_compact_recompute_rejects_mask_length_mismatch(monkeypatch):
    trainer, _ = _make_trainer(monkeypatch, [array("d", [-0.25, -0.5])], enabled=True)
    trajectory = _trajectory()
    trajectory["tokens"] = array("q", trajectory["tokens"])
    trajectory["loss_mask"] = array("b", [0, 1])
    with pytest.raises(ValueError, match="token-aligned|length mismatch"):
        run_coro(trainer._recompute_logprobs([trajectory], {}))
