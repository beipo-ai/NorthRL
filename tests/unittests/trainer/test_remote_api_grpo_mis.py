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

"""MIS integration with real GRPO preparation and router-replay hooks."""

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import MISConfig
from northrl.weaver.weaver_service_holder import ForwardLogprobsResult

_RECORD_REF = {
    "schema": "weaver.router_replay.r2_manifest_handle.v1",
    "index_set_uri": "weaver://actor/router-replay/record-1",
    "manifest_uri": "weaver://actor/router-replay/record-1/manifest.json",
}


class _Holder:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.events = []
        self.calls = []

    def _forward(self, tokens_batch, sampling_masks, metadata):
        self.events.append(("recompute", deepcopy(tokens_batch), deepcopy(metadata)))
        assert sampling_masks is None
        if self.fail:
            raise RuntimeError("recompute unavailable")
        return [[-4.0 + math.log(4.0 if tokens[0] == 10 else 1.0)] * 2 for tokens in tokens_batch]

    async def compute_target_logprobs_batch(
        self, tokens_batch, sampling_masks=None, datum_metadata_batch=None, *, loss_fn_config=None
    ):
        assert loss_fn_config == {"temperature": 1.0}
        return self._forward(tokens_batch, sampling_masks, datum_metadata_batch)

    async def compute_target_logprobs_batch_with_record(
        self, tokens_batch, sampling_masks=None, *, loss_fn_config=None
    ):
        assert loss_fn_config == {"temperature": 1.0}
        return ForwardLogprobsResult(
            logprobs=self._forward(tokens_batch, sampling_masks, None),
            router_replay_record_ref=deepcopy(_RECORD_REF),
        )

    async def forward_backward(self, **kwargs):
        self.events.append(("forward_backward", len(kwargs["datums_data"])))
        self.calls.append(deepcopy(kwargs))
        return {"loss": float(len(self.calls)), "metrics": {"loss": float(len(self.calls))}}

    async def optim_step(self, **kwargs):
        self.events.append(("update", None))
        return {"status": "success", "metrics": {}}


def _trainer(monkeypatch, *, mode=None, fail=False):
    monkeypatch.setattr(
        trainer_module,
        "get_data_dumper",
        lambda *args, **kwargs: SimpleNamespace(enabled=False, should_dump=lambda *args: False),
    )
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda datums: None)
    config = OmegaConf.create(
        {
            "training": {"total_steps": 4, "save": {"dir": "", "every_n_steps": 0}},
            "algorithm": {
                "type": "grpo",
                "entropy_coeff": 0.0,
                "sampling_logprob_source": "recomputed",
                "mis": {"enabled": True, "tis_mode": "mask"},
                "router_replay": {"enabled": mode is not None, "mode": mode or "R3"},
                "kl": {"mode": None},
                "optim": {
                    "lr": 2e-6,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "weight_decay": 0.01,
                    "eps": 1e-8,
                    "grad_clip_norm": 1.0,
                    "batch_size": 1,
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
            "models": {"reference": None},
            "runtime": {"debug": {"enable_data_dump": False}},
        }
    )
    trainer = RemoteApiGrpoTrainer(config)
    trainer._need_sampling = False
    holder = _Holder(fail=fail)
    trainer.set_service_holder(holder)
    return trainer, holder


def _trajectory(run_id):
    return Trajectory(
        tokens=[10 + run_id, 20, 30],
        loss_mask=[0, 1, 1],
        reward=1.0,
        is_val=False,
        extra_fields={
            "group_id": "constant-reward",
            "run_id": run_id,
            "old_log_probs": [0.0, -4.0, -4.0],
            "moe_topk_indices_ref": {
                "relative_path": f"rollout/sequence-{run_id}.safetensors",
                "sample_index": 20 + run_id,
                "future_server_field": {"version": 2},
            },
        },
    )


@pytest.mark.parametrize("mode", ["R2", "R3"])
def test_grpo_mis_reuses_replay_recompute_and_keeps_zero_advantage_targets(monkeypatch, mode):
    trainer, holder = _trainer(monkeypatch, mode=mode)
    # GRPO sorts asynchronous arrivals before recording the routing sample indices.
    trajectories = [_trajectory(run_id) for run_id in (2, 0, 1)]
    ordered = sorted(trajectories, key=lambda trajectory: trajectory["run_id"])

    metrics, path, version = run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == [
        "recompute",
        "forward_backward",
        "update",
        "forward_backward",
        "update",
    ]
    assert holder.events[0][1] == [trajectory.tokens for trajectory in ordered]
    assert metrics["batch/segment_count"] == 3
    assert metrics["batch/trainable_segment_count"] == 2  # One segment was rejected by MIS.
    # Rejected sequences are removed before optimizer batches are formed.
    assert metrics["algorithm/mis/skipped_batches"] == 0
    assert metrics["algorithm/mis/optimizer_steps"] == 2
    assert trainer._last_datum_sources == [(1, 0), (2, 0)]
    assert (path, version) == (None, None)
    assert all(float(trajectory["advantage"]) == 0.0 for trajectory in trajectories)
    assert all(trajectory.loss_mask == [0, 1, 1] for trajectory in trajectories)
    assert all(trajectory["rollout_log_probs"] == [0.0, -4.0, -4.0] for trajectory in trajectories)

    for sample_index, call in enumerate(holder.calls, start=1):
        datum = call["datums_data"][0]
        assert datum["input_tokens"][0] == 10 + sample_index
        assert datum["loss_fn_inputs"]["advantages"] == [0.0, 0.0]
        assert datum["loss_fn_inputs"]["loss_mask"] == [1.0, 1.0]
        assert datum["loss_fn_inputs"]["logprobs"] == [-4.0, -4.0]
        replay = datum["metadata"]["router_replay"]
        assert replay["mode"] == mode
        assert replay["action"] == "REPLAY"
        if mode == "R2":
            # Filtering the first sequence must not renumber recorded trajectories.
            assert replay["value_ref"] == {**_RECORD_REF, "sample_index": sample_index}
        else:
            assert replay["value_ref"] == ordered[sample_index]["moe_topk_indices_ref"]
            assert datum["metadata"] == holder.events[0][2][sample_index]


def test_grpo_mis_all_rejected_skips_training_before_batching(monkeypatch):
    trainer, holder = _trainer(monkeypatch)
    trajectories = [_trajectory(run_id) for run_id in range(2)]
    for trajectory in trajectories:
        trajectory.tokens[0] = 10  # Both actor/rollout ratios trigger MIS rejection.

    metrics, path, version = run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == ["recompute"]
    assert holder.calls == []
    assert trainer._last_datum_sources == []
    assert metrics["training/skipped_flag"] == 1
    assert metrics["batch/segment_count"] == 2
    assert metrics["batch/trainable_segment_count"] == 0
    assert metrics["batch/trainable_response_tokens_total"] == 0
    assert (path, version) == (None, None)


def test_grpo_mis_recompute_failure_is_fatal_without_router_replay(monkeypatch):
    trainer, holder = _trainer(monkeypatch, fail=True)
    trajectories = [_trajectory(run_id) for run_id in range(2)]

    with pytest.raises(RuntimeError, match="recompute unavailable"):
        run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == ["recompute"] * 4
    assert holder.calls == []
    assert all(trajectory["old_log_probs"] == [0.0, -4.0, -4.0] for trajectory in trajectories)


@pytest.mark.parametrize("mode", ["R2", "R3"])
def test_grpo_mismatch_metrics_only_preserves_all_replay_updates(monkeypatch, mode):
    trainer, holder = _trainer(monkeypatch, mode=mode)
    trainer._mis_config = MISConfig(enabled=False, get_mismatch_metrics=True)
    trajectories = [_trajectory(run_id) for run_id in (2, 0, 1)]
    ordered = sorted(trajectories, key=lambda trajectory: trajectory["run_id"])
    rollout_refs = [deepcopy(trajectory["moe_topk_indices_ref"]) for trajectory in ordered]

    metrics, path, version = run_coro(trainer.train(trajectories))

    # The ratio-4 sample is observed and trained, even though correction mode
    # would reject it. All three constant-reward GRPO samples have zero advantage.
    assert [event[0] for event in holder.events] == [
        "recompute",
        "forward_backward",
        "update",
        "forward_backward",
        "update",
        "forward_backward",
        "update",
    ]
    assert holder.events[0][1] == [trajectory.tokens for trajectory in ordered]
    assert metrics["batch/trainable_segment_count"] == 3
    assert metrics["batch/trainable_response_tokens_total"] == 6
    assert metrics["training/skipped_flag"] == 0
    assert metrics["algorithm/mis/metrics_available"] == 1
    assert metrics["algorithm/mis/mismatch_tokens"] == 6
    assert metrics["algorithm/mis/chi2_token"] == pytest.approx(5.0)
    assert "algorithm/mis/skipped_batches" not in metrics
    assert "algorithm/mis/optimizer_steps" not in metrics
    assert (path, version) == (None, None)
    assert all(float(trajectory["advantage"]) == 0.0 for trajectory in trajectories)
    assert all(trajectory.loss_mask == [0, 1, 1] for trajectory in trajectories)
    assert all("correction_weights" not in trajectory for trajectory in trajectories)
    assert all("correction_mask" not in trajectory for trajectory in trajectories)
    assert all("rollout_log_probs" not in trajectory for trajectory in trajectories)
    assert [trajectory["moe_topk_indices_ref"] for trajectory in ordered] == rollout_refs

    assert len(holder.calls) == 3
    for sample_index, call in enumerate(holder.calls):
        datum = call["datums_data"][0]
        assert datum["input_tokens"][0] == 10 + sample_index
        assert datum["loss_fn_inputs"]["advantages"] == [0.0, 0.0]
        assert datum["loss_fn_inputs"]["loss_mask"] == [1.0, 1.0]
        expected_logprob = -4.0 + math.log(4.0 if sample_index == 0 else 1.0)
        assert datum["loss_fn_inputs"]["logprobs"] == pytest.approx([expected_logprob] * 2)
        replay = datum["metadata"]["router_replay"]
        assert replay["mode"] == mode
        assert replay["action"] == "REPLAY"
        if mode == "R2":
            assert replay["value_ref"] == {**_RECORD_REF, "sample_index": sample_index}
        else:
            assert replay["value_ref"] == rollout_refs[sample_index]
            assert datum["metadata"] == holder.events[0][2][sample_index]


@pytest.mark.parametrize("mode", ["R2", "R3"])
def test_grpo_mismatch_metrics_only_keeps_replay_recompute_failure_fatal(monkeypatch, mode):
    trainer, holder = _trainer(monkeypatch, mode=mode, fail=True)
    trainer._mis_config = MISConfig(enabled=False, get_mismatch_metrics=True)
    trajectories = [_trajectory(run_id) for run_id in range(3)]

    with pytest.raises(RuntimeError, match="recompute unavailable"):
        run_coro(trainer.train(trajectories))

    assert [event[0] for event in holder.events] == ["recompute"] * 4
    assert holder.calls == []
    assert all(trajectory["old_log_probs"] == [0.0, -4.0, -4.0] for trajectory in trajectories)
    assert all("correction_weights" not in trajectory for trajectory in trajectories)
    assert all("correction_mask" not in trajectory for trajectory in trajectories)
