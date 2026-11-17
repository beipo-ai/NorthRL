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

"""Check one finite rollout batch using the existing completed-rollout trace."""

import asyncio
import json
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from northrl import debug_rollout


def test_debug_holder_forwards_weaver_scheduling_policy(monkeypatch):
    monkeypatch.setenv("WEAVER_API_KEY", "environment-key")
    config = OmegaConf.create(
        {
            "backend": {
                "type": "weaver",
                "api_key": None,
                "base_url": "https://weaver.example",
                "scheduling_policy": {"mode": "fixed", "clusters": ["cluster-1"]},
            },
            "models": {"actor": {"name": "actor", "tokenizer": "tokenizer"}},
            "runtime": {"debug": {"sampling_model_path": None}},
            "data": {"max_sequence_length": 4096},
        }
    )
    factory = Mock()
    monkeypatch.setattr("northrl.weaver.weaver_service_holder.WeaverServiceHolder", factory)

    assert debug_rollout.create_debug_holder(config) is factory.return_value

    assert factory.call_args.kwargs["sampling_only"] is True
    assert factory.call_args.kwargs["scheduling_policy"] == {
        "mode": "fixed",
        "clusters": ["cluster-1"],
    }
    assert isinstance(factory.call_args.kwargs["scheduling_policy"]["clusters"], list)


@pytest.fixture
def config_path(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": index,
                    "prompt": [{"role": "user", "content": f"Question {index}"}],
                    "ground_truth": "42",
                    "metadata": {"index": index},
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "experiment": {"name": "debug-test", "project": "tests"},
            "models": {"actor": {"name": "test-model"}},
            "data": {
                "train": {
                    "files": [str(dataset)],
                    "batch_size": 2,
                    "rollout_repeat_n": 2,
                    "shuffle": False,
                },
                "max_prompt_length": 8,
                "max_sequence_length": 16,
            },
            "validate": {"enabled": False},
            "algorithm": {"type": "grpo", "optim": {"lr": 1e-4, "batch_size": 2}},
            "rollout": {
                "num_workers": 2,
                "type": "single_turn",
                "reward_fn": "debug_test_reward",
                "parsers": {"reasoning": "think_tag", "tool": None},
            },
            "training": {"total_steps": 100, "save": {"dir": str(tmp_path / "checkpoints")}},
            "backend": {"type": "weaver"},
            "runtime": {},
        }
    )
    path = tmp_path / "config.yaml"
    OmegaConf.save(config, path)
    return path


class FakeSamplingHolder:
    """Stub the backend boundary; keep the real worker, client, parser, and reward."""

    def __init__(self):
        self.requests = []
        self.closed = False
        self.response_tokens = [20, 21, 22]

    async def configure_sampling_flags(self, **kwargs):
        pass

    async def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        return [10, 11] if tokenize else f"<user>{messages[-1]['content']}</user><assistant>"

    async def sample_from_messages(self, **kwargs):
        self.requests.append(kwargs)
        return {
            "prompt_tokens": [10, 11],
            "response_tokens": self.response_tokens,
            "response_logprobs": [-0.25] * len(self.response_tokens),
            "response": "<think>Compute carefully.</think>42",
            "finish_reason": "stop",
        }

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_backend(monkeypatch):
    from northrl.controller import NorthRLController
    from northrl.rewards import REWARD_FNS

    holder = FakeSamplingHolder()
    monkeypatch.setattr(debug_rollout, "create_debug_holder", lambda config: holder)
    monkeypatch.setattr(
        NorthRLController,
        "__init__",
        Mock(side_effect=AssertionError("debug must not start training")),
    )
    reward_calls = []

    def reward(task, result):
        reward_calls.append((task, result))
        assert result["response"] == "42"
        assert result["choices"][0]["message"]["reasoning_content"] == "Compute carefully."
        return {"reward": 1.0, "correct": True, "reason": "answer matches"}

    monkeypatch.setitem(REWARD_FNS, "debug_test_reward", reward)
    return holder, reward_calls


def read_traces(output_dir):
    return [json.loads(path.read_text()) for path in (output_dir / "rollout_traces").glob("*.json")]


@pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async"])
def test_one_batch_dumps_completed_rollouts(config_path, tmp_path, fake_backend, mode):
    holder, reward_calls = fake_backend
    config = debug_rollout.load_config(config_path, [f"training.weight_sync.mode={mode}"])
    original = OmegaConf.to_container(config, resolve=True)
    output_dir = tmp_path / "dump"
    counts = asyncio.run(debug_rollout.run_debug_rollout(config, output_dir))
    assert counts == {"success": 4, "skipped": 0, "failed": 0}
    assert len(holder.requests) == len(reward_calls) == 4
    assert holder.closed
    assert OmegaConf.to_container(config, resolve=True) == original
    traces = read_traces(output_dir)
    assert len(traces) == 4
    assert {(trace["task"]["task_id"], trace["task"]["run_id"]) for trace in traces} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    for trace in traces:
        assert "agent_result" not in trace
        assert trace["rollout_result"]["response"] == "42"
        assert trace["task"]["ground_truth"] == "42"
        assert trace["prompt_str"].endswith("</user><assistant>")
        assert trace["response"] == "42"
        assert trace["reward"] == trace["score"]["reward_score"] == 1.0
        assert trace["score"]["reason"] == "answer matches"
        assert trace["tokens"] == [10, 11, 20, 21, 22]
        assert trace["loss_mask"] == [0, 0, 1, 1, 1]
        assert trace["old_log_probs"] == [0.0, 0.0, -0.25, -0.25, -0.25]
        assert trace["finish_reason"] == "stop"
    assert all(
        path.name.startswith("single_turn_rollout_")
        for path in (output_dir / "rollout_traces").glob("*.json")
    )
    assert {path.name for path in output_dir.iterdir()} == {"rollout_traces"}


def test_reward_failure_saves_no_partial_trace_and_continues(
    config_path, tmp_path, fake_backend, monkeypatch, caplog
):
    from northrl.rewards import REWARD_FNS

    holder, _ = fake_backend

    def reward(task, result):
        if task["task_id"] == 0:
            raise RuntimeError("judge unavailable")
        return 1.0

    monkeypatch.setitem(REWARD_FNS, "debug_test_reward", reward)
    output_dir = tmp_path / "dump"
    counts = asyncio.run(
        debug_rollout.run_debug_rollout(debug_rollout.load_config(config_path), output_dir)
    )
    assert counts == {"success": 2, "skipped": 0, "failed": 2}
    assert len(holder.requests) == 4 and holder.closed
    traces = read_traces(output_dir)
    assert len(traces) == 2 and all(trace["task"]["task_id"] == 1 for trace in traces)
    assert "judge unavailable" in caplog.text
    assert {path.name for path in output_dir.iterdir()} == {"rollout_traces"}


def test_truncation_keeps_score_and_final_reward(config_path, tmp_path, fake_backend):
    holder, _ = fake_backend
    holder.response_tokens = list(range(20, 40))
    config = debug_rollout.load_config(
        config_path,
        [
            "data.train.batch_size=1",
            "data.train.rollout_repeat_n=1",
        ],
    )
    counts = asyncio.run(debug_rollout.run_debug_rollout(config, tmp_path))
    assert counts["success"] == 1
    (trace,) = read_traces(tmp_path)
    assert trace["score"]["reward_score"] == 1.0
    assert trace["reward"] == 0.0
    assert len(trace["tokens"]) == 16 and trace["is_truncated"]


def test_skipped_prompt_has_no_trace(config_path, tmp_path, fake_backend):
    holder, reward_calls = fake_backend
    config = debug_rollout.load_config(config_path, ["data.max_prompt_length=1"])
    counts = asyncio.run(debug_rollout.run_debug_rollout(config, tmp_path))
    assert counts == {"success": 0, "skipped": 4, "failed": 0}
    assert holder.requests == reward_calls == []
    assert holder.closed and read_traces(tmp_path) == []


def test_partial_batch_never_cycles(config_path):
    config = debug_rollout.load_config(
        config_path,
        [
            "data.train.batch_size=8",
            "data.train.drop_last=false",
        ],
    )
    tasks = debug_rollout.select_debug_batch(config)
    assert len(tasks) == 12 and {task["task_id"] for task in tasks} == set(range(6))


@pytest.mark.parametrize("override", ["data.train.batch_szie=1", "data.train.batch_size=0"])
def test_invalid_config_is_rejected(config_path, override):
    with pytest.raises(ValidationError):
        debug_rollout.load_config(config_path, [override])


@pytest.mark.parametrize(
    "overrides",
    [
        ["rollout.type=mock", "backend.type=mock"],
        ["models.actor.checkpoint=weaver://training-state"],
        ["training.resume.mode=auto", "training.resume.path=weaver://training-state"],
        ["runtime.debug.sampling_model_path=weaver://sampling-state"],
        ["data.train.batch_size=8"],
    ],
)
def test_unsupported_config_fails_before_backend(config_path, tmp_path, monkeypatch, overrides):
    config = debug_rollout.load_config(config_path, overrides)
    factory = Mock(side_effect=AssertionError("backend must not be initialized"))
    monkeypatch.setattr(debug_rollout, "create_debug_holder", factory)
    with pytest.raises(ValueError):
        asyncio.run(debug_rollout.run_debug_rollout(config, tmp_path))
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("status", "remote_status"),
    [("success", "task_failed"), ("failed", "env_error"), ("skipped", None)],
)
def test_debug_summary_uses_complete_trajectory_status(
    config_path, tmp_path, fake_backend, monkeypatch, status, remote_status
):
    from northrl.northrl_types import Trajectory
    from northrl.rollout_worker.single_turn_rollout_worker import SingleTurnRolloutWorker

    async def rollout(self, task):
        return self._put_trajectory(
            Trajectory(
                segments=[],
                reward=0.0 if status == "success" else None,
                status=status,
                extra_fields={"e2b_status": remote_status},
            )
        )

    monkeypatch.setattr(SingleTurnRolloutWorker, "rollout", rollout)
    counts = asyncio.run(
        debug_rollout.run_debug_rollout(debug_rollout.load_config(config_path), tmp_path)
    )
    assert counts[status] == 4
    assert sum(counts.values()) == 4
