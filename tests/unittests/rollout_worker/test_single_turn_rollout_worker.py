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

import asyncio
import copy
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import e2b
import httpx
import pytest
from e2b import CommandResult
from omegaconf import OmegaConf

from northrl.data_loader import TorchDataLoader
from northrl.remote_rollout import metrics as rollout_metrics_module
from northrl.rewards import REWARD_FNS, code_e2b
from northrl.rewards._runtime_context import reward_runtime_context, reward_stop_requested
from northrl.rollout_worker import base_rollout_worker as base_worker_module
from northrl.rollout_worker import single_turn_rollout_worker as single_turn_module
from northrl.rollout_worker.single_turn_rollout_worker import SingleTurnRolloutWorker
from northrl.utils.async_utils import run_coro
from northrl.utils.step_metrics import log_timing_metrics
from northrl.utils.verifier_metrics import log_verifier_metrics


class _FakeLlmClient:
    def __init__(self, token_count: int, prompt_str: str, finish_reason: str = "stop"):
        self.token_count = token_count
        self.prompt_str = prompt_str
        self.finish_reason = finish_reason
        self.template_tokenize_calls: list[bool] = []
        self.generate_called = False
        self.result = {
            "id": "generation-1",
            "choices": [
                {
                    "message": {"content": "unfinished reasoning \\boxed{42}"},
                    "finish_reason": self.finish_reason,
                }
            ],
            "northrl_train": {
                "prompt_tokens": [1, 2, 3],
                "response_tokens": [4, 5],
                "response_logprobs": [-0.1, -0.2],
                "response_old_logprobs": [-0.3, -0.4],
            },
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "sampling_mask": [[4], [5]],
            "moe_topk_indices_ref": "routing-shard-1",
            "generation_metadata": {"request_id": "request-1"},
        }

    async def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=False,
    ):
        del messages, tools, add_generation_prompt
        self.template_tokenize_calls.append(tokenize)
        return list(range(self.token_count)) if tokenize else self.prompt_str

    async def generate(self, messages):
        del messages
        self.generate_called = True
        return self.result


def _worker(
    client,
    max_prompt_length: int = 4,
    reward_fn: str = "math_answer_only",
    max_sequence_length: int = 16,
) -> SingleTurnRolloutWorker:
    config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": max_prompt_length,
                "max_sequence_length": max_sequence_length,
            },
            "rollout": {
                "type": "single_turn",
                "params": {},
                "reward_fn": reward_fn,
                "sampling": {"temperature": 0.6},
            },
        }
    )
    worker = SingleTurnRolloutWorker(config)
    worker._inference_client = client
    worker._save_rollout_trace = Mock()
    worker._trajectory_pool = Mock()
    worker._trajectory_pool.put_trajectory.return_value = "success"
    worker._validator = Mock()
    worker._validator.put_trajectory.return_value = "success"
    return worker


def _task():
    return {
        "prompt": [{"role": "user", "content": "What is 40 + 2?"}],
        "reward_model": {"ground_truth": "42"},
        "is_val": False,
    }


class _Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.wall_time = 1000.0

    def advance(self, seconds):
        self.elapsed += seconds
        self.wall_time += seconds

    def time(self):
        return self.wall_time

    def monotonic(self):
        return self.elapsed

    def perf_counter(self):
        return self.elapsed


@pytest.fixture
def worker_clock(monkeypatch):
    clock = _Clock()
    # Replace module references, keeping the event loop's real clock intact.
    for module in (single_turn_module, base_worker_module, rollout_metrics_module):
        monkeypatch.setattr(module, "time", clock)
    return clock


@pytest.mark.parametrize("is_val", [False, True], ids=["training", "validation"])
def test_rollout_and_verifier_timings_include_distinct_execution_boundaries(
    monkeypatch, worker_clock, is_val
):
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    client.result["runtime_metrics"] = {"provider_metric": 6, "verifier_skipped_length": True}
    client.result["reward_fn"] = "provider-field"
    original_result = copy.deepcopy(client.result)
    original_template = client.apply_chat_template
    original_generate = client.generate

    async def timed_template(*args, **kwargs):
        worker_clock.advance(1)
        return await original_template(*args, **kwargs)

    async def timed_generate(*args, **kwargs):
        worker_clock.advance(7)
        return await original_generate(*args, **kwargs)

    async def timed_reward(task, result):
        assert task["reward_fn"] is None
        assert result["runtime_metrics"] == {
            "provider_metric": 6,
            "verifier_skipped_length": True,
        }
        try:
            worker_clock.advance(2)  # Includes the verifier's own retry work.
            return {"reward": 0.25, "correct": True, "metrics": {"passed_cases": 1}}
        finally:
            worker_clock.advance(3)  # Cleanup completes before the verifier returns.
            worker_clock.wall_time -= 200  # Wall-clock adjustments must not change durations.

    client.apply_chat_template = timed_template
    client.generate = timed_generate
    monkeypatch.setitem(REWARD_FNS, "timed_reward", timed_reward)
    worker = _worker(client, reward_fn="timed_reward")
    worker._save_rollout_trace.side_effect = lambda *args: worker_clock.advance(2)
    task = {
        **_task(),
        "is_val": is_val,
        "reward_fn": None,
        "metadata": {"dataset": "mix"},
        "runtime_metrics": {"task_metric": 8, "verifier_skipped_length": True},
        "group_id": "group-1",
        "rollout_id": "rollout-1",
        "run_id": 0,
        "expected_rollouts": 1,
    }
    original_task = copy.deepcopy(task)
    if not is_val:
        original_rollout = worker.rollout

        async def rollout_with_final_cleanup(received_task):
            result = await original_rollout(received_task)
            worker_clock.advance(3)
            return result

        worker.rollout = rollout_with_final_cleanup

    operation = worker.rollout if is_val else worker.execute_rollout
    assert run_coro(operation(task)) == "success"

    destination = worker._validator if is_val else worker._trajectory_pool
    trajectory = destination.put_trajectory.call_args.args[0]
    expected_total = 16.0 if is_val else 19.0
    runtime = trajectory["runtime_metrics"]
    assert runtime["started_at"] == 1000.0
    assert runtime["finished_at"] == 800.0 + expected_total
    assert runtime["duration_seconds"] == expected_total
    assert runtime["verifier_duration_seconds"] == 5.0
    assert "verifier_skipped_length" not in runtime
    assert runtime["provider_metric"] == 6
    assert runtime["task_metric"] == 8
    if not is_val:
        assert runtime["model_requests"] == 1
        assert runtime["request_stats"]["duration_seconds"]["sum"] == 7.0
    assert trajectory["reward_fn"] == "timed_reward"
    assert trajectory["metadata"] == task["metadata"]
    assert trajectory["generation_metadata"] == client.result["generation_metadata"]
    assert trajectory["score"] == {
        "reward_score": 0.25,
        "format_correct": True,
        "correct": True,
        "metrics": {"passed_cases": 1.0},
    }
    assert task == original_task
    assert client.result == original_result
    generated = worker._save_rollout_trace.call_args.args[2]["rollout_result"]
    assert generated["reward_fn"] == "provider-field"


@pytest.mark.parametrize("is_val", [False, True], ids=["training", "validation"])
def test_length_finish_times_rollout_without_a_verifier_duration(monkeypatch, worker_clock, is_val):
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt", finish_reason="length")
    client.result["runtime_metrics"] = {"provider_metric": 6, "verifier_duration_seconds": 99}
    original_result = copy.deepcopy(client.result)
    original_generate = client.generate

    async def timed_generate(*args, **kwargs):
        worker_clock.advance(7)
        return await original_generate(*args, **kwargs)

    client.generate = timed_generate
    verifier = AsyncMock(side_effect=AssertionError("Length-finished output must not be judged"))
    monkeypatch.setitem(REWARD_FNS, "timed_reward", verifier)
    worker = _worker(client, reward_fn="timed_reward")
    task = {
        **_task(),
        "is_val": is_val,
        "reward_fn": None,
        "runtime_metrics": {"verifier_duration_seconds": 88},
    }

    operation = worker.rollout if is_val else worker.execute_rollout
    assert run_coro(operation(task)) == "success"

    destination = worker._validator if is_val else worker._trajectory_pool
    trajectory = destination.put_trajectory.call_args.args[0]
    runtime = trajectory["runtime_metrics"]
    assert runtime["started_at"] == 1000.0
    assert runtime["finished_at"] == 1007.0
    assert runtime["duration_seconds"] == 7.0
    assert runtime["verifier_skipped_length"] is True
    assert "verifier_duration_seconds" not in runtime
    assert runtime["provider_metric"] == 6
    assert trajectory["reward_fn"] == "timed_reward"
    assert trajectory.reward == 0.0
    assert trajectory["score"]["reason"] == "finish_reason_length"
    generated = worker._save_rollout_trace.call_args.args[2]["rollout_result"]
    assert generated["runtime_metrics"]["verifier_skipped_length"] is True
    assert "verifier_duration_seconds" not in generated["runtime_metrics"]
    verifier.assert_not_called()
    assert client.result == original_result


def test_token_length_guard_and_length_finish_zero_reward():
    client = _FakeLlmClient(token_count=3, prompt_str="x" * 100, finish_reason="length")

    worker = _worker(client)

    assert run_coro(worker.rollout(_task())) == "success"

    trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert trajectory.reward == 0.0
    assert trajectory["score"]["reason"] == "finish_reason_length"
    assert trajectory["score"]["correct"] is False
    assert client.template_tokenize_calls == [True, False]
    assert client.generate_called


@pytest.mark.parametrize("is_val", [False, True], ids=["training", "validation"])
@pytest.mark.parametrize("response", [None, "", " \n", "print(input())"])
def test_length_finished_code_keeps_zero_reward_trajectory_without_judging(
    monkeypatch, is_val, response
):
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt", finish_reason="length")
    client.result["choices"][0]["message"]["content"] = response
    verifier = AsyncMock(side_effect=AssertionError("Length-finished code must not be judged"))
    monkeypatch.setitem(REWARD_FNS, "code_e2b", verifier)
    worker = _worker(client)
    task = {**_task(), "reward_fn": "code_e2b", "is_val": is_val}

    assert run_coro(worker.rollout(task)) == "success"

    destination = worker._validator if is_val else worker._trajectory_pool
    other = worker._trajectory_pool if is_val else worker._validator
    other.put_trajectory.assert_not_called()
    trajectory = destination.put_trajectory.call_args.args[0]
    assert trajectory.reward == 0.0
    assert trajectory["score"] == {
        "reward_score": 0.0,
        "format_correct": False,
        "correct": False,
        "reason": "finish_reason_length",
    }
    assert trajectory["finish_reason"] == "length"
    assert trajectory.tokens == [1, 2, 3, 4, 5]
    assert trajectory.loss_mask == [0, 0, 0, 1, 1]
    assert trajectory["old_log_probs"] == [0.0, 0.0, 0.0, -0.3, -0.4]
    assert client.result["choices"][0]["message"]["content"] == response
    verifier.assert_not_called()


def test_non_length_completion_without_content_is_still_skipped():
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt", finish_reason="stop")
    client.result["choices"][0]["message"]["content"] = None
    worker = _worker(client)

    assert run_coro(worker.rollout(_task())) is None
    worker._trajectory_pool.put_trajectory.assert_not_called()
    worker._validator.put_trajectory.assert_not_called()


def test_token_length_guard_rejects_token_long_character_short_prompt():
    client = _FakeLlmClient(token_count=5, prompt_str="x")

    worker = _worker(client)

    assert run_coro(worker.rollout(_task())) is None
    worker._trajectory_pool.put_trajectory.assert_not_called()
    assert client.template_tokenize_calls == [True]
    assert not client.generate_called


@pytest.mark.parametrize("reference_fields", [{}, {"label": None}], ids=["missing", "null"])
def test_single_turn_worker_supports_rewards_without_reference(monkeypatch, reference_fields):
    task = {
        "prompt": [{"role": "user", "content": "Reply with a short greeting."}],
        "metadata": {"instruction": "short greeting"},
        **reference_fields,
    }
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    client.result["choices"][0]["message"]["content"] = "Hello!"

    async def instruction_reward(received_task, generated):
        assert received_task is task
        assert received_task["metadata"] == {"instruction": "short greeting"}
        assert generated["response"] == "Hello!"
        return {"reward": 1.0, "correct": True}

    monkeypatch.setitem(REWARD_FNS, "instruction_reward", instruction_reward)
    worker = _worker(client, reward_fn="instruction_reward")

    assert run_coro(worker.rollout(task)) == "success"

    trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert trajectory.reward == 1.0
    assert trajectory["ground_truth"] == ""
    assert trajectory["metadata"] == task["metadata"]
    assert trajectory["score"]["correct"] is True


def test_sample_reward_fn_overrides_config_and_preserves_result_details(monkeypatch):
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    task = {**_task(), "reward_fn": "sample_reward"}

    def sample_reward(received_task, generated):
        assert received_task is task
        assert generated["response"] == "unfinished reasoning \\boxed{42}"
        assert generated["finish_reason"] == "stop"
        assert generated["prompt_str"] == "rendered prompt"
        for key, value in client.result.items():
            assert generated[key] == value
        return {
            "reward": 0.25,
            "correct": True,
            "metrics": {"partial_credit": 0.25},
            "reason": "sample-specific judge",
            "details": {"matched": "42"},
        }

    def configured_reward(task, result):
        pytest.fail("Configured reward must not run when the sample selects its own function")

    monkeypatch.setitem(REWARD_FNS, "sample_reward", sample_reward)
    monkeypatch.setitem(REWARD_FNS, "configured_reward", configured_reward)
    worker = _worker(client, reward_fn="configured_reward")

    assert run_coro(worker.rollout(task)) == "success"

    trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
    expected_score = {
        "reward_score": 0.25,
        "format_correct": True,
        "correct": True,
        "metrics": {"partial_credit": 0.25},
        "reason": "sample-specific judge",
        "details": {"matched": "42"},
    }
    assert trajectory.reward == 0.25
    assert trajectory["score"] == expected_score
    assert trajectory["finish_reason"] == "stop"
    assert trajectory["reward_fn"] == "sample_reward"
    assert trajectory["generation_metadata"] == client.result["generation_metadata"]
    generated = worker._save_rollout_trace.call_args.args[2]["rollout_result"]
    assert generated["score"] == expected_score
    assert generated["response"] == "unfinished reasoning \\boxed{42}"
    for key, value in client.result.items():
        assert generated[key] == value


@pytest.mark.parametrize("sample_fields", [{}, {"reward_fn": None}], ids=["missing", "null"])
def test_missing_sample_reward_fn_uses_config(monkeypatch, sample_fields):
    calls = []

    def configured_reward(task, result):
        calls.append((task, result["response"]))
        return 0.5

    monkeypatch.setitem(REWARD_FNS, "configured_reward", configured_reward)
    task = {**_task(), **sample_fields}
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")

    worker = _worker(client, reward_fn="configured_reward")

    assert run_coro(worker.rollout(task)) == "success"
    trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert trajectory.reward == 0.5
    assert trajectory["reward_fn"] == "configured_reward"
    assert calls == [(task, "unfinished reasoning \\boxed{42}")]


@pytest.mark.parametrize("name", ["unregistered_reward", ""], ids=["unknown", "empty"])
def test_invalid_sample_reward_fn_does_not_fall_back(monkeypatch, name):
    def configured_reward(task, result):
        pytest.fail("An explicit invalid reward name must not fall back to configuration")

    monkeypatch.setitem(REWARD_FNS, "configured_reward", configured_reward)
    task = {**_task(), "reward_fn": name}
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")

    worker = _worker(client, reward_fn="configured_reward")

    with pytest.raises(KeyError) as raised:
        run_coro(worker.rollout(task))

    assert raised.value.args == (name,)


def test_jsonl_label_and_reward_fn_reach_worker_and_trajectory(tmp_path):
    records = [
        {"prompt": _task()["prompt"], "label": "42", "reward_fn": "math_with_format"},
        {"prompt": _task()["prompt"], "label": "42"},
    ]
    data_path = tmp_path / "mixed-rewards.jsonl"
    data_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    loader = TorchDataLoader(
        OmegaConf.create(
            {
                "files": [str(data_path)],
                "batch_size": 2,
                "rollout_repeat_n": 1,
                "prompt_key": "prompt",
                "filter_prompts": False,
                "shuffle": False,
                "drop_last": False,
                "max_prompt_length": 4,
            }
        )
    )
    client = _FakeLlmClient(token_count=3, prompt_str="<think>")
    worker = _worker(client, reward_fn="math_answer_only")
    tasks = [loader.get_next_item(), loader.get_next_item()]
    assert tasks[0]["reward_fn"] == "math_with_format"
    assert "reward_fn" not in tasks[1]

    for task in tasks:
        assert run_coro(worker.rollout(task)) == "success"

    trajectories = [call.args[0] for call in worker._trajectory_pool.put_trajectory.call_args_list]
    # Format scoring rejects unfinished thinking; answer-only scoring uses the boxed answer.
    assert [trajectory.reward for trajectory in trajectories] == [0.0, 1.0]
    for task, trajectory in zip(tasks, trajectories):
        assert trajectory["ground_truth"] == task["label"]
        assert str(trajectory["ground_truth"]) == "42"
        assert trajectory.tokens == [1, 2, 3, 4, 5]
        assert trajectory.loss_mask == [0, 0, 0, 1, 1]
        assert trajectory["old_log_probs"] == [0.0, 0.0, 0.0, -0.3, -0.4]
        assert trajectory["sampling_mask"] == [[4], [5]]
        assert trajectory["moe_topk_indices_ref"] == "routing-shard-1"
        assert trajectory["finish_reason"] == "stop"
        assert trajectory["score"]["reward_score"] == trajectory.reward
    assert loader.get_next_item() is None


@pytest.mark.parametrize("group_id", [float("nan"), float("inf")], ids=["nan", "infinite"])
def test_validation_nonfinite_group_identity_becomes_one_complete_task(group_id):
    task = {**_task(), "is_val": True, "group_id": group_id, "run_id": 7}
    worker = _worker(_FakeLlmClient(token_count=3, prompt_str="rendered prompt"))

    assert run_coro(worker.rollout(task)) == "success"

    trajectory = worker._validator.put_trajectory.call_args.args[0]
    assert isinstance(trajectory["group_id"], str) and trajectory["group_id"]
    assert trajectory["expected_rollouts"] == 1
    assert trajectory["run_id"] == 0
    assert task["group_id"] is group_id
    assert task["run_id"] == 7
    metrics = {}
    log_timing_metrics([trajectory], metrics, prefix="val/")
    assert metrics["val/group/duration_seconds_observed_count"] == 1


@pytest.mark.parametrize("validation_repeat_n", [1, 2], ids=["single", "repeated"])
@pytest.mark.parametrize(
    "group_fields", [{}, {"group_id": None}, {"group_id": ""}], ids=["missing", "null", "empty"]
)
def test_validation_loader_samples_keep_distinct_complete_timing_groups(
    tmp_path, worker_clock, validation_repeat_n, group_fields
):
    records = [
        {
            "prompt": [{"role": "user", "content": question}],
            "label": "42",
            "sample_id": sample_id,
            "run_id": 7,
            **group_fields,
        }
        for question, sample_id in (("What is 40 + 2?", "first"), ("What is 21 * 2?", "second"))
    ]
    data_path = tmp_path / "validation.jsonl"
    data_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    loader = TorchDataLoader(
        OmegaConf.create(
            {
                "files": [str(data_path)],
                "batch_size": 2,
                "validation_repeat_n": validation_repeat_n,
                "prompt_key": "prompt",
                "filter_prompts": False,
                "shuffle": False,
                "drop_last": False,
                "max_prompt_length": 4,
            }
        ),
        is_validate=True,
    )
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    original_generate = client.generate

    async def timed_generate(*args, **kwargs):
        worker_clock.advance(2)
        return await original_generate(*args, **kwargs)

    client.generate = timed_generate
    worker = _worker(client)
    tasks = [{**loader.get_next_item(), "is_val": True} for _ in range(2 * validation_repeat_n)]
    original_tasks = copy.deepcopy(tasks)
    assert loader.get_next_item() is None

    for task in tasks:
        assert run_coro(worker.rollout(task)) == "success"

    trajectories = [call.args[0] for call in worker._validator.put_trajectory.call_args_list]
    assert tasks == original_tasks
    assert len({trajectory["group_id"] for trajectory in trajectories}) == 2
    for task, trajectory in zip(tasks, trajectories, strict=True):
        assert trajectory["group_id"] not in (None, "")
        assert trajectory["expected_rollouts"] == validation_repeat_n
        assert trajectory["sample_id"] == task["sample_id"]
        if validation_repeat_n == 1:
            # Eager DataFrame readers can materialize JSON null as NaN.
            raw_group = task.get("group_id")
            assert raw_group in (None, "") or (
                isinstance(raw_group, float) and raw_group != raw_group
            )
            assert task["run_id"] == 7
            assert trajectory["run_id"] == 0
        else:
            assert trajectory["group_id"] == task["group_id"]
            assert trajectory["run_id"] == task["run_id"]
            assert trajectory["rollout_id"] == task["rollout_id"]
    metrics = {}
    log_timing_metrics(trajectories, metrics, prefix="val/")
    assert metrics["val/group/duration_seconds_observed_count"] == 2
    assert metrics["val/group/duration_seconds_mean"] == 2.0 * validation_repeat_n


@pytest.mark.parametrize("referenced_mask", [False, True], ids=["inline", "reference"])
@pytest.mark.parametrize("is_val", [False, True], ids=["training", "validation"])
def test_truncated_rollout_preserves_alignment_and_routes_by_task(referenced_mask, is_val):
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    if referenced_mask:
        client.result["sampling_mask"] = {
            "sampling_mask_ref": {"uri": "memory://mask-shard", "token_count": 2}
        }
    worker = _worker(client, max_sequence_length=4)
    task = {**_task(), "is_val": is_val, "group_id": "group-1", "task_id": 7}

    assert run_coro(worker.rollout(task)) == "success"

    destination = worker._validator if is_val else worker._trajectory_pool
    other = worker._trajectory_pool if is_val else worker._validator
    other.put_trajectory.assert_not_called()
    destination.put_trajectory.assert_called_once()
    trajectory = destination.put_trajectory.call_args.args[0]
    assert trajectory.tokens == [1, 2, 3, 4]
    assert trajectory.loss_mask == [0, 0, 0, 1]
    assert trajectory["old_log_probs"] == [0.0, 0.0, 0.0, -0.3]
    assert trajectory.reward == 0.0
    assert trajectory.is_val is is_val
    assert trajectory["is_truncated"] is True
    assert trajectory["group_id"] == "group-1"
    assert trajectory["task_id"] == 7
    assert trajectory["moe_topk_indices_ref"] == "routing-shard-1"
    if referenced_mask:
        assert trajectory["sampling_mask"] == {
            "sampling_mask_ref": {"uri": "memory://mask-shard", "token_count": 2},
            "sampling_mask_slice": {"start": 0, "stop": 1},
        }
    else:
        assert trajectory["sampling_mask"] == [[4]]
    trace = worker._save_rollout_trace.call_args
    assert trace.args[0] == "single_turn_rollout"
    assert trace.args[2]["rollout_result"]["reward"] == 0.0
    assert trace.args[2]["sampling_mask"] == trajectory["sampling_mask"]


def test_rollout_falls_back_to_raw_logprobs_without_separate_sampling_logprobs():
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    del client.result["northrl_train"]["response_old_logprobs"]
    worker = _worker(client)

    assert run_coro(worker.rollout(_task())) == "success"

    trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert trajectory["old_log_probs"] == [0.0, 0.0, 0.0, -0.1, -0.2]


@pytest.mark.parametrize("upload_failures", [0, 1, 2])
def test_prepared_code_e2b_label_dispatches_from_sample(monkeypatch, upload_failures):
    task = {
        "prompt": [{"role": "user", "content": "Echo standard input."}],
        "reward_fn": "code_e2b",
        "label": {
            "language": "python",
            "input_mode": "stdio",
            "cases": [
                {"stdin": "a\n", "expected_stdout": "a\n"},
                {"stdin": "b\n", "expected_stdout": "b\n"},
            ],
        },
    }
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt", finish_reason="stop")
    client.generate = AsyncMock(wraps=client.generate)
    client.result["choices"][0]["message"][
        "content"
    ] = "<think>considering the problem</think>\n```python\nprint(input())\n```"
    result = {
        "accepted": False,
        "score": 0.5,
        "tests": [{"passed": True}, {"passed": False}],
    }
    warmup_handle = SimpleNamespace(
        pid=123,
        wait=AsyncMock(return_value=CommandResult(stdout="", stderr="", exit_code=0, error=None)),
        disconnect=AsyncMock(),
    )
    sandbox = SimpleNamespace(
        files=SimpleNamespace(write=AsyncMock()),
        commands=SimpleNamespace(
            run=AsyncMock(
                side_effect=[
                    warmup_handle,
                    CommandResult(
                        stdout=json.dumps(result),
                        stderr="",
                        exit_code=0,
                        error=None,
                    ),
                ]
            )
        ),
        kill=AsyncMock(),
    )
    failed_sandboxes = [
        SimpleNamespace(
            files=SimpleNamespace(write=AsyncMock(side_effect=httpx.ReadTimeout("upload timeout"))),
            commands=SimpleNamespace(run=AsyncMock()),
            kill=AsyncMock(),
        )
        for _ in range(upload_failures)
    ]
    create = AsyncMock(side_effect=[*failed_sandboxes, sandbox])
    backoff = AsyncMock()
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    monkeypatch.setattr(code_e2b.random, "uniform", Mock(return_value=1.5))
    monkeypatch.setattr(code_e2b.asyncio, "sleep", backoff)
    monkeypatch.setenv("E2B_API_KEY", "test-api-key")
    for variable in ("E2B_TEMPLATE", "E2B_DOMAIN", "E2B_COMMAND_TIMEOUT", "E2B_REQUEST_TIMEOUT"):
        monkeypatch.delenv(variable, raising=False)
    worker = _worker(client, reward_fn="math_answer_only")

    assert run_coro(worker.rollout(task)) == "success"

    uploaded = json.loads(sandbox.files.write.await_args.args[1])
    assert uploaded == {**task["label"], "code": "print(input())"}
    client.generate.assert_awaited_once()
    assert create.await_count == upload_failures + 1
    assert backoff.await_count == upload_failures
    for failed in failed_sandboxes:
        assert failed.files.write.await_args.args == sandbox.files.write.await_args.args
        assert (
            failed.files.write.await_args.kwargs["request_timeout"]
            < sandbox.files.write.await_args.kwargs["request_timeout"]
        )
        failed.commands.run.assert_not_awaited()
        failed.kill.assert_awaited_once()
    trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert trajectory.reward == 0.0
    assert trajectory["score"]["reward_score"] == 0.0
    assert trajectory["score"]["reason"] == "failed_tests"
    assert trajectory["ground_truth"] == task["label"]
    assert trajectory["score"]["correct"] is False
    assert trajectory["score"]["sandbox_result"] == result
    assert trajectory["score"]["metrics"]["total_cases"] == 2
    if upload_failures:
        assert trajectory["score"]["metrics"]["upload_failure_count"] == upload_failures
        assert trajectory["score"]["metrics"]["upload_retry_count"] == upload_failures
    warmup_handle.wait.assert_awaited_once()
    warmup_handle.disconnect.assert_awaited_once()
    sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize("is_val", [False, True])
@pytest.mark.parametrize("task_reward_fn", ["missing", None, "test_override"])
def test_verifier_failure_metrics_use_the_effective_reward_fn(monkeypatch, is_val, task_reward_fn):
    async def judge(_task, _result):
        return {"reward": 0.0, "correct": False, "reason": "judge_error"}

    monkeypatch.setitem(REWARD_FNS, "test_default", judge)
    monkeypatch.setitem(REWARD_FNS, "test_override", judge)
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    client.result["reward_fn"] = "provider-field"
    worker = _worker(client, reward_fn="test_default")
    task = {**_task(), "is_val": is_val}
    if task_reward_fn != "missing":
        task["reward_fn"] = task_reward_fn

    assert run_coro(worker.rollout(task)) == "success"

    destination = worker._validator if is_val else worker._trajectory_pool
    parent = destination.put_trajectory.call_args.args[0]
    name = "test_override" if task_reward_fn == "test_override" else "test_default"
    metrics = {}
    log_verifier_metrics([parent], metrics)
    assert parent["reward_fn"] == name
    assert metrics["verifier/service_error_count"] == 1
    assert metrics[f"reward_fn/{name}/verifier/judge_error_count"] == 1
    assert not any("provider-field" in key for key in metrics)


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_reward_runtime_context_tracks_live_stop_and_resets_on_exit(monkeypatch, outcome):
    client = _FakeLlmClient(token_count=3, prompt_str="rendered prompt")
    worker = _worker(client, reward_fn="test_context")
    task = _task()
    original = copy.deepcopy(task)

    async def judge(_task, _result):
        assert not reward_stop_requested()
        worker.request_stop()
        assert reward_stop_requested()  # Not the value at entry, but the live event.
        if outcome == "error":
            raise RuntimeError("judge failed")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return {"reward": 1.0}

    monkeypatch.setitem(REWARD_FNS, "test_context", judge)

    async def run():
        outer = Event()
        outer.set()
        with reward_runtime_context(outer):
            if outcome == "success":
                result = await worker._generate_and_evaluate(task)
                assert result["reward"] == 1.0
            else:
                error_type = RuntimeError if outcome == "error" else asyncio.CancelledError
                with pytest.raises(error_type):
                    await worker._generate_and_evaluate(task)
            assert reward_stop_requested()  # The outer binding is restored.
        assert not reward_stop_requested()

    asyncio.run(run())
    assert task == original
    assert client.generate_called


def test_concurrent_worker_reward_stop_contexts_do_not_leak(monkeypatch):
    stopped_worker = _worker(_FakeLlmClient(3, "prompt"), reward_fn="context_stopped")
    active_worker = _worker(_FakeLlmClient(3, "prompt"), reward_fn="context_active")

    async def run():
        stopped = asyncio.Event()
        checked = asyncio.Event()

        async def stopped_judge(_task, _result):
            stopped_worker.request_stop()
            assert reward_stop_requested()
            stopped.set()
            await checked.wait()
            assert reward_stop_requested()
            return {"reward": 0.0}

        async def active_judge(_task, _result):
            await stopped.wait()
            assert not reward_stop_requested()
            checked.set()
            return {"reward": 1.0}

        monkeypatch.setitem(REWARD_FNS, "context_stopped", stopped_judge)
        monkeypatch.setitem(REWARD_FNS, "context_active", active_judge)
        results = await asyncio.gather(
            stopped_worker._generate_and_evaluate(_task()),
            active_worker._generate_and_evaluate(_task()),
        )
        assert [result["reward"] for result in results] == [0.0, 1.0]
        assert not reward_stop_requested()

    asyncio.run(run())
