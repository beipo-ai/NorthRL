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

import pytest

from northrl.rewards import REWARD_FNS, evaluate_reward
from northrl.rewards import reasoning_gym as reasoning_gym_module
from northrl.rewards.reasoning_gym import (
    ReasoningGymDependencyError,
    _arc_strict,
    _extract_answer,
    _rush_hour_sim,
    rule_reasoning_gym_reward,
)


def _task(source_dataset="unit", **entry_fields):
    return {
        "label": None,
        "metadata": {
            "answer": entry_fields.pop("answer", None),
            "metadata": {"source_dataset": source_dataset, **entry_fields.pop("inner", {})},
            **entry_fields,
        },
    }


def _score(task, response):
    return rule_reasoning_gym_reward(task, {"response": response})


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"reasoning \boxed{first} then \boxed{last}", "last"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("reasoning <answer> final answer </answer> trailing", "final answer"),
        ("plain full response", "plain full response"),
        (r"broken \boxed{answer", ""),
    ],
)
def test_extract_answer_matches_bp_priority(response, expected):
    assert _extract_answer(response) == expected


def test_boxed_answer_has_priority_over_answer_tag():
    assert _extract_answer(r"<answer>tag</answer> then \boxed{boxed}") == "boxed"


def test_generic_scorer_receives_extracted_answer_and_complete_entry(monkeypatch):
    task = _task("generic", answer="oracle", question="question")
    received = {}

    def factory(name):
        assert name == "generic"

        def scorer(answer, entry):
            received["answer"] = answer
            received["entry"] = entry
            return 1.0

        return scorer

    monkeypatch.setattr(reasoning_gym_module, "_load_score_factory", lambda: factory)
    result = _score(task, r"work \boxed{oracle}")

    assert received == {"answer": "oracle", "entry": task["metadata"]}
    assert result == {
        "reward": 1.0,
        "correct": True,
        "reason": "reasoning_gym/generic/pass",
        "metrics": {"reasoning_gym_raw_score": 1.0},
    }


@pytest.mark.parametrize(
    "source_dataset",
    [
        "ab",
        "binary_matrix",
        "caesar_cipher",
        "circuit_logic",
        "futoshiki",
        "game_of_life",
        "jugs",
        "kakurasu",
        "knights_knaves",
        "sokoban",
        "string_manipulation",
        "sudoku",
        "tower_of_hanoi",
    ],
)
def test_all_real_generic_subtasks_resolve_in_pinned_library(source_dataset):
    score_function = reasoning_gym_module._load_score_factory()(source_dataset)
    assert callable(score_function)


@pytest.mark.parametrize(
    ("raw_score", "expected_reward", "expected_correct"),
    [
        (1.0, 1.0, True),
        (0.999, 1.0, True),
        (0.998, 0.0, False),
        (0.5, 0.0, False),
        (0.0, 0.0, False),
    ],
)
def test_non_boxnet_scores_are_strictly_binarized(
    monkeypatch, raw_score, expected_reward, expected_correct
):
    monkeypatch.setattr(
        reasoning_gym_module,
        "_load_score_factory",
        lambda: lambda _name: lambda _answer, _entry: raw_score,
    )

    result = _score(_task("generic"), "answer")

    assert result["reward"] == expected_reward
    assert result["correct"] is expected_correct
    assert result["metrics"] == {"reasoning_gym_raw_score": float(raw_score)}


def test_boxnet_preserves_continuous_reward(monkeypatch):
    monkeypatch.setattr(
        reasoning_gym_module,
        "_load_score_factory",
        lambda: lambda _name: lambda _answer, _entry: 0.625,
    )

    result = _score(_task("boxnet"), "partial solution")

    assert result == {
        "reward": 0.625,
        "correct": False,
        "reason": "reasoning_gym/boxnet/fail",
        "metrics": {"reasoning_gym_raw_score": 0.625},
    }


@pytest.mark.parametrize("raw_score", [None, "1", {}, float("nan"), float("inf")])
def test_invalid_scorer_values_warn_and_become_zero(monkeypatch, caplog, raw_score):
    monkeypatch.setattr(
        reasoning_gym_module,
        "_load_score_factory",
        lambda: lambda _name: lambda _answer, _entry: raw_score,
    )

    result = _score(_task("generic"), "answer")

    assert result["reward"] == 0.0
    assert result["correct"] is False
    assert result["metrics"] == {"reasoning_gym_raw_score": 0.0}
    assert "assigning reward 0" in caplog.text


def test_scorer_failure_warns_and_becomes_zero(monkeypatch, caplog):
    def failing_factory(_name):
        raise RuntimeError("broken scorer")

    monkeypatch.setattr(reasoning_gym_module, "_load_score_factory", lambda: failing_factory)
    result = _score(_task("generic"), "answer")

    assert result == {
        "reward": 0.0,
        "correct": False,
        "reason": "reasoning_gym/generic/scorer_error",
        "metrics": {"reasoning_gym_raw_score": 0.0},
    }
    assert "broken scorer" in caplog.text


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("1 2\n3 4", 1.0),
        ("1 2\n3 5", 0.0),
        ("1 2 3\n4", 0.0),
        ("1 two\n3 4", 0.0),
        ("", 0.0),
    ],
)
def test_arc_strict_requires_the_exact_integer_grid(answer, expected):
    entry = {"metadata": {"output": [[1, 2], [3, 4]]}}
    assert _arc_strict(answer, entry) == expected


@pytest.mark.parametrize("source_dataset", ["arc_agi", "rearc"])
def test_arc_tasks_bypass_the_library_factory(monkeypatch, source_dataset):
    monkeypatch.setattr(
        reasoning_gym_module,
        "_load_score_factory",
        lambda: pytest.fail("ARC must use the bp strict-grid fix"),
    )
    task = _task(source_dataset, inner={"output": [[1, 2], [3, 4]]})

    assert _score(task, "<answer>1 2\n3 4</answer>")["reward"] == 1.0
    assert _score(task, "<answer>1 2\n4 3</answer>")["reward"] == 0.0


def test_simple_rush_hour_solution_and_failures():
    empty_row = "oooooo"
    entry = {
        "metadata": {
            "board_config": empty_row * 2 + "oAAooo" + empty_row * 3,
        }
    }

    assert _rush_hour_sim("A+3", entry) == 1.0
    assert _rush_hour_sim("A+2", entry) == 0.0
    assert _rush_hour_sim("Z+1 A+3", entry) == 0.0
    assert _rush_hour_sim("A+4", entry) == 0.0


def test_rush_hour_collision_is_rejected():
    entry = {
        "metadata": {
            "board_config": "oooooo" * 2 + "oAABBo" + "oooooo" * 3,
        }
    }
    assert _rush_hour_sim("A+3", entry) == 0.0


@pytest.mark.parametrize(
    ("task", "result", "error"),
    [
        ({}, {"response": "answer"}, KeyError),
        ({"metadata": None}, {"response": "answer"}, TypeError),
        ({"metadata": {}}, {"response": "answer"}, TypeError),
        ({"metadata": {"metadata": {}}}, {"response": "answer"}, ValueError),
        (
            {"metadata": {"metadata": {"source_dataset": ""}}},
            {"response": "answer"},
            ValueError,
        ),
        (_task("unit"), {}, KeyError),
        (_task("unit"), {"response": None}, TypeError),
    ],
)
def test_invalid_contract_fails_fast(task, result, error):
    with pytest.raises(error):
        rule_reasoning_gym_reward(task, result)


def test_missing_dependency_fails_instead_of_assigning_zero(monkeypatch):
    reasoning_gym_module._load_score_factory.cache_clear()

    def missing_distribution(_name):
        raise reasoning_gym_module.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(reasoning_gym_module.importlib_metadata, "version", missing_distribution)
    with pytest.raises(ReasoningGymDependencyError, match="reasoning-gym==0.1.25"):
        _score(_task("generic"), "answer")
    reasoning_gym_module._load_score_factory.cache_clear()


def test_wrong_dependency_version_fails_clearly(monkeypatch):
    reasoning_gym_module._load_score_factory.cache_clear()
    monkeypatch.setattr(reasoning_gym_module.importlib_metadata, "version", lambda _name: "9.9.9")
    with pytest.raises(ReasoningGymDependencyError, match="found 9.9.9"):
        _score(_task("generic"), "answer")
    reasoning_gym_module._load_score_factory.cache_clear()


def test_reward_is_registered_and_normalized_by_dispatcher(monkeypatch):
    assert REWARD_FNS["rule_reasoning_gym"] is rule_reasoning_gym_reward
    monkeypatch.setattr(
        reasoning_gym_module,
        "_load_score_factory",
        lambda: lambda _name: lambda _answer, _entry: 1.0,
    )
    task = {**_task("generic"), "reward_fn": "rule_reasoning_gym"}

    assert asyncio.run(evaluate_reward(task, {"response": "answer"}))["reward"] == 1.0


def test_three_real_dataset_samples_score_correct_and_incorrect_answers():
    binary_answer = (
        "4 4 3 2 1 2 1 0 1 1\n3 3 2 1 0 1 1 0 0 0\n"
        "2 2 1 0 0 1 0 1 0 0\n1 1 2 1 1 1 0 0 1 0\n"
        "0 0 1 2 2 1 0 1 0 1\n1 1 0 1 2 1 1 2 1 0\n"
        "0 1 0 0 1 0 1 1 0 1\n0 1 1 1 1 0 1 0 0 1\n"
        "1 0 1 0 1 1 0 1 0 1\n1 0 1 1 2 2 1 2 1 2"
    )
    arc_answer = "4 4 6 1 9 4\n9 9 8 8 9 4\n1 8 8 8 8 6\n" "6 8 8 8 8 1\n4 9 8 8 9 9\n4 9 1 6 4 4"
    by_source = {
        "binary_matrix": {
            "reward_fn": "rule_reasoning_gym",
            "label": binary_answer,
            "metadata": {
                "answer": binary_answer,
                "metadata": {"source_dataset": "binary_matrix"},
            },
        },
        "arc_agi": {
            "reward_fn": "rule_reasoning_gym",
            "label": arc_answer,
            "metadata": {
                "answer": arc_answer,
                "metadata": {
                    "source_dataset": "arc_agi",
                    "output": [
                        [4, 4, 6, 1, 9, 4],
                        [9, 9, 8, 8, 9, 4],
                        [1, 8, 8, 8, 8, 6],
                        [6, 8, 8, 8, 8, 1],
                        [4, 9, 8, 8, 9, 9],
                        [4, 9, 1, 6, 4, 4],
                    ],
                },
            },
        },
        "rush_hour": {
            "reward_fn": "rule_reasoning_gym",
            "label": None,
            "metadata": {
                "answer": None,
                "metadata": {
                    "source_dataset": "rush_hour",
                    "board_config": "oxIJCCooIJDDoAAJKLoxFFKLoooooooGGHHo",
                },
            },
        },
    }

    for source in ("binary_matrix", "arc_agi"):
        sample = by_source[source]
        correct = f"<answer>{sample['metadata']['answer']}</answer>"
        assert asyncio.run(evaluate_reward(sample, {"response": correct}))["reward"] == 1.0
        assert (
            asyncio.run(evaluate_reward(sample, {"response": "<answer>wrong</answer>"}))["reward"]
            == 0.0
        )

    rush_hour = by_source["rush_hour"]
    correct_moves = (
        "A-1 G-1 H-1 I+1 K+2 F+1 I+1 L+2 F+1 J+2 D-4 I-1 "
        "J-2 F-2 K-1 H+1 K-2 F+1 I+2 A+1 L-1 H+1 L-2 F+1 "
        "J+1 C-2 J+2 A+1 K-1 A+1 L-1 A+1"
    )
    assert rush_hour["label"] is None
    assert asyncio.run(evaluate_reward(rush_hour, {"response": correct_moves}))["reward"] == 1.0
    assert asyncio.run(evaluate_reward(rush_hour, {"response": "A+1"}))["reward"] == 0.0
