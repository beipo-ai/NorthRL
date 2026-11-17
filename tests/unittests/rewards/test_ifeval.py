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
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from northrl.rewards import REWARD_FNS, evaluate_reward
from northrl.rewards.ifeval import _answer_text, _instruction_registry, ifeval_reward


def _task(*instructions, prompt_text="Follow the instructions."):
    return {
        "metadata": {
            "instructions": list(instructions),
            "prompt_text": prompt_text,
        }
    }


def _score(task, response):
    return ifeval_reward(task, {"response": response})


def test_all_constraints_must_pass():
    task = _task(
        {"id": "keywords:word_once", "kwargs": {"keyword": "chemistry"}},
        {"id": "punctuation:no_comma", "kwargs": {}},
    )

    assert _score(task, "Chemistry is useful.") == {
        "reward": 1.0,
        "correct": True,
        "reason": "pass",
        "metrics": {
            "ifeval_constraints_total": 2.0,
            "ifeval_constraints_failed": 0.0,
            "ifeval_constraints_unknown": 0.0,
        },
    }
    assert _score(task, "Chemistry is useful, especially here.")["reward"] == 0.0
    assert _score(task, "No target word appears.")["metrics"]["ifeval_constraints_failed"] == 1.0


def test_final_answer_cleanup_excludes_reasoning_and_eos():
    task = _task({"id": "punctuation:no_comma", "kwargs": {}})

    assert _score(task, "reasoning, has a comma</think>Final answer.<|im_end|>\n")["reward"] == 1.0
    assert (
        _score(task, "reasoning, has a comma###Response Final answer.<|eot_id|>")["reward"] == 1.0
    )
    assert _answer_text("No marker.<|endoftext|>") == "No marker."


def test_no_instructions_returns_zero_without_loading_registry(monkeypatch):
    monkeypatch.setattr(
        "northrl.rewards.ifeval._instruction_registry",
        lambda: pytest.fail("registry should not be loaded"),
    )

    assert _score({"metadata": {"instructions": []}}, "anything") == {
        "reward": 0.0,
        "correct": False,
        "reason": "no_instructions",
        "metrics": {
            "ifeval_constraints_total": 0.0,
            "ifeval_constraints_failed": 0.0,
            "ifeval_constraints_unknown": 0.0,
        },
    }


def test_unknown_id_and_missing_required_kwargs_return_zero(caplog):
    with caplog.at_level(logging.WARNING):
        unknown = _score(_task({"id": "unknown:checker", "kwargs": {}}), "anything")
        missing = _score(
            _task({"id": "keywords:word_once", "kwargs": {}}),
            "chemistry",
        )

    assert unknown["reason"] == "unknown_verifier"
    assert unknown["metrics"]["ifeval_constraints_unknown"] == 1.0
    assert missing["reason"] == "fail_ifeval"
    assert missing["metrics"]["ifeval_constraints_failed"] == 1.0
    assert "Unknown IFEval verifier" in caplog.text
    assert "missing required kwargs" in caplog.text


def test_checker_index_error_is_scored_as_failed_constraint():
    task = _task(
        {
            "id": "last_word:last_word_answer",
            "kwargs": {"last_word": "suggestion"},
        }
    )

    assert _score(task, "") == {
        "reward": 0.0,
        "correct": False,
        "reason": "fail_ifeval",
        "metrics": {
            "ifeval_constraints_total": 1.0,
            "ifeval_constraints_failed": 1.0,
            "ifeval_constraints_unknown": 0.0,
        },
    }


@pytest.mark.parametrize(
    "instruction_id,kwargs,answer",
    [
        ("language:response_language", {"language": "en"}, ""),
        ("change_case:english_capital", {}, "HELLO"),
        ("change_case:english_lowercase", {}, "hello"),
    ],
)
def test_language_detection_failure_logs_response_and_check_context(
    monkeypatch, caplog, instruction_id, kwargs, answer
):
    from northrl.rewards.ifevalg import instructions

    def fail_detection(_value):
        raise instructions.langdetect.LangDetectException(0, "No features in text.")

    monkeypatch.setattr(instructions.langdetect, "detect", fail_detection)
    response = f"<think>Reasoning text.</think> {answer}<|im_end|>\n"
    task = _task({"id": instruction_id, "kwargs": kwargs})
    task.update(task_id="language-task", group_id="language-group", label="not-for-logging")

    score = ifeval_reward(task, {"response": response, "finish_reason": "stop"})

    # The vendored rules accept language detection failures; diagnostics must
    # preserve that behavior and the original text before answer cleanup.
    assert score["reward"] == 1.0
    records = [record for record in caplog.records if "Unable to detect language" in record.msg]
    assert len(records) == 1
    assert records[0].args[1:] == (
        instruction_id,
        "language-task",
        "language-group",
        "stop",
        response,
        answer,
        answer,
    )
    assert "not-for-logging" not in caplog.text


def test_language_failure_context_is_isolated_between_workers_and_reset(monkeypatch, caplog):
    from northrl.rewards.ifevalg import instructions

    _instruction_registry()
    barrier = Barrier(2)

    def fail_detection(value):
        if value != "outside":
            barrier.wait(timeout=10)
        raise instructions.langdetect.LangDetectException(0, "No features in text.")

    monkeypatch.setattr(instructions.langdetect, "detect", fail_detection)

    def score_task(task_id):
        task = _task({"id": "language:response_language", "kwargs": {"language": "en"}})
        task.update(task_id=task_id, group_id=f"group-{task_id}")
        response = f"<think>Reasoning {task_id}.</think>{task_id}<|im_end|>"
        return ifeval_reward(task, {"response": response, "finish_reason": "stop"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        scores = list(executor.map(score_task, ("first", "second")))

    assert all(score["reward"] == 1.0 for score in scores)
    records = [record for record in caplog.records if "Unable to detect language" in record.msg]
    assert {record.args[2] for record in records} == {"first", "second"}
    for record in records:
        task_id = record.args[2]
        assert record.args[3:] == (
            f"group-{task_id}",
            "stop",
            f"<think>Reasoning {task_id}.</think>{task_id}<|im_end|>",
            task_id,
            task_id,
        )

    # A bare checker invocation must not inherit a completed reward's context.
    checker = instructions.ResponseLanguageChecker("language:response_language")
    checker.configure({"language": "en"})
    assert checker.check_following("outside") is True
    assert caplog.records[-1].args[1:] == (None, None, None, None, None, None, "outside")


def test_concurrent_first_language_checks_wait_for_loaded_profiles(monkeypatch, caplog):
    from langdetect import detector_factory

    load_started = Event()
    release_load = Event()
    second_started = Event()
    original_load = detector_factory.DetectorFactory.load_profile

    def delayed_load(factory, directory):
        load_started.set()
        assert release_load.wait(5), "Language profile initialization was never released"
        original_load(factory, directory)

    monkeypatch.setattr(detector_factory, "_factory", None)
    monkeypatch.setattr(detector_factory.DetectorFactory, "load_profile", delayed_load)
    _instruction_registry.cache_clear()

    def score_language(*, second=False):
        if second:
            second_started.set()
        task = _task({"id": "language:response_language", "kwargs": {"language": "fr"}})
        return _score(task, "This answer is entirely written in English.")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(score_language)
            try:
                assert load_started.wait(2)
                second = executor.submit(score_language, second=True)
                assert second_started.wait(2)
                # An incomplete factory raises LangDetectException; the vendored
                # checker accepts that exception and would award incorrect credit.
                with pytest.raises(TimeoutError):
                    second.result(timeout=0.1)
            finally:
                release_load.set()
            assert first.result(timeout=5)["reward"] == 0.0
            assert second.result(timeout=5)["reward"] == 0.0
        assert "Unable to detect language" not in caplog.text
    finally:
        _instruction_registry.cache_clear()


@pytest.mark.parametrize(
    ("task", "result", "error"),
    [
        ({}, {"response": "answer"}, KeyError),
        ({"metadata": None}, {"response": "answer"}, TypeError),
        ({"metadata": {"instructions": {}}}, {"response": "answer"}, TypeError),
        (_task("not a mapping"), {"response": "answer"}, TypeError),
        (_task({"id": ""}), {"response": "answer"}, ValueError),
        (_task(), {}, KeyError),
        (_task(), {"response": None}, TypeError),
    ],
)
def test_malformed_fundamental_data_fails_fast(task, result, error):
    with pytest.raises(error):
        ifeval_reward(task, result)


def test_full_registry_and_local_dependencies_are_available():
    registry = _instruction_registry()

    assert len(registry) == 54
    assert all(isinstance(name, str) and name for name in registry)
    assert "language:response_language" in registry
    assert "length_constraints:number_sentences" in registry


def test_reward_is_registered_and_normalized_by_dispatcher():
    assert REWARD_FNS["ifeval"] is ifeval_reward
    task = {
        "reward_fn": "ifeval",
        "metadata": {"instructions": [{"id": "punctuation:no_comma", "kwargs": {}}]},
    }

    result = asyncio.run(evaluate_reward(task, {"response": "A valid answer."}))
    assert result["reward"] == 1.0
    assert result["correct"] is True
    assert result["reason"] == "pass"


# Fixed test samples cover the dataset contracts without reading recipe data.
_IFEVAL_CASES = (
    (
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "Mention chemistry exactly once and use no commas.",
                }
            ],
            "label": None,
            "reward_fn": "ifeval",
            "metadata": {
                "id": "test_if_multi_word_once",
                "data_source": "if_multi",
                "prompt_text": "Mention chemistry exactly once and use no commas.",
                "instructions": [
                    {"id": "keywords:word_once", "kwargs": {"keyword": "chemistry"}},
                    {"id": "punctuation:no_comma", "kwargs": {}},
                ],
            },
        },
        "Chemistry is useful.",
        "Chemistry helps us understand chemistry.",
    ),
    (
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "Write exactly three bullet points and include chemistry.",
                }
            ],
            "label": None,
            "reward_fn": "ifeval",
            "metadata": {
                "id": "test_if_swiss_bullets",
                "data_source": "if_swiss",
                "prompt_text": "Write exactly three bullet points and include chemistry.",
                "instructions": [
                    {"id": "detectable_format:number_bullet_lists", "kwargs": {"num_bullets": 3}},
                    {"id": "keywords:existence", "kwargs": {"keywords": ["chemistry"]}},
                ],
            },
        },
        "* Chemistry explains reactions.\n* Plants need water.\n* Animals need food.",
        "* Chemistry explains reactions.\n* Plants need water.",
    ),
    (
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "End with a postscript starting P.S. and use no commas.",
                }
            ],
            "label": None,
            "reward_fn": "ifeval",
            "metadata": {
                "id": "test_if_swiss_postscript",
                "data_source": "if_swiss",
                "prompt_text": "End with a postscript starting P.S. and use no commas.",
                "instructions": [
                    {
                        "id": "detectable_content:postscript",
                        "kwargs": {"postscript_marker": "P.S."},
                    },
                    {"id": "punctuation:no_comma", "kwargs": {}},
                ],
            },
        },
        "Meet me at noon.\nP.S. Bring a book.",
        "Meet me at noon.\nBring a book.",
    ),
)


def test_fixed_samples_follow_standard_ifeval_contract():
    samples = [sample for sample, _, _ in _IFEVAL_CASES]

    assert len(samples) == 3
    assert {sample["metadata"]["data_source"] for sample in samples} == {
        "if_multi",
        "if_swiss",
    }
    assert len({sample["metadata"]["id"] for sample in samples}) == 3
    assert all(isinstance(sample["prompt"], list) for sample in samples)
    assert all(sample["label"] is None for sample in samples)
    assert all(sample["reward_fn"] == "ifeval" for sample in samples)


@pytest.mark.parametrize(
    "sample,valid_response,invalid_response",
    _IFEVAL_CASES,
    ids=["if_multi_word_once", "if_swiss_bullets", "if_swiss_postscript"],
)
@pytest.mark.parametrize("accepted", [True, False], ids=["valid", "one_rule_broken"])
def test_fixed_samples_score_with_real_checkers(sample, valid_response, invalid_response, accepted):
    response = valid_response if accepted else invalid_response

    score = asyncio.run(evaluate_reward(sample, {"response": response}))

    assert score["reward"] == float(accepted)
    assert score["correct"] is accepted
    assert score["reason"] == ("pass" if accepted else "fail_ifeval")
    assert score["metrics"] == {
        "ifeval_constraints_total": 2.0,
        "ifeval_constraints_failed": float(not accepted),
        "ifeval_constraints_unknown": 0.0,
    }
