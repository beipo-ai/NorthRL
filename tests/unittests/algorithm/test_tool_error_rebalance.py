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

"""Step-scoped rebalance invariants and conservative diagnostic attribution."""

from array import array
from copy import deepcopy

import numpy as np
import pytest

from northrl.algorithm.advantage import build_postprocessor
from northrl.algorithm.advantage.tool_error_rebalance import (
    ToolErrorRebalancePostprocessor,
    clear_postprocessing_cache,
)
from northrl.northrl_types import Trajectory
from northrl.utils.algorithm_config import PostprocessingConfig
from northrl.utils.trajectory_sequences import iter_trajectory_sequences

PREFIX = "algorithm/advantage_postprocess/"


def _process(trajectories, config, metrics=None):
    mode = None if config is None else config.get("mode")
    # The factory consumes validated config in production. Construct directly
    # here to exercise runtime parameter guards and unsupported-mode rejection.
    component = build_postprocessor(
        PostprocessingConfig.model_construct(
            mode=None if mode == "" else mode,
            params=(
                {}
                if config is None
                else {key: value for key, value in config.items() if key != "mode"}
            ),
        )
    )
    return (
        trajectories
        if component is None
        else component.process(trajectories, {} if metrics is None else metrics)
    )


def _annotation(start=1, end=2, *, reasons=None, **overrides):
    reasons = [] if reasons is None else reasons
    return {
        "version": 1,
        "turn_id": f"turn-{start}",
        "response_span": [start, end],
        "diagnostics": {
            "checked": True,
            "malformed": bool(reasons),
            "detected": 1,
            "invalid": int(bool(reasons)),
            "reasons": reasons,
        },
        "finish_reason": "stop",
        "truncated": False,
        "tools_declared": True,
        "source": "northgate",
        **overrides,
    }


def _trajectory(advantage=1, *, error=False, **extra):
    reasons = ["invalid_arguments_json"] if error else []
    return Trajectory(
        tokens=[10, 11],
        loss_mask=[0, 1],
        reward=1,
        extra_fields={
            "advantage": advantage,
            "turn_annotations": [_annotation(reasons=reasons)],
            **extra,
        },
    )


def _outputs(trajectories):
    return [
        advantage
        for trajectory in trajectories
        for sequence in iter_trajectory_sequences(trajectory)
        for advantage in sequence["token_advantages"]
    ]


def test_one_rebalance_redistributes_advantages_between_trajectories():
    trajectories = [
        _trajectory(1, error=True),
        _trajectory(2),
        _trajectory(-1, error=True),
        _trajectory(-2),
    ]
    for index, trajectory in enumerate(trajectories):
        trajectory["group_id"] = str(index)
        trajectory["model_tag"] = f"model-{index}"
    metrics = {}
    assert _process(trajectories, {"mode": "tool_error_rebalance"}, metrics) is trajectories
    assert _outputs(trajectories) == pytest.approx([0, 3, -2, -1])
    assert metrics[PREFIX + "positive_scale"] == 1.5
    assert metrics[PREFIX + "negative_scale"] == 0.5
    assert (
        metrics[PREFIX + "positive_advantage_sum_before"]
        == metrics[PREFIX + "positive_advantage_sum_after"]
        == 3
    )
    assert (
        metrics[PREFIX + "negative_advantage_abs_sum_before"]
        == metrics[PREFIX + "negative_advantage_abs_sum_after"]
        == 3
    )
    assert [trajectory["advantage"] for trajectory in trajectories] == [1, 2, -1, -2]


def test_token_signs_zero_advantages_unknowns_and_masked_context():
    trajectory = Trajectory(
        tokens=list(range(9)),
        loss_mask=[0, 1, 1, 1, 0, 1, 1, 1, 1],
        extra_fields={
            "token_advantages": [1, -1, 0, 3, -4, 100, -100],
            "turn_annotations": [
                _annotation(1, 4, reasons=["missing_tool_name"]),
                _annotation(5, 7),
                _annotation(7, 9, diagnostics=None),
            ],
        },
    )
    metrics = {}
    _process([trajectory], {"mode": "tool_error_rebalance"}, metrics)
    assert trajectory["token_advantages"] == pytest.approx([0, -2, 0, 4, -3, 100, -100])
    assert metrics[PREFIX + "response_tokens"] == 7
    assert metrics[PREFIX + "diagnostic_coverage"] == pytest.approx(5 / 7)
    for removed in (
        "unknown_tokens",
        "annotation_coverage",
        "annotated_tokens",
        "known_tokens",
        "positive_advantage_sum_change",
        "negative_advantage_abs_sum_change",
        "net_advantage_sum_change",
    ):
        assert PREFIX + removed not in metrics
    assert metrics[PREFIX + "zero_advantage_error_tokens"] == 1
    assert trajectory.loss_mask == [0, 1, 1, 1, 0, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "annotation,error",
    [
        (_annotation(reasons=["incomplete_tool_call"]), True),
        (_annotation(reasons=["incomplete_tool_call"], finish_reason="length"), False),
        (_annotation(reasons=["incomplete_tool_call"], finish_reason=None), False),
        (_annotation(reasons=["incomplete_tool_call"], finish_reason="tool_calls"), True),
        (_annotation(reasons=["incomplete_tool_call"], finish_reason="end_turn"), True),
        (_annotation(reasons=["incomplete_tool_call"], finish_reason="tool_use"), True),
        (_annotation(reasons=["invalid_arguments_json"], truncated=True), False),
        (_annotation(reasons=["invalid_arguments_json"], finish_reason="length"), False),
        (_annotation(reasons=["invalid_arguments_json"], finish_reason="max_tokens"), False),
        (_annotation(reasons=["invalid_arguments_json"], finish_reason="stop_sequence"), False),
        (_annotation(reasons=["undeclared_tool"], tools_declared=False), False),
        (_annotation(reasons=["undeclared_tool"], tools_declared=True), True),
        (_annotation(reasons=["future_reason"]), False),
        (_annotation(diagnostics={"checked": False, "malformed": False, "reasons": []}), False),
        (_annotation(diagnostics=None), False),
        (_annotation(source="other_gateway", reasons=["missing_tool_name"]), False),
        (
            _annotation(
                diagnostics={"checked": False, "malformed": True, "reasons": ["missing_tool_name"]}
            ),
            True,
        ),
        (
            _annotation(
                diagnostics={
                    "checked": False,
                    "malformed": False,
                    "invalid": 1,
                    "reasons": ["missing_tool_name"],
                }
            ),
            True,
        ),
    ],
)
def test_unknown_or_external_failures_retain_original_advantages(annotation, error):
    trajectories = [
        _trajectory(-1, turn_annotations=[annotation]),
        _trajectory(-2),
    ]
    metrics = {}
    _process(trajectories, {"mode": "tool_error_rebalance"}, metrics)
    assert _outputs(trajectories) == ([-2, -1] if error else [-1, -2])
    assert metrics[PREFIX + "error_tokens"] == int(error)


def test_clamping_and_missing_compensation_pools_report_advantage_sums():
    trajectories = [
        _trajectory(4, error=True),
        _trajectory(1),
        _trajectory(-4, error=True),
        _trajectory(-1),
    ]
    metrics = {}
    _process(trajectories, {"mode": "tool_error_rebalance"}, metrics)
    assert _outputs(trajectories) == [0, 2, -8, -0.5]
    assert metrics[PREFIX + "positive_scale_clamped"] == 1
    assert metrics[PREFIX + "negative_scale_clamped"] == 1
    assert metrics[PREFIX + "positive_advantage_sum_before"] == 5
    assert metrics[PREFIX + "positive_advantage_sum_after"] == 2
    assert metrics[PREFIX + "negative_advantage_abs_sum_before"] == 5
    assert metrics[PREFIX + "negative_advantage_abs_sum_after"] == 8.5

    all_error = [_trajectory(1, error=True), _trajectory(-1, error=True)]
    _process(all_error, {"mode": "tool_error_rebalance"}, metrics)
    assert _outputs(all_error) == [0, -2]
    assert metrics[PREFIX + "positive_scale"] == metrics[PREFIX + "negative_scale"] == 1
    assert metrics[PREFIX + "positive_no_clean_tokens"] == 1
    assert metrics[PREFIX + "negative_no_clean_tokens"] == 1
    assert metrics[PREFIX + "positive_scale_clamped"] == 0
    assert metrics[PREFIX + "negative_scale_clamped"] == 0


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"mode": None},
        {"mode": ""},
        {"mode": None, "negative_multiplier": -10},
        {"mode": "", "negative_multiplier": -10},
    ],
)
def test_empty_mode_leaves_trajectories_and_metrics_untouched(config):
    trajectories = [_trajectory(1, error=True), _trajectory(2)]
    before = deepcopy(trajectories)
    metrics = {"existing": 4}
    assert _process(trajectories, config, metrics) is trajectories
    assert trajectories == before and metrics == {"existing": 4}


@pytest.mark.parametrize("mode", ["off", "shadow", "apply", "typo", False, 0])
def test_unknown_modes_rejected_without_mutation(mode):
    trajectories = [_trajectory(1, error=True), _trajectory(2)]
    before = deepcopy(trajectories)
    metrics = {"existing": 4}
    with pytest.raises(ValueError, match="Unknown advantage postprocessor"):
        _process(trajectories, {"mode": mode}, metrics)
    assert trajectories == before and metrics == {"existing": 4}


def test_tool_error_rebalance_replays_from_base():
    trajectories = [_trajectory(1, error=True), _trajectory(2)]
    metrics = {}
    _process(trajectories, {"mode": "tool_error_rebalance"}, metrics)
    assert _outputs(trajectories) == [0, 3]
    assert PREFIX + "applied" not in metrics
    applied = deepcopy(trajectories)
    _process(trajectories, {"mode": "tool_error_rebalance"}, metrics)
    assert trajectories == applied
    assert [trajectory["base_token_advantages"] for trajectory in trajectories] == [[1], [2]]


def test_segment_boundaries_and_excluded_trajectories_cannot_change_rebalance():
    flat = [
        _trajectory(1, error=True),
        _trajectory(2),
        _trajectory(-1, error=True),
        _trajectory(-2),
    ]
    nested = [
        Trajectory(
            segments=[
                {
                    "tokens": item.tokens,
                    "loss_mask": item.loss_mask,
                    "token_advantages": [item["advantage"]],
                    "turn_annotations": item["turn_annotations"],
                }
                for item in flat
            ]
        )
    ]
    skipped = _trajectory(1000, error=True, group_skipped=True)
    failed = _trajectory(-1000, error=True)
    failed.status = "failed"
    excluded_before = deepcopy([skipped, failed])
    _process(flat + [skipped, failed], {"mode": "tool_error_rebalance"})
    _process(nested, {"mode": "tool_error_rebalance"})
    assert _outputs(flat) == _outputs(nested)
    assert [skipped, failed] == excluded_before


@pytest.mark.parametrize(
    "changes",
    [
        {"loss_mask": [0]},
        {"loss_mask": [0, 2]},
        {"loss_mask": [1, 1]},
        {"token_advantages": []},
        {"token_advantages": [float("nan")]},
        {"turn_annotations": [_annotation(1, 3)]},
        {"turn_annotations": [_annotation(1, 1)]},
        {"turn_annotations": [_annotation(0, 2)]},
        {"loss_mask": [0, 0]},
        {"turn_annotations": [_annotation(1, 2), _annotation(0, 2)]},
    ],
)
def test_invalid_inputs_fail_before_any_sequence_is_mutated(changes):
    first, invalid = _trajectory(1, error=True), _trajectory(2)
    for key, value in changes.items():
        invalid[key] = value
    before = deepcopy([first, invalid])
    with pytest.raises(ValueError):
        _process([first, invalid], {"mode": "tool_error_rebalance"})
    assert first == before[0]
    assert "base_token_advantages" not in invalid


def test_no_errors_is_identity_even_with_mixed_signs_and_empty_sequences():
    trajectories = [_trajectory(2), _trajectory(-3), _trajectory(0), Trajectory(segments=[])]
    metrics = {}
    _process(trajectories, {"mode": "tool_error_rebalance"}, metrics)
    assert _outputs(trajectories) == [2, -3, 0]
    assert metrics[PREFIX + "positive_scale"] == metrics[PREFIX + "negative_scale"] == 1
    assert (
        metrics[PREFIX + "positive_advantage_sum_before"]
        == metrics[PREFIX + "positive_advantage_sum_after"]
        == 2
    )
    assert (
        metrics[PREFIX + "negative_advantage_abs_sum_before"]
        == metrics[PREFIX + "negative_advantage_abs_sum_after"]
        == 3
    )


@pytest.mark.parametrize("value", [1, 0, float("inf"), float("nan")])
def test_negative_multiplier_must_be_finite_and_greater_than_one(value):
    with pytest.raises(ValueError, match="negative_multiplier"):
        _process([_trajectory()], {"mode": "tool_error_rebalance", "negative_multiplier": value})


def test_compact_sequences_keep_packed_advantages_after_tool_error_rebalance_and_replay():
    trajectories = [_trajectory(1, error=True), _trajectory(2)]
    for trajectory in trajectories:
        trajectory.tokens = array("q", trajectory.tokens)
        trajectory.loss_mask = array("b", trajectory.loss_mask)
    for _ in range(2):
        _process(trajectories, {"mode": "tool_error_rebalance"})
        assert _outputs(trajectories) == [0, 3]
        for trajectory in trajectories:
            for field in ("base_token_advantages", "token_advantages"):
                assert isinstance(trajectory[field], array)
                assert trajectory[field].typecode == "d"


def test_factory_uses_tool_error_rebalance_component_params():
    from northrl.utils.config_schema import AlgorithmConfig

    config = AlgorithmConfig.model_validate(
        {
            "type": "grpo",
            "optim": {"lr": 1e-5},
            "advantage": {
                "postprocessing": {
                    "mode": "tool_error_rebalance",
                    "params": {
                        "negative_multiplier": 3,
                        "positive_scale_max": 4,
                        "negative_scale_min": 0.25,
                    },
                }
            },
        }
    )
    assert build_postprocessor(config.advantage.postprocessing) == ToolErrorRebalancePostprocessor(
        3, 4, 0.25
    )
    with pytest.raises(ValueError, match="Extra inputs"):
        AlgorithmConfig.model_validate(
            {**config.model_dump(), "advantage_postprocessing": {"mode": "tool_error_rebalance"}}
        )


@pytest.mark.parametrize(
    "params",
    [
        {"negative_multiplier": 1},
        {"negative_multiplier": float("inf")},
        {"positive_scale_max": 0.5},
        {"negative_scale_min": 0},
        {"negative_scale_min": 1.1},
        {"unknown": 1},
    ],
)
def test_tool_error_rebalance_config_rejects_invalid_parameters(params):
    from northrl.utils.config_schema import AlgorithmConfig

    with pytest.raises(ValueError):
        AlgorithmConfig.model_validate(
            {
                "type": "grpo",
                "optim": {"lr": 1e-5},
                "advantage": {"postprocessing": {"mode": "tool_error_rebalance", "params": params}},
            }
        )


def test_sft_cannot_select_advantage_postprocessing():
    from northrl.utils.config_schema import AlgorithmConfig

    with pytest.raises(ValueError, match="requires an advantage estimator"):
        AlgorithmConfig.model_validate(
            {
                "type": "sft",
                "optim": {"lr": 1e-5},
                "advantage": {"postprocessing": {"mode": "tool_error_rebalance"}},
            }
        )


@pytest.mark.parametrize("mode", [None, ""])
@pytest.mark.parametrize("params", [{"negative_multiplier": 1}, {"unknown_typo": 1}])
def test_disabled_postprocessing_rejects_parameters(mode, params):
    from northrl.utils.config_schema import AlgorithmConfig

    with pytest.raises(ValueError):
        AlgorithmConfig.model_validate(
            {
                "type": "grpo",
                "optim": {"lr": 1e-5},
                "advantage": {"postprocessing": {"mode": mode, "params": params}},
            }
        )


def test_disabled_postprocessing_has_empty_params():
    from northrl.utils.config_schema import AlgorithmConfig

    config = AlgorithmConfig(
        type="grpo", optim={"lr": 1e-5}, advantage={"postprocessing": {"mode": None}}
    )
    assert config.advantage.postprocessing.model_dump() == {"mode": None, "params": {}}


def test_token_estimates_take_precedence_over_diagnostic_scalar_without_labels():
    parents = [
        _trajectory(100, error=True, token_advantages=[1]),
        _trajectory(-100, token_advantages=[2]),
    ]
    token_estimates = [parent["token_advantages"] for parent in parents]
    assert ToolErrorRebalancePostprocessor().process(parents, {}) is parents
    assert _outputs(parents) == [0, 3]
    assert token_estimates == [[1], [2]]
    assert [parent["advantage"] for parent in parents] == [100, -100]


@pytest.mark.parametrize("mode", ["grpo", "opd"])
def test_standalone_fresh_estimation_explicitly_resets_saved_base(mode):
    from northrl.algorithm.advantage import build_estimator
    from northrl.utils.algorithm_config import EstimatorConfig

    parents = [_trajectory(1, error=True), _trajectory(2)]
    for run, parent in enumerate(parents):
        parent["group_id"] = "g"
        parent["run_id"] = run
        parent.reward = float(run * 4)
    processor = ToolErrorRebalancePostprocessor()
    processor.process(parents, {})
    clear_postprocessing_cache(parents)
    assert all(
        "base_token_advantages" not in parent and "token_advantages" not in parent
        for parent in parents
    )
    estimator = build_estimator(
        EstimatorConfig(
            mode=mode,
            params={"norm_adv_by_std": False} if mode == "grpo" else {"kl_penalty_coef": 0},
        )
    )
    estimated = estimator.estimate(parents, {}, {})
    assert processor.process(estimated, {}) is estimated
    expected = [[-2], [2]] if mode == "grpo" else [[0], [0]]
    assert [list(parent["base_token_advantages"]) for parent in parents] == expected


def test_cache_cleanup_preserves_unprocessed_token_estimates_and_buffers():
    parents = [_trajectory(100, token_advantages=[1])]
    parent = parents[0]
    original = (parent.tokens, parent.loss_mask, parent["token_advantages"])
    clear_postprocessing_cache(parents)
    assert parent.tokens is original[0]
    assert parent.loss_mask is original[1]
    assert parent["token_advantages"] is original[2]


@pytest.mark.parametrize("compact", [False, True])
def test_scalar_fast_path_matches_explicit_tokens_bitwise_and_all_metrics(compact):
    parents = []
    for value in (0.1, -0.1, 1e110, -1e110, 5e-324, -5e-324, 0.0, -0.0):
        mask = [int(index % 7 != 0) for index in range(257)]
        positions = [index for index, active in enumerate(mask) if active]
        annotations = [
            _annotation(index, index + 1, reasons=["missing_tool_name"] if offset % 3 == 0 else [])
            for offset, index in enumerate(positions[::4])
        ]
        parent = Trajectory(
            tokens=array("q", range(257)) if compact else list(range(257)),
            loss_mask=array("b", mask) if compact else mask,
            extra_fields={"advantage": value, "turn_annotations": list(reversed(annotations))},
        )
        parents.append(parent)
    explicit = deepcopy(parents)
    for parent in explicit:
        values = [parent["advantage"]] * sum(parent.loss_mask)
        parent["token_advantages"] = array("d", values) if compact else values
    processor = ToolErrorRebalancePostprocessor(
        negative_multiplier=2.1, positive_scale_max=1.3, negative_scale_min=0.25
    )
    for _ in range(2):
        scalar_metrics, token_metrics = {}, {}
        processor.process(parents, scalar_metrics)
        processor.process(explicit, token_metrics)
        assert scalar_metrics == token_metrics
        for scalar_parent, token_parent in zip(parents, explicit, strict=True):
            for field in ("base_token_advantages", "token_advantages"):
                assert (
                    np.asarray(scalar_parent[field], dtype=np.float64).tobytes()
                    == np.asarray(token_parent[field], dtype=np.float64).tobytes()
                )


def test_unknown_compact_output_keeps_independent_frozen_base_and_original_inputs():
    source = array("d", [2.0])
    parent = _trajectory(
        99, token_advantages=source, turn_annotations=[_annotation(diagnostics=None)]
    )
    parent.tokens, parent.loss_mask = array("q", parent.tokens), array("b", parent.loss_mask)
    tokens, mask = parent.tokens, parent.loss_mask
    processor = ToolErrorRebalancePostprocessor()
    processor.process([parent], {})
    saved = parent["base_token_advantages"]
    assert saved is not parent["token_advantages"] and saved is not source
    parent["token_advantages"][0] = 999
    processor.process([parent], {})
    assert list(parent["base_token_advantages"]) == list(parent["token_advantages"]) == [2.0]
    assert source == array("d", [2.0])
    assert parent["base_token_advantages"] is not saved
    assert parent.tokens is tokens and parent.loss_mask is mask


def test_late_nonfinite_output_does_not_commit_earlier_staged_buffers_or_metrics():
    parents = [_trajectory(1, error=True), _trajectory(-1e308, error=True)]
    before = deepcopy(parents)
    metrics = {"existing": 7.0}
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite"):
        ToolErrorRebalancePostprocessor().process(parents, metrics)
    assert parents == before
    assert metrics == {"existing": 7.0}


@pytest.mark.parametrize("compact", [False, True])
def test_zero_response_sequence_still_stores_empty_independent_arrays(compact):
    sequence = {"tokens": [10, 11], "loss_mask": [0, 0], "turn_annotations": []}
    if compact:
        sequence["tokens"] = array("q", sequence["tokens"])
        sequence["loss_mask"] = array("b", sequence["loss_mask"])
    parent = Trajectory(segments=[sequence])
    metrics = {}
    ToolErrorRebalancePostprocessor().process([parent], metrics)
    assert list(sequence["base_token_advantages"]) == list(sequence["token_advantages"]) == []
    assert sequence["base_token_advantages"] is not sequence["token_advantages"]
    assert metrics[PREFIX + "response_tokens"] == 0
