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

"""Grouped pass@k / avg@k aggregation of per-sample validation correctness.

Pure aggregation: groups validation trajectories by prompt and computes
pass@k / avg@k from each sample's ``score.correct`` (the per-sample grade the
rollout already produced; see the sample's ``reward_fn`` or ``rollout.reward_fn``).
This module only *combines* scores — it does not grade.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from northrl.northrl_types import Trajectory


def mean(values: list[float]) -> float:
    """Return arithmetic mean, or 0.0 for an empty bucket."""
    return sum(values) / len(values) if values else 0.0


def pass_at_k(*, n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator for grouped validation."""
    if n <= 0 or k <= 0:
        return 0.0
    k = min(k, n)
    c = max(0, min(c, n))
    if n - c < k:
        return 1.0
    product = 1.0
    for value in range(n - c + 1, n + 1):
        product *= 1.0 - k / value
    return 1.0 - product


@dataclass(frozen=True)
class _GroupedPromptResult:
    """Per-prompt grouped validation result after scoring up to ``k`` generations."""

    observed_generations: int
    complete: bool
    pass_at_1: float
    pass_at_k: float
    avg_at_k: float


def _dataset_name(trajectory: Trajectory) -> str:
    dataset = trajectory.get("dataset_name") or trajectory.get("data_source") or "all"
    return str(dataset)


def _group_id(trajectory: Trajectory, fallback_index: int) -> str:
    group_id = trajectory.get("group_id") or trajectory.get("uid")
    if group_id:
        return str(group_id)
    # Validation data carries a stable per-dataset ``index`` but is not repeated
    # when validation_repeat_n == 1. Some rollout workers populate a synthetic
    # ``task_id=0`` default; if task_id won priority, all prompts in a dataset
    # would collapse into one group. Prefer the real dataset index when present.
    index = trajectory.get("index")
    if index is not None:
        return f"index-{index}"
    task_id = trajectory.get("task_id")
    if task_id is not None:
        return f"task-{task_id}"
    return f"trajectory-{fallback_index}"


def _run_id(trajectory: Trajectory, fallback_index: int) -> int:
    run_id = trajectory.get("run_id", fallback_index)
    if hasattr(run_id, "item"):
        try:
            run_id = run_id.item()
        except Exception:  # pragma: no cover - defensive for unusual tensor wrappers
            pass
    try:
        return int(run_id)
    except (TypeError, ValueError):
        return fallback_index


def _sample_correct(trajectory: Trajectory) -> float:
    """Per-sample correctness already produced by the rollout's grader.

    Prefer the ``score.correct`` dict (e.g. single_turn); fall back to the
    trajectory ``reward`` (e.g. SimpleRolloutWorker, which stores a 0/1 reward).
    """
    score = trajectory.get("score")
    if isinstance(score, dict) and "correct" in score:
        try:
            correctness = float(score["correct"])
        except (TypeError, ValueError):
            return 0.0
        return float(isfinite(correctness) and correctness != 0.0)
    reward = trajectory.get("reward")
    if reward is None:
        return 0.0
    try:
        value = float(reward)
        return float(isfinite(value) and value >= 1.0)
    except (TypeError, ValueError):
        return 0.0


def _score_group(ordered_trajectories: list[Trajectory], *, k: int) -> _GroupedPromptResult:
    """Aggregate one prompt group using the standard pass@k / avg@k definitions."""
    scores = [_sample_correct(traj) for traj in ordered_trajectories[:k]]
    observed_generations = len(scores)
    while len(scores) < k:
        # Missing generations should not make pass@k optimistic.
        scores.append(0.0)

    num_correct = int(sum(1 for score in scores if score > 0.0))
    return _GroupedPromptResult(
        observed_generations=observed_generations,
        complete=observed_generations >= k,
        pass_at_1=pass_at_k(n=k, c=num_correct, k=1),
        pass_at_k=pass_at_k(n=k, c=num_correct, k=k),
        avg_at_k=mean(scores),
    )


def _add_metric_bucket(
    metrics: dict[str, float],
    prefix: str,
    results: list[_GroupedPromptResult],
    *,
    k: int,
) -> None:
    pass_at_1_values = [result.pass_at_1 for result in results]
    pass_at_k_values = [result.pass_at_k for result in results]
    avg_at_k_values = [result.avg_at_k for result in results]
    observed_generations = [float(result.observed_generations) for result in results]
    complete_count = sum(1 for result in results if result.complete)

    metrics[f"{prefix}/pass@1"] = mean(pass_at_1_values)
    metrics[f"{prefix}/pass@{k}"] = mean(pass_at_k_values)
    metrics[f"{prefix}/avg@{k}"] = mean(avg_at_k_values)
    metrics[f"{prefix}/num_prompts"] = float(len(results))
    metrics[f"{prefix}/num_generations"] = float(sum(observed_generations))
    metrics[f"{prefix}/observed_generations/mean"] = mean(observed_generations)
    metrics[f"{prefix}/complete_group_ratio"] = complete_count / len(results) if results else 0.0


def compute_grouped_pass_at_k_metrics(
    trajectories: list[Trajectory],
    k: int,
    *,
    metric_prefix: str = "val",
) -> dict[str, float]:
    """Compute grouped pass@k / avg@k metrics from per-sample validation correctness."""
    if k <= 0:
        raise ValueError("grouped pass@k requires k >= 1 (set validate.data.validation_repeat_n)")

    grouped: defaultdict[tuple[str, str], list[tuple[int, Trajectory]]] = defaultdict(list)
    for fallback_index, trajectory in enumerate(trajectories):
        grouped[(_dataset_name(trajectory), _group_id(trajectory, fallback_index))].append(
            (_run_id(trajectory, fallback_index), trajectory)
        )

    dataset_results: defaultdict[str, list[_GroupedPromptResult]] = defaultdict(list)
    duplicate_run_count = 0

    for (dataset_name, _), run_items in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        run_to_trajectory: dict[int, Trajectory] = {}
        for run_id, trajectory in sorted(run_items, key=lambda item: item[0]):
            if run_id in run_to_trajectory:
                duplicate_run_count += 1
                continue
            run_to_trajectory[run_id] = trajectory

        ordered_trajectories = [trajectory for _, trajectory in sorted(run_to_trajectory.items())]
        dataset_results[dataset_name].append(_score_group(ordered_trajectories, k=k))

    metrics: dict[str, float] = {
        f"{metric_prefix}/k": float(k),
        f"{metric_prefix}/duplicate_run_count": float(duplicate_run_count),
    }
    for dataset_name, results in dataset_results.items():
        _add_metric_bucket(metrics, f"{metric_prefix}/{dataset_name}", results, k=k)

    return metrics
