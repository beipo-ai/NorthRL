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

"""
Validator for NorthRL framework
"""

import logging
import threading
from collections import defaultdict

from omegaconf import DictConfig

from .base_module import NorthRLModule
from .data_loader.data_loader import BaseDataLoader
from .northrl_types import Trajectory
from .utils.step_metrics import finite_number, log_timing_metrics
from .utils.trajectory_sequences import iter_trajectory_sequences
from .utils.validation_metrics import compute_grouped_pass_at_k_metrics
from .utils.verifier_metrics import log_verifier_metrics

logger = logging.getLogger(__name__)


class Validator(NorthRLModule):
    """
    Validator collects validation trajectories, computes metrics, and logs results.
    Unlike TrajectoryPool, this does not need batching logic - just simple collection.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the validator

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self._config = config
        self._trajectories: list[Trajectory] = []
        self._lock = threading.Lock()
        self._validate_dataloader: BaseDataLoader = None  # type: ignore  # Set via set_module_references()

    def set_module_references(self, validate_dataloader: BaseDataLoader):
        """
        Set the module references for the validator.
        """
        self._validate_dataloader = validate_dataloader

    def put_trajectory(self, trajectory: Trajectory) -> str:
        """
        Store one complete validation trajectory

        Args:
            trajectory: Trajectory to store

        Returns:
            'success' to match the signature of TrajectoryPool.put_trajectory
        """
        with self._lock:
            self._trajectories.append(trajectory)
        return "success"

    def is_complete(self) -> bool:
        """
        Check if all validation trajectories collected: dataloader is drained and rollout workers are quiescent

        Returns:
            True if all expected trajectories are collected
        """
        return (
            self._validate_dataloader.is_finished()
            and self._activity_tracker.is_rollout_worker_quiescent()
        )

    def compute_and_log_metrics(self) -> dict[str, float]:
        """Publish this validation snapshot with explicit score denominators.

        Task scores must be recognized outcomes or registered on the parent via
        metric_score_names. Runtime counters sharing the legacy score dict do
        not automatically become validation task-quality metrics.
        """
        with self._lock:
            trajectories, self._trajectories = self._trajectories, []

        if not trajectories:
            logger.error("No trajectories to compute metrics")
            raise ValueError("No trajectories to compute metrics")

        score_values: dict[str, list[float]] = defaultdict(list)
        dataset_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        dataset_counts: dict[str, int] = defaultdict(int)
        dataset_length_counts: dict[str, int] = defaultdict(int)
        known_scores = {"reward", "correct", "swebench_resolved", "resolved"}
        for trajectory in trajectories:
            raw_score = trajectory.get("score", {})
            score = dict(raw_score) if isinstance(raw_score, dict) else {}
            if trajectory.reward is not None:
                score["reward"] = trajectory.reward
            registered = set(trajectory.get("metric_score_names", ()))
            names = (set(score) & known_scores) | registered
            dataset = str(trajectory.get("dataset_name") or trajectory.get("data_source") or "all")
            dataset_counts[dataset] += 1
            dataset_length_counts[dataset] += trajectory.get("finish_reason") == "length"
            for name in sorted(names):
                values = score_values[name]
                dataset_values = dataset_scores[dataset][name]
                raw_value = score.get(name)
                if name != "reward" and isinstance(raw_value, bool):
                    raw_value = int(raw_value)
                value = finite_number(raw_value)
                if value is not None:
                    values.append(value)
                    dataset_values.append(value)

        metrics: dict[str, float] = {}
        log_verifier_metrics(trajectories, metrics, prefix="val/")
        log_timing_metrics(trajectories, metrics, prefix="val/")
        for name, values in score_values.items():
            metrics[f"val/{name}_observed_count"] = len(values)
            if values:
                metrics[f"val/{name}_mean"] = sum(values) / len(values)
        for dataset, scores in dataset_scores.items():
            for name, values in scores.items():
                metrics[f"val/{dataset}/{name}_observed_count"] = len(values)
                if values:
                    metrics[f"val/{dataset}/{name}_mean"] = sum(values) / len(values)

        metrics["val/trajectory_count"] = len(trajectories)
        length_count = sum(dataset_length_counts.values())
        length_fraction = length_count / len(trajectories)
        metrics["val/finish_reason_length_count"] = length_count
        metrics["val/finish_reason_length_fraction"] = length_fraction
        logger.info(
            "[finish-reason] validation: length=%d/%d (%.2f%%)",
            length_count,
            len(trajectories),
            length_fraction * 100,
        )
        for dataset, count in dataset_counts.items():
            dataset_length_count = dataset_length_counts[dataset]
            dataset_length_fraction = dataset_length_count / count
            metrics[f"val/{dataset}/trajectory_count"] = count
            metrics[f"val/{dataset}/finish_reason_length_count"] = dataset_length_count
            metrics[f"val/{dataset}/finish_reason_length_fraction"] = dataset_length_fraction
            logger.info(
                "[finish-reason] validation dataset=%s: length=%d/%d (%.2f%%)",
                dataset,
                dataset_length_count,
                count,
                dataset_length_fraction * 100,
            )
        metrics["val/segment_count"] = sum(
            sum(1 for _ in iter_trajectory_sequences(trajectory)) for trajectory in trajectories
        )

        if str(self._config.validate.get("aggregation", "mean")) == "pass_at_k":
            k = int(self._config.validate.data.validation_repeat_n)
            grouped = compute_grouped_pass_at_k_metrics(trajectories, k)
            # avg@k and pass@1 have identical definitions for the configured k
            # slots. Keep the canonical pass@1 field in experiment history.
            metrics.update({key: value for key, value in grouped.items() if "/avg@" not in key})

        # Log metrics via activity tracker
        self._activity_tracker.experiment_logger_post(backend="wandb", data=metrics, commit=True)

        logger.info(f"Validation metrics: {metrics}")

        return metrics
