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

import logging
import math
import threading
import time
import uuid
from collections import Counter, OrderedDict
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .error_reporter import ErrorReporter, ErrorSeverity
from .remote_rollout.metrics import (
    LIVE_METRICS_INTERVAL_SECONDS,
    MODEL_REQUEST_GAUGES,
    REQUEST_DURATION_BUCKET_METRICS,
    TOOL_FORMAT_REASONS,
)
from .remote_rollout.tool_errors import ENV_TOOL_ERROR_CATEGORIES, TOOL_ERROR_CATEGORIES
from .utils.tracking import Tracking

logger = logging.getLogger(__name__)


class ActivityTracker:
    """
    Centralized tracker of in-flight work across modules.

    Modules communicate with this tracker through ActivityTrackerProxy instances.
    The tracker provides start/end methods for activity tracking and is used by
    the controller to monitor system quiescence and reported errors.
    """

    # Executor callbacks are synchronous. This window also tolerates recent
    # duplicate terminal events and late starts without retaining all run IDs.
    _MAX_RECENT_E2B_ROLLOUTS = 4096

    experiment_logger: Tracking

    def __init__(self, config: DictConfig, max_errors: int = 1000) -> None:
        self._cv = threading.Condition()
        self._count: int = 0
        self._by_module: dict[str, int] = {}
        self._tokens: dict[str, str] = {}
        self._error_reporter: ErrorReporter = ErrorReporter(max_errors=max_errors)
        self._module_refs: dict[str, Any] = {}  # module_name -> module reference for metrics
        self._rollout_worker_modules: set[str] = set()  # Track which modules are rollout workers
        self._current_training_step: int = 0  # Track current training step
        self._config = config

        self.experiment_logger = Tracking(
            project_name=self._config.experiment.project,
            experiment_name=self._config.experiment.name,
            default_backend=self._config.runtime.logger.backends,
            config=OmegaConf.to_container(config, resolve=True),
        )

        # Rollout progress is refreshed by the controller on its main thread.
        progress_config = self._config.runtime.progress
        self._progress_update_interval = progress_config.update_interval_s
        self._progress_log_interval = progress_config.log_every_n_steps
        self._progress_use_tqdm = progress_config.use_tqdm
        self._progress_next_update_at: float | None = None
        self._progress_update_count = 0
        self._progress_last_metrics: dict[str, Any] | None = None

        self._e2b_started_at = time.monotonic()
        self._e2b_next_flush_at: float | None = None
        self._e2b_seen_event = False
        self._e2b_active_rollout_ids: set[str] = set()
        self._e2b_finished_rollout_ids: OrderedDict[str, None] = OrderedDict()
        self._e2b_segments_per_rollout: Counter[int] = Counter()
        # These flags span attempts, then are released at logical completion.
        self._e2b_tool_rollout_flags: dict[str, set[str]] = {}
        self._e2b_tool_rollout_counts = {
            name: 0
            for name in (
                "observed_rollouts",
                "unobserved_rollouts",
                "env_issue_rollouts",
                "timeout_observed_rollouts",
                "timeout_env_issue_rollouts",
            )
        }
        self._e2b_attempts: dict[tuple[str, int], dict[str, Any]] = {}
        self._e2b_sandboxes: dict[tuple[str, int], dict[str, Any]] = {}
        self._e2b_metric_names = {
            "model_turns": "trace/model_turns",
            "generated_tokens": "trace/generated_tokens",
            "max_context_tokens": "trace/max_context_tokens",
            "trace_segments": "trace/segments",
            "model_requests_started": "inference/request_count",
            "model_requests_completed": "inference/completed_count",
            **{
                f"model_requests_{status}": f"inference/{status}_count"
                for status in ("success", "error", "timeout", "cancelled")
            },
            "model_request_seconds_sum": "inference/request_seconds_sum",
            "model_request_seconds_max": "inference/request_seconds_max",
            **{name: f"inference/{name}" for name in REQUEST_DURATION_BUCKET_METRICS},
            **{
                f"tool_{name}": f"tool/{name}"
                for name in (
                    "calls_detected",
                    "calls_emitted",
                    "calls_invalid",
                    "results_observed",
                    "result_errors",
                    "format_error_turns",
                    "format_unknown_turns",
                    "call_counts_unknown_turns",
                    *(f"format_reason_{reason}_turns" for reason in TOOL_FORMAT_REASONS),
                    "arguments_json_errors",
                    "arguments_type_errors",
                )
            },
        }
        self._e2b_metric_names.update(
            {
                f"tool_error_{category}": f"tool/error_{category}"
                for category in TOOL_ERROR_CATEGORIES
            }
        )
        self._e2b_metric_totals: dict[str, float] = {name: 0.0 for name in self._e2b_metric_names}
        self._e2b_counters: dict[str, float] = {
            name: 0
            for name in (
                "attempt_count",
                "retry_count",
                "completed_rollouts",
                "successful_rollouts",
                "task_failed_rollouts",
                "overlength_count",
                "env_error_count",
                "completed_attempts",
                "env_error_attempts",
                "timeout_count",
                "timeout_sandbox",
                "poll_error_count",
                "cleanup_error_count",
                "attempt_duration_seconds",
            )
        }

    def start(self, module: str, work: str) -> str:  # pylint: disable=unused-argument
        token = str(uuid.uuid4())
        with self._cv:
            self._count += 1
            self._by_module[module] = self._by_module.get(module, 0) + 1
            self._tokens[token] = module
        return token

    def end(self, token: str) -> None:
        with self._cv:
            module: str | None = self._tokens.pop(token, None)
            if module is None:
                return
            self._count -= 1
            self._by_module[module] = max(0, self._by_module.get(module, 0) - 1)
            if self._count == 0:
                self._cv.notify_all()

    def is_quiescent(self) -> bool:
        with self._cv:
            return self._count == 0

    def is_rollout_worker_quiescent(self) -> bool:
        """Check if all registered rollout workers are currently idle (no active work)

        Returns:
            bool: True if all rollout workers have no active work, False otherwise
        """
        with self._cv:
            for module_name in self._rollout_worker_modules:
                if self._by_module.get(module_name, 0) > 0:
                    return False
            return True

    def wait_quiescent(self, timeout: float | None = None) -> bool:
        with self._cv:
            end_time = None if timeout is None else (time.time() + timeout)
            while self._count != 0:
                remaining = None if end_time is None else max(0.0, end_time - time.time())
                if end_time is not None and remaining is not None and remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True

    def register_module(
        self, module_name: str, module_ref: Any, is_rollout_worker: bool = False
    ) -> None:
        """Register a module reference for progress metrics.

        Args:
            module_name: Name of the module
            module_ref: Reference to the module (for progress metrics)
            is_rollout_worker: Whether this module is a rollout worker (for special monitoring)
        """
        with self._cv:
            self._module_refs[module_name] = module_ref
            if is_rollout_worker:
                self._rollout_worker_modules.add(module_name)
            self._by_module[module_name] = 0

    def report_exception(
        self,
        module: str,
        work: str,
        exception: Exception,
        severity: ErrorSeverity | None = None,
    ) -> str:
        """
        Report an exception through the error reporter

        Args:
            module: Module name where the exception occurred
            work: Work context where the exception occurred
            exception: The exception that was raised
            severity: Severity level (defaults to ERROR)

        Returns:
            str: Error ID of the reported exception
        """
        severity = severity or ErrorSeverity.ERROR
        return self._error_reporter.report_exception(module, work, exception, severity)

    # -- module running status summary --
    def get_running_status_summary(self) -> str:
        """Get a human-readable summary of current activity status"""
        with self._cv:
            if self._count == 0:
                return "System is quiescent (no active work)"

            module_running_status = []
            for module, count in self._by_module.items():
                if count > 0:
                    module_running_status.append(f"{module}: {count}")

            return f"Active work items: {self._count} total ({', '.join(module_running_status)})"

    def get_error_health_status(self) -> dict[str, Any]:
        """Get health status from the error reporter"""
        return self._error_reporter.get_health_status()

    # -- training step tracking --
    def set_training_step(self, step: int) -> None:
        """
        Set the current training step

        Args:
            step: The current training step
        """
        with self._cv:
            self._current_training_step = step

        if step == 1 and self.experiment_logger.feishu_logger is not None:
            self.experiment_logger.feishu_logger.post(
                content=f'"Experiment: {self._config.experiment.project}/{self._config.experiment.name}"'
                f"begin training, wandb: {self.experiment_logger.wandb_url} ",
                title="Begin!",
            )

    def get_training_step(self) -> int:
        """
        Get the current training step

        Returns:
            int: The current training step
        """
        with self._cv:
            return self._current_training_step

    def experiment_logger_post(self, backend: str, **kwargs) -> None:
        if backend == "feishu":
            if self.experiment_logger.feishu_logger is not None:
                self.experiment_logger.feishu_logger.post(
                    content=kwargs["content"], title=kwargs["title"]
                )
        elif backend == "wandb":
            if "step" not in kwargs:
                kwargs["step"] = self._current_training_step
            log_kwargs = {"data": kwargs["data"], "step": kwargs["step"]}
            if "commit" in kwargs:
                log_kwargs["commit"] = kwargs["commit"]
            self.experiment_logger.log(**log_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def experiment_logger_flush(self, step: int) -> None:
        """Publish a completed training round after all its metrics are available."""
        self.experiment_logger.flush(step)

    # -- live remote rollout telemetry --
    def record_e2b_event(self, event: dict[str, Any]) -> None:
        """Aggregate executor events in memory; workers never call W&B directly."""
        now = time.monotonic()
        kind = event.get("kind")
        key = (str(event.get("rollout_id", "")), int(event.get("attempt", 1)))
        with self._cv:
            self._e2b_seen_event = True
            if kind == "rollout_finished":
                if key[0] in self._e2b_finished_rollout_ids:
                    return
                self._e2b_finished_rollout_ids[key[0]] = None
                if len(self._e2b_finished_rollout_ids) > self._MAX_RECENT_E2B_ROLLOUTS:
                    # Totals persist; only historical deduplication is bounded.
                    # Events older than the window are treated as new events.
                    self._e2b_finished_rollout_ids.popitem(last=False)
                self._e2b_active_rollout_ids.discard(key[0])
                self._e2b_counters["completed_rollouts"] += 1
                self._finish_e2b_tool_rollout(key[0])
                status = event.get("status")
                if status == "env_error":
                    self._e2b_counters["env_error_count"] += 1
                if status == "task_failed":
                    self._e2b_counters["task_failed_rollouts"] += 1
                if status == "success":
                    self._e2b_counters["successful_rollouts"] += 1
                if event.get("termination_reason") == "context_length_exceeded":
                    self._e2b_counters["overlength_count"] += 1
                segments = event.get("trajectory_segments")
                # Final results are authoritative even when the last heartbeat
                # was missed. Environment failures invalidate their GRPO groups;
                # retries and missing observations are not zero-segment samples.
                # Explicit zero-segment task outcomes are still complete rollouts.
                if (
                    status in ("success", "task_failed")
                    and isinstance(segments, int)
                    and not isinstance(segments, bool)
                    and segments >= 0
                ):
                    self._e2b_segments_per_rollout[segments] += 1
                return
            if kind == "attempt_started":
                if key[0] in self._e2b_finished_rollout_ids:
                    return
                # A logical rollout stays active across sequential retries.
                self._e2b_active_rollout_ids.add(key[0])
                if key not in self._e2b_attempts:
                    self._e2b_attempts[key] = {
                        "started_at": now,
                        "metrics": {},
                    }
                    self._e2b_counters["attempt_count"] += 1
                    self._e2b_counters["retry_count"] += int(key[1] > 1)
                return
            if kind == "sandbox_closed":
                # A cleanup confirmation can arrive after its attempt has ended.
                sandbox = self._e2b_sandboxes.get(key)
                if event.get("confirmed", False):
                    self._e2b_sandboxes.pop(key, None)
                elif sandbox is not None or key in self._e2b_attempts:
                    self._e2b_counters["cleanup_error_count"] += 1
                    if sandbox is not None:
                        sandbox["unknown"] = True
                return
            attempt = self._e2b_attempts.get(key)
            if attempt is None:
                return
            if kind == "sandbox_created":
                # Keep creation history after sandbox_closed removes the live entry.
                attempt["sandbox_created"] = True
                self._e2b_sandboxes[key] = {
                    "created_at": now,
                    "last_heartbeat_at": None,
                    "heartbeat": None,
                    "timeout_seconds": float(event.get("timeout_seconds", 0)),
                    "unknown": False,
                }
            elif kind == "snapshot":
                if key[0] in self._e2b_finished_rollout_ids:
                    return
                heartbeat = event.get("heartbeat")
                previous_heartbeat = attempt.get("heartbeat")
                if isinstance(heartbeat, (int, float)):
                    if previous_heartbeat is not None and heartbeat <= previous_heartbeat:
                        return
                    attempt["heartbeat"] = heartbeat
                sandbox = self._e2b_sandboxes.get(key)
                if (
                    sandbox is not None
                    and heartbeat is not None
                    and heartbeat != sandbox["heartbeat"]
                ):
                    sandbox["heartbeat"] = heartbeat
                    sandbox["last_heartbeat_at"] = now
                    sandbox["unknown"] = False
                for name, value in event.get("metrics", {}).items():
                    if name not in self._e2b_metric_names and name not in MODEL_REQUEST_GAUGES:
                        continue
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        continue
                    if name in MODEL_REQUEST_GAUGES:
                        attempt["metrics"][name] = value
                        continue
                    previous = attempt["metrics"].get(name, 0)
                    # Snapshots contain cumulative counters; repeated polling must
                    # not count the same model turn or tool result twice.
                    if name in ("max_context_tokens", "model_request_seconds_max"):
                        self._e2b_metric_totals[name] = max(
                            self._e2b_metric_totals.get(name, 0), value
                        )
                    else:
                        self._e2b_metric_totals[name] = self._e2b_metric_totals.get(name, 0) + max(
                            0, value - previous
                        )
                    attempt["metrics"][name] = max(previous, value)
                flags = self._e2b_tool_rollout_flags.setdefault(key[0], set())
                if attempt["metrics"].get("tool_results_observed", 0) > 0:
                    flags.add("observed")
                if any(
                    attempt["metrics"].get(f"tool_error_{category}", 0) > 0
                    for category in ENV_TOOL_ERROR_CATEGORIES
                ):
                    flags.add("env_issue")
            elif kind == "poll_error":
                self._e2b_counters["poll_error_count"] += 1
                sandbox = self._e2b_sandboxes.get(key)
                if sandbox is not None:
                    sandbox["unknown"] = True
            elif kind == "attempt_finished":
                timed_out = bool(event.get("timed_out", False))
                if timed_out and key[0] not in self._e2b_finished_rollout_ids:
                    self._e2b_tool_rollout_flags.setdefault(key[0], set()).add("timed_out")
                self._e2b_counters["completed_attempts"] += 1
                self._e2b_counters["timeout_count"] += int(timed_out)
                self._e2b_counters["timeout_sandbox"] += int(
                    timed_out and attempt.get("sandbox_created", False)
                )
                self._e2b_counters["env_error_attempts"] += int(event.get("status") == "env_error")
                self._e2b_counters["attempt_duration_seconds"] += max(
                    0.0, float(event.get("duration_seconds", now - attempt["started_at"]))
                )
                self._e2b_attempts.pop(key, None)

    def _finish_e2b_tool_rollout(self, rollout_id: str) -> None:
        """Count observation coverage and environment clues once per rollout."""
        flags = self._e2b_tool_rollout_flags.pop(rollout_id, set())
        if "observed" not in flags:
            self._e2b_tool_rollout_counts["unobserved_rollouts"] += 1
            return
        self._e2b_tool_rollout_counts["observed_rollouts"] += 1
        self._e2b_tool_rollout_counts["env_issue_rollouts"] += int("env_issue" in flags)
        if "timed_out" in flags:
            self._e2b_tool_rollout_counts["timeout_observed_rollouts"] += 1
            self._e2b_tool_rollout_counts["timeout_env_issue_rollouts"] += int("env_issue" in flags)

    def get_e2b_metrics(self) -> dict[str, Any]:
        """Snapshot totals and live gauges with a short lock held."""
        now = time.monotonic()
        with self._cv:
            # E2B sandboxes have a fixed lifetime; failed cleanup observations
            # must not stay in live gauges forever after that lifetime ends.
            expired = [
                key
                for key, sandbox in self._e2b_sandboxes.items()
                if sandbox["timeout_seconds"] > 0
                and now - sandbox["created_at"] >= sandbox["timeout_seconds"]
            ]
            for key in expired:
                self._e2b_sandboxes.pop(key, None)
            metrics: dict[str, Any] = {
                f"live/e2b/{name}": value for name, value in self._e2b_counters.items()
            }
            metrics["live/e2b/valid_rollouts"] = (
                self._e2b_counters["successful_rollouts"]
                + self._e2b_counters["task_failed_rollouts"]
            )
            sandboxes = list(self._e2b_sandboxes.values())
            metrics["live/e2b/active_sandboxes"] = len(sandboxes)
            metrics["live/e2b/active_rollouts"] = len(self._e2b_active_rollout_ids)
            metrics["live/e2b/unknown_sandboxes"] = sum(bool(item["unknown"]) for item in sandboxes)
            metrics["live/e2b/heartbeat_pending_sandboxes"] = sum(
                item["last_heartbeat_at"] is None for item in sandboxes
            )
            heartbeat_ages = [
                now - item["last_heartbeat_at"]
                for item in sandboxes
                if item["last_heartbeat_at"] is not None
            ]
            metrics["live/e2b/heartbeat_age_seconds_max"] = max(heartbeat_ages, default=0.0)
            metrics["live/e2b/sandbox_age_seconds_max"] = max(
                (now - item["created_at"] for item in sandboxes), default=0.0
            )
            remaining = [
                max(0.0, item["timeout_seconds"] - (now - item["created_at"]))
                for item in sandboxes
                if item["timeout_seconds"] > 0
            ]
            if remaining:
                metrics["live/e2b/ttl_remaining_seconds_min"] = min(remaining)
            completed = self._e2b_counters["completed_rollouts"]
            metrics["live/e2b/overlength_ratio"] = (
                self._e2b_counters["overlength_count"] / completed if completed else 0.0
            )
            if completed:
                metrics["live/e2b/env_error_rate"] = (
                    self._e2b_counters["env_error_count"] / completed
                )
            completed_attempts = self._e2b_counters["completed_attempts"]
            if completed_attempts:
                metrics["live/e2b/timeout_rate"] = (
                    self._e2b_counters["timeout_count"] / completed_attempts
                )
                metrics["live/e2b/attempt_duration_seconds_mean"] = (
                    self._e2b_counters["attempt_duration_seconds"] / completed_attempts
                )
            for name, value in self._e2b_metric_totals.items():
                if name not in REQUEST_DURATION_BUCKET_METRICS:
                    metrics[f"live/{self._e2b_metric_names[name]}"] = value
            request_count = self._e2b_metric_totals["model_requests_completed"]
            if request_count:
                metrics["live/inference/request_seconds_mean"] = (
                    self._e2b_metric_totals["model_request_seconds_sum"] / request_count
                )
                # Merge cumulative buckets before estimating the percentile;
                # rollout averages would give short and long rollouts equal weight.
                rank = math.ceil(0.95 * request_count)
                for name, upper_bound in REQUEST_DURATION_BUCKET_METRICS.items():
                    if self._e2b_metric_totals[name] >= rank:
                        metrics["live/inference/request_seconds_p95"] = min(
                            upper_bound, self._e2b_metric_totals["model_request_seconds_max"]
                        )
                        break
            active_metrics = [
                attempt["metrics"] for attempt in self._e2b_attempts.values() if attempt["metrics"]
            ]
            metrics["live/inference/inflight"] = sum(
                snapshot.get("model_request_inflight", 0) for snapshot in active_metrics
            )
            metrics["live/inference/oldest_inflight_seconds"] = max(
                (
                    snapshot.get("model_request_oldest_inflight_seconds", 0)
                    for snapshot in active_metrics
                    if snapshot.get("model_request_inflight", 0) > 0
                ),
                default=0.0,
            )
            metrics["live/e2b/active_attempts_with_metrics"] = len(active_metrics)
            for source, target in (
                ("model_turns", "trace/active_model_turns"),
                ("generated_tokens", "trace/active_generated_tokens"),
                ("trace_segments", "trace/active_segments"),
                ("tool_calls_emitted", "tool/active_calls_emitted"),
            ):
                values = [snapshot.get(source, 0) for snapshot in active_metrics]
                metrics[f"live/{target}_mean"] = sum(values) / len(values) if values else 0.0
                metrics[f"live/{target}_max"] = max(values, default=0.0)
            segment_counts = self._e2b_segments_per_rollout
            recorded_rollouts = sum(segment_counts.values())
            metrics["live/trace/rollouts_with_segments"] = recorded_rollouts - segment_counts[0]
            metrics["live/trace/rollouts_without_segments"] = segment_counts[0]
            if recorded_rollouts:
                metrics["live/trace/segments_per_trajectory_mean"] = (
                    sum(segments * count for segments, count in segment_counts.items())
                    / recorded_rollouts
                )
                metrics["live/trace/segments_per_rollout_max"] = max(segment_counts)
                metrics["live/trace/split_trajectory_ratio"] = (
                    sum(count for segments, count in segment_counts.items() if segments > 1)
                    / recorded_rollouts
                )
                rank = math.ceil(0.95 * recorded_rollouts)
                cumulative = 0
                for segments, count in sorted(segment_counts.items()):
                    cumulative += count
                    if cumulative >= rank:
                        metrics["live/trace/segments_per_rollout_p95"] = segments
                        break
            known_format_turns = (
                self._e2b_metric_totals["model_turns"]
                - self._e2b_metric_totals["tool_format_unknown_turns"]
            )
            if known_format_turns > 0:
                metrics["live/tool/format_error_turn_rate"] = (
                    self._e2b_metric_totals["tool_format_error_turns"] / known_format_turns
                )
            tool_results = self._e2b_metric_totals["tool_results_observed"]
            if tool_results:
                metrics["live/tool/result_error_rate"] = (
                    self._e2b_metric_totals["tool_result_errors"] / tool_results
                )
            metrics.update(
                {
                    f"live/tool/{name}": value
                    for name, value in self._e2b_tool_rollout_counts.items()
                }
            )
            tool_counts = self._e2b_tool_rollout_counts
            for name, numerator, denominator in (
                ("observation_coverage", tool_counts["observed_rollouts"], completed),
                (
                    "env_issue_rollout_rate",
                    tool_counts["env_issue_rollouts"],
                    tool_counts["observed_rollouts"],
                ),
                (
                    "timeout_env_issue_rollout_rate",
                    tool_counts["timeout_env_issue_rollouts"],
                    tool_counts["timeout_observed_rollouts"],
                ),
            ):
                if denominator:
                    metrics[f"live/tool/{name}"] = numerator / denominator
            return metrics

    def flush_e2b_metrics(self, force: bool = False) -> float | None:
        """Write runtime monitoring logs periodically, independent of training steps."""
        now = time.monotonic()
        if not self._e2b_seen_event:
            return None
        if not force and self._e2b_next_flush_at is not None and now < self._e2b_next_flush_at:
            return self._e2b_next_flush_at
        try:
            self.experiment_logger.log_live(
                self.get_e2b_metrics(), elapsed_seconds=now - self._e2b_started_at
            )
        except Exception:
            logger.warning("Failed to write live monitoring log", exc_info=True)
        finally:
            self._e2b_next_flush_at = time.monotonic() + LIVE_METRICS_INTERVAL_SECONDS
        return self._e2b_next_flush_at

    # -- progress monitoring --
    def get_progress_metrics(self) -> dict[str, Any]:
        """
        Collect progress metrics from all modules for progress monitoring.
        Uses _module_refs to find DataLoader, TrajectoryPool, and WeightSyncController.

        Returns:
            dict with keys:
                - active_rollout_workers: Number of rollout workers currently active
                - dataloader: Progress info from dataloader (if available)
                - rollout_round: Identity-based counts for the current ordered input batch
                - rollout_continuous: Cumulative completion and current cross-batch activity
                - trajectory_pool: Progress info from trajectory pool (if available)
                - sync_mode: Current weight sync mode
        """
        metrics: dict[str, Any] = {}

        # Get active rollout worker count
        with self._cv:
            active_workers = sum(
                self._by_module.get(module_name, 0) for module_name in self._rollout_worker_modules
            )
            metrics["active_rollout_workers"] = active_workers

        # Find singleton modules from _module_refs (they are registered with "-0" suffix)
        # The value in _module_refs is the ret list, which contains the actual module as the only element
        # Module names use lowercase with underscores (from role enum values)
        dataloader_list = self._module_refs.get("data_loader-0")
        trajectory_pool_list = self._module_refs.get("trajectory_pool-0")
        weight_sync_controller_list = self._module_refs.get("weight_sync_controller-0")

        # Extract the actual module (first element of the list)
        dataloader = dataloader_list[0] if dataloader_list and len(dataloader_list) > 0 else None
        trajectory_pool = (
            trajectory_pool_list[0]
            if trajectory_pool_list and len(trajectory_pool_list) > 0
            else None
        )
        weight_sync_controller = (
            weight_sync_controller_list[0]
            if weight_sync_controller_list and len(weight_sync_controller_list) > 0
            else None
        )

        # Get dataloader progress (if available)
        if dataloader is not None:
            try:
                dataloader_info = dataloader.get_progress_info()
                metrics["dataloader"] = dataloader_info
                get_round = getattr(dataloader, "get_rollout_progress", None)
                if get_round is not None:
                    metrics["rollout_round"] = get_round(
                        grouped=self._config.algorithm.type in {"grpo", "opd"}
                    )
                get_continuous = getattr(dataloader, "get_continuous_rollout_progress", None)
                if get_continuous is not None:
                    metrics["rollout_continuous"] = get_continuous()
            except Exception as e:
                logger.warning(f"Could not get dataloader progress: {e}", exc_info=True)
                metrics["dataloader"] = None
        else:
            logger.debug("DataLoader not found in module registry")
            metrics["dataloader"] = None

        # Get trajectory pool progress (if available)
        if trajectory_pool is not None:
            try:
                pool_info = trajectory_pool.get_progress_info()
                metrics["trajectory_pool"] = pool_info
            except Exception as e:
                logger.warning(f"Could not get trajectory pool progress: {e}", exc_info=True)
                metrics["trajectory_pool"] = None
        else:
            logger.debug("TrajectoryPool not found in module registry")
            metrics["trajectory_pool"] = None

        # Get sync mode (if available)
        if weight_sync_controller is not None:
            try:
                sync_mode = weight_sync_controller.sync_mode
                metrics["sync_mode"] = sync_mode
            except Exception as e:
                logger.warning(f"Could not get sync mode: {e}", exc_info=True)
                metrics["sync_mode"] = None
        else:
            logger.debug("WeightSyncController not found in module registry")
            metrics["sync_mode"] = None

        return metrics

    def update_progress(self) -> float:
        """Refresh progress when due and return the next update's monotonic timestamp.

        Called only by the controller's main thread.
        """
        next_live_at = self.flush_e2b_metrics()
        now = time.monotonic()
        if self._progress_next_update_at is not None and now < self._progress_next_update_at:
            return (
                min(self._progress_next_update_at, next_live_at)
                if next_live_at is not None
                else self._progress_next_update_at
            )

        try:
            metrics = self.get_progress_metrics()
            self._progress_last_metrics = metrics
            self._update_progress_display(metrics)
            self._progress_update_count += 1
        except Exception as e:
            logger.error(f"Error updating rollout progress: {e}", exc_info=True)
        finally:
            self._progress_next_update_at = time.monotonic() + self._progress_update_interval
        return (
            min(self._progress_next_update_at, next_live_at)
            if next_live_at is not None
            else self._progress_next_update_at
        )

    def _update_progress_display(self, metrics: dict[str, Any]) -> None:
        """
        Update progress display with latest metrics

        Args:
            metrics: Progress metrics
        """
        sync_mode = metrics.get("sync_mode")
        round_info = metrics.get("rollout_round")
        continuous_info = metrics.get("rollout_continuous")
        pool_info = metrics.get("trajectory_pool")
        grouped = self._config.algorithm.type in {"grpo", "opd"}
        display_parts = []

        # All ordered-round fields come from one identity-based snapshot. Pool
        # consumption and worker activity cannot make completion counts regress.
        if sync_mode in {"sync", "batch-async"} and isinstance(round_info, dict):
            display_parts.append(
                f"Rollouts: {round_info['finished']}/{round_info['total']} finished"
                f" | {round_info['in_progress']} in progress | {round_info['queued']} queued"
            )
            if grouped:
                display_parts.append(
                    f"Groups: {round_info['groups_complete']}/{round_info['groups_total']} complete"
                )
            display_parts.append(f"Segments: {round_info['ready_segments']} ready for training")
        elif sync_mode == "fully-async" and isinstance(continuous_info, dict):
            display_parts.append(
                f"Rollouts: {continuous_info['finished_total']} finished total"
                f" | {continuous_info['in_progress']} in progress | {continuous_info['queued']} queued"
            )
            # Inventory is per model: separate pools cannot jointly satisfy one
            # training batch. Before the first result, the actor pool is empty.
            if pool_info is None:
                display_parts.append("Pool: unavailable")
                instances: dict[str, Any] = {}
            else:
                instances = pool_info.get("instances") or {
                    "actor": {"groups_per_batch": self._config.data.train.batch_size}
                }
            for model_tag, instance in instances.items():
                label = f"Pool [{model_tag}]" if len(instances) > 1 else "Pool"
                if grouped:
                    display_parts.append(
                        f"{label} groups: {instance.get('ready_groups', 0)} ready"
                        f" | {instance.get('incomplete_groups', 0)} partially returned"
                        f" | {instance['groups_per_batch']} per batch"
                    )
                display_parts.append(
                    f"{label} segments: {instance.get('ready_sequences', 0)} ready for training"
                )
        else:
            # Custom loaders without rollout identity tracking retain a clearly
            # scoped activity/inventory view without a fabricated completion ratio.
            display_parts.append(f"Workers: {metrics.get('active_rollout_workers', 0)} active")
            if pool_info:
                if grouped:
                    display_parts.append(f"Pool groups: {pool_info.get('ready_groups', 0)} ready")
                ready_segments = pool_info.get(
                    "ready_sequences", pool_info.get("ready_segments", 0)
                )
                display_parts.append(f"Pool segments: {ready_segments} ready for training")

        if self._progress_update_count % self._progress_log_interval == 0:
            header = f"Progress [{sync_mode}]" if sync_mode else "Progress"
            message = header + "\n  " + "\n  ".join(display_parts)
            if self._progress_use_tqdm:
                try:
                    from tqdm import tqdm

                    tqdm.write(message)
                except ImportError:
                    logger.info(message)
            else:
                logger.info(message)


class ActivityTrackerProxy:
    """
    Local proxy for ActivityTracker that provides the same interface but communicates
    with a centralized ActivityTracker.

    This proxy is stateless and simply forwards all calls to the central tracker
    using direct local calls.
    """

    def __init__(self, central_tracker: Any):
        """
        Initialize the proxy with a reference to the central tracker.

        Args:
            central_tracker: Reference to the central ActivityTracker
        """
        self._central_tracker: ActivityTracker = central_tracker

    def track(self, module: str, work: str, auto_report_errors: bool = True) -> "_ProxyTrackCtx":
        """
        Create a context manager for tracking work and optionally reporting errors.

        Args:
            module: Module name doing the work
            work: Description of the work being done
            auto_report_errors: Whether to automatically report exceptions to error reporter

        Returns:
            Context manager that tracks activity and optionally reports errors
        """
        return ActivityTrackerProxy._ProxyTrackCtx(
            self._central_tracker, module, work, auto_report_errors
        )

    class _ProxyTrackCtx:
        """
        Context manager that forwards tracking calls to the central ActivityTracker.
        """

        def __init__(
            self,
            central_tracker: Any,
            module: str,
            work: str,
            auto_report_errors: bool = True,
        ) -> None:
            self._central_tracker = central_tracker
            self._token: str | None = None
            self._module = module
            self._work = work
            self._auto_report_errors = auto_report_errors

        def __enter__(self):
            self._token = self._central_tracker.start(self._module, self._work)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            # Report exception automatically if enabled and exception occurred
            if self._auto_report_errors and exc_type is not None and exc_value is not None:
                try:
                    self._central_tracker.report_exception(self._module, self._work, exc_value)
                except Exception:
                    # Don't let error reporting interfere with original exception
                    pass

            # End activity tracking
            if self._token is not None:
                self._central_tracker.end(self._token)

            # Don't suppress the original exception
            return False

    def is_rollout_worker_quiescent(self) -> bool:
        """Check if all registered rollout workers are currently idle (no active work)

        Returns:
            bool: True if all rollout workers have no active work, False otherwise
        """
        return self._central_tracker.is_rollout_worker_quiescent()

    def set_training_step(self, step: int) -> None:
        """
        Set the current training step in the central tracker

        Args:
            step: The current training step
        """
        self._central_tracker.set_training_step(step)

    def get_training_step(self) -> int:
        """
        Get the current training step from the central tracker

        Returns:
            int: The current training step
        """
        return self._central_tracker.get_training_step()

    def record_e2b_event(self, event: dict[str, Any]) -> None:
        self._central_tracker.record_e2b_event(event)

    def experiment_logger_post(self, backend: str, **kwargs) -> None:
        if backend == "wandb":
            assert "data" in kwargs, "data is required for wandb"
        elif backend == "feishu":
            assert "content" in kwargs, "content is required for feishu"
            assert "title" in kwargs, "title is required for feishu"
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self._central_tracker.experiment_logger_post(backend, **kwargs)

    def experiment_logger_flush(self, step: int) -> None:
        self._central_tracker.experiment_logger_flush(step)

    def get_progress_metrics(self) -> dict[str, Any]:
        """
        Get progress metrics from the central tracker

        Returns:
            dict: Progress metrics from all modules
        """
        return self._central_tracker.get_progress_metrics()
