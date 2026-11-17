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

"""Run one task rollout in E2B and emit its NorthRL training segments."""

from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, ListConfig
from pydantic import JsonValue, TypeAdapter

from ..northrl_types import Trajectory
from ..remote_rollout.contracts import RemoteRolloutResult, RolloutInput
from ..remote_rollout.e2b_executor import PreparedE2BArtifacts, run_rollout_in_e2b
from ..remote_rollout.execution_budgets import ExecutionBudgets
from ..remote_rollout.execution_profiles import get_e2b_execution_configs
from ..utils.async_utils import await_coro
from ..utils.config_utils import get_algorithm_loss_mode, get_derived_sampling_flags
from ..utils.profiling import result_profile_block
from .base_rollout_worker import BaseRolloutWorker

if TYPE_CHECKING:
    from ..weaver.weaver_service_holder import WeaverServiceHolder

logger = logging.getLogger(__name__)

_MODEL_TAG = "actor"
_TASK_JSON_ADAPTER = TypeAdapter(dict[str, Any])


class E2BRolloutWorker(BaseRolloutWorker):
    """Bridge NorthRL rollout tasks to one-task E2B templates."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        if config.backend.type != "weaver":
            raise ValueError("E2BRolloutWorker requires backend.type='weaver'")
        if config.training.weight_sync.mode not in {"sync", "fully-async"}:
            raise ValueError("E2BRolloutWorker requires weight_sync.mode='sync' or 'fully-async'")
        if get_algorithm_loss_mode(config) == "score_centering":
            raise ValueError("NorthGate E2B rollouts do not support score_centering references")
        if get_derived_sampling_flags(config)["return_moe_topk_indices"]:
            raise ValueError("NorthGate E2B rollouts do not support R3 router replay references")

        self._uses_execution_profiles = "profiles" in config.rollout.execution
        self._execution_configs = get_e2b_execution_configs(config)
        self._timeouts_by_profile: dict[str, int] = {}
        self._input_fields_by_profile: dict[str, tuple[str, ...]] = {}
        for name, execution_config in self._execution_configs.items():
            execution = execution_config.rollout.execution
            budgets = ExecutionBudgets.from_config(execution_config)

            input_fields = execution.get("input_fields")
            if (
                not isinstance(input_fields, (list, ListConfig))
                or any(not isinstance(field, str) or not field.strip() for field in input_fields)
                or len(set(input_fields)) != len(input_fields)
            ):
                raise ValueError(
                    "rollout.execution.input_fields must be a list of unique, non-empty field names"
                )
            self._timeouts_by_profile[name] = budgets.agent_seconds
            self._input_fields_by_profile[name] = tuple(input_fields)
        self._service_holder: WeaverServiceHolder | None = None
        self._prepared_artifacts: PreparedE2BArtifacts | None = None
        self._prepared_artifacts_by_profile: dict[str, PreparedE2BArtifacts] = {}

    def set_prepared_artifacts(self, artifacts: PreparedE2BArtifacts) -> None:
        """Receive the controller's shared snapshot before starting rollouts."""

        self._prepared_artifacts = artifacts

    def set_prepared_artifacts_by_profile(self, artifacts: dict[str, PreparedE2BArtifacts]) -> None:
        """Receive independent source and evaluator snapshots for each profile."""

        if artifacts.keys() != self._execution_configs.keys():
            raise ValueError(
                "Prepared E2B artifact keys must match the configured execution profiles"
            )
        self._prepared_artifacts_by_profile = dict(artifacts)

    def init_inference_service_client(
        self, service_holder: WeaverServiceHolder | None = None
    ) -> None:
        """Keep the shared Weaver holder; inference itself runs inside E2B."""

        if service_holder is None:
            raise ValueError("E2BRolloutWorker requires a WeaverServiceHolder")
        self._service_holder = service_holder

    async def rollout(self, task: dict[str, Any]) -> str:
        profile_name = ""
        if self._uses_execution_profiles:
            task_profile = task.get("rollout_profile")
            if not isinstance(task_profile, str) or task_profile not in self._execution_configs:
                raise ValueError(
                    "E2B rollout task.rollout_profile must name a configured execution profile"
                )
            profile_name = task_profile
        execution_config = self._execution_configs[profile_name]
        template = task.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError("E2B rollout task requires a non-empty template")
        if self._service_holder is None:
            raise RuntimeError("E2BRolloutWorker has not received a WeaverServiceHolder")

        gateway = await await_coro(
            self._service_holder.get_current_gateway_snapshot(
                timeout_seconds=execution_config.rollout.execution.get(
                    "gateway_publication_timeout_seconds", 300
                )
            )
        )
        weight_version = gateway["weight_version"]
        # Each task recipe chooses its inputs; task hooks interpret them.
        # Normalize selected fields only, leaving all other dataset columns local.
        task_input: dict[str, JsonValue] = json.loads(
            _TASK_JSON_ADAPTER.dump_json(
                {
                    key: task[key]
                    for key in self._input_fields_by_profile[profile_name]
                    if key in task
                }
            )
        )
        metadata: dict[str, JsonValue] = {
            key: task[key]
            for key in ("task_id", "group_id", "run_id", "expected_rollouts", "rollout_profile")
            if key in task
        }
        metadata["weight_version"] = weight_version
        if self._is_running_validate:
            metadata["is_validation"] = True
        request = RolloutInput(
            rollout_id=task.get("rollout_id") or uuid.uuid4().hex,
            task=task_input,
            metadata=metadata,
        )
        trace_output_path = (
            Path(self._rollout_trace_dir) / f"e2b_rollout_{request.rollout_id}.json.gz"
            if self._rollout_trace_dir is not None
            else None
        )

        live_options: dict[str, Any] = {}
        if self._activity_tracker is not None:
            live_options["event_callback"] = self._activity_tracker.record_e2b_event
        prepared_artifacts = self._prepared_artifacts_by_profile.get(profile_name)
        if not self._uses_execution_profiles and prepared_artifacts is None:
            prepared_artifacts = self._prepared_artifacts
        if prepared_artifacts is not None:
            live_options["prepared_artifacts"] = prepared_artifacts

        result = await run_rollout_in_e2b(
            request,
            execution_config,
            gateway=gateway,
            template=template,
            timeout_seconds=self._timeouts_by_profile[profile_name],
            trace_output_path=trace_output_path,
            **live_options,
        )

        with result_profile_block("driver.result-postprocess", rollout_id=result.rollout_id):
            trajectory = self._trajectory_from_result(result, task, weight_version)
        return self._put_trajectory(trajectory)

    def _trajectory_from_result(
        self, result: RemoteRolloutResult, task: dict[str, Any], weight_version: str
    ) -> Trajectory:
        """Build training data in Driver after the remote result has been restored."""

        from northrl.remote_rollout.metrics import runtime_metrics_from_snapshot

        runtime_metrics = runtime_metrics_from_snapshot(result.metrics)
        runtime_metrics["rollout_timed_out"] = result.metrics.get("e2b_timed_out") == 1.0
        for source, target in (
            ("sandbox_retries", "sandbox_retries"),
            ("rollout_started_at", "started_at"),
            ("rollout_finished_at", "finished_at"),
            ("rollout_duration_seconds", "duration_seconds"),
        ):
            if source in result.metrics:
                runtime_metrics[target] = result.metrics[source]

        # Infrastructure failures are terminal rollout results too. Discard their
        # partial data, but publish the failed parent so the GRPO group can finish.
        env_error = result.status == "env_error"
        task_failed = result.status == "task_failed"
        remote_segments = [] if env_error else result.segments
        if env_error:
            logger.warning(
                "E2B rollout %s failed with an environment error: %s: %s",
                result.rollout_id,
                result.error_type,
                result.error_message,
            )
        oversized_segments = [
            index
            for index, remote in enumerate(remote_segments)
            if len(remote.tokens) > self._max_sequence_length
        ]
        if oversized_segments:
            logger.warning(
                "E2B rollout %s contains segments exceeding "
                "data.max_sequence_length=%d; truncating segment_indices=%s "
                "and resetting the shared reward from %s to 0.0",
                result.rollout_id,
                self._max_sequence_length,
                oversized_segments,
                result.reward,
            )

        # The evaluator produces one reward for the complete E2B run, shared by
        # every recorded segment. Once any segment is truncated that
        # reward no longer describes the emitted training data, so fail closed
        # for the entire run instead of retaining the evaluator's reward.
        reward = None if env_error else (0.0 if oversized_segments else result.reward)
        timed_out = result.metrics.get("e2b_timed_out") == 1.0
        context_exhausted = result.termination_reason == "context_length_exceeded"
        # Request counters, histograms, and gauges are aggregated by the live
        # tracker. Trajectory scores are averaged per rollout, which would give
        # these request metrics a different meaning. Keep the source metrics
        # intact for trace recovery and live telemetry.
        score = {
            **{
                key: value
                for key, value in result.metrics.items()
                if not key.startswith(("model_request_", "model_requests_"))
            },
            # Emit an explicit zero on normal runs so this legacy score metric
            # remains meaningful instead of becoming 1.0 whenever any timeout
            # is present in a batch.
            "e2b_timed_out": float(timed_out),
            "e2b_task_failed": float(task_failed),
            "e2b_env_error": float(env_error),
            "reward": reward,
        }
        if env_error:
            # Infrastructure failures have no evaluated outcome; a numeric zero
            # would incorrectly enter rollout reward/score aggregates.
            score.pop("reward", None)
            score.pop("swebench_resolved", None)
        if oversized_segments and "swebench_resolved" in score:
            # This worker's evaluator uses swebench_resolved as its task score.
            # The emitted, truncated training data must not retain that score.
            score["swebench_resolved"] = 0.0

        if env_error and "tool_results" in runtime_metrics:
            runtime_metrics["tool_unassigned"] = runtime_metrics["tool_results"]
        segments = []
        for segment_index, remote in enumerate(remote_segments):
            tokens, loss_mask, old_log_probs, is_truncated = self._check_and_truncate(
                remote.tokens,
                remote.loss_mask,
                remote.old_log_probs,
            )
            if is_truncated and "tool_unassigned" in runtime_metrics:
                runtime_metrics["tool_unassigned"] += remote.segment_metrics.get("tool_results", 0)
            sampling_mask = remote.sampling_mask
            if is_truncated and sampling_mask is not None:
                # sampling_mask is response-aligned rather than token-aligned.
                from northrl.utils.sampling_mask_refs import trim_sampling_mask

                sampling_mask = trim_sampling_mask(sampling_mask, sum(loss_mask))

            if is_truncated or context_exhausted:
                finish_reason = "length"
            elif timed_out:
                finish_reason = "timeout"
            elif task_failed:
                finish_reason = "error"
            else:
                finish_reason = "stop"

            turn_annotations = []
            for annotation in remote.turn_annotations:
                start, stop = cast(list[int], annotation["response_span"])
                if start >= len(tokens):
                    continue
                copied = deepcopy(annotation)
                if stop > len(tokens):
                    copied["response_span"] = [start, len(tokens)]
                    copied["truncated"] = True
                turn_annotations.append(copied)

            segments.append(
                {
                    "segment_index": segment_index,
                    "segment_metrics": {} if is_truncated else dict(remote.segment_metrics),
                    "tokens": tokens,
                    "loss_mask": loss_mask,
                    "finish_reason": finish_reason,
                    "is_truncated": is_truncated,
                    "sampling_mask": sampling_mask,
                    "old_log_probs": old_log_probs,
                    "turn_annotations": turn_annotations,
                }
            )

        # A scored task failure is a valid zero-reward sample. An evaluated
        # rollout can also have zero segments; its reward still participates
        # once in the complete group's GRPO statistics.
        trajectory = Trajectory(
            segments=segments,
            reward=reward,
            is_val=task.get("is_val", False),
            status="failed" if env_error else "success",
            extra_fields={
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
                "rollout_id": result.rollout_id,
                "expected_rollouts": task.get("expected_rollouts", 1),
                "data_source": task.get("data_source", task.get("dataset_name", "")),
                **(
                    {"rollout_profile": task["rollout_profile"]}
                    if "rollout_profile" in task
                    else {}
                ),
                "temperature": self._config.rollout.sampling.temperature,
                "e2b_rollout_id": result.rollout_id,
                "runtime_metrics": runtime_metrics,
                "e2b_timed_out": timed_out,
                "e2b_status": result.status,
                "e2b_termination_reason": result.termination_reason,
                "e2b_length_details": result.length_details,
                "e2b_error_type": result.error_type,
                "e2b_error_message": result.error_message,
                "model_tag": _MODEL_TAG,
                "score": score,
                "weight_version": weight_version,
            },
        )
        return trajectory
