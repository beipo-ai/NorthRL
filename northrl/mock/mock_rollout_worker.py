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
Mock Rollout Worker for testing purposes
"""

import asyncio
import copy
import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from ..base_module import NorthRLModule
from ..northrl_types import Trajectory
from ..rollout_worker.base_rollout_worker import BaseRolloutWorker
from .mock_inference_service_client import MockInferenceServiceClient

logger = logging.getLogger(__name__)


def _deterministic_reward(*parts: object) -> float:
    """
    Map inputs deterministically to a binary reward (0.0 or 1.0).
    Stable across processes/runs (unlike Python's built-in hash()).
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Use 1 bit -> deterministic binary reward.
    return float(digest[0] & 1)


class MockRolloutWorker(BaseRolloutWorker):
    """
    Mock Rollout Worker for testing purposes.
    Uses MockInferenceServiceClient instead of real inference service to avoid network dependencies.
    """

    def __init__(self, config: DictConfig):  # pylint: disable=super-init-not-called
        # Initialize NorthRLModule (grandparent)
        NorthRLModule.__init__(self)  # pylint: disable=non-parent-init-called

        # Initialize attributes from BaseRolloutWorker without calling its __init__
        # to avoid creating a real inference service client
        self._config = config
        self._stop_event = threading.Event()
        self._idle_wakeup = None
        self._task_future = None
        self._max_sequence_length = getattr(
            getattr(config, "data", None), "max_sequence_length", 4096
        )
        self._rollout_trace_dir = None

        # Use MockInferenceServiceClient instead of real client
        self._inference_client: MockInferenceServiceClient = MockInferenceServiceClient(config)  # type: ignore

        # Initialize other attributes that would be set by BaseRolloutWorker
        self._trajectory_pool = None  # type: ignore
        self._dataloader = None  # type: ignore
        self._weight_sync_controller = None  # type: ignore
        self._validate_dataloader = None  # type: ignore
        self._validator = None  # type: ignore
        self._next_task = None
        self._is_running_validate = False
        self._pool_capacity_waiting = False

        # Mock-specific attributes
        self._processed_count: int = 0
        self._mock_delay: float = 0.1

        # Complete trajectory loading from file
        self._trajectory_load_path = config.get("trajectory_load_path", None)
        self._trajectory_format = config.get("trajectory_format", "jsonl")  # "pt" or "jsonl"
        self._loaded_trajectories: list[Trajectory] = []
        self._traj_key_map: dict[tuple, Trajectory] = {}  # Map (group_id, run_id) -> trajectory
        self._trajectory_index: int = 0
        self._all_loaded_into_pool: bool = False  # True after bulk-loading all trajectories

        if self._trajectory_load_path:
            if not Path(self._trajectory_load_path).is_file():
                raise FileNotFoundError(self._trajectory_load_path)
            logger.info(f"Begin loading trajectories from {self._trajectory_load_path}")
            self._load_trajectories()
            logger.info(
                f"Finished loading trajectories from {self._trajectory_load_path}. Building traj_key_map..."
            )
            self._build_traj_key_map()
            logger.info(
                f"MockRolloutWorker initialized with {len(self._loaded_trajectories)} loaded trajectories from {self._trajectory_load_path} (format: {self._trajectory_format})"
            )

            # Warn if using multiple workers with trajectory loading (wasteful)
            num_workers = config.get("resource", {}).get("num_workers", 1)
            if num_workers > 1:
                logger.warning(
                    f"MockRolloutWorker is loading trajectories on {num_workers} workers. "
                    "This is wasteful as each worker loads the entire file. "
                    "Consider setting rollout_worker.resource.num_workers=1 for trajectory reuse."
                )
        else:
            logger.info(
                "MockRolloutWorker initialized with MockInferenceServiceClient (generating mock trajectories)"
            )

    def _load_trajectories(self) -> None:
        """Load trajectories from dump file.

        Supports two formats:
        - "pt": PyTorch tensor format (from DataDumper)
        - "jsonl": JSON Lines format (legacy)
        """
        if not self._trajectory_load_path:
            return

        try:
            if self._trajectory_format == "pt":
                self._load_trajectories_pt()
            elif self._trajectory_format == "jsonl":
                self._load_trajectories_jsonl()
            else:
                raise ValueError(f"Unsupported trajectory format: {self._trajectory_format}")
            if not self._loaded_trajectories:
                raise ValueError("Trajectory replay file contains no complete rollout results")
            logger.info(
                f"Loaded {len(self._loaded_trajectories)} trajectories from {self._trajectory_load_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load trajectories from {self._trajectory_load_path}: {e}")
            self._loaded_trajectories = []
            raise

    def _build_traj_key_map(self) -> None:
        """Build mapping from (group_id, run_id) to trajectory for multi-worker replay."""
        for traj in self._loaded_trajectories:
            group_id = traj.get("group_id", None)
            run_id = traj.get("run_id", None)
            if group_id is not None and run_id is not None:
                key = (str(group_id), int(run_id))
                self._traj_key_map[key] = traj
        if self._traj_key_map:
            logger.debug(
                f"Built traj_key_map with {len(self._traj_key_map)} entries: {list(self._traj_key_map.keys())}"
            )

    def _restore_dump_rows(self, rows: list[dict], format_version: int = 1) -> list[Trajectory]:
        """Read flat/optional-segment dumps and restore executions from legacy flat rows."""
        trajectories = []
        legacy: dict[tuple, Trajectory] = {}
        for raw_row in rows:
            row = self._convert_extra_fields(raw_row)
            extra = row.get("extra_fields", {})
            if "segments" in row or format_version >= 3:
                segments = row.get("segments")
                if segments is not None:
                    # Version 2 stored sequence metadata inside extra_fields.
                    segments = [
                        {
                            **segment.get("extra_fields", {}),
                            **{
                                key: value
                                for key, value in segment.items()
                                if key != "extra_fields"
                            },
                        }
                        for segment in segments
                    ]
                trajectories.append(
                    Trajectory(
                        tokens=row.get("tokens"),
                        loss_mask=row.get("loss_mask"),
                        segments=segments,
                        reward=row.get("reward"),
                        is_val=row.get("is_val", False),
                        status=row.get("status", "success"),
                        extra_fields=extra,
                    )
                )
                continue
            group_id, run_id = extra.get("group_id"), extra.get("run_id")
            if group_id is None or run_id is None:
                raise ValueError("Legacy flat dumps require group_id and run_id")
            key = (group_id, run_id)
            previous = legacy.get(key)
            if previous is None:
                previous = Trajectory(
                    tokens=row["tokens"],
                    loss_mask=row["loss_mask"],
                    reward=row["reward"],
                    is_val=row["is_val"],
                    extra_fields=extra.copy(),
                )
                legacy[key] = previous
                trajectories.append(previous)
                continue
            if previous.reward != row["reward"] or previous.is_val != row["is_val"]:
                raise ValueError("Legacy segments of a trajectory disagree on the final outcome")
            for field in ("rollout_id", "e2b_rollout_id"):
                if previous.get(field) != extra.get(field):
                    raise ValueError("Legacy run slot contains multiple rollout identities")
            if previous.segments is None:
                previous.segments = [
                    {
                        **previous.extra_fields,
                        "tokens": previous.tokens,
                        "loss_mask": previous.loss_mask,
                    }
                ]
                previous.tokens = previous.loss_mask = None
            previous.segments.append(
                {**extra, "tokens": row["tokens"], "loss_mask": row["loss_mask"]}
            )
        return trajectories

    def _load_trajectories_pt(self) -> None:
        """Load current trajectories or historical nested/flat .pt dumps."""
        data = torch.load(self._trajectory_load_path)
        self._loaded_trajectories.extend(
            self._restore_dump_rows(data.get("trajectories", []), data.get("format_version", 1))
        )

    def _load_trajectories_jsonl(self) -> None:
        """Load a file as one set so legacy executions can span JSONL lines."""
        rows = []
        format_version = None
        with open(self._trajectory_load_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    version = data.get("format_version", 1)
                    if format_version is not None and version != format_version:
                        raise ValueError("A replay file must use a single trajectory dump version")
                    format_version = version
                    rows.extend(data.get("trajectories", []))
        self._loaded_trajectories.extend(self._restore_dump_rows(rows, format_version or 1))

    def _convert_extra_fields(self, value: Any) -> Any:
        """Normalize tensors recursively, including metadata inside nested segment lists."""
        if isinstance(value, torch.Tensor):
            return value.item() if value.dim() == 0 else value.tolist()
        if isinstance(value, dict):
            return {key: self._convert_extra_fields(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._convert_extra_fields(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._convert_extra_fields(item) for item in value)
        return value

    async def execute_rollout(self, task: dict[str, Any]) -> str | None:
        """File replay owns its saved rollout identities, not the outer trigger task."""
        if self._loaded_trajectories:
            return await self.rollout(task)
        return await super().execute_rollout(task)

    def _replay_loaded_rollouts(self, task: dict[str, Any]) -> None:
        """Replay complete saved trajectories without adopting the trigger task's identity."""
        expected = task.get("expected_rollouts")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise ValueError("File replay requires the task's positive expected_rollouts")
        groups: dict[Any, dict[int, Trajectory]] = {}
        for trajectory in self._loaded_trajectories:
            group_id, run_id = trajectory.get("group_id"), trajectory.get("run_id")
            if (
                group_id is None
                or group_id == ""
                or isinstance(run_id, bool)
                or not isinstance(run_id, int)
            ):
                raise ValueError("File replay requires group_id and integer run_id")
            saved_expected = trajectory.get("expected_rollouts", expected)
            if (
                isinstance(saved_expected, bool)
                or not isinstance(saved_expected, int)
                or saved_expected != expected
            ):
                raise ValueError("Saved expected_rollouts does not match the replay task")
            group = groups.setdefault(group_id, {})
            if run_id in group:
                raise ValueError("File replay contains duplicate trajectories for one run slot")
            group[run_id] = trajectory
        if any(set(group) != set(range(expected)) for group in groups.values()):
            raise ValueError("File replay group does not contain all expected rollout slots")
        self._all_loaded_into_pool = True
        for group in groups.values():
            group_id = uuid.uuid4().hex
            for run_id, source in group.items():
                trajectory = copy.deepcopy(source)
                identity = {
                    "group_id": group_id,
                    "run_id": run_id,
                    "rollout_id": uuid.uuid4().hex,
                    "expected_rollouts": expected,
                    "model_tag": "actor",
                }
                trajectory.extra_fields.update(identity)
                for segment in trajectory.segments or []:
                    segment.update(identity)
                if self._put_trajectory(trajectory) != "success":
                    raise RuntimeError("Failed to replay a complete trajectory")
                self._processed_count += 1

    async def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Mock implementation of a single step operation.
        If trajectories are loaded from file, replay them; otherwise generate mock trajectories.

        Args:
            task: Task to process
        """
        # Simulate processing time
        await asyncio.sleep(self._mock_delay)

        if self._loaded_trajectories:
            if not self._all_loaded_into_pool:
                self._replay_loaded_rollouts(task)
            # Subsequent trigger tasks only drain the loader; replay has already
            # published every independent execution's complete trajectory.
            return None

        # Otherwise, generate mock trajectory
        # GRPO grouping fields: propagate from task (preferred), otherwise generate defaults.
        # - group_id: identifies the group of repeated rollouts for the same prompt
        # - run_id: identifies the sample index within the group
        # - uid: some configs / pools use uid as the grouping key
        mock_group_id = task.get("group_id") or task.get("uid")
        if not mock_group_id:
            # Deterministic fallback to keep mock runs reproducible.
            mock_group_id = (
                f"mock-{hashlib.sha256(task['prompt'].encode('utf-8')).hexdigest()[:16]}"
            )
        run_id = int(task.get("run_id", 0))
        task_id = task.get("task_id")
        uid = task.get("uid") or mock_group_id
        # Generate mock tokens and loss_mask (simple mock: 10 prompt tokens, 5 response tokens)
        prompt_tokens = list(range(100, 110))
        response_tokens = list(range(200, 205))
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        # Create one ordinary trajectory without a segment wrapper
        mock_trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=_deterministic_reward(task.get("is_val", False), run_id, task["prompt"]),
            is_val=task.get("is_val", False),
            extra_fields={
                # GRPO-required fields
                "group_id": mock_group_id,
                "run_id": run_id,
                "uid": uid,
                # Stable ordering key (preferred over uuid-like group_id)
                "task_id": task_id,
                "prompt": task["prompt"],
                "response": f"Mock response (group_id={mock_group_id}, run_id={run_id})",
                "old_log_probs": [0.0] * len(tokens),
            },
        )

        self._processed_count += 1
        logger.info(f"Mock rollout worker: Successfully processed step #{self._processed_count}")

        return self._put_trajectory(mock_trajectory)
