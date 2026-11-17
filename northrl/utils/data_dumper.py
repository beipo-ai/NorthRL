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
Data Dumper for collecting training debug data.

This module provides a unified interface for dumping training data including:

Remote API Mode:
- Prepared trajectories (after GRPO advantage computation)
- Datums (converted data for Weaver service)
- Training metrics (returned from Weaver forward_backward)
"""

import json
import logging
import threading
from array import array
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from northrl.utils.trajectory_sequences import iter_trajectory_sequences

logger = logging.getLogger(__name__)

# Exact types keep subclasses with custom ``tolist``/``item`` methods on the
# conversion path. These built-in values are already safe to serialize.
_SERIALIZABLE_SCALAR_TYPES = frozenset((int, float, bool, str, type(None)))


def _json_array(value):
    """Materialize one packed field at a time while streaming JSON to disk."""
    if isinstance(value, (array, torch.Tensor)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DataDumper:
    """
    Thread-safe data dumper for training debug data.

    Supports both Trainer-side (single process) and Worker-side (multi-process) dumping.
    Files are organized by step and worker rank.
    """

    _instances: dict[str, "DataDumper"] = {}
    _lock = threading.Lock()

    def __init__(self, config: DictConfig | dict | None = None, rank: int = 0):
        """
        Initialize the DataDumper.

        Args:
            config: Configuration containing debug settings
            rank: Worker rank (0 for Trainer-side, actual rank for Worker-side)
        """
        self._rank = rank
        self._write_lock = threading.Lock()

        # Extract debug config
        if config is None:
            self._enabled = False
            self._dump_dir = None
            self._remote_api_options: dict[str, bool] = {}
            self._format = "pt"
            self._dump_every_n = 1
            return

        # Handle both DictConfig and dict.
        # NOTE: In NorthRL, modules like Trainer are often instantiated with a *sub-config*
        # (e.g., config.trainer). In that case, `runtime.debug` lives at the root level
        # and is accessible via OmegaConf parent chain. We try to resolve that here so
        # callers don't need to re-plumb root config everywhere.
        def _read_runtime_debug(cfg: Any) -> Any:
            if cfg is None:
                return {}
            if isinstance(cfg, DictConfig):
                return (cfg.get("runtime", {}) or {}).get("debug", {}) or {}
            return (cfg.get("runtime", {}) or {}).get("debug", {}) or {}

        debug_config: Any = _read_runtime_debug(config)
        if not debug_config and isinstance(config, DictConfig):
            parent = getattr(config, "_parent", None)
            if parent is not None:
                try:
                    debug_config = _read_runtime_debug(parent)
                except Exception:  # pylint: disable=broad-exception-caught
                    debug_config = {}

        # Resolve interpolations if any (e.g., ${oc.env:EXPERIMENT_PATH,...})
        if isinstance(debug_config, DictConfig):
            debug_config = OmegaConf.to_container(debug_config, resolve=True) or {}

        self._enabled = debug_config.get("enable_data_dump", False)

        if not self._enabled:
            self._dump_dir = None
            self._remote_api_options = {}
            self._format = "pt"
            self._dump_every_n = 1
            return

        # Set up dump directory
        dump_dir = debug_config.get("dump_dir", "debug_dump")
        self._dump_dir = Path(dump_dir)
        self._dump_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for remote API debug dumps.
        subdirs = ["prepared_trajectories", "datums", "training_metrics"]
        for subdir in subdirs:
            (self._dump_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Remote API specific options
        self._remote_api_options = debug_config.get("remote_api_dump_options", {})

        self._format = debug_config.get("dump_format", "pt")
        self._dump_every_n = debug_config.get("dump_every_n_steps", 1)

        logger.info(
            f"DataDumper initialized: enabled={self._enabled}, "
            f"dump_dir={self._dump_dir}, format={self._format}, "
            f"dump_every_n={self._dump_every_n}, rank={self._rank}"
        )

    @classmethod
    def get_instance(
        cls, config: DictConfig | dict | None = None, rank: int = 0, key: str = "default"
    ) -> "DataDumper":
        """
        Get or create a DataDumper instance (singleton per key).

        Args:
            config: Configuration containing debug settings
            rank: Worker rank
            key: Instance key for different contexts

        Returns:
            DataDumper instance
        """
        instance_key = f"{key}_{rank}"
        with cls._lock:
            if instance_key not in cls._instances:
                cls._instances[instance_key] = cls(config, rank)
            return cls._instances[instance_key]

    @classmethod
    def reset_instances(cls):
        """Reset all instances (for testing)."""
        with cls._lock:
            cls._instances.clear()

    @property
    def enabled(self) -> bool:
        """Check if dumping is enabled."""
        return self._enabled

    def should_dump(self, data_type: str, step: int) -> bool:
        """
        Check if data should be dumped for the given type and step.

        Args:
            data_type: Type of data
                Remote API: "prepared_trajectories", "datums", "training_metrics"
            step: Current training step

        Returns:
            True if data should be dumped
        """
        if not self._enabled:
            return False
        if step % self._dump_every_n != 0:
            return False

        return self._remote_api_options.get(data_type, False)

    # =========================================================================
    # Remote API Mode Dump Methods
    # =========================================================================

    def dump_prepared_trajectories(
        self,
        step: int,
        trajectories: list,
        advantages: list[float] | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump complete trajectories and their nested training segments.

        Args:
            step: Current training step
            trajectories: List of complete Trajectory objects with computed advantages
            advantages: Optional list of computed advantages
            extra_info: Optional additional information
        """
        if not self.should_dump("prepared_trajectories", step):
            return

        sequence_count = sum(
            sum(1 for _ in iter_trajectory_sequences(trajectory)) for trajectory in trajectories
        )
        data = {
            "step": step,
            "format_version": 3,
            "num_sequences": sequence_count,
            "num_segments": sequence_count,  # Legacy sequence-count name
            "num_trajectories": len(trajectories),
            "trajectories": [self._serialize_trajectory(t) for t in trajectories],
        }
        if advantages is not None:
            data["advantages"] = advantages
        if extra_info:
            data["extra_info"] = extra_info

        self._write_file("prepared_trajectories", f"step_{step:06d}", data)
        logger.info(f"Dumped {len(trajectories)} prepared trajectories for step {step}")

    def dump_rollout_round(
        self, step: int, completed_step: int, records: list[dict[str, Any]]
    ) -> None:
        """Save the parent-result index alongside this batch's prepared dump.

        The prepared-trajectories option and frequency also control this index.
        ``step`` matches the prepared/datums filename; ``completed_step`` is the
        completed-round counter used by experiment logging.
        """
        if not self.should_dump("prepared_trajectories", step) or self._dump_dir is None:
            return

        payload = {"step": step, "training_step": completed_step, "rollouts": records}
        with self._write_lock:
            directory = self._dump_dir / "rollout_rounds"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"step_{step:06d}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    self._to_json_serializable(payload),
                    indent=2,
                    default=lambda value: (
                        value.tolist() if hasattr(value, "tolist") else str(value)
                    ),
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

    def dump_datums(
        self,
        step: int,
        datums_data: list[dict[str, Any]],
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump converted datums for Weaver service (remote_api mode).

        Args:
            step: Current training step
            datums_data: List of datum dictionaries to be sent to Weaver
            extra_info: Optional additional information
        """
        if not self.should_dump("datums", step):
            return

        data = {
            "step": step,
            "num_datums": len(datums_data),
            "datums": [self._serialize_datum(d) for d in datums_data],
        }
        if extra_info:
            data["extra_info"] = extra_info

        self._write_file("datums", f"step_{step:06d}", data)
        logger.info(f"Dumped {len(datums_data)} datums for step {step}")

    def dump_training_metrics(
        self,
        step: int,
        metrics: dict[str, Any],
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump training metrics from Weaver forward_backward response (remote_api mode).

        Args:
            step: Current training step
            metrics: Metrics dictionary returned from Weaver
            extra_info: Optional additional information
        """
        if not self.should_dump("training_metrics", step):
            return

        data = {
            "step": step,
            "metrics": self._convert_metrics(metrics),
        }
        if extra_info:
            data["extra_info"] = extra_info

        self._write_file("training_metrics", f"step_{step:06d}", data)
        logger.info(f"Dumped training metrics for step {step}: {list(metrics.keys())}")

    def _serialize_datum(self, datum: dict[str, Any]) -> dict[str, Any]:
        """
        Serialize a datum dictionary, converting tensors to serializable format.

        Args:
            datum: Datum dictionary

        Returns:
            Serialized dictionary
        """
        result: dict[str, Any] = {}
        for k, v in datum.items():
            if isinstance(v, dict):
                result[k] = self._serialize_datum(v)
            elif isinstance(v, (list, tuple)):
                result[k] = self._serialize_sequence(v)
            else:
                result[k] = self._to_serializable(v)
        return result

    def _serialize_sequence(self, values: list | tuple) -> list:
        """Copy dense scalar lists without dispatching every token recursively."""
        return [
            value if type(value) in _SERIALIZABLE_SCALAR_TYPES else self._to_serializable(value)
            for value in values
        ]

    def _convert_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Convert metrics dictionary to serializable format.

        Args:
            metrics: Metrics dictionary

        Returns:
            Serializable metrics dictionary
        """
        result: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    result[k] = float(v.item())
                else:
                    result[k] = v.detach().cpu().tolist()
            elif isinstance(v, dict):
                result[k] = self._convert_metrics(v)
            elif isinstance(v, (float, int, str, bool, type(None))):
                result[k] = v
            else:
                try:
                    result[k] = float(v)
                except (TypeError, ValueError):
                    result[k] = str(v)
        return result

    def _write_file(self, subdir: str, filename: str, data: dict[str, Any]) -> None:
        """
        Write data to file in the specified format.

        Args:
            subdir: Subdirectory name
            filename: Base filename (without extension)
            data: Data dictionary to write
        """
        if self._dump_dir is None:
            return

        with self._write_lock:
            if self._format == "pt":
                filepath = self._dump_dir / subdir / f"{filename}.pt"
                torch.save(data, filepath)
            elif self._format == "jsonl":
                filepath = self._dump_dir / subdir / f"{filename}.jsonl"
                # Convert tensors to lists for JSON serialization
                json_data = self._to_json_serializable(data)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, default=_json_array)
                    f.write("\n")
            else:
                raise ValueError(f"Unsupported dump format: {self._format}")

    # Extra-field keys that the train() function actually consumes.
    # Everything else (ground_truth, finish_reason, prompt, response, …)
    # is only useful at rollout time and would waste dump size + memory on reload.
    _TRAIN_RELEVANT_EXTRA_FIELDS: set[str] = {
        # Training denominator / constrained action space
        "old_log_probs",
        "sampling_mask",
        # GRPO grouping / ordering
        "group_id",
        "uid",
        "run_id",
        "task_id",
        "rollout_id",
        "resampled_from_rollout_id",
        "expected_rollouts",
        "model_tag",
        "group_skipped",
        "e2b_rollout_id",
        "e2b_timed_out",
        "is_truncated",
        "finish_reason",
        "weight_version",
        "advantage",
        "token_advantages",
        "base_token_advantages",
        "turn_annotations",
        "ref_logprobs",
        "correction_weights",
        "correction_mask",
        "rollout_logprobs_valid",
        "sft_batch_id",
        # Metrics logging
        "score",
        # Batch metadata (extracted by Batch.from_trajectories)
        "temperature",
        # router-replay: opaque ref to the server-offloaded MoE routing shard
        "moe_topk_indices_ref",
        # Managed SFT contains only safe public identity/batching metadata.
        "sample_ref",
        "datum_id",
        "input_token_count",
        "content_visibility",
        "loss_fn_inputs",
    }

    def _serialize_trajectory(self, traj) -> dict[str, Any]:
        """Keep ordinary trajectories flat and optional segments as flat dictionaries."""

        def extras(fields: dict) -> dict:
            return {
                key: self._to_serializable(value)
                for key, value in fields.items()
                if key in self._TRAIN_RELEVANT_EXTRA_FIELDS
            }

        result = {
            "reward": self._to_serializable(traj.reward),
            "is_val": bool(traj.is_val),
            "status": traj.status,
            "extra_fields": extras(traj.extra_fields),
        }
        if traj.segments is None:
            result["tokens"] = self._to_serializable(traj.tokens)
            result["loss_mask"] = self._to_serializable(traj.loss_mask)
        else:
            result["segments"] = [
                {
                    "tokens": self._to_serializable(segment["tokens"]),
                    "loss_mask": self._to_serializable(segment["loss_mask"]),
                    **extras(segment),
                }
                for segment in traj.segments
            ]
        return result

    def _to_serializable(self, v: Any) -> Any:
        """
        Convert a value to a serializable format.

        Args:
            v: Value to convert

        Returns:
            Serializable value
        """
        if type(v) in _SERIALIZABLE_SCALAR_TYPES:
            return v
        if isinstance(v, array):
            if self._format == "jsonl":
                return v
            dtype = {"q": torch.int64, "b": torch.int8, "d": torch.float64}.get(v.typecode)
            if dtype is not None:
                # Share packed storage instead of boxing every scalar in the batch.
                # Tensors also load with weights_only=True in the existing replay path.
                return torch.frombuffer(v, dtype=dtype) if v else torch.empty(0, dtype=dtype)
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        if hasattr(v, "tolist"):
            return self._to_serializable(v.tolist())
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (list, tuple)):
            return self._serialize_sequence(v)
        if isinstance(v, dict):
            return {k: self._to_serializable(val) for k, val in v.items()}
        return v

    def _to_json_serializable(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert all values in a dictionary to JSON-serializable format.

        Args:
            data: Dictionary with potentially non-serializable values

        Returns:
            Dictionary with JSON-serializable values
        """
        return {key: self._to_serializable(value) for key, value in data.items()}


def get_data_dumper(
    config: DictConfig | dict | None = None, rank: int = 0, key: str = "default"
) -> DataDumper:
    """
    Get or create a DataDumper instance.

    Convenience function for getting a DataDumper instance.

    Args:
        config: Configuration containing debug settings
        rank: Worker rank
        key: Instance key for different contexts

    Returns:
        DataDumper instance
    """
    return DataDumper.get_instance(config, rank, key)
