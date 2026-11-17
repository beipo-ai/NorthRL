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
Type definitions for NorthRL framework
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

# Type aliases for better readability
ModelTag = str  # Type alias for model tags

_MAX_WEAVER_DATUM_ID_LENGTH = 255


@dataclass(init=False)
class Trajectory:
    """A complete rollout with direct sequence fields or nested sequence dicts.

    Ordinary trajectories store tokens and loss_mask directly, preserving the
    original constructor and dictionary interface. Multi-sequence executions
    instead supply ``segments``: flat dictionaries holding tokens, loss_mask,
    and sequence metadata. Reward, status and execution identity belong to this
    parent object in either form. A result without sequences may use segments=[].

    Internal managed-SFT carriers are the one token-free exception: they set
    ``extra_fields["sample_ref"]`` (plus ``input_token_count`` and ``datum_id``)
    and carry neither tokens nor segments, because Weaver materializes the
    server-private tokens and loss mask. See ``is_managed_sample_ref``.
    """

    tokens: list[int] | None
    loss_mask: list[int] | None
    reward: float | None
    is_val: bool
    extra_fields: dict[str, Any]
    segments: list[dict[str, Any]] | None
    status: str

    def __init__(
        self,
        tokens: list[int] | None = None,
        loss_mask: list[int] | None = None,
        reward: float | None = None,
        is_val: bool = False,
        extra_fields: dict[str, Any] | None = None,
        *,
        segments: list[dict[str, Any]] | None = None,
        status: str = "success",
    ) -> None:
        if segments is not None and (tokens is not None or loss_mask is not None):
            raise ValueError("Pass either segments or tokens/loss_mask, not both")
        if (tokens is None) != (loss_mask is None):
            raise ValueError("tokens and loss_mask must be provided together")
        if segments is not None:
            for sequence in segments:
                if not isinstance(sequence, dict):
                    raise TypeError("segments must contain flat sequence dictionaries")
                if "tokens" not in sequence or "loss_mask" not in sequence:
                    raise ValueError("Each sequence dictionary requires tokens and loss_mask")
        self.tokens = tokens
        self.loss_mask = loss_mask
        self.reward = reward
        self.is_val = is_val
        self.extra_fields = extra_fields if extra_fields is not None else {}
        self.segments = segments
        self.status = status
        self._validate_datum_identity()

    def _validate_datum_identity(self) -> None:
        """Enforce token-vs-SampleRef XOR for the internal SFT carrier."""

        sample_ref = self.extra_fields.get("sample_ref")
        datum_id = self.extra_fields.get("datum_id")
        if datum_id is not None:
            if not isinstance(datum_id, str) or not datum_id.strip():
                raise ValueError("datum_id must be a non-empty string when provided")
            datum_id = datum_id.strip()
            if len(datum_id) > _MAX_WEAVER_DATUM_ID_LENGTH:
                raise ValueError(
                    "datum_id must be at most " f"{_MAX_WEAVER_DATUM_ID_LENGTH} characters"
                )
            self.extra_fields["datum_id"] = datum_id
        if sample_ref is None:
            return

        if self.tokens is not None or self.loss_mask is not None or self.segments is not None:
            raise ValueError(
                "A managed SampleRef trajectory cannot carry tokens, loss_mask or segments"
            )
        if not isinstance(sample_ref, Mapping):
            raise TypeError("Managed SampleRef trajectory requires a mapping sample_ref")
        expected_fields = {"dataset", "version", "sample_idx"}
        if set(sample_ref) != expected_fields:
            raise ValueError(
                "Managed SampleRef trajectory must contain exactly dataset, version, sample_idx"
            )
        for field_name in ("dataset", "version"):
            field_value = sample_ref[field_name]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"Managed SampleRef {field_name} must be a non-empty string")
        sample_idx = sample_ref["sample_idx"]
        if not isinstance(sample_idx, int) or isinstance(sample_idx, bool) or sample_idx < 0:
            raise ValueError("Managed SampleRef sample_idx must be a non-negative integer")
        if not isinstance(self.extra_fields.get("input_token_count"), int) or isinstance(
            self.extra_fields.get("input_token_count"), bool
        ):
            raise ValueError("Managed SampleRef trajectory requires integer input_token_count")
        if self.extra_fields["input_token_count"] <= 0:
            raise ValueError("Managed SampleRef input_token_count must be positive")
        if datum_id is None:
            raise ValueError("Managed SampleRef trajectory requires datum_id")
        loss_fn_inputs = self.extra_fields.get("loss_fn_inputs", {})
        if not isinstance(loss_fn_inputs, Mapping):
            raise TypeError("Managed SampleRef loss_fn_inputs must be a mapping")

    @property
    def is_managed_sample_ref(self) -> bool:
        """Whether this is the token-free internal carrier for a SampleRef Datum."""

        return self.extra_fields.get("sample_ref") is not None

    def __getitem__(self, key: str) -> Any:
        if key in ("tokens", "loss_mask", "reward", "is_val", "segments", "status", "extra_fields"):
            return getattr(self, key)
        return self.extra_fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in ("tokens", "loss_mask", "reward", "is_val", "segments", "status", "extra_fields"):
            if key == "segments" and value is not None:
                if self.tokens is not None or self.loss_mask is not None:
                    raise ValueError("Pass either segments or tokens/loss_mask, not both")
            elif key in ("tokens", "loss_mask") and value is not None and self.segments is not None:
                raise ValueError("Pass either segments or tokens/loss_mask, not both")
            setattr(self, key, value)
        else:
            self.extra_fields[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    "tokens",
                    "loss_mask",
                    "reward",
                    "is_val",
                    "segments",
                    "status",
                    *self.extra_fields,
                ]
            )
        )

    def items(self) -> list[tuple[str, Any]]:
        # This implements items(); calling self.items() here would recurse.
        # pylint: disable=consider-using-dict-items
        return [(key, self[key]) for key in self.keys()]

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore historical flat pickles without wrapping their sequence."""
        Trajectory.__init__(
            self,
            tokens=state.get("tokens"),
            loss_mask=state.get("loss_mask"),
            segments=state.get("segments"),
            reward=state.get("reward"),
            is_val=state.get("is_val", False),
            status=state.get("status", "success"),
            extra_fields=state.get("extra_fields"),
        )


class NorthRLRole(Enum):
    """Identify local module roles for controller initialization and activity tracking."""

    ROLLOUT_WORKER = "rollout_worker"
    TRAINER = "trainer"
    TRAJECTORY_POOL = "trajectory_pool"
    WEIGHT_SYNC_CONTROLLER = "weight_sync_controller"
    DATA_LOADER = "data_loader"
    VALIDATE_DATALOADER = "validate_dataloader"
    VALIDATOR = "validator"


@dataclass
class Batch:
    """A field-based row container; batch_size counts current rows, including padding."""

    values: dict[
        str, Any
    ]  # Tensor or non-Tensor iterable data, length should be metadata['batch_size']
    metadata: dict[str, Any]  # Metadata about the batch
    pad_size: int = 0

    def __len__(self) -> int:
        assert "batch_size" in self.metadata, "batch_size must be in metadata"
        return self.metadata["batch_size"]

    def copy(self) -> "Batch":
        return Batch(self.values.copy(), self.metadata.copy())

    def reorder(self, perm: list[int] | torch.Tensor) -> "Batch":
        """
        Reorder per-sample entries in this batch using a permutation.

        This reorders:
        - Tensor values shaped like (batch_size, ...)
        - list / numpy.ndarray values with length/first-dim == batch_size
        - metadata entries that look per-sample (list/ndarray with length/first-dim == batch_size)
        """
        bsz = len(self)
        if isinstance(perm, torch.Tensor):
            perm_list = perm.detach().cpu().tolist()
        else:
            perm_list = list(perm)

        if len(perm_list) != bsz:
            raise ValueError(f"Permutation length {len(perm_list)} != batch_size {bsz}")

        def _reorder_value(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                if v.dim() >= 1 and v.shape[0] == bsz:
                    return v[perm_list]
                return v
            if isinstance(v, list):
                if len(v) == bsz:
                    return [v[i] for i in perm_list]
                return v
            if isinstance(v, np.ndarray):
                if v.shape and v.shape[0] == bsz:
                    return v[perm_list]
                return v
            return v

        new_values = {k: _reorder_value(v) for k, v in self.values.items()}
        new_metadata = {k: _reorder_value(v) for k, v in self.metadata.items()}
        return Batch(values=new_values, metadata=new_metadata)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Batch to a single dictionary containing both values and metadata.
        If there are overlapping keys, metadata keys will overwrite values keys.
        """
        ret = {**self.values, **self.metadata}
        return ret

    def pad_to_world_size(self, world_size: int) -> "Batch":
        """Pad the batch to be divisible by world_size using zero-padding."""
        batch_size = len(self)
        if batch_size % world_size != 0:
            pad_size = world_size - (batch_size % world_size)
            self.pad_size = pad_size
            for key, value in self.values.items():
                if isinstance(value, torch.Tensor):
                    pad_shape = (pad_size,) + value.shape[1:]
                    self.values[key] = torch.cat(
                        [value, torch.zeros(pad_shape, dtype=value.dtype, device=value.device)],
                        dim=0,
                    )
                else:
                    if isinstance(value, list):
                        # Zero-pad with None for non-tensor lists
                        self.values[key] = value + [None] * pad_size
                    else:
                        value_array = np.asarray(value, dtype=object)
                        padding = np.zeros(pad_size, dtype=object)
                        self.values[key] = np.concatenate([value_array, padding], axis=0)
            self.metadata["batch_size"] = batch_size + pad_size
        return self

    def unpad(self) -> "Batch":
        """Remove zero-padding added by pad_to_world_size().

        Returns self unchanged if pad_size == 0.
        """
        if self.pad_size == 0:
            return self
        real_size = len(self) - self.pad_size
        for key, value in self.values.items():
            if isinstance(value, torch.Tensor):
                self.values[key] = value[:real_size]
            elif isinstance(value, list):
                self.values[key] = value[:real_size]
            elif isinstance(value, np.ndarray):
                self.values[key] = value[:real_size]
        self.metadata["batch_size"] = real_size
        self.pad_size = 0
        return self

    @classmethod
    def from_trajectories(
        cls, trajectories: list[Trajectory], model_tag: "ModelTag | None" = None
    ) -> "Batch":
        """
        Create one batch row per complete trajectory, preserving nested sequences.

        Args:
            trajectories: List of Trajectory dataclass instances
            model_tag: Optional model tag for the batch metadata

        Returns:
            A Batch object containing the trajectories
        """
        if not trajectories:
            raise ValueError("Cannot create batch from empty trajectories")

        first_keys = set(trajectories[0].keys())
        same_schema = all(set(trajectory.keys()) == first_keys for trajectory in trajectories[1:])
        values: dict[str, Any]
        if same_schema:
            # Preserve the historical flat columns for uniform trajectory fields.
            values = {
                key: [trajectory[key] for trajectory in trajectories]
                for key in trajectories[0].keys()
            }
        else:
            # Ordinary and multi-sequence parents may own different metadata.
            # Keep each row's dictionary, rather than inventing absent None keys.
            values = {
                key: [getattr(trajectory, key) for trajectory in trajectories]
                for key in ("tokens", "loss_mask", "segments", "reward", "is_val", "status")
            }
            values["extra_fields"] = [trajectory.extra_fields.copy() for trajectory in trajectories]

        # Merge tensors into 2D tensors
        for key, value_list in values.items():
            if value_list and all(isinstance(value, torch.Tensor) for value in value_list):
                if all(t.dim() == 1 for t in value_list):
                    # All tensors are 1D, stack them to create 2D tensor
                    values[key] = torch.stack(value_list, dim=0)
                elif all(t.dim() == 2 for t in value_list):
                    # All tensors are 2D, stack them to create 3D tensor, then reshape to 2D
                    stacked = torch.stack(value_list, dim=0)
                    batch_size, _, _ = stacked.shape
                    values[key] = stacked.view(batch_size, -1)
                else:
                    # Mixed dimensions or higher dimensions, raise error
                    raise ValueError(f"Tensors for key '{key}' have inconsistent dimensions")

        # Determine model_tag from trajectories if not provided
        if model_tag is None:
            model_tag = trajectories[0].get("model_tag", "default")

        # Add batch_size to the metadata
        metadata = {
            "batch_size": len(trajectories),
            "model_tag": model_tag,
            "temperature": trajectories[0].get("temperature", 1.0),
        }

        return cls(values=values, metadata=metadata)

    def to_trajectories(self) -> list[Trajectory]:
        """Restore one complete trajectory per row, retaining nested sequences.

        Call unpad() before conversion. Generic batches lacking the trajectory
        fields cannot be converted; conversion must not invent rollout results.
        """
        if self.pad_size:
            raise ValueError("Call Batch.unpad() before restoring trajectories")
        has_sequence_fields = {"tokens", "loss_mask"}.issubset(
            self.values
        ) or "segments" in self.values
        if not {"reward", "is_val"}.issubset(self.values) or not has_sequence_fields:
            raise ValueError(
                "Batch rows must contain trajectory sequence fields, reward and is_val"
            )
        trajectories = []
        for i in range(len(self)):
            row: dict[str, Any] = {
                key: value[i] if isinstance(value, (torch.Tensor, list, np.ndarray)) else value
                for key, value in self.values.items()
            }
            if row["is_val"] is None or row.get("status", "success") is None:
                raise ValueError("Batch contains padding or incomplete rows; call unpad() first")
            trajectory_fields = {
                key: row.pop(key)
                for key in ("tokens", "loss_mask", "reward", "is_val", "segments", "status")
                if key in row
            }
            if "extra_fields" in row:
                stored_extra = row.pop("extra_fields")
                if not isinstance(stored_extra, dict):
                    raise TypeError("A trajectory row's extra_fields must be a dictionary")
                extra_fields = stored_extra.copy()
                extra_fields.update(row)
            else:
                # Legacy flat batches may carry these defaults only in metadata.
                extra_fields = row
                for key in ("model_tag", "temperature"):
                    if key in self.metadata:
                        extra_fields.setdefault(key, self.metadata[key])
            trajectories.append(Trajectory(**trajectory_fields, extra_fields=extra_fields))
        return trajectories

    @staticmethod
    def remove_redundant_left_padding(
        data: "Batch",
        pad_token_id: int,
        fields: list[str] | None = None,
        anchor_field: str = "input_ids",
        max_strip_threshold: int = -1,
    ) -> "Batch":
        """Remove redundant left padding tokens that are common across all sequences.

        Args:
            data: The batch to process
            pad_token_id: The ID of the padding token
            fields: List of field names to strip padding from. If None, strips from all 2D tensors
            anchor_field: The field to use for determining padding length
            max_strip_threshold: Maximum number of tokens to strip. If -1, no limit

        Returns:
            A new Batch with left padding removed
        """
        # If anchor_field not in data.values, return data as is
        if anchor_field not in data.values:
            return data

        anchor_tensor = data.values[anchor_field]

        # If not a tensor or not 2D, return as is
        if not isinstance(anchor_tensor, torch.Tensor) or len(anchor_tensor.shape) != 2:
            return data

        # Find left padding count for each sequence
        padding_mask = anchor_tensor == pad_token_id
        non_pad_mask = ~padding_mask
        cumsum = non_pad_mask.cumsum(dim=1)
        left_pad_counts = (cumsum == 0).sum(dim=1).tolist()

        # Find minimum padding count across all sequences
        min_pad_count = min(left_pad_counts)

        if min_pad_count == 0:
            return data

        # Apply max_strip_threshold if specified
        if max_strip_threshold >= 0:
            min_pad_count = min(min_pad_count, max_strip_threshold)

        # Create new values dict with padding stripped
        new_values = {}
        for key, value in data.values.items():
            # Only strip 2D tensors/arrays in specified fields (or all if fields is None)
            if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                if fields is None or key in fields:
                    new_values[key] = value[:, min_pad_count:]
                else:
                    new_values[key] = value
            else:
                new_values[key] = value

        # Return new Batch with stripped values
        return Batch(values=new_values, metadata=data.metadata.copy())

    @staticmethod
    def remove_redundant_right_padding(
        data: "Batch",
        pad_token_id: int,
        fields: list[str] | None = None,
        anchor_field: str = "input_ids",
        max_strip_threshold: int = -1,
    ) -> "Batch":
        """Remove redundant right padding tokens that are common across all sequences.

        Args:
            data: The batch to process
            pad_token_id: The ID of the padding token
            fields: List of field names to strip padding from. If None, strips from all 2D tensors
            anchor_field: The field to use for determining padding length
            max_strip_threshold: Maximum number of tokens to strip. If -1, no limit

        Returns:
            A new Batch with right padding removed
        """
        # If anchor_field not in data.values, return data as is
        if anchor_field not in data.values:
            return data

        anchor_tensor = data.values[anchor_field]

        # If not a tensor or not 2D, return as is
        if not isinstance(anchor_tensor, torch.Tensor) or len(anchor_tensor.shape) != 2:
            return data

        # Find right padding count for each sequence by flipping and counting from the start
        padding_mask = anchor_tensor == pad_token_id
        reversed_mask = padding_mask.flip(dims=[1])
        cumsum = (~reversed_mask).cumsum(dim=1)
        right_pad_counts = (cumsum == 0).sum(dim=1)

        # Find minimum padding count across all sequences
        min_pad_count = int(right_pad_counts.min().item())

        if min_pad_count == 0:
            return data

        # Apply max_strip_threshold if specified
        if max_strip_threshold >= 0:
            min_pad_count = min(min_pad_count, max_strip_threshold)

        # Create new values dict with padding stripped
        new_values = {}
        for key, value in data.values.items():
            # Only strip 2D tensors/arrays in specified fields (or all if fields is None)
            if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                if fields is None or key in fields:
                    new_values[key] = value[:, :-min_pad_count]
                else:
                    new_values[key] = value
            else:
                new_values[key] = value

        # Return new Batch with stripped values
        return Batch(values=new_values, metadata=data.metadata.copy())

    def trim_for_backend(self, required_tensor_keys: list[str]) -> "Batch":
        """Trim the batch to only include required tensor keys for the train service backend.

        This removes all non-tensor values and unused tensor values from the batch
        to reduce network traffic and GPU memory when sending to the backend.

        Only tensor values whose keys appear in ``required_tensor_keys`` are kept.
        Keys listed in ``required_tensor_keys`` that are not present in the batch
        are silently skipped (they may be optional fields).

        Metadata is preserved as-is since the backend reads configuration from it.

        Args:
            required_tensor_keys: List of tensor key names that the backend operation
                actually consumes.

        Returns:
            A new Batch containing only the required tensor values and the original metadata.
        """
        trimmed_values = {}
        for key in required_tensor_keys:
            if key in self.values and isinstance(self.values[key], torch.Tensor):
                trimmed_values[key] = self.values[key]
        return Batch(values=trimmed_values, metadata=self.metadata.copy())

    def to_nextrainer_batch(self) -> dict[str, Any]:
        """
        Prepare batch for NexTrainer
        """
        tensor_batch = {}
        non_tensor_batch = {}

        # Separate tensor and non-tensor values
        for key, value in self.values.items():
            if isinstance(value, torch.Tensor):
                tensor_batch[key] = value
            else:
                non_tensor_batch[key] = np.asarray(value, dtype=object)

        batch = {
            "batch": tensor_batch,
            "non_tensor_batch": non_tensor_batch,
            "meta_info": self.metadata,
        }
        return batch
