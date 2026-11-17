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

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from weaver import types

CustomLossCallback = Callable[
    [Sequence[Any], list[torch.Tensor]], tuple[torch.Tensor, dict[str, Any]]
]
CustomLossFactory = Callable[[Mapping[str, Any]], CustomLossCallback]


@dataclass(frozen=True, slots=True)
class CustomLossSpec:
    """Serializable description of a Weaver client-side custom loss."""

    name: str
    params: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


def parse_custom_loss_spec(raw_spec: Any) -> CustomLossSpec | None:
    """Normalize trainer config into a serializable custom loss spec."""
    if raw_spec is None:
        return None

    if isinstance(raw_spec, str):
        name = raw_spec.strip()
        if not name:
            raise ValueError("custom_loss string must be non-empty")
        return CustomLossSpec(name=name, params={})

    if not isinstance(raw_spec, Mapping):
        raise TypeError("custom_loss must be a string or mapping")

    raw_name = raw_spec.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("custom_loss.name must be a non-empty string")
    name = raw_name.strip()

    params = raw_spec.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise TypeError("custom_loss.params must be a mapping")

    return CustomLossSpec(name=name, params=dict(params))


def resolve_custom_loss(spec: CustomLossSpec | Mapping[str, Any]) -> CustomLossCallback:
    """Resolve a serializable spec into a Weaver custom-loss callback."""
    normalized_spec = spec if isinstance(spec, CustomLossSpec) else parse_custom_loss_spec(spec)
    if normalized_spec is None:
        raise ValueError("custom_loss spec is required")

    factory = _CUSTOM_LOSS_FACTORIES.get(normalized_spec.name)
    if factory is not None:
        return factory(normalized_spec.params)

    loaded = _load_import_path(_CUSTOM_LOSS_IMPORTS.get(normalized_spec.name, normalized_spec.name))
    if _looks_like_callback(loaded) and not normalized_spec.params:
        return loaded

    callback = loaded(normalized_spec.params)
    if not callable(callback):
        raise TypeError(
            f"custom loss factory '{normalized_spec.name}' must return a callable callback"
        )
    return callback


def _load_import_path(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(
            f"Unknown custom loss '{path}'. Use a registered alias or an import path 'module:factory'."
        )
    module_name, attr_name = path.split(":", 1)
    if not module_name or not attr_name:
        raise ValueError(f"Invalid custom loss import path: {path}")

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name, None)
    if value is None:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr_name}'")
    if not callable(value):
        raise TypeError(f"Imported custom loss target '{path}' is not callable")
    return value


def _looks_like_callback(value: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    positional_params = [
        param
        for param in signature.parameters.values()
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional_params) >= 2


def _loss_input_to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _resolve_targets(datum: Any) -> list[Any]:
    raw_targets = datum.loss_fn_inputs.get("target_tokens")
    if raw_targets is None:
        return list(datum.model_input.to_ints())
    value = _loss_input_to_list(raw_targets)
    return list(value)


def _resolve_required_tensor(
    datum: Any,
    field_names: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    shape: torch.Size,
) -> torch.Tensor:
    for field_name in field_names:
        raw_value = datum.loss_fn_inputs.get(field_name)
        if raw_value is None:
            continue
        tensor = torch.as_tensor(_loss_input_to_list(raw_value), dtype=dtype, device=device)
        if tensor.shape != shape:
            raise ValueError(
                f"{field_name} shape {tuple(tensor.shape)} does not match logprobs shape {tuple(shape)}"
            )
        return tensor
    joined = "/".join(field_names)
    raise ValueError(f"{joined} must be provided for custom loss '{field_names[0]}'.")


def _build_importance_sampling_factory(*, truncated: bool) -> CustomLossFactory:
    def factory(params: Mapping[str, Any]) -> CustomLossCallback:
        entropy_coeff = float(
            params.get(
                "entropy_coeff", params.get("entropy_coef", params.get("entropy_coefficient", 0.0))
            )
        )
        if entropy_coeff != 0.0:
            raise ValueError(
                "Weaver custom importance_sampling does not support entropy regularization "
                "with the current forward_backward_custom API. Set entropy_coeff=0."
            )
        truncation_c = float(params.get("truncation_c", params.get("truncation_clip_c", 10.0)))

        def callback(
            data: Sequence[Any],
            logprob_tensors: list[torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            if len(data) != len(logprob_tensors):
                raise ValueError(
                    f"Expected {len(data)} logprob tensors, got {len(logprob_tensors)}"
                )

            per_sample_losses: list[torch.Tensor] = []
            valid_ratio_slices: list[torch.Tensor] = []
            valid_logprob_diff_slices: list[torch.Tensor] = []
            total_tokens = 0

            for datum, logprobs in zip(data, logprob_tensors):
                if logprobs.numel() == 0:
                    raise ValueError("Each datum must contain at least one target token.")

                sampling_logprobs = _resolve_required_tensor(
                    datum,
                    ("sampling_logprobs", "logprobs"),
                    dtype=logprobs.dtype,
                    device=logprobs.device,
                    shape=logprobs.shape,
                )
                advantages = _resolve_required_tensor(
                    datum,
                    ("advantages",),
                    dtype=logprobs.dtype,
                    device=logprobs.device,
                    shape=logprobs.shape,
                )
                loss_mask = _resolve_required_tensor(
                    datum,
                    ("loss_mask",),
                    dtype=torch.bool,
                    device=logprobs.device,
                    shape=logprobs.shape,
                )

                logprob_diff = logprobs - sampling_logprobs
                prob_ratio = torch.exp(logprob_diff)
                if truncated:
                    prob_ratio = torch.clamp(prob_ratio, max=truncation_c)

                elementwise_loss = -(prob_ratio * advantages)
                masked_loss = torch.where(
                    loss_mask, elementwise_loss, torch.zeros_like(elementwise_loss)
                )
                sample_token_count = loss_mask.sum().to(dtype=logprobs.dtype).clamp(min=1.0)
                per_sample_losses.append(masked_loss.sum() / sample_token_count)

                total_tokens += int(loss_mask.sum().item())
                valid_ratio_slices.append(prob_ratio[loss_mask])
                valid_logprob_diff_slices.append(logprob_diff[loss_mask])

            total_loss = torch.stack(per_sample_losses).mean()
            metrics: dict[str, Any] = {
                "loss": float(total_loss.detach().cpu()),
                "loss:sum": float(total_loss.detach().cpu()),
                "tokens": total_tokens,
                "num_samples": len(logprob_tensors),
                "entropy_coeff": 0.0,
            }

            non_empty_ratios = [tensor for tensor in valid_ratio_slices if tensor.numel() > 0]
            if non_empty_ratios:
                valid_ratios = torch.cat(non_empty_ratios)
                metrics["ratio:mean"] = float(valid_ratios.mean().detach().cpu())
                metrics["ratio:max"] = float(valid_ratios.max().detach().cpu())
                metrics["ratio:min"] = float(valid_ratios.min().detach().cpu())
                metrics["ratio:std"] = float(valid_ratios.std(unbiased=False).detach().cpu())
                metrics["ratio:gt1_frac"] = float(
                    (valid_ratios > 1.0).float().mean().detach().cpu()
                )
                metrics["ratio:gt2_frac"] = float(
                    (valid_ratios > 2.0).float().mean().detach().cpu()
                )
                metrics["ratio:gt5_frac"] = float(
                    (valid_ratios > 5.0).float().mean().detach().cpu()
                )

            non_empty_diffs = [tensor for tensor in valid_logprob_diff_slices if tensor.numel() > 0]
            if non_empty_diffs:
                valid_diffs = torch.cat(non_empty_diffs)
                metrics["logprob_diff:mean"] = float(valid_diffs.mean().detach().cpu())
                metrics["logprob_diff:std"] = float(valid_diffs.std(unbiased=False).detach().cpu())
                metrics["logprob_diff:min"] = float(valid_diffs.min().detach().cpu())
                metrics["logprob_diff:max"] = float(valid_diffs.max().detach().cpu())

            return total_loss, metrics

        return callback

    return factory


def build_direct_logprob_loss(params: Mapping[str, Any]) -> CustomLossCallback:
    """Simple generic loss used for smoke tests and examples."""
    scale = float(params.get("scale", 1.0))

    def callback(
        data: Sequence[Any],
        logprob_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if len(data) != len(logprob_tensors):
            raise ValueError(f"Expected {len(data)} logprob tensors, got {len(logprob_tensors)}")

        per_sample_losses = []
        total_tokens = 0
        for datum, logprobs in zip(data, logprob_tensors):
            targets = _resolve_targets(datum)
            if len(targets) != logprobs.numel():
                raise ValueError(
                    f"target_tokens length {len(targets)} does not match logprobs shape {tuple(logprobs.shape)}"
                )
            per_sample_losses.append(-scale * logprobs.mean())
            total_tokens += int(logprobs.numel())

        total_loss = torch.stack(per_sample_losses).mean()
        return total_loss, {
            "loss": float(total_loss.detach().cpu()),
            "loss:sum": float(total_loss.detach().cpu()),
            "tokens": total_tokens,
            "num_samples": len(logprob_tensors),
            "scale": scale,
        }

    return callback


_CUSTOM_LOSS_FACTORIES: dict[str, CustomLossFactory] = {
    "importance_sampling": _build_importance_sampling_factory(truncated=False),
    "truncated_importance_sampling": _build_importance_sampling_factory(truncated=True),
    "direct_logprob": build_direct_logprob_loss,
}


_CUSTOM_LOSS_IMPORTS: dict[str, str] = {
    "cispo": "northrl.weaver.cispo_loss:build_cispo_loss",
    "gspo": "northrl.algorithm.gspo_loss:build_gspo_loss",
}
