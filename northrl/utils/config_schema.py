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

"""Pydantic schema for NorthRL config (new format).

Design and decisions: see CONFIG_REFACTOR.md.

This module defines:
  * 10 top-level config blocks (experiment / models / data / validate /
    algorithm / rollout / training / backend / runtime)
  * Field-level constraints (Literal, ge/le, min_length, ...)
  * Cross-block validators (KL ↔ reference, OPD ↔ teachers,
    lora.enabled ↔ rank, max_tokens validations, custom_loss ↔ entropy_coeff,
    filter_prompts ↔ tokenizer)

All models forbid extra fields (`extra="forbid"`) so typos surface as errors,
not silent no-ops.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml  # type: ignore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from northrl.remote_rollout.network_policy_config import normalize_network_policy
from northrl.utils.algorithm_config import (
    AdvantageConfig,
    ClipParams,
    KLParams,
    LossConfig,
)
from northrl.utils.config_utils import validate_mis_sampling
from northrl.utils.e2b_volume_mounts import validate_volume_mounts


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ============================================================
# experiment
# ============================================================


class ExperimentConfig(_StrictModel):
    name: str = Field(..., description="实验名（每次必改）；进 wandb run / 日志路径")
    project: str = Field(..., description="项目名（同项目下可固定）")
    seed: int = 42


# ============================================================
# models
# ============================================================


class LoraConfig(_StrictModel):
    enabled: bool
    rank: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _check_rank_matches_enabled(self) -> "LoraConfig":
        if self.enabled and self.rank is None:
            raise ValueError("models.actor.lora.enabled=true 时 rank 必填（int ≥ 1）")
        if not self.enabled and self.rank is not None:
            raise ValueError(
                "models.actor.lora.enabled=false 时 rank 必须不写或 null"
                "（rank 无意义；写了会导致用户改 rank 测调参时'配置藏雷'）"
            )
        return self


class ActorModelConfig(_StrictModel):
    name: str = Field(..., description="weaver 认得的模型名（用户给 HF 名最方便）")
    tokenizer: str | None = Field(
        None,
        description=(
            "HF tokenizer name_or_path；null = backend (weaver) 决定。"
            " 下游消费方：① weaver_service_holder（始终需要）② dataloader (filter_prompts=true)"
            "。后者 null 时启动报错。"
        ),
    )
    checkpoint: str | None = None
    lora: LoraConfig | None = None


class ReferenceModelConfig(_StrictModel):
    name: str | None = Field(None, description="null = inherit models.actor.name")
    checkpoint: str | None = None


class TeacherModelConfig(_StrictModel):
    name: str
    checkpoint: str | None = None


class ModelsConfig(_StrictModel):
    actor: ActorModelConfig
    reference: ReferenceModelConfig | None = None
    teachers: list[TeacherModelConfig] | None = None


# ============================================================
# data
# ============================================================


class LocalTrainSourceConfig(_StrictModel):
    type: Literal["local"]
    files: list[str] = Field(..., min_length=1)
    ratio: float = Field(..., gt=0)
    name: str | None = None
    rollout_profile: str | None = None

    @field_validator("name", "rollout_profile")
    @classmethod
    def _validate_source_name(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError(
                "source name and rollout_profile must be nonblank without boundary whitespace"
            )
        return value

    @field_validator("ratio")
    @classmethod
    def _finite_ratio(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("source ratio must be finite")
        return value


class ManagedTrainSourceConfig(_StrictModel):
    type: Literal["managed"]
    dataset: str
    version: str
    ratio: float = Field(..., gt=0)

    @field_validator("ratio")
    @classmethod
    def _finite_ratio(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("source ratio must be finite")
        return value

    @field_validator("dataset", "version")
    @classmethod
    def _validate_identity(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name
        if not value:
            raise ValueError("managed dataset name and version must not be blank")
        if value != value.strip():
            raise ValueError(f"managed dataset {field_name} must not have boundary whitespace")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"managed dataset {field_name} must be valid UTF-8") from exc
        max_length = 160 if field_name == "dataset" else 128
        if len(value) > max_length:
            raise ValueError(
                f"managed dataset {field_name} must be at most {max_length} characters"
            )
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"managed dataset {field_name} must be a single safe path segment")
        if any(unicodedata.category(character) == "Cc" for character in value):
            raise ValueError(f"managed dataset {field_name} must not contain control characters")
        return value


TrainSourceConfig = Annotated[
    LocalTrainSourceConfig | ManagedTrainSourceConfig,
    Field(discriminator="type"),
]


class TrainDataConfig(_StrictModel):
    # ``files`` retains concatenation; ``sources`` selects weighted RL sampling
    # or deterministic SFT mixing, depending on the algorithm.
    files: list[str] | None = None
    sources: list[TrainSourceConfig] | None = None
    total_samples: int | None = Field(None, ge=1)
    seed: int | None = None
    managed_variant_ready_timeout_seconds: float = Field(300.0, gt=0)
    managed_variant_poll_initial_seconds: float = Field(1.0, gt=0)
    managed_variant_poll_max_seconds: float = Field(10.0, gt=0)
    batch_size: int = Field(..., ge=1)
    rollout_repeat_n: int | None = Field(None, ge=1)
    group_order: Literal["fifo", "lifo"] = Field(
        "fifo",
        description="Grouped rollout pool selection by group completion order; unused for SFT",
    )
    prompt_key: str = "prompt"
    filter_prompts: bool = False
    shuffle: bool = True
    drop_last: bool = True
    window_samples: int | None = Field(
        None,
        ge=1,
        description="SFT only: close each optimizer window after this many samples",
    )
    render_chat: str | None = Field(
        None,
        description="SFT renderer in 'path/to/file.py:function' form",
    )

    @field_validator("render_chat")
    @classmethod
    def _strip_render_chat(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("data.train.render_chat must not be blank")
        return value

    @field_validator(
        "managed_variant_ready_timeout_seconds",
        "managed_variant_poll_initial_seconds",
        "managed_variant_poll_max_seconds",
    )
    @classmethod
    def _finite_managed_variant_timing(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("managed variant polling values must be finite")
        return value

    @model_validator(mode="after")
    def _check_source_selection(self) -> "TrainDataConfig":
        if self.managed_variant_poll_initial_seconds > self.managed_variant_poll_max_seconds:
            raise ValueError(
                "data.train.managed_variant_poll_initial_seconds must not exceed "
                "data.train.managed_variant_poll_max_seconds"
            )
        has_files = self.files is not None
        has_sources = self.sources is not None
        if has_files == has_sources:
            raise ValueError("data.train must set exactly one of files or sources")
        if has_files:
            if not self.files:
                raise ValueError("data.train.files must not be empty")
            if self.total_samples is not None:
                raise ValueError("data.train.total_samples is valid only with data.train.sources")
            return self

        sources = self.sources
        if not isinstance(sources, list) or not sources:
            raise ValueError("data.train.sources must not be empty")
        identities: set[tuple[Any, ...]] = set()
        for source in sources:  # pylint: disable=not-an-iterable
            if isinstance(source, LocalTrainSourceConfig):
                identity: tuple[Any, ...] = ("local", *source.files)
            else:
                identity = ("managed", source.dataset, source.version)
            if identity in identities:
                raise ValueError(f"data.train.sources contains duplicate source {identity!r}")
            identities.add(identity)
        return self


class DataConfig(_StrictModel):
    train: TrainDataConfig
    max_prompt_length: int | None = Field(
        None, ge=1, description="Required outside SFT; omit for SFT"
    )
    max_sequence_length: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _check_length_ordering(self) -> "DataConfig":
        if self.max_prompt_length is None:
            return self
        if self.max_prompt_length > self.max_sequence_length:
            raise ValueError(
                f"data.max_prompt_length ({self.max_prompt_length}) 不能超过 "
                f"data.max_sequence_length ({self.max_sequence_length})"
            )
        if self.max_sequence_length - self.max_prompt_length < 1:
            raise ValueError(
                f"data.max_sequence_length ({self.max_sequence_length}) - "
                f"data.max_prompt_length ({self.max_prompt_length}) 必须 ≥ 1 "
                f"（否则没空间生成 token）"
            )
        return self


# ============================================================
# validate
# ============================================================


class ValidateDatasetConfig(_StrictModel):
    files: list[str] = Field(..., min_length=1)
    batch_size: int = Field(..., ge=1)


class ValidateDataConfig(_StrictModel):
    datasets: dict[str, ValidateDatasetConfig] = Field(..., min_length=1)
    validation_repeat_n: int = Field(
        1, ge=1, description="每个验证 prompt 采样几条；即 grouped pass@k 的 k"
    )


class ValidateSamplingConfig(_StrictModel):
    reasoning_effort: str | None = Field(None, min_length=1)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    min_p: float = 0.0
    max_tokens: int | None = Field(
        None,
        ge=1,
        description="可选；E2B 未配置/None 时保留 agent 预算，显式值覆盖；其他推理路径缺省为序列与提示词长度之差",
    )


class ValidateScheduleConfig(_StrictModel):
    before_train: bool = False
    every_n_steps: int = Field(0, ge=0, description="0 = 训练中不评估")


class ValidateConfig(_StrictModel):
    enabled: bool = True
    data: ValidateDataConfig | None = None
    sampling: ValidateSamplingConfig = Field(default_factory=ValidateSamplingConfig)
    schedule: ValidateScheduleConfig = Field(default_factory=ValidateScheduleConfig)
    # pass_at_k：按 prompt 分组算 pass@k / avg@k（默认 mean = 逐 key 均值）
    aggregation: Literal["mean", "pass_at_k"] = "mean"

    @model_validator(mode="after")
    def _check_data_when_enabled(self) -> "ValidateConfig":
        if self.enabled and self.data is None:
            raise ValueError("validate.data is required when validate.enabled=true")
        return self


# ============================================================
# algorithm
# ============================================================


class LRScheduleConfig(_StrictModel):
    style: Literal["cosine"] = "cosine"
    min_lr: float = Field(..., ge=0)
    warmup_fraction: float = Field(..., ge=0, lt=1)


class OptimConfig(_StrictModel):
    lr: float = Field(..., gt=0)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    batch_size: int | None = Field(
        None,
        ge=1,
        description="每次 optimizer 更新的 Datum 数；sync、batch-async、fully-async 中省略或 null 使用当前整批；SFT 使用完整 token-budget batch",
    )
    mini_batch_size: int | None = Field(
        None,
        ge=1,
        description="单次 actor forward-only / forward_backward 的最大 Datum 数量；不写则不按条数拆分请求",
    )
    submit_ahead: int = Field(0, ge=0, strict=True, description="Weaver 请求预提交数；0 保持串行")
    schedule: LRScheduleConfig | None = None

    @model_validator(mode="after")
    def _check_lr_schedule(self) -> "OptimConfig":
        if self.schedule is not None and self.schedule.min_lr > self.lr:
            raise ValueError("algorithm.optim.schedule.min_lr cannot exceed algorithm.optim.lr")
        return self


class KLConfig(_StrictModel):
    mode: Literal["in_loss", "in_reward"] | None = None
    params: KLParams = Field(default_factory=KLParams)
    actor_to_ref_sync_freq: int = Field(0, ge=0)
    ref_logprobs_batch_size: int = Field(0, ge=0)

    @property
    def enabled(self) -> bool:
        return self.mode is not None


class RouterReplayConfig(_StrictModel):
    enabled: bool = False
    mode: Literal["R2", "R3"] = "R3"
    # Routing shape (num_layers / topk / expert bounds) and fail-fast behavior
    # live in weaver-trainer: it validates the replay indices against the real
    # model and errors on any mismatch. NorthRL only transports the indices. The
    # GPFS ref root is read from WEAVER_PAYLOAD_REF_ROOT / WEAVER_ROUTER_REPLAY_REF_ROOT
    # by the R3 writer (server) and the trainer reader, not from here.


class OPDConfig(_StrictModel):
    teacher_service_timeout_s: float = Field(1200.0, gt=0)
    # 教师 logprob 来源："inference_service"=推理服务 prefill（默认，对齐 tinker/slime）；
    # "training_service"=训练服务 forward_logprob（trainer kernel，对齐 opd-http-opt）。
    teacher_logprob_source: Literal["inference_service", "training_service"] = "inference_service"


class MISConfig(_StrictModel):
    """Rollout correction and diagnostics from frozen rollout and trainer logprobs."""

    enabled: bool = False
    get_mismatch_metrics: bool = False
    tis_mode: Literal["none", "truncate", "clip", "mask"] = "truncate"
    tis_level: Literal["token", "sequence", "geometric"] = "token"
    tis_lower_bound: float | None = Field(None, gt=0, allow_inf_nan=False)
    tis_upper_bound: float = Field(2.0, gt=0, allow_inf_nan=False)
    use_rs: bool = False
    rs_level: Literal["token", "sequence", "geometric"] = "token"
    rs_lower_bound: float | None = Field(None, gt=0, allow_inf_nan=False)
    rs_upper_bound: float | None = Field(None, gt=0, allow_inf_nan=False)
    rs_veto_threshold: float | None = Field(None, gt=0, allow_inf_nan=False)
    tis_batch_normalize: bool = False

    @model_validator(mode="after")
    def _check_active_stages(self) -> "MISConfig":
        if not self.enabled:
            return self

        tis_lower = (
            self.tis_lower_bound if self.tis_lower_bound is not None else 1.0 / self.tis_upper_bound
        )
        if self.tis_mode in {"clip", "mask"} and tis_lower >= self.tis_upper_bound:
            raise ValueError(
                "algorithm.mis requires effective tis_lower_bound < tis_upper_bound "
                "for tis_mode='clip' or 'mask'."
            )
        if self.tis_mode != "none" and self.tis_level == "geometric" and self.tis_batch_normalize:
            raise ValueError(
                "algorithm.mis.tis_batch_normalize is incompatible with "
                "tis_level='geometric' when TIS is enabled."
            )
        if self.use_rs:
            rs_lower = self.rs_lower_bound if self.rs_lower_bound is not None else tis_lower
            rs_upper = (
                self.rs_upper_bound if self.rs_upper_bound is not None else self.tis_upper_bound
            )
            if rs_lower >= rs_upper:
                raise ValueError(
                    "algorithm.mis requires effective rs_lower_bound < rs_upper_bound "
                    "when use_rs=true (unset RS bounds inherit TIS bounds)."
                )
        return self


class AlgorithmConfig(_StrictModel):
    type: Literal["grpo", "opd", "sft"]
    advantage: AdvantageConfig = Field(default_factory=AdvantageConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    sampling_logprob_source: Literal["inference", "recomputed"] = "recomputed"

    optim: OptimConfig
    entropy_coeff: float = 1.0e-4
    kl: KLConfig = Field(default_factory=KLConfig)
    router_replay: RouterReplayConfig = Field(default_factory=RouterReplayConfig)
    opd: OPDConfig = Field(default_factory=OPDConfig)
    mis: MISConfig = Field(default_factory=MISConfig)

    @model_validator(mode="before")
    @classmethod
    def _resolve_component_defaults(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        resolved = deepcopy(dict(value))
        advantage = resolved.setdefault("advantage", {})
        if isinstance(advantage, Mapping):
            advantage = dict(advantage)
            resolved["advantage"] = advantage
            estimator = advantage.setdefault("estimator", {})
            if isinstance(estimator, Mapping):
                estimator = dict(estimator)
                advantage["estimator"] = estimator
                flow = resolved.get("type")
                estimator.setdefault("mode", flow if flow in ("grpo", "opd") else None)
        loss = resolved.setdefault("loss", {})
        if isinstance(loss, Mapping):
            loss = dict(loss)
            resolved["loss"] = loss
            loss.setdefault(
                "mode", "cross_entropy" if resolved.get("type") == "sft" else "importance_sampling"
            )
            # A fixed-budget PPO bound can use ratio or equal low/high values.
            reduction = loss.get("reduction", {})
            params = loss.get("params", {})
            if (
                isinstance(reduction, Mapping)
                and reduction.get("mode") == "fixed_response_budget"
                and isinstance(params, Mapping)
                and "clip" in params
            ):
                clip = ClipParams.model_validate(params["clip"])
                if (
                    clip.ratio_low is not None
                    and clip.ratio_low == clip.ratio_high
                    and isfinite(clip.ratio_low)
                    and clip.ratio_low > 0
                    and all(
                        bound is None for bound in (clip.ratio, clip.ratio_c, clip.truncation_c)
                    )
                ):
                    loss["params"] = {
                        **params,
                        "clip": {
                            **clip.model_dump(),
                            "ratio": clip.ratio_low,
                            "ratio_low": None,
                            "ratio_high": None,
                        },
                    }
        return resolved

    @model_validator(mode="after")
    def _check_cispo_loss_contract(self) -> "AlgorithmConfig":
        if self.loss.mode == "custom" and (
            self.loss.params["name"] == "cispo" or self.loss.params["base_loss_fn"] == "cispo"
        ):
            raise ValueError("Select CISPO with algorithm.loss.mode='cispo'")
        if self.loss.mode != "cispo":
            return self
        if self.type != "grpo":
            raise ValueError("CISPO requires algorithm.type='grpo'")
        if self.entropy_coeff != 0 or self.kl.enabled:
            raise ValueError("CISPO requires entropy_coeff=0 and KL disabled")
        if self.mis.enabled:
            raise ValueError("CISPO does not support algorithm.mis.enabled=true")
        if self.optim.submit_ahead:
            raise ValueError("CISPO requires optim.submit_ahead=0")
        if self.loss.reduction.mode != "token_mean":
            raise ValueError("CISPO requires loss.reduction.mode='token_mean'")
        return self

    @model_validator(mode="after")
    def _check_trajectory_mean_owner(self) -> "AlgorithmConfig":
        if self.loss.reduction.mode == "trajectory_mean" and self.loss.mode != "gspo":
            raise ValueError("trajectory_mean is currently supported only by GSPO")
        return self

    @model_validator(mode="after")
    def _check_gspo_loss_contract(self) -> "AlgorithmConfig":
        if self.loss.mode == "custom" and (
            self.loss.params["name"] == "gspo" or self.loss.params["base_loss_fn"] == "gspo"
        ):
            raise ValueError("Select GSPO with algorithm.loss.mode='gspo'")
        if self.loss.mode != "gspo":
            return self
        if self.advantage.estimator.mode not in {"grpo", "rloo"}:
            raise ValueError("GSPO requires a scalar trajectory estimator (grpo or rloo)")
        if self.advantage.postprocessing.mode is not None:
            raise ValueError("GSPO requires disabled advantage postprocessing")
        if self.type != "grpo":
            raise ValueError("GSPO requires algorithm.type='grpo'")
        if self.entropy_coeff != 0 or self.kl.enabled:
            raise ValueError("GSPO requires entropy_coeff=0 and KL disabled")
        if self.mis.enabled:
            raise ValueError("GSPO does not support algorithm.mis.enabled=true")
        if self.optim.submit_ahead:
            raise ValueError("GSPO requires optim.submit_ahead=0")
        if self.loss.reduction.mode != "trajectory_mean":
            raise ValueError("GSPO requires loss.reduction.mode='trajectory_mean'")
        if self.optim.batch_size is not None:
            raise ValueError("GSPO requires optim.batch_size=null for complete trajectories")
        return self

    @model_validator(mode="after")
    def _check_estimator_flow(self) -> "AlgorithmConfig":
        if self.advantage.estimator.mode in {"reinforce_plus_plus", "reinforce_plus_plus_baseline"}:
            if self.type != "grpo":
                raise ValueError("REINFORCE++ advantage estimators require algorithm.type='grpo'")
            return self
        expected = {"grpo": "grpo", "opd": "opd", "sft": None}[self.type]
        if self.advantage.estimator.mode != expected and not (
            self.type == "grpo" and self.advantage.estimator.mode == "rloo"
        ):
            raise ValueError(
                f"algorithm.type={self.type!r} requires advantage.estimator.mode={expected!r}; "
                "other estimators are not implemented"
            )
        if self.advantage.estimator.mode is None and self.advantage.postprocessing.mode is not None:
            raise ValueError("Advantage postprocessing requires an advantage estimator.")
        if self.advantage.estimator.mode is not None and self.loss.mode == "cross_entropy":
            raise ValueError(
                "cross_entropy loss does not consume advantages; use algorithm.type='sft'."
            )
        return self

    @model_validator(mode="after")
    def _check_reinforce_plus_plus_contract(self) -> "AlgorithmConfig":
        if self.advantage.estimator.mode not in {
            "reinforce_plus_plus",
            "reinforce_plus_plus_baseline",
        }:
            return self
        if self.loss.mode not in {"importance_sampling", "ppo_clip"}:
            raise ValueError("REINFORCE++ requires native importance_sampling or ppo_clip loss")
        if self.loss.reduction.mode != "token_mean":
            raise ValueError("REINFORCE++ requires loss.reduction.mode='token_mean'")
        return self

    @model_validator(mode="after")
    def _check_dr_grpo_contract(self) -> "AlgorithmConfig":
        if self.loss.reduction.mode != "fixed_response_budget":
            return self
        if self.type != "grpo" or self.loss.mode != "ppo_clip":
            raise ValueError("dr-grpo requires algorithm.type='grpo' and loss.mode='ppo_clip'.")
        estimator = self.advantage.estimator
        if estimator.mode != "grpo" or estimator.params["norm_adv_by_std"]:
            raise ValueError(
                "dr-grpo requires advantage.estimator.mode='grpo' and "
                "advantage.estimator.params.norm_adv_by_std=false."
            )
        clip = ClipParams.model_validate(self.loss.params["clip"])
        if clip.ratio is None or not isfinite(clip.ratio) or clip.ratio <= 0:
            raise ValueError(
                "dr-grpo requires an explicit finite positive algorithm.loss.params.clip.ratio."
            )
        if any(
            value is not None
            for value in (clip.ratio_low, clip.ratio_high, clip.ratio_c, clip.truncation_c)
        ):
            raise ValueError(
                "dr-grpo supports symmetric clipping via algorithm.loss.params.clip.ratio "
                "or equal ratio_low/ratio_high; ratio_c and truncation_c must be omitted or null."
            )
        if self.entropy_coeff != 0 or self.kl.enabled:
            raise ValueError("dr-grpo requires KL disabled and entropy_coeff=0.")
        if self.optim.batch_size is not None:
            raise ValueError(
                "dr-grpo requires algorithm.optim.batch_size=null to preserve complete "
                "trajectories; use optim.mini_batch_size to split forward/backward requests."
            )
        return self

    @model_validator(mode="after")
    def _check_prompt_mean_contract(self) -> "AlgorithmConfig":
        if self.loss.reduction.mode != "prompt_mean":
            return self
        if self.type != "grpo":
            raise ValueError(
                "algorithm.loss.reduction.mode='prompt_mean' requires algorithm.type='grpo'."
            )
        if self.loss.mode == "custom":
            raise ValueError(
                "algorithm.loss.reduction.mode='prompt_mean' does not support algorithm.loss.mode='custom'."
            )
        if self.loss.mode not in {"importance_sampling", "ppo_clip"}:
            raise ValueError(
                "algorithm.loss.reduction.mode='prompt_mean' requires the native "
                "importance_sampling or ppo_clip loss."
            )
        if self.entropy_coeff != 0:
            raise ValueError(
                "algorithm.loss.reduction.mode='prompt_mean' requires algorithm.entropy_coeff=0."
            )
        if self.kl.enabled:
            raise ValueError(
                "algorithm.loss.reduction.mode='prompt_mean' requires algorithm.kl.mode=null."
            )
        if self.optim.batch_size is not None:
            raise ValueError(
                "algorithm.loss.reduction.mode='prompt_mean' requires algorithm.optim.batch_size=null "
                "(or omitted) to keep complete groups in one optimizer update; "
                "use algorithm.optim.mini_batch_size to limit forward/backward request size."
            )
        return self

    @model_validator(mode="after")
    def _check_mis_training_contract(self) -> "AlgorithmConfig":
        if not self.mis.enabled:
            if self.mis.get_mismatch_metrics and self.type != "grpo":
                raise ValueError(
                    "algorithm.mis.get_mismatch_metrics=true requires algorithm.type='grpo'."
                )
            return self
        if self.type != "grpo":
            raise ValueError("algorithm.mis.enabled=true requires algorithm.type='grpo'.")
        if self.sampling_logprob_source != "recomputed":
            raise ValueError(
                "algorithm.mis.enabled=true requires "
                "algorithm.sampling_logprob_source='recomputed'."
            )
        if self.loss.mode == "custom":
            raise ValueError(
                "algorithm.mis.enabled=true is incompatible with algorithm.loss.mode='custom'."
            )
        if self.loss.mode not in {"importance_sampling", "ppo_clip"}:
            raise ValueError(
                "algorithm.mis.enabled=true requires the native "
                "importance_sampling or ppo_clip loss."
            )
        return self

    @model_validator(mode="after")
    def _check_router_replay_logprob_source(self) -> "AlgorithmConfig":
        if not self.router_replay.enabled:
            return self
        if self.router_replay.mode == "R2" and self.sampling_logprob_source != "recomputed":
            raise ValueError(
                "algorithm.router_replay.mode=R2 requires "
                "algorithm.sampling_logprob_source='recomputed' so RECORD runs "
                "during trainer-side logprob recompute."
            )
        # R3 replays the sampler's routing in both trainer forwards: first when
        # recomputing the IS denominator and again during the loss forward. This
        # keeps both sides of the ratio on the rollout-selected expert paths while
        # avoiding sampler/trainer logprob mismatch.
        if self.router_replay.mode == "R3" and self.sampling_logprob_source != "recomputed":
            raise ValueError(
                "algorithm.router_replay.mode=R3 requires "
                "algorithm.sampling_logprob_source='recomputed' so the IS denominator "
                "is produced by trainer.forward_logprob while replaying the rollout routing."
            )
        return self


# ============================================================
# rollout
# ============================================================


class SamplingConfig(_StrictModel):
    reasoning_effort: str | None = Field(None, min_length=1)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] = Field(default_factory=list)
    min_tokens: int = Field(0, ge=0)
    max_tokens: int | None = Field(
        None,
        ge=1,
        description="可选；E2B 未配置/None 时保留 agent 预算，显式值覆盖；其他推理路径缺省为序列与提示词长度之差",
    )


class BuiltinResponseParserConfig(_StrictModel):
    type: Literal["builtin"] = "builtin"
    reasoning: str | None = "think_tag"
    tool: str | None = "qwen25"

    @field_validator("reasoning", "tool")
    @classmethod
    def _strip_optional_parser_names(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CustomResponseParserConfig(_StrictModel):
    type: Literal["custom"] = "custom"
    module_path: str = Field(
        ...,
        min_length=1,
        description="Python 文件；相对路径以 recipe yaml 所在目录为基准",
    )

    @field_validator("module_path")
    @classmethod
    def _strip_module_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


ParsersConfig = Annotated[
    BuiltinResponseParserConfig | CustomResponseParserConfig,
    Field(discriminator="type"),
]


class RolloutConfig(_StrictModel):
    type: str | None = None
    custom_module_path: str | None = Field(None, description="相对 recipe yaml 路径")
    custom_class_name: str = "CustomRolloutWorker"
    params: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    num_workers: int = Field(..., ge=1)
    reward_fn: str | None = Field(
        None, description="Default reward name when the sample has no reward_fn"
    )
    max_concurrent_requests: int | None = Field(
        None,
        ge=1,
        description=(
            "Max in-flight Weaver sampler requests from rollout. "
            "None keeps the limit equal to rollout.num_workers; set lower than "
            "num_workers to keep many rollout coroutines while protecting the sampler."
        ),
    )
    max_ready_groups: int | None = Field(
        128,
        ge=1,
        description="Pause rollout admission when this many complete groups await training.",
    )
    keep_sampling_mask: bool = Field(
        False, description="Request and retain sampling masks returned by Weaver."
    )
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    generation_extra: dict[str, Any] = Field(default_factory=dict)
    parsers: ParsersConfig = Field(default_factory=BuiltinResponseParserConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_rollout_config(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        if "agent" in value:
            raise ValueError(
                "rollout.agent is no longer supported; use rollout.type, rollout.params, "
                "rollout.custom_module_path and rollout.custom_class_name, then remove rollout.agent"
            )
        if "resource" in value:
            raise ValueError(
                "rollout.resource is no longer supported; remove its unused workers_per_node "
                "and agents_per_worker fields and use rollout.num_workers for concurrency"
            )
        if value.get("type") == "agent":
            raise ValueError(
                "rollout.type='agent' is no longer supported; use rollout.type='single_turn'"
            )
        if value.get("type") == "sft":
            raise ValueError(
                "rollout.type='sft' is no longer supported; "
                "set algorithm.type='sft' and remove the top-level rollout block"
            )
        return value

    @field_validator("parsers", mode="before")
    @classmethod
    def _normalize_legacy_parsers_config(cls, value: Any) -> Any:
        """Keep the pre-discriminator built-in parser config working."""
        if not isinstance(value, Mapping) or "type" in value:
            return value

        legacy = dict(value)
        legacy["type"] = "builtin"
        if not legacy.get("reasoning"):
            legacy["reasoning"] = "think_tag"
        if not legacy.get("tool"):
            legacy["tool"] = "qwen25"
        return legacy


# ============================================================
# E2B rollout entrypoint
# ============================================================


class E2BEntrypointConfig(_StrictModel):
    """Host-side task script and optional workspace/evaluation paths."""

    target: str = Field(..., min_length=1)
    task_files_root: str | None = None
    workspace: str | None = None

    @field_validator("target")
    @classmethod
    def _strip_target(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rollout.execution.entrypoint.target must not be blank")
        return value

    @field_validator("task_files_root", "workspace")
    @classmethod
    def _strip_optional_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_paths(self) -> "E2BEntrypointConfig":
        if not Path(self.target).is_absolute():
            raise ValueError("rollout.execution.entrypoint.target must be an absolute host path")
        if self.task_files_root is not None and not Path(self.task_files_root).is_absolute():
            raise ValueError(
                "rollout.execution.entrypoint.task_files_root must be an absolute host path"
            )
        if self.workspace is not None and not Path(self.workspace).is_absolute():
            raise ValueError("entrypoint.workspace must be an absolute sandbox path")
        if Path(self.target).suffix != ".py":
            raise ValueError("rollout.execution.entrypoint.target must point to a .py file")
        return self


# ============================================================
# training
# ============================================================


class SaveConfig(_StrictModel):
    every_n_steps: int = Field(0, ge=0, description="0 = 不周期 save（仅末尾）")
    final: bool = Field(True, description="Save the final step on normal completion")
    dir: str


class ResumeConfig(_StrictModel):
    mode: Literal["disable", "auto", "from_path"] = "disable"
    path: str = Field(
        "",
        description="Checkpoint for from_path; fallback for auto when no indexed checkpoint exists",
    )
    skip_dataloader: bool = True
    max_retries: int = Field(
        1,
        ge=0,
        description="训练异常退出后重建 controller 并从最新 checkpoint 自动恢复的次数",
    )


class WeightSyncConfig(_StrictModel):
    mode: Literal["sync", "batch-async", "fully-async", "no-sync"] = "sync"
    staleness_threshold: int = Field(0, ge=0)
    freeze_inflight: bool = True
    pause_mode: Literal["retract", "in_place"] | None = Field(
        None,
        description="Pause the full-FT Weaver sampler during fully-async weight synchronization",
    )
    sampler_ttl_s: int = Field(86400, ge=1)


class TrainingConfig(_StrictModel):
    total_steps: int | None = Field(None, ge=1)
    epochs: int | None = Field(None, ge=1)
    save: SaveConfig
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    weight_sync: WeightSyncConfig = Field(default_factory=WeightSyncConfig)

    @model_validator(mode="after")
    def _check_training_length(self) -> "TrainingConfig":
        if (self.total_steps is None) == (self.epochs is None):
            raise ValueError("exactly one of training.total_steps and training.epochs must be set")
        return self


# ============================================================
# backend
# ============================================================


class WeaverMetricsStoreConfig(_StrictModel):
    """SDK-owned local JSONL storage; does not control Trainer collection or W&B."""

    enabled: bool = Field(True, strict=True)
    path: str | None = None


class BackendConfig(_StrictModel):
    type: Literal["weaver", "mock"] = "weaver"
    performance_tier: Literal["normal", "fast", "flash"] | None = Field(
        None, description="Weaver actor 训练吞吐档位；null = 使用服务端默认值"
    )
    base_url: str | None = Field(
        None,
        description="ENV WEAVER_BASE_URL 必填；yaml 仅在 env 未设时作 fallback",
    )
    api_key: str | None = Field(
        None,
        description="ENV WEAVER_API_KEY 必填；yaml 仅在 env 未设时作 fallback",
    )
    metrics_store: WeaverMetricsStoreConfig = Field(default_factory=WeaverMetricsStoreConfig)
    scheduling_policy: dict[str, Any] | None = None
    max_retries: int = Field(3, ge=0)
    request_timeout_s: float = Field(1000.0, gt=0)


# ============================================================
# runtime
# ============================================================


class FeishuConfig(_StrictModel):
    enabled: bool = False
    url: str | None = None

    @model_validator(mode="after")
    def _check_url_when_enabled(self) -> "FeishuConfig":
        if self.enabled and not self.url:
            raise ValueError("runtime.logger.feishu.enabled=true 时 url 必填")
        return self


class LoggerConfig(_StrictModel):
    backends: list[Literal["console", "wandb", "mlflow", "swanlab"]] = Field(
        default_factory=lambda: ["console", "wandb"]
    )
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)


class HealthConfig(_StrictModel):
    exception_policy: Literal["stop_on_error", "stop_on_critical", "continue"] = "stop_on_error"
    exception_check_interval_s: float = Field(1.0, gt=0)
    memory_log_interval_s: float = Field(10.0, gt=0)


class ProgressConfig(_StrictModel):
    update_interval_s: float = Field(1.0, gt=0)
    log_every_n_steps: int = Field(10, ge=1)
    use_tqdm: bool = True


class DebugConfig(_StrictModel):
    enable_data_dump: bool = False
    dump_dir: str | None = None
    dump_every_n_steps: int = Field(1, ge=1, strict=True)
    remote_api_dump_options: dict[
        Literal["prepared_trajectories", "datums", "training_metrics"], bool
    ] = Field(default_factory=dict)
    sampling_model_path: str | None = Field(
        None,
        min_length=1,
        description="Explicit Weaver sampling checkpoint for northrl.debug_rollout; null = base model",
    )


class RuntimeConfig(_StrictModel):
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    progress: ProgressConfig = Field(default_factory=ProgressConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_launcher(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "launcher" in value:
            raise ValueError(
                "runtime.launcher is no longer supported; source environment scripts "
                "in your shell or use bash run.sh with NORTHRL_ENV_FILE"
            )
        return value


# ============================================================
# top-level
# ============================================================


class NorthRLConfig(_StrictModel):
    """Top-level NorthRL config (9 first-level blocks)."""

    # 重新声明 ConfigDict 以叠加 populate_by_name=True（让 alias `validate` 在 yaml 里依然可用）
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    experiment: ExperimentConfig
    models: ModelsConfig
    data: DataConfig
    # yaml key 仍是 `validate`；python 属性用 `validate_` 避开 pydantic BaseModel 同名属性告警
    validate_: ValidateConfig = Field(..., alias="validate")
    algorithm: AlgorithmConfig
    rollout: RolloutConfig
    training: TrainingConfig
    backend: BackendConfig
    runtime: RuntimeConfig

    # === cross-block validators ===

    @model_validator(mode="before")
    @classmethod
    def _derive_sft_fixed_values(cls, value: Any) -> Any:
        """Resolve fixed SFT values before validating nested blocks.

        SFT always uses an internal single-worker rollout config, so a user
        rollout block is warned about and replaced. Other fixed values are
        filled only when omitted so the after-validator can reject conflicts.
        SFT's obsolete ``rollout_repeat_n`` is ignored with a warning because
        SFT batches are built from a token budget instead.
        """
        if not isinstance(value, Mapping):
            return value
        resolved = dict(value)
        algorithm = value.get("algorithm")
        algorithm_type = algorithm.get("type") if isinstance(algorithm, Mapping) else None
        is_sft = algorithm_type == "sft"
        if is_sft:
            if "rollout" in value:
                warnings.warn(
                    "rollout is ignored for SFT; remove the top-level rollout block because "
                    "NorthRL always uses one internal SFT worker",
                    UserWarning,
                    stacklevel=2,
                )
            resolved["rollout"] = {"num_workers": 1}
        data = value.get("data")
        if isinstance(data, Mapping):
            resolved_data = dict(data)
            train = data.get("train")
            if isinstance(train, Mapping):
                resolved_train = dict(train)
                if is_sft and "rollout_repeat_n" in resolved_train:
                    if resolved_train["rollout_repeat_n"] is not None:
                        warnings.warn(
                            "data.train.rollout_repeat_n is not configurable for SFT and will be ignored",
                            UserWarning,
                            stacklevel=2,
                        )
                    resolved_train.pop("rollout_repeat_n", None)
                resolved_data["train"] = resolved_train
            resolved["data"] = resolved_data

        if is_sft:
            training = value.get("training")
            if isinstance(training, Mapping):
                resolved_training = dict(training)
                weight_sync = training.get("weight_sync")
                if "weight_sync" not in training:
                    resolved_training["weight_sync"] = {"mode": "no-sync"}
                elif isinstance(weight_sync, Mapping):
                    resolved_weight_sync = dict(weight_sync)
                    resolved_weight_sync.setdefault("mode", "no-sync")
                    resolved_training["weight_sync"] = resolved_weight_sync
                resolved["training"] = resolved_training

        return resolved

    @model_validator(mode="after")
    def _check_rollout_selection(self) -> "NorthRLConfig":
        if self.algorithm.type != "sft" and self.rollout.type is None:
            raise ValueError("rollout.type is required unless algorithm.type='sft'")
        if self.rollout.type != "e2b" and (
            self.rollout.sampling.reasoning_effort is not None
            or self.validate_.sampling.reasoning_effort is not None
        ):
            raise ValueError("sampling.reasoning_effort is only supported for rollout.type='e2b'")
        return self

    @model_validator(mode="after")
    def _check_rloo_rollout_count(self) -> "NorthRLConfig":
        if self.algorithm.advantage.estimator.mode == "rloo":
            repeat_n = self.data.train.rollout_repeat_n
            if repeat_n is None or repeat_n < 2:
                raise ValueError("RLOO requires data.train.rollout_repeat_n>=2")
        return self

    @model_validator(mode="after")
    def _check_reinforce_plus_plus_groups(self) -> "NorthRLConfig":
        if self.algorithm.advantage.estimator.mode == "reinforce_plus_plus_baseline":
            repeat_n = self.data.train.rollout_repeat_n
            if repeat_n is None or repeat_n < 2:
                raise ValueError(
                    "reinforce_plus_plus_baseline requires data.train.rollout_repeat_n>=2"
                )
        return self

    @model_validator(mode="after")
    def _check_sampling_pause(self) -> "NorthRLConfig":
        weight_sync = self.training.weight_sync
        if weight_sync.pause_mode is None:
            return self
        if self.algorithm.type == "sft" or weight_sync.mode != "fully-async":
            raise ValueError("training.weight_sync.pause_mode requires fully-async RL training")
        if self.backend.type != "weaver":
            raise ValueError("training.weight_sync.pause_mode requires backend.type='weaver'")
        if self.models.actor.lora is not None and self.models.actor.lora.enabled:
            raise ValueError("training.weight_sync.pause_mode requires full_ft (LoRA disabled)")
        if not weight_sync.freeze_inflight:
            raise ValueError("training.weight_sync.pause_mode requires freeze_inflight=true")
        return self

    @model_validator(mode="after")
    def _check_cispo_backend(self) -> "NorthRLConfig":
        if self.algorithm.loss.mode == "cispo" and self.backend.type != "weaver":
            raise ValueError("CISPO requires backend.type='weaver'")
        return self

    @model_validator(mode="after")
    def _check_gspo_backend(self) -> "NorthRLConfig":
        if self.algorithm.loss.mode == "gspo" and self.backend.type != "weaver":
            raise ValueError("GSPO requires backend.type='weaver'")
        return self

    @model_validator(mode="after")
    def _check_score_centering(self) -> "NorthRLConfig":
        a = self.algorithm
        if a.loss.mode == "custom" and a.loss.params["base_loss_fn"] == "score_centering":
            raise ValueError("score_centering does not support custom loss")
        if a.loss.mode != "score_centering":
            return self
        if a.type != "grpo" or a.sampling_logprob_source != "inference":
            raise ValueError("score_centering requires GRPO with sampling_logprob_source=inference")
        if a.router_replay.enabled or a.mis.enabled or a.mis.get_mismatch_metrics:
            raise ValueError("score_centering does not support router replay, MIS or custom loss")
        if a.entropy_coeff != 0 or a.kl.enabled:
            raise ValueError("score_centering requires entropy_coeff=0 and KL disabled")
        rollout = self.rollout
        if rollout.type != "simple" or rollout.custom_module_path:
            raise ValueError(
                "score_centering currently requires the built-in simple rollout worker"
            )
        sampling = self.rollout.sampling
        if (
            sampling.temperature,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.repetition_penalty,
            sampling.min_tokens,
        ) != (1, 1, -1, 0, 1, 0):
            raise ValueError("score_centering requires temperature=1 and unfiltered sampling")
        if self.backend.type != "weaver":
            raise ValueError("score_centering requires the Weaver backend")
        if self.training.weight_sync.mode != "sync":
            raise ValueError(
                "score_centering currently requires synchronous weight synchronization"
            )
        return self

    @model_validator(mode="after")
    def _check_optimizer_batch_size(self) -> "NorthRLConfig":
        if (
            self.algorithm.type != "sft"
            and self.training.weight_sync.mode not in {"sync", "batch-async", "fully-async"}
            and self.algorithm.optim.batch_size is None
        ):
            raise ValueError(
                "algorithm.optim.batch_size is required for RL unless "
                "training.weight_sync.mode is 'sync', 'batch-async', or 'fully-async'; "
                "omitted or null uses all Datums in the batch delivered by the trajectory pool"
            )
        return self

    @model_validator(mode="after")
    def _check_mis_rollout_support(self) -> "NorthRLConfig":
        if not (self.algorithm.mis.enabled or self.algorithm.mis.get_mismatch_metrics):
            return self
        if self.rollout.keep_sampling_mask:
            raise ValueError(
                "algorithm.mis correction or mismatch metrics require "
                "rollout.keep_sampling_mask=false: sampling masks are not supported"
            )
        sampling = self.rollout.sampling
        validate_mis_sampling(
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            temperature=sampling.temperature,
            repetition_penalty=sampling.repetition_penalty,
        )
        return self

    @model_validator(mode="after")
    def _check_e2b_execution(self) -> "NorthRLConfig":
        rollout = self.rollout
        if rollout.type != "e2b":
            return self
        if rollout.params:
            raise ValueError(
                "rollout.params is no longer supported for E2B; move its fields to "
                "rollout.execution and remove rollout.params"
            )
        stage_budget_fields = (
            "preparation_timeout_seconds",
            "agent_timeout_seconds",
            "agent_finalize_timeout_seconds",
            "evaluation_timeout_seconds",
        )

        publication_timeout = rollout.execution.get("gateway_publication_timeout_seconds", 300)
        if (
            isinstance(publication_timeout, bool)
            or not isinstance(publication_timeout, (int, float))
            or not 0 < publication_timeout < float("inf")
        ):
            raise ValueError(
                "rollout.execution.gateway_publication_timeout_seconds must be positive and finite"
            )
        if "timeout_retries" in rollout.execution:
            raise ValueError(
                "rollout.execution.timeout_retries is no longer supported; "
                "use env_error_retries for environment failures"
            )

        def validate_execution(options: dict[str, Any], location: str) -> dict[str, Any]:
            if options.get("result_processing", "thread") not in ("thread", "process"):
                raise ValueError(f"{location}.result_processing must be 'thread' or 'process'")
            if not isinstance(options.get("token_prefix_reuse", False), bool):
                raise ValueError(f"{location}.token_prefix_reuse must be a boolean")
            if "network_policy" in options:
                try:
                    options = {
                        **options,
                        "network_policy": normalize_network_policy(options["network_policy"]),
                    }
                except ValueError as exc:
                    raise ValueError(f"{location}.{exc}") from exc
            if "volume_mounts" in options:
                validate_volume_mounts(
                    options["volume_mounts"], location=f"{location}.volume_mounts"
                )
            entrypoint = options.get("entrypoint")
            if entrypoint is None:
                raise ValueError(f"{location}.entrypoint is required when rollout.type='e2b'")
            validated = E2BEntrypointConfig.model_validate(entrypoint)
            if "input_fields" not in options:
                raise ValueError(f"{location}.input_fields is required when rollout.type='e2b'")
            input_fields = options["input_fields"]
            if (
                not isinstance(input_fields, list)
                or any(not isinstance(field, str) or not field.strip() for field in input_fields)
                or len(set(input_fields)) != len(input_fields)
            ):
                raise ValueError(
                    f"{location}.input_fields must be a list of unique non-empty field names"
                )
            for key in ("timeout_seconds", *stage_budget_fields):
                if key not in options:
                    continue
                timeout = options[key]
                if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
                    raise ValueError(f"{location}.{key} must be a positive integer")
            retries = options.get("env_error_retries", 3)
            if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
                raise ValueError(f"{location}.env_error_retries must be a non-negative integer")
            return {
                **options,
                "entrypoint": validated.model_dump(exclude_none=True),
                "env_error_retries": retries,
            }

        execution = dict(rollout.execution)
        if execution.get("result_processing", "thread") not in ("thread", "process"):
            raise ValueError("rollout.execution.result_processing must be 'thread' or 'process'")
        token_prefix_reuse = execution.get("token_prefix_reuse", False)
        if not isinstance(token_prefix_reuse, bool):
            raise ValueError("rollout.execution.token_prefix_reuse must be a boolean")
        execution["token_prefix_reuse"] = token_prefix_reuse
        # Validate shared mounts even when every profile replaces them.
        if "volume_mounts" in execution:
            validate_volume_mounts(execution["volume_mounts"])
        if "network_policy" in execution:
            try:
                execution["network_policy"] = normalize_network_policy(execution["network_policy"])
            except ValueError as exc:
                raise ValueError(f"rollout.execution.{exc}") from exc
        execution["gateway_publication_timeout_seconds"] = publication_timeout
        if "profiles" not in execution:
            rollout.execution = validate_execution(execution, "rollout.execution")
            return self

        profiles = execution.pop("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("rollout.execution.profiles must be a nonempty mapping")
        if "entrypoint" in execution or "input_fields" in execution:
            raise ValueError(
                "rollout.execution.profiles cannot be combined with global entrypoint or input_fields"
            )
        allowed_fields = {
            "result_processing",
            "token_prefix_reuse",
            "entrypoint",
            "input_fields",
            "volume_mounts",
            "network_policy",
            "timeout_seconds",
            "env_error_retries",
            "tokenizer",
            *stage_budget_fields,
        }
        normalized_profiles = {}
        for name, profile in profiles.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError(
                    "rollout.execution.profiles names must be nonblank without boundary whitespace"
                )
            location = f"rollout.execution.profiles.{name}"
            if not isinstance(profile, dict):
                raise ValueError(f"{location} must be a mapping")
            unknown_fields = set(profile) - allowed_fields
            if unknown_fields:
                raise ValueError(
                    f"{location} contains unsupported fields: {sorted(unknown_fields)}"
                )
            validated_profile = validate_execution({**execution, **profile}, location)
            normalized_profiles[name] = {
                **profile,
                "entrypoint": validated_profile["entrypoint"],
            }
            if "network_policy" in profile:
                normalized_profiles[name]["network_policy"] = validated_profile["network_policy"]
        rollout.execution = {**execution, "profiles": normalized_profiles}
        return self

    @model_validator(mode="after")
    def _check_epoch_training_support(self) -> "NorthRLConfig":
        if self.training.epochs is not None and self.algorithm.type != "sft":
            raise ValueError("training.epochs is only supported when algorithm.type='sft'")
        return self

    @model_validator(mode="after")
    def _check_lr_schedule_support(self) -> "NorthRLConfig":
        if self.algorithm.optim.schedule is not None and self.algorithm.type != "sft":
            raise ValueError(
                "algorithm.optim.schedule is currently supported only when algorithm.type='sft'"
            )
        return self

    @model_validator(mode="after")
    def _check_kl_reference_pairing(self) -> "NorthRLConfig":
        if self.algorithm.kl.enabled and self.models.reference is None:
            raise ValueError(
                "algorithm.kl.mode 非 null 时 models.reference 必须存在（提供 ref_model 用于算 KL）"
            )
        if self.models.reference is not None and not self.algorithm.kl.enabled:
            raise ValueError(
                "models.reference 已配置但 algorithm.kl.mode=null——配了不用，删掉 models.reference 或打开 KL"
            )
        return self

    @model_validator(mode="after")
    def _check_opd_teachers_pairing(self) -> "NorthRLConfig":
        if self.algorithm.type == "opd" and not self.models.teachers:
            raise ValueError("algorithm.type=opd 时 models.teachers 必须非空")
        if self.models.teachers and self.algorithm.type != "opd":
            raise ValueError(
                f"models.teachers 已配置但 algorithm.type={self.algorithm.type!r}（非 opd）——配了不用"
            )
        return self

    @model_validator(mode="after")
    def _check_max_tokens_upper_bound(self) -> "NorthRLConfig":
        if self.algorithm.type == "sft":
            return self
        if self.data.max_prompt_length is None:
            raise ValueError("data.max_prompt_length is required unless algorithm.type='sft'")
        max_gen = self.data.max_sequence_length - self.data.max_prompt_length
        for label, mt in (
            ("rollout.sampling.max_tokens", self.rollout.sampling.max_tokens),
            ("validate.sampling.max_tokens", self.validate_.sampling.max_tokens),
        ):
            if mt is not None and mt > max_gen:
                raise ValueError(
                    f"{label} ({mt}) 不能超过 data.max_sequence_length - data.max_prompt_length "
                    f"({max_gen})"
                )
        return self

    @model_validator(mode="after")
    def _check_custom_loss_entropy_zero(self) -> "NorthRLConfig":
        if self.algorithm.loss.mode == "custom" and self.algorithm.entropy_coeff != 0:
            raise ValueError(
                "algorithm.loss.mode='custom' 时 algorithm.entropy_coeff 必须为 0 "
                "（custom_loss 路径不接 entropy 项，写非 0 会被默默忽略）"
            )
        return self

    @model_validator(mode="after")
    def _check_tokenizer_requirements(self) -> "NorthRLConfig":
        rollout = self.rollout
        if rollout.type is None:
            return self
        # filter_prompts=True 必须有本地 tokenizer（dataloader 长度过滤要用）
        if self.data.train.filter_prompts and not self.models.actor.tokenizer:
            raise ValueError(
                "data.train.filter_prompts=true 时 models.actor.tokenizer 必填 "
                "（dataloader 要用 tokenizer 算 prompt 长度做过滤）"
            )
        return self

    @model_validator(mode="after")
    def _check_sft_sampler_free_pipeline(self) -> "NorthRLConfig":
        # Let the dedicated router-replay validator below report its more
        # specific incompatibility before these generic pipeline invariants.
        if self.algorithm.router_replay.enabled:
            return self
        if self.algorithm.type != "sft":
            return self
        if self.training.weight_sync.mode != "no-sync":
            raise ValueError(
                "algorithm.type='sft' requires "
                "training.weight_sync.mode='no-sync' because this pipeline does not "
                "provision an inference sampler"
            )
        if self.validate_.enabled:
            raise ValueError(
                "algorithm.type='sft' requires validate.enabled=false; "
                "the current validation path requires an inference sampler"
            )
        return self

    @model_validator(mode="after")
    def _check_sft_data_options(self) -> "NorthRLConfig":
        train = self.data.train
        is_sft = self.algorithm.type == "sft"
        if not is_sft:
            if train.render_chat is not None:
                raise ValueError("data.train.render_chat is only valid when algorithm.type='sft'")
            if train.window_samples is not None:
                raise ValueError(
                    "data.train.window_samples is only valid when algorithm.type='sft'"
                )
            if train.sources is not None:
                if any(isinstance(source, ManagedTrainSourceConfig) for source in train.sources):
                    raise ValueError("managed sources are supported only when algorithm.type='sft'")
                if self.algorithm.type != "grpo":
                    raise ValueError("local RL data.train.sources requires algorithm.type='grpo'")
                if train.total_samples is not None:
                    raise ValueError(
                        "data.train.total_samples is only valid for SFT; use training.total_steps for RL"
                    )
                names: set[str] = set()
                profiles = self.rollout.execution.get("profiles", {})
                for source in train.sources:
                    if not isinstance(source, LocalTrainSourceConfig):
                        continue
                    if source.name is None:
                        raise ValueError("each RL data.train.sources entry requires a name")
                    if source.name in names:
                        raise ValueError(f"duplicate RL source name: {source.name!r}")
                    names.add(source.name)
                    if source.rollout_profile is not None:
                        if self.rollout.type != "e2b" or source.rollout_profile not in profiles:
                            raise ValueError(
                                f"source {source.name!r} references unknown E2B rollout_profile {source.rollout_profile!r}"
                            )
            if train.rollout_repeat_n is None:
                raise ValueError("data.train.rollout_repeat_n is required outside SFT")
            return self

        if train.sources is not None:
            if train.total_samples is None:
                raise ValueError("data.train.total_samples is required with SFT data.train.sources")
            if any(
                isinstance(source, LocalTrainSourceConfig)
                and (source.name is not None or source.rollout_profile is not None)
                for source in train.sources
            ):
                raise ValueError("local source name and rollout_profile are only valid for RL")

        # Weaver binds this complete pre-shift token limit to the model. A
        # causal training sample needs at least two tokens to yield one
        # positive input_token_count after autoregressive shifting.
        if self.data.max_sequence_length < 2:
            raise ValueError("algorithm.type='sft' requires data.max_sequence_length >= 2")

        # Router replay has a dedicated validator with a more actionable
        # compatibility error. Let it report that error first.
        if self.algorithm.router_replay.enabled and self.algorithm.router_replay.mode == "R3":
            return self

        if self.algorithm.loss.mode != "cross_entropy":
            raise ValueError("algorithm.type='sft' requires algorithm.loss.mode='cross_entropy'")
        if self.backend.type != "weaver":
            raise ValueError("algorithm.type='sft' requires backend.type='weaver'")
        if not train.shuffle and train.sources is not None:
            raise ValueError(
                "Mixed SFT requires data.train.shuffle=true because global "
                "deterministic shuffle is part of the versioned mixing contract"
            )
        if train.window_samples is not None and train.sources is not None:
            raise ValueError("data.train.window_samples is supported only with data.train.files")
        if not train.drop_last:
            raise ValueError("SFT requires data.train.drop_last=true")
        if train.filter_prompts:
            raise ValueError(
                "SFT does not support data.train.filter_prompts; "
                "overlong samples are truncated to data.max_sequence_length"
            )
        local_files = list(train.files or [])
        if train.sources is not None:
            local_files.extend(
                file_name
                for source in train.sources
                if isinstance(source, LocalTrainSourceConfig)
                for file_name in source.files
            )
        unsupported = [
            file_name for file_name in local_files if Path(file_name).suffix.lower() != ".jsonl"
        ]
        if unsupported:
            raise ValueError(
                "SFT streaming supports only JSONL files; " f"unsupported={unsupported}"
            )
        if train.render_chat is not None and train.sources is not None and not local_files:
            raise ValueError(
                "data.train.render_chat applies only to local sources and cannot be set "
                "for a managed-only mixture"
            )
        return self

    @model_validator(mode="after")
    def _check_router_replay_rollout_support(self) -> "NorthRLConfig":
        rr = self.algorithm.router_replay
        if not rr.enabled:
            return self
        if self.algorithm.type != "grpo":
            raise ValueError(
                "algorithm.router_replay.enabled=true requires algorithm.type='grpo'; "
                f"algorithm.type={self.algorithm.type!r} does not consume replay metadata."
            )
        if self.backend.type != "weaver":
            raise ValueError(
                "algorithm.router_replay.enabled=true requires backend.type='weaver'; "
                f"backend.type={self.backend.type!r} has no router-replay transport."
            )
        if rr.mode != "R3":
            return self

        # R3 requires a rollout worker that propagates the sampler-produced
        # ``moe_topk_indices_ref`` onto every trajectory. Other built-ins and
        # arbitrary custom workers do not declare that capability, so reject
        # them here rather than failing after rollout at the training boundary.
        supported_r3_workers = {"simple", "single_turn"}
        rollout = self.rollout
        if rollout.type is None:
            raise ValueError("rollout.type is required for R3 router replay")
        if rollout.custom_module_path or rollout.type not in supported_r3_workers:
            raise ValueError(
                "algorithm.router_replay.mode=R3 requires a rollout worker that "
                "propagates moe_topk_indices_ref; supported rollout.type values "
                f"are {sorted(supported_r3_workers)}, got {rollout.type!r}."
            )
        return self


# ============================================================
# loader
# ============================================================


def load_yaml_config(path: str | Path) -> NorthRLConfig:
    """Load and validate a yaml against the NorthRL schema.

    Raises pydantic.ValidationError on any schema violation, with a structured
    list of error locations / messages.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return NorthRLConfig.model_validate(raw)


# Resolve forward refs introduced by `from __future__ import annotations`.
# Must come AFTER all sub-models are defined.
NorthRLConfig.model_rebuild()
