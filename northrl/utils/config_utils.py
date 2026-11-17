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
Configuration utilities for handling OmegaConf DictConfig operations.
"""

import logging
import os
from typing import NotRequired, TypedDict

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

SAMPLING_LOGPROB_SOURCE_INFERENCE = "inference"
SAMPLING_LOGPROB_SOURCE_RECOMPUTED = "recomputed"


def dump_resolved_config(config: DictConfig) -> None:
    """
    Dump the fully resolved configuration to EXPERIMENT_PATH for reproducibility.

    The resolved config will have all OmegaConf interpolations (${...}) expanded to their
    actual values, making it suitable for direct use as a standalone experiment config.

    Args:
        config: The DictConfig to dump (should already be resolved)

    The config will be saved to: ${EXPERIMENT_PATH}/train_config_resolved.yaml
    """
    experiment_path = os.environ.get("EXPERIMENT_PATH")
    if experiment_path:
        resolved_config_path = os.path.join(experiment_path, "train_config_resolved.yaml")
        try:
            os.makedirs(experiment_path, exist_ok=True)
            with open(resolved_config_path, "w", encoding="utf-8") as f:
                # Save fully resolved config as YAML
                # This config can be used directly as another experiment config
                OmegaConf.save(config, f, resolve=True)
            logger.info(f"Saved resolved config to: {resolved_config_path}")
        except Exception as e:
            logger.warning(f"Failed to save resolved config: {e}")
    else:
        logger.warning("EXPERIMENT_PATH not set, skipping resolved config dump")


def validate_mis_sampling(
    *,
    top_p: float,
    top_k: int,
    min_p: float,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
) -> None:
    """Require the sampling convention used for correction and mismatch diagnostics.

    Recompute and training have not been verified to apply the same sampling
    mask. Sampler old_logprobs semantics under temperature or repetition
    transforms are also unverified; these are current integration limits.
    """
    if top_p != 1.0 or top_k > 0 or min_p != 0.0:
        raise ValueError(
            "algorithm.mis correction or mismatch metrics require top_p=1, top_k<=0, and min_p=0: "
            "recompute and training sampling_mask semantics have not been verified to match"
        )
    if temperature != 1.0 or repetition_penalty != 1.0:
        raise ValueError(
            "algorithm.mis correction or mismatch metrics require temperature=1 and repetition_penalty=1: "
            "sampler old_logprobs semantics for modified sampling distributions "
            "have not been verified"
        )


class SamplingFlags(TypedDict):
    """Keyword arguments for optional sampler outputs."""

    return_sampling_mask: bool
    return_old_logprob: bool
    return_moe_topk_indices: bool
    topk_output_logprobs: NotRequired[int]


def get_algorithm_loss_mode(config: DictConfig) -> str:
    """Read the selected objective from resolved canonical configuration."""
    return str(config.algorithm.loss.mode)


def get_derived_sampling_flags(config: DictConfig) -> SamplingFlags:
    """Derive rollout sampling API flags from explicit mask retention and training needs.

    New schema (see CONFIG_REFACTOR.md):
      sampling_logprob_source lives at algorithm.sampling_logprob_source;
      sampling mask retention is controlled by rollout.keep_sampling_mask.
    """
    return_sampling_mask = bool(getattr(config.rollout, "keep_sampling_mask", False))
    source = config.algorithm.sampling_logprob_source
    mis = getattr(config.algorithm, "mis", None)
    needs_rollout_logprobs = bool(
        mis is not None
        and (getattr(mis, "enabled", False) or getattr(mis, "get_mismatch_metrics", False))
    )
    # R3 router replay replays the sampler's MoE routing during the training
    # forward, so the sampler must offload and return the routing ref. R2 records
    # routing trainer-side during recompute and needs no sampler flag.
    router_replay = getattr(config.algorithm, "router_replay", None)
    return_moe_topk_indices = bool(
        router_replay is not None
        and getattr(router_replay, "enabled", False)
        and str(getattr(router_replay, "mode", "R3")) == "R3"
    )
    if get_algorithm_loss_mode(config) == "score_centering":
        topk = config.algorithm.loss.params.get("topk", 128)
        return {
            "return_sampling_mask": return_sampling_mask,
            "return_old_logprob": False,
            "return_moe_topk_indices": False,
            "topk_output_logprobs": topk,
        }
    return {
        "return_sampling_mask": return_sampling_mask,
        # Correction and mismatch diagnostics retain the sampler's probability
        # alongside separately evaluated trainer logprobs, regardless of the
        # denominator source used by the loss.
        "return_old_logprob": source == SAMPLING_LOGPROB_SOURCE_INFERENCE or needs_rollout_logprobs,
        "return_moe_topk_indices": return_moe_topk_indices,
    }


def use_weaver(config: DictConfig) -> bool:
    """
    Check if Weaver is used in the configuration.
    """
    return config.backend.type == "weaver"
