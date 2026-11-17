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
Pytest configuration and shared fixtures.

All fixtures now produce configs shaped per the new 9-block schema
(see CONFIG_REFACTOR.md). Workers / trainers / dataloaders read root
NorthRLConfig directly post-refactor; these fixtures provide minimal
root configs that satisfy the field paths each unit test touches.
"""

import pytest
from omegaconf import OmegaConf


def _make_root_config(**overrides):
    """Build a minimal root NorthRLConfig shape with optional per-block overrides.

    Only the field paths actually consumed by the unit tests are populated;
    schema validation is intentionally not run on these fixtures (most tests
    exercise narrow code paths and don't need the full schema to be valid).
    """
    cfg = {
        "experiment": {"name": "test_experiment", "project": "test_project", "seed": 42},
        "models": {
            "actor": {
                "name": "mock_model",
                "tokenizer": None,
                "checkpoint": None,
                "lora": None,
            },
            "reference": None,
            "teachers": None,
        },
        "validate": {
            "enabled": False,
            "data": {"datasets": {"default": {"files": ["test_eval.parquet"], "batch_size": 4}}},
            "sampling": {},
            "schedule": {},
        },
        "data": {
            "train": {
                "files": ["test.parquet"],
                "batch_size": 4,
                "rollout_repeat_n": 1,
                "prompt_key": "prompt",
                "filter_prompts": False,
                "shuffle": False,
                "drop_last": False,
            },
            "max_prompt_length": 256,
            "max_sequence_length": 1024,
        },
        "algorithm": {
            "type": "grpo",
            "sampling_logprob_source": "recomputed",
            "optim": {
                "lr": 1.0e-4,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1.0e-8,
                "weight_decay": 0.01,
                "grad_clip_norm": 1.0,
                "batch_size": 4,
                "mini_batch_size": None,
            },
            "entropy_coeff": 0.0,
            "kl": {"mode": None},
            "loss": {
                "mode": "importance_sampling",
                "params": {"clip": {}},
                "reduction": {"mode": "token_mean"},
            },
        },
        "rollout": {
            "num_workers": 1,
            "sampling": {
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "max_tokens": 512,
            },
            "generation_extra": {},
            "parsers": {"tool": None, "reasoning": None},
            "reward_fn": "math_with_format",
            "type": "single_turn",
            "params": {},
        },
        "training": {
            "total_steps": 1,
            "save": {"every_n_steps": 0, "dir": "/tmp/test_ckpt"},
            "resume": {"mode": "disable", "path": "", "skip_dataloader": True},
            "weight_sync": {
                "mode": "sync",
                "staleness_threshold": 0,
                "freeze_inflight": False,
                "sampler_ttl_s": 86400,
            },
        },
        "backend": {
            "type": "mock",
            "base_url": "http://localhost:8000",
            "api_key": "mock_key",
            "max_retries": 1,
            "request_timeout_s": 30,
        },
        "runtime": {
            "logger": {"backends": ["console"], "feishu": {"enabled": False, "url": None}},
            "health": {
                "exception_policy": "stop_on_error",
                "exception_check_interval_s": 1.0,
                "memory_log_interval_s": 10.0,
            },
            "progress": {
                "update_interval_s": 1.0,
                "log_every_n_steps": 10,
                "use_tqdm": False,
            },
            "debug": {"enable_data_dump": False, "dump_dir": None},
        },
    }
    for block, values in overrides.items():
        cfg.setdefault(block, {}).update(values)
    return OmegaConf.create(cfg)


@pytest.fixture
def basic_config():
    """Minimal config kept for tests that only read a couple of fields."""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "vocab_size": 151936,
            "max_prompt_length": 256,
            "max_response_length": 128,
            "model_tag": "test_model",
        }
    )


@pytest.fixture
def data_loader_config():
    """Sub-config shape that MockDataLoader and TorchDataLoader accept directly.

    Matches the dict produced by ``Controller._derive_train_dataloader_config``
    in the production path.
    """
    return OmegaConf.create(
        {
            "files": ["test.parquet"],
            "batch_size": 4,
            "rollout_repeat_n": 1,
            "prompt_key": "prompt",
            "filter_prompts": False,
            "shuffle": False,
            "drop_last": False,
            "max_prompt_length": 256,
            "tokenizer": None,
            "seed": 42,
            # Production step-based RL derives True; keep this generic fixture finite.
            "cycle": False,
            "keep_batch_order": True,
            # MockDataLoader-only
            "mock_api_type": "completion",
            "num_workers": 1,
        }
    )


@pytest.fixture
def rollout_worker_config():
    """Root NorthRLConfig for rollout-worker tests.

    Post-refactor, workers read the root config directly (see Phase 2.A);
    the fixture must provide ``data.max_sequence_length``, ``algorithm.type``,
    ``backend.type``, ``rollout.sampling.*``, etc.
    """
    return _make_root_config()


@pytest.fixture
def train_batch_pool_config():
    """Configuration for train batch pool testing."""
    return OmegaConf.create({"max_batches": 10, "model_tag": "test_model"})
