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

"""CPU replay of saved or synthetic complete RL trajectories; no clients or GPU work.

Select the source under test via PYTHONPATH. The loader is the existing
train-only replay kit. --synthetic needs no saved data or loader; restoration
and equivalence hashing are outside timers.
"""

# This harness intentionally measures the production trainer phases directly.
# pylint: disable=protected-access

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import resource
import struct
import sys
import time
from array import array
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from northrl.algorithm.advantage import build_estimator, build_postprocessor
from northrl.algorithm.advantage.tool_error_rebalance import clear_postprocessing_cache
from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.trainer.router_replay_handler import RouterReplayHandler
from northrl.utils.config_schema import AlgorithmConfig
from northrl.utils.logging_utils import log_rollout_metrics
from northrl.utils.step_metrics import log_trainable_metrics
from northrl.weaver.loss_adapter import build_loss_adapter


def digest(value):
    hasher = hashlib.sha256()

    def add(item):
        if isinstance(item, Trajectory):
            add(vars(item))
        elif isinstance(item, array):
            hasher.update(item.typecode.encode())
            hasher.update(memoryview(item).cast("B"))
        elif isinstance(item, torch.Tensor):
            hasher.update(str(item.dtype).encode())
            hasher.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            hasher.update(struct.pack("!Q", len(item)))
            for part in item:
                add(part)
        elif isinstance(item, float):
            hasher.update(struct.pack("!d", item))
        else:
            hasher.update(json.dumps(item, sort_keys=True).encode())

    add(value)
    return hasher.hexdigest()


def synthetic_batch(args):
    """Make independent compact segments with complete four-parent groups."""
    trajectories = []
    length = args.tokens_per_sequence
    prompt = length - args.response_tokens_per_sequence
    for parent in range(args.parents):
        segments = []
        for segment in range(args.segments_per_parent):
            error = (parent + segment) % 3 == 0
            segments.append(
                {
                    "tokens": array("i", [17]) * length,
                    "loss_mask": array("b", [0]) * prompt
                    + array("b", [1]) * args.response_tokens_per_sequence,
                    "old_log_probs": array("d", [0.0]) + array("d", [-0.2]) * (length - 1),
                    "turn_annotations": [
                        {
                            "version": 1,
                            "turn_id": f"turn-{parent}-{segment}",
                            "response_span": [prompt, length],
                            "source": "northgate",
                            "finish_reason": "stop",
                            "truncated": False,
                            "tools_declared": True,
                            "diagnostics": {
                                "checked": True,
                                "malformed": error,
                                "invalid": int(error),
                                "reasons": ["missing_tool_name"] if error else [],
                            },
                        }
                    ],
                }
            )
        trajectories.append(
            Trajectory(
                segments=segments,
                reward=float(parent % 4),
                extra_fields={
                    "model_tag": "actor",
                    "group_id": parent // 4,
                    "task_id": parent // 4,
                    "run_id": parent % 4,
                    "expected_rollouts": 4,
                    "rollout_id": f"rollout-{parent}",
                },
            )
        )
    return trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--parents", type=int, default=64)
    parser.add_argument("--segments-per-parent", type=int, default=4)
    parser.add_argument("--tokens-per-sequence", type=int, default=32768)
    parser.add_argument("--response-tokens-per-sequence", type=int, default=16384)
    parser.add_argument("--estimator", default=None)
    parser.add_argument("--postprocessing", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.config and any(
        value is not None for value in (args.estimator, args.postprocessing, args.gamma)
    ):
        parser.error("Use --config or the algorithm selector flags, not both")
    if args.synthetic:
        if args.manifest or args.loader_dir:
            parser.error("--synthetic cannot use --manifest or --loader-dir")
        if args.parents < 4 or args.parents % 4 or args.segments_per_parent < 1:
            parser.error(
                "Synthetic data requires complete groups of four and positive segment counts"
            )
        if not 0 < args.response_tokens_per_sequence < args.tokens_per_sequence:
            parser.error("Synthetic sequences require both context and response tokens")
        entries = [{"input_stage": "synthetic_rollout"}]

        def load_batch(_entry):
            return synthetic_batch(args)

    else:
        if not all((args.config, args.manifest, args.loader_dir)):
            parser.error("Saved replay requires --config, --manifest and --loader-dir")
        sys.path.insert(0, str(args.loader_dir))
        from train_only_probe import (  # pylint: disable=import-outside-toplevel
            load_batch,
            read_manifest,
        )

        entries = read_manifest(args.manifest)["batches"]

    if torch.cuda.is_available():
        raise RuntimeError("Hide CUDA for CPU replay")
    logging.basicConfig(level=logging.WARNING)
    config = (
        OmegaConf.load(args.config)
        if args.config
        else OmegaConf.create(
            {
                "algorithm": {
                    "type": "grpo",
                    "optim": {"lr": 1e-6},
                    "advantage": {
                        "estimator": {
                            "mode": args.estimator or "grpo",
                            "params": {"gamma": args.gamma} if args.gamma is not None else {},
                        },
                        "postprocessing": {"mode": args.postprocessing},
                    },
                    "loss": {
                        "mode": "ppo_clip",
                        "params": {"clip": {"ratio": 0.2}},
                        "reduction": {"mode": "token_mean"},
                    },
                }
            }
        )
    )
    algorithm = AlgorithmConfig.model_validate(
        OmegaConf.to_container(config.algorithm, resolve=True)
    )
    if algorithm.type != "grpo" or algorithm.kl.enabled:
        parser.error("This CPU replay requires the GRPO training flow and KL disabled")
    trainer = RemoteApiGrpoTrainer.__new__(RemoteApiGrpoTrainer)
    trainer._kl_enabled = False
    trainer._algorithm_config = algorithm
    trainer._advantage_estimator = build_estimator(algorithm.advantage.estimator)
    # Feature branches add concrete postprocessors to this same factory.
    postprocessor = build_postprocessor(  # pylint: disable=assignment-from-none
        algorithm.advantage.postprocessing
    )
    adapter = build_loss_adapter(algorithm)
    trainer._loss_agg_mode = adapter.runtime_reduction
    trainer._loss_fn = adapter.backend_name
    trainer._train_step = 0
    trainer._router_replay = RouterReplayHandler(config.algorithm)
    records = []
    for repeat in range(args.repeat):
        for index, entry in enumerate(entries):
            start = time.perf_counter()
            trajectories = load_batch(entry)
            restore = time.perf_counter() - start
            input_hash = digest(trajectories)
            metrics = {}
            start = time.perf_counter()
            log_rollout_metrics(trajectories, metrics)
            rollout_metrics = time.perf_counter() - start
            start = time.perf_counter()
            trajectories = asyncio.run(trainer._prepare_trajectories(trajectories, metrics))
            inputs = asyncio.run(trainer._prepare_advantage_inputs(trajectories, metrics))
            preparation = time.perf_counter() - start
            start = time.perf_counter()
            clear_postprocessing_cache(trajectories)
            trajectories = trainer._advantage_estimator.estimate(trajectories, inputs, metrics)
            estimation = time.perf_counter() - start
            start = time.perf_counter()
            if postprocessor is not None:
                trajectories = postprocessor.process(trajectories, metrics)
            postprocessing = time.perf_counter() - start
            del inputs
            start = time.perf_counter()
            datums = trainer._convert_to_datums(trajectories)
            conversion = time.perf_counter() - start
            payload_hash = digest(datums)
            start = time.perf_counter()
            counts = log_trainable_metrics(
                trajectories, datums, metrics, datum_sources=trainer._last_datum_sources
            )
            trainable_metrics = time.perf_counter() - start
            start = time.perf_counter()
            normalized = trainer._normalize_advantages_for_sum(datums, token_counts=counts)
            normalization = time.perf_counter() - start
            row = {
                "repeat": repeat,
                "batch": index,
                "input_stage": entry["input_stage"],
                "source": str(Path(sys.modules["northrl"].__file__).resolve()),
                "estimator": algorithm.advantage.estimator.mode,
                "postprocessing": algorithm.advantage.postprocessing.mode,
                "parents": len(trajectories),
                "datums": len(datums),
                "input_tokens": sum(len(d["input_tokens"]) for d in datums),
                "restore_s": restore,
                "rollout_metrics_s": rollout_metrics,
                "trainable_metrics_s": trainable_metrics,
                "input_sha256": input_hash,
                "prepared_sha256": digest(trajectories),
                "token_counts_sha256": digest(counts),
                "prepare_s": preparation + estimation,
                "prepare_inputs_s": preparation,
                "estimate_s": estimation,
                "postprocess_s": postprocessing,
                "conversion_s": conversion,
                "normalization_s": normalization,
                "payload_sha256": payload_hash,
                "normalized_sha256": digest(normalized),
                "advantages_sha256": digest([t["advantage"] for t in trajectories]),
                "sources_sha256": digest(trainer._last_datum_sources),
                "metrics": metrics,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "torch_threads": torch.get_num_threads(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "host": platform.node(),
                "thread_env": {
                    key: os.getenv(key)
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            }
            records.append(row)
            args.output.write_text(json.dumps(records, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
            del trajectories, datums, normalized, metrics


if __name__ == "__main__":
    main()
