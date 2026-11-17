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

"""Run one rollout batch and save the workers' existing completed-result JSON traces."""

import argparse
import asyncio
import copy
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .utils.async_utils import await_coro
from .utils.config_schema import NorthRLConfig
from .utils.config_utils import get_derived_sampling_flags

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a training recipe, apply optional dotted overrides, and validate its schema."""
    path = Path(config_path).resolve()
    raw = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides or []))
    validated = NorthRLConfig.model_validate(OmegaConf.to_container(raw, resolve=True))
    config = OmegaConf.create(validated.model_dump(by_alias=True))
    config._config_file_path = str(path)  # pylint: disable=protected-access
    return config


def select_debug_batch(config: DictConfig) -> list[dict[str, Any]]:
    """Use the production loader's first batch, including filtering, shuffle and repeats."""
    from .data_loader import TorchDataLoader

    train = config.data.train
    loader_config = OmegaConf.create(OmegaConf.to_container(train, resolve=True))
    loader_config.max_prompt_length = config.data.max_prompt_length
    loader_config.tokenizer = config.models.actor.tokenizer
    loader_config.seed = config.experiment.seed
    loader_config.cycle = False
    loader_config.keep_batch_order = True
    loader = TorchDataLoader(loader_config)
    tasks = []
    try:
        for _ in range(train.batch_size * train.rollout_repeat_n):
            task = loader.get_next_item()
            if task is None:
                break
            tasks.append(copy.deepcopy(task))
    finally:
        loader.close()
    return tasks


def create_debug_holder(config: DictConfig) -> Any:
    """Provision only a sampler, keeping training clients and updates out of this run."""
    if config.backend.type == "mock":
        return None
    from .weaver.weaver_service_holder import WeaverServiceHolder

    api_key = os.environ.get("WEAVER_API_KEY") or config.backend.api_key
    if not api_key:
        raise ValueError("Set WEAVER_API_KEY or backend.api_key before running rollout debug")
    os.environ["WEAVER_API_KEY"] = api_key
    scheduling_policy = config.backend.get("scheduling_policy")
    if scheduling_policy is not None:
        scheduling_policy = OmegaConf.to_container(scheduling_policy, resolve=True)
    return WeaverServiceHolder(
        base_model=config.models.actor.name,
        tokenizer_path=config.models.actor.tokenizer,
        base_url=os.environ.get("WEAVER_BASE_URL") or config.backend.base_url,
        sampling_only=True,
        model_path=config.runtime.debug.sampling_model_path,
        max_sequence_length=config.data.max_sequence_length,
        scheduling_policy=scheduling_policy,
    )


class _TrajectorySink:
    """Accept results after the worker has saved its existing rollout trace."""

    def __init__(self) -> None:
        self.status: str | None = None

    def put_trajectory(self, trajectory: Any) -> str:
        """Keep the completed outcome for the debug summary."""
        self.status = trajectory.status
        return "success" if self.status == "success" else "fail"


async def run_debug_rollout(config: DictConfig, output_dir: Path) -> dict[str, int]:
    """Run one worker sequentially over one batch; log failures and close the sampler."""
    rollout = config.rollout
    if config.algorithm.type == "sft":
        raise ValueError("SFT does not generate rollouts")
    if rollout.custom_module_path or rollout.type not in {"single_turn", "simple"}:
        raise ValueError("Rollout debug supports local single_turn and simple workers")
    sampling_path = config.runtime.debug.sampling_model_path
    if sampling_path and (config.backend.type != "weaver" or not config.models.actor.tokenizer):
        raise ValueError(
            "A sampling checkpoint requires backend.type=weaver and models.actor.tokenizer"
        )
    resume = config.training.resume
    index = Path(config.training.save.dir) / "weaver_checkpoints.jsonl"
    has_resume = resume.mode == "from_path" or (
        resume.mode == "auto"
        and (bool(resume.path) or (index.is_file() and index.stat().st_size > 0))
    )
    if not sampling_path and (config.models.actor.checkpoint or has_resume):
        raise ValueError(
            "Use runtime.debug.sampling_model_path for saved sampling weights, or clear "
            "models.actor.checkpoint and set training.resume.mode=disable for the base model"
        )

    from .controller import seed_everything
    from .rollout_worker import SimpleRolloutWorker, SingleTurnRolloutWorker

    config = copy.deepcopy(config)
    config.runtime.debug.enable_data_dump = True
    config.runtime.debug.dump_dir = str(Path(output_dir).resolve())
    config.training.weight_sync.freeze_inflight = False
    seed_everything(config.experiment.seed)
    tasks = select_debug_batch(config)
    if not tasks:
        raise ValueError("First batch is empty; check data files, filter_prompts and drop_last")

    holder = None
    previous_api_key = os.environ.get("WEAVER_API_KEY")
    counts = {"success": 0, "skipped": 0, "failed": 0}
    try:
        holder = create_debug_holder(config)
        if holder is not None:
            await await_coro(holder.configure_sampling_flags(**get_derived_sampling_flags(config)))
        worker_types: dict[str, type] = {
            "single_turn": SingleTurnRolloutWorker,
            "simple": SimpleRolloutWorker,
        }
        worker = worker_types[rollout.type](config)
        worker.init_inference_service_client(holder)
        # Use the same complete-outcome publication path as normal rollout workers.
        sink = _TrajectorySink()
        worker.set_module_references(sink, None, None, None, sink)
        worker.set_module_name("debug-rollout")
        for task in tasks:
            sink.status = None
            try:
                result = await worker.execute_rollout(task)
                status = sink.status or ("success" if result == "success" else "skipped")
                counts[status] += 1
            except Exception:  # pylint: disable=broad-exception-caught
                counts["failed"] += 1
                logger.exception(
                    "Rollout failed for task %s, run %s", task.get("task_id"), task.get("run_id")
                )
    finally:
        try:
            if holder is not None:
                await await_coro(holder.close())
        finally:
            if previous_api_key is None:
                os.environ.pop("WEAVER_API_KEY", None)
            else:
                os.environ["WEAVER_API_KEY"] = previous_api_key
    return counts


def main(argv: list[str] | None = None) -> int:
    """Accept one training YAML and optional key=value overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_name = "rollout-" + uuid.uuid4().hex[:8]
    default_dir = Path.cwd() / "debug_dump" / run_name
    previous_experiment_path = os.environ.get("EXPERIMENT_PATH")
    os.environ.setdefault("EXPERIMENT_PATH", str(default_dir))
    try:
        config = load_config(args.config, args.overrides)
        output_dir = (
            Path(config.runtime.debug.dump_dir).expanduser() / run_name
            if config.runtime.debug.dump_dir
            else default_dir
        ).resolve()
        logger.info("Rollout traces: %s", output_dir / "rollout_traces")
        counts = asyncio.run(run_debug_rollout(config, output_dir))
        print(f"{counts}; rollout traces: {output_dir / 'rollout_traces'}")
        return int(counts["failed"] > 0 or counts["success"] == 0)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Rollout debug failed")
        return 1
    finally:
        if previous_experiment_path is None:
            os.environ.pop("EXPERIMENT_PATH", None)


if __name__ == "__main__":
    raise SystemExit(main())
