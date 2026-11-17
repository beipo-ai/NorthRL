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

"""Local CPU replay of mixed, bursty result arrivals with real packed-array IPC.

Optional --results-dir loads saved result.json.gz artifacts without network access.
Otherwise use synthetic single/independent/prefix-chain segments and terminal
outcomes. Expanded sizes, not compressed byte counts, define the fixture scale.
Select the NorthRL and SDK sources via PYTHONPATH. This is not a Server/GPU replay.
"""

# Exercise the existing adapters and foreground builder without creating clients.
# pylint: disable=protected-access

import argparse
import asyncio
import hashlib
import json
import logging
import platform
import statistics
import struct
import sys
import tempfile
import time
import tomllib
from array import array
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
import weaver
from omegaconf import OmegaConf
from pydantic import BaseModel

from northrl.northrl_types import Trajectory
from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.contracts import RolloutInput
from northrl.remote_rollout.result_processing import result_processing, result_size_fields
from northrl.remote_rollout.trace_processing import trace_snapshot_writer
from northrl.remote_rollout.wire import (
    CompactRolloutResult,
    CompactRolloutTrace,
    decode_result,
    dump_artifact,
    load_result_gzip,
)
from northrl.rollout_worker.e2b_rollout_worker import E2BRolloutWorker
from northrl.trajectory_pool import GroupedTrajectoryStore
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums
from northrl.utils.profiling import result_profiling_enabled
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


def digest(value):
    """All fields and packed bytes, evaluated only outside measured intervals."""
    hasher = hashlib.sha256()

    def add(item):
        if isinstance(item, BaseModel):
            add({key: getattr(item, key) for key in type(item).model_fields})
        elif isinstance(item, Trajectory):
            add(vars(item))
        elif isinstance(item, array):
            hasher.update(item.typecode.encode())
            hasher.update(memoryview(item).cast("B"))
        elif isinstance(item, torch.Tensor):
            add(str(item.dtype))
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


def synthetic_results(scale):
    specs = [
        ("single", [62870], "success"),
        ("prefix-chain", [18725, 37450, 56175, 74900], "success"),
        ("independent", [65536] * 4, "success"),
        ("sampling-mask", [4096], "success"),
        ("task-failed", [32768, 98304], "task_failed"),
        ("env-error", [], "env_error"),
    ]
    payloads = []
    for name, sizes, status in specs:
        segments = []
        previous = 0
        for index, size in enumerate(sizes):
            length = max(4, round(size * scale))
            prefix = min(previous, length // 3) if name == "prefix-chain" and index else 0
            start = max(prefix, length // 2, 1)
            segments.append(
                {
                    "base_segment": index - 1 if prefix else None,
                    "prefix_len": prefix,
                    "suffix_tokens": [(i * 7919 + index) % 32000 for i in range(length - prefix)],
                    "response_spans": [[start, length]],
                    "response_log_probs": [-0.125] * (length - start),
                    "old_response_log_probs": {"ref": "response_log_probs"},
                    "segment_metrics": {"tool_results": index},
                }
            )
            if name == "sampling-mask":
                tokens = segments[-1]["suffix_tokens"]
                segments[-1]["sampling_mask"] = [
                    [token, (token + 1) % 32000] for token in tokens[start:]
                ]
            previous = length
        model = CompactRolloutResult(
            schema_version=2,
            rollout_id=name,
            segments=segments,
            status=status,
            reward=1.0 if status == "success" else 0.0,
            messages=[{"role": "assistant", "content": "synthetic fixture"}],
        )
        payloads.append((name, dump_artifact(model)))
    return payloads


def foreground_inputs(scale):
    inputs = []
    # Exercise different representations/objective inputs through the real
    # converter and SDK builder, including response-aligned OPD advantages.
    for kind, size in [("packed", 62870), ("list", 187250), ("ref", 262144), ("opd", 65536)]:
        length = max(4, round(size * scale))
        split = length // 2
        fields = {
            "tokens": array("q", (i % 32000 for i in range(length))),
            "loss_mask": array("b", [0]) * split + array("b", [1]) * (length - split),
            "old_log_probs": array("d", [0.0]) * split + array("d", [-0.125]) * (length - split),
        }
        if kind == "list":
            fields = {key: list(value) for key, value in fields.items()}
        trajectory = Trajectory(
            tokens=fields["tokens"],
            loss_mask=fields["loss_mask"],
            reward=1.0,
            extra_fields={"advantage": 1.0, "old_log_probs": fields["old_log_probs"]},
        )
        if kind == "ref":
            trajectory["ref_logprobs"] = fields["old_log_probs"]
        if kind == "opd":
            trajectory["token_advantages"] = [0.5] * (length - split)
        inputs.append(trajectory)
    return inputs


def foreground(inputs):
    wall, cpu = time.perf_counter(), time.thread_time()
    data = convert_trajectories_to_datums(inputs, drop_zero_advantage_datums=False)
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    datums = holder._build_weaver_datums(data)
    return (
        data,
        datums,
        {
            "wall_seconds": time.perf_counter() - wall,
            "thread_cpu_seconds": time.thread_time() - cpu,
        },
    )


def foreground_digest(output):
    data, datums, _ = output
    return digest([data, [(d.model_input.to_ints(), d.loss_fn_inputs, d.metadata) for d in datums]])


def worker():
    instance = E2BRolloutWorker.__new__(E2BRolloutWorker)
    instance._config = OmegaConf.create({"rollout": {"sampling": {"temperature": 1.0}}})
    instance._max_sequence_length = 1 << 40  # Preserve saved fixtures, never truncate them.
    return instance


def sandbox(payload):
    async def read(*args, **kwargs):
        return payload

    return SimpleNamespace(files=SimpleNamespace(read=read))


async def replay(mode, payloads, references, inputs, args, root):
    bridge = worker()
    jobs = []
    expected = []
    # Prepare identities, reference outputs, destination paths and stores before
    # timing. Each group has one rollout; actual hashing/publication is timed.
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(result_processing(mode))
        for cycle in range(args.cycles):
            for index, ((name, payload), reference) in enumerate(zip(payloads, references)):
                job_id = len(jobs)
                task = {"group_id": f"g-{job_id}", "run_id": 0, "expected_rollouts": 1}
                trajectory = bridge._trajectory_from_result(reference, task, "replay")
                store = GroupedTrajectoryStore(OmegaConf.create({"group_size": 1}), "actor")
                reference_store = GroupedTrajectoryStore(
                    OmegaConf.create({"group_size": 1}), "actor"
                )
                assert reference_store.put_trajectory(trajectory)
                delivered = reference_store.take_groups(num_groups=1)
                expected.append((digest(reference), digest(delivered[0])))
                arrival = (job_id // args.burst_size) * args.burst_interval
                jobs.append((arrival, name, payload, reference.rollout_id, task, store))
        snapshots = []
        if args.snapshots:
            for index, (_, payload) in enumerate(payloads):
                model = load_result_gzip(payload)
                if not model.segments:
                    continue
                trace = dump_artifact(
                    CompactRolloutTrace(
                        schema_version=2,
                        rollout_id=model.rollout_id,
                        segments=model.segments,
                        metrics=model.metrics,
                    )
                )
                destination = root / f"snapshot-{index}.gz"
                writer = await stack.enter_async_context(
                    trace_snapshot_writer(destination, model.rollout_id)
                )
                snapshots.append((index * args.burst_interval, trace, writer, destination))
        # Warm actual decode + IPC and snapshot pools while owning the same
        # contexts used below. Warm-up does not enter the measured interval.
        await e2b_executor._read_result(
            sandbox(payloads[0][1]), RolloutInput(rollout_id=references[0].rollout_id, task={})
        )
        if snapshots:
            _, trace, _, _ = snapshots[0]
            warm_writer = await stack.enter_async_context(
                trace_snapshot_writer(root / "warm.gz", snapshots[0][2]._rollout_id)
            )
            await e2b_executor._download_file(
                sandbox(trace), e2b_executor._TRACE_PATH, decode=warm_writer.save
            )
        completed = []
        observed_arrivals = []
        started = time.perf_counter()

        async def result_job(index, job):
            arrival, name, payload, rollout_id, task, store = job
            await asyncio.sleep(max(0.0, started + arrival - time.perf_counter()))
            observed_arrivals.append(
                {
                    "job": index,
                    "case": name,
                    "scheduled_seconds": arrival,
                    "observed_seconds": time.perf_counter() - started,
                }
            )
            result = await e2b_executor._read_result(
                sandbox(payload), RolloutInput(rollout_id=rollout_id, task={})
            )
            trajectory = bridge._trajectory_from_result(result, task, "replay")
            assert store.put_trajectory(trajectory)
            delivered = store.take_groups(num_groups=1)
            completed.append(index)
            return result, delivered[0]

        async def snapshot_job(entry):
            arrival, trace, writer, destination = entry
            await asyncio.sleep(max(0.0, started + arrival - time.perf_counter()))
            await e2b_executor._download_file(
                sandbox(trace), e2b_executor._TRACE_PATH, decode=writer.save, priority="background"
            )
            return trace, destination

        async def foreground_job():
            await asyncio.sleep(max(0.0, started + args.foreground_delay - time.perf_counter()))
            output = await asyncio.to_thread(foreground, inputs)
            return output, time.perf_counter() - started, len(jobs) - len(completed)

        # Both families start together; background completion includes snapshots.
        async def background():
            # On failure, drain every admitted callback before pool ownership
            # ends, just as the rollout wrapper does on cancellation.
            async with asyncio.TaskGroup() as group:
                result_tasks = [group.create_task(result_job(i, job)) for i, job in enumerate(jobs)]
                snapshot_tasks = [group.create_task(snapshot_job(s)) for s in snapshots]
            results = [task.result() for task in result_tasks]
            saved = [task.result() for task in snapshot_tasks]
            return results, saved, time.perf_counter() - started

        async with asyncio.TaskGroup() as group:
            probe_task = group.create_task(foreground_job())
            background_task = group.create_task(background())
        probe, foreground_done, pending = probe_task.result()
        results, saved, background_done = background_task.result()
        all_done = time.perf_counter() - started
        # Full-field/array, group delivery, snapshot and SDK checks occur after
        # *both* timed completion milestones, never during concurrent work.
        assert [(digest(r), digest(t)) for r, t in results] == expected
        assert all(destination.read_bytes() == trace for trace, destination in saved)
        return {
            "mode": mode,
            "foreground": probe[2],
            "foreground_completion_seconds": foreground_done,
            "all_background_completion_seconds": background_done,
            "all_work_completion_seconds": all_done,
            "pending_results_at_foreground_completion": pending,
            "result_jobs": len(jobs),
            "snapshot_jobs": len(snapshots),
            "observed_arrivals": observed_arrivals,
            "foreground_sha256": foreground_digest(probe),
            "all_fields_packed_bytes_group_delivery_snapshot_parity": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--foreground-scale",
        type=float,
        help="scale foreground independently of background expanded sizes",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument("--burst-interval", type=float, default=0.1)
    parser.add_argument("--foreground-delay", type=float, default=0.05)
    parser.add_argument("--snapshots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.foreground_scale is None:
        args.foreground_scale = args.scale
    if min(args.scale, args.foreground_scale, args.repeat, args.cycles, args.burst_size) <= 0:
        parser.error("scale, repeat, cycles and burst-size must be positive")
    if args.burst_interval < 0 or args.foreground_delay < 0:
        parser.error("arrival times must be nonnegative")
    if result_profiling_enabled():
        parser.error("unset NORTHRL_PROFILE_RESULTS for the timed performance check")
    logging.basicConfig(level=logging.ERROR)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payloads = (
        [(path.name, path.read_bytes()) for path in sorted(args.results_dir.rglob("*.gz"))]
        if args.results_dir
        else synthetic_results(args.scale)
    )
    if not payloads:
        parser.error("no result fixtures")
    references = [decode_result(load_result_gzip(payload), packed=True) for _, payload in payloads]
    inputs = foreground_inputs(args.foreground_scale)
    expected_foreground = foreground_digest(foreground(inputs))
    sdk_project = Path(weaver.__file__).resolve().parent.parent / "pyproject.toml"
    report = {
        "limits": "Local CPU only; in-memory remote reads, warmed workers, synthetic arrival schedule. "
        "No real Server, HTTP, GPU or matched online background state; no online speedup estimate.",
        "fixture_source": "saved results" if args.results_dir else "synthetic mixed expanded sizes",
        "cases": [
            {
                "case": name,
                "status": ref.status,
                **result_size_fields(payload, ref),
                "segment_token_positions": [len(s.tokens) for s in ref.segments],
                "has_sampling_mask": any(s.sampling_mask is not None for s in ref.segments),
                "packed_array_types": (
                    {
                        key: getattr(ref.segments[0], key).typecode
                        for key in ("tokens", "loss_mask", "old_log_probs")
                    }
                    if ref.segments
                    else {}
                ),
            }
            for (name, payload), ref in zip(payloads, references)
        ],
        "foreground_cases": ["packed GRPO", "list GRPO", "reference logprobs", "OPD"],
        "foreground_token_positions": [len(t.tokens) for t in inputs],
        "source": str(Path(e2b_executor.__file__).resolve()),
        "sdk_source": str(Path(weaver.__file__).resolve()),
        "sdk_source_version": (
            tomllib.loads(sdk_project.read_text())["project"]["version"]
            if sdk_project.exists()
            else None
        ),
        "python": sys.version,
        "host": platform.node(),
        "torch_threads": torch.get_num_threads(),
        "arrival_schedule": {
            "burst_size": args.burst_size,
            "interval_seconds": args.burst_interval,
            "foreground_delay_seconds": args.foreground_delay,
        },
        "admission": "actual DownloadQueue, two result callbacks, four snapshot callbacks, "
        "one result process and two snapshot processes shared per loop",
        "runs": [],
    }
    for repeat in range(args.repeat):
        # Alternate order to reduce systematic shared-host warmup/load bias.
        for mode in ("thread", "process") if repeat % 2 == 0 else ("process", "thread"):
            with tempfile.TemporaryDirectory(prefix="northrl-result-replay-") as directory:
                row = asyncio.run(replay(mode, payloads, references, inputs, args, Path(directory)))
            assert row["foreground_sha256"] == expected_foreground
            row["repeat"] = repeat
            report["runs"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {key: value for key, value in row.items() if key != "observed_arrivals"}
                ),
                flush=True,
            )
    report["medians"] = {
        mode: {
            key: statistics.median(row[key] for row in report["runs"] if row["mode"] == mode)
            for key in (
                "foreground_completion_seconds",
                "all_background_completion_seconds",
                "all_work_completion_seconds",
            )
        }
        for mode in ("thread", "process")
    }
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
