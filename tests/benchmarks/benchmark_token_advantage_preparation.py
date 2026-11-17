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

"""Compare response-only Datum expansion using reproducible compact trajectories.

Run legacy and compact in separate processes to compare peak RSS. Construction,
GC and payload hashing are excluded from conversion timings. For example:

    PYTHONPATH=. python tests/benchmarks/benchmark_token_advantage_preparation.py \
        --implementation compact --output /tmp/token-advantage-compact.json

Repeat with --implementation legacy for the original Python expansion and
zero check, or --advantage scalar for the existing scalar preparation path.
The default 64 parents * 4 sequences * 32768 tokens matches the long-context
synthetic trainer benchmark. This is a CPU conversion benchmark, not an
end-to-end rollout or remote training measurement.
"""

import argparse
import gc
import hashlib
import json
import resource
import statistics
import time
from array import array
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from northrl.northrl_types import Trajectory
from northrl.utils import finetune_service_utils as conversion


def legacy_expand(values, masks):
    """The original indexed response-only expansion, retained as the baseline."""
    output = []
    index = 0
    for mask in masks:
        if mask == 1 and index < len(values):
            output.append(float(values[index]))
            index += 1
        else:
            output.append(0.0)
    return output


def make_trajectories(parents, segments, length, advantage):
    output = []
    for parent_index in range(parents):
        sequences = []
        value = (parent_index % 4 - 1.5) / 2
        for _ in range(segments):
            sequence = {
                "tokens": array("i", [17]) * length,
                "loss_mask": array("b", [0]) * (length // 2)
                + array("b", [1]) * (length - length // 2),
                "old_log_probs": array("d", [0.0]) + array("d", [-0.2]) * (length - 1),
            }
            if advantage == "token":
                sequence["token_advantages"] = array("d", [value]) * (length - length // 2)
            sequences.append(sequence)
        output.append(Trajectory(segments=sequences, reward=1, extra_fields={"advantage": value}))
    return output


def digest(datums):
    result = hashlib.sha256()
    for datum in datums:
        result.update(memoryview(datum["input_tokens"]).cast("B"))
        for key in ("target_tokens", "logprobs", "advantages", "loss_mask"):
            result.update(key.encode())
            result.update(memoryview(datum["loss_fn_inputs"][key]).cast("B"))
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("legacy", "compact"), default="compact")
    parser.add_argument("--advantage", choices=("scalar", "token"), default="token")
    parser.add_argument("--parents", type=int, default=64)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.parents, args.segments, args.repeat) < 1 or args.length < 2:
        parser.error("parents, segments and repeat must be positive; length must be at least 2")

    override = (
        patch.multiple(
            conversion,
            _expand_response_advantages=legacy_expand,
            _has_nonzero_advantages=lambda values: any(value != 0.0 for value in values),
        )
        if args.implementation == "legacy"
        else nullcontext()
    )
    rows = []
    with override:
        conversion.convert_trajectories_to_datums(make_trajectories(4, 1, 128, args.advantage))
        for repeat in range(args.repeat):
            trajectories = make_trajectories(
                args.parents, args.segments, args.length, args.advantage
            )
            gc.collect()
            start = time.perf_counter()
            datums = conversion.convert_trajectories_to_datums(trajectories)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "repeat": repeat,
                    "conversion_s": elapsed,
                    "payload_sha256": digest(datums),
                    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                }
            )
            del trajectories, datums
    result = {
        "implementation": args.implementation,
        "advantage": args.advantage,
        "parents": args.parents,
        "sequences": args.parents * args.segments,
        "total_tokens": args.parents * args.segments * args.length,
        "median_conversion_s": statistics.median(row["conversion_s"] for row in rows),
        "rows": rows,
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
