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

"""Opt-in large CPU replay; fixture supplies per-request source sequence lengths."""

import io
import json
import os
import time
from array import array
from pathlib import Path

import numpy as np
import pytest
from weaver.tensor_transport import materialize_http_tensor_payloads

from northrl.utils.trajectory_alignment import align_target_logprobs_to_tokens
from northrl.weaver.weaver_service_holder import _extract_forward_logprobs_batch_from_result


@pytest.mark.skipif(
    not os.getenv("NORTHRL_OUTPUT_REPLAY_FIXTURE"), reason="large CPU replay is opt-in"
)
@pytest.mark.parametrize("workload", ["85cf", "c5ff"])
@pytest.mark.parametrize("output_format", ["list", "torch"])
def test_large_forward_output_cpu_replay(workload, output_format):
    fixture = json.loads(Path(os.environ["NORTHRL_OUTPUT_REPLAY_FIXTURE"]).read_text())[workload]
    timings = dict(decode=0.0, extract=0.0, align=0.0, compact=0.0, release=0.0)
    retained = []
    for call in fixture["calls"]:
        lengths = call["lengths"]
        count = sum(length - 1 for length in lengths)
        raw = np.arange(count, dtype=np.float32)
        raw %= 8191
        raw *= np.float32(-0.00123)
        raw[::997] = np.float32(-0.0)
        pack = io.BytesIO(raw.tobytes())
        tokens = [array("q", [1]) * length for length in lengths]
        outputs, offset = [], 0
        for length in lengths:
            size = (length - 1) * 4
            outputs.append(
                {
                    "logprobs": {
                        "$tensor": {
                            "format": "raw-tensor",
                            "codec": "raw",
                            "dtype": "float32",
                            "shape": [length - 1],
                            "size_bytes": size,
                            "offset": offset,
                        }
                    }
                }
            )
            offset += size
        start = time.perf_counter()
        response = materialize_http_tensor_payloads(
            {"result": {"loss_fn_outputs": outputs}}, pack, tensor_output_format=output_format
        )
        decoded = time.perf_counter()
        target = _extract_forward_logprobs_batch_from_result(response)
        extracted = time.perf_counter()
        full = [
            align_target_logprobs_to_tokens(tokens=t, target_logprobs=v)
            for t, v in zip(tokens, target, strict=True)
        ]
        aligned = time.perf_counter()
        # The recompute consumer retains both arrays, with no second copy of arrays.
        full = [v if isinstance(v, array) else array("d", v) for v in full]
        target = [v if isinstance(v, array) else array("d", v) for v in target]
        compacted = time.perf_counter()
        del response
        released = time.perf_counter()
        for key, duration in zip(
            timings,
            [
                decoded - start,
                extracted - decoded,
                aligned - extracted,
                compacted - aligned,
                released - compacted,
            ],
            strict=True,
        ):
            timings[key] += duration
        offset = 0
        for length, a, b in zip(lengths, full, target, strict=True):
            expected = raw[offset : offset + length - 1].astype(np.float64)
            assert a[0] == 0 and len(a) == length and len(b) == length - 1
            np.testing.assert_array_equal(
                np.frombuffer(a, dtype=np.uint64)[1:], expected.view(np.uint64)
            )
            np.testing.assert_array_equal(
                np.frombuffer(b, dtype=np.uint64), expected.view(np.uint64)
            )
            offset += length - 1
        retained.append((full, target))
    print(
        "CPU_REPLAY "
        + json.dumps(
            {
                "workload": workload,
                "format": output_format,
                "seconds": sum(timings.values()),
                "stages": timings,
            }
        )
    )
