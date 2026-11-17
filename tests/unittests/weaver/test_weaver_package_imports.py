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

import subprocess
import sys
from pathlib import Path


def test_service_holder_import_does_not_load_inference_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    code = r"""
import sys


class BlockInferenceDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"omegaconf", "openai"}:
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None


sys.meta_path.insert(0, BlockInferenceDependencies())

from northrl.weaver import WeaverServiceHolder as PublicWeaverServiceHolder
import northrl.weaver as weaver_package
from northrl.weaver.weaver_service_holder import (
    WeaverServiceHolder,
    _extract_forward_logprobs_batch_from_result,
)

assert PublicWeaverServiceHolder is WeaverServiceHolder
assert "northrl.weaver.weaver_inference_service_client" not in sys.modules
assert "omegaconf" not in sys.modules
assert "openai" not in sys.modules
assert dir(weaver_package).count("WeaverServiceHolder") == 1
assert callable(_extract_forward_logprobs_batch_from_result)
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
