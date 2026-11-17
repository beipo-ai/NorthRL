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

"""The E2B runtime must not import the optional training stack."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_remote_rollout_import_does_not_load_training_modules() -> None:
    code = """
import importlib.util
import sys

original_find_spec = importlib.util.find_spec

def find_spec(fullname, package=None):
    if fullname.split(".", 1)[0] == "pandas":
        return None
    return original_find_spec(fullname, package)

class BlockOptionalDependencies:
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname.split(".", 1)[0] == "pandas":
            raise ModuleNotFoundError(fullname)
        return None

importlib.util.find_spec = find_spec
sys.meta_path.insert(0, BlockOptionalDependencies())
from northrl.remote_rollout.e2b_runtime import E2BRolloutRuntime

assert E2BRolloutRuntime is not None
assert "northrl.controller" not in sys.modules
assert "northrl.data_loader" not in sys.modules
assert "northrl.rollout_worker" not in sys.modules

from northrl import Batch
from northrl.northrl_types import Batch as ExpectedBatch
assert Batch is ExpectedBatch
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
