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

import pytest

from northrl.weight_sync.versioning import (
    ACTOR_MODEL_TAG,
    DEFAULT_WEIGHT_VERSION,
    format_weight_version,
)


def test_format_weight_version_uses_canonical_label() -> None:
    assert ACTOR_MODEL_TAG == "actor"
    assert DEFAULT_WEIGHT_VERSION == "default"
    assert format_weight_version("actor", 0) == "northrl_actor_v000000"
    assert format_weight_version("actor", 42) == "northrl_actor_v000042"


def test_format_weight_version_rejects_negative_version() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_weight_version("actor", -1)
