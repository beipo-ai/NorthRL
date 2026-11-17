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

"""Managed SFT policy-carrier serialization tests."""

from northrl.northrl_types import Trajectory
from northrl.utils.data_dumper import DataDumper


def test_managed_trajectory_dump_preserves_internal_content_visibility() -> None:
    trajectory = Trajectory(
        tokens=None,
        loss_mask=None,
        reward=0.0,
        is_val=False,
        extra_fields={
            "sample_ref": {
                "dataset": "hq-math",
                "version": "2026-08",
                "sample_idx": 3,
            },
            "datum_id": "managed-3",
            "input_token_count": 7,
            "content_visibility": "public",
            "loss_fn_inputs": {},
        },
    )

    serialized = DataDumper()._serialize_trajectory(trajectory)  # pylint: disable=protected-access

    assert serialized["extra_fields"]["content_visibility"] == "public"
    assert "content_visibility" not in serialized["extra_fields"]["sample_ref"]
