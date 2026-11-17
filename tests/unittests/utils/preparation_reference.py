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

"""Historical preparation formulas used only as unit-test comparison oracles.

Keep these expressions independent of the optimized implementation so the
equivalence tests can detect numerical changes. No diagnostic scripts, saved
converter files, datasets or services are required.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from northrl.utils import finetune_service_utils as conversion


def legacy_masked_arrays(
    values: list[float], loss_mask: list[int]
) -> tuple[list[float], list[float]]:
    """Apply the two original per-position array-building expressions."""
    return (
        [float(value) * float(mask) if mask else 0.0 for value, mask in zip(values, loss_mask)],
        [float(mask) for mask in loss_mask],
    )


@contextmanager
def use_legacy_preparation() -> Iterator[None]:
    """Use main's original array formulas without replacing its advantage cache."""
    with patch.object(conversion, "_build_aligned_masked_arrays", legacy_masked_arrays):
        yield
