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

from collections import Counter

import pytest

from northrl.data_loader.sft_mixing import (
    MIX_SHUFFLE_ALGORITHM,
    MixSource,
    allocate_target_counts,
    build_epoch_order,
    mixing_plan_fingerprint,
    occurrence_datum_id,
)


def test_largest_remainder_allocation_normalizes_weights_and_breaks_ties_by_order() -> None:
    assert allocate_target_counts(11, [0.8, 0.2]) == [9, 2]
    assert allocate_target_counts(2, [1.0, 1.0, 1.0]) == [1, 1, 0]


def test_allocation_rejects_non_finite_weights() -> None:
    with pytest.raises(ValueError, match="finite"):
        allocate_target_counts(10, [1.0, float("inf")])


def test_repeat_remainder_and_global_shuffle_are_deterministic() -> None:
    sources = [
        MixSource(identity={"name": "a"}, size=2, target_count=5),
        MixSource(identity={"name": "b"}, size=7, target_count=3),
    ]

    first = build_epoch_order(sources, seed=42, epoch=0)
    second = build_epoch_order(sources, seed=42, epoch=0)

    assert first == second
    assert len(first) == 8
    assert Counter(item.source_ordinal for item in first) == {0: 5, 1: 3}
    first_source_indices = Counter(item.sample_idx for item in first if item.source_ordinal == 0)
    assert sorted(first_source_indices.values()) == [2, 3]
    assert len({item.sample_idx for item in first if item.source_ordinal == 1}) == 3


@pytest.mark.parametrize(
    ("size", "target_count"),
    [(5, 2), (5, 5), (5, 6), (5, 13), (2, 9)],
)
def test_source_membership_matches_ms_swift_repeat_and_remainder_semantics(
    size: int, target_count: int
) -> None:
    """Match ms-swift's count contract without coupling to its NumPy RNG.

    ms-swift repeats every source row ``count // size`` times, then selects
    ``count % size`` distinct rows for the remainder. NorthRL intentionally uses
    its own frozen RNG because the virtual plan must resume independently of
    NumPy and Hugging Face versions.
    """

    order = build_epoch_order(
        [MixSource(identity={"name": "source"}, size=size, target_count=target_count)],
        seed=20260904,
        epoch=0,
    )
    observed = Counter(item.sample_idx for item in order)
    full_repeats, remainder = divmod(target_count, size)
    counts = [observed[index] for index in range(size)]

    assert len(order) == target_count
    assert set(observed).issubset(range(size))
    assert sum(count == full_repeats + 1 for count in counts) == remainder
    assert sum(count == full_repeats for count in counts) == size - remainder


def test_concat_then_global_shuffle_has_a_frozen_cross_source_golden_order() -> None:
    sources = [
        MixSource(identity={"name": "a"}, size=3, target_count=8),
        MixSource(identity={"name": "b"}, size=5, target_count=2),
        MixSource(identity={"name": "c"}, size=2, target_count=4),
    ]

    order = build_epoch_order(sources, seed=20260904, epoch=0)

    assert [(item.source_ordinal, item.sample_idx) for item in order] == [
        (2, 1),
        (0, 0),
        (0, 2),
        (0, 1),
        (0, 1),
        (2, 0),
        (0, 2),
        (1, 1),
        (2, 1),
        (0, 0),
        (2, 0),
        (0, 0),
        (1, 4),
        (0, 2),
    ]


def test_epoch_and_seed_are_domain_separated() -> None:
    sources = [MixSource(identity={"name": "a"}, size=30, target_count=30)]

    epoch_zero = build_epoch_order(sources, seed=42, epoch=0)

    assert epoch_zero != build_epoch_order(sources, seed=42, epoch=1)
    assert epoch_zero != build_epoch_order(sources, seed=43, epoch=0)
    assert MIX_SHUFFLE_ALGORITHM == "sha256-counter-fisher-yates-v1"


def test_plan_fingerprint_and_occurrence_ids_cover_order_inputs() -> None:
    sources = [MixSource(identity={"name": "a", "version": "v1"}, size=2, target_count=4)]
    fingerprint = mixing_plan_fingerprint(sources, seed=7, total_samples=4)

    first = occurrence_datum_id(fingerprint, epoch=0, position=0)
    repeat_of_same_ref = occurrence_datum_id(fingerprint, epoch=0, position=2)

    # This persisted ID predates the package rename and must remain reproducible.
    assert first == "nexrl-e4d0facacc065fe594a2b2144618b7fe0c2e1337cf4b6dffc44626a3bc21148e"
    assert first == occurrence_datum_id(fingerprint, epoch=0, position=0)
    assert first != repeat_of_same_ref
    assert fingerprint != mixing_plan_fingerprint(sources, seed=8, total_samples=4)
