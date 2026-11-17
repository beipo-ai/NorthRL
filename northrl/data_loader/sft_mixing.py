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

"""Deterministic virtual-index planning for mixed SFT datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any, MutableSequence, Sequence, TypeVar

MIX_PLAN_VERSION = 1
MIX_SHUFFLE_ALGORITHM = "sha256-counter-fisher-yates-v1"

_T = TypeVar("_T")
_UINT64_RANGE = 1 << 64


@dataclass(frozen=True, slots=True)
class MixSource:
    """Safe source identity and its planned contribution to one epoch."""

    identity: dict[str, Any]
    size: int
    target_count: int


@dataclass(frozen=True, slots=True)
class VirtualSample:
    """One logical sample occurrence before materialization."""

    source_ordinal: int
    sample_idx: int


class _Sha256CounterRng:
    """Small, frozen random stream used only for dataset permutations.

    Every 64-bit draw is SHA-256(key || counter)[:8]. Rejection sampling makes
    ``randbelow`` unbiased, and Fisher-Yates below defines the permutation
    independently of Python's ``random`` implementation and version.
    """

    def __init__(self, key_payload: dict[str, Any]) -> None:
        encoded = json.dumps(
            key_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        self._key = hashlib.sha256(encoded).digest()
        self._counter = 0

    def _uint64(self) -> int:
        block = hashlib.sha256(
            self._key + self._counter.to_bytes(16, byteorder="big", signed=False)
        ).digest()
        self._counter += 1
        return int.from_bytes(block[:8], byteorder="big", signed=False)

    def randbelow(self, upper_bound: int) -> int:
        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        limit = _UINT64_RANGE - (_UINT64_RANGE % upper_bound)
        while True:
            value = self._uint64()
            if value < limit:
                return value % upper_bound

    def shuffle(self, values: MutableSequence[_T]) -> None:
        for index in range(len(values) - 1, 0, -1):
            swap_index = self.randbelow(index + 1)
            values[index], values[swap_index] = values[swap_index], values[index]


def allocate_target_counts(total_samples: int, ratios: Sequence[float]) -> list[int]:
    """Allocate integer quotas with largest remainder and stable tie-breaking."""

    if not isinstance(total_samples, int) or isinstance(total_samples, bool) or total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if not ratios or any(not isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("source ratios must be non-empty, finite, and positive")

    # Convert the user-visible decimal forms to exact rationals so quota
    # rounding does not depend on binary floating-point edge behavior.
    rational_ratios = [Fraction(str(ratio)) for ratio in ratios]
    ratio_sum = sum(rational_ratios)
    exact = [Fraction(total_samples) * ratio / ratio_sum for ratio in rational_ratios]
    targets = [int(value) for value in exact]
    remainder = total_samples - sum(targets)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(exact[index] - targets[index]), index),
    )
    for index in order[:remainder]:
        targets[index] += 1
    return targets


def build_epoch_order(
    sources: Sequence[MixSource],
    *,
    seed: int,
    epoch: int,
) -> list[VirtualSample]:
    """Build repeat/sample/concat/global-shuffle virtual order for one epoch."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    mixed: list[VirtualSample] = []
    for source_ordinal, source in enumerate(sources):
        if source.size <= 0:
            raise ValueError(f"source {source_ordinal} size must be positive")
        if source.target_count < 0:
            raise ValueError(f"source {source_ordinal} target_count must be non-negative")

        full_repeats, remainder = divmod(source.target_count, source.size)
        selected = list(range(source.size)) * full_repeats
        if remainder:
            candidates = list(range(source.size))
            _rng(
                seed=seed,
                epoch=epoch,
                stage="source-remainder",
                source_ordinal=source_ordinal,
            ).shuffle(candidates)
            selected.extend(candidates[:remainder])
        mixed.extend(VirtualSample(source_ordinal, sample_idx) for sample_idx in selected)

    _rng(seed=seed, epoch=epoch, stage="global-shuffle").shuffle(mixed)
    return mixed


def mixing_plan_fingerprint(
    sources: Sequence[MixSource],
    *,
    seed: int,
    total_samples: int,
) -> str:
    """Fingerprint every stable input needed to regenerate epoch orders."""

    payload = {
        "version": MIX_PLAN_VERSION,
        "shuffle_algorithm": MIX_SHUFFLE_ALGORITHM,
        "seed": seed,
        "total_samples": total_samples,
        "sources": [
            {
                "identity": source.identity,
                "size": source.size,
                "target_count": source.target_count,
            }
            for source in sources
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def occurrence_datum_id(plan_fingerprint: str, *, epoch: int, position: int) -> str:
    """Return a stable ID for one occurrence, including repeated references."""

    # Keep persisted occurrence IDs stable across the package rename.
    payload = f"nexrl-mix-datum-v1\0{plan_fingerprint}\0{epoch}\0{position}"
    return f"nexrl-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _rng(
    *,
    seed: int,
    epoch: int,
    stage: str,
    source_ordinal: int | None = None,
) -> _Sha256CounterRng:
    payload: dict[str, Any] = {
        "algorithm": MIX_SHUFFLE_ALGORITHM,
        "seed": seed,
        "epoch": epoch,
        "stage": stage,
    }
    if source_ordinal is not None:
        payload["source_ordinal"] = source_ordinal
    return _Sha256CounterRng(payload)
