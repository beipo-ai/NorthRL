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

"""Exact routing table for the 35 task families in the full SynLogic data."""

from collections.abc import Mapping

from .base import Verifier
from .verifiers.arc_agi import ArcAGIVerifier
from .verifiers.arrow_maze import ArrowMazeVerifier
from .verifiers.boolean_expressions import BooleanExpressionsVerifier
from .verifiers.buggy_tables import BuggyTableVerifier
from .verifiers.calcudoko import CalcudokoVerifier
from .verifiers.campsite import CampsiteVerifier
from .verifiers.cipher import CipherVerifier
from .verifiers.cryptarithm import CryptarithmVerifier
from .verifiers.dyck_language import DyckLanguageVerifier
from .verifiers.dyck_language_errors import DyckLanguageErrorsVerifier
from .verifiers.dyck_language_reasoning_errors import DyckLanguageReasoningErrorsVerifier
from .verifiers.futoshiki import FutoshikiVerifier
from .verifiers.goods_exchange import GoodsExchangeVerifier
from .verifiers.kukurasu import KukurasuVerifier
from .verifiers.math_path import MathPathVerifier
from .verifiers.mathador import GameOf24Verifier
from .verifiers.minesweeper import MinesweeperVerifier
from .verifiers.norinori import NorinoriVerifier
from .verifiers.number_wall import NumberWallVerifier
from .verifiers.numbrix import NumbrixVerifier
from .verifiers.object_counting import ObjectCountingVerifier
from .verifiers.object_properties import ObjectPropertiesVerifier
from .verifiers.operation import OperationVerifier
from .verifiers.skyscraper_puzzle import SkyscraperPuzzleVerifier
from .verifiers.space_reasoning import SpaceReasoningVerifier
from .verifiers.space_reasoning_tree import SpaceReasoningTreeVerifier
from .verifiers.star_placement_puzzle import StarPlacementPuzzleVerifier
from .verifiers.sudoku import SudokuVerifier
from .verifiers.survo import SurvoVerifier
from .verifiers.time_sequence import TimeSequenceVerifier
from .verifiers.web_of_lies import WebOfLiesVerifier
from .verifiers.word_sorting import WordSortingVerifier
from .verifiers.word_sorting_mistake import WordSortingMistakeVerifier
from .verifiers.wordscapes import WordscapesVerifier
from .verifiers.zebra_puzzle import ZebraPuzzleVerifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


VERIFIER_TYPES: Mapping[str, type[Verifier]] = {
    "arc_agi": ArcAGIVerifier,
    "arrow_maze": ArrowMazeVerifier,
    "boolean_expressions": BooleanExpressionsVerifier,
    "buggy_tables": BuggyTableVerifier,
    "calcudoko": CalcudokoVerifier,
    "campsite": CampsiteVerifier,
    "cipher": CipherVerifier,
    "cryptarithm": CryptarithmVerifier,
    "dyck_language": DyckLanguageVerifier,
    "dyck_language_errors": DyckLanguageErrorsVerifier,
    "dyck_language_reasoning_errors": DyckLanguageReasoningErrorsVerifier,
    "futoshiki": FutoshikiVerifier,
    "goods_exchange": GoodsExchangeVerifier,
    "kukurasu": KukurasuVerifier,
    "math_path": MathPathVerifier,
    "mathador": GameOf24Verifier,
    "minesweeper": MinesweeperVerifier,
    "norinori": NorinoriVerifier,
    "number_wall": NumberWallVerifier,
    "numbrix": NumbrixVerifier,
    "object_counting": ObjectCountingVerifier,
    "object_properties": ObjectPropertiesVerifier,
    "operation": OperationVerifier,
    "skyscraper_puzzle": SkyscraperPuzzleVerifier,
    "space_reasoning": SpaceReasoningVerifier,
    "space_reasoning_tree": SpaceReasoningTreeVerifier,
    "star_placement_puzzle": StarPlacementPuzzleVerifier,
    "sudoku": SudokuVerifier,
    "survo": SurvoVerifier,
    "time_sequence": TimeSequenceVerifier,
    "web_of_lies": WebOfLiesVerifier,
    "word_sorting": WordSortingVerifier,
    "word_sorting_mistake": WordSortingMistakeVerifier,
    "wordscapes": WordscapesVerifier,
    "zebra_puzzle": ZebraPuzzleVerifier,
}


def get_verifier(data_source: str) -> Verifier:
    """Construct the exact verifier for ``data_source`` or fail fast."""
    try:
        verifier_type = VERIFIER_TYPES[data_source]
    except KeyError as error:
        raise ValueError(f"Unknown SynLogic data_source: {data_source!r}") from error
    return verifier_type()
