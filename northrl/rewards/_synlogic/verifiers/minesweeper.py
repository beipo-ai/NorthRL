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

import re
from typing import List, Tuple

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class MinesweeperVerifier(Verifier):
    """
    Verifier for Minesweeper puzzle
    扫雷游戏验证器
    """

    def verify(self, data: Data, test_answer: str, **kwargs) -> bool:
        del kwargs
        try:
            predicted_mines = self.extract_answer(test_answer)
            expected_mines = data.metadata["current_mines"]
            if set((tuple(mine) for mine in predicted_mines)) == set(
                (tuple(mine) for mine in expected_mines)
            ):
                return True
            return False
        except Exception:
            return False

    def extract_answer(self, test_solution: str) -> List[Tuple[int, int]]:
        """从模型的响应中提取地雷坐标
        Extract mine coordinates from the model's response"""
        response = test_solution
        patterns = [
            "\\[\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)(?:\\s*,\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\))*\\s*\\]",
            "\\[\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\](?:\\s*,\\s*\\[\\s*\\d+\\s*,\\s*\\d+\\s*\\])*\\s*\\]",
            "\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)(?:\\s*,\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\))*",
        ]
        for pattern in patterns:
            coords = []
            for match in re.finditer(pattern, response):
                try:
                    coord_pattern = "(?:\\(|\\[)\\s*(\\d+)\\s*,\\s*(\\d+)\\s*(?:\\)|\\])"
                    for coord_match in re.finditer(coord_pattern, match.group(0)):
                        i, j = (int(coord_match.group(1)), int(coord_match.group(2)))
                        coords.append((i, j))
                except Exception:
                    continue
            if coords:
                return coords
        number_pairs = re.findall("(\\d+)[^\\d]+(\\d+)", response)
        if number_pairs:
            return [(int(i), int(j)) for i, j in number_pairs]
        return []
