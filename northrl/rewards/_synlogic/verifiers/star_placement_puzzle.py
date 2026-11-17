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

import ast
import re

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class StarPlacementPuzzleVerifier(Verifier):
    """
    星星放置游戏验证器，用于验证模型提供的解答是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否符合星星放置游戏的规则

        @param data: 包含游戏信息的Data对象
        @param star_coords: 通过extract_answer提取的星星坐标字典 {区域: [(行,列), ...]}
        @return: 回答是否正确的布尔值
        """
        test_solution = test_answer
        try:
            star_coords = self.extract_answer(test_solution)
            metadata = data.metadata
            n = metadata["n"]
            k = metadata["k"]
            region_grid = metadata["region_grid"]
            if not star_coords:
                return False
            star_grid = [[0 for _ in range(n)] for _ in range(n)]
            for region, coords in star_coords.items():
                for coord in coords:
                    row, col = coord
                    if row < 0 or row >= n or col < 0 or (col >= n):
                        return False
                    star_grid[row][col] = 1
            for row in star_grid:
                pass
            for i in range(n):
                stars_in_row = sum(star_grid[i])
                if stars_in_row != k:
                    return False
            for j in range(n):
                stars_in_col = sum((star_grid[i][j] for i in range(n)))
                if stars_in_col != k:
                    return False
            regions: dict[str, list[tuple[int, int]]] = {}
            for i in range(n):
                for j in range(n):
                    region = region_grid[i][j]
                    if region not in regions:
                        regions[region] = []
                    regions[region].append((i, j))
            for region, cells in regions.items():
                stars_in_region = sum((star_grid[i][j] for i, j in cells))
                if stars_in_region != k:
                    return False
            for i in range(n):
                for j in range(n):
                    if star_grid[i][j] == 1:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = (i + di, j + dj)
                                if 0 <= ni < n and 0 <= nj < n and (star_grid[ni][nj] == 1):
                                    return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取星星坐标

        @param test_solution: 模型的完整回答
        @return: 提取的星星坐标字典 {区域: [(行,列), ...]}
        """
        try:
            python_match = re.search("```python\\s*\\n(.*?)\\n\\s*```", test_solution, re.DOTALL)
            if not python_match:
                return None
            code_content = python_match.group(1)
            try:
                dict_match = re.search("\\{[^{}]*\\}", code_content, re.DOTALL)
                if dict_match:
                    dict_str = dict_match.group(0)
                    try:
                        coords_dict = ast.literal_eval(dict_str)
                        if isinstance(coords_dict, dict):
                            result = {}
                            for region, coords in coords_dict.items():
                                result[region] = [(row - 1, col - 1) for row, col in coords]
                            return result
                    except (ValueError, SyntaxError):
                        pass
                assign_match = re.search("(\\w+)\\s*=\\s*(\\{[^{}]*\\})", code_content, re.DOTALL)
                if assign_match:
                    dict_str = assign_match.group(2)
                    try:
                        coords_dict = ast.literal_eval(dict_str)
                        if isinstance(coords_dict, dict):
                            result = {}
                            for region, coords in coords_dict.items():
                                result[region] = [(row - 1, col - 1) for row, col in coords]
                            return result
                    except (ValueError, SyntaxError):
                        pass
            except Exception:
                pass
            return None
        except Exception:
            return None
