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

import json
import re
from collections import deque

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class NumberWallVerifier(Verifier):
    """
    Verifier for Number Wall puzzle
    数字墙拼图验证器
    """

    def verify(self, data: Data, test_answer: str, **kwargs) -> bool:
        del kwargs
        try:
            solution_grid = self.extract_answer(test_answer)
            if not solution_grid:
                return False
            original_grid = data.metadata["grid"]
            n = data.metadata["n"]
            if len(solution_grid) != n:
                return False
            for row in solution_grid:
                if len(row) != n:
                    return False
                for cell in row:
                    if not (isinstance(cell, int) or cell in ["X", "A"]):
                        return False
            if not self._check_original_numbers(original_grid, solution_grid):
                return False
            if not self._check_wall_layout(solution_grid):
                return False
            if not self._check_islands(solution_grid):
                return False
            if not self._check_diagonal_borders(solution_grid):
                return False
            return True
        except Exception:
            return False

    def _check_original_numbers(self, original_grid, solution_grid):
        """检查原始数字是否在解决方案中保留"""
        for i, original_row in enumerate(original_grid):
            for j, original_cell in enumerate(original_row):
                if isinstance(original_cell, int):
                    if original_cell != solution_grid[i][j]:
                        return False
        return True

    def _check_wall_layout(self, grid):
        """检查墙壁布局是否有效（没有2×2或更大的连续墙块）"""
        n = len(grid)
        for i in range(n - 1):
            for j in range(n - 1):
                if (
                    grid[i][j] == "A"
                    and grid[i][j + 1] == "A"
                    and (grid[i + 1][j] == "A")
                    and (grid[i + 1][j + 1] == "A")
                ):
                    return False
        return True

    def _check_islands(self, grid):
        """检查岛屿划分是否有效"""
        n = len(grid)
        visited = set()
        for i in range(n):
            for j in range(n):
                if (i, j) not in visited and grid[i][j] != "A":
                    island_cells = []
                    island_number = None
                    queue = deque([(i, j)])
                    visited.add((i, j))
                    while queue:
                        r, c = queue.popleft()
                        island_cells.append((r, c))
                        if isinstance(grid[r][c], int):
                            if island_number is not None:
                                return False
                            island_number = grid[r][c]
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = (r + dr, c + dc)
                            if (
                                0 <= nr < n
                                and 0 <= nc < n
                                and ((nr, nc) not in visited)
                                and (grid[nr][nc] != "A")
                            ):
                                queue.append((nr, nc))
                                visited.add((nr, nc))
                    if island_number is None:
                        return False
                    if len(island_cells) != island_number:
                        return False
        return True

    def _check_diagonal_borders(self, grid):
        """检查是否有斜线边（对角相邻的不同岛屿）"""
        n = len(grid)
        island_map = {}
        island_id = 0
        visited = set()
        for i in range(n):
            for j in range(n):
                if grid[i][j] != "A" and (i, j) not in visited:
                    queue = deque([(i, j)])
                    visited.add((i, j))
                    while queue:
                        r, c = queue.popleft()
                        island_map[r, c] = island_id
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = (r + dr, c + dc)
                            if (
                                0 <= nr < n
                                and 0 <= nc < n
                                and (grid[nr][nc] != "A")
                                and ((nr, nc) not in visited)
                            ):
                                queue.append((nr, nc))
                                visited.add((nr, nc))
                    island_id += 1
        for i in range(n - 1):
            for j in range(n - 1):
                if (
                    grid[i][j] != "A"
                    and grid[i + 1][j + 1] != "A"
                    and (grid[i][j + 1] == "A")
                    and (grid[i + 1][j] == "A")
                ):
                    if island_map.get((i, j)) != island_map.get((i + 1, j + 1)):
                        return False
                if (
                    grid[i][j + 1] != "A"
                    and grid[i + 1][j] != "A"
                    and (grid[i][j] == "A")
                    and (grid[i + 1][j + 1] == "A")
                ):
                    if island_map.get((i, j + 1)) != island_map.get((i + 1, j)):
                        return False
        return True

    def extract_answer(self, test_solution: str):
        """从模型的响应中提取答案网格"""
        response = test_solution
        grid_pattern = "\\[\\s*\\[(?:\\s*(?:\"[XA]\"|\\'[XA]\\'|[0-9]+|\"[0-9]+\"|\\'[0-9]+\\')\\s*,\\s*)*\\s*(?:\"[XA]\"|\\'[XA]\\'|[0-9]+|\"[0-9]+\"|\\'[0-9]+\\')\\s*\\]\\s*(?:,\\s*\\[(?:\\s*(?:\"[XA]\"|\\'[XA]\\'|[0-9]+|\"[0-9]+\"|\\'[0-9]+\\')\\s*,\\s*)*\\s*(?:\"[XA]\"|\\'[XA]\\'|[0-9]+|\"[0-9]+\"|\\'[0-9]+\\')\\s*\\]\\s*)*\\]"
        matches = re.findall(grid_pattern, response)
        if matches:
            grid_str = matches[-1]
            try:
                cleaned_grid_str = grid_str.replace("\n", "").replace("\r", "").strip()
                grid = json.loads(cleaned_grid_str)
                for i, row in enumerate(grid):
                    for j, cell in enumerate(row):
                        if isinstance(cell, str) and cell.isdigit():
                            grid[i][j] = int(cell)
                return grid
            except json.JSONDecodeError:
                try:
                    import ast

                    grid = ast.literal_eval(cleaned_grid_str)
                    for i, row in enumerate(grid):
                        for j, cell in enumerate(row):
                            if isinstance(cell, str) and cell.isdigit():
                                grid[i][j] = int(cell)
                    return grid
                except Exception:
                    pass
        else:
            pass
        return None
