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
from collections import defaultdict

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class NorinoriVerifier(Verifier):
    """
    Norinori 游戏的验证器
    检查提交的答案是否符合 Norinori 游戏规则
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证 Norinori 游戏的答案

        参数:
        data -- 游戏数据，包含区域网格等信息
        test_solution -- 用户提交的答案，应为多米诺坐标列表

        返回:
        bool -- 答案是否正确
        """
        test_solution = test_answer
        try:
            region_grid = data.metadata["region_grid"]
            n = len(region_grid)
            dominoes = self._parse_answer(test_solution)
            if dominoes is None:
                return False
            if not self._check_domino_shapes(dominoes):
                return False
            covered = [[False for _ in range(n)] for _ in range(n)]
            for domino in dominoes:
                for i, j in domino:
                    i -= 1
                    j -= 1
                    if i < 0 or i >= n or j < 0 or (j >= n):
                        return False
                    if covered[i][j]:
                        return False
                    covered[i][j] = True
            if not self._check_domino_adjacency(dominoes, n):
                return False
            region_coverage: defaultdict[str, int] = defaultdict(int)
            for i in range(n):
                for j in range(n):
                    if covered[i][j] and region_grid[i][j] != "X":
                        region_coverage[region_grid[i][j]] += 1
            for count in region_coverage.values():
                if count != 2:
                    return False
            for i in range(n):
                for j in range(n):
                    if region_grid[i][j] == "X" and (not covered[i][j]):
                        return False
            return True
        except Exception:
            return False

    def _parse_answer(self, test_solution: str):
        """
        解析答案字符串，提取多米诺坐标

        参数:
        test_solution -- 答案字符串

        返回:
        list -- 多米诺坐标列表，如果格式不正确则返回None
        """
        try:
            pattern = "\\[\\((\\d+),\\s*(\\d+)\\),\\s*\\((\\d+),\\s*(\\d+)\\)\\]"
            matches = re.findall(pattern, test_solution)
            if not matches:
                pattern = "\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)\\s*,\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)"
                matches = re.findall(pattern, test_solution)
            dominoes = []
            for match in matches:
                i1, j1, i2, j2 = map(int, match)
                dominoes.append([(i1, j1), (i2, j2)])
            return dominoes
        except Exception:
            return None

    def _check_domino_shapes(self, dominoes):
        """
        检查所有多米诺是否都是1×2或2×1的形状

        参数:
        dominoes -- 多米诺坐标列表

        返回:
        bool -- 是否所有多米诺都符合形状要求
        """
        for domino in dominoes:
            if len(domino) != 2:
                return False
            (i1, j1), (i2, j2) = domino
            if not (i1 == i2 and abs(j1 - j2) == 1 or (j1 == j2 and abs(i1 - i2) == 1)):
                return False
        return True

    def _check_domino_adjacency(self, dominoes, n):
        """
        检查多米诺之间是否相邻

        参数:
        dominoes -- 多米诺坐标列表
        n -- 网格大小

        返回:
        bool -- 是否所有多米诺都不相邻
        """
        grid = [[-1 for _ in range(n + 2)] for _ in range(n + 2)]
        for idx, domino in enumerate(dominoes):
            for i, j in domino:
                grid[i][j] = idx
        for idx, domino in enumerate(dominoes):
            for i, j in domino:
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = (i + di, j + dj)
                    if 1 <= ni <= n and 1 <= nj <= n:
                        if grid[ni][nj] != -1 and grid[ni][nj] != idx:
                            return False
        return True

    def extract_answer(self, test_solution: str, strict=False):
        """
        从回答中提取答案

        参数:
        test_solution -- 用户的回答
        strict -- 是否严格模式

        返回:
        str -- 提取的答案
        """
        del strict
        answer_patterns = [
            "\\[\\s*\\[\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\)\\s*,\\s*\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\)\\s*\\]",
            "答案是\\s*(.*?)\\s*$",
            "answer is\\s*(.*?)\\s*$",
            "solution is\\s*(.*?)\\s*$",
        ]
        for pattern in answer_patterns:
            matches = re.findall(pattern, test_solution, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[-1]
        return test_solution
