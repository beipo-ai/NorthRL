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

"""
Wordscapes verifier module for the reasonreason framework.
"""
import re

from ..base import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START, Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic

debug_mode = False


class WordscapesVerifier(Verifier):
    """
    Verifier for Wordscapes game
    """

    def verify(self, data, test_answer: str):
        """
        Verify whether the test answer is consistent with the gold answer

        Args:
            data: WordscapesData
            test_solution: str containing the solution

        Returns:
            float: Score between 0 and 1
        """
        test_solution = test_answer
        try:
            extracted_answer = self.extract_answer(test_solution)
            if not extracted_answer:
                return False
            grid = data.metadata["grid"]
            across_words = data.metadata["across_words"]
            down_words = data.metadata["down_words"]
            if len(extracted_answer) != len(grid):
                return False
            for i, grid_row in enumerate(grid):
                if len(extracted_answer[i]) != len(grid_row):
                    return False
            for i, grid_row in enumerate(grid):
                for j, grid_cell in enumerate(grid_row):
                    if grid_cell == "0" and extracted_answer[i][j].strip():
                        return False
                    if grid_cell == "X" and (not extracted_answer[i][j].strip()):
                        return False
            for word in across_words:
                found = False
                for extracted_row in extracted_answer:
                    row_str = "".join(extracted_row).replace(" ", "").lower()
                    if word.lower() in row_str:
                        found = True
                        break
                if not found and word:
                    return 0
            for word in down_words:
                found = False
                for j, _ in enumerate(extracted_answer[0]):
                    col = []
                    for extracted_row in extracted_answer:
                        if j < len(extracted_row):
                            col.append(extracted_row[j])
                    col_str = "".join(col).replace(" ", "").lower()
                    if word.lower() in col_str:
                        found = True
                        break
                if not found and word:
                    return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        """
        Extract the answer from the test solution

        Args:
            test_solution: str

        Returns:
            list: 2D grid of the answer or None if extraction fails
        """
        try:
            if THOUGHT_DELIMITER_START in test_solution and THOUGHT_DELIMITER_END in test_solution:
                thought_end_pos = test_solution.rfind(THOUGHT_DELIMITER_END)
                if thought_end_pos >= 0:
                    test_solution = test_solution[thought_end_pos + len(THOUGHT_DELIMITER_END) :]
            grid_pattern = re.search(
                "\\[\\s*\\[(?:\\s*\\[)?(.+?)(?:\\]\\s*)?\\]\\s*\\]", test_solution, re.DOTALL
            )
            if not grid_pattern:
                return None
            grid_text = grid_pattern.group(1)
            rows = []
            split_rows = re.split("\\],\\s*\\[", grid_text)
            for row_text in split_rows:
                row_text = row_text.strip().strip("[],")
                chars = []
                char_matches = re.findall("\\\"([^\\\"]*)\\\"|\\'([^\\']*)\\'|([^,\\s]+)", row_text)
                for match in char_matches:
                    char = next((x for x in match if x), "")
                    if char in ("0", ""):
                        char = " "
                    chars.append(char)
                if chars:
                    rows.append(chars)
            if not rows or not all(rows):
                return None
            return rows
        except Exception:
            return None
