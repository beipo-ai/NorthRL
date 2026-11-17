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

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class GoodsExchangeVerifier(Verifier):
    """
    验证器用于检查物品交换游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的回答字符串
        @return: 回答是否正确的布尔值
        """
        test_solution = test_answer
        try:
            test_answer = self.extract_answer(test_solution)
            correct_answer = data.metadata["owns_after"]
            model_ownership = self._parse_answer(test_answer)
            correct_ownership = self._parse_answer(correct_answer)
            is_correct = self._compare_answers(model_ownership, correct_ownership)
            if is_correct:
                pass
            else:
                self._print_difference(model_ownership, correct_ownership)
            return is_correct
        except Exception:
            return False

    def _parse_answer(self, answer_str):
        """
        解析答案字符串为物品归属字典

        @param answer_str: 答案字符串，格式为"(('人1','物品1'),('人2','物品2'),...)"或"(人1,物品1),(人2,物品2),..."
        @return: 归属关系字典 {人: 物品}
        """
        if not answer_str:
            return {}
        result = {}
        try:
            answer_str = answer_str.strip()
            import ast as _ast  # [0717] 模型输出禁 eval:期望 tuple 字面量

            pairs = _ast.literal_eval(answer_str)
            if isinstance(pairs, tuple):
                for pair in pairs:
                    if isinstance(pair, tuple) and len(pair) == 2:
                        person, item = pair
                        result[person.strip()] = item.strip()
                return result
        except Exception:
            if answer_str.startswith("("):
                answer_str = answer_str[1:]
            if answer_str.endswith(")"):
                answer_str = answer_str[:-1]
            person_item_pairs = []
            current_pair = ""
            bracket_count = 0
            for char in answer_str:
                if char == "(":
                    bracket_count += 1
                    current_pair += char
                elif char == ")":
                    bracket_count -= 1
                    current_pair += char
                    if bracket_count == 0:
                        person_item_pairs.append(current_pair)
                        current_pair = ""
                elif char == "," and bracket_count == 0:
                    continue
                else:
                    current_pair += char
            for pair in person_item_pairs:
                pair = pair.strip()
                if pair.startswith("("):
                    pair = pair[1:]
                if pair.endswith(")"):
                    pair = pair[:-1]
                try:
                    parts = []
                    quote_count = 0
                    current = ""
                    for char in pair:
                        if char in "\"'" and (len(current) == 0 or current[-1] != "\\"):
                            quote_count = 1 - quote_count
                        if char == "," and quote_count == 0:
                            parts.append(current.strip())
                            current = ""
                        else:
                            current += char
                    if current:
                        parts.append(current.strip())
                    if len(parts) >= 2:
                        person = parts[0].strip().strip("'\"")
                        item = parts[1].strip().strip("'\"")
                        result[person] = item
                except Exception:
                    pass
        return result

    def _compare_answers(self, model_ownership, correct_ownership):
        """
        比较两个归属关系字典是否相同

        @param model_ownership: 模型回答的归属关系
        @param correct_ownership: 正确的归属关系
        @return: 是否完全一致
        """
        if len(model_ownership) != len(correct_ownership):
            return False
        model_lower_to_original = {person.lower(): person for person in model_ownership}
        for person in correct_ownership:
            if person.lower() not in model_lower_to_original:
                return False
            model_person = model_lower_to_original[person.lower()]
            if model_ownership[model_person].lower() != correct_ownership[person].lower():
                return False
        return True

    def _print_difference(self, model_ownership, correct_ownership):
        """
        打印两个归属关系之间的差异

        @param model_ownership: 模型回答的归属关系
        @param correct_ownership: 正确的归属关系
        """
        model_lower_to_original = {person.lower(): person for person in model_ownership}
        correct_lower_to_original = {person.lower(): person for person in correct_ownership}
        for person in correct_ownership:
            person_lower = person.lower()
            if person_lower not in model_lower_to_original:
                pass
            else:
                model_person = model_lower_to_original[person_lower]
                if model_ownership[model_person].lower() != correct_ownership[person].lower():
                    pass
        for person in model_ownership:
            if person.lower() not in correct_lower_to_original:
                pass

    def extract_answer(self, test_solution):
        """从文本中提取答案。

        Args:
            text (str): 输入文本

        Returns:
            str: 提取的答案，格式为 "(('人1','物品1'),('人2','物品2'),...)"
        """
        text = test_solution
        if not text:
            return ""
        code_block_pattern = "```python\\s*\\n(.*?)\\n```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        if code_blocks:
            last_block = code_blocks[-1].strip()
            if last_block.startswith("(") and last_block.endswith(")"):
                return last_block
        return ""
