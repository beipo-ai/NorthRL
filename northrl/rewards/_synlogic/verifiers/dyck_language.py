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

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class DyckLanguageVerifier(Verifier):
    """
    验证器用于检查Dyck Language游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的回答字符串
        @return: 回答是否正确的布尔值
        """
        try:
            full_sequence = data.metadata["full_sequence"]
            extracted_answer = self.extract_answer(test_answer)
            is_correct = extracted_answer == full_sequence
            if is_correct:
                pass
            else:
                pass
            return is_correct
        except Exception:
            return False

    def extract_answer(self, test_solution: str) -> str:
        """
        从模型的回答中提取括号序列答案

        @param test_solution: 模型的完整回答
        @return: 提取的答案
        """
        if not test_solution:
            return ""

        def clean_text(text: str) -> str:
            """清理文本，处理转义字符和空白字符"""
            text = "".join(text.split())
            text = text.replace("\\n", "")
            text = text.replace("\\t", "")
            text = text.replace("\\r", "")
            text = text.replace("\\\\", "\\")
            if len(text) >= 2:
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                elif text.startswith("'") and text.endswith("'"):
                    text = text[1:-1]
            return text

        return clean_text(test_solution)
