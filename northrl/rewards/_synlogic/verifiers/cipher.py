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


class CipherVerifier(Verifier):
    """
    Verifier for Cipher
    """

    def verify(self, data: Data, test_answer: str):
        """The scoring function for cipher."""
        test_solution = test_answer
        try:
            answer = self.extract_answer(test_solution)
            answer = answer.split("```python")[-1].split("```")[0].strip()
            import ast as _ast  # [0717] 模型输出与 gold 都用 literal_eval 解析矩阵字面量

            answer = _ast.literal_eval(answer)
            gold_answer = data.answer
            gold_answer = (
                _ast.literal_eval(gold_answer) if isinstance(gold_answer, str) else gold_answer
            )
            if answer == gold_answer:
                return 1.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def extract_answer(self, test_solution: str):
        return test_solution
