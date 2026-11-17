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

"""Unit tests for the reasoning parser factory."""

from __future__ import annotations

import pytest

from northrl.utils import reasoning_parser as reasoning_parser_module


@pytest.mark.parametrize("parser_type", ["qwen3-thinking", "qwen3_thinking"])
def test_qwen3_thinking_uses_sglang_forced_reasoning_parser(
    parser_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeSglangReasoningParser:
        def __init__(
            self,
            *,
            model_type: str,
            stream_reasoning: bool,
            force_reasoning: bool | None,
        ) -> None:
            captured.update(
                {
                    "model_type": model_type,
                    "stream_reasoning": stream_reasoning,
                    "force_reasoning": force_reasoning,
                }
            )
            self.detector = object()

    monkeypatch.setattr(reasoning_parser_module, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(
        reasoning_parser_module,
        "SglangReasoningParser",
        _FakeSglangReasoningParser,
        raising=False,
    )

    parser = reasoning_parser_module.create_reasoning_parser(parser_type)

    assert isinstance(parser, reasoning_parser_module.NorthRLReasoningParser)
    assert captured == {
        "model_type": "qwen3-thinking",
        "stream_reasoning": False,
        "force_reasoning": None,
    }
