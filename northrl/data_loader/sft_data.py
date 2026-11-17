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

"""Minimal SFT conversion: messages in, aligned token IDs and loss mask out."""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import math
import numbers
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from northrl.utils.path_utils import resolve_path_from_config

RenderChat = Callable[[dict[str, Any], Any], tuple[list[int], list[int]]]

logger = logging.getLogger(__name__)


def load_render_chat(reference: str, config_file_path: str | None) -> RenderChat:
    """Load ``path.py:function`` relative to the recipe YAML."""

    module_path, separator, function_name = reference.rpartition(":")
    if not separator or not module_path or not function_name:
        raise ValueError("data.train.render_chat must use 'path/to/file.py:function' format")

    resolved = resolve_path_from_config(module_path, config_file_path)
    if resolved is None:
        raise ValueError(f"Could not resolve SFT render_chat module: {module_path}")
    path = Path(resolved).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SFT render_chat module not found: {path}")

    spec = importlib.util.spec_from_file_location(
        f"_northrl_sft_render_chat_{uuid.uuid4().hex}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SFT render_chat module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    callback = getattr(module, function_name, None)
    if not callable(callback):
        raise TypeError(f"SFT render_chat '{reference}' is not callable")
    if inspect.iscoroutinefunction(callback):
        raise TypeError(f"SFT render_chat '{reference}' must be synchronous")
    return callback


def default_render_chat(row: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int]]:
    """Render cumulative chat text and train only assistant text segments."""

    messages_value = _present_value(row.get("messages"))
    if messages_value is None:
        raise ValueError("default SFT render_chat requires row['messages']")
    if isinstance(messages_value, (str, bytes)) or not isinstance(messages_value, Sequence):
        raise TypeError("row['messages'] must be a sequence of message objects")
    if not all(isinstance(message, Mapping) for message in messages_value):
        raise TypeError("row['messages'] must contain message objects")
    messages = [dict(message) for message in messages_value]
    if not messages:
        raise ValueError("row['messages'] must not be empty")
    _normalize_tool_call_arguments(messages)
    tools = _present_value(row.get("tools"))

    input_ids: list[int] = []
    loss_mask: list[int] = []
    history: list[dict[str, Any]] = []
    rendered_text = ""

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            history.append(message)
            continue
        if index == 0:
            raise ValueError("row['messages'] must not start with an assistant message")
        if messages[index - 1].get("role") == "assistant":
            raise ValueError("row['messages'] must not contain consecutive assistant messages")

        prompt_text = _render_chat_template(
            tokenizer,
            history,
            tools=tools,
            add_generation_prompt=True,
        )
        prompt_segment = _appended_text(
            rendered_text,
            prompt_text,
            context=f"prompt before messages[{index}]",
        )
        _append_text_segment(
            input_ids,
            loss_mask,
            prompt_segment,
            mask_value=0,
            tokenizer=tokenizer,
        )

        history.append(message)
        round_text = _render_chat_template(
            tokenizer,
            history,
            tools=tools,
            add_generation_prompt=False,
        )
        assistant_segment = _appended_text(
            prompt_text,
            round_text,
            context=f"assistant messages[{index}]",
        )
        appended_tokens = _append_text_segment(
            input_ids,
            loss_mask,
            assistant_segment,
            mask_value=1,
            tokenizer=tokenizer,
        )
        if appended_tokens == 0:
            raise ValueError(
                f"The tokenizer chat template rendered no completion for assistant messages[{index}]"
            )
        rendered_text = round_text

    if messages[-1].get("role") != "assistant":
        tail_text = _render_chat_template(
            tokenizer,
            history,
            tools=tools,
            add_generation_prompt=False,
        )
        tail_segment = _appended_text(
            rendered_text,
            tail_text,
            context="trailing non-assistant messages",
        )
        _append_text_segment(
            input_ids,
            loss_mask,
            tail_segment,
            mask_value=0,
            tokenizer=tokenizer,
        )

    return input_ids, loss_mask


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> None:
    """Decode JSON-string tool arguments before applying a chat template."""

    for message_index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, (str, bytes)) or not isinstance(tool_calls, Sequence):
            continue

        normalized_tool_calls: list[Any] = []
        for tool_call_index, tool_call_value in enumerate(tool_calls):
            if not isinstance(tool_call_value, Mapping):
                normalized_tool_calls.append(tool_call_value)
                continue

            tool_call = dict(tool_call_value)
            function_value = tool_call.get("function")
            if not isinstance(function_value, Mapping) or "arguments" not in function_value:
                normalized_tool_calls.append(tool_call)
                continue

            function = dict(function_value)
            arguments = function["arguments"]
            path = f"messages[{message_index}].tool_calls[{tool_call_index}]"
            path += ".function.arguments"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} must contain valid JSON") from exc
            if not isinstance(arguments, Mapping):
                raise TypeError(f"{path} must be a JSON object or a JSON-encoded object")

            function["arguments"] = dict(arguments)
            tool_call["function"] = function
            normalized_tool_calls.append(tool_call)

        message["tool_calls"] = normalized_tool_calls


def _render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: Any | None,
    add_generation_prompt: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(tokenize=False) must return a string")
    return rendered


def _appended_text(previous: str, current: str, *, context: str) -> str:
    if not current.startswith(previous):
        raise ValueError(
            "The tokenizer chat template must preserve previously rendered text when appending "
            f"{context}. Configure data.train.render_chat for this model"
        )
    return current[len(previous) :]


def _append_text_segment(
    input_ids: list[int],
    loss_mask: list[int],
    text: str,
    *,
    mask_value: int,
    tokenizer: Any,
) -> int:
    if not text:
        return 0
    # Keep prompt/assistant boundaries explicit so one token cannot span both masks.
    segment_ids = _integer_list(
        "input_ids",
        tokenizer.encode(text, add_special_tokens=False),
    )
    input_ids.extend(segment_ids)
    loss_mask.extend([mask_value] * len(segment_ids))
    return len(segment_ids)


def prepare_sft_item(
    item: dict[str, Any],
    *,
    tokenizer: Any | None,
    render_chat: RenderChat | None,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Normalize one tokenized or messages-format SFT record."""

    input_ids = _present_value(item.get("input_ids"))
    if input_ids is not None:
        loss_mask = _present_value(item.get("loss_mask"))
        if loss_mask is None:
            raise ValueError("tokenized SFT data requires loss_mask")
    else:
        messages = _present_value(item.get("messages"))
        if render_chat is None and messages is None:
            raise ValueError("SFT data requires either input_ids or messages")
        if tokenizer is None:
            raise ValueError("SFT render_chat requires models.actor.tokenizer")
        callback = render_chat or default_render_chat
        result = callback(dict(item), tokenizer)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("render_chat must return (input_ids, loss_mask)")
        input_ids, loss_mask = result

    tokens = _integer_list("input_ids", input_ids)
    mask = _binary_mask(loss_mask)
    if len(tokens) != len(mask):
        raise ValueError(
            f"input_ids and loss_mask must have equal length: {len(tokens)} != {len(mask)}"
        )
    if not tokens:
        raise ValueError("SFT input_ids must not be empty")

    original_length = len(tokens)
    tokens = tokens[:max_sequence_length]
    mask = mask[:max_sequence_length]
    if original_length > max_sequence_length:
        logger.warning(
            "Truncated SFT sample %s from %d to max_sequence_length=%d",
            item.get("index", "<unknown>"),
            original_length,
            max_sequence_length,
        )
    mask[0] = 0
    if not any(mask):
        raise ValueError("SFT loss_mask must select at least one token")

    prepared = dict(item)
    prepared.pop("messages", None)
    prepared.pop("labels", None)
    prepared["input_ids"] = tokens
    prepared["loss_mask"] = mask
    return prepared


def _present_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, numbers.Real) and math.isnan(float(value)):
        return None
    return value


def _integer_list(name: str, values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if not all(isinstance(value, numbers.Integral) for value in values):
        raise TypeError(f"{name} must contain integers")
    return [int(value) for value in values]


def _binary_mask(values: Any) -> list[int]:
    mask = _integer_list("loss_mask", values)
    if any(value not in (0, 1) for value in mask):
        raise ValueError("loss_mask must contain only 0 or 1")
    return mask
