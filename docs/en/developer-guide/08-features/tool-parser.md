English | [简体中文](../../../zh/developer-guide/08-features/tool-parser.md) | [Documentation index](../../index.md)

# Tool Parser Framework

The Tool Parser Framework provides a flexible, extensible system for parsing tool calls from different model formats. Different language models use different formats for tool calls, and this framework allows you to easily switch between parsers or add new ones via configuration.

## Dependencies

The tool parser now uses [SGLang's](https://github.com/sgl-project/sglang) function call parser implementation under the hood. To use the full functionality, install with:

```bash
pip install 'NorthRL[sglang]'
```

If SGLang is not installed, NorthRL will fall back to a minimal parser that only handles basic `<tool_call>JSON</tool_call>` format.

## Overview

Previously, tool parsing logic was hardcoded in the service holders (Weaver and Tinker). The tool parser framework refactors this into a centralized, configurable system that separates concerns:

- **Service Holders** return the raw model response
- **Response Parsers** select the built-in pipeline or a custom callback
- **Inference Service Client** invokes parsing and returns OpenAI-format results

## Architecture

The framework consists of three main components:

### 1. SGLang Format Detectors

NorthRL uses SGLang's `BaseFormatDetector` implementations for parsing. SGLang provides detectors for:
- `Qwen25Detector` - Qwen 2.5/3.0 multi-line JSON format
- `Qwen3CoderDetector` - Qwen 3 Coder XML-style parameters
- `DeepSeekV3Detector` - DeepSeek V3 format
- `DeepSeekV31Detector` - DeepSeek V3.1 special Unicode tokens
- `DeepSeekV32Detector` - DeepSeek V3.2 format
- `GptOssDetector` - GPT OSS/Harmony commentary format
- `Llama32Detector` - Llama 3.x format
- `MistralDetector` - Mistral format
- `PythonicDetector` - Pythonic tool call format
- `KimiK2Detector` - Kimi K2 format
- `Step3Detector` - Step3 format
- `GlmDetector` - GLM format
- And many more...

### 2. SglangToolParserAdapter

NorthRL's adapter wraps SGLang detectors to provide a unified interface:
- Converts NorthRL tool dicts to SGLang `Tool` objects
- Converts SGLang `ToolCallItem` to NorthRL `ToolCallItem` format
- Handles tool name validation when tools are provided

### 3. Factory Function

`create_tool_parser(parser_type: str)` instantiates parsers by type:

```python
from northrl.utils.tool_parser import create_tool_parser

parser = create_tool_parser("qwen25")
```

### 4. Core Types

Data classes for structured results (maintained for backward compatibility):

```python
@dataclass
class ToolCallItem:
    id: str                    # Unique identifier
    type: str                  # "function"
    function: dict[str, Any]   # {"name": str, "arguments": str}

@dataclass
class ParseResult:
    tool_calls: list[ToolCallItem] | None
    is_valid: bool
```

**Note:** Internally, SGLang uses a different format (`tool_index`, `name`, `parameters`), but NorthRL's adapter transparently converts to the OpenAI-compatible format above.

## Supported Formats

All formats are provided by SGLang's detectors. See the [Config Options](#built-in-parser-options) table below for the complete list.

### Qwen 2.5/3.0 (`qwen25`)

Multi-line JSON format supporting multiple calls:

```xml
<tool_call>
{"name": "get_weather", "arguments": {"city": "NYC"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"operation": "add", "a": 5, "b": 3}}
</tool_call>
```

**Config:** `qwen25`, `qwen`, or `qwen3`

**Use case:** Qwen family models, multiple tool calls per response

### Qwen 3 Coder (`qwen3_coder`)

XML-style function and parameter tags:

```xml
<tool_call>
<function=execute_bash>
<parameter=command>
pwd && ls
</parameter>
<parameter=flag>
-la
</parameter>
</function>
</tool_call>
```

**Config:** `qwen3_coder`

**Use case:** Qwen 3 Coder models, multi-parameter functions with text content

**Features:**
- Multiple named parameters per function
- Text-based parameter values (not just JSON)
- Safe value parsing (JSON → Python literals → raw string)

### DeepSeek V3.1 (`deepseekv31`)

Special Unicode tokens with JSON arguments:

```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

Multiple calls:

```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>calculate<｜tool▁sep｜>{"operation": "add"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

**Config:** `deepseekv31`, `deepseek_v31`, or `deepseek`

**Use case:** DeepSeek V3.1 models

**Features:**
- Special Unicode tokens for delimiting
- Function name followed by JSON arguments
- Supports multiple tool calls in one response

### GPT OSS (`gpt-oss`)

Harmony-style commentary channel format:

```
<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>
```

With namespace:

```
<|channel|>commentary to=tools.calculate<|constrain|>json<|message|>{"operation": "add", "a": 5, "b": 3}<|call|>
```

**Config:** `gpt-oss`

**Use case:** GPT OSS/T4-style models with harmony format

**Features:**
- Commentary channel with `to=` directive
- Supports namespaced functions (e.g., `functions.name`, `tools.name`)
- JSON constraint specification

**Note:** This is a simplified version for basic parsing. Full streaming support requires HarmonyParser.

## Configuration

`rollout.parsers.type` explicitly selects one of two response-parsing modes.
Fields from the other mode are rejected during config validation.

### Built-in response parser

Use NorthRL's reasoning parser followed by its tool parser:

```yaml
rollout:
  parsers:
    type: builtin
    reasoning: think_tag
    tool: qwen25
```

`reasoning` defaults to `think_tag` and `tool` defaults to `qwen25`, so the
shortest equivalent config is:

```yaml
rollout:
  parsers:
    type: builtin
```

Omitting a field selects its default. Setting a field to null disables that
stage:

```yaml
rollout:
  parsers:
    type: builtin
    reasoning: null  # Do not extract reasoning content.
    tool: qwen25
```

The built-in parser processes the raw response in this order:

1. Extract reasoning and cleaned content, unless `reasoning` is null.
2. Parse tool calls from the cleaned content, unless `tool` is null.
3. Return the final OpenAI-compatible assistant message.

Legacy built-in configs without `type` remain accepted for compatibility.
Their old semantics are preserved: a missing, null, or blank `reasoning` or
`tool` value selects its default. The null-means-disabled rule applies only
when `type: builtin` is explicit. New recipes should always select the mode
explicitly.

### Custom response parser

A custom parser completely replaces the built-in reasoning and tool pipeline:

```yaml
rollout:
  parsers:
    type: custom
    module_path: parsers/response_parser.py
```

`module_path` is the only custom-mode option. It must point to a `.py` file and
a relative path is resolved from the directory containing the recipe YAML.
The module must define a synchronous function named `parse_response`:

```python
from collections.abc import Mapping
from typing import Any


def parse_response(
    *,
    response: str,
    tools: list[dict[str, Any]],
) -> Mapping[str, Any]:
    return {
        "role": "assistant",
        "content": response.strip(),
    }
```

NorthRL calls the function with keyword arguments.

| Input | Type | Meaning |
|---|---|---|
| `response` | `str` | Raw model output before any reasoning or tool parsing |
| `tools` | `list[dict[str, Any]]` | A deep copy of the request's OpenAI-compatible tool definitions; `[]` when absent |

The return value is the final assistant message. NorthRL copies it and validates
this boundary:

| Output field | Type | Required | Meaning |
|---|---|---|---|
| `role` | `"assistant"` | No | Defaults to `"assistant"`; any other value is rejected |
| `content` | `str \| None` | Conditional | Text shown to the caller |
| `reasoning_content` | `str \| None` | No | Structured reasoning content |
| `tool_calls` | `list[Mapping[str, Any]] \| None` | Conditional | OpenAI-compatible tool calls |
| other fields | `Any` | No | Preserved without rewriting |

The result must contain non-null `content` or at least one tool call. For tool
calls, the custom parser owns the complete OpenAI shape:

```python
import json


def parse_response(*, response, tools):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"query": response}),
                },
            }
        ],
    }
```

NorthRL does not generate tool-call IDs, rewrite function names, or serialize
`function.arguments`; arguments must already be a JSON string.

The parser file is loaded directly rather than as a Python package. Its
normal dependencies must be importable, and package-relative imports from the
recipe directory are not supported. Import errors, callback exceptions,
async/awaitable results, and invalid messages fail immediately without falling
back to the built-in parser.

Response parsing applies to `generate()` and `generate_with_token()`.
`completion()` continues to return raw text.

### Built-in parser options

Built-in reasoning values currently include `think_tag`, `think`, `default`,
`deepseek_r1`, `qwen3`, `qwen3-thinking`, `kimi`, `minimax_append_think`,
`step3`, `step3p5`, `nano_v3`, `gpt_oss`, `glm45`, `deepseek_v3`, `kimi_k2`,
`minimax`, and `interns1`. Hyphens and underscores are interchangeable for
reasoning names.

Built-in tool values are listed below. They are passed directly to SGLang, so
spelling, hyphens, and underscores must match its registered names.


**Important:** Use SGLang's parser names directly in your config. No mapping or aliases.

| Parser | Config Value | Format | Notes |
|--------|-------------|--------|-------|
| Qwen 2.5/3.0 | `qwen25` or `qwen` | Multi-line JSON with tags | Default |
| Qwen 3 Coder | `qwen3_coder` | XML-style parameters | |
| DeepSeek V3 | `deepseekv3` | DeepSeek V3 format | |
| DeepSeek V3.1 | `deepseekv31` | Unicode tokens | |
| DeepSeek V3.2 | `deepseekv32` | DeepSeek V3.2 format | |
| GPT OSS | `gpt-oss` | Harmony commentary | |
| Llama 3.x | `llama3` | Llama tool format | |
| Mistral | `mistral` | Mistral format | |
| Pythonic | `pythonic` | Python function calls | |
| Kimi K2 | `kimi_k2` | Kimi K2 format | |
| Step3 | `step3` | Step3 format | |
| Step3.5 | `step3p5` | Step3.5 format | |
| GLM | `glm` or `glm45` | GLM format | |
| GLM 4.7 | `glm47` | GLM 4.7 format | |
| Hermes | `hermes` | Hermes format | |
| InternLM | `interns1` | InternLM S1 format | |
| MiniMax M2 | `minimax-m2` | MiniMax M2 format | |
| Trinity | `trinity` | Trinity format | |
| GigaChat3 | `gigachat3` | GigaChat3 format | |
| MiMo | `mimo` | MiMo format | |
| LFM2 | `lfm2` | LFM2 format | |

**Note:** All parsers are powered by SGLang's detectors. Parser names must match SGLang's `FunctionCallParser.ToolCallParserEnum` exactly.

## Usage

### Basic Usage

```python
from northrl.utils.tool_parser import create_tool_parser

# Create a parser
parser = create_tool_parser("qwen25")

# Define available tools (optional, for validation)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }
]

# Parse response text
response = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
result = parser.parse(response, tools=tools)

if result.is_valid and result.tool_calls:
    for tool_call in result.tool_calls:
        print(f"Function: {tool_call.function['name']}")
        print(f"Arguments: {tool_call.function['arguments']}")
        print(f"ID: {tool_call.id}")
```

**Note:** Passing `tools` is optional but recommended for validation. SGLang will validate tool names against the provided tools list.

### Integration with NorthRL

Remote service holders return the raw model response and do not parse reasoning
or tool calls. `RemoteApiInferenceServiceClient` owns the single response-parsing
entry point:

```python
self._response_parser = create_response_parser(
    config.rollout.parsers,
    config_file_path=getattr(config, "_config_file_path", None),
)

message = self._response_parser.parse(raw_response, tools)
```

The factory returns either the composed built-in pipeline or the recipe-local
custom parser. Both accept the same raw response and return the same assistant
message shape.

## Extending parsing

### Add a built-in tool format

Built-in tool detectors come from SGLang. `create_tool_parser()` passes the
configured name to `FunctionCallParser.ToolCallParserEnum` and adapts its result
to NorthRL's `ParseResult`. Add or register the detector in the supported SGLang
version, then document the exact config name here. NorthRL does not maintain a
second set of model-specific detector classes.

### Add a recipe-local response format

Use `type: custom` when a recipe needs parsing behavior that is not a reusable
SGLang tool format. The recipe module receives the complete raw response and
returns the final assistant message, so no NorthRL factory registration is needed.

## Architecture

- Remote service holders return raw model text and token metadata.
- `RemoteApiInferenceServiceClient` invokes exactly one `ResponseParser`.
- The built-in response parser composes the reasoning and tool parsers.
- The custom response parser invokes the recipe's fixed `parse_response` hook.
- Both response parsers return the same assistant-message shape.

## File Structure

```text
northrl/utils/
├── reasoning_parser/        # Built-in reasoning adapters
├── tool_parser/             # Built-in SGLang tool adapters
└── response_parser.py       # Unified built-in/custom response pipeline
```

## Reference

This implementation is inspired by SGLang's tool parser framework (`.vibe/tool_parser/`), adapted for NorthRL's architecture and needs.

## See Also

- [Reasoning Parser](reasoning-parser.md) - Extract and structure reasoning content
