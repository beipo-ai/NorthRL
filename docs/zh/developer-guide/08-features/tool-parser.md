[English](../../../en/developer-guide/08-features/tool-parser.md) | 简体中文 | [文档目录](../../index.md)

<a id="tool-parser-framework"></a>

# 工具解析器框架

工具解析器框架提供灵活、可扩展的系统，用于解析不同模型格式的工具调用。不同语言模型使用不同工具调用格式，此框架允许通过配置轻松切换解析器或添加新解析器。

<a id="dependencies"></a>

## 依赖

工具解析器目前在底层使用 [SGLang](https://github.com/sgl-project/sglang) 的函数调用解析实现。使用完整功能时，请执行以下安装命令：

```bash
pip install 'NorthRL[sglang]'
```

若未安装 SGLang，NorthRL 回退到仅处理基本 `<tool_call>JSON</tool_call>` 格式的最简解析器。

<a id="overview"></a>

## 概述

此前，工具解析逻辑硬编码在服务持有器（Weaver 和 Tinker）中。工具解析器框架将其重构为集中、可配置的系统，分离各项职责：

- **服务持有器**返回原始模型响应
- **响应解析器**选择内置流水线或自定义回调
- **推理服务客户端**调用解析逻辑并返回 OpenAI 格式结果

<a id="architecture"></a>

## 架构

框架包含三个主要组件：

<a id="1-sglang-format-detectors"></a>

### 1. SGLang 格式检测器

NorthRL 使用 SGLang 的 `BaseFormatDetector` 实现进行解析。SGLang 提供以下检测器：
- `Qwen25Detector` - Qwen 2.5/3.0 多行 JSON 格式
- `Qwen3CoderDetector` - Qwen 3 Coder XML 风格参数
- `DeepSeekV3Detector` - DeepSeek V3 格式
- `DeepSeekV31Detector` - DeepSeek V3.1 特殊 Unicode token
- `DeepSeekV32Detector` - DeepSeek V3.2 格式
- `GptOssDetector` - GPT OSS/Harmony commentary 格式
- `Llama32Detector` - Llama 3.x 格式
- `MistralDetector` - Mistral 格式
- `PythonicDetector` - Python 风格工具调用格式
- `KimiK2Detector` - Kimi K2 格式
- `Step3Detector` - Step3 格式
- `GlmDetector` - GLM 格式
- 以及更多类型……

<a id="2-sglangtoolparseradapter"></a>

### 2. SglangToolParserAdapter

NorthRL 适配器封装 SGLang 检测器，提供统一接口：
- 将 NorthRL 工具字典转换为 SGLang `Tool` 对象
- 将 SGLang `ToolCallItem` 转换为 NorthRL `ToolCallItem` 格式
- 提供工具列表时，执行工具名称验证

<a id="3-factory-function"></a>

### 3. 工厂函数

`create_tool_parser(parser_type: str)` 按类型实例化解析器：

```python
from northrl.utils.tool_parser import create_tool_parser

parser = create_tool_parser("qwen25")
```

<a id="4-core-types"></a>

### 4. 核心类型

用于结构化结果的数据类（保留以向后兼容）：

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

**注意：**SGLang 内部使用不同格式（`tool_index`、`name`、`parameters`），但 NorthRL 适配器会透明转换为上述 OpenAI 兼容格式。

<a id="supported-formats"></a>

## 支持的格式

所有格式均由 SGLang 检测器提供。完整列表请参阅下方[配置选项](#built-in-parser-options)表。

<a id="qwen-2530-qwen25"></a>

### Qwen 2.5/3.0（`qwen25`）

支持多个调用的多行 JSON 格式：

```xml
<tool_call>
{"name": "get_weather", "arguments": {"city": "NYC"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"operation": "add", "a": 5, "b": 3}}
</tool_call>
```

**配置：**`qwen25`、`qwen` 或 `qwen3`

**适用场景：**Qwen 系列模型，单个响应中包含多个工具调用

<a id="qwen-3-coder-qwen3_coder"></a>

### Qwen 3 Coder（`qwen3_coder`）

XML 风格的函数与参数标签：

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

**配置：**`qwen3_coder`

**适用场景：**Qwen 3 Coder 模型，具有文本内容的多参数函数

**功能：**
- 每个函数支持多个命名参数
- 基于文本的参数值（不限于 JSON）
- 安全的值解析（JSON → Python 字面量 → 原始字符串）

<a id="deepseek-v31-deepseekv31"></a>

### DeepSeek V3.1（`deepseekv31`）

带 JSON 参数的特殊 Unicode token：

```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

多个调用：

```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>calculate<｜tool▁sep｜>{"operation": "add"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

**配置：**`deepseekv31`、`deepseek_v31` 或 `deepseek`

**适用场景：**DeepSeek V3.1 模型

**功能：**
- 使用特殊 Unicode token 分隔
- 函数名称后跟 JSON 参数
- 支持单个响应中包含多个工具调用

<a id="gpt-oss-gpt-oss"></a>

### GPT OSS（`gpt-oss`）

Harmony 风格的 commentary 通道格式：

```
<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>
```

包含命名空间：

```
<|channel|>commentary to=tools.calculate<|constrain|>json<|message|>{"operation": "add", "a": 5, "b": 3}<|call|>
```

**配置：**`gpt-oss`

**适用场景：**使用 harmony 格式的 GPT OSS/T4 风格模型

**功能：**
- 带 `to=` 指令的 commentary 通道
- 支持命名空间函数（如 `functions.name`、`tools.name`）
- JSON 约束声明

**注意：**这是用于基本解析的简化版本。完整流式支持需要 HarmonyParser。

<a id="configuration"></a>

## 配置

`rollout.parsers.type` 显式选择两种响应解析模式之一。
配置验证会拒绝另一模式的字段。

<a id="built-in-response-parser"></a>

### 内置响应解析器

依次使用 NorthRL 的推理内容解析器和工具解析器：

```yaml
rollout:
  parsers:
    type: builtin
    reasoning: think_tag
    tool: qwen25
```

`reasoning` 默认为 `think_tag`，`tool` 默认为 `qwen25`，因此
最短等效配置为：

```yaml
rollout:
  parsers:
    type: builtin
```

省略字段时使用默认值。将字段设为 null 可禁用对应
阶段：

```yaml
rollout:
  parsers:
    type: builtin
    reasoning: null  # Do not extract reasoning content.
    tool: qwen25
```

内置解析器按以下顺序处理原始响应：

1. 提取推理内容和清理后的文本，除非 `reasoning` 为 null。
2. 从清理后的文本解析工具调用，除非 `tool` 为 null。
3. 返回最终的 OpenAI 兼容助手消息。

为保持兼容，仍接受不带 `type` 的旧版内置配置。
保留其旧语义：`reasoning` 或 `tool` 缺失、为 null 或
空白时使用默认值。null 表示禁用的规则仅在
显式设置 `type: builtin` 时适用。新配方应始终显式
选择模式。

<a id="custom-response-parser"></a>

### 自定义响应解析器

自定义解析器完全替代内置推理内容和工具解析流水线：

```yaml
rollout:
  parsers:
    type: custom
    module_path: parsers/response_parser.py
```

`module_path` 是自定义模式唯一选项，必须指向 `.py` 文件，
相对路径基于配方 YAML 所在目录解析。
模块必须定义名为 `parse_response` 的同步函数：

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

NorthRL 使用关键字参数调用该函数。

| 输入 | 类型 | 含义 |
|---|---|---|
| `response` | `str` | 任何推理内容或工具解析之前的原始模型输出 |
| `tools` | `list[dict[str, Any]]` | 请求中 OpenAI 兼容工具定义的深拷贝；缺失时为 `[]` |

返回值是最终助手消息。NorthRL 复制该值，并验证
以下接口约定：

| 输出字段 | 类型 | 必需 | 含义 |
|---|---|---|---|
| `role` | `"assistant"` | 否 | 默认为 `"assistant"`；拒绝其他任何值 |
| `content` | `str \| None` | 有条件 | 向调用方展示的文本 |
| `reasoning_content` | `str \| None` | 否 | 结构化推理内容 |
| `tool_calls` | `list[Mapping[str, Any]] \| None` | 有条件 | OpenAI 兼容工具调用 |
| 其他字段 | `Any` | 否 | 保留且不改写 |

结果必须包含非 null 的 `content` 或至少一个工具调用。对于工具
调用，自定义解析器负责生成完整 OpenAI 结构：

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

NorthRL 不生成工具调用 ID、不改写函数名，也不序列化
`function.arguments`；参数必须已经是 JSON 字符串。

解析器文件直接加载，不作为 Python 包加载。其
普通依赖必须可导入，且不支持从配方目录进行
包相对导入。导入错误、回调异常、
异步/可等待结果以及无效消息都会立即失败，
不会回退到内置解析器。

响应解析适用于 `generate()` 和 `generate_with_token()`。
`completion()` 继续返回原始文本。

<a id="built-in-parser-options"></a>

### 内置解析器选项

内置 reasoning 值目前包括 `think_tag`、`think`、`default`、
`deepseek_r1`、`qwen3`、`qwen3-thinking`、`kimi`、`minimax_append_think`、
`step3`、`step3p5`、`nano_v3`、`gpt_oss`、`glm45`、`deepseek_v3`、`kimi_k2`、
`minimax` 和 `interns1`。reasoning 名称中的连字符和下划线
可互换。

内置 tool 值如下。它们直接传给 SGLang，因此
拼写、连字符和下划线必须与其注册名称一致。


**重要：**直接在配置中使用 SGLang 解析器名称，不提供映射或别名。

| 解析器 | 配置值 | 格式 | 备注 |
|--------|-------------|--------|-------|
| Qwen 2.5/3.0 | `qwen25` 或 `qwen` | 带标签的多行 JSON | 默认 |
| Qwen 3 Coder | `qwen3_coder` | XML 风格参数 | |
| DeepSeek V3 | `deepseekv3` | DeepSeek V3 格式 | |
| DeepSeek V3.1 | `deepseekv31` | Unicode token | |
| DeepSeek V3.2 | `deepseekv32` | DeepSeek V3.2 格式 | |
| GPT OSS | `gpt-oss` | Harmony commentary | |
| Llama 3.x | `llama3` | Llama 工具格式 | |
| Mistral | `mistral` | Mistral 格式 | |
| Pythonic | `pythonic` | Python 函数调用 | |
| Kimi K2 | `kimi_k2` | Kimi K2 格式 | |
| Step3 | `step3` | Step3 格式 | |
| Step3.5 | `step3p5` | Step3.5 格式 | |
| GLM | `glm` 或 `glm45` | GLM 格式 | |
| GLM 4.7 | `glm47` | GLM 4.7 格式 | |
| Hermes | `hermes` | Hermes 格式 | |
| InternLM | `interns1` | InternLM S1 格式 | |
| MiniMax M2 | `minimax-m2` | MiniMax M2 格式 | |
| Trinity | `trinity` | Trinity 格式 | |
| GigaChat3 | `gigachat3` | GigaChat3 格式 | |
| MiMo | `mimo` | MiMo 格式 | |
| LFM2 | `lfm2` | LFM2 格式 | |

**注意：**所有解析器均由 SGLang 检测器提供。解析器名称必须与 SGLang 的 `FunctionCallParser.ToolCallParserEnum` 完全一致。

<a id="usage"></a>

## 使用

<a id="basic-usage"></a>

### 基础用法

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

**注意：**`tools` 可选，但建议传入以便验证。SGLang 会根据提供的工具列表验证工具名称。

<a id="integration-with-northrl"></a>

### 与 NorthRL 集成

远程服务持有器返回原始模型响应，不解析推理内容
或工具调用。`RemoteApiInferenceServiceClient` 提供唯一的响应解析
入口：

```python
self._response_parser = create_response_parser(
    config.rollout.parsers,
    config_file_path=getattr(config, "_config_file_path", None),
)

message = self._response_parser.parse(raw_response, tools)
```

工厂返回组合后的内置流水线或配方本地的
自定义解析器。两者均接收相同原始响应，并返回相同助手
消息结构。

<a id="extending-parsing"></a>

## 扩展解析

<a id="add-a-built-in-tool-format"></a>

### 添加内置工具格式

内置工具检测器来自 SGLang。`create_tool_parser()` 将配置
名称传给 `FunctionCallParser.ToolCallParserEnum`，并将结果适配为
NorthRL 的 `ParseResult`。在受支持 SGLang 版本中添加或注册
检测器，然后在此记录准确配置名称。NorthRL 不维护
另一套模型专属检测器类。

<a id="add-a-recipe-local-response-format"></a>

### 添加配方本地响应格式

当配方需要无法作为可复用 SGLang 工具格式的解析行为时，
使用 `type: custom`。配方模块接收完整原始响应，
返回最终助手消息，因此无需在 NorthRL 工厂中注册。

<a id="architecture-1"></a>

## 架构

- 远程服务持有器返回原始模型文本与 token 元数据。
- `RemoteApiInferenceServiceClient` 仅调用一个 `ResponseParser`。
- 内置响应解析器组合推理内容和工具解析器。
- 自定义响应解析器调用配方固定的 `parse_response` 钩子。
- 两种响应解析器返回相同助手消息结构。

<a id="file-structure"></a>

## 文件结构

```text
northrl/utils/
├── reasoning_parser/        # Built-in reasoning adapters
├── tool_parser/             # Built-in SGLang tool adapters
└── response_parser.py       # Unified built-in/custom response pipeline
```

<a id="reference"></a>

## 参考

此实现受 SGLang 工具解析器框架（`.vibe/tool_parser/`）启发，并针对 NorthRL 架构和需求进行了适配。

<a id="see-also"></a>

## 另请参阅

- [推理内容解析器](reasoning-parser.md) - 提取并结构化推理内容
