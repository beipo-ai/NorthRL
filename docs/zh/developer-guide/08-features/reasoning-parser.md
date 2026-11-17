[English](../../../en/developer-guide/08-features/reasoning-parser.md) | 简体中文 | [文档目录](../../index.md)

<a id="reasoning-parser"></a>

# 推理内容解析器

推理内容解析器从模型响应中提取思考/推理内容，并放入独立字段，与 OpenAI o1 模型格式一致。

<a id="dependencies"></a>

## 依赖

推理内容解析器目前在底层使用 [SGLang](https://github.com/sgl-project/sglang) 的推理解析实现。使用完整功能时，请执行以下安装命令：

```bash
pip install 'NorthRL[sglang]'
```

若未安装 SGLang，NorthRL 回退到仅处理基本 `<think>...</think>` 标签的最简解析器。

<a id="overview"></a>

## 概述

模型使用思维链推理时，会输出以特殊标签包裹的思考过程。推理内容解析器：

1. 提取推理内容
2. 从主内容中移除推理标签
3. 返回干净的结构化响应
4. 支持多种模型专属格式
5. 支持流式增量解析

<a id="format"></a>

## 格式

**模型输出：**
```
<think>reasoning content here</think>actual response<tool_call>...</tool_call>
```

**解析后响应：**
```python
{
  "content": "actual response",           # Clean content only
  "reasoning_content": "reasoning...",    # Separated reasoning
  "tool_calls": [...]                     # Parsed tool calls
}
```

<a id="configuration"></a>

## 配置

推理内容解析器使用检测器类型处理不同模型格式，
是内置响应解析器的一个阶段：

```yaml
rollout:
  parsers:
    type: builtin
    reasoning: think_tag  # Default
```

设置 `reasoning: null` 可禁用此阶段。设置 `type: custom` 时，
配方本地的 `parse_response` 回调替代推理内容解析和工具解析。

<a id="supported-detector-types"></a>

### 支持的检测器类型

| 检测器 | 格式 | 适用模型 | 流式 |
|----------|--------|---------|-----------|
| `think_tag` | `<think>...</think>` | 默认/通用 |  |
| `qwen3` | `<think>...</think>` | Qwen3、MiniMax |  |
| `qwen3-thinking` | `(<think>)?...</think>` | 强制推理的 Qwen3 |  |
| `deepseek_r1` | `(<think>)?...</think>` | DeepSeek R1（强制推理） |  |
| `deepseek_v3` | `<think>...</think>` | DeepSeek V3 |  |
| `kimi` | `◁think▷...◁/think▷` | Kimi Thinking 模型 |  |
| `kimi_k2` | `<think>...</think>` | Kimi K2 |  |
| `step3` | `<think>...</think>` | Step3 模型 |  |
| `step3p5` | `<think>...</think>` | Step3.5 模型 |  |
| `nano_v3` | `<think>...</think>` | NanoV3 模型 |  |
| `minimax` | `<think>...</think>` | MiniMax |  |
| `minimax_append_think` | 特殊格式 | MiniMax（前置标签） |  |
| `glm45` | `<think>...</think>` | GLM-4.5 |  |
| `gpt-oss` | Harmony 格式 | GPT-OSS/T4 模型 |  |
| `interns1` | `<think>...</think>` | InternLM S1 |  |

**注意：**所有检测器类型（旧版 `think_tag` 除外）均由 SGLang 提供并支持流式解析。

**示例：**
```yaml
# For Qwen3 models
rollout:
  parsers:
    type: builtin
    reasoning: qwen3

# For Qwen3 models whose opening <think> tag is part of the prompt
rollout:
  parsers:
    type: builtin
    reasoning: qwen3-thinking

# For DeepSeek R1
rollout:
  parsers:
    type: builtin
    reasoning: deepseek_r1

# For Kimi model
rollout:
  parsers:
    type: builtin
    reasoning: kimi
```

<a id="usage"></a>

## 使用

<a id="automatic-integration"></a>

### 自动集成

无需修改代码。推理内容解析器已自动集成：

1. **服务持有器**返回原始模型响应
2. **推理客户端**运行所配置的内置响应流水线
3. 若存在推理内容，**响应**包含 `reasoning_content` 字段

<a id="basic-example"></a>

### 基础示例

**输入：**
```python
messages = [{"role": "user", "content": "What's 2+2?"}]
response = client.generate(messages)
```

**输出：**
```python
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 4.",
      "reasoning_content": "Let me calculate: 2 + 2 = 4"
    }
  }]
}
```

对于兼容 Anthropic 的远程 rollout，适配器将解析后的 Weaver
`reasoning_content` 映射为提供商原生 `thinking` 块，并在
下次请求中映射回来。该块使用未签名的本地占位签名，
适配器将此块转换回 Weaver 输入时忽略该签名。

<a id="programmatic-usage"></a>

### 编程使用

也可直接使用推理内容解析器：

```python
from northrl.utils.reasoning_parser import create_reasoning_parser

# Create parser with specific detector
parser = create_reasoning_parser("qwen3")

# Parse complete text
result = parser.parse(text)
print(result.reasoning_content)  # Extracted reasoning
print(result.cleaned_content)    # Clean response
print(result.is_valid)           # Parsing success

# Or get as tuple
reasoning, content = parser.parse_non_stream(text)
```

<a id="streaming-usage"></a>

### 流式使用

流式应用应使用启用了流式功能的检测器型解析器：

```python
from northrl.utils.reasoning_parser import create_reasoning_parser

# Create streaming parser
parser = create_reasoning_parser(
    parser_type="qwen3",
    stream_reasoning=True
)

# Process streaming chunks
for chunk in stream:
    reasoning, content = parser.parse_stream_chunk(chunk)
    if reasoning:
        print(f"Thinking: {reasoning}")
    if content:
        print(f"Response: {content}")
```

**注意：**`think_tag` 检测器（默认）不支持流式解析。请使用 `qwen3` 或其他检测器类型获得流式支持。

<a id="content-cleanup"></a>

## 内容清理

解析器按以下顺序处理内容：

1. **服务持有器**返回原始模型响应
2. **推理客户端**对原始响应应用推理内容解析器
3. **工具解析器**从已清理推理内容的文本中解析工具调用
4. 返回字段分离的干净内容

这确保 `content` 字段仅包含实际响应，不包含推理或工具调用标记。

<a id="architecture"></a>

## 架构

<a id="component-flow"></a>

### 组件流程

```
┌─────────────────────────────────────────────────────┐
│           Service Holder (Tinker/Weaver)            │
│  - Returns the raw model response                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          Inference Service Client                    │
│  - Creates the response parser from rollout config  │
│  - Parses reasoning, then tool calls                 │
│  - Adds reasoning_content to message                │
└─────────────────────────────────────────────────────┘
```

<a id="parser-hierarchy"></a>

### 解析器层级

```
BaseReasoningFormatDetector (streaming support)
├── DeepSeekR1Detector
├── Qwen3Detector
├── KimiDetector
├── Step3Detector
├── NanoV3Detector
└── MiniMaxAppendThinkDetector

BaseReasoningParser (legacy)
└── ThinkTagReasoningParser

ReasoningParser (main wrapper)
└── Wraps detector instances
```

<a id="advanced-features"></a>

## 高级功能

<a id="force-reasoning-mode"></a>

### 强制推理模式

某些模型始终输出推理内容。对于生成
提示词已包含起始 `<think>` 标签的 Qwen3 模型，在 YAML 中使用强制推理
检测器：

```yaml
rollout:
  parsers:
    type: builtin
    reasoning: qwen3-thinking
```

编程使用时，直接传入 `force_reasoning` 参数：

```python
parser = create_reasoning_parser(
    parser_type="qwen3",
    force_reasoning=True  # Assume reasoning until </think>
)
```

<a id="custom-detector-implementation"></a>

### 自定义检测器实现

添加新推理格式的支持：

1. 创建继承 `BaseReasoningFormatDetector` 的检测器类
2. 实现 `detect_and_parse()` 和 `parse_streaming_increment()`
3. 添加到 `create_reasoning_parser()` 中的检测器映射

```python
from northrl.utils.reasoning_parser import BaseReasoningFormatDetector

class CustomDetector(BaseReasoningFormatDetector):
    def __init__(self, stream_reasoning=True, force_reasoning=False, **kwargs):
        super().__init__(
            think_start_token="<custom>",
            think_end_token="</custom>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            **kwargs
        )
```

<a id="backward-compatibility"></a>

## 向后兼容性

实现保持完全向后兼容：

- 旧版 `ThinkTagReasoningParser` 仍可用
- 默认 `think_tag` 解析器不变
- 现有配置继续有效
- API 无破坏性变更

<a id="choosing-the-right-detector"></a>

## 选择合适的检测器

| 场景 | 推荐检测器 | 原因 |
|----------|---------------------|---------|
| 通用/未知 | `think_tag` | 稳定、简单，适用于多数情况 |
| Qwen3 系列 | `qwen3` | 为 Qwen3 模型优化，支持流式解析 |
| 提示词含起始标签的 Qwen3 | `qwen3-thinking` | 强制推理直到 `</think>` |
| DeepSeek R1 | `deepseek_r1` | 处理 R1 的强制推理模式 |
| Kimi 模型 | `kimi` | 支持 Kimi 的特殊标记 |
| 流式应用 | `qwen3` 或其他类型 | 除 `think_tag` 外均支持流式解析 |
| 稳定生产环境 | `think_tag` | 经过实践检验，行为稳定 |

<a id="troubleshooting"></a>

## 故障排查

<a id="reasoning-not-extracted"></a>

### 未提取推理内容

**问题：**缺少 `reasoning_content` 字段

**解决方案：**
1. 检查模型是否输出 `</think>` 标签
2. 确认配置中的检测器类型正确
3. 提示词提供 `<think>` 且模型输出从推理内部开始时，使用 `qwen3-thinking`
4. 启用推理客户端调试日志以检查解析过程

<a id="incorrect-parsing"></a>

### 解析不正确

**问题：**推理内容包含不需要的文本

**解决方案：**
1. 尝试模型专属检测器，而非 `think_tag`
2. 检查模型格式是否符合检测器预期
3. 启用调试日志查看解析步骤

<a id="streaming-not-working"></a>

### 流式解析无效

**问题：**流式解析器不返回增量结果

**解决方案：**确保未使用不支持流式解析的 `think_tag` 检测器。使用 `qwen3` 或其他检测器类型。

<a id="performance-considerations"></a>

## 性能考虑

- **旧版解析器（`think_tag`）**：基于正则表达式，处理较短响应速度快
- **检测器型解析器**：使用字符串操作，为流式解析优化
- **内存**：开销极小，检测器维护小型缓冲区
- **延迟**：流式检测器每个数据块增加的耗时小于 1 毫秒

<a id="see-also"></a>

## 另请参阅

- [工具解析器](tool-parser.md) - 工具调用解析框架（架构类似）
