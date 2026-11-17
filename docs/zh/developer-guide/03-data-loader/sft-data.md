[English](../../../en/developer-guide/03-data-loader/sft-data.md) | 简体中文 | [文档目录](../../index.md)

<a id="supervised-fine-tuning-data"></a>

# 监督微调数据

NorthRL 的 SFT 路径使用 `SFTStreamingDataLoader` 和 `SFTRolloutWorker`。
它不执行 LLM 推理：数据加载器流式读取源记录，并将
其转换为按 token 对齐的 `input_ids` 和 `loss_mask`，工作进程随后将每条
记录作为独立的 `Datum` 传给 Weaver。NorthRL 不在本地打包 SFT
样本；训练时的打包由 Weaver 负责。

<a id="configuration"></a>

## 配置

将 `algorithm.type` 设置为 `sft`。以下片段展示 SFT 专属
字段；请保留配方中其他必需的模型、优化器、后端和运行时字段。
SFT 应省略整个顶层 `rollout` 配置块。
控制器始终只创建一个内部直通式
`SFTRolloutWorker`；如果显式提供 `rollout`，NorthRL 会发出
`UserWarning`，并忽略整个配置块。

```yaml
models:
  actor:
    name: Qwen/Qwen3-8B
    tokenizer: Qwen/Qwen3-8B

data:
  train:
    files:
      - ./train.jsonl
    batch_size: 4
    shuffle: true
    drop_last: true
    # Optional; paths are resolved relative to this recipe YAML.
    # render_chat: ./custom_sft.py:render_chat
  max_sequence_length: 8192

algorithm:
  type: sft

training:
  # Set exactly one of total_steps or epochs.
  epochs: 3

validate:
  enabled: false
```

对于 SFT，`algorithm.loss.mode` 可省略，并推导为 `cross_entropy`；
选择其他 `algorithm.loss.mode`（包括 `custom`）都会导致
验证失败。

对于 SFT，应省略 `data.train.rollout_repeat_n`；此选项不可配置。如果
提供了该选项，NorthRL 会发出警告并忽略它。同时应省略
`training.weight_sync`：SFT 会推导出权重同步模式 `no-sync`。

SFT 要求 `drop_last: true`。混合 sources 还要求 `shuffle: true`；使用 `files` 时，
`shuffle: false` 每个 epoch 按文件顺序读取，可用预先排好序的数据流复现其他训练器的
样本顺序。其他值无法通过配置模式验证。SFT 不支持 `filter_prompts`；长度处理始终
采用下文介绍的截断行为。

`data.train.window_samples`（仅限 files）让每个优化器窗口在达到该样本数时结束，
使每步样本数固定，相当于全局 batch size；下文的 token 预算仍然约束每个窗口。

SFT 训练长度可通过步数或 epoch 数指定，但不能同时指定：

```yaml
training:
  total_steps: 1000  # token-budget optimizer updates
# or: epochs: 3      # complete shuffled passes over the source
```

`messages` 输入需要 `models.actor.tokenizer`。该分词器不会用于
重新分词已经包含 `input_ids` 和 `loss_mask` 的数据行。

`data.train.files` 中的相对路径和自定义渲染器的相对路径，均
相对于配方 YAML 解析，而非进程工作目录。
已存在的本地分词器路径也按同样方式解析；其他分词器
字符串保持为远程模型 ID。

<a id="supported-input-formats"></a>

## 支持的输入格式

仅支持 JSONL（`.jsonl`）文件。记录按需读取，
不会一次性将完整数据集加载到内存。数据集可以
使用以下两种数据行格式之一。

<a id="tokenized-rows"></a>

### 已分词的数据行

同时提供 `input_ids` 和 `loss_mask`：

```json
{"input_ids": [151643, 872, 198, 9707], "loss_mask": [0, 0, 0, 1]}
```

这两个数组必须：

- 包含整数；
- 长度相同且非零；
- `loss_mask` 仅使用 `0` 和 `1`；
- 通过 `1` 选中至少一个可训练 token。

`loss_mask[i]` 表示 `input_ids[i]` 是否参与训练
损失。NorthRL 始终将 `loss_mask[0]` 设为 `0`，因为第一个 token 没有
可用于预测它的前置 token。

`labels` 不能替代 `loss_mask`。数据行包含 `input_ids` 但缺少
`loss_mask` 时会立即失败，即使其中包含 `labels` 也是如此。

<a id="raw-message-rows"></a>

### 原始消息数据行

提供 OpenAI 风格的 `messages` 数组：

```json
{"messages": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4"}]}
```

省略 `data.train.render_chat` 时，NorthRL 读取 `row["messages"]`，并
将对话拆分为独立轮次，每轮均以一条
`assistant` 消息结束。对于每个完整轮次，它会渲染两个 token 序列：

```python
prompt_ids = tokenizer.apply_chat_template(
    round_messages[:-1],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=False,
)
round_ids = tokenizer.apply_chat_template(
    round_messages,
    add_generation_prompt=False,
    tokenize=True,
    return_dict=False,
)
```

`prompt_ids` 必须是 `round_ids` 的严格前缀。NorthRL 将整个
提示前缀赋值为 `0`，包括
`add_generation_prompt=True` 添加的 assistant 头部。剩余后缀赋值为 `1`，因此
assistant 补全内容及其轮次结束 token 都参与损失计算。
末尾没有 assistant 响应的轮次会使用全零
掩码渲染。聊天模板负责将 assistant 头部放入
`prompt_ids`，并将轮次结束标记放入 `round_ids`。

每轮独立渲染后追加到样本中。BOS
token 只应位于合并序列的开头，因此渲染后续轮次时产生的重复
前导 BOS 不会再次追加。

这种基于角色的构造方式不使用 Transformers 的
`return_assistant_tokens_mask`，也不要求聊天模板
包含 `{% generation %}` 块。它支持常见的消息顺序，即
非 assistant 提示后跟随一条 assistant 响应。以 `assistant` 开头
或包含连续 assistant 消息的对话，
需要使用自定义渲染器。

<a id="custom-chat-rendering-and-loss-masks"></a>

## 自定义聊天渲染与损失掩码

将 `data.train.render_chat` 设置为 Python 文件和函数。相对模块
路径将相对于配方 YAML 解析：

```yaml
data:
  train:
    render_chat: ./custom_sft.py:render_chat
```

该函数必须同步执行，并遵守以下约定：

```python
def render_chat(row, tokenizer) -> tuple[list[int], list[int]]:
    return input_ids, loss_mask
```

`row` 是从当前 JSONL 记录加载的完整字典，
而不仅是其中的 `messages` 字段。这样，自定义渲染器就可以使用
`prompt`、`answer` 或元数据等字段。默认渲染器要求
存在 `row["messages"]`，并使用上述以 assistant 结尾的轮次行为。
其他消息布局或模型专属的损失边界应使用自定义渲染器。
返回的 `loss_mask` 遵循与已分词输入相同的二值、长度和
非空规则。

<a id="length-handling"></a>

## 长度处理

NorthRL 会将 `input_ids` 和 `loss_mask` 一同截断到
`data.max_sequence_length`。然后将掩码首值设为 `0`，并
要求剩余值至少包含一个 `1`。截断会发出警告，而不会
导致失败。截断后没有可训练 token 的数据行仍然无效。

<a id="token-budget-batches-and-weaver-packing"></a>

## 按 Token 预算分批与 Weaver 打包

每个训练请求中，NorthRL 使用以下 token 预算：

```text
token_budget = data.max_sequence_length * data.train.batch_size
```

每个 Datum 占用 `len(input_ids) - 1` 个下一 token 预测位置，
并计入该预算。

准备好的数据行会不断累积，只要总量不超过预算。如果加入下一
行会超过预算，则发送当前批次，并用该行开始下一
批。每行都保持为独立的 `Datum`；NorthRL 从不拼接
`input_ids` 或 `loss_mask`。Weaver 接收完整的一组数据，并执行
实际打包。

在 epoch 结束时，由于 SFT 要求
`drop_last: true`，不完整的尾部会被丢弃，绝不会与下一 epoch 的记录合并。
每个 epoch 开始前都会确定性地重新打乱源数据。

使用 `training.epochs` 时，加载器在丢弃最后一个指定
epoch 的尾部后停止。使用 `training.total_steps` 时，会继续进入新的 epoch，直到
达到步数限制。

<a id="prefetch-and-exact-resume"></a>

## 预取与精确恢复

Weaver 训练批次 N 时，一个后台工作进程读取、渲染或分词，
并准备批次 N+1。始终只有一个预取批次，因此准备工作
不会无限超前于训练。

只有训练成功的批次才会推进已提交的源数据游标。
检查点会记录该游标（epoch 和位置），以及数据集和
分词流水线的指纹。恢复时重建相同的 epoch 顺序，
并从精确的已提交源位置继续；仅预取但未训练的
批次会重新处理，而不会被跳过。修改源文件、分词器、
渲染器引用或实现、序列长度、token 预算、打乱
种子或打乱行为，都会导致恢复验证失败，避免静默
读取不同数据。JSONL 指纹包含文件内容；本地
分词器目录会对分词器资源生成指纹，
但不会对模型权重计算哈希。

对于通过远程模型 ID 引用的分词器，指纹包含其
引用和 Transformers 版本。如果恢复时也必须拒绝同一远程 ID 背后的上游变更，
请使用不可变的本地分词器快照。

<a id="data-flow"></a>

## 数据流

```text
messages ── default/custom render_chat ──┐
                                        ├─ input_ids + loss_mask
tokenized row ───────────────────────────┘
                                                   │
                                                   ▼
                               SFTStreamingDataLoader token-budget group
                                                   │
                                                   ▼
              SFTRolloutWorker → independent Trajectories → Weaver packing/training
```

无效数据行会报告其数据集索引。常见原因包括缺少
`loss_mask`、不支持的 assistant 消息布局、轮次提示
不是完整轮次的 token 前缀、token 与掩码长度不一致、
掩码包含非二值元素，或没有可训练 token。
