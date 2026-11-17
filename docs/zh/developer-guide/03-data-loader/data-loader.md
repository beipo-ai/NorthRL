[English](../../../en/developer-guide/03-data-loader/data-loader.md) | 简体中文 | [文档目录](../../index.md)

<a id="data-loader"></a>

# 数据加载器

数据加载器在训练和验证期间为 rollout 工作进程提供输入数据。NorthRL 包含内置数据加载器，以及用于自定义实现的抽象接口。

<a id="overview"></a>

## 概览

**用途**：管理数据集加载、打乱，以及向 rollout 工作进程分发数据

**主要特性**：
- 顺序分发任务，可选数据源打乱
- 有序加载时显式释放下一批输入
- 分离验证数据与训练数据
- 线程安全操作
- 可配置的数据打乱和批处理

<a id="basedataloader"></a>

## BaseDataLoader

**位置**：`northrl/data_loader/data_loader.py`

所有数据加载器的**基类**，提供核心接口。

<a id="constructor"></a>

### 构造函数

```python
def __init__(self, config: DictConfig, is_validate: bool = False)
```

**参数**：
- `config`：数据加载器配置
- `is_validate`：此数据加载器是否用于验证（影响行为和跟踪）

**属性**：
- `_is_validate`：表示验证模式的标志

<a id="abstract-methods"></a>

### 抽象方法

子类必须实现以下方法：

<a id="get_next_item---dictstr-any--none"></a>

#### `get_next_item() -> dict[str, Any] | None`

按顺序获取下一个数据项。

```python
item = dataloader.get_next_item()
if item is not None:
    process(item)
```

**返回值**：以字典形式返回一个 rollout 请求；当前无法发出请求时返回 `None`。对于有序加载器，`None` 也可能表示当前输入批次已领完，正在等待 `release_next_batch()` 开放下一批，不代表数据集一定耗尽。用 `is_finished()` 区分数据源结束和暂时等待。

<a id="is_finished---bool"></a>

#### `is_finished() -> bool`

对于 TorchDataLoader，检查有限数据源是否耗尽，以及 buffer 中的请求是否全部领完。这不表示已领取的 rollout 都已完成。循环的 RL 加载器在训练继续期间不会报告结束。

```python
if dataloader.is_finished():
    print("所有源数据请求均已领取")
```

**返回值**：数据源和任务 buffer 均已耗尽时为 `True`

<a id="can_return_item---bool"></a>

#### `can_return_item() -> bool`

对于 `SequentialDataLoader`，此方法只检查当前 buffer 是否还有未领取的数据项，
不会读取数据源，也不会填充空 buffer。

```python
has_buffered_item = dataloader.can_return_item()  # 当前 buffer 的快照
item = dataloader.get_next_item()                # 可能读取首批或下一批
```

第一次读取前，即使数据集非空，`can_return_item()` 也为 `False`。
不要要求它返回 `True` 才调用 `get_next_item()`。
Rollout worker 直接调用 `get_next_item()`，由加载器处理批次放行。

<a id="reset"></a>

#### `reset()`

将数据加载器重置为初始状态。

```python
dataloader.reset()
# Can iterate through data again
```

**用法**：在验证周期之间调用

<a id="convenience-methods"></a>

### 便捷方法

<a id="release_next_batch"></a>

#### `release_next_batch()`

对于有序加载器，当前 buffer 领完后，此方法读取并开放下一批输入。调用方负责等待当前轮次处理完成。在 RL 中，权重同步控制器会在相应的训练、权重同步或验证边界之后放行。Fully-async 加载器通过 `get_next_item()` 自动读取下一批，不需要显式放行。

```python
dataloader.release_next_batch()
```

<a id="subscribe_readycallback---callable-none--none"></a>

#### `subscribe_ready(callback) -> Callable[[], None] | None`

可选的就绪订阅接口。支持通知时返回取消订阅函数，否则返回 `None`，让 worker
继续使用轮询兜底。`SequentialDataLoader` 在 `release_next_batch()` 和 `reset()`
之后、数据锁之外发出通知。回调可能运行在发布任务的线程中，应只安排唤醒，不领取任务。
订阅方需要在订阅后和每次唤醒后重新检查 loader 状态。自定义实现必须在每个可能让
空 loader 重新返回任务的状态变化之后通知；通知只表示状态可能改变，不预留任务。

<a id="torchdataloader"></a>

## TorchDataLoader

**位置**：`northrl/data_loader/torch_data_loader.py`

基于 PyTorch DataLoader 的**具体实现**，用于文件数据集。

<a id="features"></a>

### 特性

- 将 Parquet、CSV 和 JSONL 文件加载到同一个经过过滤的数据集中
- 使用专用的确定性打乱生成器
- 基于步数的强化学习训练会自动跨 epoch 循环
- 验证保持有限遍历
- 将当前 epoch 的尾部与下一 epoch 的头部拼接，保持训练批次完整
- 支持在迭代前按提示长度过滤

<a id="single-turn-reward-data"></a>

### 单轮奖励数据

对于 `rollout.type: single_turn`，每行数据包含提示、
可选的奖励名称，以及该奖励所需的参考数据。使用 JSONL（每行一个 JSON 对象）
或具有相同列的 Parquet，以保留嵌套消息和元数据。
加载器也可读取 CSV，但不会将 CSV 单元格中的 JSON 字符串解码为
消息列表或字典。

| 字段 | 是否必需 | 含义 |
| --- | --- | --- |
| `prompt` | 是 | 聊天消息列表，例如 `[{"role": "user", "content": "What is 40 + 2?"}]`。此智能体必须使用字面顶层键 `prompt`。 |
| `label` | 所选奖励需要的参考数据 | 使用非空字符串，例如数学题使用 `"42"`，`rule_nemotron_mcqa` 使用 `"D"`。其他标签类型取决于所选奖励。`code_e2b` 要求包含完整测试用例的预处理判题请求字典。`recast` 和 `verinstruct` 不需要标签。下文介绍的旧版数学答案字段也受支持。 |
| `reward_fn` | 未配置默认值时必需 | 已注册的短名称，例如 `math_answer_only`。当前行的值会覆盖 `rollout.reward_fn`。 |
| `metadata` | 否 | 包含任务专属信息的字典，以 `task["metadata"]` 传给奖励函数。`verinstruct` 从该字典读取硬约束和指令，见下文。 |

例如，将以下五行保存为 `train.jsonl`：

```jsonl
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "label": "42", "reward_fn": "math_answer_only", "metadata": {"source": "arithmetic"}}
{"prompt": [{"role": "user", "content": "What is 6 * 7? Put the final answer in \\boxed{}."}], "label": "42", "reward_fn": "math_with_format"}
{"prompt": [{"role": "user", "content": "What is 20 + 22? Put the final answer in \\boxed{}."}], "label": "42"}
{"prompt": [{"role": "user", "content": "Give exactly three suggestions for learning Python, in Chinese."}], "reward_fn": "recast"}
{"prompt": [{"role": "user", "content": "Which option is correct? A. First B. Second. End with Answer: \\boxed{A/B}."}], "label": "B", "reward_fn": "rule_nemotron_mcqa"}
```

在现有训练配方中设置以下字段：

```yaml
data:
  train:
    files:
      - ./train.jsonl
    prompt_key: prompt

rollout:
  reward_fn: math_answer_only
  type: single_turn
```

第一行使用 `math_answer_only`，第二行使用 `math_with_format`，
第三行使用配置的默认值 `math_answer_only`。第四行使用
`recast`，无需参考答案即可判断指令遵循情况；使用此混合数据集前，
请先配置其 [LLM 裁判](../11-configuration-reference/rollout-config.md#recast-reward)。
每个生成响应都会由其所在行选择的函数独立评分。输入数据集中无需保存生成响应
或数值奖励。

奖励选择遵循以下规则：

1. 非 null 的顶层 `reward_fn` 会精确选择该名称。
2. 省略该值或设为 `null` 时使用 `rollout.reward_fn`。如果每行都指定了
   奖励，则可以省略配置中的默认值。
3. 如果两个位置均未提供名称，评估会抛出 `KeyError("reward_fn")`。
4. 未知名称或显式空字符串会引发错误，即使配置了
   默认值也是如此。CSV 空单元格遵循 pandas 的缺失值解析规则，视为未提供；
   JSONL 中的 `"reward_fn": ""` 则仍被视为显式空字符串。

用顶层 `reward_fn` 选择已注册的奖励。将 `data.train.prompt_key` 设置为其他名称，不会为此智能体
将该列重命名为 `prompt`。

奖励名称、评分规则和 LLM 要求见
[单轮奖励选择](../11-configuration-reference/rollout-config.md#single-turn-reward-selection)。
Nemotron MCQA 数据行使用 `rule_nemotron_mcqa`。

代码数据使用 `reward_fn: code_e2b`，在 `label` 中提供完整判题请求。
字段、示例、沙箱配置和评分见
[Code E2B 奖励](../11-configuration-reference/rollout-config.md#code-e2b-reward)。

对于 `verinstruct`，请将 Python 硬约束检查器放入
`metadata.hard_functions`。每个字符串定义一个
`check_following(instruction, response)` 函数，在响应满足该约束时
返回真值。例如，下面是一行 JSONL；检查器字符串内的 `\n`
表示换行：

```jsonl
{"prompt": [{"role": "user", "content": "Write a greeting using exactly three words."}], "reward_fn": "verinstruct", "metadata": {"instruction": "Write a greeting using exactly three words.", "hard_functions": ["def check_following(instruction, response):\n    return len(response.split()) == 3"]}}
```

使用 `metadata.instruction` 向检查器提供指令；
也支持 `metadata.prompt_text`。回退值为纯字符串形式的
`prompt`，如果提示是聊天消息列表，则使用空字符串。单轮 rollout worker 数据应显式设置 `metadata.instruction`。
`hard_functions` 缺失、为 null 或为空时，
将跳过硬检查，仅运行 LLM 检查。支持 Parquet 格式的检查器字符串数组。
`label` 为可选字段，不参与使用。

检查器在新建且禁用网络访问的 E2B 沙箱中运行。NorthRL
会为每个需要硬检查的数据行创建并清理沙箱；无需硬检查的数据行
不需要 E2B 凭据。只有全部硬检查通过后，
才会由 LLM 判断指令遵循情况。参见 [VerInstruct 奖励](../11-configuration-reference/rollout-config.md#verinstruct-reward)，
了解 E2B 和裁判配置、响应规范化，以及失败处理。

对于旧版数学数据，适配器首先使用非 null 的 `label`；否则
读取 `ground_truth`，若 `ground_truth` 缺失或为空，再读取 `reward_model.ground_truth`。
新数据集优先使用 `label`。显式为空的 `label` 不会
回退到旧版字段。以下数据行也被接受，并使用配置的
默认奖励：

```jsonl
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "ground_truth": "42"}
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "reward_model": {"ground_truth": "42"}}
```

此约定描述当前的单轮奖励路径。SimpleRolloutWorker
和 E2B 保留各自已有的评估路径；SFT 有独立的
[数据约定](sft-data.md)。

<a id="constructor-1"></a>

### 构造函数

```python
def __init__(self, config: DictConfig, is_validate: bool = False)
```

**训练配置**：

```yaml
data:
  train:
    files:
      - /path/to/train.jsonl
    batch_size: 32
    rollout_repeat_n: 16
    prompt_key: prompt
    filter_prompts: true
    shuffle: true
    drop_last: true
  max_prompt_length: 4096

training:
  total_steps: 3000
```

`cycle` 不是面向用户的选项。控制器会为非 SFT、
基于步数的训练推导此值。每个数据集路径只需列出一次；加载器会自动开启
新 epoch，直到达到 `training.total_steps`。

<a id="implementation-details"></a>

### 实现细节

<a id="shuffling"></a>

#### 数据打乱

- Epoch 0 使用 `experiment.seed + 0`
- Epoch 1 使用 `experiment.seed + 1`，依此类推
- 每个 epoch 对完整的过滤后数据集恰好进行一次随机排列
- `shuffle: false` 按数据集原始顺序循环

<a id="epoch-boundaries"></a>

#### Epoch 边界

训练过程中的提示流实际上是无限的，其停止条件为
`training.total_steps`。如果某个 epoch 剩余提示不足 `batch_size`，
加载器会递增 `epoch_id`、重新打乱数据，并从下一个
epoch 补齐批次。例如，`batch_size: 32` 时，包含 29 个数据项的尾部会与
新 epoch 的前 3 个数据项拼接。不会产出不完整的训练批次，
也不会丢弃 epoch 尾部的提示。

验证加载器不会循环。它们保留普通的有限遍历 `drop_last`
语义，并在单次遍历结束后通过 `is_finished()` 报告完成。

<a id="resume"></a>

#### 恢复

非 SFT 恢复继续使用 `skip_batches(global_step)`。由于每个
epoch 的顺序是 `experiment.seed + epoch_id` 的确定性函数，
跳过操作可以跨越 epoch 边界，并重建相同的数据游标。

<a id="filtering"></a>

#### 过滤

- 过滤超过 `max_prompt_length` 的样本
- 记录过滤掉的样本数量
- 确保提示能放入模型上下文窗口

<a id="mixturedataloader"></a>

## MixtureDataLoader

**位置**：`northrl/data_loader/mixture_data_loader.py`

GRPO 训练设置 `data.train.sources` 时，Controller 选择此加载器。它通过
`FileDataset` 读取各本地源，按正数 `ratio` 随机选择来源，再推进该源独立
打乱的循环序列。`experiment.seed` 控制来源抽样和源内顺序。不要求每批
包含所有来源；配置的权重表示长期抽样概率。

加载器将原始输入交给 `SequentialDataLoader`，由后者创建新组并按全局
`rollout_repeat_n` 展开请求。它从来源名称添加 `data_source`，从来源配置
添加可选的 `rollout_profile`，并保留原始 `task_id`。因此，同一组内所有
执行始终使用同一题目和执行 profile。

按 step 训练会持续循环至 `training.total_steps`。重置和跳过批次能重建
确定性抽样序列，但异步完成顺序及在途任务不在此恢复约定内。来源字段及
MiMo/SWE 五类混合配置见
[混合 RL 任务数据源](../11-configuration-reference/data-config.md#mixing-rl-task-sources)。

<a id="sftstreamingdataloader"></a>

## SFTStreamingDataLoader

**位置**：`northrl/data_loader/sft_streaming_data_loader.py`

当 `algorithm.type: sft` 时，控制器选择此仅用于训练的加载器。
应省略整个顶层 `rollout` 配置块：NorthRL 始终只创建一个
内部 `SFTRolloutWorker`，如果提供该配置块，则发出警告并忽略整个块。
有关数据行约定、配置和自定义 `render_chat` 回调，
请参见 [SFT 数据](sft-data.md)。

<a id="features-1"></a>

### 特性

- 按需流式读取 JSONL 源记录
- 在每个 epoch 确定性地重新打乱记录
- 按以下 token 预算组成请求：
  `max_sequence_length * batch_size`
- 每个样本保持为独立 `Datum`，由 Weaver 进行打包
- 丢弃 EOF 处不完整的尾部，不带入下一个 epoch
- 仅在训练成功后提交精确的源数据游标，并从
  该游标恢复
- 当前批次在 Weaver 中训练时，仅预先准备并分词
  一个后续批次

<a id="mock-dataloader"></a>

## 模拟 DataLoader

**位置**：`northrl/mock/mock_data_loader.py`

使用生成的模拟数据的**测试实现**。

<a id="features-2"></a>

### 特性

- 动态生成模拟数据
- 可配置样本数量
- 无外部文件依赖
- 适合测试和开发

<a id="configuration"></a>

### 配置

这是供测试直接实例化的内部类，生产 recipe 不通过 `data.type` 选择它。
与其他顺序加载器一样，它提供 `_fetch_batch_data()`，复用任务 buffer 处理逻辑。

<a id="usage-patterns"></a>

## 使用模式

<a id="basic-data-loading"></a>

### 基本数据加载

Worker 已有的 `_get_rollout_task()` 会直接调用当前使用的加载器。
对于本地加载器，基本操作如下：

```python
item = dataloader.get_next_item()
if item is not None:
    process(item)
elif dataloader.is_finished():
    print("数据源已结束")
else:
    print("等待下一批输入放行")
```

不要为了添加 `can_return_item()` 检查而覆盖 worker 的调度逻辑。
`get_next_item()` 会自行读取首批数据。在 `sync` 或 `batch-async` 模式下，
当前批次的请求全部领完后，它返回 `None`，直到权重同步控制器调用
`release_next_batch()`。在 `fully-async` 模式下，下次调用会自动读取下一批输入。
每次调用领取的是一个 rollout 请求，不是整个组，也不是已完成的 trajectory。

<a id="validation-cycle"></a>

### 验证周期

Controller 在 `_start_validate()` 内重置验证加载器，然后才允许 worker
领取验证任务。`_end_validate()` 等待验证完成、记录结果、将 worker 切回训练，
并通知权重同步控制器。

```python
# NorthRLController
def _run_validate(self, model_tag):
    self._start_validate(model_tag)
    self._end_validate(model_tag)
```

<a id="custom-data-loader-implementation"></a>

## 自定义数据加载器实现

<a id="minimal-implementation"></a>

### 最小实现

继承 `SequentialDataLoader`，只实现数据源部分，复用它的锁、任务 buffer、
重复展开、重置、恢复和批次放行逻辑。此示例按文件顺序读取 JSONL，
不实现 TorchDataLoader 的数据打乱或提示词过滤。内部 `cycle` 标志支持
持续提供 RL 输入，验证数据则保持有限遍历。

```python
import json
from typing import Any

from northrl.data_loader import SequentialDataLoader
from omegaconf import DictConfig


class MyDataLoader(SequentialDataLoader):
    """JSONL source with the standard task buffer and repeat handling."""

    def __init__(self, config: DictConfig, is_validate: bool = False):
        super().__init__(config, is_validate)
        self._data: list[dict[str, Any]] = []
        for filename in config.files:
            with open(filename, encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        item = json.loads(line)
                        item.setdefault("task_id", len(self._data))
                        self._data.append(item)
        self._cycle = bool(config.get("cycle", False)) and not is_validate
        self._drop_last = bool(config.get("drop_last", False))
        self._index = 0

    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        # Called under SequentialDataLoader's lock; return original inputs.
        batch = []
        while self._data and len(batch) < self._batch_size:
            if self._index == len(self._data):
                if not self._cycle:
                    break
                self._index = 0
            batch.append(self._data[self._index].copy())
            self._index += 1
        if not self._cycle and self._drop_last and len(batch) < self._batch_size:
            return []
        return batch

    def _reset_iterator(self) -> None:
        self._index = 0

    def is_finished(self) -> bool:
        with self._lock:
            source_done = not self._data or (
                not self._cycle and self._index >= len(self._data)
            )
            return source_done and self._buffer_index >= len(self._data_buffer)
```

`_fetch_batch_data()` 返回至多 `batch_size` 个原始输入。不要在此处重复输入或
分配执行身份：`_try_fetch_batch()` 会把每个输入展开成 `rollout_repeat_n` 个请求，
并设置新的 `group_id`、`0..R-1` 的 `run_id`、各自唯一的 `rollout_id`
和 `expected_rollouts=R`。因此，同一数据项再次被抽取时会形成独立的新组。
`_reset_iterator()` 只重置数据源游标；继承的 `reset()` 还会清空任务 buffer
及其索引。`is_finished()` 同时检查数据源已结束和 buffer 内的请求已领完。

<a id="register-custom-dataloader"></a>

### 接入自定义加载器

当前 Controller 的 `_get_module_class_for_role()` 为使用 `files` 的 RL
选择 `TorchDataLoader`，为使用 `sources` 的 GRPO 选择 `MixtureDataLoader`，
为 SFT 选择 `SFTStreamingDataLoader`。不存在公开的
`data.type` 选择项或 DataLoader `MODULE_REGISTRY`。接入此示例需要修改
该选择逻辑，或由嵌入式调用方直接实例化。传入
`_derive_train_dataloader_config()` 派生的内部配置，其中包括 `files` 或 `sources`、
`batch_size`、`rollout_repeat_n`、`cycle` 和 `keep_batch_order`。

保留继承的 `get_next_item()` 和 `release_next_batch()`，同一个数据源实现
便可同时支持 sync 的有序放行和 fully-async 的自动续取。

<a id="configuration-reference"></a>

## 配置参考

以下片段可合并到已有的 RL recipe：

```yaml
experiment:
  seed: 42
models:
  actor:
    tokenizer: /path/to/model
data:
  train:
    files:
      - /path/to/train.jsonl
    batch_size: 32
    rollout_repeat_n: 8
    shuffle: true
    drop_last: true
  max_prompt_length: 4096
  max_sequence_length: 8192
training:
  total_steps: 3000
  weight_sync:
    mode: sync
```

`data.train.batch_size=B` 统计原始输入数，`rollout_repeat_n=R`
将其展开成 `B × R` 个 rollout 请求。对于 GRPO/OPD，Controller 为轨迹池
派生 `groups_per_batch=B` 和 `group_size=R`。这里统计的是完整 rollout 组，
一条含多个训练序列的 trajectory 不会被计成多条 rollout。因此，最终交给训练的
token 序列或 Datum 数量可能不等于 `B × R`。

`fully-async` 下的分组 GRPO/OPD 会跨输入批次等待 B 个有效完整组。
超过一半成员失败，或有效成员没有训练目标的组不占名额，worker 会继续领取新请求；
trainer 在训练前补齐可修复组。成功零段成员仍有效，也可作为复制来源。reward 为零或组内相同
不会导致拒绝。这里固定的是每个 trainer 轮次在 pool 准入时的组数，后续
MIS/OPD 过滤及 optimizer 拆批仍可能改变实际训练数据。有限源耗尽且所有
worker 静止后，不足 B 个有效组的尾批会丢弃，不训练也不推进 `train_step`；
其他分组模式保留原有的完整组交付和有限源尾批行为。

`keep_batch_order` 和 `cycle` 是内部派生字段，不是 recipe 配置项。
Tokenizer 来自 `models.actor.tokenizer`，打乱种子来自 `experiment.seed`。
批次单位、不同模式及验证配置详见
[数据加载器配置](../11-configuration-reference/data-config.md)。

<a id="best-practices"></a>

## 最佳实践

<a id="data-organization"></a>

### 数据组织

1. **使用 Parquet 格式**，实现高效存储和加载
2. **将大型数据集拆分**为多个文件，以便并行加载
3. **包含所有必需字段**（prompt、answer、id 等）
4. 训练前**验证数据格式**

<a id="configuration-2"></a>

### 配置

1. **设置合适的 max_prompt_length**，过滤过长提示
2. **设置 `experiment.seed`**，使数据集打乱顺序可复现
3. **启用数据打乱**，改善训练效果
4. **结合设置 `data.train.batch_size` 与 `rollout_repeat_n`**，控制 rollout 任务量

<a id="performance"></a>

### 性能

1. **监控缓冲区大小**（避免过量使用内存）
2. **平衡文件数量**与文件大小
3. 对数据文件**使用 SSD 或快速存储**
4. 尽可能离线**预处理数据**

<a id="error-handling"></a>

### 错误处理

1. 训练开始前**验证文件路径**
2. 妥善**处理缺失文件**
3. **检查数据格式**的兼容性
4. 清晰地**记录加载错误**

<a id="troubleshooting"></a>

## 故障排查

<a id="issue-no-data-items-available"></a>

### 问题："No data items available"

**症状**：工作进程空闲，没有任务在处理

**解决方法**：
1. 检查 `data.train.files` 路径是否正确
2. 验证文件非空
3. 检查 `max_prompt_length` 过滤是否过于严格
4. 确保文件格式兼容

<a id="issue-data-loading-is-slow"></a>

### 问题："Data loading is slow"

**症状**：Rollout 工作进程利用率低

**解决方法**：
1. 先分析文件读取和提示词过滤耗时，再调整 rollout 并发
2. 使用更快的存储（SSD、并行文件系统）
3. 通过压缩减小文件大小
4. 将数据预加载到缓冲区

<a id="issue-out-of-memory"></a>

### 问题："Out of memory"

**症状**：系统在加载数据时崩溃

**解决方法**：
1. 减小 PyTorch DataLoader 的 `batch_size`
2. 处理更小的文件
3. 过滤或预处理过大的记录
4. 更严格地按长度过滤样本

<a id="next-steps"></a>

## 后续步骤

- 了解[轨迹池](../04-trajectory-pool/trajectory-pool.md)如何进行轨迹批处理
- 了解消费数据的 [Rollout 工作进程](../05-rollout-workers/overview.md)
- 探索[配置参考](../11-configuration-reference/data-config.md)，查看所有选项
- 查阅[最佳实践](../12-best-practices/module-development.md)，开发自定义加载器
