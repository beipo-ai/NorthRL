[English](../../../en/developer-guide/06-trainers/remote-api-trainers.md) | 简体中文 | [文档目录](../../index.md)

<a id="remote-api-trainers"></a>

# 远程 API 训练器

[算法组件](algorithm-components.md)介绍公共配置结构，以及估计、后处理、loss
和归一化之间的职责边界。trainer 直接调用具体组件，下文介绍其数学行为与 token
对齐契约。

pool 与准备接口始终使用完整 `list[Trajectory]`。普通轨迹直接保存 tokens，仅分段结果含内部记录；token 计算与 Datum 转换在内部遍历对应序列。`batch/segment_count` 统计输入保存段数，`batch/trainable_segment_count` 统计 Datum/mask 后的有效段数；optimizer 的 `batch_size` 和 `mini_batch_size` 按 Datum 计数。详见[轨迹术语说明](../04-trajectory-pool/trajectory-semantics.md)。

远程 API 训练器使用 Weaver 后端，在外部基础设施上训练。这些训练器接收完整轨迹，并将保留的 token 序列转换为服务 Datum。

> **注意：对数概率的形状约定。**修改轨迹到
> Datum 的传递、KL/ref_model，或任何涉及 `logprobs` /
> `ref_logprobs` / `old_log_probs` 的逻辑时，请先阅读
> [`.ai-instructions/developing/logprobs-conventions.md`](../../../../.ai-instructions/developing/logprobs-conventions.md)
> 中的说明。混淆完整长度与目标对齐的区别会导致静默错误
> （不抛异常，只产生错误梯度）。

<a id="architecture"></a>

## 架构

```python
BaseTrainer
    ↓
RemoteApiTrainer (abstract)
    ↓
    ┌────────────────────────────┐
    ↓                            ↓
RemoteApiGrpoTrainer    RemoteApiCrossEntropyTrainer
```

<a id="remoteapitrainer"></a>

## RemoteApiTrainer

远程 API 训练后端的基类，位于 `northrl/trainer/remote_api_trainer.py`。

<a id="purpose"></a>

### 用途

提供远程 API 后端的核心训练循环：
1. 算法专属的轨迹准备
2. 转换为服务 datum 格式
3. 远程 API 通信
4. 指标收集与记录



| 边界 | 远程训练器约定 |
| --- | --- |
| 输入与准备 | 完整 `list[Trajectory]`，直接保存 token 字段或包含内部序列字典 |
| 分词 | worker 提供 token ID；Datum 转换负责对齐输入、目标和 mask |
| 批处理 | NorthRL 组织 optimizer batch 和请求；服务端管理设备微批次 |
| 准备与数学计算 | `_prepare_trajectories()` 准备完整轨迹，随后由 `train()` 中的 estimator 计算 advantage |
| 服务载荷 | 序列级 Datum；设备 padding 和 packing 由服务端处理 |

<a id="constructor"></a>

### 构造函数

```python
def __init__(self, config: DictConfig)
```

**关键初始化操作：**
- 提取训练超参数（lr、beta1、beta2 等）
- 创建具体 estimator、可选 postprocessor 和 backend loss adapter
- 稍后通过 `set_service_holder()` 设置服务持有器

紧凑轨迹缓冲区在目标对齐、mask 和标量 advantage 展开时保持紧凑。
数值 token 统计使用 NumPy，避免逐 token 创建 Python 对象；
对齐检查、float64 数值和 estimator 公式不变。

SDK 支持时，HTTP-binary 的 Datum 构造也会将输入 tokens 保存在自有数值缓冲区中。
访问公开的可变 token 列表时才生成列表；默认 JSON 和旧版 SDK 保持原有路径与校验。

<a id="optimizer-and-request-batching"></a>

### 优化器批次与请求拆分

`algorithm.optim.batch_size` 指定每次 RL 优化器更新的目标 Datum 数量。
可选的 `algorithm.optim.mini_batch_size` 指定批次内每个前向/反向请求
最多使用的 Datum 数量。完整批次先完成归一化，再拆分请求；
全部请求成功完成后才执行一次优化器更新。配置说明见
[训练器配置](../11-configuration-reference/trainer-config.md#optimizer-batches-and-forwardbackward-requests)。

SDK 支持 `tensor_output_format` 时，Weaver 重计算请求选择 HTTP-binary 的 CPU tensor 输出。
NorthRL 批量复制为独立的 float64 数组，并为紧凑轨迹补上开头的无上下文零值。
旧版 SDK 和 inline JSON 仍可使用；target 对齐、R2/R3 元数据和 MIS 检查不变。

`algorithm.optim.submit_ahead: 1` 为 Weaver 内置损失启用一个请求的预提交。
重算和 FB 阶段可将下一请求的准备、提交与远端执行重叠。默认 `0` 保持串行。
结果保持输入顺序；失败时等待已提交工作结束，不执行 OPT。整批归一化和
先重算、后 FB 的边界不变；暂不支持自定义损失。
首请求的准备和末请求的完成仍在关键路径上；只有一个请求时没有可重叠的工作。
整批转换和归一化仍在 FB 前完成。SFT 保留其现有独立请求流水线。

重算只重试失败的前向数据块，首次尝试失败后最多重试三次，每次重试前记录 warning。
成功块的结果会保留；全部数据块校验通过后才写回对数概率和路由重放记录。
取消异常不重试。此机制不为 FB 或优化器更新增加重试。

RL 使用 `training.weight_sync.mode: sync`、`batch-async` 或 `fully-async` 时，省略 `batch_size`
或设为 `null`，会将已交付批次转换后的全部 Datum 放入一个 optimizer batch，
不打乱、不补齐。三种模式的分组 RL 都交付 B = `data.train.batch_size` 个完整组，
每组包含 R = `data.train.rollout_repeat_n` 条完整 trajectory。不足 B 组返回
`None`，有积压时其余组继续排队；pool 不另等 worker 静止或 loader buffer 耗尽。
调度器仍控制输入放行，包括 `batch-async` 的 staleness 策略。公开 RL loader
会跨 epoch 循环填满 B 个输入；只有有限源耗尽且 worker 静止后才允许较小的
完整组尾批。所选 estimator 在 Datum 转换之前接收完整已交付 step。
`mini_batch_size` 只拆分前向/反向请求，所有请求累积
梯度后才执行一次优化器更新。被跳过的批次（包括没有可训练 Datum 的批次）仍不
更新参数或导出权重。

显式正整数保留现有 Datum 级拆批规则：RL 批次需要拆分时，NorthRL 先打乱 Datum，
再用尾批自身的记录重复补齐不足的最后一个 optimizer batch；无需拆分时保留原批次。
`no-sync` RL 必须提供正整数。mini-batch 保持所属 optimizer batch 的顺序，
不额外 padding。advantage 计算和可选后处理在拆批前基于完整 step 完成；
整数拆批仍可能将同一轨迹的序列或同一比较组的成员分散到不同 optimizer batch。
每次交付只做一次更新不改变 trajectory/segment 表达或损失权重。SFT 每次更新仍使用完整
token budget 批次，`batch_size` 可省略或设为 `null`，另见下文。

`algorithm.loss.reduction.mode: prompt_mean` 要求本批完整组共用一次 optimizer 更新
（`optim.batch_size: null`）。trainer 在拆请求前，跨同一 `(model_tag, group_id)`
的所有 rollout 和 segment 统计原始 response token，以 `1 / (G * max(T_g, 1))`
加权策略损失。MIS rejection 不缩小分母；零 advantage 和成功零 token 组仍计入
`G`；修复组按替换后的成员计算，不可修复组排除。此模式仅支持 GRPO 原生 `importance_sampling` 或 `ppo_clip`，
要求 entropy 系数为零，并关闭 KL、custom loss。详见[配置及示例](../11-configuration-reference/trainer-config.md#prompt-mean-loss-aggregation)。

<a id="trajectory-ordering"></a>

**当前排序局限：** 一批交付结果中若有多个 group 使用相同 `task_id`，排序仍依赖
随机 group UUID 来区分这些组。因此，即使 shuffle seed 相同，UUID 的变化也可能
改变 Datum 顺序及 optimizer batch 的成员。这个可复现性问题尚未修复；完整轨迹模型
没有提供稳定的采样顺序字段。

<a id="main-training-flow"></a>

### 主要训练流程

以下顺序直接体现在
[`RemoteApiTrainer.train()`](../../../../northrl/trainer/remote_api_trainer.py)
中。这是执行顺序示意，不是可替换完整实现的代码：

```text
记录完整结果 -> 修复或筛除失败组
排列 trajectory，准备各 estimator 专用输入；reference/teacher 服务调用
可选 actor logprob 重算 / mismatch 指标
若 estimator 存在：
    clear_postprocessing_cache(trajectories)
    trajectories = estimator.estimate(trajectories, inputs, metrics)
    若 postprocessor 存在：
        trajectories = postprocessor.process(trajectories, metrics)
prepared trajectory dump -> MIS -> sequence 转 Datum
划分 optimizer batch -> loss_adapter.normalize_advantages_for_sum()
拆分请求 -> 前向/反向梯度累积 -> optimizer 更新
按需导出权重 -> 返回 metrics, sampling_path, weight_version
```

`__init__()` 调用 `build_estimator()`、`build_postprocessor()`、
`build_loss_adapter()`。estimator 和可选 postprocessor 在 Datum 分批前接收
完整 step，trainer 与这些直接调用之间没有 execution plan、注册表查询或 step
分发器。SFT 没有 estimator。adapter 管理 native/custom 载荷及支持的 sum-loss
缩放，[`algorithm/reductions.py`](../../../../northrl/algorithm/reductions.py)
保存纯分母计算；所有服务调用仍由 trainer 管理。

`BaseTrainer._main_loop()` 等待 `train()`，仅记录 `metrics`，并在
增加训练步数计数器之前，将导出的路径和名称传给 WSC。过滤掉的
批次返回 `(metrics, None, None)`，不执行训练或导出权重。
每个 Weaver 持有器协程都通过 `await_coro(...)` 等待，使
Trainer 编排保留在专用事件循环中，同时让绑定事件循环的
Weaver 客户端及操作句柄在 NorthRL 的 SDK 事件循环上运行。

<a id="key-methods"></a>

### 关键方法

<a id="_prepare_trajectories-abstract"></a>

#### _prepare_trajectories()（抽象方法）

```python
@abstractmethod
async def _prepare_trajectories(
    self,
    trajectories: list[Trajectory],
    metrics: dict[str, Any]
) -> list[Trajectory]
```

在概率服务调用与 advantage 估计前执行 trainer 专属的准备。子类实现此钩子，
estimator 则由 `train()` 另行调用。

**参数：**
- `trajectories`：完整 rollout 结果，包括失败、跳过以及成功的零序列结果
- `metrics`：用于填入指标的字典

**返回值：**准备后的完整 `list[Trajectory]`；普通 token 字段和嵌套记录始终保留所属执行关系。

**用途：**准备、校验 trajectory，内部 sequence 仍归属于原 trajectory。需要的概率输入就绪后，
由具体 estimator 计算 advantage，不在此钩子中执行公式。

<a id="set_service_holder"></a>

#### set_service_holder()

```python
def set_service_holder(self, service_holder: WeaverServiceHolder) -> None
```

设置用于 API 通信的远程服务持有器。

**调用方：**初始化期间的 Controller

<a id="remoteapigrpotrainer"></a>

## RemoteApiGrpoTrainer

`RemoteApiGrpoTrainer` 为远程 API 后端准备完整比较组并转换训练数据，
位于 `northrl/trainer/remote_api_grpo_trainer.py`。基类训练流程调用
`algorithm.advantage.estimator.mode` 指定的 estimator，然后执行可选后处理、
MIS 和 Datum 转换。公共数据流及可用组件见
[组件调用链](algorithm-components.md#step-data-flow-and-alignment)。

<a id="_prepare_trajectories-implementation"></a>

### _prepare_trajectories() 实现

此钩子调用 `prepare_grpo_trajectories()`，按任务标识、组标识和 `run_id` 排列
trajectory，检查 rollout ID 是否重复。提供 `expected_rollouts` 时，还检查
预期的每个 `run_id` 是否恰好出现一次。有失败 rollout 的组会被标记为跳过；
未跳过组中的 reward 必须是有限数值。此钩子还把 trajectory 的元数据复制到
内部 sequence，并记录 rollout logprob 指标，不计算 advantage。
随后基类训练方法用本轮准备好的完整数据调用所选 estimator 的 `estimate()`。
参见[组件调用链](algorithm-components.md#step-data-flow-and-alignment)。

下文公式和标量 advantage 行为对应 `algorithm.advantage.estimator.mode: grpo`。

这个接口接收完整 `Trajectory` 对象。**GRPO baseline 按完整 trajectory 计数，每条只计一次，不能按 segment 计数。** 对一个包含 N 条不同完整 trajectory、最终 reward 为 R_1 到 R_N 的比较组：

```text
b_g     = (R_1 + ... + R_N) / N
sigma_g = std(R_1, ..., R_N)
A_i     = (R_i - b_g) / (sigma_g + epsilon)
```

标准差也使用同一组 N 个 reward。trajectory i 的所有 segment 共享 A_i；仅改变分段数量，不能改变 R_i、b_g 或 A_i。reward 只属于完整轨迹，segment 不保存 reward 副本。标准差及单 trajectory 组的具体约定见下文。

**完整 trajectory 统计：** estimator 对每个唯一 `rollout_id` 读取一次完整轨迹的 `Trajectory.reward`，并将 advantage 保存在同一完整轨迹对象上。token 计算随后使用完整轨迹的 advantage，并将它应用于内部各段。N > 1 时在这 N 个 reward 上计算样本标准差（分母 N−1），沿用主 GRPO 路径的约定；N = 1 时该 trajectory 的 advantage 为 0。

`use_run_ids` 是历史参数名。两种设置始终按 `(model_tag, group_id)` 分组，
并在提供预期槽位时校验 `run_id`；它不控制是否分组，也不会先把多个 segment
平均成一次 rollout。GRPO estimator 使用 `True` 对应的计算：

| `use_run_ids` | N > 1 时的标准差 | reward 计算精度 | 完整轨迹对象 `advantage` 值 |
| --- | --- | --- | --- |
| `True`（主路径） | 样本标准差，分母 N−1 | `torch.float32` | 标量 tensor |
| `False` | 总体标准差，分母 N | `torch.float64` | Python `float` |

两种设置都将单 trajectory 组的 advantage 设为 0，并返回完整轨迹列表。

完整组有 R 个原始成员；完整轨迹的状态非 `success`、最终 reward 无效或已标记跳过的成员
计为失败，数量为 F。F ≤ R/2 时保留全部有效成员，均匀随机、无放回抽取 F 个不同
有效成员，各复制一次补齐失败槽位；F > R/2 时整组不训练。分组训练流程在
记录原始结果之后、计算 advantage 或重算 logprob 之前执行此规则。有效零序列
成员仍保留在比较组中，也可被复制，只是不产生 Datum。
零分或相同 reward 仍有效；已评分的 E2B `task_failed` 对应完整轨迹的状态 `success`。

pool 保留原始组的 `list[Trajectory]`。fully-async 排除不可修复的组和有效成员
没有训练目标的组；其他模式仍交付这些原始结果。只有 `None` 表示批次未就绪。
trainer 过滤后没有可训练序列时跳过 optimizer 更新，但推进训练步数。
batch 调度、optimizer batch 和请求边界遵循上文规则。

原始 rollout reward/score 和输入计数在复制前记录；`batch/trajectory_count` 统计
原始 outcome 数，`batch/segment_count` 统计原始保存序列数。advantage 统计针对
修复后的训练组，使用所选 estimator 计算出的 advantage。
`batch/resampled_group_count` 统计修复组数，`batch/resampled_trajectory_count`
统计被补齐的失败槽位数。`algorithm/trajectory/advantage_token_weighted_mean`
仍按 token 加权；`batch/trainable_segment_count` 统计 Datum/mask 后的有效段数，
包括替换成员，但不含 optimizer 补齐。详见[训练 step 指标与 W&B 查看方式](../../metrics.md)。

此训练流程保留有训练 token、但 advantage 为零的样本。可选 KL 或熵项仍可能
在掩码允许的位置参与计算。采样器以
`finish_reason: length` 停止的响应保留 rollout 损失掩码，数学智能体
正常评判可见响应，而非强制将奖励设为 `0`。

advantage 估计、可选后处理和 MIS 计算之后，公共
`convert_trajectories_to_datums()` 应用 MIS 并检查最终 `loss_mask`。
全零 mask 的 sequence 不生成 Datum 或来源记录，包括被 MIS 全部拒绝的
sequence。随后只对生成的 Datum 拆分 optimizer batch；此过滤不重新计算 advantage。

若需兼容 Slime 的数学评分，请设置
`rollout.reward_fn: math_answer_only`（或在样本上设置 `reward_fn: math_answer_only`）。它检查完整
响应中的方框答案，应用与 Slime 的 `rm_type=math` 相同的 MathD/SymPy
等价判定规则，并且仅返回 `0.0` 或 `1.0`；think 标签不会
影响结果。单轮数学智能体通过推理服务聊天模板返回的 token ID
执行 `data.max_prompt_length` 限制，
而非使用渲染后字符串的字符数。

<a id="configuration-example"></a>

### 配置示例


<a id="router-replay-weaver-moe"></a>

### 路由重放（Weaver MoE）

`RemoteApiGrpoTrainer` 可传递 MoE 路由引用，无需在 NorthRL 中加载或
验证路由张量。路由形状、并行放置、持久化及重放验证
仍以 Weaver 训练器为准。

- **R3（rollout 重放）：**要求
  `sampling_logprob_source: recomputed`。采样器卸载每个序列的
  路由，并返回不透明引用 `moe_topk_indices_ref`。NorthRL 在
  对应序列记录上保留该引用，并以 `datum.metadata.router_replay.value_ref` 发送给
  两次训练器前向计算：重新计算 IS 分母的 `forward_logprob`
  调用，以及随后计算当前策略分子的
  损失前向计算。因此两次前向都使用 rollout 选择的专家路径，
  且比率两侧都使用训练器计算的对数概率。
- **R2（重计算重放）：**要求
  `sampling_logprob_source: recomputed`。产生重计算对数概率的
  同一次训练器前向计算也执行 `RECORD`。其结果显式
  包含对数概率与已记录路由集引用；在转换为 datum 之前，NorthRL
  立即将该引用和原始记录索引绑定到
  每条序列上。

对于两种模式，若已启用的重放路径缺少必需引用，当前步骤都会失败。
重放元数据在调试 datum 转储及优化器批次拆分之前附加，
因此可观测性工具与训练看到相同载荷。

```yaml
algorithm:
  type: grpo
  sampling_logprob_source: recomputed
  router_replay:
    enabled: true
    mode: R3                          # R2 or R3
```

<a id="remoteapicrossentropytrainer"></a>

## RemoteApiCrossEntropyTrainer

在 Weaver 后端实现监督学习，位于 `northrl/trainer/remote_api_cross_entropy_trainer.py`。

<a id="purpose-1"></a>

### 用途

适用于以下监督微调任务：
- 轨迹包含参考输出
- 无需计算优势值
- 直接使用交叉熵损失

<a id="_prepare_trajectories-implementation-1"></a>

### _prepare_trajectories() 实现

```python
async def _prepare_trajectories(
    self, trajectories: list[Trajectory], metrics: dict[str, Any]
) -> list[Trajectory]:
    # No reward/advantage computation; retain the sequence loss masks.
    return trajectories
```

**原因：**对于监督学习，`loss_mask` 字段已标明哪些 token 应参与损失计算。无需计算优势值或处理奖励。

<a id="configuration-example-1"></a>

### 配置示例

```yaml
algorithm:
  type: sft

backend:
  type: weaver

validate:
  enabled: false

training:
  total_steps: 100
```

SFT 也可使用 epoch 上限：

```yaml
training:
  epochs: 3
```

`algorithm.type: sft` 选择 `RemoteApiCrossEntropyTrainer`，推导出
`algorithm.loss.mode: cross_entropy`，且不创建推理
采样器。应省略整个顶层 `rollout` 块；Controller 始终
只创建一个内部 `SFTRolloutWorker`。若提供了 `rollout`，
NorthRL 发出 `UserWarning` 并忽略整个块。若未指定，NorthRL 还会推导出
`training.weight_sync.mode: no-sync`。SFT 不支持配置 `rollout_repeat_n`；
显式设置会触发警告并被忽略。
目前必须禁用依赖采样器的验证。

必须且只能提供 `training.total_steps` 与 `training.epochs` 中的一项。
`training.epochs` 目前仅支持 SFT。

<a id="sft-service-datum-format"></a>

## SFT 服务 Datum 格式

SFT 使用 `convert_trajectories_to_datums_cross_entropy()` 向 Weaver 发送
预先分词的下一 token 预测数组：

```python
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums_cross_entropy

datums = convert_trajectories_to_datums_cross_entropy(trajectories)
# Each datum has exactly this protocol:
{
    "input_tokens": [...],
    "loss_fn_inputs": {
        "target_tokens": [...],
        "loss_mask": [...],
    },
}
```

对于具有 token 对齐的 `tokens` 和 `loss_mask` 的轨迹，生成的
数组按如下方式与目标对齐：

```text
input_tokens = tokens[:-1]
target_tokens = tokens[1:]
loss_mask = trajectory.loss_mask[1:]
```

SFT Datum 中没有提示词/响应字符串、优势值或对数概率。
Weaver 直接接收 token ID；不会对 SFT 记录进行分词。

在此转换之前，`SFTStreamingDataLoader` 已按
以下每请求预算对独立条目分组：

```text
token_budget = data.max_sequence_length * data.train.batch_size
datum contribution = len(input_ids) - 1
```

训练器将整个 token 预算分组保留在一个优化器
步骤中。NorthRL 将每个条目保留为独立 Datum，Weaver 在
训练请求内部执行实际的序列打包。

<a id="key-features"></a>

## 主要功能

<a id="trajectory-based-processing"></a>

### 从完整 Trajectory 到 Datum

准备阶段始终返回完整 trajectory。普通结果直接保存 tokens；分段结果才使用内部字典列表：

```python
trajectory = Trajectory(
    tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0,
    extra_fields={"old_log_probs": [0.0, -0.2, -0.3], "advantage": 0.5},
)
datums = convert_trajectories_to_datums([trajectory])
assert trajectory.segments is None
```

KL、OPD、重算 logprob 和 MIS 也在内部遍历对应序列，不要求先构造公开的段对象。主路径不经过 `Batch`。

prepared trajectory dump 使用 `format_version: 3`：普通结果保持顶层 token 字段；多段结果保存内部 `segments` 字典。`num_trajectories` 统计完整执行，`num_sequences` 统计序列；读取器兼容旧格式。

### Weaver 服务持有器

对于 `backend.type: weaver`，Controller 创建并注入 Weaver
服务持有器：

```python
if config.backend.type == "weaver":
    service_holder = WeaverServiceHolder(config)
    trainer.set_service_holder(service_holder)
```

SFT 跳过推理采样器的创建，但优化器请求仍使用
Weaver 训练服务。

<a id="automatic-trainer-selection"></a>

### 自动选择训练器

Controller 直接根据 `config.algorithm.type` 选择训练器；
没有独立的公共训练器选择项：

```python
trainer_registry = {
    "grpo": RemoteApiGrpoTrainer,
    "opd": RemoteApiOpdTrainer,
    "sft": RemoteApiCrossEntropyTrainer,
}
trainer = trainer_registry[config.algorithm.type](config)
```

生产训练使用 `backend.type: weaver`。其中，监督
训练仅通过 `algorithm.type: sft` 选择。

<a id="sft-usage-example"></a>

## SFT 使用示例

受支持的监督训练路径使用标准顶层选择项：

```yaml
algorithm:
  type: sft

backend:
  type: weaver

training:
  total_steps: 100

validate:
  enabled: false
```

此示例仅展示选择项；省略了模型、数据、优化器和保存
设置。不存在 `trainer.type`、`trainer.train_service`、Tinker 或

对于自定义算法，可扩展 `RemoteApiTrainer`：

```python
from northrl.trainer import RemoteApiTrainer
from northrl.northrl_types import Trajectory

class MyCustomRemoteTrainer(RemoteApiTrainer):
    async def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict
    ) -> list[Trajectory]:
        # Your custom algorithm logic
        for traj in trajectories:
            # Add custom fields
            traj["custom_field"] = compute_custom_value(traj)

        return trajectories
```


<a id="comparison-grpo-vs-sft"></a>

## 训练器对比

| 功能 | RemoteApiGrpoTrainer | RemoteApiCrossEntropyTrainer |
|---------|----------------------|------------------------------|
| **训练类型** | 强化学习 | 监督学习 |
| **损失函数** | 配置指定的策略损失 | cross_entropy |
| **优势值计算** | 配置指定的 estimator | 否 |
| **比较组** | 按 model_tag 和 group_id；run_id 标识尝试槽位 | 不分组 |
| **适用场景** | 具有奖励信号的强化学习 | 监督微调 |
| **需要 reward** | 是 | 否 |

<a id="related-documentation"></a>

## 相关文档

- [不匹配修正（MIS）](mis.md) - 重计算对数概率、重要性权重与拒绝掩码
