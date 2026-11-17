[English](../../../en/developer-guide/11-configuration-reference/data-config.md) | 简体中文 | [文档目录](../../index.md)

<a id="data-loader-configuration"></a>

# 数据加载器配置

公开的训练输入配置位于 `data.train`。Controller 为使用 `files` 的 rollout
训练选择 `TorchDataLoader`，为使用 `sources` 的 GRPO 训练选择
`MixtureDataLoader`，为 `algorithm.type: sft` 选择 `SFTStreamingDataLoader`；
不存在公开的 `data.type` 选择项。

## Rollout 训练

以下配置片段可合并到已有的 RL recipe：

```yaml
experiment:
  seed: 42
models:
  actor:
    tokenizer: "${oc.env:NORTHRL_MODEL_PATH}/model"
data:
  train:
    files:
      - "${oc.env:NORTHRL_DATA_PATH}/train1.parquet"
      - "${oc.env:NORTHRL_DATA_PATH}/train2.parquet"
    batch_size: 32
    rollout_repeat_n: 8
    group_order: fifo
    prompt_key: prompt
    filter_prompts: false
    shuffle: true
    drop_last: true
  max_prompt_length: 4096
  max_sequence_length: 12288
training:
  total_steps: 3000
  weight_sync:
    mode: sync
```

<a id="files"></a>

### files

`data.train.files` 是非空文件路径列表，未设置 `sources` 时必填。
`files` 与 `sources` 必须且只能设置一个。RL 加载器支持 Parquet、CSV
和 JSONL，并将所列文件拼接成数据集。每条路径只需列一次；按 step 训练的
RL 会自动循环数据集，直到满足训练停止条件。环境变量使用 `${oc.env:NAME}`，
分片文件应逐个列出路径。

<a id="mixing-rl-task-sources"></a>

### 混合 RL 任务数据源

用 `data.train.sources` 替换 `files`，按权重随机抽取任务类型：

```yaml
data:
  train:
    sources:
      - type: local
        name: mimo_code
        files: [/data/mimo/code.jsonl]
        ratio: 1
        rollout_profile: mimo_code
      - type: local
        name: mimo_arvo
        files: [/data/mimo/arvo.jsonl]
        ratio: 1
        rollout_profile: mimo_arvo
      - type: local
        name: mimo_terminal
        files: [/data/mimo/terminal.jsonl]
        ratio: 1
        rollout_profile: mimo_terminal
      - type: local
        name: mimo_general
        files: [/data/mimo/general.jsonl]
        ratio: 1
        rollout_profile: mimo_general
      - type: local
        name: swe
        files: [/data/swe/train.jsonl]
        ratio: 1
        rollout_profile: swe
    batch_size: 8
    rollout_repeat_n: 8
    shuffle: true
```

RL 数据源混合要求 `algorithm.type: grpo`，且只支持 `type: local`。
每个源必须配置唯一的非空 `name`、非空
`files` 和正数 `ratio`。每次抽取原始题目，先以 `ratio / sum(ratios)` 的
概率选择数据源，再从该源独立打乱的循环序列中取下一题。上例每类概率均为
20%；若要 SWE 占 50%、MiMo 整体占 50%，将 SWE 的 `ratio` 设为 `4`，
四类 MiMo 保持 `1`。比例表示长期抽样概率，不要求每批精确配额，
也不由文件中的题目数量决定。

`batch_size` 仍统计原始输入数，所有来源共用一个 `rollout_repeat_n`。每次
抽中的题目独立展开为完整 rollout 组，同一组不会混入其他题目。各数据源
独立循环，直到达到 `training.total_steps`。`total_samples` 和
`data.train.seed` 仅用于 SFT；RL 使用 `experiment.seed`。设为
`shuffle: false` 后，来源仍按权重随机选择，但每个源内部按文件顺序循环。

加载器为每条运行时任务添加 `data_source: <source.name>`，保留原始
`task_id`。若源配置了 `rollout_profile`，加载器自动添加此字段，因此单类
JSONL 文件无需修改。若数据行已有不同的 `rollout_profile`，加载会报错。
若同一文件包含多种执行配置，省略源级字段，在每行中设置 `rollout_profile`。
Worker 路由配置见 [E2B 执行 profiles](rollout-config.md#execution-profiles)。

恢复时保持种子、来源顺序、文件及权重不变。同步训练通过
`skip_batches(global_step)` 重建抽样序列；异步完成顺序及在途任务不会作为
sampler 状态保存到检查点。检查可复现性时，先使用
`training.weight_sync.mode: sync`。

<a id="batch_size"></a>

### batch_size

`data.train.batch_size` 是必填的正整数。对于 RL，用 `B` 表示该值，
用 `R` 表示 `data.train.rollout_repeat_n`：

| 阶段 | 单位与数量 |
| --- | --- |
| 加载器读取的源批次 | 重复展开前的 `B` 个原始输入 |
| 加载器任务 buffer | `B × R` 个独立 rollout 请求 |
| GRPO/OPD 采样组 | `B` 个组，每组等待 `R` 个完整 rollout 结果 |
| 训练 token 序列 | 普通 trajectory 对应一条序列；多序列 trajectory 对应其嵌套序列，总数可能不同于 `B × R` |

Controller 由这些公开字段派生轨迹池内部的 `groups_per_batch=B` 和
`group_size=R`。分组 RL 在 `sync`、`batch-async` 和 `fully-async` 下统一
调用 `take_groups(num_groups=B)`：返回 B 个完整组（B × R 条完整轨迹），
不足则返回 `None`。常规取组不另等 loader buffer 耗尽或 worker 静止，
即使有积压，单次也最多取 B 组。

对于 `fully-async` 下的分组 GRPO/OPD，Controller 还会派生内部配置
`require_trainable_groups=true`。只有可修复的完整组才计入 B：失败成员不超过一半，
且有效成员中至少有一条长度大于 1、`loss_mask[1:]` 中存在有效训练目标的序列。
trainer 随后按轨迹池文档中的规则补齐失败槽位。成功零段成员仍有效，也可被复制。
reward 为零或组内 reward 相同不会导致拒绝。被拒绝的组单独汇总，
不占就绪组和 `max_ready_groups` 背压名额；pool 继续收集直到凑齐 B 个
有效组，拒绝组本身不会推进 `train_step`。配置 `rollout.max_ready_groups`
后，fully-async 的实际水位至少为 B，避免背压阻止凑齐一批。丢弃指标及日志
详见[轨迹池](../04-trajectory-pool/trajectory-pool.md#groupedtrajectorystore)。

调度器仍控制输入放行：`sync` 和 `batch-async` 通常每次放行一个包含 B 组
的输入批次；`fully-async` 可跨输入批次取组。公开 RL loader 会跨 epoch
循环填满 B 个输入。对于底层有限源，pool 确认 `is_finished()` 明确返回
`True` 且 worker 全部静止后，fully-async 分组 GRPO/OPD 会丢弃不足 B 个
有效组的尾批，不训练也不推进 `train_step`；其他分组模式仍交付较小的
完整组尾批。不足 B 组时，当前 buffer 耗尽且 worker 静止后仍有不完整组，
也会报错。

Simple/SFT 则等当前 loader 整轮已领完且 worker 静止后，交付该轮全部
trajectory；数量可随 SFT token budget 变化。

其他分组模式仍交付失败、跳过或成功但没有训练序列的组，可能消费训练
轮次却不执行 optimizer 更新。fully-async 保证的是每个 trainer 轮次在 pool
准入时有 B 个有效组；后续 MIS 拒绝和 OPD 依赖 advantage 的过滤仍可能减少
训练数据，序列或 Datum 数量也不一定等于 B × R。

`algorithm.optim.batch_size` 是后续按训练 Datum 数划分的 optimizer 边界：
正整数表示拆成多次更新；在这三种 RL 模式下，省略或设为 `null` 表示
已交付批次的全部 Datum 一次更新。`algorithm.optim.mini_batch_size`
限制每次前向/反向请求的 Datum 数。二者都不定义加载器输入数或 GRPO 组大小。
optimizer 拆批可能把同一组的 Datum 分到多次更新，因此 B 不是每次 optimizer
更新的组数保证。详见 [Remote API 训练器](../06-trainers/remote-api-trainers.md)。

<a id="rollout_repeat_n"></a>

### rollout_repeat_n

`data.train.rollout_repeat_n` 在非 SFT 训练中是必填正整数，表示每次抽取
一个输入后独立执行 rollout 的次数，不表示对话轮数或嵌套 token 序列数。

```yaml
data:
  train:
    rollout_repeat_n: 8  # 每组独立执行八次 rollout
```

每次抽取产生新的 `group_id`；组内各次执行具有 `0..R-1` 的 `run_id`、
各自唯一的 `rollout_id` 和 `expected_rollouts=R`。
`task_id` 标识数据源中的任务，因此再次抽取同一任务会生成新组，不会复用
先前执行的身份。

SFT 应省略此字段；显式设置会发出警告并被忽略。
验证独立使用 `validate.data.validation_repeat_n`。

<a id="group_order"></a>

### group_order

`data.train.group_order` 控制轨迹池优先取出哪些完整 GRPO/OPD 组，可选
`fifo`（默认）和 `lifo`。Controller 将其传入内部 pool 的 `group_order`；
recipe 使用公开字段：

```yaml
data:
  train:
    group_order: lifo  # 优先最近完成的组
```

排序依据是最后一个所需 rollout 到齐、整个组进入 ready 队列的顺序。
例如 A、B、C 依次完成，下批需要两个组时，`fifo` 取 A、B，`lifo` 取 C、B。
两种方式都不拆组，也不改变组内 `run_id` 顺序。

就绪组超过 B 时，这个选项影响批次成员，最常见于 `fully-async` 积压时。
所有模式每次最多取 B 组；`sync` 和 `batch-async` 通常每次放行一个 B 组输入批次。
GRPO trainer 随后会重新排序交付的 trajectories，因此 pool 顺序不保证
optimizer 执行顺序。
“最近完成”不代表最近发起，也不代表使用最新模型权重采样。持续积压时，
`lifo` 可能不断消费新完成组，让老组长期等待。此设置不影响 SFT 使用的非分组存储。

<a id="keep_batch_order"></a>

### keep_batch_order（内部字段）

此字段由 Controller 派生，不是公开的 `data.train` 配置项：

| 模式 | 内部值 | 何时开放下一批输入 |
| --- | --- | --- |
| RL `sync` / `batch-async` | `true` | 当前轮次完成后，通过 `release_next_batch()` 放行 |
| RL `fully-async` / `no-sync` | `false` | `get_next_item()` 发现当前任务 buffer 已领完时自动读取 |
| SFT | `true` | 通过 SFT 批次提交与放行流程 |

首批数据由 `get_next_item()` 自动读取，不需要显式放行。
Worker 每次领取的都是一个请求；有序加载不表示某个 worker 一次领取整个批次。
关闭此门控也不会关闭轨迹池对 GRPO/OPD 完整组的要求。

<a id="prompt_key"></a>
<a id="filter_prompts"></a>

### prompt_key 与 filter_prompts

`data.train.prompt_key` 默认是 `prompt`，选择加载器处理提示词时使用的列，
不会替 worker 重命名数据字段。`data.train.filter_prompts` 默认是 `false`。
在 RL 中启用后，会在遍历前排除超过 `data.max_prompt_length` 的提示词。

SFT 必须保持 `filter_prompts=false`；过长样本按
`data.max_sequence_length` 截断。

## 长度限制与 Tokenizer

- `data.max_prompt_length`：RL 必填的正整数提示词 token 上限；SFT 省略。
- `data.max_sequence_length`：必填的正整数总序列长度上限，RL 中必须大于 `max_prompt_length`。
- `models.actor.tokenizer`：tokenizer 名称或路径，不是 `data.tokenizer_path`。
- `rollout.sampling.max_tokens`：可选的生成长度上限。E2B 未配置或为 `null` 时保留 agent 预算，显式值覆盖；其他推理路径仍由 `max_sequence_length - max_prompt_length` 派生默认值。不存在 `data.max_response_length` 字段。

<a id="shuffle"></a>
<a id="seed"></a>

### shuffle 与 seed

`data.train.shuffle` 默认为 `true`。TorchDataLoader 使用独立的打乱生成器，
种子为 `experiment.seed + epoch_id`；`experiment.seed` 默认是 `42`。
设为 `shuffle: false` 后，按源数据顺序循环。这些设置控制数据集顺序，
不保证不同运行中的异步完成顺序或训练顺序相同。

<a id="drop_last"></a>

### drop_last

`data.train.drop_last` 默认为 `true`。有限的 TorchDataLoader 会据此丢弃
末尾不足 `batch_size` 的源批次。按 step 训练的 RL 使用内部派生的 `cycle`
标志，将 epoch 尾部与下一 epoch 的输入拼成完整批次，不丢弃尾部输入。
不要在公开 recipe 中添加 `cycle`。

生产验证加载器有限遍历，并在内部固定使用 `shuffle=false`、
`drop_last=false` 和 `filter_prompts=false`。

## SFT 数据

Controller 自动选择 `SFTStreamingDataLoader`：

```yaml
models:
  actor:
    name: Qwen/Qwen3-8B
    tokenizer: Qwen/Qwen3-8B
data:
  train:
    files:
      - ./train.jsonl
      - ./more-train.jsonl
    batch_size: 8
    shuffle: true
    drop_last: true
  max_sequence_length: 4096
algorithm:
  type: sft
validate:
  enabled: false
```

SFT 的 `batch_size` 是 token 预算
`data.max_sequence_length × data.train.batch_size` 中的乘数，不是固定的
数据行、rollout 组或 Datum 数量。加载器流式读取 JSONL，要求
`shuffle=true` 和 `drop_last=true`，并丢弃不足预算的 epoch 尾批，不跨到
下一个 epoch 补齐。Weaver 负责打包独立的 Datum。

检查点保存已提交的数据源游标。训练批次 N 时，加载器提前准备并分词一批数据。
可选的 `data.train.render_chat` 使用 `path/to/file.py:function` 选择渲染函数。

省略顶层 `rollout` 块。Controller 创建一个内部 `SFTRolloutWorker`；
显式提供该块会发出警告并被忽略。SFT 的 `algorithm.loss.mode` 默认派生为
`cross_entropy`。数据行格式和恢复行为详见 [SFT 数据](../03-data-loader/sft-data.md)。

<a id="validation-data-configuration"></a>

## 验证数据

验证在 `validate.data.datasets` 下配置命名数据集，单独设置重复次数：

```yaml
validate:
  enabled: true
  schedule:
    before_train: true
    every_n_steps: 100
  data:
    datasets:
      held_out:
        files:
          - "${oc.env:NORTHRL_DATA_PATH}/test.parquet"
        batch_size: 16
    validation_repeat_n: 4
  aggregation: pass_at_k
```

每个数据集的 `batch_size` 统计验证重复展开前的原始输入数。
`validation_repeat_n` 默认为 `1`；此例为每个输入执行四次，用于分组评估，
不使用训练的重复次数。验证复用已配置的 tokenizer 和长度限制。
SFT 设置 `validate.enabled=false`，因为当前验证路径需要推理采样器。

<a id="troubleshooting"></a>

## 故障排查

- **启动后没有任务**：直接调用 `get_next_item()`；`can_return_item()` 只检查现有 buffer，初始值是 `False`。
- **有序批次之间返回 `None`**：检查 `is_finished()`，等待放行不等于数据源结束。
- **rollout 数量与预期不同**：区分 `B` 个原始输入、`B × R` 个请求和完整结果。
- **Datum 数量与预期不同**：检查嵌套序列、零序列结果和跳过的组；`drop_last` 不保证训练序列数固定。
- **文件不存在**：检查 `data.train.files`、环境变量以及逐个列出的分片路径。

<a id="related-documentation"></a>

## 相关文档

- [数据加载器](../03-data-loader/data-loader.md) — 实现与自定义数据源
- [轨迹池](../04-trajectory-pool/trajectory-pool.md) — 完整组收集
- [配置设置](../01-getting-started/configuration-setup.md) — 环境变量
