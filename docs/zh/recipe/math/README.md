[English](../../../en/recipe/math/README.md) | 简体中文 | [文档目录](../../index.md)

<a id="single-turn-math-task"></a>

# 单轮数学任务

本目录包含使用 Weaver 后端训练单轮数学问题模型的配方。

<a id="task-description"></a>

## 任务说明

单轮数学任务要求模型接收一道数学题，并在单次回复中给出正确解答。奖励通过规则评估计算，将生成答案与标准答案比较。

<a id="configuration-files"></a>

## 配置文件

- **`weaver.yaml`** - Weaver 训练，LoRA 微调（rank 32）
- **`weaver_full_ft.yaml`** - Weaver 训练，全参数微调（不使用 LoRA，学习率更低）

两个文件都采用当前的九块配置结构（`experiment` / `models` / `data` / `validate` / `algorithm` / `rollout` / `training` / `backend` / `runtime`），且各自包含完整配置。

它们使用 `rollout.type: single_turn` 和数学任务的 `reward_fn`。

<a id="configuration-requirements"></a>

## 配置要求

这些配方通过 `${oc.env:...}` 读取环境变量。直接执行时，请在运行控制器的环境中设置：

```bash
export NORTHRL_DATA_PATH=/path/to/your/data        # training/test parquet root
export EXPERIMENT_PATH=/path/to/output           # checkpoint output dir
export WEAVER_API_KEY=sk-your-weaver-api-key      # or set in the env, not the yaml
export WEAVER_BASE_URL=https://your-weaver-host   # weaver service endpoint
```

**可选：**
- `export FEISHU_WEBHOOK_URL=https://your-webhook-url`（用于通知）

在同一本地环境中安装运行依赖并准备数据访问。如果将环境变量保存在私有 shell 脚本中，请在启动 Python 前手动 source。

<a id="data-format"></a>

## 数据格式

提供的配方指向 Parquet 文件。修改 `data.train.files` 也可使用 JSONL；
两种格式使用相同字段：

- `prompt`：包含 `role` 和 `content` 的聊天消息列表，不能是纯字符串。
- `label`：参考答案，建议使用字符串。也支持旧字段 `ground_truth` 和
  `reward_model.ground_truth`。
- `reward_fn`：可选的行级奖励简称，会覆盖配方中的
  `rollout.reward_fn`；这些配方的默认值为 `math_with_format`。
- `metadata`：可选任务信息，会传递给奖励函数。

例如，以下 JSONL 行选择对 boxed 答案进行二元评分：

```json
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "label": "42", "reward_fn": "math_answer_only"}
```

若希望所有省略 `reward_fn` 或将其设为 `null` 的行都使用该奖励，请设置：

```yaml
rollout:
  reward_fn: math_answer_only
```

同一文件的不同数据行可选择不同的已注册奖励。未知
名称或显式空字符串会报错。详见
[数据格式与混合奖励示例](../../developer-guide/03-data-loader/data-loader.md#single-turn-reward-data)
以及[支持的奖励名称](../../developer-guide/11-configuration-reference/rollout-config.md#single-turn-reward-selection)。
bp-rl-judge 的 `math` 奖励行为不同，未在这里注册。

提供的配方需要以下文件：
```
${NORTHRL_DATA_PATH}/math/
├── level1_train.parquet
├── level2_train.parquet
├── level3_train.parquet
├── level4_train.parquet
├── level5_train.parquet
├── level1_test.parquet
├── level2_test.parquet
├── level3_test.parquet
├── level4_test.parquet
└── level5_test.parquet
```

<a id="usage"></a>

## 用法

按照[本地快速入门](../../developer-guide/01-getting-started/quick-start.md#run-locally)安装运行环境并准备服务访问。设置上述变量后，从仓库根目录运行：

```bash
export TRAIN_CONFIG="$(pwd)/recipe/math/weaver.yaml"
# For full fine-tuning, select recipe/math/weaver_full_ft.yaml instead.
mkdir -p "$EXPERIMENT_PATH"
python -m northrl.main \
  --config-path="$(dirname "$TRAIN_CONFIG")" \
  --config-name="$(basename "${TRAIN_CONFIG%.*}")" \
  'runtime.logger.backends=[console]'
```

<a id="key-configuration-parameters"></a>

## 主要配置参数

- `data.train.batch_size`：每批提示词数（默认：32）
- `data.train.rollout_repeat_n`：每个提示词的回复数（默认：8）
- `training.total_steps`：总训练步数（默认：100）
- `rollout.num_workers`：并行 Rollout Worker 数（默认：256）
- `rollout.sampling.temperature`：采样温度（默认：1.0）
- `algorithm.optim.lr`：学习率（weaver：1e-4 / full_ft：2e-6）
- `models.actor.lora.enabled`：LoRA 或全参数微调（weaver：true / full_ft：false）
- `data.max_prompt_length` / `data.max_sequence_length`：4096 / 12288（推导出的生成预算为 8192）
