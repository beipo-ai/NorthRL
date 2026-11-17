[English](../en/debug_rollout.md) | 简体中文 | [文档目录](index.md)

<a id="debug-one-rollout-batch"></a>

# 调试单个 Rollout 批次

使用现有训练 YAML 运行一个 rollout 批次并保存完成的结果，
整个过程无需启动控制器或训练器：

```bash
python -m northrl.debug_rollout --config recipe/math/weaver.yaml \
  data.train.batch_size=2 \
  data.train.rollout_repeat_n=2 \
  runtime.debug.dump_dir=/tmp/math-rollout-debug
```

请在训练所用的 Python 环境中，从仓库根目录运行。先在当前 shell 中设置
数据集路径、Weaver 凭据和评判模型，也可以手动加载自己的环境脚本。
若未设置 `EXPERIMENT_PATH`，命令在解析 YAML 时会提供一个回退值。
你也可以直接在 YAML 中设置覆盖项。

命令顺序处理首批 `batch_size` 个提示词，每个尝试 `rollout_repeat_n` 次。
上述示例共尝试四次。训练数据加载器的
shuffle、过滤和 `drop_last` 设置同样生效，
异步训练配置也不例外。命令不会进入下一批次或下一个 epoch。如果
数据集小于批次大小，请减小批次或设置 `data.train.drop_last=false`。

每次调用都会打印 `runtime.debug.dump_dir` 下新建的目录
（默认 `./debug_dump`）。成功的 rollout 使用 Worker 现有的转储格式：
`rollout_traces/*.json`。对于 `single_turn`，每个文件包含：

- `task`：原始消息、标准答案、ID 和元数据。
- `prompt_str`：渲染后的提示词；`rollout_result.northrl_train.prompt_tokens`
  存储采样器返回的提示词 token。
- `response` 和 `rollout_result`：解析后的生成结果及已有结果字段。
- `reward` 和 `score`：最终奖励与评分详情。截断可能将
  最终奖励设为零，而 `score.reward_score` 保留评估得分。
- `tokens`、`loss_mask`、`old_log_probs`、token 数量、`finish_reason` 和
  `is_truncated`：Worker 的轨迹数据与完成状态。

生成或奖励计算错误会写入日志，剩余样本仍会继续尝试。
失败或跳过的样本没有完整执行记录。命令不会额外捕获
中间请求、模板源文件或评判模型调用，也不会生成单独报告。

支持的 Worker 为 `single_turn` 和 `simple`；`simple` 保留其已有的
答案标签奖励与执行记录字段。E2B、自定义 Worker、mock 和 SFT
均不受支持。Weaver 仅为采样而初始化。检查基础模型时，
不要设置 `models.actor.checkpoint`，并使用 `training.resume.mode=disable`。
使用已保存权重时，将 `runtime.debug.sampling_model_path` 设置为 Weaver 采样
检查点，并将 `models.actor.tokenizer` 设置为对应分词器。训练
检查点不会自动转换：显式检查点或已有恢复配置
必须提供采样路径，以避免使用错误权重采样。
