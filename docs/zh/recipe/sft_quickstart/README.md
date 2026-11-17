[English](../../../en/recipe/sft_quickstart/README.md) | 简体中文 | [文档目录](../../index.md)

<a id="sft-快速上手"></a>

# SFT 快速上手

从包含本 recipe 的 NorthRL 仓库开始。需要一台有 Python 3.12+ 的 Linux 或 macOS 机器、
访问 PyPI 和 Hugging Face 的网络，以及能够训练目标模型的 Weaver 账号。
当前机器运行驱动，Weaver 运行 GPU 训练；无需本地 GPU 或预装 NorthRL 环境。

<a id="setup"></a>

## 1. 创建自己的运行环境

在仓库根目录执行一次：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$(uname -s)" = Linux ]; then
  .venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -e ".[core]"
cp recipe/sft_quickstart/.env.example .env
```

Linux 安装 CPU 版 PyTorch 避免本地 CUDA 依赖，macOS 使用标准 PyPI wheel。
启动脚本自动使用 `.venv/bin/python`，无需激活环境。
先只修改 `.env` 中的两行即可使用仓库自带的数据：

```dotenv
WEAVER_BASE_URL=https://your-weaver-host
WEAVER_API_KEY=your-real-weaver-key
```

默认使用公开的 `Qwen/Qwen3.5-0.8B:262144` Weaver entry，
从 Hugging Face 的 `Qwen/Qwen3.5-0.8B` 下载 tokenizer。
模型权重保留在 Weaver 侧，账号需有该 entry 的权限。仓库中的 8 行数据用于跑通流程，不能用于提升模型质量。
`.env` 已被 Git 忽略，使用 shell 赋值语法；包含空格的值需加引号。

<a id="1-配置训练数据"></a>

## 2. 使用自带样例或放入自己的数据

默认 `.env` 已可使用：

```dotenv
SFT_DATA_PATH=recipe/sft_quickstart/train.example.jsonl
SFT_MODEL=Qwen/Qwen3.5-0.8B:262144
SFT_TOKENIZER=Qwen/Qwen3.5-0.8B
SFT_BATCH_SIZE=4
SFT_MAX_LENGTH=4096
SFT_EPOCHS=1
```

使用自己的数据时，将文件放到仓库根目录的 `data/train.jsonl`，并修改
`SFT_DATA_PATH=data/train.jsonl`。每行是一条对话：

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

只有 assistant 回复正文及其结束 token 参与 loss。也支持使用对应模型 tokenizer 生成的等长数组：

```json
{"input_ids":[1,2,3,4],"loss_mask":[0,0,1,1]}
```

`SFT_BATCH_SIZE` 是每个 optimizer step 的样本数，数据行数至少达到该值，否则调小它。
每个 epoch 末尾不足一批的数据会丢弃；`SFT_MAX_LENGTH` 截断单条样本。
`.env` 中的数据及输出路径默认相对仓库根目录，也可以使用绝对路径。数据由本地驱动读取和处理。

<a id="2-在-yaml-中配置-weaver"></a>

## 选用 Qwen3.8-Flash-Next（可选）

如果 Weaver 账号有该模型的权限，在 `.env` 中修改：

```dotenv
SFT_MODEL=Qwen/Qwen3.8-Flash-Next:262144
SFT_TOKENIZER=Qwen/Qwen3.8-Flash-Next
SFT_MAX_LENGTH=262144
SFT_LR=5.0e-6
```

tokenizer 自动从 Hugging Face 下载，不需要内部模型目录，也不需要下载模型权重。
服务端启动可能需要数分钟。
两个模型的配置都使用全参数微调（`lora.enabled: false`）。

<a id="3-启动训练"></a>

## 3. 检查并启动

```bash
bash recipe/sft_quickstart/run.sh --check
bash recipe/sft_quickstart/run.sh
```

`--check` 会对所有数据行渲染、tokenize，并统计一个 epoch 的 optimizer window 数量。
需要时下载 tokenizer，但不连接 Weaver、不启动训练。无效数据和不足一批的数据会在训练前报错。
普通启动命令重复这一检查，然后开始训练并打印运行目录。训练期间保持驱动进程运行，长任务建议放在 `tmux` 中。
启动训练时检查服务访问和模型权限，所需训练资源由 Weaver 管理。

只跑两步、不保存 checkpoint 时：

```bash
bash recipe/sft_quickstart/run.sh   training.epochs=null +training.total_steps=2 training.save.final=false
```

这会在 Weaver 上执行实际训练。默认的 8 行样例及 batch size 4 每个 epoch 恰好产生两步。

<a id="monitoring"></a>

## 4. 查看进度和结果

把 `RUN` 替换为脚本打印的实际目录：

```bash
RUN=logs/sft-THE-PRINTED-RUN-ID
tail -f "$RUN/northrl.log"
grep 'Training step .* metrics:' "$RUN/northrl.log" | tail -3
cat "$RUN/exit_code"
```

每步关注 `weaver/loss`、`weaver/grad_norm` 和 `weaver/update_successful`。
日志也会打印 Weaver model ID。启动可能需要数分钟。
命令退出后才生成 `exit_code`；值为 `0` 且出现 `NorthRL training completed` 表示驱动正常完成。
没有退出码文件或有报错时应检查原因；服务可用不代表训练已经成功。

详细指标在 `$RUN/weaver-metrics/metrics-*/` 下，生效配置在 `$RUN/train_config_resolved.yaml` 中。

### 可选：W&B loss 曲线

在 `.env` 中填入自己的 W&B 凭证，然后开启该 backend：

```dotenv
WANDB_HOST=https://api.wandb.ai
WANDB_KEY=your-wandb-api-key
```

```bash
bash recipe/sft_quickstart/run.sh 'runtime.logger.backends=[console,wandb]'
```

日志会打印 W&B run 链接。训练 loss 对应 W&B 中带编号前缀的 `weaver/loss` 指标，
横轴为 `training/train_step`。安装的 W&B SDK 同时支持旧格式及新的长 API key。

<a id="4-训练步数与-checkpoint"></a>

## Checkpoint 与再次运行

普通训练结束会保存可恢复的最终 checkpoint，其服务端路径记录在
`$RUN/ckpt/weaver_checkpoints.jsonl` 中；权重不会下载到该目录。
上面的 smoke 命令关闭了保存。周期保存可追加 `training.save.every_n_steps=100`。
训练长度只能选择 epochs 或 total_steps 其中一个。

每次启动默认生成新的 `logs/sft-...` 目录，也可以在 `.env` 中设置 `EXPERIMENT_PATH`，
每次使用新目录；脚本拒绝覆盖旧运行的日志。再次执行本快速上手会创建新模型，
显式断点恢复见[数据说明](../../developer-guide/03-data-loader/sft-data.md#prefetch-and-exact-resume)。

<a id="5-sft-配置约束"></a>

## 更多配置

`SFT_BATCH_SIZE` 也设置 token budget 的倍数，每步不超过
`SFT_MAX_LENGTH × SFT_BATCH_SIZE` 个移位后的输入 token。保持 `drop_last: true`。
只有明确需要按预定顺序回放时才设置 `SFT_SHUFFLE=false`。
SFT 使用 cross-entropy、token-mean loss 和无权重同步模式，无需填写 `rollout` 或 `training.weight_sync`。
自定义 renderer、mask 及多个数据文件等用法见 [SFT 数据说明](../../developer-guide/03-data-loader/sft-data.md)。
