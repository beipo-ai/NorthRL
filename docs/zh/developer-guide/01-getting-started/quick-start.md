[English](../../../en/developer-guide/01-getting-started/quick-start.md) | 简体中文 | [文档目录](../../index.md)

<a id="quick-start-guide"></a>

# 快速入门指南

在本地 CPU 机器上运行 NorthRL 驱动，模型训练和采样由远端 Weaver 完成。本页使用[数学配方](../../../../recipe/math/weaver.yaml)；若想先用自带的小数据集和现成的本地启动脚本跑通流程，请看 [SFT 快速上手](../../recipe/sft_quickstart/README.md)。

<a id="prerequisites"></a>

## 前提条件

- Python 3.12+、可写的输出目录，以及配方所需的依赖。
- Weaver 服务地址、API key，以及所选模型的训练权限。
- 能够在本地读取配方数据和 tokenizer，或能够联网下载 tokenizer。

<a id="installation"></a>

## 安装

在 Linux 或 macOS 上从新检出的仓库开始：

```bash
git clone https://github.com/beipo-ai/NorthRL.git
cd NorthRL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname -s)" = Linux ]; then
  python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -e ".[core]" pyarrow
```

`core` 扩展包含训练运行时和 Weaver SDK，`pyarrow` 用于读取数学配方的 Parquet 文件。任务需要的其他依赖也应安装到这个环境中。

<a id="run-locally"></a>

## 本地运行

数学配方需要 `NORTHRL_DATA_PATH` 下的 `math/level{1..5}_train.parquet` 及对应的 `level{1..5}_test.parquet` 文件。在 `models.actor.name` 中选择 Weaver 账号可用的模型。替换占位值后，从仓库根目录使用 Bash 运行：

```bash
export EXPERIMENT_PATH="$(pwd)/logs/math-$(date +%Y%m%d-%H%M%S)"
export NORTHRL_DATA_PATH="/path/to/your/data"
export WEAVER_BASE_URL="https://your-weaver-host"
export WEAVER_API_KEY="your-weaver-api-key"
mkdir -p "$EXPERIMENT_PATH"
bash run.sh recipe/math/weaver.yaml \
  'runtime.logger.backends=[console]'
```

`run.sh` 使用上面导出的环境变量，用所选 YAML 启动 `northrl.main`，不会自动加载 `.env`。需要加载环境文件时，使用 `bash run.sh --env-file /path/to/env.sh recipe/math/weaver.yaml`，详见[环境文件](configuration-setup.md#environment-files)。脚本使用当前激活环境中的 `python`；`NORTHRL_PATH` 未设置或为空时，自动设为仓库根目录。运行依赖需要提前安装。示例只使用 console 日志，因此不需要 W&B 凭据。

<a id="minimal-setup-for-a-new-task"></a>

## 使用自己的任务

从符合任务类型的完整配方开始，修改模型、数据、奖励和训练设置，再执行 `bash run.sh /path/to/recipe.yaml`。如需使用 `--env-file`，将它放在配方路径前。E2B 任务还需按照[任务接入指南](../05-rollout-workers/e2b-task-integration.md)准备模板、任务 hook、NorthGate 路由和服务凭据。[最小 E2B 训练示例](../../recipe/e2b_minimal/README.md)提供任务数据、完整训练配置和本地启动命令。

<a id="verify-installation"></a>
<a id="monitoring"></a>

## 日志与结果

训练输出默认显示在终端。如需同时保存日志，将上面的启动命令替换为：

```bash
set -o pipefail
bash run.sh recipe/math/weaver.yaml \
  'runtime.logger.backends=[console]' 2>&1 | tee northrl.log
```

终端输出会保存到当前目录的 `northrl.log`。`set -o pipefail` 确保 Python 失败时整个管道也返回失败。请同时检查命令退出状态和 `NorthRL training completed` 日志；`run.sh` 不会写入 `exit_code` 文件，独立的 SFT 脚本会写入该文件。

NorthRL 在实验目录中保存生效配置 `train_config_resolved.yaml`。启用检查点保存时，元数据记录服务端路径，模型权重仍由 Weaver 管理。训练期间应保持驱动运行，例如使用 `tmux`。

<a id="next-steps"></a>

## 下一步

- 在[本地配置](configuration-setup.md)中了解服务访问、数据和日志设置。
- 阅读[数学配方指南](../../recipe/math/README.md)或[核心架构](../02-core-architecture/overview.md)。
