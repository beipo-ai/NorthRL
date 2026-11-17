[English](../../../en/developer-guide/01-getting-started/configuration-setup.md) | 简体中文 | [文档目录](../../index.md)

<a id="configuration-setup"></a>

# 本地配置

NorthRL 的驱动运行在本地 Python 环境中，通过远端服务执行模型。先按照[快速入门](quick-start.md#installation)安装运行环境，再按[本地运行](quick-start.md#run-locally)启动完整配方。

<a id="service-access"></a>

## 服务访问

在运行 NorthRL 的 shell 中设置凭据：

```bash
export WEAVER_BASE_URL="https://your-weaver-host"
export WEAVER_API_KEY="your-weaver-api-key"
```

账号需要有 `models.actor.name` 所选模型的访问权限，以及所需的 LoRA 或全参数微调能力。`models.actor.tokenizer` 可以指定本地 tokenizer 目录或可下载的模型 ID，应与模型匹配。部分配方在该字段为 null 时使用服务提供的 tokenizer。

E2B 智能体任务还需要 `E2B_API_KEY`、`NORTHGATE_API_KEY` 和相应沙箱模板的访问权限。Weaver 与 NorthGate 需要提供所选 sampler 及其训练 token 数据。任务 hook、数据和模板依赖仍需适配，详见[任务接入](../05-rollout-workers/e2b-task-integration.md)。

<a id="data-and-output"></a>

## 数据与输出

```bash
export NORTHRL_DATA_PATH="/path/to/your/data"
export EXPERIMENT_PATH="/path/to/a/new-experiment"
mkdir -p "$EXPERIMENT_PATH"
```

数学配方使用 `NORTHRL_DATA_PATH`，其他配方可能使用不同变量。请检查 YAML 中的每个 `${oc.env:...}` 表达式。驱动需要能读取数据并写入输出目录；每次独立运行应使用新的实验目录。

使用 `bash run.sh /path/to/recipe.yaml` 选择配方，脚本会将 `TRAIN_CONFIG` 和 Hydra 的配置路径设为同一份源 YAML。自定义任务模块及任务文件需要能够在本地访问。

<a id="environment-files"></a>

## 环境文件

启动配方前，可以通过以下两种方式准备环境。

**指定环境文件。** 将服务凭据和本地路径写入 Bash 文件，例如 `/path/to/env.sh`：

```bash
WEAVER_BASE_URL="https://your-weaver-host"
WEAVER_API_KEY="your-weaver-api-key"
NORTHRL_DATA_PATH="/path/to/your/data"
EXPERIMENT_PATH="/path/to/a/new-experiment"
```

从仓库根目录运行，将 `--env-file` 放在配方路径前：

```bash
bash run.sh --env-file /path/to/env.sh recipe/math/weaver.yaml \
  'runtime.logger.backends=[console]'
```

脚本先 `source` 这个文件并自动导出其中的变量，再启动 Python。文件中的赋值会覆盖继承的值；指定的文件必须存在且可读。不要将凭据提交到 Git。也支持 `--env-file=/path/to/env.sh` 写法。

**使用当前终端中已导出的变量。** 环境已经配置好时，直接启动：

```bash
bash run.sh recipe/math/weaver.yaml 'runtime.logger.backends=[console]'
```

这些变量需要通过 `export NAME=value` 导出。如果手动执行 `source /path/to/env.sh`，文件里的赋值应使用 `export`；也可以执行 `set -a; source /path/to/env.sh; set +a` 自动导出。不指定环境文件时，`run.sh` 不读取任何环境文件，包括仓库根目录的 `.env`。

已有的 `NORTHRL_ENV_FILE=/path/to/env.sh bash run.sh recipe/math/weaver.yaml` 写法仍然支持，也属于显式指定环境文件。`--env-file` 的优先级高于 `NORTHRL_ENV_FILE`。切换到直接使用终端环境时，请先取消之前设置的 `NORTHRL_ENV_FILE`。

配方和环境文件的相对路径均以执行命令时的目录为准。加载环境文件前，脚本为 `NORTHRL_PATH` 提供默认值，即 `run.sh` 所在的仓库根目录；已提供的值会保留。环境文件可以用它拼接路径，也可以重新赋值；最终未设置或为空时仍使用仓库根目录。训练 YAML 继续使用 `${oc.env:...}` 引用。Hydra 覆盖参数放在配方路径后，如上面的日志设置。

脚本使用当前 `PATH` 中的 `python`，包括已激活的虚拟环境；也可以用 `PYTHON=/path/to/python` 指定解释器。脚本不会自动选择仓库中的 `.venv`、安装依赖或保存终端输出。独立的 [SFT 快速上手脚本](../../recipe/sft_quickstart/README.md)保留自己的 `.env` 加载、解释器选择、预检查和日志处理，直接运行即可。

<a id="logging"></a>

## 日志

在配方路径后传入 `'runtime.logger.backends=[console]'`，即可只使用终端日志。需要 W&B 时，先配置账号并启用相应后端：

```bash
export WANDB_KEY="your-wandb-api-key"
export WANDB_HOST="https://api.wandb.ai"
```

然后传入 `'runtime.logger.backends=[console,wandb]'`。[快速入门的日志示例](quick-start.md#monitoring)使用 `tee` 保存终端输出。生效配方和检查点元数据保存在 `EXPERIMENT_PATH` 下，模型权重由远端 Weaver 管理。

<a id="publishing-images"></a>

## 可选的镜像发布

如需从本仓库向 Docker Hub 发布运行镜像，将 GitHub Actions 仓库变量 `NORTHRL_IMAGE_REPOSITORY` 设为目标镜像仓库地址，不带 tag，并配置 `DOCKERHUB_USERNAME` 和 `DOCKERHUB_TOKEN` 仓库 Secret。发布工作流使用 GitHub 托管的 runner；未设置 `NORTHRL_IMAGE_REPOSITORY` 时，发布目标检查会失败。

<a id="optional-issue-automation"></a>

## 可选的 Issue 自动化

Issue 关联和里程碑工作流默认使用本仓库的 `GITHUB_TOKEN`，写操作限于本仓库。如需跨仓库联动，配置对目标仓库有权限的 `ISSUE_AUTOMATION_TOKEN`；已有的 `PROJECT_TOKEN` Secret 仍可作为回退。使用这些功能不需要组织 Project，也不需要另外部署反馈仓库或知识库。

<a id="troubleshooting"></a>

## 常见问题

- **缺少环境变量：** 在终端中导出变量，或在通过 `--env-file` 指定的文件中设置变量，并检查配方中的 `${oc.env:...}` 引用。同目录的 `.env` 或 `env.sh` 不会自动加载。
- **缺少 Python 模块：** 在运行驱动的同一环境中安装 `core` 扩展及任务依赖。
- **模型不可用或请求未授权：** 检查 Weaver 服务地址、key、模型标识及训练权限。
- **找不到 tokenizer 或数据：** 检查本地路径和下载权限；远端训练服务不会自动提供本地文件。
- **没有保存终端日志：** 直接入口默认输出到终端；使用[快速入门](quick-start.md#monitoring)中的 `tee` 命令保存。
