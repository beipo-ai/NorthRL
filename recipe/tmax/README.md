# tmax 终端任务（Claude Code）

资源准备：[English](../../docs/en/recipe/advanced-recipes/README.md#tmax) | [简体中文](../../docs/zh/recipe/advanced-recipes/README.md#tmax)

把 tmax 终端任务接入 [E2B 任务](../../docs/zh/developer-guide/05-rollout-workers/e2b-task-integration.md)，在沙箱里用 Claude Code 做 agent。每道题使用自己准备的 E2B 模板，由数据行的 `template` 指定。

本仓库提供任务入口和训练配置，不包含任务数据、题目环境源码、模板构建工具或评测包。运行前需要准备这些资源，不能只填写 API key 就直接使用。

| 内容 | 位置 | 阶段 |
| --- | --- | --- |
| 题面 | 数据行的 `instruction` | agent（`input_fields: [instruction]`） |
| 运行环境 | 题目模板：Ubuntu 22.04，默认用户 `user`（免密 sudo），工作目录 `/home/user` | agent |
| 打分脚本 `run-tests.sh` | 评测包 `<assets-root>/<task_id>/run-tests.sh`，测试内嵌其中 | 只在 agent 结束后上传 |

reward 是二值：`run-tests.sh` 输出 `SWEBench results starts here / PASSED|FAILED / SWEBench results ends here`，PASSED 记 1，FAILED 记 0。

## 运行方式

[`northrl/rollout_entrypoints/tmax_claude_code_rollout.py`](../../northrl/rollout_entrypoints/tmax_claude_code_rollout.py) 实现三个阶段：

- `prepare`（root）：删掉模板里烘进去的旧打分文件 `/tests`、`/logs/verifier`；把 Claude Code 从共享 E2B 卷拷到 `/dev/shm`，然后卸载卷，避免 agent 改动别的 rollout 共用的二进制；在 NorthRL 运行时装好之后，用模板自带的 `/usr/bin/python3` 记录打分工具链的哈希（`/root/.tmax_integrity`），打分时 `run-tests.sh` 会重新核对。
- `run_agent`：以模板用户 `user` 运行 `claude --print --model northrl`，工作目录 `/home/user`，stream-json 输出只留在 root 进程和 `/root/northrl-tmax/` 下，agent 读不到。运行时长由平台的 agent 预算 `rollout.execution.agent_timeout_seconds`（1 小时）控制，预算用完时平台取消 `run_agent`，只杀 Claude Code 的进程组，然后照常打分；正常结束时 agent 自己起的后台服务保留，供测试检查。
- `evaluate`（root）：运行评测包里的 `run-tests.sh`（默认 20 分钟超时），按结果块给 0/1。缺结果块或打分超时抛 `RolloutEnvironmentError`，不当作 0 分训练；工具链被改动时 `run-tests.sh` 自己判 FAILED，指标里记 `tmax_integrity_mismatch`。

可以按题覆盖的字段（加进数据行并放进 `input_fields` 才会生效）：`evaluator_timeout_sec`、`install_timeout_sec`、`effort`、`compact_window`、`max_output_tokens`，默认值见入口里的 `_DEFAULTS`。

## 强制思考

Claude Code 只发 Anthropic 格式的请求。入口通过 `CLAUDE_CODE_EXTRA_BODY` 在每个请求里带上 `output_config: {"effort": "high"}`，NorthGate 把它映射成 `reasoning_effort=high`，生成提示以 `<think>\n` 结尾。`chat_template_kwargs` 和顶层 `reasoning_effort` 在 `/v1/messages` 上不生效。

Claude Code 会把上一轮的 thinking 块原样放回历史，但前提是响应里的模型名和它请求时用的一致；relay 因此在 Anthropic 响应里把采样器的模型名改回客户端请求的 `northrl`，否则 Claude Code 会删掉历史里的思考内容，每次调用都变成一个新的训练片段。

## 准备数据

训练前准备训练 JSONL 和逐题评测包，分别通过 `NORTHRL_TMAX_TRAIN_DATA` 和 `NORTHRL_TMAX_TASK_FILES_ROOT` 指定。

JSONL 每行包含 `{task_id, template, instruction, band}`：`task_id` 用于查找评测包，`template` 是当前 E2B API key 可以访问的模板 ID 或别名，`instruction` 是给 agent 的题面，`band` 是题目分档元数据。例如：

```json
{"task_id":"tmax__task_000002_81d660f3__56b98a95","template":"tmax__task_000002_81d660f3__56b98a95","instruction":"完成题面要求的终端任务。","band":"mid"}
```

评测包按 `<NORTHRL_TMAX_TASK_FILES_ROOT>/<task_id>/run-tests.sh` 存放，每个目录只含测试脚本的真实拷贝，不使用软链接。评测包由平台在 agent 结束后上传。

如果所选模型或 NorthGate 采样通道只接受文本，Claude Code 读取图片后发出的多模态请求会被拒绝，对应 rollout 会作为环境错误处理。准备数据时应检查题目文件和构建过程，移除依赖图片或视频输入的任务；agent 自行生成并读取图片的情况仍可能发生。训练步数应按自己的数据规模设置。

## 沙箱模板与 Claude Code 卷

**题目模板**：每道题一个 E2B 模板，数据行的 `template` 必须指向你有权限使用的对应模板。

- 模板用题目自带的环境源码构建：`Dockerfile`（基于 Ubuntu 22.04）加 `post_install.sh`，装好了这道题的依赖和文件。默认用户是 `user`（免密 sudo），工作目录 `/home/user`，规格约 1 核 CPU、2 GB 内存、5 GB 磁盘。
- 每次 rollout 都从模板新起一个沙箱，用完即销毁。
- 自行构建或取得符合上述环境要求的模板，并确认它们在所用 E2B 部署中可由自己的 `E2B_API_KEY` 启动。本仓库不提供预先发布的题目模板。
- 不要把隐藏打分脚本放入模板：`run-tests.sh` 在 agent 结束后才上传。入口还会清理旧模板可能遗留的 `/tests`、`/logs/verifier`；准备模板时仍需检查其他位置，避免把评测内容暴露给 agent。

**Claude Code 卷**：此入口通过 E2B 卷获取 `claude` 二进制，不依赖模板内预装的 Claude Code：

- `rollout.execution.volume_mounts` 把卷挂到 `/mnt/ccbin`。
- `prepare` 把 `/mnt/ccbin/claude` 整个顺序拷到 `/dev/shm`，然后卸载卷，agent 改不到团队共享的那份。之所以整个拷，是因为高并发下直接在共享卷上按需分页读二进制，可能慢到几百秒。
- 在自己的 E2B 部署中准备一个 API key 有权限挂载的卷：
  1. 使用该部署支持的卷管理方式创建一个卷；
  2. 起任意一个沙箱并挂上这个卷（例如挂到 `/mnt/ccbin`），把 Linux x64 的 Claude Code 原生二进制放到卷的根目录，文件名为 `claude`，并加上可执行权限；
  3. 在沙箱里运行 `/mnt/ccbin/claude --version`，确认能启动；
  4. 训练时把 `NORTHRL_TMAX_CC_VOLUME` 设成这个卷名。
- 所用 E2B 部署必须支持此 recipe 使用的卷挂载能力；仅有模板访问权限不足以运行它。

## 训练配置

[`train.yaml`](train.yaml) 的默认设置：每步 32 题 × 16 条、温度 1.0、单次调用 16k token、上下文 128k、每条 1 小时、lr 1e-6、Adam β2 0.98、weight decay 0.01。Claude Code 的二进制来自 `rollout.execution.volume_mounts` 挂到 `/mnt/ccbin` 的 E2B 卷。三段平台预算：准备 1200 秒（建沙箱和拷 Claude Code）、agent 3600 秒、评测 1500 秒（`run-tests.sh` 自身限 1200 秒，再加上传评测包的时间）。需要的环境变量：

| 变量 | 含义 |
| --- | --- |
| `NORTHRL_PATH` | NorthRL checkout 的绝对路径；根目录 `run.sh` 默认自动设置，直接启动 Python 时需自行导出 |
| `NORTHRL_TMAX_MODEL` | 必填，Weaver 服务中可训练的模型名 |
| `NORTHRL_TMAX_TOKENIZER` | 必填，与该模型匹配的 Hugging Face tokenizer ID 或本地目录 |
| `NORTHRL_TMAX_TRAIN_DATA` | 预处理后的训练 JSONL，格式见「准备数据」 |
| `NORTHRL_TMAX_TASK_FILES_ROOT` | 逐题评测包根目录 |
| `NORTHRL_TMAX_CC_VOLUME` | 必填，自己准备的、含 `claude` 二进制且可挂载的 E2B 卷名 |
| `NORTHRL_TMAX_TOTAL_STEPS` | 训练步数，默认 109；按自己的数据规模调整 |
| `NORTHRL_TMAX_BATCH_SIZE` / `NORTHRL_TMAX_REPEAT_N` / `NORTHRL_TMAX_ROLLOUT_WORKERS` | 冒烟时缩小规模用 |

宿主另需 `WEAVER_API_KEY`、`NORTHGATE_API_KEY`，以及能访问 tmax 模板的 E2B 配置：

- 能访问所需题目模板和 Claude Code 卷的 `E2B_API_KEY`，见上文「沙箱模板与 Claude Code 卷」。
- 用自建 E2B 部署时，还要把 `E2B_DOMAIN`（或 `E2B_API_URL`）指向该部署，否则会连到默认的 E2B 云。
- 按实际网络环境配置代理；只有需要直接连接的服务地址才加入 `NO_PROXY`/`no_proxy`。

将上述变量及 `WEAVER_BASE_URL`、`EXPERIMENT_PATH` 写入自己的环境文件后，从仓库根目录启动：

```bash
bash run.sh --env-file /path/to/env.sh recipe/tmax/train.yaml \
  'runtime.logger.backends=[console]'
```

环境变量已在终端导出时，可省略 `--env-file /path/to/env.sh`。首次试跑先用上表中的批量、重复次数、并发数和训练步数变量缩小规模。环境文件加载规则见[本地配置](../../docs/zh/developer-guide/01-getting-started/configuration-setup.md#environment-files)。

## 执行与训练行为

- Agent 使用 Claude Code，其工具、系统提示和上下文压缩策略会影响任务表现；对比实验时应保持这些设置一致。
- 没有 R3（E2B 路径不支持），用训练侧重算 log-prob（`sampling_logprob_source: recomputed`），一批一次更新。
- 平台每轮按模板重新渲染整段历史。NorthGate 把 Claude Code 回传的工具参数按键名字母序重排时，若与采样时的顺序不一致，这一轮会切成新的训练片段。
