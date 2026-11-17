# NorthRL Documentation / 文档

| Language / 语言 | Overview / 项目介绍 | All documentation / 完整目录 | Developer guide / 开发者指南 |
| --- | --- | --- | --- |
| English | [Overview](en/README.md) | [Documentation index](en/index.md) | [Developer guide](en/developer-guide/README.md) |
| 简体中文 | [项目介绍](zh/README.md) | [文档目录](zh/index.md) | [开发者指南](zh/developer-guide/README.md) |

## Agent training / Agent 训练

| Start here / 从这里开始 | English | 简体中文 |
| --- | --- | --- |
| Run a minimal example / 运行最小示例 | [Example](en/recipe/e2b_minimal/README.md) | [示例](zh/recipe/e2b_minimal/README.md) |
| Connect your own agent / 接入自己的 Agent | [Integration guide](en/developer-guide/05-rollout-workers/e2b-task-integration.md) | [接入指南](zh/developer-guide/05-rollout-workers/e2b-task-integration.md) |
| Look up hooks and return values / 查询接口与返回值 | [Task contract](en/developer-guide/05-rollout-workers/e2b-task-contract.md) | [任务接口](zh/developer-guide/05-rollout-workers/e2b-task-contract.md) |

Start with the [Quick Start](en/developer-guide/01-getting-started/quick-start.md) to run NorthRL locally. [Local Configuration](en/developer-guide/01-getting-started/configuration-setup.md) covers service credentials, data paths, and logging.

先看[快速入门](zh/developer-guide/01-getting-started/quick-start.md)，在本地运行 NorthRL。[本地配置](zh/developer-guide/01-getting-started/configuration-setup.md)介绍服务凭据、数据路径和日志设置。

English and Simplified Chinese pages use matching paths under `docs/en/` and `docs/zh/`. Use the language links at the top of each page to switch versions. Shared images live in `docs/imgs/`. Commands and configuration paths are relative to the repository root unless a page says otherwise; code examples and API identifiers are kept unchanged in translations.

英文与简体中文文档分别位于 `docs/en/` 和 `docs/zh/`，文件路径一一对应。每页顶部均可切换语言，共享图片位于 `docs/imgs/`。除非页面另有说明，命令和配置路径均以仓库根目录为基准；译文保留代码示例和 API 标识符。

When adding or changing a page, update both language versions and their indexes together. Keep documentation links within the reader's language and preserve explicit section anchors. Keep local installation, environment, and launch instructions in the getting-started guides.

新增或修改文档时，请同步更新两种语言及对应目录；文档之间优先链接到同语言页面，并保留显式章节锚点。本地安装、环境和启动说明统一维护在快速入门章节中。

## Monitoring and training reference / 监控与训练参考

| Topic / 内容 | English | 简体中文 |
| --- | --- | --- |
| Training step metrics / 训练步指标 | [Read](en/metrics.md) | [阅读](zh/metrics.md) |
| Runtime monitoring / 运行监控 | [Read](en/developer-guide/08-features/live-metrics.md) | [阅读](zh/developer-guide/08-features/live-metrics.md) |
| Tasks and trajectories / 任务与轨迹 | [Read](en/developer-guide/04-trajectory-pool/trajectory-semantics.md) | [阅读](zh/developer-guide/04-trajectory-pool/trajectory-semantics.md) |

All topic pages live under `en/` and `zh/`. This directory contains only the shared documentation entry point and images.

专题正文统一放在 `en/` 和 `zh/` 下；本层只保留共用文档入口和图片目录。
