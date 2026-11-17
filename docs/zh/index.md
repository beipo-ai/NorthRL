[English](../en/index.md) | 简体中文 | [NorthRL](README.md)

# 文档目录

以下每篇文档均有英文和简体中文版本，可通过页面顶部链接切换到对应语言。

## 项目介绍与指南

- [NorthRL](README.md)
- [训练 step 指标与 W&B 查看方式](metrics.md)
- [Debug Data Dump 使用指南](debug_data_dump_usage.md)
- [调试单个 Rollout 批次](debug_rollout.md)
- [NorthRL 算法开发者用户指南](user-guide.md)

## 开始使用

- [接入自己的 Agent](developer-guide/05-rollout-workers/e2b-task-integration.md)
- [快速入门指南](developer-guide/01-getting-started/quick-start.md)
- [本地配置](developer-guide/01-getting-started/configuration-setup.md)

## 配方

- [最小 Agent 训练示例](recipe/e2b_minimal/README.md)
- [Reward rollout debug](recipe/debug_rollout/README.md)
- [单轮数学任务](recipe/math/README.md)
- [SFT 快速上手](recipe/sft_quickstart/README.md)
- [进阶 Recipe 的资源准备](recipe/advanced-recipes/README.md)

## 测试

- [NorthRL 测试套件](tests/README.md)

## 开发者指南

### 01-getting-started

- [本地配置](developer-guide/01-getting-started/configuration-setup.md)
- [快速入门指南](developer-guide/01-getting-started/quick-start.md)
### 02-core-architecture

- [活动跟踪系统](developer-guide/02-core-architecture/activity-tracking.md)
- [NorthRL 控制器](developer-guide/02-core-architecture/controller.md)
- [核心数据类型](developer-guide/02-core-architecture/data-types.md)
- [核心架构概览](developer-guide/02-core-architecture/overview.md)
### 03-data-loader

- [数据加载器](developer-guide/03-data-loader/data-loader.md)
- [监督微调数据](developer-guide/03-data-loader/sft-data.md)
### 04-trajectory-pool

- [Task、rollout 与 trajectory](developer-guide/04-trajectory-pool/trajectory-semantics.md)
- [轨迹池](developer-guide/04-trajectory-pool/trajectory-pool.md)
### 05-rollout-workers

- [Agent 任务接入](developer-guide/05-rollout-workers/e2b-task-integration.md)
- [E2B 任务接口](developer-guide/05-rollout-workers/e2b-task-contract.md)
- [E2B 执行机制与轨迹](developer-guide/05-rollout-workers/e2b-runtime.md)
- [Rollout 工作进程基类](developer-guide/05-rollout-workers/base-rollout-worker.md)
- [创建自定义 Rollout 工作进程](developer-guide/05-rollout-workers/custom-workers.md)
- [评估器](developer-guide/05-rollout-workers/evaluators.md)
- [Rollout 工作进程概览](developer-guide/05-rollout-workers/overview.md)
### 06-trainers

- [完整 rollout 与训练批次](developer-guide/06-trainers/sync-rollout-semantics.md)
- [Score centering（实验性）](developer-guide/06-trainers/score-centering.md)
- [算法组件](developer-guide/06-trainers/algorithm-components.md)
- [Rollout 与训练不匹配修正（MIS）](developer-guide/06-trainers/mis.md)
- [远程 API 训练器](developer-guide/06-trainers/remote-api-trainers.md)
### 08-features

- [运行监控](developer-guide/08-features/live-metrics.md)
- [Rollout 与训练耗时](developer-guide/08-features/rollout-timing-metrics.md)
- [模型生成与奖励评测故障](developer-guide/08-features/verifier-metrics.md)
- [错误处理](developer-guide/08-features/error-handling.md)
- [推理内容解析器](developer-guide/08-features/reasoning-parser.md)
- [工具解析器框架](developer-guide/08-features/tool-parser.md)
- [验证](developer-guide/08-features/validation.md)
- [权重同步](developer-guide/08-features/weight-synchronization.md)
### 11-configuration-reference

- [E2B 高级配置](developer-guide/11-configuration-reference/e2b-advanced.md)
- [E2B 执行时限与恢复](developer-guide/11-configuration-reference/e2b-execution-budgets.md)
- [数据加载器配置](developer-guide/11-configuration-reference/data-config.md)
- [Rollout Worker 配置](developer-guide/11-configuration-reference/rollout-config.md)
- [训练器配置](developer-guide/11-configuration-reference/trainer-config.md)
### 12-best-practices

- [模块开发](developer-guide/12-best-practices/module-development.md)
- [NorthRL 开发者指南](developer-guide/README.md)
