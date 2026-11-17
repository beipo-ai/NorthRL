[English](../../en/developer-guide/README.md) | 简体中文 | [文档目录](../index.md)

<a id="northrl-developer-guide"></a>

# NorthRL 开发者指南



本指南介绍 NorthRL 的任务接入、训练配置、架构和扩展开发。
在本地运行 NorthRL，通过 API 调用模型训练与采样服务。

<a id="-documentation-structure"></a>

## 文档结构

<a id="-01-getting-started"></a>

###  01. 快速入门
快速开始指南与配置设置。

- [快速开始](01-getting-started/quick-start.md) - 本地安装、启动和日志
- [本地配置](01-getting-started/configuration-setup.md) - 服务凭据、数据路径和本地环境

<a id="-02-core-architecture"></a>

###  02. 核心架构
系统架构与基础组件。

- [概览](02-core-architecture/overview.md) - 架构图、组件概览
- [控制器](02-core-architecture/controller.md) - NorthRLController、模块注册表、生命周期
- [数据类型](02-core-architecture/data-types.md) - 核心类型（Trajectory、Batch 等）
- [活动跟踪](02-core-architecture/activity-tracking.md) - 监控与日志

<a id="-03-data-management"></a>

###  03. 数据管理
数据加载与管理。

- [数据加载器](03-data-loader/data-loader.md) - BaseDataLoader、TorchDataLoader、自定义加载器
- [SFT 数据](03-data-loader/sft-data.md) - token 化输入与 messages 输入、损失掩码、自定义聊天渲染

<a id="-04-trajectory-pool"></a>

###  04. 轨迹池
轨迹收集与批处理。

- [Task、rollout 与 trajectory](04-trajectory-pool/trajectory-semantics.md) - 任务身份、完整轨迹、内部序列与比较组
- [轨迹池](04-trajectory-pool/trajectory-pool.md) - TrajectoryPool、存储类型、分组策略

<a id="-05-rollout-workers"></a>

###  05. Rollout Worker
Rollout Worker 架构与实现。

- [Agent 任务接入](05-rollout-workers/e2b-task-integration.md) - 接入已有 Agent 并配置训练
- [E2B 任务接口](05-rollout-workers/e2b-task-contract.md) - 任务函数、返回值与评测约定
- [E2B 执行机制](05-rollout-workers/e2b-runtime.md) - 执行阶段、模型路由、轨迹与日志
- [概览](05-rollout-workers/overview.md) - 架构、工作流程、Worker 层次结构
- [基础 Rollout Worker](05-rollout-workers/base-rollout-worker.md) - 核心接口、生命周期、方法
- [自定义 Worker](05-rollout-workers/custom-workers.md) - 创建任务专用 Worker
- [评估器](05-rollout-workers/evaluators.md) - 评估模式与最佳实践

<a id="-06-trainers"></a>

###  06. 训练器
训练器架构与算法集成。

- [完整 rollout 与训练批次](06-trainers/sync-rollout-semantics.md) - 更新边界、任务结果与批次统计
- [Score centering（实验性）](06-trainers/score-centering.md) - Weaver 约束与实验性评分配置
- [算法组件](06-trainers/algorithm-components.md) - 公共组件职责、数据流与算法指南
- [远程 API 训练器](06-trainers/remote-api-trainers.md) - RemoteApiTrainer、GRPO、CrossEntropy

<a id="-07-services"></a>

<a id="-08-features"></a>

###  08. 功能
贯穿多个模块的功能。

- [运行监控](08-features/live-metrics.md) - 实时沙箱、模型请求与工具执行状态
- [Rollout 与训练耗时](08-features/rollout-timing-metrics.md) - 等待、执行与阶段重叠的统计口径
- [模型生成与奖励评测故障](08-features/verifier-metrics.md) - 失败、超时与重试计数
- [权重同步](08-features/weight-synchronization.md) - WeightSyncController、同步模式、协调机制
- [验证](08-features/validation.md) - Validator、验证周期、指标计算
- [错误处理](08-features/error-handling.md) - ErrorReporter 与错误策略
- [工具解析器](08-features/tool-parser.md) - 适配不同模型格式的灵活工具调用解析框架
- [推理内容解析器](08-features/reasoning-parser.md) - 提取并结构化模型响应中的推理内容

<a id="-09-recipes"></a>

###  09. 配方
配方开发与结构。

- [最小 Agent 训练示例](../recipe/e2b_minimal/README.md) - 任务代码、数据、完整配置与本地启动
- [数学任务配方](../recipe/math/README.md) - 单轮生成与奖励配置

<a id="-11-configuration-reference"></a>

###  11. 配置参考
完整配置文档。

- [数据配置](11-configuration-reference/data-config.md) - 数据加载器配置选项
- [E2B 高级配置](11-configuration-reference/e2b-advanced.md) - 历史 token 复用与 Agent 网络策略
- [E2B 执行时限](11-configuration-reference/e2b-execution-budgets.md) - 各阶段时限、重试与结果恢复
- [Rollout 配置](11-configuration-reference/rollout-config.md) - Rollout Worker 配置
- [训练器配置](11-configuration-reference/trainer-config.md) - 按类型说明训练器配置

<a id="-12-best-practices"></a>

###  12. 最佳实践
开发指南与模式。

- [模块开发](12-best-practices/module-development.md) - 编写 NorthRL 模块


<a id="-quick-navigation"></a>

##  快速导航

<a id="for-new-users"></a>

### 新用户

1. 从[快速开始](01-getting-started/quick-start.md)入手
2. 按[本地配置](01-getting-started/configuration-setup.md)准备服务凭据、数据路径和运行环境
3. 阅读[架构概览](02-core-architecture/overview.md)

<a id="for-task-developers"></a>

### 任务开发者

1. 跑通[最小 Agent 训练示例](../recipe/e2b_minimal/README.md)。
2. 按[接入指南](05-rollout-workers/e2b-task-integration.md)连接自己的 Agent。
3. 根据需要查阅[任务接口](05-rollout-workers/e2b-task-contract.md)和[评测说明](05-rollout-workers/evaluators.md)。

<a id="for-algorithm-developers"></a>

### 算法开发者

1. 阅读[算法组件](06-trainers/algorithm-components.md)，了解算法、配置和数据流。

<a id="-key-concepts"></a>

##  关键概念

<a id="architecture"></a>

### 架构
- **模块化设计**：清晰划分职责
- **服务抽象**：为不同后端提供统一 API
- **活动跟踪**：全面的监控系统
- **并发控制**：使 rollout 并发符合驱动内存与远端服务容量

<a id="data-flow"></a>

### 数据流
```
DataLoader → RolloutWorker → TrajectoryPool → Trainer
                ↓
            Evaluator
```

<a id="weight-synchronization"></a>

### 权重同步
```
Trainer → WeightSync → InferenceService
             ↓
         Validator
```

<a id="lifecycle"></a>

### 生命周期
```
Controller → Initialize → Run → Monitor → Stop
                ↓
         [ Training Loop ]
```

<a id="-development-workflow"></a>

##  开发流程

<a id="setting-up-a-new-task"></a>

### 设置新任务

1. **创建配方结构**：
   ```bash
   mkdir -p recipe/my_task
   ```

2. **配置任务**：
   - 创建 `my_task.yaml`（配方配置）
   - 选择单轮 `reward_fn`，或在 `rollout.py` 中实现远程 `run_agent` 和 `evaluate` 钩子

3. **运行训练**：按[快速入门](01-getting-started/quick-start.md)使用你的任务配置运行。

<a id="implementing-a-custom-worker"></a>

### 实现自定义 Worker

创建 `BaseRolloutWorker` 子类，实现 `async rollout(task)`，通过
`_put_trajectory()` 提交完整 `Trajectory`。在配方中设置
`rollout.custom_module_path` 和 `rollout.custom_class_name` 以加载该类。
接口约定详见[自定义 Worker](05-rollout-workers/custom-workers.md)。

<a id="implementing-a-custom-trainer"></a>

### 实现自定义训练器

1. **创建训练器类**：

2. **在控制器中注册**：
   ```python
   MODULE_REGISTRY[NorthRLRole.TRAINER]["my_trainer"] = MyTrainer
   ```

<a id="-contributing"></a>

##  参与贡献

添加新功能时：

1. 在相应章节**更新相关文档**
2. **添加示例**来展示用法
3. 如新增配置，**更新配置参考**
4. 如引入新模式，**补充最佳实践**

<a id="-related-documentation"></a>

##  相关文档

- [主 README](../README.md) - 项目概览
- [用户指南](../user-guide.md) - 端到端教程
- [源码 API](../../../northrl/) - Python 实现与文档字符串

<a id="-getting-help"></a>

##  获取帮助

- **Issues**：通过 GitHub Issues 报告缺陷和提出功能请求
- **Discussions**：通过 GitHub Discussions 提问
- **示例**：完整示例见 `recipe/` 目录
- **测试**：用法示例见 `tests/` 目录

<a id="legend"></a>

## 图例

-  文档完整
-  编写中 / 占位页
-  需要审阅或更新

---

**最后更新**：2026-01-16
