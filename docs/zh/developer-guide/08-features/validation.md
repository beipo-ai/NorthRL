[English](../../../en/developer-guide/08-features/validation.md) | 简体中文 | [文档目录](../../index.md)

<a id="validation"></a>

# 验证

<a id="overview"></a>

## 概述

`Validator` 对每次验证 rollout 收集一个完整 `Trajectory`，按完整轨迹对象聚合 score。内部序列数量不会增加某次执行的统计权重。验证使用独立收集器，不要求组成训练比较组。

**位置**：`northrl/validator.py`

<a id="key-concepts"></a>

## 关键概念

<a id="validation-cycle"></a>

### 验证周期

```
Weight Sync → Reset Validate DataLoader → Rollout Workers Generate → Collect Trajectories → Compute Metrics
```

设置 `validate.enabled: true` 并提供验证数据集。周期验证通过
`validate.schedule.every_n_steps` 配置，WeightSyncController 在发布模型版本时
使用该间隔。schedule 配置片段如下：

```yaml
validate:
  schedule:
    before_train: false
    every_n_steps: 5  # Validate every 5 published model versions
```

<a id="completion-criteria"></a>

### 完成条件

满足以下条件时，验证完成：
1. 验证数据加载器已结束（全部数据已消耗）
2. 所有 rollout 工作进程均已静止（没有正在执行的工作）

<a id="core-methods"></a>

## 核心方法

<a id="constructor"></a>

### 构造函数

```python
def __init__(self, config: DictConfig)
```

使用配置初始化验证器。

**参数**：
- `config`：配置字典

<a id="set_module_references"></a>

### set_module_references

```python
def set_module_references(self, validate_dataloader: BaseDataLoader)
```

设置对验证数据加载器的引用。

**参数**：
- `validate_dataloader`：提供验证数据的 DataLoader

<a id="put_trajectory"></a>

### put_trajectory

```python
def put_trajectory(self, trajectory: Trajectory) -> str
```

存储 rollout 工作进程提交的验证轨迹。

**参数**：
- `trajectory`：一个完整 `Trajectory`，验证分数放在 `extra_fields["score"]`

**返回值**：`"success"`（与 TrajectoryPool 接口一致）

<a id="is_complete"></a>

### is_complete

```python
def is_complete(self) -> bool
```

检查是否已收集全部验证轨迹。

**返回值**：验证完成时为 `True`

**逻辑**：
```python
return (
    execute(self._validate_dataloader.is_finished)
    and self._activity_tracker.is_rollout_worker_quiescent()
)
```

<a id="compute_and_log_metrics"></a>

### compute_and_log_metrics

```python
def compute_and_log_metrics(self) -> dict[str, float]
```

计算验证指标，并通过活动跟踪器记录。

**返回值**：计算所得指标的字典，带 `"val/"` 前缀

**流程**：
1. 收集所有轨迹的 score 值
2. 按 score 键和数据来源分组
3. 计算每个键的均值
4. 为指标名称添加 `"val/"` 前缀
5. 记录到实验跟踪工具（如 wandb）
6. 清空已收集轨迹

**指标键**：
- `val/{score_key}`：在该 score 键有数值的完整轨迹之间求均值
- `val/{data_source}_{score_key}`：在每个数据来源内按相同方式求均值
- `val/trajectory_count`：收到的完整验证轨迹数
- `val/num_samples`：同一轨迹计数的兼容名称
- `val/sequence_count`：这些轨迹包含的 token 序列数；普通结果贡献一条，分段结果贡献内部序列数，空结果贡献零条
- `val/segment_count`：`val/sequence_count` 的兼容名称

<a id="trajectory-score-structure"></a>

## 轨迹分数结构

验证提交完整对象。`score` 字典和 `data_source` 放在完整轨迹对象上，多序列结果也使用同样的完整轨迹的分数：

```python
from northrl.northrl_types import Trajectory

trajectory = Trajectory(
    tokens=[1, 2, 3],
    loss_mask=[0, 1, 1],
    reward=0.85,
    is_val=True,
    extra_fields={
        "data_source": "dataset_name",
        "score": {"reward": 0.85, "accuracy": 1.0, "custom_metric": 0.9},
    },
)
validator.put_trajectory(trajectory)
```

<a id="validation-workflow"></a>

## 验证工作流程

<a id="1-controller-triggers-validation"></a>

### 1. 控制器触发验证

当 `rollout_model_version % validate_freq == 0` 时：

```python
# In WeightSyncController.train_worker_notify_weight_update
if self._validate_freq > 0 and rollout_model_version % self._validate_freq == 0:
    self._waiting_for_validation = True
    # Trainer waits; sync and batch-async leave the next data batch unreleased.
```

<a id="2-controller-runs-validation"></a>

### 2. 控制器运行验证

领取另一个训练批次之前，训练器检查是否有验证
在等待。它在活动跟踪之外等待，在此期间被报告为
空闲。这也适用于 `fully-async`，其中验证开始时可能已有
训练批次排队。

```python
# In NorthRLController._run_validate
def _run_validate(self, model_tag):
    self._start_validate(model_tag)
    self._end_validate(model_tag)
```

`_start_validate()` 在将工作进程切换到验证之前重置验证输入。
`_end_validate()` 一边刷新进度并检查
工作进程错误，一边等待完成，然后计算指标并将工作进程切回训练。
它还调用 `WeightSyncController.end_validate()` 清除验证等待状态。
在 `fully-async` 中，数据加载器设置 `keep_batch_order: false`，并通过
`get_next_item()` 自动获取下一批次；`release_next_batch()`
不执行操作，`end_validate()` 也不调用它。训练器的验证等待
防止其消耗排队中的训练批次。训练需要继续时，
sync 和 batch-async 调用 `_release_next_batch()`，向工作进程放行下一数据批次。
轨迹池没有权重同步写入门控，始终接受已完成的结果。
最后一轮训练后，验证完成不会放行另一个批次。

<a id="3-rollout-workers-process-validation-data"></a>

### 3. Rollout 工作进程处理验证数据

Rollout 工作进程从验证输入生成完整轨迹。自定义 worker 从 task 设置 `is_val`，调用 `_put_trajectory` 后，该方法将对象路由至 validator：

```python
# In a custom worker's async rollout(task), after generating/evaluating:
trajectory = Trajectory(
    tokens=tokens,
    loss_mask=loss_mask,
    reward=final_reward,
    is_val=task.get("is_val", False),
    extra_fields={"score": scores, "data_source": task.get("data_source", "")},
)
return self._put_trajectory(trajectory)
```

<a id="4-validator-computes-metrics"></a>

### 4. 验证器计算指标

收集全部轨迹后：

```python
metrics = execute(self._validator.compute_and_log_metrics)
# Example output:
# {
#   "val/reward": 0.85,
#   "val/accuracy": 0.92,
#   "val/dataset1_reward": 0.83,
#   "val/dataset2_reward": 0.87,
#   "val/trajectory_count": 100,
#   "val/num_samples": 100,
#   "val/sequence_count": 140,
#   "val/segment_count": 140
# }
```

<a id="configuration"></a>

## 配置

<a id="validator-config"></a>

### 验证器配置

Controller 使用根配置创建内置 `Validator`。验证选项统一放在 `validate` 下，
没有 `validator.type` 选择器。以下示例是合并到已有 recipe 的配置片段。

<a id="validation-dataloader"></a>

### 验证 DataLoader

```yaml
validate:
  enabled: true
  data:
    validation_repeat_n: 1
    datasets:
      dev:
        files: [/data/validation.parquet]
        batch_size: 1
```

<a id="validation-frequency"></a>

### 验证频率

```yaml
validate:
  schedule:
    before_train: false
    every_n_steps: 5  # 0 disables periodic validation only
```

<a id="validation-vs-training-dataloaders"></a>

## 验证与训练数据加载器

Controller 从 `data.train` 派生训练输入，从 `validate.data.datasets` 派生
验证输入。`_derive_validate_dataloader_config()` 将每个命名数据集的 `files`
和 `batch_size` 转为内部 loader 配置，继承 `data.train.prompt_key`、actor 的
tokenizer 和 `data.max_prompt_length`，并将 `shuffle`、`drop_last`、
`filter_prompts` 固定为 `false`。

Controller 创建 `MultiDatasetValidationDataLoader`，由它为每个数据集创建
`TorchDataLoader(..., is_validate=True)`。每轮验证有限遍历一次，reset 后开始新一轮。
不要直接实例化抽象 `BaseDataLoader`，也不要在 `validate.data` 下添加 `epochs`；
每个验证 prompt 的尝试数由 `validation_repeat_n` 控制。训练输入在 RL 下使用
`TorchDataLoader`，SFT 下使用 `SFTStreamingDataLoader`。

**主要区别**：
- 训练数据加载器：RL 循环读取至 `training.total_steps`；SFT 使用配置的 step 或 epoch 上限
- 验证数据加载器：单次遍历，不打乱；验证期间重置并由工作进程选择

<a id="integration-example"></a>

## 集成示例

<a id="recipe-configuration"></a>

### 配方配置

```yaml
# Data and validation fragment for an existing RL recipe
data:
  train:
    files: [/data/train.parquet]
    batch_size: 1
    rollout_repeat_n: 2
    prompt_key: prompt
    shuffle: true
  max_prompt_length: 1024
  max_sequence_length: 2048

validate:
  enabled: true
  data:
    datasets:
      dev:
        files: [/data/validation.parquet]
        batch_size: 1
    validation_repeat_n: 1
  aggregation: mean
  schedule:
    before_train: false
    every_n_steps: 10
```

<a id="custom-evaluator-with-validation-scores"></a>

### 带验证分数的自定义评估器

E2B 评估器随奖励返回标量指标：

```python
from northrl.remote_rollout.contracts import EvaluationResult

result = EvaluationResult(
    reward=1.0,
    metrics={"accuracy": 1.0, "length": 2.0},
)
```

validator 读取 `Trajectory.extra_fields["score"]`，不会直接读取 evaluator 的返回值。`E2BRolloutWorker` 会将任务指标与奖励复制到该字典。自定义 worker 需要显式填入，例如 `{"reward": result.reward, **result.metrics}`。包含这些键时，验证指标为：
- `val/reward`：平均奖励
- `val/accuracy`：平均准确率
- `val/length`：平均答案长度

<a id="related-documentation"></a>

## 相关文档

- [核心架构](../02-core-architecture/overview.md) - 系统概述
- [权重同步](weight-synchronization.md) - 权重同步协调
- [活动跟踪](../02-core-architecture/activity-tracking.md) - 日志与监控
- [数据加载器](../03-data-loader/data-loader.md) - 数据加载系统
- [评估器](../05-rollout-workers/evaluators.md) - 评估模式
