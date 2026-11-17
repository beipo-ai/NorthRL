[English](../en/debug_data_dump_usage.md) | 简体中文 | [文档目录](index.md)

<a id="debug-data-dump-使用指南"></a>

# Debug Data Dump 使用指南

在完整的 Weaver 训练配方中设置 `runtime.debug`，即可保存本地驱动运行时的数据。若只想查看一个 rollout 批次、不进行训练，请使用[单批次 Rollout 调试](debug_rollout.md)。

<a id="enable-data-dump"></a>

## 启用数据转储

按照[快速入门](developer-guide/01-getting-started/quick-start.md#run-locally)准备本地 Python 环境、数据和服务凭据。将以下设置合并到配方已有的 `runtime` 块中：

```yaml
runtime:
  debug:
    enable_data_dump: true
    dump_dir: "${oc.env:EXPERIMENT_PATH}/debug_dump"
    dump_every_n_steps: 1
    remote_api_dump_options:
      prepared_trajectories: true
      datums: true
      training_metrics: true
```

将 `TRAIN_CONFIG` 指向修改后的 YAML，设置可写的 `EXPERIMENT_PATH`，再从仓库根目录运行：

```bash
python -m northrl.main \
  --config-path="$(dirname "$TRAIN_CONFIG")" \
  --config-name="$(basename "${TRAIN_CONFIG%.*}")" \
  'runtime.logger.backends=[console]'
```

这会通过 Weaver 执行实际训练。转储设置不能替代模型权限、数据及其他配方配置。`dump_every_n_steps` 控制保存间隔；训练器执行到相应阶段时，才会保存已启用的数据类别。

<a id="dump-directory-structure"></a>

## 转储内容

```text
debug_dump/
├── prepared_trajectories/
│   └── step_000000.pt
├── datums/
│   └── step_000000.pt
└── training_metrics/
    └── step_000000.pt
```

- `prepared_trajectories`：优势计算后准备好的完整 rollout 结果，包括 token 数据和训练信号。
- `datums`：转换后发送给 Weaver 训练 API 的数据。
- `training_metrics`：训练服务返回的指标。

当前准备好的轨迹转储使用 `format_version=3`：普通轨迹将 token 字段保留在顶层，分段结果则使用内部 `segments` 字典。请将这些文件作为训练数据，与生成它们的实验一起保存。

<a id="trajectory-replay-contract"></a>

## 在开发工具中回放

内部 mock worker 可以在构造时接收回放配置并加载已保存轨迹。这是开发 API，并非本地训练命令的回放选项：公开配方 schema 没有提供 `trajectory_load_path` 或 `trajectory_format` 字段。

回放会恢复完整轨迹，并检查每个保存的组是否恰好包含各个预期 `run_id` 槽位的一条轨迹。保存的 `expected_rollouts` 必须与触发任务一致。缺少成员、重复槽位或旧格式中的结果不一致都会报错。只有能够重建完整执行及比较组时，读取器才接受 v2 嵌套格式和更早的扁平格式。

每个 worker 实例在第一次触发时回放整份文件一次。各组获得新的 `group_id`，轨迹获得新的 `rollout_id`，原 `run_id` 槽位保持不变。后续触发只继续消耗数据加载器。开发回放请使用一个 worker，避免多个 worker 分别加载并回放同一文件。

<a id="troubleshooting"></a>

## 常见问题

- 目录为空时，检查 `enable_data_dump`、类别开关、保存间隔，以及训练是否执行到了对应阶段。
- 开启转储仍会调用 Weaver。只需要 rollout 调试时，请使用独立的单批次命令。
- NorthRL 的本地转储选项不会启用远端服务内部的调试设置；那些设置由服务运营方管理。
