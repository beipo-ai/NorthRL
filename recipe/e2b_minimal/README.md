# Minimal Agent task example / 最小 Agent 任务示例

[English walkthrough](../../docs/en/recipe/e2b_minimal/README.md) · [中文说明](../../docs/zh/recipe/e2b_minimal/README.md)

[`rollout.py`](rollout.py) is the shared task module: `run_agent` calls the model once, and `evaluate` compares the response with a hidden answer. NorthRL loads these hooks in an E2B sandbox; this file is not a standalone training program.

The walkthrough provides prerequisites, sample data, a complete training configuration, local startup commands, and checks for answers, rewards, and training results. To integrate your own Agent, follow the [English guide](../../docs/en/developer-guide/05-rollout-workers/e2b-task-integration.md) or [中文接入指南](../../docs/zh/developer-guide/05-rollout-workers/e2b-task-integration.md).

[`rollout.py`](rollout.py) 是中英文说明共用的任务代码：`run_agent` 调用一次模型，`evaluate` 将回答与隐藏答案比较。NorthRL 在 E2B 沙箱中加载这两个接口；这个文件本身不是训练启动程序。

说明页提供环境要求、样例数据、完整训练配置、本地启动命令，以及回答、奖励和训练结果的检查方法。接入自己的 Agent 时，请参考上方接入指南。
