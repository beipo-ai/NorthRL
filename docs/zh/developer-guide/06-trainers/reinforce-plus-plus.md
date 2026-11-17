[English](../../../en/developer-guide/06-trainers/reinforce-plus-plus.md) | 简体中文 | [文档目录](../../index.md)

# REINFORCE++ 与 REINFORCE++-baseline

通过 `algorithm.advantage.estimator.mode` 选择估计器，并设置
`algorithm.type: grpo`。factory 构造 `northrl.algorithm.advantage.reinforce_plus_plus` 中的
`ReinforcePlusPlusEstimator`，trainer 直接调用
`estimate(trajectories, {}, metrics)`；返回原 `list[Trajectory]`，
各 sequence 保存自己的 response-only advantage。
两者使用原生 `importance_sampling` 或 `ppo_clip` 损失。
在此配置下，省略 estimator 时使用默认的 `grpo` 估计器，详见[算法组件](algorithm-components.md)。

<a id="configuration"></a>

## 配置

将 `recipe/math/weaver.yaml` 复制为
`recipe/math/weaver_reinforce_plus_plus.yaml`。保留原有模型、数据路径、rollout、
backend 和 runtime 设置，将 `algorithm` 块替换为下列内容。此处是完整 recipe
的覆盖配置，不是独立的训练配置。

```yaml
algorithm:
  type: grpo
  advantage:
    estimator:
      mode: reinforce_plus_plus
      params:
        gamma: 1.0
    postprocessing:
      mode: null
  loss:
    mode: ppo_clip
    params:
      clip:
        ratio: 0.2
    reduction:
      mode: token_mean
  sampling_logprob_source: recomputed
  entropy_coeff: 0.0
  optim:
    lr: 1.0e-6
    batch_size: null
    mini_batch_size: 16
  kl:
    mode: null
```

math recipe 已设置 `data.train.rollout_repeat_n: 8`。普通 REINFORCE++ 也允许
设为 `1`。使用 baseline 版本时，只需改为
`advantage.estimator.mode: reinforce_plus_plus_baseline`，保留其 `params.gamma: 1.0`，并确保
`data.train.rollout_repeat_n >= 2`。配置好原 recipe 所需的服务和环境变量后运行：

```bash
export TRAIN_CONFIG="$(pwd)/recipe/math/weaver_reinforce_plus_plus.yaml"
mkdir -p "$EXPERIMENT_PATH"
python -m northrl.main \
  --config-path="$(dirname "$TRAIN_CONFIG")" \
  --config-name="$(basename "${TRAIN_CONFIG%.*}")" \
  'runtime.logger.backends=[console]'
```

两种估计器均要求 `loss.reduction.mode: token_mean`，使用原生
`loss.mode: importance_sampling` 或 `ppo_clip`；不支持 custom callback。
它们不支持 `algorithm.type: opd` 或 `sft`。只有普通 REINFORCE++ 可以使用非默认
`gamma`，取值必须为 0 到 1 之间的有限数。切换 advantage 估计器不会改变 PPO
clipping、entropy 或 MIS 配置。

请在 `algorithm.advantage.estimator` 中配置 `mode` 和 `params.gamma`。
REINFORCE++ 使用全 step token whitening，因此其 estimator 参数不包含 GRPO 的
`norm_adv_by_std` 选项。

<a id="optional-advantage-postprocessing"></a>

## 可选的 advantage 后处理

两种版本均支持[工具错误优势重分配](advantage-processing.md)。启用时，只需将上面
配置中的 `algorithm.advantage.postprocessing` 替换为：

```yaml
postprocessing:
  mode: tool_error_rebalance
  params:
    negative_multiplier: 2.0
    positive_scale_max: 2.0
    negative_scale_min: 0.5
```

重分配在 MIS 或拆批之前，对完整 step 已 whitening 的 response-only token advantage
执行，不会再次 whitening。后处理器保存这些 base advantage 以支持重放。
每次重新估计前，trainer 都会清除保存的 base 和 shaped 值，包括本次已关闭后处理的情况。

<a id="returns-and-the-group-baseline"></a>

## Return 与组内 baseline

将完整 trajectory 所有 segment 的有效 response token 按顺序记为
`j = 1, ..., T`，不计入 prompt、工具 observation、masked token 和 padding。
设 `R` 为完整轨迹的最终 outcome reward。token reward 为：

```text
r_j = R if j == T else 0
```

普通 REINFORCE++ 计算 reward-to-go：

```text
G_j = r_j + gamma * G_(j+1)，其中 G_(T+1) = 0
```

每条完整轨迹只添加一次终点 outcome。return 跨 segment 边界传播；
masked observation 不会清空累计 return，也不消耗 discount 步数。
当 `gamma: 1.0` 时，一条完整轨迹的所有有效 token
在 whitening 前都使用该完整轨迹的 outcome reward。

baseline 版本使用每条完整轨迹的 outcome reward，再减去所属
`(model_tag, group_id)` 比较组的 inclusive mean，即包含自身的均值：

```text
B_g = mean_i_in_group(R_i)
G_ij = R_i - B_g
```

此版本不使用 leave-one-out baseline，不除以组内 reward 标准差，也不折扣
return。每条完整轨迹只贡献一个 reward，与 segment 数量无关。成功但没有
有效 token 的完整轨迹仍参与组内 baseline，但不向 whitening 或 loss
贡献 token。
后处理和 MIS 计算完成后，公共转换函数检查最终 `loss_mask`，全零的 sequence
不生成训练 Datum 或来源记录。此训练流程保留有有效 token 但 advantage 为零的
sequence。这一步不改变已经算好的 baseline 和 whitening；转换设置见
[公共数据流](algorithm-components.md#step-data-flow-and-alignment)。

两种版本都在计算 return 前执行完整组校验和失败组修复。baseline 配置
要求每组至少有两个预期成员；底层估计器会将 singleton 的中心化 score 安全置零。
原始 rollout reward 指标描述替换前的结果。

<a id="global-token-whitening-and-batching"></a>

## 全 step token whitening 与拆批

两种版本都在修复后的整个已交付训练 step 内，对所有有效 response token
的 `G` 做一次 whitening。设有效 token 数为 `N`：

```text
mu = sum(G) / N
variance = sum((G - mu)^2) / (N - 1)
A = (G - mu) / sqrt(variance + 1e-8)
```

这里使用无偏样本方差，并按 token 加权：更长的回答会为这些统计量贡献更多
观测值。`N <= 1` 时 advantage 为零；masked 位置的训练 advantage 为零。
每个 segment 保存自己的 response-token `token_advantages` 向量；完整轨迹
的标量 `advantage` 仅用于诊断，取其有效 token advantage 的均值，空轨迹取零。
compact token buffer 输出 response-only 的 `array('d')` advantage，list 输入仍输出
list。token 和 mask 处理尽可能直接使用 typed buffer，whitening 保留 float64 精度。
折扣 return 跨 segment 传播，不通过 gamma 的幂做除法，长 response 和很小的
gamma 仍保持数值稳定。estimator 直接读取原 token 和 mask buffer，统计有效
response token 数量，并为 mask 校验、return 和 whitening 分配计算缓冲区。

whitening 使用 trajectory 的 reward 计算，早于
advantage 后处理、MIS/rejection、Datum
shuffle、optimizer 拆批和尾批 padding。拒绝部分 token 不会触发剩余 advantage
的重新 whitening。正整数 `algorithm.optim.batch_size` 用于选择固定大小的
optimizer batch；已交付批次超过该大小时，在 whitening 后执行打乱、拆分和尾批补齐。
随后，每个 optimizer batch 单独进行 token-mean loss 除法归一化。
修改 `mini_batch_size` 只改变请求大小，不改变
whitening 统计量或 optimizer 更新次数。示例使用 `batch_size: null`，将一次
交付的批次放入一次更新；详见[拆批说明](../11-configuration-reference/trainer-config.md#optimizer-batches-and-forwardbackward-requests)。

<a id="kl-modes"></a>

## KL 模式

这两个 estimator 不计算 KL，也不读取 actor/reference logprob。
`kl.mode: in_loss` 添加独立 loss 项，需要配置 `models.reference`；
`kl.mode: in_reward` 尚未实现，trainer 初始化时会拒绝它。

<a id="input-production-and-consumption"></a>

## 从 trajectory 到 advantage 的数据流

两种 REINFORCE++ 版本直接读取 `trajectories`，公共 estimator 接口的
`inputs` 参数传空字典：

```text
RemoteApiGrpoTrainer._prepare_advantage_inputs()
  返回 {}
    ↓
RemoteApiTrainer.train()
  按配置为训练重算 actor logprob；清除后处理缓存
    ↓
ReinforcePlusPlusEstimator.estimate(trajectories, {}, metrics)
  从 estimator 实例读取 gamma 和 baseline
    ↓
compute_reinforce_plus_plus_advantages(trajectories, baseline=..., gamma=...)
  计算 return、全 step whitening，写回各 sequence 的 token_advantages
    ↓
可选工具错误后处理 → 计算 MIS 权重和拒绝 mask
    ↓
convert_trajectories_to_datums()
  应用 MIS 得到最终 loss_mask；跳过全零 mask；生成 Datum 和对应来源记录
    ↓
拆分 optimizer batch
```

[trainer](../../../../northrl/trainer/remote_api_grpo_trainer.py) 不为这两个 estimator
准备额外输入。按配置执行的 actor 重算和 reference 评分服务于训练 loss，不会改变
estimator 的输入。[estimator](../../../../northrl/algorithm/advantage/reinforce_plus_plus.py)
将以下数据交给 `compute_reinforce_plus_plus_advantages()`：

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| outcome reward | 每条完整轨迹的 `reward` | 终点 return 或组内 baseline |
| 分组信息 | 完整轨迹的 `extra_fields` 中的 `model_tag`、`group_id` | 组内 baseline 和校验 |
| token 与 mask | 原 trajectory 或其 segment 字典 | 选择有效 response 位置 |
| `gamma`、`baseline` | 由 estimator 配置构造的实例 | 折扣和版本选择 |

数学函数直接遍历成功且未被跳过的完整轨迹中的原 sequence，无须额外传递 sequence
或概率载荷。它把结果写到原 sequence 的 `token_advantages`，自身返回 `None`；
`estimate()` 随后记录指标并返回原 `list[Trajectory]`，供后处理和训练数据转换继续使用。
