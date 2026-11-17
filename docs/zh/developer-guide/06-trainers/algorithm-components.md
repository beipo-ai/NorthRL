[English](../../../en/developer-guide/06-trainers/algorithm-components.md) | 简体中文 | [文档索引](../../index.md)

# 算法组件

本文介绍公共组件模型和训练数据流。具体算法的公式、可运行配置与支持组合，
由各自的[算法指南](#algorithm-guides)说明。

<a id="canonical-configuration"></a>

## 组件职责

trainer 根据已验证的配置创建组件，并按顺序调用。算法名称不一定对应独立的
trainer：advantage 估计、后处理、loss 与 reduction 分别选择。

| 配置 | 职责 |
| --- | --- |
| `algorithm.type` | 选择 trainer 类型及其输入准备流程。 |
| `algorithm.advantage.estimator` | 根据 trajectory 和额外评分输入计算 advantage。 |
| `algorithm.advantage.postprocessing` | 可选地变换估计后的 advantage。 |
| `algorithm.loss` | 定义优化目标。 |
| `algorithm.loss.reduction` | 定义如何把各项贡献汇总为 optimizer batch 的标量 loss。 |
| `algorithm.kl` | 配置 reference policy 正则。 |
| `algorithm.optim` | 配置 optimizer 更新与请求分批。 |

estimator、postprocessing、loss、reduction 和 KL 使用 `mode` 与 `params`
映射选择组件及其参数。配置 schema 验证完整组合；可以分别选择不代表能够任意
组合。省略后处理或将其 mode 设为 `null` 时关闭后处理。
设置 `algorithm.kl.mode: null` 时关闭 reference KL。

模型声明放在 `models` 下。后端设置、optimizer 分批和训练控制见
[训练器配置](../11-configuration-reference/trainer-config.md)。

<a id="step-data-flow-and-alignment"></a>

## Step 数据流

worker 将完整 trajectory 发布到 pool。trainer 领取一批数据，作为一个训练
step 传给 `train(trajectories)`：

```text
本次交付的 list[Trajectory]
  -> 准备 trajectory 及所需的模型评分
  -> 在完整的已准备 step 上估计 advantage
  -> 可选的 advantage 后处理
  -> 应用配置的采样修正，将 sequence 转换为 Datum
  -> 组成 optimizer batch，应用所选 loss 与 reduction
  -> 拆分服务请求，累积梯度，执行 optimizer 更新
```

advantage 估计的范围是本次 step 准备后的完整列表。后续拆分 optimizer batch
或服务请求，不会按更小的子集重新估计。如何定义比较组、如何处理不可用 rollout，
由所选 estimator 决定。

一次 rollout 的结果对应一条完整轨迹（`Trajectory`）。代码有时将它称为
`parent`，因为它保存了这次 rollout 的各个训练 segment。

| 术语 | 含义 |
| --- | --- |
| prompt group（比较组） | 同一道题在一个比较组中采样得到的多次 rollout。 |
| 完整轨迹（trajectory） | 一次 rollout 的结果，包含该次执行的 reward 和 status。 |
| segment（段） | 这条轨迹内部保存的一段 token 序列。 |
| Datum | 一个保留的训练序列转换后的训练服务输入。 |

例如，同一道题采样两次，就得到同组的两条完整轨迹。它们分别保存两个和一个
可训练 segment 时，转换后是三个 Datum，完整轨迹仍然是两条。
一次包含多轮对话和工具调用的 rollout 仍对应一条完整轨迹；一次工具调用
不一定产生新 segment。未分段的轨迹直接保存 tokens。

按完整轨迹计算的 loss 需要保留各 segment 与原 rollout 的归属关系。
转换、过滤和分批行为见[远程 API 训练器](remote-api-trainers.md)，
数据模型见[轨迹语义](../04-trajectory-pool/trajectory-semantics.md)。

<a id="data-passed-between-components"></a>

## 公共数据接口

| 值 | 含义 |
| --- | --- |
| `trajectories` | 包含完整轨迹的 `list[Trajectory]`。 |
| estimator 的 `inputs` | trainer 准备的额外数据；不需要时为 `{}`。 |
| 完整轨迹数据 | reward、status 和 rollout/group 身份。 |
| sequence 数据 | 普通 trajectory 或单个 segment 的 tokens、mask、logprob 等字段。 |
| 标量 advantage | 完整轨迹上的一个 `advantage`，由其各训练 sequence 共享。 |
| token advantage | 每个 sequence 上仅包含 response 位置的 `token_advantages`。 |
| 组件返回值 | 在原对象上保存计算字段的 trajectory 列表；列表可重新排序。 |
| `metrics` | 原地更新的共享字典。 |

estimator 决定输出 advantage 的粒度。可选后处理消费该输出；未启用后处理时
保留估计结果。转换训练数据时，显式 token advantage 按 target 位置对齐；
没有时把完整轨迹的标量广播到 response token。所选 loss 决定接受哪种表示。

组件的公共调用接口为：

```python
trajectories = estimator.estimate(trajectories, inputs, metrics)
trajectories = postprocessor.process(trajectories, metrics)  # when enabled
```

这些组件基于传入数据计算。额外输入准备、外部服务调用、请求拆分和 optimizer
更新由 trainer 负责。

<a id="loss-and-reduction"></a>

## Loss 与 reduction

loss 定义优化目标中的各项贡献，reduction 定义它们的权重，以及如何汇总成一个
optimizer batch 的标量。token、Datum、完整 trajectory 和 prompt group 是不同
的聚合单位；将 trajectory 拆成 segment，不应隐式改变目标所选择的聚合单位。

原生 loss 通过服务端 loss 接口执行，自定义 loss 通过 callback 执行。
adapter 将配置转换成相应协议，并在协议需要的位置处理归一化。
在同一 optimizer batch 内拆分请求，不会产生新的平均范围。

### 自定义 loss callback

使用 `loss.mode: custom` 选择 callback，其 `loss.params` 块为：

```yaml
name: my_package.losses:build_loss
params: {}
base_loss_fn: importance_sampling
clip: {}
```

`name` 可以是已注册的别名或可导入的 `module:factory`；内层 `params` 映射传给
factory。`base_loss_fn` 和外层 `clip` 是 adapter 字段，不作为 callback 参数。
`module_path` 不会加载 recipe 相对路径的 Python 文件。自定义 loss 要求 entropy
为零并关闭 submit-ahead。按名称选择服务端原生 loss 则使用 `loss.mode: native`
和 `params.name`。

<a id="where-to-read-the-code"></a>

## 代码与扩展入口

| 文件 | 职责 |
| --- | --- |
| [`trainer/remote_api_trainer.py`](../../../../northrl/trainer/remote_api_trainer.py) | 准备输入、调用组件、转换数据并组织训练。 |
| [`algorithm/advantage/types.py`](../../../../northrl/algorithm/advantage/types.py) | estimator 和 postprocessor 接口。 |
| [`algorithm/advantage/factory.py`](../../../../northrl/algorithm/advantage/factory.py) | 根据已验证的配置创建组件。 |
| [`weaver/loss_adapter.py`](../../../../northrl/weaver/loss_adapter.py) | 将 loss 配置和数据适配到服务协议。 |
| [`algorithm/reductions.py`](../../../../northrl/algorithm/reductions.py) | 计算公共归一化分母。 |

<a id="adding-an-algorithm-and-python-api-changes"></a>

新增 estimator 或 postprocessor 时，实现对应接口、增加 factory 分支与配置校验，
并在独立指南中说明行为。需要额外外部数据时扩展 trainer 输入准备。测试输出对齐、
支持组合，以及请求拆分不改变计算结果。最后将指南加入下方目录。

<a id="supported-configurations"></a>
<a id="algorithm-guides"></a>

## 算法指南

- [GRPO](grpo.md)
- [RLOO](rloo.md)
- [REINFORCE++ 与 REINFORCE++-baseline](reinforce-plus-plus.md)
- [OPD](opd.md)
- [Dr.GRPO](dr-grpo.md)
- [CISPO](cispo.md)
- [GSPO](gspo.md)

<a id="future-integrations"></a>

## 相关指南

- [Advantage 后处理](advantage-processing.md)
- [Rollout 与训练不匹配修正](mis.md)
- [远程 API 训练器](remote-api-trainers.md)
- [训练器配置](../11-configuration-reference/trainer-config.md)
