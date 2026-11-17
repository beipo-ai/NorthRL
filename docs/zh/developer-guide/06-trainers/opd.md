[English](../../../en/developer-guide/06-trainers/opd.md) | 简体中文 | [文档索引](../../index.md)

# On-policy distillation（OPD）

[返回算法组件总览](algorithm-components.md)

OPD 使用学生模型自身生成的 rollout，以教师 logprob 提供逐 token 学习信号。
设置 `algorithm.type: opd` 和 `algorithm.advantage.estimator.mode: opd`。
它生成仅含 response 位置的 `token_advantages`，不计算奖励比较组的标量。

<a id="advantage-definition"></a>

## Advantage 定义

对每个 segment，记 `t` 为 target 位置（`tokens[1:]`），`m_t` 为对应的 target
loss mask，`c = kl_penalty_coef`，`gamma = kl_discount_factor`：

```text
d_t = -c * (student_logprob_t - teacher_logprob_t) * m_t
A_t = d_t + gamma * A_(t+1)          # terminal future advantage is zero
```

`gamma: 0` 时，每个 response advantage 就是该位置的 `d_t`。gamma 为正时，
折扣累加沿所有 target 位置推进，包括被 mask 的 context。这些位置的局部信号
为零，但仍计入折扣距离。递推在每个 segment 内重新开始，不跨 segment 传递
后续 advantage。只将 `m_t == 1` 的位置按 response 顺序写入 `token_advantages`。

<a id="configuration"></a>

## 配置

完整的 [Weaver OPD recipe](../../../../recipe/on_policy_distill/weaver_opd.yaml)
在 `models.actor` 声明学生模型，在 `models.teachers` 声明教师。以下是 estimator
和 loss 的配置片段，不是完整 recipe：

```yaml
algorithm:
  type: opd
  advantage:
    estimator:
      mode: opd
      params:
        kl_penalty_coef: 1.0
        kl_discount_factor: 0.0
    postprocessing:
      mode: null
  loss:
    mode: importance_sampling
    reduction:
      mode: token_mean
  opd:
    teacher_logprob_source: inference_service
    teacher_service_timeout_s: 1200.0
  sampling_logprob_source: inference
  kl:
    mode: null
  optim:
    lr: 1.0e-4
```

两个 estimator 参数的默认值分别为 `kl_penalty_coef: 1.0` 和
`kl_discount_factor: 0.0`。系数设为零时跳过教师评分，并生成零 advantage。
教师服务设置属于 `algorithm.opd`；`teacher_logprob_source` 可选
`inference_service` 或 `training_service`。`algorithm.kl` 下的 reference-policy
KL 与 OPD 教师信号相互独立。

<a id="data-and-scope"></a>

## 数据与计算范围

OPD 要求 `models.teachers` 非空。当前评分路径使用列表中的第一个教师，不按
数据集选择不同教师。与 RLOO 不同，OPD 不要求重复生成的比较组；示例使用
`data.train.rollout_repeat_n: 1`。

[`RemoteApiOpdTrainer`](../../../../northrl/trainer/remote_api_opd_trainer.py)
为 [`OPDEstimator`](../../../../northrl/algorithm/advantage/opd.py) 准备
`OPDInputs` 字典：

| 字段 | 系数非零时需要的值 |
| --- | --- |
| `sequences` | 按本轮遍历顺序排列的原始训练 sequence 对象。 |
| `student_logprobs` | 在 actor 重算前保留的 rollout 学生评分数组。 |
| `teacher_logprobs` | 按相同 sequence 顺序排列的教师评分数组。 |

每行概率都按完整 token sequence 对齐，包括位置零的占位；estimator 在计算
advantage 前将其对齐到 target。缺失字段、行数不匹配或 sequence 对象身份变化
都会报错。某行概率缺失或对齐失败时，该 sequence 的 advantage 置零。

recipe 使用 `sampling_logprob_source: inference`。如果启用 actor 重算，OPD
仍使用保留的 rollout 学生评分计算 advantage，重算后的 `old_log_probs` 则作为
loss 分母。重算替换字段时，保留原数组引用即可；如果生产者会原地修改数组，
准备阶段必须保存快照。教师调用及其失败处理由 trainer 负责，estimator 只计算
advantage。可选的[工具错误优势重分配](advantage-processing.md)在此之后执行。
