[English](../../../en/recipe/debug_rollout/README.md) | 简体中文 | [文档目录](../../index.md)

<a id="reward-rollout-debug"></a>

# Reward rollout debug

用内置的 8 条样本检查生成结果、chat template 和 reward。每条样本生成一次，
完成一个 batch 后退出，不执行训练更新。

<a id="运行"></a>

## 运行

先按仓库安装说明准备好 NorthRL Python 环境，然后在**仓库根目录**执行：

```bash
cp -n recipe/debug_rollout/env.sh.example recipe/debug_rollout/env.sh
```

填写 `env.sh` 中的 Weaver、LLM judge 和 E2B 配置。该文件已加入 Git 忽略规则。
随后运行：

```bash
source recipe/debug_rollout/env.sh
python -m northrl.debug_rollout --config recipe/debug_rollout/rewards.yaml
```

`debug_rollout` 不会自动加载 `env.sh`，运行前需要手动 source。

默认使用 `Qwen/Qwen3-30B-A3B-Base`，直接创建 base-model sampling client，
无需提供 sampling checkpoint。Tokenizer 默认从同名 Hugging Face 仓库加载；
也可以指定包含 chat template 的本地 tokenizer 目录：

```bash
python -m northrl.debug_rollout --config recipe/debug_rollout/rewards.yaml \
  models.actor.tokenizer=/path/to/tokenizer
```

更换模型时，可通过 `models.actor.name` 覆盖默认模型，并确保 tokenizer 与模型匹配。
最大生成长度为 4096 tokens，可用 `rollout.sampling.max_tokens` 覆盖。

<a id="样本与评测"></a>

## 样本与评测

[samples.jsonl](../../../../recipe/debug_rollout/samples.jsonl) 自带全部输入数据，每条样本通过顶层 `reward_fn` 选择评测函数。

| 顺序 | reward_fn | 内容 | 依赖 |
| --- | --- | --- | --- |
| 1 | math_rule_llm_answer_only | 中文数学题；user 消息 | MathD/SymPy，规则未判对时 LLM |
| 2 | math_rule_llm_answer_only | 数学题；system + user 消息 | MathD/SymPy，规则未判对时 LLM |
| 3 | math_rule_llm_answer_only | 从 mix.jsonl 提取的几何题 | MathD/SymPy，规则未判对时 LLM |
| 4 | math_rule_llm_answer_only | 从 mix.jsonl 提取的函数值域题 | MathD/SymPy，规则未判对时 LLM |
| 5 | verinstruct | 电影标题列表；无 hard functions | LLM judge |
| 6 | verinstruct | 简历文档标题；4 个 hard checks | E2B + LLM judge |
| 7 | rule_dr_sci | 从 mix.jsonl 提取的生物选择题；label B | 本地 MathD/SymPy |
| 8 | rule_dr_sci | 从 mix.jsonl 提取的物理数值题；label 0.134 | 本地 MathD/SymPy |

4 条数学样本只在本地规则未判对时调用 LLM。此时以及第 5、6 条运行时需要
`LLM_JUDGE_URL`、`LLM_JUDGE_MODEL`，必要时设置 `LLM_JUDGE_API_KEY`。
VerInstruct 在有 `hard_functions` 时先通过 E2B 执行 Python 检查，失败直接给 0；
硬检查通过或没有硬检查时，再调用 RECAST LLM judge 给出 0/1 评分。

第 6 条还需要 `E2B_API_KEY`。`E2B_TEMPLATE` 可指定模板名称或 ID，模板必须包含可执行的
`python3`；留空时使用服务的默认模板。使用自定义 E2B 服务时，还需设置
`E2B_API_URL` 和 `E2B_DOMAIN`。

第 7、8 条使用纯本地 `rule_dr_sci`：只读取第一个正确闭合、大小写敏感的
`<answer>...</answer>`，把其中内容临时包装成 `\boxed{...}`，再通过
MathD/SymPy 与顶层 `label` 做数学等价比较。缺少或空 answer tag、错误答案均给
0，等价答案给 1。这两条同样不需要额外环境变量、LLM judge 或沙箱。

<a id="查看结果"></a>

## 查看结果

每次运行的 JSON 文件保存在：

```text
outputs/debug_rollout/rollout-<id>/rollout_traces/*.json
```

可通过 `runtime.debug.dump_dir=/path/to/output` 更改输出目录。

- `prompt_str`：渲染 chat template 后的输入。
- `response`：模型回答。
- `reward`、`score`：最终 reward、评测结果及失败原因。
- `finish_reason`：生成结束原因，包括是否达到生成长度限制。
- `is_truncated`：是否因 `max_sequence_length` 发生序列截断。
- `rollout_result.northrl_train`：prompt/response tokens 和 log probabilities。

生成或 reward 服务报错时，该条不会得到完整 trace；具体异常见终端日志，后面的样本继续执行。
