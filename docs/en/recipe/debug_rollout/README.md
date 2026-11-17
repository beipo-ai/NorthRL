English | [简体中文](../../../zh/recipe/debug_rollout/README.md) | [Documentation index](../../index.md)

<a id="reward-rollout-debug"></a>

# Reward Rollout Debugging

Use the 8 built-in samples to inspect generation, chat templates, and rewards. Each sample is generated once;
the command exits after one batch without performing training updates.

<a id="运行"></a>

## Run

Prepare the NorthRL Python environment following the repository installation instructions, then run from the **repository root**:

```bash
cp -n recipe/debug_rollout/env.sh.example recipe/debug_rollout/env.sh
```

Fill in the Weaver, LLM judge, and E2B settings in `env.sh`. This file is ignored by Git.
Then run:

```bash
source recipe/debug_rollout/env.sh
python -m northrl.debug_rollout --config recipe/debug_rollout/rewards.yaml
```

`debug_rollout` does not load `env.sh` automatically; source it manually before running.

The default model is `Qwen/Qwen3-30B-A3B-Base`. A base-model sampling client is created directly,
so no sampling checkpoint is needed. The tokenizer is loaded from the matching Hugging Face repository by default;
you can also specify a local tokenizer directory containing a chat template:

```bash
python -m northrl.debug_rollout --config recipe/debug_rollout/rewards.yaml \
  models.actor.tokenizer=/path/to/tokenizer
```

To change models, override `models.actor.name` and ensure the tokenizer matches the model.
The maximum generation length is 4096 tokens, configurable through `rollout.sampling.max_tokens`.

<a id="样本与评测"></a>

## Samples and Evaluation

[samples.jsonl](../../../../recipe/debug_rollout/samples.jsonl) includes all inputs. Each sample selects its evaluator through the top-level `reward_fn`.

| Order | reward_fn | Content | Dependencies |
| --- | --- | --- | --- |
| 1 | math_rule_llm_answer_only | Chinese math problem; user message | MathD/SymPy; LLM if rules do not accept the answer |
| 2 | math_rule_llm_answer_only | Math problem; system + user messages | MathD/SymPy; LLM if rules do not accept the answer |
| 3 | math_rule_llm_answer_only | Geometry problem extracted from mix.jsonl | MathD/SymPy; LLM if rules do not accept the answer |
| 4 | math_rule_llm_answer_only | Function range problem extracted from mix.jsonl | MathD/SymPy; LLM if rules do not accept the answer |
| 5 | verinstruct | Movie title list; no hard functions | LLM judge |
| 6 | verinstruct | Resume document headings; 4 hard checks | E2B + LLM judge |
| 7 | rule_dr_sci | Biology multiple-choice problem extracted from mix.jsonl; label B | Local MathD/SymPy |
| 8 | rule_dr_sci | Physics numerical problem extracted from mix.jsonl; label 0.134 | Local MathD/SymPy |

The 4 math samples call an LLM only when local rules do not accept the answer. In that case, and for samples 5 and 6,
set `LLM_JUDGE_URL`, `LLM_JUDGE_MODEL`, and, if needed, `LLM_JUDGE_API_KEY`.
When `hard_functions` are present, VerInstruct first runs Python checks through E2B and returns 0 immediately on failure.
If hard checks pass or none are provided, it calls the RECAST LLM judge for a 0/1 score.

Sample 6 also requires `E2B_API_KEY`. `E2B_TEMPLATE` can specify a template name or ID; the template must contain an executable
`python3`. Leave it empty to use the service's default template. For a custom E2B service, also set
`E2B_API_URL` and `E2B_DOMAIN`.

Samples 7 and 8 use the entirely local `rule_dr_sci`: it reads only the first correctly closed, case-sensitive
`<answer>...</answer>` tag, temporarily wraps the content in `\boxed{...}`, and then uses
MathD/SymPy to compare mathematical equivalence with the top-level `label`. Missing or empty answer tags and incorrect answers receive
0; equivalent answers receive 1. These samples require no additional environment variables, LLM judge, or sandbox.

<a id="查看结果"></a>

## Inspect Results

Each run saves JSON files under:

```text
outputs/debug_rollout/rollout-<id>/rollout_traces/*.json
```

Change the output directory with `runtime.debug.dump_dir=/path/to/output`.

- `prompt_str`: Input after rendering the chat template.
- `response`: Model response.
- `reward`, `score`: Final reward, evaluation result, and failure reasons.
- `finish_reason`: Reason generation ended, including reaching the generation length limit.
- `is_truncated`: Whether the sequence was truncated due to `max_sequence_length`.
- `rollout_result.northrl_train`: Prompt/response tokens and log probabilities.

If generation or the reward service fails, that sample has no complete trace. See terminal logs for details; remaining samples continue.
