English | [简体中文](../../../zh/developer-guide/11-configuration-reference/rollout-config.md) | [Documentation index](../../index.md)

# Rollout Worker Configuration

Configuration options for rollout workers.

For E2B rollouts, set `rollout.sampling.reasoning_effort: high` to override the
agent's effort on every request. The relay writes OpenAI `reasoning_effort` or
Anthropic `output_config.effort`; supported values depend on the gateway/model.
Omitted or `null` preserves the agent's setting. Validation uses the independent
`validate.sampling.reasoning_effort` setting. Other rollout types reject this option.

## Sampling Mask Retention

`rollout.keep_sampling_mask` defaults to `false` and directly controls Weaver's
`return_sampling_mask` request flag. Set it to `true` to request masks and retain
them in rollout trajectories for training. Sampling filters (`top_p`, `top_k`,
and `min_p`) do not enable mask retention automatically.

```yaml
rollout:
  keep_sampling_mask: true
  sampling:
    top_p: 0.95
```

Full fine-tuning with this option enabled requires a Weaver SDK that supports
`sampling_mask_transport="ref"`. Keep this option disabled when using
`algorithm.mis.enabled` or `algorithm.mis.get_mismatch_metrics`, since MIS does
not support sampling masks.

## Single-turn Reward Selection

For dataset fields, JSONL/Parquet examples, and mixed per-row rewards, see
[Single-turn Reward Data](../03-data-loader/data-loader.md#single-turn-reward-data).

The single-turn rollout worker evaluates each generated response immediately. It uses the
sample's top-level `reward_fn` when present; otherwise it uses `rollout.reward_fn`.
A missing or null data value uses the configured default. An unknown name or an
empty string raises an error instead of falling back. Pandas missing values in
file-backed datasets are treated as absent. CSV parsing retains pandas' default
missing-value rules, including empty cells.

```yaml
rollout:
  num_workers: 128
  reward_fn: math_answer_only
  type: single_turn
```

```json
{"prompt": [{"role": "user", "content": "What is 40 + 2?"}], "label": "42", "reward_fn": "math_with_format"}
```

This sample uses `math_with_format` even though the configured default is `math_answer_only`.
`label` contains the reference answer; existing `ground_truth` and
`reward_model.ground_truth` data remain supported.

| Reward name | Existing implementation |
| --- | --- |
| `math_with_format` | Original math rule scoring, including its 0.1 format reward |
| `math_answer_only` | Full-response boxed-answer math scoring, binary 0/1 |
| `math_llm` | LLM judgment of the answer after the thinking block |
| `math_rule_llm` | Rule scoring first, then LLM judgment when needed |
| `math_rule_llm_answer_only` | Last-boxed MathD/SymPy scoring, then a bp-compatible binary LLM fallback |
| `math_boxed_llm` | LLM judgment of the extracted boxed answer |
| `tinker_aime_scalar_match` | Existing AIME scalar comparison |
| `rule_nemotron_mcqa` | Local extraction and exact comparison of a Nemotron multiple-choice answer, binary 0/1 |
| `rule_dr_sci` | Local `<answer>...</answer>` extraction and MathD/SymPy equivalence scoring, binary 0/1 |
| `rule_reasoning_gym` | Local ReasoningGym task scoring with audited ARC and Rush Hour fixes; binary 0/1 except continuous `boxnet` |
| `reasoning360` | Routes seven Reasoning360 sources to four local rule families or a STEM LLM judge, binary 0/1 |
| `synlogic` | Routes 35 SynLogic games to bundled local rule verifiers, binary 0/1 |
| `ifeval` | Local evaluation of structured IFEvalG constraints, all-of binary 0/1 |
| `if_verify` | Runs the row's `evaluate(response)` Python checker in an E2B sandbox, binary 0/1 |
| `code_e2b` | E2B evaluation of prepared code labels; all cases must pass, binary 0/1 |
| `recast` | LLM instruction-following judgment of the complete response, binary 0/1 |
| `verinstruct` | Sandboxed hard-constraint checks followed by LLM instruction-following judgment, binary 0/1 |

The four LLM-capable modes (`math_llm`, `math_rule_llm`,
`math_rule_llm_answer_only`, and `math_boxed_llm`)
require `LLM_JUDGE_URL` and `LLM_JUDGE_MODEL` in the rollout process environment.
`LLM_JUDGE_API_KEY` is optional. `math_llm` also requires thinking start/end markers
in the rendered prompt plus response; `math_rule_llm` inherits this requirement
when its rule check does not pass. `math_boxed_llm` judges the extracted boxed
answer without that thinking-marker requirement. `math_rule_llm_answer_only`
selects the region after the later of the final `</think>` or `###Response`
marker, or the complete response when neither marker exists. It first compares
the final boxed answer with `label` using MathD/SymPy. A rule-negative result,
missing boxed answer, or rule exception falls back to the LLM judge. SymPy runs
directly in the reward coroutine without a soft timeout, matching `math_rule_llm`.
The fallback uses a `[[1]]`/`[[0]]` verdict and all
successful judging paths return a binary reward. If that fallback cannot be
initialized, requested, or parsed after retries, it emits a warning and returns
0; the other LLM modes retain their existing failure behavior.

SimpleRolloutWorker and E2B use their own evaluation paths.

### Nemotron MCQA Reward

`rule_nemotron_mcqa` is a synchronous local reward and requires a non-empty
string `label`. It first uses the last complete, case-sensitive
`<answer>...</answer>` block. When that format is absent, it uses the text after
the last case-insensitive `Answer:` marker. If the selected segment contains a
`\boxed{...}` command, the first boxed body in that segment is used. Edge
punctuation and backslashes are normalized before a case-insensitive exact
comparison with `label`.

Missing answer markers, an empty extracted answer, a wrong choice, or malformed
answer formatting receives 0. A matching choice receives 1. The reward does not
call an LLM, sandbox, SymPy, or external service and does not read
`metadata.style` or `metadata.rm_type`; select it explicitly with the row's
`"reward_fn": "rule_nemotron_mcqa"` or with `rollout.reward_fn`.

### ReasoningGym Reward

`rule_reasoning_gym` is a synchronous local reward backed by
`reasoning-gym==0.1.25`. It reads the task name from
`metadata.metadata.source_dataset` and passes the extracted final answer plus the
complete outer `metadata` entry to that task's registered scorer. The top-level
`label` is not used and may be null. Responses prefer the final balanced
`\boxed{...}` answer, then the first `<answer>...</answer>` block, and otherwise
use the complete response.

The `arc_agi` and `rearc` tasks use an exact integer-grid comparison, and
`rush_hour` uses a local move simulator, matching bp-rl-judge's fixes for bugs in
the corresponding library scorers. Other tasks use the pinned library scorer.
Scores at least `0.999` become reward 1 and smaller scores become 0; `boxnet` is
the sole exception and preserves its continuous raw score. This reward needs no
LLM, sandbox, API key, or recipe parameter. The pinned package includes a
compiled `pycosat` dependency, so the Driver image must install it in an
environment with a C build tool or from a prebuilt Python 3.12 wheel.

### Reasoning360 Reward

`reasoning360` is one async public reward with an internal, exact
`metadata.data_source` route:

| Data source | Verifier |
| --- | --- |
| `logic__graph_logical_dataset` | Last answer-tag alphabetic value, case-insensitive exact comparison |
| `logic__zebra_puzzle_dataset` | Safely parsed table cells, all target cells must match |
| `logic__ordering_puzzle_dataset` | Safely parsed ordered string list, exact order required |
| `simulation__barc` | Strict two-dimensional integer-grid comparison |
| `simulation__arcagi1` | Strict two-dimensional integer-grid comparison |
| `simulation__arcagi2` | Strict two-dimensional integer-grid comparison |
| `stem_web` | Extracted final answer judged by an OpenAI-compatible LLM |

The first six source names use four synchronous local algorithms. `stem_web`
requires a non-empty string `label` and `metadata.question`; it accepts the first
`<answer>...</answer>` region or the final balanced `\boxed{...}`/`\fbox{...}`
answer. It then asks the judge only whether that answer is equivalent to the
reference and treats `Final Decision: Yes` as correct.

Configure the STEM path with the existing shared variables:

```bash
export LLM_JUDGE_URL=https://your-openai-compatible-service/v1
export LLM_JUDGE_MODEL=your-judge-model
export LLM_JUDGE_API_KEY=your-key
export LLM_JUDGE_TIMEOUT_S=180
```

No Reasoning360-specific recipe parameter or new Python package is required.
Local parsing failures and wrong answers return 0. Missing or invalid dataset
fields and unknown source names raise an error. Exhausted STEM judge requests
emit a warning and return 0.
Every Reasoning360 path is finally converted to a strict binary reward.

### SynLogic Reward

`synlogic` is a synchronous local reward covering all 35 game names in the
canonical SynLogic hard dataset. A row selects the public reward with its
top-level `reward_fn`, then selects the game-specific verifier with the exact
`metadata.data_source` value. `metadata.game_data_str` must be a serialized JSON
object containing the puzzle question, answer, difficulty, and game-specific
metadata. The top-level `label` is unused and may be empty.

NorthRL first tries the final balanced `\boxed{...}` answer, then the first
`<answer>...</answer>` region, or otherwise the full response. When that
representation does not pass, the verifier also receives the original full
response. This accommodates both bare-answer rules and game rules that parse
their own code block or answer wrapper. A finite raw score of at least `0.999`
becomes reward 1; all other valid scores become 0. There is no continuous-score
exception.

All 35 verifiers and their exact registry are bundled under
`northrl.rewards._synlogic`; production code does not load a GPFS reward hub or
call bp-rl-judge. Model-controlled literals and arithmetic expressions are
parsed with bounded, allow-listed AST evaluators rather than Python `eval`.
The `operation` game uses `math-verify==0.9.0`, which is part of NorthRL's core
dependencies. SynLogic requires no LLM, E2B, API key, external data file, or
reward-specific recipe parameter.

### IFEval Reward

`ifeval` is a synchronous local reward for mechanically checkable instruction
constraints. It reads `metadata.instructions`, where every item has a verifier
`id` and its explicit `kwargs`. The vendored IFEvalG registry provides 54 rule
types covering keywords, counts, language, punctuation, length, and structured
formats. Every constraint must pass for reward 1; any failed or unknown verifier
gives reward 0. `label` is unused.

Before checking, NorthRL keeps the text after the last `</think>`, or after the last
`###Response` when no thinking marker exists, then removes trailing whitespace
and repeated EOS tokens. This prevents reasoning text from affecting constraints
such as no-comma, exact word count, and JSON formatting.

The Python environment needs `nltk==3.9.4` and `langdetect==1.0.9`, which are in
the `core` dependency group. NLTK's `punkt` and `punkt_tab` data are separate
runtime resources and must be prepared once in a persistent directory:

```bash
export NLTK_DATA=/path/to/persistent/nltk_data
python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab
```

Keep `NLTK_DATA` exported in the shell running NorthRL. IFEval requires no API key, LLM judge, sandbox,
or reward-specific recipe parameter. Select it with a row's
`"reward_fn": "ifeval"` or set `rollout.reward_fn: ifeval` as the default.

### IfVerify Reward

`if_verify` is an async binary reward for rows that provide their own Python
checker in `metadata.checker`. The source must define a synchronous
`evaluate(response)` function. NorthRL runs it only in a fresh E2B sandbox; the
dataset source never executes in the driver process. A truthy return gives 1,
and a falsy return gives 0. `label` is unused.

Before checking, NorthRL keeps the text after the last `</think>`, or after the
last `###Response` when no thinking marker exists, then removes trailing
whitespace and repeated `<|im_end|>`, `<|endoftext|>`, and `<|eot_id|>` tokens.
The normalized answer is JSON-serialized into the sandbox harness so quotes,
newlines, and code-like text in the model response remain data.

Configure E2B in the process that runs NorthRL:

```bash
export E2B_API_KEY=your-e2b-api-key
# Optional: E2B_TEMPLATE=your-python-template
# Optional for a custom service: E2B_API_URL and E2B_DOMAIN
```

The lifecycle and execution limits match the sandbox stage of `verinstruct`:
internet access is disabled, the command runs as
`python3 -I /tmp/if_verify_check.py`, the command timeout is 10 seconds, and
combined stdout/stderr is capped at 2,000,000 UTF-8 bytes. The harness also
applies CPU, address-space, file-size, and process limits. The sandbox is always
given a bounded cleanup attempt.

A missing or blank checker gives zero with reason `no_checker`. A falsy checker
gives `fail_checker`; checker syntax/runtime errors or malformed output give
`checker_error`. A timeout during sandbox creation, upload, or execution retries
the full check once in a fresh sandbox, after cleanup of the first attempt. Each
attempt has its own 60-second total deadline and 10-second command timeout.
If the retry completes, its verdict determines the reward; a second timeout
gives zero with reason `timeout_after_retry`. Invalid input types and non-timeout
E2B service/configuration failures raise. Select the reward with a
row's `"reward_fn": "if_verify"` or set `rollout.reward_fn: if_verify` as the
default. No reward-specific recipe parameter or LLM judge is required.

### Code E2B Reward

For prepared LeetCode, LiveCodeBench, and TACO datasets, set `reward_fn: code_e2b`
per row or `rollout.reward_fn: code_e2b` as the default. Each row's `label` contains
the complete judge request:

```jsonl
{"prompt": [{"role": "user", "content": "Read two integers and print their sum. Return Python code."}], "reward_fn": "code_e2b", "label": {"language": "python", "input_mode": "stdio", "cases": [{"stdin": "2 3\n", "expected_stdout": "5\n"}, {"stdin": "-2 2\n", "expected_stdout": "0\n"}], "run_timeout": 2, "compile_timeout": 10}}
```

Required label fields are `language`, `input_mode` (`stdio` or `function_call`),
and non-empty `cases`. Optional fields include `fn_name`, `generated_checker`,
`compile_timeout`, `run_timeout`, `checker_timeout`, and `code_template` for
wrappers or embedded tests. Templates and function-call mode support Python only.

The answer uses the last fenced code block after the final `</think>`, or the
remaining text if unfenced. A `code_template` must contain `__MODEL_CODE__`,
which is replaced with `repr(code)`; other judge fields pass through unchanged.
The old `leetcode` reward was removed. Convert legacy `metadata.test_cases` or
`metadata.test_cases_path` data to prepared labels.

Set these variables in the rollout process environment:

| Variable | Meaning |
| --- | --- |
| `E2B_API_KEY` | Required E2B credential. |
| `E2B_TEMPLATE` | Required, non-empty ID or alias of a prepared code-judge template accessible to your E2B account. |
| `E2B_DOMAIN` | Optional E2B service domain. |
| `E2B_REQUEST_TIMEOUT` | Positive SDK request timeout in seconds; defaults to 60. |
| `E2B_UPLOAD_REQUEST_TIMEOUT` | Positive base upload timeout in seconds; defaults to the larger of 300 and `E2B_REQUEST_TIMEOUT`. Each upload retry adds 50% of this base, up to 3 times the base. Enforced locally and passed to the SDK. |
| `E2B_COMMAND_TIMEOUT` | Optional positive per-batch command timeout in seconds; otherwise estimated from case count and compile/run/checker budgets, reserving upload and cleanup time within the initial default one-hour sandbox lifetime. |

The upload has a separate local deadline, so the SDK transport cannot extend its
budget by combining read and write timeouts. Creation and other SDK requests
continue to use `E2B_REQUEST_TIMEOUT`; the judge command keeps its own timeout.
Each sandbox attempt reserves `2 * rpc + upload + min(rpc, 30) + 30` seconds in
addition to the command budget, where `rpc` is the request timeout and `upload`
is the upload timeout for the current retry. The whole-attempt budget and sandbox lifetime are
`max(1200, ceil(command + reserve))` seconds: short batches receive 20 minutes,
while longer batches retain their derived budget. Each retry starts a fresh
attempt budget. With the default request and base upload settings, the estimated
command budget for legacy uploaded batches keeps the initial sandbox lifetime within one hour. Upload retries
increase the reserved time without shortening the judge command budget. These
retries or explicitly longer settings may require a longer-lived sandbox.

Provide your own prepared code-judge template and set `E2B_TEMPLATE` explicitly.
NorthRL does not include the template build assets or the sandbox-side judge
implementation. The template must provide `/root/miniconda3/bin/python` and
`/opt/e2b-sandboxfusion/judge.py --stdio` with the prepared request/result protocol.
These sandbox paths are part of the evaluator contract; the default E2B base
template is insufficient. Missing or blank `E2B_TEMPLATE` fails before sandbox
creation. Empty model answers retain their zero-reward result without a sandbox.

Without a shared reference, all cases run in sequential batches of up to 100 cases or 10 MiB. A single case
may exceed the byte target; each request and result has a hard limit of 1 GiB.
`reward` is 1 only when every case passes; any failed case gives 0.
`sandbox_result.score` retains the case-weighted judge score for diagnostics only.

Transient upload failures allow five attempts total, each in a fresh sandbox
with the same payload. The four retries use exponential backoff with jitter:
5–10, 10–20, 20–40, then 30–60 seconds. With the default base upload timeout,
the five upload deadlines are 300, 450, 600, 750, and 900 seconds. Only upload
failures advance this deadline; each batch starts from the base timeout.
Creation and command warmup each retain
two retries; a legacy batch command timeout retries twice, with a third timeout giving
zero reward for the answer. Retries affect only the failed batch and preserve
the judge inputs and scoring rules.

Caller cancellation, worker shutdown, and expiry of the whole-attempt deadline
do not start another attempt. Exhausted setup retries, other E2B service
failures, invalid inputs, incomplete results, and inconsistent result fields
raise errors without a reward. Batch diagnostics include `upload_timeout_s`
alongside the payload size, attempt number, and elapsed stage time.

### Shared testcase storage for code evaluation

A prepared row can retain its complete `label.cases` while adding
`metadata.cases_ref`. This selects one sandbox per candidate answer for the
entire task, rather than one sandbox per 100-case/10-MiB upload batch:

```json
{"protocol_version":1,"manifest_path":"/mnt/data/mixrl-eval/v1/manifests/<sha256>.json","manifest_sha256":"<sha256>","case_count":29}
```

The reference above is the value of `metadata.cases_ref`. Dataset preparation
and vePFS publication happen before training, outside NorthRL. Prepared JSONL
files preserve the original labels and all cases; non-code rows remain unchanged.
The manifest binds a case JSONL file, every case's byte range and SHA256, the
original judge settings, and explicit resource budgets. Hashes cover canonical
UTF-8 JSON; case record hashes include their trailing newline. Publish and verify
these immutable files before training. The shared volume name is deployment
configuration, not dataset data.

The protocol-v1 reader is maintained and deployed outside NorthRL. NorthRL does
not distribute the reader or its publication tools. You must provide
these assets on a volume accessible to your E2B account. The mount point
`/mnt/data` and the content-addressed paths under `/mnt/data/mixrl-eval/v1`
are fixed by protocol v1; they do not identify a shared public volume.

Use a verified reader already published at
`/mnt/data/mixrl-eval/v1/readers/<sha256>.py`, and configure the host with its
volume, path, and SHA256:

| Variable | Meaning |
| --- | --- |
| `E2B_VEPFS_VOLUME` | Existing volume containing the verified publication. |
| `E2B_VEPFS_READER_PATH` | Content-addressed reader path shown above. |
| `E2B_VEPFS_READER_SHA256` | SHA256 of the published reader bytes. |

The E2B service must support named `volume_mounts`; the client requires
`e2b>=2.50.0,<2.51`. Only code and verified references travel with each request.
Small descriptors are base64-encoded in a shell-quoted bootstrap; larger ones
use a small JSON upload. Neither `label.cases` nor the reader is uploaded per
evaluation. Tests embedded in `code_template` still travel as part of the prepared
code. Missing configuration, invalid references, changed settings,
hash mismatches, and incomplete results fail rather than silently falling back.

The reader streams the task file into a private sandbox directory, checks its
hashes, closes shared files, and unmounts vePFS before running candidate code.
It parses and judges one complete case at a time, preserving order and coverage.
The existing SandboxFusion judge still compares the complete outputs. The
resource-aware runner drains stdout concurrently to avoid large-output pipe
backpressure. After judging, replies omit repeated testcase text and large
output values, retaining per-case verdicts, scores, execution status, resource
information, and bounded diagnostic previews. Stdout and fetched answer files
have a 1 GiB hard limit; exceeding it produces an output-limit verdict. Stderr
is drained completely while retaining at most 16 MiB, so verbose stderr alone
does not fail an otherwise correct answer.

The manifest records distinct run, compile, and checker memory budgets in MiB,
with one CPU of aggregate quota by default. Preserve declared task memory limits;
use the protocol-v1 default of 1024 MiB for missing run limits and separate
1024 MiB budgets for compilation and checkers. Convert source memory units to
MiB explicitly during data preparation. Time limits remain those of the original
task. Candidate and checker processes run under separate unprivileged identities;
reference files stay private. A per-case OOM is detected using that execution's cgroup
counter delta, not historical counters from previous cases.

Whole-task command budgets use the complete case count without the legacy
one-hour default cap; `E2B_COMMAND_TIMEOUT` remains an explicit override. The
existing 24-hour attempt ceiling is checked rather than silently truncating a
larger estimate. Whole-task command timeouts and retryable execution failures share
a budget of three failed runs, with each retry using a fresh sandbox. If the final
failure is a run-stage timeout, the answer receives zero reward with
`reason: "timeout_after_retry"` and `timed_out: true`. Missing case verdicts remain
unevaluated (`sandbox_result.score: null`, `tests: []`); timeout and retry counts
remain available in metrics. A final non-timeout failure, an exhausted setup
failure, or expiry of the whole-attempt deadline still raises an evaluation error.
Sandbox cleanup failures are logged without blocking eligible retries. A genuine
per-case TLE/MLE remains a failed case, and all remaining cases continue. Binary training reward is still
1 only when every case passes; the raw average score is diagnostic.

### Single-turn Worker Configuration

Set `rollout.type: single_turn` for `SingleTurnRolloutWorker`. It generates one
response, selects a reward using the rules above, and creates the training
trajectory. Python integrations can import it from `northrl.rollout_worker`.

Configure worker options directly under `rollout`. `rollout.num_workers` controls
worker concurrency. Custom workers use `rollout.custom_module_path` and default
to the class name `CustomRolloutWorker`; set `rollout.custom_class_name` explicitly
when the class uses another name.

SFT selects its internal worker through `algorithm.type: sft` and should omit
the entire `rollout` block. Single-turn debug traces use `single_turn_rollout` as
their type and `rollout_result` for the generation result.

### Recast Reward

`recast` is an ordinary async reward function in `northrl/rewards/recast.py`.
It uses bp-rl-judge's RECAST instruction-following prompt and calls the configured
judge model directly. No bp judge gateway is needed.

A row may omit `label` entirely:

```json
{"prompt": [{"role": "user", "content": "Give exactly three suggestions for learning Python, in Chinese."}], "reward_fn": "recast"}
```

To use it by default for rows without a reward name, set these fields in an
existing training recipe:

```yaml
rollout:
  type: single_turn
  reward_fn: recast
```

Configure the judge in the environment of the process that runs NorthRL:

```bash
export LLM_JUDGE_URL=http://judge-host:8000/v1
export LLM_JUDGE_MODEL=your-served-judge-model
# Optional: LLM_JUDGE_API_KEY and LLM_JUDGE_TIMEOUT_S
```

| Variable | Meaning |
| --- | --- |
| `LLM_JUDGE_URL` | Required OpenAI-compatible base URL, normally ending in `/v1`. For multiple endpoints, separate URLs with newlines; all must serve the configured model. |
| `LLM_JUDGE_MODEL` | Required served model name. |
| `LLM_JUDGE_API_KEY` | Optional API key; defaults to `_`. |
| `LLM_JUDGE_TIMEOUT_S` | Optional positive finite request timeout in seconds; defaults to 180 for `recast`. |

The judge request uses temperature 0 and a maximum of 8192 output tokens.
It sends the dataset prompt and the complete generated response, including any
think or EOS text. Chat-message arrays are converted to lists and represented
as strings, preserving the original roles and message order. It does not use
`label`, `metadata`, or the tokenizer-rendered `prompt_str`, and does not require
boxed answers or thinking tags.

The last `[[0]]` or `[[1]]` token in the judge response determines the reward;
whitespace inside the brackets is allowed. If response content is empty, the
judge's `reasoning_content` is used. The returned score includes `correct` and
`reason`: `pass`, `fail_llm`, or `no_verdict`. No verdict gives 0 without a retry.

Request failures and malformed API responses receive at most two total attempts.
The first endpoint is selected randomly and the next attempt rotates to the next
one. A persistent failure raises `RuntimeError` with the original cause; invalid
configuration fails before sending a request. This follows NorthRL's exception
propagation policy. bp's gateway instead converts uncaught adapter failures into
zero rewards; that gateway policy is not reproduced here. Each request client
is closed, and proxy environment variables are bypassed for judge requests.

Rollout workers run in the local NorthRL process and read its environment. Export the judge variables in the current shell or source your private environment script before running `python -m northrl.main`.

### VerInstruct Reward

`verinstruct` is an ordinary async reward function in
`northrl/rewards/verinstruct.py`. It runs the row's Python hard-constraint checkers
in an E2B sandbox through the async E2B SDK, then calls the LLM directly for
soft constraints. No bp judge gateway is needed. Select it per row with
`"reward_fn": "verinstruct"`, or set the default in an existing training recipe:

```yaml
rollout:
  type: single_turn
  reward_fn: verinstruct
```

See the [data example](../03-data-loader/data-loader.md#single-turn-reward-data)
for `metadata.instruction` and `metadata.hard_functions`. Each checker defines
`check_following(instruction, response)`. The hard instruction is the first
truthy value of `metadata.instruction` or `metadata.prompt_text`; otherwise it is
`task["prompt"]` for a plain string, or `""` for a chat-message list. Supply
`metadata.instruction` explicitly when using the single-turn rollout worker.

Before either stage, the reward keeps the text after the last `</think>`;
if that marker is absent, it keeps the text after the last `###Response`.
It trims surrounding whitespace and repeated trailing `<|im_end|>`,
`<|endoftext|>`, and `<|eot_id|>` tokens. Tokens inside the answer remain intact.
Both hard and soft checks receive this same normalized response. `label` is
unused.

Configure E2B and the judge in the NorthRL process environment:

```bash
export E2B_API_KEY=your-e2b-api-key
# Optional: E2B_TEMPLATE=your-python-template
export LLM_JUDGE_URL=http://judge-host:8000/v1
export LLM_JUDGE_MODEL=your-served-if-verifier-model
# Optional: LLM_JUDGE_API_KEY and LLM_JUDGE_TIMEOUT_S
```

`E2B_API_KEY` authenticates with hosted E2B. `E2B_TEMPLATE` optionally selects a
sandbox template; otherwise the SDK's base template is used. Custom templates
must provide `python3`. The SDK also supports `E2B_DOMAIN` and `E2B_API_URL` for
E2B SDK.

Missing, null, or empty `hard_functions` skips E2B entirely, so those rows do not
require E2B credentials. The LLM uses the same `LLM_JUDGE_*` settings as `recast`,
including its default 180-second request timeout. Configure an IF-Verifier model
to match bp's VerInstruct judge. All rewards in one process share these settings;
selecting `verinstruct` does not automatically select a different model from
`recast` or the math LLM rewards. Export both E2B and judge settings in the local driver environment before launching.

For each row with hard checks, NorthRL creates a fresh `AsyncSandbox` with internet
access disabled, writes the checker harness to `/tmp/verinstruct_check.py`, and
runs `python3 -I /tmp/verinstruct_check.py`. All checkers run sequentially in this
single command. The first false return or checker exception stops evaluation and
gives `reward: 0`, `correct: false`, and `reason: "fail_hard"`; no LLM request is
sent. Success requires both exit code 0 and stdout equal to `True` after stripping
surrounding whitespace. Additional checker output therefore fails the hard stage.
Other nonzero exits also fail the hard stage, except 126 and 127: these indicate
an unusable Python command or template and raise an error.

The command has a 10-second timeout. Sandbox creation, file upload, and command
execution together have a 60-second deadline; SDK request limits are 30 seconds
for creation and 10 seconds for upload and command execution. SDK output callbacks
cap combined stdout and stderr at 2,000,000 UTF-8 bytes. Once a sandbox is
created, NorthRL attempts to kill it afterward with a separate 10-second cleanup
deadline and request timeout. Cleanup failures are logged without replacing the score or original
error; the sandbox's 60-second lifetime provides fallback cleanup. A timeout
during creation, upload, or execution retries the hard checks once in a fresh
sandbox after cleanup of the first attempt. Both attempts use the same checker
source and independent time/output limits. A successful retry follows the normal
hard-then-soft scoring rules. A second timeout gives `reward: 0`, `correct: false`,
and `reason: "timeout_after_retry"`, without sending an LLM request. Cleanup
timeouts do not trigger retries. The SDK may also retry transport requests internally.

The harness applies 10-second CPU, 1 GiB address-space, 64 MiB file-size, and
64-process limits and disables Python socket creation. E2B provides the sandbox
isolation, with network access disabled when the sandbox is created.

After hard checks pass, the soft stage reuses the ordinary `recast_reward`
function with the original dataset prompt and the normalized response. The last
`[[0]]` or `[[1]]` in the judge reply selects the binary reward. The result's
`reason` is `pass`, `fail_soft`, or `no_verdict`; no verdict gives zero. Soft
requests use the [Recast retry and error rules](#recast-reward).

Invalid data or configuration, non-timeout E2B API errors, exceeded output limits,
and persistent LLM request failures raise an error. Cancellation propagates
without a reward-level retry. Only repeated sandbox timeouts are converted to
zero rewards by the retry policy above.

## Rollout Worker Types

### SimpleRolloutWorker (`type: "simple"`)

Basic LLM completion worker.


### SingleTurnRolloutWorker (`type: "single_turn"`)

Single-response generation, reward evaluation, and trajectory construction.

```yaml
rollout:
  num_workers: 128
  reward_fn: math_with_format
  sampling:
    temperature: 0.7
  type: single_turn
```

### E2BRolloutWorker (`type: "e2b"`)

Runs task-defined agent and evaluation hooks in an E2B sandbox. Configure the
entrypoint, input fields, and execution budgets under `rollout.execution`.


### Internal SFTRolloutWorker

Pass-through worker for supervised fine-tuning. It converts prepared,
independent Datums from `SFTStreamingDataLoader` into Trajectories without LLM
inference; Weaver performs packing during training.

```yaml
algorithm:
  type: sft
```

This worker is an internal bridge, not a public rollout-worker type. Omit the
entire top-level `rollout` block for SFT. The Controller always creates exactly
one `SFTRolloutWorker`; if the block is supplied, NorthRL emits a `UserWarning`
and ignores it in full. `algorithm.loss.mode` may be omitted and is derived as
`cross_entropy`.

## Configuration Options

### type

**Type:** `string`
**Options:** `"single_turn"`, `"simple"`, `"e2b"`, `"mock"`
**Default:** Required

Rollout worker implementation.

- `simple`: Basic LLM completion
- `single_turn`: One response with a selectable reward function
- `e2b`: Execute a remote rollout entrypoint in an E2B sandbox
- `mock`: Mock worker for testing

### num_workers

**Type:** `int`
**Default:** Required

Number of rollout worker instances.

**Guidelines:**
- CPU-bound: 4-16 workers
- I/O-bound (LLM API): 32-256 workers
- Workers are local coroutines; size concurrency to the driver and remote-service limits
- Consider data throughput needs

**Example:**
```yaml
# Small scale
num_workers: 8

# Medium scale
num_workers: 64

# Large scale
num_workers: 256
```

### need_llm_inference

**Type:** `bool`
**Default:** `true`

Whether worker needs LLM inference service.

**When `true`:**
- Worker uses LLMServiceClient
- Requires inference service configured
- Supports weight synchronization

**When `false`:**
- Worker doesn't call LLM
- Used for testing or non-LLM tasks

### temperature

**Type:** `float`
**Default:** `1.0`

Sampling temperature for LLM generation.

**Values:**
- `0.0`: Greedy (deterministic)
- `0.1-0.5`: Low variance
- `0.7-1.0`: Balanced (common)
- `>1.0`: High variance

**Use Cases:**
```yaml
# Deterministic (evaluation)
temperature: 0.0

# Low variance (precise tasks)
temperature: 0.3

# Balanced (general training)
temperature: 0.7

# Exploration (diverse responses)
temperature: 1.0
```

### top_p

**Type:** `float`
**Default:** `1.0`

Nucleus sampling probability for LLM generation. Weaver forwards this with the request-level
sampling parameters. Backends that do not support this filter may ignore it and log a warning.

```yaml
top_p: 0.95
```

### top_k

**Type:** `int`
**Default:** `-1`

Top-k sampling cutoff for LLM generation. Use `-1` to leave top-k disabled when supported
by the backend. Backends that do not support this filter may ignore it and log a warning.

```yaml
top_k: 50
```

## Custom Worker Options

### custom_module_path

**Type:** `string`
**Default:** None (optional)

Path to a custom worker implementation, configured as `rollout.custom_module_path`.

**Format:** Relative to recipe directory or absolute path

**Example:**
```yaml
custom_module_path: "agent_workspace/my_worker.py"
```

**Custom Worker:**

Implement `async rollout(task)` on a `BaseRolloutWorker` subclass. Submit the
complete trajectory using `_put_trajectory()` and return its result. See
[Custom Workers](../05-rollout-workers/custom-workers.md).

### custom_class_name

**Type:** `string`
**Default:** `"CustomRolloutWorker"`

Class name in the custom worker module, configured as `rollout.custom_class_name`.

**Example:**
```yaml
custom_class_name: "MyWorker"
```

## Worker Parameters

### execution

**Type:** `dict`
**Default:** `{}`

E2B execution options under `rollout.execution`. The E2B worker reads its remote
entrypoint, selected input fields, and timeout settings here.

```yaml
rollout:
  type: e2b
  num_workers: 4
  execution:
    input_fields: [instruction, interface]
    agent_timeout_seconds: 3600
    gateway_publication_timeout_seconds: 300
    token_prefix_reuse: false
    env_error_retries: 3
    volume_mounts:  # Optional shared volumes
      /mnt/shared: shared-data
    entrypoint:
      target: /path/to/NorthRL/northrl/rollout_entrypoints/swerebench_claude_code_rollout.py
      task_files_root: /data/swe/evaluation
```

`execution.result_processing` accepts `thread` (default) or `process`. Opting
into `process` moves final-result gzip/JSON validation and packed-array expansion
to one resident spawn worker shared by concurrent rollouts on the same driver
loop. Complete results, including packed arrays, cross IPC back to the driver;
pool ownership and duplicate detection stay in the driver. This adds process
startup, memory and IPC costs, so CPU replay speedups are not end-to-end training
guarantees. The existing two-slot result admission, downloaded-result deadlines
and cancellation drain remain in effect. The last rollout owner joins the worker.
The setting can also be overridden per execution profile. It does not change
periodic snapshot downloads or partial training-trace recovery.

Set `NORTHRL_PROFILE_RESULTS=1` in the Driver environment to enable low-volume
result diagnostics at INFO level (disabled by default). Each `driver.result-decode`
record includes the effective mode, compressed bytes, expanded token positions
(sum of segment lengths), segment count and parse/restore wall and executing-thread
CPU time. Process mode also records the child PID and parent round-trip wall time.
The round trip includes worker startup on the first call, serialization, transport,
child queueing and scheduling; subtracting child parse/restore time does **not**
measure pure IPC.

`driver.result-postprocess` measures Driver-side result-to-trajectory construction.
`driver.result-pool-insert` measures actual pool publication, including locking,
grouping and duplicate detection, after the training wrapper releases its staged
trajectory. Both report wall and executing-thread CPU time. Download admission,
read and processing-queue timings remain in `driver.file-download`. The records
contain identities, counts and timings, not tensor contents; enabling them does
not alter admission, cancellation or result validation.

For local performance validation, `tests/benchmarks/benchmark_result_processing.py`
exercises the real result-download adapter, packed-array IPC, grouped pool and
foreground conversion/SDK builder. It reports per-fixture expanded sizes,
foreground completion, all-background completion and observed burst arrivals;
parity checks run after both timed phases. Its default mixed fixtures are
synthetic. `--results-dir` accepts a directory containing only saved full-result
gzip files to replay real sizes and types. Select NorthRL/SDK sources with
`PYTHONPATH`; the report records their paths and the SDK source version. Remote
reads are in memory and arrival schedules are synthetic, so this check does not
establish a Server/GPU speedup or reproduce online background state.

`execution.volume_mounts` maps absolute, normalized sandbox paths other than `/`
to non-empty volume names. Set it globally under `execution` or in an execution
profile. A profile that omits the field inherits the global mapping; an explicit
mapping replaces the entire mapping, and `{}` or `null` clears it.

Volumes are mounted when each sandbox is created, with the same mapping used on
environment retries. Mounted data is shared and writable. NorthRL does not
automatically copy it into sandbox storage or unmount it; a task's `prepare` hook
can perform those operations if needed. Leaving the setting unconfigured
preserves the existing SDK call without a volume-mount argument.

For optional agent tools, set `NORTHRL_E2B_AGENT_ENV` on the driver to a
comma-separated list of host environment variable **names**, such as
`SERPER_API_KEY,SERPER_API_URL`. Only selected, defined values reach the root
agent-stage command, including its preparation hook. They are not sent to
bootstrap or evaluation, or serialized into the task/config payload or trace
artifacts. The task hook controls which values its non-root agent inherits.
An unset or empty selector forwards nothing extra; invalid or duplicate names
and reserved Python/runtime/model-gateway variables are rejected before sandbox
creation. Use `NORTHRL_E2B_EVALUATOR_ENV` separately for evaluator-only settings.
See the [JobBench recipe](../../../../recipe/jobbench/README.md) and its `env.sh`
for reference-mount and Serper configuration. Keep actual keys in private host
environment settings, not YAML or dataset fields.

The host uses the Weaver SDK to manage the current sampler and waits for its
NorthGate `gateway.state` to become `published`. It then passes the returned
`gateway.base_url` and `gateway.model` to the sandbox; no manual gateway address or
model configuration is needed. `gateway_publication_timeout_seconds` bounds this
centralized wait and defaults to 300 seconds. Startup confirms publication after
the sampler warmup and before starting workers. Weight synchronization waits for
the candidate sampler's publication before switching its client/version and
releasing the next batch. Successful publication is cached for the session's
lifetime; normal rollouts do not periodically query its state. Failure preserves
the previous snapshot and cleans up the unactivated candidate session.

The cache does not automatically detect idle retirement. Centralized recovery
can call `warmup_sampling(refresh_gateway=True)` to wake the sampler with a direct
Weaver probe and refresh publication. Alternatively, issue a direct Weaver
sampling or logprobs request followed by `refresh_gateway_snapshot()`, or replace
the sampler. NorthGate retries, metadata GETs, and heartbeats cannot wake it.
Publication checks report `retiring` / `retired` explicitly instead of sampling
automatically; expired or terminated samplers must be replaced. Running rollouts
retain their original session/version; recovery applies to subsequent tasks.

The sandbox relay manages NorthGate retries internally: at most 3 attempts per
request, with a 300-second recovery window from the start of the request for
reading error responses and backoff. No further retries start after this window.
The connection timeout is 120 seconds, capped by the remaining recovery window.
Temporary transport failures before receiving response headers, HTTP 429, and
HTTP 5xx use one- and two-second backoff bases with jitter and honor `Retry-After`;
HTTP 404 fails immediately. If `Retry-After` exceeds the remaining window, the
request fails instead of retrying early. Waiting for successful response headers,
successful response bodies, and SSE generation remain subject to the existing
request and agent execution limits, even after the recovery window expires.
Failures while reading a successful response body or stream are never replayed.
Logs preserve the actual HTTP status, SSE error code/type, and request ID.

Set `WEAVER_API_KEY` and `NORTHGATE_API_KEY` on the host. The HTTP relay forwards
OpenAI and Anthropic requests and records `token_data` from the responses. Agents keep
using the local API address and `model: northrl`; the relay selects the published
sampler. Explicit `rollout.sampling.max_tokens` (`validate.sampling` during
validation) overrides the agent's token budget without clipping it to remaining
context. When omitted or null, E2B preserves the agent's budget. NorthGate
owns context checks.

E2B does not use `execution.tokenizer`, `execution.chat_template`, or
`rollout.parsers`. Configure the model tokenizer, chat template, and output parser
in the server's NorthGate codec profile and enable sampler auto-publication.
The host's `models.actor.tokenizer` remains available for training.

`execution.token_prefix_reuse` is a boolean, defaulting to `false`. It applies
to OpenAI Chat Completions and Anthropic Messages requests, with JSON or SSE responses;
the relay does not expose the Responses API. Enable it only when NorthGate supports
`history_tokens` on the endpoint in use; see
[NorthGate: reuse historical tokens](https://northgate.xiaobei.top/docs/backend/features/weaver-sample.html#复用历史-token).
Execution profiles may override the common setting; training and validation use
the same setting. The configuration name remains `token_prefix_reuse`; the relay uses
`history_tokens` to reuse its cached token history.

Before matching or forwarding Anthropic requests, the relay removes `cache_control`
from the request root, direct system/message content blocks, and outer tool
definitions. This also applies when `token_prefix_reuse` is disabled. Moving prompt-cache
breakpoints between turns therefore does not invalidate an otherwise identical history.
These explicit prompt-cache directives are not forwarded to NorthGate; `history_tokens`
reuse remains independent. Tool inputs, input schemas, and other nested business data
retain any same-named fields. OpenAI requests are unaffected.

With token reuse enabled, Anthropic history comparison also tolerates the removal of
newline-only text before tool calls. SGLang's `qwen3` reasoning and `qwen3_coder` tool
parsers retain the `"\n\n"` between `</think>` and `<tool_call>` as content, while
Claude Code can omit the corresponding text block from the next request. NorthRL removes
these blocks only from the cache comparison copy, on both insertion and lookup; it
does not change SGLang or `llm-t2t-go` parsing, forwarded messages, JSON/SSE responses,
or recorded messages, tokens, logprobs, and masks. The reused tokens still include the
original newline tokens. There is no additional configuration option.

This exception applies only to an Anthropic assistant content list containing
`tool_use`, when every text block precedes the first `tool_use`, has only `type` and
`text` fields, and contains a nonempty string made entirely of `\r`/`\n`. All such
text blocks are omitted together. Any real text, space/tab, empty text, extra block
metadata, or text after a tool call keeps the entire content list exact. Thinking,
tool inputs, nested content, and OpenAI messages remain exact. If two histories
become equivalent but have different original tokens, the cache marks them ambiguous
and does not reuse that history or fall back to a shorter match.

The relay saves completed message histories and their actual prompt/response tokens.
Before the next request, it compares the API, message prefix, and request settings
such as system, tools, and thinking. Anthropic version and beta headers also participate
in matching. A matching, unambiguous history supplies the top-level field
`history_tokens: [...]`, containing the previous actual prompt tokens followed by its
response tokens; otherwise the field is omitted. Full messages still accompany the
request, without a `message_count` field. Only successfully recorded responses
ending with OpenAI `stop`/`tool_calls` or Anthropic `end_turn`/`tool_use` are reusable;
streams must finish with a fully reconstructed assistant message.
OpenAI requests with nonempty `stop`, `sampling_params.stop`, or
`sampling_params.stop_token_ids` do not seed the reuse cache: their finish reason does
not establish that the model emitted its required terminator. Anthropic responses
ending with `stop_sequence` are also ineligible.

The history must be nonempty, contain nonnegative token IDs, and end with the model's
termination token from the previous response. Truncated output without that terminator
cannot be reused. Keep the model, tokenizer, and template configuration fixed throughout
each rollout.

NorthGate renders the full request, counts occurrences of the history's final token ID
in `history_tokens`, and cuts the rendering after the same number of terminators. It
encodes the remaining new messages and generation prefix and appends them to the
unchanged history. The template must preserve the terminator count for the same history
across renders. NorthGate retains no session history; the client is responsible for
matching messages to tokens. Invalid history or more history terminators than the
rendered request contains produces `400 invalid_request` before sampling.

The relay rejects returned prompt tokens that do not preserve the supplied prefix.
The recorder continues to use actual returned token data and strict token-prefix
checks. See [E2B task integration](e2b-advanced.md#token-prefix-reuse).

`rollout.execution.input_fields` is required for E2B without profiles and must be a list of unique,
non-empty strings. It selects the dataloader fields copied into `RolloutInput.task`;
`[]` sends no task fields. Missing fields are skipped, and the task script validates
its required inputs and builds the prompt. SWE selects `[instruction, interface]`;
other tasks can select fields such as `[question, context]`. `task_id` is carried
separately in metadata for evaluation and need not be selected. Unselected data
fields are not uploaded as task inputs; do not select hidden answers or tests.

`execution.entrypoint.target` and the optional `task_files_root` must be absolute
host paths; relative paths and `~` are not accepted. Replace `/path/to/NorthRL` in
the example with your checkout's absolute path.

E2B tasks set `execution.entrypoint.target` to a host-side Python script with
`run_agent` and `evaluate` hooks. Evaluation assets are uploaded after the agent
exits. `execution.env_error_retries` controls retries for environment failures.
Each rollout produces one complete trajectory containing its segments; a scored
task failure is a valid zero-reward outcome. See [E2B task integration](../05-rollout-workers/e2b-task-contract.md#scoring-and-failures).

Agent duration is controlled by `rollout.execution.agent_timeout_seconds`.
MiMo hooks no longer read the dataset field `agent_timeout_sec`; existing rows
may retain that unused field, but it should not be selected in `input_fields`.
An expired Agent budget cancels the hook and allows cleanup under its separate
finalization budget. Evaluator timeouts remain independent.

#### Execution Profiles

To run different task families in one E2B worker pool, replace the global
`entrypoint` and `input_fields` with named `rollout.execution.profiles`:

```yaml
rollout:
  type: e2b
  execution:
    agent_timeout_seconds: 3600
    env_error_retries: 3
    profiles:
      mimo_code:
        input_fields: [instruction, cwd, agent_user, max_turns]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/code.py
          task_files_root: /data/mimo/evaluation/code
      mimo_arvo:
        input_fields: [instruction, cwd, agent_user, max_turns, description]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/arvo.py
      mimo_terminal:
        input_fields: [instruction, cwd, agent_user, max_turns]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/terminal.py
          task_files_root: /data/mimo/evaluation/terminal
      mimo_general:
        input_fields: [instruction, cwd, agent_user, max_turns]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/general.py
          task_files_root: /data/mimo/evaluation/general
      swe:
        input_fields: [instruction, interface]
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/swerebench_claude_code_rollout.py
          task_files_root: /data/swe/evaluation
```

Each profile must supply its own complete `entrypoint` and `input_fields`.
Global `execution.entrypoint` / `input_fields` cannot be combined with
`execution.profiles`. A profile's `agent_timeout_seconds` overrides the common
value; omitting both uses 3600 seconds unless the legacy `timeout_seconds` alias
is configured. Profiles can also override `preparation_timeout_seconds`,
`agent_finalize_timeout_seconds`, and `evaluation_timeout_seconds`, as well as
`env_error_retries`, `result_processing`, `token_prefix_reuse`, and the legacy
`tokenizer` option. Omitted fields inherit
their common values. Gateway publication settings remain common to the worker pool.
All profiles share the actor, model tokenizer, rollout sampling settings, and trainer.

Every training and validation task in profile mode must contain a known
`rollout_profile`. For single-family source files, set it on
[`data.train.sources`](data-config.md#mixing-rl-task-sources) and let the loader
inject the field. With `data.train.files` or validation files, add it to each
row. Missing or unknown profile names fail rather than falling back to another
entrypoint.

The Controller prepares execution code and evaluation assets separately for
each profile. The worker selects both using the task's `rollout_profile`, then
uses the original `task_id` within that profile's evaluation root. Different
families can therefore reuse the same `task_id` without sharing assets. A
profile never inherits another profile's evaluation directory. The selected
`input_fields` alone determine task inputs uploaded to the sandbox;
`rollout_profile` and `task_id` remain routing/evaluation metadata unless
explicitly selected. Omitting `profiles` retains the single-entrypoint behavior.

Before sampler warmup, E2B artifact preparation logs each execution profile's
start, completion, and elapsed time at INFO level. While packing evaluation
archives, progress reports completed/total tasks, archive bytes, and elapsed
time at most once every five seconds, between completed tasks.

### params

**Type:** `dict`
**Default:** `{}`

`rollout.params` holds worker-specific settings for custom and non-E2B workers
that read this field. It is independent of `rollout.execution`. E2B requires
`params` to be empty and reads its settings from `execution`.

## Length Constraints

### max_prompt_length

**Type:** `int`
**Default:** Inherited from `data.max_prompt_length`

Maximum prompt length in tokens.

**Example:**
```yaml
max_prompt_length: ${data.max_prompt_length}
```

### max_response_length

**Type:** `int`
**Default:** Inherited from `data.max_response_length`

Maximum response length in tokens.

**Example:**
```yaml
max_response_length: ${data.max_response_length}
```

## Common Patterns

### Simple LLM Task


### Math Reasoning

```yaml
rollout:
  num_workers: 128
  reward_fn: math_with_format
  sampling:
    temperature: 1.0
  type: single_turn
```

### Testing Configuration


## Worker Scaling Guidelines

### Development

```yaml
rollout:
  num_workers: 8  # Small scale for debugging
```

### Moderate Concurrency

```yaml
rollout:
  num_workers: 64  # Concurrent local rollout workers
```

### Higher Concurrency

```yaml
rollout:
  num_workers: 256  # Requires sufficient driver and service capacity
```

## Temperature Guidelines

### Task-Based

```yaml
# Exact matching (code, math)
temperature: 0.0

# Reasoning tasks
temperature: 0.7

# Creative tasks
temperature: 1.0

# Exploration training
temperature: 1.0
```

## Troubleshooting

### Workers Idle

**Symptom:** Low throughput, workers not processing

**Solutions:**
- Check data loader has items available
- Verify LLM service is accessible
- Review worker logs for errors
- Check whether the next data batch is waiting for weight synchronization or validation

### High Error Rate

**Symptom:** Many failed trajectories

**Solutions:**
- Check LLM service status
- Verify evaluator implementation
- Review prompt format
- Check max_response_length sufficient

### Custom Worker Not Found

**Symptom:** `ModuleNotFoundError` or `AttributeError`

**Solutions:**
- Verify `custom_module_path` is correct
- Check `custom_class_name` matches class
- Ensure file is in correct location
- Make local dependencies importable in the worker process

## Related Documentation

- [Rollout Workers](../05-rollout-workers/overview.md) - Worker architecture
- [Custom Workers](../05-rollout-workers/custom-workers.md) - Implementing custom workers
- [Evaluators](../05-rollout-workers/evaluators.md) - Evaluation patterns
