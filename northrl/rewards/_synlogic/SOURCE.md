# Vendored SynLogic verifier subset

The task verifier logic in this directory is adapted from the MIT-licensed
SynLogic project (`MiniMax-AI/SynLogic`) and the locally audited bp-rl-judge
copy used by the source dataset. Only the 35 verifier paths needed for reward
scoring are retained. Dataset generation, command-line tools, BBH/BBeH/GPQA
scorers, and dynamic GPFS imports are intentionally excluded.

NorthRL-specific changes include package-local imports, explicit task routing,
bounded input parsing, replacement of model-controlled `eval`, data-contract
validation, and a normalized strict 0/1 reward interface.
