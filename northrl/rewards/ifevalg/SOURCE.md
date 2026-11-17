# Vendored IFEvalG rules

`instructions.py`, `instructions_registry.py`, and `instructions_util.py` are
vendored from the IFEvalG rule library used by `bp-rl-judge` under the
Apache License 2.0. Their Google Research copyright and license headers are
preserved.

NorthRL makes only deployment-neutral adaptations to the vendored rule layer:

- `absl.logging` is replaced with Python's standard `logging` module;
- `immutabledict` is replaced with a standard-library read-only mapping;
- `langdetect.DetectorFactory.seed` is fixed to zero for deterministic checks;
- formatting is normalized to NorthRL's Black/isort settings.

NorthRL-specific data validation, answer cleanup, all-of aggregation, diagnostics,
and reward dispatch are implemented separately in `northrl/rewards/ifeval.py`.
