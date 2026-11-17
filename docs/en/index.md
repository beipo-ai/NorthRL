English | [简体中文](../zh/index.md) | [NorthRL](README.md)

# Documentation Index

Every page below has an English and a Simplified Chinese version. Use the language links on each page to switch to its counterpart.

## Overview and guides

- [NorthRL](README.md)
- [Step metrics in W&B](metrics.md)
- [Debug Data Dump Guide](debug_data_dump_usage.md)
- [Debug one rollout batch](debug_rollout.md)
- [NorthRL User Guide for Algorithm Developers](user-guide.md)

## Getting started

- [Integrate your agent](developer-guide/05-rollout-workers/e2b-task-integration.md)
- [Quick Start Guide](developer-guide/01-getting-started/quick-start.md)
- [Local Configuration](developer-guide/01-getting-started/configuration-setup.md)

## Recipes

- [Minimal agent training example](recipe/e2b_minimal/README.md)
- [Reward Rollout Debugging](recipe/debug_rollout/README.md)
- [Single Turn Math Task](recipe/math/README.md)
- [SFT Quick Start](recipe/sft_quickstart/README.md)
- [Resource requirements for advanced recipes](recipe/advanced-recipes/README.md)

## Testing

- [NorthRL Test Suite](tests/README.md)

## Developer guide

### 01-getting-started

- [Local Configuration](developer-guide/01-getting-started/configuration-setup.md)
- [Quick Start Guide](developer-guide/01-getting-started/quick-start.md)
### 02-core-architecture

- [Activity Tracking System](developer-guide/02-core-architecture/activity-tracking.md)
- [NorthRL Controller](developer-guide/02-core-architecture/controller.md)
- [Core Data Types](developer-guide/02-core-architecture/data-types.md)
- [Core Architecture Overview](developer-guide/02-core-architecture/overview.md)
### 03-data-loader

- [Data Loader](developer-guide/03-data-loader/data-loader.md)
- [Supervised Fine-Tuning Data](developer-guide/03-data-loader/sft-data.md)
### 04-trajectory-pool

- [Task, rollout, and trajectory](developer-guide/04-trajectory-pool/trajectory-semantics.md)
- [Trajectory Pool](developer-guide/04-trajectory-pool/trajectory-pool.md)
### 05-rollout-workers

- [Agent task integration](developer-guide/05-rollout-workers/e2b-task-integration.md)
- [E2B task contract](developer-guide/05-rollout-workers/e2b-task-contract.md)
- [E2B runtime and traces](developer-guide/05-rollout-workers/e2b-runtime.md)
- [Base Rollout Worker](developer-guide/05-rollout-workers/base-rollout-worker.md)
- [Creating Custom Rollout Workers](developer-guide/05-rollout-workers/custom-workers.md)
- [Evaluators](developer-guide/05-rollout-workers/evaluators.md)
- [Rollout Workers Overview](developer-guide/05-rollout-workers/overview.md)
### 06-trainers

- [Complete rollouts and training batches](developer-guide/06-trainers/sync-rollout-semantics.md)
- [Score centering (experimental)](developer-guide/06-trainers/score-centering.md)
- [Algorithm Components](developer-guide/06-trainers/algorithm-components.md)
- [Rollout and Training Mismatch Correction (MIS)](developer-guide/06-trainers/mis.md)
- [Remote API Trainers](developer-guide/06-trainers/remote-api-trainers.md)
### 08-features

- [Runtime monitoring](developer-guide/08-features/live-metrics.md)
- [Rollout and trainer timing](developer-guide/08-features/rollout-timing-metrics.md)
- [Generation and reward failures](developer-guide/08-features/verifier-metrics.md)
- [Error Handling](developer-guide/08-features/error-handling.md)
- [Reasoning Parser](developer-guide/08-features/reasoning-parser.md)
- [Tool Parser Framework](developer-guide/08-features/tool-parser.md)
- [Validation](developer-guide/08-features/validation.md)
- [Weight Synchronization](developer-guide/08-features/weight-synchronization.md)
### 11-configuration-reference

- [E2B advanced configuration](developer-guide/11-configuration-reference/e2b-advanced.md)
- [E2B execution budgets and recovery](developer-guide/11-configuration-reference/e2b-execution-budgets.md)
- [Data Loader Configuration](developer-guide/11-configuration-reference/data-config.md)
- [Rollout Worker Configuration](developer-guide/11-configuration-reference/rollout-config.md)
- [Trainer Configuration](developer-guide/11-configuration-reference/trainer-config.md)
### 12-best-practices

- [Module Development](developer-guide/12-best-practices/module-development.md)
- [NorthRL Developer Guide](developer-guide/README.md)
