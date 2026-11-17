English | [简体中文](../../zh/developer-guide/README.md) | [Documentation index](../index.md)

# NorthRL Developer Guide



This guide covers NorthRL task integration, training configuration, architecture,
and extension development. Run NorthRL locally and access model training and
sampling through APIs.

## Documentation Structure

###  01. Getting Started
Quickstart guides and configuration setup.

- [Quick Start](01-getting-started/quick-start.md) - Local installation, launch, and logs
- [Local Configuration](01-getting-started/configuration-setup.md) - Service credentials, data paths, and local environment

###  02. Core Architecture
System architecture and foundational components.

- [Overview](02-core-architecture/overview.md) - Architecture diagram, component overview
- [Controller](02-core-architecture/controller.md) - NorthRLController, module registry, lifecycle
- [Data Types](02-core-architecture/data-types.md) - Core types (Trajectory, Batch, etc.)
- [Activity Tracking](02-core-architecture/activity-tracking.md) - Monitoring and logging

###  03. Data Management
Data loading and management.

- [Data Loader](03-data-loader/data-loader.md) - BaseDataLoader, TorchDataLoader, custom loaders
- [SFT Data](03-data-loader/sft-data.md) - Tokenized/messages input, loss masks, and custom chat rendering

###  04. Trajectory Pool
Trajectory collection and batching.

- [Task, rollout, and trajectory](04-trajectory-pool/trajectory-semantics.md) - Task identities, complete rollouts, segments, and comparison groups
- [Trajectory Pool](04-trajectory-pool/trajectory-pool.md) - TrajectoryPool, store types, grouping strategies

###  05. Rollout Workers
Rollout worker architecture and implementation.

- [Agent Task Integration](05-rollout-workers/e2b-task-integration.md) - Connect an existing agent and configure training
- [E2B Task Contract](05-rollout-workers/e2b-task-contract.md) - Hooks, return values, and evaluation rules
- [E2B Runtime](05-rollout-workers/e2b-runtime.md) - Execution phases, model routing, traces, and logs
- [Overview](05-rollout-workers/overview.md) - Architecture, workflow, worker hierarchy
- [Base Rollout Worker](05-rollout-workers/base-rollout-worker.md) - Core interface, lifecycle, methods
- [Custom Workers](05-rollout-workers/custom-workers.md) - Creating task-specific workers
- [Evaluators](05-rollout-workers/evaluators.md) - Evaluation patterns and best practices

###  06. Trainers
Trainer architecture and algorithm integration.

- [Complete rollouts and training batches](06-trainers/sync-rollout-semantics.md) - Update boundaries, task outcomes, and batch accounting
- [Score centering (experimental)](06-trainers/score-centering.md) - Weaver constraints and experimental scoring configuration
- [Algorithm Components](06-trainers/algorithm-components.md) - Shared component responsibilities, data flow, and algorithm guides
- [Remote API Trainers](06-trainers/remote-api-trainers.md) - RemoteApiTrainer, GRPO, CrossEntropy

###  08. Features
Cross-cutting features.

- [Runtime monitoring](08-features/live-metrics.md) - Live sandbox, request, and tool activity
- [Rollout and trainer timing](08-features/rollout-timing-metrics.md) - Wait time, execution time, and overlapping stages
- [Generation and reward failures](08-features/verifier-metrics.md) - Failure, timeout, and retry counters
- [Weight Synchronization](08-features/weight-synchronization.md) - WeightSyncController, sync modes, coordination
- [Validation](08-features/validation.md) - Validator, validation cycles, metrics computation
- [Error Handling](08-features/error-handling.md) - ErrorReporter and error policies
- [Tool Parser](08-features/tool-parser.md) - Flexible tool call parsing framework for different model formats
- [Reasoning Parser](08-features/reasoning-parser.md) - Extract and structure reasoning content from model responses

###  09. Recipes
Recipe development and structure.

- [Minimal Agent Training](../recipe/e2b_minimal/README.md) - Task code, data, complete configuration, and local launch
- [Math Recipe](../recipe/math/README.md) - Single-turn generation and reward configuration

###  11. Configuration Reference
Complete configuration documentation.

- [Data Config](11-configuration-reference/data-config.md) - Data loader configuration options
- [E2B Advanced Configuration](11-configuration-reference/e2b-advanced.md) - History-token reuse and agent network policies
- [E2B Execution Budgets](11-configuration-reference/e2b-execution-budgets.md) - Stage deadlines, retries, and result recovery
- [Rollout Config](11-configuration-reference/rollout-config.md) - Rollout worker configuration
- [Trainer Config](11-configuration-reference/trainer-config.md) - Trainer configuration by type

###  12. Best Practices
Development guidelines and patterns.

- [Module Development](12-best-practices/module-development.md) - Writing NorthRL modules


##  Quick Navigation

### For New Users

1. Start with [Quick Start](01-getting-started/quick-start.md)
2. Prepare service credentials, data paths, and your environment with [Local Configuration](01-getting-started/configuration-setup.md)
3. Review [Architecture Overview](02-core-architecture/overview.md)

### For Task Developers

1. Run the [minimal agent training example](../recipe/e2b_minimal/README.md).
2. Follow [Agent Task Integration](05-rollout-workers/e2b-task-integration.md) to connect your agent.
3. Consult the [task contract](05-rollout-workers/e2b-task-contract.md) and [evaluation guide](05-rollout-workers/evaluators.md) as needed.

### For Algorithm Developers

1. Review [Algorithm Components](06-trainers/algorithm-components.md) for algorithms, configuration, and data flow.

##  Key Concepts

### Architecture
- **Modular Design**: Clean separation of concerns
- **Service Abstraction**: Unified API for different backends
- **Activity Tracking**: Comprehensive monitoring system
- **Concurrency Control**: Rollout concurrency matched to driver memory and remote service capacity

### Data Flow
```
DataLoader → RolloutWorker → TrajectoryPool → Trainer
                ↓
            Evaluator
```

### Weight Synchronization
```
Trainer → WeightSync → InferenceService
             ↓
         Validator
```

### Lifecycle
```
Controller → Initialize → Run → Monitor → Stop
                ↓
         [ Training Loop ]
```

##  Development Workflow

### Setting Up a New Task

1. **Create recipe structure**:
   ```bash
   mkdir -p recipe/my_task
   ```

2. **Configure task**:
   - Create `my_task.yaml` (recipe config)
   - Select a single-turn `reward_fn`, or implement remote `run_agent` and `evaluate` hooks in `rollout.py`

3. **Run training**: Follow the [Quick Start](01-getting-started/quick-start.md) with your task's configuration.

### Implementing a Custom Worker

Create a `BaseRolloutWorker` subclass that implements `async rollout(task)` and
submits a complete `Trajectory` with `_put_trajectory()`. Set
`rollout.custom_module_path` and `rollout.custom_class_name` to load it from your
recipe. See [Custom Workers](05-rollout-workers/custom-workers.md) for the contract.

### Implementing a Custom Trainer

1. **Create trainer class**:

2. **Register in controller**:
   ```python
   MODULE_REGISTRY[NorthRLRole.TRAINER]["my_trainer"] = MyTrainer
   ```

##  Contributing

When adding new features:

1. **Update relevant docs** in appropriate section
2. **Add examples** to demonstrate usage
3. **Update configuration reference** if new config added
4. **Add to best practices** if introducing new patterns

##  Related Documentation

- [Main README](../README.md) - Project overview
- [User Guide](../user-guide.md) - End-to-end tutorial
- [Source API](../../../northrl/) - Python implementation and docstrings

##  Getting Help

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Examples**: See `recipe/` directory for complete examples
- **Tests**: See `tests/` directory for usage examples

## Legend

-  Complete documentation
-  Work in progress / Placeholder
-  Needs review/update

---

**Last Updated**: 2026-01-16
