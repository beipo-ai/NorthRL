English | [简体中文](../../zh/tests/README.md) | [Documentation index](../index.md)

# NorthRL Test Suite

This directory contains the test suite for the NorthRL framework.

## Structure

The test directory is organized into two main categories:

```
tests/
├── __init__.py
├── conftest.py                    # Top-level pytest configuration
├── README.md                      # This file
├── unittests/                     # Unit tests for framework components
│   ├── __init__.py
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_base_module.py       # Tests for base_module
│   ├── test_northrl_types.py       # Tests for northrl_types
│   ├── test_train_batch_pool.py  # Tests for train_batch_pool
│   ├── mock/                     # Tests for mock components
│   │   ├── __init__.py
│   │   ├── test_mock_data_loader.py
│   │   ├── test_mock_rollout_worker.py
│   │   ├── test_mock_algorithm_processor.py
│   │   └── test_mock_llm_service_client.py
│   ├── data_loader/              # Tests for data_loader module
│   │   ├── __init__.py
│   │   ├── test_base_data_loader.py
│   │   ├── test_torch_data_loader.py
│   │   └── test_data/
│   │       └── test_dataset.parquet
│   ├── rollout_worker/           # Tests for rollout_worker module
│   │   ├── __init__.py
│   │   ├── test_base_rollout_worker.py
│   │   └── test_simple_rollout_worker.py
│   ├── train_worker/             # Tests for train_worker module
│   │   └── __init__.py
│   ├── trajectory_pool/          # Tests for trajectory_pool module
│   │   └── __init__.py
│   ├── algorithm_processor/      # Tests for algorithm_processor module
│   │   └── __init__.py
│   └── utils/                    # Tests for utils module
│       ├── __init__.py
│       └── test_config_utils.py
└── lint/                         # Lint and code quality check scripts
    ├── __init__.py
    └── check_license_header.py   # License header compliance checker (pre-commit hook)
```

### Test Categories

- **unittests/**: Contains unit tests for all framework components. The structure mirrors the `northrl` module hierarchy.
- **lint/**: Contains scripts for code quality checks and linting (used by pre-commit hooks and CI/CD).

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run only unit tests
```bash
pytest tests/unittests/
```

### Run specific test file
```bash
pytest tests/unittests/test_northrl_types.py
```

### Run with verbose output
```bash
pytest -v tests/
```

### Run tests with markers
```bash
pytest -m unit tests/
pytest -m integration tests/
```

## CPU performance checks

The [preparation replay](../../../tests/benchmarks/benchmark_trajectory_preparation.py)
can use saved trajectories or generate independent compact buffers without clients:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python tests/benchmarks/benchmark_trajectory_preparation.py \
  --synthetic --repeat 3 --output /tmp/grpo-cpu.json
```

The default synthetic step has 64 complete trajectories, four segments per complete trajectory, and 32,768
positions per segment (half context, half response). Vary `--parents`,
`--segments-per-parent`, `--tokens-per-sequence` and
`--response-tokens-per-sequence` independently to expose segment-count and
context-length overhead. Select `--estimator reinforce_plus_plus --gamma 1`
and/or `--postprocessing tool_error_rebalance` to measure those components.
The existing `--config`, `--manifest`, `--loader-dir` saved-data mode is retained;
these CPU-only runs require KL disabled and do not run actor recomputation.

Reports separate preparation, estimation, postprocessing, Datum conversion,
metrics and normalization. `prepare_s` retains the combined preparation/estimation
meaning. Input construction/restoration and hashes are outside timers; RSS is the
process-wide high-water mark, including imports and input data. Use the same data,
interpreter and thread settings for comparisons. These measurements do not include
RPC/GPU time or concurrent rollout contention. Smaller optimizer/request batches
do not reduce the complete-step estimation or postprocessing population.

## Writing Tests

### Unit Tests

1. Place test files in `tests/unittests/` in a directory that mirrors the module structure
2. Name test files with the `test_` prefix
3. Name test functions with the `test_` prefix
4. Use the provided fixtures from `conftest.py`
5. Keep tests simple and focused on a single behavior
6. Use mock components from `northrl.mock` instead of `unittest.mock` when possible

### Lint Scripts

1. Place lint scripts in `tests/lint/`
2. Lint scripts check code quality, formatting, and compliance
3. These scripts are typically called by pre-commit hooks and CI/CD pipelines

## Example Test

```python
def test_example_function(basic_config):
    """Test description"""
    # Arrange
    obj = SomeClass(basic_config)

    # Act
    result = obj.some_method()

    # Assert
    assert result is not None
```

## Available Fixtures

- `basic_config`: Basic configuration for general testing
- `data_loader_config`: Configuration for data loader testing
- `rollout_worker_config`: Configuration for rollout worker testing
- `train_batch_pool_config`: Configuration for train batch pool testing
