[English](../../en/tests/README.md) | 简体中文 | [文档目录](../index.md)

<a id="northrl-test-suite"></a>

# NorthRL 测试套件

本目录包含 NorthRL 框架的测试套件。

<a id="structure"></a>

## 结构

测试目录分为两个主要类别：

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

<a id="test-categories"></a>

### 测试类别

- **unittests/**：包含框架所有组件的单元测试，结构与 `northrl` 模块层次对应。
- **lint/**：包含代码质量检查和 lint 脚本，供 pre-commit 钩子和 CI/CD 使用。

<a id="running-tests"></a>

## 运行测试

<a id="run-all-tests"></a>

### 运行所有测试
```bash
pytest tests/
```

<a id="run-only-unit-tests"></a>

### 仅运行单元测试
```bash
pytest tests/unittests/
```

<a id="run-specific-test-file"></a>

### 运行指定测试文件
```bash
pytest tests/unittests/test_northrl_types.py
```

<a id="run-with-verbose-output"></a>

### 输出详细信息
```bash
pytest -v tests/
```

<a id="run-tests-with-markers"></a>

### 按标记运行测试
```bash
pytest -m unit tests/
pytest -m integration tests/
```

## CPU 性能检查

[准备阶段回放](../../../tests/benchmarks/benchmark_trajectory_preparation.py)
既可读取保存的轨迹，也可生成独立 compact buffer，无需连接训练服务：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python tests/benchmarks/benchmark_trajectory_preparation.py \
  --synthetic --repeat 3 --output /tmp/grpo-cpu.json
```

默认合成 step 包含 64 条完整轨迹，每条完整轨迹四段，每段 32,768 个位置，
context 与 response 各半。可独立调整 `--parents`、`--segments-per-parent`、
`--tokens-per-sequence`、`--response-tokens-per-sequence`，区分 segment 数量和
context 长度带来的开销。选择 `--estimator reinforce_plus_plus --gamma 1`
和/或 `--postprocessing tool_error_rebalance` 即可测量对应组件。
原来的 `--config`、`--manifest`、`--loader-dir` 保存数据回放入口保留；
CPU 回放要求关闭 KL，不执行 actor 重算。

报告分别记录准备、估计、后处理、Datum 转换、指标与归一化时间；`prepare_s`
保留准备与估计合计的原含义。数据构造/恢复与哈希不计入计时；RSS 是包含导入和
输入数据的整个进程内存峰值。比较时应使用相同数据、解释器和线程设置。
这些结果不含 RPC/GPU 和并发 rollout 竞争；调小 optimizer/request batch
不会缩小完整 step 的 advantage 计算与后处理范围。

<a id="writing-tests"></a>

## 编写测试

<a id="unit-tests"></a>

### 单元测试

1. 将测试文件放在 `tests/unittests/` 下与模块结构对应的目录中
2. 测试文件名使用 `test_` 前缀
3. 测试函数名使用 `test_` 前缀
4. 使用 `conftest.py` 提供的 fixture
5. 保持测试简单，聚焦单一行为
6. 尽可能使用 `northrl.mock` 中的模拟组件，而非 `unittest.mock`

<a id="lint-scripts"></a>

### Lint 脚本

1. 将 lint 脚本放在 `tests/lint/`
2. Lint 脚本检查代码质量、格式和规范符合情况
3. 这些脚本通常由 pre-commit 钩子和 CI/CD 流水线调用

<a id="example-test"></a>

## 测试示例

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

<a id="available-fixtures"></a>

## 可用 Fixture

- `basic_config`：用于通用测试的基础配置
- `data_loader_config`：用于数据加载器测试的配置
- `rollout_worker_config`：用于 Rollout Worker 测试的配置
- `train_batch_pool_config`：用于训练批次池测试的配置
