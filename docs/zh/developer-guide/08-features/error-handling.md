[English](../../../en/developer-guide/08-features/error-handling.md) | 简体中文 | [文档目录](../../index.md)

<a id="error-handling"></a>

# 错误处理

<a id="overview"></a>

## 概述

`ErrorReporter` 为 NorthRL 提供集中错误跟踪和健康监控。它收集所有模块的错误、维护错误历史，并提供健康状态查询。

**位置**：`northrl/error_reporter.py`

<a id="key-concepts"></a>

## 关键概念

<a id="centralized-error-tracking"></a>

### 集中错误跟踪

所有模块向同一个 `ErrorReporter` 实例报告错误：

```
Module A → ErrorReporter ← Module B
            ↓
      Health Status Query
```

<a id="error-severity-levels"></a>

### 错误严重级别

```python
class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
```

- **INFO**：信息性事件（非故障）
- **WARNING**：不会停止执行的潜在问题
- **ERROR**：需要关注的实际错误

<a id="errorinfo"></a>

## ErrorInfo

包含错误详细信息的数据结构：

```python
@dataclass
class ErrorInfo:
    error_id: str                      # Unique identifier (UUID)
    timestamp: float                   # When error occurred
    module_name: str                   # Module reporting error
    work_context: str                  # What was being done
    severity: ErrorSeverity            # Severity level
    message: str                       # Error message
    details: dict[str, Any]            # Additional context
    exception_type: str | None         # Exception class name
    exception_traceback: str | None    # Full traceback
```

<a id="core-methods"></a>

## 核心方法

<a id="constructor"></a>

### 构造函数

```python
def __init__(self, max_errors: int = 1000)
```

使用最大错误历史容量初始化错误报告器。

**参数**：
- `max_errors`：内存中保留的最大错误数（超出时移除最早的错误）

<a id="report_exception"></a>

### report_exception

```python
def report_exception(
    self,
    module_name: str,
    work_context: str,
    exception: Exception,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    details: dict[str, Any] | None = None
) -> str
```

报告异常错误。

**参数**：
- `module_name`：发生错误的模块名称
- `work_context`：正在执行的工作描述
- `exception`：异常对象
- `severity`：错误严重级别
- `details`：附加上下文信息

**返回值**：唯一错误 ID（UUID 字符串）

**示例**：

```python
try:
    result = process_data(item)
except ValueError as e:
    error_id = error_reporter.report_exception(
        module_name="rollout_worker",
        work_context="processing_data_item",
        exception=e,
        severity=ErrorSeverity.ERROR,
        details={"item_id": item.get("id")}
    )
```

<a id="report_error"></a>

### report_error

```python
def report_error(
    self,
    module_name: str,
    work_context: str,
    message: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    details: dict[str, Any] | None = None
) -> str
```

报告非异常错误。

**参数**：
- `module_name`：发生错误的模块名称
- `work_context`：正在执行的工作描述
- `message`：错误消息
- `severity`：错误严重级别
- `details`：附加上下文信息

**返回值**：唯一错误 ID（UUID 字符串）

**示例**：

```python
if response.status_code != 200:
    error_id = error_reporter.report_error(
        module_name="inference_service",
        work_context="http_request",
        message=f"HTTP request failed: {response.status_code}",
        severity=ErrorSeverity.WARNING,
        details={"url": url, "status": response.status_code}
    )
```

<a id="get_health_status"></a>

### get_health_status

```python
def get_health_status(self) -> dict[str, Any]
```

根据近期错误（最近 5 分钟）获取系统整体健康状态。

**返回值**：包含健康状态的字典

**返回内容**：
```python
{
    "status": "healthy" | "warning" | "error",
    "message": "X errors, Y warnings in last 5 minutes",
    "recent_error_count": 5,      # Total recent errors
    "error_level_count": 2,        # ERROR severity
    "warning_level_count": 3       # WARNING severity
}
```

**状态逻辑**：
- **healthy**：近期没有错误或警告
- **warning**：近期有警告但没有错误
- **error**：近期有 ERROR 级别错误

<a id="internal-methods"></a>

## 内部方法

<a id="_get_errors"></a>

### _get_errors

```python
def _get_errors(
    self,
    module_name: str | None = None,
    severity: ErrorSeverity | None = None,
    since: float | None = None
) -> list[ErrorInfo]
```

按条件过滤查询错误。

**参数**：
- `module_name`：按模块过滤
- `severity`：按严重级别过滤
- `since`：仅查询此时间戳之后的错误

**返回值**：ErrorInfo 对象列表（最新在前）

<a id="_cleanup_old_errors"></a>

### _cleanup_old_errors

超过 `max_errors` 时自动移除最早的错误。

<a id="integration-example"></a>

## 集成示例

<a id="module-level-error-reporting"></a>

### 模块级错误报告

```python
from northrl.error_reporter import ErrorReporter, ErrorSeverity

class MyRolloutWorker(BaseRolloutWorker):
    def __init__(self, config):
        super().__init__(config)
        self._error_reporter = ErrorReporter()

    def _run_single_work(self, data_item):
        try:
            result = self.process_item(data_item)
            return result
        except Exception as e:
            self._error_reporter.report_exception(
                module_name=self.get_module_name(),
                work_context="run_single_work",
                exception=e,
                details={"data_item_id": data_item.get("id")}
            )
            raise  # Re-raise after reporting
```

<a id="health-monitoring"></a>

### 健康监控

```python
# In monitoring loop
def check_system_health():
    health = error_reporter.get_health_status()

    if health["status"] == "error":
        logger.error(f"System unhealthy: {health['message']}")
        # Trigger alerts
    elif health["status"] == "warning":
        logger.warning(f"System warnings: {health['message']}")
    else:
        logger.info("System healthy")
```

<a id="error-query"></a>

### 错误查询

```python
# Get recent errors from specific module
from northrl.error_reporter import ErrorSeverity

recent_time = time.time() - 600  # Last 10 minutes
errors = error_reporter._get_errors(
    module_name="trainer",
    severity=ErrorSeverity.ERROR,
    since=recent_time
)

for error in errors:
    print(f"[{error.timestamp}] {error.work_context}: {error.message}")
```

<a id="error-handling-patterns"></a>

## 错误处理模式

<a id="pattern-1-report-and-continue"></a>

### 模式 1：报告并继续

适用于可继续执行的非关键错误：

```python
try:
    optional_operation()
except Exception as e:
    error_reporter.report_exception(
        module_name="worker",
        work_context="optional_operation",
        exception=e,
        severity=ErrorSeverity.WARNING
    )
    # Continue execution
```

<a id="pattern-2-report-and-re-raise"></a>

### 模式 2：报告并重新抛出

适用于应向上传播的关键错误：

```python
try:
    critical_operation()
except Exception as e:
    error_reporter.report_exception(
        module_name="worker",
        work_context="critical_operation",
        exception=e,
        severity=ErrorSeverity.ERROR
    )
    raise  # Propagate the error
```

<a id="pattern-3-report-and-retry"></a>

### 模式 3：报告并重试

适用于带重试逻辑的暂时性错误：

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        result = unreliable_operation()
        break
    except Exception as e:
        error_reporter.report_exception(
            module_name="worker",
            work_context=f"operation_attempt_{attempt}",
            exception=e,
            severity=ErrorSeverity.WARNING,
            details={"attempt": attempt, "max_retries": max_retries}
        )
        if attempt == max_retries - 1:
            raise  # Last attempt failed
```

<a id="best-practices"></a>

## 最佳实践

<a id="1-provide-context"></a>

### 1. 提供上下文

始终提供有意义的 `work_context` 和 `details`：

```python
# Good
error_reporter.report_error(
    module_name="data_loader",
    work_context="loading_batch_3",
    message="Dataset file not found",
    details={"file_path": path, "batch_id": 3}
)

# Bad
error_reporter.report_error(
    module_name="loader",
    work_context="load",
    message="error"
)
```

<a id="2-choose-appropriate-severity"></a>

### 2. 选择适当严重级别

- **ERROR**：影响训练的故障
- **WARNING**：不会停止执行的问题
- **INFO**：值得关注的事件（较少使用，通常不为此使用 ErrorReporter）

<a id="3-include-relevant-details"></a>

### 3. 提供相关细节

添加有助于调试问题的详细信息：

```python
details = {
    "data_item_id": item["id"],
    "model_tag": model_tag,
    "attempt": retry_count,
    "response_code": response.status_code
}
```

<a id="4-log-and-report-separately"></a>

### 4. 分别记录日志与报告错误

ErrorReporter 自动记录日志，但也可添加上下文日志：

```python
try:
    operation()
except Exception as e:
    logger.debug(f"Operation failed, will retry: {e}")
    error_reporter.report_exception(...)
```

<a id="limitations"></a>

## 限制

<a id="not-a-replacement-for-logging"></a>

### 不能替代日志

ErrorReporter 补充而不替代标准日志：
- 使用日志记录常规事件与调试信息
- 使用 ErrorReporter 处理实际错误与健康监控

<a id="memory-bounded"></a>

### 内存有界

内存中仅保留 `max_errors` 条记录。长期分析应使用日志基础设施。

<a id="thread-safe-only"></a>

### 仅保证线程安全

ErrorReporter 在本地驱动进程内保证线程安全。各进程拥有独立的实例，不会自动汇总其他进程的错误。

<a id="related-documentation"></a>

## 相关文档

- [核心架构](../02-core-architecture/overview.md) - 系统概述
- [活动跟踪](../02-core-architecture/activity-tracking.md) - 监控与日志
- [最佳实践：调试](../../debug_rollout.md) - 调试策略
