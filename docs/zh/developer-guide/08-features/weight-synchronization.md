[English](../../../en/developer-guide/08-features/weight-synchronization.md) | 简体中文 | [文档目录](../../index.md)

<a id="weight-synchronization"></a>

# 权重同步

<a id="overview"></a>

## 概述

`WeightSyncController` 管理 NorthRL 训练服务与推理服务之间的模型权重同步。它协调同步时机、在更新期间阻塞服务，并管理验证触发。

**位置**：`northrl/weight_sync/weight_sync_controller.py`

<a id="key-concepts"></a>

## 关键概念

<a id="synchronization-lifecycle"></a>

### 同步生命周期

```
Training Complete → Check Sync Mode → Sync Weights → Update Version → Continue Training
                                           ↓
                                    Validation (if needed)
```

<a id="model-version-tracking"></a>

### 模型版本跟踪

同步控制器为每个模型维护两个进程本地计数器：
- **train_model_version**：已完成且更新了权重的训练轮数
- **rollout_model_version**：最近一次成功同步所达到的训练轮数

训练器使用已完成的训练步数命名导出权重，例如
`northrl_actor_v000001`。持有器成对保存活动采样客户端的路径和名称。
每个请求将服务器报告的版本与为该请求捕获的名称
比较，独立于控制器计数器。

采样器快照标签使用已完成的 `train_step`。被过滤的轮次
不导出权重，因此标签可能存在间隔。

重启或恢复后，两个计数器重新从 `0` 开始；恢复的
检查点成为该进程的采样器基准。全局训练进度
由 `train_step` 独立跟踪。

<a id="service-states"></a>

### 服务状态

`RolloutServiceState` 跟踪同步状态：
- **running**：服务运行中，无需同步
- **need_sync**：当前训练轮结束后需同步；下一数据批次仍不可用
- **syncing**：正在同步

<a id="synchronization-modes"></a>

## 同步模式

NorthRL 通过 `training.weight_sync.mode` 支持四种同步模式：

<a id="1-no-sync-no-synchronization"></a>

### 1. no-sync（不同步）

不向 rollout 采样器发布或同步权重。

**适用场景**：纯 SFT 或无需更新推理服务权重的流水线

**权衡**：rollout 模型不会收到更新后的权重。

```yaml
training:
  weight_sync:
    mode: no-sync
```

对于 `algorithm.type: sft`，省略 `weight_sync` 块时，NorthRL
推导出 `no-sync`。显式配置冲突模式会被拒绝。

**注意**：Controller 对非 SFT 的
`no-sync` 流水线推导出 `keep_batch_order: false`。对 SFT 则推导出 `keep_batch_order: true`，使
同一时间仅有一个 token 预算批次处于活动状态，直到其优化器步骤提交来源游标，
且 SFT 训练器调用 `release_next_batch()`。SFT 不使用权重同步
状态机控制此数据批次门控。

<a id="2-sync-synchronous"></a>

### 2. sync（同步）

每个训练步骤后，所有 rollout 工作进程阻塞，直到权重同步完成。

**适用场景**：追求最高训练稳定性与即时权重更新

**权衡**：频繁阻塞导致吞吐量降低

```yaml
weight:
  sync_mode: "sync"
```

<a id="3-fully-async-fully-asynchronous"></a>

### 3. fully-async（完全异步）

Rollout 与训练重叠运行。每轮更新权重后，包括
最后一轮，训练器导出权重并将该轮路径和
名称传给控制器进行同步。跳过的轮次不导出或
同步权重。训练器等待导出和采样客户端替换
完成后，才开始下一轮训练。

控制器对完全异步训练推导出 `keep_batch_order: false`。
当前缓冲区耗尽时，`get_next_item()` 就加载下一批次；
不会等待训练或批次就绪通知。对于这些加载器，
`release_next_batch()` 不执行操作，因此批次就绪通知不会调用
它。轨迹池在整个训练及同步期间都接受结果。

设置 `freeze_inflight: false` 时，新请求可继续使用旧采样客户端，
直到替换成功。设置 `freeze_inflight: true` 时，请求在
采样前检查控制器，若正在同步则等待。已经启动的请求
将响应与各自捕获的版本
比较，即使期间活动客户端已改变。

```yaml
training:
  weight_sync:
    mode: fully-async
    freeze_inflight: false
```

<a id="optional-engine-pause"></a>

#### 可选的引擎暂停

`training.weight_sync.pause_mode` 支持 `retract`、`in_place` 和 `null`
（默认）。省略或设为 `null` 时保持原有行为，不暂停引擎。
启用任一模式都要求 `fully-async` RL（`grpo` 或 `opd`）、
`backend.type: weaver`、`freeze_inflight: true` 和 `full_ft`：
省略 `models.actor.lora`、设为 `null`，或设为 `enabled: false` 且不配置 LoRA rank。
暂停作用于整个推理引擎，因此仅支持全参数微调，不能用于共享引擎中的单个 LoRA 适配器。

```yaml
models:
  actor:
    lora: null
backend:
  type: weaver
training:
  weight_sync:
    mode: fully-async
    freeze_inflight: true
    pause_mode: retract  # 或 in_place；null 禁用引擎暂停
```

权重导出完成后，持有器对当前采样客户端调用 `pause_generation(mode=...)`，
将其替换为使用已导出权重的客户端，再对**当前**客户端调用 `continue_generation()`。
即使暂停或替换失败，`finally` 块也会尝试恢复；替换未成功时使用旧客户端。
同步失败会向上传播，权重同步控制器继续保持阻塞状态。

暂停期间请求的排队与调度由 Weaver 负责。已提交的采样操作可以跨越权重更新保持待完成状态，
NorthRL 继续等待同一操作的结果。采样保留原有的并发信号量、重试，以及捕获的客户端和版本。
推理入口保留原有的 `freeze_inflight` 检查，已经通过检查的请求可与权重同步交叠。
持有器不增加提交与更新的互斥锁，也不会等待正在提交或生成的请求完成后才暂停。

| `pause_mode` | 引擎行为 |
|---|---|
| `null` / 省略 | 不暂停引擎，保持原有同步行为 |
| `retract` | 撤回正在执行的请求，恢复后重新调度；不保证保留已生成的 token 或 KV 状态 |
| `in_place` | 保留暂停的请求及其 KV 状态 |

`in_place` 要求后端支持在保留请求及其 KV 状态时更新权重。
目前没有在保留这些请求的同时清空缓存的受支持方式。
NorthRL 不会自动回退到 `retract` 或其他模式。

<a id="4-batch-async-batch-asynchronous"></a>

### 4. batch-async（批次异步）

仅当过期程度超过阈值时，工作进程才阻塞。

**适用场景**：平衡吞吐量与权重的新鲜度

**权衡**：可配置过期阈值

```yaml
weight:
  sync_mode: "batch-async"
  staleness_threshold: 2  # Block if behind by 2+ versions
```

<a id="weight-sync-methods"></a>

## 权重同步方法

NorthRL 通过 `sync_method` 配置支持多种权重传输方法：

<a id="network"></a>

### network

通过 HTTP 向 vLLM 或 SGLang 后端传输权重。

```yaml
weight_sync_controller:
  sync_method: "network"
  sync_weight_path: "/path/to/weights"
```

**后端**：vLLM、SGLang

<a id="disk"></a>

### disk

从磁盘路径更新权重。对于 SGLang，通过 sglang-router 查询所有工作进程并并行更新。

```yaml
weight_sync_controller:
  sync_method: "disk"
  sync_weight_path: "/path/to/weights"
```

**后端**：SGLang（配合 sglang-router）

设置 `weight_type: "sglang_disk"` 时，磁盘同步路径调用 SGLang
`/update_weights_from_dcp`，并随
`checkpoint_dir` 一起发送 `weight_version`。其他 SGLang 磁盘更新向
`/update_weights_from_disk` 传递相同标签。

<a id="rollout-weight-version-validation"></a>

## Rollout 权重版本验证

Weaver 请求在采样前捕获采样客户端及其预期权重名称。
响应的 `weight_version` 必须与捕获的名称匹配，即使
请求执行期间有更新的客户端生效。响应版本缺失或为空
会抛出错误。预期名称未知的仅推理持有器
仍要求响应版本，但跳过相等比较。

版本不匹配会通过推理客户端和
rollout 工作进程抛出 `WeightVersionMismatchError`。不会转为警告、自动权重修复或
再次 rollout 尝试。工作进程通过现有
活动跟踪器报告失败；控制器默认 `stop_on_error` 策略会停止运行。
启动预热执行相同严格比较。主入口现有的
`training.resume.max_retries` 设置仍控制失败后重启整个训练
运行；此模式不改变该行为。

<a id="tinker"></a>

### tinker

使用 Tinker 服务后端同步权重。

```yaml
weight_sync_controller:
  sync_method: "tinker"
```

<a id="weaver"></a>

### weaver

使用 Weaver 服务后端同步权重。

```yaml
weight_sync_controller:
  sync_method: "weaver"
```

<a id="core-methods"></a>

## 核心方法

<a id="constructor"></a>

### 构造函数

```python
def __init__(self, config: DictConfig)
```

使用配置初始化权重同步控制器。

**参数**：
- `config`：包含 sync_mode、sync_method、validate_freq 的配置

<a id="check_rollout_service_status"></a>

### check_rollout_service_status

```python
def check_rollout_service_status(self, model_tag: ModelTag) -> Literal["continue", "block"]
```

由 `InferenceServiceClient` 在每次请求前调用，检查服务应继续还是阻塞。

**返回值**：
- `"continue"`：服务可继续
- `"block"`：服务应阻塞并重试

<a id="trajectory_pool_notify_batch_ready"></a>

### trajectory_pool_notify_batch_ready

```python
def trajectory_pool_notify_batch_ready(self, model_tag: ModelTag) -> None
```

训练器从 TrajectoryPool 取出就绪批次时调用。检查
训练此批次之前是否可放行下一数据批次。

**逻辑**：
1. 在 `sync` 中，或 `batch-async` 过期程度超过阈值时，将状态设为
   `"need_sync"`，保持下一数据批次未放行
2. 在 `batch-async` 低于或等于阈值时，调用 `_release_next_batch()`，使
   工作进程可在本批次训练期间收集下一批次
3. 在 `fully-async` 中，保持数据加载不变；它已自动继续
4. 在 `no-sync` 中，直接返回，不改变数据加载

<a id="train_worker_notify_weight_update"></a>

### train_worker_notify_weight_update

```python
def train_worker_notify_weight_update(
    self,
    worker_name: str,
    model_tag: ModelTag,
    *,
    sampling_path: str | None,
    weight_version: str | None,
    weights_updated: bool = True,
    continue_training: bool = True,
) -> None
```

每轮训练完成时调用。在需要时协调同步
发布权重。

**流程**：
1. 若 `weights_updated` 为 false，保持两个版本不变，将
   `"need_sync"` 状态恢复为 `"running"`，然后执行共用的放行判定
2. 否则，在参数更新后增加 `train_model_version`
3. 状态为 `"need_sync"` 或模式为 `fully-async` 时，使用
   提供的 `sampling_path` 和 `weight_version` 同步
4. 仅在同步成功后更新 `rollout_model_version`
5. 同步后检查验证频率；若应执行验证，
   将放行交给 `end_validate()`
6. 更新与跳过的轮次使用相同最终判定：若批次因同步而
   暂停且训练将继续，则调用 `_release_next_batch()`。
   最后一轮保持下一批次阻塞。`fully-async` 不需要放行

<a id="_release_next_batch"></a>

### _release_next_batch

此辅助方法向工作进程放行下一个有序数据批次：

```python
self._dataloader.release_next_batch()
```

池没有权重同步写入门控。在 `sync` 和 `batch-async` 中，
`keep_batch_order: true` 阻止工作进程在下一数据批次放行之前
获取它。分组 RL 在所有模式下都直接请求 B = `data.train.batch_size`
个完整组，pool 不另做 worker 静止或 loader buffer 耗尽检查。
Simple/SFT 则等当前 loader 整轮已领取完且 worker 静止后再整轮交付。
常规池/存储锁仍保护共享数据，防止并发读写冲突。

存在三个放行点：`batch-async` 未超过阈值时的训练前；
保留批次的更新或跳过轮次结束后；以及
验证后。后两种情况仅在训练将继续时放行。

<a id="sync_weight_to_rollout_service"></a>

### sync_weight_to_rollout_service

```python
def sync_weight_to_rollout_service(
    self, model_tag: ModelTag, sampling_path: str | None, weight_version: str | None
) -> None
```

发布 `WeaverServiceHolder` 保存的权重，并在发布成功后更新
已同步训练计数器。Weaver 同时要求路径与
名称；mock 后端接受 `None` 进行模拟更新。失败时抛出
异常。

控制器保留 `train_model_version` 和 `rollout_model_version`，用于批次
调度与验证。`save_weights_for_sampler(name)` 返回导出
路径，不在持有器中存储。`RemoteApiTrainer.train()` 返回
`(metrics, sampling_path, weight_version)`；`BaseTrainer` 仅记录指标，
并将路径和名称传给此控制器。到同步时机时，
控制器调用 `holder.update_sampling_client(sampling_path, weight_version)`。

不需要同步的轮次保持活动采样器不变，
也不保留导出路径供后续轮次使用。下一训练轮
导出自己的权重并传递对应参数。初始化与
检查点恢复也将各自导出结果直接传给更新调用。

持有器仅保留活动采样路径/名称对。创建新客户端
使用本次调用参数，即使期间发生另一导出。
仅创建成功后才替换活动客户端和路径/名称对。调用方可
使用相同路径和名称重试失败更新；初始化失败后重试
会重新导出，而非复用缓存中的待处理导出。

每个采样请求同时捕获客户端和预期版本。
持有器将返回版本与该请求的预期版本比较，
不匹配时抛出 `WeightVersionMismatchError`。错误通过
推理客户端和 rollout 工作进程传播，不自动修复或重试任务。
启动预热也会因版本不匹配而失败。没有已知
预期版本的仅推理持有器跳过此比较。

E2B 接受 `sync` 和 `fully-async`。它将任务开始时捕获的路径传给
远程执行器，并在轨迹上保留该任务版本。随后
本地采样器变化不会使结果失效。远程结果不提供实际权重
版本，因此此路径不验证 E2B 服务使用的版本。

<a id="validation-coordination"></a>

## 验证协调

权重同步控制器协调按固定间隔触发的验证运行。

<a id="configuration"></a>

### 配置

```yaml
weight_sync_controller:
  validate_freq: 5  # Run validation every 5 training steps
```

<a id="process"></a>

### 流程

同步成功后，若已启用验证且
`rollout_model_version % validate_freq == 0`：
1. 设置 `_waiting_for_validation = True`
2. 训练器在领取下一训练批次前等待，等待期间在
   活动跟踪中保持空闲；sync 和 batch-async 保持下一
   数据批次未放行
3. 控制器使用已同步采样器运行验证
4. 控制器调用 `end_validate()`，允许训练恢复

<a id="end_validate"></a>

### end_validate

```python
def end_validate(self, model_tag: ModelTag) -> None
```

控制器在验证完成后调用。清除验证等待，
并在训练继续时调用 `_release_next_batch()`，开始 rollout
计时并放行下一数据批次。在 `fully-async` 中，数据加载器自动获取
批次，因此此方法仅清除验证等待。最后一
轮不放行另一个批次。

<a id="rolloutservicestate"></a>

## RolloutServiceState

跟踪每个模型同步状态的数据结构：

```python
@dataclass
class RolloutServiceState:
    # State tracking
    state: Literal["need_sync", "syncing", "running"] = "running"
    rollout_model_version: int = 0
    train_model_version: int = 0

    # Service info
    model_name: str = ""
    weight_type: str = ""
    weight_path: str = ""
    backend: str = ""
    base_url: str = ""
```

<a id="configuration-reference"></a>

## 配置参考

```yaml
weight_sync_controller:
  type: "weight_sync_controller"

  # Synchronization mode
  sync_mode: "sync"  # "no-sync" | "sync" | "fully-async" | "batch-async"

  # Staleness threshold (for batch-async mode)
  staleness_threshold: 2

  # Weight sync method
  sync_method: "disk"  # "network" | "disk" | "tinker" | "weaver" | "mock"

  # Weight path (for network/disk methods)
  sync_weight_path: "/path/to/weights"

  # Validation frequency (0 = disabled)
  validate_freq: 5

  # Inference service config (model info)
  inference_service:
    model_tag: "default"
    model: "model-name"
    weight_type: "full"
    backend: "sglang"
    base_url: "http://localhost:8000"
```

**`training.weight_sync.mode` 推导行为：**

| mode | `keep_batch_order` | 发布采样器权重 |
|------|--------------------|-------------------------|
| `no-sync` | `false` | 否 |
| `sync` | `true` | 是 |
| `batch-async` | `true` | 是 |
| `fully-async` | `false` | 是 |

<a id="integration-example"></a>

## 集成示例

控制器初始化并连接权重同步控制器：

```python
# In NorthRLController.__init__
self.weight_sync_controller = WeightSyncController(config.weight_sync_controller)

# Set module references
self.weight_sync_controller.set_module_references(
    dataloader=self.dataloader,
    trajectory_pool=self.trajectory_pool
)

# For Weaver backend
if config.inference_service.backend == "weaver":
    self.weight_sync_controller.set_weaver_service_holder(self.weaver_service_holder)
```

<a id="related-documentation"></a>

## 相关文档

- [核心架构](../02-core-architecture/overview.md) - 系统架构
- [轨迹池](../04-trajectory-pool/trajectory-pool.md) - 轨迹收集
- [验证](validation.md) - 验证协调
