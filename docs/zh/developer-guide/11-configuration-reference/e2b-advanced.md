[English](../../../en/developer-guide/11-configuration-reference/e2b-advanced.md) | 简体中文 | [文档目录](../../index.md)

# E2B 高级配置：历史 token 与网络策略

先使用[接入指南](../05-rollout-workers/e2b-task-integration.md)完成一次任务训练，再按需要启用本文配置。基础 execution 字段和 profile 规则见[Rollout 配置](rollout-config.md)；时限见[执行时限与恢复](e2b-execution-budgets.md)。

<a id="token-prefix-reuse"></a>

## 历史 token 前缀复用

`rollout.execution.token_prefix_reuse` 默认 `false`，只接受布尔值，execution profile
可覆盖。此功能支持 OpenAI Chat Completions 和 Anthropic Messages 接口的普通响应与
流式响应；relay 不提供 Responses 接口。开启前须确认 NorthGate 已支持所用接口的
`history_tokens`；协议约定见
[NorthGate：复用历史 token](https://northgate.xiaobei.top/docs/backend/features/weaver-sample.html#复用历史-token)。
配置名仍为 `token_prefix_reuse`，relay 通过 `history_tokens` 复用缓存的历史 tokens。

<a id="anthropic-normalization"></a>

## Anthropic 历史兼容处理

relay 会先移除 Anthropic 请求顶层、system/message 的直接内容块和工具定义外层
的 `cache_control`，避免 Claude Code 移动缓存标记后使相同历史匹配失败。这些显式
prompt-cache 指令不会转发给 NorthGate；未开启历史 token 复用时也执行清理。
工具 input、input_schema 及其他嵌套业务数据中的同名字段会保留，OpenAI 请求不受影响。

开启 token 复用后，Anthropic assistant 历史还有一项仅用于缓存比较的兼容处理：
content 列表含 `tool_use`，且所有 text 块都在第一个 `tool_use` 之前、字段只有
`type`/`text`、正文非空且仅含 `\r`/`\n` 时，写入和查询缓存都会从比较用副本中
移除这些 text 块。SGLang 的 `qwen3` reasoning + `qwen3_coder` tool parser 会将
`</think>` 与 `<tool_call>` 之间的 `"\n\n"` 保留为 content，Claude Code 回填历史
时可能删除对应 text 块；此兼容处理使两种历史仍能命中相同的原始 tokens。
无需额外开关，也不改变 SGLang/`llm-t2t-go` 的解析结果。

真实正文、空格/tab、空字符串、额外 text 块字段或工具调用之后的 text 都会保留
整个 content 列表的精确比较；thinking、工具 input、嵌套内容和 OpenAI messages
也不归一化。相同缓存键若对应不同 tokens，仍标记歧义，不复用也不回退到更短历史。
转发的 messages、JSON/SSE 响应、录制的 messages/tokens/logprobs/mask 均不受此
处理影响；复用的 `history_tokens` 仍包含原始换行 tokens。

```yaml
rollout:
  execution:
    token_prefix_reuse: true
```

<a id="cache-matching"></a>

## 缓存匹配

NorthRL relay 保存完整成功轮次的 messages 和实际 prompt/response tokens。下一轮
请求前，比较接口、历史 messages 前缀及 system、tools、thinking 等请求设置；
Anthropic 的 `anthropic-version`、`anthropic-beta` 请求头也参与匹配。
有唯一的 token 历史时，发送
顶层字段 `history_tokens: [...]`，值为上一轮实际 prompt tokens 与 response tokens
顺序拼接后的完整数组，否则省略该字段。例如缓存了
`[user, assistant]` 对应的原始 tokens，收到 `[user, assistant, tool]` 时，前两条
消息和请求设置相同即可携带该历史；完整 messages 仍随请求发送，不发送 `message_count`。
Anthropic 工具结果仍保留在原生 user 消息的 `tool_result` 内容块中。

<a id="reuse-guards"></a>

## 允许复用的条件

只有正常完成且成功录制的响应可供后续复用：OpenAI 的完成原因须为 `stop` 或
`tool_calls`，Anthropic 须为 `end_turn` 或 `tool_use`。流式响应还须完整结束，且能完整
重建 assistant 消息。截断、失败、取消或无法识别的增量不会产生新的缓存条目。
OpenAI 请求若配置了非空的 `stop`、`sampling_params.stop` 或
`sampling_params.stop_token_ids`，响应不写入复用缓存，因为完成原因不能证明模型输出了
所需结束符；Anthropic 的 `stop_sequence` 完成原因也不可复用。
`history_tokens` 必须非空、各 ID 非负，最后一个 ID 须为上一轮模型停止生成时输出的
结束符；未以结束符结尾的截断输出不能复用。一次 rollout 内须固定模型、tokenizer
和模板配置。

<a id="server-contract"></a>

## 服务端约束

NorthGate 不保存会话历史，消息与 `history_tokens` 的对应关系由客户端保证。服务端
先渲染完整请求，以 `history_tokens` 的最后一个 ID 为结束符，统计它在该数组中出现的
次数 n，再从渲染结果的第 n 个结束符之后切开。切点之后的新消息与生成前缀编码后，
接在原样保留的 `history_tokens` 之后交给 Weaver。同一历史在两轮渲染中的结束符个数
须一致；使用的模板须满足这一约束。

若历史不满足约束，或其中结束符多于渲染结果中的结束符，NorthGate 在采样前返回
`400 invalid_request`。relay 也会拒绝未保留所传历史的 `token_data.prompt_tokens`。

recorder 始终记录实际返回的 tokens、logprobs 和 mask，沿用严格 token 前缀检查。
此开关改变模型下一轮的实际输入，与[轨迹落盘的前缀压缩](../05-rollout-workers/e2b-runtime.md#trace-schema)无关。

<a id="network-policy"></a>

## Agent 网络策略

网络规则由宿主的 `rollout.execution.network_policy` 配置。SWE、MiMo code／terminal／general／arvo、JobBench、tmax 和 AutomationBench 已接入，在 `prepare` 完成依赖与用户准备后启用规则。ATC 的启动逻辑位于模板内，现有模板尚未接入，需要单独升级；它及其他未接入入口收到启用的策略（非空黑名单或 `local_only`）时，runtime 会在启动 agent 前报环境错误。

<a id="network-modes"></a>

## 选择网络模式

对于无需公网的代码任务，使用 `mode: local_only`：

```yaml
rollout:
  execution:
    profiles:
      swe_rebench:
        network_policy:
          mode: local_only
      mimo_code:
        network_policy:
          mode: local_only
```

该模式禁止 agent UID 的全部非 loopback IPv4/IPv6 流量，不启动外网转发代理；保留 `127.0.0.1` 模型网关和本地测试服务，仍封禁 E2B root 命令服务的 49983 端口。它不是任意模型服务域名的白名单，也不只允许模型端口。可信 bootstrap／prepare 仍可联网；agent 启动后自行执行的 npm、pip、curl 等下载均受限制，所需依赖必须提前准备。MiMo code 的测试进程使用同一 UID，也受限制；SWE 的 root evaluator 不受该 UID 策略限制。`local_only` 不允许搭配非空 `deny_domains`，省略名单或空名单都不会关闭该模式。

策略随 worker 的配置和源码包分发，编辑本地 YAML 不会热更新已创建的 worker 或 sandbox。切换模式后需重新创建相关 worker／sandbox，并避免旧策略的在飞 rollout 混入新策略统计。

<a id="network-overrides"></a>

## 公共、profile 与逐题覆盖

可设置公共策略、profile 策略和逐题覆盖。下面两个 profile 均使用 SWE 入口，省略其原有的 `entrypoint`、`input_fields` 等配置：

```yaml
rollout:
  execution:
    network_policy:
      deny_domains: [github.com, githubusercontent.com]
    profiles:
      swe_github:
        network_policy:
          deny_domains: [github.com, githubusercontent.com, gitlab.com]
          task_overrides:
            project__issue-123:
              deny_domains: [bitbucket.org]
            project__issue-456:
              deny_domains: []
      swe_gitlab:
        network_policy:
          deny_domains: [gitlab.com]
```

profile 未配置 `network_policy` 时继承公共策略；显式配置时替换整个策略，包括 `task_overrides`，不递归合并。之后按 `metadata.task_id` 精确匹配逐题覆盖，每项须显式包含 `deny_domains` 或设置 `mode: local_only`，完整替换默认模式和名单。因此 `swe_github` 中的 `project__issue-123` 只禁止 Bitbucket，`project__issue-456` 关闭限制；在 `local_only` 下逐题覆盖为 `deny_domains: []` 也会显式解除该题限制。单一任务无需 profiles，直接在公共 `network_policy` 下配置即可。

省略 `mode` 时使用兼容现有配置的 `denylist` 模式。缺少策略，或 `denylist` 下最终 `deny_domains: []`，表示关闭：不安装代理或防火墙，也不改变 agent 环境和启动命令；`null` 不是关闭语法。宿主从可信配置和 `metadata.task_id` 选择模式和名单，只将本题的结果写入 root 私有运行配置，题目输入和 agent 不能覆盖它。

名单只接受域名，不接受 URL、端口、IP 或 CIDR。每项同时匹配域名本身和全部子域，`*.github.com` 与 `github.com` 等价；大小写、国际化域名和末尾的点会规范化，重复项会去重。SWE recipe 显式配置 `github.com`、`githubusercontent.com`、`gitlab.com`、`bitbucket.org`，其他 recipe 不会自动继承这四项。

<a id="custom-network-entrypoints"></a>

## 接入自定义任务入口

自定义入口使用公共组件接入，不需要改变任务 hook 的签名：

```python
from northrl.remote_rollout.network_policy import (
    agent_network_command,
    agent_network_environment,
    prepare_network_policy,
)


async def prepare(input_data, workspace):
    # Install dependencies and create the non-root agent_user first.
    await prepare_network_policy(agent_user)


async def run_agent(input_data, workspace, model_environment):
    # Build your agent's environment and runuser command here, then wrap them.
    agent_environment = agent_network_environment(environment, agent_user)
    agent_command = agent_network_command(command, agent_user)
    # Launch with agent_environment/agent_command, then clean up descendants.
```

runtime 为 prepare、run_agent 和 evaluate 提供相同的可信策略上下文，入口只传入受限用户名。示例中的 `agent_user`、`environment` 和 `command` 由入口定义；`command` 须以 `["runuser", "--user", agent_user, "--preserve-environment", "--"]` 开头，启用策略后替换为清除权限的 `setpriv` 命令。prepare 和启动包装必须同时接入，agent 及其子进程使用同一非 root UID。MiMo code 的测试使用该 UID，因此 evaluate 同样包装测试命令和代理环境。

AutomationBench 在启用策略时使用模板已有的非 root `user` 账户运行 agent；未配置策略或 `denylist` 为空时保留原启动方式。启用策略后，所有已接入入口的 agent 都不能通过 `sudo` 提权，包括原本允许 sudo 的 tmax 模板；需要 root 操作的题目应按任务配置决定是否启用。

<a id="network-requirements"></a>

## 模板与运行环境要求

模板必须提供 `nft`、`setpriv`，`denylist` 另需 `useradd` 创建代理用户；prepare 以 root 运行，拥有 `CAP_NET_ADMIN`，内核支持 `nf_tables` 的 `ip`、`ip6`、`meta skuid` 和 `reject`。两套地址族规则原子安装，不依赖 `xt_owner` 或 `inet`。公共 bootstrap 安装 `proxy.py==2.4.10`，系统工具由模板提供；`local_only` 不运行代理，也不依赖代理健康检查。代理或任一地址族规则安装、核验失败时，prepare 报 `RolloutEnvironmentError`。每个 sandbox 独立保存规则及模式状态，启动前核对模式、用户和规则，不能重用另一任务或另一模式的状态。

`denylist` 代理使用独立的非 root UID，规则和状态由 root 管理。agent 公网直连被禁止，HTTP/HTTPS 经代理访问公网 80/443；代理拒绝黑名单、IP 字面量和非公网解析结果。启动包装在 `denylist` 下设置代理变量，在 `local_only` 下移除继承的代理变量；两种模式都会清除附加组和 capabilities、启用 `no_new_privs`。loopback 保留模型 gateway 和本地测试访问，单独拒绝 E2B root 命令服务的 49983 端口。删除代理变量或改用 SSH 不能恢复直连，限制持续到 sandbox 销毁。

<a id="network-boundaries"></a>

## 限制的作用范围

黑名单不覆盖允许域名上的镜像或转发服务，不解析 CONNECT 隧道内的 TLS SNI／HTTP Host；`local_only` 会阻断 agent 直连这些公网服务。两种模式都不限制本地 Git 对象或预装的答案副本。规则按 UID 生效，root evaluator、独立 judge 用户及其他 UID 的工具服务保留原有网络权限，包括这些服务代 agent 发出的请求；MiMo general 的 MCP 服务和 ARVO 的验证服务也遵循这一边界。保留 loopback 时需审查模板里的其他本地代理／工具服务，不能把该模式视为整个 sandbox 离线或完整的 reward hack 防护。
