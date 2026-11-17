English | [简体中文](../../../zh/developer-guide/11-configuration-reference/e2b-advanced.md) | [Documentation index](../../index.md)

# Advanced E2B configuration: token history and network policy

Complete a first task run with the [integration guide](../05-rollout-workers/e2b-task-integration.md) before enabling these options. See [rollout configuration](rollout-config.md) for basic execution fields/profile inheritance and [execution budgets](e2b-execution-budgets.md) for deadlines.

<a id="token-prefix-reuse"></a>

## Reuse historical tokens

`rollout.execution.token_prefix_reuse` defaults to `false`, accepts only booleans, and can be overridden by execution profiles. It supports OpenAI Chat Completions and Anthropic Messages with ordinary and streamed responses; the relay does not implement the Responses API. Before enabling it, confirm that NorthGate supports `history_tokens` for the chosen API. See the [NorthGate history-token protocol](https://northgate.xiaobei.top/docs/backend/features/weaver-sample.html#复用历史-token).

The setting remains named `token_prefix_reuse`; requests use the `history_tokens` field to reuse cached original tokens.

<a id="anthropic-normalization"></a>

## Anthropic history compatibility

The relay removes protocol `cache_control` from the Anthropic request root, direct system/message content blocks, and outer tool definitions before forwarding or history matching. This prevents Claude Code's moved cache markers from making otherwise identical histories mismatch. It happens even when token reuse is disabled. Nested business data, including tool `input` and `input_schema` fields named `cache_control`, is preserved. OpenAI requests are unaffected.

With reuse enabled, one additional normalization applies **only to the cache-comparison copy** of Anthropic assistant messages. If a content list contains `tool_use`, and every text block is before the first tool call, has exactly `type`/`text` fields, and contains a nonempty string consisting only of `\r`/`\n`, those text blocks are removed from both cache writes and lookups. SGLang's `qwen3` reasoning plus `qwen3_coder` tool parser can preserve `"\n\n"` between `</think>` and `<tool_call>`, while Claude Code drops it when replaying history. This normalization lets both histories match the same original tokens without an extra switch or a parser change in SGLang/`llm-t2t-go`.

Actual body text, spaces/tabs, empty strings, extra text-block fields, or text after a tool call keep the entire content list subject to exact comparison. Thinking, tool inputs, nested content, and OpenAI messages are not normalized. If the same cache key maps to different token histories, it remains ambiguous: the relay neither reuses it nor falls back to a shorter history.

Forwarded messages, JSON/SSE responses, recorded messages/tokens/logprobs/masks are unchanged. Reused `history_tokens` still contain the original newline tokens.

```yaml
rollout:
  execution:
    token_prefix_reuse: true
```

<a id="cache-matching"></a>

## Cache matching

The relay caches messages and actual prompt/response tokens from complete successful turns. Before the next request, it compares the API, historical message prefix, and effective request settings such as system, tools, and thinking. Anthropic's `anthropic-version` and `anthropic-beta` headers also participate.

For a unique matching token history, it sends top-level `history_tokens: [...]` containing the previous prompt tokens followed by response tokens. Otherwise, it omits the field. For example, a cached `[user, assistant]` can match the prefix of `[user, assistant, tool]` when messages and settings match. Full messages are still sent; there is no `message_count` field. Anthropic tool results remain native `tool_result` blocks inside user messages.

<a id="reuse-guards"></a>

## Conditions for reuse

Only normally completed, successfully recorded responses populate the cache:

- OpenAI finish reasons must be `stop` or `tool_calls`; Anthropic reasons must be `end_turn` or `tool_use`.
- A stream must finish completely and allow reconstruction of the complete assistant message. Truncated, failed, cancelled, or unrecognized incremental output adds no cache entry.
- OpenAI requests with nonempty `stop`, `sampling_params.stop`, or `sampling_params.stop_token_ids` do not populate the cache: their finish reason cannot prove that the required terminator was generated. Anthropic `stop_sequence` completion is also ineligible.
- `history_tokens` must be nonempty, contain nonnegative IDs, and end with the terminator generated at the prior turn's completion. Truncated output without this terminator cannot be reused.
- Keep model, tokenizer, and template settings fixed within a rollout.

<a id="server-contract"></a>

## Server constraints

NorthGate does not store conversation history; the client is responsible for the correspondence between messages and `history_tokens`. The server renders the complete request, takes the history's last token ID as its terminator, counts its occurrences in the supplied history, and cuts the rendered text after the same occurrence. It tokenizes the new messages/generation prefix after that point and appends them to the unchanged historical tokens before calling Weaver. The template must preserve the terminator count for the same history across renderings.

Invalid history, including more terminators than the rendered request contains, produces `400 invalid_request` before sampling. The relay also rejects returned `token_data.prompt_tokens` that do not preserve the supplied history prefix.

The recorder always stores actual returned tokens, log probabilities, and masks and retains strict token-prefix checks. This option changes the next model input; it is separate from [prefix compression in stored traces](../05-rollout-workers/e2b-runtime.md#trace-schema).

<a id="network-policy"></a>

## Agent network policy

The host sets `rollout.execution.network_policy`. SWE, MiMo code/terminal/general/arvo, JobBench, tmax, and AutomationBench integrate the policy after dependency/user setup in `prepare`. ATC launches from its template, which still needs a separate upgrade. ATC and other unintegrated entrypoints fail with an environment error before agent startup when given an enabled policy: a nonempty denylist or `local_only`.

<a id="network-modes"></a>

## Choose a network mode

For code tasks that do not need public networking, use `local_only`:

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

This mode blocks all non-loopback IPv4/IPv6 traffic for the agent UID and starts no outbound proxy. It preserves the `127.0.0.1` model relay and local test services, while blocking E2B's root command service on port `49983`. It is not an allowlist for arbitrary model-service domains, nor does it permit only the model port.

Trusted bootstrap/prepare can still use the network. Agent-started `npm`, `pip`, `curl`, and other downloads are restricted, so prepare dependencies beforehand. MiMo code tests use the same UID and are restricted; the SWE root evaluator is outside this policy. `local_only` cannot have nonempty `deny_domains`; an omitted/empty list does not disable it.

Workers receive policy through their configuration/source package. Editing local YAML does not update existing workers or sandboxes. Recreate them after changing policy and avoid mixing in-flight results from the old policy into new-policy statistics.

<a id="network-overrides"></a>

## Common, profile, and per-task overrides

Policies can be common, profile-specific, and task-specific. Both profiles below use the SWE entrypoint; their existing `entrypoint`, `input_fields`, and other settings are omitted:

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

A profile without `network_policy` inherits the common policy. An explicitly supplied policy replaces it completely, including `task_overrides`; there is no recursive merge. The selected policy then matches `metadata.task_id` exactly. Each task override must explicitly provide `deny_domains` or `mode: local_only` and replaces the default mode/list completely.

Thus `project__issue-123` under `swe_github` blocks only Bitbucket, while `project__issue-456` disables restrictions. A task override of `deny_domains: []` also explicitly disables an inherited `local_only` policy. A single-task recipe can set common `network_policy` directly without profiles.

Omitted `mode` means `denylist` for compatibility. Missing policy, or a final empty `deny_domains` list in `denylist` mode, disables restrictions: no proxy/firewall is installed, and agent environment/command remain unchanged. `null` is not the disable syntax. The host resolves trusted configuration and `metadata.task_id` into root-private runtime configuration; task input and the agent cannot override that resolved result.

Lists accept domains only, not URLs, ports, IP addresses, or CIDR ranges. Entries match both the domain and all subdomains; `*.github.com` equals `github.com`. Case, internationalized domains, and trailing dots are normalized and duplicates removed. The SWE recipe explicitly lists `github.com`, `githubusercontent.com`, `gitlab.com`, and `bitbucket.org`; other recipes do not automatically inherit those entries.

<a id="custom-network-entrypoints"></a>

## Integrate a custom entrypoint

Use the shared components without changing hook signatures:

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

Runtime supplies the same trusted policy context to `prepare`, `run_agent`, and `evaluate`; the entrypoint supplies the restricted username. Define `agent_user`, `environment`, and `command` in your entrypoint. The original command must start with `["runuser", "--user", agent_user, "--preserve-environment", "--"]`; enabled policy replaces this with a privilege-dropping `setpriv` command.

Integrate both preparation and launch wrappers. The agent and descendants must use the same non-root UID. MiMo code tests use that UID, so their evaluation commands and proxy environment are wrapped too.

AutomationBench uses the template's existing non-root `user` account with enabled policy, retaining its original launch behavior when policy is absent or the denylist is empty. Enabled policy prevents `sudo` escalation in every integrated entrypoint, including tmax templates that ordinarily allow sudo. Decide per task whether to enable it when root operations are required.

<a id="network-requirements"></a>

## Template and runtime requirements

Templates must provide `nft` and `setpriv`; `denylist` additionally needs `useradd` for the proxy account. `prepare` runs as root with `CAP_NET_ADMIN`. The kernel must support `nf_tables` with `ip`, `ip6`, `meta skuid`, and `reject`. Both address families are installed atomically; this does not depend on `xt_owner` or `inet`.

Shared bootstrap installs `proxy.py==2.4.10`; templates supply system tools. `local_only` neither runs a proxy nor requires proxy health checks. Proxy setup or installation/verification failure for either address family raises `RolloutEnvironmentError`. Each sandbox keeps its own mode/rules; startup checks mode, user, and rules, and cannot reuse another task's or mode's state.

In `denylist` mode, the proxy has a separate non-root UID, with root-owned rules/state. The agent cannot connect directly to the public network; HTTP/HTTPS uses the proxy on public ports 80/443. The proxy rejects listed domains, IP literals, and non-public DNS resolutions. Launch wrapping sets proxy variables for `denylist` and removes inherited proxy variables for `local_only`. Both clear supplementary groups/capabilities and enable `no_new_privs`.

Loopback remains available for the model relay and local tests except E2B's root command service on port `49983`. Removing proxy variables or using SSH cannot restore direct networking. Restrictions last until sandbox termination.

<a id="network-boundaries"></a>

## Scope of restrictions

A denylist does not cover mirrors/forwarders on permitted domains or inspect TLS SNI/HTTP Host inside CONNECT tunnels. `local_only` blocks the agent's direct connections to those public services. Neither mode restricts local Git objects or preinstalled answer copies.

Rules apply by UID. Root evaluators, separate judge accounts, and tools under other UIDs retain their networking, including requests made on the agent's behalf. This also applies to MiMo general MCP services and ARVO verification services. Review other local proxies/tool services when preserving loopback: these modes do not make the whole sandbox offline or provide complete protection against reward hacking.
