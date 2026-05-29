# Prefix 重排工具改造指导

本文档是给下一个 agent 的改造说明。它的任务不是从零发明一个新项目，而是在理解当前 `autogen_prefix_tree` 工具的基础上，重新设计下一版实现路线：从旧的 HTTP 代理模式，改造成插在 AutoGen `model_client` 前面的 prompt 重排中间件。

请下一个 agent 先读本文档，再去读旧代码。旧代码可以作为经验来源，但不要沿着旧 HTTP proxy 继续做主实现。

## 1. 当前工具是什么

当前 `autogen_prefix_tree` 的核心实现是一个 OpenAI-compatible HTTP 代理：

```text
AutoGen Agent / Team
  -> autogen_prefix_tree.proxy
  -> OpenAI-compatible API
```

它的优点是接入轻，AutoGen 只要把 `base_url` 指向本地代理，就能经过插件。代理会截取 HTTP JSON 请求，分析 `messages/tools/model`，尝试重排可共享内容，再把请求转发给云端 API。

但这个模式有一个根本限制：HTTP 代理看到的是 AutoGen 已经压平后的 OpenAI-compatible JSON，它能无损转发请求本身，却不能无损还原 AutoGen 内部语义。

具体来说，HTTP JSON 很难可靠判断：

- system prompt 中哪部分是 agent role，哪部分是 team policy。
- 哪些内容来自 memory 注入，哪些是用户任务。
- 哪些消息在 AutoGen 内部是 `SystemMessage`、`UserMessage`、`AssistantMessage` 或 tool result。
- 哪些 tool 是公共工具，哪些是 agent 私有工具。
- 哪些历史来自 group chat，哪些是当前轮指令。

因此，旧 HTTP 代理适合做早期验证，但不适合作为下一版主路线。

## 2. 要改造成什么

下一版要改成 AutoGen 原生模型调用层中间件。新的调用链应是：

```text
AutoGen Agent / Team
  -> PrefixReorderClient
  -> 原始 OpenAIChatCompletionClient
  -> 云端 API
```

也就是说，插件只出现在一个地方：原始 `OpenAIChatCompletionClient` 前面。

它要做的事情只有三件：

1. 捕获 AutoGen 即将交给模型客户端的原始 `LLMMessage`、tools 和 model args。
2. 根据三模块算法生成重排后的 messages。
3. 把重排结果交回原始 `OpenAIChatCompletionClient`，由它继续转成 HTTP 请求并调用云端 API。

它不应该：

- 修改 AutoGen 源码。
- monkey patch AutoGen 内部函数。
- 自己发 HTTP 请求。
- 替代 `OpenAIChatCompletionClient` 的模型适配能力。
- 参与 agent 创建、team 调度、tool 执行或 memory 存储。
- 修改模型响应。

这个工具本质上是“模型客户端前置中间件”，不是独立网络代理。

## 3. 当前阶段只做最小接入

第一版只考虑最简单、最直接的接入方式：写一个插件类，暴露一个接口，把它接到原本工程项目的 `model_client` 前面。

```text
inner_client = OpenAIChatCompletionClient(...)
model_client = PrefixReorderClient(inner_client)
agent = AssistantAgent(..., model_client=model_client)
```

这里的 `PrefixReorderClient` 不是新的模型客户端实现，也不是新的 HTTP 服务。它只是一个前置包装层，负责在每次模型调用前做这条链路：

```text
接收原始 model_client 调用参数
  -> 获取原始 prompt / LLMMessage / tools / model args
  -> 交给 Local Prompt Compiler 构造 IR
  -> 交给 Hierarchical Prefix Planner 生成重排计划
  -> 交给 Cache-Utility Validator 做安全与一致性验证
  -> 如果验证通过，把重排后的 messages 交还给 inner_client
  -> 如果验证失败，把原始 messages 原样交还给 inner_client
```

也就是说，原工程里原本是：

```text
Agent -> OpenAIChatCompletionClient -> HTTP API
```

改造后变成：

```text
Agent -> PrefixReorderClient -> OpenAIChatCompletionClient -> HTTP API
```

`OpenAIChatCompletionClient` 仍然负责把 messages 转成 HTTP 请求并发送给云端 API。`PrefixReorderClient` 只负责在它前面拿到原始 prompt、按三模块算法调整顺序、再把结果还给它。

第一版不要设计复杂的接入体系。暂时不做：

- 配置式 provider。
- 工厂函数封装。
- 多框架统一抽象层。
- 独立 HTTP proxy 服务。

这些可以作为后续扩展。第一版只要把 `PrefixReorderClient(inner_client)` 这条路线设计清楚，并确保它能稳定接到现有 AutoGen 项目的 `model_client` 前面。

## 4. 三模块主线

下一版仍然沿用 `AgentTeam共享Prefix.pdf` 里的三模块设计：

```text
Local Prompt Compiler
  -> Hierarchical Prefix Planner
  -> Cache-Utility Validator
```

### 4.1 Local Prompt Compiler

职责：把 AutoGen 原始模型输入转换成 block-level semantic IR。

输入来自 `PrefixReorderClient` 捕获到的模型调用参数，至少包括：

- `LLMMessage` 序列。
- tools 参数。
- model 调用参数。
- 可获得的 agent/source 信息。

Compiler 要识别：

- block 边界。
- block 语义类型。
- block 是否跨 agent 共享或重复。
- block 是否适合前置。
- block 的潜在风险。

IR 中每个 block 至少应包含：

- `block_id`
- `source_message_id`
- `source_role`
- `source_type`
- `agent_or_source`
- `text_or_payload`
- `content_hash`
- `semantic_type`
- `movability`
- `share_scope`
- `risk_tags`
- `original_position`

推荐语义类型：

- `global_task_background`
- `shared_context`
- `team_policy`
- `shared_tool_description`
- `role_identity`
- `private_memory`
- `private_tool_permission`
- `conversation_history`
- `tool_result`
- `current_user_instruction`
- `current_turn_instruction`
- `output_format`
- `unknown`

推荐移动性：

- `safe_prefix`
- `conditional_prefix`
- `local_only`
- `order_sensitive`
- `never_move`

Compiler 可以使用 marker，但不能只依赖 marker。AutoGen 试点版本应优先利用 `LLMMessage` 类型、message source/name、tools 参数等结构化信息。

如果无法可靠识别某段内容，就保守标记为 `order_sensitive` 或 `local_only`。

### 4.2 Hierarchical Prefix Planner

职责：根据 semantic IR 生成重排计划。

Planner 的概念结构仍然是：

```text
global shared prefix
  -> subgroup / phase shared prefix
  -> agent-specific suffix
```

但第一版不需要实现复杂树优化。先做保守规则即可：

- 全局共享、稳定、低风险的 block 可以前置。
- 子组共享内容可以作为后续扩展，第一版只保留设计口径。
- agent 私有、顺序敏感、高风险内容必须留在原位置或 agent-specific 区域。

Planner 应输出 plan，而不是直接改消息。plan 至少包含：

- 哪些 block 被移动。
- 每个 block 的原位置和新位置。
- 移动原因。
- 风险标签。
- 是否需要回退为 no-op。

默认禁止移动：

- agent role identity。
- 最新用户指令。
- 当前轮指令。
- assistant/tool 历史顺序。
- tool result。
- 私有 memory。
- 私有 tool 权限。
- 未识别的大段混合内容。

### 4.3 Cache-Utility Validator

职责：验证重排是否可接受。

第一版至少做静态验证：

- 重排前后 block hash multiset 必须一致。
- tools hash 必须一致。
- model args hash 必须一致。
- 禁止移动的 block 没有被非法移动。
- 没有新增解释性文本。
- 没有删除内容。
- 没有改写内容。

验证失败时必须回退原始 messages，并把回退原因写入 trace。

后续可以扩展缓存收益估计和任务效用验证：

- baseline vs rewritten prefix hit proxy。
- provider cached tokens。
- latency。
- 任务成功率。
- tool 调用正确性。
- 角色遵循度。
- 输出格式一致性。

## 5. Strict Reorder 是第一版核心

第一版的核心不是最大化 cache hit，而是做干净的控制变量实验。

Strict reorder 必须满足：

- 不改写原文。
- 不删除原文。
- 不新增解释性文本。
- 不改变 tools。
- 不改变 model 参数。
- 不改变 response 处理。
- 不改变 agent/team 调度。
- 不改变云端 API endpoint。
- 尽量保持 message 边界，除非后续明确设计非 strict 模式。

也就是说，第一版要尽量做到：

```text
内容不变，参数不变，tools 不变，只改变允许移动 block 的顺序。
```

特别注意：旧 HTTP 代理中的 `PREFIX_TREE_LAYOUT_START`、`GLOBAL_PREFIX_START` 等包装标签，不应出现在 strict mode 里，因为它们会改变模型看到的内容。

## 6. 旧代码怎么参考

当前目录里的旧实现可以作为参考材料：

- `compiler.py`：参考 block IR 和 marker 分类经验。
- `planner.py`：参考 prefix tree/global-subgroup-suffix 思路。
- `cache_oracle.py`：参考本地 cache hit 估计方式。
- `proxy.py`：仅作为旧 HTTP 代理参考，不作为新版主入口。

但是，下一个 agent 不应继续沿着 HTTP proxy 做主实现。

另外，`cache_hit_proxy` 是另一个独立仓库，不属于本项目。后续如果需要复用它，应作为外部依赖或独立引用处理，不要把它当作新仓库的一部分直接混入。

## 7. 推荐最小文件结构

第一版可以保持很小：

```text
multiagent_prefix_recoder/
  autogen_client.py      # PrefixReorderClient，包装 inner model client
  compiler.py            # Local Prompt Compiler
  planner.py             # Hierarchical Prefix Planner
  validator.py           # Cache-Utility Validator
  types.py               # IR / plan / validation dataclass
  trace.py               # hash 与 trace 记录
```

如果暂时仍在旧 `autogen_prefix_tree` 目录下推进，也可以先用类似命名：

```text
autogen_prefix_tree/
  native_client.py
  native_compiler.py
  native_planner.py
  native_validator.py
  native_types.py
  native_trace.py
```

概念上，新项目最终应面向多 multi-agent 框架；AutoGen 只是第一阶段适配对象。

## 8. Trace 要求

默认不保存 raw prompt。

默认 trace 保存 hash、计数、移动计划和验证结果即可。建议字段：

- `request_id`
- `agent_id`
- `framework = autogen`
- `inner_client_type`
- `transformation_applied`
- `fallback_applied`
- `fallback_reason`
- `original_message_count`
- `rewritten_message_count`
- `block_count`
- `moved_block_count`
- `forbidden_move_count`
- `hash_multiset_equal`
- `tools_hash_equal`
- `model_args_hash_equal`
- `estimated_cached_tokens`
- `estimated_cache_hit_rate`

只有显式 debug 模式才允许保存 raw content。

## 9. 第一版验收标准

下一个 agent 设计详细计划时，应至少覆盖这些验收项：

- 实现一个 `PrefixReorderClient(inner_client)`。
- 不修改 AutoGen 源码。
- 不自己发 HTTP。
- `create()` 能捕获原始 `LLMMessage` 并委托 inner client。
- 如 AutoGen 需要，也考虑 `create_stream()` 的委托路径。
- 能构造 block-level semantic IR。
- 能生成保守重排 plan。
- Validator 能检查 hash multiset、tools、model args 不变。
- 验证失败自动回退原始 messages。
- trace 默认不保存 raw prompt。
- strict mode 不新增包装文本。
- 旧 HTTP proxy 不再作为主实现路线。

## 10. 交付位置

用户已经建立新仓库：

```text
git@github.com:2426664247/Multiagent-prefix-recoder.git
```

后续正式改造工作应放到这个仓库中。当前仓库/目录中的旧 `autogen_prefix_tree` 只作为历史参考和可借鉴代码来源。

## 11. 一句话总结

下一个 agent 要做的是：把现有 prefix 重排工具从“HTTP 层代理”改造成“AutoGen model client 前置中间件”。第一版只实现 `PrefixReorderClient(inner_client)` 这一条接入路线，通过 Local Prompt Compiler、Hierarchical Prefix Planner、Cache-Utility Validator 三模块，在模型调用前完成严格、保守、可验证的 prompt block 顺序调整，然后把结果交还给原始 `OpenAIChatCompletionClient` 继续请求云端 API。
