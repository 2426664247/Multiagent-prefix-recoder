# Prefix 重排工具改造指导

本文档是给下一个 agent 的改造说明。它要指导下一步如何把现有 prefix 重排工具，从旧的 HTTP 代理模式，改造成插在 AutoGen `model_client` 前面的 prompt 重排插件。

下一个 agent 不需要迁移旧实现。旧代码已经通过 git 历史保留，正式实现可以从干净工作区重新开始。当前最重要的不是复用旧代码，而是把三模块主线设计清楚：`Local Prompt Compiler -> Hierarchical Prefix Planner -> Cache-Utility Validator`。

## 1. 当前工具的问题

旧工具是 OpenAI-compatible HTTP 代理：

```text
AutoGen Agent / Team
  -> autogen_prefix_tree.proxy
  -> OpenAI-compatible API
```

这个方案接入轻，只要把 AutoGen 的 `base_url` 指向本地代理即可。但 HTTP 代理只能看到 AutoGen 已经压平成 OpenAI-compatible JSON 之后的请求。它能无损转发 HTTP 请求本身，却不能无损还原 AutoGen 内部语义。

典型丢失的信息包括：

- system prompt 中哪部分是 agent role，哪部分是 team policy。
- 哪些内容来自 memory 注入，哪些是用户任务。
- 哪些消息在 AutoGen 内部是 `SystemMessage`、`UserMessage`、`AssistantMessage` 或 tool result。
- 哪些 tool 是公共工具，哪些是 agent 私有工具。
- 哪些历史来自 group chat，哪些是当前轮指令。

因此，旧 HTTP 代理可以作为历史记录或早期验证参考，但不应作为新版主路线。

## 2. 新版工具的接入形态

下一版要做成 AutoGen 模型调用层前置插件：

```text
AutoGen Agent / Team
  -> PrefixReorderClient
  -> 原始 OpenAIChatCompletionClient
  -> 云端 API
```

第一版只考虑这一种最小接入方式：

```text
inner_client = OpenAIChatCompletionClient(...)
model_client = PrefixReorderClient(inner_client)
agent = AssistantAgent(..., model_client=model_client)
```

`PrefixReorderClient` 不是新的模型客户端实现，也不是 HTTP 服务。它只是一个前置包装层：

```text
接收原始 model_client 调用参数
  -> 获取原始 prompt / LLMMessage / tools / model args
  -> 交给 Local Prompt Compiler 构造 IR
  -> 交给 Hierarchical Prefix Planner 生成重排计划
  -> 交给 Cache-Utility Validator 验证
  -> 验证通过：把重排后的 messages 交给 inner_client
  -> 验证失败：把原始 messages 原样交给 inner_client
```

新版插件不应该：

- 修改 AutoGen 源码。
- monkey patch AutoGen 内部函数。
- 自己发送 HTTP 请求。
- 参与 agent 创建、team 调度、tool 执行或 memory 存储。
- 修改模型响应。
- 第一版设计 provider、工厂函数、多框架抽象层或独立 HTTP proxy 服务。

## 3. 三模块主线

三模块是这个工具的核心。下一个 agent 的详细计划应围绕这三个模块展开，而不是围绕旧代理代码展开。

```text
Local Prompt Compiler
  -> Hierarchical Prefix Planner
  -> Cache-Utility Validator
```

当前讨论后的三模块定义必须明确：

- `Compiler`：把 prompt 拆成带语义和风险属性的 IR 块。
- `Planner`：把可共享、可前置的 IR 块组织成 prefix tree。
- `Validator`：检查这棵树是否安全、是否有收益、是否应该回退。

这里的 IR block 不是“已经决定要复用的块”，而是“可供复用判断和重排规划的候选语义块”。每个 IR block 都要携带语义类型、来源、共享范围、可移动性、风险标签和内容指纹。Planner 再从这些候选块中挑选真正适合进入 prefix tree 的内容。

目标 prefix tree 的直觉结构是：

```text
root: 所有 agent 共享的全局 prompt
  -> subgroup node: 某些 agent 子组共享的 prompt
    -> leaf: 单个 agent 的特殊内容
```

越靠近 root 的 block，被更多 agent 和更多请求复用；越靠近 leaf 的 block，越偏向 agent-specific。每次发请求时，都按 root -> subgroup -> leaf 的顺序组织 prompt，从而让不同 agent 的请求在开头形成尽可能长的一致前缀。

这里追求的不是无约束的最大 cache hit，而是在语义安全、可移动性、工具权限和消息顺序约束下的最大可用 prefix cache 收益。

### 3.1 Local Prompt Compiler

`Local Prompt Compiler` 负责把 AutoGen 原始模型输入拆成 block-level semantic IR。它回答的问题是：当前请求里有哪些上下文块，每个块是什么语义，来自哪里，是否可能共享，是否可能前置，有什么风险。

Compiler 不决定最终 prefix tree，也不直接移动内容。它只产出结构化 IR，让后续 Planner 能判断哪些块有复用价值、哪些块必须保守处理。

**输入**

来自 `PrefixReorderClient` 捕获到的模型调用参数：

- `LLMMessage` 序列。
- tools 参数。
- model 调用参数。
- 可获得的 agent/source 信息。
- 可获得的 session/team 标识。

**输出**

输出一组语义 block。每个 block 至少包含：

- `block_id`：稳定唯一标识。
- `source_message_id`：来自哪条原始 message。
- `source_role`：system/user/assistant/tool 等角色。
- `source_type`：AutoGen 内部消息类型。
- `agent_or_source`：消息来源 agent 或 source。
- `text_or_payload`：文本或结构化 payload。
- `content_hash`：内容指纹，用于验证内容未被改写。
- `semantic_type`：语义类型。
- `movability`：可移动性。
- `share_scope`：共享范围判断。
- `risk_tags`：风险标签。
- `original_position`：原始位置。

**推荐语义类型**

- `global_task_background`：全局任务背景。
- `shared_context`：多个 agent 共享的上下文。
- `team_policy`：团队协作规则。
- `shared_tool_description`：公共工具说明。
- `role_identity`：agent 身份、职责、人设。
- `private_memory`：私有记忆或私有状态。
- `private_tool_permission`：私有工具权限。
- `conversation_history`：历史对话。
- `tool_result`：工具执行结果。
- `current_user_instruction`：最新用户指令。
- `current_turn_instruction`：当前轮指令。
- `output_format`：输出格式要求。
- `unknown`：无法可靠识别的内容。

**推荐可移动性**

- `safe_prefix`：可以进入共享前缀。
- `conditional_prefix`：满足条件时可以前置。
- `local_only`：只能保留在本 agent 局部区域。
- `order_sensitive`：顺序敏感，默认不移动。
- `never_move`：禁止移动。

**Compiler 的设计要求**

- 第一版优先使用 AutoGen typed `LLMMessage`、message source/name、tools 参数等结构化信息。
- marker 可以作为辅助，但不能作为唯一依据。
- 如果一个 system message 内部混合了 role、policy、task、format 等内容，Compiler 要尝试切分 block。
- 如果无法可靠切分，整段内容应保守标为 `order_sensitive` 或 `local_only`。
- 对 private memory、role identity、private tool permission、latest/current instruction 要默认保守。
- Compiler 不直接重排 prompt，只生成 IR。

### 3.2 Hierarchical Prefix Planner

`Hierarchical Prefix Planner` 负责把可共享、可前置的 IR block 组织成 prefix tree。它回答的问题是：哪些 block 可以进入 root 全局前缀，哪些 block 只适合进入子组前缀，哪些 block 必须留在 agent-specific suffix。

Planner 的概念结构仍然是：

```text
global shared prefix
  -> subgroup / phase shared prefix
  -> agent-specific suffix
```

这棵树表达的是共享范围：root 一定是所有相关 agent 都相同的全局 prompt；root 之后可以接若干子组共享 prompt；最后叶子节点保留每个 agent 的特殊内容。越靠近树根，block 被使用的次数越多，也越可能形成跨 agent 的 prefix cache 命中。

**输入**

- 当前请求的 IR blocks。
- 当前 session 已观察到的历史 block 指纹和来源。
- Compiler 给出的 `semantic_type`、`movability`、`share_scope`、`risk_tags`。
- 可选的 token 长度估计或 cache benefit 估计。

**输出**

Planner 不直接改 messages，而是输出 plan。plan 至少包含：

- `moved_blocks`：被移动的 block 列表。
- `kept_blocks`：保持原位的 block 列表。
- `new_order`：重排后的 block 顺序。
- `move_reason`：每个移动的理由。
- `risk_notes`：风险说明。
- `fallback_required`：是否建议回退。
- `fallback_reason`：回退原因。

**默认禁止移动**

- agent role identity。
- 最新用户指令。
- 当前轮指令。
- assistant/tool 历史顺序。
- tool result。
- 私有 memory。
- 私有 tool 权限。
- 未识别的大段混合内容。

**Planner 的保守策略**

- 没有明确共享证据，不移动。
- 有共享证据但有角色、权限、隐私风险，不移动。
- 移动后会跨越最新用户指令，不移动。
- 移动后会改变 tool call / tool result 对应关系，不移动。
- 只对 block 做顺序规划，不改写 block 内容。
- 生成后的 prompt 顺序应体现 root -> subgroup -> leaf 的路径展开。

**需要特别想清楚的问题**

- 共享范围如何判断：完全相同 hash、规范化后相同、还是语义等价。
- 第一版是否只支持 exact hash sharing，语义等价留到后续。
- 如何处理第一条冷启动请求：没有历史共享证据时是否只记录不重排。
- 多 agent 调用顺序不同，如何避免把后出现的私有内容错误提升为全局内容。
- Planner 是否需要输出可解释理由，方便用户审查。

### 3.3 Cache-Utility Validator

`Cache-Utility Validator` 负责检查 Planner 构建出的 prefix tree 是否可以采用。它不是简单的日志模块，而是安全阀和反馈器。它回答的问题是：这棵树是否安全，是否可能带来缓存收益，是否应该回退或要求 Planner 重建。

第一版 Validator 重点做静态验证和回退机制。任务效用评估可以先留接口。

**输入**

- 原始 messages。
- Compiler 生成的 IR。
- Planner 生成的 plan。
- 根据 plan 生成的 rewritten messages。
- 原始 tools 和 model args。

**必须验证**

- 重排前后 block hash multiset 一致。
- tools hash 一致。
- model args hash 一致。
- 禁止移动的 block 没有被移动。
- 没有新增解释性文本。
- 没有删除内容。
- 没有改写内容。
- message role 和必要边界没有被意外破坏。

**回退规则**

只要出现以下情况，就必须回退原始 messages：

- hash 不一致。
- tools 或 model args 被修改。
- Planner 移动了 `never_move` / `local_only` / `order_sensitive` block。
- rewritten messages 无法被 inner client 接受。
- Compiler 或 Planner 抛出异常。
- 无法证明重排只改变允许移动 block 的顺序。

**缓存收益判断**

第一版可以先做轻量估计，不必依赖外部服务：

- 记录重排前后的 prefix 相同长度。
- 记录历史请求中可复用的 block hash。
- 记录理论上可能命中的 shared prefix token 数。

后续可以再接真实 provider 的 cached tokens、latency 和任务质量评估。

**需要特别想清楚的问题**

- Validator 是 fail closed：不确定就回退。
- 回退也要被记录，方便后续知道是 Compiler、Planner 还是验证失败。
- 验证逻辑要独立于 Planner，不能 Planner 说安全就直接信任。
- 验证报告要能让用户知道为什么这次没有重排。

## 4. 工作区与旧代码处理

下一步可以直接清空当前实现工作区，从一个干净的 Python 项目结构开始。不要为了兼容旧 HTTP 代理而保留旧目录结构，也不要把旧代码搬进新实现里。

旧实现的历史价值通过 git 记录保留即可。后续 agent 如果确实需要回看旧思路，可以通过版本历史或旧仓库记录查看，而不是在新实现里继续引用旧模块。

新的实现应以 `PrefixReorderClient(inner_client)` 为中心，从零组织文件、类型和测试。

`cache_hit_proxy` 是另一个独立仓库，不属于本项目。后续如果需要复用它，应作为外部依赖或独立引用处理，不要把它当作新仓库的一部分直接混入。

代码注释要求：

- 关键设计注释、复杂逻辑注释、TODO 注释尽量使用中文，方便用户直接审阅。
- 不需要给显而易见的代码写空泛注释。
- 重要边界条件要用中文说明为什么不能移动某些 block。

## 5. 第一版验收标准

验收标准要围绕三模块和 wrapper 链路设计。

**PrefixReorderClient**

- 能包装一个已有 `OpenAIChatCompletionClient` 或兼容 inner client。
- 不修改 AutoGen 源码。
- 不自己发送 HTTP。
- `create()` 能捕获原始 `LLMMessage`、tools、model args，并最终委托 inner client。
- 如 AutoGen 需要，`create_stream()` 也要有清晰委托策略。
- 任一模块失败时，能原样回退到 inner client。

**Local Prompt Compiler**

- 能把一次 AutoGen 模型调用转换成 block-level IR。
- IR 中包含来源、语义类型、移动性、hash、原始位置。
- 无法识别的内容会保守标记为不可移动。
- tools、model args、message source 等结构化信息会被纳入判断。

**Hierarchical Prefix Planner**

- 能接收 IR 并输出可解释 plan。
- plan 中清楚标明哪些 block 移动、哪些保留、为什么。
- 默认禁止移动 role identity、latest instruction、tool result、private memory、private tool permission、未知混合内容。
- 第一版允许只支持 exact hash sharing，不强求语义等价。

**Cache-Utility Validator**

- 能验证重排前后 block hash multiset 一致。
- 能验证 tools 和 model args 不变。
- 能发现非法移动并回退。
- 能输出回退原因。
- 不确定时默认回退。

**工程质量**

- 关键注释使用中文，特别是安全边界、回退原因和 block 不可移动规则。
- 不提交实验输出、密钥、虚拟环境、外部仓库。
- 有清晰 git commit 记录。
- 第一版 MVP 完成后打一个明确 tag。

## 6. Git 版本控制要求

后续 agent 必须认真做版本控制。这个项目后续会继续迭代，多阶段 commit 和 tag 很重要。

**基础规则**

- 开始正式实现前，先确认当前 `main` 干净。
- 如果要清空旧工作区，先确认旧版本已经在远端仓库和 tag 中可追溯。
- 每完成一个清晰阶段就提交一次 commit，不要把大量无关改动堆到一个提交里。
- commit message 可以用中文或英文，但必须清楚说明目的。
- 不要提交密钥、配置私密信息、虚拟环境、实验输出、缓存目录或外部仓库。

**建议 commit 阶段**

- `docs: clarify module design`：完善三模块设计文档。
- `chore: reset workspace for native client`：清理旧结构，准备干净项目。
- `feat: add PrefixReorderClient skeleton`：建立 wrapper 骨架。
- `feat: add prompt compiler IR`：实现 Compiler 和 IR。
- `feat: add conservative prefix planner`：实现 Planner。
- `feat: add validation fallback`：实现 Validator 和回退。
- `test: add autogen wrapper smoke tests`：补最小测试。

**建议 tag**

- `guide-v1`：当前指导文档版本。
- `autogen-wrapper-plan-v1`：详细实现计划完成。
- `autogen-wrapper-skeleton`：wrapper 骨架完成。
- `compiler-ir-mvp`：Compiler/IR 最小版完成。
- `planner-mvp`：Planner 最小版完成。
- `validator-mvp`：Validator 最小版完成。
- `autogen-wrapper-mvp`：三模块打通的第一版 MVP。

tag 名称可以调整，但必须语义清楚。

## 7. 交付位置

用户已经建立新仓库：

```text
git@github.com:2426664247/Multiagent-prefix-recoder.git
```

后续正式改造工作应放到这个仓库中。可以从干净工作区开始，不需要保留当前目录中的旧 `autogen_prefix_tree` 实现。

## 8. 一句话总结

下一个 agent 要做的是：把现有 prefix 重排工具从“HTTP 层代理”改造成“AutoGen model client 前置插件”。第一版只实现 `PrefixReorderClient(inner_client)` 这一条接入路线，并重点设计 `Local Prompt Compiler`、`Hierarchical Prefix Planner`、`Cache-Utility Validator` 三个模块。插件在模型调用前完成保守、可验证的 prompt block 顺序调整，然后把结果交还给原始 `OpenAIChatCompletionClient` 继续请求云端 API。
