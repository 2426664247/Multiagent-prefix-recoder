# AutoGen Prefix Tree / Validator 工作报告

日期：2026-06-05  
工作目录：`F:\CodexProject\MutilAgent`  
工作分支：`codex/autogen-validator-tree-eval`  
旧版回滚点：`main` 当前 HEAD 为 `1a8f067`，已有 tag `autogen-wrapper-mvp`

## 1. 本轮目标

根据当前研究总结和老师建议，本轮工作聚焦在 AutoGen 版本的前缀重排插件：

1. 保留旧版 git 版本，避免后续无法回滚。
2. 在 AutoGen `model_client` 前置 wrapper 路线基础上，补足显式 prefix tree。
3. 让 Validator 不只是做安全检查，还能辅助 planner 判断 tree 是否可采用、是否有 cache utility。
4. 参考 AutoGen 官方评测思路，规划后续真实 API / benchmark 验证路线。
5. 详细记录做了什么、试错过程和当前限制。

## 2. Git 和环境隔离

已确认当前仓库顶层为：

```text
F:/CodexProject/MutilAgent
```

为了不污染其他 git：

- 只在该仓库内执行 git 操作。
- 没有修改全局 git 配置。
- 没有进入或修改其他仓库。
- 新工作放在新分支 `codex/autogen-validator-tree-eval`。
- 旧版仍保留在 `main` / `autogen-wrapper-mvp` tag 上。

为了不污染系统 Python / conda：

- 创建并使用仓库本地虚拟环境 `.venv`。
- 后续测试命令显式使用 `.venv\Scripts\python.exe`。
- 安装的依赖只进入 `.venv`：`pytest`、`autogen-core` 及其依赖。
- `.venv` 已被 `.gitignore` 忽略，没有进入 git。

## 3. 基线检查和试错

先跑了一次全仓库测试：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果失败，原因不是当前插件测试失败，而是 pytest 默认递归收集了外部源码快照和旧实验目录，例如：

- `tmp/framework-src/**`
- `cache_hit_proxy/**`

这些目录里的测试依赖各自框架的完整开发环境，当前仓库不是要跑它们。因此第一次全仓库收集出现 249 个 collection error。

随后只跑当前插件测试：

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

结果：

```text
9 passed
```

之后新增了项目级 `pytest.ini`，限制默认测试入口只收集 `tests/`。再次运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
11 passed
```

继续补充 telemetry 后再次运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
13 passed
```

继续补充离线 utility analyzer 后再次运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
14 passed
```

继续补充 semantic guard 接口后再次运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
17 passed
```

继续补充离线 microbenchmark harness 后再次运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
18 passed
```

并实际运行了一次命令行 microbenchmark：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.microbench `
  --telemetry tmp\prefix_microbench\requests.jsonl `
  --summary tmp\prefix_microbench\summary.json `
  --repeats 1
```

输出摘要：

```json
{"request_count": 3, "applied_count": 2, "fallback_count": 0, "total_estimated_gain_chars": 894}
```

第一次运行时出现过一个 `runpy` warning，原因是 `autogen_prefix_tree.__init__` 顶层提前 import 了 `microbench`，再用 `python -m autogen_prefix_tree.microbench` 会导致模块预加载。随后把 `microbench` 从顶层 `__init__` 导出移除，改为通过 `autogen_prefix_tree.microbench` 子模块导入，CLI warning 消失。

继续补充 AutoGenBench readiness checker 后再次运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
22 passed
```

严格 readiness 检查：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd .
```

当前结果：

```text
python_environment: ok
docker: ok, Docker version 28.5.1
api_config: missing
autogenbench: cli smoke ok
autogen_core: ok
ready: false
```

随后在 `.venv` 中安装了 `autogenbench 0.0.3`。第一次安装会带入 `pyautogen 0.10.0`，但 `autogenbench --help` 报错：

```text
ModuleNotFoundError: No module named 'autogen'
```

原因是 AutoGenBench 0.0.3 仍依赖 AutoGen 0.2 时代的旧 `autogen` API，而 `pyautogen 0.10.0` 已迁移到新的 package 形态。之后在本仓库 `.venv` 中固定：

```powershell
.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"
```

此后 `autogenbench --help` 可以正常启动。现在真实 AutoGenBench/API 评测只剩一个硬阻塞：准备 `OPENAI_API_KEY` 或 `OAI_CONFIG_LIST`。

## 4. 参考的 AutoGen 官方评测思路

我查阅了 AutoGen 官方资料，主要参考：

- AutoGenBench 官方博客：https://microsoft.github.io/autogen/0.2/blog/2024/01/25/AutoGenBench/
- AutoGenBench README：https://github.com/microsoft/autogen/blob/0.2/samples/tools/autogenbench/README.md
- AutoGen tracing / observability 文档：https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tracing.html

对本插件最有用的三条原则是：

1. Repetition：LLM 和 agent 运行有随机性，benchmark 要多次重复，不能只看单次结果。
2. Isolation：每个任务应在隔离环境里跑，AutoGenBench 推荐 Docker，避免任务之间互相污染。
3. Instrumentation：不仅看最终成功率，还要保存 logs / telemetry，方便后续算自定义指标。

因此后续真实实验不应只报告“cache 命中率提高了”，而要同时记录：

- benchmark 原生质量指标，例如 HumanEval / AutoGenBench task success。
- request-level cache telemetry，例如 input tokens、cached tokens、latency、cost。
- 插件内部 telemetry，例如移动了哪些 semantic block、prefix tree 结构、Validator 是否 fallback。
- 重复实验和 paired A/B 结果。

## 5. 代码改动

### 5.1 IR：新增 prefix tree 数据结构

文件：`autogen_prefix_tree/ir.py`

新增：

- `PrefixTreeNode`
- `PrefixTree`

树的目标形态：

```text
root: global_shared_prefix
  -> subgroup: subgroup_shared_prefix
    -> leaf: agent_local_suffix
```

这样 planner 不再只是输出一个线性的 `new_order`，而是显式说明哪些 block 位于全局共享前缀、子组共享前缀、agent 本地后缀。

### 5.2 Planner：生成 prefix tree 和 cacheable prefix path

文件：`autogen_prefix_tree/planner.py`

主要变化：

- `PrefixPlan` 新增：
  - `cacheable_prefix_blocks`
  - `prefix_tree`
- `HierarchicalPrefixPlanner` 新增候选 block 排序规则。
- 保守策略不变：只有同一 session 里已经观察到 exact hash 的共享 system block 才能进入共享前缀。
- 没有历史证据时仍然只 observe，不重排。
- 计划中保留可解释的 `move_reason`，包含 share scope 和 semantic type。

当前排序优先级：

```text
global_task_background
shared_context
team_policy
shared_tool_description
output_format
```

这仍然是保守 MVP，不做语义等价合并，也不移动 latest user instruction、tool result、assistant history、private memory 等动态/敏感块。

### 5.3 Validator：补足 tree 验证和 cache utility 估计

文件：`autogen_prefix_tree/validator.py`

新增：

- `CacheUtilityEstimate`
- `ValidationReport.utility_estimate`
- `CacheUtilityValidator(min_estimated_gain_chars=1)`

Validator 现在额外检查：

- `prefix_tree` 是否存在。
- prefix tree 是否覆盖当前请求里的全部 block。
- `cacheable_prefix_blocks` 是否真的位于 `new_order` 开头。
- 重排前后 block id multiset 是否一致。
- 重排前后 block hash multiset 是否一致。
- tools hash 是否一致。
- model args hash 是否一致。
- forbidden block 是否被 planner 直接移动。
- message 数量、类型、source 是否保持。
- 重写后的 system message 内容是否严格等于 planner 指定的 block 拼接结果。
- 若发生移动，是否有最小 cache utility 估计增益。

`CacheUtilityEstimate` 当前记录的是轻量字符级估计：

- `original_prefix_chars`
- `rewritten_prefix_chars`
- `moved_block_chars`
- `estimated_gain_chars`
- `prefix_fingerprint_before`
- `prefix_fingerprint_after`

注意：这还不是 provider 返回的真实 cached tokens。后续接真实 API 时，需要把它和 API usage 里的 cached tokens、latency、cost 对齐。

### 5.4 测试：从 9 个增加到 20 个

文件：`tests/test_native_prefix_reorder.py`

新增/增强的验证：

- planner 输出的 prefix tree 覆盖全部 block。
- `cacheable_prefix_blocks` 位于新顺序开头。
- wrapper 成功重排后，`ValidationReport.utility_estimate.estimated_gain_chars > 0`。
- Validator 能拒绝 block 覆盖不完整的坏 tree。
- Validator 能拒绝被篡改的 rewritten system content。
- telemetry sink 能收到两次请求记录。
- telemetry JSON 序列化后不包含测试 prompt 正文。
- JSONL logger 能写入请求记录。
- pipeline exception fallback 也会写 telemetry。
- telemetry summary 能统计 apply/fallback/reuse 和 validation reason。
- semantic guard 通过时允许重排并写 telemetry。
- semantic guard 拒绝时强制 fallback。
- semantic guard 抛异常时 fail closed。
- microbenchmark 能写出 telemetry JSONL 和 summary JSON。
- readiness checker 能识别 API 配置缺失，且不输出 secret 值。
- readiness checker 会实际执行 `autogenbench --help`，能发现 console script 存在但依赖不兼容的问题。

最终测试结果：

```text
22 passed
```

### 5.5 测试配置

新增文件：`pytest.ini`

作用：

- 默认只收集 `tests/`。
- 避免误跑 `tmp/framework-src` 和旧实验目录里的第三方测试。

### 5.6 README 更新

文件：`autogen_prefix_tree/README.md`

补充当前实现状态：

- planner 现在生成显式 prefix tree。
- Validator 现在输出 utility estimate。
- `PrefixReorderClient` 现在支持可选 telemetry sink / JSONL logger。
- 推荐使用 `.venv\Scripts\python.exe -m pytest -q` 进行本地验证。

### 5.7 Telemetry：为真实 benchmark 做请求级记录

新增文件：`autogen_prefix_tree/telemetry.py`

`PrefixReorderClient` 现在支持可选 telemetry：

```python
records = []
client = PrefixReorderClient(inner_client, telemetry_sink=records.append)
```

或者写 JSONL：

```python
client = PrefixReorderClient(
    inner_client,
    telemetry_log_path="runs/prefix_reorder_requests.jsonl",
)
```

默认不会写文件，只保留最近一次 `last_telemetry_record`。这样本地调试方便，也不会无意中生成大量日志。

记录字段包括：

- `session_id`
- `request_index`
- `operation`
- 重排前后 message 类型和数量
- `blocks_moved`
- moved block 的 semantic type、movability、share scope、source role
- `cacheable_prefix_blocks`
- `prefix_tree`
- Validator 的 `applied`、`fallback`、`reason`
- `utility_estimate`
- tools / model args hash

为了避免 benchmark trace 泄露任务 prompt，当前 telemetry 不记录完整 prompt 正文，也不记录 block 文本内容，只记录 block id 和结构化 metadata。

### 5.8 离线 Utility Analyzer

在 `autogen_prefix_tree/telemetry.py` 中继续新增：

- `TelemetrySummary`
- `load_jsonl_telemetry(path)`
- `summarize_telemetry(records)`

这一步的目的不是替代真实 provider cached tokens，而是在还没有 API key / AutoGenBench 真实运行之前，先让本地和后续 benchmark telemetry 能被稳定汇总。

当前 summary 字段包括：

- `request_count`
- `applied_count` / `applied_rate`
- `fallback_count` / `fallback_rate`
- `no_rewrite_count`
- `validation_reason_counts`
- `moved_block_count`
- `total_original_prefix_chars`
- `total_rewritten_prefix_chars`
- `total_estimated_gain_chars`
- `total_moved_block_chars`
- `reusable_prefix_request_count`
- `repeated_prefix_request_count` / `repeated_prefix_rate`
- `unique_reusable_prefix_count`
- `average_estimated_gain_chars`

新增测试用三次 AutoGen-style 请求模拟 cold / warm / repeated warm：

```text
planner   -> no_rewrite_needed
engineer  -> validated
reviewer  -> validated
```

测试确认：

- 三次请求都能被 JSONL 加载。
- validation reason 统计为 `{"no_rewrite_needed": 1, "validated": 2}`。
- 两次 reusable prefix request 中有一次是重复 signature。
- `applied_rate == 2/3`。

这个 analyzer 可以直接用于后续 AutoGenBench A/B：插件侧先输出 JSONL，再用 summary 看规则覆盖率、fallback 原因和可复用前缀是否稳定形成。

### 5.9 Semantic Guard：为本地小模型判断预留接口

新增文件：`autogen_prefix_tree/semantic_guard.py`

新增：

- `SemanticGuardReport`
- `SemanticGuard` protocol

设计目的：

老师建议后续可以加本地小模型辅助 semantic 规则判断。当前没有绑定某个具体模型，而是先把 Validator 的扩展点做出来。后续无论用本地小模型、规则 judge，还是一个混合 judge，都可以实现同一个 `evaluate(...)` 接口。

接入方式：

```python
validator = CacheUtilityValidator(semantic_guard=local_judge)
client = PrefixReorderClient(inner_client, validator=validator)
```

执行规则：

- 只有静态不变量检查通过后才调用 semantic guard。
- 只有发生 block 移动时才调用 guard，冷启动 no-rewrite 不调用。
- guard 返回 `passed=False` 时，Validator 回退原始 messages。
- guard 抛异常时也回退，原因形如 `semantic_guard_failed:exception:RuntimeError`。
- telemetry 会记录 `semantic_guard` report，但不记录 prompt 正文。

新增测试覆盖：

- `PassingSemanticGuard`：确认本地 judge 通过时，重排仍然生效。
- `RejectingSemanticGuard`：确认 judge 不确定时，原始 messages 被传给 inner client。
- `ThrowingSemanticGuard`：确认 judge 抛异常时 fail closed。

这一步没有真正调用本地小模型，但完成了 Validator 与本地语义判断模块的协作接口。后续接模型时，需要重点设计 judge prompt：例如询问当前 agent 身份、最新用户指令、private/shared 边界、tool result 对应关系是否仍然清楚。

### 5.10 Offline Microbenchmark Harness

新增文件：`autogen_prefix_tree/microbench.py`

这个 harness 用 AutoGen typed `SystemMessage` 构造一个最小三 agent 请求序列：

```text
planner -> engineer -> reviewer
```

每个请求都包含：

- role-specific instruction
- shared user task
- shared groupchat context
- team policy
- shared tool schema
- current turn instruction

它使用 `_NoopClient` 作为 inner client，不调用网络、不读取 API key、不产生真实模型输出。目的只是验证插件链路：

```text
SystemMessage
  -> LocalPromptCompiler
  -> HierarchicalPrefixPlanner
  -> CacheUtilityValidator
  -> PrefixReorderClient telemetry
  -> summarize_telemetry
```

命令：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.microbench `
  --telemetry tmp\prefix_microbench\requests.jsonl `
  --summary tmp\prefix_microbench\summary.json `
  --repeats 1
```

一次默认运行的期望形态：

```text
request_count = 3
no_rewrite_count = 1
applied_count = 2
fallback_count = 0
validation_reason_counts = {"no_rewrite_needed": 1, "validated": 2}
```

这不是替代真实 AutoGenBench，而是给真实 API 前提供一个本地 smoke test：如果未来改 compiler/planner/validator 后 microbenchmark 都无法形成 warm prefix，就不应直接跑昂贵的真实 API。

### 5.11 AutoGenBench/API Readiness Checker

新增文件：`autogen_prefix_tree/readiness.py`

目的：

真实 API benchmark 前需要确认环境已经具备：

- 本仓库本地 `.venv`
- Docker
- API 配置
- `autogenbench`
- `autogen_core`

命令：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd .
```

如果只想允许 API 配置缺失，但继续检查 AutoGenBench/Docker 等其他条件：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --allow-missing-api
```

安全边界：

- 只报告 `OPENAI_API_KEY` / `OAI_CONFIG_LIST` 是否存在。
- 不读取 `OAI_CONFIG_LIST` 内容。
- 不打印任何密钥值。

当前严格检查结果是 not ready：

```text
api_config = missing
autogenbench = cli smoke ok
```

这一步让真实评测的阻塞条件变成可执行清单，而不是口头说明。下一步一旦提供 API 配置，可先执行：

```powershell
.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"
.venv\Scripts\autogenbench.exe clone HumanEval
cd HumanEval
..\.venv\Scripts\autogenbench.exe run --subsample 0.1 --repeat 3 Tasks/human_eval_two_agents.jsonl
```

## 6. 当前没有完成的真实 API / benchmark 工作

我检查了当前环境：

```text
OPENAI_API_KEY=missing
OAI_CONFIG_LIST=missing
OAI_CONFIG_LIST_FILE=missing
Docker version 28.5.1
autogenbench: cli smoke ok in .venv
```

结论：

- Docker 已安装。
- AutoGenBench CLI 已安装并能启动。
- 但当前没有可用 API 配置。
- 因此本轮没有跑真实 AutoGenBench，也没有伪造任何 cached token / latency / cost 结果。

后续要跑真实 AutoGenBench 时，若需要重建 `.venv`，建议固定兼容依赖：

```powershell
.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"
```

然后准备 `OAI_CONFIG_LIST` 或 `OPENAI_API_KEY`，再按 AutoGenBench 官方方式跑：

```powershell
.venv\Scripts\autogenbench.exe clone HumanEval
cd HumanEval
..\.venv\Scripts\autogenbench.exe run --subsample 0.1 --repeat 3 Tasks/human_eval_two_agents.jsonl
..\.venv\Scripts\autogenbench.exe tabulate Results/human_eval_two_agents
```

正式 A/B 应至少分两组：

```text
baseline: 原始 AutoGen model_client
plugin: PrefixReorderClient(inner_client)
```

更好的三组：

```text
baseline
safe_prefix
full_plugin
```

## 7. 下一步建议

### 7.1 先补请求级 telemetry

建议给 `PrefixReorderClient` 增加可选 JSONL logger，字段包括：

```text
session_id
request_index
agent/source
message_roles_before
message_roles_after
blocks_moved
cacheable_prefix_blocks
prefix_tree_leaf_path
validation_reason
fallback
utility_estimate
tools_hash
model_args_hash
```

接真实 API 后再加入：

```text
input_tokens
cached_tokens
output_tokens
latency_ms
cost_estimate
api_error
```

### 7.2 再做 AutoGenBench A/B harness

先做 HumanEval / two-agent 小子集，验证插件不破坏 AutoGen 官方 benchmark 的运行链路。

实验设计：

- 同一 task 成对运行 baseline 和 plugin。
- `repeat >= 3`。
- 固定 model、temperature、tool 权限、预算、benchmark commit。
- 保存每次请求的 plugin telemetry。
- 报告 task success delta、cached token delta、latency delta。

### 7.3 再扩展到更敏感的 multi-agent 数据集

AutoGenBench/HumanEval 主要验证 coding agent workflow，不足以覆盖所有语义风险。后续建议按已有研究文档扩展：

- Silo-Bench：检查分布式信息整合和通信顺序是否被破坏。
- CooperBench：检查多 coding agent 协作、merge compatibility 和 unit tests。
- HiddenBench：检查 private/shared 信息边界和 hidden facts 是否被忽略。
- MultiAgentBench / MARBLE：做综合多场景 smoke test。

### 7.4 本地小模型 semantic rule

当前 compiler / validator 仍是规则和 exact hash。老师提到的本地小模型辅助 semantic 判断，可以放到下一阶段：

- 第一阶段只作为 validator 的旁路诊断，不直接放行移动。
- 让小模型回答“当前 agent 是谁”“最新用户指令是什么”“private memory 是否泄露到 shared prefix”等问题。
- 只有当规则检查和小模型检查都通过时，才考虑把某些 conditional block 提升到更高 scope。

## 8. 当前限制

1. Compiler 仍主要依赖 marker / 文本规则，真实 AutoGen system message 的泛化能力还要用 trace 验证。
2. Planner 只支持 exact hash sharing，不支持规范化 hash 或语义等价。
3. `subgroup` 结构已经有数据模型，但当前 compiler 还没有可靠推断 subgroup scope。
4. `rewrite_messages` 仍只支持移动同一个 system message 内的文本 block。
5. Utility 估计和 summary 现在仍是字符级 / signature 级 proxy，不是真实 tokenizer / provider cached token。
6. Semantic guard 现在是接口和 fake judge 测试，还没有接真实本地小模型。
7. readiness checker 显示当前没有 API 配置；AutoGenBench CLI 已在 `.venv` 中通过 smoke test，但还没有真实 API cache / latency / cost 结果。

## 9. 本轮产物清单

代码：

- `autogen_prefix_tree/ir.py`
- `autogen_prefix_tree/planner.py`
- `autogen_prefix_tree/validator.py`
- `autogen_prefix_tree/__init__.py`
- `autogen_prefix_tree/client.py`
- `autogen_prefix_tree/telemetry.py`
- `autogen_prefix_tree/semantic_guard.py`
- `autogen_prefix_tree/microbench.py`
- `autogen_prefix_tree/readiness.py`

测试：

- `tests/test_native_prefix_reorder.py`
- `pytest.ini`

文档：

- `autogen_prefix_tree/README.md`
- `reports/autogen_validator_tree_work_report_2026-06-05.md`

验证：

```text
.venv\Scripts\python.exe -m pytest -q
22 passed
```

命令行 smoke：

```text
.venv\Scripts\python.exe -m autogen_prefix_tree.microbench --telemetry tmp\prefix_microbench\requests.jsonl --summary tmp\prefix_microbench\summary.json --repeats 1
request_count=3, applied_count=2, fallback_count=0
```

Readiness：

```text
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd .
ready=false, api_config=missing, autogenbench=cli_smoke_ok
```
