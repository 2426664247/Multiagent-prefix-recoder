# Agent Handoff - AutoGen Prefix Reorder Plugin

日期：2026-06-05  
仓库：`F:\CodexProject\MutilAgent`  
当前分支：`codex/autogen-validator-tree-eval`  
交接原因：当前 agent 停止继续开发，仅整理已完成工作和后续接手路径。

## 1. 当前目标

用户正在做一个 AutoGen 优先的 prefix reorder 插件。核心判断是：直接截取框架发给云端 API 的 HTTP 请求，不能无损还原原始 prompt，所以插件应接在 AutoGen 把结构化 message 转成 HTTP request 之前。

当前方向：

1. 先适配 AutoGen，不急着扩展到所有 multi-agent 框架。
2. 自动识别 shared task、shared context、team policy、shared tool description、output format 等共享块。
3. 将可安全复用的共享块排到 prompt 前面，提高 provider prompt cache 命中。
4. Planner 不只输出线性重排结果，还要构建 prefix tree。
5. Validator 要和 planner 配合，验证 tree、重排安全性和 cache utility。
6. 后续用 AutoGen 官方 benchmark / 真实 API 评估效果。
7. 再考虑本地小模型辅助 semantic guard，验证规则是否在真实 case 中泛化。

用户额外要求：

- 不污染其他 git 仓库。
- 使用 conda 或虚拟环境；当前实际使用的是仓库本地 `.venv`。
- 保留旧版本，方便随时回滚。
- 详细记录做过的工作、试错过程和后续计划。

## 2. Git 状态

旧版本保留点：

- `main` / `origin/main`：`1a8f067 feat: add native prefix reorder client mvp`
- tag：`autogen-wrapper-mvp` 指向 `1a8f067`

当前工作分支：

```text
codex/autogen-validator-tree-eval
```

最近提交：

```text
840b0b9 chore: harden autogenbench readiness
f70b633 feat: add autogenbench readiness check
bdcb4fa feat: add offline prefix microbenchmark
9497da7 feat: add semantic guard hook
0580636 feat: summarize prefix telemetry utility
b2a125c feat: add prefix reorder telemetry
bda416c feat: add autogen prefix tree validator
1a8f067 feat: add native prefix reorder client mvp
```

重要提醒：工作区里有一批未跟踪研究资料和旧实验目录，接手时不要误 stage：

```text
AgentTeam共享Prefix.pdf
AgentTeam语义前缀缓存初步研究.md
Prefix重排插件设计_PPT大纲.md
autogen/
autogen_cache_probe/
autogen_groupchat_kvcache_experiment/
deepseek-cache-verifier/
deepseek_agent_cache_case_study/
framework_cache_shapes_experiment/
multiagent_benchmark_notes/
worklog.md
```

这些不是本轮新开发提交的一部分。后续 git 操作建议始终显式指定文件路径，不要 `git add .`。

## 3. 环境状态

当前使用仓库本地虚拟环境：

```powershell
.venv\Scripts\python.exe
```

已验证：

- `.venv` 是当前仓库本地环境。
- Docker 可用：`Docker version 28.5.1, build e180ab8`
- `autogen_core` 已安装。
- `autogenbench` 已安装并能通过 CLI smoke test。
- 真实 API 配置缺失：没有检测到 `OPENAI_API_KEY` / `OAI_CONFIG_LIST`。

AutoGenBench 兼容性试错：

1. 安装 `autogenbench 0.0.3` 后，默认带入 `pyautogen 0.10.0`。
2. 执行 `autogenbench --help` 报错：

```text
ModuleNotFoundError: No module named 'autogen'
```

3. 原因：`autogenbench 0.0.3` 依赖 AutoGen 0.2 时代的旧 `autogen` API，而 `pyautogen 0.10.0` 已迁移到新 package 形态。
4. 修复方式是在本仓库 `.venv` 中固定：

```powershell
.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"
```

注意：不要装到全局 Python，不要污染系统环境。

## 4. 已完成的主要代码工作

### 4.1 Prefix Tree IR

文件：`autogen_prefix_tree/ir.py`

新增：

- `PrefixTreeNode`
- `PrefixTree`

目标结构：

```text
root: global_shared_prefix
  -> subgroup: subgroup_shared_prefix
    -> leaf: agent_local_suffix
```

当前 tree 已作为 planner 输出的一部分，validator 会检查覆盖关系。

### 4.2 Planner

文件：`autogen_prefix_tree/planner.py`

已完成：

- `PrefixPlan` 增加 `cacheable_prefix_blocks`。
- `PrefixPlan` 增加 `prefix_tree`。
- planner 会按共享 block 生成 prefix tree。
- 仍然是保守策略：只有同一 session 中已经观察过 exact hash 的共享 system block 才进入可复用前缀。
- 没有历史复用证据时只 observe，不重排。

当前优先重排的共享语义类型：

```text
global_task_background
shared_context
team_policy
shared_tool_description
output_format
```

### 4.3 Validator

文件：`autogen_prefix_tree/validator.py`

已完成：

- `CacheUtilityEstimate`
- `ValidationReport.utility_estimate`
- tree coverage 检查
- `cacheable_prefix_blocks` 是否位于新顺序开头的检查
- block id/hash multiset 不变检查
- tools / model args hash 不变检查
- forbidden block 不被移动检查
- rewritten system content 与 planner 指定 block 拼接结果严格一致检查
- 最小 cache utility 增益检查
- semantic guard fail-closed

当前 utility 仍然是字符级 proxy，不是真实 provider cached tokens。

### 4.4 PrefixReorderClient Telemetry

文件：

- `autogen_prefix_tree/client.py`
- `autogen_prefix_tree/telemetry.py`

已完成：

- `PrefixReorderClient.last_telemetry_record`
- 可选 `telemetry_sink`
- 可选 `telemetry_log_path`
- `JsonlTelemetryLogger`
- `TelemetrySummary`
- `load_jsonl_telemetry`
- `summarize_telemetry`

Telemetry 只记录结构化元数据，不记录 prompt 正文，避免 benchmark trace 泄露任务内容。

### 4.5 Semantic Guard 接口

文件：`autogen_prefix_tree/semantic_guard.py`

已完成：

- `SemanticGuardReport`
- `SemanticGuard` protocol
- Validator 支持可选 `semantic_guard`

当前只是接口和 fake judge 测试，还没有接真实本地小模型。

### 4.6 Offline Microbenchmark

文件：`autogen_prefix_tree/microbench.py`

用途：在没有 API key 的情况下，先本地验证 AutoGen typed message -> compiler -> planner -> validator -> telemetry 链路。

命令：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.microbench `
  --telemetry tmp\prefix_microbench\requests.jsonl `
  --summary tmp\prefix_microbench\summary.json `
  --repeats 1
```

最近验证结果：

```text
request_count=3
applied_count=2
fallback_count=0
total_estimated_gain_chars=894
```

### 4.7 AutoGenBench Readiness

文件：`autogen_prefix_tree/readiness.py`

已完成：

- 检查是否使用本仓库 `.venv`。
- 检查 Docker。
- 检查 API 配置是否存在，但不打印密钥值。
- 检查 `autogen_core`。
- 检查 `autogenbench --help` 是否能实际启动，而不是只检查包是否存在。
- 输出下一步建议命令，显式使用 `.venv` 下的入口。

严格 readiness 现状：

```text
python_environment: ok
docker: ok
api_config: missing
autogenbench: cli smoke ok
autogen_core: ok
ready: false
```

允许 API 缺失时：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --allow-missing-api
```

结果应为 `ready=true`。

## 5. 已完成的测试

当前测试范围由 `pytest.ini` 限定为 `tests/`，避免误收集旧实验目录和外部源码快照。

最近通过：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
22 passed
```

主要测试文件：

- `tests/test_native_prefix_reorder.py`
- `tests/test_microbench.py`
- `tests/test_readiness.py`

新增/覆盖内容包括：

- planner 输出 prefix tree。
- validator 检查 tree 覆盖。
- validator 拒绝 rewritten system content 被篡改。
- utility estimate 大于 0。
- telemetry 不记录 prompt 正文。
- telemetry summary 能统计 apply/fallback/repeated prefix。
- semantic guard 通过、拒绝、异常三种路径。
- microbenchmark 能输出 JSONL 和 summary JSON。
- readiness 不泄露 secret。
- readiness 能识别 broken `autogenbench --help`。

## 6. 详细报告位置

主要工作报告：

```text
reports/autogen_validator_tree_work_report_2026-06-05.md
```

README：

```text
autogen_prefix_tree/README.md
```

这两个文件已经记录了实现状态、试错过程、验证命令和当前限制。

## 7. 还没有完成的工作

### 7.1 真实 API / AutoGenBench 未跑

原因：当前没有 `OPENAI_API_KEY` / `OAI_CONFIG_LIST`。

不要伪造以下指标：

- real cached tokens
- real latency
- real cost
- real AutoGenBench task success delta

这些必须等真实 API 配置可用后再跑。

### 7.2 AutoGenBench A/B Harness 未完成

还需要把当前 `PrefixReorderClient` 接到 AutoGenBench 的真实运行链路里，做 baseline vs plugin 对照。

建议实验分组：

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

### 7.3 本地小模型 Semantic Guard 未接入

当前只是接口。后续可接本地小模型或规则 judge，检查：

- 当前 agent 身份是否保持清晰。
- 最新用户指令是否仍然可识别。
- private/shared 边界是否被破坏。
- tool result 与对应上下文是否仍然绑定。
- conditional prefix 是否可提升到更高 scope。

建议先只做旁路诊断，不要一开始就让小模型直接放行高风险重排。

### 7.4 Tree 仍需增强

当前 tree 已经存在，但还比较保守：

- `subgroup` 结构有数据模型，但 compiler 还没有可靠推断 subgroup scope。
- planner 只支持 exact hash sharing。
- 不支持 normalized hash。
- 不支持语义等价共享。
- `rewrite_messages` 仍只支持同一个 system message 内部 block 重排。

## 8. 下一个 Agent 建议操作顺序

### Step 1：不要先改代码，先确认状态

```powershell
git status --short --branch
git log --oneline --decorate -10
```

确认仍在：

```text
codex/autogen-validator-tree-eval
```

不要 `git add .`。

### Step 2：确认本地环境

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd .
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --allow-missing-api
```

如果 `autogenbench --help` 又失败，先修 `.venv`：

```powershell
.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"
```

### Step 3：如果用户提供 API 配置，再跑真实 benchmark

准备 `OPENAI_API_KEY` 或 `OAI_CONFIG_LIST` 后：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd .
.venv\Scripts\autogenbench.exe clone HumanEval
cd HumanEval
..\.venv\Scripts\autogenbench.exe run --subsample 0.1 --repeat 3 Tasks/human_eval_two_agents.jsonl
..\.venv\Scripts\autogenbench.exe tabulate Results/human_eval_two_agents
```

注意：正式实验前最好先小样本 smoke。不要一次性跑大规模任务烧预算。

### Step 4：做 A/B 接入

需要把 AutoGenBench 的 model client 构造处改成可切换：

```text
baseline -> OpenAIChatCompletionClient(...)
plugin   -> PrefixReorderClient(OpenAIChatCompletionClient(...), telemetry_log_path=...)
```

建议输出：

```text
runs/autogenbench_baseline/...
runs/autogenbench_plugin/...
runs/prefix_reorder_requests.jsonl
runs/prefix_reorder_summary.json
```

这些输出目录如果会产生大文件，应先确认 `.gitignore`，不要误提交。

### Step 5：补真实 utility 报告

真实 API 跑完后，报告至少包含：

- task success delta
- input tokens
- cached tokens
- output tokens
- latency
- cost
- plugin applied/fallback rate
- fallback reason counts
- repeated reusable prefix rate
- semantic block 类型分布

当前已有 telemetry summary 只能证明规则会触发，不能证明真实 provider cache 收益。

## 9. 接手时的风险点

1. `autogen/` 是当前仓库里的未跟踪目录，可能影响 Python import 行为。执行 AutoGenBench 时要注意 cwd 和 `PYTHONPATH`。
2. 不要把 `.venv`、`tmp/`、benchmark results、API config、trace 文件提交进 git。
3. readiness 只检查 secret 是否存在，不会也不应该打印 secret 内容。
4. 当前 compiler 仍依赖 marker / 文本规则，真实 AutoGen case 的泛化要靠 benchmark telemetry 证明。
5. 真实 API 评测前，先跑 microbenchmark。若 microbenchmark 不能形成 warm prefix，不要直接跑真实 API。
6. 任何声称“提升 cache 命中率比例”的结论，都必须来自真实 provider usage 或严格 A/B 数据。

## 10. 最短接手摘要

当前代码层面已经完成：

- AutoGen 前置 wrapper MVP
- prefix tree IR
- planner tree output
- validator tree/utility 检查
- telemetry JSONL 和 summary
- semantic guard 接口
- offline microbenchmark
- AutoGenBench readiness + CLI smoke

当前阻塞：

```text
缺少 OPENAI_API_KEY / OAI_CONFIG_LIST
```

下一步真正要做：

```text
拿到 API 配置 -> 跑 AutoGenBench 小样本 smoke -> 做 baseline/plugin A/B -> 汇总真实 cached tokens/latency/cost/task success -> 再决定是否扩展 semantic guard 和其他框架。
```

