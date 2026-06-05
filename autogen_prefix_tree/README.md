# AutoGen Native Prefix Reorder MVP

这个包现在只实现一条接入路线：

```python
inner_client = OpenAIChatCompletionClient(...)
model_client = PrefixReorderClient(inner_client)
```

`PrefixReorderClient` 在调用 inner client 之前执行：

```text
LocalPromptCompiler -> HierarchicalPrefixPlanner -> CacheUtilityValidator
```

第一版是保守 MVP：

- 只处理 AutoGen `ChatCompletionClient` 层输入。
- 不启动 HTTP proxy，不发送 HTTP 请求。
- 不修改模型响应。
- 只支持 exact hash sharing。
- 不确定时原样回退到 inner client。

## 当前新增能力

Planner 现在不只返回线性 `new_order`，还会生成一棵显式 prefix tree：

```text
root: global_shared_prefix
  -> subgroup: subgroup_shared_prefix
    -> leaf: agent_local_suffix
```

第一版仍然只把已经在同一 session 中出现过、且被 compiler 标为 `safe_prefix` / `conditional_prefix` 的共享 system text block 放入共享前缀。没有历史复用证据时只 observe，不重排。

Validator 现在会额外检查：

- prefix tree 是否覆盖当前请求里的全部 block。
- `cacheable_prefix_blocks` 是否真的位于新顺序开头。
- 重写后的 system message 是否严格等于 planner 指定的 block 拼接结果。
- 是否有最小 cache utility 增益估计。

`ValidationReport.utility_estimate` 会给出轻量指标，包括原始/重写后可复用前缀字符数、移动 block 字符数、估计增益和前后 block 指纹。后续接真实 API 时，可以把这些字段和 provider 返回的 cached tokens、latency、cost 一起记录。

## Semantic Guard

`CacheUtilityValidator` 支持可选 `semantic_guard`，用于后续接本地小模型或规则 judge。它不会替代静态安全检查，只会在静态检查通过后作为额外安全闸。

```python
from autogen_prefix_tree import CacheUtilityValidator, SemanticGuardReport


class LocalJudge:
    def evaluate(self, **kwargs):
        return SemanticGuardReport(
            passed=True,
            reason="semantic_invariants_hold",
            checks=("agent_identity", "latest_instruction"),
            model_name="local-small-model",
            confidence=0.9,
        )


validator = CacheUtilityValidator(semantic_guard=LocalJudge())
model_client = PrefixReorderClient(inner_client, validator=validator)
```

规则：

- 只有发生 block 移动时才调用 guard。
- guard 返回 `passed=False` 时回退原始 messages。
- guard 抛异常时也回退，reason 形如 `semantic_guard_failed:exception:RuntimeError`。
- telemetry 会记录 guard report，但不记录 prompt 正文。

## Telemetry

`PrefixReorderClient` 支持可选请求级 telemetry。默认不会写文件，只把最近一次记录保存在 `last_telemetry_record`，方便调试。

内存 sink：

```python
records = []
model_client = PrefixReorderClient(inner_client, telemetry_sink=records.append)
```

JSONL 文件：

```python
model_client = PrefixReorderClient(
    inner_client,
    telemetry_log_path="runs/prefix_reorder_requests.jsonl",
)
```

记录内容只包含结构化元数据，不包含 prompt 正文。主要字段：

- `session_id` / `request_index` / `operation`
- 重排前后的 message 类型和数量
- `blocks_moved`、移动 block 的语义类型、可移动性和共享范围
- `cacheable_prefix_blocks`
- `prefix_tree`
- Validator 的 `applied` / `fallback` / `reason`
- `utility_estimate`
- tools / model args hash

这套字段用于后续 AutoGenBench 或真实 API A/B 测试：插件侧记录 planner/validator 行为，provider 侧再补充 cached tokens、latency 和 cost。

离线汇总：

```python
from autogen_prefix_tree import load_jsonl_telemetry, summarize_telemetry

records = load_jsonl_telemetry("runs/prefix_reorder_requests.jsonl")
summary = summarize_telemetry(records)
print(summary.to_dict())
```

当前 summary 是 provider-independent 的估计指标，主要用于没有真实 API usage 时先判断规则是否常触发、是否频繁 fallback、是否形成重复可复用前缀：

- `request_count`
- `applied_count` / `applied_rate`
- `fallback_count` / `fallback_rate`
- `validation_reason_counts`
- `moved_block_count`
- `total_estimated_gain_chars`
- `reusable_prefix_request_count`
- `repeated_prefix_request_count` / `repeated_prefix_rate`
- `unique_reusable_prefix_count`

## Offline Microbenchmark

在没有 API key / AutoGenBench 环境时，可以先跑包内 microbenchmark，验证 AutoGen typed message -> compiler -> planner -> validator -> telemetry 的链路：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.microbench `
  --telemetry tmp\prefix_microbench\requests.jsonl `
  --summary tmp\prefix_microbench\summary.json `
  --repeats 1
```

它使用 no-op inner client，不会调用网络，也不会记录 prompt 正文。输出的 `summary.json` 可用于快速检查：

- cold request 是否 `no_rewrite_needed`
- warm request 是否 `validated`
- `applied_count`
- `fallback_count`
- `total_estimated_gain_chars`
- repeated reusable prefix 是否形成

## Evaluation Readiness

真实 AutoGenBench / API 运行前可以先检查环境：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd .
```

检查项：

- 是否在本仓库 `.venv` 中运行
- Docker 是否可用
- `OPENAI_API_KEY` / `OAI_CONFIG_LIST` 是否存在
- `autogenbench --help` 是否能通过 CLI smoke test
- `autogen_core` 是否安装

AutoGenBench 0.0.3 依赖旧版 `autogen` API。若 `autogenbench --help` 报 `ModuleNotFoundError: No module named 'autogen'`，在本仓库 `.venv` 中固定兼容版本：

```powershell
.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"
```

这个命令只报告密钥配置是否存在，不读取或打印密钥内容。当前如果还没有 API 配置，可以先用宽松模式放过 API 缺失，但继续检查 AutoGenBench/Docker 等其他前置条件：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --allow-missing-api
```

## 本地验证

建议始终在仓库本地虚拟环境中运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

项目根目录的 `pytest.ini` 已限制默认收集范围为 `tests/`，避免误收集 `tmp/framework-src` 等外部源码快照里的测试。
