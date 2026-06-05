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

## 本地验证

建议始终在仓库本地虚拟环境中运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

项目根目录的 `pytest.ini` 已限制默认收集范围为 `tests/`，避免误收集 `tmp/framework-src` 等外部源码快照里的测试。
