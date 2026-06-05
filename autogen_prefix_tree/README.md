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

## 本地验证

建议始终在仓库本地虚拟环境中运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

项目根目录的 `pytest.ini` 已限制默认收集范围为 `tests/`，避免误收集 `tmp/framework-src` 等外部源码快照里的测试。
