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

