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

## 当前阶段

当前阶段是 Utility Validator 训练前的框架搭建：实现 prefix tree planner、placement-level feedback、hard validator、cache gain report、replay log 和 HumanEval utility goldset builder。此阶段不训练 Utility Validator，也不训练 Planner/ranking 模型；所有 planner scoring 都是规则式、可解释、可替换的接口。

## 当前新增能力

Planner 现在的主语义输出是 `PrefixTreeCandidate`。为兼容旧 wrapper，`plan()` 仍返回 `PrefixPlan`，但其中会挂载 `prefix_tree_candidate`、`placements`、`planner_score_breakdown` 和 `cache_gain_report`。

```text
root: global_shared_prefix
  -> subgroup: subgroup_shared_prefix
    -> leaf: agent_local_suffix
```

`PrefixTreeCandidate` 表达：

- global shared prefix、subgroup shared prefix、agent-local suffix；
- 每个 agent 从 root 到 leaf 的 `agent_paths`；
- 每个 block 的 `BlockPlacement`、target scope、风险和 placement score；
- `estimated_cache_gain`、`planner_score` 和 breakdown；
- 可回放的 block order 与 materialized prompt hash/text（telemetry 默认不写正文）。

`generate_candidates(...)` 会生成 conservative、balanced、aggressive 三类候选。当前仍只把已经在同一 session 中出现过、且通过 hard-gate 规则的 exact-hash shared block 放入共享前缀；没有历史复用证据时只 observe，不重排。`materialize_prompt(candidate, agent_id)` 会按 prefix tree 路径展开指定 agent 的最终 prompt，并检查不丢 block、不篡改、不重复。

Validator 现在会额外检查：

- prefix tree 是否覆盖当前请求里的全部 block。
- `cacheable_prefix_blocks` 是否真的位于新顺序开头。
- 重写后的 system message 是否严格等于 planner 指定的 block 拼接结果。
- 是否有最小 cache utility 增益估计。

`ValidationReport.utility_estimate` 会给出轻量指标，包括原始/重写后可复用前缀字符数、移动 block 字符数、估计增益和前后 block 指纹。后续接真实 API 时，可以把这些字段和 provider 返回的 cached tokens、latency、cost 一起记录。

Cache gain 现在优先来自 prefix tree / shared prefix：report 会记录 longest common prefix tokens、global prefix tokens、subgroup prefix tokens、每个 prefix tree node 的 cache contribution 和 candidate 总 `estimated_cache_gain`。

## Unified CacheEstimator

The Cache Hit Increased Gate now reads a unified `CacheEstimator` interface:

```python
from autogen_prefix_tree import CacheHitProxyEstimator, CacheUtilityValidator, PrefixReorderClient

validator = CacheUtilityValidator(cache_estimator="cache_hit_proxy")
model_client = PrefixReorderClient(inner_client, cache_estimator="cache_hit_proxy")

validator = CacheUtilityValidator(cache_estimator=CacheHitProxyEstimator())
```

Supported estimators:

- `PrefixTreeEstimator` is the default and preserves the existing offline prefix-tree/shared-prefix behavior.
- `CacheHitProxyEstimator` dynamically loads `cache_hit_proxy/cache_estimator.py` through an adapter, does not modify `cache_hit_proxy/`, and uses prompt-safe hash-derived token units instead of recording prompt text.
- `ProviderTelemetryEstimator` is a placeholder for future provider-reported cached tokens, latency, and cost.

If `cache_hit_proxy` is configured but unavailable or its interface does not match, the adapter fails gracefully and falls back to `PrefixTreeEstimator`; the report sets `used_fallback=true` and records the reason. Offline estimates are not real provider cached-token measurements. Real cached tokens, latency, and cost still require future provider telemetry.

`ValidationReport`, request telemetry, replay records, feedback records, and goldset/training feature exports now include `cache_estimate_report` with `estimator_name`, `estimator_available`, `used_fallback`, `cached_tokens_delta`, `estimated_cache_gain`, contribution fields, reason, and warnings. The older `cache_gain_report` remains available for compatibility.

No-network smoke:

```powershell
$env:PYTHONPATH = "resource"
.venv\Scripts\python.exe -m autogen_prefix_tree.cache_estimator_smoke `
  --estimator prefix-tree `
  --output-dir tmp\cache_estimator_smoke

$env:PYTHONPATH = "resource"
.venv\Scripts\python.exe -m autogen_prefix_tree.cache_estimator_smoke `
  --estimator cache-hit-proxy `
  --output-dir tmp\cache_estimator_smoke_proxy
```

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

也可以接一个本地 OpenAI-compatible judge。这个 guard 会把重排前后的消息内容发送到配置的本地模型服务做语义不变量判断，因此只应指向受控本地服务；telemetry 仍然只记录 prompt-safe 的判断结果：

```python
from autogen_prefix_tree import CacheUtilityValidator, OpenAICompatibleSemanticGuard

guard = OpenAICompatibleSemanticGuard(
    base_url="http://127.0.0.1:11434/v1",
    model="local-small-model",
    min_confidence=0.75,
)
validator = CacheUtilityValidator(semantic_guard=guard)
model_client = PrefixReorderClient(inner_client, validator=validator)
```

AutoGen component YAML 也可以开启：

```yaml
model_config:
  provider: autogen_prefix_tree.client.PrefixReorderClient
  component_type: model
  config:
    inner_client: ...
    enable_natural_language_segmentation: false
    semantic_guard:
      provider: openai-compatible
      base_url: http://127.0.0.1:11434/v1
      model: local-small-model
      min_confidence: 0.75
```

规则：

- 只有发生 block 移动时才调用 guard。
- guard 返回 `passed=False` 时回退原始 messages。
- guard 抛异常时也回退，reason 形如 `semantic_guard_failed:exception:RuntimeError`。
- 本地模型置信度低于 `min_confidence` 时也回退，reason 为 `semantic_guard_failed:local_judge_low_confidence`。
- telemetry 会记录 guard report，但不记录 prompt 正文。

## Feedback-Driven Planner Update

当前主链路已经从单纯的固定语义分类规则，推进为更明确的反馈驱动结构：

```text
LocalPromptCompiler -> HierarchicalPrefixPlanner
  -> CacheUtilityValidator
     -> Hard Constraint Gate
     -> Utility Preservation Gate
     -> Cache Hit Increased Gate
  -> PlannerFeedbackLearner
```

固定 `semantic_type` 仍然保留为解释性 metadata 和保守 fallback hint，但 planner 的长期优化入口已经转向 prompt-safe feedback。`PromptBlock` 现在会额外保留：

- `has_hard_risk`：是否命中不可交给 utility 事后判断的硬风险；
- `dependency_refs`：顺序、角色边界、工具结果、私有记忆等依赖提示；
- `summary`：不含 prompt 正文的简短 block 元信息。

Validator 分层如下：

```text
Hard Constraint Gate
  -> Utility Preservation Gate
  -> Cache Hit Increased Gate
  -> Accept / Rollback
```

第一层仍然 fail closed，检查 block 不丢失/不篡改、tools/tool choice/model args 不变、不可移动块不移动、prefix tree 覆盖完整、重写消息与 plan 严格匹配等硬约束。只有通过硬约束的 candidate 才进入 utility gate。

`UtilityPreservationReport` 已作为接口落地：

```json
{
  "is_utility_preserved": null,
  "confidence": 0.0,
  "reason": "utility_validator_not_configured",
  "checks": ["hard_constraints_passed", "interface_reserved"],
  "utility_status": "not_configured"
}
```

当前默认不训练、不调用 utility 小模型；未配置 `utility_validator` 时，utility gate 只记录 `not_configured`，不会把 utility preserved 记成 true。当前 `accepted` 只表示 hard constraint passed + cache gain increased，不表示 utility success。如果注入的本地 utility validator 返回 `is_utility_preserved=false`，validator 会直接回滚，reason 形如：

```text
utility_preservation_failed:<reason>
```

只有 utility preserved 且离线估计的 cacheable prefix 字符数增加时，重排才会被接受；否则回滚。

`PlannerFeedbackLearner` 会记录 candidate-level 与 placement-level feedback。Candidate feedback 包括 run/candidate/scenario、accepted/applied/fallback、hard constraint、utility status、cache gain 和 validator report；placement feedback 包括 placement_id、block hash、原/目标 scope、目标 agent group、risk/dependency、cache contribution、hard/cache/utility result 和 `placement_label`。

当前未训练 Utility Validator 时，accepted placement 最多标记为 `weak_positive`，不会产生 `utility_positive`。查询接口包括 `success_rate_for_placement(...)`、`weak_success_rate_for_placement(...)`、`rejection_rate_for_placement(...)`、`get_historical_score_for_placement(...)`、`get_feedback_summary_by_scope(...)`、`export_replay_dataset(...)`。它当前只做记录和汇总，不训练模型，也不自动改变 validator 行为。可以在 AutoGen wrapper 中写 JSONL：

```python
from autogen_prefix_tree import PrefixReorderClient

model_client = PrefixReorderClient(
    inner_client,
    telemetry_log_path="runs/prefix_reorder_telemetry.jsonl",
    feedback_log_path="runs/planner_feedback.jsonl",
)
```

OpenAI-compatible adapter 也会在 telemetry 中写入 `utility_preservation` 与 `planner_feedback` 字段。

### HumanEval Utility Annotation Template

本次还新增了 HumanEval 本地人工标注样例框架，用来为后续 `LocalUtilityValidator` 训练或评估准备数据。默认输出 prompt-safe metadata，不写原始 prompt 正文：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.utility_goldset_builder `
  --input datasets\HumanEval\Tasks\human_eval_TwoAgents.jsonl `
  --input datasets\HumanEval\Tasks\human_eval_GroupChatThreeAgents_sample2.jsonl `
  --output experiments\humaneval-utility-goldset\utility_goldset_template.jsonl `
  --summary experiments\humaneval-utility-goldset\summary.json `
  --max-rows 8
```

如果需要人工本地标注时查看原始 / 重排 block 文本，可以显式加 `--include-text`，并把输出文件保留在本地实验目录，不提交：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.utility_goldset_builder `
  --input datasets\HumanEval\Tasks\human_eval_TwoAgents.jsonl `
  --output experiments\humaneval-utility-goldset\utility_goldset_template.local.jsonl `
  --summary experiments\humaneval-utility-goldset\summary.local.json `
  --max-rows 8 `
  --include-text
```

模板行会包含：

- original / reordered block metadata；
- `prefix_tree_candidate`；
- `materialized_reordered_prompts`（默认只写 hash，`--include-text` 时写本地正文）；
- `placement_changes`；
- `hard_constraint_report`；
- `estimated_cache_gain`；
- moved block summary；
- hard constraint 是否通过；
- `expected_is_utility_preserved` 人工标注占位；
- `annotator_reason`、`oracle_result` 和 `label_status=unlabeled`；
- annotation focus；
- cache hit increased 离线估计。

这些样例只面向“已经通过硬约束但可能影响任务效果”的合法重排，不用于训练硬约束违规检测。

## Replay Log

`autogen_prefix_tree.replay` 提供可回放数据工具：

- `build_replay_run_record(...)` / `ReplayRunStore.save_replay_run(...)`
- `load_replay_run(run_id_or_path)`
- `replay_candidate(candidate_id, ...)`
- `revalidate_candidate(candidate_id, validator, ...)`
- `export_utility_labeling_samples(...)`
- `export_planner_training_samples(...)`

Replay 记录会保存 original prompt hashes、compiler blocks、block metadata、所有 `PrefixTreeCandidate`、materialized prompts（默认 hash）、block placements、planner score breakdown、hard validator report、utility validator report、cache gain report、final decision、fallback reason 和 feedback records。默认不保存 prompt 正文；只有本地实验显式 `include_text=True` 时才写正文。

`enable_natural_language_segmentation` 是实验开关，默认关闭。开启后只对未标注的自然语言 system prompt 做保守行级切块，用于 `baseline / rule-only / nl-segmentation` A/B；正式结论仍必须由真实 provider 与任务结果验证。

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

`PrefixReorderClient` 也可以作为 AutoGen component config 被 `ChatCompletionClient.load_component(...)` 直接加载：

```yaml
model_config:
  provider: autogen_prefix_tree.client.PrefixReorderClient
  component_type: model
  config:
    session_id: agbench-humaneval-plugin
    telemetry_log_path: runs/prefix_reorder_telemetry.jsonl
    inner_client:
      provider: autogen_ext.models.openai.OpenAIChatCompletionClient
      config:
        model: gpt-4o-mini
```

这条路径适合 AutoGenBench 模板，因为官方 scenario 通常已经写成：

```python
model_client = ChatCompletionClient.load_component(config["model_config"])
```

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

## Offline Dataset Coverage Evaluation

老师关心的一个关键问题是：当前 prompts 拆解和 validator 规则在真实 case 里是否普遍存在。没有 API key 时，可以先用真实 JSONL 请求做离线覆盖评估：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.dataset_eval `
  --input runs\openai_compatible_requests.jsonl `
  --telemetry tmp\dataset_eval\telemetry.jsonl `
  --summary tmp\dataset_eval\summary.json `
  --require-messages
```

输入可以是 OpenAI-compatible chat request JSONL，顶层或 `body` / `request_body` / `payload` 内包含 `messages` 即可。评估器会执行：

```text
OpenAI-compatible messages -> AutoGen typed messages
  -> LocalPromptCompiler
  -> HierarchicalPrefixPlanner
  -> CacheUtilityValidator
  -> prompt-safe telemetry / summary
```

输出字段包括：

- `supported_request_count`
- `semantic_coverage_supported`
- `semantic_type_counts`
- `movability_counts`
- `share_scope_counts`
- `moved_semantic_type_counts`
- `validation_reason_counts`
- `applied_count` / `fallback_count`
- `total_estimated_gain_chars`
- `rule_gap_diagnostics`

评估器不保存 prompt 正文，只保存结构化 block metadata、block id、prefix tree、validator reason 和字符级 utility proxy。

`rule_gap_diagnostics` 用于定位规则泛化缺口，不会改变插件行为。它只输出 prompt-safe 元数据，例如重复出现但当前未进入共享前缀的 system block 哈希、出现次数、语义类型、可移动性和自然语言 hint 计数。MagenticOneCoderAgent 当前会出现一个重复的 `role_identity/local_only/agent` system block，并带有 `policy_like_instruction`、`tool_or_code_policy`、`verification_policy` 等 hint；这说明它可能需要被进一步切成 role identity 与共享策略/工具使用规则，而不是直接整体提升到共享前缀。

### OpenAI-compatible Request Adapter

除了 AutoGen `ChatCompletionClient` wrapper，也可以在更通用的 OpenAI-compatible request body 层做离线重排准备。这适合后续包到不同框架的 OpenAI-compatible client/package 层，而不是从 HTTP trace 反推 prompt：

```python
from autogen_prefix_tree import OpenAICompatibleRequestAdapter

adapter = OpenAICompatibleRequestAdapter(session_id="framework-run")

cold = adapter.rewrite_request_body(openai_chat_body_1)
warm = adapter.rewrite_request_body(openai_chat_body_2)

body_to_send = warm.rewritten_body
```

也可以用 JSONL CLI 做离线 smoke：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.openai_request_adapter `
  --input runs\openai_compatible_bodies.jsonl `
  --output tmp\openai_request_adapter\rewritten_bodies.jsonl `
  --telemetry tmp\openai_request_adapter\telemetry.jsonl `
  --session-id openai-compatible-smoke `
  --enable-natural-language-segmentation `
  --semantic-guard-base-url http://127.0.0.1:11434/v1 `
  --semantic-guard-model local-small-model
```

这个 adapter 默认不发网络请求，只返回重排后的 request body 和 prompt-safe telemetry。`--enable-natural-language-segmentation` 是显式实验开关，默认关闭；若启用 `--semantic-guard-base-url`，它会额外调用本地 OpenAI-compatible judge。它保留原始 body 的非 prompt 字段，只在 validator 通过时替换 `messages`；`api_key` / `authorization` 等 secret 字段不会写入 telemetry。和 planner 当前策略一致，同一 session 的第一条重复块通常只 observe，后续 warm request 才可能 rewrite。

如果想更接近真实框架接入方式，可以启动一个本地 OpenAI-compatible forwarding proxy，然后把框架的 `base_url` 指向这个 proxy：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy `
  --host 127.0.0.1 `
  --port 8788 `
  --upstream-base-url http://real-upstream-host:8000 `
  --telemetry tmp\openai_forward_proxy\telemetry.jsonl `
  --session-id proxy-smoke `
  --semantic-guard-base-url http://127.0.0.1:11434/v1 `
  --semantic-guard-model local-small-model
```

这个 proxy 只处理 `POST /v1/chat/completions`：收到请求后先调用 `OpenAICompatibleRequestAdapter`，再把重排后的 body 转发给上游，并透传 `Authorization`。测试里用本地 fake upstream 验证了 warm request 会在转发前改写、secret/prompt 不进入 telemetry。

proxy telemetry 会混合两类 prompt-safe JSONL 记录：

- `prefix-openai-request-adapter-telemetry-v1`：planner/validator 行为、移动 block、prefix tree 和字符级 utility proxy。
- `prefix-forward-proxy-provider-telemetry-v1`：上游状态码、延迟、`usage.prompt_tokens`、cached tokens、completion tokens、total tokens，以及 provider 若返回则记录的 `actual_cost_usd`。

provider telemetry 可以直接喂给 `dataset_eval` 汇总：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.dataset_eval `
  --input tmp\openai_forward_proxy\telemetry.jsonl `
  --summary tmp\openai_forward_proxy\provider_summary.json
```

当前这些字段已由 fake upstream 测试验证；真实 cached tokens、latency、cost 仍必须来自实际上游 provider 的响应或日志。

provider usage 解析已兼容常见 OpenAI-compatible 形态：`prompt_tokens` / `completion_tokens` / `total_tokens`，`input_tokens` / `output_tokens`，顶层 `cached_tokens` / `cached_prompt_tokens` / `prompt_cache_hit_tokens` / `cache_hit_tokens` / `input_cached_tokens`，以及 `prompt_tokens_details`、`input_token_details`、`input_tokens_details` 下的 `cached_tokens` / `cache_read` / `cached` 等字段。若 provider 没有返回 `total_tokens`，会用 prompt/input tokens 加 completion/output tokens 作为 fallback；`cost_usd`、`cost`、`total_cost_usd`、`total_cost` 会被统一汇总为 cost 字段。`dataset_eval` 同样支持从嵌套 `usage` 对象或 proxy 已归一化的 `actual_*` 字段汇总 provider trace，且 token/cost 字段可以是数字或数字字符串。

如果要单独验证 OpenAI-compatible proxy 与本地 semantic guard judge 的接线，可以跑 no-network smoke：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.semantic_guard_proxy_smoke `
  --output-dir tmp\semantic_guard_proxy_smoke `
  --session-id semantic-guard-proxy-smoke `
  --judge-mode reject
```

`--judge-mode` 支持 `accept`、`reject` 和 `low-confidence`。这个 smoke 会启动 fake upstream、fake local OpenAI-compatible judge 和一个带 semantic guard 的 forwarding proxy，然后发送一冷一热两条请求。输出的 `semantic_guard_proxy_smoke_summary.json` 只包含 prompt-safe 元数据，例如 `adapter_validation_reasons`、`provider_rewrite_applied`、`warm_semantic_guard` 和 `warm_provider_rewrite_applied`。它用于验证 guard 放行/拒绝/低置信度 fail-closed 是否能在正式 proxy 路径里落 telemetry；因为 upstream 和 judge 都是 fake，它不证明真实 provider 指标或真实 semantic model 质量。

没有 API 配置时，可以先跑三组本地 proxy smoke，验证 baseline / rule-only / nl-segmentation 三条 OpenAI-compatible URL 路径、provider telemetry 解析和 `ab_eval` 报告生成是否能串起来：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.three_proxy_smoke `
  --output-dir tmp\three_proxy_smoke `
  --session-id three-proxy-smoke
```

这个 smoke 会启动一个本地 fake upstream 和三个本地 proxy。fake upstream 使用带 `/v1` 后缀的 base URL，用来防止真实 provider 常见配置 `https://.../v1` 被错误拼成 `/v1/v1/chat/completions`。输出包括三组 provider telemetry、`baseline_vs_rule_only` 和 `baseline_vs_nl_segmentation` 两份 A/B summary/report，以及顶层 `three_proxy_smoke_summary.json`。这些产物会显式标记 `fake_upstream=true`、`real_provider_metrics_available=false`；它们只证明链路和汇总器可用，不能证明真实 cache、latency、cost 或任务成功率收益。

如果已经用 `agbench_legacy_suite` 生成了 `legacy_suite_manifest.json`，也可以让 smoke 直接写入 manifest 规定的 artifact 路径，并自动跑 `agbench_legacy_collect` 和 `agbench_legacy_runbook`：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.three_proxy_smoke `
  --output-dir tmp\three_proxy_smoke `
  --session-id three-proxy-smoke `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json
```

To run both no-network preflight smokes and refresh the prompt-safe reports in one step:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight_smokes `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json `
  --output-dir runs\agbench_legacy_prefix_suite\preflight_smokes
```

This command runs `three_proxy_smoke --manifest`, runs `semantic_guard_proxy_smoke` with the manifest's fake judge settings, then rewrites the legacy runbook and real A/B preflight summaries. Its summary is `legacy_preflight_smokes_summary.json`; it is prompt-safe and explicitly marks `fake_upstream=true`, `fake_local_judge=true`, and `real_provider_metrics_available=false`. It is a wiring rehearsal only, not evidence for real cache, latency, cost, or task-success gains.

这种模式会生成 fake tabulate CSV、task result JSONL、provider telemetry、A/B summary/report 和 runbook summary。runbook 会把 fake artifacts 标为 warning，并把 `next_action` 设为 `run_real_ab_after_fake_smoke`，避免把本地 smoke 误当成真实 A/B 已完成。

`agbench_legacy_suite` 生成的 manifest 现在也会把这条命令写入 `suggested_commands.preflight_fake_smoke`，runbook 会在 `verify_suite` 之后、`start_three_proxies` 之前显示 `preflight_fake_smoke` 步骤。这个步骤是推荐预演，不替代真实三组 AutoGenBench 运行。

拿到 baseline、rule-only 和 nl-segmentation 的 provider/proxy telemetry 后，可以用 A/B 汇总器分别计算两组差值：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.ab_eval `
  --baseline runs\agbench_legacy_prefix_suite\artifacts\baseline_provider_telemetry.jsonl `
  --plugin runs\agbench_legacy_prefix_suite\artifacts\plugin_rule_only_provider_telemetry.jsonl `
  --summary runs\agbench_legacy_prefix_suite\reports\ab_baseline_vs_rule_only_summary.json

.venv\Scripts\python.exe -m autogen_prefix_tree.ab_eval `
  --baseline runs\agbench_legacy_prefix_suite\artifacts\baseline_provider_telemetry.jsonl `
  --plugin runs\agbench_legacy_prefix_suite\artifacts\plugin_nl_segmentation_provider_telemetry.jsonl `
  --summary runs\agbench_legacy_prefix_suite\reports\ab_baseline_vs_nl_segmentation_summary.json
```

如果同时拿到了 benchmark 任务级结果，可以一并合并：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.ab_eval `
  --baseline runs\agbench_legacy_prefix_suite\artifacts\baseline_provider_telemetry.jsonl `
  --plugin runs\agbench_legacy_prefix_suite\artifacts\plugin_rule_only_provider_telemetry.jsonl `
  --baseline-tasks runs\agbench_legacy_prefix_suite\artifacts\baseline_task_results.jsonl `
  --plugin-tasks runs\agbench_legacy_prefix_suite\artifacts\plugin_rule_only_task_results.jsonl `
  --summary runs\agbench_legacy_prefix_suite\reports\ab_baseline_vs_rule_only_summary.json `
  --report-md runs\agbench_legacy_prefix_suite\reports\ab_baseline_vs_rule_only_report.md `
  --min-success-rate-delta -0.02 `
  --max-p95-latency-relative-change 0.0 `
  --max-cost-per-success-relative-change 0.0 `
  --bootstrap-iterations 1000 `
  --bootstrap-seed 20260605
```

`ab_eval` 会输出 prompt tokens、cached tokens、cache hit ratio、completion tokens、total tokens、成本、平均延迟和 p50/p95/p99 延迟的 baseline/plugin/delta。提供 `--baseline-tasks` / `--plugin-tasks` 后，还会从 JSONL、JSON 或 CSV 任务结果中抽取 `task_id`、`success` / `passed`、`score` / `task_score` 等字段，输出 success rate、average score、按 task id 配对后的 delta，以及 paired bootstrap 95% CI。`combined_efficiency` 会给出 cost/latency per successful task。默认 gates 为：paired bootstrap success rate delta CI 下界不低于 -2pp、p95 latency 不上升、cost per successful task 不上升；缺少可用 CI 或真实字段时 gate 为 `unknown`，不会伪装成通过。`--report-md` 会额外生成一份可读 Markdown 表格报告，gate 表会标出 evidence、observed CI 下界、threshold 和 CI。summary/report 只保存聚合指标，不写 prompt 正文或任务正文。真实 cached tokens、cost、latency 和 task success 仍必须来自真实 provider 与 AutoGenBench 或对应 benchmark 的结果文件。

### Capturing AutoGen Requests

为了避免 HTTP 层无法无损还原 prompt，真实 AutoGen case 应在 `ChatCompletionClient.create(...)` 前一层采集 typed messages：

```python
from autogen_prefix_tree import RequestCaptureClient

inner_client = OpenAIChatCompletionClient(...)
model_client = RequestCaptureClient(
    inner_client,
    capture_log_path="runs/autogen_messages.jsonl",
    session_id="agbench-humaneval-capture",
)
```

这会把 AutoGen typed `SystemMessage` / `UserMessage` / `AssistantMessage` / `FunctionExecutionResultMessage` 的 JSON dump 写入 JSONL，`dataset_eval` 可以直接读取：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.dataset_eval `
  --input runs\autogen_messages.jsonl `
  --telemetry runs\prefix_rule_coverage.jsonl `
  --summary runs\prefix_rule_coverage_summary.json `
  --require-messages
```

如果要同时运行插件并采集插件收到的原始 typed request，可以这样包：

```python
base_client = OpenAIChatCompletionClient(...)
captured_base = RequestCaptureClient(
    base_client,
    capture_log_path="runs/autogen_messages.jsonl",
    session_id="agbench-humaneval-plugin",
)
model_client = PrefixReorderClient(
    captured_base,
    session_id="agbench-humaneval-plugin",
    telemetry_log_path="runs/prefix_reorder_telemetry.jsonl",
)
```

注意：`RequestCaptureClient` 默认保存 message content，因为这是验证语义规则泛化所必需的。只应在受控 benchmark 环境使用，不要把含 prompt 的 capture JSONL 提交进 git。若只需要链路 smoke，可设置 `include_message_content=False`；但 redacted capture 不能用于语义覆盖评估。

### Offline AutoGen AgentChat Smoke

没有 API key 时，可以用包内 `StaticResponseClient` 跑一个真实 AutoGen AgentChat capture smoke。它不访问网络，但会经过 `AssistantAgent` / `RoundRobinGroupChat` / `RequestCaptureClient` / `dataset_eval` 这条链路：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.autogen_capture_smoke `
  --capture tmp\autogen_capture_smoke\marked_capture.jsonl `
  --summary tmp\autogen_capture_smoke\marked_summary.json `
  --telemetry tmp\autogen_capture_smoke\marked_telemetry.jsonl `
  --prompt-style marked
```

`marked` 风格使用显式 section marker，用于验证正向链路。当前结果应表现为 2 条可评估请求、第一条 observe、第二条 validated，并移动 `global_task_background`、`shared_context`、`team_policy`。

还可以跑一个 MagenticOne-like 的自然语言 system prompt：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.autogen_capture_smoke `
  --capture tmp\autogen_capture_smoke\magentic_capture.jsonl `
  --summary tmp\autogen_capture_smoke\magentic_summary.json `
  --telemetry tmp\autogen_capture_smoke\magentic_telemetry.jsonl `
  --prompt-style magentic_one_like
```

这个负例当前应表现为 `applied_count=0`、`reusable_prefix_request_count=0`。规则会把自然语言 prompt 主要识别为 `role_identity` / `current_user_instruction`，而不是 `shared_context` / `team_policy` 等共享前缀类别。这说明 marker-based 规则能跑通链路，但对官方风格自然语言 prompt 的泛化还不足，后续需要更丰富的分类规则或本地小模型 semantic guard。

如果当前解释器安装了 `autogen_ext.agents.magentic_one`，可以直接捕获官方 `MagenticOneCoderAgent` 的 system prompt。当前仓库 `.venv` 缺少该扩展包，但 `autogen/.venv` 源码环境可运行：

```powershell
autogen\.venv\Scripts\python.exe -m autogen_prefix_tree.magentic_one_capture_smoke `
  --capture tmp\magentic_one_coder_capture_cli\messages.jsonl `
  --summary tmp\magentic_one_coder_capture_cli\coverage_summary.json `
  --telemetry tmp\magentic_one_coder_capture_cli\coverage_telemetry.jsonl `
  --session-id magentic-one-coder-cli
```

这个 smoke 仍然使用 `StaticResponseClient`，不访问网络。当前官方 MagenticOneCoderAgent 结果也是 `applied_count=0`、`reusable_prefix_request_count=0`，语义类型主要为 `role_identity`、`current_user_instruction` 和少量 `conversation_history`。`rule_gap_diagnostics` 会进一步显示重复的自然语言 system block 带有 policy/tool/verification hints，说明后续需要在真实 prompt capture 上统计自然语言共享块的漏识别。

### Source Prompt Corpus

还可以直接扫描本地 AutoGen 源码快照里的 system prompt / selector prompt，生成 OpenAI-compatible JSONL 后喂给 `dataset_eval`：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.source_prompt_corpus `
  --source-root tmp\framework-src\autogen\python\packages `
  --requests tmp\source_prompt_corpus\autogen_prompts.jsonl `
  --summary tmp\source_prompt_corpus\summary.json `
  --telemetry tmp\source_prompt_corpus\telemetry.jsonl `
  --candidates tmp\source_prompt_corpus\semantic_candidates.jsonl `
  --session-id autogen-source-prompts
```

这个入口会从本地框架源码中保守抽取 prompt-like 字符串，默认排除 tests。当前覆盖包括 Python AST 里的 `system_message=`、`selector_prompt`、`*_SYSTEM_MESSAGE`、短但强提示字段名如 `instructions` / `sys_prompt` 的默认值、静态拼接字符串 / 可静态还原的 f-string，以及 JSON prompt 资源里的显式 `system_message` / `selector_prompt` / `instructions` 和顶层 `slices`、`memory`、`planning` 等 prompt namespace。JSON 抽取会跳过普通 UI/schema `description` 和 `source_code`，避免把组件说明或工具源码当作 system prompt。输出 summary 会附带 `source_prompt_corpus.prompt_sources`，只包含源码路径、符号名、hash、字符数和行数，不包含 prompt 正文。

`--candidates` 会额外写出一个语义候选 JSONL，用于本地小模型或人工标注。默认只包含 candidate id、来源、hash、候选语义 hint、confidence、risk tags 和空 label 字段，不写候选文本。只有显式加 `--include-candidate-text` 才会输出候选文本。

当前本地 AutoGen 源码快照可抽取 18 个 prompt，来自 7 个源码文件。当前规则在这个 corpus 上 `applied_count=0`、`reusable_prefix_request_count=0`，同时 `rule_gap_diagnostics` 在全部 non-prefix system block 中找到 33 个语义候选：`tool_or_code_policy=11`、`verification_policy=9`、`team_policy=7`、`procedure_policy=5`、`long_stable_instruction=1`。这个结果说明 marker-based 规则不能代表真实 AutoGen prompt 的普遍覆盖率，后续应先做语义切块/分类评估，再考虑重排。

候选 JSONL 可以先用保守规则 judge 做离线标签评估，输出格式后续可替换成本地小模型 judge：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.candidate_label_eval `
  --input tmp\source_prompt_corpus\semantic_candidates.jsonl `
  --output tmp\source_prompt_corpus\semantic_candidates_labeled.jsonl `
  --summary tmp\source_prompt_corpus\semantic_candidate_label_summary.json
```

在当前 AutoGen corpus 上，保守规则 judge 对 37 条候选给出：`accept=25`、`review=11`、`reject=1`。这些标签仍然只是旁路证据，不会直接放行重排。

如果本机已经有 OpenAI-compatible 的小模型服务，也可以显式导出候选文本后让模型参与标签评估：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.source_prompt_corpus `
  --source-root tmp\framework-src\autogen\python\packages `
  --requests tmp\source_prompt_corpus\autogen_prompts_with_text.jsonl `
  --summary tmp\source_prompt_corpus\summary_with_text.json `
  --telemetry tmp\source_prompt_corpus\telemetry_with_text.jsonl `
  --candidates tmp\source_prompt_corpus\semantic_candidates_with_text.jsonl `
  --include-candidate-text
```

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.candidate_label_eval `
  --input tmp\source_prompt_corpus\semantic_candidates_with_text.jsonl `
  --output tmp\source_prompt_corpus\semantic_candidates_labeled_local_model.jsonl `
  --summary tmp\source_prompt_corpus\semantic_candidate_label_summary_local_model.json `
  --judge openai-compatible `
  --local-judge-base-url http://127.0.0.1:11434/v1 `
  --local-judge-model local-small-model
```

本地模型 judge 只是离线候选标注入口，不会改变 `PrefixReorderClient` 的默认行为。即使模型返回 `accept`，如果模型置信度低于 `--min-confidence`，或候选带有 agent identity、turn-specific、user-interaction 等高风险标签，也会被静态规则降为 `review`。

启用模型 judge 后，label summary 会额外记录 `rule_label_counts`、`model_label_counts`、`rule_to_model_label_counts`、`model_to_final_label_counts`、`rule_to_final_label_counts` 和 `static_safety_clamp_count`，用于观察本地模型与保守规则的分歧，以及静态安全钳制把多少模型 `accept` 降为 `review`。这些字段只记录标签转移计数，不包含候选文本。测试里已经用本地 fake OpenAI-compatible HTTP server 验证 `/v1/chat/completions` 调用和 JSON 标签解析；真实模型质量仍需要接入本机模型服务后再评估。

最后可以估计这些已标注候选的字符级复用潜力：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.candidate_utility_eval `
  --input tmp\source_prompt_corpus\semantic_candidates_labeled.jsonl `
  --summary tmp\source_prompt_corpus\semantic_candidate_utility_summary.json
```

默认只统计 `accept` 候选。当前 18 个 AutoGen source prompt 上，25 个 accepted 候选里只有 4 组 exact-hash 重复，估计可复用字符为 218。加 `--include-review` 后为 363。这个数字只是离线字符级潜力，不是 provider cached tokens。

### Offline Semantic Pipeline

上面几步也可以用一条命令串起来，适合反复扫描同一份 AutoGen 源码快照，并把“源码 prompt 抽取 -> 规则覆盖评估 -> 语义候选导出 -> 候选标签评估 -> 字符级 utility proxy”写到同一个输出目录：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_pipeline `
  --source-root tmp\framework-src\autogen\python\packages `
  --output-dir tmp\offline_semantic_pipeline\autogen_source `
  --session-id autogen-source-offline-semantic
```

输出目录会包含：

- `source_prompts.jsonl`
- `dataset_summary.json` / `dataset_telemetry.jsonl`
- `semantic_candidates.jsonl`
- `semantic_candidates_labeled.jsonl`
- `candidate_label_summary.json`
- `review_worklist.jsonl` / `review_worklist_summary.json`
- `candidate_utility_summary.json`
- `candidate_utility_summary_with_review.json`
- `pipeline_summary.json`

当前 AutoGen source prompt corpus 的 pipeline 结果和分步命令一致：18 个 prompt 来自 7 个源码文件，当前重排规则 `applied_count=0`、`reusable_prefix_request_count=0`；候选导出得到 37 条语义候选，保守 judge 标为 `accept=25`、`review=11`、`reject=1`；accepted-only 的 exact repeated 字符级可复用估计为 218，包含 review 作为上界时为 363。`pipeline_summary.json` 会显式标记 `real_provider_metrics_available=false`，因为这条 pipeline 不访问真实 provider，不能给出 cached tokens、latency、cost 或 task success 结论。

`pipeline_summary.json` 还会输出 `semantic_rule_gap_summary`，把当前规则覆盖和候选语义证据放在一起看。若出现 `applied_count=0`、`reusable_prefix_request_count=0`，但 `accepted_candidate_count>0`，则 `rule_gap_observed=true`。这表示真实源码 prompt 中存在保守规则认可的共享 policy/tool/verification 候选，但当前 planner/compiler 还没有把它们转成可移动 prefix block。这是规则泛化缺口证据，不是 provider cache 收益证据。

如果要评估下一阶段“自然语言 prompt 语义切块”能否缓解这个 gap，可以显式开启离线分析开关。这个开关默认关闭，不影响线上 `PrefixReorderClient`：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_pipeline `
  --source-root tmp\framework-src\autogen\python\packages `
  --output-dir tmp\offline_semantic_pipeline\autogen_source_segmented `
  --session-id autogen-source-segmented `
  --enable-natural-language-segmentation
```

在当前 AutoGen source snapshot 上，开启后 `natural_language_segmentation_enabled=true`，`applied_count=1`、`reusable_prefix_request_count=1`、`moved_semantic_type_counts={"shared_tool_description": 2}`、`total_estimated_gain_chars=92`。这说明严格行级切块能把少量低风险自然语言 tool/policy 句子变成真实可移动 block，但覆盖仍很低，只能作为下一阶段规则增强证据，不能当作真实 provider cache 收益。

如果只想把 rule-only、nl-segmentation 和对比报告一次性跑完，可以用离线 suite：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_suite `
  --source-root tmp\framework-src\autogen\python\packages `
  --output-dir tmp\offline_semantic_suite\autogen_source `
  --session-id autogen-source-offline-suite
```

For the legacy AutoGenBench path, `agbench_legacy_suite` also writes this command to
`suggested_commands.offline_semantic_suite`. The legacy runbook treats
`offline_semantic_suite_summary.json` as a prompt-safe readiness artifact, and
`agbench_legacy_preflight` now recommends `run_offline_semantic_suite` before
starting real provider A/B runs when the suite and API marker are otherwise ready.

它会生成 `rule_only\pipeline_summary.json`、`nl_segmentation\pipeline_summary.json`、`reports\offline_rule_vs_nl_summary.json`、`reports\offline_rule_vs_nl_report.md` 和顶层 `offline_semantic_suite_summary.json`。顶层 summary/report 是 prompt-safe 的，但两个 pipeline 下的 `source_prompts.jsonl` 会包含源码 prompt 文本；如果加 `--include-candidate-text`，`semantic_candidates.jsonl` 也会包含候选文本。这条 suite 仍然只做离线覆盖和字符级 utility proxy，不访问真实 provider。

如果要一次性比较多个本地框架源码快照，可以跑矩阵汇总：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_matrix `
  --source autogen=tmp\framework-src\autogen\python\packages `
  --source semantic_kernel=tmp\framework-src\semantic-kernel\python `
  --source crewai=tmp\framework-src\crewAI\lib `
  --output-dir tmp\offline_semantic_matrix\frameworks `
  --session-id frameworks-offline-matrix
```

矩阵 summary/report 仍然是 prompt-safe 的，只汇总每个 source 的 prompt 数、supported request、rule-only / nl-segmentation applied count、candidate/review count 和 rule-gap 状态。矩阵还会汇总 `review_queue_diagnostics`：review 候选的语义 hint、risk tags、label reason、top source files 和本地 judge 优先级，用来决定后续小模型语义判断应该优先覆盖哪些风险类别。每个 source 子目录下的 `source_prompts.jsonl` 仍含 prompt 文本，应留在本地输出目录。

两组 pipeline summary 可以用离线比较器汇总成一份 prompt-safe 对照报告：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_compare `
  --baseline tmp\offline_semantic_pipeline\autogen_source_gap_v2\pipeline_summary.json `
  --candidate tmp\offline_semantic_pipeline\autogen_source_segmented_v2\pipeline_summary.json `
  --baseline-label rule-only `
  --candidate-label nl-segmentation `
  --summary tmp\offline_semantic_pipeline\autogen_rule_vs_nl_summary_v3.json `
  --report-md tmp\offline_semantic_pipeline\autogen_rule_vs_nl_report_v3.md
```

当前对比结果：`applied_count_delta=1`、`reusable_prefix_request_count_delta=1`、`total_estimated_gain_chars_delta=92`、`rule_gap_resolved=true`，但 `review_queue_still_present=true`。`experiment_gate.recommendation` 因此是 `run_real_ab_smoke_with_review_candidates_disabled`：可以进入小规模真实 A/B smoke，但 review 候选仍不应自动提升到共享前缀。这仍然只是离线聚合比较，不能替代 provider/benchmark A/B。

`offline_compare` 的 summary/report 还会带 `Candidate Label Diagnostics`，展示 `judge_mode`、`label_counts`、`rule_to_model_label_counts`、`model_to_final_label_counts`、`rule_to_final_label_counts` 和 `static_safety_clamp_count`。这方便比较 rule-only、nl-segmentation、本地模型 judge 等 pipeline 变体时判断模型是否真的补充了规则覆盖，还是主要产生被静态钳制的 `accept->review`。

`judge=rule` 路径也会填充 `rule_label_counts` 和 `rule_to_final_label_counts`，因此无本地模型的真实源码矩阵仍能报告保守规则本身的 accept/review/reject 分布。

对应的 `experiment_gate` 也会输出 `local_judge_safety_clamps`：未启用本地模型时为 `unknown`；启用本地模型且没有 `accept->review` 钳制时为 `pass`；存在 `static_safety_clamp_count>0` 或 `model_to_final_label_counts["accept->review"]>0` 时为 `warn`，提醒这些候选不应直接自动提升。

这些 label diagnostics 也会进入 `delta`，例如 `model_to_final_label_counts_delta` 和 `static_safety_clamp_count_delta`。Markdown 表会显示 Baseline / Candidate / Delta 三列，用于直接观察候选 pipeline 比 baseline 多了多少 `accept->review` 钳制或其它标签转移。

如果要在 pipeline 中启用本地小模型 judge，需要同时输出候选文本：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_healthcheck `
  --base-url http://127.0.0.1:11434/v1 `
  --model local-small-model `
  --summary tmp\local_judge_healthcheck\summary.json
```

Run this prompt-safe healthcheck before a real local-judge batch. It sends one
synthetic candidate to the configured OpenAI-compatible `/chat/completions`
endpoint, verifies that the response can be parsed as JSON with an allowed
`label` and a usable `confidence`, and writes no candidate text or API key to
the summary. `ready=true` only means the endpoint/model/schema path is usable
for a small batch; it does not prove broad semantic-model quality or provider
cache performance.

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_pipeline `
  --source-root tmp\framework-src\autogen\python\packages `
  --output-dir tmp\offline_semantic_pipeline\autogen_source_local_judge `
  --session-id autogen-source-local-judge `
  --include-candidate-text `
  --judge openai-compatible `
  --local-judge-base-url http://127.0.0.1:11434/v1 `
  --local-judge-model local-small-model `
  --local-judge-scope review `
  --max-local-judge-calls 50
```

`--local-judge-scope` controls which candidates are sent to the local
OpenAI-compatible judge. The default is `all` for backward compatibility. The
recommended triage mode is `review`: first label every candidate with the
conservative rule judge, then send only rule-`review` rows to the local model.
Rule-`accept` and rule-`reject` rows keep their rule labels and are marked with
`local_judge_action=skipped_by_scope`; model-called rows are marked with
`local_judge_action=model_called`.

`--max-local-judge-calls` optionally caps actual local model calls for a small
smoke run. Rows that would exceed the cap keep their conservative rule labels
and are marked `local_judge_action=skipped_by_budget`. Missing candidate text
does not consume this budget because no model request is made.
When a budget is set, candidates are selected deterministically by
`priority_review_risk_confidence_chars_v1`: rule-`review` rows first, then more
high-risk boundary tags, lower classifier confidence, larger candidates, and
stable source/id tie breakers. The budget is therefore a priority sample, not an
input-order prefix.

The prompt-safe label summary now records `local_judge_scope`,
`max_local_judge_calls`, `local_judge_budget_strategy`,
`local_judge_action_counts`,
`local_judge_skipped_by_scope_label_counts`, and
`local_judge_skipped_by_budget_label_counts`. These counts make local judge cost
and coverage explicit without storing candidate text in the summary. Static
safety clamps still apply: a local model `accept` is downgraded to `review` when
low confidence, unsupported semantic hints, or high-risk boundary tags remain.
When a budget is set, `local_judge_budget_diagnostics` also reports
`eligible_for_budget_count`, `model_called_count`, `skipped_by_budget_count`,
`budget_exhausted`, and `budget_coverage_rate`. `offline_compare` exposes the
same field and adds a `local_judge_budget_coverage` experiment gate; the gate is
`warn` when the budget was exhausted, so a small local-model smoke is not
mistaken for full local-judge coverage.

`offline_semantic_matrix` also aggregates these diagnostics across framework
sources under `aggregate.local_judge_budget_diagnostics`. The matrix-level
summary reports how many sources had budget diagnostics, how many exhausted
their budget, total eligible/model-called/skipped counts, aggregate budget
coverage, strategy/max-call distributions, and per-source coverage rows. The
Markdown matrix report includes the same prompt-safe fields in a `Local Judge
Budget` section, so a multi-framework local-model smoke can be audited without
opening candidate-text artifacts.

`local_judge_effectiveness_diagnostics` summarizes whether the local judge
actually resolves conservative rule-`review` candidates. It reports the number
of rule-review candidates, how many were model-called, how many ended as
`accept` or `reject`, how many remain `review`, plus overall and called-only
resolution rates and label/action count maps. `offline_compare` also adds a
`local_judge_review_resolution` gate. This is still semantic triage evidence
only; it does not make provider performance claims.

The matrix summary/report also includes prompt-safe extraction diagnostics under
`aggregate.extraction_counts` and the Markdown `Prompt Extraction Diagnostics`
section. These counts show which source prompt extraction rules matched real
framework code, such as `prompt_named_constant`, `prompt_text_constant`,
`json_prompt_value`, and keyword/argument-default prompt fields.

For a no-network rehearsal of that matrix path, use the fake local judge smoke:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_local_judge_matrix_smoke `
  --source autogen=tmp\framework-src\autogen\python\packages `
  --source agentscope=tmp\framework-src\agentscope\src `
  --source crewai=tmp\framework-src\crewAI\lib `
  --output-dir tmp\offline_local_judge_matrix_smoke\frameworks_budget50 `
  --session-id frameworks-fake-local-judge-budget50 `
  --local-judge-scope review `
  --max-local-judge-calls 50 `
  --fake-judge-mode accept
```

This command starts a local fake OpenAI-compatible judge, runs the offline
semantic matrix with candidate text kept under the output directory, and writes
`offline_local_judge_matrix_smoke_summary.json`. The summary is prompt-safe and
records fake-judge request counts, semantic-hint/risk-tag counts, matrix budget
diagnostics, and `matrix_digest.candidate_label_diagnostics` for rule/model/final
label transitions plus local-judge review resolution. It validates wiring,
budget accounting, and diagnostic propagation only; it does not measure real
local-model quality or provider performance.

A no-network 11-framework run is recorded under
`tmp\offline_local_judge_matrix_smoke\frameworks11_review_resolution_20260606`.
In that smoke, the fake judge returned `accept` for every called review
candidate, but static safety clamps still left most rule-`review` candidates in
review: 40/236 were resolved overall, 40/160 model-called review candidates
were resolved, and 120 model accepts were clamped back to review. This is useful
for validating the evidence path, not for claiming real local-model quality.

To claim local-model semantic quality, first build a sampled gold-set labeling
template from real semantic candidates:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_builder `
  --input runs\agbench_legacy_prefix_suite\offline_semantic_suite\rule_only\semantic_candidates_labeled.jsonl `
  --input runs\agbench_legacy_prefix_suite\offline_semantic_suite\nl_segmentation\semantic_candidates_labeled.jsonl `
  --max-rows 60 `
  --seed 20260605 `
  --output runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_template.jsonl `
  --summary runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_template_summary.json
```

The default template is prompt-safe metadata only. For local human annotation,
rerun with `--include-text --require-text`; if candidate files were generated
without text, also pass the matching `--source-prompts` files so the builder can
recover candidate text by `parent_hash` and `text_hash`:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_builder `
  --input runs\agbench_legacy_prefix_suite\offline_semantic_suite\rule_only\semantic_candidates_labeled.jsonl `
  --input runs\agbench_legacy_prefix_suite\offline_semantic_suite\nl_segmentation\semantic_candidates_labeled.jsonl `
  --source-prompts runs\agbench_legacy_prefix_suite\offline_semantic_suite\rule_only\source_prompts.jsonl `
  --source-prompts runs\agbench_legacy_prefix_suite\offline_semantic_suite\nl_segmentation\source_prompts.jsonl `
  --include-text `
  --require-text `
  --max-rows 60 `
  --seed 20260605 `
  --output runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_template.with_text.local.jsonl `
  --summary runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_template.with_text.local.summary.json
```

Keep that with-text artifact local. The repo `.gitignore` excludes generated
`runs/**/source_prompts.jsonl`, with-text local JSONL, local annotation CSVs,
and labeled local gold JSONL so prompt/candidate text is not accidentally
committed; keep prompt-safe summaries/reports for shareable evidence. For easier
human annotation, export it to a local CSV, fill `expected_label` as `accept`,
`review`, or `reject`, then import the labeled CSV back to JSONL:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv export `
  --input runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_template.with_text.local.jsonl `
  --output runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_annotation.local.csv `
  --summary runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_annotation_export.local.summary.json
```

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv import `
  --input runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_annotation.labeled.local.csv `
  --output runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_labeled.local.jsonl `
  --summary runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_annotation_import.local.summary.json
```

The CSV contains raw candidate text and should stay local. The import command
returns nonzero and does not write the labeled JSONL unless every row has text
and `expected_label` is one of `accept`, `review`, or `reject`; use
`--allow-incomplete-output` only for local debugging. Before any model calls,
validate the completed local gold set:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_validate `
  --input runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_labeled.local.jsonl `
  --min-samples 20 `
  --min-label-count 1 `
  --summary runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_validation_summary.json
```

Then run the separate gold-label evaluation:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_quality_eval `
  --input runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_labeled.local.jsonl `
  --base-url <real-local-judge-base-url> `
  --model <real-local-judge-model> `
  --min-samples 20 `
  --min-accuracy 0.75 `
  --min-macro-f1 0.70 `
  --output runs\agbench_legacy_prefix_suite\artifacts\local_judge_quality_eval_predictions.jsonl `
  --summary runs\agbench_legacy_prefix_suite\artifacts\local_judge_quality_eval_summary.json
```

The completed gold JSONL must include `text` plus `expected_label` or
`gold_label`. The prediction output is prompt-safe: it writes text hashes,
labels, confidence, correctness, and compact metadata, but not the gold text.
`agbench_legacy_runbook` and `agbench_legacy_preflight` expose the preparation
artifact as `local_judge_goldset_template_ready` and the actual quality gate as
`local_judge_quality_eval_ready` plus `local_judge_goldset_chain_ready`. The
scoped claim `local_model_semantic_quality_on_gold_set` is allowed only when the
quality eval passes and the read-only chain verifier confirms the human-label
pipeline order; broad cross-framework semantic-rule generalization still
requires broader gold evaluation.

After `labels-from-csv`, `validate-labels`, `apply-labels`, `import`,
`local_judge_goldset_validate`, and `local_judge_quality_eval` have all produced
summaries, run the read-only chain verifier:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_chain_verify `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json `
  --summary runs\agbench_legacy_prefix_suite\artifacts\local_judge_goldset_chain_verification_summary.json `
  --report-md runs\agbench_legacy_prefix_suite\reports\local_judge_goldset_chain_verification.md
```

This verifier reads only prompt-safe summaries. It checks that manual labels
come from `expected_label`, suggestions were not auto-applied, `validate-labels`
passed before `apply-labels`, `apply-labels --require-complete` wrote the
labeled CSV, the import and gold-set validation succeeded, and quality eval ran
after that validated gold set. It does not read candidate text, call the local
judge, or prove provider metrics.

`agbench_legacy_runbook` and `agbench_legacy_preflight` also carry these
candidate label diagnostics into `offline_local_judge_matrix_status` and
`offline_local_judge_matrix_smoke_status`, so the real A/B preflight summary can
show rule-vs-local-judge label transitions and review-resolution rates without
opening nested matrix files.
They also carry `prompt_extraction_diagnostics.extraction_counts` in
`offline_semantic_suite_status`, `offline_local_judge_matrix_status`, and
`offline_local_judge_matrix_smoke_status`, so the same preflight summary can
show which prompt extraction rules matched the real source suite or matrix
before any provider-budget run.

The legacy AutoGenBench manifest/runbook now includes the same check as
`suggested_commands.offline_local_judge_matrix_smoke`. It also includes
`suggested_commands.local_judge_healthcheck`, and the real A/B preflight checks
`local_judge_healthcheck_ready` before the fake local-judge matrix smoke. The
recommended gate order is now:

```text
offline_semantic_suite_ready
local_judge_healthcheck_ready
local_judge_goldset_chain_ready
offline_local_judge_matrix_ready
offline_local_judge_matrix_smoke_ready
semantic_guard_fake_smoke_ready
real provider A/B artifacts
```

`agbench_legacy_preflight_smokes` can optionally run the real one-request local
judge healthcheck when a base URL and model are configured, and can also run the
real local-judge matrix when `--run-real-local-judge-matrix` is set. The real
matrix sends candidate text to the configured local OpenAI-compatible judge;
the fake matrix smoke stays no-network and only rehearses wiring/budget
accounting.

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight_smokes `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json `
  --output-dir runs\agbench_legacy_prefix_suite\preflight_smokes `
  --local-judge-base-url http://127.0.0.1:11434/v1 `
  --local-judge-model local-small-model `
  --run-real-local-judge-matrix `
  --local-judge-source autogen=tmp\framework-src\autogen\python\packages `
  --local-judge-source agentscope=tmp\framework-src\agentscope\src `
  --local-judge-source crewai=tmp\framework-src\crewAI\lib `
  --max-local-judge-calls 50
```

The generated suite also exposes the real local-judge matrix as a standalone
command:

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_matrix `
  --source autogen=tmp\framework-src\autogen\python\packages `
  --source agentscope=tmp\framework-src\agentscope\src `
  --source crewai=tmp\framework-src\crewAI\lib `
  --output-dir runs\agbench_legacy_prefix_suite\offline_local_judge_matrix `
  --session-id agbench-legacy-humaneval-offline-local-judge-matrix `
  --include-candidate-text `
  --judge openai-compatible `
  --local-judge-base-url http://127.0.0.1:11434/v1 `
  --local-judge-model local-small-model `
  --local-judge-scope review `
  --max-local-judge-calls 50
```

The matrix aggregate/report includes prompt-safe local-judge label diagnostics:
`rule_label_counts`, `model_label_counts`, `rule_to_model_label_counts`,
`model_to_final_label_counts`, `rule_to_final_label_counts`, and
`static_safety_clamp_count`. These counts are aggregated from the
`nl_segmentation` candidate-label path and show where the conservative rules and
local model agree, disagree, or where static safety clamps downgrade model
`accept` results to `review`. They do not include candidate text.
It also includes `local_judge_effectiveness_diagnostics`, so the matrix can show
how many rule-`review` candidates the local judge resolved versus left in the
review queue.

如果只想先生成本地小模型 / 人工复核的 review 队列，可以从 `semantic_candidates_labeled.jsonl` 导出 worklist。默认 worklist 是 prompt-safe 的，只含 hash、来源、风险标签、label reason、优先级和 recommended action，不含候选文本：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.review_worklist `
  --input tmp\offline_semantic_pipeline\autogen_source\nl_segmentation\semantic_candidates_labeled.jsonl `
  --output tmp\offline_semantic_pipeline\autogen_source\nl_segmentation\review_worklist.jsonl `
  --summary tmp\offline_semantic_pipeline\autogen_source\nl_segmentation\review_worklist_summary.json
```

真正要喂给本地 judge 或人工复核时，再显式加 `--include-text`：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.review_worklist `
  --input tmp\offline_semantic_pipeline\autogen_source\nl_segmentation\semantic_candidates_labeled.jsonl `
  --output tmp\offline_semantic_pipeline\autogen_source\nl_segmentation\review_worklist_with_text.jsonl `
  --summary tmp\offline_semantic_pipeline\autogen_source\nl_segmentation\review_worklist_with_text_summary.json `
  --include-text
```

`offline_semantic_pipeline` 现在会自动生成 prompt-safe `review_worklist.jsonl` 和 `review_worklist_summary.json`；只有在 pipeline 自身加了 `--include-candidate-text` 时，才会额外生成含文本的 `review_worklist_with_text.jsonl`，并在 suite/matrix 的 `prompt_text_artifacts` 里标出。

Review worklist rows also carry prompt-safe local-judge metadata when present:
`local_judge_action`, `rule_label`, `model_label`, and `model_confidence`.
`review_worklist_summary.json` aggregates `local_judge_action_counts`, so the
remaining review queue can be split into rows that still need candidate text,
rows skipped by `--local-judge-scope review`, rows where the local judge errored,
rows skipped by the local judge budget, and rows already judged by the model.
Recommended actions distinguish these cases, including
`defer_local_judge_until_budget_available` for budget-deferred rows and
`review_model_accept_static_clamp` for model `accept` results that were
downgraded by static safety rules.

### Generating AutoGenBench Configs

`autogenbench 0.0.3` 的官方 `run` 命令使用旧式 `OAI_CONFIG_LIST`，因此最贴近官方 benchmark 的路径是让 baseline、rule-only plugin 和 nl-segmentation plugin 都通过本地 `openai_forward_proxy` 访问真实 upstream。baseline proxy 使用 `--disabled` 只转发并记录 provider telemetry，rule-only proxy 启用当前规则重排，nl-segmentation proxy 额外启用自然语言 system prompt 行级切块。

先克隆官方 HumanEval 任务，再用其中的 task JSONL 生成三份独立 scenario：
```powershell
.venv\Scripts\autogenbench.exe clone HumanEval
```

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_suite `
  --output-dir runs\agbench_legacy_prefix_suite `
  --suite-id agbench-legacy-humaneval `
  --model deepseek-v4-pro `
  --upstream-base-url https://api.deepseek.com/v1 `
  --baseline-proxy-base-url http://127.0.0.1:8787/v1 `
  --rule-proxy-base-url http://127.0.0.1:8788/v1 `
  --nl-proxy-base-url http://127.0.0.1:8789/v1 `
  --scenario-jsonl HumanEval\Tasks\human_eval_two_agents.jsonl `
  --api-key-placeholder '${OPENAI_API_KEY}' `
  --temperature 0
```

输出包括：

- `OAI_CONFIG_LIST.baseline.json`：baseline 连接禁用重排的 pass-through proxy。
- `OAI_CONFIG_LIST.plugin_rule_only.json`：plugin rule-only 组连接默认 prefix proxy。
- `OAI_CONFIG_LIST.plugin_nl_segmentation.json`：plugin nl-segmentation 组连接启用自然语言切块的 prefix proxy。
- `OAI_CONFIG_LIST.combined.json`：包含三组 tags，便于手动筛选。
- `scenarios\agbench-legacy-humaneval.baseline.jsonl`、`scenarios\agbench-legacy-humaneval.plugin_rule_only.jsonl`、`scenarios\agbench-legacy-humaneval.plugin_nl_segmentation.jsonl`：三份文件名不同的任务 JSONL，避免 `autogenbench 0.0.3` 按 scenario 文件名复用同一个 `Results` 目录而跳过后续 A/B 组。
- `legacy_suite_manifest.json`：prompt-safe 清单和后续命令。

运行前需要开三个 proxy。baseline proxy 只转发并记录 provider telemetry：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy `
  --host 127.0.0.1 `
  --port 8787 `
  --upstream-base-url https://api.deepseek.com/v1 `
  --telemetry runs\agbench_legacy_prefix_suite\artifacts\baseline_provider_telemetry.jsonl `
  --session-id agbench-legacy-humaneval-baseline-proxy `
  --disabled
```

rule-only proxy 启用默认前缀重排：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy `
  --host 127.0.0.1 `
  --port 8788 `
  --upstream-base-url https://api.deepseek.com/v1 `
  --telemetry runs\agbench_legacy_prefix_suite\artifacts\plugin_rule_only_provider_telemetry.jsonl `
  --session-id agbench-legacy-humaneval-plugin-rule-only-proxy
```

nl-segmentation proxy 额外开启自然语言切块：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy `
  --host 127.0.0.1 `
  --port 8789 `
  --upstream-base-url https://api.deepseek.com/v1 `
  --telemetry runs\agbench_legacy_prefix_suite\artifacts\plugin_nl_segmentation_provider_telemetry.jsonl `
  --session-id agbench-legacy-humaneval-plugin-nl-segmentation-proxy `
  --enable-natural-language-segmentation
```

然后按官方 AutoGenBench 0.0.3 方式分别运行三组：

```powershell
.venv\Scripts\autogenbench.exe run `
  -c runs\agbench_legacy_prefix_suite\OAI_CONFIG_LIST.baseline.json `
  --model baseline `
  --subsample 0.1 `
  --repeat 3 `
  runs\agbench_legacy_prefix_suite\scenarios\agbench-legacy-humaneval.baseline.jsonl

.venv\Scripts\autogenbench.exe run `
  -c runs\agbench_legacy_prefix_suite\OAI_CONFIG_LIST.plugin_rule_only.json `
  --model plugin_rule_only `
  --subsample 0.1 `
  --repeat 3 `
  runs\agbench_legacy_prefix_suite\scenarios\agbench-legacy-humaneval.plugin_rule_only.jsonl

.venv\Scripts\autogenbench.exe run `
  -c runs\agbench_legacy_prefix_suite\OAI_CONFIG_LIST.plugin_nl_segmentation.json `
  --model plugin_nl_segmentation `
  --subsample 0.1 `
  --repeat 3 `
  runs\agbench_legacy_prefix_suite\scenarios\agbench-legacy-humaneval.plugin_nl_segmentation.jsonl
```

每组跑完后用官方 tabulate 输出 CSV，再用 collector 转成 `ab_eval` 需要的任务 JSONL 并生成两组 A/B 报告：

```powershell
.venv\Scripts\autogenbench.exe tabulate Results\agbench-legacy-humaneval.baseline -c > runs\agbench_legacy_prefix_suite\reports\baseline_tabulate.csv
.venv\Scripts\autogenbench.exe tabulate Results\agbench-legacy-humaneval.plugin_rule_only -c > runs\agbench_legacy_prefix_suite\reports\plugin_rule_only_tabulate.csv
.venv\Scripts\autogenbench.exe tabulate Results\agbench-legacy-humaneval.plugin_nl_segmentation -c > runs\agbench_legacy_prefix_suite\reports\plugin_nl_segmentation_tabulate.csv

.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_collect `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json `
  --baseline-tabulate-csv runs\agbench_legacy_prefix_suite\reports\baseline_tabulate.csv `
  --rule-tabulate-csv runs\agbench_legacy_prefix_suite\reports\plugin_rule_only_tabulate.csv `
  --nl-tabulate-csv runs\agbench_legacy_prefix_suite\reports\plugin_nl_segmentation_tabulate.csv `
  --summary runs\agbench_legacy_prefix_suite\reports\legacy_collection_summary.json
```

生成后可静态校验：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_suite_verify `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json `
  --summary runs\agbench_legacy_prefix_suite\reports\legacy_suite_verification_summary.json
```

也可以生成一份 prompt-safe 的真实 A/B runbook。它只读取 manifest 和 artifact 状态，不启动 proxy、不访问 API、不打印 key；用于检查命令顺序、API marker、三组 artifact 是否已经就绪：
```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_runbook `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json `
  --summary runs\agbench_legacy_prefix_suite\reports\legacy_runbook_summary.json `
  --report-md runs\agbench_legacy_prefix_suite\reports\legacy_runbook.md
```

如果要把 API 配置作为强制门槛，可以加 `--require-api-config`。缺少 `OPENAI_API_KEY` / `OAI_CONFIG_LIST` 时该命令返回非零；有 API marker 且 suite 结构正确时，`next_action` 会推进到 `start_proxies_and_run_three_autogenbench_variants`。真实跑完三组 proxy、三组 AutoGenBench 和 tabulate 后，runbook 会把下一步推进到 `run_agbench_legacy_collect` 或 `inspect_ab_reports`。

runbook 还会输出 `artifact_quality`：provider telemetry 会用 `dataset_eval` 检查是否能形成 `provider_trace`，tabulate CSV 会用 legacy collector 的 parser 检查是否能展开任务行，task JSONL 会用 `ab_eval` 的 task parser 检查是否有 success/score 指标，A/B summary/report 会检查 schema、gates 和 Markdown 关键段落。若 artifact 文件存在但不可解释，`next_action` 会变成 `fix_or_rerun_unreadable_artifacts`，避免把“文件落盘”误当成“真实 A/B 证据可用”。

如果只想得到一份更短的真实 A/B 前置门禁报告，可以跑 `agbench_legacy_preflight`。它聚合 legacy-proxy readiness、manifest/runbook 状态和 artifact quality，并区分两类状态：`ready_to_start_real_ab` 表示环境、API marker 和 suite 结构已经足够开始真实三组 A/B；`ready_to_claim_real_results` 表示真实 provider metrics、task results 和 A/B reports 都已可用。fake-upstream smoke 产物会被标成 `fake_artifacts_detected=true`，不会被算作真实结果：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight `
  --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json `
  --summary runs\agbench_legacy_prefix_suite\reports\legacy_real_ab_preflight_summary.json `
  --report-md runs\agbench_legacy_prefix_suite\reports\legacy_real_ab_preflight.md
```

When environment/API/suite checks pass but the real-source offline semantic suite
has not been run yet, this preflight returns `next_action=run_offline_semantic_suite`
and suggests the manifest's `offline_semantic_suite` command. This gate checks
rule coverage on real source prompts before spending provider budget; it is still
offline evidence and does not prove real cached-token, latency, cost, or task-success
changes.

这条 legacy 路径在 OpenAI-compatible URL 层介入，适配老师建议的“先在 openai 请求 package / compatible URL 层改”。限制是它只能看到 AutoGen 0.2 已经发出的 OpenAI-compatible request body；若需要无损 typed message 语义覆盖评估，仍应使用前面 `RequestCaptureClient` 或下面新版 `model_config` scenario 路径。

如果已有 AutoGenBench 的 baseline YAML，例如：

```yaml
model_config:
  provider: autogen_ext.models.openai.OpenAIChatCompletionClient
  config:
    model: deepseek-v4-pro
    base_url: http://host.docker.internal:8788/v1
    api_key: local-placeholder
```

可以用配置生成器产出 capture 或 plugin 版本，不需要修改 benchmark `scenario.py`：

```powershell
$env:PYTHONPATH = "F:\CodexProject\MutilAgent"
$env:AUTOGEN_ALLOWED_PROVIDER_NAMESPACES = "autogen_prefix_tree"
```

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_config `
  --base-config runs\baseline.yaml `
  --output runs\capture.yaml `
  --mode capture `
  --session-id agbench-humaneval-capture `
  --capture-log-path runs\autogen_messages.jsonl
```

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_config `
  --base-config runs\baseline.yaml `
  --output runs\plugin.yaml `
  --mode plugin `
  --session-id agbench-humaneval-plugin `
  --capture-log-path runs\autogen_messages_plugin.jsonl `
  --telemetry-log-path runs\prefix_reorder_telemetry.jsonl `
  --enable-natural-language-segmentation
```

正式 A/B 建议单独生成两份 plugin YAML：一份不带 `--enable-natural-language-segmentation` 作为 rule-only 组，另一份带该开关作为 nl-segmentation 组。

如果本机已有 OpenAI-compatible 小模型 judge，也可以直接生成带 semantic guard 的 plugin YAML：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_config `
  --base-config runs\baseline.yaml `
  --output runs\plugin_with_guard.yaml `
  --mode plugin `
  --session-id agbench-humaneval-plugin-guard `
  --capture-log-path runs\autogen_messages_plugin_guard.jsonl `
  --telemetry-log-path runs\prefix_reorder_telemetry_guard.jsonl `
  --semantic-guard-base-url http://127.0.0.1:11434/v1 `
  --semantic-guard-model local-small-model `
  --semantic-guard-min-confidence 0.75
```

没有 API 配置但想让 benchmark 场景先跑到 prompt capture 层时，可以生成 `static_capture` 配置。它会把原始 `model_config` 替换为本地静态响应 client：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_config `
  --base-config runs\baseline.yaml `
  --output runs\static_capture.yaml `
  --mode static_capture `
  --session-id agbench-static-capture `
  --capture-log-path runs\autogen_static_messages.jsonl
```

生成后的 YAML 仍然只有 `model_config`，所以 AutoGenBench HumanEval 模板里的 `ChatCompletionClient.load_component(config["model_config"])` 可以直接加载。

如果要一次性准备完整 A/B 套件，可以用 suite generator 生成 baseline、capture、rule-only plugin、nl-segmentation plugin、static_capture，以及可选 semantic guard 配置：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_suite `
  --base-config runs\baseline.yaml `
  --output-dir runs\agbench_prefix_suite `
  --suite-id agbench-prefix-humaneval
```

如果本机已有 OpenAI-compatible 小模型 judge，可以加 guard 组：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_suite `
  --base-config runs\baseline.yaml `
  --output-dir runs\agbench_prefix_suite `
  --suite-id agbench-prefix-humaneval `
  --include-guard `
  --semantic-guard-base-url http://127.0.0.1:11434/v1 `
  --semantic-guard-model local-small-model
```

输出目录会包含：

- `baseline.yaml`
- `capture.yaml`
- `plugin_rule_only.yaml`
- `plugin_nl_segmentation.yaml`
- `plugin_guard.yaml`（仅 `--include-guard` 时生成）
- `static_capture.yaml`
- `suite_manifest.json`

`suite_manifest.json` 是 prompt-safe 的实验清单，只记录配置路径、capture/telemetry/provider/task result 预期路径、readiness 前置条件，以及后续 `dataset_eval`、`offline_compare`、`ab_eval` 建议命令。它会显式标记 `real_provider_metrics_available=false`；真实 cached tokens、latency、cost 和 task success 仍必须来自后续真实 provider / AutoGenBench A/B 运行。

生成后可以先做一次不跑 benchmark 的静态校验：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.agbench_suite_verify `
  --manifest runs\agbench_prefix_suite\suite_manifest.json `
  --summary runs\agbench_prefix_suite\reports\suite_verification_summary.json
```

这个 verifier 会检查 manifest schema、各组 YAML 是否存在、是否能通过 `ChatCompletionClient.load_component(...)` 加载、rule-only / nl-segmentation / guard 开关是否和 manifest 一致。默认不要求 capture/provider/task 结果文件已经存在；真实运行后可以加 `--require-capture-artifacts`、`--require-provider-artifacts`、`--require-task-artifacts` 来确认产物是否已经落盘。

需要注意：`autogenbench 0.0.3 run` 的官方 CLI 主要读取旧式 `OAI_CONFIG_LIST`，而这里生成的 `model_config` YAML 面向的是较新的 AgentChat scenario 模板，即模板里显式调用：

```python
model_client = ChatCompletionClient.load_component(config["model_config"])
```

因此 suite generator 解决的是“同一套新版 scenario 模板下 baseline / wrapper config 的可重复生成和校验”，不是把旧式 AutoGenBench 0.2 `OAI_CONFIG_LIST` 流程自动改写成 component YAML 流程。

在 AutoGen 源码环境或较新的 AutoGen 版本中，`ChatCompletionClient.load_component(...)` 默认只信任 `autogen_core`、`autogen_agentchat`、`autogen_ext` 等命名空间。加载本插件 wrapper 前必须设置：

```powershell
$env:AUTOGEN_ALLOWED_PROVIDER_NAMESPACES = "autogen_prefix_tree"
```

如果在单独的 AutoGenBench 环境运行，还需要让该解释器能 import 本仓库：

```powershell
$env:PYTHONPATH = "F:\CodexProject\MutilAgent"
```

如果输入是旧的 proxy trace，只有 `original_prompt_hash`、token、latency、cached tokens 而没有 `messages`，评估器只能汇总 provider cache 指标：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.dataset_eval `
  --input autogen_prefix_tree_results\agbench_humaneval\...\proxy_traces\plugin.jsonl `
  --summary tmp\dataset_eval\provider_trace_summary.json
```

这种情况下 `semantic_coverage_supported=false`，不能用来证明 prompt 拆解规则的泛化能力。它只能说明真实 provider cache、latency、token usage 等请求级效果。

## Evaluation Readiness

真实 AutoGenBench / API 运行前可以先检查环境：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd .
```

`readiness` 支持三种模式：

- `--mode legacy-proxy`：当前最贴近官方 `autogenbench 0.0.3` 的 OAI_CONFIG_LIST + `openai_forward_proxy` 路径；不要求 `autogen_ext.models.openai` 或 `AUTOGEN_ALLOWED_PROVIDER_NAMESPACES`。
- `--mode component`：新版 `model_config` / `ChatCompletionClient.load_component(...)` 路径；需要 `autogen_ext.models.openai` 和 `AUTOGEN_ALLOWED_PROVIDER_NAMESPACES=autogen_prefix_tree`。
- `--mode all`：默认全量检查，用于同时准备两条路径。

检查项：

- 是否在本仓库 `.venv` 中运行
- Docker 是否可用
- `OPENAI_API_KEY` / `OAI_CONFIG_LIST` 是否存在
- `autogenbench --help` 是否能通过 CLI smoke test
- `autogen_core` 是否安装
- `autogen_ext.models.openai` 是否安装
- `AUTOGEN_ALLOWED_PROVIDER_NAMESPACES` 是否允许 `autogen_prefix_tree`

AutoGenBench 0.0.3 依赖旧版 `autogen` API。若 `autogenbench --help` 报 `ModuleNotFoundError: No module named 'autogen'`，在本仓库 `.venv` 中固定兼容版本：

```powershell
.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"
```

如果只准备官方 AutoGenBench 0.0.3 legacy 路径，建议先跑：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --mode legacy-proxy --allow-missing-api
```

如果 readiness 在 `component` 或 `all` 模式下报 `autogen_ext.models.openai not installed`，安装与当前 `autogen-core` 匹配的 OpenAI extension，例如：

```powershell
.venv\Scripts\python.exe -m pip install "autogen-ext[openai]==0.7.5"
```

加载本插件 wrapper 前还需要允许本仓库 provider namespace：

```powershell
$env:AUTOGEN_ALLOWED_PROVIDER_NAMESPACES = "autogen_prefix_tree"
```

这个命令只报告密钥配置是否存在，不读取或打印密钥内容。当前如果还没有 API 配置，可以先用宽松模式放过 API 缺失，但继续检查所选路径的其他前置条件：

```powershell
.venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --mode component --allow-missing-api
```

## 本地验证

建议始终在仓库本地虚拟环境中运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

项目根目录的 `pytest.ini` 已限制默认收集范围为 `tests/`，避免误收集 `tmp/framework-src` 等外部源码快照里的测试。
