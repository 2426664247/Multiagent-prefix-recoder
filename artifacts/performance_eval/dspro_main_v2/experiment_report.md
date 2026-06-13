# DS Pro Performance Evaluation

> Superseded for planner-performance claims: this v2 run manually post-processed the optimized prompt to force a shared global prefix at token 0. It is useful as a cache-mechanism / front-loaded-prefix ablation, but it is not a valid measurement of the current planner alone. Use a planner-only run where `force_front_loaded_common_prefix=false` for the primary planner result.

## Scope

- Task set: `datasets/utility_validator/tasks/perf_eval_dspro/utility_tasks.jsonl`
- Backend: `dsapi`
- Model: `deepseek-v4-pro`
- Cache isolation tag: `dspro-main-v2-20260614`
- Prompt text stored in artifacts: false
- API key stored in artifacts: false
- Training performed: false
- Cache telemetry field: DeepSeek `prompt_cache_hit_tokens` parsed into `cached_tokens`.

## Overall Metrics

- API calls: 200
- Total input/output/cached tokens: 113417 / 29131 / 25088
- Estimated cost USD: 0.259434
- Average latency seconds: 2.350
- Baseline cache hit rate: 0.0000
- Ours cache hit rate: 0.4291
- Cache hit rate lift: 42.91 pp
- Baseline success rate: 0.6400
- Ours success rate: 0.6400
- Utility preservation rate: 1.0000
- Utility drop: 0.0000

## Per-source Metrics

| source | baseline cache | ours cache | lift pp | baseline success | ours success | preservation | drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | 0.0000 | 0.2995 | 29.95 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| synthetic_agent | 0.0000 | 0.4689 | 46.89 | 0.5000 | 0.5000 | 1.0000 | 0.0000 |
| format_protocol | 0.0000 | 0.4700 | 47.00 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| role_privacy | 0.0000 | 0.4744 | 47.44 | 0.7000 | 0.7000 | 1.0000 | 0.0000 |
| history_state | 0.0000 | 0.4890 | 48.90 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Checkpoints

- Checkpoint count: 4
- Checkpoint errors: 0

## Figures

- `figures/cache_hit_rate_comparison.png` / `.svg`
- `figures/utility_comparison.png` / `.svg`
- `figures/per_source_cache_hit_rate.png` / `.svg`
- `figures/per_source_utility.png` / `.svg`
- `figures/token_cost_latency_summary.png` / `.svg`
- `figures/cache_utility_tradeoff.png` / `.svg`

## Token Budget

- Estimated input tokens before run: 109954
- Actual input tokens: 113417

## Smoke Tests

- Long-prefix cache smoke: `artifacts/performance_eval/dspro_cache_smoke/smoke_summary.json`; repeated DS Pro prompt hit `8064` cached tokens on calls 2 and 3.
- Short-prefix cache smoke: `artifacts/performance_eval/dspro_cache_smoke/short_prefix_smoke_summary.json`; shared 289-token prompt hit `256` cached tokens on calls 2 and 3.
- Main-path smoke: `artifacts/performance_eval/dspro_main_path_smoke/smoke_summary.json`; patched optimized path produced nonzero cache hits before the full v2 run.

## Pytest

- Default `pytest -q`: `324 passed, 1 warning in 73.88s`.
