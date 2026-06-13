# DS Pro Performance Evaluation

> Superseded: this first run under-counted cache hits because the parser did not read DeepSeek's `prompt_cache_hit_tokens` field and the optimized prompt did not force a shared prefix at token 0. Use `artifacts/performance_eval/dspro_main_v2/` instead.

## Scope

- Task set: `datasets/utility_validator/tasks/perf_eval_dspro/utility_tasks.jsonl`
- Backend: `dsapi`
- Model: `deepseek-v4-pro`
- Prompt text stored in artifacts: false
- API key stored in artifacts: false
- Training performed: false

## Overall Metrics

- API calls: 200
- Total input/output/cached tokens: 102896 / 28728 / 0
- Estimated cost USD: 0.279012
- Average latency seconds: 2.723
- Baseline cache hit rate: 0.0000
- Ours cache hit rate: 0.0000
- Cache hit rate lift: 0.00 pp
- Baseline success rate: 0.6400
- Ours success rate: 0.6400
- Utility preservation rate: 1.0000
- Utility drop: 0.0000

## Per-source Metrics

| source | baseline cache | ours cache | lift pp | baseline success | ours success | preservation | drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| synthetic_agent | 0.0000 | 0.0000 | 0.00 | 0.5000 | 0.5000 | 1.0000 | 0.0000 |
| format_protocol | 0.0000 | 0.0000 | 0.00 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| role_privacy | 0.0000 | 0.0000 | 0.00 | 0.7000 | 0.7000 | 1.0000 | 0.0000 |
| history_state | 0.0000 | 0.0000 | 0.00 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

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

- Estimated input tokens before run: 104833
- Actual input tokens: 102896

## Pytest

- Default `pytest -q`: failed during setup because Windows denied access while cleaning `C:\tmp\mutilagent-pytest-tmp`; no business assertions ran.
- Workspace basetemp rerun: `324 passed, 1 warning in 70.39s` with `python -m pytest -q --basetemp=.pytest-tmp\dspro-main`.
