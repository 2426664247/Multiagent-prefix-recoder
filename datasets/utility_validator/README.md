# Utility Validator Dataset

This directory is the unified dataset area for training a future Utility
Validator. It is split into task metadata, oracle labels, training features,
schemas, and reports.

## Directory Purpose

- `schema/`: JSON schemas for Utility Task, Utility Label, Planner Candidate,
  and Training Feature rows.
- `tasks/{smoke,train,valid,test}/`: prompt-safe Utility Task JSONL rows.
- `labels/{smoke,train,valid,test}/`: oracle result labels comparing original
  and reordered prompts.
- `features/{smoke,train,valid,test}/`: structured training features derived
  mostly from placement metadata, risk tags, dependencies, scenario type, and
  cache gain.
- `reports/`: smoke summaries, build summaries, and future cost summaries.

## Prompt-Safe vs Include-Text Artifacts

Default outputs do not store prompt bodies. They store hashes, refs, source
metadata, placement metadata, oracle reports, and cache-gain metadata.

`--include-text` is a local-only option. Any row or local artifact produced with
that option is marked `prompt_text_included=true` and should not be treated as a
prompt-safe artifact.

## Smoke Command

```powershell
$env:PYTHONPATH = "resource"
.venv\Scripts\python.exe -m autogen_prefix_tree.utility_dataset_builder `
  --repo-root . `
  --output-root datasets\utility_validator `
  --source all `
  --mode smoke `
  --backend fake `
  --max-tasks 15
```

This validates the full no-network chain:

1. generate Utility Tasks;
2. call `HierarchicalPrefixPlanner` for `PrefixTreeCandidate`;
3. materialize original/reordered prompts in memory;
4. use the fake backend;
5. run oracles;
6. export Utility Labels;
7. export Training Features;
8. write `reports/smoke_summary.json`.

## Future DS API Runs

The DS API backend is reserved and must be explicitly selected with:

```powershell
--backend dsapi --max-api-calls 5 --confirm-cost-aware
```

Small real smoke runs estimate input/output tokens, cached tokens when returned,
latency, and estimated cost, then write `reports/real_smoke_summary.json`.
They stop at `--max-api-calls`, and API keys must come from environment
variables or private config only.

Example:

```powershell
$env:PYTHONPATH = "resource"
.venv\Scripts\python.exe -m autogen_prefix_tree.utility_dataset_builder `
  --repo-root . `
  --output-root datasets\utility_validator `
  --source format_protocol `
  --mode smoke_real `
  --backend dsapi `
  --max-tasks 3 `
  --max-api-calls 5 `
  --confirm-cost-aware
```

Use `format_protocol` first because three tasks can fit in five API calls when
unchanged prompts reuse the original run.

## Current Label Status

Current smoke labels are fake pipeline labels from `fake_smoke_oracle`. They are
useful for checking schemas and the end-to-end builder, but they are not real
training data and must not be used as final Utility Validator labels.
