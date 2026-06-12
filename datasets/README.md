# Utility Validator Datasets

This directory separates raw/source data from first-stage Utility Validator
training artifacts.

## Layout

- `HumanEval/`: existing AutoGenBench-style HumanEval tasks. This source is
  readable and reused by the new `sources/humaneval` loader; it is not deleted
  or treated as the only dataset source.
- `sources/`: source-specific raw and processed areas. These are inputs or local
  intermediate artifacts, not final training splits.
- `utility_validator/`: unified task, label, feature, schema, and report outputs
  for Utility Validator dataset construction.

## Source vs Training Data

Source data may contain prompt bodies, tests, private task text, or local
artifacts. Training data under `utility_validator/` defaults to prompt-safe
metadata and structured features. Prompt text is only written when a command uses
`--include-text`; those local artifacts are marked with
`prompt_text_included=true`.

## Smoke Build

No-network smoke data can be regenerated with:

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

The smoke run writes:

- `datasets/utility_validator/tasks/smoke/utility_tasks.jsonl`
- `datasets/utility_validator/labels/smoke/utility_labels.jsonl`
- `datasets/utility_validator/features/smoke/training_features.jsonl`
- `datasets/utility_validator/reports/smoke_summary.json`

Small real DS API smoke labeling must be explicit and cost-aware:

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

Real smoke labels go to `labels/smoke_real/`, features to
`features/smoke_real/`, and cost summary to `reports/real_smoke_summary.json`.

## Cost Control

The default backend is `fake` and makes zero real API calls. A future DS API run
must be explicit: `--backend dsapi --max-api-calls N --confirm-cost-aware`.
API keys must come from environment/config and must never be written into this
directory.

The current smoke labels are `fake_smoke_oracle` labels for pipeline validation
only. They are not real Utility Validator training labels.
