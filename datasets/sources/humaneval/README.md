# HumanEval Source

`datasets/HumanEval/` is the existing source of HumanEval-style tasks. The new
loader reads the JSONL files under `datasets/HumanEval/Tasks/`, checks for
prompt, test, and entry point fields, and converts a small sample into unified
Utility Task rows.

Prompt and test text are not copied into default smoke outputs. When
`--include-text` is used, local artifacts are written under
`datasets/sources/humaneval/processed/local_artifacts/` and rows are marked with
`prompt_text_included=true`.
