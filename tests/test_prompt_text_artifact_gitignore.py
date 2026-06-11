from __future__ import annotations

from pathlib import Path


def test_generated_prompt_text_artifacts_are_gitignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    required_patterns = {
        "runs/**/source_prompts.jsonl",
        "runs/**/semantic_candidates*.jsonl",
        "runs/**/review_worklist_with_text*.jsonl",
        "runs/**/*with_text*.jsonl",
        "runs/**/*.local.csv",
        "runs/**/*.local.md",
        "runs/**/local_judge_goldset_annotation_worklist.local.jsonl",
        "runs/**/local_judge_goldset_annotation_labels.local.jsonl",
        "runs/**/local_judge_goldset_labeled.local.jsonl",
    }

    for pattern in required_patterns:
        assert pattern in gitignore
