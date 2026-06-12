from __future__ import annotations

import json

from autogen_prefix_tree.utility_goldset_builder import build_humaneval_utility_goldset_template, main


def test_humaneval_utility_goldset_template_is_prompt_safe_by_default(tmp_path) -> None:
    input_path = tmp_path / "human_eval_two_agents.jsonl"
    output_path = tmp_path / "utility_template.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(input_path, [_task("HumanEval_0")])

    result = build_humaneval_utility_goldset_template(
        input_paths=[input_path],
        output_path=output_path,
        summary_path=summary_path,
        max_rows=1,
        scenario_ids=("two_agents",),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    serialized_rows = json.dumps(rows, ensure_ascii=False)
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "prefix-utility-goldset-template-item-v1"
    assert row["expected_is_utility_preserved"] is None
    assert row["manual_labeling_required"] is True
    assert row["prompt_text_included"] is False
    assert row["validation_reason"] == "validated"
    assert row["cache_hit_increased"] is True
    assert row["moved_block_count"] > 0
    assert "def has_close_elements" not in serialized_rows
    assert "Shared project context" not in serialized_rows
    assert "def has_close_elements" not in serialized_summary
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["gold_text_written"] is False
    assert result.summary["expected_label_pending_count"] == 1
    assert json.loads(summary_path.read_text(encoding="utf-8"))["row_count"] == 1


def test_humaneval_utility_goldset_template_can_include_text_for_local_annotation(tmp_path) -> None:
    input_path = tmp_path / "human_eval_two_agents.jsonl"
    output_path = tmp_path / "utility_template.local.jsonl"
    _write_jsonl(input_path, [_task("HumanEval_0")])

    result = build_humaneval_utility_goldset_template(
        input_paths=[input_path],
        output_path=output_path,
        include_text=True,
        max_rows=1,
        scenario_ids=("two_agents",),
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["prompt_text_included"] is True
    assert any("text" in block for block in row["original_block_order"])
    assert "def has_close_elements" in json.dumps(row, ensure_ascii=False)
    assert result.summary["gold_text_written"] is True
    assert result.summary["text_artifact_note"].startswith("output JSONL contains prompt text")


def test_humaneval_utility_goldset_cli_writes_template_and_summary(tmp_path) -> None:
    input_path = tmp_path / "human_eval_two_agents.jsonl"
    output_path = tmp_path / "utility_template.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(input_path, [_task("HumanEval_0")])

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--max-rows",
            "1",
            "--scenario-id",
            "two_agents",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["row_count"] == 1
    assert summary["scenario_ids"] == ["two_agents"]


def _task(task_id: str) -> dict:
    return {
        "id": task_id,
        "template": "datasets/HumanEval/Templates/TwoAgents",
        "substitutions": {
            "scenario.py": {
                "__ENTRY_POINT__": "has_close_elements",
                "__SELECTION_METHOD__": "auto",
            },
            "prompt.txt": {
                "__PROMPT__": (
                    "from typing import List\n\n\n"
                    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
                    "    \"\"\"Return True when any two numbers are closer than threshold.\"\"\"\n"
                ),
            },
            "coding/my_tests.py": {
                "__TEST__": "def check(candidate):\n    assert candidate([1.0, 1.1], 0.2) is True\n",
            },
        },
    }


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
