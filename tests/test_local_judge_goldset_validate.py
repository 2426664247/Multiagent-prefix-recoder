from __future__ import annotations

import json

from autogen_prefix_tree.local_judge_goldset_validate import main, validate_local_judge_goldset


def test_goldset_validate_passes_balanced_labeled_text_without_leaking_text(tmp_path) -> None:
    gold = tmp_path / "local_judge_goldset_labeled.local.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_gold(
        gold,
        [
            ("accept-1", "accept", "Stable reusable verification policy."),
            ("review-1", "review", "Agent identity boundary needs human review."),
            ("reject-1", "reject", "Current turn user-specific request."),
        ],
    )

    result = validate_local_judge_goldset(
        input_path=gold,
        summary_path=summary_path,
        min_samples=3,
        min_label_count=1,
    )

    serialized = json.dumps(result.summary, ensure_ascii=False)
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-validation-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["ready"] is True
    assert result.summary["ready_for_local_judge_quality_eval"] is True
    assert result.summary["valid_row_count"] == 3
    assert result.summary["invalid_row_count"] == 0
    assert result.summary["expected_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert result.summary["gold_text_written"] is False
    assert result.summary["recommendation"] == "run_local_judge_quality_eval"
    assert "Stable reusable verification policy" not in serialized
    assert "Agent identity boundary" not in serialized
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_validate_rejects_missing_text_and_invalid_labels(tmp_path) -> None:
    gold = tmp_path / "bad_gold.jsonl"
    _write_jsonl(
        gold,
        [
            {"id": "row-1", "expected_label": "accept"},
            {"id": "row-2", "expected_label": None, "text": "Has text but missing label."},
            {"id": "row-3", "gold_label": "maybe", "text": "Invalid label text."},
        ],
    )

    result = validate_local_judge_goldset(input_path=gold, min_samples=1, min_label_count=1)

    assert result.summary["ready"] is False
    assert result.summary["valid_row_count"] == 0
    assert result.summary["invalid_row_count"] == 3
    assert result.summary["missing_text_count"] == 1
    assert result.summary["invalid_expected_label_count"] == 2
    assert result.summary["recommendation"] == "fix_missing_text_or_expected_labels"
    assert "Has text but missing label." not in json.dumps(result.summary, ensure_ascii=False)


def test_goldset_validate_requires_min_samples_and_label_balance(tmp_path) -> None:
    gold = tmp_path / "unbalanced_gold.jsonl"
    _write_gold(
        gold,
        [
            ("accept-1", "accept", "Stable shared instruction."),
            ("accept-2", "accept", "Another stable shared instruction."),
        ],
    )

    too_small = validate_local_judge_goldset(input_path=gold, min_samples=3, min_label_count=1)
    unbalanced = validate_local_judge_goldset(input_path=gold, min_samples=2, min_label_count=1)

    assert too_small.summary["ready"] is False
    assert too_small.summary["recommendation"] == "add_more_gold_labeled_rows"
    assert unbalanced.summary["ready"] is False
    assert unbalanced.summary["labels_with_min_count"] == 1
    assert unbalanced.summary["recommendation"] == "add_label_balance_before_quality_eval"


def test_goldset_validate_cli_exit_codes(tmp_path) -> None:
    good = tmp_path / "good.jsonl"
    bad = tmp_path / "bad.jsonl"
    _write_gold(
        good,
        [
            ("accept-1", "accept", "Stable reusable verification policy."),
            ("review-1", "review", "Agent identity boundary needs human review."),
            ("reject-1", "reject", "Current turn user-specific request."),
        ],
    )
    _write_jsonl(bad, [{"id": "bad", "expected_label": "accept"}])

    passed = main(["--input", str(good), "--min-samples", "3", "--summary", str(tmp_path / "good_summary.json")])
    failed = main(["--input", str(bad), "--min-samples", "1", "--summary", str(tmp_path / "bad_summary.json")])

    assert passed == 0
    assert failed == 1
    assert json.loads((tmp_path / "good_summary.json").read_text(encoding="utf-8"))["ready"] is True
    assert json.loads((tmp_path / "bad_summary.json").read_text(encoding="utf-8"))["ready"] is False


def _write_gold(path, rows) -> None:
    _write_jsonl(
        path,
        [
            {
                "id": example_id,
                "expected_label": label,
                "semantic_hint": "verification_policy",
                "risk_tags": [],
                "text": text,
            }
            for example_id, label, text in rows
        ],
    )


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
