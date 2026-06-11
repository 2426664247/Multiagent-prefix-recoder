from __future__ import annotations

import csv
import json

from autogen_prefix_tree.local_judge_goldset_csv import (
    apply_goldset_annotation_labels,
    build_goldset_annotation_review_plan,
    export_goldset_template_csv,
    export_goldset_labels_template,
    export_goldset_labels_from_csv,
    import_goldset_labeled_csv,
    inspect_goldset_annotation_progress,
    main,
    merge_goldset_annotation_suggestions,
    suggest_goldset_annotation_labels,
    validate_goldset_labels_jsonl,
)
from autogen_prefix_tree.local_judge_goldset_validate import validate_local_judge_goldset


def test_goldset_csv_export_is_local_text_artifact_with_prompt_safe_summary(tmp_path) -> None:
    template = tmp_path / "local_judge_goldset_template.with_text.local.jsonl"
    annotation_csv = tmp_path / "annotation.local.csv"
    summary_path = tmp_path / "export_summary.json"
    _write_template(
        template,
        [
            ("gold-1", None, "Stable verification policy text."),
            ("gold-2", None, "Agent identity boundary text."),
        ],
    )

    result = export_goldset_template_csv(
        input_path=template,
        output_path=annotation_csv,
        summary_path=summary_path,
    )

    csv_text = annotation_csv.read_text(encoding="utf-8-sig")
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert "Stable verification policy text." in csv_text
    assert "Agent identity boundary text." in csv_text
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-csv-export-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["csv_text_written"] is True
    assert result.summary["gold_text_written"] is False
    assert result.summary["row_count"] == 2
    assert result.summary["expected_label_pending_count"] == 2
    assert "Stable verification policy text." not in serialized_summary
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_import_outputs_labeled_jsonl_that_validator_accepts(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.labeled.local.csv"
    labeled_jsonl = tmp_path / "local_judge_goldset_labeled.local.jsonl"
    summary_path = tmp_path / "import_summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "accept", "Stable verification policy text."),
            ("gold-2", "review", "Agent identity boundary text."),
            ("gold-3", "reject", "Turn-specific request text."),
        ],
    )

    result = import_goldset_labeled_csv(
        input_path=annotation_csv,
        output_path=labeled_jsonl,
        summary_path=summary_path,
    )
    validation = validate_local_judge_goldset(
        input_path=labeled_jsonl,
        min_samples=3,
        min_label_count=1,
    )

    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-csv-import-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["gold_text_written"] is True
    assert result.summary["expected_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert result.summary["ready_for_local_judge_goldset_validate"] is True
    assert validation.summary["ready"] is True
    assert "Stable verification policy text." not in serialized_summary
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_cli_export_and_import(tmp_path) -> None:
    template = tmp_path / "template.jsonl"
    exported = tmp_path / "annotation.csv"
    imported = tmp_path / "labeled.jsonl"
    _write_template(template, [("gold-1", None, "Reusable rule text.")])

    export_exit = main(["export", "--input", str(template), "--output", str(exported)])
    _fill_label(exported, "accept")
    import_exit = main(["import", "--input", str(exported), "--output", str(imported)])

    row = json.loads(imported.read_text(encoding="utf-8"))
    assert export_exit == 0
    assert import_exit == 0
    assert row["expected_label"] == "accept"
    assert row["text"] == "Reusable rule text."


def test_goldset_csv_progress_writes_prompt_safe_guide(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    summary_path = tmp_path / "progress.json"
    guide_path = tmp_path / "annotation_guide.md"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "accept", "Stable verification policy text."),
            ("gold-2", "", "Agent identity boundary text."),
            ("gold-3", "maybe", "Turn-specific request text."),
        ],
    )

    result = inspect_goldset_annotation_progress(
        input_path=annotation_csv,
        summary_path=summary_path,
        guide_path=guide_path,
        min_samples=3,
        min_label_count=1,
    )

    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    guide_text = guide_path.read_text(encoding="utf-8")
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-annotation-progress-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["row_count"] == 3
    assert result.summary["rows_with_expected_label_count"] == 1
    assert result.summary["expected_label_pending_count"] == 1
    assert result.summary["invalid_expected_label_count"] == 1
    assert result.summary["ready_for_labeled_jsonl_import"] is False
    assert result.summary["recommendation"] == "fill_expected_label_for_pending_rows"
    assert "Stable verification policy text." not in serialized_summary
    assert "Agent identity boundary text." not in guide_text
    assert "Local Judge Gold-Set Annotation Guide" in guide_text
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_progress_reports_prompt_safe_coverage_counts(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    guide_path = tmp_path / "annotation_guide.md"
    worklist_path = tmp_path / "annotation_worklist.jsonl"
    _write_annotation_csv(
        annotation_csv,
        [
            (
                "gold-1",
                "",
                "Stable verification policy text.",
                "accept",
                "verification_policy",
                ["conditional_instruction"],
                "high",
            ),
            (
                "gold-2",
                "maybe",
                "Agent identity boundary text.",
                "review",
                "team_policy",
                ["agent_identity_boundary", "conditional_instruction"],
                "high",
            ),
            (
                "gold-3",
                "reject",
                "Turn-specific request text.",
                "reject",
                "tool_or_code_policy",
                [],
                "low",
            ),
        ],
    )

    result = inspect_goldset_annotation_progress(
        input_path=annotation_csv,
        guide_path=guide_path,
        worklist_path=worklist_path,
        min_samples=3,
        min_label_count=1,
    )

    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    guide_text = guide_path.read_text(encoding="utf-8")
    worklist_rows = [
        json.loads(line)
        for line in worklist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert result.summary["source_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert result.summary["semantic_hint_counts"] == {
        "team_policy": 1,
        "tool_or_code_policy": 1,
        "verification_policy": 1,
    }
    assert result.summary["risk_tag_counts"] == {
        "agent_identity_boundary": 1,
        "conditional_instruction": 2,
        "unknown": 1,
    }
    assert result.summary["pending_source_label_counts"] == {"accept": 1}
    assert result.summary["pending_semantic_hint_counts"] == {"verification_policy": 1}
    assert result.summary["pending_risk_tag_counts"] == {"conditional_instruction": 1}
    assert result.summary["invalid_source_label_counts"] == {"review": 1}
    assert result.summary["invalid_semantic_hint_counts"] == {"team_policy": 1}
    assert result.summary["invalid_risk_tag_counts"] == {
        "agent_identity_boundary": 1,
        "conditional_instruction": 1,
    }
    assert result.summary["worklist_path"] == str(worklist_path)
    assert result.summary["worklist_written"] is True
    assert result.summary["worklist_row_count"] == 2
    assert result.summary["worklist_pending_count"] == 1
    assert result.summary["worklist_invalid_count"] == 1
    assert [row["label_status"] for row in worklist_rows] == ["pending", "invalid"]
    assert worklist_rows[0]["csv_row_number"] == 2
    assert worklist_rows[0]["id"] == "gold-1"
    assert worklist_rows[0]["text_hash"] == "hash-gold-1"
    assert worklist_rows[1]["expected_label"] == "maybe"
    assert "text" not in worklist_rows[0]
    assert "source_path" not in worklist_rows[0]
    assert "symbol" not in worklist_rows[0]
    assert "Stable verification policy text." not in serialized_summary
    assert "Agent identity boundary text." not in guide_text
    assert "Stable verification policy text." not in worklist_path.read_text(encoding="utf-8")
    assert "pending_semantic_hint_counts" not in guide_text
    assert "Pending semantic hints" in guide_text


def test_goldset_csv_progress_cli_returns_zero_for_incomplete_progress(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    summary_path = tmp_path / "progress.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Reusable rule text.")])

    exit_code = main(["progress", "--input", str(annotation_csv), "--summary", str(summary_path)])

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["ready_for_labeled_jsonl_import"] is False
    assert summary["expected_label_pending_count"] == 1


def test_goldset_csv_apply_labels_updates_csv_from_prompt_safe_jsonl(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    labels_jsonl = tmp_path / "annotation_labels.local.jsonl"
    labeled_csv = tmp_path / "annotation.labeled.local.csv"
    summary_path = tmp_path / "apply_summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "", "Stable verification policy text."),
            ("gold-2", "", "Agent identity boundary text."),
            ("gold-3", "", "Turn-specific request text."),
        ],
    )
    _write_label_jsonl(
        labels_jsonl,
        [
            ("gold-1", "hash-gold-1", "accept"),
            ("gold-2", "hash-gold-2", "review"),
            ("gold-3", "hash-gold-3", "reject"),
        ],
    )

    result = apply_goldset_annotation_labels(
        input_path=annotation_csv,
        labels_path=labels_jsonl,
        output_path=labeled_csv,
        summary_path=summary_path,
        require_complete=True,
    )
    imported = import_goldset_labeled_csv(
        input_path=labeled_csv,
        output_path=tmp_path / "local_judge_goldset_labeled.local.jsonl",
    )

    rows = _read_csv(labeled_csv)
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-label-apply-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["output_written"] is True
    assert result.summary["ready_for_labeled_jsonl_import"] is True
    assert result.summary["expected_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert [row["expected_label"] for row in rows] == ["accept", "review", "reject"]
    assert imported.summary["ready_for_local_judge_goldset_validate"] is True
    assert "Stable verification policy text." not in serialized_summary
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_labels_template_is_prompt_safe_and_apply_ready(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    worklist_path = tmp_path / "annotation_worklist.local.jsonl"
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    summary_path = tmp_path / "labels_template_summary.json"
    labeled_csv = tmp_path / "annotation.labeled.local.csv"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "", "Stable verification policy text.", "accept", "verification_policy", ["conditional_instruction"]),
            ("gold-2", "", "Agent identity boundary text.", "review", "team_policy", ["agent_identity_boundary"]),
        ],
    )
    inspect_goldset_annotation_progress(
        input_path=annotation_csv,
        worklist_path=worklist_path,
    )

    result = export_goldset_labels_template(
        input_path=worklist_path,
        output_path=labels_path,
        summary_path=summary_path,
    )
    label_rows = _read_jsonl(labels_path)
    for row, label in zip(label_rows, ("accept", "review"), strict=True):
        row["expected_label"] = label
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in label_rows),
        encoding="utf-8",
    )
    applied = apply_goldset_annotation_labels(
        input_path=annotation_csv,
        labels_path=labels_path,
        output_path=labeled_csv,
        require_complete=True,
    )

    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    serialized_labels = labels_path.read_text(encoding="utf-8")
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-labels-template-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["row_count"] == 2
    assert result.summary["expected_label_pending_count"] == 2
    assert result.summary["ready_for_apply_labels"] is True
    assert [row["expected_label"] for row in label_rows] == ["accept", "review"]
    assert all("text" not in row for row in label_rows)
    assert all("source_path" not in row for row in label_rows)
    assert all("symbol" not in row for row in label_rows)
    assert applied.summary["output_written"] is True
    assert "Stable verification policy text." not in serialized_summary
    assert "Agent identity boundary text." not in serialized_labels
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_labels_template_cli_writes_prompt_safe_summary(tmp_path) -> None:
    worklist_path = tmp_path / "annotation_worklist.local.jsonl"
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    summary_path = tmp_path / "labels_template_summary.json"
    _write_worklist_jsonl(
        worklist_path,
        [
            ("gold-1", "hash-gold-1", "verification_policy", ["conditional_instruction"]),
            ("gold-2", "hash-gold-2", "team_policy", ["agent_identity_boundary"]),
        ],
    )

    exit_code = main(
        [
            "labels-template",
            "--input",
            str(worklist_path),
            "--output",
            str(labels_path),
            "--summary",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(labels_path)
    assert exit_code == 0
    assert summary["row_count"] == 2
    assert summary["ready_for_apply_labels"] is True
    assert rows[0]["id"] == "gold-1"
    assert rows[0]["text_hash"] == "hash-gold-1"
    assert rows[0]["expected_label"] == ""


def test_goldset_csv_validate_labels_reports_prompt_safe_pending_status(tmp_path) -> None:
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    summary_path = tmp_path / "labels_validation_summary.json"
    _write_label_jsonl(
        labels_path,
        [
            ("gold-1", "hash-gold-1", "accept"),
            ("gold-2", "hash-gold-2", ""),
            ("gold-3", "hash-gold-3", "maybe"),
        ],
    )

    exit_code = main(
        [
            "validate-labels",
            "--input",
            str(labels_path),
            "--summary",
            str(summary_path),
            "--require-complete",
            "--min-samples",
            "3",
            "--min-label-count",
            "1",
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    serialized_summary = json.dumps(summary, ensure_ascii=False)
    assert exit_code == 1
    assert summary["schema_version"] == "prefix-local-judge-goldset-labels-validation-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["row_count"] == 3
    assert summary["valid_label_row_count"] == 1
    assert summary["expected_label_pending_count"] == 1
    assert summary["invalid_label_row_count"] == 1
    assert summary["ready_for_apply_labels"] is False
    assert summary["recommendation"] == "fix_invalid_expected_label_values"
    assert "Stable verification policy text." not in serialized_summary
    assert "source_path" not in summary
    assert "symbol" not in summary


def test_goldset_csv_validate_labels_accepts_complete_balanced_labels(tmp_path) -> None:
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    summary_path = tmp_path / "labels_validation_summary.json"
    _write_label_jsonl(
        labels_path,
        [
            ("gold-1", "hash-gold-1", "accept"),
            ("gold-2", "hash-gold-2", "reject"),
            ("gold-3", "hash-gold-3", "review"),
        ],
    )

    result = validate_goldset_labels_jsonl(
        input_path=labels_path,
        summary_path=summary_path,
        require_complete=True,
        min_samples=3,
        min_label_count=1,
    )
    exit_code = main(
        [
            "validate-labels",
            "--input",
            str(labels_path),
            "--require-complete",
            "--min-samples",
            "3",
            "--min-label-count",
            "1",
        ]
    )

    assert result.summary["ready_for_apply_labels"] is True
    assert result.summary["ready_for_goldset_import_after_apply"] is True
    assert result.summary["expected_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert result.summary["recommendation"] == "run_apply_labels_require_complete"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary
    assert exit_code == 0


def test_goldset_csv_validate_labels_audits_rule_self_confirmation(tmp_path) -> None:
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    _write_label_jsonl(
        labels_path,
        [(f"gold-{index}", f"hash-gold-{index}", "accept", "accept") for index in range(20)],
    )

    result = validate_goldset_labels_jsonl(
        input_path=labels_path,
        require_complete=True,
        min_samples=20,
        min_label_count=0,
    )

    assert result.summary["ready_for_apply_labels"] is True
    assert result.summary["source_expected_label_audited_count"] == 20
    assert result.summary["source_expected_label_match_count"] == 20
    assert result.summary["source_expected_label_mismatch_count"] == 0
    assert result.summary["source_expected_label_match_rate"] == 1.0
    assert result.summary["source_expected_label_mismatch_counts"] == {}
    assert result.summary["possible_rule_self_confirmation"] is True
    assert result.summary["gold_labels_independent_from_rules_ready"] is False


def test_goldset_csv_validate_labels_audits_rule_label_disagreements(tmp_path) -> None:
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    _write_label_jsonl(
        labels_path,
        [
            ("gold-1", "hash-gold-1", "accept", "accept"),
            ("gold-2", "hash-gold-2", "reject", "accept"),
            ("gold-3", "hash-gold-3", "review", "review"),
        ],
    )

    result = validate_goldset_labels_jsonl(
        input_path=labels_path,
        require_complete=True,
        min_samples=3,
        min_label_count=1,
    )

    assert result.summary["ready_for_apply_labels"] is True
    assert result.summary["source_expected_label_audited_count"] == 3
    assert result.summary["source_expected_label_match_count"] == 2
    assert result.summary["source_expected_label_mismatch_count"] == 1
    assert result.summary["source_expected_label_mismatch_counts"] == {"accept->reject": 1}
    assert result.summary["possible_rule_self_confirmation"] is False
    assert result.summary["gold_labels_independent_from_rules_ready"] is True


def test_goldset_csv_suggest_labels_writes_prompt_safe_local_model_suggestions(tmp_path, monkeypatch) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    summary_path = tmp_path / "suggestions_summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            (
                "gold-1",
                "",
                "Agent identity boundary text.",
                "review",
                "team_policy",
                ["agent_identity_boundary"],
            )
        ],
    )

    def fake_model(_config, _row):
        return {"label": "accept", "reason": "looks reusable", "confidence": 0.91}

    monkeypatch.setattr("autogen_prefix_tree.candidate_label_eval._call_openai_compatible_judge", fake_model)

    result = suggest_goldset_annotation_labels(
        input_path=annotation_csv,
        output_path=suggestions_path,
        summary_path=summary_path,
        base_url="http://127.0.0.1:11434/v1",
        model="local-test-model",
    )

    rows = _read_jsonl(suggestions_path)
    serialized = suggestions_path.read_text(encoding="utf-8")
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-label-suggestions-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["suggestions_ready_for_manual_review"] is True
    assert result.summary["ready_for_apply_labels"] is False
    assert result.summary["suggestion_count"] == 1
    assert result.summary["model_called_count"] == 1
    assert result.summary["static_safety_clamp_count"] == 1
    assert result.summary["suggested_label_counts"] == {"review": 1}
    assert result.summary["model_to_suggested_label_counts"] == {"accept->review": 1}
    assert rows[0]["suggested_label"] == "review"
    assert rows[0]["model_label"] == "accept"
    assert rows[0]["static_safety_clamped"] is True
    assert "Agent identity boundary text." not in serialized
    assert "source_path" not in rows[0]
    assert "symbol" not in rows[0]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_suggest_labels_cli_returns_success_for_reviewable_suggestions(tmp_path, monkeypatch) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    summary_path = tmp_path / "suggestions_summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Stable policy text.", "accept")])

    def fake_model(_config, _row):
        return {"label": "accept", "reason": "stable", "confidence": 0.91}

    monkeypatch.setattr("autogen_prefix_tree.candidate_label_eval._call_openai_compatible_judge", fake_model)

    exit_code = main(
        [
            "suggest-labels",
            "--input",
            str(annotation_csv),
            "--output",
            str(suggestions_path),
            "--summary",
            str(summary_path),
            "--base-url",
            "http://127.0.0.1:11434/v1",
            "--model",
            "local-test-model",
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["suggestions_ready_for_manual_review"] is True
    assert summary["ready_for_apply_labels"] is False


def test_goldset_csv_suggest_labels_preserves_prompt_safe_local_judge_error_type(tmp_path, monkeypatch) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    summary_path = tmp_path / "suggestions_summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Stable policy text.", "accept")])

    def fake_model(_config, _row):
        raise ValueError("invalid_model_label")

    monkeypatch.setattr("autogen_prefix_tree.candidate_label_eval._call_openai_compatible_judge", fake_model)

    result = suggest_goldset_annotation_labels(
        input_path=annotation_csv,
        output_path=suggestions_path,
        summary_path=summary_path,
        base_url="http://127.0.0.1:11434/v1",
        model="local-test-model",
    )

    rows = _read_jsonl(suggestions_path)
    serialized = suggestions_path.read_text(encoding="utf-8")
    assert result.summary["suggestions_ready_for_manual_review"] is False
    assert result.summary["local_judge_error_count"] == 1
    assert result.summary["label_reason_code_counts"] == {"local_judge_error:valueerror": 1}
    assert rows[0]["suggested_label"] == "review"
    assert rows[0]["label_reason_code"] == "local_judge_error:valueerror"
    assert "Stable policy text." not in serialized


def test_goldset_csv_merge_suggestions_writes_local_review_csv_without_filling_expected_labels(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    summary_path = tmp_path / "suggestion_review_summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            (
                "gold-1",
                "",
                "Stable verification policy text.",
                "accept",
                "verification_policy",
                ["conditional_instruction"],
            ),
            (
                "gold-2",
                "review",
                "Agent identity boundary text.",
                "review",
                "team_policy",
                ["agent_identity_boundary"],
            ),
        ],
    )
    _write_suggestion_jsonl(
        suggestions_path,
        [
            ("gold-1", "hash-gold-1", "accept", "accept", False),
            ("gold-2", "hash-gold-2", "reject", "reject", False),
        ],
    )

    result = merge_goldset_annotation_suggestions(
        input_path=annotation_csv,
        suggestions_path=suggestions_path,
        output_path=review_csv,
        summary_path=summary_path,
    )

    rows = _read_csv(review_csv)
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-suggestion-review-csv-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["ready_for_manual_review"] is True
    assert result.summary["ready_for_apply_labels"] is False
    assert result.summary["ready_for_goldset_import_after_apply"] is False
    assert result.summary["csv_text_written"] is True
    assert result.summary["row_count"] == 2
    assert result.summary["matched_suggestion_count"] == 2
    assert result.summary["expected_label_pending_count"] == 1
    assert result.summary["expected_label_counts"] == {"review": 1}
    assert result.summary["suggested_label_counts"] == {"accept": 1, "reject": 1}
    assert [row["suggested_label"] for row in rows] == ["accept", "reject"]
    assert [row["expected_label"] for row in rows] == ["", "review"]
    assert rows[0]["text"] == "Stable verification policy text."
    assert rows[0]["suggestion_notes"] == "use_as_manual_review_input_not_gold_label"
    assert "Stable verification policy text." not in serialized_summary
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_merge_suggestions_cli_returns_success_for_complete_join(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    summary_path = tmp_path / "suggestion_review_summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Stable policy text.", "accept")])
    _write_suggestion_jsonl(suggestions_path, [("gold-1", "hash-gold-1", "accept", "accept", False)])

    exit_code = main(
        [
            "merge-suggestions",
            "--input",
            str(annotation_csv),
            "--suggestions",
            str(suggestions_path),
            "--output",
            str(review_csv),
            "--summary",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_csv(review_csv)
    assert exit_code == 0
    assert summary["ready_for_manual_review"] is True
    assert summary["ready_for_apply_labels"] is False
    assert rows[0]["text"] == "Stable policy text."
    assert rows[0]["expected_label"] == ""
    assert rows[0]["suggested_label"] == "accept"


def test_goldset_csv_merge_suggestions_cli_returns_nonzero_for_incomplete_join(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    summary_path = tmp_path / "suggestion_review_summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Stable policy text.", "accept")])
    _write_suggestion_jsonl(suggestions_path, [("gold-2", "hash-gold-2", "accept", "accept", False)])

    exit_code = main(
        [
            "merge-suggestions",
            "--input",
            str(annotation_csv),
            "--suggestions",
            str(suggestions_path),
            "--output",
            str(review_csv),
            "--summary",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert summary["ready_for_manual_review"] is False
    assert summary["missing_suggestion_count"] == 1
    assert summary["unmatched_suggestion_count"] == 1
    assert review_csv.exists() is True


def test_goldset_csv_review_plan_prioritizes_disagreement_without_prompt_text(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    review_plan = tmp_path / "annotation_review_plan.local.jsonl"
    summary_path = tmp_path / "review_plan_summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "", "Stable verification policy text.", "accept", "verification_policy", []),
            (
                "gold-2",
                "",
                "Agent identity boundary text.",
                "review",
                "team_policy",
                ["agent_identity_boundary"],
            ),
            ("gold-3", "maybe", "Turn-specific request text.", "reject", "tool_or_code_policy", []),
        ],
    )
    _write_suggestion_jsonl(
        suggestions_path,
        [
            ("gold-1", "hash-gold-1", "accept", "accept", False),
            ("gold-2", "hash-gold-2", "review", "accept", True),
            ("gold-3", "hash-gold-3", "review", "review", False),
        ],
    )
    merge_goldset_annotation_suggestions(
        input_path=annotation_csv,
        suggestions_path=suggestions_path,
        output_path=review_csv,
    )

    result = build_goldset_annotation_review_plan(
        input_path=review_csv,
        output_path=review_plan,
        summary_path=summary_path,
    )

    rows = _read_jsonl(review_plan)
    serialized_plan = review_plan.read_text(encoding="utf-8")
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-review-plan-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["ready_for_manual_review"] is True
    assert result.summary["suggested_labels_not_auto_applied"] is True
    assert result.summary["plan_row_count"] == 3
    assert result.summary["plan_disagreement_row_count"] == 2
    assert result.summary["invalid_expected_label_count"] == 1
    assert rows[0]["id"] == "gold-3"
    assert rows[0]["expected_label_status"] == "invalid"
    assert rows[1]["id"] == "gold-2"
    assert "high_risk_boundary" in rows[1]["review_reasons"]
    assert "model_label_clamped_or_adjusted" in rows[1]["review_reasons"]
    assert rows[1]["suggested_label"] == "review"
    assert rows[1]["model_label"] == "accept"
    assert all("text" not in row for row in rows)
    assert all("source_path" not in row for row in rows)
    assert all("symbol" not in row for row in rows)
    assert "Stable verification policy text." not in serialized_plan
    assert "Agent identity boundary text." not in serialized_summary
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_review_plan_cli_writes_prompt_safe_plan(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    review_plan = tmp_path / "annotation_review_plan.local.jsonl"
    summary_path = tmp_path / "review_plan_summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Stable policy text.", "accept")])
    _write_suggestion_jsonl(suggestions_path, [("gold-1", "hash-gold-1", "accept", "accept", False)])
    merge_goldset_annotation_suggestions(
        input_path=annotation_csv,
        suggestions_path=suggestions_path,
        output_path=review_csv,
    )

    exit_code = main(
        [
            "review-plan",
            "--input",
            str(review_csv),
            "--output",
            str(review_plan),
            "--summary",
            str(summary_path),
            "--max-rows",
            "1",
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(review_plan)
    assert exit_code == 0
    assert summary["ready_for_manual_review"] is True
    assert summary["max_rows"] == 1
    assert summary["plan_row_count"] == 1
    assert rows[0]["id"] == "gold-1"
    assert rows[0]["recommended_action"] == "read_candidate_text_and_fill_expected_label"


def test_goldset_csv_review_plan_cli_succeeds_when_annotation_complete(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    review_plan = tmp_path / "annotation_review_plan.local.jsonl"
    summary_path = tmp_path / "review_plan_summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "accept", "Stable policy text.", "accept")])
    _write_suggestion_jsonl(suggestions_path, [("gold-1", "hash-gold-1", "accept", "accept", False)])
    merge_goldset_annotation_suggestions(
        input_path=annotation_csv,
        suggestions_path=suggestions_path,
        output_path=review_csv,
    )

    exit_code = main(
        [
            "review-plan",
            "--input",
            str(review_csv),
            "--output",
            str(review_plan),
            "--summary",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["annotation_complete"] is True
    assert summary["ready_for_manual_review"] is False
    assert summary["plan_row_count"] == 0
    assert summary["recommendation"] == "review_plan_complete_run_labels_from_csv"
    assert review_plan.read_text(encoding="utf-8") == ""


def test_goldset_csv_labels_from_review_csv_exports_prompt_safe_manual_labels_only(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    summary_path = tmp_path / "labels_from_csv_summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "", "Stable verification policy text.", "accept"),
            ("gold-2", "", "Agent identity boundary text.", "review"),
            ("gold-3", "", "Turn-specific request text.", "reject"),
        ],
    )
    _write_suggestion_jsonl(
        suggestions_path,
        [
            ("gold-1", "hash-gold-1", "accept", "accept", False),
            ("gold-2", "hash-gold-2", "review", "review", False),
            ("gold-3", "hash-gold-3", "review", "review", True),
        ],
    )
    merge_goldset_annotation_suggestions(
        input_path=annotation_csv,
        suggestions_path=suggestions_path,
        output_path=review_csv,
    )
    _fill_labels_by_id(review_csv, {"gold-1": "accept", "gold-2": "review", "gold-3": "reject"})

    result = export_goldset_labels_from_csv(
        input_path=review_csv,
        output_path=labels_path,
        summary_path=summary_path,
        min_samples=3,
        min_label_count=1,
    )
    validation = validate_goldset_labels_jsonl(
        input_path=labels_path,
        require_complete=True,
        min_samples=3,
        min_label_count=1,
    )

    rows = _read_jsonl(labels_path)
    serialized_labels = labels_path.read_text(encoding="utf-8")
    serialized_summary = json.dumps(result.summary, ensure_ascii=False)
    assert result.summary["schema_version"] == "prefix-local-judge-goldset-labels-from-csv-summary-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["row_count"] == 3
    assert result.summary["valid_label_row_count"] == 3
    assert result.summary["expected_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert result.summary["suggested_label_counts"] == {"accept": 1, "review": 2}
    assert result.summary["expected_to_suggested_label_counts"] == {
        "accept->accept": 1,
        "reject->review": 1,
        "review->review": 1,
    }
    assert result.summary["ready_for_apply_labels"] is True
    assert result.summary["ready_for_goldset_import_after_apply"] is True
    assert result.summary["labels_jsonl_text_written"] is False
    assert result.summary["suggested_labels_not_auto_applied"] is True
    assert validation.summary["ready_for_apply_labels"] is True
    assert [row["expected_label"] for row in rows] == ["accept", "review", "reject"]
    assert rows[2]["suggested_label"] == "review"
    assert all("text" not in row for row in rows)
    assert all("source_path" not in row for row in rows)
    assert all("symbol" not in row for row in rows)
    assert "Stable verification policy text." not in serialized_labels
    assert "Agent identity boundary text." not in serialized_summary
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary


def test_goldset_csv_labels_from_review_csv_does_not_auto_apply_suggestions(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    suggestions_path = tmp_path / "suggestions.local.jsonl"
    review_csv = tmp_path / "annotation_suggestion_review.local.csv"
    labels_path = tmp_path / "annotation_labels.local.jsonl"
    summary_path = tmp_path / "labels_from_csv_summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Stable policy text.", "accept")])
    _write_suggestion_jsonl(suggestions_path, [("gold-1", "hash-gold-1", "accept", "accept", False)])
    merge_goldset_annotation_suggestions(
        input_path=annotation_csv,
        suggestions_path=suggestions_path,
        output_path=review_csv,
    )

    exit_code = main(
        [
            "labels-from-csv",
            "--input",
            str(review_csv),
            "--output",
            str(labels_path),
            "--summary",
            str(summary_path),
        ]
    )

    rows = _read_jsonl(labels_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert rows[0]["expected_label"] == ""
    assert rows[0]["suggested_label"] == "accept"
    assert summary["expected_label_pending_count"] == 1
    assert summary["suggested_label_counts"] == {"accept": 1}
    assert summary["ready_for_apply_labels"] is False
    assert summary["suggested_labels_not_auto_applied"] is True


def test_goldset_csv_apply_labels_refuses_invalid_or_unmatched_prompt_safe_rows(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    labels_jsonl = tmp_path / "annotation_labels.local.jsonl"
    labeled_csv = tmp_path / "annotation.labeled.local.csv"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Reusable rule text.")])
    _write_label_jsonl(
        labels_jsonl,
        [
            ("gold-1", "hash-gold-1", "maybe"),
            ("gold-missing", "hash-gold-missing", "accept"),
        ],
    )

    result = apply_goldset_annotation_labels(
        input_path=annotation_csv,
        labels_path=labels_jsonl,
        output_path=labeled_csv,
        require_complete=True,
    )

    assert result.summary["output_written"] is False
    assert result.summary["invalid_label_row_count"] == 1
    assert result.summary["unmatched_label_count"] == 1
    assert result.summary["recommendation"] == "fix_label_jsonl_before_applying"
    assert labeled_csv.exists() is False


def test_goldset_csv_apply_labels_cli_returns_nonzero_until_complete(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    labels_jsonl = tmp_path / "annotation_labels.local.jsonl"
    labeled_csv = tmp_path / "annotation.labeled.local.csv"
    summary_path = tmp_path / "apply_summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "", "Stable policy text."),
            ("gold-2", "", "Turn-specific request text."),
        ],
    )
    _write_label_jsonl(labels_jsonl, [("gold-1", "hash-gold-1", "accept")])

    exit_code = main(
        [
            "apply-labels",
            "--input",
            str(annotation_csv),
            "--labels",
            str(labels_jsonl),
            "--output",
            str(labeled_csv),
            "--summary",
            str(summary_path),
            "--require-complete",
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert summary["output_written"] is False
    assert summary["matched_label_count"] == 1
    assert summary["expected_label_pending_count"] == 1
    assert summary["recommendation"] == "fill_remaining_expected_label_values"
    assert labeled_csv.exists() is False


def test_goldset_csv_import_cli_returns_nonzero_until_labels_are_complete(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    imported = tmp_path / "labeled.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_annotation_csv(
        annotation_csv,
        [
            ("gold-1", "", "Reusable rule text."),
            ("gold-2", "accept", "Stable policy text."),
        ],
    )

    exit_code = main(
        [
            "import",
            "--input",
            str(annotation_csv),
            "--output",
            str(imported),
            "--summary",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert summary["ready_for_local_judge_goldset_validate"] is False
    assert summary["expected_label_pending_count"] == 1
    assert summary["invalid_expected_label_count"] == 0
    assert summary["output_written"] is False
    assert imported.exists() is False
    assert summary["recommendation"] == "fill_expected_label_in_csv_before_import"


def test_goldset_csv_import_cli_returns_nonzero_for_invalid_labels(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    imported = tmp_path / "labeled.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_annotation_csv(annotation_csv, [("gold-1", "maybe", "Reusable rule text.")])

    exit_code = main(
        [
            "import",
            "--input",
            str(annotation_csv),
            "--output",
            str(imported),
            "--summary",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert summary["ready_for_local_judge_goldset_validate"] is False
    assert summary["expected_label_pending_count"] == 0
    assert summary["invalid_expected_label_count"] == 1
    assert summary["output_written"] is False
    assert imported.exists() is False
    assert summary["recommendation"] == "fix_invalid_expected_label_values"


def test_goldset_csv_import_can_explicitly_write_incomplete_probe_output(tmp_path) -> None:
    annotation_csv = tmp_path / "annotation.local.csv"
    imported = tmp_path / "labeled.jsonl"
    _write_annotation_csv(annotation_csv, [("gold-1", "", "Reusable rule text.")])

    result = import_goldset_labeled_csv(
        input_path=annotation_csv,
        output_path=imported,
        allow_incomplete_output=True,
    )

    assert result.summary["ready_for_local_judge_goldset_validate"] is False
    assert result.summary["allow_incomplete_output"] is True
    assert result.summary["output_written"] is True
    assert imported.exists() is True


def _write_template(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "prefix-local-judge-goldset-template-item-v1",
                    "id": row_id,
                    "expected_label": label,
                    "label_options": ["accept", "reject", "review"],
                    "semantic_hint": "verification_policy",
                    "risk_tags": [],
                    "source_label": "review",
                    "text_hash": f"hash-{row_id}",
                    "text": text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row_id, label, text in rows
        ),
        encoding="utf-8",
    )


def _write_annotation_csv(path, rows) -> None:
    fieldnames = [
        "id",
        "expected_label",
        "annotation_notes",
        "label_options",
        "semantic_hint",
        "risk_tags",
        "source_label",
        "rule_label",
        "model_label",
        "review_priority",
        "recommended_action",
        "source_path",
        "symbol",
        "extraction",
        "parent_hash",
        "text_hash",
        "text",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_id = row[0]
            label = row[1]
            text = row[2]
            source_label = row[3] if len(row) > 3 else "review"
            semantic_hint = row[4] if len(row) > 4 else "verification_policy"
            risk_tags = row[5] if len(row) > 5 else []
            review_priority = row[6] if len(row) > 6 else ""
            writer.writerow(
                {
                    "id": row_id,
                    "expected_label": label,
                    "annotation_notes": "",
                    "label_options": json.dumps(["accept", "reject", "review"]),
                    "semantic_hint": semantic_hint,
                    "risk_tags": json.dumps(risk_tags),
                    "source_label": source_label,
                    "review_priority": review_priority,
                    "text_hash": f"hash-{row_id}",
                    "text": text,
                }
            )


def _write_label_jsonl(path, rows) -> None:
    def _row_dict(row):
        row_id, text_hash, label = row[:3]
        payload = {
            "schema_version": "prefix-local-judge-goldset-annotation-label-v1",
            "id": row_id,
            "text_hash": text_hash,
            "expected_label": label,
        }
        if len(row) > 3:
            payload["source_label"] = row[3]
        return payload

    path.write_text(
        "".join(
            json.dumps(_row_dict(row), ensure_ascii=False, sort_keys=True)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_suggestion_jsonl(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "prefix-local-judge-goldset-label-suggestion-v1",
                    "id": row_id,
                    "text_hash": text_hash,
                    "suggested_label": suggested_label,
                    "model_label": model_label,
                    "model_confidence": 0.91,
                    "confidence_meets_min": True,
                    "rule_label": "review",
                    "source_label": "review",
                    "local_judge_action": "model_called",
                    "label_reason_code": "model_label",
                    "static_safety_clamped": static_safety_clamped,
                    "semantic_hint": "verification_policy",
                    "risk_tags": [],
                    "risk_tag_count": 0,
                    "char_count": 120,
                    "line_count": 1,
                    "existing_expected_label": None,
                    "recommended_action": "use_as_manual_review_input_not_gold_label",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row_id, text_hash, suggested_label, model_label, static_safety_clamped in rows
        ),
        encoding="utf-8",
    )


def _write_worklist_jsonl(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "prefix-local-judge-goldset-annotation-worklist-item-v1",
                    "csv_row_number": index + 2,
                    "id": row_id,
                    "label_options": ["accept", "reject", "review"],
                    "label_status": "pending",
                    "recommended_action": "fill_expected_label",
                    "risk_tags": risk_tags,
                    "semantic_hint": semantic_hint,
                    "source_label": "review",
                    "text_hash": text_hash,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for index, (row_id, text_hash, semantic_hint, risk_tags) in enumerate(rows)
        ),
        encoding="utf-8",
    )


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fill_label(path, label: str) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    for row in rows:
        row["expected_label"] = label
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fill_labels_by_id(path, labels_by_id: dict[str, str]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id in labels_by_id:
            row["expected_label"] = labels_by_id[row_id]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
