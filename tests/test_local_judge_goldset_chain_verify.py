from __future__ import annotations

import json

from autogen_prefix_tree.local_judge_goldset_chain_verify import (
    main,
    verify_local_judge_goldset_chain,
)


def test_goldset_chain_verify_accepts_complete_manual_label_chain(tmp_path) -> None:
    paths = _write_chain_summaries(tmp_path, apply_ready=True)
    summary_path = tmp_path / "chain_summary.json"
    report_path = tmp_path / "chain_report.md"

    result = verify_local_judge_goldset_chain(
        labels_from_csv_summary_path=paths["labels_from_csv"],
        labels_validation_summary_path=paths["labels_validation"],
        apply_labels_summary_path=paths["apply_labels"],
        import_summary_path=paths["import_goldset"],
        goldset_validation_summary_path=paths["goldset_validation"],
        quality_eval_summary_path=paths["quality_eval"],
        summary_path=summary_path,
        report_path=report_path,
    )
    exit_code = main(
        [
            "--labels-from-csv-summary",
            str(paths["labels_from_csv"]),
            "--labels-validation-summary",
            str(paths["labels_validation"]),
            "--apply-labels-summary",
            str(paths["apply_labels"]),
            "--import-summary",
            str(paths["import_goldset"]),
            "--goldset-validation-summary",
            str(paths["goldset_validation"]),
            "--quality-eval-summary",
            str(paths["quality_eval"]),
        ]
    )

    assert result.summary["schema_version"] == "prefix-local-judge-goldset-chain-verification-v1"
    assert result.summary["prompt_safe_summary"] is True
    assert result.summary["ready"] is True
    assert result.summary["chain_ready"] is True
    assert result.summary["next_action"] == "goldset_chain_ready_for_semantic_quality_claim"
    assert result.summary["blocking_reasons"] == []
    assert "local_goldset_chain_ready" in {claim["id"] for claim in result.summary["allowed_claims"]}
    assert "real_provider_cache_or_task_success" in {claim["id"] for claim in result.summary["prohibited_claims"]}
    assert "Stable verification policy text" not in json.dumps(result.summary, ensure_ascii=False)
    assert "Local Judge Gold-Set Chain Verification" in report_path.read_text(encoding="utf-8")
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result.summary
    assert exit_code == 0


def test_goldset_chain_verify_blocks_quality_claim_when_apply_labels_missing(tmp_path) -> None:
    paths = _write_chain_summaries(tmp_path, apply_ready=False)

    result = verify_local_judge_goldset_chain(
        labels_from_csv_summary_path=paths["labels_from_csv"],
        labels_validation_summary_path=paths["labels_validation"],
        apply_labels_summary_path=paths["apply_labels"],
        import_summary_path=paths["import_goldset"],
        goldset_validation_summary_path=paths["goldset_validation"],
        quality_eval_summary_path=paths["quality_eval"],
    )

    failed = {check["name"] for check in result.summary["checks"] if not check["ok"]}
    assert result.summary["ready"] is False
    assert result.summary["chain_ready"] is False
    assert result.summary["next_action"] == "run_apply_labels_require_complete"
    assert "apply_labels:ready" in failed
    assert "order:import_goldset_after_apply_labels" in failed
    assert "local_goldset_chain_ready" in {claim["id"] for claim in result.summary["prohibited_claims"]}


def test_goldset_chain_verify_distinguishes_low_quality_eval_from_missing_eval(tmp_path) -> None:
    paths = _write_chain_summaries(tmp_path, apply_ready=True, quality_ready=False)

    result = verify_local_judge_goldset_chain(
        labels_from_csv_summary_path=paths["labels_from_csv"],
        labels_validation_summary_path=paths["labels_validation"],
        apply_labels_summary_path=paths["apply_labels"],
        import_summary_path=paths["import_goldset"],
        goldset_validation_summary_path=paths["goldset_validation"],
        quality_eval_summary_path=paths["quality_eval"],
    )

    quality_step = next(step for step in result.summary["steps"] if step["name"] == "quality_eval")
    assert result.summary["ready"] is False
    assert result.summary["next_action"] == "improve_or_recalibrate_local_judge_quality_eval"
    assert quality_step["status"] == "warn"
    assert "below semantic-quality thresholds" in quality_step["evidence"]


def test_goldset_chain_verify_reads_manifest_paths_and_suggests_next_command(tmp_path) -> None:
    paths = _write_chain_summaries(tmp_path, apply_ready=False)
    manifest = tmp_path / "legacy_suite_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "prefix-legacy-agbench-suite-manifest-v1",
                "local_judge": {
                    "goldset_annotation_labels_from_csv_summary_path": str(paths["labels_from_csv"]),
                    "goldset_annotation_labels_validation_summary_path": str(paths["labels_validation"]),
                    "goldset_annotation_apply_summary_path": str(paths["apply_labels"]),
                    "goldset_annotation_import_summary_path": str(paths["import_goldset"]),
                    "goldset_validation_summary_path": str(paths["goldset_validation"]),
                    "quality_eval_summary_path": str(paths["quality_eval"]),
                },
                "suggested_commands": {
                    "local_judge_goldset_template": [
                        "local_judge_goldset_csv labels-from-csv",
                        "local_judge_goldset_csv validate-labels",
                        "local_judge_goldset_csv apply-labels --require-complete",
                        "local_judge_goldset_csv import",
                    ],
                    "local_judge_goldset_validate": ["local_judge_goldset_validate"],
                    "local_judge_quality_eval": ["local_judge_quality_eval"],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = verify_local_judge_goldset_chain(manifest_path=manifest)

    assert result.summary["next_action"] == "run_apply_labels_require_complete"
    assert result.summary["suggested_commands"] == ["local_judge_goldset_csv apply-labels --require-complete"]


def test_goldset_chain_verify_rejects_rule_self_confirming_labels(tmp_path) -> None:
    paths = _write_chain_summaries(tmp_path, apply_ready=True, independent_labels=False)

    result = verify_local_judge_goldset_chain(
        labels_from_csv_summary_path=paths["labels_from_csv"],
        labels_validation_summary_path=paths["labels_validation"],
        apply_labels_summary_path=paths["apply_labels"],
        import_summary_path=paths["import_goldset"],
        goldset_validation_summary_path=paths["goldset_validation"],
        quality_eval_summary_path=paths["quality_eval"],
    )

    labels_validation = next(step for step in result.summary["steps"] if step["name"] == "labels_validation")
    assert result.summary["ready"] is False
    assert result.summary["next_action"] == "run_validate_labels_require_complete"
    assert labels_validation["status"] == "fail"
    assert "independent" in labels_validation["evidence"]


def _write_chain_summaries(
    tmp_path,
    *,
    apply_ready: bool,
    independent_labels: bool = True,
    quality_ready: bool = True,
):
    paths = {
        "labels_from_csv": tmp_path / "labels_from_csv_summary.json",
        "labels_validation": tmp_path / "labels_validation_summary.json",
        "apply_labels": tmp_path / "apply_summary.json",
        "import_goldset": tmp_path / "import_summary.json",
        "goldset_validation": tmp_path / "goldset_validation_summary.json",
        "quality_eval": tmp_path / "quality_eval_summary.json",
    }
    _write_json(
        paths["labels_from_csv"],
        {
            "schema_version": "prefix-local-judge-goldset-labels-from-csv-summary-v1",
            "prompt_safe_summary": True,
            "output_written": True,
            "row_count": 3,
            "valid_label_row_count": 3,
            "expected_label_pending_count": 0,
            "invalid_label_row_count": 0,
            "rows_missing_id_count": 0,
            "rows_missing_text_hash_count": 0,
            "duplicate_label_key_count": 0,
            "expected_label_counts": {"accept": 1, "reject": 1, "review": 1},
            "sample_count_ready": True,
            "label_balance_ready": True,
            "ready_for_apply_labels": True,
            "ready_for_goldset_import_after_apply": True,
            "labels_jsonl_text_written": False,
            "suggested_labels_not_auto_applied": True,
            "recommendation": "run_validate_labels_then_apply_labels",
        },
    )
    _write_json(
        paths["labels_validation"],
        {
            "schema_version": "prefix-local-judge-goldset-labels-validation-summary-v1",
            "prompt_safe_summary": True,
            "require_complete": True,
            "row_count": 3,
            "valid_label_row_count": 3,
            "expected_label_pending_count": 0,
            "invalid_label_row_count": 0,
            "rows_missing_id_count": 0,
            "rows_missing_text_hash_count": 0,
            "duplicate_label_key_count": 0,
            "expected_label_counts": {"accept": 1, "reject": 1, "review": 1},
            "possible_rule_self_confirmation": not independent_labels,
            "gold_labels_independent_from_rules_ready": independent_labels,
            "structure_ready": True,
            "complete_labels_ready": True,
            "sample_count_ready": True,
            "label_balance_ready": True,
            "ready_for_apply_labels": True,
            "ready_for_goldset_import_after_apply": True,
            "recommendation": "run_apply_labels_require_complete",
        },
    )
    _write_json(
        paths["apply_labels"],
        {
            "schema_version": "prefix-local-judge-goldset-label-apply-summary-v1",
            "prompt_safe_summary": True,
            "require_complete": True,
            "output_written": apply_ready,
            "row_count": 3,
            "rows_with_text_count": 3 if apply_ready else 0,
            "rows_missing_text_count": 0 if apply_ready else 3,
            "rows_with_expected_label_count": 3 if apply_ready else 0,
            "expected_label_pending_count": 0 if apply_ready else 3,
            "invalid_expected_label_count": 0,
            "expected_label_counts": {"accept": 1, "reject": 1, "review": 1} if apply_ready else {},
            "ready_for_labeled_jsonl_import": apply_ready,
            "recommendation": "import_labeled_csv_then_validate_goldset"
            if apply_ready
            else "fill_remaining_expected_label_values",
        },
    )
    _write_json(
        paths["import_goldset"],
        {
            "schema_version": "prefix-local-judge-goldset-csv-import-summary-v1",
            "prompt_safe_summary": True,
            "output_written": True,
            "allow_incomplete_output": False,
            "row_count": 3,
            "rows_with_text_count": 3,
            "rows_missing_text_count": 0,
            "expected_label_counts": {"accept": 1, "reject": 1, "review": 1},
            "expected_label_pending_count": 0,
            "invalid_expected_label_count": 0,
            "ready_for_local_judge_goldset_validate": True,
            "recommendation": "run_local_judge_goldset_validate",
        },
    )
    _write_json(
        paths["goldset_validation"],
        {
            "schema_version": "prefix-local-judge-goldset-validation-summary-v1",
            "prompt_safe_summary": True,
            "row_count": 3,
            "valid_row_count": 3,
            "invalid_row_count": 0,
            "expected_label_counts": {"accept": 1, "reject": 1, "review": 1},
            "ready": True,
            "ready_for_local_judge_quality_eval": True,
            "recommendation": "run_local_judge_quality_eval",
        },
    )
    _write_json(
        paths["quality_eval"],
        {
            "schema_version": "prefix-local-judge-quality-eval-summary-v1",
            "prompt_safe_summary": True,
            "sample_count": 3,
            "attempted_count": 3,
            "evaluated_count": 3,
            "model_called_count": 3,
            "parse_error_count": 0,
            "low_confidence_count": 0,
            "correct_count": 3 if quality_ready else 1,
            "accuracy": 1.0 if quality_ready else 1 / 3,
            "macro_f1": 1.0 if quality_ready else 0.2,
            "expected_label_counts": {"accept": 1, "reject": 1, "review": 1},
            "ready": quality_ready,
            "semantic_quality_metrics_available": True,
            "recommendation": "semantic_quality_claim_allowed_for_gold_set_only"
            if quality_ready
            else "improve_or_recalibrate_local_judge_accuracy",
        },
    )
    return paths


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
