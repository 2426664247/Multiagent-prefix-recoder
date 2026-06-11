from __future__ import annotations

import json

import autogen_prefix_tree.local_judge_quality_eval as quality_eval
from autogen_prefix_tree.local_judge_quality_eval import main, run_local_judge_quality_eval


def test_local_judge_quality_eval_passes_gold_set_without_leaking_text(tmp_path, monkeypatch) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(
        gold,
        [
            ("accept-1", "accept", "Stable shared verification rule."),
            ("review-1", "review", "Agent identity instruction that needs review."),
            ("reject-1", "reject", "Turn-specific user request."),
        ],
    )

    def fake_call(config, row):
        if row["candidate_id"].startswith("accept"):
            return {"label": "accept", "reason": "stable", "confidence": 0.91}
        if row["candidate_id"].startswith("review"):
            return {"label": "review", "reason": "boundary", "confidence": 0.88}
        return {"label": "reject", "reason": "not reusable", "confidence": 0.9}

    monkeypatch.setattr(quality_eval, "_call_openai_compatible_judge", fake_call)

    result = run_local_judge_quality_eval(
        input_path=gold,
        base_url="http://127.0.0.1:11434/v1",
        model="local-quality-model",
        api_key="secret-quality-key",
        min_samples=3,
        min_accuracy=0.9,
        min_macro_f1=0.9,
        output_path=tmp_path / "predictions.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    predictions = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8")
    assert summary["schema_version"] == "prefix-local-judge-quality-eval-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["ready"] is True
    assert summary["semantic_quality_metrics_available"] is True
    assert summary["sample_count"] == 3
    assert summary["evaluated_count"] == 3
    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["expected_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert summary["predicted_label_counts"] == {"accept": 1, "reject": 1, "review": 1}
    assert summary["semantic_quality_claim_allowed"] is True
    assert summary["quality_diagnostics"]["failure_mode_codes"] == []
    assert summary["quality_diagnostics"]["predicted_label_collapse"] is False
    assert summary["quality_diagnostics"]["raw_model_label_collapse"] is False
    assert summary["confusion_matrix"]["accept"]["accept"] == 1
    assert summary["recommendation"] == "semantic_quality_claim_allowed_for_gold_set_only"
    assert summary["gold_text_written"] is False
    assert "secret-quality-key" not in serialized
    assert "Stable shared verification rule" not in serialized
    assert "Agent identity instruction" not in serialized
    assert "Turn-specific user request" not in serialized
    assert "Stable shared verification rule" not in predictions
    assert "text_hash" in predictions

    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written == summary


def test_local_judge_quality_eval_blocks_low_accuracy_claim(tmp_path, monkeypatch) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(
        gold,
        [
            ("accept-1", "accept", "Stable shared verification rule."),
            ("review-1", "review", "Agent identity instruction that needs review."),
            ("reject-1", "reject", "Turn-specific user request."),
        ],
    )

    def fake_call(config, row):
        return {"label": "accept", "reason": "over accepts", "confidence": 0.91}

    monkeypatch.setattr(quality_eval, "_call_openai_compatible_judge", fake_call)

    result = run_local_judge_quality_eval(
        input_path=gold,
        base_url="http://127.0.0.1:11434/v1",
        model="local-quality-model",
        min_samples=3,
        min_accuracy=0.9,
        min_macro_f1=0.9,
    )

    summary = result.summary
    assert summary["ready"] is False
    assert summary["accuracy"] == 1 / 3
    assert summary["macro_f1"] < 0.9
    assert summary["semantic_quality_claim_allowed"] is False
    assert summary["failure_mode_codes"] == [
        "accuracy_below_min",
        "macro_f1_below_min",
        "raw_model_accuracy_below_min",
        "raw_model_macro_f1_below_min",
        "predicted_label_collapse",
        "raw_model_label_collapse",
        "missing_reject_predictions",
        "missing_review_predictions",
        "raw_model_missing_reject_predictions",
        "raw_model_missing_review_predictions",
        "reject_zero_recall",
        "review_zero_recall",
        "raw_model_reject_zero_recall",
        "raw_model_review_zero_recall",
    ]
    diagnostics = summary["quality_diagnostics"]
    assert diagnostics["predicted_label_collapse"] is True
    assert diagnostics["raw_model_label_collapse"] is True
    assert diagnostics["missing_predicted_labels"] == ["reject", "review"]
    assert diagnostics["missing_raw_model_labels"] == ["reject", "review"]
    assert diagnostics["reject_recall"] == 0.0
    assert diagnostics["raw_model_reject_recall"] == 0.0
    assert diagnostics["accept_false_positive_count"] == 2
    assert diagnostics["raw_model_accept_false_positive_count"] == 2
    mismatch = summary["mismatch_diagnostics"]
    assert mismatch["prompt_safe_summary"] is True
    assert mismatch["mismatch_count"] == 2
    assert mismatch["mismatch_label_transition_counts"] == {
        "reject->accept": 1,
        "review->accept": 1,
    }
    assert mismatch["mismatch_semantic_hint_counts"] == {"verification_policy": 2}
    assert mismatch["mismatch_risk_tag_count_counts"] == {"0": 2}
    assert summary["recommendation"] == "improve_or_recalibrate_local_judge_accuracy"


def test_local_judge_quality_eval_cli_exit_codes(tmp_path, monkeypatch) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(
        gold,
        [
            ("accept-1", "accept", "Stable shared verification rule."),
            ("review-1", "review", "Agent identity instruction that needs review."),
            ("reject-1", "reject", "Turn-specific user request."),
        ],
    )

    def fake_call(config, row):
        return {"label": row["candidate_id"].split("-", 1)[0], "reason": "matches", "confidence": 0.9}

    monkeypatch.setattr(quality_eval, "_call_openai_compatible_judge", fake_call)

    passed = main(
        [
            "--input",
            str(gold),
            "--base-url",
            "http://127.0.0.1:11434/v1",
            "--model",
            "local-quality-model",
            "--min-samples",
            "3",
            "--min-accuracy",
            "0.9",
            "--min-macro-f1",
            "0.9",
            "--summary",
            str(tmp_path / "cli-summary.json"),
        ]
    )
    failed = main(
        [
            "--input",
            str(gold),
            "--base-url",
            "http://127.0.0.1:11434/v1",
            "--model",
            "local-quality-model",
            "--min-samples",
            "4",
            "--summary",
            str(tmp_path / "cli-failed-summary.json"),
        ]
    )

    assert passed == 0
    assert failed == 1
    assert json.loads((tmp_path / "cli-summary.json").read_text(encoding="utf-8"))["ready"] is True
    assert json.loads((tmp_path / "cli-failed-summary.json").read_text(encoding="utf-8"))["ready"] is False


def _write_gold(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": example_id,
                    "expected_label": label,
                    "semantic_hint": "verification_policy",
                    "risk_tags": [],
                    "text": text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for example_id, label, text in rows
        ),
        encoding="utf-8",
    )
