from __future__ import annotations

import json

from autogen_prefix_tree.local_judge_policy_eval import main, run_local_judge_policy_eval


def test_local_judge_policy_eval_compares_static_model_and_hybrid_policies(tmp_path) -> None:
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_gold(
        gold,
        [
            ("row-0", "accept", "accept"),
            ("row-1", "reject", "reject"),
            ("row-2", "review", "review"),
        ],
    )
    _write_predictions(
        predictions,
        [
            ("accept", "accept"),
            ("accept", "review"),
            ("accept", "review"),
        ],
    )

    result = run_local_judge_policy_eval(
        input_path=gold,
        predictions_path=predictions,
        min_accuracy=0.9,
        min_macro_f1=0.9,
        summary_path=tmp_path / "summary.json",
        report_path=tmp_path / "report.md",
    )

    summary = result.summary
    by_policy = {item["policy_name"]: item for item in summary["policy_summaries"]}
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["schema_version"] == "prefix-local-judge-policy-eval-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["row_count"] == 3
    assert summary["joined_prediction_count"] == 3
    assert summary["ready"] is True
    assert summary["best_ready_policy"] == "source_label"
    assert by_policy["source_label"]["ready"] is True
    assert by_policy["rule_label"]["ready"] is True
    assert by_policy["raw_model_label"]["ready"] is False
    assert by_policy["raw_model_label"]["predicted_label_counts"] == {"accept": 3}
    assert "missing_reject_predictions" in by_policy["raw_model_label"]["failure_mode_codes"]
    assert by_policy["clamped_model_label"]["predicted_label_counts"] == {"accept": 1, "review": 2}
    assert by_policy["fail_closed_rule_and_model_accept"]["predicted_label_counts"] == {
        "accept": 1,
        "reject": 1,
        "review": 1,
    }
    raw_diagnostics = by_policy["raw_model_label"]["mismatch_diagnostics"]
    assert raw_diagnostics["mismatch_count"] == 2
    assert raw_diagnostics["mismatch_label_transition_counts"] == {
        "reject->accept": 1,
        "review->accept": 1,
    }
    assert summary["policy_failure_diagnostics"]["raw_model_label"][
        "top_mismatch_semantic_hints"
    ] == {"verification_policy": 2}
    assert "Failure Diagnostics" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "stable shared verification rule" not in serialized
    assert "stable shared verification rule" not in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary


def test_local_judge_policy_eval_cli_exit_code_reflects_any_ready_policy(tmp_path) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(gold, [("row-0", "accept", "review")])

    failed = main(
        [
            "--input",
            str(gold),
            "--min-accuracy",
            "0.9",
            "--min-macro-f1",
            "0.9",
            "--summary",
            str(tmp_path / "failed.json"),
        ]
    )

    assert failed == 1
    assert json.loads((tmp_path / "failed.json").read_text(encoding="utf-8"))["ready"] is False


def _write_gold(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": example_id,
                    "expected_label": expected,
                    "source_label": rule_label,
                    "rule_label": rule_label,
                    "semantic_hint": "verification_policy",
                    "risk_tags": [],
                    "text": "stable shared verification rule",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for example_id, expected, rule_label in rows
        ),
        encoding="utf-8",
    )


def _write_predictions(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "input_index": index,
                    "raw_model_label": raw,
                    "predicted_label": clamped,
                    "text_hash": f"hash-{index}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for index, (raw, clamped) in enumerate(rows)
        ),
        encoding="utf-8",
    )
