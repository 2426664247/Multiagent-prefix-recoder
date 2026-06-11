from __future__ import annotations

import json

from autogen_prefix_tree.offline_local_judge_matrix_smoke import (
    main,
    run_offline_local_judge_matrix_smoke,
)


PROMPT = '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.

When using code, indicate the script type in the code block.

Select the next step in the plan before responding.

Plan the next step and report progress clearly.

Reply with concise final results.
"""
'''


def test_offline_local_judge_matrix_smoke_runs_fake_judge_without_prompt_text(tmp_path) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "agent.py").write_text(PROMPT, encoding="utf-8")
    (source_b / "agent.py").write_text(PROMPT, encoding="utf-8")

    result = run_offline_local_judge_matrix_smoke(
        sources=[("framework-a", source_a), ("framework-b", source_b)],
        output_dir=tmp_path / "smoke",
        session_id="fake-local-judge-matrix-smoke",
        local_judge_scope="review",
        max_local_judge_calls=1,
        fake_judge_mode="accept",
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    budget = summary["matrix_digest"]["local_judge_budget_diagnostics"]
    candidate_diagnostics = summary["matrix_digest"]["candidate_label_diagnostics"]
    assert summary["schema_version"] == "prefix-offline-local-judge-matrix-smoke-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["fake_local_judge"] is True
    assert summary["fake_local_judge_mode"] == "accept"
    assert summary["fake_local_judge_request_count"] == 4
    assert summary["fake_local_judge_model_label_counts"] == {"accept": 4}
    assert summary["fake_local_judge_semantic_hint_counts"] == {"procedure_policy": 4}
    assert summary["matrix_digest"]["completed_source_count"] == 2
    assert summary["matrix_digest"]["prompt_extraction_diagnostics"] == {
        "extraction_counts": {"prompt_named_constant": 2}
    }
    assert summary["matrix_digest"]["local_judge_action_counts"] == {
        "model_called": 2,
        "skipped_by_scope": 4,
        "skipped_by_budget": 2,
    }
    assert candidate_diagnostics == {
        "rule_label_counts": {"accept": 4, "review": 4},
        "model_label_counts": {"accept": 2},
        "rule_to_model_label_counts": {"review->accept": 2},
        "model_to_final_label_counts": {"accept->accept": 2},
        "rule_to_final_label_counts": {
            "accept->accept": 4,
            "review->accept": 2,
            "review->review": 2,
        },
        "static_safety_clamp_count": 0,
        "local_judge_effectiveness_diagnostics": {
            "schema_version": "prefix-matrix-local-judge-effectiveness-diagnostics-v1",
            "source_with_effectiveness_diagnostics_count": 2,
            "rule_review_candidate_count": 4,
            "model_called_rule_review_count": 2,
            "resolved_rule_review_count": 2,
            "called_resolved_rule_review_count": 2,
            "remaining_rule_review_count": 2,
            "resolution_rate": 0.5,
            "called_resolution_rate": 1.0,
            "rule_review_final_label_counts": {"accept": 2, "review": 2},
            "rule_review_model_label_counts": {"accept": 2},
            "rule_review_local_judge_action_counts": {
                "model_called": 2,
                "skipped_by_budget": 2,
            },
        },
    }
    assert budget["schema_version"] == "prefix-matrix-local-judge-budget-diagnostics-v1"
    assert budget["budget_limited_source_count"] == 2
    assert budget["budget_exhausted_source_count"] == 2
    assert budget["total_eligible_for_budget_count"] == 4
    assert budget["total_model_called_count"] == 2
    assert budget["total_skipped_by_budget_count"] == 2
    assert budget["aggregate_budget_coverage_rate"] == 0.5
    assert summary["real_provider_metrics_available"] is False
    assert "fake local OpenAI-compatible judge" in summary["real_provider_metrics_note"]
    assert "helpful AI assistant" not in serialized
    assert "code block" not in serialized
    assert "Plan the next step" not in serialized
    assert "text" not in summary["fake_local_judge_semantic_hint_counts"]

    written = json.loads(open(result.summary_path, encoding="utf-8").read())
    assert written["matrix_summary_path"] == result.matrix_summary_path
    matrix_summary = json.loads(open(result.matrix_summary_path, encoding="utf-8").read())
    assert matrix_summary["include_candidate_text"] is True
    report = open(result.matrix_report_path, encoding="utf-8").read()
    assert "Candidate Label Diagnostics" in report
    assert "local_judge_effectiveness_diagnostics" in report
    assert "Local Judge Budget" in report


def test_offline_local_judge_matrix_smoke_cli_writes_summary(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "agent.py").write_text(PROMPT, encoding="utf-8")
    output_dir = tmp_path / "smoke-cli"

    exit_code = main(
        [
            "--source",
            f"framework={source_root}",
            "--output-dir",
            str(output_dir),
            "--session-id",
            "fake-local-judge-matrix-cli",
            "--max-local-judge-calls",
            "1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (output_dir / "offline_local_judge_matrix_smoke_summary.json").read_text(encoding="utf-8")
    )
    assert summary["session_id"] == "fake-local-judge-matrix-cli"
    assert summary["fake_local_judge"] is True
    assert summary["matrix_digest"]["supported_source_count"] == 1
