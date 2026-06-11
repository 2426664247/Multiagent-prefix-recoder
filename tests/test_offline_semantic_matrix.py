from __future__ import annotations

import json

import autogen_prefix_tree.candidate_label_eval as label_eval
from autogen_prefix_tree.offline_semantic_matrix import main, run_offline_semantic_matrix


PROMPT = '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.

When using code, indicate the script type in the code block.

Verify the answer carefully and include evidence.

Reply with concise final results.
"""
'''


def test_offline_semantic_matrix_runs_multiple_sources_without_prompt_text(tmp_path) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "agent_a.py").write_text(PROMPT, encoding="utf-8")
    (source_a / "agent_b.py").write_text(PROMPT, encoding="utf-8")
    (source_b / "agent.py").write_text(PROMPT, encoding="utf-8")

    result = run_offline_semantic_matrix(
        sources=[("framework-a", source_a), ("framework-b", source_b), ("missing", tmp_path / "missing")],
        output_dir=tmp_path / "matrix",
        session_id="matrix-test",
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["schema_version"] == "prefix-offline-semantic-matrix-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["real_provider_metrics_available"] is False
    assert summary["source_count"] == 3
    assert summary["aggregate"]["completed_source_count"] == 2
    assert summary["aggregate"]["failed_source_count"] == 1
    assert summary["aggregate"]["supported_source_count"] == 2
    assert summary["aggregate"]["total_prompt_count"] == 3
    assert summary["aggregate"]["extraction_counts"] == {"prompt_named_constant": 3}
    assert summary["aggregate"]["total_nl_segmentation_applied_count"] == 0
    assert summary["aggregate"]["total_estimated_gain_chars_delta"] > 0
    assert summary["aggregate"]["rule_gap_resolved_source_count"] >= 1
    assert summary["aggregate"]["review_queue_diagnostics"]["schema_version"] == (
        "prefix-matrix-review-queue-diagnostics-v1"
    )
    budget = summary["aggregate"]["local_judge_budget_diagnostics"]
    assert budget["schema_version"] == "prefix-matrix-local-judge-budget-diagnostics-v1"
    assert budget["source_with_budget_diagnostics_count"] == 0
    assert budget["aggregate_budget_coverage_rate"] is None
    assert summary["aggregate"]["review_queue_diagnostics"]["local_judge_priority_counts"]
    assert summary["aggregate"]["recommendation"] in {
        "run_small_real_ab_smoke_keep_review_candidates_disabled",
        "run_small_real_ab_smoke",
    }
    assert any(row["status"] == "fail" and row["error"] == "source_root_missing" for row in summary["sources"])
    assert summary["prompt_text_artifacts"]
    assert "helpful AI assistant" not in serialized
    assert "code block" not in serialized
    report = open(result.report_path, encoding="utf-8").read()
    assert "Offline Semantic Matrix" in report
    assert "Prompt Extraction Diagnostics" in report
    assert "prompt_named_constant" in report
    assert "Review Queue" in report
    assert "Local Judge Budget" in report


def test_offline_semantic_matrix_aggregates_local_judge_budget_diagnostics(tmp_path, monkeypatch) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "agent.py").write_text(PROMPT, encoding="utf-8")
    (source_b / "agent.py").write_text(PROMPT, encoding="utf-8")
    seen_candidate_ids: list[str] = []

    def fake_call(config, candidate_row):
        seen_candidate_ids.append(candidate_row["candidate_id"])
        return {"label": "accept", "reason": "stable shared instruction", "confidence": 0.93}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    result = run_offline_semantic_matrix(
        sources=[("framework-a", source_a), ("framework-b", source_b)],
        output_dir=tmp_path / "matrix",
        session_id="matrix-budget-test",
        include_candidate_text=True,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
        max_local_judge_calls=1,
    )

    aggregate = result.summary["aggregate"]
    budget = aggregate["local_judge_budget_diagnostics"]
    assert budget["schema_version"] == "prefix-matrix-local-judge-budget-diagnostics-v1"
    assert budget["source_with_budget_diagnostics_count"] == 2
    assert budget["budget_limited_source_count"] == 2
    assert budget["budget_exhausted_source_count"] == 2
    assert budget["budget_limited"] is True
    assert budget["budget_exhausted"] is True
    assert budget["total_eligible_for_budget_count"] == 6
    assert budget["total_model_called_count"] == 2
    assert budget["total_skipped_by_budget_count"] == 4
    assert budget["aggregate_budget_coverage_rate"] == 2 / 6
    assert budget["budget_strategy_counts"] == {"priority_review_risk_confidence_chars_v1": 2}
    assert budget["max_local_judge_calls_counts"] == {"1": 2}
    assert [row["label"] for row in budget["budget_exhausted_sources"]] == ["framework-a", "framework-b"]
    assert aggregate["local_judge_action_counts"] == {"model_called": 2, "skipped_by_budget": 4}
    assert aggregate["local_judge_skipped_by_budget_label_counts"] == {"accept": 4}
    assert aggregate["rule_label_counts"] == {"accept": 6}
    assert aggregate["model_label_counts"] == {"accept": 2}
    assert aggregate["rule_to_model_label_counts"] == {"accept->accept": 2}
    assert aggregate["model_to_final_label_counts"] == {"accept->accept": 2}
    assert aggregate["rule_to_final_label_counts"] == {"accept->accept": 6}
    assert aggregate["static_safety_clamp_count"] == 0
    assert aggregate["local_judge_effectiveness_diagnostics"]["source_with_effectiveness_diagnostics_count"] == 2
    assert aggregate["local_judge_effectiveness_diagnostics"]["rule_review_candidate_count"] == 0
    assert aggregate["local_judge_effectiveness_diagnostics"]["resolution_rate"] is None
    assert all(row["local_judge_budget_diagnostics"]["budget_exhausted"] for row in result.summary["sources"])
    assert len(seen_candidate_ids) == 4
    report = open(result.report_path, encoding="utf-8").read()
    assert "Local Judge Budget" in report
    assert "Candidate Label Diagnostics" in report
    assert "rule_to_model_label_counts" in report
    assert "aggregate_budget_coverage_rate" in report


def test_offline_semantic_matrix_aggregates_label_transition_clamps(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "agent.py").write_text(PROMPT, encoding="utf-8")

    def fake_call(config, candidate_row):
        return {"label": "accept", "reason": "model accepted everything", "confidence": 0.51}

    monkeypatch.setattr(label_eval, "_call_openai_compatible_judge", fake_call)

    result = run_offline_semantic_matrix(
        sources=[("framework", source_root)],
        output_dir=tmp_path / "matrix-clamps",
        session_id="matrix-clamp-test",
        include_candidate_text=True,
        judge="openai-compatible",
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-test-model",
        max_local_judge_calls=3,
    )

    aggregate = result.summary["aggregate"]
    assert aggregate["model_label_counts"] == {"accept": 3}
    assert aggregate["rule_to_model_label_counts"] == {"accept->accept": 3}
    assert aggregate["model_to_final_label_counts"] == {"accept->review": 3}
    assert aggregate["rule_to_final_label_counts"] == {"accept->review": 3}
    assert aggregate["static_safety_clamp_count"] == 3
    assert "local_judge_effectiveness_diagnostics" in aggregate
    source = result.summary["sources"][0]
    assert source["model_to_final_label_counts"] == {"accept->review": 3}
    report = open(result.report_path, encoding="utf-8").read()
    assert "static_safety_clamp_count" in report
    assert "accept->review" in report


def test_offline_semantic_matrix_cli_writes_summary(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "agent.py").write_text(PROMPT, encoding="utf-8")
    output_dir = tmp_path / "matrix-cli"

    exit_code = main(
        [
            "--source",
            f"framework={source_root}",
            "--output-dir",
            str(output_dir),
            "--session-id",
            "matrix-cli",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "offline_semantic_matrix_summary.json").read_text(encoding="utf-8"))
    assert summary["session_id"] == "matrix-cli"
    assert summary["sources"][0]["label"] == "framework"
    assert summary["sources"][0]["status"] == "pass"
    assert (output_dir / "reports" / "offline_semantic_matrix_report.md").exists()
