from __future__ import annotations

import json

from autogen_prefix_tree.offline_semantic_suite import main, run_offline_semantic_suite


def test_offline_semantic_suite_runs_rule_and_nl_pipelines_without_prompt_text_in_summary(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    prompt_module = '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.

When using code, indicate the script type in the code block.

Verify the answer carefully and include evidence.

Reply with concise final results.
"""
'''
    (source_root / "agent_a.py").write_text(prompt_module, encoding="utf-8")
    (source_root / "agent_b.py").write_text(prompt_module, encoding="utf-8")

    result = run_offline_semantic_suite(
        source_root=source_root,
        output_dir=tmp_path / "suite",
        session_id="offline-suite-test",
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["schema_version"] == "prefix-offline-semantic-suite-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["real_provider_metrics_available"] is False
    assert summary["rule_only"]["prompt_count"] == 2
    assert summary["nl_segmentation"]["prompt_count"] == 2
    assert summary["nl_segmentation"]["extraction_counts"] == {"prompt_named_constant": 2}
    assert summary["nl_segmentation"]["review_queue_diagnostics"]["schema_version"] == (
        "prefix-review-queue-diagnostics-v1"
    )
    assert summary["rule_only"]["applied_count"] == 0
    assert summary["rule_only"]["rule_gap_observed"] is True
    assert summary["nl_segmentation"]["rule_label_counts"]
    assert summary["nl_segmentation"]["rule_to_final_label_counts"]
    assert summary["nl_segmentation"]["static_safety_clamp_count"] == 0
    assert summary["nl_segmentation"]["candidate_promotion_policy"]["automatic_promotion_allowed_labels"] == ["accept"]
    assert summary["nl_segmentation"]["candidate_promotion_policy"]["automatic_promotion_includes_review"] is False
    assert summary["nl_segmentation"]["candidate_review_upper_bound_policy"][
        "review_candidates_used_for_upper_bound_only"
    ] is True
    assert "local_judge_effectiveness_diagnostics" in summary["nl_segmentation"]
    assert summary["nl_segmentation"]["applied_count"] == 0
    assert summary["nl_segmentation"]["reusable_prefix_request_count"] >= 1
    assert summary["comparison"]["delta"]["applied_count_delta"] == 0
    assert summary["comparison"]["delta"]["reusable_prefix_request_count_delta"] >= 1
    assert summary["comparison"]["delta"]["total_estimated_gain_chars_delta"] > 0
    assert summary["comparison"]["delta"]["rule_gap_resolved"] is True
    assert summary["comparison"]["experiment_gate"]["performance_conclusion_allowed"] is False
    assert summary["recommendation"] == "run_real_ab_smoke"
    assert summary["prompt_text_artifacts"]
    assert "helpful AI assistant" not in serialized
    assert "code block" not in serialized

    artifacts = summary["artifacts"]
    assert json.loads(open(artifacts["rule_only_pipeline_summary"], encoding="utf-8").read())[
        "natural_language_segmentation_enabled"
    ] is False
    assert json.loads(open(artifacts["nl_segmentation_pipeline_summary"], encoding="utf-8").read())[
        "natural_language_segmentation_enabled"
    ] is True
    assert "Offline Prefix Rule Comparison" in open(artifacts["comparison_report_md"], encoding="utf-8").read()


def test_offline_semantic_suite_cli_writes_summary(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "agent.py").write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.

When using code, indicate the script type in the code block.

Verify the answer carefully and include evidence.

Reply with concise final results.
"""
''',
        encoding="utf-8",
    )
    output_dir = tmp_path / "suite-cli"

    exit_code = main(
        [
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_dir),
            "--session-id",
            "offline-suite-cli",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "offline_semantic_suite_summary.json").read_text(encoding="utf-8"))
    assert summary["session_id"] == "offline-suite-cli"
    assert summary["artifacts"]["comparison_summary"].endswith("offline_rule_vs_nl_summary.json")
    assert summary["real_provider_metrics_available"] is False


def test_offline_semantic_suite_marks_review_text_worklist_as_prompt_text_artifact(tmp_path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "agent.py").write_text(
        '''
SYSTEM_MESSAGE = """
You are a helpful AI assistant.
When using code, indicate the script type in the code block.
"""
''',
        encoding="utf-8",
    )

    result = run_offline_semantic_suite(
        source_root=source_root,
        output_dir=tmp_path / "suite",
        session_id="offline-suite-text-worklist",
        include_candidate_text=True,
    )

    artifact_paths = [artifact["path"] for artifact in result.summary["prompt_text_artifacts"]]
    assert any(path.endswith("review_worklist_with_text.jsonl") for path in artifact_paths)
