from __future__ import annotations

import json

from autogen_prefix_tree.offline_compare import compare_offline_pipeline_summaries, main


def test_offline_compare_summarizes_rule_gap_without_prompt_text(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"
    baseline_path.write_text(json.dumps(_pipeline_summary(applied=0, reusable=0, gap=True, gain=0)), encoding="utf-8")
    candidate_path.write_text(json.dumps(_pipeline_summary(applied=1, reusable=1, gap=False, gain=92)), encoding="utf-8")

    result = compare_offline_pipeline_summaries(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        baseline_label="rule-only",
        candidate_label="nl-segmentation",
        summary_path=summary_path,
        report_path=report_path,
    )

    assert result.summary["baseline"]["label"] == "rule-only"
    assert result.summary["candidate"]["label"] == "nl-segmentation"
    assert result.summary["delta"]["applied_count_delta"] == 1
    assert result.summary["delta"]["reusable_prefix_request_count_delta"] == 1
    assert result.summary["delta"]["total_estimated_gain_chars_delta"] == 92
    assert result.summary["delta"]["rule_gap_resolved"] is True
    assert result.summary["delta"]["model_to_final_label_counts_delta"]["accept->review"] == 1
    assert result.summary["delta"]["static_safety_clamp_count_delta"] == 1
    assert result.summary["baseline"]["judge_mode"] == "rule"
    assert result.summary["candidate"]["local_judge_scope"] == "review"
    assert result.summary["candidate"]["max_local_judge_calls"] == 1
    assert result.summary["candidate"]["local_judge_budget_strategy"] == "priority_review_risk_confidence_chars_v1"
    assert result.summary["candidate"]["local_judge_budget_diagnostics"]["budget_exhausted"] is True
    assert result.summary["candidate"]["local_judge_effectiveness_diagnostics"]["resolution_rate"] == 0.0
    assert result.summary["candidate"]["candidate_promotion_policy"]["automatic_promotion_allowed_labels"] == ["accept"]
    assert result.summary["candidate"]["candidate_promotion_policy"]["automatic_promotion_includes_review"] is False
    assert result.summary["candidate"]["candidate_review_upper_bound_policy"][
        "review_candidates_used_for_upper_bound_only"
    ] is True
    assert result.summary["delta"]["local_judge_effectiveness_diagnostics_delta"]["rule_review_candidate_count_delta"] == 1
    assert result.summary["candidate"]["local_judge_action_counts"] == {
        "model_called": 1,
        "skipped_by_budget": 1,
        "skipped_by_scope": 2,
    }
    assert result.summary["delta"]["local_judge_action_counts_delta"] == {
        "model_called": 1,
        "skipped_by_budget": 1,
        "skipped_by_scope": 2,
    }
    assert result.summary["delta"]["local_judge_skipped_by_budget_label_counts_delta"] == {"review": 1}
    assert result.summary["candidate"]["model_to_final_label_counts"]["accept->review"] == 1
    assert result.summary["candidate"]["static_safety_clamp_count"] == 1
    gate = result.summary["experiment_gate"]
    assert gate["recommendation"] == "run_real_ab_smoke_with_review_candidates_disabled"
    assert gate["performance_conclusion_allowed"] is False
    assert _gate_item(gate, "same_supported_corpus")["status"] == "pass"
    assert _gate_item(gate, "candidate_increases_offline_rewrites")["status"] == "pass"
    assert _gate_item(gate, "rule_gap_resolution")["status"] == "pass"
    assert _gate_item(gate, "review_queue")["status"] == "warn"
    assert _gate_item(gate, "review_candidate_promotion_policy")["status"] == "pass"
    assert _gate_item(gate, "local_judge_safety_clamps")["status"] == "warn"
    assert _gate_item(gate, "local_judge_budget_coverage")["status"] == "warn"
    assert _gate_item(gate, "local_judge_review_resolution")["status"] == "warn"
    assert _gate_item(gate, "real_provider_metrics")["status"] == "unknown"
    assert result.summary["real_provider_metrics_available"] is False
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "prefix-offline-pipeline-comparison-v1"
    assert written["experiment_gate"]["schema_version"] == "prefix-offline-experiment-gate-v1"
    report = report_path.read_text(encoding="utf-8")
    assert "Offline Prefix Rule Comparison" in report
    assert "Experiment Gate" in report
    assert "Candidate Label Diagnostics" in report
    assert "local_judge_scope" in report
    assert "max_local_judge_calls" in report
    assert "local_judge_budget_strategy" in report
    assert "local_judge_budget_diagnostics" in report
    assert "local_judge_effectiveness_diagnostics" in report
    assert "candidate_promotion_policy" in report
    assert "review_candidate_promotion_policy" in report
    assert "local_judge_action_counts" in report
    assert "model_to_final_label_counts" in report
    assert "accept->review" in report
    assert "run_real_ab_smoke_with_review_candidates_disabled" in report
    assert "Real cached tokens" in report
    assert "secret prompt text" not in json.dumps(result.summary, ensure_ascii=False)
    assert "secret prompt text" not in report


def test_offline_compare_blocks_when_review_candidates_would_be_promoted(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_pipeline_summary(applied=0, reusable=0, gap=True, gain=0)), encoding="utf-8")
    candidate = _pipeline_summary(applied=1, reusable=1, gap=False, gain=92)
    candidate["semantic_rule_gap_summary"]["candidate_promotion_policy"] = {
        "schema_version": "prefix-candidate-promotion-policy-v1",
        "automatic_promotion_allowed_labels": ["accept", "review"],
        "review_candidate_policy": "unsafe_auto_promote_review",
        "review_candidates_used_for_upper_bound_only": False,
        "automatic_promotion_includes_review": True,
    }
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = compare_offline_pipeline_summaries(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )

    gate = result.summary["experiment_gate"]
    assert _gate_item(gate, "review_candidate_promotion_policy")["status"] == "fail"
    assert gate["recommendation"] == "fix_review_candidate_promotion_policy_before_ab"


def test_offline_compare_cli_writes_outputs(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    summary_path = tmp_path / "summary.json"
    baseline_path.write_text(json.dumps(_pipeline_summary(applied=0, reusable=0, gap=True, gain=0)), encoding="utf-8")
    candidate_path.write_text(json.dumps(_pipeline_summary(applied=1, reusable=1, gap=False, gain=92)), encoding="utf-8")

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--baseline-label",
            "rule-only",
            "--candidate-label",
            "nl-segmentation",
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["delta"]["rule_gap_resolved"] is True
    assert written["experiment_gate"]["recommendation"] == "run_real_ab_smoke_with_review_candidates_disabled"


def _pipeline_summary(*, applied: int, reusable: int, gap: bool, gain: int) -> dict:
    return {
        "schema_version": "prefix-offline-semantic-pipeline-summary-v1",
        "natural_language_segmentation_enabled": applied > 0,
        "source_prompt_corpus": {
            "prompt_count": 2,
            "source_file_count": 1,
        },
        "dataset_eval": {
            "supported_request_count": 2,
            "applied_count": applied,
            "reusable_prefix_request_count": reusable,
            "total_estimated_gain_chars": gain,
            "prompt_sample": "secret prompt text",
        },
        "candidate_utility_eval": {
            "eligible_candidate_chars": 100,
            "promotion_policy": {
                "schema_version": "prefix-candidate-promotion-policy-v1",
                "automatic_promotion_allowed_labels": ["accept"],
                "review_candidate_policy": "excluded_from_automatic_promotion",
                "review_candidates_used_for_upper_bound_only": False,
                "automatic_promotion_includes_review": False,
            },
        },
        "candidate_utility_eval_with_review": {
            "eligible_candidate_chars": 150,
            "promotion_policy": {
                "schema_version": "prefix-candidate-promotion-policy-v1",
                "automatic_promotion_allowed_labels": ["accept"],
                "review_candidate_policy": "upper_bound_only_require_local_judge_or_manual_review",
                "review_candidates_used_for_upper_bound_only": True,
                "automatic_promotion_includes_review": False,
            },
        },
        "candidate_label_eval": {
            "judge_mode": "openai-compatible" if applied else "rule",
            "local_judge_scope": "review" if applied else "all",
            "max_local_judge_calls": 1 if applied else None,
            "local_judge_budget_strategy": "priority_review_risk_confidence_chars_v1" if applied else None,
            "local_judge_budget_diagnostics": {
                "schema_version": "prefix-local-judge-budget-diagnostics-v1",
                "budget_limited": True,
                "max_local_judge_calls": 1,
                "budget_strategy": "priority_review_risk_confidence_chars_v1",
                "eligible_for_budget_count": 2,
                "model_called_count": 1,
                "skipped_by_budget_count": 1,
                "budget_exhausted": True,
                "budget_coverage_rate": 0.5,
            }
            if applied
            else {},
            "local_judge_effectiveness_diagnostics": {
                "schema_version": "prefix-local-judge-effectiveness-diagnostics-v1",
                "rule_review_candidate_count": 1,
                "model_called_rule_review_count": 1,
                "resolved_rule_review_count": 0,
                "called_resolved_rule_review_count": 0,
                "remaining_rule_review_count": 1,
                "resolution_rate": 0.0,
                "called_resolution_rate": 0.0,
                "rule_review_final_label_counts": {"review": 1},
                "rule_review_model_label_counts": {"accept": 1},
                "rule_review_local_judge_action_counts": {
                    "model_called": 1,
                    "skipped_by_budget": 1,
                },
            }
            if applied
            else {},
            "label_counts": {"accept": 2, "review": 1},
            "local_judge_action_counts": {"model_called": 1, "skipped_by_budget": 1, "skipped_by_scope": 2}
            if applied
            else {},
            "local_judge_skipped_by_scope_label_counts": {"accept": 2} if applied else {},
            "local_judge_skipped_by_budget_label_counts": {"review": 1} if applied else {},
            "rule_label_counts": {"accept": 2, "review": 1} if applied else {},
            "model_label_counts": {"accept": 3} if applied else {},
            "rule_to_model_label_counts": {"accept->accept": 2, "review->accept": 1} if applied else {},
            "model_to_final_label_counts": {"accept->accept": 2, "accept->review": 1} if applied else {},
            "rule_to_final_label_counts": {"accept->accept": 2, "review->review": 1} if applied else {},
            "static_safety_clamp_count": 1 if applied else 0,
        },
        "semantic_rule_gap_summary": {
            "supported_request_count": 2,
            "applied_count": applied,
            "applied_rate": applied / 2,
            "reusable_prefix_request_count": reusable,
            "candidate_count": 3,
            "accepted_candidate_count": 2,
            "review_candidate_count": 1,
            "accepted_candidate_parent_block_count": 2,
            "accepted_or_review_candidate_parent_block_count": 3,
            "accepted_estimated_reusable_chars": 10,
            "accepted_or_review_estimated_reusable_chars": 20,
            "accepted_repeated_group_count": 1,
            "candidate_promotion_policy": {
                "schema_version": "prefix-candidate-promotion-policy-v1",
                "automatic_promotion_allowed_labels": ["accept"],
                "review_candidate_policy": "excluded_from_automatic_promotion",
                "review_candidates_used_for_upper_bound_only": False,
                "automatic_promotion_includes_review": False,
            },
            "candidate_review_upper_bound_policy": {
                "schema_version": "prefix-candidate-promotion-policy-v1",
                "automatic_promotion_allowed_labels": ["accept"],
                "review_candidate_policy": "upper_bound_only_require_local_judge_or_manual_review",
                "review_candidates_used_for_upper_bound_only": True,
                "automatic_promotion_includes_review": False,
            },
            "rule_gap_observed": gap,
            "review_queue_observed": True,
        },
    }


def _gate_item(gate: dict, name: str) -> dict:
    return next(item for item in gate["items"] if item["name"] == name)
