from __future__ import annotations

import json

from autogen_prefix_tree.static_rule_calibration_eval import (
    main,
    run_static_rule_calibration_eval,
)


def test_static_rule_calibration_eval_reports_prompt_safe_metadata_signal(tmp_path) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(
        gold,
        [
            ("accept-a", "accept", "accept", "verification_policy", [], "tmp/framework-src/autogen/pkg/a.py"),
            ("accept-b", "accept", "accept", "verification_policy", [], "tmp/framework-src/autogen/pkg/b.py"),
            (
                "reject-a",
                "reject",
                "review",
                "verification_policy",
                ["agent_identity_boundary"],
                "tmp/framework-src/autogen/pkg/c.py",
            ),
            (
                "reject-b",
                "reject",
                "review",
                "verification_policy",
                ["agent_identity_boundary"],
                "tmp/framework-src/autogen/pkg/d.py",
            ),
            (
                "review-a",
                "review",
                "accept",
                "tool_or_code_policy",
                ["conditional_instruction"],
                "tmp/framework-src/autogen/pkg/e.py",
            ),
            (
                "review-b",
                "review",
                "accept",
                "tool_or_code_policy",
                ["conditional_instruction"],
                "tmp/framework-src/autogen/pkg/f.py",
            ),
        ],
    )

    result = run_static_rule_calibration_eval(
        input_path=gold,
        min_accuracy=0.8,
        min_macro_f1=0.8,
        production_min_support=1,
        production_min_purity=0.67,
        summary_path=tmp_path / "summary.json",
        report_path=tmp_path / "report.md",
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["schema_version"] == "prefix-static-rule-calibration-eval-summary-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["ready"] is True
    assert summary["best_production_candidate_policy"].startswith(
        "metadata_calibration:semantic_hint+risk_tag_set"
    )
    assert summary["top_policies"][0]["accuracy"] == 1.0
    assert summary["top_policies"][0]["macro_f1"] == 1.0
    assert summary["production_readiness_diagnostics"]["status"] == "production_candidate_ready"
    assert summary["gold_support_diagnostics"]["prompt_safe_summary"] is True
    assert summary["gold_support_diagnostics"]["production_ready_bucket_count"] > 0
    assert summary["validator_rule_candidates"]["prompt_safe_summary"] is True
    assert summary["validator_rule_candidates"]["candidate_rule_count"] > 0
    assert {
        candidate["validator_action"]
        for candidate in summary["validator_rule_candidates"]["candidate_rules"]
    } >= {"allow_accept", "force_reject", "prefer_review"}
    gate = summary["validator_rule_candidates"]["promotion_gate"]
    assert gate["automatic_promotion_ready"] is False
    assert "manual_code_review_required" in gate["blocking_reasons"]
    assert "real_provider_ab_required" in gate["blocking_reasons"]
    assert "allow_accept_candidates_require_stronger_evidence" in gate["blocking_reasons"]
    overlap = summary["validator_rule_candidates"]["overlap_diagnostics"]
    assert overlap["prompt_safe_summary"] is True
    assert overlap["overlap_pair_count"] > 0
    simulation = summary["validator_rule_candidates"]["fail_closed_simulation"]
    assert simulation["prompt_safe_summary"] is True
    assert simulation["candidate_rule_count"] > 0
    assert simulation["matched_count"] > 0
    assert simulation["safe_for_automatic_validator_promotion"] is False
    assert simulation["automation_policy"] == "diagnostic_only_until_manual_review_and_real_ab"
    review_plan = summary["validator_rule_candidates"]["validator_review_plan"]
    assert review_plan["schema_version"] == "prefix-static-rule-validator-review-plan-v1"
    assert review_plan["prompt_safe_summary"] is True
    assert review_plan["review_item_count"] > 0
    assert review_plan["blocking_review_item_count"] > 0
    assert review_plan["review_items"][0]["priority_rank"] == 1
    assert review_plan["automation_policy"] == "manual_review_only_before_validator_code_changes"
    subset = summary["validator_rule_candidates"]["conservative_fail_closed_subset"]
    assert subset["schema_version"] == "prefix-static-rule-conservative-fail-closed-subset-v1"
    assert subset["prompt_safe_summary"] is True
    assert subset["safe_for_automatic_validator_promotion"] is False
    assert subset["automation_policy"] == "manual_trial_only_until_code_review_and_real_ab"
    assert subset["included_rule_count"] + subset["excluded_rule_count"] == subset["source_fail_closed_rule_count"]
    assert "allow_accept_requires_stronger_evidence" not in subset["exclusion_reason_counts"]
    shadow_trial = summary["validator_rule_candidates"]["shadow_trial_plan"]
    assert shadow_trial["schema_version"] == "prefix-static-rule-shadow-trial-plan-v1"
    assert shadow_trial["prompt_safe_summary"] is True
    assert shadow_trial["trial_mode"] == "shadow_only"
    assert shadow_trial["shadow_trial_ready"] is True
    assert shadow_trial["trial_rule_count"] == subset["included_rule_count"]
    assert shadow_trial["offline_simulation_mismatch_count"] == subset["simulation"]["mismatch_count"]
    assert shadow_trial["validator_behavior_change_allowed"] is False
    assert shadow_trial["safe_for_automatic_validator_promotion"] is False
    assert shadow_trial["automation_policy"] == (
        "shadow_only_no_behavior_change_until_manual_review_and_real_ab"
    )
    assert shadow_trial["trial_rules"][0]["trial_rule_id"].startswith("shadow_rule:")
    assert "real_provider_ab_with_shadow_telemetry" in shadow_trial["required_preconditions"]
    assert "provider_cached_tokens" in shadow_trial["telemetry_fields"]
    source_generalization = summary["validator_rule_candidates"]["source_generalization_diagnostics"]
    assert source_generalization["schema_version"] == "prefix-static-rule-source-generalization-diagnostics-v1"
    assert source_generalization["prompt_safe_summary"] is True
    assert source_generalization["candidate_with_multi_source_file_count"] > 0
    assert source_generalization["generalization_claim_allowed"] is False
    assert source_generalization["automation_policy"] == (
        "diagnostic_only_until_broader_cross_framework_gold_eval"
    )
    assert "stable shared verification rule" not in serialized
    assert "stable shared verification rule" not in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Validator Rule Candidates" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Automatic promotion ready" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Validator Review Plan" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Conservative Fail-Closed Subset" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Validator Shadow Trial Plan" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Source Generalization Diagnostics" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary


def test_static_rule_calibration_eval_separates_exploratory_from_production_candidate(tmp_path) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(
        gold,
        [
            ("accept-a", "accept", "accept", "verification_policy", [], "tmp/framework-src/autogen/pkg/a.py"),
            ("accept-b", "accept", "accept", "verification_policy", [], "tmp/framework-src/autogen/pkg/b.py"),
            (
                "reject-a",
                "reject",
                "review",
                "verification_policy",
                ["agent_identity_boundary"],
                "tmp/framework-src/autogen/pkg/c.py",
            ),
            (
                "reject-b",
                "reject",
                "review",
                "verification_policy",
                ["agent_identity_boundary"],
                "tmp/framework-src/autogen/pkg/d.py",
            ),
            (
                "review-a",
                "review",
                "accept",
                "tool_or_code_policy",
                ["conditional_instruction"],
                "tmp/framework-src/autogen/pkg/e.py",
            ),
            (
                "review-b",
                "review",
                "accept",
                "tool_or_code_policy",
                ["conditional_instruction"],
                "tmp/framework-src/autogen/pkg/f.py",
            ),
        ],
    )

    result = run_static_rule_calibration_eval(
        input_path=gold,
        min_accuracy=0.9,
        min_macro_f1=0.9,
        production_min_support=3,
        production_min_purity=0.67,
    )

    assert result.summary["exploratory_ready"] is True
    assert result.summary["ready"] is False
    assert result.summary["best_ready_policy"] is not None
    assert result.summary["best_production_candidate_policy"] is None
    assert result.summary["recommendation"] == (
        "metadata_signal_exists_but_needs_more_gold_support_before_validator_promotion"
    )
    readiness = result.summary["production_readiness_diagnostics"]
    assert readiness["status"] == "exploratory_ready_only"
    assert readiness["missing_threshold_reason_counts"]["min_support_below_production"] > 0
    assert readiness["missing_threshold_reason_counts"]["min_purity_below_production"] > 0
    assert readiness["top_ready_nonproduction_policies"][0]["missing_production_support"] == 2
    assert readiness["recommendation"] == "collect_more_gold_labels_or_raise_support_for_exploratory_metadata_rules"
    support = result.summary["gold_support_diagnostics"]
    assert support["feature_bucket_count"] > 0
    assert support["priority_buckets"]
    assert support["priority_buckets"][0]["needed_additional_labels"] >= 0
    assert support["recommendation"].startswith("review_production_ready_buckets") or support[
        "recommendation"
    ].startswith("add_gold_labels_to_priority_buckets")
    candidates = result.summary["validator_rule_candidates"]
    assert candidates["schema_version"] == "prefix-static-rule-validator-candidates-v1"
    assert candidates["prompt_safe_summary"] is True
    assert candidates["status"] == "no_candidate_rules"
    assert candidates["candidate_rule_count"] == 0
    assert candidates["conflict_bucket_count"] > 0
    assert candidates["conflict_buckets"][0]["automation_policy"] == "manual_review_or_more_gold_labels"
    assert candidates["promotion_gate"]["automatic_promotion_ready"] is False
    assert "no_candidate_rules" in candidates["promotion_gate"]["blocking_reasons"]
    assert candidates["fail_closed_simulation"]["candidate_rule_count"] == 0
    assert candidates["fail_closed_simulation"]["matched_count"] == 0
    assert candidates["validator_review_plan"]["review_item_count"] > 0
    assert candidates["validator_review_plan"]["recommended_next_review_type"] == "gold_support_conflict_bucket"
    assert candidates["validator_review_plan"]["recommendation"] == "add_gold_labels_or_split_high_conflict_buckets"
    assert candidates["conservative_fail_closed_subset"]["included_rule_count"] == 0
    assert candidates["conservative_fail_closed_subset"]["recommendation"] == (
        "no_conservative_fail_closed_rules_after_exclusions"
    )
    assert candidates["shadow_trial_plan"]["shadow_trial_ready"] is False
    assert candidates["shadow_trial_plan"]["trial_rule_count"] == 0
    assert candidates["shadow_trial_plan"]["recommendation"] == (
        "resolve_conservative_subset_blockers_before_shadow_trial"
    )
    assert candidates["source_generalization_diagnostics"]["generalization_claim_allowed"] is False


def test_static_rule_calibration_eval_cli_exit_code_reflects_production_ready(tmp_path) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(
        gold,
        [
            ("accept-a", "accept", "accept", "verification_policy", [], "tmp/framework-src/autogen/pkg/a.py"),
            ("accept-b", "accept", "accept", "verification_policy", [], "tmp/framework-src/autogen/pkg/b.py"),
        ],
    )

    exit_code = main(
        [
            "--input",
            str(gold),
            "--production-min-support",
            "1",
            "--min-accuracy",
            "0.9",
            "--min-macro-f1",
            "0.9",
            "--summary",
            str(tmp_path / "summary.json"),
        ]
    )

    assert exit_code == 0
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["ready"] is True


def test_static_rule_shadow_trial_uses_full_conservative_subset_not_display_slice(tmp_path) -> None:
    gold = tmp_path / "gold.jsonl"
    rows = []
    for index in range(13):
        risk_tags = [f"shadow_risk_{index}", "agent_identity_boundary"]
        rows.extend(
            [
                (
                    f"reject-{index}-a",
                    "reject",
                    "review",
                    "verification_policy",
                    risk_tags,
                    f"tmp/framework-src/autogen/pkg/reject_{index}_a.py",
                ),
                (
                    f"reject-{index}-b",
                    "reject",
                    "review",
                    "verification_policy",
                    risk_tags,
                    f"tmp/framework-src/autogen/pkg/reject_{index}_b.py",
                ),
            ]
        )
    _write_gold(gold, rows)

    result = run_static_rule_calibration_eval(
        input_path=gold,
        min_accuracy=0.9,
        min_macro_f1=0.9,
        production_min_support=2,
        production_min_purity=1.0,
    )

    subset = result.summary["validator_rule_candidates"]["conservative_fail_closed_subset"]
    shadow_trial = result.summary["validator_rule_candidates"]["shadow_trial_plan"]
    assert subset["included_rule_count"] > 12
    assert len(subset["included_candidate_rules"]) == subset["included_rule_count"]
    assert shadow_trial["trial_rule_count"] == subset["included_rule_count"]
    assert len(shadow_trial["trial_rules"]) == shadow_trial["trial_rule_count"]


def _write_gold(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": example_id,
                    "expected_label": expected,
                    "source_label": rule_label,
                    "rule_label": rule_label,
                    "semantic_hint": semantic_hint,
                    "risk_tags": risk_tags,
                    "source_path": source_path,
                    "extraction": "prompt_named_constant",
                    "text": "stable shared verification rule",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for example_id, expected, rule_label, semantic_hint, risk_tags, source_path in rows
        ),
        encoding="utf-8",
    )
