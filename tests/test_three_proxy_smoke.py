from __future__ import annotations

import json

from autogen_prefix_tree.agbench_legacy_suite import generate_legacy_agbench_suite
from autogen_prefix_tree.three_proxy_smoke import main, run_three_proxy_smoke


def test_three_proxy_smoke_runs_baseline_rule_and_nl_variants(tmp_path) -> None:
    result = run_three_proxy_smoke(output_dir=tmp_path / "smoke", session_id="three-proxy-test")

    summary = result.summary
    assert summary["schema_version"] == "prefix-three-proxy-smoke-summary-v1"
    assert summary["fake_upstream"] is True
    assert summary["real_provider_metrics_available"] is False
    assert summary["upstream_request_count"] == 6
    assert summary["upstream_paths"] == ["/v1/chat/completions"]

    providers = summary["provider_summaries"]
    assert providers["baseline"]["record_count"] == 2
    assert providers["baseline"]["transformed_count"] == 0
    assert providers["plugin_rule_only"]["transformed_count"] == 1
    assert providers["plugin_nl_segmentation"]["transformed_count"] == 0
    assert providers["plugin_rule_only"]["actual_cached_tokens"] > providers["baseline"]["actual_cached_tokens"]
    assert providers["plugin_nl_segmentation"]["actual_cached_tokens"] == providers["baseline"]["actual_cached_tokens"]

    rule_delta = summary["ab_summaries"]["baseline_vs_rule_only"]["delta"]
    nl_delta = summary["ab_summaries"]["baseline_vs_nl_segmentation"]["delta"]
    assert rule_delta["transformed_count_delta"] == 1
    assert nl_delta["transformed_count_delta"] == 0
    assert rule_delta["actual_cached_tokens_delta"] > 0
    assert nl_delta["actual_cached_tokens_delta"] == 0
    assert summary["ab_summaries"]["baseline_vs_rule_only"]["fake_upstream"] is True
    assert summary["ab_summaries"]["baseline_vs_rule_only"]["real_provider_metrics_available"] is False
    assert summary["ab_summaries"]["baseline_vs_nl_segmentation"]["real_provider_metrics_available"] is False
    assert "fake upstream" in summary["real_provider_metrics_note"].lower()

    written = json.loads((tmp_path / "smoke" / "reports" / "three_proxy_smoke_summary.json").read_text(encoding="utf-8"))
    assert written["upstream_paths"] == ["/v1/chat/completions"]
    rule_summary = json.loads(
        (tmp_path / "smoke" / "reports" / "ab_baseline_vs_rule_only_summary.json").read_text(encoding="utf-8")
    )
    assert rule_summary["fake_upstream"] is True
    assert rule_summary["real_provider_metrics_available"] is False
    assert (tmp_path / "smoke" / "reports" / "ab_baseline_vs_nl_segmentation_report.md").exists()


def test_three_proxy_smoke_cli_writes_summary(tmp_path) -> None:
    output_dir = tmp_path / "cli-smoke"

    exit_code = main(["--output-dir", str(output_dir), "--session-id", "three-proxy-cli"])

    assert exit_code == 0
    summary = json.loads((output_dir / "reports" / "three_proxy_smoke_summary.json").read_text(encoding="utf-8"))
    assert summary["session_id"] == "three-proxy-cli"
    assert summary["provider_summaries"]["plugin_rule_only"]["transformed_count"] == 1


def test_three_proxy_smoke_can_fill_legacy_manifest_artifacts(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-three-proxy-smoke",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_shadow_plan(suite.manifest["local_judge"]["shadow_trial_plan_path"])

    result = run_three_proxy_smoke(
        output_dir=tmp_path / "smoke",
        session_id="legacy-three-proxy-smoke",
        manifest_path=suite.manifest_path,
    )

    summary = result.summary
    manifest = suite.manifest
    assert summary["manifest_path"] == suite.manifest_path
    assert summary["legacy_collection_summary_path"]
    assert summary["legacy_runbook_next_action"] == "run_offline_semantic_suite"
    assert summary["legacy_runbook_artifact_quality_warn_count"] >= 1
    assert summary["shadow_trial_plan_path"] == suite.manifest["local_judge"]["shadow_trial_plan_path"]
    assert summary["shadow_trial_summaries"]["baseline"]["supported"] is False
    assert summary["shadow_trial_summaries"]["plugin_rule_only"]["supported"] is True
    assert summary["shadow_trial_summaries"]["plugin_rule_only"]["record_with_match_count"] >= 1
    assert summary["shadow_trial_summaries"]["plugin_rule_only"]["matched_rule_observation_count"] >= 1
    assert summary["shadow_trial_summaries"]["plugin_nl_segmentation"]["supported"] is True

    baseline_provider = _read_jsonl(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"])
    assert baseline_provider
    assert len(_provider_rows(baseline_provider)) == 2
    assert all(row["fake_upstream"] is True for row in baseline_provider)
    assert _read_jsonl(manifest["oai_config_lists"]["baseline"]["task_results_path"])

    rule_summary = json.loads(open(manifest["recommended_ab_eval"]["rule_only_summary"], encoding="utf-8").read())
    nl_summary = json.loads(open(manifest["recommended_ab_eval"]["nl_segmentation_summary"], encoding="utf-8").read())
    collection_summary = json.loads(open(summary["legacy_collection_summary_path"], encoding="utf-8").read())
    runbook_summary = json.loads(open(manifest["recommended_runbook"]["summary"], encoding="utf-8").read())
    assert rule_summary["fake_upstream"] is True
    assert rule_summary["real_provider_metrics_available"] is False
    assert nl_summary["real_provider_metrics_available"] is False
    assert collection_summary["fake_upstream"] is True
    assert collection_summary["real_provider_metrics_available"] is False
    assert runbook_summary["real_provider_metrics_available"] is False
    assert runbook_summary["next_action"] == "run_offline_semantic_suite"
    assert any(item.get("fake_upstream") for item in runbook_summary["artifact_quality"])

    rerun = run_three_proxy_smoke(
        output_dir=tmp_path / "smoke",
        session_id="legacy-three-proxy-smoke",
        manifest_path=suite.manifest_path,
    )
    assert rerun.summary["provider_summaries"]["baseline"]["record_count"] == 2
    assert rerun.summary["provider_summaries"]["plugin_rule_only"]["record_count"] == 2
    assert rerun.summary["shadow_trial_summaries"]["plugin_rule_only"]["record_with_match_count"] >= 1
    assert len(_read_jsonl(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"])) == len(baseline_provider)
    assert len(_provider_rows(_read_jsonl(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"]))) == 2


def _task_jsonl(tmp_path):
    tasks_dir = tmp_path / "Tasks"
    tasks_dir.mkdir(exist_ok=True)
    template = tasks_dir / "template.py"
    template.write_text("print('template')\n", encoding="utf-8")
    path = tasks_dir / "human_eval_two_agents.jsonl"
    path.write_text(
        json.dumps({"id": "HumanEval_0", "template": "template.py", "substitutions": {}}) + "\n",
        encoding="utf-8",
    )
    return path


def _read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _provider_rows(rows):
    return [row for row in rows if row.get("schema_version") == "prefix-forward-proxy-provider-telemetry-v1"]


def _write_shadow_plan(path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": "prefix-static-rule-calibration-eval-summary-v1",
                    "prompt_safe_summary": True,
                    "row_count": 2,
                    "min_accuracy": 0.75,
                    "min_macro_f1": 0.7,
                    "production_min_support": 2,
                    "production_min_purity": 0.67,
                    "expected_label_counts": {"review": 2},
                    "policy_summaries": [
                        {
                            "policy_name": "rule_label",
                            "accuracy": 0.5,
                            "macro_f1": 0.5,
                            "ready": False,
                            "production_candidate": False,
                        }
                    ],
                    "top_policies": [
                        {
                            "policy_name": "metadata_calibration:semantic_hint+risk_tag_set",
                            "accuracy": 0.8,
                            "macro_f1": 0.8,
                            "ready": True,
                            "production_candidate": False,
                            "calibration_coverage_rate": 1.0,
                            "predicted_label_counts": {"review": 2},
                        }
                    ],
                    "best_ready_policy": "metadata_calibration:semantic_hint+risk_tag_set",
                    "best_production_candidate_policy": None,
                    "production_readiness_diagnostics": {
                        "schema_version": "prefix-static-rule-production-readiness-diagnostics-v1",
                        "prompt_safe_summary": True,
                        "status": "exploratory_ready_only",
                        "top_ready_nonproduction_policies": [],
                        "missing_threshold_reason_counts": {},
                        "recommendation": "collect_more_gold_labels_or_raise_support_for_exploratory_metadata_rules",
                    },
                    "gold_support_diagnostics": {
                        "schema_version": "prefix-static-rule-gold-support-diagnostics-v1",
                        "prompt_safe_summary": True,
                        "feature_bucket_count": 0,
                        "production_ready_bucket_count": 0,
                        "priority_buckets": [],
                        "recommendation": "build_goldset_before_static_rule_calibration",
                    },
                    "validator_rule_candidates": {
                        "schema_version": "prefix-static-rule-validator-candidates-v1",
                        "prompt_safe_summary": True,
                        "candidate_rule_count": 1,
                        "safe_accept_rule_count": 0,
                        "reject_rule_count": 0,
                        "review_rule_count": 1,
                        "conflict_bucket_count": 0,
                        "candidate_rules": [],
                        "conflict_buckets": [],
                        "overlap_diagnostics": {
                            "schema_version": "prefix-static-rule-candidate-overlap-diagnostics-v1",
                            "prompt_safe_summary": True,
                            "overlap_pair_count": 0,
                            "conflicting_overlap_pair_count": 0,
                            "same_action_overlap_pair_count": 0,
                            "conflicting_overlap_pairs": [],
                        },
                        "fail_closed_simulation": {
                            "schema_version": "prefix-static-rule-validator-fail-closed-simulation-v1",
                            "prompt_safe_summary": True,
                            "candidate_rule_count": 1,
                            "matched_count": 0,
                            "mismatch_count": 0,
                            "sample_mismatches": [],
                            "safe_for_automatic_validator_promotion": False,
                            "automation_policy": "diagnostic_only_until_manual_review_and_real_ab",
                        },
                        "promotion_gate": {
                            "schema_version": "prefix-static-rule-validator-promotion-gate-v1",
                            "prompt_safe_summary": True,
                            "automatic_promotion_ready": False,
                            "manual_review_ready": True,
                            "blocking_reasons": ["manual_code_review_required", "real_provider_ab_required"],
                        },
                        "validator_review_plan": {
                            "schema_version": "prefix-static-rule-validator-review-plan-v1",
                            "prompt_safe_summary": True,
                            "review_item_count": 0,
                            "blocking_review_item_count": 0,
                            "priority_reason_counts": {},
                            "review_items": [],
                        },
                        "conservative_fail_closed_subset": {
                            "schema_version": "prefix-static-rule-conservative-fail-closed-subset-v1",
                            "prompt_safe_summary": True,
                            "included_rule_count": 1,
                            "excluded_rule_count": 0,
                            "included_candidate_rules": [],
                            "excluded_candidate_rules": [],
                            "simulation": {
                                "schema_version": "prefix-static-rule-validator-fail-closed-simulation-v1",
                                "prompt_safe_summary": True,
                                "candidate_rule_count": 1,
                                "matched_count": 0,
                                "mismatch_count": 0,
                                "sample_mismatches": [],
                                "safe_for_automatic_validator_promotion": False,
                            },
                            "manual_trial_ready": True,
                            "safe_for_automatic_validator_promotion": False,
                        },
                        "shadow_trial_plan": {
                            "schema_version": "prefix-static-rule-shadow-trial-plan-v1",
                            "trial_mode": "shadow_only",
                            "trial_rules": [
                                {
                                    "trial_rule_id": "shadow_rule:team-policy-none",
                                    "feature_set": "semantic_hint+risk_tag_set",
                                    "features": {"semantic_hint": "team_policy", "risk_tag_set": "none"},
                                    "shadow_action": "prefer_review",
                                    "support": 4,
                                    "purity": 1.0,
                                }
                            ],
                        }
                    },
                    "ready": False,
                    "exploratory_ready": True,
                    "semantic_quality_claim_allowed": False,
                    "real_provider_metrics_available": False,
                    "gold_text_written": False,
                    "recommendation": "metadata_signal_exists_but_needs_more_gold_support_before_validator_promotion",
                    "limits": "metadata-only test fixture",
                },
                ensure_ascii=False,
            )
        )
