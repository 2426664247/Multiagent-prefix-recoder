from __future__ import annotations

import json
from types import SimpleNamespace

import autogen_prefix_tree.agbench_legacy_preflight as preflight
from autogen_prefix_tree.agbench_legacy_preflight import build_legacy_agbench_preflight, main
from autogen_prefix_tree.agbench_legacy_suite import generate_legacy_agbench_suite


def test_legacy_preflight_reports_missing_api_without_secret_values(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-missing-api",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _patch_readiness(monkeypatch, ready=False, api_present=False)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={},
        summary_path=tmp_path / "preflight.json",
        report_path=tmp_path / "preflight.md",
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["schema_version"] == "prefix-legacy-real-ab-preflight-v1"
    assert summary["prompt_safe_summary"] is True
    assert summary["gates"]["environment_ready"] is False
    assert summary["gates"]["suite_structural_ready"] is True
    assert summary["gates"]["api_config_present"] is False
    assert summary["gates"]["ready_to_start_real_ab"] is False
    assert summary["next_action"] == "set_api_config_before_real_run"
    assert summary["local_only_next_actions"][0]["id"] == "run_offline_semantic_suite"
    assert any("offline_semantic_suite" in command for command in summary["local_only_next_actions"][0]["commands"])
    assert "missing_api_config" in summary["blocking_reasons"]
    gap_names = {gap["name"] for gap in summary["evidence_gaps"]}
    assert "provider_api_config" in gap_names
    assert "local_judge_config" in gap_names
    assert "real_provider_metrics" in gap_names
    assert summary["claim_gate"]["real_provider_claims_allowed"] is False
    assert summary["claim_gate"]["semantic_quality_claims_allowed"] is False
    assert summary["claim_gate"]["semantic_chain_ready"] is False
    assert "real_cache_hit_or_cached_token_improvement" in summary["claim_gate"]["prohibited_claim_ids"]
    assert "real_latency_or_cost_savings" in summary["claim_gate"]["prohibited_claim_ids"]
    assert "real_task_success_improvement" in summary["claim_gate"]["prohibited_claim_ids"]
    assert "real_local_model_semantic_quality" in summary["claim_gate"]["prohibited_claim_ids"]
    assert summary["allowed_claims"] == summary["claim_gate"]["allowed_claims"]
    assert summary["prohibited_claims"] == summary["claim_gate"]["prohibited_claims"]
    assert "sk-secret" not in serialized
    report_text = (tmp_path / "preflight.md").read_text(encoding="utf-8")
    assert "Evidence Gaps" in report_text
    assert "Claim Gate" in report_text
    assert "real_cache_hit_or_cached_token_improvement" in report_text
    assert "provider_api_config" in report_text
    assert "Legacy AutoGenBench Real A/B Preflight" in report_text
    assert "Local-Only Next Actions" in report_text


def test_legacy_preflight_prioritizes_offline_semantic_suite_before_replacing_fake_smoke(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-fake",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = suite.manifest
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = manifest["oai_config_lists"][variant]
        _write_provider(info["provider_telemetry_path"], fake=True)
        _write_text(info["tabulate_csv_path"], "Task Id,Trial 0 Success\nHumanEval_0,True\n")
        _write_task_results(info["task_results_path"])
    for field in ("rule_only_summary", "nl_segmentation_summary"):
        _write_ab_summary(manifest["recommended_ab_eval"][field], fake=True)
    for field in ("rule_only_report_md", "nl_segmentation_report_md"):
        _write_text(manifest["recommended_ab_eval"][field], "# Prefix Reorder A/B Summary\n\n## Gates\n")
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["collection_inputs_ready"] is True
    assert summary["gates"]["ab_reports_ready"] is True
    assert summary["gates"]["fake_artifacts_detected"] is True
    assert summary["gates"]["real_provider_metrics_available"] is False
    assert summary["gates"]["ready_to_claim_real_results"] is False
    assert summary["next_action"] == "run_offline_semantic_suite"
    assert "fake_smoke_artifacts_do_not_prove_real_metrics" in summary["blocking_reasons"]
    assert "offline_semantic_suite_not_run" in summary["blocking_reasons"]
    assert summary["claim_gate"]["real_provider_claims_allowed"] is False
    assert "fake_three_proxy_smoke_wiring" in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_provider_ab_metrics" not in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_cache_hit_or_cached_token_improvement" in summary["claim_gate"]["prohibited_claim_ids"]
    assert any("offline_semantic_suite" in command for command in summary["suggested_commands"])
    assert "sk-secret-value" not in serialized


def test_legacy_preflight_allows_only_shadow_trial_wiring_claim_from_plugin_telemetry(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-shadow-trial",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = suite.manifest
    _write_provider(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"], fake=False)
    _write_provider_with_shadow(
        manifest["oai_config_lists"]["plugin_rule_only"]["provider_telemetry_path"],
        rule_id="shadow_rule:rule-only",
        action="prefer_review",
    )
    _write_provider_with_shadow(
        manifest["oai_config_lists"]["plugin_nl_segmentation"]["provider_telemetry_path"],
        rule_id="shadow_rule:nl",
        action="force_reject",
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["shadow_trial_wiring_ready"] is True
    assert "shadow_trial_telemetry_wiring" in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_provider_ab_metrics" not in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_cache_hit_or_cached_token_improvement" in summary["claim_gate"]["prohibited_claim_ids"]
    assert "broad_semantic_rule_generalization" in summary["claim_gate"]["prohibited_claim_ids"]
    serialized = json.dumps(summary, ensure_ascii=False)
    assert "sk-secret-value" not in serialized


def test_legacy_preflight_suggests_replacing_fake_smoke_after_semantic_chain_ready(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-fake-after-semantic-chain",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    manifest = suite.manifest
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = manifest["oai_config_lists"][variant]
        _write_provider(info["provider_telemetry_path"], fake=True)
        _write_text(info["tabulate_csv_path"], "Task Id,Trial 0 Success\nHumanEval_0,True\n")
        _write_task_results(info["task_results_path"])
    for field in ("rule_only_summary", "nl_segmentation_summary"):
        _write_ab_summary(manifest["recommended_ab_eval"][field], fake=True)
    for field in ("rule_only_report_md", "nl_segmentation_report_md"):
        _write_text(manifest["recommended_ab_eval"][field], "# Prefix Reorder A/B Summary\n\n## Gates\n")
    _write_semantic_guard_smoke_summary(
        tmp_path / "legacy" / "preflight_semantic_guard_proxy_smoke" / "semantic_guard_proxy_smoke_summary.json"
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_goldset_chain_ready_summaries(tmp_path)
    _write_offline_local_judge_matrix_summary(
        tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    )
    _write_offline_local_judge_matrix_smoke_summary(
        tmp_path
        / "legacy"
        / "offline_local_judge_matrix_smoke"
        / "offline_local_judge_matrix_smoke_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["fake_artifacts_detected"] is True
    assert summary["gates"]["offline_semantic_suite_ready"] is True
    assert summary["gates"]["local_judge_config_ready"] is True
    assert summary["gates"]["local_judge_healthcheck_ready"] is True
    assert summary["gates"]["offline_local_judge_matrix_ready"] is True
    assert summary["gates"]["offline_local_judge_matrix_smoke_ready"] is True
    assert summary["gates"]["semantic_guard_fake_smoke_ready"] is True
    assert summary["next_action"] == "run_real_ab_after_fake_smoke"
    assert "fake_smoke_artifacts_do_not_prove_real_metrics" in summary["blocking_reasons"]
    assert "offline_semantic_suite_not_run" not in summary["blocking_reasons"]
    gap_names = {gap["name"] for gap in summary["evidence_gaps"]}
    assert "real_provider_artifacts" in gap_names
    assert "real_provider_metrics" in gap_names
    assert summary["claim_gate"]["semantic_chain_ready"] is True
    assert summary["claim_gate"]["real_provider_claims_allowed"] is False
    assert "local_judge_review_resolution_diagnostics" in summary["claim_gate"]["allowed_claim_ids"]
    assert "review_candidates_excluded_from_automatic_promotion" in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_provider_ab_metrics" not in summary["claim_gate"]["allowed_claim_ids"]
    assert summary["claim_gate"]["semantic_quality_claims_allowed"] is True
    assert "local_model_semantic_quality_on_gold_set" in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_local_model_semantic_quality" not in summary["claim_gate"]["prohibited_claim_ids"]
    assert "real_cache_hit_or_cached_token_improvement" in summary["claim_gate"]["prohibited_claim_ids"]
    assert any("openai_forward_proxy" in command for command in summary["suggested_commands"])


def test_legacy_preflight_keeps_missing_gold_validation_as_quality_gap_not_provider_blocker(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-gold-validation-gap",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    manifest = suite.manifest
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = manifest["oai_config_lists"][variant]
        _write_provider(info["provider_telemetry_path"], fake=True)
        _write_text(info["tabulate_csv_path"], "Task Id,Trial 0 Success\nHumanEval_0,True\n")
        _write_task_results(info["task_results_path"])
    for field in ("rule_only_summary", "nl_segmentation_summary"):
        _write_ab_summary(manifest["recommended_ab_eval"][field], fake=True)
    for field in ("rule_only_report_md", "nl_segmentation_report_md"):
        _write_text(manifest["recommended_ab_eval"][field], "# Prefix Reorder A/B Summary\n\n## Gates\n")
    _write_semantic_guard_smoke_summary(
        tmp_path / "legacy" / "preflight_semantic_guard_proxy_smoke" / "semantic_guard_proxy_smoke_summary.json"
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json"
    )
    _write_offline_local_judge_matrix_summary(
        tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    )
    _write_offline_local_judge_matrix_smoke_summary(
        tmp_path
        / "legacy"
        / "offline_local_judge_matrix_smoke"
        / "offline_local_judge_matrix_smoke_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    gap_names = {gap["name"] for gap in summary["evidence_gaps"]}
    assert summary["gates"]["local_judge_goldset_template_ready"] is True
    assert summary["gates"]["local_judge_goldset_annotation_ready"] is False
    assert summary["gates"]["local_judge_goldset_import_ready"] is False
    assert summary["gates"]["local_judge_goldset_validation_ready"] is False
    assert summary["gates"]["local_judge_quality_eval_ready"] is False
    assert summary["next_action"] == "complete_local_judge_goldset_annotation"
    assert "local_judge_goldset_template" not in gap_names
    assert "local_judge_goldset_annotation" in gap_names
    assert "local_judge_goldset_import" in gap_names
    assert "local_judge_goldset_validation" in gap_names
    assert "local_judge_quality_eval" in gap_names
    assert "local_judge_goldset_annotation_not_ready" in summary["blocking_reasons"]
    assert "local_judge_goldset_import_not_ready" in summary["blocking_reasons"]
    assert summary["claim_gate"]["semantic_quality_claims_allowed"] is False
    assert "real_local_model_semantic_quality" in summary["claim_gate"]["prohibited_claim_ids"]
    assert any("local_judge_goldset_csv progress" in command for command in summary["suggested_commands"])
    assert not any("local_judge_goldset_csv import" in command for command in summary["suggested_commands"])


def test_legacy_preflight_accepts_real_ab_artifacts(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-real",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    manifest = suite.manifest
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = manifest["oai_config_lists"][variant]
        _write_provider(info["provider_telemetry_path"], fake=False)
        _write_text(info["tabulate_csv_path"], "Task Id,Trial 0 Success\nHumanEval_0,True\n")
        _write_task_results(info["task_results_path"])
    for field in ("rule_only_summary", "nl_segmentation_summary"):
        _write_ab_summary(manifest["recommended_ab_eval"][field], fake=False)
    for field in ("rule_only_report_md", "nl_segmentation_report_md"):
        _write_text(manifest["recommended_ab_eval"][field], "# Prefix Reorder A/B Summary\n\n## Gates\n")
    _write_semantic_guard_smoke_summary(
        tmp_path / "legacy" / "preflight_semantic_guard_proxy_smoke" / "semantic_guard_proxy_smoke_summary.json"
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_goldset_chain_ready_summaries(tmp_path)
    _write_offline_local_judge_matrix_summary(
        tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    )
    _write_offline_local_judge_matrix_smoke_summary(
        tmp_path
        / "legacy"
        / "offline_local_judge_matrix_smoke"
        / "offline_local_judge_matrix_smoke_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        report_path=tmp_path / "legacy" / "reports" / "legacy_real_ab_preflight.md",
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["ready_to_claim_real_results"] is True
    assert summary["gates"]["real_provider_metrics_available"] is True
    assert summary["gates"]["local_judge_config_ready"] is True
    assert summary["gates"]["offline_semantic_suite_status"]["prompt_extraction_diagnostics"] == {
        "extraction_counts": {"json_prompt_value": 2, "prompt_named_constant": 3}
    }
    assert summary["gates"]["offline_local_judge_matrix_status"]["candidate_label_diagnostics"] == {
        "rule_label_counts": {"accept": 10, "review": 8},
        "model_label_counts": {"accept": 8},
        "rule_to_model_label_counts": {"review->accept": 8},
        "model_to_final_label_counts": {"accept->accept": 8},
        "rule_to_final_label_counts": {"accept->accept": 10, "review->accept": 8},
        "static_safety_clamp_count": 0,
        "local_judge_effectiveness_diagnostics": {
            "schema_version": "prefix-matrix-local-judge-effectiveness-diagnostics-v1",
            "source_with_effectiveness_diagnostics_count": 3,
            "rule_review_candidate_count": 8,
            "model_called_rule_review_count": 8,
            "resolved_rule_review_count": 8,
            "called_resolved_rule_review_count": 8,
            "remaining_rule_review_count": 0,
            "resolution_rate": 1.0,
            "called_resolution_rate": 1.0,
            "rule_review_final_label_counts": {"accept": 8},
            "rule_review_model_label_counts": {"accept": 8},
            "rule_review_local_judge_action_counts": {"model_called": 8},
        },
    }
    assert summary["gates"]["offline_local_judge_matrix_status"]["prompt_extraction_diagnostics"] == {
        "extraction_counts": {"json_prompt_value": 6, "prompt_named_constant": 12}
    }
    assert summary["gates"]["offline_local_judge_matrix_smoke_status"]["candidate_label_diagnostics"] == {
        "rule_label_counts": {"accept": 6, "review": 6},
        "model_label_counts": {"accept": 4},
        "rule_to_model_label_counts": {"review->accept": 4},
        "model_to_final_label_counts": {"accept->accept": 4},
        "rule_to_final_label_counts": {"accept->accept": 6, "review->accept": 4, "review->review": 2},
        "static_safety_clamp_count": 0,
        "local_judge_effectiveness_diagnostics": {
            "schema_version": "prefix-matrix-local-judge-effectiveness-diagnostics-v1",
            "source_with_effectiveness_diagnostics_count": 3,
            "rule_review_candidate_count": 6,
            "model_called_rule_review_count": 4,
            "resolved_rule_review_count": 4,
            "called_resolved_rule_review_count": 4,
            "remaining_rule_review_count": 2,
            "resolution_rate": 2 / 3,
            "called_resolution_rate": 1.0,
            "rule_review_final_label_counts": {"accept": 4, "review": 2},
            "rule_review_model_label_counts": {"accept": 4},
            "rule_review_local_judge_action_counts": {"model_called": 4, "skipped_by_budget": 2},
        },
    }
    assert summary["gates"]["offline_local_judge_matrix_smoke_status"]["prompt_extraction_diagnostics"] == {
        "extraction_counts": {"json_prompt_value": 2, "prompt_named_constant": 4}
    }
    report = (tmp_path / "legacy" / "reports" / "legacy_real_ab_preflight.md").read_text(encoding="utf-8")
    assert "Offline Evidence Diagnostics" in report
    assert "nl_segmentation_applied_count" in report
    assert "nl_segmentation_total_estimated_gain_chars" in report
    assert "rule_gap_resolved" in report
    assert "recommendation" in report
    assert "rule_review_candidate_count" in report
    assert "called_resolution_rate" in report
    assert "prompt_extraction_counts" in report
    assert "semantic_guard_fake_smoke" in report
    assert "adapter_validation_reasons" in report
    assert "candidate_promotion_policy" in report
    assert "candidate_review_upper_bound_policy" in report
    assert "experiment_gate_items" in report
    assert "review_candidate_promotion_policy" in report
    assert "review_queue_risk_tag_counts" in report
    assert "agent_identity_boundary" in report
    assert "review_queue_local_judge_priority" in report
    assert "Claim Gate" in report
    assert "real_provider_ab_metrics" in report
    assert summary["claim_gate"]["semantic_chain_ready"] is True
    assert summary["claim_gate"]["real_provider_claims_allowed"] is True
    assert "real_provider_ab_metrics" in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_cache_hit_or_cached_token_improvement" not in summary["claim_gate"]["prohibited_claim_ids"]
    assert "real_latency_or_cost_savings" not in summary["claim_gate"]["prohibited_claim_ids"]
    assert "real_task_success_improvement" not in summary["claim_gate"]["prohibited_claim_ids"]
    assert summary["claim_gate"]["semantic_quality_claims_allowed"] is True
    assert "local_model_semantic_quality_on_gold_set" in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_local_model_semantic_quality" not in summary["claim_gate"]["prohibited_claim_ids"]
    assert summary["blocking_reasons"] == []
    assert summary["evidence_gaps"] == []
    assert summary["next_action"] == "inspect_ab_reports"


def test_legacy_preflight_requires_chain_verifier_after_gold_eval(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-real-with-quality",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json"
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_import_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_import.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_validation_summary.json"
    )
    _write_local_judge_quality_eval_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_quality_eval_summary.json",
        ready=True,
    )
    _write_offline_local_judge_matrix_summary(
        tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    )
    _write_offline_local_judge_matrix_smoke_summary(
        tmp_path
        / "legacy"
        / "offline_local_judge_matrix_smoke"
        / "offline_local_judge_matrix_smoke_summary.json"
    )
    _write_semantic_guard_smoke_summary(
        tmp_path / "legacy" / "preflight_semantic_guard_proxy_smoke" / "semantic_guard_proxy_smoke_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        report_path=tmp_path / "legacy" / "reports" / "legacy_real_ab_preflight.md",
    )

    summary = result.summary
    assert summary["gates"]["local_judge_goldset_template_ready"] is True
    assert summary["gates"]["local_judge_goldset_progress_ready"] is True
    assert summary["gates"]["local_judge_goldset_annotation_ready"] is True
    assert summary["gates"]["local_judge_goldset_import_ready"] is True
    assert summary["gates"]["local_judge_goldset_validation_ready"] is True
    assert summary["gates"]["local_judge_quality_eval_ready"] is True
    assert summary["gates"]["local_judge_goldset_chain_ready"] is False
    assert summary["next_action"] == "verify_local_judge_goldset_chain"
    assert summary["claim_gate"]["semantic_quality_claims_allowed"] is False
    assert "local_model_semantic_quality_on_gold_set" not in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_local_model_semantic_quality" in summary["claim_gate"]["prohibited_claim_ids"]
    assert "local_judge_goldset_chain" in {gap["name"] for gap in summary["evidence_gaps"]}
    assert any("local_judge_goldset_chain_verify" in command for command in summary["suggested_commands"])
    assert "broad_semantic_rule_generalization" in summary["claim_gate"]["prohibited_claim_ids"]
    report = (tmp_path / "legacy" / "reports" / "legacy_real_ab_preflight.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_template" in report
    assert "local_judge_goldset_progress" in report
    assert "local_judge_quality_eval" in report
    assert "local_judge_goldset_chain" in report
    assert "macro_f1" in report
    assert "raw_model_accuracy" in report
    assert "static_safety_clamp_count" in report
    assert "predicted_label_collapse" in report
    assert "raw_model_reject_recall" in report


def test_legacy_preflight_reports_static_rule_calibration_after_failed_quality_eval(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-static-calibration",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json"
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_import_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_import.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_validation_summary.json"
    )
    _write_local_judge_quality_eval_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_quality_eval_summary.json",
        ready=False,
    )
    _write_static_rule_calibration_eval_summary(
        tmp_path / "legacy" / "artifacts" / "static_rule_calibration_eval_summary.json",
        ready=False,
        exploratory_ready=True,
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        allow_missing_api=True,
        report_path=tmp_path / "legacy" / "reports" / "legacy_real_ab_preflight.md",
    )

    summary = result.summary
    assert summary["gates"]["local_judge_quality_eval_ready"] is False
    assert summary["gates"]["static_rule_calibration_eval_ready"] is False
    assert summary["gates"]["static_rule_calibration_eval_status"]["exploratory_ready"] is True
    gate = summary["gates"]["static_rule_calibration_eval_status"]["validator_rule_candidates"]["promotion_gate"]
    assert gate["automatic_promotion_ready"] is False
    assert "overlapping_candidate_rules_have_conflicting_actions" in gate["blocking_reasons"]
    simulation = summary["gates"]["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "fail_closed_simulation"
    ]
    assert simulation["matched_count"] == 2
    assert simulation["mismatch_count"] == 1
    assert simulation["safe_for_automatic_validator_promotion"] is False
    review_plan = summary["gates"]["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "validator_review_plan"
    ]
    assert review_plan["review_item_count"] == 3
    assert review_plan["recommended_next_review_type"] == "conflicting_overlap_pair"
    assert review_plan["review_items"][1]["input_index"] == 2
    subset = summary["gates"]["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "conservative_fail_closed_subset"
    ]
    assert subset["included_rule_count"] == 0
    assert subset["excluded_rule_count"] == 1
    assert subset["simulation"]["candidate_rule_count"] == 0
    assert subset["manual_trial_ready"] is False
    shadow_trial = summary["gates"]["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "shadow_trial_plan"
    ]
    assert shadow_trial["prompt_safe_summary"] is True
    assert shadow_trial["trial_mode"] == "shadow_only"
    assert shadow_trial["shadow_trial_ready"] is False
    assert shadow_trial["validator_behavior_change_allowed"] is False
    source_generalization = summary["gates"]["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "source_generalization_diagnostics"
    ]
    assert source_generalization["candidate_with_multi_source_file_count"] == 2
    assert source_generalization["generalization_claim_allowed"] is False
    assert summary["next_action"] == "collect_more_gold_labels_or_recalibrate_validator_rules"
    action_by_id = {action["id"]: action for action in summary["local_only_next_actions"]}
    assert "collect_more_gold_labels_or_recalibrate_validator_rules" in action_by_id
    assert any("static_rule_calibration_eval" in command for command in summary["suggested_commands"])
    assert "static_rule_metadata_calibration_diagnostics" in summary["claim_gate"]["allowed_claim_ids"]
    assert "local_model_semantic_quality_on_gold_set" not in summary["claim_gate"]["allowed_claim_ids"]
    assert "real_local_model_semantic_quality" in summary["claim_gate"]["prohibited_claim_ids"]
    assert "broad_semantic_rule_generalization" in summary["claim_gate"]["prohibited_claim_ids"]
    assert "local_judge_quality_eval" in {gap["name"] for gap in summary["evidence_gaps"]}
    report = (tmp_path / "legacy" / "reports" / "legacy_real_ab_preflight.md").read_text(encoding="utf-8")
    assert "static_rule_calibration_eval" in report
    assert "baseline_rule_accuracy" in report
    assert "automatic_promotion_ready" in report
    assert "overlapping_candidate_rules_have_conflicting_actions" in report
    assert "fail_closed_simulation" in report
    assert "validator_review_plan" in report
    assert "conservative_fail_closed_subset" in report
    assert "shadow_trial_plan" in report
    assert "source_generalization_diagnostics" in report
    assert "conflicting_overlap_pair" in report
    assert "diagnostic_only_until_manual_review_and_real_ab" in report


def test_legacy_preflight_suggests_static_rule_calibration_when_quality_eval_failed(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-static-calibration-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json"
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_import_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_import.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_validation_summary.json"
    )
    _write_local_judge_quality_eval_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_quality_eval_summary.json",
        ready=False,
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    summary = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        allow_missing_api=True,
    ).summary

    assert summary["next_action"] == "run_static_rule_calibration_eval"
    action_by_id = {action["id"]: action for action in summary["local_only_next_actions"]}
    assert "run_static_rule_calibration_eval" in action_by_id
    assert any("static_rule_calibration_eval" in command for command in summary["suggested_commands"])


def test_legacy_preflight_reads_synthesized_chain_summary_for_old_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-old-chain-summary",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    manifest_path = tmp_path / "legacy" / "legacy_suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["suggested_commands"].pop("local_judge_goldset_chain_verify", None)
    manifest["local_judge"].pop("goldset_chain_verification_summary_path", None)
    manifest["local_judge"].pop("goldset_chain_verification_report_path", None)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_local_judge_goldset_chain_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_chain_verification_summary.json",
        ready=False,
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    chain_status = result.summary["gates"]["local_judge_goldset_chain_status"]
    assert result.summary["gates"]["local_judge_goldset_chain_ready"] is False
    assert chain_status["status"] == "warn"
    assert chain_status["next_action"] == "run_apply_labels_require_complete"
    assert chain_status["blocking_reasons"] == ["apply_labels:ready"]
    assert chain_status["evidence"] != "artifact missing or empty"


def test_legacy_preflight_suggests_offline_semantic_suite_before_real_ab(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-offline-suite-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["offline_semantic_suite_ready"] is False
    assert summary["gates"]["semantic_guard_fake_smoke_ready"] is False
    assert "offline_semantic_suite_not_run" in summary["blocking_reasons"]
    assert "semantic_guard_fake_smoke_not_run" in summary["blocking_reasons"]
    assert summary["next_action"] == "run_offline_semantic_suite"
    assert any("offline_semantic_suite" in command for command in summary["suggested_commands"])
    assert any("--source-root <autogen-source-root>" in command for command in summary["suggested_commands"])


def test_legacy_preflight_suggests_configure_local_judge_after_offline_suite(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-local-judge-matrix-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["offline_semantic_suite_ready"] is True
    assert summary["gates"]["local_judge_config_ready"] is False
    assert summary["gates"]["local_judge_healthcheck_ready"] is False
    assert summary["gates"]["offline_local_judge_matrix_smoke_ready"] is False
    assert summary["gates"]["semantic_guard_fake_smoke_ready"] is False
    assert "offline_semantic_suite_not_run" not in summary["blocking_reasons"]
    assert "local_judge_not_configured" in summary["blocking_reasons"]
    assert "local_judge_healthcheck_not_run" in summary["blocking_reasons"]
    assert "offline_local_judge_matrix_smoke_not_run" in summary["blocking_reasons"]
    assert "semantic_guard_fake_smoke_not_run" in summary["blocking_reasons"]
    assert summary["next_action"] == "configure_local_judge"
    assert any("--local-judge-base-url <real-local-judge-base-url>" in command for command in summary["suggested_commands"])


def test_legacy_preflight_lists_parallel_local_actions_when_api_missing_and_csv_labels_pending(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-local-parallel-actions",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json"
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=False,
    )
    _write_local_judge_goldset_labels_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_labels.local.summary.json"
    )
    _write_local_judge_goldset_suggestions_summary(
        tmp_path
        / "legacy"
        / "artifacts"
        / "local_judge_goldset_annotation_label_suggestions.local.summary.json"
    )
    _write_local_judge_goldset_suggestion_review_summary(
        tmp_path
        / "legacy"
        / "artifacts"
        / "local_judge_goldset_annotation_suggestion_review.local.summary.json"
    )
    _write_local_judge_goldset_review_plan_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_review_plan.local.summary.json"
    )
    _write_local_judge_goldset_labels_from_csv_summary(
        tmp_path
        / "legacy"
        / "artifacts"
        / "local_judge_goldset_annotation_labels_from_csv.local.summary.json",
        ready=False,
    )
    _write_local_judge_goldset_labels_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_labels_validation.local.summary.json",
        ready=False,
    )
    _write_local_judge_goldset_import_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_import.local.summary.json",
        ready=False,
    )
    _patch_readiness(monkeypatch, ready=False, api_present=False)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={},
    )

    summary = result.summary
    action_by_id = {action["id"]: action for action in summary["local_only_next_actions"]}
    assert summary["next_action"] == "set_api_config_before_real_run"
    assert "configure_local_judge" in action_by_id
    assert "complete_local_judge_goldset_annotation" in action_by_id
    assert "run_local_judge_healthcheck" not in action_by_id
    assert "run_local_judge_quality_eval" not in action_by_id
    assert summary["gates"]["local_judge_goldset_annotation_ready"] is False
    assert summary["gates"]["local_judge_goldset_progress_status"]["expected_label_pending_count"] == 2
    assert summary["gates"]["local_judge_goldset_progress_status"]["pending_semantic_hint_counts"] == {
        "verification_policy": 2
    }
    assert summary["gates"]["local_judge_goldset_progress_status"]["worklist_written"] is True
    assert summary["gates"]["local_judge_goldset_progress_status"]["worklist_row_count"] == 2
    assert summary["gates"]["local_judge_goldset_labels_template_ready"] is True
    assert summary["gates"]["local_judge_goldset_labels_template_status"]["ready_for_apply_labels"] is True
    assert summary["gates"]["local_judge_goldset_labels_template_status"]["label_status_counts"] == {"pending": 3}
    assert summary["gates"]["local_judge_goldset_suggestions_ready"] is True
    assert summary["gates"]["local_judge_goldset_suggestions_status"]["suggestion_count"] == 3
    assert summary["gates"]["local_judge_goldset_suggestions_status"]["model_called_count"] == 3
    assert summary["gates"]["local_judge_goldset_suggestions_status"]["ready_for_apply_labels"] is False
    assert summary["gates"]["local_judge_goldset_suggestion_review_ready"] is True
    assert summary["gates"]["local_judge_goldset_suggestion_review_status"]["matched_suggestion_count"] == 3
    assert summary["gates"]["local_judge_goldset_suggestion_review_status"]["ready_for_apply_labels"] is False
    assert summary["gates"]["local_judge_goldset_review_plan_ready"] is True
    assert summary["gates"]["local_judge_goldset_review_plan_status"]["plan_disagreement_row_count"] == 1
    assert summary["gates"]["local_judge_goldset_review_plan_status"]["suggested_labels_not_auto_applied"] is True
    assert summary["gates"]["local_judge_goldset_labels_from_csv_ready"] is False
    assert summary["gates"]["local_judge_goldset_labels_from_csv_status"]["expected_label_pending_count"] == 2
    assert summary["gates"]["local_judge_goldset_labels_from_csv_status"]["suggested_labels_not_auto_applied"] is True
    assert summary["gates"]["local_judge_goldset_labels_validation_ready"] is False
    assert summary["gates"]["local_judge_goldset_labels_validation_status"]["expected_label_pending_count"] == 2
    assert summary["gates"]["local_judge_goldset_labels_validation_status"]["source_expected_label_match_count"] == 1
    assert summary["gates"]["local_judge_goldset_labels_validation_status"]["source_expected_label_mismatch_count"] == 0
    assert summary["gates"]["local_judge_goldset_labels_validation_status"]["source_expected_label_mismatch_counts"] == {}
    assert summary["gates"]["local_judge_goldset_labels_validation_status"]["possible_rule_self_confirmation"] is False
    assert "Pending labels: 2" in action_by_id["complete_local_judge_goldset_annotation"]["reason"]
    assert "labels template ready: true" in action_by_id["complete_local_judge_goldset_annotation"]["reason"]
    assert "local suggestions ready: true" in action_by_id["complete_local_judge_goldset_annotation"]["reason"]
    assert "suggestion review CSV ready: true" in action_by_id["complete_local_judge_goldset_annotation"]["reason"]
    assert "labels-from-csv ready: false" in action_by_id["complete_local_judge_goldset_annotation"]["reason"]
    assert "not gold labels" in action_by_id["complete_local_judge_goldset_annotation"]["reason"]
    assert any(
        "local_judge_goldset_csv progress" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert any(
        "local_judge_goldset_csv merge-suggestions" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert any(
        "local_judge_goldset_csv review-plan" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert any(
        "local_judge_goldset_csv labels-from-csv" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert any(
        "local_judge_goldset_csv labels-template" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert any(
        "local_judge_goldset_csv suggest-labels" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert any(
        "local_judge_goldset_csv validate-labels" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert any(
        "local_judge_goldset_csv apply-labels" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )
    assert not any(
        "local_judge_goldset_csv import" in command
        for command in action_by_id["complete_local_judge_goldset_annotation"]["commands"]
    )


def test_legacy_preflight_suggests_local_judge_healthcheck_after_configured_offline_suite(
    monkeypatch,
    tmp_path,
) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-local-judge-healthcheck-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["offline_semantic_suite_ready"] is True
    assert summary["gates"]["local_judge_config_ready"] is True
    assert summary["gates"]["local_judge_healthcheck_ready"] is False
    assert "local_judge_not_configured" not in summary["blocking_reasons"]
    assert summary["next_action"] == "run_local_judge_healthcheck"
    assert any("local_judge_healthcheck" in command for command in summary["suggested_commands"])


def test_legacy_preflight_blocks_failed_offline_semantic_suite_gate(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-offline-suite-failed-gate",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json",
        unsafe_review_promotion=True,
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["artifact_quality_failed"] is True
    assert summary["gates"]["ready_to_start_real_ab"] is False
    assert summary["gates"]["offline_semantic_suite_ready"] is False
    assert "unreadable_or_invalid_artifacts" in summary["blocking_reasons"]
    assert summary["next_action"] == "fix_or_rerun_unreadable_artifacts"


def test_legacy_preflight_suggests_real_local_judge_matrix_after_healthcheck(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-local-judge-matrix-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_goldset_chain_ready_summaries(tmp_path)
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["offline_semantic_suite_ready"] is True
    assert summary["gates"]["local_judge_config_ready"] is True
    assert summary["gates"]["local_judge_healthcheck_ready"] is True
    assert summary["gates"]["offline_local_judge_matrix_ready"] is False
    assert summary["gates"]["offline_local_judge_matrix_smoke_ready"] is False
    assert summary["gates"]["semantic_guard_fake_smoke_ready"] is False
    assert "local_judge_healthcheck_not_run" not in summary["blocking_reasons"]
    assert "offline_local_judge_matrix_not_run" in summary["blocking_reasons"]
    assert "offline_local_judge_matrix_smoke_not_run" in summary["blocking_reasons"]
    assert "semantic_guard_fake_smoke_not_run" in summary["blocking_reasons"]
    assert summary["next_action"] == "run_offline_local_judge_matrix"
    assert any("offline_semantic_matrix" in command for command in summary["suggested_commands"])


def test_legacy_preflight_suggests_fake_local_judge_matrix_smoke_after_real_matrix(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-local-judge-matrix-smoke-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_goldset_chain_ready_summaries(tmp_path)
    _write_offline_local_judge_matrix_summary(
        tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["offline_semantic_suite_ready"] is True
    assert summary["gates"]["local_judge_config_ready"] is True
    assert summary["gates"]["local_judge_healthcheck_ready"] is True
    assert summary["gates"]["offline_local_judge_matrix_ready"] is True
    assert summary["gates"]["offline_local_judge_matrix_smoke_ready"] is False
    assert summary["gates"]["semantic_guard_fake_smoke_ready"] is False
    assert "offline_local_judge_matrix_not_run" not in summary["blocking_reasons"]
    assert "offline_local_judge_matrix_smoke_not_run" in summary["blocking_reasons"]
    assert summary["next_action"] == "run_offline_local_judge_matrix_smoke"
    assert any("offline_local_judge_matrix_smoke" in command for command in summary["suggested_commands"])


def test_legacy_preflight_suggests_semantic_guard_fake_smoke_after_local_judge_matrix(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-semantic-guard-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json"
    )
    _write_goldset_chain_ready_summaries(tmp_path)
    _write_offline_local_judge_matrix_summary(
        tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    )
    _write_offline_local_judge_matrix_smoke_summary(
        tmp_path
        / "legacy"
        / "offline_local_judge_matrix_smoke"
        / "offline_local_judge_matrix_smoke_summary.json"
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    result = build_legacy_agbench_preflight(
        manifest_path=suite.manifest_path,
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
    )

    summary = result.summary
    assert summary["gates"]["ready_to_start_real_ab"] is True
    assert summary["gates"]["offline_semantic_suite_ready"] is True
    assert summary["gates"]["local_judge_config_ready"] is True
    assert summary["gates"]["local_judge_healthcheck_ready"] is True
    assert summary["gates"]["offline_local_judge_matrix_ready"] is True
    assert summary["gates"]["offline_local_judge_matrix_smoke_ready"] is True
    assert summary["gates"]["semantic_guard_fake_smoke_ready"] is False
    assert "offline_semantic_suite_not_run" not in summary["blocking_reasons"]
    assert "local_judge_healthcheck_not_run" not in summary["blocking_reasons"]
    assert "offline_local_judge_matrix_not_run" not in summary["blocking_reasons"]
    assert "offline_local_judge_matrix_smoke_not_run" not in summary["blocking_reasons"]
    assert "semantic_guard_fake_smoke_not_run" in summary["blocking_reasons"]
    assert summary["next_action"] == "run_semantic_guard_fake_smoke"
    assert any("semantic_guard_proxy_smoke" in command for command in summary["suggested_commands"])


def test_legacy_preflight_cli_require_real_results_returns_nonzero_when_missing(monkeypatch, tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="preflight-cli",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _patch_readiness(monkeypatch, ready=True, api_present=True)

    exit_code = main(["--manifest", suite.manifest_path, "--cwd", str(tmp_path), "--require-real-results"])

    assert exit_code == 1


def test_legacy_preflight_redacts_real_keys_without_rewriting_task_success_text() -> None:
    value = preflight._redact_sensitive("sk-secret-value and task-success gains")

    assert value == "sk-REDACTED and task-success gains"


def _patch_readiness(monkeypatch, *, ready: bool, api_present: bool) -> None:
    detail = "present: OPENAI_API_KEY" if api_present else "missing OPENAI_API_KEY/OAI_CONFIG_LIST"
    api_ok = api_present or ready
    report = {
        "ready": ready,
        "mode": "legacy-proxy",
        "checks": [
            {"name": "python_environment", "ok": True, "detail": "ok"},
            {"name": "docker", "ok": True, "detail": "ok"},
            {"name": "api_config", "ok": api_ok, "detail": detail},
            {"name": "autogenbench", "ok": True, "detail": "ok"},
            {"name": "autogen_core", "ok": True, "detail": "ok"},
        ],
        "suggested_commands": ["readiness-command"],
    }
    monkeypatch.setattr(preflight, "check_evaluation_readiness", lambda **kwargs: SimpleNamespace(to_dict=lambda: report))


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


def _write_text(path, text: str) -> None:
    target = __import__("pathlib").Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _write_provider(path, *, fake: bool) -> None:
    row = {
        "schema_version": "prefix-forward-proxy-provider-telemetry-v1",
        "request_id": "request-1",
        "actual_prompt_tokens": 100,
        "actual_cached_tokens": 10,
        "actual_completion_tokens": 10,
        "actual_total_tokens": 110,
        "latency_seconds": 1.0,
        "error": None,
    }
    if fake:
        row["fake_upstream"] = True
    _write_text(path, json.dumps(row) + "\n")


def _write_provider_with_shadow(path, *, rule_id: str, action: str) -> None:
    rows = [
        {
            "schema_version": "prefix-openai-request-adapter-telemetry-v1",
            "shadow_trial": {
                "schema_version": "prefix-shadow-trial-telemetry-v1",
                "matched_rule_count": 1,
                "unsupported_rule_count": 0,
                "trial_action_counts": {action: 1},
                "matched_rules": [{"trial_rule_id": rule_id, "shadow_action": action}],
                "validator_behavior_change_allowed": False,
            },
        },
        {
            "schema_version": "prefix-forward-proxy-provider-telemetry-v1",
            "request_id": f"request-{rule_id}",
            "actual_prompt_tokens": 100,
            "actual_cached_tokens": 10,
            "actual_completion_tokens": 10,
            "actual_total_tokens": 110,
            "latency_seconds": 1.0,
            "rewrite_applied": True,
            "error": None,
        },
    ]
    _write_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def _write_task_results(path) -> None:
    _write_text(path, json.dumps({"task_id": "HumanEval_0::trial_0", "success": True}) + "\n")


def _write_ab_summary(path, *, fake: bool) -> None:
    value = {
        "schema_version": "prefix-reorder-ab-eval-summary-v1",
        "real_provider_metrics_available": not fake,
        "task_metrics_available": True,
        "gates": {"overall_status": "pass"},
    }
    if fake:
        value["fake_upstream"] = True
    _write_text(path, json.dumps(value) + "\n")


def _write_semantic_guard_smoke_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-semantic-guard-proxy-smoke-summary-v1",
                "fake_upstream": True,
                "fake_local_judge": True,
                "real_provider_metrics_available": False,
                "real_semantic_model_metrics_available": False,
                "request_count": 2,
                "judge_request_count": 1,
                "upstream_request_count": 2,
                "adapter_validation_reasons": [
                    "no_rewrite_needed",
                    "semantic_guard_failed:private_boundary_uncertain",
                ],
            }
        )
        + "\n",
    )


def _write_offline_semantic_suite_summary(path, *, unsafe_review_promotion: bool = False) -> None:
    gate_items = [
        {
            "name": "review_candidate_promotion_policy",
            "status": "fail" if unsafe_review_promotion else "pass",
            "evidence": "review candidates are incorrectly allowed into automatic promotion"
            if unsafe_review_promotion
            else "review candidates remain upper-bound only",
        }
    ]
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-offline-semantic-suite-summary-v1",
                "prompt_safe_summary": True,
                "real_provider_metrics_available": False,
                "rule_only": {
                    "supported_request_count": 2,
                    "applied_count": 2,
                },
                "nl_segmentation": {
                    "supported_request_count": 2,
                    "applied_count": 3,
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
                    "review_queue_diagnostics": {
                        "schema_version": "prefix-review-queue-diagnostics-v1",
                        "review_candidate_count": 2,
                        "review_parent_block_count": 2,
                        "review_source_file_count": 1,
                        "review_candidate_chars": 123,
                        "semantic_hint_counts": {"procedure_policy": 1, "team_policy": 1},
                        "risk_tag_counts": {"agent_identity_boundary": 1},
                        "label_reason_counts": {"high_risk_boundary:agent_identity_boundary": 1},
                        "local_judge_priority": "high_risk_review_with_local_judge_and_static_clamps",
                    },
                    "extraction_counts": {"json_prompt_value": 2, "prompt_named_constant": 3},
                },
                "comparison": {
                    "delta": {"rule_gap_resolved": True},
                    "experiment_gate": {
                        "schema_version": "prefix-offline-experiment-gate-v1",
                        "performance_conclusion_allowed": False,
                        "recommendation": "offline coverage supports a small real A/B smoke",
                        "items": gate_items,
                    },
                },
            }
        )
        + "\n",
    )


def _write_offline_local_judge_matrix_smoke_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-offline-local-judge-matrix-smoke-summary-v1",
                "prompt_safe_summary": True,
                "fake_local_judge": True,
                "fake_local_judge_request_count": 12,
                "real_provider_metrics_available": False,
                "matrix_digest": {
                    "completed_source_count": 3,
                    "supported_source_count": 3,
                    "recommendation": "run_small_real_ab_smoke_keep_review_candidates_disabled",
                    "local_judge_budget_diagnostics": {
                        "budget_exhausted_source_count": 2,
                        "aggregate_budget_coverage_rate": 0.5,
                    },
                    "prompt_extraction_diagnostics": {
                        "extraction_counts": {
                            "json_prompt_value": 2,
                            "prompt_named_constant": 4,
                        },
                    },
                    "candidate_label_diagnostics": {
                        "rule_label_counts": {"accept": 6, "review": 6},
                        "model_label_counts": {"accept": 4},
                        "rule_to_model_label_counts": {"review->accept": 4},
                        "model_to_final_label_counts": {"accept->accept": 4},
                        "rule_to_final_label_counts": {
                            "accept->accept": 6,
                            "review->accept": 4,
                            "review->review": 2,
                        },
                        "static_safety_clamp_count": 0,
                        "local_judge_effectiveness_diagnostics": {
                            "schema_version": "prefix-matrix-local-judge-effectiveness-diagnostics-v1",
                            "source_with_effectiveness_diagnostics_count": 3,
                            "rule_review_candidate_count": 6,
                            "model_called_rule_review_count": 4,
                            "resolved_rule_review_count": 4,
                            "called_resolved_rule_review_count": 4,
                            "remaining_rule_review_count": 2,
                            "resolution_rate": 2 / 3,
                            "called_resolution_rate": 1.0,
                            "rule_review_final_label_counts": {"accept": 4, "review": 2},
                            "rule_review_model_label_counts": {"accept": 4},
                            "rule_review_local_judge_action_counts": {
                                "model_called": 4,
                                "skipped_by_budget": 2,
                            },
                        },
                    },
                },
            }
        )
        + "\n",
    )


def _write_offline_local_judge_matrix_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-offline-semantic-matrix-summary-v1",
                "prompt_safe_summary": True,
                "judge_mode": "openai-compatible",
                "include_candidate_text": True,
                "real_provider_metrics_available": False,
                "aggregate": {
                    "completed_source_count": 3,
                    "supported_source_count": 3,
                    "recommendation": "run_small_real_ab_smoke_keep_review_candidates_disabled",
                    "local_judge_action_counts": {
                        "model_called": 8,
                        "skipped_by_budget": 2,
                    },
                    "local_judge_budget_diagnostics": {
                        "budget_exhausted_source_count": 1,
                        "aggregate_budget_coverage_rate": 0.8,
                    },
                    "extraction_counts": {
                        "json_prompt_value": 6,
                        "prompt_named_constant": 12,
                    },
                    "rule_label_counts": {"accept": 10, "review": 8},
                    "model_label_counts": {"accept": 8},
                    "rule_to_model_label_counts": {"review->accept": 8},
                    "model_to_final_label_counts": {"accept->accept": 8},
                    "rule_to_final_label_counts": {
                        "accept->accept": 10,
                        "review->accept": 8,
                    },
                    "static_safety_clamp_count": 0,
                    "local_judge_effectiveness_diagnostics": {
                        "schema_version": "prefix-matrix-local-judge-effectiveness-diagnostics-v1",
                        "source_with_effectiveness_diagnostics_count": 3,
                        "rule_review_candidate_count": 8,
                        "model_called_rule_review_count": 8,
                        "resolved_rule_review_count": 8,
                        "called_resolved_rule_review_count": 8,
                        "remaining_rule_review_count": 0,
                        "resolution_rate": 1.0,
                        "called_resolution_rate": 1.0,
                        "rule_review_final_label_counts": {"accept": 8},
                        "rule_review_model_label_counts": {"accept": 8},
                        "rule_review_local_judge_action_counts": {"model_called": 8},
                    },
                },
            }
        )
        + "\n",
    )


def _write_local_judge_healthcheck_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-healthcheck-summary-v1",
                "prompt_safe_summary": True,
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "local-candidate-judge",
                "request_sent": True,
                "response_parse_ok": True,
                "allowed_label": True,
                "label": "accept",
                "confidence": 0.91,
                "confidence_meets_min": True,
                "synthetic_label_matches_expected": True,
                "ready": True,
                "recommendation": "run_small_local_judge_batch",
                "real_provider_metrics_available": False,
            }
        )
        + "\n",
    )


def _write_local_judge_quality_eval_summary(path, *, ready: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-quality-eval-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "gold.jsonl",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "local-candidate-judge",
                "api_key_configured": False,
                "timeout_seconds": 30.0,
                "min_confidence": 0.66,
                "min_samples": 20,
                "min_accuracy": 0.75,
                "min_macro_f1": 0.7,
                "sample_count": 20,
                "attempted_count": 20,
                "evaluated_count": 20,
                "model_called_count": 20,
                "invalid_gold_count": 0,
                "parse_error_count": 0,
                "low_confidence_count": 0,
                "correct_count": 18,
                "accuracy": 0.9,
                "macro_f1": 0.88,
                "raw_model_correct_count": 16,
                "raw_model_accuracy": 0.8,
                "raw_model_macro_f1": 0.78,
                "static_safety_clamp_count": 2,
                "static_safety_clamp_corrected_count": 2,
                "static_safety_clamp_worsened_count": 0,
                "static_safety_clamp_net_correct_delta": 2,
                "expected_label_counts": {"accept": 8, "reject": 6, "review": 6},
                "predicted_label_counts": {"accept": 8, "reject": 6, "review": 6},
                "raw_model_label_counts": {"accept": 10, "reject": 5, "review": 5},
                "raw_model_to_predicted_label_counts": {"accept->accept": 8, "accept->review": 2},
                "confusion_matrix": {},
                "per_label_metrics": {},
                "quality_diagnostics": {
                    "schema_version": "prefix-local-judge-quality-diagnostics-v1",
                    "prompt_safe_summary": True,
                    "expected_label_coverage_count": 3,
                    "predicted_label_coverage_count": 3,
                    "raw_model_label_coverage_count": 3,
                    "missing_predicted_labels": [],
                    "missing_raw_model_labels": [],
                    "zero_recall_labels": [],
                    "raw_model_zero_recall_labels": [],
                    "per_label_recall": {"accept": 1.0, "reject": 1.0, "review": 1.0},
                    "raw_model_per_label_recall": {"accept": 1.0, "reject": 0.83, "review": 0.83},
                    "predicted_label_collapse": False,
                    "raw_model_label_collapse": False,
                    "max_predicted_label_rate": 0.4,
                    "raw_model_max_label_rate": 0.5,
                    "accept_false_positive_count": 0,
                    "raw_model_accept_false_positive_count": 2,
                    "reject_recall": 1.0,
                    "raw_model_reject_recall": 0.83,
                    "review_recall": 1.0,
                    "raw_model_review_recall": 0.83,
                    "failure_mode_codes": [],
                },
                "failure_mode_codes": [],
                "ready": ready,
                "semantic_quality_metrics_available": True,
                "real_provider_metrics_available": False,
                "gold_text_written": False,
                "recommendation": "semantic_quality_claim_allowed_for_gold_set_only"
                if ready
                else "improve_or_recalibrate_local_judge_accuracy",
            }
        )
        + "\n",
    )


def _write_static_rule_calibration_eval_summary(path, *, ready: bool, exploratory_ready: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-static-rule-calibration-eval-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_labeled.local.jsonl",
                "row_count": 60,
                "min_accuracy": 0.75,
                "min_macro_f1": 0.7,
                "production_min_support": 2,
                "production_min_purity": 0.67,
                "expected_label_counts": {"accept": 22, "reject": 21, "review": 17},
                "policy_summaries": [
                    {
                        "policy_name": "rule_label",
                        "policy_family": "baseline",
                        "ready": False,
                        "accuracy": 0.4667,
                        "macro_f1": 0.4385,
                        "predicted_label_counts": {"accept": 28, "reject": 4, "review": 28},
                        "failure_mode_codes": ["accuracy_below_min", "macro_f1_below_min"],
                        "automation_policy": "baseline_reference",
                    }
                ],
                "top_policies": [
                    {
                        "policy_name": (
                            "metadata_calibration:semantic_hint+risk_tag_set:"
                            "support>=1:purity>=0.5:fallback=rule_label"
                        ),
                        "policy_family": "metadata_calibration",
                        "ready": True,
                        "production_candidate": False,
                        "automation_policy": "exploratory_only_low_support_or_purity",
                        "accuracy": 0.8333,
                        "macro_f1": 0.8192,
                        "predicted_label_counts": {"accept": 24, "reject": 21, "review": 15},
                        "failure_mode_codes": [],
                    }
                ],
                "best_ready_policy": (
                    "metadata_calibration:semantic_hint+risk_tag_set:"
                    "support>=1:purity>=0.5:fallback=rule_label"
                )
                if exploratory_ready
                else None,
                "best_production_candidate_policy": "metadata_calibration:semantic_hint+risk_tag_set:support>=2:purity>=0.67:fallback=review"
                if ready
                else None,
                "production_readiness_diagnostics": {
                    "schema_version": "prefix-static-rule-production-readiness-diagnostics-v1",
                    "prompt_safe_summary": True,
                    "status": "production_candidate_ready" if ready else "exploratory_ready_only",
                    "best_ready_policy": (
                        "metadata_calibration:semantic_hint+risk_tag_set:"
                        "support>=1:purity>=0.5:fallback=rule_label"
                    ),
                    "best_production_candidate_policy": "metadata_calibration:semantic_hint+risk_tag_set:support>=2:purity>=0.67:fallback=review"
                    if ready
                    else None,
                    "ready_nonproduction_policy_count": 1 if not ready else 0,
                    "production_min_support": 2,
                    "production_min_purity": 0.67,
                    "missing_threshold_reason_counts": {"min_support_below_production": 1}
                    if not ready
                    else {},
                    "top_ready_nonproduction_policies": [
                        {
                            "policy_name": (
                                "metadata_calibration:semantic_hint+risk_tag_set:"
                                "support>=1:purity>=0.5:fallback=rule_label"
                            ),
                            "automation_policy": "exploratory_only_low_support_or_purity",
                            "min_support": 1,
                            "min_purity": 0.5,
                            "missing_production_support": 1,
                            "missing_production_purity": 0.17,
                            "missing_threshold_reasons": [
                                "min_support_below_production",
                                "min_purity_below_production",
                            ],
                            "accuracy": 0.8333,
                            "macro_f1": 0.8192,
                        }
                    ],
                    "recommendation": "collect_more_gold_labels_or_raise_support_for_exploratory_metadata_rules",
                },
                "gold_support_diagnostics": {
                    "schema_version": "prefix-static-rule-gold-support-diagnostics-v1",
                    "prompt_safe_summary": True,
                    "feature_bucket_count": 3,
                    "support_ready_bucket_count": 1,
                    "purity_ready_bucket_count": 2,
                    "production_ready_bucket_count": 0,
                    "production_min_support": 2,
                    "production_min_purity": 0.67,
                    "feature_set_bucket_counts": {"semantic_hint+risk_tag_set": 3},
                    "priority_buckets": [
                        {
                            "feature_set": "semantic_hint+risk_tag_set",
                            "features": {
                                "semantic_hint": "tool_or_code_policy",
                                "risk_tag_set": "conditional_instruction",
                            },
                            "support": 1,
                            "expected_label_counts": {"review": 1},
                            "dominant_label": "review",
                            "purity": 1.0,
                            "support_ready": False,
                            "purity_ready": True,
                            "production_ready": False,
                            "tie_for_dominant_label": False,
                            "needed_additional_labels": 1,
                            "needed_for_support": 1,
                            "needed_for_purity": 0,
                            "purity_gap": 0.0,
                        }
                    ],
                    "recommendation": "add_gold_labels_to_priority_buckets_until_support>=2_and_purity>=0.67",
                },
                "validator_rule_candidates": {
                    "schema_version": "prefix-static-rule-validator-candidates-v1",
                    "prompt_safe_summary": True,
                    "status": "candidate_rules_available",
                    "candidate_rule_count": 1,
                    "safe_accept_rule_count": 0,
                    "reject_rule_count": 1,
                    "review_rule_count": 0,
                    "conflict_bucket_count": 1,
                    "production_min_support": 2,
                    "production_min_purity": 0.67,
                    "candidate_rules": [
                        {
                            "feature_set": "semantic_hint+risk_tag_set",
                            "features": {
                                "semantic_hint": "verification_policy",
                                "risk_tag_set": "agent_identity_boundary",
                            },
                            "label": "reject",
                            "validator_action": "force_reject",
                            "support": 2,
                            "purity": 1.0,
                            "expected_label_counts": {"reject": 2},
                            "promotion_reason": "meets_support_and_purity_metadata_thresholds",
                            "automation_policy": "review_before_validator_promotion",
                        }
                    ],
                    "conflict_buckets": [
                        {
                            "feature_set": "semantic_hint+risk_tag_set",
                            "features": {
                                "semantic_hint": "tool_or_code_policy",
                                "risk_tag_set": "conditional_instruction",
                            },
                            "support": 1,
                            "dominant_label": "review",
                            "purity": 1.0,
                            "expected_label_counts": {"review": 1},
                            "needed_additional_labels": 1,
                            "reason": "support_below_threshold",
                            "automation_policy": "manual_review_or_more_gold_labels",
                        }
                    ],
                    "overlap_diagnostics": {
                        "schema_version": "prefix-static-rule-candidate-overlap-diagnostics-v1",
                        "prompt_safe_summary": True,
                        "overlap_pair_count": 1,
                        "conflicting_overlap_pair_count": 1,
                        "same_action_overlap_pair_count": 0,
                        "conflicting_overlap_pairs": [
                            {
                                "relation": "left_subset_of_right",
                                "conflicting_actions": True,
                                "left_action": "force_reject",
                                "right_action": "allow_accept",
                                "left_feature_set": "risk_tag_set",
                                "right_feature_set": "semantic_hint+risk_tag_set",
                                "left_features": {"risk_tag_set": "conditional_instruction"},
                                "right_features": {
                                    "semantic_hint": "verification_policy",
                                    "risk_tag_set": "conditional_instruction",
                                },
                                "left_support": 4,
                                "right_support": 2,
                                "left_purity": 1.0,
                                "right_purity": 1.0,
                            }
                        ],
                    },
                    "fail_closed_simulation": {
                        "schema_version": "prefix-static-rule-validator-fail-closed-simulation-v1",
                        "prompt_safe_summary": True,
                        "policy_name": "validator_fail_closed_candidates:fallback=rule_label",
                        "evaluated_count": 3,
                        "candidate_rule_count": 1,
                        "matched_count": 2,
                        "matched_rate": 2 / 3,
                        "action_counts": {"force_reject": 2},
                        "accuracy": 2 / 3,
                        "macro_f1": 0.5,
                        "predicted_label_counts": {"reject": 2, "review": 1},
                        "mismatch_count": 1,
                        "mismatch_rate": 0.5,
                        "sample_mismatches": [
                            {
                                "input_index": 2,
                                "expected_label": "review",
                                "predicted_label": "reject",
                                "validator_action": "force_reject",
                                "candidate_feature_set": "semantic_hint+risk_tag_set",
                                "candidate_features": {
                                    "semantic_hint": "verification_policy",
                                    "risk_tag_set": "agent_identity_boundary",
                                },
                                "candidate_support": 2,
                                "candidate_purity": 1.0,
                            }
                        ],
                        "safe_for_automatic_validator_promotion": False,
                        "automation_policy": "diagnostic_only_until_manual_review_and_real_ab",
                        "recommendation": "inspect_fail_closed_candidate_mismatches_before_validator_changes",
                    },
                    "promotion_gate": {
                        "schema_version": "prefix-static-rule-validator-promotion-gate-v1",
                        "prompt_safe_summary": True,
                        "automatic_promotion_ready": False,
                        "manual_review_ready": True,
                        "fail_closed_candidate_count": 1,
                        "safe_accept_candidate_count": 0,
                        "blocking_reasons": [
                            "manual_code_review_required",
                            "real_provider_ab_required",
                            "gold_support_conflict_buckets_remain",
                            "overlapping_candidate_rules_have_conflicting_actions",
                        ],
                        "recommendation": "resolve_overlapping_candidate_actions_before_validator_changes",
                    },
                    "validator_review_plan": {
                        "schema_version": "prefix-static-rule-validator-review-plan-v1",
                        "prompt_safe_summary": True,
                        "review_item_count": 3,
                        "blocking_review_item_count": 2,
                        "priority_reason_counts": {
                            "conflicting_overlap_pair": 1,
                            "fail_closed_simulation_mismatch": 1,
                            "gold_support_conflict_bucket": 1,
                        },
                        "recommended_next_review_type": "conflicting_overlap_pair",
                        "review_items": [
                            {
                                "priority_rank": 1,
                                "priority_group": "p0_conflicting_candidate_overlap",
                                "review_type": "conflicting_overlap_pair",
                                "review_reason": "overlapping_candidate_rules_have_conflicting_actions",
                                "relation": "left_subset_of_right",
                                "left_action": "force_reject",
                                "right_action": "allow_accept",
                                "left_feature_set": "risk_tag_set",
                                "right_feature_set": "semantic_hint+risk_tag_set",
                                "left_features": {"risk_tag_set": "conditional_instruction"},
                                "right_features": {
                                    "semantic_hint": "verification_policy",
                                    "risk_tag_set": "conditional_instruction",
                                },
                                "left_support": 4,
                                "right_support": 2,
                                "left_purity": 1.0,
                                "right_purity": 1.0,
                            },
                            {
                                "priority_rank": 2,
                                "priority_group": "p1_fail_closed_mismatch",
                                "review_type": "fail_closed_simulation_mismatch",
                                "review_reason": "fail_closed_candidate_changes_gold_label",
                                "input_index": 2,
                                "expected_label": "review",
                                "predicted_label": "reject",
                                "validator_action": "force_reject",
                                "candidate_feature_set": "semantic_hint+risk_tag_set",
                                "candidate_features": {
                                    "semantic_hint": "verification_policy",
                                    "risk_tag_set": "agent_identity_boundary",
                                },
                                "candidate_support": 2,
                                "candidate_purity": 1.0,
                            },
                            {
                                "priority_rank": 3,
                                "priority_group": "p3_high_needed_label_conflict_bucket",
                                "review_type": "gold_support_conflict_bucket",
                                "review_reason": "needs_more_gold_support_or_rule_split",
                                "feature_set": "semantic_hint+risk_tag_set",
                                "features": {
                                    "semantic_hint": "tool_or_code_policy",
                                    "risk_tag_set": "conditional_instruction",
                                },
                                "support": 1,
                                "dominant_label": "review",
                                "purity": 1.0,
                                "expected_label_counts": {"review": 1},
                                "needed_additional_labels": 1,
                                "reason": "support_below_threshold",
                                "automation_policy": "manual_review_or_more_gold_labels",
                            },
                        ],
                        "automation_policy": "manual_review_only_before_validator_code_changes",
                        "recommendation": "resolve_conflicting_candidate_overlaps_first",
                    },
                    "conservative_fail_closed_subset": {
                        "schema_version": "prefix-static-rule-conservative-fail-closed-subset-v1",
                        "prompt_safe_summary": True,
                        "source_candidate_rule_count": 2,
                        "source_fail_closed_rule_count": 1,
                        "included_rule_count": 0,
                        "excluded_rule_count": 1,
                        "exclusion_reason_counts": {
                            "conflicting_overlap_pair": 1,
                            "fail_closed_simulation_mismatch": 1,
                        },
                        "included_candidate_rules": [],
                        "excluded_candidate_rules": [
                            {
                                "feature_set": "semantic_hint+risk_tag_set",
                                "features": {
                                    "semantic_hint": "verification_policy",
                                    "risk_tag_set": "agent_identity_boundary",
                                },
                                "label": "reject",
                                "validator_action": "force_reject",
                                "support": 2,
                                "purity": 1.0,
                                "expected_label_counts": {"reject": 2},
                                "promotion_reason": "meets_support_and_purity_metadata_thresholds",
                                "automation_policy": "review_before_validator_promotion",
                                "subset_policy": "conservative_fail_closed_only",
                                "subset_decision": "excluded",
                                "exclusion_reasons": [
                                    "conflicting_overlap_pair",
                                    "fail_closed_simulation_mismatch",
                                ],
                            }
                        ],
                        "simulation": {
                            "schema_version": "prefix-static-rule-validator-fail-closed-simulation-v1",
                            "prompt_safe_summary": True,
                            "policy_name": "validator_fail_closed_candidates:fallback=rule_label",
                            "evaluated_count": 3,
                            "candidate_rule_count": 0,
                            "matched_count": 0,
                            "matched_rate": 0.0,
                            "action_counts": {},
                            "accuracy": 1.0,
                            "macro_f1": 1.0,
                            "predicted_label_counts": {"accept": 1, "reject": 1, "review": 1},
                            "mismatch_count": 0,
                            "mismatch_rate": None,
                            "sample_mismatches": [],
                            "safe_for_automatic_validator_promotion": False,
                            "automation_policy": "diagnostic_only_until_manual_review_and_real_ab",
                            "recommendation": "no_fail_closed_candidate_coverage",
                        },
                        "manual_trial_ready": False,
                        "safe_for_automatic_validator_promotion": False,
                        "automation_policy": "manual_trial_only_until_code_review_and_real_ab",
                        "recommendation": "no_conservative_fail_closed_rules_after_exclusions",
                    },
                    "shadow_trial_plan": {
                        "schema_version": "prefix-static-rule-shadow-trial-plan-v1",
                        "prompt_safe_summary": True,
                        "trial_mode": "shadow_only",
                        "shadow_trial_ready": False,
                        "trial_rule_count": 0,
                        "trial_action_counts": {},
                        "offline_simulation_mismatch_count": 0,
                        "offline_simulation_matched_count": 0,
                        "offline_simulation_matched_rate": 0.0,
                        "source_diverse_rule_count": 0,
                        "weak_source_support_rule_count": 0,
                        "source_conflict_rule_count": 0,
                        "trial_rules": [],
                        "required_preconditions": [
                            "manual_code_review",
                            "real_provider_ab_with_shadow_telemetry",
                            "inspect_source_conflicts",
                            "keep_validator_behavior_unchanged",
                        ],
                        "telemetry_fields": [
                            "trial_rule_id",
                            "shadow_action",
                            "matched_candidate_count",
                            "fallback_reason",
                            "provider_cached_tokens",
                            "latency_ms",
                            "task_success",
                        ],
                        "validator_behavior_change_allowed": False,
                        "safe_for_automatic_validator_promotion": False,
                        "automation_policy": "shadow_only_no_behavior_change_until_manual_review_and_real_ab",
                        "recommendation": "resolve_conservative_subset_blockers_before_shadow_trial",
                    },
                    "source_generalization_diagnostics": {
                        "schema_version": "prefix-static-rule-source-generalization-diagnostics-v1",
                        "prompt_safe_summary": True,
                        "candidate_rule_count": 2,
                        "candidate_with_multi_source_group_count": 1,
                        "candidate_with_multi_source_file_count": 2,
                        "conservative_subset_rule_count": 0,
                        "conservative_subset_multi_source_group_count": 0,
                        "conservative_subset_multi_source_file_count": 0,
                        "weak_source_support_count": 1,
                        "source_conflict_count": 1,
                        "top_candidate_source_summaries": [
                            {
                                "feature_set": "semantic_hint+risk_tag_set",
                                "features": {
                                    "semantic_hint": "verification_policy",
                                    "risk_tag_set": "agent_identity_boundary",
                                },
                                "label": "reject",
                                "validator_action": "force_reject",
                                "support": 2,
                                "purity": 1.0,
                                "expected_label_counts": {"reject": 2},
                                "matched_count": 2,
                                "source_group_count": 1,
                                "source_file_hash_count": 2,
                                "source_group_counts": {"framework:autogen/package:pkg": 2},
                                "sample_source_path_hashes": ["abc123", "def456"],
                                "extraction_counts": {"prompt_named_constant": 2},
                                "expected_label_counts_by_source": {"reject": 2},
                                "source_label_counts_by_source": {"review": 2},
                                "expected_label_conflict": False,
                                "source_label_conflict": False,
                            }
                        ],
                        "weak_source_support_examples": [],
                        "source_conflict_examples": [],
                        "generalization_claim_allowed": False,
                        "automation_policy": "diagnostic_only_until_broader_cross_framework_gold_eval",
                        "recommendation": "collect_more_source_diverse_gold_labels_before_validator_rule_changes",
                    },
                    "recommendation": "use_candidates_for_fail_closed_reject_or_review_only",
                },
                "ready": ready,
                "exploratory_ready": exploratory_ready,
                "semantic_quality_claim_allowed": ready,
                "real_provider_metrics_available": False,
                "gold_text_written": False,
                "recommendation": "review_calibrated_metadata_rules_before_validator_promotion"
                if ready
                else "metadata_signal_exists_but_needs_more_gold_support_before_validator_promotion",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_template_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-template-summary-v1",
                "prompt_safe_summary": True,
                "input_paths": ["semantic_candidates_labeled.jsonl"],
                "output_path": "local_judge_goldset_template.jsonl",
                "include_text": False,
                "require_text": False,
                "gold_text_written": False,
                "real_provider_metrics_available": False,
                "max_rows": 60,
                "seed": 20260605,
                "strategy": "stratified",
                "filters": {"labels": [], "semantic_hints": [], "risk_tags": [], "source_contains": []},
                "input_row_count": 6,
                "filtered_row_count": 6,
                "selected_count": 3,
                "selected_with_text_count": 0,
                "selected_missing_text_count": 3,
                "expected_label_pending_count": 3,
                "expected_label_counts": {},
                "label_options": ["accept", "reject", "review"],
                "source_label_counts": {"accept": 1, "review": 2},
                "semantic_hint_counts": {"team_policy": 1, "verification_policy": 2},
                "source_schema_counts": {"prefix-semantic-candidate-v1": 3},
                "risk_tag_counts": {"agent_identity_boundary": 1},
                "text_artifact_note": "output JSONL is prompt-safe metadata only",
                "ready_for_local_judge_quality_eval": False,
                "manual_labeling_required": True,
                "recommendation": "rerun_with_include_text_for_human_labeling_then_fill_expected_label",
            }
        )
        + "\n",
    )


def _write_goldset_chain_ready_summaries(tmp_path) -> None:
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json"
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_labels_from_csv_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_labels_from_csv.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_labels_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_labels_validation.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_apply_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_apply.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_import_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_import.local.summary.json",
        ready=True,
    )
    _write_local_judge_goldset_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_validation_summary.json"
    )
    _write_local_judge_quality_eval_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_quality_eval_summary.json",
        ready=True,
    )
    _write_local_judge_goldset_chain_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_chain_verification_summary.json",
        ready=True,
    )


def _write_local_judge_goldset_progress_summary(path, *, ready: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-annotation-progress-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation.local.csv",
                "guide_path": "local_judge_goldset_annotation_guide.local.md",
                "worklist_path": "local_judge_goldset_annotation_worklist.local.jsonl",
                "row_count": 3,
                "rows_with_text_count": 3,
                "rows_missing_text_count": 0,
                "rows_with_expected_label_count": 3 if ready else 1,
                "expected_label_pending_count": 0 if ready else 2,
                "invalid_expected_label_count": 0,
                "expected_label_counts": {"accept": 1, "reject": 1, "review": 1} if ready else {"accept": 1},
                "source_label_counts": {"review": 3},
                "semantic_hint_counts": {"verification_policy": 3},
                "review_priority_counts": {"unknown": 3},
                "risk_tag_counts": {"unknown": 3},
                "pending_source_label_counts": {} if ready else {"review": 2},
                "pending_semantic_hint_counts": {} if ready else {"verification_policy": 2},
                "pending_review_priority_counts": {} if ready else {"unknown": 2},
                "pending_risk_tag_counts": {} if ready else {"unknown": 2},
                "invalid_source_label_counts": {},
                "invalid_semantic_hint_counts": {},
                "invalid_review_priority_counts": {},
                "invalid_risk_tag_counts": {},
                "annotation_completion_rate": 1.0 if ready else 1 / 3,
                "min_samples": 3,
                "min_label_count": 1,
                "labels_with_min_count": 3 if ready else 1,
                "required_label_count": 3,
                "label_balance_ready": ready,
                "label_options": ["accept", "reject", "review"],
                "ready_for_labeled_jsonl_import": ready,
                "ready_for_local_judge_goldset_validate_after_import": ready,
                "ready_for_local_judge_quality_eval_after_import": ready,
                "worklist_written": True,
                "worklist_row_count": 0 if ready else 2,
                "worklist_pending_count": 0 if ready else 2,
                "worklist_invalid_count": 0,
                "worklist_schema_version": "prefix-local-judge-goldset-annotation-worklist-item-v1",
                "text_artifact_note": "local CSV contains raw candidate text",
                "recommendation": "import_labeled_csv_then_validate_goldset"
                if ready
                else "fill_expected_label_for_pending_rows",
                "limits": "progress only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_labels_template_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-labels-template-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation_worklist.local.jsonl",
                "output_path": "local_judge_goldset_annotation_labels.local.jsonl",
                "output_written": True,
                "row_count": 3,
                "expected_label_pending_count": 3,
                "expected_label_counts": {},
                "rows_missing_id_count": 0,
                "rows_missing_text_hash_count": 0,
                "duplicate_label_key_count": 0,
                "label_status_counts": {"pending": 3},
                "source_label_counts": {"review": 3},
                "semantic_hint_counts": {"verification_policy": 3},
                "risk_tag_counts": {"unknown": 3},
                "ready_for_apply_labels": True,
                "recommendation": "fill_expected_label_values_then_run_apply_labels",
                "text_artifact_note": "labels template is prompt-safe and omits candidate text, source_path, and symbol",
                "limits": "labels template only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_suggestions_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-label-suggestions-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation.local.csv",
                "output_path": "local_judge_goldset_annotation_label_suggestions.local.jsonl",
                "output_written": True,
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "local-candidate-judge",
                "api_key_configured": False,
                "timeout_seconds": 90,
                "min_confidence": 0.66,
                "max_rows": None,
                "source_row_count": 3,
                "suggestion_count": 3,
                "model_called_count": 3,
                "local_judge_error_count": 0,
                "rows_missing_text_count": 0,
                "static_safety_clamp_count": 1,
                "suggested_label_counts": {"accept": 1, "reject": 1, "review": 1},
                "model_label_counts": {"accept": 2, "reject": 1},
                "rule_label_counts": {"review": 3},
                "source_label_counts": {"review": 3},
                "local_judge_action_counts": {"model_called": 3},
                "label_reason_code_counts": {"model_accept_clamped": 1, "model_label": 2},
                "model_to_suggested_label_counts": {"accept->accept": 1, "accept->review": 1, "reject->reject": 1},
                "rule_to_suggested_label_counts": {"review->accept": 1, "review->reject": 1, "review->review": 1},
                "source_to_suggested_label_counts": {"review->accept": 1, "review->reject": 1, "review->review": 1},
                "suggestions_ready_for_manual_review": True,
                "ready_for_apply_labels": False,
                "ready_for_goldset_import_after_apply": False,
                "text_artifact_note": "suggestions JSONL omits candidate text",
                "recommendation": "manually_review_suggestions_then_fill_expected_label_values",
                "limits": "suggestions only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_suggestion_review_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-suggestion-review-csv-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation.local.csv",
                "suggestions_path": "local_judge_goldset_annotation_label_suggestions.local.jsonl",
                "output_path": "local_judge_goldset_annotation_suggestion_review.local.csv",
                "output_written": True,
                "row_count": 3,
                "suggestion_row_count": 3,
                "matched_suggestion_count": 3,
                "missing_suggestion_count": 0,
                "unmatched_suggestion_count": 0,
                "duplicate_csv_key_count": 0,
                "duplicate_suggestion_key_count": 0,
                "suggestions_missing_key_count": 0,
                "expected_label_pending_count": 2,
                "expected_label_counts": {"review": 1},
                "suggested_label_counts": {"accept": 1, "reject": 1, "review": 1},
                "suggestion_matches_source_label_counts": {
                    "review->accept": 1,
                    "review->reject": 1,
                    "review->review": 1,
                },
                "ready_for_manual_review": True,
                "ready_for_apply_labels": False,
                "ready_for_goldset_import_after_apply": False,
                "csv_text_written": True,
                "text_artifact_note": "review CSV contains candidate text",
                "recommendation": "manually_fill_expected_label_using_review_csv",
                "limits": "suggestion review only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_review_plan_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-review-plan-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation_suggestion_review.local.csv",
                "output_path": "local_judge_goldset_annotation_review_plan.local.jsonl",
                "output_written": True,
                "max_rows": None,
                "row_count": 3,
                "plan_row_count": 2,
                "expected_label_pending_count": 2,
                "invalid_expected_label_count": 0,
                "source_label_counts": {"accept": 1, "review": 2},
                "rule_label_counts": {"accept": 1, "review": 2},
                "model_label_counts": {"accept": 2, "review": 1},
                "suggested_label_counts": {"accept": 1, "review": 2},
                "plan_expected_label_status_counts": {"pending": 2},
                "plan_review_reason_counts": {
                    "high_risk_boundary": 1,
                    "missing_expected_label": 2,
                    "source_rule_model_or_suggestion_disagree": 1,
                },
                "plan_high_risk_tag_counts": {"agent_identity_boundary": 1},
                "plan_semantic_hint_counts": {"team_policy": 1, "verification_policy": 1},
                "plan_disagreement_row_count": 1,
                "plan_high_risk_row_count": 1,
                "ready_for_manual_review": True,
                "ready_for_apply_labels": False,
                "ready_for_goldset_import_after_apply": False,
                "suggested_labels_not_auto_applied": True,
                "text_artifact_note": "review plan omits candidate text",
                "recommendation": "manually_fill_expected_label_using_prioritized_review_plan",
                "limits": "review plan only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_labels_from_csv_summary(path, *, ready: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-labels-from-csv-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation_suggestion_review.local.csv",
                "output_path": "local_judge_goldset_annotation_labels.local.jsonl",
                "output_written": True,
                "row_count": 3,
                "valid_label_row_count": 3 if ready else 1,
                "expected_label_pending_count": 0 if ready else 2,
                "invalid_label_row_count": 0,
                "rows_missing_id_count": 0,
                "rows_missing_text_hash_count": 0,
                "duplicate_label_key_count": 0,
                "expected_label_counts": {"accept": 1, "reject": 1, "review": 1} if ready else {"accept": 1},
                "min_samples": 3,
                "min_label_count": 1,
                "sample_count_ready": ready,
                "label_balance_ready": ready,
                "suggested_label_counts": {"accept": 1, "review": 2},
                "expected_to_suggested_label_counts": {"accept->accept": 1, "reject->review": 1, "review->review": 1}
                if ready
                else {"accept->accept": 1, "unknown->review": 2},
                "ready_for_apply_labels": ready,
                "ready_for_goldset_import_after_apply": ready,
                "csv_text_read": True,
                "labels_jsonl_text_written": False,
                "suggested_labels_not_auto_applied": True,
                "text_artifact_note": "output labels JSONL omits candidate text",
                "recommendation": "run_validate_labels_then_apply_labels" if ready else "fill_remaining_expected_label_values",
                "limits": "labels-from-csv only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_labels_validation_summary(path, *, ready: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-labels-validation-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation_labels.local.jsonl",
                "require_complete": True,
                "row_count": 3,
                "valid_label_row_count": 3 if ready else 1,
                "expected_label_pending_count": 0 if ready else 2,
                "invalid_label_row_count": 0,
                "rows_missing_id_count": 0,
                "rows_missing_text_hash_count": 0,
                "duplicate_label_key_count": 0,
                "expected_label_counts": {"accept": 1, "reject": 1, "review": 1} if ready else {"accept": 1},
                "source_label_counts": {"review": 3},
                "source_expected_label_match_count": 2 if ready else 1,
                "source_expected_label_mismatch_count": 1 if ready else 0,
                "source_expected_label_audited_count": 3 if ready else 1,
                "source_expected_label_match_rate": 2 / 3 if ready else 1.0,
                "source_expected_label_mismatch_counts": {"accept->reject": 1} if ready else {},
                "possible_rule_self_confirmation": False,
                "gold_labels_independent_from_rules_ready": ready,
                "semantic_hint_counts": {"verification_policy": 3},
                "risk_tag_counts": {"unknown": 3},
                "label_status_counts": {"pending": 3},
                "label_options": ["accept", "reject", "review"],
                "min_samples": 3,
                "min_label_count": 1,
                "labels_with_min_count": 3 if ready else 1,
                "structure_ready": True,
                "complete_labels_ready": ready,
                "sample_count_ready": ready,
                "label_balance_ready": ready,
                "ready_for_apply_labels": ready,
                "ready_for_goldset_import_after_apply": ready,
                "recommendation": "run_apply_labels_require_complete"
                if ready
                else "fill_remaining_expected_label_values",
                "text_artifact_note": "labels validation is prompt-safe and omits candidate text, source_path, and symbol",
                "limits": "labels validation only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_apply_summary(path, *, ready: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-label-apply-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation.local.csv",
                "labels_path": "local_judge_goldset_annotation_labels.local.jsonl",
                "output_path": "local_judge_goldset_annotation.labeled.local.csv",
                "require_complete": True,
                "output_written": ready,
                "row_count": 3,
                "rows_with_text_count": 3 if ready else 0,
                "rows_missing_text_count": 0 if ready else 3,
                "rows_with_expected_label_count": 3 if ready else 0,
                "expected_label_pending_count": 0 if ready else 3,
                "invalid_expected_label_count": 0,
                "expected_label_counts": {"accept": 1, "reject": 1, "review": 1} if ready else {},
                "ready_for_labeled_jsonl_import": ready,
                "label_row_count": 3,
                "valid_label_row_count": 3 if ready else 0,
                "pending_label_row_count": 0 if ready else 3,
                "invalid_label_row_count": 0,
                "missing_label_key_count": 0,
                "duplicate_label_key_count": 0,
                "unmatched_label_count": 0,
                "duplicate_csv_key_count": 0,
                "matched_label_count": 3 if ready else 0,
                "label_expected_counts": {"accept": 1, "reject": 1, "review": 1} if ready else {},
                "recommendation": "import_labeled_csv_then_validate_goldset"
                if ready
                else "fill_remaining_expected_label_values",
                "limits": "apply-labels only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_chain_summary(path, *, ready: bool) -> None:
    step_names = [
        "labels_from_csv",
        "labels_validation",
        "apply_labels",
        "import_goldset",
        "goldset_validation",
        "quality_eval",
    ]
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-chain-verification-v1",
                "prompt_safe_summary": True,
                "ready": ready,
                "chain_ready": ready,
                "next_action": "goldset_chain_ready_for_semantic_quality_claim"
                if ready
                else "run_apply_labels_require_complete",
                "steps": [
                    {
                        "name": name,
                        "exists": True,
                        "ready": ready,
                        "status": "pass" if ready else "warn",
                        "evidence": "ready" if ready else "not ready",
                    }
                    for name in step_names
                ],
                "checks": [
                    {"name": f"{name}:ready", "ok": ready, "detail": "ready" if ready else "not ready"}
                    for name in step_names
                ],
                "blocking_reasons": [] if ready else ["apply_labels:ready"],
                "allowed_claims": [],
                "prohibited_claims": [],
                "real_provider_metrics_available": False,
                "limits": "chain verifier only",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_validation_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-validation-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_labeled.local.jsonl",
                "min_samples": 3,
                "min_label_count": 1,
                "row_count": 3,
                "valid_row_count": 3,
                "invalid_row_count": 0,
                "missing_text_count": 0,
                "invalid_expected_label_count": 0,
                "expected_label_counts": {"accept": 1, "reject": 1, "review": 1},
                "error_counts": {},
                "labels_with_min_count": 3,
                "label_options": ["accept", "reject", "review"],
                "gold_text_written": False,
                "real_provider_metrics_available": False,
                "ready": True,
                "ready_for_local_judge_quality_eval": True,
                "recommendation": "run_local_judge_quality_eval",
            }
        )
        + "\n",
    )


def _write_local_judge_goldset_import_summary(path, *, ready: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-goldset-csv-import-summary-v1",
                "prompt_safe_summary": True,
                "input_path": "local_judge_goldset_annotation.labeled.local.csv",
                "output_path": "local_judge_goldset_labeled.local.jsonl",
                "output_written": ready,
                "allow_incomplete_output": False,
                "gold_text_written": True,
                "row_count": 3,
                "rows_with_text_count": 3,
                "rows_missing_text_count": 0,
                "expected_label_counts": {"accept": 1, "reject": 1, "review": 1} if ready else {},
                "expected_label_pending_count": 0 if ready else 3,
                "invalid_expected_label_count": 0,
                "label_options": ["accept", "reject", "review"],
                "ready_for_local_judge_goldset_validate": ready,
                "recommendation": "run_local_judge_goldset_validate"
                if ready
                else "fill_expected_label_in_csv_before_import",
                "limits": "local labels only",
            }
        )
        + "\n",
    )
