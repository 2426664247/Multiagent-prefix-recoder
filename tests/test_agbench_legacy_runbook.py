from __future__ import annotations

import json

from autogen_prefix_tree.agbench_legacy_runbook import build_legacy_agbench_runbook, main
from autogen_prefix_tree.agbench_legacy_suite import generate_legacy_agbench_suite


def test_legacy_runbook_is_prompt_safe_and_orders_real_run_steps(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        api_key_placeholder="${OPENAI_API_KEY}",
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        summary_path=tmp_path / "runbook.json",
        report_path=tmp_path / "runbook.md",
        env={},
    )

    summary = result.summary
    assert summary["schema_version"] == "prefix-legacy-agbench-runbook-v1"
    assert summary["structural_ready"] is True
    assert summary["api_config_present"] is False
    assert summary["ready_to_start_real_run"] is False
    assert summary["next_action"] == "set_api_config_before_real_run"
    assert [step["name"] for step in summary["steps"]] == [
        "verify_suite",
        "preflight_fake_smoke",
        "semantic_guard_fake_smoke",
        "offline_semantic_suite",
        "local_judge_healthcheck",
        "local_judge_goldset_template",
        "local_judge_goldset_validate",
        "local_judge_quality_eval",
        "local_judge_policy_eval",
        "static_rule_calibration_eval",
        "local_judge_goldset_chain_verify",
        "offline_local_judge_matrix",
        "offline_local_judge_matrix_smoke",
        "start_three_proxies",
        "run_autogenbench_variants",
        "tabulate_results",
        "collect_and_compare",
    ]
    assert len(summary["steps"][1]["commands"]) == 1
    assert "three_proxy_smoke" in summary["steps"][1]["commands"][0]
    assert len(summary["steps"][2]["commands"]) == 1
    assert "semantic_guard_proxy_smoke" in summary["steps"][2]["commands"][0]
    assert len(summary["steps"][3]["commands"]) == 1
    assert "offline_semantic_suite" in summary["steps"][3]["commands"][0]
    assert len(summary["steps"][4]["commands"]) == 1
    assert "local_judge_healthcheck" in summary["steps"][4]["commands"][0]
    assert len(summary["steps"][5]["commands"]) == 12
    assert "local_judge_goldset_builder" in summary["steps"][5]["commands"][0]
    assert "--include-text" in summary["steps"][5]["commands"][1]
    assert "local_judge_goldset_csv export" in summary["steps"][5]["commands"][2]
    assert "local_judge_goldset_csv progress" in summary["steps"][5]["commands"][3]
    assert "local_judge_goldset_csv labels-template" in summary["steps"][5]["commands"][4]
    assert "local_judge_goldset_csv suggest-labels" in summary["steps"][5]["commands"][5]
    assert "local_judge_goldset_csv merge-suggestions" in summary["steps"][5]["commands"][6]
    assert "local_judge_goldset_csv review-plan" in summary["steps"][5]["commands"][7]
    assert "local_judge_goldset_csv labels-from-csv" in summary["steps"][5]["commands"][8]
    assert "local_judge_goldset_csv validate-labels" in summary["steps"][5]["commands"][9]
    assert "local_judge_goldset_csv apply-labels" in summary["steps"][5]["commands"][10]
    assert "local_judge_goldset_csv import" in summary["steps"][5]["commands"][11]
    assert len(summary["steps"][6]["commands"]) == 1
    assert "local_judge_goldset_validate" in summary["steps"][6]["commands"][0]
    assert len(summary["steps"][7]["commands"]) == 1
    assert "local_judge_quality_eval" in summary["steps"][7]["commands"][0]
    assert len(summary["steps"][8]["commands"]) == 1
    assert "local_judge_policy_eval" in summary["steps"][8]["commands"][0]
    assert len(summary["steps"][9]["commands"]) == 1
    assert "static_rule_calibration_eval" in summary["steps"][9]["commands"][0]
    assert len(summary["steps"][10]["commands"]) == 1
    assert "local_judge_goldset_chain_verify" in summary["steps"][10]["commands"][0]
    assert len(summary["steps"][11]["commands"]) == 1
    assert "offline_semantic_matrix" in summary["steps"][11]["commands"][0]
    assert len(summary["steps"][12]["commands"]) == 1
    assert "offline_local_judge_matrix_smoke" in summary["steps"][12]["commands"][0]
    assert len(summary["steps"][13]["commands"]) == 3
    assert summary["steps"][13]["manual_long_running"] is True
    assert len(summary["steps"][14]["commands"]) == 3
    assert all("scenarios" in command for command in summary["steps"][14]["commands"])
    assert "sk-secret" not in json.dumps(summary, ensure_ascii=False)
    report = (tmp_path / "runbook.md").read_text(encoding="utf-8")
    assert "Legacy AutoGenBench A/B Runbook" in report
    assert "preflight_fake_smoke" in report
    assert "semantic_guard_fake_smoke" in report
    assert "offline_semantic_suite" in report
    assert "local_judge_healthcheck" in report
    assert "local_judge_goldset_template" in report
    assert "local_judge_goldset_validate" in report
    assert "local_judge_quality_eval" in report
    assert "local_judge_policy_eval" in report
    assert "static_rule_calibration_eval" in report
    assert "local_judge_goldset_chain_verify" in report
    assert "offline_local_judge_matrix" in report
    assert "offline_local_judge_matrix_smoke" in report
    assert "start_three_proxies" in report
    assert "API key" in report or "API" in report


def test_legacy_runbook_detects_collection_and_ab_report_readiness(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-ready",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = suite.manifest
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = manifest["oai_config_lists"][variant]
        _write_provider(info["provider_telemetry_path"], cached=10)
        _write_text(info["tabulate_csv_path"], "Task Id,Trial 0 Success\nHumanEval_0,True\n")
        _write_task_results(info["task_results_path"])
    for field in ("rule_only_summary", "nl_segmentation_summary"):
        _write_ab_summary(manifest["recommended_ab_eval"][field])
    for field in ("rule_only_report_md", "nl_segmentation_report_md"):
        _write_text(manifest["recommended_ab_eval"][field], "# Prefix Reorder A/B Summary\n\n## Gates\n")

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    serialized = json.dumps(summary, ensure_ascii=False)
    assert summary["ready_to_start_real_run"] is True
    assert summary["collection_inputs_ready"] is True
    assert summary["ab_reports_ready"] is True
    assert summary["next_action"] == "run_offline_semantic_suite"
    assert summary["artifact_quality_fail_count"] == 0
    assert all(item["status"] in {"pass", "pending"} for item in summary["artifact_quality"])
    assert "sk-secret-value" not in serialized


def test_legacy_runbook_summarizes_shadow_trial_wiring_from_plugin_provider_telemetry(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-shadow-trial",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = suite.manifest
    _write_provider(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"], cached=0)
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

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    status = summary["shadow_trial_wiring_status"]
    assert summary["shadow_trial_wiring_ready"] is True
    assert status["shadow_trial_supported_artifact_count"] == 2
    assert status["shadow_trial_record_with_match_count"] == 2
    assert status["shadow_trial_matched_rule_observation_count"] == 2
    assert status["shadow_trial_action_counts"] == {"force_reject": 1, "prefer_review": 1}
    baseline_item = next(
        item for item in summary["artifact_quality"] if item["name"] == "baseline:provider_telemetry_path"
    )
    assert baseline_item["shadow_trial_supported"] is False


def test_legacy_runbook_accepts_project_deepseek_config_without_printing_key(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-project-config",
        model="deepseek-v4-pro",
        upstream_base_url="https://api.deepseek.com/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_dir.joinpath("config.txt").write_text("deepseek=ds-secret-value\ndeepseek-v4-pro\n", encoding="utf-8")

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={},
        require_api_config=True,
    )

    serialized = json.dumps(result.summary, ensure_ascii=False)
    api_check = next(check for check in result.summary["checks"] if check["name"] == "api_config")
    assert result.summary["api_config_present"] is True
    assert "project DeepSeek config" in api_check["detail"]
    assert "ds-secret-value" not in serialized


def test_legacy_runbook_warns_on_semantic_guard_fake_smoke_without_blocking_real_provider_readiness(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-semantic-guard-smoke",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = suite.manifest
    smoke_summary_path = tmp_path / "legacy" / "preflight_semantic_guard_proxy_smoke" / "semantic_guard_proxy_smoke_summary.json"
    _write_semantic_guard_smoke_summary(smoke_summary_path)

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    guard_item = next(item for item in summary["artifact_quality"] if item["name"] == "semantic_guard_fake_smoke:summary")
    assert guard_item["status"] == "warn"
    assert guard_item["fake_upstream"] is True
    assert guard_item["fake_local_judge"] is True
    assert summary["semantic_guard_fake_smoke_ready"] is True
    assert summary["semantic_guard_fake_smoke_status"]["ready"] is True
    assert summary["provider_or_ab_fake_artifacts_detected"] is False
    assert summary["next_action"] == "run_offline_semantic_suite"
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "semantic_guard_fake_smoke" in report
    assert "adapter_validation_reasons" in report
    assert "semantic_guard_failed:private_boundary_uncertain" in report


def test_legacy_runbook_marks_offline_semantic_suite_ready_without_fake_provider_flag(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-offline-suite",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    summary_path = tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json"
    _write_offline_semantic_suite_summary(summary_path)

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    suite_item = next(item for item in summary["artifact_quality"] if item["name"] == "offline_semantic_suite:summary")
    assert suite_item["status"] == "pass"
    assert summary["offline_semantic_suite_ready"] is True
    assert summary["offline_semantic_suite_status"]["status"] == "pass"
    assert summary["offline_semantic_suite_status"]["rule_only_applied_count"] == 2
    assert summary["offline_semantic_suite_status"]["nl_segmentation_applied_count"] == 3
    assert summary["offline_semantic_suite_status"]["rule_supported_request_count"] == 2
    assert summary["offline_semantic_suite_status"]["nl_supported_request_count"] == 2
    assert summary["offline_semantic_suite_status"]["rule_only_candidate_count"] == 4
    assert summary["offline_semantic_suite_status"]["nl_segmentation_candidate_count"] == 5
    assert summary["offline_semantic_suite_status"]["nl_segmentation_total_estimated_gain_chars"] == 92
    assert summary["offline_semantic_suite_status"]["candidate_promotion_policy"][
        "automatic_promotion_allowed_labels"
    ] == ["accept"]
    assert summary["offline_semantic_suite_status"]["candidate_review_upper_bound_policy"][
        "review_candidates_used_for_upper_bound_only"
    ] is True
    assert summary["offline_semantic_suite_status"]["experiment_gate_items"] == [
        {
            "name": "review_candidate_promotion_policy",
            "status": "pass",
            "evidence": "review candidates remain upper-bound only",
        }
    ]
    assert summary["offline_semantic_suite_status"]["review_queue_diagnostics"] == {
        "schema_version": "prefix-review-queue-diagnostics-v1",
        "review_candidate_count": 2,
        "review_parent_block_count": 2,
        "review_source_file_count": 1,
        "review_candidate_chars": 123,
        "semantic_hint_counts": {"procedure_policy": 1, "team_policy": 1},
        "risk_tag_counts": {"agent_identity_boundary": 1},
        "label_reason_counts": {"high_risk_boundary:agent_identity_boundary": 1},
        "local_judge_priority": "high_risk_review_with_local_judge_and_static_clamps",
    }
    assert summary["offline_semantic_suite_status"]["prompt_extraction_diagnostics"] == {
        "extraction_counts": {"json_prompt_value": 2, "prompt_named_constant": 3}
    }
    report = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    ).report_path
    assert report is not None
    report_text = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "nl_segmentation_total_estimated_gain_chars" in report_text
    assert "rule_gap_resolved" in report_text
    assert "recommendation" in report_text
    assert "candidate_promotion_policy" in report_text
    assert "candidate_review_upper_bound_policy" in report_text
    assert "experiment_gate_items" in report_text
    assert "review_candidate_promotion_policy" in report_text
    assert "review_queue_risk_tag_counts" in report_text
    assert "agent_identity_boundary" in report_text
    assert "review_queue_local_judge_priority" in report_text
    assert summary["provider_or_ab_fake_artifacts_detected"] is False


def test_legacy_runbook_marks_example_local_judge_config_not_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-example-local-judge",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-small-model",
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    assert summary["local_judge_config_ready"] is False
    assert summary["local_judge_config_status"]["configured"] is False
    assert summary["local_judge_config_status"]["reason"] == (
        "local judge manifest marks configured=false"
    )


def test_legacy_runbook_blocks_failed_offline_semantic_suite_gate(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-offline-suite-failed-gate",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_offline_semantic_suite_summary(
        tmp_path / "legacy" / "offline_semantic_suite" / "offline_semantic_suite_summary.json",
        unsafe_review_promotion=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    suite_item = next(item for item in summary["artifact_quality"] if item["name"] == "offline_semantic_suite:summary")
    assert suite_item["status"] == "fail"
    assert suite_item["failed_experiment_gate_items"] == ["review_candidate_promotion_policy"]
    assert summary["offline_semantic_suite_ready"] is False
    assert summary["artifact_quality_fail_count"] == 1
    assert summary["next_action"] == "fix_or_rerun_unreadable_artifacts"


def test_legacy_runbook_marks_offline_local_judge_matrix_smoke_ready_without_fake_provider_flag(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-matrix",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    summary_path = (
        tmp_path
        / "legacy"
        / "offline_local_judge_matrix_smoke"
        / "offline_local_judge_matrix_smoke_summary.json"
    )
    _write_offline_local_judge_matrix_smoke_summary(summary_path)

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "offline_local_judge_matrix_smoke:summary"
    )
    assert item["status"] == "warn"
    assert item["fake_local_judge"] is True
    assert item["fake_local_judge_request_count"] == 12
    assert summary["offline_local_judge_matrix_smoke_ready"] is True
    assert summary["offline_local_judge_matrix_smoke_status"]["status"] == "warn"
    assert summary["offline_local_judge_matrix_smoke_status"]["completed_source_count"] == 3
    assert summary["offline_local_judge_matrix_smoke_status"]["budget_exhausted_source_count"] == 2
    assert summary["offline_local_judge_matrix_smoke_status"]["aggregate_budget_coverage_rate"] == 0.5
    assert summary["offline_local_judge_matrix_smoke_status"]["prompt_extraction_diagnostics"] == {
        "extraction_counts": {"json_prompt_value": 2, "prompt_named_constant": 4}
    }
    assert summary["offline_local_judge_matrix_smoke_status"]["candidate_label_diagnostics"] == {
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
    assert summary["provider_or_ab_fake_artifacts_detected"] is False


def test_legacy_runbook_marks_offline_local_judge_matrix_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-real-local-judge-matrix",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    summary_path = tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    _write_offline_local_judge_matrix_summary(summary_path, model_called=8)

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "offline_local_judge_matrix:summary")
    assert item["status"] == "pass"
    assert item["model_called_count"] == 8
    assert summary["offline_local_judge_matrix_ready"] is True
    assert summary["offline_local_judge_matrix_status"]["status"] == "pass"
    assert summary["offline_local_judge_matrix_status"]["completed_source_count"] == 3
    assert summary["offline_local_judge_matrix_status"]["model_called_count"] == 8
    assert summary["offline_local_judge_matrix_status"]["prompt_extraction_diagnostics"] == {
        "extraction_counts": {"json_prompt_value": 6, "prompt_named_constant": 12}
    }
    assert summary["offline_local_judge_matrix_status"]["candidate_label_diagnostics"] == {
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
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "Offline Evidence Diagnostics" in report
    assert "rule_review_candidate_count" in report
    assert "called_resolution_rate" in report
    assert "prompt_extraction_counts" in report
    assert summary["provider_or_ab_fake_artifacts_detected"] is False


def test_legacy_runbook_rejects_rule_only_matrix_as_local_judge_evidence(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-rule-only-matrix",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    summary_path = tmp_path / "legacy" / "offline_local_judge_matrix" / "offline_semantic_matrix_summary.json"
    _write_offline_local_judge_matrix_summary(summary_path, model_called=0, judge_mode="rule")

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "offline_local_judge_matrix:summary")
    assert item["status"] == "fail"
    assert "judge_mode=rule" in item["evidence"]
    assert summary["offline_local_judge_matrix_ready"] is False
    assert summary["next_action"] == "fix_or_rerun_unreadable_artifacts"


def test_legacy_runbook_marks_local_judge_healthcheck_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-healthcheck",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json",
        label="accept",
        ready=True,
        matches_expected=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_healthcheck:summary")
    assert item["status"] == "pass"
    assert item["healthcheck_ready"] is True
    assert item["label"] == "accept"
    assert summary["local_judge_healthcheck_ready"] is True
    assert summary["local_judge_healthcheck_status"]["status"] == "pass"
    assert summary["local_judge_healthcheck_status"]["model"] == "local-candidate-judge"
    assert summary["provider_or_ab_fake_artifacts_detected"] is False


def test_legacy_runbook_warns_when_local_judge_healthcheck_label_differs(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-healthcheck-warn",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json",
        label="review",
        ready=True,
        matches_expected=False,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_healthcheck:summary")
    assert item["status"] == "warn"
    assert item["healthcheck_ready"] is True
    assert item["synthetic_label_matches_expected"] is False
    assert summary["local_judge_healthcheck_ready"] is True
    assert summary["local_judge_healthcheck_status"]["status"] == "warn"


def test_legacy_runbook_marks_local_judge_quality_eval_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-quality",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_quality_eval_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_quality_eval_summary.json",
        ready=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_quality_eval:summary")
    assert item["status"] == "pass"
    assert item["quality_eval_ready"] is True
    assert item["accuracy"] == 0.9
    assert item["macro_f1"] == 0.88
    assert item["raw_model_accuracy"] == 0.8
    assert item["static_safety_clamp_count"] == 2
    assert item["static_safety_clamp_corrected_count"] == 2
    assert item["static_safety_clamp_net_correct_delta"] == 2
    assert item["quality_diagnostics"]["predicted_label_collapse"] is False
    assert item["quality_diagnostics"]["reject_recall"] == 1.0
    assert item["failure_mode_codes"] == []
    assert summary["local_judge_quality_eval_ready"] is True
    assert summary["local_judge_quality_eval_status"]["status"] == "pass"
    assert summary["local_judge_quality_eval_status"]["quality_diagnostics"][
        "raw_model_label_coverage_count"
    ] == 3
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_quality_eval" in report
    assert "macro_f1" in report
    assert "raw_model_accuracy" in report
    assert "static_safety_clamp_count" in report
    assert "predicted_label_collapse" in report
    assert "raw_model_reject_recall" in report


def test_legacy_runbook_marks_static_rule_calibration_eval_exploratory(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-static-calibration",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_static_rule_calibration_eval_summary(
        tmp_path / "legacy" / "artifacts" / "static_rule_calibration_eval_summary.json",
        ready=False,
        exploratory_ready=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "static_rule_calibration_eval:summary")
    assert item["status"] == "warn"
    assert item["exploratory_ready"] is True
    assert item["static_rule_calibration_eval_ready"] is False
    assert item["baseline_rule_accuracy"] == 0.4667
    assert item["baseline_rule_macro_f1"] == 0.4385
    assert item["best_ready_policy"] == "metadata_calibration:semantic_hint+risk_tag_set:support>=1:purity>=0.5:fallback=rule_label"
    assert item["best_production_candidate_policy"] is None
    assert summary["static_rule_calibration_eval_ready"] is False
    assert summary["static_rule_calibration_eval_status"]["exploratory_ready"] is True
    assert summary["static_rule_calibration_eval_status"]["top_policy_metric_table"][0]["macro_f1"] == 0.8192
    assert summary["static_rule_calibration_eval_status"]["production_readiness_diagnostics"][
        "missing_threshold_reason_counts"
    ] == {"min_support_below_production": 1}
    assert summary["static_rule_calibration_eval_status"]["gold_support_diagnostics"]["priority_buckets"][0][
        "needed_additional_labels"
    ] == 1
    assert summary["static_rule_calibration_eval_status"]["validator_rule_candidates"]["candidate_rule_count"] == 1
    assert summary["static_rule_calibration_eval_status"]["validator_rule_candidates"]["candidate_rules"][0][
        "validator_action"
    ] == "force_reject"
    assert summary["static_rule_calibration_eval_status"]["validator_rule_candidates"]["conflict_buckets"][0][
        "needed_additional_labels"
    ] == 1
    gate = summary["static_rule_calibration_eval_status"]["validator_rule_candidates"]["promotion_gate"]
    assert gate["automatic_promotion_ready"] is False
    assert "overlapping_candidate_rules_have_conflicting_actions" in gate["blocking_reasons"]
    overlap = summary["static_rule_calibration_eval_status"]["validator_rule_candidates"]["overlap_diagnostics"]
    assert overlap["conflicting_overlap_pair_count"] == 1
    simulation = summary["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "fail_closed_simulation"
    ]
    assert simulation["matched_count"] == 2
    assert simulation["mismatch_count"] == 1
    assert simulation["safe_for_automatic_validator_promotion"] is False
    review_plan = summary["static_rule_calibration_eval_status"]["validator_rule_candidates"]["validator_review_plan"]
    assert review_plan["prompt_safe_summary"] is True
    assert review_plan["review_item_count"] == 3
    assert review_plan["recommended_next_review_type"] == "conflicting_overlap_pair"
    assert review_plan["review_items"][0]["review_reason"] == "overlapping_candidate_rules_have_conflicting_actions"
    subset = summary["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "conservative_fail_closed_subset"
    ]
    assert subset["prompt_safe_summary"] is True
    assert subset["included_rule_count"] == 0
    assert subset["excluded_rule_count"] == 1
    assert subset["simulation"]["candidate_rule_count"] == 0
    assert subset["safe_for_automatic_validator_promotion"] is False
    shadow_trial = summary["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "shadow_trial_plan"
    ]
    assert shadow_trial["prompt_safe_summary"] is True
    assert shadow_trial["trial_mode"] == "shadow_only"
    assert shadow_trial["shadow_trial_ready"] is False
    assert shadow_trial["trial_rule_count"] == 0
    assert shadow_trial["validator_behavior_change_allowed"] is False
    source_generalization = summary["static_rule_calibration_eval_status"]["validator_rule_candidates"][
        "source_generalization_diagnostics"
    ]
    assert source_generalization["prompt_safe_summary"] is True
    assert source_generalization["candidate_with_multi_source_file_count"] == 2
    assert source_generalization["generalization_claim_allowed"] is False
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "static_rule_calibration_eval" in report
    assert "baseline_rule_accuracy" in report
    assert "production_readiness_diagnostics" in report
    assert "gold_support_diagnostics" in report
    assert "validator_rule_candidates" in report
    assert "validator_review_plan" in report
    assert "conservative_fail_closed_subset" in report
    assert "shadow_trial_plan" in report
    assert "source_generalization_diagnostics" in report
    assert "conflicting_overlap_pair" in report
    assert "force_reject" in report
    assert "automatic_promotion_ready" in report
    assert "overlapping_candidate_rules_have_conflicting_actions" in report
    assert "fail_closed_simulation" in report
    assert "diagnostic_only_until_manual_review_and_real_ab" in report
    assert "needed_additional_labels" in report
    assert "metadata_signal_exists_but_needs_more_gold_support_before_validator_promotion" in report


def test_legacy_runbook_marks_local_judge_goldset_chain_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-chain",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_chain_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_chain_verification_summary.json",
        ready=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_goldset_chain:summary")
    assert item["status"] == "pass"
    assert item["goldset_chain_ready"] is True
    assert item["step_ready_counts"] == {"ready": 6, "total": 6}
    assert summary["local_judge_goldset_chain_ready"] is True
    assert summary["local_judge_goldset_chain_status"]["status"] == "pass"
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_chain" in report
    assert "step_ready_counts" in report


def test_legacy_runbook_synthesizes_goldset_chain_verify_command_for_old_manifest(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-old-chain-command",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
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

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    step = next(step for step in result.summary["steps"] if step["name"] == "local_judge_goldset_chain_verify")
    assert step["command_count"] == 1
    assert "local_judge_goldset_chain_verify" in step["commands"][0]
    assert str(manifest_path) in step["commands"][0]
    assert "artifacts\\local_judge_goldset_chain_verification_summary.json" in step["commands"][0]
    assert "reports\\local_judge_goldset_chain_verification.md" in step["commands"][0]
    item = next(
        item for item in result.summary["artifact_quality"] if item["name"] == "local_judge_goldset_chain:summary"
    )
    assert item["status"] == "warn"
    assert item["goldset_chain_ready"] is False
    assert item["blocking_reasons"] == ["apply_labels:ready"]


def test_legacy_runbook_marks_local_judge_goldset_template_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json",
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_goldset_template:summary")
    assert item["status"] == "pass"
    assert item["goldset_template_ready"] is True
    assert item["selected_count"] == 3
    assert item["expected_label_pending_count"] == 3
    assert summary["local_judge_goldset_template_ready"] is True
    assert summary["local_judge_goldset_template_status"]["status"] == "pass"
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_template" in report
    assert "expected_label_pending_count" in report


def test_legacy_runbook_marks_local_judge_goldset_validation_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-validation",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_validation_summary.json",
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_goldset_validation:summary")
    assert item["status"] == "pass"
    assert item["goldset_validation_ready"] is True
    assert item["valid_row_count"] == 3
    assert summary["local_judge_goldset_validation_ready"] is True
    assert summary["local_judge_goldset_validation_status"]["status"] == "pass"
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_validation" in report
    assert "valid_row_count" in report


def test_legacy_runbook_marks_local_judge_goldset_import_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-import",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_import_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_import.local.summary.json",
        ready=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_goldset_import:summary")
    assert item["status"] == "pass"
    assert item["goldset_import_ready"] is True
    assert item["expected_label_pending_count"] == 0
    assert item["output_written"] is True
    assert summary["local_judge_goldset_import_ready"] is True
    assert summary["local_judge_goldset_import_status"]["status"] == "pass"
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_import" in report
    assert "ready_for_validate" in report


def test_legacy_runbook_marks_local_judge_goldset_progress(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-progress",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=False,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_goldset_progress:summary")
    assert item["status"] == "warn"
    assert item["goldset_progress_ready"] is False
    assert item["expected_label_pending_count"] == 2
    assert summary["local_judge_goldset_progress_ready"] is True
    assert summary["local_judge_goldset_annotation_ready"] is False
    assert summary["local_judge_goldset_progress_status"]["status"] == "warn"
    assert summary["local_judge_goldset_progress_status"]["annotation_ready"] is False
    assert summary["local_judge_goldset_progress_status"]["ready_for_labeled_jsonl_import"] is False
    assert summary["local_judge_goldset_progress_status"]["pending_semantic_hint_counts"] == {
        "verification_policy": 2
    }
    assert summary["local_judge_goldset_progress_status"]["worklist_written"] is True
    assert summary["local_judge_goldset_progress_status"]["worklist_row_count"] == 2
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_progress" in report
    assert "annotation_completion_rate" in report
    assert "pending_semantic_hint_counts" in report
    assert "worklist_row_count" in report


def test_legacy_runbook_marks_local_judge_goldset_labels_template(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-labels-template",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_labels_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_labels.local.summary.json"
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "local_judge_goldset_labels_template:summary"
    )
    assert item["status"] == "pass"
    assert item["goldset_labels_template_ready"] is True
    assert item["row_count"] == 3
    assert item["expected_label_pending_count"] == 3
    assert item["duplicate_label_key_count"] == 0
    assert summary["local_judge_goldset_labels_template_ready"] is True
    assert summary["local_judge_goldset_labels_template_status"]["ready_for_apply_labels"] is True
    assert summary["local_judge_goldset_labels_template_status"]["label_status_counts"] == {"pending": 3}
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_labels_template" in report
    assert "ready_for_apply_labels" in report
    assert "duplicate_label_key_count" in report


def test_legacy_runbook_marks_local_judge_goldset_suggestions(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-suggestions",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_local_judge_goldset_suggestions_summary(
        tmp_path
        / "legacy"
        / "artifacts"
        / "local_judge_goldset_annotation_label_suggestions.local.summary.json"
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "local_judge_goldset_suggestions:summary"
    )
    assert item["status"] == "pass"
    assert item["goldset_suggestions_ready"] is True
    assert item["suggestion_count"] == 3
    assert item["model_called_count"] == 3
    assert item["static_safety_clamp_count"] == 1
    assert item["ready_for_apply_labels"] is False
    assert summary["local_judge_goldset_suggestions_ready"] is True
    assert summary["local_judge_goldset_suggestions_status"]["suggestions_ready_for_manual_review"] is True
    assert summary["local_judge_goldset_suggestions_status"]["ready_for_apply_labels"] is False
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_suggestions" in report
    assert "suggestions_ready_for_manual_review" in report
    assert "static_safety_clamp_count" in report


def test_legacy_runbook_marks_local_judge_goldset_suggestion_review(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-suggestion-review",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_local_judge_goldset_suggestion_review_summary(
        tmp_path
        / "legacy"
        / "artifacts"
        / "local_judge_goldset_annotation_suggestion_review.local.summary.json"
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "local_judge_goldset_suggestion_review:summary"
    )
    assert item["status"] == "pass"
    assert item["goldset_suggestion_review_ready"] is True
    assert item["row_count"] == 3
    assert item["matched_suggestion_count"] == 3
    assert item["expected_label_pending_count"] == 2
    assert item["ready_for_apply_labels"] is False
    assert summary["local_judge_goldset_suggestion_review_ready"] is True
    assert summary["local_judge_goldset_suggestion_review_status"]["ready_for_manual_review"] is True
    assert summary["local_judge_goldset_suggestion_review_status"]["ready_for_apply_labels"] is False
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_suggestion_review" in report
    assert "ready_for_manual_review" in report
    assert "matched_suggestion_count" in report


def test_legacy_runbook_marks_local_judge_goldset_review_plan(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-review-plan",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_local_judge_goldset_review_plan_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_review_plan.local.summary.json"
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "local_judge_goldset_review_plan:summary"
    )
    assert item["status"] == "pass"
    assert item["goldset_review_plan_ready"] is True
    assert item["plan_row_count"] == 2
    assert item["plan_disagreement_row_count"] == 1
    assert item["suggested_labels_not_auto_applied"] is True
    assert summary["local_judge_goldset_review_plan_ready"] is True
    assert summary["local_judge_goldset_review_plan_status"]["ready"] is True
    assert summary["local_judge_goldset_review_plan_status"]["plan_high_risk_row_count"] == 1
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_review_plan" in report
    assert "plan_disagreement_row_count" in report
    assert "suggested_labels_not_auto_applied" in report


def test_legacy_runbook_marks_completed_local_judge_goldset_review_plan(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-review-plan-complete",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_local_judge_goldset_review_plan_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_review_plan.local.summary.json",
        annotation_complete=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    assert summary["local_judge_goldset_review_plan_ready"] is True
    assert summary["local_judge_goldset_review_plan_status"]["annotation_complete"] is True
    assert summary["local_judge_goldset_review_plan_status"]["plan_row_count"] == 0
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "local_judge_goldset_review_plan:summary"
    )
    assert item["status"] == "pass"
    assert item["evidence"] == "prompt-safe review plan is complete; no pending expected_label rows remain"


def test_legacy_runbook_marks_local_judge_goldset_labels_from_csv(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-labels-from-csv",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
    )
    _write_local_judge_goldset_labels_from_csv_summary(
        tmp_path
        / "legacy"
        / "artifacts"
        / "local_judge_goldset_annotation_labels_from_csv.local.summary.json",
        ready=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "local_judge_goldset_labels_from_csv:summary"
    )
    assert item["status"] == "pass"
    assert item["goldset_labels_from_csv_ready"] is True
    assert item["valid_label_row_count"] == 3
    assert item["ready_for_apply_labels"] is True
    assert item["suggested_labels_not_auto_applied"] is True
    assert summary["local_judge_goldset_labels_from_csv_ready"] is True
    assert summary["local_judge_goldset_labels_from_csv_status"]["expected_label_counts"] == {
        "accept": 1,
        "reject": 1,
        "review": 1,
    }
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_labels_from_csv" in report
    assert "suggested_labels_not_auto_applied" in report


def test_legacy_runbook_marks_local_judge_goldset_labels_validation(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-labels-validation",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_labels_validation_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_labels_validation.local.summary.json",
        ready=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        report_path=tmp_path / "legacy" / "reports" / "legacy_runbook.md",
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(
        item
        for item in summary["artifact_quality"]
        if item["name"] == "local_judge_goldset_labels_validation:summary"
    )
    assert item["status"] == "pass"
    assert item["goldset_labels_validation_ready"] is True
    assert summary["local_judge_goldset_labels_validation_ready"] is True
    assert summary["local_judge_goldset_labels_validation_status"]["ready_for_apply_labels"] is True
    assert summary["local_judge_goldset_labels_validation_status"]["ready_for_goldset_import_after_apply"] is True
    assert summary["local_judge_goldset_labels_validation_status"]["source_expected_label_match_count"] == 2
    assert summary["local_judge_goldset_labels_validation_status"]["source_expected_label_mismatch_count"] == 1
    assert summary["local_judge_goldset_labels_validation_status"]["source_expected_label_mismatch_counts"] == {
        "accept->reject": 1
    }
    assert summary["local_judge_goldset_labels_validation_status"]["possible_rule_self_confirmation"] is False
    assert summary["local_judge_goldset_labels_validation_status"]["gold_labels_independent_from_rules_ready"] is True
    report = (tmp_path / "legacy" / "reports" / "legacy_runbook.md").read_text(encoding="utf-8")
    assert "local_judge_goldset_labels_validation" in report
    assert "ready_for_goldset_import_after_apply" in report
    assert "source_expected_label_mismatch_count" in report


def test_legacy_runbook_prioritizes_annotation_before_goldset_import(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-annotation-before-import",
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
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json",
        label="accept",
        ready=True,
        matches_expected=True,
    )
    _write_local_judge_goldset_template_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_template_summary.json"
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=False,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    assert summary["offline_semantic_suite_ready"] is True
    assert summary["local_judge_config_ready"] is True
    assert summary["local_judge_healthcheck_ready"] is True
    assert summary["local_judge_goldset_template_ready"] is True
    assert summary["local_judge_goldset_annotation_ready"] is False
    assert summary["local_judge_goldset_import_ready"] is False
    assert summary["next_action"] == "complete_local_judge_goldset_annotation"


def test_legacy_runbook_marks_local_judge_goldset_annotation_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-annotation-ready",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_progress_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_progress.local.summary.json",
        ready=True,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    assert summary["local_judge_goldset_progress_ready"] is True
    assert summary["local_judge_goldset_annotation_ready"] is True
    assert summary["local_judge_goldset_progress_status"]["annotation_ready"] is True
    assert summary["local_judge_goldset_progress_status"]["ready_for_labeled_jsonl_import"] is True


def test_legacy_runbook_warns_when_goldset_csv_import_has_pending_labels(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-goldset-import-pending",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_goldset_import_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_goldset_annotation_import.local.summary.json",
        ready=False,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_goldset_import:summary")
    assert item["status"] == "warn"
    assert item["goldset_import_ready"] is False
    assert item["expected_label_pending_count"] == 3
    assert item["output_written"] is False
    assert summary["local_judge_goldset_import_ready"] is False
    assert summary["next_action"] == "run_offline_semantic_suite"


def test_legacy_runbook_fails_when_local_judge_healthcheck_not_ready(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-local-judge-healthcheck-fail",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    _write_local_judge_healthcheck_summary(
        tmp_path / "legacy" / "artifacts" / "local_judge_healthcheck_summary.json",
        label="maybe",
        ready=False,
        matches_expected=False,
    )

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    item = next(item for item in summary["artifact_quality"] if item["name"] == "local_judge_healthcheck:summary")
    assert item["status"] == "fail"
    assert item["healthcheck_ready"] is False
    assert summary["local_judge_healthcheck_ready"] is False
    assert summary["artifact_quality_fail_count"] == 1
    assert summary["next_action"] == "fix_or_rerun_unreadable_artifacts"


def test_legacy_runbook_detects_unreadable_real_run_artifacts(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-bad-artifacts",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = suite.manifest
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = manifest["oai_config_lists"][variant]
        _write_text(info["provider_telemetry_path"], "{}\n")
        _write_text(info["tabulate_csv_path"], "Task Id,Trial 0 Success\nHumanEval_0,True\n")
        _write_task_results(info["task_results_path"])
    for field in ("rule_only_summary", "nl_segmentation_summary"):
        _write_ab_summary(manifest["recommended_ab_eval"][field])
    for field in ("rule_only_report_md", "nl_segmentation_report_md"):
        _write_text(manifest["recommended_ab_eval"][field], "# Prefix Reorder A/B Summary\n\n## Gates\n")

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    assert summary["collection_inputs_ready"] is True
    assert summary["ab_reports_ready"] is True
    assert summary["next_action"] == "fix_or_rerun_unreadable_artifacts"
    assert summary["artifact_quality_fail_count"] == 3
    failed = [item for item in summary["artifact_quality"] if item["status"] == "fail"]
    assert all(item["name"].endswith(":provider_telemetry_path") for item in failed)


def test_legacy_runbook_warns_on_fake_smoke_artifacts(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-fake-artifacts",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = suite.manifest
    for variant in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = manifest["oai_config_lists"][variant]
        _write_provider(info["provider_telemetry_path"], cached=10)
        _append_fake_flag(info["provider_telemetry_path"])
        _write_text(info["tabulate_csv_path"], "Task Id,Trial 0 Success\nHumanEval_0,True\n")
        _write_task_results(info["task_results_path"])
    for field in ("rule_only_summary", "nl_segmentation_summary"):
        _write_ab_summary(manifest["recommended_ab_eval"][field])
        _mark_fake_ab_summary(manifest["recommended_ab_eval"][field])
    for field in ("rule_only_report_md", "nl_segmentation_report_md"):
        _write_text(manifest["recommended_ab_eval"][field], "# Prefix Reorder A/B Summary\n\n## Gates\n")

    result = build_legacy_agbench_runbook(
        manifest_path=suite.manifest_path,
        env={"OPENAI_API_KEY": "sk-secret-value"},
        require_api_config=True,
    )

    summary = result.summary
    assert summary["collection_inputs_ready"] is True
    assert summary["ab_reports_ready"] is True
    assert summary["artifact_quality_fail_count"] == 0
    assert summary["artifact_quality_warn_count"] >= 1
    assert summary["real_provider_metrics_available"] is False
    assert summary["next_action"] == "run_offline_semantic_suite"
    assert any(item.get("fake_upstream") for item in summary["artifact_quality"])


def test_legacy_runbook_cli_require_api_config_returns_nonzero_when_missing(tmp_path) -> None:
    suite = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-runbook-cli",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )

    exit_code = main(["--manifest", suite.manifest_path, "--require-api-config"])

    assert exit_code == 1


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


def _write_provider(path, *, cached: int) -> None:
    row = {
        "schema_version": "prefix-forward-proxy-provider-telemetry-v1",
        "request_id": f"request-{cached}",
        "actual_prompt_tokens": 100,
        "actual_cached_tokens": cached,
        "actual_completion_tokens": 10,
        "actual_total_tokens": 110,
        "latency_seconds": 1.0,
        "error": None,
    }
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


def _write_ab_summary(path) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-reorder-ab-eval-summary-v1",
                "real_provider_metrics_available": True,
                "task_metrics_available": True,
                "gates": {"overall_status": "pass"},
            }
        )
        + "\n",
    )


def _append_fake_flag(path) -> None:
    target = __import__("pathlib").Path(path)
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["fake_upstream"] = True
        rows.append(row)
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _mark_fake_ab_summary(path) -> None:
    target = __import__("pathlib").Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    value["fake_upstream"] = True
    value["real_provider_metrics_available"] = False
    target.write_text(json.dumps(value) + "\n", encoding="utf-8")


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
                    "candidate_count": 4,
                    "label_counts": {"accept": 3, "review": 1},
                    "total_estimated_gain_chars": 20,
                },
                "nl_segmentation": {
                    "supported_request_count": 2,
                    "applied_count": 3,
                    "candidate_count": 5,
                    "label_counts": {"accept": 3, "review": 2},
                    "total_estimated_gain_chars": 92,
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


def _write_offline_local_judge_matrix_summary(path, *, model_called: int, judge_mode: str = "openai-compatible") -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-offline-semantic-matrix-summary-v1",
                "prompt_safe_summary": True,
                "judge_mode": judge_mode,
                "include_candidate_text": True,
                "real_provider_metrics_available": False,
                "aggregate": {
                    "completed_source_count": 3,
                    "supported_source_count": 3,
                    "recommendation": "run_small_real_ab_smoke_keep_review_candidates_disabled",
                    "local_judge_action_counts": {
                        "model_called": model_called,
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
                    "model_label_counts": {"accept": model_called},
                    "rule_to_model_label_counts": {"review->accept": model_called},
                    "model_to_final_label_counts": {"accept->accept": model_called},
                    "rule_to_final_label_counts": {
                        "accept->accept": 10,
                        "review->accept": model_called,
                    },
                    "static_safety_clamp_count": 0,
                    "local_judge_effectiveness_diagnostics": {
                        "schema_version": "prefix-matrix-local-judge-effectiveness-diagnostics-v1",
                        "source_with_effectiveness_diagnostics_count": 3,
                        "rule_review_candidate_count": model_called,
                        "model_called_rule_review_count": model_called,
                        "resolved_rule_review_count": model_called,
                        "called_resolved_rule_review_count": model_called,
                        "remaining_rule_review_count": 0,
                        "resolution_rate": 1.0 if model_called else None,
                        "called_resolution_rate": 1.0 if model_called else None,
                        "rule_review_final_label_counts": {"accept": model_called} if model_called else {},
                        "rule_review_model_label_counts": {"accept": model_called} if model_called else {},
                        "rule_review_local_judge_action_counts": {"model_called": model_called}
                        if model_called
                        else {},
                    },
                },
            }
        )
        + "\n",
    )


def _write_local_judge_healthcheck_summary(path, *, label: str, ready: bool, matches_expected: bool) -> None:
    _write_text(
        path,
        json.dumps(
            {
                "schema_version": "prefix-local-judge-healthcheck-summary-v1",
                "prompt_safe_summary": True,
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "local-candidate-judge",
                "request_sent": ready,
                "response_parse_ok": ready,
                "allowed_label": label in {"accept", "review", "reject"},
                "label": label,
                "confidence": 0.91 if ready else None,
                "confidence_meets_min": ready,
                "synthetic_label_matches_expected": matches_expected,
                "ready": ready,
                "error_reason": None if ready else "invalid_model_label",
                "recommendation": "run_small_local_judge_batch"
                if matches_expected
                else "inspect_synthetic_label_before_large_batch",
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
                "next_action": "goldset_chain_ready_for_semantic_quality_claim"
                if ready
                else "run_apply_labels_require_complete",
                "blocking_reasons": [] if ready else ["apply_labels:ready"],
                "real_provider_metrics_available": False,
                "limits": "chain verifier only",
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


def _write_local_judge_goldset_review_plan_summary(path, *, annotation_complete: bool = False) -> None:
    plan_row_count = 0 if annotation_complete else 2
    pending_count = 0 if annotation_complete else 2
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
                "plan_row_count": plan_row_count,
                "expected_label_pending_count": pending_count,
                "invalid_expected_label_count": 0,
                "source_label_counts": {"accept": 1, "review": 2},
                "rule_label_counts": {"accept": 1, "review": 2},
                "model_label_counts": {"accept": 2, "review": 1},
                "suggested_label_counts": {"accept": 1, "review": 2},
                "plan_expected_label_status_counts": {} if annotation_complete else {"pending": 2},
                "plan_review_reason_counts": {}
                if annotation_complete
                else {
                    "high_risk_boundary": 1,
                    "missing_expected_label": 2,
                    "source_rule_model_or_suggestion_disagree": 1,
                },
                "plan_high_risk_tag_counts": {} if annotation_complete else {"agent_identity_boundary": 1},
                "plan_semantic_hint_counts": {}
                if annotation_complete
                else {"team_policy": 1, "verification_policy": 1},
                "plan_disagreement_row_count": 0 if annotation_complete else 1,
                "plan_high_risk_row_count": 0 if annotation_complete else 1,
                "ready_for_manual_review": not annotation_complete,
                "annotation_complete": annotation_complete,
                "ready_for_apply_labels": False,
                "ready_for_goldset_import_after_apply": False,
                "suggested_labels_not_auto_applied": True,
                "text_artifact_note": "review plan omits candidate text",
                "recommendation": "review_plan_complete_run_labels_from_csv"
                if annotation_complete
                else "manually_fill_expected_label_using_prioritized_review_plan",
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
