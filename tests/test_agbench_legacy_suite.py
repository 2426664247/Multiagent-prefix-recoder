from __future__ import annotations

import json

from autogen import config_list_from_json

from autogen_prefix_tree.agbench_legacy_suite import generate_legacy_agbench_suite, main
from autogen_prefix_tree.agbench_legacy_suite_verify import main as verify_main
from autogen_prefix_tree.agbench_legacy_suite_verify import verify_legacy_agbench_suite


def test_generate_legacy_agbench_suite_writes_oai_config_lists(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-smoke",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        baseline_proxy_base_url="http://127.0.0.1:8787/v1",
        rule_proxy_base_url="http://127.0.0.1:8788/v1",
        nl_proxy_base_url="http://127.0.0.1:8789/v1",
        extra_config={"temperature": 0, "max_tokens": 16},
    )

    manifest = result.manifest
    baseline = config_list_from_json(manifest["oai_config_lists"]["baseline"]["path"])
    rule = config_list_from_json(manifest["oai_config_lists"]["plugin_rule_only"]["path"])
    nl = config_list_from_json(manifest["oai_config_lists"]["plugin_nl_segmentation"]["path"])
    combined = config_list_from_json(manifest["oai_config_lists"]["combined"]["path"])

    assert baseline[0]["model"] == "gpt-test"
    assert baseline[0]["base_url"] == "http://127.0.0.1:8787/v1"
    assert baseline[0]["api_key"] == "${OPENAI_API_KEY}"
    assert baseline[0]["temperature"] == 0
    assert "baseline" in baseline[0]["tags"]
    assert rule[0]["base_url"] == "http://127.0.0.1:8788/v1"
    assert "plugin_rule_only" in rule[0]["tags"]
    assert nl[0]["base_url"] == "http://127.0.0.1:8789/v1"
    assert "plugin_nl_segmentation" in nl[0]["tags"]
    assert len(combined) == 3
    assert manifest["real_provider_metrics_available"] is False
    assert manifest["proxies"]["baseline"]["rewrite_enabled"] is False
    assert manifest["proxies"]["plugin_rule_only"]["rewrite_enabled"] is True
    assert manifest["proxies"]["plugin_rule_only"]["natural_language_segmentation_enabled"] is False
    assert manifest["proxies"]["plugin_nl_segmentation"]["rewrite_enabled"] is True
    assert manifest["proxies"]["plugin_nl_segmentation"]["natural_language_segmentation_enabled"] is True
    assert manifest["scenario_files"]["baseline"]["result_dir_name"] == "legacy-smoke.baseline"
    assert manifest["scenario_files"]["plugin_rule_only"]["result_dir_name"] == "legacy-smoke.plugin_rule_only"
    assert manifest["scenario_files"]["plugin_nl_segmentation"]["result_dir_name"] == "legacy-smoke.plugin_nl_segmentation"
    assert len({item["path"] for item in manifest["scenario_files"].values()}) == 3
    scenario_row = json.loads(next(iter(open(manifest["scenario_files"]["baseline"]["path"], encoding="utf-8"))))
    assert scenario_row["template"].endswith("template.py")
    assert ":" in scenario_row["template"] or scenario_row["template"].startswith("/")


def test_generate_legacy_agbench_suite_manifest_is_prompt_safe(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-safe",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        api_key_placeholder="${MY_API_KEY}",
    )

    serialized = json.dumps(result.manifest, ensure_ascii=False)
    assert result.manifest["prompt_safe_manifest"] is True
    assert result.manifest["secret_policy"]["api_key_value_not_written"] is True
    assert "sk-" not in serialized
    assert "Bearer " not in serialized
    assert "${MY_API_KEY}" in serialized


def test_generate_legacy_agbench_suite_can_include_guard_proxy_command(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-guard",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        include_guard=True,
        semantic_guard_base_url="http://127.0.0.1:11434/v1",
        semantic_guard_model="local-semantic-judge",
    )

    baseline_proxy_command, rule_proxy_command, nl_proxy_command = result.manifest["suggested_commands"]["proxy"]
    assert "--disabled" in baseline_proxy_command
    assert "--enable-natural-language-segmentation" not in rule_proxy_command
    assert "--enable-natural-language-segmentation" in nl_proxy_command
    assert "--shadow-trial-plan" not in baseline_proxy_command
    assert "--shadow-trial-plan" in rule_proxy_command
    assert "--shadow-trial-plan" in nl_proxy_command
    assert "static_rule_calibration_eval_summary.json" in rule_proxy_command
    assert "--semantic-guard-base-url http://127.0.0.1:11434/v1" in rule_proxy_command
    assert "--semantic-guard-model local-semantic-judge" in nl_proxy_command
    assert result.manifest["proxies"]["plugin_rule_only"]["semantic_guard_enabled"] is True
    assert result.manifest["proxies"]["plugin_nl_segmentation"]["semantic_guard_enabled"] is True


def test_generate_legacy_agbench_suite_adds_project_deepseek_proxy_auth_for_deepseek_upstream(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-deepseek",
        model="deepseek-v4-pro",
        upstream_base_url="https://api.deepseek.com/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )

    commands = result.manifest["suggested_commands"]["proxy"]
    assert all("--use-project-deepseek-config" in command for command in commands)
    assert all("--reset-telemetry" in command for command in commands)
    serialized = json.dumps(result.manifest, ensure_ascii=False)
    assert "ds-secret" not in serialized


def test_generate_legacy_agbench_suite_can_include_local_judge_healthcheck_config(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-local-judge",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
        local_judge_base_url="http://127.0.0.1:11434/v1",
        local_judge_model="local-candidate-judge",
        local_judge_api_key_env="LOCAL_JUDGE_API_KEY",
    )

    manifest = result.manifest
    command = manifest["suggested_commands"]["local_judge_healthcheck"][0]
    assert manifest["local_judge"]["configured"] is True
    assert manifest["local_judge"]["base_url"] == "http://127.0.0.1:11434/v1"
    assert manifest["local_judge"]["model"] == "local-candidate-judge"
    assert manifest["local_judge"]["api_key_env"] == "LOCAL_JUDGE_API_KEY"
    assert "--base-url http://127.0.0.1:11434/v1" in command
    assert "--model local-candidate-judge" in command
    assert "--api-key-env LOCAL_JUDGE_API_KEY" in command
    assert "local_judge_healthcheck_summary.json" in command
    goldset_commands = manifest["suggested_commands"]["local_judge_goldset_template"]
    assert len(goldset_commands) == 12
    assert "local_judge_goldset_builder" in goldset_commands[0]
    assert "semantic_candidates_labeled.jsonl" in goldset_commands[0]
    assert "local_judge_goldset_template_summary.json" in goldset_commands[0]
    assert "--include-text" in goldset_commands[1]
    assert "--require-text" in goldset_commands[1]
    assert "--source-prompts" in goldset_commands[1]
    assert "source_prompts.jsonl" in goldset_commands[1]
    assert "local_judge_goldset_csv export" in goldset_commands[2]
    assert "local_judge_goldset_annotation.local.csv" in goldset_commands[2]
    assert "local_judge_goldset_csv progress" in goldset_commands[3]
    assert "local_judge_goldset_annotation_progress.local.summary.json" in goldset_commands[3]
    assert "local_judge_goldset_annotation_guide.local.md" in goldset_commands[3]
    assert "local_judge_goldset_annotation_worklist.local.jsonl" in goldset_commands[3]
    assert "local_judge_goldset_csv labels-template" in goldset_commands[4]
    assert "local_judge_goldset_annotation_labels.local.jsonl" in goldset_commands[4]
    assert "local_judge_goldset_csv suggest-labels" in goldset_commands[5]
    assert "local_judge_goldset_annotation.local.csv" in goldset_commands[5]
    assert "local_judge_goldset_annotation_label_suggestions.local.jsonl" in goldset_commands[5]
    assert "local_judge_goldset_annotation_label_suggestions.local.summary.json" in goldset_commands[5]
    assert "--base-url http://127.0.0.1:11434/v1" in goldset_commands[5]
    assert "--model local-candidate-judge" in goldset_commands[5]
    assert "--timeout 90" in goldset_commands[5]
    assert "--api-key-env LOCAL_JUDGE_API_KEY" in goldset_commands[5]
    assert "local_judge_goldset_csv merge-suggestions" in goldset_commands[6]
    assert "local_judge_goldset_annotation_label_suggestions.local.jsonl" in goldset_commands[6]
    assert "local_judge_goldset_annotation_suggestion_review.local.csv" in goldset_commands[6]
    assert "local_judge_goldset_annotation_suggestion_review.local.summary.json" in goldset_commands[6]
    assert "local_judge_goldset_csv review-plan" in goldset_commands[7]
    assert "local_judge_goldset_annotation_suggestion_review.local.csv" in goldset_commands[7]
    assert "local_judge_goldset_annotation_review_plan.local.jsonl" in goldset_commands[7]
    assert "local_judge_goldset_annotation_review_plan.local.summary.json" in goldset_commands[7]
    assert "local_judge_goldset_csv labels-from-csv" in goldset_commands[8]
    assert "local_judge_goldset_annotation_suggestion_review.local.csv" in goldset_commands[8]
    assert "local_judge_goldset_annotation_labels.local.jsonl" in goldset_commands[8]
    assert "local_judge_goldset_annotation_labels_from_csv.local.summary.json" in goldset_commands[8]
    assert "local_judge_goldset_csv validate-labels" in goldset_commands[9]
    assert "local_judge_goldset_annotation_labels.local.jsonl" in goldset_commands[9]
    assert "--require-complete" in goldset_commands[9]
    assert "local_judge_goldset_csv apply-labels" in goldset_commands[10]
    assert "local_judge_goldset_annotation_labels.local.jsonl" in goldset_commands[10]
    assert "local_judge_goldset_annotation.labeled.local.csv" in goldset_commands[10]
    assert "local_judge_goldset_csv import" in goldset_commands[11]
    assert "local_judge_goldset_annotation.labeled.local.csv" in goldset_commands[11]
    quality_command = manifest["suggested_commands"]["local_judge_quality_eval"][0]
    validate_command = manifest["suggested_commands"]["local_judge_goldset_validate"][0]
    assert "local_judge_goldset_validate" in validate_command
    assert "local_judge_goldset_labeled.local.jsonl" in validate_command
    assert "local_judge_goldset_validation_summary.json" in validate_command
    chain_verify_command = manifest["suggested_commands"]["local_judge_goldset_chain_verify"][0]
    static_calibration_command = manifest["suggested_commands"]["static_rule_calibration_eval"][0]
    assert "local_judge_goldset_chain_verify" in chain_verify_command
    assert "--manifest" in chain_verify_command
    assert "local_judge_goldset_chain_verification_summary.json" in chain_verify_command
    assert "local_judge_goldset_chain_verification.md" in chain_verify_command
    assert "static_rule_calibration_eval" in static_calibration_command
    assert "local_judge_goldset_labeled.local.jsonl" in static_calibration_command
    assert "--production-min-support 2" in static_calibration_command
    assert "--production-min-purity 0.67" in static_calibration_command
    assert "static_rule_calibration_eval_summary.json" in static_calibration_command
    assert "static_rule_calibration_eval.md" in static_calibration_command
    assert "local_judge_quality_eval" in quality_command
    assert "--input" in quality_command
    assert "local_judge_goldset_labeled.local.jsonl" in quality_command
    assert "--base-url http://127.0.0.1:11434/v1" in quality_command
    assert "--model local-candidate-judge" in quality_command
    assert "--api-key-env LOCAL_JUDGE_API_KEY" in quality_command
    assert "local_judge_quality_eval_summary.json" in quality_command
    assert manifest["local_judge"]["goldset_template_path"].endswith("local_judge_goldset_template.jsonl")
    assert manifest["local_judge"]["goldset_template_summary_path"].endswith("local_judge_goldset_template_summary.json")
    assert manifest["local_judge"]["goldset_annotation_csv_path"].endswith("local_judge_goldset_annotation.local.csv")
    assert manifest["local_judge"]["goldset_annotation_export_summary_path"].endswith(
        "local_judge_goldset_annotation_export.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_progress_summary_path"].endswith(
        "local_judge_goldset_annotation_progress.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_guide_path"].endswith(
        "local_judge_goldset_annotation_guide.local.md"
    )
    assert manifest["local_judge"]["goldset_annotation_worklist_path"].endswith(
        "local_judge_goldset_annotation_worklist.local.jsonl"
    )
    assert manifest["local_judge"]["goldset_annotation_labels_path"].endswith(
        "local_judge_goldset_annotation_labels.local.jsonl"
    )
    assert manifest["local_judge"]["goldset_annotation_labels_summary_path"].endswith(
        "local_judge_goldset_annotation_labels.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_labels_from_csv_summary_path"].endswith(
        "local_judge_goldset_annotation_labels_from_csv.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_suggestions_path"].endswith(
        "local_judge_goldset_annotation_label_suggestions.local.jsonl"
    )
    assert manifest["local_judge"]["goldset_annotation_suggestions_summary_path"].endswith(
        "local_judge_goldset_annotation_label_suggestions.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_suggestion_review_csv_path"].endswith(
        "local_judge_goldset_annotation_suggestion_review.local.csv"
    )
    assert manifest["local_judge"]["goldset_annotation_suggestion_review_summary_path"].endswith(
        "local_judge_goldset_annotation_suggestion_review.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_review_plan_path"].endswith(
        "local_judge_goldset_annotation_review_plan.local.jsonl"
    )
    assert manifest["local_judge"]["goldset_annotation_review_plan_summary_path"].endswith(
        "local_judge_goldset_annotation_review_plan.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_labels_validation_summary_path"].endswith(
        "local_judge_goldset_annotation_labels_validation.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_labeled_csv_path"].endswith(
        "local_judge_goldset_annotation.labeled.local.csv"
    )
    assert manifest["local_judge"]["goldset_annotation_apply_summary_path"].endswith(
        "local_judge_goldset_annotation_apply.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_annotation_import_summary_path"].endswith(
        "local_judge_goldset_annotation_import.local.summary.json"
    )
    assert manifest["local_judge"]["goldset_labeled_path"].endswith("local_judge_goldset_labeled.local.jsonl")
    assert manifest["local_judge"]["goldset_validation_summary_path"].endswith("local_judge_goldset_validation_summary.json")
    assert manifest["local_judge"]["quality_eval_summary_path"].endswith("local_judge_quality_eval_summary.json")
    assert manifest["local_judge"]["static_rule_calibration_eval_summary_path"].endswith(
        "static_rule_calibration_eval_summary.json"
    )
    assert manifest["local_judge"]["static_rule_calibration_eval_report_path"].endswith(
        "static_rule_calibration_eval.md"
    )
    assert manifest["local_judge"]["shadow_trial_plan_path"].endswith(
        "static_rule_calibration_eval_summary.json"
    )
    assert manifest["local_judge"]["goldset_chain_verification_summary_path"].endswith(
        "local_judge_goldset_chain_verification_summary.json"
    )
    assert manifest["local_judge"]["goldset_chain_verification_report_path"].endswith(
        "local_judge_goldset_chain_verification.md"
    )


def test_generate_legacy_agbench_suite_marks_missing_local_judge_unconfigured(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-local-judge-missing",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )

    manifest = result.manifest
    assert manifest["local_judge"]["configured"] is False
    assert manifest["local_judge"]["base_url"] is None
    assert manifest["local_judge"]["model"] is None


def test_verify_legacy_agbench_suite_accepts_generated_suite(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-verify",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )

    verified = verify_legacy_agbench_suite(manifest_path=result.manifest_path)

    assert verified.summary["ready"] is True
    assert verified.summary["schema_version"] == "prefix-legacy-agbench-suite-verification-v1"
    assert _check(verified.summary, "oai_config_list:baseline:load")["ok"] is True
    assert _check(verified.summary, "oai_config_list:plugin_rule_only:load")["ok"] is True
    assert _check(verified.summary, "oai_config_list:plugin_nl_segmentation:load")["ok"] is True
    assert _check(verified.summary, "oai_config_list:combined:entry_count")["ok"] is True
    assert _check(verified.summary, "manifest:nl_segmentation_enabled")["ok"] is True
    assert _check(verified.summary, "scenario_files:result_dirs_distinct")["ok"] is True
    assert _check(verified.summary, "scenario_file:baseline:template_paths_rewritten_absolute")["ok"] is True


def test_verify_legacy_agbench_suite_reports_missing_required_artifacts(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-artifacts",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )

    verified = verify_legacy_agbench_suite(
        manifest_path=result.manifest_path,
        require_provider_artifacts=True,
        require_task_artifacts=True,
    )

    assert verified.summary["ready"] is False
    failed = {check["name"] for check in verified.summary["checks"] if not check["ok"]}
    assert "artifact:baseline:provider_telemetry_path" in failed
    assert "artifact:plugin_rule_only:task_results_path" in failed
    assert "artifact:plugin_nl_segmentation:task_results_path" in failed


def test_agbench_legacy_suite_cli_writes_manifest(tmp_path) -> None:
    output_dir = tmp_path / "legacy"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--suite-id",
            "legacy-cli",
            "--model",
            "gpt-test",
            "--upstream-base-url",
            "https://upstream.example/v1",
            "--task-jsonl",
            str(_task_jsonl(tmp_path)),
            "--temperature",
            "0",
            "--max-tokens",
            "32",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "legacy_suite_manifest.json").read_text(encoding="utf-8"))
    baseline = config_list_from_json(manifest["oai_config_lists"]["baseline"]["path"])
    assert baseline[0]["temperature"] == 0
    assert baseline[0]["max_tokens"] == 32


def test_agbench_legacy_suite_cli_output_verifies_with_three_proxy_commands(tmp_path) -> None:
    output_dir = tmp_path / "legacy"
    summary_path = output_dir / "reports" / "verify_summary.json"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--suite-id",
            "legacy-cli-verify",
            "--model",
            "gpt-test",
            "--upstream-base-url",
            "https://upstream.example/v1",
            "--task-jsonl",
            str(_task_jsonl(tmp_path)),
        ]
    )
    verify_exit_code = verify_main(
        [
            "--manifest",
            str(output_dir / "legacy_suite_manifest.json"),
            "--summary",
            str(summary_path),
        ]
    )

    manifest = json.loads((output_dir / "legacy_suite_manifest.json").read_text(encoding="utf-8"))
    proxy_commands = manifest["suggested_commands"]["proxy"]
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert verify_exit_code == 0
    assert len(proxy_commands) == 3
    assert "--disabled" in proxy_commands[0]
    assert all("--reset-telemetry" in command for command in proxy_commands)
    assert "--enable-natural-language-segmentation" not in proxy_commands[1]
    assert "--enable-natural-language-segmentation" in proxy_commands[2]
    assert len(manifest["suggested_commands"]["tabulate"]) == 3
    assert "three_proxy_smoke" in manifest["suggested_commands"]["preflight_fake_smoke"][0]
    assert "--manifest" in manifest["suggested_commands"]["preflight_fake_smoke"][0]
    assert "semantic_guard_proxy_smoke" in manifest["suggested_commands"]["semantic_guard_fake_smoke"][0]
    assert "--judge-mode reject" in manifest["suggested_commands"]["semantic_guard_fake_smoke"][0]
    assert "offline_semantic_suite" in manifest["suggested_commands"]["offline_semantic_suite"][0]
    assert "--source-root <autogen-source-root>" in manifest["suggested_commands"]["offline_semantic_suite"][0]
    assert "local_judge_healthcheck" in manifest["suggested_commands"]["local_judge_healthcheck"][0]
    assert "local_judge_goldset_validate" in manifest["suggested_commands"]["local_judge_goldset_validate"][0]
    assert "local_judge_goldset_chain_verify" in manifest["suggested_commands"]["local_judge_goldset_chain_verify"][0]
    assert "local_judge_quality_eval" in manifest["suggested_commands"]["local_judge_quality_eval"][0]
    assert "static_rule_calibration_eval" in manifest["suggested_commands"]["static_rule_calibration_eval"][0]
    assert manifest["local_judge"]["configured"] is False
    assert "--base-url <local-judge-base-url>" in manifest["suggested_commands"]["local_judge_healthcheck"][0]
    assert "--model <local-judge-model>" in manifest["suggested_commands"]["local_judge_healthcheck"][0]
    assert "local_judge_goldset_builder" in manifest["suggested_commands"]["local_judge_goldset_template"][0]
    assert "--include-text" in manifest["suggested_commands"]["local_judge_goldset_template"][1]
    assert "--source-prompts" in manifest["suggested_commands"]["local_judge_goldset_template"][1]
    assert "local_judge_goldset_csv export" in manifest["suggested_commands"]["local_judge_goldset_template"][2]
    assert "local_judge_goldset_csv progress" in manifest["suggested_commands"]["local_judge_goldset_template"][3]
    assert "local_judge_goldset_csv labels-template" in manifest["suggested_commands"]["local_judge_goldset_template"][4]
    assert "local_judge_goldset_csv suggest-labels" in manifest["suggested_commands"]["local_judge_goldset_template"][5]
    assert "local_judge_goldset_csv merge-suggestions" in manifest["suggested_commands"]["local_judge_goldset_template"][6]
    assert "local_judge_goldset_csv review-plan" in manifest["suggested_commands"]["local_judge_goldset_template"][7]
    assert "local_judge_goldset_csv labels-from-csv" in manifest["suggested_commands"]["local_judge_goldset_template"][8]
    assert "local_judge_goldset_csv validate-labels" in manifest["suggested_commands"]["local_judge_goldset_template"][9]
    assert "local_judge_goldset_csv apply-labels" in manifest["suggested_commands"]["local_judge_goldset_template"][10]
    assert "local_judge_goldset_csv import" in manifest["suggested_commands"]["local_judge_goldset_template"][11]
    assert "local_judge_goldset_labeled.local.jsonl" in manifest["suggested_commands"]["local_judge_quality_eval"][0]
    assert manifest["local_judge"]["healthcheck_summary_path"].endswith("local_judge_healthcheck_summary.json")
    assert manifest["local_judge"]["goldset_template_summary_path"].endswith("local_judge_goldset_template_summary.json")
    assert manifest["local_judge"]["quality_eval_predictions_path"].endswith("local_judge_quality_eval_predictions.jsonl")
    assert "offline_local_judge_matrix_smoke" in manifest["suggested_commands"]["offline_local_judge_matrix_smoke"][0]
    assert "--source autogen=<autogen-source-root>" in manifest["suggested_commands"]["offline_local_judge_matrix_smoke"][0]
    assert "--max-local-judge-calls 50" in manifest["suggested_commands"]["offline_local_judge_matrix_smoke"][0]
    assert "offline_semantic_matrix" in manifest["suggested_commands"]["offline_local_judge_matrix"][0]
    assert "--judge openai-compatible" in manifest["suggested_commands"]["offline_local_judge_matrix"][0]
    assert "--include-candidate-text" in manifest["suggested_commands"]["offline_local_judge_matrix"][0]
    assert "--local-judge-base-url <local-judge-base-url>" in manifest["suggested_commands"]["offline_local_judge_matrix"][0]
    assert "agbench_legacy_preflight_smokes" in manifest["suggested_commands"]["preflight_smokes"][0]
    assert "--manifest" in manifest["suggested_commands"]["preflight_smokes"][0]
    assert "agbench_legacy_preflight" in manifest["suggested_commands"]["real_ab_preflight"][0]
    assert "legacy_real_ab_preflight_summary.json" in manifest["suggested_commands"]["real_ab_preflight"][0]
    assert "agbench_legacy_collect" in manifest["suggested_commands"]["collect"][0]
    assert "agbench_legacy_runbook" in manifest["suggested_commands"]["runbook"][0]
    assert manifest["recommended_runbook"]["summary"].endswith("legacy_runbook_summary.json")
    assert manifest["recommended_runbook"]["report_md"].endswith("legacy_runbook.md")
    assert written_summary["ready"] is True
    assert _check(written_summary, "manifest:proxy_ports_distinct")["ok"] is True


def _check(summary, name: str):
    return next(check for check in summary["checks"] if check["name"] == name)


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
