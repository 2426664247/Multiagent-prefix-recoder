from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LegacyAgBenchSuiteResult:
    output_dir: str
    manifest_path: str
    manifest: dict[str, Any]


def generate_legacy_agbench_suite(
    *,
    output_dir: str | Path,
    suite_id: str,
    model: str,
    upstream_base_url: str,
    task_jsonl_path: str | Path | None = None,
    baseline_proxy_base_url: str = "http://127.0.0.1:8787/v1",
    rule_proxy_base_url: str = "http://127.0.0.1:8788/v1",
    nl_proxy_base_url: str = "http://127.0.0.1:8789/v1",
    api_key_env: str = "OPENAI_API_KEY",
    api_key_placeholder: str = "${OPENAI_API_KEY}",
    extra_config: Mapping[str, Any] | None = None,
    include_guard: bool = False,
    semantic_guard_base_url: str | None = None,
    semantic_guard_model: str | None = None,
    local_judge_base_url: str | None = None,
    local_judge_model: str | None = None,
    local_judge_api_key_env: str | None = None,
    proxy_host: str = "127.0.0.1",
    baseline_proxy_port: int = 8787,
    rule_proxy_port: int = 8788,
    nl_proxy_port: int = 8789,
) -> LegacyAgBenchSuiteResult:
    if include_guard and (not semantic_guard_base_url or not semantic_guard_model):
        raise ValueError("semantic_guard_base_url and semantic_guard_model are required when include_guard=True")
    if task_jsonl_path is None:
        raise ValueError("task_jsonl_path is required so each A/B variant gets a distinct AutoGenBench scenario file")
    out = Path(output_dir)
    artifacts = out / "artifacts"
    reports = out / "reports"
    scenarios = out / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    scenarios.mkdir(parents=True, exist_ok=True)

    base_entry = _base_config_entry(
        model=model,
        base_url=baseline_proxy_base_url,
        api_key_placeholder=api_key_placeholder,
        tags=[f"{suite_id}-baseline", "baseline", model],
        extra_config=extra_config,
    )
    rule_entry = _base_config_entry(
        model=model,
        base_url=rule_proxy_base_url,
        api_key_placeholder=api_key_placeholder,
        tags=[f"{suite_id}-plugin-rule-only", "plugin_rule_only", "plugin", "prefix-reorder", model],
        extra_config=extra_config,
    )
    nl_entry = _base_config_entry(
        model=model,
        base_url=nl_proxy_base_url,
        api_key_placeholder=api_key_placeholder,
        tags=[f"{suite_id}-plugin-nl-segmentation", "plugin_nl_segmentation", "plugin", "prefix-reorder", model],
        extra_config=extra_config,
    )

    baseline_path = out / "OAI_CONFIG_LIST.baseline.json"
    rule_path = out / "OAI_CONFIG_LIST.plugin_rule_only.json"
    nl_path = out / "OAI_CONFIG_LIST.plugin_nl_segmentation.json"
    combined_path = out / "OAI_CONFIG_LIST.combined.json"
    _write_json(baseline_path, [base_entry])
    _write_json(rule_path, [rule_entry])
    _write_json(nl_path, [nl_entry])
    _write_json(combined_path, [base_entry, rule_entry, nl_entry])
    scenario_files = _scenario_files(
        scenarios_dir=scenarios,
        suite_id=suite_id,
        task_jsonl_path=Path(task_jsonl_path) if task_jsonl_path is not None else None,
    )

    manifest = {
        "schema_version": "prefix-legacy-agbench-suite-manifest-v1",
        "suite_id": suite_id,
        "output_dir": str(out),
        "artifacts_dir": str(artifacts),
        "reports_dir": str(reports),
        "prompt_safe_manifest": True,
        "secret_policy": {
            "api_key_env": api_key_env,
            "api_key_written_as_placeholder": api_key_placeholder,
            "api_key_value_not_written": True,
        },
        "autogenbench_version_family": "0.0.3-legacy-oai-config-list",
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "This suite only generates OAI_CONFIG_LIST files and proxy commands. "
            "Cached tokens, latency, cost, and task success require real AutoGenBench/provider runs."
        ),
        "oai_config_lists": {
            "baseline": {
                "path": str(baseline_path),
                "model": model,
                "base_url": baseline_proxy_base_url,
                "upstream_base_url": upstream_base_url,
                "tags": base_entry["tags"],
                "scenario_path": scenario_files["baseline"]["path"],
                "tabulate_csv_path": str(reports / "baseline_tabulate.csv"),
                "provider_telemetry_path": str(artifacts / "baseline_provider_telemetry.jsonl"),
                "task_results_path": str(artifacts / "baseline_task_results.jsonl"),
            },
            "plugin_rule_only": {
                "path": str(rule_path),
                "model": model,
                "base_url": rule_proxy_base_url,
                "upstream_base_url": upstream_base_url,
                "tags": rule_entry["tags"],
                "scenario_path": scenario_files["plugin_rule_only"]["path"],
                "tabulate_csv_path": str(reports / "plugin_rule_only_tabulate.csv"),
                "provider_telemetry_path": str(artifacts / "plugin_rule_only_provider_telemetry.jsonl"),
                "task_results_path": str(artifacts / "plugin_rule_only_task_results.jsonl"),
            },
            "plugin_nl_segmentation": {
                "path": str(nl_path),
                "model": model,
                "base_url": nl_proxy_base_url,
                "upstream_base_url": upstream_base_url,
                "tags": nl_entry["tags"],
                "scenario_path": scenario_files["plugin_nl_segmentation"]["path"],
                "tabulate_csv_path": str(reports / "plugin_nl_segmentation_tabulate.csv"),
                "provider_telemetry_path": str(artifacts / "plugin_nl_segmentation_provider_telemetry.jsonl"),
                "task_results_path": str(artifacts / "plugin_nl_segmentation_task_results.jsonl"),
            },
            "combined": {
                "path": str(combined_path),
                "tags": ["baseline", "plugin_rule_only", "plugin_nl_segmentation"],
            },
        },
        "scenario_files": scenario_files,
        "proxies": {
            "host": proxy_host,
            "baseline": {
                "port": baseline_proxy_port,
                "base_url": baseline_proxy_base_url,
                "upstream_base_url": upstream_base_url,
                "telemetry_path": str(artifacts / "baseline_provider_telemetry.jsonl"),
                "session_id": f"{suite_id}-baseline-proxy",
                "rewrite_enabled": False,
            },
            "plugin_rule_only": {
                "port": rule_proxy_port,
                "base_url": rule_proxy_base_url,
                "upstream_base_url": upstream_base_url,
                "telemetry_path": str(artifacts / "plugin_rule_only_provider_telemetry.jsonl"),
                "session_id": f"{suite_id}-plugin-rule-only-proxy",
                "rewrite_enabled": True,
                "natural_language_segmentation_enabled": False,
                "semantic_guard_enabled": include_guard,
                "semantic_guard_base_url": semantic_guard_base_url if include_guard else None,
                "semantic_guard_model": semantic_guard_model if include_guard else None,
            },
            "plugin_nl_segmentation": {
                "port": nl_proxy_port,
                "base_url": nl_proxy_base_url,
                "upstream_base_url": upstream_base_url,
                "telemetry_path": str(artifacts / "plugin_nl_segmentation_provider_telemetry.jsonl"),
                "session_id": f"{suite_id}-plugin-nl-segmentation-proxy",
                "rewrite_enabled": True,
                "natural_language_segmentation_enabled": True,
                "semantic_guard_enabled": include_guard,
                "semantic_guard_base_url": semantic_guard_base_url if include_guard else None,
                "semantic_guard_model": semantic_guard_model if include_guard else None,
            },
        },
        "local_judge": {
            "configured": _local_judge_configured(
                base_url=local_judge_base_url,
                model=local_judge_model,
            ),
            "base_url": local_judge_base_url,
            "model": local_judge_model,
            "api_key_env": local_judge_api_key_env,
            "healthcheck_summary_path": str(artifacts / "local_judge_healthcheck_summary.json"),
            "goldset_template_path": str(artifacts / "local_judge_goldset_template.jsonl"),
            "goldset_template_summary_path": str(artifacts / "local_judge_goldset_template_summary.json"),
            "goldset_annotation_csv_path": str(artifacts / "local_judge_goldset_annotation.local.csv"),
            "goldset_annotation_export_summary_path": str(
                artifacts / "local_judge_goldset_annotation_export.local.summary.json"
            ),
            "goldset_annotation_progress_summary_path": str(
                artifacts / "local_judge_goldset_annotation_progress.local.summary.json"
            ),
            "goldset_annotation_guide_path": str(artifacts / "local_judge_goldset_annotation_guide.local.md"),
            "goldset_annotation_worklist_path": str(
                artifacts / "local_judge_goldset_annotation_worklist.local.jsonl"
            ),
            "goldset_annotation_labels_path": str(
                artifacts / "local_judge_goldset_annotation_labels.local.jsonl"
            ),
            "goldset_annotation_labels_summary_path": str(
                artifacts / "local_judge_goldset_annotation_labels.local.summary.json"
            ),
            "goldset_annotation_labels_from_csv_summary_path": str(
                artifacts / "local_judge_goldset_annotation_labels_from_csv.local.summary.json"
            ),
            "goldset_annotation_suggestions_path": str(
                artifacts / "local_judge_goldset_annotation_label_suggestions.local.jsonl"
            ),
            "goldset_annotation_suggestions_summary_path": str(
                artifacts / "local_judge_goldset_annotation_label_suggestions.local.summary.json"
            ),
            "goldset_annotation_suggestion_review_csv_path": str(
                artifacts / "local_judge_goldset_annotation_suggestion_review.local.csv"
            ),
            "goldset_annotation_suggestion_review_summary_path": str(
                artifacts / "local_judge_goldset_annotation_suggestion_review.local.summary.json"
            ),
            "goldset_annotation_review_plan_path": str(
                artifacts / "local_judge_goldset_annotation_review_plan.local.jsonl"
            ),
            "goldset_annotation_review_plan_summary_path": str(
                artifacts / "local_judge_goldset_annotation_review_plan.local.summary.json"
            ),
            "goldset_annotation_labels_validation_summary_path": str(
                artifacts / "local_judge_goldset_annotation_labels_validation.local.summary.json"
            ),
            "goldset_annotation_labeled_csv_path": str(
                artifacts / "local_judge_goldset_annotation.labeled.local.csv"
            ),
            "goldset_annotation_apply_summary_path": str(
                artifacts / "local_judge_goldset_annotation_apply.local.summary.json"
            ),
            "goldset_annotation_import_summary_path": str(
                artifacts / "local_judge_goldset_annotation_import.local.summary.json"
            ),
            "goldset_labeled_path": str(artifacts / "local_judge_goldset_labeled.local.jsonl"),
            "goldset_validation_summary_path": str(artifacts / "local_judge_goldset_validation_summary.json"),
            "quality_eval_summary_path": str(artifacts / "local_judge_quality_eval_summary.json"),
            "quality_eval_predictions_path": str(artifacts / "local_judge_quality_eval_predictions.jsonl"),
            "policy_eval_summary_path": str(artifacts / "local_judge_policy_eval_summary.json"),
            "policy_eval_report_path": str(reports / "local_judge_policy_eval.md"),
            "static_rule_calibration_eval_summary_path": str(
                artifacts / "static_rule_calibration_eval_summary.json"
            ),
            "static_rule_calibration_eval_report_path": str(reports / "static_rule_calibration_eval.md"),
            "shadow_trial_plan_path": str(artifacts / "static_rule_calibration_eval_summary.json"),
            "goldset_chain_verification_summary_path": str(
                artifacts / "local_judge_goldset_chain_verification_summary.json"
            ),
            "goldset_chain_verification_report_path": str(
                reports / "local_judge_goldset_chain_verification.md"
            ),
        },
        "suggested_commands": _suggested_commands(
            suite_id=suite_id,
            manifest_path=out / "legacy_suite_manifest.json",
            reports_dir=reports,
            baseline_path=baseline_path,
            rule_path=rule_path,
            nl_path=nl_path,
            baseline_scenario_path=Path(scenario_files["baseline"]["path"]),
            rule_scenario_path=Path(scenario_files["plugin_rule_only"]["path"]),
            nl_scenario_path=Path(scenario_files["plugin_nl_segmentation"]["path"]),
            artifacts_dir=artifacts,
            proxy_host=proxy_host,
            baseline_proxy_port=baseline_proxy_port,
            rule_proxy_port=rule_proxy_port,
            nl_proxy_port=nl_proxy_port,
            upstream_base_url=upstream_base_url,
            include_guard=include_guard,
            semantic_guard_base_url=semantic_guard_base_url,
            semantic_guard_model=semantic_guard_model,
            local_judge_base_url=local_judge_base_url,
            local_judge_model=local_judge_model,
            local_judge_api_key_env=local_judge_api_key_env,
        ),
        "recommended_ab_eval": {
            "baseline_provider_telemetry": str(artifacts / "baseline_provider_telemetry.jsonl"),
            "plugin_rule_only_provider_telemetry": str(artifacts / "plugin_rule_only_provider_telemetry.jsonl"),
            "plugin_nl_segmentation_provider_telemetry": str(artifacts / "plugin_nl_segmentation_provider_telemetry.jsonl"),
            "baseline_task_results": str(artifacts / "baseline_task_results.jsonl"),
            "plugin_rule_only_task_results": str(artifacts / "plugin_rule_only_task_results.jsonl"),
            "plugin_nl_segmentation_task_results": str(artifacts / "plugin_nl_segmentation_task_results.jsonl"),
            "rule_only_summary": str(reports / "legacy_ab_baseline_vs_rule_only_summary.json"),
            "rule_only_report_md": str(reports / "legacy_ab_baseline_vs_rule_only_report.md"),
            "nl_segmentation_summary": str(reports / "legacy_ab_baseline_vs_nl_segmentation_summary.json"),
            "nl_segmentation_report_md": str(reports / "legacy_ab_baseline_vs_nl_segmentation_report.md"),
        },
        "recommended_runbook": {
            "summary": str(reports / "legacy_runbook_summary.json"),
            "report_md": str(reports / "legacy_runbook.md"),
        },
    }
    manifest_path = out / "legacy_suite_manifest.json"
    _write_json(manifest_path, manifest)
    return LegacyAgBenchSuiteResult(output_dir=str(out), manifest_path=str(manifest_path), manifest=manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate legacy AutoGenBench 0.0.3 OAI_CONFIG_LIST files for baseline/proxy A/B runs."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument(
        "--task-jsonl",
        "--scenario-jsonl",
        dest="task_jsonl",
        required=True,
        help=(
            "Official AutoGenBench task JSONL to copy into three variant scenario files. "
            "This avoids autogenbench 0.0.3 result directory collisions across A/B groups."
        ),
    )
    parser.add_argument("--baseline-proxy-base-url", default="http://127.0.0.1:8787/v1")
    parser.add_argument("--rule-proxy-base-url", default="http://127.0.0.1:8788/v1")
    parser.add_argument("--nl-proxy-base-url", default="http://127.0.0.1:8789/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--api-key-placeholder",
        default="${OPENAI_API_KEY}",
        help="Placeholder written into generated JSON. The real key should stay in the runtime environment.",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--price", nargs=2, type=float, metavar=("PROMPT_PER_1K", "COMPLETION_PER_1K"))
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--baseline-proxy-port", type=int, default=8787)
    parser.add_argument("--rule-proxy-port", type=int, default=8788)
    parser.add_argument("--nl-proxy-port", type=int, default=8789)
    parser.add_argument("--include-guard", action="store_true")
    parser.add_argument("--semantic-guard-base-url")
    parser.add_argument("--semantic-guard-model")
    parser.add_argument("--local-judge-base-url")
    parser.add_argument("--local-judge-model")
    parser.add_argument("--local-judge-api-key-env")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_legacy_agbench_suite(
        output_dir=args.output_dir,
        suite_id=args.suite_id,
        model=args.model,
        upstream_base_url=args.upstream_base_url,
        task_jsonl_path=args.task_jsonl,
        baseline_proxy_base_url=args.baseline_proxy_base_url,
        rule_proxy_base_url=args.rule_proxy_base_url,
        nl_proxy_base_url=args.nl_proxy_base_url,
        api_key_env=args.api_key_env,
        api_key_placeholder=args.api_key_placeholder,
        extra_config=_extra_config_from_args(args),
        include_guard=args.include_guard,
        semantic_guard_base_url=args.semantic_guard_base_url,
        semantic_guard_model=args.semantic_guard_model,
        local_judge_base_url=args.local_judge_base_url,
        local_judge_model=args.local_judge_model,
        local_judge_api_key_env=args.local_judge_api_key_env,
        proxy_host=args.proxy_host,
        baseline_proxy_port=args.baseline_proxy_port,
        rule_proxy_port=args.rule_proxy_port,
        nl_proxy_port=args.nl_proxy_port,
    )
    print(json.dumps(result.manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _base_config_entry(
    *,
    model: str,
    base_url: str,
    api_key_placeholder: str,
    tags: Sequence[str],
    extra_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key_placeholder,
        "tags": list(tags),
    }
    if extra_config:
        entry.update(deepcopy(dict(extra_config)))
    return entry


def _project_provider_proxy_args(upstream_base_url: str) -> str:
    if "api.deepseek.com" not in upstream_base_url.lower():
        return ""
    return " --use-project-deepseek-config"


def _local_judge_configured(*, base_url: str | None, model: str | None) -> bool:
    return bool(
        base_url
        and model
        and not _is_placeholder_value(base_url)
        and not _is_placeholder_value(model)
        and model != "local-small-model"
    )


def _is_placeholder_value(value: str | None) -> bool:
    if not value:
        return True
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _extra_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if args.temperature is not None:
        extra["temperature"] = args.temperature
    if args.max_tokens is not None:
        extra["max_tokens"] = args.max_tokens
    if args.timeout is not None:
        extra["timeout"] = args.timeout
    if args.price is not None:
        extra["price"] = list(args.price)
    return extra


def _suggested_commands(
    *,
    suite_id: str,
    manifest_path: Path,
    reports_dir: Path,
    baseline_path: Path,
    rule_path: Path,
    nl_path: Path,
    baseline_scenario_path: Path,
    rule_scenario_path: Path,
    nl_scenario_path: Path,
    artifacts_dir: Path,
    proxy_host: str,
    baseline_proxy_port: int,
    rule_proxy_port: int,
    nl_proxy_port: int,
    upstream_base_url: str,
    include_guard: bool,
    semantic_guard_base_url: str | None,
    semantic_guard_model: str | None,
    local_judge_base_url: str | None,
    local_judge_model: str | None,
    local_judge_api_key_env: str | None,
) -> dict[str, list[str]]:
    baseline_proxy_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy "
        rf"--host {proxy_host} "
        rf"--port {baseline_proxy_port} "
        rf"--upstream-base-url {upstream_base_url} "
        rf"--telemetry {artifacts_dir}\baseline_provider_telemetry.jsonl "
        r"--reset-telemetry "
        rf"--session-id {suite_id}-baseline-proxy "
        r"--disabled"
    )
    rule_proxy_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy "
        rf"--host {proxy_host} "
        rf"--port {rule_proxy_port} "
        rf"--upstream-base-url {upstream_base_url} "
        rf"--telemetry {artifacts_dir}\plugin_rule_only_provider_telemetry.jsonl "
        r"--reset-telemetry "
        rf"--session-id {suite_id}-plugin-rule-only-proxy "
        rf"--shadow-trial-plan {artifacts_dir}\static_rule_calibration_eval_summary.json"
    )
    nl_proxy_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy "
        rf"--host {proxy_host} "
        rf"--port {nl_proxy_port} "
        rf"--upstream-base-url {upstream_base_url} "
        rf"--telemetry {artifacts_dir}\plugin_nl_segmentation_provider_telemetry.jsonl "
        r"--reset-telemetry "
        rf"--session-id {suite_id}-plugin-nl-segmentation-proxy "
        rf"--shadow-trial-plan {artifacts_dir}\static_rule_calibration_eval_summary.json "
        r"--enable-natural-language-segmentation"
    )
    project_config_args = _project_provider_proxy_args(upstream_base_url)
    if project_config_args:
        baseline_proxy_command += project_config_args
        rule_proxy_command += project_config_args
        nl_proxy_command += project_config_args
    if include_guard and semantic_guard_base_url and semantic_guard_model:
        guard_args = (
            rf" --semantic-guard-base-url {semantic_guard_base_url} "
            rf"--semantic-guard-model {semantic_guard_model}"
        )
        rule_proxy_command += guard_args
        nl_proxy_command += guard_args
    healthcheck_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_healthcheck "
        rf"--base-url {local_judge_base_url or '<local-judge-base-url>'} "
        rf"--model {local_judge_model or '<local-judge-model>'} "
        rf"--summary {artifacts_dir}\local_judge_healthcheck_summary.json"
    )
    if local_judge_api_key_env:
        healthcheck_command += rf" --api-key-env {local_judge_api_key_env}"
    goldset_template_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_builder "
        rf"--input {manifest_path.parent}\offline_semantic_suite\rule_only\semantic_candidates_labeled.jsonl "
        rf"--input {manifest_path.parent}\offline_semantic_suite\nl_segmentation\semantic_candidates_labeled.jsonl "
        rf"--max-rows 60 "
        rf"--seed 20260605 "
        rf"--output {artifacts_dir}\local_judge_goldset_template.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_template_summary.json"
    )
    goldset_template_with_text_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_builder "
        rf"--input {manifest_path.parent}\offline_semantic_suite\rule_only\semantic_candidates_labeled.jsonl "
        rf"--input {manifest_path.parent}\offline_semantic_suite\nl_segmentation\semantic_candidates_labeled.jsonl "
        rf"--source-prompts {manifest_path.parent}\offline_semantic_suite\rule_only\source_prompts.jsonl "
        rf"--source-prompts {manifest_path.parent}\offline_semantic_suite\nl_segmentation\source_prompts.jsonl "
        rf"--include-text "
        rf"--require-text "
        rf"--max-rows 60 "
        rf"--seed 20260605 "
        rf"--output {artifacts_dir}\local_judge_goldset_template.with_text.local.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_template.with_text.local.summary.json"
    )
    goldset_csv_export_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv export "
        rf"--input {artifacts_dir}\local_judge_goldset_template.with_text.local.jsonl "
        rf"--output {artifacts_dir}\local_judge_goldset_annotation.local.csv "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_export.local.summary.json"
    )
    goldset_csv_progress_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv progress "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation.local.csv "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_progress.local.summary.json "
        rf"--guide-md {artifacts_dir}\local_judge_goldset_annotation_guide.local.md "
        rf"--worklist-jsonl {artifacts_dir}\local_judge_goldset_annotation_worklist.local.jsonl "
        r"--min-samples 20 "
        r"--min-label-count 1"
    )
    goldset_csv_labels_template_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv labels-template "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation_worklist.local.jsonl "
        rf"--output {artifacts_dir}\local_judge_goldset_annotation_labels.local.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_labels.local.summary.json"
    )
    goldset_csv_suggest_labels_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv suggest-labels "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation.local.csv "
        rf"--output {artifacts_dir}\local_judge_goldset_annotation_label_suggestions.local.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_label_suggestions.local.summary.json "
        rf"--base-url {local_judge_base_url or '<local-judge-base-url>'} "
        rf"--model {local_judge_model or '<local-judge-model>'} "
        r"--timeout 90"
    )
    if local_judge_api_key_env:
        goldset_csv_suggest_labels_command += rf" --api-key-env {local_judge_api_key_env}"
    goldset_csv_merge_suggestions_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv merge-suggestions "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation.local.csv "
        rf"--suggestions {artifacts_dir}\local_judge_goldset_annotation_label_suggestions.local.jsonl "
        rf"--output {artifacts_dir}\local_judge_goldset_annotation_suggestion_review.local.csv "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_suggestion_review.local.summary.json"
    )
    goldset_csv_review_plan_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv review-plan "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation_suggestion_review.local.csv "
        rf"--output {artifacts_dir}\local_judge_goldset_annotation_review_plan.local.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_review_plan.local.summary.json"
    )
    goldset_csv_labels_from_csv_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv labels-from-csv "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation_suggestion_review.local.csv "
        rf"--output {artifacts_dir}\local_judge_goldset_annotation_labels.local.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_labels_from_csv.local.summary.json "
        r"--min-samples 20 "
        r"--min-label-count 1"
    )
    goldset_csv_validate_labels_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv validate-labels "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation_labels.local.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_labels_validation.local.summary.json "
        r"--require-complete "
        r"--min-samples 20 "
        r"--min-label-count 1"
    )
    goldset_csv_import_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv import "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation.labeled.local.csv "
        rf"--output {artifacts_dir}\local_judge_goldset_labeled.local.jsonl "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_import.local.summary.json"
    )
    goldset_csv_apply_labels_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_csv apply-labels "
        rf"--input {artifacts_dir}\local_judge_goldset_annotation.local.csv "
        rf"--labels {artifacts_dir}\local_judge_goldset_annotation_labels.local.jsonl "
        rf"--output {artifacts_dir}\local_judge_goldset_annotation.labeled.local.csv "
        rf"--summary {artifacts_dir}\local_judge_goldset_annotation_apply.local.summary.json "
        r"--require-complete"
    )
    quality_eval_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_quality_eval "
        rf"--input {artifacts_dir}\local_judge_goldset_labeled.local.jsonl "
        rf"--base-url {local_judge_base_url or '<local-judge-base-url>'} "
        rf"--model {local_judge_model or '<local-judge-model>'} "
        r"--min-samples 20 "
        r"--min-accuracy 0.75 "
        r"--min-macro-f1 0.70 "
        rf"--output {artifacts_dir}\local_judge_quality_eval_predictions.jsonl "
        rf"--summary {artifacts_dir}\local_judge_quality_eval_summary.json"
    )
    if local_judge_api_key_env:
        quality_eval_command += rf" --api-key-env {local_judge_api_key_env}"
    policy_eval_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_policy_eval "
        rf"--input {artifacts_dir}\local_judge_goldset_labeled.local.jsonl "
        rf"--predictions {artifacts_dir}\local_judge_quality_eval_predictions.jsonl "
        r"--min-accuracy 0.75 "
        r"--min-macro-f1 0.70 "
        rf"--summary {artifacts_dir}\local_judge_policy_eval_summary.json "
        rf"--report-md {reports_dir}\local_judge_policy_eval.md"
    )
    static_rule_calibration_eval_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.static_rule_calibration_eval "
        rf"--input {artifacts_dir}\local_judge_goldset_labeled.local.jsonl "
        r"--min-accuracy 0.75 "
        r"--min-macro-f1 0.70 "
        r"--production-min-support 2 "
        r"--production-min-purity 0.67 "
        rf"--summary {artifacts_dir}\static_rule_calibration_eval_summary.json "
        rf"--report-md {reports_dir}\static_rule_calibration_eval.md"
    )
    goldset_validate_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_validate "
        rf"--input {artifacts_dir}\local_judge_goldset_labeled.local.jsonl "
        rf"--min-samples 20 "
        rf"--min-label-count 1 "
        rf"--summary {artifacts_dir}\local_judge_goldset_validation_summary.json"
    )
    goldset_chain_verify_command = (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_chain_verify "
        rf"--manifest {manifest_path} "
        rf"--summary {artifacts_dir}\local_judge_goldset_chain_verification_summary.json "
        rf"--report-md {reports_dir}\local_judge_goldset_chain_verification.md"
    )
    return {
        "generate": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_suite --output-dir <out> --suite-id {suite_id} --model <model> --upstream-base-url <upstream-base-url> --scenario-jsonl <Tasks\human_eval_two_agents.jsonl>",
        ],
        "verify": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_suite_verify --manifest {manifest_path} --summary {reports_dir}\legacy_suite_verification_summary.json",
        ],
        "preflight_fake_smoke": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.three_proxy_smoke --output-dir {manifest_path.parent}\preflight_three_proxy_smoke --session-id {suite_id}-preflight-smoke --manifest {manifest_path}",
        ],
        "semantic_guard_fake_smoke": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.semantic_guard_proxy_smoke --output-dir {manifest_path.parent}\preflight_semantic_guard_proxy_smoke --session-id {suite_id}-semantic-guard-smoke --judge-mode reject",
        ],
        "preflight_smokes": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight_smokes --manifest {manifest_path} --output-dir {manifest_path.parent}\preflight_smokes",
        ],
        "offline_semantic_suite": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_suite --source-root <autogen-source-root> --output-dir {manifest_path.parent}\offline_semantic_suite --session-id {suite_id}-offline-semantic-suite",
        ],
        "local_judge_healthcheck": [
            healthcheck_command,
        ],
        "local_judge_goldset_template": [
            goldset_template_command,
            goldset_template_with_text_command,
            goldset_csv_export_command,
            goldset_csv_progress_command,
            goldset_csv_labels_template_command,
            goldset_csv_suggest_labels_command,
            goldset_csv_merge_suggestions_command,
            goldset_csv_review_plan_command,
            goldset_csv_labels_from_csv_command,
            goldset_csv_validate_labels_command,
            goldset_csv_apply_labels_command,
            goldset_csv_import_command,
        ],
        "local_judge_goldset_validate": [
            goldset_validate_command,
        ],
        "local_judge_goldset_chain_verify": [
            goldset_chain_verify_command,
        ],
        "local_judge_quality_eval": [
            quality_eval_command,
        ],
        "local_judge_policy_eval": [
            policy_eval_command,
        ],
        "static_rule_calibration_eval": [
            static_rule_calibration_eval_command,
        ],
        "offline_local_judge_matrix_smoke": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.offline_local_judge_matrix_smoke --source autogen=<autogen-source-root> --source agentscope=<agentscope-source-root> --source crewai=<crewai-source-root> --output-dir {manifest_path.parent}\offline_local_judge_matrix_smoke --session-id {suite_id}-offline-local-judge-matrix-smoke --local-judge-scope review --max-local-judge-calls 50 --fake-judge-mode accept",
        ],
        "offline_local_judge_matrix": [
            (
                rf".venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_matrix "
                rf"--source autogen=<autogen-source-root> "
                rf"--source agentscope=<agentscope-source-root> "
                rf"--source crewai=<crewai-source-root> "
                rf"--output-dir {manifest_path.parent}\offline_local_judge_matrix "
                rf"--session-id {suite_id}-offline-local-judge-matrix "
                rf"--include-candidate-text "
                rf"--judge openai-compatible "
                rf"--local-judge-base-url {local_judge_base_url or '<local-judge-base-url>'} "
                rf"--local-judge-model {local_judge_model or '<local-judge-model>'} "
                rf"--local-judge-scope review "
                rf"--max-local-judge-calls 50"
                + (rf" --local-judge-api-key-env {local_judge_api_key_env}" if local_judge_api_key_env else "")
            ),
        ],
        "runbook": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_runbook --manifest {manifest_path} --summary {reports_dir}\legacy_runbook_summary.json --report-md {reports_dir}\legacy_runbook.md",
        ],
        "real_ab_preflight": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight --manifest {manifest_path} --summary {reports_dir}\legacy_real_ab_preflight_summary.json --report-md {reports_dir}\legacy_real_ab_preflight.md",
        ],
        "proxy": [baseline_proxy_command, rule_proxy_command, nl_proxy_command],
        "autogenbench": [
            rf".venv\Scripts\autogenbench.exe run -c {baseline_path} --model baseline --subsample 0.1 --repeat 3 {baseline_scenario_path}",
            rf".venv\Scripts\autogenbench.exe run -c {rule_path} --model plugin_rule_only --subsample 0.1 --repeat 3 {rule_scenario_path}",
            rf".venv\Scripts\autogenbench.exe run -c {nl_path} --model plugin_nl_segmentation --subsample 0.1 --repeat 3 {nl_scenario_path}",
        ],
        "tabulate": [
            rf".venv\Scripts\autogenbench.exe tabulate Results\{baseline_scenario_path.stem} -c > {reports_dir}\baseline_tabulate.csv",
            rf".venv\Scripts\autogenbench.exe tabulate Results\{rule_scenario_path.stem} -c > {reports_dir}\plugin_rule_only_tabulate.csv",
            rf".venv\Scripts\autogenbench.exe tabulate Results\{nl_scenario_path.stem} -c > {reports_dir}\plugin_nl_segmentation_tabulate.csv",
        ],
        "collect": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_collect --manifest {manifest_path} --baseline-tabulate-csv {reports_dir}\baseline_tabulate.csv --rule-tabulate-csv {reports_dir}\plugin_rule_only_tabulate.csv --nl-tabulate-csv {reports_dir}\plugin_nl_segmentation_tabulate.csv --summary {reports_dir}\legacy_collection_summary.json",
        ],
        "ab_eval": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.ab_eval --baseline {artifacts_dir}\baseline_provider_telemetry.jsonl --plugin {artifacts_dir}\plugin_rule_only_provider_telemetry.jsonl --baseline-tasks {artifacts_dir}\baseline_task_results.jsonl --plugin-tasks {artifacts_dir}\plugin_rule_only_task_results.jsonl --summary {reports_dir}\legacy_ab_baseline_vs_rule_only_summary.json --report-md {reports_dir}\legacy_ab_baseline_vs_rule_only_report.md --min-success-rate-delta -0.02 --max-p95-latency-relative-change 0.0 --max-cost-per-success-relative-change 0.0 --bootstrap-iterations 1000 --bootstrap-seed 20260605",
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.ab_eval --baseline {artifacts_dir}\baseline_provider_telemetry.jsonl --plugin {artifacts_dir}\plugin_nl_segmentation_provider_telemetry.jsonl --baseline-tasks {artifacts_dir}\baseline_task_results.jsonl --plugin-tasks {artifacts_dir}\plugin_nl_segmentation_task_results.jsonl --summary {reports_dir}\legacy_ab_baseline_vs_nl_segmentation_summary.json --report-md {reports_dir}\legacy_ab_baseline_vs_nl_segmentation_report.md --min-success-rate-delta -0.02 --max-p95-latency-relative-change 0.0 --max-cost-per-success-relative-change 0.0 --bootstrap-iterations 1000 --bootstrap-seed 20260605",
        ],
    }


def _scenario_files(
    *,
    scenarios_dir: Path,
    suite_id: str,
    task_jsonl_path: Path | None,
) -> dict[str, dict[str, Any]]:
    names = {
        "baseline": f"{suite_id}.baseline.jsonl",
        "plugin_rule_only": f"{suite_id}.plugin_rule_only.jsonl",
        "plugin_nl_segmentation": f"{suite_id}.plugin_nl_segmentation.jsonl",
    }
    if task_jsonl_path is None:
        raise ValueError("task_jsonl_path is required")
    source_text = _read_variant_scenario_text(task_jsonl_path)
    result: dict[str, dict[str, Any]] = {}
    for label, filename in names.items():
        path = scenarios_dir / filename
        path.write_text(source_text, encoding="utf-8")
        result[label] = {
            "path": str(path),
            "source_task_jsonl": str(task_jsonl_path),
            "created": True,
            "result_dir_name": path.stem,
            "template_paths_rewritten_absolute": True,
        }
    return result


def _read_variant_scenario_text(task_jsonl_path: Path) -> str:
    base_dir = task_jsonl_path.parent.resolve()
    lines: list[str] = []
    with task_jsonl_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"{task_jsonl_path} must contain JSON objects")
            record["template"] = _absolute_template_value(record.get("template"), base_dir=base_dir)
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def _absolute_template_value(value: Any, *, base_dir: Path) -> Any:
    if isinstance(value, str):
        return _absolute_template_path(value, base_dir=base_dir)
    if isinstance(value, list):
        converted: list[Any] = []
        for item in value:
            if isinstance(item, str):
                converted.append(_absolute_template_path(item, base_dir=base_dir))
            elif isinstance(item, list) and item and isinstance(item[0], str):
                row = list(item)
                row[0] = _absolute_template_path(row[0], base_dir=base_dir)
                converted.append(row)
            else:
                converted.append(item)
        return converted
    return value


def _absolute_template_path(value: str, *, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / value).resolve())


def _write_json(path: Path, value: Any) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
