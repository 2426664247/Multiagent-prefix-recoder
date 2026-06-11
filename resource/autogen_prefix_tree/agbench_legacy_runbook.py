from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ab_eval import summarize_task_results
from .agbench_legacy_collect import parse_autogenbench_tabulate_csv
from .dataset_eval import evaluate_dataset
from .project_api_config import load_project_provider_config


VARIANTS = ("baseline", "plugin_rule_only", "plugin_nl_segmentation")


@dataclass(frozen=True)
class LegacyAgBenchRunbookResult:
    manifest_path: str
    summary_path: str | None
    report_path: str | None
    summary: dict[str, Any]


def build_legacy_agbench_runbook(
    *,
    manifest_path: str | Path,
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    require_api_config: bool = False,
) -> LegacyAgBenchRunbookResult:
    manifest_file = Path(manifest_path)
    manifest = _load_json_mapping(manifest_file)
    effective_env = os.environ if env is None else env
    checks = _checks(manifest=manifest, manifest_file=manifest_file, env=effective_env, require_api_config=require_api_config)
    artifacts = _artifact_status(manifest=manifest, manifest_file=manifest_file)
    artifact_quality = _artifact_quality_status(artifacts)
    artifact_quality_failed = any(item.get("status") == "fail" for item in artifact_quality)
    provider_artifact_fake = any(_is_provider_or_ab_fake_artifact(item) for item in artifact_quality)
    semantic_guard_fake_smoke = _semantic_guard_fake_smoke_status(artifact_quality)
    offline_semantic_suite = _offline_semantic_suite_status(artifact_quality)
    local_judge_healthcheck = _local_judge_healthcheck_status(artifact_quality)
    local_judge_goldset_template = _local_judge_goldset_template_status(artifact_quality)
    local_judge_goldset_progress = _local_judge_goldset_progress_status(artifact_quality)
    local_judge_goldset_labels_template = _local_judge_goldset_labels_template_status(artifact_quality)
    local_judge_goldset_suggestions = _local_judge_goldset_suggestions_status(artifact_quality)
    local_judge_goldset_suggestion_review = _local_judge_goldset_suggestion_review_status(artifact_quality)
    local_judge_goldset_review_plan = _local_judge_goldset_review_plan_status(artifact_quality)
    local_judge_goldset_labels_from_csv = _local_judge_goldset_labels_from_csv_status(artifact_quality)
    local_judge_goldset_labels_validation = _local_judge_goldset_labels_validation_status(artifact_quality)
    local_judge_goldset_annotation_ready = bool(
        local_judge_goldset_progress.get("ready_for_labeled_jsonl_import") is True
    )
    local_judge_goldset_import = _local_judge_goldset_import_status(artifact_quality)
    local_judge_goldset_validation = _local_judge_goldset_validation_status(artifact_quality)
    local_judge_quality_eval = _local_judge_quality_eval_status(artifact_quality)
    local_judge_policy_eval = _local_judge_policy_eval_status(artifact_quality)
    static_rule_calibration_eval = _static_rule_calibration_eval_status(artifact_quality)
    local_judge_goldset_chain = _local_judge_goldset_chain_status(artifact_quality)
    offline_local_judge_matrix = _offline_local_judge_matrix_status(artifact_quality)
    offline_local_judge_matrix_smoke = _offline_local_judge_matrix_smoke_status(artifact_quality)
    shadow_trial_wiring = _shadow_trial_wiring_status(artifact_quality)
    local_judge_config = _local_judge_config_status(manifest)
    structural_ready = all(check["ok"] for check in checks if check["name"] != "api_config")
    api_config_present = bool(next((check for check in checks if check["name"] == "api_config"), {}).get("present"))
    collection_ready = _collection_inputs_ready(artifacts)
    ab_reports_ready = _ab_reports_ready(artifacts)
    summary = {
        "schema_version": "prefix-legacy-agbench-runbook-v1",
        "manifest_path": str(manifest_file),
        "suite_id": manifest.get("suite_id"),
        "prompt_safe_summary": True,
        "checks": checks,
        "structural_ready": structural_ready,
        "api_config_present": api_config_present,
        "ready_to_start_real_run": structural_ready and api_config_present,
        "collection_inputs_ready": collection_ready,
        "ab_reports_ready": ab_reports_ready,
        "next_action": _next_action(
            structural_ready=structural_ready,
            api_config_present=api_config_present,
            artifact_quality_failed=artifact_quality_failed,
            artifact_quality_fake=provider_artifact_fake,
            offline_semantic_suite_ready=offline_semantic_suite["ready"],
            local_judge_config_ready=local_judge_config["ready"],
            local_judge_healthcheck_ready=local_judge_healthcheck["ready"],
            local_judge_goldset_template_ready=local_judge_goldset_template["ready"],
            local_judge_goldset_annotation_ready=local_judge_goldset_annotation_ready,
            local_judge_goldset_import_ready=local_judge_goldset_import["ready"],
            local_judge_goldset_validation_ready=local_judge_goldset_validation["ready"],
            local_judge_quality_eval_ready=local_judge_quality_eval["ready"],
            local_judge_quality_eval_ran=_local_quality_eval_ran(local_judge_quality_eval),
            static_rule_calibration_eval_ran=_static_rule_calibration_eval_ran(static_rule_calibration_eval),
            local_judge_goldset_chain_ready=local_judge_goldset_chain["ready"],
            offline_local_judge_matrix_ready=offline_local_judge_matrix["ready"],
            offline_local_judge_matrix_smoke_ready=offline_local_judge_matrix_smoke["ready"],
            semantic_guard_fake_smoke_ready=semantic_guard_fake_smoke["ready"],
            collection_ready=collection_ready,
            ab_reports_ready=ab_reports_ready,
        ),
        "steps": _runbook_steps(manifest, manifest_file=manifest_file),
        "artifacts": artifacts,
        "artifact_quality": artifact_quality,
        "artifact_quality_fail_count": sum(1 for item in artifact_quality if item.get("status") == "fail"),
        "artifact_quality_warn_count": sum(1 for item in artifact_quality if item.get("status") == "warn"),
        "provider_or_ab_fake_artifacts_detected": provider_artifact_fake,
        "semantic_guard_fake_smoke_ready": semantic_guard_fake_smoke["ready"],
        "semantic_guard_fake_smoke_status": semantic_guard_fake_smoke,
        "offline_semantic_suite_ready": offline_semantic_suite["ready"],
        "offline_semantic_suite_status": offline_semantic_suite,
        "local_judge_config_ready": local_judge_config["ready"],
        "local_judge_config_status": local_judge_config,
        "local_judge_healthcheck_ready": local_judge_healthcheck["ready"],
        "local_judge_healthcheck_status": local_judge_healthcheck,
        "local_judge_goldset_template_ready": local_judge_goldset_template["ready"],
        "local_judge_goldset_template_status": local_judge_goldset_template,
        "local_judge_goldset_progress_ready": local_judge_goldset_progress["ready"],
        "local_judge_goldset_labels_template_ready": local_judge_goldset_labels_template["ready"],
        "local_judge_goldset_labels_template_status": local_judge_goldset_labels_template,
        "local_judge_goldset_suggestions_ready": local_judge_goldset_suggestions["ready"],
        "local_judge_goldset_suggestions_status": local_judge_goldset_suggestions,
        "local_judge_goldset_suggestion_review_ready": local_judge_goldset_suggestion_review["ready"],
        "local_judge_goldset_suggestion_review_status": local_judge_goldset_suggestion_review,
        "local_judge_goldset_review_plan_ready": local_judge_goldset_review_plan["ready"],
        "local_judge_goldset_review_plan_status": local_judge_goldset_review_plan,
        "local_judge_goldset_labels_from_csv_ready": local_judge_goldset_labels_from_csv["ready"],
        "local_judge_goldset_labels_from_csv_status": local_judge_goldset_labels_from_csv,
        "local_judge_goldset_labels_validation_ready": local_judge_goldset_labels_validation["ready"],
        "local_judge_goldset_labels_validation_status": local_judge_goldset_labels_validation,
        "local_judge_goldset_annotation_ready": local_judge_goldset_annotation_ready,
        "local_judge_goldset_progress_status": local_judge_goldset_progress,
        "local_judge_goldset_import_ready": local_judge_goldset_import["ready"],
        "local_judge_goldset_import_status": local_judge_goldset_import,
        "local_judge_goldset_validation_ready": local_judge_goldset_validation["ready"],
        "local_judge_goldset_validation_status": local_judge_goldset_validation,
        "local_judge_quality_eval_ready": local_judge_quality_eval["ready"],
        "local_judge_quality_eval_status": local_judge_quality_eval,
        "local_judge_policy_eval_ready": local_judge_policy_eval["ready"],
        "local_judge_policy_eval_status": local_judge_policy_eval,
        "static_rule_calibration_eval_ready": static_rule_calibration_eval["ready"],
        "static_rule_calibration_eval_status": static_rule_calibration_eval,
        "local_judge_goldset_chain_ready": local_judge_goldset_chain["ready"],
        "local_judge_goldset_chain_status": local_judge_goldset_chain,
        "offline_local_judge_matrix_ready": offline_local_judge_matrix["ready"],
        "offline_local_judge_matrix_status": offline_local_judge_matrix,
        "offline_local_judge_matrix_smoke_ready": offline_local_judge_matrix_smoke["ready"],
        "offline_local_judge_matrix_smoke_status": offline_local_judge_matrix_smoke,
        "shadow_trial_wiring_ready": shadow_trial_wiring["ready"],
        "shadow_trial_wiring_status": shadow_trial_wiring,
        "real_provider_metrics_available": ab_reports_ready and not provider_artifact_fake,
        "real_provider_metrics_note": (
            "This runbook only checks structure, API markers, command order, and artifact presence. "
            "Real cached tokens, latency, cost, and task success require the three proxy and AutoGenBench runs."
        ),
    }
    summary_target = _write_json(summary_path, summary)
    report_target = _write_text(report_path, render_markdown_runbook(summary))
    return LegacyAgBenchRunbookResult(
        manifest_path=str(manifest_file),
        summary_path=str(summary_target) if summary_target else None,
        report_path=str(report_target) if report_target else None,
        summary=summary,
    )


def render_markdown_runbook(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Legacy AutoGenBench A/B Runbook",
        "",
        f"Suite: `{_format_value(summary.get('suite_id'))}`",
        f"Next action: **{_format_value(summary.get('next_action'))}**",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in summary.get("checks") if isinstance(summary.get("checks"), list) else ():
        if isinstance(check, Mapping):
            lines.append(
                f"| {check.get('name')} | {'pass' if check.get('ok') else 'fail'} | {_format_value(check.get('detail'))} |"
            )
    lines.extend(["", "## Steps", ""])
    for step in summary.get("steps") if isinstance(summary.get("steps"), list) else ():
        if not isinstance(step, Mapping):
            continue
        lines.extend([f"### {step.get('order')}. {step.get('name')}", "", _format_value(step.get("purpose")), ""])
        commands = step.get("commands") if isinstance(step.get("commands"), list) else []
        if commands:
            lines.append("```powershell")
            lines.extend(str(command) for command in commands)
            lines.append("```")
            lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            "| Artifact | Exists | Size bytes | Path |",
            "|---|---:|---:|---|",
        ]
    )
    for artifact in summary.get("artifacts") if isinstance(summary.get("artifacts"), list) else ():
        if isinstance(artifact, Mapping):
            lines.append(
                f"| {artifact.get('name')} | {_format_value(artifact.get('exists'))} | "
                f"{_format_value(artifact.get('size_bytes'))} | `{artifact.get('path')}` |"
            )
    lines.extend(
        [
            "",
            "## Artifact Quality",
            "",
            "| Artifact | Status | Evidence |",
            "|---|---|---|",
        ]
    )
    for item in summary.get("artifact_quality") if isinstance(summary.get("artifact_quality"), list) else ():
        if isinstance(item, Mapping):
            lines.append(
                f"| {item.get('name')} | {_format_value(item.get('status'))} | {_format_value(item.get('evidence'))} |"
            )
    lines.extend(_offline_evidence_markdown(summary))
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This file is prompt-safe and does not include API key values.",
            "- Proxy commands are long-running processes and should be started in separate terminals.",
            "- This runbook does not execute AutoGenBench and does not prove provider cache, latency, cost, or task success gains.",
            "",
        ]
    )
    return "\n".join(lines)


def _offline_evidence_markdown(summary: Mapping[str, Any]) -> list[str]:
    rows: list[tuple[str, str, Any]] = []
    suite = (
        summary.get("offline_semantic_suite_status")
        if isinstance(summary.get("offline_semantic_suite_status"), Mapping)
        else {}
    )
    if suite:
        review_queue = (
            suite.get("review_queue_diagnostics")
            if isinstance(suite.get("review_queue_diagnostics"), Mapping)
            else {}
        )
        rows.extend(
            [
                ("offline_semantic_suite", "status", suite.get("status")),
                ("offline_semantic_suite", "rule_supported_request_count", suite.get("rule_supported_request_count")),
                ("offline_semantic_suite", "nl_supported_request_count", suite.get("nl_supported_request_count")),
                ("offline_semantic_suite", "rule_only_applied_count", suite.get("rule_only_applied_count")),
                (
                    "offline_semantic_suite",
                    "nl_segmentation_applied_count",
                    suite.get("nl_segmentation_applied_count"),
                ),
                (
                    "offline_semantic_suite",
                    "rule_only_total_estimated_gain_chars",
                    suite.get("rule_only_total_estimated_gain_chars"),
                ),
                (
                    "offline_semantic_suite",
                    "nl_segmentation_total_estimated_gain_chars",
                    suite.get("nl_segmentation_total_estimated_gain_chars"),
                ),
                ("offline_semantic_suite", "rule_only_candidate_count", suite.get("rule_only_candidate_count")),
                (
                    "offline_semantic_suite",
                    "nl_segmentation_candidate_count",
                    suite.get("nl_segmentation_candidate_count"),
                ),
                (
                    "offline_semantic_suite",
                    "rule_only_review_candidate_count",
                    suite.get("rule_only_review_candidate_count"),
                ),
                (
                    "offline_semantic_suite",
                    "nl_segmentation_review_candidate_count",
                    suite.get("nl_segmentation_review_candidate_count"),
                ),
                ("offline_semantic_suite", "rule_gap_resolved", suite.get("rule_gap_resolved")),
                ("offline_semantic_suite", "recommendation", suite.get("recommendation")),
                ("offline_semantic_suite", "candidate_promotion_policy", suite.get("candidate_promotion_policy")),
                (
                    "offline_semantic_suite",
                    "candidate_review_upper_bound_policy",
                    suite.get("candidate_review_upper_bound_policy"),
                ),
                ("offline_semantic_suite", "experiment_gate_items", suite.get("experiment_gate_items")),
                (
                    "offline_semantic_suite",
                    "review_queue_review_parent_block_count",
                    review_queue.get("review_parent_block_count"),
                ),
                (
                    "offline_semantic_suite",
                    "review_queue_review_source_file_count",
                    review_queue.get("review_source_file_count"),
                ),
                (
                    "offline_semantic_suite",
                    "review_queue_review_candidate_chars",
                    review_queue.get("review_candidate_chars"),
                ),
                (
                    "offline_semantic_suite",
                    "review_queue_semantic_hint_counts",
                    review_queue.get("semantic_hint_counts"),
                ),
                (
                    "offline_semantic_suite",
                    "review_queue_risk_tag_counts",
                    review_queue.get("risk_tag_counts"),
                ),
                (
                    "offline_semantic_suite",
                    "review_queue_label_reason_counts",
                    review_queue.get("label_reason_counts"),
                ),
                (
                    "offline_semantic_suite",
                    "review_queue_local_judge_priority",
                    review_queue.get("local_judge_priority"),
                ),
                (
                    "offline_semantic_suite",
                    "prompt_extraction_counts",
                    _diagnostic_extraction_counts(suite),
                ),
            ]
        )
    local_judge = (
        summary.get("local_judge_config_status")
        if isinstance(summary.get("local_judge_config_status"), Mapping)
        else {}
    )
    if local_judge:
        rows.extend(
            [
                ("local_judge_config", "ready", local_judge.get("ready")),
                ("local_judge_config", "configured", local_judge.get("configured")),
                ("local_judge_config", "base_url", local_judge.get("base_url")),
                ("local_judge_config", "model", local_judge.get("model")),
                ("local_judge_config", "reason", local_judge.get("reason")),
            ]
        )
    goldset = (
        summary.get("local_judge_goldset_template_status")
        if isinstance(summary.get("local_judge_goldset_template_status"), Mapping)
        else {}
    )
    if goldset:
        rows.extend(
            [
                ("local_judge_goldset_template", "status", goldset.get("status")),
                ("local_judge_goldset_template", "selected_count", goldset.get("selected_count")),
                (
                    "local_judge_goldset_template",
                    "expected_label_pending_count",
                    goldset.get("expected_label_pending_count"),
                ),
                ("local_judge_goldset_template", "include_text", goldset.get("include_text")),
                ("local_judge_goldset_template", "gold_text_written", goldset.get("gold_text_written")),
                (
                    "local_judge_goldset_template",
                    "selected_recovered_text_count",
                    goldset.get("selected_recovered_text_count"),
                ),
                ("local_judge_goldset_template", "source_label_counts", goldset.get("source_label_counts")),
                ("local_judge_goldset_template", "semantic_hint_counts", goldset.get("semantic_hint_counts")),
                ("local_judge_goldset_template", "risk_tag_counts", goldset.get("risk_tag_counts")),
                ("local_judge_goldset_template", "recommendation", goldset.get("recommendation")),
            ]
        )
    import_status = (
        summary.get("local_judge_goldset_import_status")
        if isinstance(summary.get("local_judge_goldset_import_status"), Mapping)
        else {}
    )
    progress = (
        summary.get("local_judge_goldset_progress_status")
        if isinstance(summary.get("local_judge_goldset_progress_status"), Mapping)
        else {}
    )
    if progress:
        rows.extend(
            [
                ("local_judge_goldset_progress", "status", progress.get("status")),
                ("local_judge_goldset_progress", "row_count", progress.get("row_count")),
                (
                    "local_judge_goldset_progress",
                    "rows_with_expected_label_count",
                    progress.get("rows_with_expected_label_count"),
                ),
                (
                    "local_judge_goldset_progress",
                    "expected_label_pending_count",
                    progress.get("expected_label_pending_count"),
                ),
                (
                    "local_judge_goldset_progress",
                    "invalid_expected_label_count",
                    progress.get("invalid_expected_label_count"),
                ),
                (
                    "local_judge_goldset_progress",
                    "annotation_completion_rate",
                    progress.get("annotation_completion_rate"),
                ),
                (
                    "local_judge_goldset_progress",
                    "ready_for_labeled_jsonl_import",
                    progress.get("ready_for_labeled_jsonl_import"),
                ),
                (
                    "local_judge_goldset_progress",
                    "annotation_ready",
                    progress.get("annotation_ready"),
                ),
                ("local_judge_goldset_progress", "source_label_counts", progress.get("source_label_counts")),
                ("local_judge_goldset_progress", "semantic_hint_counts", progress.get("semantic_hint_counts")),
                ("local_judge_goldset_progress", "risk_tag_counts", progress.get("risk_tag_counts")),
                (
                    "local_judge_goldset_progress",
                    "pending_source_label_counts",
                    progress.get("pending_source_label_counts"),
                ),
                (
                    "local_judge_goldset_progress",
                    "pending_semantic_hint_counts",
                    progress.get("pending_semantic_hint_counts"),
                ),
                (
                    "local_judge_goldset_progress",
                    "pending_risk_tag_counts",
                    progress.get("pending_risk_tag_counts"),
                ),
                ("local_judge_goldset_progress", "worklist_written", progress.get("worklist_written")),
                ("local_judge_goldset_progress", "worklist_row_count", progress.get("worklist_row_count")),
                ("local_judge_goldset_progress", "worklist_pending_count", progress.get("worklist_pending_count")),
                ("local_judge_goldset_progress", "worklist_invalid_count", progress.get("worklist_invalid_count")),
                ("local_judge_goldset_progress", "recommendation", progress.get("recommendation")),
            ]
        )
    labels_template = (
        summary.get("local_judge_goldset_labels_template_status")
        if isinstance(summary.get("local_judge_goldset_labels_template_status"), Mapping)
        else {}
    )
    if labels_template:
        rows.extend(
            [
                ("local_judge_goldset_labels_template", "status", labels_template.get("status")),
                ("local_judge_goldset_labels_template", "row_count", labels_template.get("row_count")),
                (
                    "local_judge_goldset_labels_template",
                    "expected_label_pending_count",
                    labels_template.get("expected_label_pending_count"),
                ),
                (
                    "local_judge_goldset_labels_template",
                    "rows_missing_id_count",
                    labels_template.get("rows_missing_id_count"),
                ),
                (
                    "local_judge_goldset_labels_template",
                    "rows_missing_text_hash_count",
                    labels_template.get("rows_missing_text_hash_count"),
                ),
                (
                    "local_judge_goldset_labels_template",
                    "duplicate_label_key_count",
                    labels_template.get("duplicate_label_key_count"),
                ),
                (
                    "local_judge_goldset_labels_template",
                    "label_status_counts",
                    labels_template.get("label_status_counts"),
                ),
                (
                    "local_judge_goldset_labels_template",
                    "ready_for_apply_labels",
                    labels_template.get("ready_for_apply_labels"),
                ),
                ("local_judge_goldset_labels_template", "recommendation", labels_template.get("recommendation")),
            ]
        )
    suggestions = (
        summary.get("local_judge_goldset_suggestions_status")
        if isinstance(summary.get("local_judge_goldset_suggestions_status"), Mapping)
        else {}
    )
    if suggestions:
        rows.extend(
            [
                ("local_judge_goldset_suggestions", "status", suggestions.get("status")),
                ("local_judge_goldset_suggestions", "suggestion_count", suggestions.get("suggestion_count")),
                ("local_judge_goldset_suggestions", "source_row_count", suggestions.get("source_row_count")),
                ("local_judge_goldset_suggestions", "model_called_count", suggestions.get("model_called_count")),
                (
                    "local_judge_goldset_suggestions",
                    "local_judge_error_count",
                    suggestions.get("local_judge_error_count"),
                ),
                (
                    "local_judge_goldset_suggestions",
                    "rows_missing_text_count",
                    suggestions.get("rows_missing_text_count"),
                ),
                (
                    "local_judge_goldset_suggestions",
                    "static_safety_clamp_count",
                    suggestions.get("static_safety_clamp_count"),
                ),
                (
                    "local_judge_goldset_suggestions",
                    "suggested_label_counts",
                    suggestions.get("suggested_label_counts"),
                ),
                ("local_judge_goldset_suggestions", "model_label_counts", suggestions.get("model_label_counts")),
                (
                    "local_judge_goldset_suggestions",
                    "local_judge_action_counts",
                    suggestions.get("local_judge_action_counts"),
                ),
                (
                    "local_judge_goldset_suggestions",
                    "suggestions_ready_for_manual_review",
                    suggestions.get("suggestions_ready_for_manual_review"),
                ),
                ("local_judge_goldset_suggestions", "ready_for_apply_labels", suggestions.get("ready_for_apply_labels")),
                ("local_judge_goldset_suggestions", "recommendation", suggestions.get("recommendation")),
            ]
        )
    suggestion_review = (
        summary.get("local_judge_goldset_suggestion_review_status")
        if isinstance(summary.get("local_judge_goldset_suggestion_review_status"), Mapping)
        else {}
    )
    if suggestion_review:
        rows.extend(
            [
                ("local_judge_goldset_suggestion_review", "status", suggestion_review.get("status")),
                ("local_judge_goldset_suggestion_review", "row_count", suggestion_review.get("row_count")),
                (
                    "local_judge_goldset_suggestion_review",
                    "suggestion_row_count",
                    suggestion_review.get("suggestion_row_count"),
                ),
                (
                    "local_judge_goldset_suggestion_review",
                    "matched_suggestion_count",
                    suggestion_review.get("matched_suggestion_count"),
                ),
                (
                    "local_judge_goldset_suggestion_review",
                    "missing_suggestion_count",
                    suggestion_review.get("missing_suggestion_count"),
                ),
                (
                    "local_judge_goldset_suggestion_review",
                    "unmatched_suggestion_count",
                    suggestion_review.get("unmatched_suggestion_count"),
                ),
                (
                    "local_judge_goldset_suggestion_review",
                    "expected_label_pending_count",
                    suggestion_review.get("expected_label_pending_count"),
                ),
                (
                    "local_judge_goldset_suggestion_review",
                    "suggested_label_counts",
                    suggestion_review.get("suggested_label_counts"),
                ),
                (
                    "local_judge_goldset_suggestion_review",
                    "ready_for_manual_review",
                    suggestion_review.get("ready_for_manual_review"),
                ),
                (
                    "local_judge_goldset_suggestion_review",
                    "ready_for_apply_labels",
                    suggestion_review.get("ready_for_apply_labels"),
                ),
                ("local_judge_goldset_suggestion_review", "recommendation", suggestion_review.get("recommendation")),
            ]
        )
    review_plan = (
        summary.get("local_judge_goldset_review_plan_status")
        if isinstance(summary.get("local_judge_goldset_review_plan_status"), Mapping)
        else {}
    )
    if review_plan:
        rows.extend(
            [
                ("local_judge_goldset_review_plan", "status", review_plan.get("status")),
                ("local_judge_goldset_review_plan", "plan_row_count", review_plan.get("plan_row_count")),
                (
                    "local_judge_goldset_review_plan",
                    "expected_label_pending_count",
                    review_plan.get("expected_label_pending_count"),
                ),
                (
                    "local_judge_goldset_review_plan",
                    "invalid_expected_label_count",
                    review_plan.get("invalid_expected_label_count"),
                ),
                (
                    "local_judge_goldset_review_plan",
                    "plan_disagreement_row_count",
                    review_plan.get("plan_disagreement_row_count"),
                ),
                (
                    "local_judge_goldset_review_plan",
                    "plan_high_risk_row_count",
                    review_plan.get("plan_high_risk_row_count"),
                ),
                (
                    "local_judge_goldset_review_plan",
                    "plan_review_reason_counts",
                    review_plan.get("plan_review_reason_counts"),
                ),
                (
                    "local_judge_goldset_review_plan",
                    "suggested_labels_not_auto_applied",
                    review_plan.get("suggested_labels_not_auto_applied"),
                ),
                ("local_judge_goldset_review_plan", "recommendation", review_plan.get("recommendation")),
            ]
        )
    labels_from_csv = (
        summary.get("local_judge_goldset_labels_from_csv_status")
        if isinstance(summary.get("local_judge_goldset_labels_from_csv_status"), Mapping)
        else {}
    )
    if labels_from_csv:
        rows.extend(
            [
                ("local_judge_goldset_labels_from_csv", "status", labels_from_csv.get("status")),
                ("local_judge_goldset_labels_from_csv", "row_count", labels_from_csv.get("row_count")),
                (
                    "local_judge_goldset_labels_from_csv",
                    "valid_label_row_count",
                    labels_from_csv.get("valid_label_row_count"),
                ),
                (
                    "local_judge_goldset_labels_from_csv",
                    "expected_label_pending_count",
                    labels_from_csv.get("expected_label_pending_count"),
                ),
                (
                    "local_judge_goldset_labels_from_csv",
                    "expected_label_counts",
                    labels_from_csv.get("expected_label_counts"),
                ),
                (
                    "local_judge_goldset_labels_from_csv",
                    "suggested_label_counts",
                    labels_from_csv.get("suggested_label_counts"),
                ),
                (
                    "local_judge_goldset_labels_from_csv",
                    "ready_for_apply_labels",
                    labels_from_csv.get("ready_for_apply_labels"),
                ),
                (
                    "local_judge_goldset_labels_from_csv",
                    "suggested_labels_not_auto_applied",
                    labels_from_csv.get("suggested_labels_not_auto_applied"),
                ),
                ("local_judge_goldset_labels_from_csv", "recommendation", labels_from_csv.get("recommendation")),
            ]
        )
    labels_validation = (
        summary.get("local_judge_goldset_labels_validation_status")
        if isinstance(summary.get("local_judge_goldset_labels_validation_status"), Mapping)
        else {}
    )
    if labels_validation:
        rows.extend(
            [
                ("local_judge_goldset_labels_validation", "status", labels_validation.get("status")),
                ("local_judge_goldset_labels_validation", "row_count", labels_validation.get("row_count")),
                (
                    "local_judge_goldset_labels_validation",
                    "valid_label_row_count",
                    labels_validation.get("valid_label_row_count"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "expected_label_pending_count",
                    labels_validation.get("expected_label_pending_count"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "invalid_label_row_count",
                    labels_validation.get("invalid_label_row_count"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "duplicate_label_key_count",
                    labels_validation.get("duplicate_label_key_count"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "expected_label_counts",
                    labels_validation.get("expected_label_counts"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "source_expected_label_match_count",
                    labels_validation.get("source_expected_label_match_count"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "source_expected_label_mismatch_count",
                    labels_validation.get("source_expected_label_mismatch_count"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "source_expected_label_audited_count",
                    labels_validation.get("source_expected_label_audited_count"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "source_expected_label_match_rate",
                    labels_validation.get("source_expected_label_match_rate"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "source_expected_label_mismatch_counts",
                    labels_validation.get("source_expected_label_mismatch_counts"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "possible_rule_self_confirmation",
                    labels_validation.get("possible_rule_self_confirmation"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "gold_labels_independent_from_rules_ready",
                    labels_validation.get("gold_labels_independent_from_rules_ready"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "ready_for_apply_labels",
                    labels_validation.get("ready_for_apply_labels"),
                ),
                (
                    "local_judge_goldset_labels_validation",
                    "ready_for_goldset_import_after_apply",
                    labels_validation.get("ready_for_goldset_import_after_apply"),
                ),
                ("local_judge_goldset_labels_validation", "recommendation", labels_validation.get("recommendation")),
            ]
        )
    if import_status:
        rows.extend(
            [
                ("local_judge_goldset_import", "status", import_status.get("status")),
                ("local_judge_goldset_import", "row_count", import_status.get("row_count")),
                (
                    "local_judge_goldset_import",
                    "expected_label_pending_count",
                    import_status.get("expected_label_pending_count"),
                ),
                (
                    "local_judge_goldset_import",
                    "invalid_expected_label_count",
                    import_status.get("invalid_expected_label_count"),
                ),
                ("local_judge_goldset_import", "rows_missing_text_count", import_status.get("rows_missing_text_count")),
                ("local_judge_goldset_import", "output_written", import_status.get("output_written")),
                ("local_judge_goldset_import", "ready_for_validate", import_status.get("ready_for_validate")),
                ("local_judge_goldset_import", "recommendation", import_status.get("recommendation")),
            ]
        )
    validation = (
        summary.get("local_judge_goldset_validation_status")
        if isinstance(summary.get("local_judge_goldset_validation_status"), Mapping)
        else {}
    )
    if validation:
        rows.extend(
            [
                ("local_judge_goldset_validation", "status", validation.get("status")),
                ("local_judge_goldset_validation", "row_count", validation.get("row_count")),
                ("local_judge_goldset_validation", "valid_row_count", validation.get("valid_row_count")),
                ("local_judge_goldset_validation", "invalid_row_count", validation.get("invalid_row_count")),
                ("local_judge_goldset_validation", "missing_text_count", validation.get("missing_text_count")),
                (
                    "local_judge_goldset_validation",
                    "invalid_expected_label_count",
                    validation.get("invalid_expected_label_count"),
                ),
                ("local_judge_goldset_validation", "expected_label_counts", validation.get("expected_label_counts")),
                ("local_judge_goldset_validation", "recommendation", validation.get("recommendation")),
            ]
        )
    quality = (
        summary.get("local_judge_quality_eval_status")
        if isinstance(summary.get("local_judge_quality_eval_status"), Mapping)
        else {}
    )
    if quality:
        rows.extend(
            [
                ("local_judge_quality_eval", "status", quality.get("status")),
                ("local_judge_quality_eval", "sample_count", quality.get("sample_count")),
                ("local_judge_quality_eval", "evaluated_count", quality.get("evaluated_count")),
                ("local_judge_quality_eval", "model_called_count", quality.get("model_called_count")),
                ("local_judge_quality_eval", "accuracy", quality.get("accuracy")),
                ("local_judge_quality_eval", "macro_f1", quality.get("macro_f1")),
                ("local_judge_quality_eval", "raw_model_accuracy", quality.get("raw_model_accuracy")),
                ("local_judge_quality_eval", "raw_model_macro_f1", quality.get("raw_model_macro_f1")),
                (
                    "local_judge_quality_eval",
                    "static_safety_clamp_count",
                    quality.get("static_safety_clamp_count"),
                ),
                (
                    "local_judge_quality_eval",
                    "static_safety_clamp_corrected_count",
                    quality.get("static_safety_clamp_corrected_count"),
                ),
                (
                    "local_judge_quality_eval",
                    "static_safety_clamp_worsened_count",
                    quality.get("static_safety_clamp_worsened_count"),
                ),
                (
                    "local_judge_quality_eval",
                    "static_safety_clamp_net_correct_delta",
                    quality.get("static_safety_clamp_net_correct_delta"),
                ),
                ("local_judge_quality_eval", "min_samples", quality.get("min_samples")),
                ("local_judge_quality_eval", "min_accuracy", quality.get("min_accuracy")),
                ("local_judge_quality_eval", "min_macro_f1", quality.get("min_macro_f1")),
                ("local_judge_quality_eval", "parse_error_count", quality.get("parse_error_count")),
                ("local_judge_quality_eval", "low_confidence_count", quality.get("low_confidence_count")),
                ("local_judge_quality_eval", "expected_label_counts", quality.get("expected_label_counts")),
                ("local_judge_quality_eval", "predicted_label_counts", quality.get("predicted_label_counts")),
                ("local_judge_quality_eval", "raw_model_label_counts", quality.get("raw_model_label_counts")),
                (
                    "local_judge_quality_eval",
                    "raw_model_to_predicted_label_counts",
                    quality.get("raw_model_to_predicted_label_counts"),
                ),
                ("local_judge_quality_eval", "failure_mode_codes", quality.get("failure_mode_codes")),
                ("local_judge_quality_eval", "recommendation", quality.get("recommendation")),
            ]
        )
        diagnostics = (
            quality.get("quality_diagnostics")
            if isinstance(quality.get("quality_diagnostics"), Mapping)
            else {}
        )
        if diagnostics:
            rows.extend(
                [
                    (
                        "local_judge_quality_eval",
                        "predicted_label_coverage_count",
                        diagnostics.get("predicted_label_coverage_count"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "raw_model_label_coverage_count",
                        diagnostics.get("raw_model_label_coverage_count"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "missing_predicted_labels",
                        diagnostics.get("missing_predicted_labels"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "missing_raw_model_labels",
                        diagnostics.get("missing_raw_model_labels"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "predicted_label_collapse",
                        diagnostics.get("predicted_label_collapse"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "raw_model_label_collapse",
                        diagnostics.get("raw_model_label_collapse"),
                    ),
                    ("local_judge_quality_eval", "reject_recall", diagnostics.get("reject_recall")),
                    (
                        "local_judge_quality_eval",
                        "raw_model_reject_recall",
                        diagnostics.get("raw_model_reject_recall"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "accept_false_positive_count",
                        diagnostics.get("accept_false_positive_count"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "raw_model_accept_false_positive_count",
                        diagnostics.get("raw_model_accept_false_positive_count"),
                    ),
                ]
            )
        mismatch_diagnostics = (
            quality.get("mismatch_diagnostics")
            if isinstance(quality.get("mismatch_diagnostics"), Mapping)
            else {}
        )
        if mismatch_diagnostics:
            rows.extend(
                [
                    (
                        "local_judge_quality_eval",
                        "mismatch_label_transition_counts",
                        mismatch_diagnostics.get("mismatch_label_transition_counts"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "mismatch_semantic_hint_counts",
                        mismatch_diagnostics.get("mismatch_semantic_hint_counts"),
                    ),
                    (
                        "local_judge_quality_eval",
                        "mismatch_risk_tag_count_counts",
                        mismatch_diagnostics.get("mismatch_risk_tag_count_counts"),
                    ),
                ]
            )
    policy_eval = (
        summary.get("local_judge_policy_eval_status")
        if isinstance(summary.get("local_judge_policy_eval_status"), Mapping)
        else {}
    )
    if policy_eval:
        rows.extend(
            [
                ("local_judge_policy_eval", "status", policy_eval.get("status")),
                ("local_judge_policy_eval", "row_count", policy_eval.get("row_count")),
                (
                    "local_judge_policy_eval",
                    "joined_prediction_count",
                    policy_eval.get("joined_prediction_count"),
                ),
                (
                    "local_judge_policy_eval",
                    "missing_prediction_count",
                    policy_eval.get("missing_prediction_count"),
                ),
                ("local_judge_policy_eval", "best_ready_policy", policy_eval.get("best_ready_policy")),
                ("local_judge_policy_eval", "best_overall_policy", policy_eval.get("best_overall_policy")),
                ("local_judge_policy_eval", "policy_metric_table", policy_eval.get("policy_metric_table")),
                (
                    "local_judge_policy_eval",
                    "policy_failure_diagnostics",
                    policy_eval.get("policy_failure_diagnostics"),
                ),
                ("local_judge_policy_eval", "recommendation", policy_eval.get("recommendation")),
            ]
        )
    static_calibration = (
        summary.get("static_rule_calibration_eval_status")
        if isinstance(summary.get("static_rule_calibration_eval_status"), Mapping)
        else {}
    )
    if static_calibration:
        rows.extend(
            [
                ("static_rule_calibration_eval", "status", static_calibration.get("status")),
                ("static_rule_calibration_eval", "row_count", static_calibration.get("row_count")),
                (
                    "static_rule_calibration_eval",
                    "exploratory_ready",
                    static_calibration.get("exploratory_ready"),
                ),
                ("static_rule_calibration_eval", "ready", static_calibration.get("ready")),
                (
                    "static_rule_calibration_eval",
                    "best_ready_policy",
                    static_calibration.get("best_ready_policy"),
                ),
                (
                    "static_rule_calibration_eval",
                    "best_production_candidate_policy",
                    static_calibration.get("best_production_candidate_policy"),
                ),
                (
                    "static_rule_calibration_eval",
                    "baseline_rule_accuracy",
                    static_calibration.get("baseline_rule_accuracy"),
                ),
                (
                    "static_rule_calibration_eval",
                    "baseline_rule_macro_f1",
                    static_calibration.get("baseline_rule_macro_f1"),
                ),
                (
                    "static_rule_calibration_eval",
                    "top_policy_metric_table",
                    static_calibration.get("top_policy_metric_table"),
                ),
                (
                    "static_rule_calibration_eval",
                    "production_readiness_diagnostics",
                    static_calibration.get("production_readiness_diagnostics"),
                ),
                (
                    "static_rule_calibration_eval",
                    "gold_support_diagnostics",
                    static_calibration.get("gold_support_diagnostics"),
                ),
                (
                    "static_rule_calibration_eval",
                    "validator_rule_candidates",
                    static_calibration.get("validator_rule_candidates"),
                ),
                (
                    "static_rule_calibration_eval",
                    "validator_review_plan",
                    (
                        static_calibration.get("validator_rule_candidates", {}).get("validator_review_plan")
                        if isinstance(static_calibration.get("validator_rule_candidates"), Mapping)
                        else None
                    ),
                ),
                (
                    "static_rule_calibration_eval",
                    "conservative_fail_closed_subset",
                    (
                        static_calibration.get("validator_rule_candidates", {}).get(
                            "conservative_fail_closed_subset"
                        )
                        if isinstance(static_calibration.get("validator_rule_candidates"), Mapping)
                        else None
                    ),
                ),
                (
                    "static_rule_calibration_eval",
                    "shadow_trial_plan",
                    (
                        static_calibration.get("validator_rule_candidates", {}).get("shadow_trial_plan")
                        if isinstance(static_calibration.get("validator_rule_candidates"), Mapping)
                        else None
                    ),
                ),
                (
                    "static_rule_calibration_eval",
                    "source_generalization_diagnostics",
                    (
                        static_calibration.get("validator_rule_candidates", {}).get(
                            "source_generalization_diagnostics"
                        )
                        if isinstance(static_calibration.get("validator_rule_candidates"), Mapping)
                        else None
                    ),
                ),
                ("static_rule_calibration_eval", "recommendation", static_calibration.get("recommendation")),
            ]
        )
    chain = (
        summary.get("local_judge_goldset_chain_status")
        if isinstance(summary.get("local_judge_goldset_chain_status"), Mapping)
        else {}
    )
    if chain:
        rows.extend(
            [
                ("local_judge_goldset_chain", "status", chain.get("status")),
                ("local_judge_goldset_chain", "step_ready_counts", chain.get("step_ready_counts")),
                ("local_judge_goldset_chain", "failed_check_count", chain.get("failed_check_count")),
                ("local_judge_goldset_chain", "blocking_reasons", chain.get("blocking_reasons")),
                ("local_judge_goldset_chain", "next_action", chain.get("next_action")),
            ]
        )
    guard = (
        summary.get("semantic_guard_fake_smoke_status")
        if isinstance(summary.get("semantic_guard_fake_smoke_status"), Mapping)
        else {}
    )
    if guard:
        rows.extend(
            [
                ("semantic_guard_fake_smoke", "status", guard.get("status")),
                ("semantic_guard_fake_smoke", "fake_upstream", guard.get("fake_upstream")),
                ("semantic_guard_fake_smoke", "fake_local_judge", guard.get("fake_local_judge")),
                ("semantic_guard_fake_smoke", "request_count", guard.get("request_count")),
                ("semantic_guard_fake_smoke", "judge_request_count", guard.get("judge_request_count")),
                ("semantic_guard_fake_smoke", "upstream_request_count", guard.get("upstream_request_count")),
                (
                    "semantic_guard_fake_smoke",
                    "adapter_validation_reasons",
                    guard.get("adapter_validation_reasons"),
                ),
            ]
        )
    for key, label in (
        ("offline_local_judge_matrix_status", "offline_local_judge_matrix"),
        ("offline_local_judge_matrix_smoke_status", "offline_local_judge_matrix_smoke"),
    ):
        status = summary.get(key) if isinstance(summary.get(key), Mapping) else {}
        if not status:
            continue
        diagnostics = (
            status.get("candidate_label_diagnostics")
            if isinstance(status.get("candidate_label_diagnostics"), Mapping)
            else {}
        )
        effectiveness = (
            diagnostics.get("local_judge_effectiveness_diagnostics")
            if isinstance(diagnostics.get("local_judge_effectiveness_diagnostics"), Mapping)
            else {}
        )
        rows.extend(
            [
                (label, "status", status.get("status")),
                (label, "completed_source_count", status.get("completed_source_count")),
                (label, "supported_source_count", status.get("supported_source_count")),
                (label, "prompt_extraction_counts", _diagnostic_extraction_counts(status)),
                (label, "rule_label_counts", diagnostics.get("rule_label_counts")),
                (label, "model_to_final_label_counts", diagnostics.get("model_to_final_label_counts")),
                (label, "static_safety_clamp_count", diagnostics.get("static_safety_clamp_count")),
                (label, "rule_review_candidate_count", effectiveness.get("rule_review_candidate_count")),
                (label, "model_called_rule_review_count", effectiveness.get("model_called_rule_review_count")),
                (label, "resolved_rule_review_count", effectiveness.get("resolved_rule_review_count")),
                (label, "remaining_rule_review_count", effectiveness.get("remaining_rule_review_count")),
                (label, "resolution_rate", effectiveness.get("resolution_rate")),
                (label, "called_resolution_rate", effectiveness.get("called_resolution_rate")),
            ]
        )
    if not rows:
        return []
    lines = [
        "",
        "## Offline Evidence Diagnostics",
        "",
        "| Source | Field | Value |",
        "|---|---|---|",
    ]
    lines.extend(f"| {source} | {field} | {_format_value(value)} |" for source, field, value in rows)
    return lines


def _diagnostic_extraction_counts(value: Mapping[str, Any]) -> Any:
    diagnostics = value.get("prompt_extraction_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    return diagnostics.get("extraction_counts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a prompt-safe runbook for a generated legacy AutoGenBench suite.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--report-md")
    parser.add_argument("--require-api-config", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_legacy_agbench_runbook(
        manifest_path=args.manifest,
        summary_path=args.summary,
        report_path=args.report_md,
        require_api_config=args.require_api_config,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    if args.require_api_config and not result.summary.get("ready_to_start_real_run"):
        return 1
    return 0 if result.summary.get("structural_ready") else 1


def _checks(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
    env: Mapping[str, str],
    require_api_config: bool,
) -> list[dict[str, Any]]:
    checks = [
        _check("manifest_schema", manifest.get("schema_version") == "prefix-legacy-agbench-suite-manifest-v1", str(manifest.get("schema_version"))),
        _check("manifest_prompt_safe", manifest.get("prompt_safe_manifest") is True, str(manifest.get("prompt_safe_manifest"))),
        _check(
            "manifest_secret_policy",
            bool((manifest.get("secret_policy") or {}).get("api_key_value_not_written"))
            if isinstance(manifest.get("secret_policy"), Mapping)
            else False,
            "api_key_value_not_written=true expected",
        ),
    ]
    api_markers = []
    if env.get("OPENAI_API_KEY"):
        api_markers.append("OPENAI_API_KEY")
    if env.get("OAI_CONFIG_LIST"):
        api_markers.append("OAI_CONFIG_LIST env")
    if _manifest_targets_deepseek_upstream(manifest):
        project_config = load_project_provider_config(cwd=manifest_file.parent.parent, env=env)
        if project_config is None:
            project_config = load_project_provider_config(cwd=Path.cwd(), env=env)
        if project_config is not None:
            api_markers.append(project_config.redacted_marker())
    api_present = bool(api_markers)
    api_check = _check(
        "api_config",
        api_present or not require_api_config,
        "present: " + ", ".join(api_markers)
        if api_present
        else "missing OPENAI_API_KEY/OAI_CONFIG_LIST/project DeepSeek config",
    )
    api_check["present"] = api_present
    checks.append(api_check)

    config_lists = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    for label in (*VARIANTS, "combined"):
        info = config_lists.get(label) if isinstance(config_lists.get(label), Mapping) else {}
        path = _artifact_path(info.get("path"), manifest_file=manifest_file)
        checks.append(_check(f"oai_config:{label}", bool(path and path.exists()), str(path) if path else "missing path"))

    scenario_files = manifest.get("scenario_files") if isinstance(manifest.get("scenario_files"), Mapping) else {}
    result_dir_names: list[str] = []
    for label in VARIANTS:
        info = scenario_files.get(label) if isinstance(scenario_files.get(label), Mapping) else {}
        path = _artifact_path(info.get("path"), manifest_file=manifest_file)
        checks.append(_check(f"scenario:{label}", bool(path and path.exists()), str(path) if path else "missing path"))
        if isinstance(info.get("result_dir_name"), str):
            result_dir_names.append(str(info["result_dir_name"]))
    checks.append(
        _check(
            "scenario_result_dirs_distinct",
            len(result_dir_names) == 3 and len(set(result_dir_names)) == 3,
            ", ".join(result_dir_names) if result_dir_names else "missing",
        )
    )
    return checks


def _manifest_targets_deepseek_upstream(manifest: Mapping[str, Any]) -> bool:
    config_lists = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    for info in config_lists.values():
        if isinstance(info, Mapping) and "api.deepseek.com" in str(info.get("upstream_base_url") or "").lower():
            return True
    proxies = manifest.get("proxies") if isinstance(manifest.get("proxies"), Mapping) else {}
    for info in proxies.values():
        if isinstance(info, Mapping) and "api.deepseek.com" in str(info.get("upstream_base_url") or "").lower():
            return True
    return False


def _runbook_steps(manifest: Mapping[str, Any], *, manifest_file: Path) -> list[dict[str, Any]]:
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    commands = _commands_with_compat_defaults(manifest=manifest, manifest_file=manifest_file, commands=commands)
    return [
        _step(1, "verify_suite", "Verify generated OAI configs, scenarios, proxy metadata, and prompt-safe manifest.", commands.get("verify")),
        _step(
            2,
            "preflight_fake_smoke",
            (
                "Run the no-network three-proxy fake-upstream smoke into the manifest artifact paths. "
                "This validates wiring only and must be followed by real provider runs."
            ),
            commands.get("preflight_fake_smoke"),
        ),
        _step(
            3,
            "semantic_guard_fake_smoke",
            (
                "Run the no-network fake local semantic judge smoke against the forwarding proxy. "
                "This validates guard fail-closed wiring only and does not prove real semantic model quality."
            ),
            commands.get("semantic_guard_fake_smoke"),
        ),
        _step(
            4,
            "offline_semantic_suite",
            (
                "Run rule-only and nl-segmentation offline source-prompt coverage over a real local source tree. "
                "This validates rule coverage before spending provider budget and does not prove provider metrics."
            ),
            commands.get("offline_semantic_suite"),
        ),
        _step(
            5,
            "local_judge_healthcheck",
            (
                "Healthcheck the real OpenAI-compatible local judge endpoint with one synthetic candidate. "
                "This validates endpoint/model/schema before any candidate-text batch."
            ),
            commands.get("local_judge_healthcheck"),
        ),
        _step(
            6,
            "local_judge_goldset_template",
            (
                "Build a sampled human-labeling template from real semantic candidates/review worklists. "
                "The default artifact is metadata-only; use the with-text and CSV commands locally for annotation."
            ),
            commands.get("local_judge_goldset_template"),
        ),
        _step(
            7,
            "local_judge_goldset_validate",
            (
                "Validate the manually labeled local gold JSONL before model calls. "
                "This checks text, expected labels, sample count, and label balance without sending prompts anywhere."
            ),
            commands.get("local_judge_goldset_validate"),
        ),
        _step(
            8,
            "local_judge_quality_eval",
            (
                "Evaluate the configured local judge against a gold label JSONL. "
                "This is the only local-model semantic-quality claim gate and remains scoped to the supplied gold set."
            ),
            commands.get("local_judge_quality_eval"),
        ),
        _step(
            9,
            "local_judge_policy_eval",
            (
                "Compare static rule labels, raw local-model labels, clamped model labels, and conservative "
                "hybrid policies against the same gold set. This is diagnostic evidence for Validator design."
            ),
            commands.get("local_judge_policy_eval"),
        ),
        _step(
            10,
            "static_rule_calibration_eval",
            (
                "Evaluate prompt-safe metadata-only static rule calibration with leave-one-out cross-validation. "
                "This diagnoses whether semantic_hint/risk_tag patterns generalize in the gold set before Validator promotion."
            ),
            commands.get("static_rule_calibration_eval"),
        ),
        _step(
            11,
            "local_judge_goldset_chain_verify",
            (
                "Read prompt-safe summaries for labels-from-csv, validate-labels, apply-labels, import, "
                "gold-set validation, and quality eval. This verifies the local human-label evidence chain."
            ),
            commands.get("local_judge_goldset_chain_verify"),
        ),
        _step(
            12,
            "offline_local_judge_matrix",
            (
                "Run a real local-judge semantic matrix over framework source trees. "
                "This sends candidate text to the configured local OpenAI-compatible judge."
            ),
            commands.get("offline_local_judge_matrix"),
        ),
        _step(
            13,
            "offline_local_judge_matrix_smoke",
            (
                "Run a no-network fake local-judge matrix over real source trees. "
                "This validates local-judge wiring and budget coverage before spending provider budget."
            ),
            commands.get("offline_local_judge_matrix_smoke"),
        ),
        _step(14, "start_three_proxies", "Start baseline, rule-only, and nl-segmentation proxies in separate long-running terminals.", commands.get("proxy")),
        _step(15, "run_autogenbench_variants", "Run official AutoGenBench once per generated scenario file to avoid result directory collisions.", commands.get("autogenbench")),
        _step(16, "tabulate_results", "Convert each AutoGenBench Results directory into a CSV table.", commands.get("tabulate")),
        _step(17, "collect_and_compare", "Flatten tabulate CSV repeats, join provider telemetry, and produce two A/B reports.", commands.get("collect")),
    ]


def _step(order: int, name: str, purpose: str, commands: Any) -> dict[str, Any]:
    command_list = [_redact_command(str(command)) for command in commands] if isinstance(commands, list) else []
    return {
        "order": order,
        "name": name,
        "purpose": purpose,
        "commands": command_list,
        "command_count": len(command_list),
        "manual_long_running": name == "start_three_proxies",
    }


def _commands_with_compat_defaults(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
    commands: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(commands)
    manifest_path = str(manifest_file)
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    static_existing = result.get("static_rule_calibration_eval")
    if not isinstance(static_existing, list) or not static_existing:
        goldset_path = local_judge.get("goldset_labeled_path")
        static_summary_path = local_judge.get("static_rule_calibration_eval_summary_path")
        static_report_path = local_judge.get("static_rule_calibration_eval_report_path")
        base_dir = manifest_file.parent
        if not isinstance(goldset_path, str) or not goldset_path:
            goldset_path = str(base_dir / "artifacts" / "local_judge_goldset_labeled.local.jsonl")
        if not isinstance(static_summary_path, str) or not static_summary_path:
            static_summary_path = str(base_dir / "artifacts" / "static_rule_calibration_eval_summary.json")
        if not isinstance(static_report_path, str) or not static_report_path:
            static_report_path = str(base_dir / "reports" / "static_rule_calibration_eval.md")
        result["static_rule_calibration_eval"] = [
            (
                rf".venv\Scripts\python.exe -m autogen_prefix_tree.static_rule_calibration_eval "
                rf"--input {goldset_path} "
                r"--min-accuracy 0.75 "
                r"--min-macro-f1 0.70 "
                r"--production-min-support 2 "
                r"--production-min-purity 0.67 "
                rf"--summary {static_summary_path} "
                rf"--report-md {static_report_path}"
            )
        ]
    existing = result.get("local_judge_goldset_chain_verify")
    if isinstance(existing, list) and existing:
        return result
    summary_path = local_judge.get("goldset_chain_verification_summary_path")
    report_path = local_judge.get("goldset_chain_verification_report_path")
    base_dir = manifest_file.parent
    if not isinstance(summary_path, str) or not summary_path:
        summary_path = str(base_dir / "artifacts" / "local_judge_goldset_chain_verification_summary.json")
    if not isinstance(report_path, str) or not report_path:
        report_path = str(base_dir / "reports" / "local_judge_goldset_chain_verification.md")
    result["local_judge_goldset_chain_verify"] = [
        (
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.local_judge_goldset_chain_verify "
            rf"--manifest {manifest_path} "
            rf"--summary {summary_path} "
            rf"--report-md {report_path}"
        )
    ]
    return result


def _artifact_status(*, manifest: Mapping[str, Any], manifest_file: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    config_lists = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    for label in VARIANTS:
        info = config_lists.get(label) if isinstance(config_lists.get(label), Mapping) else {}
        for field in ("provider_telemetry_path", "tabulate_csv_path", "task_results_path"):
            artifacts.append(_artifact_record(name=f"{label}:{field}", value=info.get(field), manifest_file=manifest_file))
    recommended = manifest.get("recommended_ab_eval") if isinstance(manifest.get("recommended_ab_eval"), Mapping) else {}
    for field in ("rule_only_summary", "rule_only_report_md", "nl_segmentation_summary", "nl_segmentation_report_md"):
        artifacts.append(_artifact_record(name=f"ab_eval:{field}", value=recommended.get(field), manifest_file=manifest_file))
    semantic_guard_smoke_path = _semantic_guard_fake_smoke_summary_path(manifest=manifest, manifest_file=manifest_file)
    artifacts.append(
        _artifact_record(
            name="semantic_guard_fake_smoke:summary",
            value=semantic_guard_smoke_path,
            manifest_file=manifest_file,
        )
    )
    offline_semantic_suite_path = _offline_semantic_suite_summary_path(manifest=manifest, manifest_file=manifest_file)
    artifacts.append(
        _artifact_record(
            name="offline_semantic_suite:summary",
            value=offline_semantic_suite_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_healthcheck_path = _local_judge_healthcheck_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_healthcheck:summary",
            value=local_judge_healthcheck_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_template_path = _local_judge_goldset_template_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_template:summary",
            value=local_judge_goldset_template_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_progress_path = _local_judge_goldset_progress_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_progress:summary",
            value=local_judge_goldset_progress_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_labels_template_path = _local_judge_goldset_labels_template_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_labels_template:summary",
            value=local_judge_goldset_labels_template_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_suggestions_path = _local_judge_goldset_suggestions_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_suggestions:summary",
            value=local_judge_goldset_suggestions_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_suggestion_review_path = _local_judge_goldset_suggestion_review_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_suggestion_review:summary",
            value=local_judge_goldset_suggestion_review_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_review_plan_path = _local_judge_goldset_review_plan_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_review_plan:summary",
            value=local_judge_goldset_review_plan_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_labels_from_csv_path = _local_judge_goldset_labels_from_csv_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_labels_from_csv:summary",
            value=local_judge_goldset_labels_from_csv_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_labels_validation_path = _local_judge_goldset_labels_validation_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_labels_validation:summary",
            value=local_judge_goldset_labels_validation_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_import_path = _local_judge_goldset_import_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_import:summary",
            value=local_judge_goldset_import_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_validation_path = _local_judge_goldset_validation_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_validation:summary",
            value=local_judge_goldset_validation_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_quality_eval_path = _local_judge_quality_eval_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_quality_eval:summary",
            value=local_judge_quality_eval_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_policy_eval_path = _local_judge_policy_eval_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_policy_eval:summary",
            value=local_judge_policy_eval_path,
            manifest_file=manifest_file,
        )
    )
    static_rule_calibration_eval_path = _static_rule_calibration_eval_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="static_rule_calibration_eval:summary",
            value=static_rule_calibration_eval_path,
            manifest_file=manifest_file,
        )
    )
    local_judge_goldset_chain_path = _local_judge_goldset_chain_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="local_judge_goldset_chain:summary",
            value=local_judge_goldset_chain_path,
            manifest_file=manifest_file,
        )
    )
    offline_local_judge_matrix_path = _offline_local_judge_matrix_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="offline_local_judge_matrix:summary",
            value=offline_local_judge_matrix_path,
            manifest_file=manifest_file,
        )
    )
    offline_local_judge_matrix_path = _offline_local_judge_matrix_smoke_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
    )
    artifacts.append(
        _artifact_record(
            name="offline_local_judge_matrix_smoke:summary",
            value=offline_local_judge_matrix_path,
            manifest_file=manifest_file,
        )
    )
    return artifacts


def _artifact_quality_status(artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_artifact_quality_record(artifact) for artifact in artifacts]


def _is_provider_or_ab_fake_artifact(item: Mapping[str, Any]) -> bool:
    name = str(item.get("name") or "")
    if name == "semantic_guard_fake_smoke:summary":
        return False
    if name == "offline_semantic_suite:summary":
        return False
    if name == "local_judge_healthcheck:summary":
        return False
    if name == "local_judge_goldset_template:summary":
        return False
    if name == "local_judge_goldset_progress:summary":
        return False
    if name == "local_judge_goldset_labels_template:summary":
        return False
    if name == "local_judge_goldset_suggestions:summary":
        return False
    if name == "local_judge_goldset_suggestion_review:summary":
        return False
    if name == "local_judge_goldset_review_plan:summary":
        return False
    if name == "local_judge_goldset_labels_from_csv:summary":
        return False
    if name == "local_judge_goldset_labels_validation:summary":
        return False
    if name == "local_judge_goldset_import:summary":
        return False
    if name == "local_judge_goldset_validation:summary":
        return False
    if name == "local_judge_quality_eval:summary":
        return False
    if name == "local_judge_policy_eval:summary":
        return False
    if name == "static_rule_calibration_eval:summary":
        return False
    if name == "local_judge_goldset_chain:summary":
        return False
    if name == "offline_local_judge_matrix:summary":
        return False
    if name == "offline_local_judge_matrix_smoke:summary":
        return False
    return bool(item.get("fake_upstream"))


def _shadow_trial_wiring_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    plugin_items = [
        item
        for item in artifact_quality
        if isinstance(item, Mapping)
        and str(item.get("name") or "").endswith(":provider_telemetry_path")
        and str(item.get("name") or "").startswith(("plugin_rule_only", "plugin_nl_segmentation"))
    ]
    supported_items = [item for item in plugin_items if item.get("shadow_trial_supported") is True]
    record_count = sum(_num(item.get("shadow_trial_record_count")) for item in supported_items)
    record_with_match_count = sum(_num(item.get("shadow_trial_record_with_match_count")) for item in supported_items)
    matched_rule_observation_count = sum(
        _num(item.get("shadow_trial_matched_rule_observation_count")) for item in supported_items
    )
    unique_rule_ids: set[str] = set()
    action_counts: dict[str, int] = {}
    for item in supported_items:
        for rule_id in item.get("shadow_trial_unique_matched_rule_ids", []) if isinstance(item.get("shadow_trial_unique_matched_rule_ids"), list) else []:
            unique_rule_ids.add(str(rule_id))
        counts = item.get("shadow_trial_action_counts") if isinstance(item.get("shadow_trial_action_counts"), Mapping) else {}
        for action, count in counts.items():
            action_counts[str(action)] = action_counts.get(str(action), 0) + _num(count)
    ready = bool(record_with_match_count > 0 and matched_rule_observation_count > 0)
    return {
        "ready": ready,
        "status": "pass" if ready else "pending",
        "evidence": (
            "shadow trial telemetry matched rules in plugin provider telemetry"
            if ready
            else "no plugin shadow trial rule-match telemetry observed"
        ),
        "provider_artifact_count": len(plugin_items),
        "shadow_trial_supported_artifact_count": len(supported_items),
        "shadow_trial_record_count": record_count,
        "shadow_trial_record_with_match_count": record_with_match_count,
        "shadow_trial_matched_rule_observation_count": matched_rule_observation_count,
        "shadow_trial_unique_matched_rule_count": len(unique_rule_ids),
        "shadow_trial_action_counts": dict(sorted(action_counts.items())),
        "validator_behavior_change_allowed": False,
        "safe_for_automatic_validator_promotion": False,
        "limits": "Shadow trial wiring is metadata-only and does not prove Validator quality or provider utility.",
    }


def _semantic_guard_fake_smoke_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "semantic_guard_fake_smoke:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "semantic guard fake smoke artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "warn" and item.get("fake_local_judge") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "fake_upstream": bool(item.get("fake_upstream")),
        "fake_local_judge": bool(item.get("fake_local_judge")),
        "request_count": _num(item.get("request_count")),
        "judge_request_count": _num(item.get("judge_request_count")),
        "upstream_request_count": _num(item.get("upstream_request_count")),
        "adapter_validation_reasons": item.get("adapter_validation_reasons")
        if isinstance(item.get("adapter_validation_reasons"), list)
        else [],
    }


def _offline_semantic_suite_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "offline_semantic_suite:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "offline semantic suite artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass",
        "status": status,
        "evidence": item.get("evidence"),
        "recommendation": item.get("recommendation"),
        "rule_only_applied_count": _num(item.get("rule_only_applied_count")),
        "nl_segmentation_applied_count": _num(item.get("nl_segmentation_applied_count")),
        "rule_supported_request_count": _num(item.get("rule_supported_request_count")),
        "nl_supported_request_count": _num(item.get("nl_supported_request_count")),
        "rule_only_candidate_count": _num(item.get("rule_only_candidate_count")),
        "nl_segmentation_candidate_count": _num(item.get("nl_segmentation_candidate_count")),
        "rule_only_review_candidate_count": _num(item.get("rule_only_review_candidate_count")),
        "nl_segmentation_review_candidate_count": _num(item.get("nl_segmentation_review_candidate_count")),
        "rule_only_total_estimated_gain_chars": _num(item.get("rule_only_total_estimated_gain_chars")),
        "nl_segmentation_total_estimated_gain_chars": _num(item.get("nl_segmentation_total_estimated_gain_chars")),
        "candidate_promotion_policy": item.get("candidate_promotion_policy")
        if isinstance(item.get("candidate_promotion_policy"), Mapping)
        else {},
        "candidate_review_upper_bound_policy": item.get("candidate_review_upper_bound_policy")
        if isinstance(item.get("candidate_review_upper_bound_policy"), Mapping)
        else {},
        "experiment_gate_items": item.get("experiment_gate_items")
        if isinstance(item.get("experiment_gate_items"), list)
        else [],
        "review_queue_diagnostics": item.get("review_queue_diagnostics")
        if isinstance(item.get("review_queue_diagnostics"), Mapping)
        else {},
        "prompt_extraction_diagnostics": _prompt_extraction_diagnostics(item),
        "rule_gap_resolved": bool(item.get("rule_gap_resolved")),
    }


def _local_judge_config_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    base_url = local_judge.get("base_url") if isinstance(local_judge.get("base_url"), str) else None
    model = local_judge.get("model") if isinstance(local_judge.get("model"), str) else None
    configured_marker = local_judge.get("configured")
    if configured_marker is False:
        return {
            "ready": False,
            "configured": False,
            "base_url": base_url,
            "model": model,
            "reason": "local judge manifest marks configured=false",
        }
    if _local_judge_placeholder(base_url) or _local_judge_placeholder(model):
        return {
            "ready": False,
            "configured": False,
            "base_url": base_url,
            "model": model,
            "reason": "local judge base URL/model is missing or placeholder",
        }
    if model == "local-small-model":
        return {
            "ready": False,
            "configured": False,
            "base_url": base_url,
            "model": model,
            "reason": "local judge model is the example placeholder local-small-model",
        }
    return {
        "ready": True,
        "configured": True,
        "base_url": base_url,
        "model": model,
        "reason": "local judge base URL/model configured",
    }


def _local_judge_placeholder(value: str | None) -> bool:
    if not value:
        return True
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _local_judge_healthcheck_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_healthcheck:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "local judge healthcheck artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status in {"pass", "warn"} and item.get("healthcheck_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "base_url": item.get("base_url"),
        "model": item.get("model"),
        "request_sent": bool(item.get("request_sent")),
        "response_parse_ok": bool(item.get("response_parse_ok")),
        "allowed_label": bool(item.get("allowed_label")),
        "label": item.get("label"),
        "confidence": item.get("confidence"),
        "confidence_meets_min": bool(item.get("confidence_meets_min")),
        "synthetic_label_matches_expected": bool(item.get("synthetic_label_matches_expected")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_template_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_template:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "local judge gold-set template artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status in {"pass", "warn"} and item.get("goldset_template_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "selected_count": _num(item.get("selected_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "include_text": bool(item.get("include_text")),
        "gold_text_written": bool(item.get("gold_text_written")),
        "selected_recovered_text_count": _num(item.get("selected_recovered_text_count")),
        "text_recovery": item.get("text_recovery") if isinstance(item.get("text_recovery"), Mapping) else {},
        "source_label_counts": item.get("source_label_counts")
        if isinstance(item.get("source_label_counts"), Mapping)
        else {},
        "semantic_hint_counts": item.get("semantic_hint_counts")
        if isinstance(item.get("semantic_hint_counts"), Mapping)
        else {},
        "risk_tag_counts": item.get("risk_tag_counts") if isinstance(item.get("risk_tag_counts"), Mapping) else {},
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_import_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_import:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "local judge gold-set CSV import artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("goldset_import_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "rows_with_text_count": _num(item.get("rows_with_text_count")),
        "rows_missing_text_count": _num(item.get("rows_missing_text_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "invalid_expected_label_count": _num(item.get("invalid_expected_label_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "output_written": bool(item.get("output_written")),
        "allow_incomplete_output": bool(item.get("allow_incomplete_output")),
        "ready_for_validate": bool(item.get("ready_for_validate")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_labels_template_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next(
        (entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_labels_template:summary"),
        None,
    )
    if not isinstance(item, Mapping):
        return {
            "ready": False,
            "status": "pending",
            "evidence": "local judge gold-set labels template artifact missing",
        }
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("goldset_labels_template_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "rows_missing_id_count": _num(item.get("rows_missing_id_count")),
        "rows_missing_text_hash_count": _num(item.get("rows_missing_text_hash_count")),
        "duplicate_label_key_count": _num(item.get("duplicate_label_key_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "label_status_counts": item.get("label_status_counts")
        if isinstance(item.get("label_status_counts"), Mapping)
        else {},
        "source_label_counts": item.get("source_label_counts")
        if isinstance(item.get("source_label_counts"), Mapping)
        else {},
        "semantic_hint_counts": item.get("semantic_hint_counts")
        if isinstance(item.get("semantic_hint_counts"), Mapping)
        else {},
        "risk_tag_counts": item.get("risk_tag_counts") if isinstance(item.get("risk_tag_counts"), Mapping) else {},
        "ready_for_apply_labels": bool(item.get("ready_for_apply_labels")),
        "output_written": bool(item.get("output_written")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_suggestions_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next(
        (entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_suggestions:summary"),
        None,
    )
    if not isinstance(item, Mapping):
        return {
            "ready": False,
            "status": "pending",
            "evidence": "local judge gold-set label suggestions artifact missing",
        }
    status = str(item.get("status") or "pending")
    return {
        "ready": status in {"pass", "warn"} and item.get("goldset_suggestions_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "suggestion_count": _num(item.get("suggestion_count")),
        "source_row_count": _num(item.get("source_row_count")),
        "model_called_count": _num(item.get("model_called_count")),
        "local_judge_error_count": _num(item.get("local_judge_error_count")),
        "rows_missing_text_count": _num(item.get("rows_missing_text_count")),
        "static_safety_clamp_count": _num(item.get("static_safety_clamp_count")),
        "suggested_label_counts": item.get("suggested_label_counts")
        if isinstance(item.get("suggested_label_counts"), Mapping)
        else {},
        "model_label_counts": item.get("model_label_counts")
        if isinstance(item.get("model_label_counts"), Mapping)
        else {},
        "rule_label_counts": item.get("rule_label_counts")
        if isinstance(item.get("rule_label_counts"), Mapping)
        else {},
        "source_label_counts": item.get("source_label_counts")
        if isinstance(item.get("source_label_counts"), Mapping)
        else {},
        "local_judge_action_counts": item.get("local_judge_action_counts")
        if isinstance(item.get("local_judge_action_counts"), Mapping)
        else {},
        "label_reason_code_counts": item.get("label_reason_code_counts")
        if isinstance(item.get("label_reason_code_counts"), Mapping)
        else {},
        "model_to_suggested_label_counts": item.get("model_to_suggested_label_counts")
        if isinstance(item.get("model_to_suggested_label_counts"), Mapping)
        else {},
        "suggestions_ready_for_manual_review": bool(item.get("suggestions_ready_for_manual_review")),
        "ready_for_apply_labels": bool(item.get("ready_for_apply_labels")),
        "output_written": bool(item.get("output_written")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_suggestion_review_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next(
        (entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_suggestion_review:summary"),
        None,
    )
    if not isinstance(item, Mapping):
        return {
            "ready": False,
            "status": "pending",
            "evidence": "local judge gold-set suggestion review artifact missing",
        }
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("goldset_suggestion_review_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "suggestion_row_count": _num(item.get("suggestion_row_count")),
        "matched_suggestion_count": _num(item.get("matched_suggestion_count")),
        "missing_suggestion_count": _num(item.get("missing_suggestion_count")),
        "unmatched_suggestion_count": _num(item.get("unmatched_suggestion_count")),
        "duplicate_csv_key_count": _num(item.get("duplicate_csv_key_count")),
        "duplicate_suggestion_key_count": _num(item.get("duplicate_suggestion_key_count")),
        "suggestions_missing_key_count": _num(item.get("suggestions_missing_key_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "suggested_label_counts": item.get("suggested_label_counts")
        if isinstance(item.get("suggested_label_counts"), Mapping)
        else {},
        "ready_for_manual_review": bool(item.get("ready_for_manual_review")),
        "ready_for_apply_labels": bool(item.get("ready_for_apply_labels")),
        "ready_for_goldset_import_after_apply": bool(item.get("ready_for_goldset_import_after_apply")),
        "output_written": bool(item.get("output_written")),
        "csv_text_written": bool(item.get("csv_text_written")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_labels_from_csv_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next(
        (entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_labels_from_csv:summary"),
        None,
    )
    if not isinstance(item, Mapping):
        return {
            "ready": False,
            "status": "pending",
            "evidence": "local judge gold-set labels-from-csv artifact missing",
        }
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("goldset_labels_from_csv_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "valid_label_row_count": _num(item.get("valid_label_row_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "invalid_label_row_count": _num(item.get("invalid_label_row_count")),
        "duplicate_label_key_count": _num(item.get("duplicate_label_key_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "suggested_label_counts": item.get("suggested_label_counts")
        if isinstance(item.get("suggested_label_counts"), Mapping)
        else {},
        "expected_to_suggested_label_counts": item.get("expected_to_suggested_label_counts")
        if isinstance(item.get("expected_to_suggested_label_counts"), Mapping)
        else {},
        "ready_for_apply_labels": bool(item.get("ready_for_apply_labels")),
        "ready_for_goldset_import_after_apply": bool(item.get("ready_for_goldset_import_after_apply")),
        "suggested_labels_not_auto_applied": bool(item.get("suggested_labels_not_auto_applied")),
        "output_written": bool(item.get("output_written")),
        "labels_jsonl_text_written": bool(item.get("labels_jsonl_text_written")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_review_plan_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next(
        (entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_review_plan:summary"),
        None,
    )
    if not isinstance(item, Mapping):
        return {
            "ready": False,
            "status": "pending",
            "evidence": "local judge gold-set review-plan artifact missing",
        }
    status = str(item.get("status") or "pending")
    annotation_complete = item.get("annotation_complete") is True
    return {
        "ready": status in {"pass", "warn"}
        and (
            item.get("ready_for_manual_review") is True
            or annotation_complete
        ),
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "plan_row_count": _num(item.get("plan_row_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "invalid_expected_label_count": _num(item.get("invalid_expected_label_count")),
        "plan_disagreement_row_count": _num(item.get("plan_disagreement_row_count")),
        "plan_high_risk_row_count": _num(item.get("plan_high_risk_row_count")),
        "plan_review_reason_counts": item.get("plan_review_reason_counts")
        if isinstance(item.get("plan_review_reason_counts"), Mapping)
        else {},
        "plan_high_risk_tag_counts": item.get("plan_high_risk_tag_counts")
        if isinstance(item.get("plan_high_risk_tag_counts"), Mapping)
        else {},
        "ready_for_manual_review": item.get("ready_for_manual_review") is True,
        "annotation_complete": annotation_complete,
        "suggested_labels_not_auto_applied": bool(item.get("suggested_labels_not_auto_applied")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_labels_validation_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next(
        (entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_labels_validation:summary"),
        None,
    )
    if not isinstance(item, Mapping):
        return {
            "ready": False,
            "status": "pending",
            "evidence": "local judge gold-set labels validation artifact missing",
        }
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("goldset_labels_validation_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "valid_label_row_count": _num(item.get("valid_label_row_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "invalid_label_row_count": _num(item.get("invalid_label_row_count")),
        "rows_missing_id_count": _num(item.get("rows_missing_id_count")),
        "rows_missing_text_hash_count": _num(item.get("rows_missing_text_hash_count")),
        "duplicate_label_key_count": _num(item.get("duplicate_label_key_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "label_status_counts": item.get("label_status_counts")
        if isinstance(item.get("label_status_counts"), Mapping)
        else {},
        "source_expected_label_match_count": _num(item.get("source_expected_label_match_count")),
        "source_expected_label_mismatch_count": _num(item.get("source_expected_label_mismatch_count")),
        "source_expected_label_audited_count": _num(item.get("source_expected_label_audited_count")),
        "source_expected_label_match_rate": item.get("source_expected_label_match_rate"),
        "source_expected_label_mismatch_counts": item.get("source_expected_label_mismatch_counts")
        if isinstance(item.get("source_expected_label_mismatch_counts"), Mapping)
        else {},
        "possible_rule_self_confirmation": bool(item.get("possible_rule_self_confirmation")),
        "gold_labels_independent_from_rules_ready": bool(item.get("gold_labels_independent_from_rules_ready")),
        "structure_ready": bool(item.get("structure_ready")),
        "complete_labels_ready": bool(item.get("complete_labels_ready")),
        "sample_count_ready": bool(item.get("sample_count_ready")),
        "label_balance_ready": bool(item.get("label_balance_ready")),
        "ready_for_apply_labels": bool(item.get("ready_for_apply_labels")),
        "ready_for_goldset_import_after_apply": bool(item.get("ready_for_goldset_import_after_apply")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_progress_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_progress:summary"), None)
    if not isinstance(item, Mapping):
        return {
            "ready": False,
            "annotation_ready": False,
            "status": "pending",
            "evidence": "local judge gold-set annotation progress artifact missing",
        }
    status = str(item.get("status") or "pending")
    annotation_ready = bool(item.get("ready_for_labeled_jsonl_import"))
    return {
        "ready": status in {"pass", "warn"},
        "annotation_ready": annotation_ready,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "rows_with_text_count": _num(item.get("rows_with_text_count")),
        "rows_missing_text_count": _num(item.get("rows_missing_text_count")),
        "rows_with_expected_label_count": _num(item.get("rows_with_expected_label_count")),
        "expected_label_pending_count": _num(item.get("expected_label_pending_count")),
        "invalid_expected_label_count": _num(item.get("invalid_expected_label_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "source_label_counts": _int_count_map(item.get("source_label_counts")),
        "semantic_hint_counts": _int_count_map(item.get("semantic_hint_counts")),
        "review_priority_counts": _int_count_map(item.get("review_priority_counts")),
        "risk_tag_counts": _int_count_map(item.get("risk_tag_counts")),
        "pending_source_label_counts": _int_count_map(item.get("pending_source_label_counts")),
        "pending_semantic_hint_counts": _int_count_map(item.get("pending_semantic_hint_counts")),
        "pending_review_priority_counts": _int_count_map(item.get("pending_review_priority_counts")),
        "pending_risk_tag_counts": _int_count_map(item.get("pending_risk_tag_counts")),
        "invalid_source_label_counts": _int_count_map(item.get("invalid_source_label_counts")),
        "invalid_semantic_hint_counts": _int_count_map(item.get("invalid_semantic_hint_counts")),
        "invalid_review_priority_counts": _int_count_map(item.get("invalid_review_priority_counts")),
        "invalid_risk_tag_counts": _int_count_map(item.get("invalid_risk_tag_counts")),
        "annotation_completion_rate": item.get("annotation_completion_rate"),
        "label_balance_ready": bool(item.get("label_balance_ready")),
        "ready_for_labeled_jsonl_import": bool(item.get("ready_for_labeled_jsonl_import")),
        "ready_for_validate_after_import": bool(item.get("ready_for_validate_after_import")),
        "guide_path": item.get("guide_path"),
        "worklist_path": item.get("worklist_path"),
        "worklist_written": bool(item.get("worklist_written")),
        "worklist_row_count": _num(item.get("worklist_row_count")),
        "worklist_pending_count": _num(item.get("worklist_pending_count")),
        "worklist_invalid_count": _num(item.get("worklist_invalid_count")),
        "worklist_schema_version": item.get("worklist_schema_version"),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_validation_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_validation:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "local judge gold-set validation artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("goldset_validation_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "valid_row_count": _num(item.get("valid_row_count")),
        "invalid_row_count": _num(item.get("invalid_row_count")),
        "missing_text_count": _num(item.get("missing_text_count")),
        "invalid_expected_label_count": _num(item.get("invalid_expected_label_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "min_samples": _num(item.get("min_samples")),
        "min_label_count": _num(item.get("min_label_count")),
        "recommendation": item.get("recommendation"),
    }


def _local_judge_quality_eval_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_quality_eval:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "local judge quality eval artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("quality_eval_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "sample_count": _num(item.get("sample_count")),
        "evaluated_count": _num(item.get("evaluated_count")),
        "model_called_count": _num(item.get("model_called_count")),
        "accuracy": item.get("accuracy"),
        "macro_f1": item.get("macro_f1"),
        "raw_model_accuracy": item.get("raw_model_accuracy"),
        "raw_model_macro_f1": item.get("raw_model_macro_f1"),
        "static_safety_clamp_count": _num(item.get("static_safety_clamp_count")),
        "static_safety_clamp_corrected_count": _num(item.get("static_safety_clamp_corrected_count")),
        "static_safety_clamp_worsened_count": _num(item.get("static_safety_clamp_worsened_count")),
        "static_safety_clamp_net_correct_delta": _num(item.get("static_safety_clamp_net_correct_delta")),
        "min_samples": _num(item.get("min_samples")),
        "min_accuracy": item.get("min_accuracy"),
        "min_macro_f1": item.get("min_macro_f1"),
        "parse_error_count": _num(item.get("parse_error_count")),
        "low_confidence_count": _num(item.get("low_confidence_count")),
        "expected_label_counts": item.get("expected_label_counts")
        if isinstance(item.get("expected_label_counts"), Mapping)
        else {},
        "predicted_label_counts": item.get("predicted_label_counts")
        if isinstance(item.get("predicted_label_counts"), Mapping)
        else {},
        "raw_model_label_counts": item.get("raw_model_label_counts")
        if isinstance(item.get("raw_model_label_counts"), Mapping)
        else {},
        "raw_model_to_predicted_label_counts": item.get("raw_model_to_predicted_label_counts")
        if isinstance(item.get("raw_model_to_predicted_label_counts"), Mapping)
        else {},
        "quality_diagnostics": item.get("quality_diagnostics")
        if isinstance(item.get("quality_diagnostics"), Mapping)
        else {},
        "mismatch_diagnostics": item.get("mismatch_diagnostics")
        if isinstance(item.get("mismatch_diagnostics"), Mapping)
        else {},
        "failure_mode_codes": item.get("failure_mode_codes")
        if isinstance(item.get("failure_mode_codes"), list)
        else [],
        "recommendation": item.get("recommendation"),
    }


def _local_judge_policy_eval_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_policy_eval:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "local judge policy eval artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("policy_eval_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "joined_prediction_count": _num(item.get("joined_prediction_count")),
        "missing_prediction_count": _num(item.get("missing_prediction_count")),
        "best_ready_policy": item.get("best_ready_policy"),
        "best_overall_policy": item.get("best_overall_policy"),
        "policy_metric_table": item.get("policy_metric_table")
        if isinstance(item.get("policy_metric_table"), list)
        else [],
        "policy_failure_diagnostics": item.get("policy_failure_diagnostics")
        if isinstance(item.get("policy_failure_diagnostics"), Mapping)
        else {},
        "recommendation": item.get("recommendation"),
    }


def _static_rule_calibration_eval_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "static_rule_calibration_eval:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "static rule calibration eval artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("static_rule_calibration_eval_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "row_count": _num(item.get("row_count")),
        "exploratory_ready": item.get("exploratory_ready") is True,
        "best_ready_policy": item.get("best_ready_policy"),
        "best_production_candidate_policy": item.get("best_production_candidate_policy"),
        "baseline_rule_accuracy": item.get("baseline_rule_accuracy"),
        "baseline_rule_macro_f1": item.get("baseline_rule_macro_f1"),
        "top_policy_metric_table": item.get("top_policy_metric_table")
        if isinstance(item.get("top_policy_metric_table"), list)
        else [],
        "production_readiness_diagnostics": item.get("production_readiness_diagnostics")
        if isinstance(item.get("production_readiness_diagnostics"), Mapping)
        else {},
        "gold_support_diagnostics": item.get("gold_support_diagnostics")
        if isinstance(item.get("gold_support_diagnostics"), Mapping)
        else {},
        "validator_rule_candidates": item.get("validator_rule_candidates")
        if isinstance(item.get("validator_rule_candidates"), Mapping)
        else {},
        "recommendation": item.get("recommendation"),
    }


def _local_judge_goldset_chain_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "local_judge_goldset_chain:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "local judge gold-set chain verifier artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass" and item.get("goldset_chain_ready") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "step_ready_counts": item.get("step_ready_counts") if isinstance(item.get("step_ready_counts"), Mapping) else {},
        "failed_check_count": _num(item.get("failed_check_count")),
        "blocking_reasons": item.get("blocking_reasons") if isinstance(item.get("blocking_reasons"), list) else [],
        "next_action": item.get("next_action"),
        "recommendation": item.get("recommendation"),
    }


def _offline_local_judge_matrix_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next((entry for entry in artifact_quality if entry.get("name") == "offline_local_judge_matrix:summary"), None)
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "offline local-judge matrix artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "pass",
        "status": status,
        "evidence": item.get("evidence"),
        "completed_source_count": _num(item.get("completed_source_count")),
        "supported_source_count": _num(item.get("supported_source_count")),
        "model_called_count": _num(item.get("model_called_count")),
        "budget_exhausted_source_count": _num(item.get("budget_exhausted_source_count")),
        "aggregate_budget_coverage_rate": item.get("aggregate_budget_coverage_rate"),
        "prompt_extraction_diagnostics": _prompt_extraction_diagnostics(item),
        "candidate_label_diagnostics": _candidate_label_diagnostics(item),
        "recommendation": item.get("recommendation"),
    }


def _offline_local_judge_matrix_smoke_status(artifact_quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item = next(
        (entry for entry in artifact_quality if entry.get("name") == "offline_local_judge_matrix_smoke:summary"),
        None,
    )
    if not isinstance(item, Mapping):
        return {"ready": False, "status": "pending", "evidence": "offline local-judge matrix smoke artifact missing"}
    status = str(item.get("status") or "pending")
    return {
        "ready": status == "warn" and item.get("fake_local_judge") is True,
        "status": status,
        "evidence": item.get("evidence"),
        "fake_local_judge": bool(item.get("fake_local_judge")),
        "fake_local_judge_request_count": _num(item.get("fake_local_judge_request_count")),
        "completed_source_count": _num(item.get("completed_source_count")),
        "supported_source_count": _num(item.get("supported_source_count")),
        "budget_exhausted_source_count": _num(item.get("budget_exhausted_source_count")),
        "aggregate_budget_coverage_rate": item.get("aggregate_budget_coverage_rate"),
        "prompt_extraction_diagnostics": _prompt_extraction_diagnostics(item),
        "candidate_label_diagnostics": _candidate_label_diagnostics(item),
        "recommendation": item.get("recommendation"),
    }


def _artifact_quality_record(artifact: Mapping[str, Any]) -> dict[str, Any]:
    name = str(artifact.get("name") or "")
    path_value = artifact.get("path")
    base = {
        "name": name,
        "path": path_value,
        "status": "pending",
        "evidence": "artifact missing or empty",
    }
    if not artifact.get("nonempty") or not isinstance(path_value, str):
        return base

    path = Path(path_value)
    try:
        if name.endswith(":provider_telemetry_path"):
            return {**base, **_provider_telemetry_quality(path)}
        if name.endswith(":tabulate_csv_path"):
            return {**base, **_tabulate_csv_quality(path, variant=_artifact_variant(name))}
        if name.endswith(":task_results_path"):
            return {**base, **_task_results_quality(path)}
        if name.startswith("ab_eval:") and name.endswith("_summary"):
            return {**base, **_ab_summary_quality(path)}
        if name.startswith("ab_eval:") and name.endswith("_report_md"):
            return {**base, **_ab_report_quality(path)}
        if name == "semantic_guard_fake_smoke:summary":
            return {**base, **_semantic_guard_fake_smoke_quality(path)}
        if name == "offline_semantic_suite:summary":
            return {**base, **_offline_semantic_suite_quality(path)}
        if name == "local_judge_healthcheck:summary":
            return {**base, **_local_judge_healthcheck_quality(path)}
        if name == "local_judge_goldset_template:summary":
            return {**base, **_local_judge_goldset_template_quality(path)}
        if name == "local_judge_goldset_progress:summary":
            return {**base, **_local_judge_goldset_progress_quality(path)}
        if name == "local_judge_goldset_labels_template:summary":
            return {**base, **_local_judge_goldset_labels_template_quality(path)}
        if name == "local_judge_goldset_suggestions:summary":
            return {**base, **_local_judge_goldset_suggestions_quality(path)}
        if name == "local_judge_goldset_suggestion_review:summary":
            return {**base, **_local_judge_goldset_suggestion_review_quality(path)}
        if name == "local_judge_goldset_review_plan:summary":
            return {**base, **_local_judge_goldset_review_plan_quality(path)}
        if name == "local_judge_goldset_labels_from_csv:summary":
            return {**base, **_local_judge_goldset_labels_from_csv_quality(path)}
        if name == "local_judge_goldset_labels_validation:summary":
            return {**base, **_local_judge_goldset_labels_validation_quality(path)}
        if name == "local_judge_goldset_import:summary":
            return {**base, **_local_judge_goldset_import_quality(path)}
        if name == "local_judge_goldset_validation:summary":
            return {**base, **_local_judge_goldset_validation_quality(path)}
        if name == "local_judge_quality_eval:summary":
            return {**base, **_local_judge_quality_eval_quality(path)}
        if name == "local_judge_policy_eval:summary":
            return {**base, **_local_judge_policy_eval_quality(path)}
        if name == "static_rule_calibration_eval:summary":
            return {**base, **_static_rule_calibration_eval_quality(path)}
        if name == "local_judge_goldset_chain:summary":
            return {**base, **_local_judge_goldset_chain_quality(path)}
        if name == "offline_local_judge_matrix:summary":
            return {**base, **_offline_local_judge_matrix_quality(path)}
        if name == "offline_local_judge_matrix_smoke:summary":
            return {**base, **_offline_local_judge_matrix_smoke_quality(path)}
    except Exception as exc:  # pragma: no cover - exercised by broad parser failures
        return {
            **base,
            "status": "fail",
            "evidence": f"artifact parser raised {type(exc).__name__}: {exc}",
        }
    return {**base, "status": "pass", "evidence": "nonempty artifact; no specialized parser"}


def _provider_telemetry_quality(path: Path) -> dict[str, Any]:
    dataset_summary = evaluate_dataset(input_path=path).summary
    summary = dataset_summary.get("provider_trace", {})
    if not isinstance(summary, Mapping) or not summary.get("supported"):
        return {
            "status": "fail",
            "evidence": "provider telemetry is nonempty but dataset_eval found no provider_trace records",
        }
    shadow = dataset_summary.get("shadow_trial_trace") if isinstance(dataset_summary.get("shadow_trial_trace"), Mapping) else {}
    shadow_fields = _shadow_trial_quality_fields(shadow)
    if _jsonl_contains_flag(path, "fake_upstream"):
        return {
            "status": "warn",
            "evidence": "provider telemetry is from a fake upstream smoke and cannot prove real provider metrics",
            "provider_record_count": _num(summary.get("record_count")),
            "actual_prompt_tokens": _num(summary.get("actual_prompt_tokens")),
            "actual_cached_tokens": _num(summary.get("actual_cached_tokens")),
            "fake_upstream": True,
            **shadow_fields,
        }
    record_count = _num(summary.get("record_count"))
    prompt_tokens = _num(summary.get("actual_prompt_tokens"))
    cached_tokens = _num(summary.get("actual_cached_tokens"))
    if record_count <= 0:
        return {"status": "fail", "evidence": "provider_trace record_count=0"}
    if prompt_tokens <= 0:
        return {
            "status": "warn",
            "evidence": f"provider_trace records={record_count}, but prompt/input token usage was not available",
            "provider_record_count": record_count,
            "actual_prompt_tokens": prompt_tokens,
            "actual_cached_tokens": cached_tokens,
            **shadow_fields,
        }
    return {
        "status": "pass",
        "evidence": (
            f"provider_trace records={record_count}, prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}"
        ),
        "provider_record_count": record_count,
        "actual_prompt_tokens": prompt_tokens,
        "actual_cached_tokens": cached_tokens,
        "latency_supported": bool(summary.get("latency_supported")),
        "cost_supported": bool(summary.get("cost_supported")),
        **shadow_fields,
    }


def _shadow_trial_quality_fields(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not summary.get("supported"):
        return {
            "shadow_trial_supported": False,
            "shadow_trial_record_count": 0,
            "shadow_trial_record_with_match_count": 0,
            "shadow_trial_matched_rule_observation_count": 0,
            "shadow_trial_unique_matched_rule_count": 0,
            "shadow_trial_action_counts": {},
            "validator_behavior_change_allowed": False,
        }
    return {
        "shadow_trial_supported": True,
        "shadow_trial_record_count": _num(summary.get("record_count")),
        "shadow_trial_record_with_match_count": _num(summary.get("record_with_match_count")),
        "shadow_trial_matched_rule_observation_count": _num(summary.get("matched_rule_observation_count")),
        "shadow_trial_unique_matched_rule_count": _num(summary.get("unique_matched_rule_count")),
        "shadow_trial_unique_matched_rule_ids": list(summary.get("unique_matched_rule_ids"))
        if isinstance(summary.get("unique_matched_rule_ids"), list | tuple)
        else [],
        "shadow_trial_action_counts": dict(summary.get("trial_action_counts"))
        if isinstance(summary.get("trial_action_counts"), Mapping)
        else {},
        "shadow_trial_unsupported_rule_observation_count": _num(summary.get("unsupported_rule_observation_count")),
        "validator_behavior_change_allowed": bool(summary.get("validator_behavior_change_allowed")),
        "safe_for_automatic_validator_promotion": bool(summary.get("safe_for_automatic_validator_promotion")),
    }


def _tabulate_csv_quality(path: Path, *, variant: str) -> dict[str, Any]:
    rows = parse_autogenbench_tabulate_csv(path, variant=variant)
    success_evaluated = sum(1 for row in rows if row.get("success") is not None)
    if not rows:
        return {"status": "fail", "evidence": "tabulate CSV parsed but produced no task rows"}
    if success_evaluated <= 0:
        return {
            "status": "fail",
            "evidence": f"tabulate CSV produced {len(rows)} rows but no success values",
        }
    return {
        "status": "pass",
        "evidence": f"tabulate CSV task rows={len(rows)}, success_evaluated={success_evaluated}",
        "task_row_count": len(rows),
        "success_evaluated_count": success_evaluated,
    }


def _task_results_quality(path: Path) -> dict[str, Any]:
    summary = summarize_task_results(path)
    if not summary.get("supported"):
        return {"status": "fail", "evidence": "task results parsed but no task records were supported"}
    success_evaluated = _num(summary.get("success_evaluated_count"))
    score_evaluated = _num(summary.get("score_evaluated_count"))
    if success_evaluated <= 0 and score_evaluated <= 0:
        return {
            "status": "fail",
            "evidence": "task results parsed but no success or score metrics were available",
        }
    return {
        "status": "pass",
        "evidence": (
            f"task_count={_num(summary.get('task_count'))}, "
            f"success_evaluated={success_evaluated}, score_evaluated={score_evaluated}"
        ),
        "task_count": _num(summary.get("task_count")),
        "success_evaluated_count": success_evaluated,
        "score_evaluated_count": score_evaluated,
    }


def _ab_summary_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    schema_ok = value.get("schema_version") == "prefix-reorder-ab-eval-summary-v1"
    gates = value.get("gates") if isinstance(value.get("gates"), Mapping) else {}
    overall = gates.get("overall_status") if isinstance(gates, Mapping) else None
    if not schema_ok:
        return {"status": "fail", "evidence": f"unexpected A/B summary schema={value.get('schema_version')}"}
    if not overall:
        return {"status": "fail", "evidence": "A/B summary has no gates.overall_status"}
    if value.get("fake_upstream") is True:
        return {
            "status": "warn",
            "evidence": "A/B summary was generated from a fake upstream smoke and cannot prove real provider metrics",
            "overall_gate_status": overall,
            "fake_upstream": True,
        }
    real_provider = bool(value.get("real_provider_metrics_available"))
    task_metrics = bool(value.get("task_metrics_available"))
    if not real_provider or not task_metrics or overall == "unknown":
        return {
            "status": "warn",
            "evidence": (
                f"overall_gate_status={overall}, real_provider_metrics={real_provider}, "
                f"task_metrics={task_metrics}"
            ),
            "overall_gate_status": overall,
        }
    return {
        "status": "pass",
        "evidence": (
            f"overall_gate_status={overall}, real_provider_metrics={real_provider}, task_metrics={task_metrics}"
        ),
        "overall_gate_status": overall,
    }


def _ab_report_quality(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    has_expected_sections = "# Prefix Reorder A/B Summary" in text and "## Gates" in text
    return {
        "status": "pass" if has_expected_sections else "warn",
        "evidence": "Markdown report has expected A/B summary sections"
        if has_expected_sections
        else "Markdown report is nonempty but expected A/B sections were not found",
    }


def _semantic_guard_fake_smoke_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-semantic-guard-proxy-smoke-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected semantic guard smoke schema={value.get('schema_version')}",
        }
    fake_upstream = value.get("fake_upstream") is True
    fake_judge = value.get("fake_local_judge") is True
    request_count = _num(value.get("request_count"))
    judge_request_count = _num(value.get("judge_request_count"))
    upstream_request_count = _num(value.get("upstream_request_count"))
    validation_reasons = value.get("adapter_validation_reasons") if isinstance(value.get("adapter_validation_reasons"), list) else []
    if request_count <= 0 or upstream_request_count <= 0:
        return {
            "status": "fail",
            "evidence": "semantic guard smoke summary has no proxy/upstream requests",
        }
    if judge_request_count <= 0:
        return {
            "status": "fail",
            "evidence": "semantic guard smoke summary has no local judge request",
        }
    if not fake_upstream or not fake_judge:
        return {
            "status": "warn",
            "evidence": "semantic guard smoke summary is nonempty but missing fake upstream/judge markers",
            "fake_upstream": fake_upstream,
            "fake_local_judge": fake_judge,
            "request_count": request_count,
            "judge_request_count": judge_request_count,
        }
    return {
        "status": "warn",
        "evidence": (
            "semantic guard proxy smoke used fake upstream and fake local judge; "
            "it validates fail-closed wiring only"
        ),
        "fake_upstream": True,
        "fake_local_judge": True,
        "request_count": request_count,
        "judge_request_count": judge_request_count,
        "upstream_request_count": upstream_request_count,
        "adapter_validation_reasons": validation_reasons,
    }


def _offline_semantic_suite_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-offline-semantic-suite-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected offline semantic suite schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "offline semantic suite summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {
            "status": "warn",
            "evidence": "offline semantic suite should not claim real provider metrics",
        }
    rule_only = value.get("rule_only") if isinstance(value.get("rule_only"), Mapping) else {}
    nl = value.get("nl_segmentation") if isinstance(value.get("nl_segmentation"), Mapping) else {}
    comparison = value.get("comparison") if isinstance(value.get("comparison"), Mapping) else {}
    gate = comparison.get("experiment_gate") if isinstance(comparison.get("experiment_gate"), Mapping) else {}
    rule_supported = _num(rule_only.get("supported_request_count"))
    nl_supported = _num(nl.get("supported_request_count"))
    if rule_supported <= 0 or nl_supported <= 0:
        return {
            "status": "fail",
            "evidence": "offline semantic suite has no supported source-prompt requests",
            "rule_supported_request_count": rule_supported,
            "nl_supported_request_count": nl_supported,
        }
    if gate.get("schema_version") != "prefix-offline-experiment-gate-v1":
        return {"status": "fail", "evidence": "offline semantic suite comparison gate is missing or invalid"}
    if gate.get("performance_conclusion_allowed") is not False:
        return {"status": "fail", "evidence": "offline semantic suite gate must not allow performance conclusions"}
    failing_gate_items = _failing_experiment_gate_items(gate)
    if failing_gate_items:
        return {
            "status": "fail",
            "evidence": "offline semantic suite experiment gate failed: " + ", ".join(failing_gate_items),
            "recommendation": gate.get("recommendation") or value.get("recommendation"),
            "failed_experiment_gate_items": failing_gate_items,
            "experiment_gate_items": _experiment_gate_items_digest(gate),
        }
    return {
        "status": "pass",
        "evidence": (
            "offline source-prompt suite is prompt-safe and has rule-only/nl-segmentation coverage plus experiment gate"
        ),
        "recommendation": gate.get("recommendation") or value.get("recommendation"),
        "rule_only_applied_count": _num(rule_only.get("applied_count")),
        "nl_segmentation_applied_count": _num(nl.get("applied_count")),
        "rule_only_candidate_count": _num(rule_only.get("candidate_count")),
        "nl_segmentation_candidate_count": _num(nl.get("candidate_count")),
        "rule_only_review_candidate_count": _num((rule_only.get("label_counts") or {}).get("review"))
        if isinstance(rule_only.get("label_counts"), Mapping)
        else 0,
        "nl_segmentation_review_candidate_count": _num((nl.get("label_counts") or {}).get("review"))
        if isinstance(nl.get("label_counts"), Mapping)
        else 0,
        "rule_only_total_estimated_gain_chars": _num(rule_only.get("total_estimated_gain_chars")),
        "nl_segmentation_total_estimated_gain_chars": _num(nl.get("total_estimated_gain_chars")),
        "candidate_promotion_policy": nl.get("candidate_promotion_policy")
        if isinstance(nl.get("candidate_promotion_policy"), Mapping)
        else {},
        "candidate_review_upper_bound_policy": nl.get("candidate_review_upper_bound_policy")
        if isinstance(nl.get("candidate_review_upper_bound_policy"), Mapping)
        else {},
        "experiment_gate_items": _experiment_gate_items_digest(gate),
        "review_queue_diagnostics": _review_queue_digest(nl.get("review_queue_diagnostics")),
        "prompt_extraction_diagnostics": _prompt_extraction_diagnostics(nl),
        "rule_gap_resolved": bool((comparison.get("delta") or {}).get("rule_gap_resolved"))
        if isinstance(comparison.get("delta"), Mapping)
        else False,
        "rule_supported_request_count": rule_supported,
        "nl_supported_request_count": nl_supported,
    }


def _failing_experiment_gate_items(gate: Mapping[str, Any]) -> list[str]:
    items = gate.get("items")
    if not isinstance(items, list):
        return []
    failing: list[str] = []
    for item in items:
        if isinstance(item, Mapping) and item.get("status") == "fail":
            failing.append(str(item.get("name") or "unknown_gate"))
    return failing


def _experiment_gate_items_digest(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = gate.get("items")
    if not isinstance(items, list):
        return []
    digest: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        digest.append(
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "evidence": item.get("evidence"),
            }
        )
    return digest


def _review_queue_digest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "schema_version": value.get("schema_version"),
        "review_candidate_count": _num(value.get("review_candidate_count")),
        "review_parent_block_count": _num(value.get("review_parent_block_count")),
        "review_source_file_count": _num(value.get("review_source_file_count")),
        "review_candidate_chars": _num(value.get("review_candidate_chars")),
        "semantic_hint_counts": value.get("semantic_hint_counts")
        if isinstance(value.get("semantic_hint_counts"), Mapping)
        else {},
        "risk_tag_counts": value.get("risk_tag_counts") if isinstance(value.get("risk_tag_counts"), Mapping) else {},
        "label_reason_counts": value.get("label_reason_counts")
        if isinstance(value.get("label_reason_counts"), Mapping)
        else {},
        "local_judge_priority": value.get("local_judge_priority"),
    }


def _local_judge_healthcheck_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-healthcheck-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge healthcheck schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge healthcheck summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "local judge healthcheck must not claim provider metrics"}
    request_sent = value.get("request_sent") is True
    parse_ok = value.get("response_parse_ok") is True
    allowed_label = value.get("allowed_label") is True
    confidence_ok = value.get("confidence_meets_min") is True
    ready = value.get("ready") is True
    if not ready:
        return {
            "status": "fail",
            "evidence": f"local judge healthcheck not ready: {value.get('error_reason') or 'ready=false'}",
            "healthcheck_ready": False,
            "base_url": value.get("base_url"),
            "model": value.get("model"),
            "request_sent": request_sent,
            "response_parse_ok": parse_ok,
            "allowed_label": allowed_label,
            "label": value.get("label"),
            "confidence": value.get("confidence"),
            "confidence_meets_min": confidence_ok,
            "synthetic_label_matches_expected": bool(value.get("synthetic_label_matches_expected")),
            "recommendation": value.get("recommendation"),
        }
    matches_expected = value.get("synthetic_label_matches_expected") is True
    return {
        "status": "pass" if matches_expected else "warn",
        "evidence": "local judge endpoint/schema healthcheck passed"
        if matches_expected
        else "local judge endpoint/schema passed but synthetic label differs from expected",
        "healthcheck_ready": True,
        "base_url": value.get("base_url"),
        "model": value.get("model"),
        "request_sent": request_sent,
        "response_parse_ok": parse_ok,
        "allowed_label": allowed_label,
        "label": value.get("label"),
        "confidence": value.get("confidence"),
        "confidence_meets_min": confidence_ok,
        "synthetic_label_matches_expected": matches_expected,
        "recommendation": value.get("recommendation"),
    }


def _local_judge_quality_eval_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-quality-eval-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge quality eval schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge quality eval summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "local judge quality eval must not claim provider metrics"}
    if value.get("gold_text_written") is not False:
        return {"status": "fail", "evidence": "local judge quality eval summary must not write gold text"}
    ready = value.get("ready") is True
    base = {
        "quality_eval_ready": ready,
        "sample_count": _num(value.get("sample_count")),
        "evaluated_count": _num(value.get("evaluated_count")),
        "model_called_count": _num(value.get("model_called_count")),
        "accuracy": _number_or_none(value.get("accuracy")),
        "macro_f1": _number_or_none(value.get("macro_f1")),
        "raw_model_accuracy": _number_or_none(value.get("raw_model_accuracy")),
        "raw_model_macro_f1": _number_or_none(value.get("raw_model_macro_f1")),
        "raw_model_correct_count": _num(value.get("raw_model_correct_count")),
        "static_safety_clamp_count": _num(value.get("static_safety_clamp_count")),
        "static_safety_clamp_corrected_count": _num(value.get("static_safety_clamp_corrected_count")),
        "static_safety_clamp_worsened_count": _num(value.get("static_safety_clamp_worsened_count")),
        "static_safety_clamp_net_correct_delta": _num(value.get("static_safety_clamp_net_correct_delta")),
        "min_samples": _num(value.get("min_samples")),
        "min_accuracy": _number_or_none(value.get("min_accuracy")),
        "min_macro_f1": _number_or_none(value.get("min_macro_f1")),
        "parse_error_count": _num(value.get("parse_error_count")),
        "low_confidence_count": _num(value.get("low_confidence_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "predicted_label_counts": _int_count_map(value.get("predicted_label_counts")),
        "raw_model_label_counts": _int_count_map(value.get("raw_model_label_counts")),
        "raw_model_to_predicted_label_counts": _int_count_map(value.get("raw_model_to_predicted_label_counts")),
        "quality_diagnostics": _quality_diagnostics(value.get("quality_diagnostics")),
        "mismatch_diagnostics": _mismatch_diagnostics(value.get("mismatch_diagnostics")),
        "failure_mode_codes": [str(code) for code in value.get("failure_mode_codes", [])]
        if isinstance(value.get("failure_mode_codes"), list)
        else [],
        "recommendation": value.get("recommendation"),
    }
    if not ready:
        return {
            **base,
            "status": "warn",
            "evidence": "local judge quality eval ran but did not meet semantic-quality thresholds",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "local judge quality eval met supplied gold-set thresholds",
    }


def _local_judge_policy_eval_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-policy-eval-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge policy eval schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge policy eval summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "local judge policy eval must not claim provider metrics"}
    if value.get("gold_text_written") is not False:
        return {"status": "fail", "evidence": "local judge policy eval summary must not write gold text"}
    policies = value.get("policy_summaries") if isinstance(value.get("policy_summaries"), list) else []
    table = [_policy_metric_row(policy) for policy in policies if isinstance(policy, Mapping)]
    ready = value.get("ready") is True
    base = {
        "policy_eval_ready": ready,
        "row_count": _num(value.get("row_count")),
        "joined_prediction_count": _num(value.get("joined_prediction_count")),
        "missing_prediction_count": _num(value.get("missing_prediction_count")),
        "best_ready_policy": value.get("best_ready_policy"),
        "best_overall_policy": value.get("best_overall_policy"),
        "policy_metric_table": table,
        "policy_failure_diagnostics": _policy_failure_diagnostics(value.get("policy_failure_diagnostics")),
        "recommendation": value.get("recommendation"),
    }
    if not table:
        return {
            **base,
            "status": "warn",
            "evidence": "local judge policy eval ran but produced no policy summaries",
        }
    if not ready:
        return {
            **base,
            "status": "warn",
            "evidence": "local judge policy eval ran but no static/model/hybrid policy met thresholds",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "local judge policy eval found at least one policy meeting supplied gold-set thresholds",
    }


def _static_rule_calibration_eval_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-static-rule-calibration-eval-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected static rule calibration eval schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "static rule calibration eval summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "static rule calibration eval must not claim provider metrics"}
    if value.get("gold_text_written") is not False:
        return {"status": "fail", "evidence": "static rule calibration eval summary must not write gold text"}
    policies = value.get("policy_summaries") if isinstance(value.get("policy_summaries"), list) else []
    top_policies = value.get("top_policies") if isinstance(value.get("top_policies"), list) else []
    baseline_rule = next(
        (
            policy
            for policy in policies
            if isinstance(policy, Mapping) and policy.get("policy_name") == "rule_label"
        ),
        {},
    )
    ready = value.get("ready") is True
    exploratory_ready = value.get("exploratory_ready") is True
    base = {
        "static_rule_calibration_eval_ready": ready,
        "exploratory_ready": exploratory_ready,
        "row_count": _num(value.get("row_count")),
        "min_accuracy": _number_or_none(value.get("min_accuracy")),
        "min_macro_f1": _number_or_none(value.get("min_macro_f1")),
        "production_min_support": _num(value.get("production_min_support")),
        "production_min_purity": _number_or_none(value.get("production_min_purity")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "best_ready_policy": value.get("best_ready_policy"),
        "best_production_candidate_policy": value.get("best_production_candidate_policy"),
        "baseline_rule_accuracy": _number_or_none(baseline_rule.get("accuracy"))
        if isinstance(baseline_rule, Mapping)
        else None,
        "baseline_rule_macro_f1": _number_or_none(baseline_rule.get("macro_f1"))
        if isinstance(baseline_rule, Mapping)
        else None,
        "top_policy_metric_table": [_policy_metric_row(policy) for policy in top_policies if isinstance(policy, Mapping)],
        "production_readiness_diagnostics": _static_production_readiness_diagnostics(
            value.get("production_readiness_diagnostics")
        ),
        "gold_support_diagnostics": _static_gold_support_diagnostics(value.get("gold_support_diagnostics")),
        "validator_rule_candidates": _static_validator_rule_candidates(
            value.get("validator_rule_candidates")
        ),
        "recommendation": value.get("recommendation"),
    }
    if ready:
        return {
            **base,
            "status": "pass",
            "evidence": "metadata calibration found a production-threshold candidate for Validator rule review",
        }
    if exploratory_ready:
        return {
            **base,
            "status": "warn",
            "evidence": "metadata calibration found exploratory signal but no production-threshold candidate",
        }
    return {
        **base,
        "status": "warn",
        "evidence": "static rule calibration ran but metadata policies did not meet thresholds",
    }


def _policy_metric_row(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": policy.get("policy_name"),
        "ready": policy.get("ready") is True,
        "accuracy": _number_or_none(policy.get("accuracy")),
        "macro_f1": _number_or_none(policy.get("macro_f1")),
        "predicted_label_counts": _int_count_map(policy.get("predicted_label_counts")),
        "failure_mode_codes": [str(code) for code in policy.get("failure_mode_codes", [])]
        if isinstance(policy.get("failure_mode_codes"), list)
        else [],
        "automation_policy": policy.get("automation_policy"),
    }


def _static_production_readiness_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rows = value.get("top_ready_nonproduction_policies")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "status": value.get("status"),
        "ready_nonproduction_policy_count": _num(value.get("ready_nonproduction_policy_count")),
        "production_min_support": _num(value.get("production_min_support")),
        "production_min_purity": _number_or_none(value.get("production_min_purity")),
        "missing_threshold_reason_counts": _int_count_map(value.get("missing_threshold_reason_counts")),
        "top_ready_nonproduction_policies": [
            _static_production_gap_row(row)
            for row in rows
            if isinstance(rows, list) and isinstance(row, Mapping)
        ][:5],
        "recommendation": value.get("recommendation"),
    }


def _static_production_gap_row(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": value.get("policy_name"),
        "automation_policy": value.get("automation_policy"),
        "min_support": _num(value.get("min_support")),
        "min_purity": _number_or_none(value.get("min_purity")),
        "missing_production_support": _num(value.get("missing_production_support")),
        "missing_production_purity": _number_or_none(value.get("missing_production_purity")),
        "missing_threshold_reasons": [str(item) for item in value.get("missing_threshold_reasons", [])]
        if isinstance(value.get("missing_threshold_reasons"), list)
        else [],
        "accuracy": _number_or_none(value.get("accuracy")),
        "macro_f1": _number_or_none(value.get("macro_f1")),
    }


def _static_gold_support_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rows = value.get("priority_buckets")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "feature_bucket_count": _num(value.get("feature_bucket_count")),
        "support_ready_bucket_count": _num(value.get("support_ready_bucket_count")),
        "purity_ready_bucket_count": _num(value.get("purity_ready_bucket_count")),
        "production_ready_bucket_count": _num(value.get("production_ready_bucket_count")),
        "production_min_support": _num(value.get("production_min_support")),
        "production_min_purity": _number_or_none(value.get("production_min_purity")),
        "feature_set_bucket_counts": _int_count_map(value.get("feature_set_bucket_counts")),
        "priority_buckets": [
            _static_gold_support_bucket(row)
            for row in rows
            if isinstance(rows, list) and isinstance(row, Mapping)
        ][:8],
        "recommendation": value.get("recommendation"),
    }


def _static_gold_support_bucket(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_set": value.get("feature_set"),
        "features": value.get("features") if isinstance(value.get("features"), Mapping) else {},
        "support": _num(value.get("support")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "dominant_label": value.get("dominant_label"),
        "purity": _number_or_none(value.get("purity")),
        "support_ready": value.get("support_ready") is True,
        "purity_ready": value.get("purity_ready") is True,
        "production_ready": value.get("production_ready") is True,
        "tie_for_dominant_label": value.get("tie_for_dominant_label") is True,
        "needed_additional_labels": _num(value.get("needed_additional_labels")),
        "needed_for_support": _num(value.get("needed_for_support")),
        "needed_for_purity": _num(value.get("needed_for_purity")),
        "purity_gap": _number_or_none(value.get("purity_gap")),
    }


def _static_validator_rule_candidates(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    candidates = value.get("candidate_rules")
    conflicts = value.get("conflict_buckets")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "status": value.get("status"),
        "candidate_rule_count": _num(value.get("candidate_rule_count")),
        "safe_accept_rule_count": _num(value.get("safe_accept_rule_count")),
        "reject_rule_count": _num(value.get("reject_rule_count")),
        "review_rule_count": _num(value.get("review_rule_count")),
        "conflict_bucket_count": _num(value.get("conflict_bucket_count")),
        "production_min_support": _num(value.get("production_min_support")),
        "production_min_purity": _number_or_none(value.get("production_min_purity")),
        "candidate_rules": [
            _static_validator_candidate_row(row)
            for row in candidates
            if isinstance(candidates, list) and isinstance(row, Mapping)
        ][:8],
        "conflict_buckets": [
            _static_validator_conflict_row(row)
            for row in conflicts
            if isinstance(conflicts, list) and isinstance(row, Mapping)
        ][:5],
        "overlap_diagnostics": _static_validator_overlap_diagnostics(
            value.get("overlap_diagnostics")
        ),
        "fail_closed_simulation": _static_validator_fail_closed_simulation(
            value.get("fail_closed_simulation")
        ),
        "promotion_gate": _static_validator_promotion_gate(value.get("promotion_gate")),
        "validator_review_plan": _static_validator_review_plan(value.get("validator_review_plan")),
        "conservative_fail_closed_subset": _static_validator_conservative_fail_closed_subset(
            value.get("conservative_fail_closed_subset")
        ),
        "shadow_trial_plan": _static_validator_shadow_trial_plan(value.get("shadow_trial_plan")),
        "source_generalization_diagnostics": _static_validator_source_generalization(
            value.get("source_generalization_diagnostics")
        ),
        "recommendation": value.get("recommendation"),
    }


def _static_validator_candidate_row(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_set": value.get("feature_set"),
        "features": value.get("features") if isinstance(value.get("features"), Mapping) else {},
        "label": value.get("label"),
        "validator_action": value.get("validator_action"),
        "support": _num(value.get("support")),
        "purity": _number_or_none(value.get("purity")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "promotion_reason": value.get("promotion_reason"),
        "automation_policy": value.get("automation_policy"),
    }


def _static_validator_conflict_row(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_set": value.get("feature_set"),
        "features": value.get("features") if isinstance(value.get("features"), Mapping) else {},
        "support": _num(value.get("support")),
        "dominant_label": value.get("dominant_label"),
        "purity": _number_or_none(value.get("purity")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "needed_additional_labels": _num(value.get("needed_additional_labels")),
        "reason": value.get("reason"),
        "automation_policy": value.get("automation_policy"),
    }


def _static_validator_overlap_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rows = value.get("conflicting_overlap_pairs")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "overlap_pair_count": _num(value.get("overlap_pair_count")),
        "conflicting_overlap_pair_count": _num(value.get("conflicting_overlap_pair_count")),
        "same_action_overlap_pair_count": _num(value.get("same_action_overlap_pair_count")),
        "conflicting_overlap_pairs": [
            _static_validator_overlap_pair(row)
            for row in rows
            if isinstance(rows, list) and isinstance(row, Mapping)
        ][:5],
    }


def _static_validator_overlap_pair(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relation": value.get("relation"),
        "conflicting_actions": value.get("conflicting_actions") is True,
        "left_action": value.get("left_action"),
        "right_action": value.get("right_action"),
        "left_feature_set": value.get("left_feature_set"),
        "right_feature_set": value.get("right_feature_set"),
        "left_features": value.get("left_features") if isinstance(value.get("left_features"), Mapping) else {},
        "right_features": value.get("right_features") if isinstance(value.get("right_features"), Mapping) else {},
        "left_support": _num(value.get("left_support")),
        "right_support": _num(value.get("right_support")),
        "left_purity": _number_or_none(value.get("left_purity")),
        "right_purity": _number_or_none(value.get("right_purity")),
    }


def _static_validator_promotion_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "automatic_promotion_ready": value.get("automatic_promotion_ready") is True,
        "manual_review_ready": value.get("manual_review_ready") is True,
        "fail_closed_candidate_count": _num(value.get("fail_closed_candidate_count")),
        "safe_accept_candidate_count": _num(value.get("safe_accept_candidate_count")),
        "blocking_reasons": [str(item) for item in value.get("blocking_reasons", [])]
        if isinstance(value.get("blocking_reasons"), list)
        else [],
        "recommendation": value.get("recommendation"),
    }


def _static_validator_fail_closed_simulation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rows = value.get("sample_mismatches")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "policy_name": value.get("policy_name"),
        "evaluated_count": _num(value.get("evaluated_count")),
        "candidate_rule_count": _num(value.get("candidate_rule_count")),
        "matched_count": _num(value.get("matched_count")),
        "matched_rate": _number_or_none(value.get("matched_rate")),
        "action_counts": _int_count_map(value.get("action_counts")),
        "accuracy": _number_or_none(value.get("accuracy")),
        "macro_f1": _number_or_none(value.get("macro_f1")),
        "predicted_label_counts": _int_count_map(value.get("predicted_label_counts")),
        "mismatch_count": _num(value.get("mismatch_count")),
        "mismatch_rate": _number_or_none(value.get("mismatch_rate")),
        "sample_mismatches": [
            _static_validator_fail_closed_mismatch(row)
            for row in rows
            if isinstance(rows, list) and isinstance(row, Mapping)
        ][:5],
        "safe_for_automatic_validator_promotion": value.get("safe_for_automatic_validator_promotion") is True,
        "automation_policy": value.get("automation_policy"),
        "recommendation": value.get("recommendation"),
    }


def _static_validator_fail_closed_mismatch(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_index": _num(value.get("input_index")),
        "expected_label": value.get("expected_label"),
        "predicted_label": value.get("predicted_label"),
        "validator_action": value.get("validator_action"),
        "candidate_feature_set": value.get("candidate_feature_set"),
        "candidate_features": value.get("candidate_features")
        if isinstance(value.get("candidate_features"), Mapping)
        else {},
        "candidate_support": _num(value.get("candidate_support")),
        "candidate_purity": _number_or_none(value.get("candidate_purity")),
    }


def _static_validator_review_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rows = value.get("review_items")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "review_item_count": _num(value.get("review_item_count")),
        "blocking_review_item_count": _num(value.get("blocking_review_item_count")),
        "priority_reason_counts": _int_count_map(value.get("priority_reason_counts")),
        "recommended_next_review_type": value.get("recommended_next_review_type"),
        "review_items": [
            _static_validator_review_item(row)
            for row in rows
            if isinstance(rows, list) and isinstance(row, Mapping)
        ][:8],
        "automation_policy": value.get("automation_policy"),
        "recommendation": value.get("recommendation"),
    }


def _static_validator_review_item(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "priority_rank": _num(value.get("priority_rank")),
        "priority_group": value.get("priority_group"),
        "review_type": value.get("review_type"),
        "review_reason": value.get("review_reason"),
        "input_index": _num(value.get("input_index")) if "input_index" in value else None,
        "relation": value.get("relation"),
        "feature_set": value.get("feature_set"),
        "features": value.get("features") if isinstance(value.get("features"), Mapping) else {},
        "candidate_feature_set": value.get("candidate_feature_set"),
        "candidate_features": value.get("candidate_features")
        if isinstance(value.get("candidate_features"), Mapping)
        else {},
        "left_feature_set": value.get("left_feature_set"),
        "right_feature_set": value.get("right_feature_set"),
        "left_features": value.get("left_features") if isinstance(value.get("left_features"), Mapping) else {},
        "right_features": value.get("right_features") if isinstance(value.get("right_features"), Mapping) else {},
        "left_action": value.get("left_action"),
        "right_action": value.get("right_action"),
        "validator_action": value.get("validator_action"),
        "candidate_action": value.get("candidate_action"),
        "expected_label": value.get("expected_label"),
        "predicted_label": value.get("predicted_label"),
        "dominant_label": value.get("dominant_label"),
        "label": value.get("label"),
        "support": _num(value.get("support")),
        "candidate_support": _num(value.get("candidate_support")),
        "left_support": _num(value.get("left_support")),
        "right_support": _num(value.get("right_support")),
        "purity": _number_or_none(value.get("purity")),
        "candidate_purity": _number_or_none(value.get("candidate_purity")),
        "left_purity": _number_or_none(value.get("left_purity")),
        "right_purity": _number_or_none(value.get("right_purity")),
        "needed_additional_labels": _num(value.get("needed_additional_labels")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "reason": value.get("reason"),
        "promotion_reason": value.get("promotion_reason"),
        "automation_policy": value.get("automation_policy"),
    }


def _static_validator_conservative_fail_closed_subset(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    included = value.get("included_candidate_rules")
    excluded = value.get("excluded_candidate_rules")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "source_candidate_rule_count": _num(value.get("source_candidate_rule_count")),
        "source_fail_closed_rule_count": _num(value.get("source_fail_closed_rule_count")),
        "included_rule_count": _num(value.get("included_rule_count")),
        "excluded_rule_count": _num(value.get("excluded_rule_count")),
        "exclusion_reason_counts": _int_count_map(value.get("exclusion_reason_counts")),
        "included_candidate_rules": [
            _static_validator_subset_candidate(row)
            for row in included
            if isinstance(included, list) and isinstance(row, Mapping)
        ][:8],
        "excluded_candidate_rules": [
            _static_validator_subset_candidate(row)
            for row in excluded
            if isinstance(excluded, list) and isinstance(row, Mapping)
        ][:8],
        "simulation": _static_validator_fail_closed_simulation(value.get("simulation")),
        "manual_trial_ready": value.get("manual_trial_ready") is True,
        "safe_for_automatic_validator_promotion": value.get("safe_for_automatic_validator_promotion") is True,
        "automation_policy": value.get("automation_policy"),
        "recommendation": value.get("recommendation"),
    }


def _static_validator_shadow_trial_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rows = value.get("trial_rules")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "trial_mode": value.get("trial_mode"),
        "shadow_trial_ready": value.get("shadow_trial_ready") is True,
        "trial_rule_count": _num(value.get("trial_rule_count")),
        "trial_action_counts": _int_count_map(value.get("trial_action_counts")),
        "offline_simulation_mismatch_count": _num(value.get("offline_simulation_mismatch_count")),
        "offline_simulation_matched_count": _num(value.get("offline_simulation_matched_count")),
        "offline_simulation_matched_rate": _number_or_none(value.get("offline_simulation_matched_rate")),
        "source_diverse_rule_count": _num(value.get("source_diverse_rule_count")),
        "weak_source_support_rule_count": _num(value.get("weak_source_support_rule_count")),
        "source_conflict_rule_count": _num(value.get("source_conflict_rule_count")),
        "trial_rules": [
            _static_validator_shadow_trial_rule(row)
            for row in rows
            if isinstance(rows, list) and isinstance(row, Mapping)
        ][:8],
        "required_preconditions": [str(item) for item in value.get("required_preconditions", [])]
        if isinstance(value.get("required_preconditions"), list)
        else [],
        "telemetry_fields": [str(item) for item in value.get("telemetry_fields", [])]
        if isinstance(value.get("telemetry_fields"), list)
        else [],
        "validator_behavior_change_allowed": value.get("validator_behavior_change_allowed") is True,
        "safe_for_automatic_validator_promotion": value.get("safe_for_automatic_validator_promotion") is True,
        "automation_policy": value.get("automation_policy"),
        "recommendation": value.get("recommendation"),
    }


def _static_validator_shadow_trial_rule(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_rule_id": value.get("trial_rule_id"),
        "feature_set": value.get("feature_set"),
        "features": value.get("features") if isinstance(value.get("features"), Mapping) else {},
        "shadow_action": value.get("shadow_action"),
        "support": _num(value.get("support")),
        "purity": _number_or_none(value.get("purity")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "matched_count": _num(value.get("matched_count")),
        "source_group_count": _num(value.get("source_group_count")),
        "source_file_hash_count": _num(value.get("source_file_hash_count")),
        "extraction_counts": _int_count_map(value.get("extraction_counts")),
        "expected_label_counts_by_source": _int_count_map(value.get("expected_label_counts_by_source")),
        "source_label_counts_by_source": _int_count_map(value.get("source_label_counts_by_source")),
        "source_label_conflict": value.get("source_label_conflict") is True,
        "expected_label_conflict": value.get("expected_label_conflict") is True,
        "weak_source_support": value.get("weak_source_support") is True,
        "review_status": value.get("review_status"),
    }


def _static_validator_source_generalization(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    top_rows = value.get("top_candidate_source_summaries")
    weak_rows = value.get("weak_source_support_examples")
    conflict_rows = value.get("source_conflict_examples")
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "candidate_rule_count": _num(value.get("candidate_rule_count")),
        "candidate_with_multi_source_group_count": _num(value.get("candidate_with_multi_source_group_count")),
        "candidate_with_multi_source_file_count": _num(value.get("candidate_with_multi_source_file_count")),
        "conservative_subset_rule_count": _num(value.get("conservative_subset_rule_count")),
        "conservative_subset_multi_source_group_count": _num(
            value.get("conservative_subset_multi_source_group_count")
        ),
        "conservative_subset_multi_source_file_count": _num(
            value.get("conservative_subset_multi_source_file_count")
        ),
        "weak_source_support_count": _num(value.get("weak_source_support_count")),
        "source_conflict_count": _num(value.get("source_conflict_count")),
        "top_candidate_source_summaries": [
            _static_validator_source_generalization_row(row)
            for row in top_rows
            if isinstance(top_rows, list) and isinstance(row, Mapping)
        ][:8],
        "weak_source_support_examples": [
            _static_validator_source_generalization_row(row)
            for row in weak_rows
            if isinstance(weak_rows, list) and isinstance(row, Mapping)
        ][:5],
        "source_conflict_examples": [
            _static_validator_source_generalization_row(row)
            for row in conflict_rows
            if isinstance(conflict_rows, list) and isinstance(row, Mapping)
        ][:5],
        "generalization_claim_allowed": value.get("generalization_claim_allowed") is True,
        "automation_policy": value.get("automation_policy"),
        "recommendation": value.get("recommendation"),
    }


def _static_validator_source_generalization_row(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_static_validator_candidate_row(value),
        "matched_count": _num(value.get("matched_count")),
        "source_group_count": _num(value.get("source_group_count")),
        "source_file_hash_count": _num(value.get("source_file_hash_count")),
        "source_group_counts": _int_count_map(value.get("source_group_counts")),
        "sample_source_path_hashes": [str(item) for item in value.get("sample_source_path_hashes", [])]
        if isinstance(value.get("sample_source_path_hashes"), list)
        else [],
        "extraction_counts": _int_count_map(value.get("extraction_counts")),
        "expected_label_counts_by_source": _int_count_map(value.get("expected_label_counts_by_source")),
        "source_label_counts_by_source": _int_count_map(value.get("source_label_counts_by_source")),
        "expected_label_conflict": value.get("expected_label_conflict") is True,
        "source_label_conflict": value.get("source_label_conflict") is True,
    }


def _static_validator_subset_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_static_validator_candidate_row(value),
        "subset_policy": value.get("subset_policy"),
        "subset_decision": value.get("subset_decision"),
        "exclusion_reasons": [str(item) for item in value.get("exclusion_reasons", [])]
        if isinstance(value.get("exclusion_reasons"), list)
        else [],
    }


def _policy_failure_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for policy_name, diagnostic in value.items():
        if not isinstance(diagnostic, Mapping):
            continue
        result[str(policy_name)] = {
            "schema_version": diagnostic.get("schema_version"),
            "prompt_safe_summary": diagnostic.get("prompt_safe_summary") is True,
            "mismatch_count": _num(diagnostic.get("mismatch_count")),
            "top_mismatch_label_transitions": _int_count_map(
                diagnostic.get("top_mismatch_label_transitions")
            ),
            "top_mismatch_semantic_hints": _int_count_map(
                diagnostic.get("top_mismatch_semantic_hints")
            ),
            "top_mismatch_risk_tags": _int_count_map(diagnostic.get("top_mismatch_risk_tags")),
        }
    return dict(sorted(result.items()))


def _local_judge_goldset_chain_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-chain-verification-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set chain schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set chain summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "local judge gold-set chain verifier must not claim provider metrics"}
    steps = value.get("steps") if isinstance(value.get("steps"), list) else []
    checks = value.get("checks") if isinstance(value.get("checks"), list) else []
    ready_steps = sum(1 for step in steps if isinstance(step, Mapping) and step.get("ready") is True)
    failed_checks = [check for check in checks if isinstance(check, Mapping) and check.get("ok") is not True]
    ready = value.get("ready") is True and value.get("chain_ready") is True
    base = {
        "goldset_chain_ready": ready,
        "step_ready_counts": {
            "ready": ready_steps,
            "total": len(steps),
        },
        "failed_check_count": len(failed_checks),
        "blocking_reasons": value.get("blocking_reasons") if isinstance(value.get("blocking_reasons"), list) else [],
        "next_action": value.get("next_action"),
        "recommendation": value.get("next_action"),
    }
    if not ready:
        return {
            **base,
            "status": "warn",
            "evidence": "local judge gold-set chain verifier has not passed all human-label chain gates",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "local judge gold-set chain verifier passed labels, import, validation, and quality-eval gates",
    }


def _local_judge_goldset_validation_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-validation-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set validation schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set validation summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "local judge gold-set validation must not claim provider metrics"}
    if value.get("gold_text_written") is not False:
        return {"status": "fail", "evidence": "local judge gold-set validation summary must not write gold text"}
    ready = value.get("ready") is True and value.get("ready_for_local_judge_quality_eval") is True
    base = {
        "goldset_validation_ready": ready,
        "row_count": _num(value.get("row_count")),
        "valid_row_count": _num(value.get("valid_row_count")),
        "invalid_row_count": _num(value.get("invalid_row_count")),
        "missing_text_count": _num(value.get("missing_text_count")),
        "invalid_expected_label_count": _num(value.get("invalid_expected_label_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "min_samples": _num(value.get("min_samples")),
        "min_label_count": _num(value.get("min_label_count")),
        "recommendation": value.get("recommendation"),
    }
    if not ready:
        return {
            **base,
            "status": "warn",
            "evidence": "local judge gold-set validation ran but is not ready for quality eval",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "local judge gold-set labeled JSONL is ready for quality eval",
    }


def _local_judge_goldset_import_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-csv-import-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set CSV import schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set CSV import summary is not marked prompt-safe"}
    if value.get("gold_text_written") is not True:
        return {"status": "fail", "evidence": "labeled gold import must be based on rows with local candidate text"}
    ready = value.get("ready_for_local_judge_goldset_validate") is True and value.get("output_written") is True
    base = {
        "goldset_import_ready": ready,
        "row_count": _num(value.get("row_count")),
        "rows_with_text_count": _num(value.get("rows_with_text_count")),
        "rows_missing_text_count": _num(value.get("rows_missing_text_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "invalid_expected_label_count": _num(value.get("invalid_expected_label_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "output_written": value.get("output_written") is True,
        "allow_incomplete_output": value.get("allow_incomplete_output") is True,
        "ready_for_validate": value.get("ready_for_local_judge_goldset_validate") is True,
        "recommendation": value.get("recommendation"),
    }
    if base["allow_incomplete_output"]:
        return {
            **base,
            "status": "fail",
            "evidence": "CSV import wrote incomplete debug output; rerun without --allow-incomplete-output",
        }
    if not ready:
        return {
            **base,
            "status": "warn",
            "evidence": "CSV import is not ready; finish labels before gold-set validation",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "labeled CSV import wrote a gold JSONL ready for validation",
    }


def _local_judge_goldset_progress_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-annotation-progress-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set annotation progress schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set annotation progress summary is not marked prompt-safe"}
    ready_for_import = value.get("ready_for_labeled_jsonl_import") is True
    ready_after_import = value.get("ready_for_local_judge_goldset_validate_after_import") is True
    base = {
        "goldset_progress_ready": ready_for_import,
        "row_count": _num(value.get("row_count")),
        "rows_with_text_count": _num(value.get("rows_with_text_count")),
        "rows_missing_text_count": _num(value.get("rows_missing_text_count")),
        "rows_with_expected_label_count": _num(value.get("rows_with_expected_label_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "invalid_expected_label_count": _num(value.get("invalid_expected_label_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "source_label_counts": _int_count_map(value.get("source_label_counts")),
        "semantic_hint_counts": _int_count_map(value.get("semantic_hint_counts")),
        "review_priority_counts": _int_count_map(value.get("review_priority_counts")),
        "risk_tag_counts": _int_count_map(value.get("risk_tag_counts")),
        "pending_source_label_counts": _int_count_map(value.get("pending_source_label_counts")),
        "pending_semantic_hint_counts": _int_count_map(value.get("pending_semantic_hint_counts")),
        "pending_review_priority_counts": _int_count_map(value.get("pending_review_priority_counts")),
        "pending_risk_tag_counts": _int_count_map(value.get("pending_risk_tag_counts")),
        "invalid_source_label_counts": _int_count_map(value.get("invalid_source_label_counts")),
        "invalid_semantic_hint_counts": _int_count_map(value.get("invalid_semantic_hint_counts")),
        "invalid_review_priority_counts": _int_count_map(value.get("invalid_review_priority_counts")),
        "invalid_risk_tag_counts": _int_count_map(value.get("invalid_risk_tag_counts")),
        "annotation_completion_rate": _number_or_none(value.get("annotation_completion_rate")),
        "min_samples": _num(value.get("min_samples")),
        "min_label_count": _num(value.get("min_label_count")),
        "labels_with_min_count": _num(value.get("labels_with_min_count")),
        "label_balance_ready": value.get("label_balance_ready") is True,
        "ready_for_labeled_jsonl_import": ready_for_import,
        "ready_for_validate_after_import": ready_after_import,
        "guide_path": value.get("guide_path"),
        "worklist_path": value.get("worklist_path"),
        "worklist_written": value.get("worklist_written") is True,
        "worklist_row_count": _num(value.get("worklist_row_count")),
        "worklist_pending_count": _num(value.get("worklist_pending_count")),
        "worklist_invalid_count": _num(value.get("worklist_invalid_count")),
        "worklist_schema_version": value.get("worklist_schema_version"),
        "recommendation": value.get("recommendation"),
    }
    if base["row_count"] <= 0:
        return {**base, "status": "warn", "evidence": "annotation CSV progress has no rows"}
    if not ready_for_import:
        return {**base, "status": "warn", "evidence": "annotation CSV still needs labels, valid labels, or candidate text"}
    if not ready_after_import:
        return {
            **base,
            "status": "warn",
            "evidence": "annotation CSV can import but does not yet meet validation sample or label-balance thresholds",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "annotation CSV labels are complete enough to import and validate",
    }


def _local_judge_goldset_labels_template_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-labels-template-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set labels template schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set labels template summary is not marked prompt-safe"}
    ready = value.get("ready_for_apply_labels") is True and value.get("output_written") is True
    base = {
        "goldset_labels_template_ready": ready,
        "row_count": _num(value.get("row_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "rows_missing_id_count": _num(value.get("rows_missing_id_count")),
        "rows_missing_text_hash_count": _num(value.get("rows_missing_text_hash_count")),
        "duplicate_label_key_count": _num(value.get("duplicate_label_key_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "label_status_counts": _int_count_map(value.get("label_status_counts")),
        "source_label_counts": _int_count_map(value.get("source_label_counts")),
        "semantic_hint_counts": _int_count_map(value.get("semantic_hint_counts")),
        "risk_tag_counts": _int_count_map(value.get("risk_tag_counts")),
        "ready_for_apply_labels": value.get("ready_for_apply_labels") is True,
        "output_written": value.get("output_written") is True,
        "recommendation": value.get("recommendation"),
    }
    if base["row_count"] <= 0:
        return {**base, "status": "warn", "evidence": "labels template has no rows to apply"}
    if base["rows_missing_id_count"] > 0 or base["rows_missing_text_hash_count"] > 0:
        return {**base, "status": "fail", "evidence": "labels template has rows missing id or text_hash"}
    if base["duplicate_label_key_count"] > 0:
        return {**base, "status": "fail", "evidence": "labels template has duplicate id/text_hash keys"}
    if not ready:
        return {**base, "status": "warn", "evidence": "labels template is not ready for apply-labels"}
    return {
        **base,
        "status": "pass",
        "evidence": "prompt-safe labels JSONL template is ready for manual expected_label values",
    }


def _local_judge_goldset_suggestions_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-label-suggestions-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set label suggestions schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set suggestions summary is not marked prompt-safe"}
    if value.get("ready_for_apply_labels") is not False:
        return {"status": "fail", "evidence": "suggestions must not be marked ready for apply-labels"}
    if value.get("ready_for_goldset_import_after_apply") is not False:
        return {"status": "fail", "evidence": "suggestions must not unlock gold-set import"}
    ready = value.get("suggestions_ready_for_manual_review") is True and value.get("output_written") is True
    base = {
        "goldset_suggestions_ready": ready,
        "suggestion_count": _num(value.get("suggestion_count")),
        "source_row_count": _num(value.get("source_row_count")),
        "model_called_count": _num(value.get("model_called_count")),
        "local_judge_error_count": _num(value.get("local_judge_error_count")),
        "rows_missing_text_count": _num(value.get("rows_missing_text_count")),
        "static_safety_clamp_count": _num(value.get("static_safety_clamp_count")),
        "suggested_label_counts": _int_count_map(value.get("suggested_label_counts")),
        "model_label_counts": _int_count_map(value.get("model_label_counts")),
        "rule_label_counts": _int_count_map(value.get("rule_label_counts")),
        "source_label_counts": _int_count_map(value.get("source_label_counts")),
        "local_judge_action_counts": _int_count_map(value.get("local_judge_action_counts")),
        "label_reason_code_counts": _int_count_map(value.get("label_reason_code_counts")),
        "model_to_suggested_label_counts": _int_count_map(value.get("model_to_suggested_label_counts")),
        "rule_to_suggested_label_counts": _int_count_map(value.get("rule_to_suggested_label_counts")),
        "source_to_suggested_label_counts": _int_count_map(value.get("source_to_suggested_label_counts")),
        "suggestions_ready_for_manual_review": value.get("suggestions_ready_for_manual_review") is True,
        "ready_for_apply_labels": value.get("ready_for_apply_labels") is True,
        "ready_for_goldset_import_after_apply": value.get("ready_for_goldset_import_after_apply") is True,
        "output_written": value.get("output_written") is True,
        "recommendation": value.get("recommendation"),
    }
    if base["suggestion_count"] <= 0:
        return {**base, "status": "warn", "evidence": "local judge suggestions summary has no rows"}
    if base["local_judge_error_count"] > 0:
        return {**base, "status": "warn", "evidence": "local judge suggestions include model-call errors"}
    if base["rows_missing_text_count"] > 0:
        return {**base, "status": "warn", "evidence": "local judge suggestions found annotation rows missing text"}
    if not ready:
        return {**base, "status": "warn", "evidence": "local judge suggestions are not ready for manual review"}
    return {
        **base,
        "status": "pass",
        "evidence": "local judge wrote prompt-safe suggestions for manual review only",
    }


def _local_judge_goldset_suggestion_review_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-suggestion-review-csv-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set suggestion review schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set suggestion review summary is not marked prompt-safe"}
    if value.get("ready_for_apply_labels") is not False:
        return {"status": "fail", "evidence": "suggestion review CSV must not be marked ready for apply-labels"}
    if value.get("ready_for_goldset_import_after_apply") is not False:
        return {"status": "fail", "evidence": "suggestion review CSV must not unlock gold-set import"}
    ready = value.get("ready_for_manual_review") is True and value.get("output_written") is True
    base = {
        "goldset_suggestion_review_ready": ready,
        "row_count": _num(value.get("row_count")),
        "suggestion_row_count": _num(value.get("suggestion_row_count")),
        "matched_suggestion_count": _num(value.get("matched_suggestion_count")),
        "missing_suggestion_count": _num(value.get("missing_suggestion_count")),
        "unmatched_suggestion_count": _num(value.get("unmatched_suggestion_count")),
        "duplicate_csv_key_count": _num(value.get("duplicate_csv_key_count")),
        "duplicate_suggestion_key_count": _num(value.get("duplicate_suggestion_key_count")),
        "suggestions_missing_key_count": _num(value.get("suggestions_missing_key_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "suggested_label_counts": _int_count_map(value.get("suggested_label_counts")),
        "suggestion_matches_source_label_counts": _int_count_map(value.get("suggestion_matches_source_label_counts")),
        "ready_for_manual_review": value.get("ready_for_manual_review") is True,
        "ready_for_apply_labels": value.get("ready_for_apply_labels") is True,
        "ready_for_goldset_import_after_apply": value.get("ready_for_goldset_import_after_apply") is True,
        "output_written": value.get("output_written") is True,
        "csv_text_written": value.get("csv_text_written") is True,
        "recommendation": value.get("recommendation"),
    }
    if base["row_count"] <= 0:
        return {**base, "status": "warn", "evidence": "suggestion review CSV summary has no rows"}
    if base["missing_suggestion_count"] > 0 or base["unmatched_suggestion_count"] > 0:
        return {**base, "status": "warn", "evidence": "suggestion review CSV join is incomplete"}
    if (
        base["duplicate_csv_key_count"] > 0
        or base["duplicate_suggestion_key_count"] > 0
        or base["suggestions_missing_key_count"] > 0
    ):
        return {**base, "status": "fail", "evidence": "suggestion review CSV join has duplicate or missing keys"}
    if not ready:
        return {**base, "status": "warn", "evidence": "suggestion review CSV is not ready for manual review"}
    return {
        **base,
        "status": "pass",
        "evidence": "local review CSV joins candidate text with suggestions for manual labeling only",
    }


def _local_judge_goldset_review_plan_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-review-plan-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set review-plan schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge review-plan summary is not marked prompt-safe"}
    if value.get("ready_for_apply_labels") is not False:
        return {"status": "fail", "evidence": "review plan must not be marked ready for apply-labels"}
    if value.get("ready_for_goldset_import_after_apply") is not False:
        return {"status": "fail", "evidence": "review plan must not unlock gold-set import"}
    if value.get("suggested_labels_not_auto_applied") is not True:
        return {"status": "fail", "evidence": "review plan must explicitly avoid auto-applying suggestions"}
    annotation_complete = value.get("annotation_complete") is True
    ready = bool(
        value.get("output_written") is True
        and (
            value.get("ready_for_manual_review") is True
            or annotation_complete
        )
    )
    base = {
        "goldset_review_plan_ready": ready,
        "row_count": _num(value.get("row_count")),
        "plan_row_count": _num(value.get("plan_row_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "invalid_expected_label_count": _num(value.get("invalid_expected_label_count")),
        "plan_disagreement_row_count": _num(value.get("plan_disagreement_row_count")),
        "plan_high_risk_row_count": _num(value.get("plan_high_risk_row_count")),
        "source_label_counts": _int_count_map(value.get("source_label_counts")),
        "rule_label_counts": _int_count_map(value.get("rule_label_counts")),
        "model_label_counts": _int_count_map(value.get("model_label_counts")),
        "suggested_label_counts": _int_count_map(value.get("suggested_label_counts")),
        "plan_expected_label_status_counts": _int_count_map(value.get("plan_expected_label_status_counts")),
        "plan_review_reason_counts": _int_count_map(value.get("plan_review_reason_counts")),
        "plan_high_risk_tag_counts": _int_count_map(value.get("plan_high_risk_tag_counts")),
        "plan_semantic_hint_counts": _int_count_map(value.get("plan_semantic_hint_counts")),
        "ready_for_manual_review": value.get("ready_for_manual_review") is True,
        "annotation_complete": annotation_complete,
        "ready_for_apply_labels": value.get("ready_for_apply_labels") is True,
        "ready_for_goldset_import_after_apply": value.get("ready_for_goldset_import_after_apply") is True,
        "suggested_labels_not_auto_applied": value.get("suggested_labels_not_auto_applied") is True,
        "output_written": value.get("output_written") is True,
        "recommendation": value.get("recommendation"),
    }
    if base["row_count"] <= 0:
        return {**base, "status": "warn", "evidence": "review plan summary has no source rows"}
    if base["plan_row_count"] <= 0 and not annotation_complete:
        return {**base, "status": "warn", "evidence": "review plan has no pending or invalid rows"}
    if not ready:
        return {**base, "status": "warn", "evidence": "review plan is not ready for manual review"}
    if annotation_complete:
        return {
            **base,
            "status": "pass",
            "evidence": "prompt-safe review plan is complete; no pending expected_label rows remain",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "prompt-safe prioritized review plan is ready for manual expected_label work",
    }


def _local_judge_goldset_labels_from_csv_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-labels-from-csv-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set labels-from-csv schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge labels-from-csv summary is not marked prompt-safe"}
    if value.get("labels_jsonl_text_written") is not False:
        return {"status": "fail", "evidence": "labels-from-csv output must not write candidate text"}
    if value.get("suggested_labels_not_auto_applied") is not True:
        return {"status": "fail", "evidence": "labels-from-csv must explicitly avoid auto-applying suggestions"}
    ready = value.get("ready_for_apply_labels") is True and value.get("output_written") is True
    base = {
        "goldset_labels_from_csv_ready": ready,
        "row_count": _num(value.get("row_count")),
        "valid_label_row_count": _num(value.get("valid_label_row_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "invalid_label_row_count": _num(value.get("invalid_label_row_count")),
        "rows_missing_id_count": _num(value.get("rows_missing_id_count")),
        "rows_missing_text_hash_count": _num(value.get("rows_missing_text_hash_count")),
        "duplicate_label_key_count": _num(value.get("duplicate_label_key_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "suggested_label_counts": _int_count_map(value.get("suggested_label_counts")),
        "expected_to_suggested_label_counts": _int_count_map(value.get("expected_to_suggested_label_counts")),
        "sample_count_ready": value.get("sample_count_ready") is True,
        "label_balance_ready": value.get("label_balance_ready") is True,
        "ready_for_apply_labels": value.get("ready_for_apply_labels") is True,
        "ready_for_goldset_import_after_apply": value.get("ready_for_goldset_import_after_apply") is True,
        "suggested_labels_not_auto_applied": value.get("suggested_labels_not_auto_applied") is True,
        "output_written": value.get("output_written") is True,
        "labels_jsonl_text_written": value.get("labels_jsonl_text_written") is True,
        "recommendation": value.get("recommendation"),
    }
    if base["row_count"] <= 0:
        return {**base, "status": "warn", "evidence": "labels-from-csv summary has no rows"}
    if base["rows_missing_id_count"] > 0 or base["rows_missing_text_hash_count"] > 0:
        return {**base, "status": "fail", "evidence": "labels-from-csv found rows missing id or text_hash"}
    if base["duplicate_label_key_count"] > 0:
        return {**base, "status": "fail", "evidence": "labels-from-csv found duplicate id/text_hash keys"}
    if base["invalid_label_row_count"] > 0:
        return {**base, "status": "warn", "evidence": "labels-from-csv found invalid expected_label values"}
    if base["expected_label_pending_count"] > 0:
        return {**base, "status": "warn", "evidence": "labels-from-csv still has pending expected_label values"}
    if not ready:
        return {**base, "status": "warn", "evidence": "labels-from-csv output is not ready for apply-labels"}
    return {
        **base,
        "status": "pass",
        "evidence": "prompt-safe labels JSONL was exported from manual CSV expected_label values",
    }


def _local_judge_goldset_labels_validation_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-labels-validation-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set labels validation schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set labels validation summary is not marked prompt-safe"}
    ready = value.get("ready_for_apply_labels") is True
    base = {
        "goldset_labels_validation_ready": ready,
        "row_count": _num(value.get("row_count")),
        "valid_label_row_count": _num(value.get("valid_label_row_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "invalid_label_row_count": _num(value.get("invalid_label_row_count")),
        "rows_missing_id_count": _num(value.get("rows_missing_id_count")),
        "rows_missing_text_hash_count": _num(value.get("rows_missing_text_hash_count")),
        "duplicate_label_key_count": _num(value.get("duplicate_label_key_count")),
        "expected_label_counts": _int_count_map(value.get("expected_label_counts")),
        "label_status_counts": _int_count_map(value.get("label_status_counts")),
        "source_expected_label_match_count": _num(value.get("source_expected_label_match_count")),
        "source_expected_label_mismatch_count": _num(value.get("source_expected_label_mismatch_count")),
        "source_expected_label_audited_count": _num(value.get("source_expected_label_audited_count")),
        "source_expected_label_match_rate": value.get("source_expected_label_match_rate"),
        "source_expected_label_mismatch_counts": _int_count_map(value.get("source_expected_label_mismatch_counts")),
        "possible_rule_self_confirmation": value.get("possible_rule_self_confirmation") is True,
        "gold_labels_independent_from_rules_ready": value.get("gold_labels_independent_from_rules_ready") is True,
        "structure_ready": value.get("structure_ready") is True,
        "complete_labels_ready": value.get("complete_labels_ready") is True,
        "sample_count_ready": value.get("sample_count_ready") is True,
        "label_balance_ready": value.get("label_balance_ready") is True,
        "ready_for_apply_labels": ready,
        "ready_for_goldset_import_after_apply": value.get("ready_for_goldset_import_after_apply") is True,
        "recommendation": value.get("recommendation"),
    }
    if base["row_count"] <= 0:
        return {**base, "status": "warn", "evidence": "labels validation has no rows"}
    if base["rows_missing_id_count"] > 0 or base["rows_missing_text_hash_count"] > 0:
        return {**base, "status": "fail", "evidence": "labels validation found rows missing id or text_hash"}
    if base["duplicate_label_key_count"] > 0:
        return {**base, "status": "fail", "evidence": "labels validation found duplicate id/text_hash keys"}
    if base["invalid_label_row_count"] > 0:
        return {**base, "status": "warn", "evidence": "labels validation found invalid expected_label values"}
    if base["expected_label_pending_count"] > 0:
        return {**base, "status": "warn", "evidence": "labels validation still has pending expected_label values"}
    if not ready:
        return {**base, "status": "warn", "evidence": "labels validation is not ready for apply-labels"}
    return {
        **base,
        "status": "pass",
        "evidence": "prompt-safe labels JSONL is complete and ready for apply-labels",
    }


def _local_judge_goldset_template_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-local-judge-goldset-template-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected local judge gold-set template schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "local judge gold-set template summary is not marked prompt-safe"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "local judge gold-set template must not claim provider metrics"}
    base = {
        "goldset_template_ready": _num(value.get("selected_count")) > 0,
        "selected_count": _num(value.get("selected_count")),
        "expected_label_pending_count": _num(value.get("expected_label_pending_count")),
        "include_text": bool(value.get("include_text")),
        "gold_text_written": bool(value.get("gold_text_written")),
        "selected_recovered_text_count": _num(value.get("selected_recovered_text_count")),
        "text_recovery": value.get("text_recovery") if isinstance(value.get("text_recovery"), Mapping) else {},
        "source_label_counts": _int_count_map(value.get("source_label_counts")),
        "semantic_hint_counts": _int_count_map(value.get("semantic_hint_counts")),
        "risk_tag_counts": _int_count_map(value.get("risk_tag_counts")),
        "recommendation": value.get("recommendation"),
    }
    if base["selected_count"] <= 0:
        return {**base, "status": "warn", "evidence": "gold-set template ran but selected no candidates"}
    if value.get("manual_labeling_required") is not True:
        return {**base, "status": "fail", "evidence": "gold-set template must require manual labeling"}
    if value.get("ready_for_local_judge_quality_eval") is not False:
        return {
            **base,
            "status": "fail",
            "evidence": "blank gold-set template must not mark ready_for_local_judge_quality_eval=true",
        }
    if base["gold_text_written"]:
        return {
            **base,
            "status": "warn",
            "evidence": "gold-set template includes raw text for local annotation; keep it out of reports/commits",
        }
    return {
        **base,
        "status": "pass",
        "evidence": "prompt-safe gold-set labeling template is ready for manual labels",
    }


def _offline_local_judge_matrix_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-offline-semantic-matrix-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected offline local-judge matrix schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "offline local-judge matrix summary is not marked prompt-safe"}
    if value.get("judge_mode") != "openai-compatible":
        return {
            "status": "fail",
            "evidence": f"offline local-judge matrix judge_mode={value.get('judge_mode')}",
        }
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "offline local-judge matrix must not claim provider metrics"}
    if value.get("include_candidate_text") is not True:
        return {
            "status": "fail",
            "evidence": "offline local-judge matrix did not include candidate text for model calls",
        }
    aggregate = value.get("aggregate") if isinstance(value.get("aggregate"), Mapping) else {}
    completed_sources = _num(aggregate.get("completed_source_count"))
    supported_sources = _num(aggregate.get("supported_source_count"))
    action_counts = (
        aggregate.get("local_judge_action_counts")
        if isinstance(aggregate.get("local_judge_action_counts"), Mapping)
        else {}
    )
    model_called = _num(action_counts.get("model_called"))
    budget = (
        aggregate.get("local_judge_budget_diagnostics")
        if isinstance(aggregate.get("local_judge_budget_diagnostics"), Mapping)
        else {}
    )
    candidate_label_diagnostics = _matrix_candidate_label_diagnostics(aggregate)
    prompt_extraction_diagnostics = _matrix_prompt_extraction_diagnostics(aggregate)
    if completed_sources <= 0 or supported_sources <= 0:
        return {
            "status": "fail",
            "evidence": "offline local-judge matrix has no completed supported sources",
            "completed_source_count": completed_sources,
            "supported_source_count": supported_sources,
            "model_called_count": model_called,
        }
    if model_called <= 0:
        return {
            "status": "fail",
            "evidence": "offline local-judge matrix made no local judge model calls",
            "completed_source_count": completed_sources,
            "supported_source_count": supported_sources,
            "model_called_count": model_called,
        }
    return {
        "status": "pass",
        "evidence": "offline matrix used a real configured OpenAI-compatible local judge",
        "completed_source_count": completed_sources,
        "supported_source_count": supported_sources,
        "model_called_count": model_called,
        "budget_exhausted_source_count": _num(budget.get("budget_exhausted_source_count")),
        "aggregate_budget_coverage_rate": budget.get("aggregate_budget_coverage_rate"),
        "prompt_extraction_diagnostics": prompt_extraction_diagnostics,
        "candidate_label_diagnostics": candidate_label_diagnostics,
        "recommendation": aggregate.get("recommendation"),
    }


def _offline_local_judge_matrix_smoke_quality(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path)
    if value.get("schema_version") != "prefix-offline-local-judge-matrix-smoke-summary-v1":
        return {
            "status": "fail",
            "evidence": f"unexpected offline local-judge matrix smoke schema={value.get('schema_version')}",
        }
    if value.get("prompt_safe_summary") is not True:
        return {"status": "fail", "evidence": "offline local-judge matrix smoke summary is not marked prompt-safe"}
    if value.get("fake_local_judge") is not True:
        return {"status": "fail", "evidence": "offline local-judge matrix smoke did not mark fake_local_judge=true"}
    if value.get("real_provider_metrics_available") is not False:
        return {"status": "fail", "evidence": "offline local-judge matrix smoke must not claim provider metrics"}
    digest = value.get("matrix_digest") if isinstance(value.get("matrix_digest"), Mapping) else {}
    budget = (
        digest.get("local_judge_budget_diagnostics")
        if isinstance(digest.get("local_judge_budget_diagnostics"), Mapping)
        else {}
    )
    candidate_label_diagnostics = (
        _candidate_label_diagnostics(digest)
        if isinstance(digest.get("candidate_label_diagnostics"), Mapping)
        else _matrix_candidate_label_diagnostics(digest)
    )
    prompt_extraction_diagnostics = _prompt_extraction_diagnostics(digest)
    completed_sources = _num(digest.get("completed_source_count"))
    supported_sources = _num(digest.get("supported_source_count"))
    request_count = _num(value.get("fake_local_judge_request_count"))
    if completed_sources <= 0 or supported_sources <= 0:
        return {
            "status": "fail",
            "evidence": "offline local-judge matrix smoke has no completed supported sources",
            "completed_source_count": completed_sources,
            "supported_source_count": supported_sources,
        }
    if request_count <= 0:
        return {
            "status": "fail",
            "evidence": "offline local-judge matrix smoke made no fake local judge requests",
            "completed_source_count": completed_sources,
            "supported_source_count": supported_sources,
        }
    return {
        "status": "warn",
        "evidence": (
            "offline local-judge matrix smoke used a fake local judge; it validates wiring and budget "
            "coverage only"
        ),
        "fake_local_judge": True,
        "fake_local_judge_request_count": request_count,
        "completed_source_count": completed_sources,
        "supported_source_count": supported_sources,
        "budget_exhausted_source_count": _num(budget.get("budget_exhausted_source_count")),
        "aggregate_budget_coverage_rate": budget.get("aggregate_budget_coverage_rate"),
        "prompt_extraction_diagnostics": prompt_extraction_diagnostics,
        "candidate_label_diagnostics": candidate_label_diagnostics,
        "recommendation": digest.get("recommendation"),
    }


def _matrix_candidate_label_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rule_label_counts": _int_count_map(value.get("rule_label_counts")),
        "model_label_counts": _int_count_map(value.get("model_label_counts")),
        "rule_to_model_label_counts": _int_count_map(value.get("rule_to_model_label_counts")),
        "model_to_final_label_counts": _int_count_map(value.get("model_to_final_label_counts")),
        "rule_to_final_label_counts": _int_count_map(value.get("rule_to_final_label_counts")),
        "static_safety_clamp_count": _num(value.get("static_safety_clamp_count")),
        "local_judge_effectiveness_diagnostics": _local_judge_effectiveness_diagnostics(
            value.get("local_judge_effectiveness_diagnostics")
        ),
    }


def _candidate_label_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = value.get("candidate_label_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return _matrix_candidate_label_diagnostics(value)
    return _matrix_candidate_label_diagnostics(diagnostics)


def _local_judge_effectiveness_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "schema_version": str(value.get("schema_version") or ""),
        "source_with_effectiveness_diagnostics_count": _num(
            value.get("source_with_effectiveness_diagnostics_count")
        ),
        "rule_review_candidate_count": _num(value.get("rule_review_candidate_count")),
        "model_called_rule_review_count": _num(value.get("model_called_rule_review_count")),
        "resolved_rule_review_count": _num(value.get("resolved_rule_review_count")),
        "called_resolved_rule_review_count": _num(value.get("called_resolved_rule_review_count")),
        "remaining_rule_review_count": _num(value.get("remaining_rule_review_count")),
        "resolution_rate": _number_or_none(value.get("resolution_rate")),
        "called_resolution_rate": _number_or_none(value.get("called_resolution_rate")),
        "rule_review_final_label_counts": _int_count_map(value.get("rule_review_final_label_counts")),
        "rule_review_model_label_counts": _int_count_map(value.get("rule_review_model_label_counts")),
        "rule_review_local_judge_action_counts": _int_count_map(
            value.get("rule_review_local_judge_action_counts")
        ),
    }


def _matrix_prompt_extraction_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extraction_counts": _int_count_map(value.get("extraction_counts")),
    }


def _prompt_extraction_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = value.get("prompt_extraction_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return _matrix_prompt_extraction_diagnostics(value)
    return _matrix_prompt_extraction_diagnostics(diagnostics)


def _int_count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        counts[str(key)] = _num(count)
    return dict(sorted(counts.items()))


def _quality_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "expected_label_coverage_count": _num(value.get("expected_label_coverage_count")),
        "predicted_label_coverage_count": _num(value.get("predicted_label_coverage_count")),
        "raw_model_label_coverage_count": _num(value.get("raw_model_label_coverage_count")),
        "missing_predicted_labels": _string_list(value.get("missing_predicted_labels")),
        "missing_raw_model_labels": _string_list(value.get("missing_raw_model_labels")),
        "zero_recall_labels": _string_list(value.get("zero_recall_labels")),
        "raw_model_zero_recall_labels": _string_list(value.get("raw_model_zero_recall_labels")),
        "per_label_recall": _number_map(value.get("per_label_recall")),
        "raw_model_per_label_recall": _number_map(value.get("raw_model_per_label_recall")),
        "predicted_label_collapse": value.get("predicted_label_collapse") is True,
        "raw_model_label_collapse": value.get("raw_model_label_collapse") is True,
        "max_predicted_label_rate": _number_or_none(value.get("max_predicted_label_rate")),
        "raw_model_max_label_rate": _number_or_none(value.get("raw_model_max_label_rate")),
        "accept_false_positive_count": _num(value.get("accept_false_positive_count")),
        "raw_model_accept_false_positive_count": _num(value.get("raw_model_accept_false_positive_count")),
        "reject_recall": _number_or_none(value.get("reject_recall")),
        "raw_model_reject_recall": _number_or_none(value.get("raw_model_reject_recall")),
        "review_recall": _number_or_none(value.get("review_recall")),
        "raw_model_review_recall": _number_or_none(value.get("raw_model_review_recall")),
        "failure_mode_codes": _string_list(value.get("failure_mode_codes")),
    }


def _mismatch_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "schema_version": value.get("schema_version"),
        "prompt_safe_summary": value.get("prompt_safe_summary") is True,
        "mismatch_count": _num(value.get("mismatch_count")),
        "raw_model_mismatch_count": _num(value.get("raw_model_mismatch_count")),
        "mismatch_label_transition_counts": _int_count_map(
            value.get("mismatch_label_transition_counts")
        ),
        "raw_model_mismatch_label_transition_counts": _int_count_map(
            value.get("raw_model_mismatch_label_transition_counts")
        ),
        "mismatch_semantic_hint_counts": _int_count_map(value.get("mismatch_semantic_hint_counts")),
        "raw_model_mismatch_semantic_hint_counts": _int_count_map(
            value.get("raw_model_mismatch_semantic_hint_counts")
        ),
        "mismatch_risk_tag_count_counts": _int_count_map(
            value.get("mismatch_risk_tag_count_counts")
        ),
        "raw_model_mismatch_risk_tag_count_counts": _int_count_map(
            value.get("raw_model_mismatch_risk_tag_count_counts")
        ),
    }


def _number_map(value: Any) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _number_or_none(item) for key, item in sorted(value.items())}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _artifact_variant(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else "unknown"


def _artifact_record(*, name: str, value: Any, manifest_file: Path) -> dict[str, Any]:
    path = _artifact_path(value, manifest_file=manifest_file)
    exists = bool(path and path.exists())
    size_bytes = path.stat().st_size if exists and path else 0
    return {
        "name": name,
        "path": str(path) if path else None,
        "exists": exists,
        "size_bytes": size_bytes,
        "nonempty": bool(exists and size_bytes > 0),
    }


def _semantic_guard_fake_smoke_summary_path(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path | None:
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    smoke_commands = commands.get("semantic_guard_fake_smoke") if isinstance(commands.get("semantic_guard_fake_smoke"), list) else []
    if not smoke_commands:
        return None
    output_dir = _command_option_value(str(smoke_commands[0]), "--output-dir")
    if not output_dir:
        return None
    return _artifact_path(str(Path(output_dir) / "semantic_guard_proxy_smoke_summary.json"), manifest_file=manifest_file)


def _offline_semantic_suite_summary_path(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path | None:
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    suite_commands = commands.get("offline_semantic_suite") if isinstance(commands.get("offline_semantic_suite"), list) else []
    if not suite_commands:
        return None
    output_dir = _command_option_value(str(suite_commands[0]), "--output-dir")
    if not output_dir:
        return None
    return _artifact_path(str(Path(output_dir) / "offline_semantic_suite_summary.json"), manifest_file=manifest_file)


def _local_judge_healthcheck_summary_path(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = local_judge.get("healthcheck_summary_path") if isinstance(local_judge, Mapping) else None
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    healthcheck_commands = (
        commands.get("local_judge_healthcheck")
        if isinstance(commands.get("local_judge_healthcheck"), list)
        else []
    )
    if not healthcheck_commands:
        return None
    value = _command_option_value(str(healthcheck_commands[0]), "--summary")
    if not value:
        return None
    return _artifact_path(value, manifest_file=manifest_file)


def _local_judge_goldset_template_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = local_judge.get("goldset_template_summary_path") if isinstance(local_judge, Mapping) else None
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    if not template_commands:
        return None
    value = _command_option_value(str(template_commands[0]), "--summary")
    if not value:
        return None
    return _artifact_path(value, manifest_file=manifest_file)


def _local_judge_goldset_progress_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_progress_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv progress" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return None


def _local_judge_goldset_labels_template_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_labels_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv labels-template" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return None


def _local_judge_goldset_labels_validation_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_labels_validation_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv validate-labels" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return None


def _local_judge_goldset_suggestions_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_suggestions_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv suggest-labels" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return None


def _local_judge_goldset_suggestion_review_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_suggestion_review_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv merge-suggestions" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return None


def _local_judge_goldset_review_plan_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_review_plan_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv review-plan" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return manifest_file.parent / "artifacts" / "local_judge_goldset_annotation_review_plan.local.summary.json"


def _local_judge_goldset_labels_from_csv_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_labels_from_csv_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv labels-from-csv" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return None


def _local_judge_goldset_import_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("goldset_annotation_import_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    template_commands = (
        commands.get("local_judge_goldset_template")
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    for command in template_commands:
        if "local_judge_goldset_csv import" not in str(command):
            continue
        value = _command_option_value(str(command), "--summary")
        if value:
            return _artifact_path(value, manifest_file=manifest_file)
    return None


def _local_judge_goldset_validation_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = local_judge.get("goldset_validation_summary_path") if isinstance(local_judge, Mapping) else None
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    validation_commands = (
        commands.get("local_judge_goldset_validate")
        if isinstance(commands.get("local_judge_goldset_validate"), list)
        else []
    )
    if not validation_commands:
        return None
    value = _command_option_value(str(validation_commands[0]), "--summary")
    if not value:
        return None
    return _artifact_path(value, manifest_file=manifest_file)


def _local_judge_quality_eval_summary_path(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = local_judge.get("quality_eval_summary_path") if isinstance(local_judge, Mapping) else None
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    quality_commands = (
        commands.get("local_judge_quality_eval")
        if isinstance(commands.get("local_judge_quality_eval"), list)
        else []
    )
    if not quality_commands:
        return None
    value = _command_option_value(str(quality_commands[0]), "--summary")
    if not value:
        return None
    return _artifact_path(value, manifest_file=manifest_file)


def _local_judge_policy_eval_summary_path(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = local_judge.get("policy_eval_summary_path") if isinstance(local_judge, Mapping) else None
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    policy_commands = (
        commands.get("local_judge_policy_eval")
        if isinstance(commands.get("local_judge_policy_eval"), list)
        else []
    )
    if not policy_commands:
        return manifest_file.parent / "artifacts" / "local_judge_policy_eval_summary.json"
    value = _command_option_value(str(policy_commands[0]), "--summary")
    if not value:
        return manifest_file.parent / "artifacts" / "local_judge_policy_eval_summary.json"
    return _artifact_path(value, manifest_file=manifest_file)


def _static_rule_calibration_eval_summary_path(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = (
        local_judge.get("static_rule_calibration_eval_summary_path")
        if isinstance(local_judge, Mapping)
        else None
    )
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    calibration_commands = (
        commands.get("static_rule_calibration_eval")
        if isinstance(commands.get("static_rule_calibration_eval"), list)
        else []
    )
    if not calibration_commands:
        return manifest_file.parent / "artifacts" / "static_rule_calibration_eval_summary.json"
    value = _command_option_value(str(calibration_commands[0]), "--summary")
    if not value:
        return manifest_file.parent / "artifacts" / "static_rule_calibration_eval_summary.json"
    return _artifact_path(value, manifest_file=manifest_file)


def _local_judge_goldset_chain_summary_path(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    summary_path = local_judge.get("goldset_chain_verification_summary_path") if isinstance(local_judge, Mapping) else None
    if isinstance(summary_path, str) and summary_path:
        return _artifact_path(summary_path, manifest_file=manifest_file)
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    chain_commands = (
        commands.get("local_judge_goldset_chain_verify")
        if isinstance(commands.get("local_judge_goldset_chain_verify"), list)
        else []
    )
    if not chain_commands:
        return manifest_file.parent / "artifacts" / "local_judge_goldset_chain_verification_summary.json"
    value = _command_option_value(str(chain_commands[0]), "--summary")
    if not value:
        return manifest_file.parent / "artifacts" / "local_judge_goldset_chain_verification_summary.json"
    return _artifact_path(value, manifest_file=manifest_file)


def _offline_local_judge_matrix_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    matrix_commands = (
        commands.get("offline_local_judge_matrix")
        if isinstance(commands.get("offline_local_judge_matrix"), list)
        else []
    )
    if not matrix_commands:
        return None
    output_dir = _command_option_value(str(matrix_commands[0]), "--output-dir")
    if not output_dir:
        return None
    return _artifact_path(
        str(Path(output_dir) / "offline_semantic_matrix_summary.json"),
        manifest_file=manifest_file,
    )


def _offline_local_judge_matrix_smoke_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
) -> Path | None:
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    smoke_commands = (
        commands.get("offline_local_judge_matrix_smoke")
        if isinstance(commands.get("offline_local_judge_matrix_smoke"), list)
        else []
    )
    if not smoke_commands:
        return None
    output_dir = _command_option_value(str(smoke_commands[0]), "--output-dir")
    if not output_dir:
        return None
    return _artifact_path(
        str(Path(output_dir) / "offline_local_judge_matrix_smoke_summary.json"),
        manifest_file=manifest_file,
    )


def _command_option_value(command: str, option: str) -> str | None:
    parts = command.split()
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        prefix = option + "="
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _collection_inputs_ready(artifacts: Sequence[Mapping[str, Any]]) -> bool:
    required_markers = ("provider_telemetry_path", "tabulate_csv_path")
    return all(
        bool(artifact.get("nonempty"))
        for artifact in artifacts
        if any(marker in str(artifact.get("name")) for marker in required_markers)
    )


def _ab_reports_ready(artifacts: Sequence[Mapping[str, Any]]) -> bool:
    report_artifacts = [artifact for artifact in artifacts if str(artifact.get("name")).startswith("ab_eval:")]
    return bool(report_artifacts) and all(bool(artifact.get("nonempty")) for artifact in report_artifacts)


def _next_action(
    *,
    structural_ready: bool,
    api_config_present: bool,
    artifact_quality_failed: bool,
    artifact_quality_fake: bool,
    offline_semantic_suite_ready: bool,
    local_judge_config_ready: bool,
    local_judge_healthcheck_ready: bool,
    local_judge_goldset_template_ready: bool,
    local_judge_goldset_annotation_ready: bool,
    local_judge_goldset_import_ready: bool,
    local_judge_goldset_validation_ready: bool,
    local_judge_quality_eval_ready: bool,
    local_judge_quality_eval_ran: bool,
    static_rule_calibration_eval_ran: bool,
    local_judge_goldset_chain_ready: bool,
    offline_local_judge_matrix_ready: bool,
    offline_local_judge_matrix_smoke_ready: bool,
    semantic_guard_fake_smoke_ready: bool,
    collection_ready: bool,
    ab_reports_ready: bool,
) -> str:
    if not structural_ready:
        return "fix_or_regenerate_legacy_suite"
    if not api_config_present:
        return "set_api_config_before_real_run"
    if artifact_quality_failed:
        return "fix_or_rerun_unreadable_artifacts"
    if not offline_semantic_suite_ready:
        return "run_offline_semantic_suite"
    if not local_judge_config_ready:
        return "configure_local_judge"
    if not local_judge_healthcheck_ready:
        return "run_local_judge_healthcheck"
    if not local_judge_goldset_template_ready:
        return "prepare_local_judge_goldset_template"
    if not local_judge_goldset_annotation_ready:
        return "complete_local_judge_goldset_annotation"
    if not local_judge_goldset_import_ready:
        return "import_labeled_local_judge_goldset_csv"
    if not local_judge_goldset_validation_ready:
        return "validate_local_judge_goldset"
    if not local_judge_quality_eval_ready:
        if local_judge_quality_eval_ran:
            if static_rule_calibration_eval_ran:
                return "collect_more_gold_labels_or_recalibrate_validator_rules"
            return "run_static_rule_calibration_eval"
        return "run_local_judge_quality_eval"
    if not local_judge_goldset_chain_ready:
        return "verify_local_judge_goldset_chain"
    if not offline_local_judge_matrix_ready:
        return "run_offline_local_judge_matrix"
    if not offline_local_judge_matrix_smoke_ready:
        return "run_offline_local_judge_matrix_smoke"
    if not semantic_guard_fake_smoke_ready:
        return "run_semantic_guard_fake_smoke"
    if artifact_quality_fake:
        return "run_real_ab_after_fake_smoke"
    if not collection_ready:
        return "start_proxies_and_run_three_autogenbench_variants"
    if not ab_reports_ready:
        return "run_agbench_legacy_collect"
    return "inspect_ab_reports"


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _local_quality_eval_ran(value: Mapping[str, Any]) -> bool:
    status = str(value.get("status") or "")
    return status in {"pass", "warn"} and _num(value.get("evaluated_count")) > 0


def _static_rule_calibration_eval_ran(value: Mapping[str, Any]) -> bool:
    status = str(value.get("status") or "")
    return status in {"pass", "warn"} and _num(value.get("row_count")) > 0


def _artifact_path(value: Any, *, manifest_file: Path) -> Path | None:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value:
        path = Path(value)
    else:
        return None
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    manifest_relative = manifest_file.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return cwd_path


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl_contains_flag(path: Path, key: str) -> bool:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping) and value.get(key) is True:
                return True
    return False


def _write_json(path: str | Path | None, value: Mapping[str, Any]) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _write_text(path: str | Path | None, value: str) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")
    return target


def _redact_command(command: str) -> str:
    redacted = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-REDACTED", command)
    return re.sub(r"Bearer\s+\S+", "Bearer REDACTED", redacted)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _num(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
