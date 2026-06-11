from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agbench_legacy_runbook import build_legacy_agbench_runbook
from .readiness import check_evaluation_readiness


@dataclass(frozen=True)
class LegacyAgBenchPreflightResult:
    manifest_path: str | None
    summary_path: str | None
    report_path: str | None
    summary: dict[str, Any]


def build_legacy_agbench_preflight(
    *,
    manifest_path: str | Path | None = None,
    cwd: str | Path = ".",
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    allow_missing_api: bool = False,
) -> LegacyAgBenchPreflightResult:
    effective_env = os.environ if env is None else env
    readiness = check_evaluation_readiness(
        cwd=cwd,
        env=effective_env,
        require_api_config=not allow_missing_api,
        mode="legacy-proxy",
    )
    manifest_file = Path(manifest_path) if manifest_path is not None else None
    runbook_summary: dict[str, Any] | None = None
    runbook_error: str | None = None
    if manifest_file is not None and manifest_file.exists():
        try:
            runbook_summary = build_legacy_agbench_runbook(
                manifest_path=manifest_file,
                env=effective_env,
                require_api_config=not allow_missing_api,
            ).summary
        except Exception as exc:  # noqa: BLE001
            runbook_error = f"{type(exc).__name__}: {exc}"
    elif manifest_file is not None:
        runbook_error = "manifest file does not exist"

    gates = _gates(readiness=readiness.to_dict(), runbook=runbook_summary, runbook_error=runbook_error)
    blocking_reasons = _blocking_reasons(gates=gates)
    evidence_gaps = _evidence_gaps(gates=gates)
    claim_gate = _claim_gate(gates=gates, evidence_gaps=evidence_gaps)
    next_action = _next_action(gates=gates, runbook=runbook_summary, blocking_reasons=blocking_reasons)
    local_only_next_actions = _local_only_next_actions(gates=gates, runbook=runbook_summary)
    summary = _redact_sensitive(
        {
            "schema_version": "prefix-legacy-real-ab-preflight-v1",
            "prompt_safe_summary": True,
            "cwd": str(Path(cwd)),
            "manifest_path": str(manifest_file) if manifest_file is not None else None,
            "readiness": readiness.to_dict(),
            "runbook": runbook_summary,
            "runbook_error": runbook_error,
            "gates": gates,
            "blocking_reasons": blocking_reasons,
            "evidence_gaps": evidence_gaps,
            "claim_gate": claim_gate,
            "allowed_claims": claim_gate["allowed_claims"],
            "prohibited_claims": claim_gate["prohibited_claims"],
            "next_action": next_action,
            "local_only_next_actions": local_only_next_actions,
            "suggested_commands": _suggested_commands(next_action=next_action, readiness=readiness.to_dict(), runbook=runbook_summary),
            "real_provider_metrics_note": (
                "This preflight reads local readiness, manifest, runbook, and artifact metadata only. "
                "It does not execute AutoGenBench, start proxies, call an API, or prove cached-token, latency, cost, "
                "or task-success gains."
            ),
        }
    )
    summary_target = _write_json(summary_path, summary)
    report_target = _write_text(report_path, render_markdown_preflight(summary))
    return LegacyAgBenchPreflightResult(
        manifest_path=str(manifest_file) if manifest_file is not None else None,
        summary_path=str(summary_target) if summary_target else None,
        report_path=str(report_target) if report_target else None,
        summary=summary,
    )


def render_markdown_preflight(summary: Mapping[str, Any]) -> str:
    gates = summary.get("gates") if isinstance(summary.get("gates"), Mapping) else {}
    lines = [
        "# Legacy AutoGenBench Real A/B Preflight",
        "",
        f"Next action: **{_format_value(summary.get('next_action'))}**",
        f"Manifest: `{_format_value(summary.get('manifest_path'))}`",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for key in (
        "environment_ready",
        "manifest_available",
        "suite_structural_ready",
        "api_config_present",
        "fake_artifacts_detected",
        "artifact_quality_failed",
        "semantic_guard_fake_smoke_ready",
        "offline_semantic_suite_ready",
        "local_judge_config_ready",
        "local_judge_healthcheck_ready",
        "local_judge_goldset_template_ready",
        "local_judge_goldset_progress_ready",
        "local_judge_goldset_labels_template_ready",
        "local_judge_goldset_suggestions_ready",
        "local_judge_goldset_suggestion_review_ready",
        "local_judge_goldset_review_plan_ready",
        "local_judge_goldset_labels_from_csv_ready",
        "local_judge_goldset_labels_validation_ready",
        "local_judge_goldset_annotation_ready",
        "local_judge_goldset_import_ready",
        "local_judge_goldset_validation_ready",
        "local_judge_quality_eval_ready",
        "local_judge_policy_eval_ready",
        "static_rule_calibration_eval_ready",
        "local_judge_goldset_chain_ready",
        "offline_local_judge_matrix_ready",
        "offline_local_judge_matrix_smoke_ready",
        "shadow_trial_wiring_ready",
        "collection_inputs_ready",
        "ab_reports_ready",
        "real_provider_metrics_available",
        "ready_to_start_real_ab",
        "ready_to_claim_real_results",
    ):
        lines.append(f"| {key} | {_format_value(gates.get(key))} |")
    blocking = summary.get("blocking_reasons") if isinstance(summary.get("blocking_reasons"), list) else []
    lines.extend(["", "## Blocking Reasons", ""])
    if blocking:
        lines.extend(f"- `{reason}`" for reason in blocking)
    else:
        lines.append("- none")

    gaps = summary.get("evidence_gaps") if isinstance(summary.get("evidence_gaps"), list) else []
    lines.extend(["", "## Evidence Gaps", ""])
    if gaps:
        lines.extend(
            f"- `{gap.get('name')}`: {_format_value(gap.get('evidence'))}"
            for gap in gaps
            if isinstance(gap, Mapping)
        )
    else:
        lines.append("- none")

    lines.extend(_claim_gate_markdown(summary.get("claim_gate")))
    lines.extend(_local_only_next_actions_markdown(summary.get("local_only_next_actions")))

    commands = summary.get("suggested_commands") if isinstance(summary.get("suggested_commands"), list) else []
    if commands:
        lines.extend(["", "## Suggested Commands", "", "```powershell"])
        lines.extend(str(command) for command in commands)
        lines.extend(["```", ""])
    lines.extend(_offline_evidence_markdown(gates))
    lines.extend(
        [
            "## Limits",
            "",
            "- This report is prompt-safe and does not include API key values.",
            "- Fake-upstream smoke artifacts are wiring evidence only; they are not real provider metrics.",
            "- Real cache, latency, cost, and task-success conclusions require the real three-proxy AutoGenBench A/B run.",
            "",
        ]
    )
    return "\n".join(lines)


def _offline_evidence_markdown(gates: Mapping[str, Any]) -> list[str]:
    rows: list[tuple[str, str, Any]] = []
    suite = (
        gates.get("offline_semantic_suite_status")
        if isinstance(gates.get("offline_semantic_suite_status"), Mapping)
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
        gates.get("local_judge_config_status")
        if isinstance(gates.get("local_judge_config_status"), Mapping)
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
    quality = (
        gates.get("local_judge_quality_eval_status")
        if isinstance(gates.get("local_judge_quality_eval_status"), Mapping)
        else {}
    )
    chain = (
        gates.get("local_judge_goldset_chain_status")
        if isinstance(gates.get("local_judge_goldset_chain_status"), Mapping)
        else {}
    )
    goldset = (
        gates.get("local_judge_goldset_template_status")
        if isinstance(gates.get("local_judge_goldset_template_status"), Mapping)
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
        gates.get("local_judge_goldset_import_status")
        if isinstance(gates.get("local_judge_goldset_import_status"), Mapping)
        else {}
    )
    progress = (
        gates.get("local_judge_goldset_progress_status")
        if isinstance(gates.get("local_judge_goldset_progress_status"), Mapping)
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
                ("local_judge_goldset_progress", "annotation_ready", progress.get("annotation_ready")),
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
        gates.get("local_judge_goldset_labels_template_status")
        if isinstance(gates.get("local_judge_goldset_labels_template_status"), Mapping)
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
        gates.get("local_judge_goldset_suggestions_status")
        if isinstance(gates.get("local_judge_goldset_suggestions_status"), Mapping)
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
        gates.get("local_judge_goldset_suggestion_review_status")
        if isinstance(gates.get("local_judge_goldset_suggestion_review_status"), Mapping)
        else {}
    )
    if suggestion_review:
        rows.extend(
            [
                ("local_judge_goldset_suggestion_review", "status", suggestion_review.get("status")),
                ("local_judge_goldset_suggestion_review", "row_count", suggestion_review.get("row_count")),
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
        gates.get("local_judge_goldset_review_plan_status")
        if isinstance(gates.get("local_judge_goldset_review_plan_status"), Mapping)
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
        gates.get("local_judge_goldset_labels_from_csv_status")
        if isinstance(gates.get("local_judge_goldset_labels_from_csv_status"), Mapping)
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
        gates.get("local_judge_goldset_labels_validation_status")
        if isinstance(gates.get("local_judge_goldset_labels_validation_status"), Mapping)
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
        gates.get("local_judge_goldset_validation_status")
        if isinstance(gates.get("local_judge_goldset_validation_status"), Mapping)
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
        gates.get("local_judge_policy_eval_status")
        if isinstance(gates.get("local_judge_policy_eval_status"), Mapping)
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
        gates.get("static_rule_calibration_eval_status")
        if isinstance(gates.get("static_rule_calibration_eval_status"), Mapping)
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
        gates.get("semantic_guard_fake_smoke_status")
        if isinstance(gates.get("semantic_guard_fake_smoke_status"), Mapping)
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
        status = gates.get(key) if isinstance(gates.get(key), Mapping) else {}
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
    parser = argparse.ArgumentParser(
        description="Build a prompt-safe preflight report for the legacy AutoGenBench three-proxy real A/B path."
    )
    parser.add_argument("--manifest", help="Generated legacy_suite_manifest.json.")
    parser.add_argument("--cwd", default=".", help="Project root to inspect for readiness.")
    parser.add_argument("--summary")
    parser.add_argument("--report-md")
    parser.add_argument("--allow-missing-api", action="store_true")
    parser.add_argument(
        "--require-real-ready",
        action="store_true",
        help="Return non-zero unless environment, API marker, and generated suite are ready to start real A/B.",
    )
    parser.add_argument(
        "--require-real-results",
        action="store_true",
        help="Return non-zero unless real provider and task-result A/B reports are already available.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_legacy_agbench_preflight(
        manifest_path=args.manifest,
        cwd=args.cwd,
        summary_path=args.summary,
        report_path=args.report_md,
        allow_missing_api=args.allow_missing_api,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    gates = result.summary.get("gates") if isinstance(result.summary.get("gates"), Mapping) else {}
    if args.require_real_results:
        return 0 if gates.get("ready_to_claim_real_results") is True else 1
    if args.require_real_ready:
        return 0 if gates.get("ready_to_start_real_ab") is True else 1
    if not gates.get("environment_ready"):
        return 1
    if args.manifest and not gates.get("manifest_available"):
        return 1
    return 0


def _gates(*, readiness: Mapping[str, Any], runbook: Mapping[str, Any] | None, runbook_error: str | None) -> dict[str, Any]:
    checks = readiness.get("checks") if isinstance(readiness.get("checks"), Sequence) else []
    failed_readiness = [
        str(check.get("name"))
        for check in checks
        if isinstance(check, Mapping) and check.get("ok") is not True
    ]
    api_config_present = _api_config_present(readiness=readiness, runbook=runbook)
    manifest_available = runbook is not None
    suite_structural_ready = bool(runbook and runbook.get("structural_ready"))
    artifact_quality_fail_count = _num(runbook.get("artifact_quality_fail_count") if runbook else None)
    artifact_quality_warn_count = _num(runbook.get("artifact_quality_warn_count") if runbook else None)
    fake_artifacts_detected = bool(
        runbook
        and (
            runbook.get("provider_or_ab_fake_artifacts_detected") is True
            or runbook.get("next_action") == "run_real_ab_after_fake_smoke"
        )
    )
    artifact_quality_failed = artifact_quality_fail_count > 0
    real_provider_metrics_available = bool(runbook and runbook.get("real_provider_metrics_available"))
    ab_reports_ready = bool(runbook and runbook.get("ab_reports_ready"))
    collection_inputs_ready = bool(runbook and runbook.get("collection_inputs_ready"))
    semantic_guard_fake_smoke_ready = bool(runbook and runbook.get("semantic_guard_fake_smoke_ready"))
    offline_semantic_suite_ready = bool(runbook and runbook.get("offline_semantic_suite_ready"))
    local_judge_config_ready = bool(runbook and runbook.get("local_judge_config_ready"))
    local_judge_healthcheck_ready = bool(runbook and runbook.get("local_judge_healthcheck_ready"))
    local_judge_goldset_template_ready = bool(runbook and runbook.get("local_judge_goldset_template_ready"))
    local_judge_goldset_progress_ready = bool(runbook and runbook.get("local_judge_goldset_progress_ready"))
    local_judge_goldset_labels_template_ready = bool(
        runbook and runbook.get("local_judge_goldset_labels_template_ready")
    )
    local_judge_goldset_suggestions_ready = bool(
        runbook and runbook.get("local_judge_goldset_suggestions_ready")
    )
    local_judge_goldset_suggestion_review_ready = bool(
        runbook and runbook.get("local_judge_goldset_suggestion_review_ready")
    )
    local_judge_goldset_review_plan_ready = bool(
        runbook and runbook.get("local_judge_goldset_review_plan_ready")
    )
    local_judge_goldset_labels_from_csv_ready = bool(
        runbook and runbook.get("local_judge_goldset_labels_from_csv_ready")
    )
    local_judge_goldset_labels_validation_ready = bool(
        runbook and runbook.get("local_judge_goldset_labels_validation_ready")
    )
    local_judge_goldset_annotation_ready = bool(
        runbook and runbook.get("local_judge_goldset_annotation_ready")
    )
    local_judge_goldset_import_ready = bool(runbook and runbook.get("local_judge_goldset_import_ready"))
    local_judge_goldset_validation_ready = bool(runbook and runbook.get("local_judge_goldset_validation_ready"))
    local_judge_quality_eval_ready = bool(runbook and runbook.get("local_judge_quality_eval_ready"))
    local_judge_policy_eval_ready = bool(runbook and runbook.get("local_judge_policy_eval_ready"))
    static_rule_calibration_eval_ready = bool(runbook and runbook.get("static_rule_calibration_eval_ready"))
    local_judge_goldset_chain_ready = bool(runbook and runbook.get("local_judge_goldset_chain_ready"))
    offline_local_judge_matrix_ready = bool(runbook and runbook.get("offline_local_judge_matrix_ready"))
    offline_local_judge_matrix_smoke_ready = bool(
        runbook and runbook.get("offline_local_judge_matrix_smoke_ready")
    )
    shadow_trial_wiring_ready = bool(runbook and runbook.get("shadow_trial_wiring_ready"))
    environment_ready = bool(readiness.get("ready"))
    ready_to_start_real_ab = (
        environment_ready
        and manifest_available
        and suite_structural_ready
        and api_config_present
        and not artifact_quality_failed
    )
    ready_to_claim_real_results = (
        ready_to_start_real_ab
        and offline_semantic_suite_ready
        and local_judge_config_ready
        and local_judge_healthcheck_ready
        and local_judge_goldset_template_ready
        and local_judge_goldset_annotation_ready
        and local_judge_goldset_import_ready
        and local_judge_goldset_validation_ready
        and local_judge_quality_eval_ready
        and local_judge_goldset_chain_ready
        and offline_local_judge_matrix_ready
        and offline_local_judge_matrix_smoke_ready
        and semantic_guard_fake_smoke_ready
        and ab_reports_ready
        and real_provider_metrics_available
        and not fake_artifacts_detected
    )
    return {
        "environment_ready": environment_ready,
        "readiness_failed_checks": failed_readiness,
        "manifest_available": manifest_available,
        "runbook_error": runbook_error,
        "suite_structural_ready": suite_structural_ready,
        "api_config_present": api_config_present,
        "collection_inputs_ready": collection_inputs_ready,
        "ab_reports_ready": ab_reports_ready,
        "artifact_quality_fail_count": artifact_quality_fail_count,
        "artifact_quality_warn_count": artifact_quality_warn_count,
        "artifact_quality_failed": artifact_quality_failed,
        "fake_artifacts_detected": fake_artifacts_detected,
        "semantic_guard_fake_smoke_ready": semantic_guard_fake_smoke_ready,
        "semantic_guard_fake_smoke_status": runbook.get("semantic_guard_fake_smoke_status") if runbook else None,
        "offline_semantic_suite_ready": offline_semantic_suite_ready,
        "offline_semantic_suite_status": runbook.get("offline_semantic_suite_status") if runbook else None,
        "local_judge_config_ready": local_judge_config_ready,
        "local_judge_config_status": runbook.get("local_judge_config_status") if runbook else None,
        "local_judge_healthcheck_ready": local_judge_healthcheck_ready,
        "local_judge_healthcheck_status": runbook.get("local_judge_healthcheck_status") if runbook else None,
        "local_judge_goldset_template_ready": local_judge_goldset_template_ready,
        "local_judge_goldset_template_status": runbook.get("local_judge_goldset_template_status")
        if runbook
        else None,
        "local_judge_goldset_progress_ready": local_judge_goldset_progress_ready,
        "local_judge_goldset_labels_template_ready": local_judge_goldset_labels_template_ready,
        "local_judge_goldset_labels_template_status": runbook.get("local_judge_goldset_labels_template_status")
        if runbook
        else None,
        "local_judge_goldset_suggestions_ready": local_judge_goldset_suggestions_ready,
        "local_judge_goldset_suggestions_status": runbook.get("local_judge_goldset_suggestions_status")
        if runbook
        else None,
        "local_judge_goldset_suggestion_review_ready": local_judge_goldset_suggestion_review_ready,
        "local_judge_goldset_suggestion_review_status": runbook.get("local_judge_goldset_suggestion_review_status")
        if runbook
        else None,
        "local_judge_goldset_review_plan_ready": local_judge_goldset_review_plan_ready,
        "local_judge_goldset_review_plan_status": runbook.get("local_judge_goldset_review_plan_status")
        if runbook
        else None,
        "local_judge_goldset_labels_from_csv_ready": local_judge_goldset_labels_from_csv_ready,
        "local_judge_goldset_labels_from_csv_status": runbook.get("local_judge_goldset_labels_from_csv_status")
        if runbook
        else None,
        "local_judge_goldset_labels_validation_ready": local_judge_goldset_labels_validation_ready,
        "local_judge_goldset_labels_validation_status": runbook.get("local_judge_goldset_labels_validation_status")
        if runbook
        else None,
        "local_judge_goldset_annotation_ready": local_judge_goldset_annotation_ready,
        "local_judge_goldset_progress_status": runbook.get("local_judge_goldset_progress_status")
        if runbook
        else None,
        "local_judge_goldset_import_ready": local_judge_goldset_import_ready,
        "local_judge_goldset_import_status": runbook.get("local_judge_goldset_import_status")
        if runbook
        else None,
        "local_judge_goldset_validation_ready": local_judge_goldset_validation_ready,
        "local_judge_goldset_validation_status": runbook.get("local_judge_goldset_validation_status")
        if runbook
        else None,
        "local_judge_quality_eval_ready": local_judge_quality_eval_ready,
        "local_judge_quality_eval_status": runbook.get("local_judge_quality_eval_status") if runbook else None,
        "local_judge_policy_eval_ready": local_judge_policy_eval_ready,
        "local_judge_policy_eval_status": runbook.get("local_judge_policy_eval_status") if runbook else None,
        "static_rule_calibration_eval_ready": static_rule_calibration_eval_ready,
        "static_rule_calibration_eval_status": runbook.get("static_rule_calibration_eval_status")
        if runbook
        else None,
        "local_judge_goldset_chain_ready": local_judge_goldset_chain_ready,
        "local_judge_goldset_chain_status": runbook.get("local_judge_goldset_chain_status") if runbook else None,
        "offline_local_judge_matrix_ready": offline_local_judge_matrix_ready,
        "offline_local_judge_matrix_status": runbook.get("offline_local_judge_matrix_status") if runbook else None,
        "offline_local_judge_matrix_smoke_ready": offline_local_judge_matrix_smoke_ready,
        "offline_local_judge_matrix_smoke_status": runbook.get("offline_local_judge_matrix_smoke_status")
        if runbook
        else None,
        "shadow_trial_wiring_ready": shadow_trial_wiring_ready,
        "shadow_trial_wiring_status": runbook.get("shadow_trial_wiring_status") if runbook else None,
        "real_provider_metrics_available": real_provider_metrics_available,
        "ready_to_start_real_ab": ready_to_start_real_ab,
        "ready_to_claim_real_results": ready_to_claim_real_results,
    }


def _api_config_present(*, readiness: Mapping[str, Any], runbook: Mapping[str, Any] | None) -> bool:
    if runbook and runbook.get("api_config_present") is True:
        return True
    checks = readiness.get("checks") if isinstance(readiness.get("checks"), Sequence) else []
    for check in checks:
        if isinstance(check, Mapping) and check.get("name") == "api_config":
            detail = str(check.get("detail") or "")
            return bool(check.get("ok") is True and detail.startswith("present:"))
    return False


def _blocking_reasons(*, gates: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not gates.get("environment_ready"):
        failed = gates.get("readiness_failed_checks") if isinstance(gates.get("readiness_failed_checks"), list) else []
        if failed == ["api_config"] or "api_config" in failed:
            reasons.append("missing_api_config")
        else:
            reasons.append("legacy_readiness_failed")
    if not gates.get("manifest_available"):
        reasons.append("legacy_suite_manifest_missing")
    elif not gates.get("suite_structural_ready"):
        reasons.append("legacy_suite_structural_not_ready")
    if not gates.get("api_config_present"):
        reasons.append("missing_api_config")
    if gates.get("artifact_quality_failed"):
        reasons.append("unreadable_or_invalid_artifacts")
    if gates.get("fake_artifacts_detected"):
        reasons.append("fake_smoke_artifacts_do_not_prove_real_metrics")
    if gates.get("ready_to_start_real_ab") and not gates.get("offline_semantic_suite_ready"):
        reasons.append("offline_semantic_suite_not_run")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_config_ready"):
        reasons.append("local_judge_not_configured")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_healthcheck_ready"):
        reasons.append("local_judge_healthcheck_not_run")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_template_ready"):
        reasons.append("local_judge_goldset_template_not_ready")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_annotation_ready"):
        reasons.append("local_judge_goldset_annotation_not_ready")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_import_ready"):
        reasons.append("local_judge_goldset_import_not_ready")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_validation_ready"):
        reasons.append("local_judge_goldset_validation_not_ready")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_quality_eval_ready"):
        reasons.append("local_judge_quality_eval_not_ready")
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_chain_ready"):
        reasons.append("local_judge_goldset_chain_not_verified")
    if gates.get("ready_to_start_real_ab") and not gates.get("offline_local_judge_matrix_ready"):
        reasons.append("offline_local_judge_matrix_not_run")
    if gates.get("ready_to_start_real_ab") and not gates.get("offline_local_judge_matrix_smoke_ready"):
        reasons.append("offline_local_judge_matrix_smoke_not_run")
    if gates.get("ready_to_start_real_ab") and not gates.get("semantic_guard_fake_smoke_ready"):
        reasons.append("semantic_guard_fake_smoke_not_run")
    if gates.get("ready_to_start_real_ab") and not gates.get("collection_inputs_ready"):
        reasons.append("real_autogenbench_variants_not_run")
    elif gates.get("ready_to_start_real_ab") and not gates.get("ab_reports_ready"):
        reasons.append("real_ab_reports_not_collected")
    seen: set[str] = set()
    deduped: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped


def _evidence_gaps(*, gates: Mapping[str, Any]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    if not gates.get("api_config_present"):
        gaps.append(
            {
                "name": "provider_api_config",
                "evidence": "OPENAI_API_KEY, OAI_CONFIG_LIST, or project DeepSeek config is required before real provider A/B.",
            }
        )
    if not gates.get("offline_semantic_suite_ready"):
        gaps.append(
            {
                "name": "offline_semantic_suite",
                "evidence": "AutoGen source-prompt coverage must pass before spending provider budget.",
            }
        )
    if not gates.get("local_judge_config_ready"):
        gaps.append(
            {
                "name": "local_judge_config",
                "evidence": "A real local judge base URL and model are required before semantic review judging.",
            }
        )
    if not gates.get("local_judge_healthcheck_ready"):
        gaps.append(
            {
                "name": "local_judge_healthcheck",
                "evidence": "The configured local judge must pass a prompt-safe endpoint/schema healthcheck.",
            }
        )
    if not gates.get("local_judge_goldset_template_ready"):
        gaps.append(
            {
                "name": "local_judge_goldset_template",
                "evidence": (
                    "A sampled gold-set labeling template should be built from real semantic candidates before "
                    "manual labels and local-judge quality evaluation."
                ),
            }
        )
    if not gates.get("local_judge_goldset_annotation_ready"):
        gaps.append(
            {
                "name": "local_judge_goldset_annotation",
                "evidence": (
                    "The local annotation CSV should have complete, valid expected_label values before "
                    "it is imported into a labeled gold JSONL."
                ),
            }
        )
    if not gates.get("local_judge_goldset_import_ready"):
        gaps.append(
            {
                "name": "local_judge_goldset_import",
                "evidence": (
                    "The local annotation CSV should be filled with expected_label values and imported into "
                    "local_judge_goldset_labeled.local.jsonl before gold-set validation."
                ),
            }
        )
    if not gates.get("local_judge_goldset_validation_ready"):
        gaps.append(
            {
                "name": "local_judge_goldset_validation",
                "evidence": (
                    "The manually labeled local gold JSONL should pass text, expected-label, sample-count, "
                    "and label-balance validation before local-judge quality evaluation."
                ),
            }
        )
    if not gates.get("local_judge_quality_eval_ready"):
        quality_status = (
            gates.get("local_judge_quality_eval_status")
            if isinstance(gates.get("local_judge_quality_eval_status"), Mapping)
            else {}
        )
        quality_ran = _local_quality_eval_ran(quality_status)
        gaps.append(
            {
                "name": "local_judge_quality_eval",
                "evidence": (
                    "The gold-label local judge quality eval ran but is below semantic-quality thresholds; "
                    "inspect failure modes and mismatch diagnostics before using it for automatic Validator decisions."
                    if quality_ran
                    else (
                        "A gold-label local judge quality eval is required before claiming local-model "
                        "semantic judgment quality."
                    )
                ),
            }
        )
    if not gates.get("local_judge_goldset_chain_ready"):
        chain_status = (
            gates.get("local_judge_goldset_chain_status")
            if isinstance(gates.get("local_judge_goldset_chain_status"), Mapping)
            else {}
        )
        chain_next_action = str(chain_status.get("next_action") or "")
        gaps.append(
            {
                "name": "local_judge_goldset_chain",
                "evidence": (
                    "The read-only gold-set chain verifier has run and is blocked by below-threshold quality eval."
                    if chain_next_action == "improve_or_recalibrate_local_judge_quality_eval"
                    else (
                        "The read-only gold-set chain verifier should pass before claiming local semantic quality; "
                        "it checks labels-from-csv, validate-labels, apply-labels, import, gold-set validation, "
                        "and quality eval ordering."
                    )
                ),
            }
        )
    if not gates.get("offline_local_judge_matrix_ready"):
        gaps.append(
            {
                "name": "offline_local_judge_matrix",
                "evidence": "A real local-judge semantic matrix is required for local-model review-resolution evidence.",
            }
        )
    if not gates.get("offline_local_judge_matrix_smoke_ready"):
        gaps.append(
            {
                "name": "offline_local_judge_matrix_smoke",
                "evidence": "The fake local-judge matrix smoke should validate wiring and budget coverage.",
            }
        )
    if not gates.get("semantic_guard_fake_smoke_ready"):
        gaps.append(
            {
                "name": "semantic_guard_fake_smoke",
                "evidence": "The semantic guard fail-closed proxy path should be smoke-tested before real A/B.",
            }
        )
    if gates.get("fake_artifacts_detected"):
        gaps.append(
            {
                "name": "real_provider_artifacts",
                "evidence": "Fake smoke provider/A-B artifacts must be replaced by real provider AutoGenBench runs.",
            }
        )
    if not gates.get("real_provider_metrics_available"):
        gaps.append(
            {
                "name": "real_provider_metrics",
                "evidence": "Cached tokens, latency, cost, and task success must come from real provider A/B artifacts.",
            }
        )
    if gates.get("artifact_quality_failed"):
        gaps.append(
            {
                "name": "artifact_quality",
                "evidence": "Unreadable or invalid artifacts must be fixed before any real result claim.",
            }
    )
    return gaps


def _claim_gate(*, gates: Mapping[str, Any], evidence_gaps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    allowed: list[dict[str, Any]] = []
    prohibited: list[dict[str, Any]] = []
    gap_names = [
        str(gap.get("name"))
        for gap in evidence_gaps
        if isinstance(gap, Mapping) and gap.get("name") is not None
    ]
    semantic_chain_ready = _semantic_chain_ready(gates)
    real_provider_claims_allowed = bool(
        gates.get("ready_to_claim_real_results") is True
        and gates.get("real_provider_metrics_available") is True
        and gates.get("fake_artifacts_detected") is False
        and semantic_chain_ready
    )
    semantic_quality_claims_allowed = bool(
        gates.get("local_judge_quality_eval_ready") is True
        and gates.get("local_judge_goldset_chain_ready") is True
    )

    if gates.get("suite_structural_ready"):
        allowed.append(
            _claim(
                "legacy_suite_structure_ready",
                "The generated legacy AutoGenBench suite structure is available for the three-variant proxy path.",
                "manifest/runbook structural checks passed",
            )
        )
    if gates.get("offline_semantic_suite_ready"):
        allowed.append(
            _claim(
                "offline_source_prompt_coverage",
                "Offline source-prompt coverage diagnostics are available for rule-only and NL segmentation modes.",
                "offline_semantic_suite status is pass",
            )
        )
        suite = (
            gates.get("offline_semantic_suite_status")
            if isinstance(gates.get("offline_semantic_suite_status"), Mapping)
            else {}
        )
        if suite.get("rule_gap_resolved") is True:
            allowed.append(
                _claim(
                    "offline_rule_gap_resolved",
                    "NL segmentation resolves the observed rule-only reuse gap in offline estimated-gain diagnostics.",
                    "offline semantic suite reports rule_gap_resolved=true",
                )
            )
        if _review_candidate_promotion_policy_pass(suite):
            allowed.append(
                _claim(
                    "review_candidates_excluded_from_automatic_promotion",
                    "Review candidates are excluded from automatic promotion and remain upper-bound/manual-or-local-judge opportunities.",
                    "offline experiment gate review_candidate_promotion_policy passed",
                )
            )
    if gates.get("semantic_guard_fake_smoke_ready"):
        allowed.append(
            _claim(
                "semantic_guard_fake_smoke_wiring",
                "The semantic guard proxy fail-closed path has wiring-only smoke evidence.",
                "semantic_guard_fake_smoke summary is ready and uses fake local judge/upstream",
            )
        )
    if gates.get("fake_artifacts_detected") and gates.get("collection_inputs_ready") and gates.get("ab_reports_ready"):
        allowed.append(
            _claim(
                "fake_three_proxy_smoke_wiring",
                "The three-proxy collector and A/B report path has fake-upstream wiring evidence only.",
                "fake provider/A-B artifacts are present and explicitly marked fake",
            )
        )
    if semantic_chain_ready:
        allowed.append(
            _claim(
                "local_judge_review_resolution_diagnostics",
                "The configured local-judge review-resolution diagnostics chain is available.",
                "local judge config, healthcheck, real matrix, smoke matrix, offline suite, and guard smoke are ready",
            )
        )
    static_calibration = (
        gates.get("static_rule_calibration_eval_status")
        if isinstance(gates.get("static_rule_calibration_eval_status"), Mapping)
        else {}
    )
    if _static_rule_calibration_eval_ran(static_calibration):
        allowed.append(
            _claim(
                "static_rule_metadata_calibration_diagnostics",
                "Prompt-safe metadata-only static rule calibration diagnostics are available on the supplied gold set.",
                "static_rule_calibration_eval ran with leave-one-out cross-validation",
            )
        )
    if gates.get("shadow_trial_wiring_ready"):
        allowed.append(
            _claim(
                "shadow_trial_telemetry_wiring",
                "Shadow-only Validator rule matching telemetry is wired into plugin proxies.",
                "plugin provider telemetry contains shadow trial rule-match observations with validator_behavior_change_allowed=false",
            )
        )
    if real_provider_claims_allowed:
        allowed.append(
            _claim(
                "real_provider_ab_metrics",
                "Real provider A/B artifacts are available for cached-token, latency, cost, and task-success comparisons.",
                "ready_to_claim_real_results=true with real provider metrics and no fake provider/A-B artifacts",
            )
        )
    if not real_provider_claims_allowed:
        prohibited.extend(
            [
                _prohibited_claim(
                    "real_cache_hit_or_cached_token_improvement",
                    "Do not claim real cache-hit or cached-token improvement.",
                    gap_names,
                ),
                _prohibited_claim(
                    "real_latency_or_cost_savings",
                    "Do not claim real latency reduction, cost reduction, or cost savings.",
                    gap_names,
                ),
                _prohibited_claim(
                    "real_task_success_improvement",
                    "Do not claim real AutoGenBench task-success improvement or regression.",
                    gap_names,
                ),
                _prohibited_claim(
                    "final_real_performance_conclusion",
                    "Do not make a final provider-performance conclusion.",
                    gap_names,
                ),
            ]
        )
    if semantic_quality_claims_allowed:
        allowed.append(
            _claim(
                "local_model_semantic_quality_on_gold_set",
                "The local semantic judge met the configured thresholds on the supplied gold label set.",
                "local_judge_quality_eval_ready=true and local_judge_goldset_chain_ready=true",
            )
        )
    if not semantic_quality_claims_allowed:
        semantic_quality_gaps = gap_names or ["dedicated_semantic_quality_benchmark"]
        prohibited.append(
            _prohibited_claim(
                "real_local_model_semantic_quality",
                "Do not claim local-model semantic judgment quality.",
                semantic_quality_gaps,
            )
        )
    prohibited.append(
        _prohibited_claim(
            "broad_semantic_rule_generalization",
            "Do not claim broad cross-framework semantic-rule generalization from the current evidence alone.",
            ["broader_cross_framework_gold_eval"],
        )
    )
    return {
        "schema_version": "prefix-legacy-claim-gate-v1",
        "prompt_safe_summary": True,
        "real_provider_claims_allowed": real_provider_claims_allowed,
        "semantic_quality_claims_allowed": semantic_quality_claims_allowed,
        "semantic_chain_ready": semantic_chain_ready,
        "current_missing_evidence": gap_names,
        "allowed_claim_ids": [claim["id"] for claim in allowed],
        "prohibited_claim_ids": [claim["id"] for claim in prohibited],
        "allowed_claims": allowed,
        "prohibited_claims": prohibited,
    }


def _claim(identifier: str, claim: str, evidence: str) -> dict[str, str]:
    return {"id": identifier, "claim": claim, "evidence": evidence}


def _prohibited_claim(identifier: str, claim: str, gap_names: Sequence[str]) -> dict[str, Any]:
    return {
        "id": identifier,
        "claim": claim,
        "missing_evidence": list(gap_names) or ["real_provider_and_semantic_evidence_not_ready"],
    }


def _semantic_chain_ready(gates: Mapping[str, Any]) -> bool:
    return bool(
        gates.get("offline_semantic_suite_ready") is True
        and gates.get("local_judge_config_ready") is True
        and gates.get("local_judge_healthcheck_ready") is True
        and gates.get("local_judge_goldset_template_ready") is True
        and gates.get("local_judge_goldset_annotation_ready") is True
        and gates.get("local_judge_goldset_import_ready") is True
        and gates.get("local_judge_goldset_validation_ready") is True
        and gates.get("local_judge_quality_eval_ready") is True
        and gates.get("local_judge_goldset_chain_ready") is True
        and gates.get("offline_local_judge_matrix_ready") is True
        and gates.get("offline_local_judge_matrix_smoke_ready") is True
        and gates.get("semantic_guard_fake_smoke_ready") is True
    )


def _local_quality_eval_ran(value: Mapping[str, Any]) -> bool:
    status = str(value.get("status") or "")
    return status in {"pass", "warn"} and _num(value.get("evaluated_count")) > 0


def _static_rule_calibration_eval_ran(value: Mapping[str, Any]) -> bool:
    status = str(value.get("status") or "")
    return status in {"pass", "warn"} and _num(value.get("row_count")) > 0


def _review_candidate_promotion_policy_pass(suite: Mapping[str, Any]) -> bool:
    items = suite.get("experiment_gate_items") if isinstance(suite.get("experiment_gate_items"), list) else []
    for item in items:
        if (
            isinstance(item, Mapping)
            and item.get("name") == "review_candidate_promotion_policy"
            and item.get("status") == "pass"
        ):
            return True
    policy = (
        suite.get("candidate_promotion_policy")
        if isinstance(suite.get("candidate_promotion_policy"), Mapping)
        else {}
    )
    return (
        policy.get("review_candidate_policy") == "excluded_from_automatic_promotion"
        and policy.get("automatic_promotion_includes_review") is False
    )


def _claim_gate_markdown(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    lines = [
        "",
        "## Claim Gate",
        "",
        f"- real_provider_claims_allowed: `{_format_value(value.get('real_provider_claims_allowed'))}`",
        f"- semantic_quality_claims_allowed: `{_format_value(value.get('semantic_quality_claims_allowed'))}`",
        f"- semantic_chain_ready: `{_format_value(value.get('semantic_chain_ready'))}`",
        "",
        "### Allowed Claims",
        "",
    ]
    allowed = value.get("allowed_claims") if isinstance(value.get("allowed_claims"), list) else []
    if allowed:
        lines.extend(_claim_lines(allowed, evidence_key="evidence"))
    else:
        lines.append("- none")
    lines.extend(["", "### Prohibited Claims", ""])
    prohibited = value.get("prohibited_claims") if isinstance(value.get("prohibited_claims"), list) else []
    if prohibited:
        lines.extend(_claim_lines(prohibited, evidence_key="missing_evidence"))
    else:
        lines.append("- none")
    return lines


def _claim_lines(claims: Sequence[Any], *, evidence_key: str) -> list[str]:
    lines: list[str] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        evidence = claim.get(evidence_key)
        lines.append(
            f"- `{_format_value(claim.get('id'))}`: {_format_value(claim.get('claim'))} "
            f"({_format_value(evidence_key)}: {_format_value(evidence)})"
        )
    return lines


def _local_only_next_actions_markdown(value: Any) -> list[str]:
    actions = value if isinstance(value, list) else []
    lines = ["", "## Local-Only Next Actions", ""]
    if not actions:
        lines.append("- none")
        return lines
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        lines.extend(
            [
                f"### {_format_value(action.get('id'))}",
                "",
                _format_value(action.get("reason")),
                "",
            ]
        )
        commands = action.get("commands") if isinstance(action.get("commands"), list) else []
        if commands:
            lines.extend(["```powershell"])
            lines.extend(str(command) for command in commands)
            lines.extend(["```", ""])
    return lines


def _local_only_next_actions(*, gates: Mapping[str, Any], runbook: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not runbook:
        return []
    actions: list[dict[str, Any]] = []
    if gates.get("artifact_quality_failed"):
        return [
            {
                "id": "fix_or_rerun_unreadable_artifacts",
                "reason": "Some local artifacts failed parser/quality checks; repair them before continuing local or provider evaluation.",
                "commands": [],
            }
        ]
    if not gates.get("offline_semantic_suite_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="run_offline_semantic_suite",
            step_name="offline_semantic_suite",
            reason=(
                "Run the source-prompt semantic coverage suite locally before spending provider budget. "
                "This does not require provider API credentials."
            ),
        )
        return actions
    if not gates.get("local_judge_config_ready"):
        actions.append(
            {
                "id": "configure_local_judge",
                "reason": "Regenerate the suite or edit local judge config with a real local OpenAI-compatible base URL and model.",
                "commands": _suggested_commands(
                    next_action="configure_local_judge",
                    readiness={},
                    runbook=runbook,
                ),
            }
        )
    elif not gates.get("local_judge_healthcheck_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="run_local_judge_healthcheck",
            step_name="local_judge_healthcheck",
            reason="Healthcheck the configured local judge endpoint before sending candidate-text batches.",
        )

    if not gates.get("local_judge_goldset_template_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="prepare_local_judge_goldset_template",
            step_name="local_judge_goldset_template",
            reason="Build the local annotation CSV/template from offline semantic candidates.",
        )
        return actions
    progress = gates.get("local_judge_goldset_progress_status") if isinstance(gates.get("local_judge_goldset_progress_status"), Mapping) else {}
    labels_template = (
        gates.get("local_judge_goldset_labels_template_status")
        if isinstance(gates.get("local_judge_goldset_labels_template_status"), Mapping)
        else {}
    )
    suggestions = (
        gates.get("local_judge_goldset_suggestions_status")
        if isinstance(gates.get("local_judge_goldset_suggestions_status"), Mapping)
        else {}
    )
    suggestion_review = (
        gates.get("local_judge_goldset_suggestion_review_status")
        if isinstance(gates.get("local_judge_goldset_suggestion_review_status"), Mapping)
        else {}
    )
    labels_from_csv = (
        gates.get("local_judge_goldset_labels_from_csv_status")
        if isinstance(gates.get("local_judge_goldset_labels_from_csv_status"), Mapping)
        else {}
    )
    if not gates.get("local_judge_goldset_import_ready"):
        if not gates.get("local_judge_goldset_annotation_ready"):
            _append_local_action(
                actions,
                runbook=runbook,
                action_id="complete_local_judge_goldset_annotation",
                step_name="local_judge_goldset_template",
                reason=(
                    "Fill expected_label in the local annotation CSV, then rerun the progress command. "
                    f"Pending labels: {_format_value(progress.get('expected_label_pending_count'))}; "
                    f"invalid labels: {_format_value(progress.get('invalid_expected_label_count'))}; "
                    f"labels template ready: {_format_value(labels_template.get('ready_for_apply_labels'))}; "
                    f"local suggestions ready: {_format_value(suggestions.get('suggestions_ready_for_manual_review'))}. "
                    f"suggestion review CSV ready: {_format_value(suggestion_review.get('ready_for_manual_review'))}. "
                    f"labels-from-csv ready: {_format_value(labels_from_csv.get('ready_for_apply_labels'))}. "
                    "Suggestions are manual-review input only, not gold labels."
                ),
                command_filter=(
                    "local_judge_goldset_csv progress|"
                    "local_judge_goldset_csv labels-template|"
                    "local_judge_goldset_csv suggest-labels|"
                    "local_judge_goldset_csv merge-suggestions|"
                    "local_judge_goldset_csv review-plan|"
                    "local_judge_goldset_csv labels-from-csv|"
                    "local_judge_goldset_csv validate-labels|"
                    "local_judge_goldset_csv apply-labels"
                ),
            )
            return actions
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="import_labeled_local_judge_goldset_csv",
            step_name="local_judge_goldset_template",
            reason="Import the completed local annotation CSV into labeled local gold JSONL.",
            command_filter="local_judge_goldset_csv import",
        )
        return actions
    if not gates.get("local_judge_goldset_validation_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="validate_local_judge_goldset",
            step_name="local_judge_goldset_validate",
            reason="Validate the labeled local gold JSONL before local judge quality evaluation.",
        )
        return actions
    if gates.get("local_judge_healthcheck_ready") and not gates.get("local_judge_quality_eval_ready"):
        quality_status = (
            gates.get("local_judge_quality_eval_status")
            if isinstance(gates.get("local_judge_quality_eval_status"), Mapping)
            else {}
        )
        if _local_quality_eval_ran(quality_status):
            static_calibration = (
                gates.get("static_rule_calibration_eval_status")
                if isinstance(gates.get("static_rule_calibration_eval_status"), Mapping)
                else {}
            )
            if not _static_rule_calibration_eval_ran(static_calibration):
                _append_local_action(
                    actions,
                    runbook=runbook,
                    action_id="run_static_rule_calibration_eval",
                    step_name="static_rule_calibration_eval",
                    reason=(
                        "Local judge quality evaluation has run but is below thresholds. "
                        "Run the prompt-safe metadata-only calibration eval to test whether semantic_hint/risk_tag "
                        "patterns are reliable enough to inform Validator rule changes."
                    ),
                )
                return actions
            actions.append(
                {
                    "id": "collect_more_gold_labels_or_recalibrate_validator_rules",
                    "reason": (
                        "Local judge quality evaluation is below thresholds, and static metadata calibration has run. "
                        "Use mismatch, policy, and calibration diagnostics to add more gold labels, tighten Validator "
                        "rules, recalibrate the judge prompt, or try a stronger local model."
                    ),
                    "commands": _commands_for_steps(
                        runbook,
                        step_names=(
                            "local_judge_quality_eval",
                            "local_judge_policy_eval",
                            "static_rule_calibration_eval",
                        ),
                    ),
                }
            )
            return actions
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="run_local_judge_quality_eval",
            step_name="local_judge_quality_eval",
            reason="Evaluate the configured local judge against the validated gold set.",
        )
        return actions
    if gates.get("local_judge_quality_eval_ready") and not gates.get("local_judge_goldset_chain_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="verify_local_judge_goldset_chain",
            step_name="local_judge_goldset_chain_verify",
            reason="Verify the prompt-safe human-label chain before local semantic-quality claims.",
        )
        return actions
    if gates.get("local_judge_healthcheck_ready") and not gates.get("offline_local_judge_matrix_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="run_offline_local_judge_matrix",
            step_name="offline_local_judge_matrix",
            reason="Run the real local-judge semantic matrix over framework source trees.",
        )
        return actions
    if not gates.get("offline_local_judge_matrix_smoke_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="run_offline_local_judge_matrix_smoke",
            step_name="offline_local_judge_matrix_smoke",
            reason="Run the fake local-judge matrix smoke to check wiring and budget coverage.",
        )
        return actions
    if not gates.get("semantic_guard_fake_smoke_ready"):
        _append_local_action(
            actions,
            runbook=runbook,
            action_id="run_semantic_guard_fake_smoke",
            step_name="semantic_guard_fake_smoke",
            reason="Run the fake semantic-guard smoke to verify fail-closed proxy wiring.",
        )
        return actions
    return actions


def _append_local_action(
    actions: list[dict[str, Any]],
    *,
    runbook: Mapping[str, Any],
    action_id: str,
    step_name: str,
    reason: str,
    command_filter: str | None = None,
) -> None:
    actions.append(
        {
            "id": action_id,
            "reason": reason,
            "commands": _commands_for_step(runbook, step_name=step_name, command_filter=command_filter),
        }
    )


def _commands_for_step(
    runbook: Mapping[str, Any],
    *,
    step_name: str,
    command_filter: str | None = None,
) -> list[str]:
    for step in runbook.get("steps", []) if isinstance(runbook.get("steps"), list) else []:
        if not isinstance(step, Mapping) or step.get("name") != step_name:
            continue
        commands = [str(command) for command in step.get("commands", []) if isinstance(step.get("commands"), list)]
        if command_filter:
            filters = [part for part in command_filter.split("|") if part]
            return [command for command in commands if any(part in command for part in filters)]
        return commands
    return []


def _commands_for_steps(runbook: Mapping[str, Any], *, step_names: Sequence[str]) -> list[str]:
    requested = set(step_names)
    commands: list[str] = []
    for step in runbook.get("steps", []) if isinstance(runbook.get("steps"), list) else []:
        if not isinstance(step, Mapping) or step.get("name") not in requested:
            continue
        values = step.get("commands") if isinstance(step.get("commands"), list) else []
        commands.extend(str(command) for command in values)
    return commands


def _next_action(
    *,
    gates: Mapping[str, Any],
    runbook: Mapping[str, Any] | None,
    blocking_reasons: Sequence[str],
) -> str:
    if not gates.get("environment_ready"):
        if "missing_api_config" in blocking_reasons:
            return "set_api_config_before_real_run"
        return "fix_legacy_readiness_checks"
    if not gates.get("manifest_available"):
        return "generate_legacy_agbench_suite"
    if not gates.get("suite_structural_ready"):
        return "fix_or_regenerate_legacy_suite"
    if not gates.get("api_config_present"):
        return "set_api_config_before_real_run"
    if gates.get("artifact_quality_failed"):
        return "fix_or_rerun_unreadable_artifacts"
    if gates.get("ready_to_start_real_ab") and not gates.get("offline_semantic_suite_ready"):
        return "run_offline_semantic_suite"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_config_ready"):
        return "configure_local_judge"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_healthcheck_ready"):
        return "run_local_judge_healthcheck"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_template_ready"):
        return "prepare_local_judge_goldset_template"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_annotation_ready"):
        return "complete_local_judge_goldset_annotation"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_import_ready"):
        return "import_labeled_local_judge_goldset_csv"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_validation_ready"):
        return "validate_local_judge_goldset"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_quality_eval_ready"):
        quality_status = (
            gates.get("local_judge_quality_eval_status")
            if isinstance(gates.get("local_judge_quality_eval_status"), Mapping)
            else {}
        )
        if _local_quality_eval_ran(quality_status):
            static_calibration = (
                gates.get("static_rule_calibration_eval_status")
                if isinstance(gates.get("static_rule_calibration_eval_status"), Mapping)
                else {}
            )
            if _static_rule_calibration_eval_ran(static_calibration):
                return "collect_more_gold_labels_or_recalibrate_validator_rules"
            return "run_static_rule_calibration_eval"
        return "run_local_judge_quality_eval"
    if gates.get("ready_to_start_real_ab") and not gates.get("local_judge_goldset_chain_ready"):
        return "verify_local_judge_goldset_chain"
    if gates.get("ready_to_start_real_ab") and not gates.get("offline_local_judge_matrix_ready"):
        return "run_offline_local_judge_matrix"
    if gates.get("ready_to_start_real_ab") and not gates.get("offline_local_judge_matrix_smoke_ready"):
        return "run_offline_local_judge_matrix_smoke"
    if gates.get("ready_to_start_real_ab") and not gates.get("semantic_guard_fake_smoke_ready"):
        return "run_semantic_guard_fake_smoke"
    if gates.get("fake_artifacts_detected"):
        return "run_real_ab_after_fake_smoke"
    if runbook and isinstance(runbook.get("next_action"), str):
        return str(runbook["next_action"])
    if not gates.get("collection_inputs_ready"):
        return "start_proxies_and_run_three_autogenbench_variants"
    if not gates.get("ab_reports_ready"):
        return "run_agbench_legacy_collect"
    return "inspect_ab_reports"


def _suggested_commands(*, next_action: str, readiness: Mapping[str, Any], runbook: Mapping[str, Any] | None) -> list[str]:
    if next_action == "set_api_config_before_real_run":
        return [
            r"# use config\config.txt DeepSeek credentials, or set $env:OPENAI_API_KEY = '<provider-key>'",
            r".venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --mode legacy-proxy",
        ]
    if next_action == "configure_local_judge":
        return [
            (
                r".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_suite "
                r"--output-dir <suite-output-dir> --suite-id <suite-id> --model <provider-model> "
                r"--upstream-base-url <provider-base-url> --scenario-jsonl <Tasks\human_eval_two_agents.jsonl> "
                r"--local-judge-base-url <real-local-judge-base-url> --local-judge-model <real-local-judge-model>"
            ),
            r".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight --manifest <suite-output-dir>\legacy_suite_manifest.json --summary <suite-output-dir>\reports\legacy_real_ab_preflight_summary.json --report-md <suite-output-dir>\reports\legacy_real_ab_preflight.md",
        ]
    if runbook:
        if next_action in {
            "improve_or_recalibrate_local_judge_quality_eval",
            "collect_more_gold_labels_or_recalibrate_validator_rules",
        }:
            return _commands_for_steps(
                runbook,
                step_names=(
                    "local_judge_quality_eval",
                    "local_judge_policy_eval",
                    "static_rule_calibration_eval",
                ),
            )
        for step in runbook.get("steps", []) if isinstance(runbook.get("steps"), list) else []:
            if not isinstance(step, Mapping):
                continue
            name = step.get("name")
            commands = step.get("commands") if isinstance(step.get("commands"), list) else []
            if next_action == "run_real_ab_after_fake_smoke" and name == "start_three_proxies":
                return [str(command) for command in commands]
            if next_action == "run_offline_semantic_suite" and name == "offline_semantic_suite":
                return [str(command) for command in commands]
            if next_action == "run_local_judge_healthcheck" and name == "local_judge_healthcheck":
                return [str(command) for command in commands]
            if next_action == "prepare_local_judge_goldset_template" and name == "local_judge_goldset_template":
                return [str(command) for command in commands]
            if next_action == "complete_local_judge_goldset_annotation" and name == "local_judge_goldset_template":
                return [
                    str(command)
                    for command in commands
                    if "local_judge_goldset_csv progress" in str(command)
                    or "local_judge_goldset_csv labels-template" in str(command)
                    or "local_judge_goldset_csv suggest-labels" in str(command)
                    or "local_judge_goldset_csv merge-suggestions" in str(command)
                    or "local_judge_goldset_csv review-plan" in str(command)
                    or "local_judge_goldset_csv labels-from-csv" in str(command)
                    or "local_judge_goldset_csv validate-labels" in str(command)
                    or "local_judge_goldset_csv apply-labels" in str(command)
                ]
            if next_action == "import_labeled_local_judge_goldset_csv" and name == "local_judge_goldset_template":
                return [
                    str(command)
                    for command in commands
                    if "local_judge_goldset_csv import" in str(command)
                ]
            if next_action == "validate_local_judge_goldset" and name == "local_judge_goldset_validate":
                return [str(command) for command in commands]
            if next_action == "run_local_judge_quality_eval" and name == "local_judge_quality_eval":
                return [str(command) for command in commands]
            if next_action == "run_static_rule_calibration_eval" and name == "static_rule_calibration_eval":
                return [str(command) for command in commands]
            if next_action == "verify_local_judge_goldset_chain" and name == "local_judge_goldset_chain_verify":
                return [str(command) for command in commands]
            if next_action == "run_offline_local_judge_matrix" and name == "offline_local_judge_matrix":
                return [str(command) for command in commands]
            if next_action == "run_offline_local_judge_matrix_smoke" and name == "offline_local_judge_matrix_smoke":
                return [str(command) for command in commands]
            if next_action == "run_semantic_guard_fake_smoke" and name == "semantic_guard_fake_smoke":
                return [str(command) for command in commands]
            if next_action == "start_proxies_and_run_three_autogenbench_variants" and name in {
                "start_three_proxies",
                "run_autogenbench_variants",
            }:
                return [str(command) for command in commands]
            if next_action == "run_agbench_legacy_collect" and name == "collect_and_compare":
                return [str(command) for command in commands]
    commands = readiness.get("suggested_commands") if isinstance(readiness.get("suggested_commands"), Sequence) else []
    return [str(command) for command in commands[:3]]


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


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, str):
        redacted = re.sub(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{6,}", "sk-REDACTED", value)
        redacted = re.sub(r"Bearer\s+\S+", "Bearer REDACTED", redacted)
        return re.sub(r"(?i)(api[_-]?key=)[^&\s]+", r"\1REDACTED", redacted)
    if isinstance(value, Mapping):
        return {str(key): _redact_sensitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive(item) for item in value)
    return value


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


if __name__ == "__main__":
    raise SystemExit(main())
