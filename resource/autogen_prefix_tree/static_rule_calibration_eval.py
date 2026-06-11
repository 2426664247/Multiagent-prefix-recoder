from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_label_eval import ALLOWED_LABELS


FEATURE_SETS: tuple[tuple[str, ...], ...] = (
    ("semantic_hint",),
    ("risk_tag_set",),
    ("semantic_hint", "risk_tag_set"),
    ("rule_label", "semantic_hint", "risk_tag_set"),
)
MIN_SUPPORT_VALUES = (1, 2, 3, 4, 5)
MIN_PURITY_VALUES = (0.50, 0.67, 0.75, 0.80, 1.00)
FALLBACK_POLICIES = ("rule_label", "review")


@dataclass(frozen=True)
class StaticRuleCalibrationEvalResult:
    input_path: str
    summary_path: str | None
    report_path: str | None
    summary: dict[str, Any]


def run_static_rule_calibration_eval(
    *,
    input_path: str | Path,
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    min_accuracy: float = 0.75,
    min_macro_f1: float = 0.70,
    production_min_support: int = 2,
    production_min_purity: float = 0.67,
    top_policy_count: int = 12,
) -> StaticRuleCalibrationEvalResult:
    source_path = Path(input_path)
    rows = tuple(_normalize_row(row, index=index) for index, row in enumerate(_load_jsonl(source_path)))
    baseline_summaries = [
        _baseline_policy_summary(
            rows,
            policy_name="source_label",
            field_name="source_label",
            min_accuracy=min_accuracy,
            min_macro_f1=min_macro_f1,
        ),
        _baseline_policy_summary(
            rows,
            policy_name="rule_label",
            field_name="rule_label",
            min_accuracy=min_accuracy,
            min_macro_f1=min_macro_f1,
        ),
    ]
    calibration_summaries = [
        _calibrated_policy_summary(
            rows,
            feature_names=feature_names,
            min_support=min_support,
            min_purity=min_purity,
            fallback_policy=fallback_policy,
            min_accuracy=min_accuracy,
            min_macro_f1=min_macro_f1,
            production_min_support=production_min_support,
            production_min_purity=production_min_purity,
        )
        for feature_names in FEATURE_SETS
        for min_support in MIN_SUPPORT_VALUES
        for min_purity in MIN_PURITY_VALUES
        for fallback_policy in FALLBACK_POLICIES
    ]
    policy_summaries = baseline_summaries + calibration_summaries
    top_policies = _top_policies(policy_summaries, limit=top_policy_count)
    best_ready = _best_policy(policy_summaries, ready_only=True, production_only=False)
    best_production_candidate = _best_policy(policy_summaries, ready_only=True, production_only=True)
    production_readiness_diagnostics = _production_readiness_diagnostics(
        policies=policy_summaries,
        best_ready=best_ready,
        best_production_candidate=best_production_candidate,
        production_min_support=production_min_support,
        production_min_purity=production_min_purity,
    )
    gold_support_diagnostics = _gold_support_diagnostics(
        rows,
        production_min_support=production_min_support,
        production_min_purity=production_min_purity,
    )
    validator_rule_candidates = _validator_rule_candidates(
        gold_support_diagnostics,
        rows,
        production_min_support=production_min_support,
        production_min_purity=production_min_purity,
    )
    summary = {
        "schema_version": "prefix-static-rule-calibration-eval-summary-v1",
        "prompt_safe_summary": True,
        "input_path": str(source_path),
        "row_count": len(rows),
        "min_accuracy": min_accuracy,
        "min_macro_f1": min_macro_f1,
        "production_min_support": production_min_support,
        "production_min_purity": production_min_purity,
        "expected_label_counts": dict(sorted(Counter(row["expected_label"] for row in rows).items())),
        "feature_sets": ["+".join(feature_names) for feature_names in FEATURE_SETS],
        "candidate_config_count": len(calibration_summaries),
        "policy_summaries": policy_summaries,
        "top_policies": top_policies,
        "best_ready_policy": best_ready,
        "best_production_candidate_policy": best_production_candidate,
        "production_readiness_diagnostics": production_readiness_diagnostics,
        "gold_support_diagnostics": gold_support_diagnostics,
        "validator_rule_candidates": validator_rule_candidates,
        "ready": best_production_candidate is not None,
        "exploratory_ready": best_ready is not None,
        "semantic_quality_claim_allowed": best_production_candidate is not None,
        "real_provider_metrics_available": False,
        "gold_text_written": False,
        "recommendation": _recommendation(
            best_ready=best_ready,
            best_production_candidate=best_production_candidate,
        ),
        "limits": (
            "This evaluates prompt-safe metadata-only static rule calibration with leave-one-out cross-validation. "
            "It does not read candidate text into reports, call a model, prove provider cache metrics, or prove "
            "broad cross-framework generalization."
        ),
    }
    summary_target = _write_json(summary_path, summary)
    report_target = _write_text(report_path, render_markdown_static_rule_calibration(summary))
    return StaticRuleCalibrationEvalResult(
        input_path=str(source_path),
        summary_path=str(summary_target) if summary_target else None,
        report_path=str(report_target) if report_target else None,
        summary=summary,
    )


def render_markdown_static_rule_calibration(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Static Rule Calibration Evaluation",
        "",
        f"Ready: `{_format_value(summary.get('ready'))}`",
        f"Exploratory ready: `{_format_value(summary.get('exploratory_ready'))}`",
        f"Best ready policy: `{_format_value(summary.get('best_ready_policy'))}`",
        f"Best production candidate: `{_format_value(summary.get('best_production_candidate_policy'))}`",
        f"Recommendation: **{_format_value(summary.get('recommendation'))}**",
        "",
        "## Top Policies",
        "",
        "| Policy | Ready | Production Candidate | Accuracy | Macro F1 | Coverage | Predicted Counts |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for policy in summary.get("top_policies") if isinstance(summary.get("top_policies"), list) else []:
        if not isinstance(policy, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(policy.get("policy_name")),
                    _format_value(policy.get("ready")),
                    _format_value(policy.get("production_candidate")),
                    _format_value(policy.get("accuracy")),
                    _format_value(policy.get("macro_f1")),
                    _format_value(policy.get("calibration_coverage_rate")),
                    _format_value(policy.get("predicted_label_counts")),
                ]
            )
            + " |"
        )
    readiness = (
        summary.get("production_readiness_diagnostics")
        if isinstance(summary.get("production_readiness_diagnostics"), Mapping)
        else {}
    )
    if readiness:
        lines.extend(
            [
                "",
                "## Production Readiness Diagnostics",
                "",
                f"Status: `{_format_value(readiness.get('status'))}`",
                f"Recommendation: **{_format_value(readiness.get('recommendation'))}**",
                "",
                "| Policy | Automation Policy | Accuracy | Macro F1 | Missing Production Support | Missing Production Purity |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for policy in readiness.get("top_ready_nonproduction_policies") if isinstance(readiness.get("top_ready_nonproduction_policies"), list) else []:
            if not isinstance(policy, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_value(policy.get("policy_name")),
                        _format_value(policy.get("automation_policy")),
                        _format_value(policy.get("accuracy")),
                        _format_value(policy.get("macro_f1")),
                        _format_value(policy.get("missing_production_support")),
                        _format_value(policy.get("missing_production_purity")),
                    ]
                )
                + " |"
            )
    support = (
        summary.get("gold_support_diagnostics")
        if isinstance(summary.get("gold_support_diagnostics"), Mapping)
        else {}
    )
    if support:
        lines.extend(
            [
                "",
                "## Gold Support Diagnostics",
                "",
                f"Recommendation: **{_format_value(support.get('recommendation'))}**",
                "",
                "| Feature Set | Features | Support | Dominant Label | Purity | Needed Additional Labels |",
                "|---|---|---:|---|---:|---:|",
            ]
        )
        for bucket in support.get("priority_buckets") if isinstance(support.get("priority_buckets"), list) else []:
            if not isinstance(bucket, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_value(bucket.get("feature_set")),
                        _format_value(bucket.get("features")),
                        _format_value(bucket.get("support")),
                        _format_value(bucket.get("dominant_label")),
                        _format_value(bucket.get("purity")),
                        _format_value(bucket.get("needed_additional_labels")),
                    ]
                )
                + " |"
            )
    candidates = (
        summary.get("validator_rule_candidates")
        if isinstance(summary.get("validator_rule_candidates"), Mapping)
        else {}
    )
    if candidates:
        promotion_gate = (
            candidates.get("promotion_gate")
            if isinstance(candidates.get("promotion_gate"), Mapping)
            else {}
        )
        overlap = (
            candidates.get("overlap_diagnostics")
            if isinstance(candidates.get("overlap_diagnostics"), Mapping)
            else {}
        )
        simulation = (
            candidates.get("fail_closed_simulation")
            if isinstance(candidates.get("fail_closed_simulation"), Mapping)
            else {}
        )
        review_plan = (
            candidates.get("validator_review_plan")
            if isinstance(candidates.get("validator_review_plan"), Mapping)
            else {}
        )
        subset = (
            candidates.get("conservative_fail_closed_subset")
            if isinstance(candidates.get("conservative_fail_closed_subset"), Mapping)
            else {}
        )
        source_generalization = (
            candidates.get("source_generalization_diagnostics")
            if isinstance(candidates.get("source_generalization_diagnostics"), Mapping)
            else {}
        )
        shadow_trial = (
            candidates.get("shadow_trial_plan")
            if isinstance(candidates.get("shadow_trial_plan"), Mapping)
            else {}
        )
        subset_simulation = (
            subset.get("simulation")
            if isinstance(subset.get("simulation"), Mapping)
            else {}
        )
        lines.extend(
            [
                "",
                "## Validator Rule Candidates",
                "",
                f"Status: `{_format_value(candidates.get('status'))}`",
                f"Recommendation: **{_format_value(candidates.get('recommendation'))}**",
                f"Automatic promotion ready: `{_format_value(promotion_gate.get('automatic_promotion_ready'))}`",
                f"Promotion blocking reasons: `{_format_value(promotion_gate.get('blocking_reasons'))}`",
                f"Conflicting overlap pairs: `{_format_value(overlap.get('conflicting_overlap_pair_count'))}`",
                f"Fail-closed simulation accuracy: `{_format_value(simulation.get('accuracy'))}`",
                f"Fail-closed matched rate: `{_format_value(simulation.get('matched_rate'))}`",
                f"Fail-closed mismatch count: `{_format_value(simulation.get('mismatch_count'))}`",
                f"Validator review items: `{_format_value(review_plan.get('review_item_count'))}`",
                f"Next review type: `{_format_value(review_plan.get('recommended_next_review_type'))}`",
                f"Conservative fail-closed included rules: `{_format_value(subset.get('included_rule_count'))}`",
                f"Conservative fail-closed excluded rules: `{_format_value(subset.get('excluded_rule_count'))}`",
                f"Conservative fail-closed mismatch count: `{_format_value(subset_simulation.get('mismatch_count'))}`",
                f"Shadow trial ready: `{_format_value(shadow_trial.get('shadow_trial_ready'))}`",
                f"Shadow trial rule count: `{_format_value(shadow_trial.get('trial_rule_count'))}`",
                f"Validator behavior change allowed: `{_format_value(shadow_trial.get('validator_behavior_change_allowed'))}`",
                f"Multi-source candidate rules: `{_format_value(source_generalization.get('candidate_with_multi_source_group_count'))}`",
                f"Source generalization claim allowed: `{_format_value(source_generalization.get('generalization_claim_allowed'))}`",
                "",
                "| Feature Set | Features | Label | Support | Purity | Action | Reason |",
                "|---|---|---|---:|---:|---|---|",
            ]
        )
        for candidate in candidates.get("candidate_rules") if isinstance(candidates.get("candidate_rules"), list) else []:
            if not isinstance(candidate, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_value(candidate.get("feature_set")),
                        _format_value(candidate.get("features")),
                        _format_value(candidate.get("label")),
                        _format_value(candidate.get("support")),
                        _format_value(candidate.get("purity")),
                        _format_value(candidate.get("validator_action")),
                        _format_value(candidate.get("promotion_reason")),
                    ]
                )
                + " |"
            )
        if review_plan:
            lines.extend(
                [
                    "",
                    "## Validator Review Plan",
                    "",
                    f"Recommendation: **{_format_value(review_plan.get('recommendation'))}**",
                    f"Automation policy: `{_format_value(review_plan.get('automation_policy'))}`",
                    "",
                    "| Rank | Type | Reason | Feature Set | Features | Action | Expected | Predicted | Support | Purity |",
                    "|---:|---|---|---|---|---|---|---|---:|---:|",
                ]
            )
            for item in review_plan.get("review_items") if isinstance(review_plan.get("review_items"), list) else []:
                if not isinstance(item, Mapping):
                    continue
                feature_set = (
                    item.get("feature_set")
                    or item.get("candidate_feature_set")
                    or item.get("left_feature_set")
                    or item.get("right_feature_set")
                )
                features = (
                    item.get("features")
                    or item.get("candidate_features")
                    or item.get("left_features")
                    or item.get("right_features")
                )
                action = (
                    item.get("validator_action")
                    or item.get("candidate_action")
                    or item.get("left_action")
                    or item.get("right_action")
                )
                support = item.get("support") or item.get("candidate_support") or item.get("left_support")
                purity = item.get("purity") or item.get("candidate_purity") or item.get("left_purity")
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _format_value(item.get("priority_rank")),
                            _format_value(item.get("review_type")),
                            _format_value(item.get("review_reason")),
                            _format_value(feature_set),
                            _format_value(features),
                            _format_value(action),
                            _format_value(item.get("expected_label")),
                            _format_value(item.get("predicted_label")),
                            _format_value(support),
                            _format_value(purity),
                        ]
                    )
                    + " |"
                )
        if subset:
            lines.extend(
                [
                    "",
                    "## Conservative Fail-Closed Subset",
                    "",
                    f"Recommendation: **{_format_value(subset.get('recommendation'))}**",
                    f"Manual trial ready: `{_format_value(subset.get('manual_trial_ready'))}`",
                    f"Automation policy: `{_format_value(subset.get('automation_policy'))}`",
                    f"Exclusion reasons: `{_format_value(subset.get('exclusion_reason_counts'))}`",
                    "",
                    "| Feature Set | Features | Action | Support | Purity | Decision | Exclusion Reasons |",
                    "|---|---|---|---:|---:|---|---|",
                ]
            )
            subset_rows = []
            if isinstance(subset.get("included_candidate_rules"), list):
                subset_rows.extend(subset.get("included_candidate_rules"))
            if isinstance(subset.get("excluded_candidate_rules"), list):
                subset_rows.extend(subset.get("excluded_candidate_rules"))
            for item in subset_rows[:20]:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _format_value(item.get("feature_set")),
                            _format_value(item.get("features")),
                            _format_value(item.get("validator_action")),
                            _format_value(item.get("support")),
                            _format_value(item.get("purity")),
                            _format_value(item.get("subset_decision")),
                            _format_value(item.get("exclusion_reasons")),
                        ]
                    )
                    + " |"
                )
        if shadow_trial:
            lines.extend(
                [
                    "",
                    "## Validator Shadow Trial Plan",
                    "",
                    f"Recommendation: **{_format_value(shadow_trial.get('recommendation'))}**",
                    f"Automation policy: `{_format_value(shadow_trial.get('automation_policy'))}`",
                    f"Trial mode: `{_format_value(shadow_trial.get('trial_mode'))}`",
                    f"Source-diverse rules: `{_format_value(shadow_trial.get('source_diverse_rule_count'))}`",
                    f"Source conflict rules: `{_format_value(shadow_trial.get('source_conflict_rule_count'))}`",
                    "",
                    "| Rule ID | Feature Set | Features | Shadow Action | Support | Purity | Source Groups | Source Files | Review Status |",
                    "|---|---|---|---|---:|---:|---:|---:|---|",
                ]
            )
            shadow_rows = shadow_trial.get("trial_rules")
            for item in shadow_rows[:20] if isinstance(shadow_rows, list) else []:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _format_value(item.get("trial_rule_id")),
                            _format_value(item.get("feature_set")),
                            _format_value(item.get("features")),
                            _format_value(item.get("shadow_action")),
                            _format_value(item.get("support")),
                            _format_value(item.get("purity")),
                            _format_value(item.get("source_group_count")),
                            _format_value(item.get("source_file_hash_count")),
                            _format_value(item.get("review_status")),
                        ]
                    )
                    + " |"
                )
        if source_generalization:
            lines.extend(
                [
                    "",
                    "## Source Generalization Diagnostics",
                    "",
                    f"Recommendation: **{_format_value(source_generalization.get('recommendation'))}**",
                    f"Automation policy: `{_format_value(source_generalization.get('automation_policy'))}`",
                    f"Candidate rules with multi-source groups: `{_format_value(source_generalization.get('candidate_with_multi_source_group_count'))}`",
                    f"Conservative subset with multi-source groups: `{_format_value(source_generalization.get('conservative_subset_multi_source_group_count'))}`",
                    f"Weak source support count: `{_format_value(source_generalization.get('weak_source_support_count'))}`",
                    f"Source conflict count: `{_format_value(source_generalization.get('source_conflict_count'))}`",
                    "",
                    "| Feature Set | Features | Action | Matched | Source Groups | Source Files | Expected Labels | Source Labels |",
                    "|---|---|---|---:|---:|---:|---|---|",
                ]
            )
            rows = source_generalization.get("top_candidate_source_summaries")
            for item in rows if isinstance(rows, list) else []:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _format_value(item.get("feature_set")),
                            _format_value(item.get("features")),
                            _format_value(item.get("validator_action")),
                            _format_value(item.get("matched_count")),
                            _format_value(item.get("source_group_count")),
                            _format_value(item.get("source_file_hash_count")),
                            _format_value(item.get("expected_label_counts_by_source")),
                            _format_value(item.get("source_label_counts_by_source")),
                        ]
                    )
                    + " |"
                )
        if candidates.get("conflict_buckets"):
            lines.extend(
                [
                    "",
                    "| Conflict Feature Set | Features | Support | Purity | Needed Additional Labels | Reason |",
                    "|---|---|---:|---:|---:|---|",
                ]
            )
            for bucket in candidates.get("conflict_buckets") if isinstance(candidates.get("conflict_buckets"), list) else []:
                if not isinstance(bucket, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _format_value(bucket.get("feature_set")),
                            _format_value(bucket.get("features")),
                            _format_value(bucket.get("support")),
                            _format_value(bucket.get("purity")),
                            _format_value(bucket.get("needed_additional_labels")),
                            _format_value(bucket.get("reason")),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Policy | Accuracy | Macro F1 | Predicted Counts | Failure Modes |",
            "|---|---:|---:|---|---|",
        ]
    )
    for policy in summary.get("policy_summaries") if isinstance(summary.get("policy_summaries"), list) else []:
        if not isinstance(policy, Mapping) or policy.get("policy_family") != "baseline":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(policy.get("policy_name")),
                    _format_value(policy.get("accuracy")),
                    _format_value(policy.get("macro_f1")),
                    _format_value(policy.get("predicted_label_counts")),
                    _format_value(policy.get("failure_mode_codes")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This report is prompt-safe and does not include candidate text or API key values.",
            "- Passing here is scoped to this gold set and cross-validation protocol only.",
            "- Real provider cache, latency, cost, and task-success claims still require provider A/B runs.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate metadata-only static rule calibration against a gold label JSONL."
    )
    parser.add_argument("--input", required=True, help="Gold JSONL with expected_label and prompt-safe metadata.")
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--min-macro-f1", type=float, default=0.70)
    parser.add_argument("--production-min-support", type=int, default=2)
    parser.add_argument("--production-min-purity", type=float, default=0.67)
    parser.add_argument("--top-policy-count", type=int, default=12)
    parser.add_argument("--summary", help="Optional prompt-safe summary JSON path.")
    parser.add_argument("--report-md", help="Optional prompt-safe Markdown report path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_static_rule_calibration_eval(
        input_path=args.input,
        summary_path=args.summary,
        report_path=args.report_md,
        min_accuracy=args.min_accuracy,
        min_macro_f1=args.min_macro_f1,
        production_min_support=args.production_min_support,
        production_min_purity=args.production_min_purity,
        top_policy_count=args.top_policy_count,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.summary.get("ready") is True else 1


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                yield value


def _normalize_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    risk_tags = tuple(sorted(str(tag) for tag in row.get("risk_tags") or ()))
    source_path = _safe_token(row.get("source_path") or "unknown")
    return {
        "input_index": index,
        "expected_label": _safe_label(row.get("expected_label") or row.get("gold_label")),
        "source_label": _safe_label(row.get("source_label")),
        "rule_label": _safe_label(row.get("rule_label") or row.get("source_label")),
        "semantic_hint": _safe_token(row.get("semantic_hint") or "unknown"),
        "risk_tag_set": "|".join(risk_tags) if risk_tags else "none",
        "risk_tag_count": len(risk_tags),
        "source_group": _source_group(source_path),
        "source_path_hash": _short_hash(source_path),
        "extraction": _safe_token(row.get("extraction") or "unknown"),
    }


def _baseline_policy_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_name: str,
    field_name: str,
    min_accuracy: float,
    min_macro_f1: float,
) -> dict[str, Any]:
    predictions = [
        str(row.get(field_name)) if row.get(field_name) in ALLOWED_LABELS else "review"
        for row in rows
    ]
    base = _metric_summary(
        rows,
        predictions=predictions,
        policy_name=policy_name,
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
    )
    return {
        **base,
        "policy_family": "baseline",
        "production_candidate": base["ready"],
        "automation_policy": "baseline_reference",
        "calibration_coverage_count": 0,
        "calibration_coverage_rate": 0.0,
    }


def _calibrated_policy_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    min_support: int,
    min_purity: float,
    fallback_policy: str,
    min_accuracy: float,
    min_macro_f1: float,
    production_min_support: int,
    production_min_purity: float,
) -> dict[str, Any]:
    predictions: list[str] = []
    covered = 0
    for holdout_index, row in enumerate(rows):
        training_rows = rows[:holdout_index] + rows[holdout_index + 1 :]
        mapping = _learn_mapping(
            training_rows,
            feature_names=feature_names,
            min_support=min_support,
            min_purity=min_purity,
        )
        row_key = _feature_key(row, feature_names)
        if row_key in mapping:
            predictions.append(mapping[row_key]["label"])
            covered += 1
        else:
            predictions.append(_fallback_label(row, fallback_policy=fallback_policy))
    full_mapping = _learn_mapping(
        rows,
        feature_names=feature_names,
        min_support=min_support,
        min_purity=min_purity,
    )
    policy_name = (
        f"metadata_calibration:{'+'.join(feature_names)}:"
        f"support>={min_support}:purity>={min_purity:g}:fallback={fallback_policy}"
    )
    base = _metric_summary(
        rows,
        predictions=predictions,
        policy_name=policy_name,
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
    )
    production_thresholds_met = min_support >= production_min_support and min_purity >= production_min_purity
    production_candidate = bool(base["ready"] and production_thresholds_met)
    return {
        **base,
        "policy_family": "metadata_calibration",
        "feature_names": list(feature_names),
        "min_support": min_support,
        "min_purity": min_purity,
        "fallback_policy": fallback_policy,
        "calibration_coverage_count": covered,
        "calibration_coverage_rate": _ratio_or_none(covered, len(rows)),
        "learned_rule_count": len(full_mapping),
        "learned_rules": _learned_rule_rows(full_mapping, feature_names=feature_names),
        "production_candidate": production_candidate,
        "automation_policy": "candidate_for_validator_rule_review"
        if production_candidate
        else (
            "exploratory_only_low_support_or_purity"
            if base["ready"]
            else "diagnostic_only_until_thresholds_pass"
        ),
    }


def _learn_mapping(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    min_support: int,
    min_purity: float,
) -> dict[tuple[str, ...], dict[str, Any]]:
    grouped: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        expected = row.get("expected_label")
        if expected in ALLOWED_LABELS:
            grouped[_feature_key(row, feature_names)][str(expected)] += 1
    mapping: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, counts in grouped.items():
        support = sum(counts.values())
        ranked = counts.most_common()
        if not ranked:
            continue
        label, count = ranked[0]
        if len(ranked) > 1 and ranked[1][1] == count:
            continue
        purity = count / support if support else 0.0
        if support >= min_support and purity >= min_purity:
            mapping[key] = {
                "label": label,
                "support": support,
                "purity": purity,
                "expected_label_counts": dict(sorted(counts.items())),
            }
    return mapping


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    predictions: Sequence[str],
    policy_name: str,
    min_accuracy: float,
    min_macro_f1: float,
) -> dict[str, Any]:
    evaluated = [
        (str(row.get("expected_label")), predicted)
        for row, predicted in zip(rows, predictions)
        if row.get("expected_label") in ALLOWED_LABELS and predicted in ALLOWED_LABELS
    ]
    confusion = _confusion_matrix(evaluated)
    metrics = _per_label_metrics(confusion)
    accuracy = _ratio_or_none(sum(1 for expected, predicted in evaluated if expected == predicted), len(evaluated))
    macro_f1 = _macro_value(metrics, "f1")
    expected_counts = Counter(expected for expected, _predicted in evaluated)
    predicted_counts = Counter(predicted for _expected, predicted in evaluated)
    failure_modes = _failure_modes(
        accuracy=accuracy,
        macro_f1=macro_f1,
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
        expected_counts=expected_counts,
        predicted_counts=predicted_counts,
        metrics=metrics,
    )
    ready = bool(
        evaluated
        and accuracy is not None
        and accuracy >= min_accuracy
        and macro_f1 is not None
        and macro_f1 >= min_macro_f1
        and not failure_modes
    )
    return {
        "schema_version": "prefix-static-rule-calibration-policy-summary-v1",
        "policy_name": policy_name,
        "ready": ready,
        "evaluated_count": len(evaluated),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "expected_label_counts": dict(sorted(expected_counts.items())),
        "predicted_label_counts": dict(sorted(predicted_counts.items())),
        "confusion_matrix": confusion,
        "per_label_metrics": metrics,
        "failure_mode_codes": failure_modes,
    }


def _confusion_matrix(rows: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    labels = sorted(ALLOWED_LABELS)
    matrix = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for expected, predicted in rows:
        if expected in ALLOWED_LABELS and predicted in ALLOWED_LABELS:
            matrix[expected][predicted] += 1
    return matrix


def _per_label_metrics(confusion: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    labels = sorted(ALLOWED_LABELS)
    for label in labels:
        tp = int(confusion.get(label, {}).get(label, 0))
        fp = sum(int(confusion.get(other, {}).get(label, 0)) for other in labels if other != label)
        fn = sum(int(confusion.get(label, {}).get(other, 0)) for other in labels if other != label)
        precision = _ratio_or_none(tp, tp + fp)
        recall = _ratio_or_none(tp, tp + fn)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion.get(label, {}).values()),
        }
    return result


def _failure_modes(
    *,
    accuracy: float | None,
    macro_f1: float | None,
    min_accuracy: float,
    min_macro_f1: float,
    expected_counts: Mapping[str, int],
    predicted_counts: Mapping[str, int],
    metrics: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    modes: list[str] = []
    if accuracy is None or accuracy < min_accuracy:
        modes.append("accuracy_below_min")
    if macro_f1 is None or macro_f1 < min_macro_f1:
        modes.append("macro_f1_below_min")
    expected_labels = sorted(label for label in ALLOWED_LABELS if expected_counts.get(label, 0) > 0)
    for label in expected_labels:
        if int(predicted_counts.get(label, 0)) == 0:
            modes.append(f"missing_{label}_predictions")
        if metrics.get(label, {}).get("recall") == 0:
            modes.append(f"{label}_zero_recall")
    return modes


def _top_policies(policies: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    top = sorted(
        policies,
        key=lambda policy: (
            _score(policy.get("macro_f1")),
            _score(policy.get("accuracy")),
            _num(policy.get("calibration_coverage_count")),
            str(policy.get("policy_name")),
        ),
        reverse=True,
    )[: max(0, limit)]
    fields = (
        "policy_name",
        "policy_family",
        "ready",
        "production_candidate",
        "automation_policy",
        "accuracy",
        "macro_f1",
        "calibration_coverage_count",
        "calibration_coverage_rate",
        "predicted_label_counts",
        "failure_mode_codes",
    )
    return [{field: policy.get(field) for field in fields} for policy in top]


def _production_readiness_diagnostics(
    *,
    policies: Sequence[Mapping[str, Any]],
    best_ready: str | None,
    best_production_candidate: str | None,
    production_min_support: int,
    production_min_purity: float,
) -> dict[str, Any]:
    ready_nonproduction = [
        policy
        for policy in policies
        if policy.get("ready") is True
        and policy.get("production_candidate") is not True
        and policy.get("policy_family") == "metadata_calibration"
    ]
    rows = [
        _production_gap_row(
            policy,
            production_min_support=production_min_support,
            production_min_purity=production_min_purity,
        )
        for policy in ready_nonproduction
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            _score(row.get("macro_f1")),
            _score(row.get("accuracy")),
            _num(row.get("calibration_coverage_count")),
            str(row.get("policy_name")),
        ),
        reverse=True,
    )
    status = (
        "production_candidate_ready"
        if best_production_candidate
        else ("exploratory_ready_only" if best_ready else "no_ready_metadata_policy")
    )
    return {
        "schema_version": "prefix-static-rule-production-readiness-diagnostics-v1",
        "prompt_safe_summary": True,
        "status": status,
        "best_ready_policy": best_ready,
        "best_production_candidate_policy": best_production_candidate,
        "ready_nonproduction_policy_count": len(ready_nonproduction),
        "production_min_support": production_min_support,
        "production_min_purity": production_min_purity,
        "top_ready_nonproduction_policies": rows[:8],
        "missing_threshold_reason_counts": dict(
            sorted(Counter(reason for row in rows for reason in row.get("missing_threshold_reasons", [])).items())
        ),
        "recommendation": _production_readiness_recommendation(
            best_ready=best_ready,
            best_production_candidate=best_production_candidate,
        ),
    }


def _production_gap_row(
    policy: Mapping[str, Any],
    *,
    production_min_support: int,
    production_min_purity: float,
) -> dict[str, Any]:
    min_support = _num(policy.get("min_support"))
    min_purity = _score(policy.get("min_purity"))
    reasons: list[str] = []
    missing_support = max(0, production_min_support - min_support)
    missing_purity = max(0.0, production_min_purity - min_purity)
    if missing_support > 0:
        reasons.append("min_support_below_production")
    if missing_purity > 0:
        reasons.append("min_purity_below_production")
    return {
        "policy_name": policy.get("policy_name"),
        "automation_policy": policy.get("automation_policy"),
        "feature_names": policy.get("feature_names") if isinstance(policy.get("feature_names"), list) else [],
        "min_support": min_support,
        "min_purity": min_purity,
        "missing_production_support": missing_support,
        "missing_production_purity": missing_purity,
        "missing_threshold_reasons": reasons,
        "accuracy": _score(policy.get("accuracy")),
        "macro_f1": _score(policy.get("macro_f1")),
        "calibration_coverage_count": _num(policy.get("calibration_coverage_count")),
        "calibration_coverage_rate": policy.get("calibration_coverage_rate"),
        "predicted_label_counts": policy.get("predicted_label_counts")
        if isinstance(policy.get("predicted_label_counts"), Mapping)
        else {},
    }


def _gold_support_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    production_min_support: int,
    production_min_purity: float,
) -> dict[str, Any]:
    bucket_rows = [
        bucket
        for feature_names in FEATURE_SETS
        for bucket in _feature_bucket_rows(
            rows,
            feature_names=feature_names,
            production_min_support=production_min_support,
            production_min_purity=production_min_purity,
        )
    ]
    priority = sorted(
        bucket_rows,
        key=lambda bucket: (
            _num(bucket.get("needed_additional_labels")) > 0,
            _score(bucket.get("purity_gap")),
            _num(bucket.get("support")),
            str(bucket.get("feature_set")),
            str(bucket.get("features")),
        ),
        reverse=True,
    )[:12]
    support_ready_count = sum(1 for bucket in bucket_rows if bucket.get("support_ready") is True)
    purity_ready_count = sum(1 for bucket in bucket_rows if bucket.get("purity_ready") is True)
    return {
        "schema_version": "prefix-static-rule-gold-support-diagnostics-v1",
        "prompt_safe_summary": True,
        "feature_bucket_count": len(bucket_rows),
        "support_ready_bucket_count": support_ready_count,
        "purity_ready_bucket_count": purity_ready_count,
        "production_ready_bucket_count": sum(1 for bucket in bucket_rows if bucket.get("production_ready") is True),
        "production_min_support": production_min_support,
        "production_min_purity": production_min_purity,
        "priority_buckets": priority,
        "all_buckets": bucket_rows,
        "feature_set_bucket_counts": dict(sorted(Counter(str(bucket.get("feature_set")) for bucket in bucket_rows).items())),
        "recommendation": _gold_support_recommendation(
            bucket_rows,
            production_min_support=production_min_support,
            production_min_purity=production_min_purity,
        ),
    }


def _validator_rule_candidates(
    gold_support: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    production_min_support: int,
    production_min_purity: float,
) -> dict[str, Any]:
    bucket_rows = (
        gold_support.get("all_buckets")
        if isinstance(gold_support.get("all_buckets"), list)
        else gold_support.get("priority_buckets")
        if isinstance(gold_support.get("priority_buckets"), list)
        else []
    )
    candidate_rows = [
        _validator_candidate_row(bucket)
        for bucket in bucket_rows
        if isinstance(bucket, Mapping) and bucket.get("production_ready") is True
    ]
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            row.get("validator_action") == "prefer_review",
            _score(row.get("purity")),
            _num(row.get("support")),
            str(row.get("feature_set")),
            str(row.get("features")),
        ),
        reverse=True,
    )
    conflict_rows = [
        _validator_conflict_row(bucket)
        for bucket in bucket_rows
        if isinstance(bucket, Mapping)
        and bucket.get("production_ready") is not True
        and (
            _num(bucket.get("support")) >= production_min_support
            or _num(bucket.get("needed_additional_labels")) > 0
        )
    ]
    conflict_rows = sorted(
        conflict_rows,
        key=lambda row: (
            _num(row.get("needed_additional_labels")),
            _num(row.get("support")),
            str(row.get("feature_set")),
            str(row.get("features")),
        ),
        reverse=True,
    )
    safe_accept_count = sum(1 for row in candidate_rows if row.get("validator_action") == "allow_accept")
    reject_count = sum(1 for row in candidate_rows if row.get("validator_action") == "force_reject")
    review_count = sum(1 for row in candidate_rows if row.get("validator_action") == "prefer_review")
    overlap_diagnostics = _validator_candidate_overlap_diagnostics(candidate_rows)
    fail_closed_simulation = _validator_fail_closed_simulation(
        rows=rows,
        candidates=candidate_rows,
    )
    promotion_gate = _validator_candidate_promotion_gate(
        candidate_count=len(candidate_rows),
        safe_accept_count=safe_accept_count,
        conflict_count=len(conflict_rows),
        overlap_diagnostics=overlap_diagnostics,
    )
    validator_review_plan = _validator_review_plan(
        candidate_rows=candidate_rows,
        conflict_rows=conflict_rows,
        overlap_diagnostics=overlap_diagnostics,
        fail_closed_simulation=fail_closed_simulation,
    )
    conservative_fail_closed_subset = _conservative_fail_closed_subset(
        rows=rows,
        candidate_rows=candidate_rows,
        overlap_diagnostics=overlap_diagnostics,
        fail_closed_simulation=fail_closed_simulation,
    )
    source_generalization = _validator_source_generalization(
        rows=rows,
        candidate_rows=candidate_rows,
        conservative_subset=conservative_fail_closed_subset,
    )
    shadow_trial_plan = _validator_shadow_trial_plan(
        rows=rows,
        conservative_subset=conservative_fail_closed_subset,
    )
    return {
        "schema_version": "prefix-static-rule-validator-candidates-v1",
        "prompt_safe_summary": True,
        "candidate_rule_count": len(candidate_rows),
        "safe_accept_rule_count": safe_accept_count,
        "reject_rule_count": reject_count,
        "review_rule_count": review_count,
        "production_min_support": production_min_support,
        "production_min_purity": production_min_purity,
        "candidate_rules": candidate_rows[:20],
        "conflict_bucket_count": len(conflict_rows),
        "conflict_buckets": conflict_rows[:12],
        "overlap_diagnostics": overlap_diagnostics,
        "fail_closed_simulation": fail_closed_simulation,
        "promotion_gate": promotion_gate,
        "validator_review_plan": validator_review_plan,
        "conservative_fail_closed_subset": conservative_fail_closed_subset,
        "shadow_trial_plan": shadow_trial_plan,
        "source_generalization_diagnostics": source_generalization,
        "status": "candidate_rules_available" if candidate_rows else "no_candidate_rules",
        "recommendation": _validator_candidate_recommendation(
            candidate_count=len(candidate_rows),
            conflict_count=len(conflict_rows),
            safe_accept_count=safe_accept_count,
        ),
        "limits": (
            "These are prompt-safe metadata-only rule candidates. They are not automatically promoted into "
            "Validator behavior without code review, larger gold support, and real provider A/B validation."
        ),
    }


def _validator_review_plan(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    overlap_diagnostics: Mapping[str, Any],
    fail_closed_simulation: Mapping[str, Any],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    overlap_pairs = (
        overlap_diagnostics.get("conflicting_overlap_pairs")
        if isinstance(overlap_diagnostics.get("conflicting_overlap_pairs"), list)
        else []
    )
    for pair in overlap_pairs:
        if not isinstance(pair, Mapping):
            continue
        items.append(
            {
                "priority_group": "p0_conflicting_candidate_overlap",
                "review_type": "conflicting_overlap_pair",
                "review_reason": "overlapping_candidate_rules_have_conflicting_actions",
                "relation": pair.get("relation"),
                "left_action": pair.get("left_action"),
                "right_action": pair.get("right_action"),
                "left_feature_set": pair.get("left_feature_set"),
                "right_feature_set": pair.get("right_feature_set"),
                "left_features": pair.get("left_features") if isinstance(pair.get("left_features"), Mapping) else {},
                "right_features": pair.get("right_features") if isinstance(pair.get("right_features"), Mapping) else {},
                "left_support": _num(pair.get("left_support")),
                "right_support": _num(pair.get("right_support")),
                "left_purity": pair.get("left_purity"),
                "right_purity": pair.get("right_purity"),
            }
        )

    mismatches = (
        fail_closed_simulation.get("sample_mismatches")
        if isinstance(fail_closed_simulation.get("sample_mismatches"), list)
        else []
    )
    for mismatch in mismatches:
        if not isinstance(mismatch, Mapping):
            continue
        items.append(
            {
                "priority_group": "p1_fail_closed_mismatch",
                "review_type": "fail_closed_simulation_mismatch",
                "review_reason": "fail_closed_candidate_changes_gold_label",
                "input_index": _num(mismatch.get("input_index")),
                "expected_label": mismatch.get("expected_label"),
                "predicted_label": mismatch.get("predicted_label"),
                "validator_action": mismatch.get("validator_action"),
                "candidate_feature_set": mismatch.get("candidate_feature_set"),
                "candidate_features": mismatch.get("candidate_features")
                if isinstance(mismatch.get("candidate_features"), Mapping)
                else {},
                "candidate_support": _num(mismatch.get("candidate_support")),
                "candidate_purity": mismatch.get("candidate_purity"),
            }
        )

    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping) or candidate.get("validator_action") != "allow_accept":
            continue
        items.append(
            {
                "priority_group": "p2_allow_accept_needs_stronger_evidence",
                "review_type": "allow_accept_candidate",
                "review_reason": "allow_accept_candidates_require_stronger_evidence",
                "feature_set": candidate.get("feature_set"),
                "features": candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {},
                "label": candidate.get("label"),
                "candidate_action": candidate.get("validator_action"),
                "support": _num(candidate.get("support")),
                "purity": candidate.get("purity"),
                "expected_label_counts": candidate.get("expected_label_counts")
                if isinstance(candidate.get("expected_label_counts"), Mapping)
                else {},
                "promotion_reason": candidate.get("promotion_reason"),
                "automation_policy": candidate.get("automation_policy"),
            }
        )

    for conflict in conflict_rows[:12]:
        if not isinstance(conflict, Mapping):
            continue
        items.append(
            {
                "priority_group": "p3_high_needed_label_conflict_bucket",
                "review_type": "gold_support_conflict_bucket",
                "review_reason": "needs_more_gold_support_or_rule_split",
                "feature_set": conflict.get("feature_set"),
                "features": conflict.get("features") if isinstance(conflict.get("features"), Mapping) else {},
                "support": _num(conflict.get("support")),
                "dominant_label": conflict.get("dominant_label"),
                "purity": conflict.get("purity"),
                "expected_label_counts": conflict.get("expected_label_counts")
                if isinstance(conflict.get("expected_label_counts"), Mapping)
                else {},
                "needed_additional_labels": _num(conflict.get("needed_additional_labels")),
                "reason": conflict.get("reason"),
                "automation_policy": conflict.get("automation_policy"),
            }
        )

    for index, item in enumerate(items, start=1):
        item["priority_rank"] = index
    reason_counts = Counter(str(item.get("review_type")) for item in items)
    return {
        "schema_version": "prefix-static-rule-validator-review-plan-v1",
        "prompt_safe_summary": True,
        "review_item_count": len(items),
        "blocking_review_item_count": sum(
            1
            for item in items
            if item.get("review_type")
            in {"conflicting_overlap_pair", "fail_closed_simulation_mismatch", "allow_accept_candidate"}
        ),
        "priority_reason_counts": dict(sorted(reason_counts.items())),
        "recommended_next_review_type": str(items[0].get("review_type")) if items else None,
        "review_items": items[:20],
        "automation_policy": "manual_review_only_before_validator_code_changes",
        "recommendation": _validator_review_plan_recommendation(items),
        "limits": (
            "This plan is prompt-safe metadata only. It prioritizes manual Validator review work and does not "
            "promote rules, call a model/provider, or prove provider cache, latency, cost, or task-success effects."
        ),
    }


def _validator_review_plan_recommendation(items: Sequence[Mapping[str, Any]]) -> str:
    review_types = [str(item.get("review_type") or "") for item in items]
    if "conflicting_overlap_pair" in review_types:
        return "resolve_conflicting_candidate_overlaps_first"
    if "fail_closed_simulation_mismatch" in review_types:
        return "inspect_fail_closed_mismatches_before_validator_changes"
    if "allow_accept_candidate" in review_types:
        return "require_more_gold_support_or_real_ab_before_allow_accept_rules"
    if "gold_support_conflict_bucket" in review_types:
        return "add_gold_labels_or_split_high_conflict_buckets"
    return "no_validator_review_items_from_current_metadata"


def _conservative_fail_closed_subset(
    *,
    rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    overlap_diagnostics: Mapping[str, Any],
    fail_closed_simulation: Mapping[str, Any],
) -> dict[str, Any]:
    exclusion_reasons_by_key: dict[tuple[str, str, tuple[tuple[str, str], ...]], set[str]] = defaultdict(set)
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            continue
        key = _candidate_rule_key(candidate)
        if candidate.get("validator_action") == "allow_accept":
            exclusion_reasons_by_key[key].add("allow_accept_requires_stronger_evidence")

    overlap_pairs = (
        overlap_diagnostics.get("conflicting_overlap_pairs")
        if isinstance(overlap_diagnostics.get("conflicting_overlap_pairs"), list)
        else []
    )
    for pair in overlap_pairs:
        if not isinstance(pair, Mapping):
            continue
        left_key = _candidate_rule_key(
            {
                "feature_set": pair.get("left_feature_set"),
                "features": pair.get("left_features") if isinstance(pair.get("left_features"), Mapping) else {},
                "validator_action": pair.get("left_action"),
            }
        )
        right_key = _candidate_rule_key(
            {
                "feature_set": pair.get("right_feature_set"),
                "features": pair.get("right_features") if isinstance(pair.get("right_features"), Mapping) else {},
                "validator_action": pair.get("right_action"),
            }
        )
        exclusion_reasons_by_key[left_key].add("conflicting_overlap_pair")
        exclusion_reasons_by_key[right_key].add("conflicting_overlap_pair")

    mismatches = (
        fail_closed_simulation.get("sample_mismatches")
        if isinstance(fail_closed_simulation.get("sample_mismatches"), list)
        else []
    )
    for mismatch in mismatches:
        if not isinstance(mismatch, Mapping):
            continue
        key = _candidate_rule_key(
            {
                "feature_set": mismatch.get("candidate_feature_set"),
                "features": mismatch.get("candidate_features")
                if isinstance(mismatch.get("candidate_features"), Mapping)
                else {},
                "validator_action": mismatch.get("validator_action"),
            }
        )
        exclusion_reasons_by_key[key].add("fail_closed_simulation_mismatch")

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            continue
        action = str(candidate.get("validator_action") or "")
        if action not in {"force_reject", "prefer_review"}:
            continue
        key = _candidate_rule_key(candidate)
        reasons = sorted(exclusion_reasons_by_key.get(key, set()))
        annotated = {
            **_validator_candidate_copy(candidate),
            "subset_policy": "conservative_fail_closed_only",
            "exclusion_reasons": reasons,
            "subset_decision": "excluded" if reasons else "included",
        }
        if reasons:
            excluded.append(annotated)
        else:
            included.append(annotated)

    simulation = _validator_fail_closed_simulation(rows=rows, candidates=included)
    exclusion_reason_counts = Counter(
        reason
        for candidate in excluded
        for reason in candidate.get("exclusion_reasons", [])
    )
    return {
        "schema_version": "prefix-static-rule-conservative-fail-closed-subset-v1",
        "prompt_safe_summary": True,
        "source_candidate_rule_count": len(candidate_rows),
        "source_fail_closed_rule_count": sum(
            1 for candidate in candidate_rows if candidate.get("validator_action") in {"force_reject", "prefer_review"}
        ),
        "included_rule_count": len(included),
        "excluded_rule_count": len(excluded),
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "included_candidate_rules": included,
        "excluded_candidate_rules": excluded,
        "simulation": simulation,
        "manual_trial_ready": len(included) > 0 and _num(simulation.get("mismatch_count")) == 0,
        "safe_for_automatic_validator_promotion": False,
        "automation_policy": "manual_trial_only_until_code_review_and_real_ab",
        "recommendation": _conservative_fail_closed_subset_recommendation(
            included_count=len(included),
            mismatch_count=_num(simulation.get("mismatch_count")),
        ),
        "limits": (
            "This is a conservative metadata-only subset for manual review. It excludes allow-accept, conflicting "
            "overlap, and observed mismatch rules, but still does not change Validator behavior or prove real "
            "provider metrics."
        ),
    }


def _candidate_rule_key(candidate: Mapping[str, Any]) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
    return (
        str(candidate.get("feature_set") or ""),
        str(candidate.get("validator_action") or ""),
        tuple(sorted((str(key), str(value)) for key, value in features.items())),
    )


def _validator_candidate_copy(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_set": candidate.get("feature_set"),
        "features": candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {},
        "label": candidate.get("label"),
        "validator_action": candidate.get("validator_action"),
        "support": _num(candidate.get("support")),
        "purity": candidate.get("purity"),
        "expected_label_counts": candidate.get("expected_label_counts")
        if isinstance(candidate.get("expected_label_counts"), Mapping)
        else {},
        "promotion_reason": candidate.get("promotion_reason"),
        "automation_policy": candidate.get("automation_policy"),
    }


def _conservative_fail_closed_subset_recommendation(*, included_count: int, mismatch_count: int) -> str:
    if included_count <= 0:
        return "no_conservative_fail_closed_rules_after_exclusions"
    if mismatch_count > 0:
        return "inspect_remaining_fail_closed_mismatches_before_manual_trial"
    return "manual_review_conservative_fail_closed_subset_before_real_ab"


def _validator_shadow_trial_plan(
    *,
    rows: Sequence[Mapping[str, Any]],
    conservative_subset: Mapping[str, Any],
) -> dict[str, Any]:
    included = (
        conservative_subset.get("included_candidate_rules")
        if isinstance(conservative_subset.get("included_candidate_rules"), list)
        else []
    )
    simulation = (
        conservative_subset.get("simulation")
        if isinstance(conservative_subset.get("simulation"), Mapping)
        else {}
    )
    trial_rules = [
        _shadow_trial_rule_row(candidate, rows=rows)
        for candidate in included
        if isinstance(candidate, Mapping)
    ]
    action_counts = Counter(str(item.get("shadow_action") or "unknown") for item in trial_rules)
    shadow_ready = (
        len(trial_rules) > 0
        and _num(simulation.get("mismatch_count")) == 0
        and conservative_subset.get("manual_trial_ready") is True
    )
    source_diverse_count = sum(
        1
        for item in trial_rules
        if _num(item.get("source_group_count")) >= 2 and _num(item.get("source_file_hash_count")) >= 2
    )
    source_conflict_count = sum(
        1
        for item in trial_rules
        if item.get("source_label_conflict") is True or item.get("expected_label_conflict") is True
    )
    weak_source_support_count = sum(
        1
        for item in trial_rules
        if _num(item.get("source_group_count")) < 2 or _num(item.get("source_file_hash_count")) < 2
    )
    return {
        "schema_version": "prefix-static-rule-shadow-trial-plan-v1",
        "prompt_safe_summary": True,
        "trial_mode": "shadow_only",
        "shadow_trial_ready": shadow_ready,
        "trial_rule_count": len(trial_rules),
        "trial_action_counts": dict(sorted(action_counts.items())),
        "offline_simulation_mismatch_count": _num(simulation.get("mismatch_count")),
        "offline_simulation_matched_count": _num(simulation.get("matched_count")),
        "offline_simulation_matched_rate": simulation.get("matched_rate"),
        "source_diverse_rule_count": source_diverse_count,
        "weak_source_support_rule_count": weak_source_support_count,
        "source_conflict_rule_count": source_conflict_count,
        "trial_rules": trial_rules,
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
        "recommendation": _shadow_trial_plan_recommendation(
            shadow_ready=shadow_ready,
            source_conflict_count=source_conflict_count,
            weak_source_support_count=weak_source_support_count,
        ),
        "limits": (
            "This plan is a prompt-safe manual shadow-trial checklist. It must not change Validator decisions "
            "until code review and real provider A/B telemetry confirm behavior and utility."
        ),
    }


def _shadow_trial_rule_row(candidate: Mapping[str, Any], *, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_summary = _candidate_source_generalization_row(candidate, rows=rows)
    rule_id_payload = json.dumps(
        {
            "feature_set": candidate.get("feature_set"),
            "features": candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {},
            "shadow_action": candidate.get("validator_action"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    source_label_conflict = source_summary.get("source_label_conflict") is True
    expected_label_conflict = source_summary.get("expected_label_conflict") is True
    weak_source_support = (
        _num(source_summary.get("source_group_count")) < 2
        or _num(source_summary.get("source_file_hash_count")) < 2
    )
    return {
        "trial_rule_id": f"shadow_rule:{_short_hash(rule_id_payload)}",
        "feature_set": candidate.get("feature_set"),
        "features": candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {},
        "shadow_action": candidate.get("validator_action"),
        "support": _num(candidate.get("support")),
        "purity": candidate.get("purity"),
        "expected_label_counts": candidate.get("expected_label_counts")
        if isinstance(candidate.get("expected_label_counts"), Mapping)
        else {},
        "matched_count": _num(source_summary.get("matched_count")),
        "source_group_count": _num(source_summary.get("source_group_count")),
        "source_file_hash_count": _num(source_summary.get("source_file_hash_count")),
        "extraction_counts": source_summary.get("extraction_counts")
        if isinstance(source_summary.get("extraction_counts"), Mapping)
        else {},
        "expected_label_counts_by_source": source_summary.get("expected_label_counts_by_source")
        if isinstance(source_summary.get("expected_label_counts_by_source"), Mapping)
        else {},
        "source_label_counts_by_source": source_summary.get("source_label_counts_by_source")
        if isinstance(source_summary.get("source_label_counts_by_source"), Mapping)
        else {},
        "source_label_conflict": source_label_conflict,
        "expected_label_conflict": expected_label_conflict,
        "weak_source_support": weak_source_support,
        "review_status": _shadow_trial_rule_review_status(
            source_label_conflict=source_label_conflict,
            expected_label_conflict=expected_label_conflict,
            weak_source_support=weak_source_support,
        ),
    }


def _shadow_trial_rule_review_status(
    *,
    source_label_conflict: bool,
    expected_label_conflict: bool,
    weak_source_support: bool,
) -> str:
    if expected_label_conflict:
        return "inspect_expected_label_conflict"
    if source_label_conflict:
        return "inspect_source_label_conflict"
    if weak_source_support:
        return "needs_more_source_diversity_before_promotion"
    return "ready_for_shadow_trial"


def _shadow_trial_plan_recommendation(
    *,
    shadow_ready: bool,
    source_conflict_count: int,
    weak_source_support_count: int,
) -> str:
    if not shadow_ready:
        return "resolve_conservative_subset_blockers_before_shadow_trial"
    if source_conflict_count > 0:
        return "run_shadow_trial_and_inspect_source_conflicts_before_validator_changes"
    if weak_source_support_count > 0:
        return "run_shadow_trial_but_collect_more_source_diverse_gold_labels_before_promotion"
    return "run_shadow_trial_before_manual_validator_code_review"


def _validator_source_generalization(
    *,
    rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    conservative_subset: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_summaries = [
        _candidate_source_generalization_row(candidate, rows=rows)
        for candidate in candidate_rows
        if isinstance(candidate, Mapping)
    ]
    candidate_summaries = sorted(
        candidate_summaries,
        key=lambda item: (
            item.get("validator_action") in {"force_reject", "prefer_review"},
            _num(item.get("source_group_count")),
            _num(item.get("source_file_hash_count")),
            _num(item.get("matched_count")),
            _score(item.get("purity")),
            str(item.get("feature_set")),
            str(item.get("features")),
        ),
        reverse=True,
    )
    included = (
        conservative_subset.get("included_candidate_rules")
        if isinstance(conservative_subset.get("included_candidate_rules"), list)
        else []
    )
    included_keys = {_candidate_rule_key(candidate) for candidate in included if isinstance(candidate, Mapping)}
    included_summaries = [
        item
        for item in candidate_summaries
        if _candidate_rule_key(item) in included_keys
    ]
    weak_items = [
        item
        for item in candidate_summaries
        if _num(item.get("source_group_count")) < 2 or _num(item.get("source_file_hash_count")) < 2
    ]
    conflict_items = [
        item
        for item in candidate_summaries
        if item.get("source_label_conflict") is True or item.get("expected_label_conflict") is True
    ]
    return {
        "schema_version": "prefix-static-rule-source-generalization-diagnostics-v1",
        "prompt_safe_summary": True,
        "candidate_rule_count": len(candidate_summaries),
        "candidate_with_multi_source_group_count": sum(
            1 for item in candidate_summaries if _num(item.get("source_group_count")) >= 2
        ),
        "candidate_with_multi_source_file_count": sum(
            1 for item in candidate_summaries if _num(item.get("source_file_hash_count")) >= 2
        ),
        "conservative_subset_rule_count": len(included_summaries),
        "conservative_subset_multi_source_group_count": sum(
            1 for item in included_summaries if _num(item.get("source_group_count")) >= 2
        ),
        "conservative_subset_multi_source_file_count": sum(
            1 for item in included_summaries if _num(item.get("source_file_hash_count")) >= 2
        ),
        "weak_source_support_count": len(weak_items),
        "source_conflict_count": len(conflict_items),
        "top_candidate_source_summaries": candidate_summaries[:12],
        "weak_source_support_examples": weak_items[:12],
        "source_conflict_examples": conflict_items[:12],
        "generalization_claim_allowed": False,
        "automation_policy": "diagnostic_only_until_broader_cross_framework_gold_eval",
        "recommendation": _source_generalization_recommendation(
            included_summaries=included_summaries,
            weak_count=len(weak_items),
            conflict_count=len(conflict_items),
        ),
        "limits": (
            "This is source-distribution metadata only. It uses source group and hashed source path counts, "
            "does not include prompt text, and cannot prove broad cross-framework generalization."
        ),
    }


def _candidate_source_generalization_row(candidate: Mapping[str, Any], *, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
    matched = [row for row in rows if _candidate_matches_row(row, features)]
    source_groups = Counter(str(row.get("source_group") or "unknown") for row in matched)
    source_hashes = Counter(str(row.get("source_path_hash") or "unknown") for row in matched)
    extraction_counts = Counter(str(row.get("extraction") or "unknown") for row in matched)
    expected_counts = Counter(str(row.get("expected_label") or "unknown") for row in matched)
    source_label_counts = Counter(str(row.get("source_label") or "unknown") for row in matched)
    return {
        **_validator_candidate_copy(candidate),
        "matched_count": len(matched),
        "source_group_count": len(source_groups),
        "source_file_hash_count": len(source_hashes),
        "source_group_counts": dict(sorted(source_groups.items())),
        "sample_source_path_hashes": sorted(source_hashes)[:8],
        "extraction_counts": dict(sorted(extraction_counts.items())),
        "expected_label_counts_by_source": dict(sorted(expected_counts.items())),
        "source_label_counts_by_source": dict(sorted(source_label_counts.items())),
        "expected_label_conflict": sum(1 for count in expected_counts.values() if count > 0) > 1,
        "source_label_conflict": sum(1 for count in source_label_counts.values() if count > 0) > 1,
    }


def _source_generalization_recommendation(
    *,
    included_summaries: Sequence[Mapping[str, Any]],
    weak_count: int,
    conflict_count: int,
) -> str:
    if not included_summaries:
        return "collect_conservative_fail_closed_candidates_before_source_generalization_review"
    if conflict_count > 0:
        return "inspect_source_conflicts_before_validator_rule_changes"
    if any(_num(item.get("source_group_count")) >= 2 for item in included_summaries):
        return "manual_review_multi_source_conservative_rules_before_real_ab"
    if weak_count > 0:
        return "collect_more_source_diverse_gold_labels_before_validator_rule_changes"
    return "manual_review_source_distribution_before_real_ab"


def _validator_candidate_overlap_diagnostics(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            relation = _candidate_overlap_relation(left, right)
            if relation is None:
                continue
            left_action = str(left.get("validator_action") or "")
            right_action = str(right.get("validator_action") or "")
            pairs.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "relation": relation,
                    "conflicting_actions": left_action != right_action,
                    "left_action": left_action,
                    "right_action": right_action,
                    "left_feature_set": left.get("feature_set"),
                    "right_feature_set": right.get("feature_set"),
                    "left_features": left.get("features") if isinstance(left.get("features"), Mapping) else {},
                    "right_features": right.get("features") if isinstance(right.get("features"), Mapping) else {},
                    "left_support": _num(left.get("support")),
                    "right_support": _num(right.get("support")),
                    "left_purity": left.get("purity"),
                    "right_purity": right.get("purity"),
                }
            )
    conflicting = [pair for pair in pairs if pair.get("conflicting_actions") is True]
    return {
        "schema_version": "prefix-static-rule-candidate-overlap-diagnostics-v1",
        "prompt_safe_summary": True,
        "overlap_pair_count": len(pairs),
        "conflicting_overlap_pair_count": len(conflicting),
        "same_action_overlap_pair_count": len(pairs) - len(conflicting),
        "conflicting_overlap_pairs": conflicting[:12],
        "sample_overlap_pairs": pairs[:12],
    }


def _candidate_overlap_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str | None:
    left_features = left.get("features") if isinstance(left.get("features"), Mapping) else {}
    right_features = right.get("features") if isinstance(right.get("features"), Mapping) else {}
    if not left_features or not right_features:
        return None
    if left_features == right_features:
        return "same_features"
    if _feature_subset(left_features, right_features):
        return "left_subset_of_right"
    if _feature_subset(right_features, left_features):
        return "right_subset_of_left"
    return None


def _feature_subset(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if len(left) >= len(right):
        return False
    for key, value in left.items():
        if key not in right or str(right.get(key)) != str(value):
            return False
    return True


def _validator_candidate_promotion_gate(
    *,
    candidate_count: int,
    safe_accept_count: int,
    conflict_count: int,
    overlap_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    blocking_reasons = ["manual_code_review_required", "real_provider_ab_required"]
    if candidate_count <= 0:
        blocking_reasons.append("no_candidate_rules")
    if safe_accept_count > 0:
        blocking_reasons.append("allow_accept_candidates_require_stronger_evidence")
    if conflict_count > 0:
        blocking_reasons.append("gold_support_conflict_buckets_remain")
    if _num(overlap_diagnostics.get("conflicting_overlap_pair_count")) > 0:
        blocking_reasons.append("overlapping_candidate_rules_have_conflicting_actions")
    return {
        "schema_version": "prefix-static-rule-validator-promotion-gate-v1",
        "prompt_safe_summary": True,
        "automatic_promotion_ready": False,
        "manual_review_ready": candidate_count > 0,
        "fail_closed_candidate_count": max(0, candidate_count - safe_accept_count),
        "safe_accept_candidate_count": safe_accept_count,
        "blocking_reasons": blocking_reasons,
        "recommendation": _validator_promotion_gate_recommendation(blocking_reasons),
        "limits": (
            "This gate intentionally prevents automatic Validator promotion from metadata-only diagnostics. "
            "Candidate rules must be reviewed and then validated in real provider A/B before production claims."
        ),
    }


def _validator_fail_closed_simulation(
    *,
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fail_closed_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("validator_action") in {"force_reject", "prefer_review"}
    ]
    fail_closed_candidates = sorted(
        fail_closed_candidates,
        key=lambda candidate: (
            len(candidate.get("features")) if isinstance(candidate.get("features"), Mapping) else 0,
            _score(candidate.get("purity")),
            _num(candidate.get("support")),
            str(candidate.get("validator_action")),
        ),
        reverse=True,
    )
    predictions: list[str] = []
    matched_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for row in rows:
        match = _first_matching_candidate(row, fail_closed_candidates)
        if match is None:
            predictions.append(_fallback_label(row, fallback_policy="rule_label"))
            continue
        action = str(match.get("validator_action") or "")
        predicted = "reject" if action == "force_reject" else "review"
        predictions.append(predicted)
        action_counts[action] += 1
        matched_rows.append(
            {
                "input_index": _num(row.get("input_index")),
                "expected_label": row.get("expected_label"),
                "predicted_label": predicted,
                "validator_action": action,
                "candidate_feature_set": match.get("feature_set"),
                "candidate_features": match.get("features") if isinstance(match.get("features"), Mapping) else {},
                "candidate_support": _num(match.get("support")),
                "candidate_purity": match.get("purity"),
                "matched_expected": row.get("expected_label") == predicted,
            }
        )
    metric = _metric_summary(
        rows,
        predictions=predictions,
        policy_name="validator_fail_closed_candidates:fallback=rule_label",
        min_accuracy=1.0,
        min_macro_f1=1.0,
    )
    matched_count = len(matched_rows)
    mismatch_rows = [row for row in matched_rows if row.get("matched_expected") is not True]
    return {
        "schema_version": "prefix-static-rule-validator-fail-closed-simulation-v1",
        "prompt_safe_summary": True,
        "policy_name": "validator_fail_closed_candidates:fallback=rule_label",
        "evaluated_count": len(rows),
        "candidate_rule_count": len(fail_closed_candidates),
        "matched_count": matched_count,
        "matched_rate": _ratio_or_none(matched_count, len(rows)),
        "action_counts": dict(sorted(action_counts.items())),
        "accuracy": metric.get("accuracy"),
        "macro_f1": metric.get("macro_f1"),
        "predicted_label_counts": metric.get("predicted_label_counts"),
        "confusion_matrix": metric.get("confusion_matrix"),
        "mismatch_count": len(mismatch_rows),
        "mismatch_rate": _ratio_or_none(len(mismatch_rows), matched_count),
        "sample_mismatches": mismatch_rows[:12],
        "safe_for_automatic_validator_promotion": False,
        "automation_policy": "diagnostic_only_until_manual_review_and_real_ab",
        "recommendation": _validator_fail_closed_simulation_recommendation(
            matched_count=matched_count,
            mismatch_count=len(mismatch_rows),
        ),
        "limits": (
            "This is an offline metadata-only simulation. It does not change Validator behavior, does not call "
            "a model/provider, and does not prove provider cache, latency, cost, or task-success effects."
        ),
    }


def _first_matching_candidate(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
        if _candidate_matches_row(row, features):
            return candidate
    return None


def _candidate_matches_row(row: Mapping[str, Any], features: Mapping[str, Any]) -> bool:
    for key, value in features.items():
        if str(row.get(str(key)) or "missing") != str(value):
            return False
    return True


def _validator_fail_closed_simulation_recommendation(
    *,
    matched_count: int,
    mismatch_count: int,
) -> str:
    if matched_count <= 0:
        return "no_fail_closed_candidate_coverage"
    if mismatch_count > 0:
        return "inspect_fail_closed_candidate_mismatches_before_validator_changes"
    return "fail_closed_candidates_match_gold_labels_but_still_require_real_ab"


def _validator_promotion_gate_recommendation(blocking_reasons: Sequence[str]) -> str:
    if "overlapping_candidate_rules_have_conflicting_actions" in blocking_reasons:
        return "resolve_overlapping_candidate_actions_before_validator_changes"
    if "allow_accept_candidates_require_stronger_evidence" in blocking_reasons:
        return "promote_only_fail_closed_candidates_until_more_evidence"
    if "gold_support_conflict_buckets_remain" in blocking_reasons:
        return "prioritize_conflict_buckets_for_more_labels_or_rule_splits"
    if "no_candidate_rules" in blocking_reasons:
        return "collect_more_gold_labels_before_validator_rule_changes"
    return "manual_review_and_real_ab_required_before_validator_changes"


def _validator_candidate_row(bucket: Mapping[str, Any]) -> dict[str, Any]:
    label = bucket.get("dominant_label")
    action_by_label = {
        "accept": "allow_accept",
        "reject": "force_reject",
        "review": "prefer_review",
    }
    return {
        "feature_set": bucket.get("feature_set"),
        "features": bucket.get("features") if isinstance(bucket.get("features"), Mapping) else {},
        "label": label,
        "validator_action": action_by_label.get(str(label), "prefer_review"),
        "support": _num(bucket.get("support")),
        "purity": bucket.get("purity"),
        "expected_label_counts": bucket.get("expected_label_counts")
        if isinstance(bucket.get("expected_label_counts"), Mapping)
        else {},
        "promotion_reason": "meets_support_and_purity_metadata_thresholds",
        "automation_policy": "review_before_validator_promotion",
    }


def _validator_conflict_row(bucket: Mapping[str, Any]) -> dict[str, Any]:
    reason = "needs_more_gold_support"
    if bucket.get("tie_for_dominant_label") is True:
        reason = "dominant_label_tie"
    elif bucket.get("purity_ready") is not True:
        reason = "purity_below_threshold"
    elif bucket.get("support_ready") is not True:
        reason = "support_below_threshold"
    return {
        "feature_set": bucket.get("feature_set"),
        "features": bucket.get("features") if isinstance(bucket.get("features"), Mapping) else {},
        "support": _num(bucket.get("support")),
        "dominant_label": bucket.get("dominant_label"),
        "purity": bucket.get("purity"),
        "expected_label_counts": bucket.get("expected_label_counts")
        if isinstance(bucket.get("expected_label_counts"), Mapping)
        else {},
        "needed_additional_labels": _num(bucket.get("needed_additional_labels")),
        "reason": reason,
        "automation_policy": "manual_review_or_more_gold_labels",
    }


def _validator_candidate_recommendation(
    *,
    candidate_count: int,
    conflict_count: int,
    safe_accept_count: int,
) -> str:
    if candidate_count <= 0:
        return "keep_validator_rules_static_and_collect_more_gold_labels"
    if safe_accept_count <= 0:
        return "use_candidates_for_fail_closed_reject_or_review_only"
    if conflict_count > candidate_count:
        return "review_candidates_and_prioritize_conflict_buckets_before_validator_promotion"
    return "review_candidate_metadata_rules_before_validator_promotion"


def _feature_bucket_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    production_min_support: int,
    production_min_purity: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        expected = row.get("expected_label")
        if expected in ALLOWED_LABELS:
            grouped[_feature_key(row, feature_names)][str(expected)] += 1
    result: list[dict[str, Any]] = []
    for key, counts in grouped.items():
        support = sum(counts.values())
        ranked = counts.most_common()
        if ranked:
            dominant_label, dominant_count = ranked[0]
            tie_for_dominant = len(ranked) > 1 and ranked[1][1] == dominant_count
        else:
            dominant_label = None
            dominant_count = 0
            tie_for_dominant = False
        purity = dominant_count / support if support else 0.0
        needed_for_support = max(0, production_min_support - support)
        needed_for_purity = _additional_labels_needed_for_purity(
            support=support,
            dominant_count=dominant_count,
            min_purity=production_min_purity,
        )
        needed = max(needed_for_support, needed_for_purity)
        support_ready = support >= production_min_support
        purity_ready = bool(not tie_for_dominant and purity >= production_min_purity)
        result.append(
            {
                "feature_set": "+".join(feature_names),
                "features": dict(zip(feature_names, key)),
                "support": support,
                "expected_label_counts": dict(sorted(counts.items())),
                "dominant_label": dominant_label,
                "purity": purity,
                "support_ready": support_ready,
                "purity_ready": purity_ready,
                "production_ready": support_ready and purity_ready,
                "tie_for_dominant_label": tie_for_dominant,
                "needed_additional_labels": needed,
                "needed_for_support": needed_for_support,
                "needed_for_purity": needed_for_purity,
                "purity_gap": max(0.0, production_min_purity - purity),
            }
        )
    return result


def _additional_labels_needed_for_purity(*, support: int, dominant_count: int, min_purity: float) -> int:
    if support <= 0:
        return 1
    if min_purity <= 0:
        return 0
    if dominant_count / support >= min_purity:
        return 0
    if min_purity >= 1.0:
        return max(0, support - dominant_count)
    needed = math.ceil((min_purity * support - dominant_count) / (1 - min_purity))
    return max(0, needed)


def _production_readiness_recommendation(
    *,
    best_ready: str | None,
    best_production_candidate: str | None,
) -> str:
    if best_production_candidate:
        return "review_production_candidate_rules_before_validator_promotion"
    if best_ready:
        return "collect_more_gold_labels_or_raise_support_for_exploratory_metadata_rules"
    return "metadata_rules_need_rework_before_more_gold_support"


def _gold_support_recommendation(
    buckets: Sequence[Mapping[str, Any]],
    *,
    production_min_support: int,
    production_min_purity: float,
) -> str:
    if not buckets:
        return "build_goldset_before_static_rule_calibration"
    if any(bucket.get("production_ready") is True for bucket in buckets):
        return "review_production_ready_buckets_with_cross_validation_before_validator_promotion"
    if any(_num(bucket.get("needed_additional_labels")) > 0 for bucket in buckets):
        return (
            "add_gold_labels_to_priority_buckets_until_support>="
            f"{production_min_support}_and_purity>={production_min_purity:g}"
        )
    return "review_bucket_conflicts_or_adjust_metadata_features"


def _best_policy(
    policies: Sequence[Mapping[str, Any]],
    *,
    ready_only: bool,
    production_only: bool,
) -> str | None:
    candidates = [
        policy
        for policy in policies
        if (not ready_only or policy.get("ready") is True)
        and (not production_only or policy.get("production_candidate") is True)
    ]
    if not candidates:
        return None
    best = sorted(
        candidates,
        key=lambda policy: (
            _score(policy.get("macro_f1")),
            _score(policy.get("accuracy")),
            _num(policy.get("calibration_coverage_count")),
            str(policy.get("policy_name")),
        ),
        reverse=True,
    )[0]
    return str(best.get("policy_name"))


def _learned_rule_rows(
    mapping: Mapping[tuple[str, ...], Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in sorted(mapping.items(), key=lambda item: (-_num(item[1].get("support")), item[0])):
        rows.append(
            {
                "features": dict(zip(feature_names, key)),
                "label": value.get("label"),
                "support": _num(value.get("support")),
                "purity": value.get("purity"),
                "expected_label_counts": value.get("expected_label_counts")
                if isinstance(value.get("expected_label_counts"), Mapping)
                else {},
            }
        )
    return rows


def _feature_key(row: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(name) or "missing") for name in feature_names)


def _fallback_label(row: Mapping[str, Any], *, fallback_policy: str) -> str:
    if fallback_policy == "review":
        return "review"
    value = row.get("rule_label") if fallback_policy == "rule_label" else row.get("source_label")
    return str(value) if value in ALLOWED_LABELS else "review"


def _recommendation(*, best_ready: str | None, best_production_candidate: str | None) -> str:
    if best_production_candidate:
        return "review_calibrated_metadata_rules_before_validator_promotion"
    if best_ready:
        return "metadata_signal_exists_but_needs_more_gold_support_before_validator_promotion"
    return "current_metadata_rules_do_not_pass_goldset_thresholds"


def _safe_label(value: Any) -> str | None:
    label = str(value or "").strip().lower()
    return label if label in ALLOWED_LABELS else None


def _safe_token(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _source_group(source_path: str) -> str:
    normalized = source_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return "unknown"
    if "framework-src" in parts:
        root_index = parts.index("framework-src")
        framework = parts[root_index + 1] if root_index + 1 < len(parts) else "unknown"
        package = None
        for marker in ("packages", "src", "lib"):
            if marker in parts:
                marker_index = parts.index(marker)
                if marker_index + 1 < len(parts):
                    package = parts[marker_index + 1]
                    break
        if package:
            return f"framework:{framework}/package:{package}"
        return f"framework:{framework}"
    return f"source_hash_group:{_short_hash('/'.join(parts[:-1]) or normalized)}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _macro_value(metrics: Mapping[str, Mapping[str, Any]], field: str) -> float | None:
    values = [value.get(field) for value in metrics.values() if isinstance(value.get(field), (int, float))]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _score(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else -1.0


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


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
