from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OfflinePipelineComparisonResult:
    baseline_path: str
    candidate_path: str
    summary_path: str | None
    report_path: str | None
    summary: dict[str, Any]


def compare_offline_pipeline_summaries(
    *,
    baseline_path: str | Path,
    candidate_path: str | Path,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> OfflinePipelineComparisonResult:
    baseline = _load_json_mapping(Path(baseline_path))
    candidate = _load_json_mapping(Path(candidate_path))
    baseline_metrics = _extract_metrics(baseline, label=baseline_label)
    candidate_metrics = _extract_metrics(candidate, label=candidate_label)
    delta = _delta_metrics(baseline_metrics, candidate_metrics)
    experiment_gate = _experiment_gate(baseline_metrics, candidate_metrics, delta)
    summary = {
        "schema_version": "prefix-offline-pipeline-comparison-v1",
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "experiment_gate": experiment_gate,
        "real_provider_metrics_available": False,
        "notes": (
            "This comparison uses offline aggregate pipeline summaries only. "
            "It does not include prompt text and cannot prove provider cache, latency, cost, or task success gains."
        ),
    }
    summary_target = _write_json(summary_path, summary)
    report_target = _write_text(report_path, render_markdown_report(summary))
    return OfflinePipelineComparisonResult(
        baseline_path=str(baseline_path),
        candidate_path=str(candidate_path),
        summary_path=str(summary_target) if summary_target else None,
        report_path=str(report_target) if report_target else None,
        summary=summary,
    )


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), Mapping) else {}
    candidate = summary.get("candidate") if isinstance(summary.get("candidate"), Mapping) else {}
    delta = summary.get("delta") if isinstance(summary.get("delta"), Mapping) else {}
    lines = [
        "# Offline Prefix Rule Comparison",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Supported requests", "supported_request_count"),
        ("Applied requests", "applied_count"),
        ("Applied rate", "applied_rate"),
        ("Reusable prefix requests", "reusable_prefix_request_count"),
        ("Rule gap observed", "rule_gap_observed"),
        ("Review queue observed", "review_queue_observed"),
        ("Accepted candidates", "accepted_candidate_count"),
        ("Review candidates", "review_candidate_count"),
        ("Accepted parent blocks", "accepted_candidate_parent_block_count"),
        ("Accepted estimated reusable chars", "accepted_estimated_reusable_chars"),
        ("Accepted or review estimated reusable chars", "accepted_or_review_estimated_reusable_chars"),
        ("Total estimated gain chars", "total_estimated_gain_chars"),
    ):
        lines.append(
            f"| {label} | {_format_value(baseline.get(key))} | "
            f"{_format_value(candidate.get(key))} | {_format_value(delta.get(key + '_delta'))} |"
        )
    gate = summary.get("experiment_gate") if isinstance(summary.get("experiment_gate"), Mapping) else {}
    lines.extend(
        [
            "",
            "## Candidate Label Diagnostics",
            "",
            "| Field | Baseline | Candidate | Delta |",
            "|---|---|---|---|",
        ]
    )
    for key in (
        "judge_mode",
        "local_judge_scope",
        "max_local_judge_calls",
        "local_judge_budget_strategy",
        "local_judge_budget_diagnostics",
        "local_judge_effectiveness_diagnostics",
        "candidate_promotion_policy",
        "candidate_review_upper_bound_policy",
        "label_counts",
        "local_judge_action_counts",
        "local_judge_skipped_by_scope_label_counts",
        "local_judge_skipped_by_budget_label_counts",
        "rule_label_counts",
        "model_label_counts",
        "rule_to_model_label_counts",
        "model_to_final_label_counts",
        "rule_to_final_label_counts",
        "static_safety_clamp_count",
    ):
        lines.append(
            f"| {key} | {_format_value(baseline.get(key))} | "
            f"{_format_value(candidate.get(key))} | {_format_value(delta.get(key + '_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Experiment Gate",
            "",
            f"Recommendation: **{_format_value(gate.get('recommendation'))}**",
            "",
            "| Gate | Status | Evidence |",
            "|---|---|---|",
        ]
    )
    for item in gate.get("items") if isinstance(gate.get("items"), list) else ():
        if isinstance(item, Mapping):
            lines.append(
                f"| {item.get('name')} | {item.get('status')} | {item.get('evidence')} |"
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This report is generated from aggregate offline summaries only.",
            "- It does not include prompt text or candidate text.",
            "- Real cached tokens, latency, cost, and task success require provider/benchmark A/B data.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two offline semantic pipeline summary JSON files.")
    parser.add_argument("--baseline", required=True, help="Baseline pipeline_summary.json.")
    parser.add_argument("--candidate", required=True, help="Candidate pipeline_summary.json.")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--summary", help="Optional comparison summary JSON output.")
    parser.add_argument("--report-md", help="Optional prompt-safe Markdown report output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_offline_pipeline_summaries(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        summary_path=args.summary,
        report_path=args.report_md,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _extract_metrics(summary: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    source = summary.get("source_prompt_corpus") if isinstance(summary.get("source_prompt_corpus"), Mapping) else {}
    dataset = summary.get("dataset_eval") if isinstance(summary.get("dataset_eval"), Mapping) else {}
    gap = (
        summary.get("semantic_rule_gap_summary")
        if isinstance(summary.get("semantic_rule_gap_summary"), Mapping)
        else {}
    )
    utility = summary.get("candidate_utility_eval") if isinstance(summary.get("candidate_utility_eval"), Mapping) else {}
    utility_review = (
        summary.get("candidate_utility_eval_with_review")
        if isinstance(summary.get("candidate_utility_eval_with_review"), Mapping)
        else {}
    )
    label_eval = (
        summary.get("candidate_label_eval")
        if isinstance(summary.get("candidate_label_eval"), Mapping)
        else {}
    )
    return {
        "label": label,
        "natural_language_segmentation_enabled": bool(summary.get("natural_language_segmentation_enabled")),
        "prompt_count": _num(source.get("prompt_count")),
        "source_file_count": _num(source.get("source_file_count")),
        "supported_request_count": _num(gap.get("supported_request_count") or dataset.get("supported_request_count")),
        "applied_count": _num(gap.get("applied_count") or dataset.get("applied_count")),
        "applied_rate": _ratio_or_existing(gap.get("applied_rate"), gap.get("applied_count"), gap.get("supported_request_count")),
        "reusable_prefix_request_count": _num(gap.get("reusable_prefix_request_count")),
        "rule_gap_observed": bool(gap.get("rule_gap_observed")),
        "review_queue_observed": bool(gap.get("review_queue_observed")),
        "candidate_count": _num(gap.get("candidate_count")),
        "accepted_candidate_count": _num(gap.get("accepted_candidate_count")),
        "review_candidate_count": _num(gap.get("review_candidate_count")),
        "accepted_candidate_parent_block_count": _num(gap.get("accepted_candidate_parent_block_count")),
        "accepted_or_review_candidate_parent_block_count": _num(
            gap.get("accepted_or_review_candidate_parent_block_count")
        ),
        "accepted_estimated_reusable_chars": _num(gap.get("accepted_estimated_reusable_chars")),
        "accepted_or_review_estimated_reusable_chars": _num(
            gap.get("accepted_or_review_estimated_reusable_chars")
        ),
        "accepted_repeated_group_count": _num(gap.get("accepted_repeated_group_count")),
        "eligible_candidate_chars": _num(utility.get("eligible_candidate_chars")),
        "eligible_with_review_candidate_chars": _num(utility_review.get("eligible_candidate_chars")),
        "total_estimated_gain_chars": _num(summary.get("dataset_eval", {}).get("total_estimated_gain_chars") if isinstance(summary.get("dataset_eval"), Mapping) else None),
        "candidate_promotion_policy": gap.get("candidate_promotion_policy")
        if isinstance(gap.get("candidate_promotion_policy"), Mapping)
        else utility.get("promotion_policy")
        if isinstance(utility.get("promotion_policy"), Mapping)
        else {},
        "candidate_review_upper_bound_policy": gap.get("candidate_review_upper_bound_policy")
        if isinstance(gap.get("candidate_review_upper_bound_policy"), Mapping)
        else utility_review.get("promotion_policy")
        if isinstance(utility_review.get("promotion_policy"), Mapping)
        else {},
        "judge_mode": str(label_eval.get("judge_mode") or "unknown"),
        "local_judge_scope": str(label_eval.get("local_judge_scope") or "unknown"),
        "max_local_judge_calls": label_eval.get("max_local_judge_calls"),
        "local_judge_budget_strategy": label_eval.get("local_judge_budget_strategy"),
        "local_judge_budget_diagnostics": label_eval.get("local_judge_budget_diagnostics")
        if isinstance(label_eval.get("local_judge_budget_diagnostics"), Mapping)
        else {},
        "local_judge_effectiveness_diagnostics": label_eval.get("local_judge_effectiveness_diagnostics")
        if isinstance(label_eval.get("local_judge_effectiveness_diagnostics"), Mapping)
        else {},
        "label_counts": _string_int_dict(label_eval.get("label_counts")),
        "local_judge_action_counts": _string_int_dict(label_eval.get("local_judge_action_counts")),
        "local_judge_skipped_by_scope_label_counts": _string_int_dict(
            label_eval.get("local_judge_skipped_by_scope_label_counts")
        ),
        "local_judge_skipped_by_budget_label_counts": _string_int_dict(
            label_eval.get("local_judge_skipped_by_budget_label_counts")
        ),
        "rule_label_counts": _string_int_dict(label_eval.get("rule_label_counts")),
        "model_label_counts": _string_int_dict(label_eval.get("model_label_counts")),
        "rule_to_model_label_counts": _string_int_dict(label_eval.get("rule_to_model_label_counts")),
        "model_to_final_label_counts": _string_int_dict(label_eval.get("model_to_final_label_counts")),
        "rule_to_final_label_counts": _string_int_dict(label_eval.get("rule_to_final_label_counts")),
        "static_safety_clamp_count": _num(label_eval.get("static_safety_clamp_count")),
    }


def _delta_metrics(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    numeric_keys = (
        "prompt_count",
        "source_file_count",
        "supported_request_count",
        "applied_count",
        "applied_rate",
        "reusable_prefix_request_count",
        "candidate_count",
        "accepted_candidate_count",
        "review_candidate_count",
        "accepted_candidate_parent_block_count",
        "accepted_or_review_candidate_parent_block_count",
        "accepted_estimated_reusable_chars",
        "accepted_or_review_estimated_reusable_chars",
        "accepted_repeated_group_count",
        "eligible_candidate_chars",
        "eligible_with_review_candidate_chars",
        "total_estimated_gain_chars",
    )
    delta: dict[str, Any] = {}
    for key in numeric_keys:
        delta[f"{key}_delta"] = _float(candidate.get(key)) - _float(baseline.get(key))
    delta["rule_gap_resolved"] = bool(baseline.get("rule_gap_observed")) and not bool(candidate.get("rule_gap_observed"))
    delta["review_queue_still_present"] = bool(candidate.get("review_queue_observed"))
    for key in (
        "label_counts",
        "local_judge_action_counts",
        "local_judge_skipped_by_scope_label_counts",
        "local_judge_skipped_by_budget_label_counts",
        "rule_label_counts",
        "model_label_counts",
        "rule_to_model_label_counts",
        "model_to_final_label_counts",
        "rule_to_final_label_counts",
    ):
        delta[f"{key}_delta"] = _dict_delta(baseline.get(key), candidate.get(key))
    delta["static_safety_clamp_count_delta"] = _num(candidate.get("static_safety_clamp_count")) - _num(
        baseline.get("static_safety_clamp_count")
    )
    delta["local_judge_effectiveness_diagnostics_delta"] = _effectiveness_delta(
        baseline.get("local_judge_effectiveness_diagnostics"),
        candidate.get("local_judge_effectiveness_diagnostics"),
    )
    return delta


def _experiment_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    items = [
        _same_supported_corpus_gate(baseline, candidate),
        _rewrite_improvement_gate(delta),
        _rule_gap_gate(baseline, candidate, delta),
        _review_queue_gate(candidate),
        _review_candidate_promotion_policy_gate(candidate),
        _local_judge_safety_gate(candidate),
        _local_judge_budget_gate(candidate),
        _local_judge_effectiveness_gate(candidate),
        {
            "name": "real_provider_metrics",
            "status": "unknown",
            "evidence": (
                "offline comparison only; run provider/AutoGenBench A/B for cached tokens, latency, cost, and task success"
            ),
        },
    ]
    return {
        "schema_version": "prefix-offline-experiment-gate-v1",
        "recommendation": _recommendation(items=items, baseline=baseline, candidate=candidate),
        "items": items,
        "prompt_safe_summary": True,
        "performance_conclusion_allowed": False,
        "performance_conclusion_reason": (
            "Offline aggregate evidence can justify a real A/B smoke, but cannot prove real cache, latency, cost, or task success gains."
        ),
    }


def _same_supported_corpus_gate(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_prompts = _num(baseline.get("prompt_count"))
    candidate_prompts = _num(candidate.get("prompt_count"))
    baseline_supported = _num(baseline.get("supported_request_count"))
    candidate_supported = _num(candidate.get("supported_request_count"))
    comparable = (
        baseline_prompts > 0
        and baseline_supported > 0
        and baseline_prompts == candidate_prompts
        and baseline_supported == candidate_supported
    )
    return {
        "name": "same_supported_corpus",
        "status": "pass" if comparable else "fail",
        "evidence": (
            f"baseline prompts={baseline_prompts}, supported={baseline_supported}; "
            f"candidate prompts={candidate_prompts}, supported={candidate_supported}"
        ),
    }


def _rewrite_improvement_gate(delta: Mapping[str, Any]) -> dict[str, Any]:
    applied_delta = _float(delta.get("applied_count_delta"))
    reusable_delta = _float(delta.get("reusable_prefix_request_count_delta"))
    gain_delta = _float(delta.get("total_estimated_gain_chars_delta"))
    improved = applied_delta > 0 or reusable_delta > 0 or gain_delta > 0
    return {
        "name": "candidate_increases_offline_rewrites",
        "status": "pass" if improved else "fail",
        "evidence": (
            f"applied_delta={_format_value(applied_delta)}, reusable_delta={_format_value(reusable_delta)}, "
            f"estimated_gain_chars_delta={_format_value(gain_delta)}"
        ),
    }


def _rule_gap_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_gap = bool(baseline.get("rule_gap_observed"))
    candidate_gap = bool(candidate.get("rule_gap_observed"))
    if baseline_gap and not candidate_gap:
        status = "pass"
        evidence = "baseline rule gap was observed and candidate summary no longer reports that gap"
    elif baseline_gap and candidate_gap:
        status = "warn"
        evidence = "rule gap remains after candidate changes; improve segmentation/classifier before large A/B"
    else:
        status = "unknown"
        evidence = (
            f"rule_gap_resolved={bool(delta.get('rule_gap_resolved'))}; "
            f"baseline_gap={baseline_gap}; candidate_gap={candidate_gap}"
        )
    return {"name": "rule_gap_resolution", "status": status, "evidence": evidence}


def _review_queue_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    review_count = _num(candidate.get("review_candidate_count"))
    if review_count == 0:
        status = "pass"
        evidence = "candidate has no review-labeled semantic candidates"
    else:
        status = "warn"
        evidence = f"candidate still has {review_count} review-labeled semantic candidates; keep them out of automatic promotion"
    return {"name": "review_queue", "status": status, "evidence": evidence}


def _review_candidate_promotion_policy_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    policy = candidate.get("candidate_promotion_policy")
    upper_bound_policy = candidate.get("candidate_review_upper_bound_policy")
    if not isinstance(policy, Mapping):
        return {
            "name": "review_candidate_promotion_policy",
            "status": "unknown",
            "evidence": "candidate promotion policy is missing from the offline summary",
        }
    allowed_labels = tuple(str(label) for label in policy.get("automatic_promotion_allowed_labels") or ())
    includes_review = bool(policy.get("automatic_promotion_includes_review"))
    if includes_review or "review" in allowed_labels:
        return {
            "name": "review_candidate_promotion_policy",
            "status": "fail",
            "evidence": (
                f"automatic promotion includes review candidates: "
                f"allowed_labels={allowed_labels}, automatic_promotion_includes_review={includes_review}"
            ),
        }
    review_count = _num(candidate.get("review_candidate_count"))
    if review_count > 0:
        upper_bound_only = (
            bool(upper_bound_policy.get("review_candidates_used_for_upper_bound_only"))
            if isinstance(upper_bound_policy, Mapping)
            else False
        )
        if not upper_bound_only:
            return {
                "name": "review_candidate_promotion_policy",
                "status": "warn",
                "evidence": (
                    f"candidate has {review_count} review candidates but no explicit upper-bound-only policy"
                ),
            }
        return {
            "name": "review_candidate_promotion_policy",
            "status": "pass",
            "evidence": (
                f"automatic promotion labels={allowed_labels}; {review_count} review candidates are upper-bound only"
            ),
        }
    return {
        "name": "review_candidate_promotion_policy",
        "status": "pass",
        "evidence": f"automatic promotion labels={allowed_labels}; no review candidates present",
    }


def _local_judge_safety_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    judge_mode = str(candidate.get("judge_mode") or "unknown")
    if judge_mode != "openai-compatible":
        return {
            "name": "local_judge_safety_clamps",
            "status": "unknown",
            "evidence": f"candidate judge_mode={judge_mode}; no local model judge diagnostics",
        }
    clamp_count = _num(candidate.get("static_safety_clamp_count"))
    model_to_final = candidate.get("model_to_final_label_counts")
    accept_to_review = (
        _num(model_to_final.get("accept->review"))
        if isinstance(model_to_final, Mapping)
        else 0
    )
    if clamp_count > 0 or accept_to_review > 0:
        return {
            "name": "local_judge_safety_clamps",
            "status": "warn",
            "evidence": (
                f"local judge accepted candidates were clamped: "
                f"static_safety_clamp_count={clamp_count}, accept_to_review={accept_to_review}"
            ),
        }
    return {
        "name": "local_judge_safety_clamps",
        "status": "pass",
        "evidence": "local judge diagnostics show no accept->review static safety clamps",
    }


def _local_judge_budget_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    judge_mode = str(candidate.get("judge_mode") or "unknown")
    diagnostics = candidate.get("local_judge_budget_diagnostics")
    if judge_mode != "openai-compatible" or not isinstance(diagnostics, Mapping):
        return {
            "name": "local_judge_budget_coverage",
            "status": "unknown",
            "evidence": f"candidate judge_mode={judge_mode}; no local judge budget diagnostics",
        }
    if not bool(diagnostics.get("budget_limited")):
        return {
            "name": "local_judge_budget_coverage",
            "status": "pass",
            "evidence": "local judge was not budget-limited",
        }
    model_called = _num(diagnostics.get("model_called_count"))
    skipped = _num(diagnostics.get("skipped_by_budget_count"))
    coverage = diagnostics.get("budget_coverage_rate")
    evidence = (
        f"model_called={model_called}, skipped_by_budget={skipped}, "
        f"coverage={_format_value(coverage)}"
    )
    if skipped > 0:
        return {
            "name": "local_judge_budget_coverage",
            "status": "warn",
            "evidence": evidence,
        }
    return {
        "name": "local_judge_budget_coverage",
        "status": "pass",
        "evidence": evidence,
    }


def _local_judge_effectiveness_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    judge_mode = str(candidate.get("judge_mode") or "unknown")
    diagnostics = candidate.get("local_judge_effectiveness_diagnostics")
    if judge_mode != "openai-compatible" or not isinstance(diagnostics, Mapping):
        return {
            "name": "local_judge_review_resolution",
            "status": "unknown",
            "evidence": f"candidate judge_mode={judge_mode}; no local judge effectiveness diagnostics",
        }
    rule_review = _num(diagnostics.get("rule_review_candidate_count"))
    called = _num(diagnostics.get("model_called_rule_review_count"))
    resolved = _num(diagnostics.get("resolved_rule_review_count"))
    called_resolved = _num(diagnostics.get("called_resolved_rule_review_count"))
    remaining = _num(diagnostics.get("remaining_rule_review_count"))
    resolution = diagnostics.get("resolution_rate")
    called_resolution = diagnostics.get("called_resolution_rate")
    evidence = (
        f"rule_review={rule_review}, model_called={called}, resolved={resolved}, "
        f"called_resolved={called_resolved}, remaining={remaining}, "
        f"resolution={_format_value(resolution)}, called_resolution={_format_value(called_resolution)}"
    )
    if rule_review == 0:
        status = "pass"
    elif called == 0:
        status = "unknown"
    elif called_resolved > 0:
        status = "pass" if remaining == 0 else "warn"
    else:
        status = "warn"
    return {"name": "local_judge_review_resolution", "status": status, "evidence": evidence}


def _recommendation(
    *,
    items: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    statuses = {str(item.get("name")): str(item.get("status")) for item in items}
    if statuses.get("same_supported_corpus") == "fail":
        return "fix_offline_inputs_before_ab"
    if statuses.get("review_candidate_promotion_policy") == "fail":
        return "fix_review_candidate_promotion_policy_before_ab"
    if statuses.get("candidate_increases_offline_rewrites") == "pass":
        if statuses.get("review_queue") == "warn":
            return "run_real_ab_smoke_with_review_candidates_disabled"
        return "run_real_ab_smoke"
    if bool(baseline.get("rule_gap_observed")) or bool(candidate.get("rule_gap_observed")):
        return "improve_semantic_rules_or_local_judge_before_ab"
    return "collect_more_real_cases_before_ab"


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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


def _ratio_or_existing(value: Any, numerator: Any, denominator: Any) -> float | None:
    if value is not None:
        return _float(value)
    denominator_float = _float(denominator)
    if denominator_float == 0:
        return None
    return _float(numerator) / denominator_float


def _num(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _string_int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _num(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _dict_delta(baseline: Any, candidate: Any) -> dict[str, int]:
    baseline_dict = _string_int_dict(baseline)
    candidate_dict = _string_int_dict(candidate)
    keys = sorted(set(baseline_dict) | set(candidate_dict))
    return {key: candidate_dict.get(key, 0) - baseline_dict.get(key, 0) for key in keys}


def _effectiveness_delta(baseline: Any, candidate: Any) -> dict[str, Any]:
    baseline_dict = baseline if isinstance(baseline, Mapping) else {}
    candidate_dict = candidate if isinstance(candidate, Mapping) else {}
    numeric_keys = (
        "source_with_effectiveness_diagnostics_count",
        "rule_review_candidate_count",
        "model_called_rule_review_count",
        "resolved_rule_review_count",
        "called_resolved_rule_review_count",
        "remaining_rule_review_count",
        "resolution_rate",
        "called_resolution_rate",
    )
    delta: dict[str, Any] = {}
    for key in numeric_keys:
        delta[f"{key}_delta"] = _float(candidate_dict.get(key)) - _float(baseline_dict.get(key))
    for key in (
        "rule_review_final_label_counts",
        "rule_review_model_label_counts",
        "rule_review_local_judge_action_counts",
    ):
        delta[f"{key}_delta"] = _dict_delta(baseline_dict.get(key), candidate_dict.get(key))
    return delta


if __name__ == "__main__":
    raise SystemExit(main())
