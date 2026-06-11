from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .offline_compare import compare_offline_pipeline_summaries
from .offline_semantic_pipeline import run_offline_semantic_pipeline


@dataclass(frozen=True)
class OfflineSemanticSuiteResult:
    output_dir: str
    summary_path: str
    summary: dict[str, Any]


def run_offline_semantic_suite(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    session_id: str = "offline-semantic-suite",
    include_tests: bool = False,
    include_candidate_text: bool = False,
    min_prompt_chars: int = 40,
    min_confidence: float = 0.66,
    judge: str = "rule",
    local_judge_base_url: str | None = None,
    local_judge_model: str | None = None,
    local_judge_api_key: str | None = None,
    local_judge_timeout: float = 30.0,
    local_judge_scope: str = "all",
    max_local_judge_calls: int | None = None,
) -> OfflineSemanticSuiteResult:
    out = Path(output_dir)
    reports = out / "reports"
    rule_dir = out / "rule_only"
    nl_dir = out / "nl_segmentation"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    rule = run_offline_semantic_pipeline(
        source_root=source_root,
        output_dir=rule_dir,
        session_id=f"{session_id}-rule-only",
        include_tests=include_tests,
        include_candidate_text=include_candidate_text,
        min_prompt_chars=min_prompt_chars,
        min_confidence=min_confidence,
        enable_natural_language_segmentation=False,
        judge=judge,
        local_judge_base_url=local_judge_base_url,
        local_judge_model=local_judge_model,
        local_judge_api_key=local_judge_api_key,
        local_judge_timeout=local_judge_timeout,
        local_judge_scope=local_judge_scope,
        max_local_judge_calls=max_local_judge_calls,
    )
    nl = run_offline_semantic_pipeline(
        source_root=source_root,
        output_dir=nl_dir,
        session_id=f"{session_id}-nl-segmentation",
        include_tests=include_tests,
        include_candidate_text=include_candidate_text,
        min_prompt_chars=min_prompt_chars,
        min_confidence=min_confidence,
        enable_natural_language_segmentation=True,
        judge=judge,
        local_judge_base_url=local_judge_base_url,
        local_judge_model=local_judge_model,
        local_judge_api_key=local_judge_api_key,
        local_judge_timeout=local_judge_timeout,
        local_judge_scope=local_judge_scope,
        max_local_judge_calls=max_local_judge_calls,
    )
    comparison = compare_offline_pipeline_summaries(
        baseline_path=rule.summary_path,
        candidate_path=nl.summary_path,
        baseline_label="rule-only",
        candidate_label="nl-segmentation",
        summary_path=reports / "offline_rule_vs_nl_summary.json",
        report_path=reports / "offline_rule_vs_nl_report.md",
    )

    summary = {
        "schema_version": "prefix-offline-semantic-suite-summary-v1",
        "prompt_safe_summary": True,
        "source_root": str(source_root),
        "output_dir": str(out),
        "session_id": session_id,
        "judge_mode": judge,
        "local_judge_scope": local_judge_scope,
        "max_local_judge_calls": max_local_judge_calls,
        "include_tests": include_tests,
        "include_candidate_text": include_candidate_text,
        "artifacts": {
            "rule_only_pipeline_summary": rule.summary_path,
            "nl_segmentation_pipeline_summary": nl.summary_path,
            "comparison_summary": comparison.summary_path,
            "comparison_report_md": comparison.report_path,
        },
        "prompt_text_artifacts": _prompt_text_artifacts(
            rule_summary=rule.summary,
            nl_summary=nl.summary,
            include_candidate_text=include_candidate_text,
        ),
        "rule_only": _pipeline_digest(rule.summary),
        "nl_segmentation": _pipeline_digest(nl.summary),
        "comparison": {
            "delta": comparison.summary.get("delta"),
            "experiment_gate": comparison.summary.get("experiment_gate"),
        },
        "recommendation": (comparison.summary.get("experiment_gate") or {}).get("recommendation")
        if isinstance(comparison.summary.get("experiment_gate"), Mapping)
        else None,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "Offline source-prompt coverage suite only. It can justify a small real A/B smoke, but cannot prove "
            "real provider cache, latency, cost, or task-success gains."
        ),
    }
    summary_path = out / "offline_semantic_suite_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return OfflineSemanticSuiteResult(output_dir=str(out), summary_path=str(summary_path), summary=summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run rule-only and natural-language-segmentation offline semantic pipelines, then compare them."
        )
    )
    parser.add_argument("--source-root", required=True, help="Local source tree to scan.")
    parser.add_argument("--output-dir", required=True, help="Directory for suite artifacts.")
    parser.add_argument("--session-id", default="offline-semantic-suite")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--include-candidate-text", action="store_true")
    parser.add_argument("--min-prompt-chars", type=int, default=40)
    parser.add_argument("--min-confidence", type=float, default=0.66)
    parser.add_argument("--judge", choices=("rule", "openai-compatible"), default="rule")
    parser.add_argument("--local-judge-base-url")
    parser.add_argument("--local-judge-model")
    parser.add_argument("--local-judge-api-key-env")
    parser.add_argument("--local-judge-timeout", type=float, default=30.0)
    parser.add_argument(
        "--local-judge-scope",
        choices=("all", "review"),
        default="all",
        help="For --judge openai-compatible, call the local judge for all candidates or only rule-review candidates.",
    )
    parser.add_argument(
        "--max-local-judge-calls",
        type=int,
        help="Optional cap on actual local model calls for --judge openai-compatible.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_offline_semantic_suite(
        source_root=args.source_root,
        output_dir=args.output_dir,
        session_id=args.session_id,
        include_tests=args.include_tests,
        include_candidate_text=args.include_candidate_text,
        min_prompt_chars=args.min_prompt_chars,
        min_confidence=args.min_confidence,
        judge=args.judge,
        local_judge_base_url=args.local_judge_base_url,
        local_judge_model=args.local_judge_model,
        local_judge_api_key=os.environ.get(args.local_judge_api_key_env) if args.local_judge_api_key_env else None,
        local_judge_timeout=args.local_judge_timeout,
        local_judge_scope=args.local_judge_scope,
        max_local_judge_calls=args.max_local_judge_calls,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _pipeline_digest(summary: Mapping[str, Any]) -> dict[str, Any]:
    source = summary.get("source_prompt_corpus") if isinstance(summary.get("source_prompt_corpus"), Mapping) else {}
    dataset = summary.get("dataset_eval") if isinstance(summary.get("dataset_eval"), Mapping) else {}
    gap = summary.get("semantic_rule_gap_summary") if isinstance(summary.get("semantic_rule_gap_summary"), Mapping) else {}
    label = summary.get("candidate_label_eval") if isinstance(summary.get("candidate_label_eval"), Mapping) else {}
    utility = summary.get("candidate_utility_eval") if isinstance(summary.get("candidate_utility_eval"), Mapping) else {}
    utility_review = (
        summary.get("candidate_utility_eval_with_review")
        if isinstance(summary.get("candidate_utility_eval_with_review"), Mapping)
        else {}
    )
    return {
        "natural_language_segmentation_enabled": bool(summary.get("natural_language_segmentation_enabled")),
        "prompt_count": _num(source.get("prompt_count")),
        "source_file_count": _num(source.get("source_file_count")),
        "extraction_counts": source.get("extraction_counts")
        if isinstance(source.get("extraction_counts"), Mapping)
        else {},
        "supported_request_count": _num(dataset.get("supported_request_count")),
        "applied_count": _num(dataset.get("applied_count")),
        "reusable_prefix_request_count": _num(dataset.get("reusable_prefix_request_count")),
        "total_estimated_gain_chars": _num(dataset.get("total_estimated_gain_chars")),
        "candidate_count": _num(label.get("candidate_count")),
        "label_counts": label.get("label_counts") if isinstance(label.get("label_counts"), Mapping) else {},
        "rule_label_counts": label.get("rule_label_counts")
        if isinstance(label.get("rule_label_counts"), Mapping)
        else {},
        "model_label_counts": label.get("model_label_counts")
        if isinstance(label.get("model_label_counts"), Mapping)
        else {},
        "rule_to_model_label_counts": label.get("rule_to_model_label_counts")
        if isinstance(label.get("rule_to_model_label_counts"), Mapping)
        else {},
        "model_to_final_label_counts": label.get("model_to_final_label_counts")
        if isinstance(label.get("model_to_final_label_counts"), Mapping)
        else {},
        "rule_to_final_label_counts": label.get("rule_to_final_label_counts")
        if isinstance(label.get("rule_to_final_label_counts"), Mapping)
        else {},
        "static_safety_clamp_count": _num(label.get("static_safety_clamp_count")),
        "local_judge_scope": str(label.get("local_judge_scope") or "unknown"),
        "max_local_judge_calls": label.get("max_local_judge_calls"),
        "local_judge_budget_strategy": label.get("local_judge_budget_strategy"),
        "local_judge_budget_diagnostics": label.get("local_judge_budget_diagnostics")
        if isinstance(label.get("local_judge_budget_diagnostics"), Mapping)
        else {},
        "local_judge_effectiveness_diagnostics": label.get("local_judge_effectiveness_diagnostics")
        if isinstance(label.get("local_judge_effectiveness_diagnostics"), Mapping)
        else {},
        "local_judge_action_counts": label.get("local_judge_action_counts")
        if isinstance(label.get("local_judge_action_counts"), Mapping)
        else {},
        "local_judge_skipped_by_scope_label_counts": label.get("local_judge_skipped_by_scope_label_counts")
        if isinstance(label.get("local_judge_skipped_by_scope_label_counts"), Mapping)
        else {},
        "local_judge_skipped_by_budget_label_counts": label.get("local_judge_skipped_by_budget_label_counts")
        if isinstance(label.get("local_judge_skipped_by_budget_label_counts"), Mapping)
        else {},
        "accepted_estimated_reusable_chars": _num(utility.get("total_estimated_reusable_chars")),
        "accepted_or_review_estimated_reusable_chars": _num(utility_review.get("total_estimated_reusable_chars")),
        "candidate_promotion_policy": gap.get("candidate_promotion_policy")
        if isinstance(gap.get("candidate_promotion_policy"), Mapping)
        else {},
        "candidate_review_upper_bound_policy": gap.get("candidate_review_upper_bound_policy")
        if isinstance(gap.get("candidate_review_upper_bound_policy"), Mapping)
        else {},
        "rule_gap_observed": bool(gap.get("rule_gap_observed")),
        "review_queue_observed": bool(gap.get("review_queue_observed")),
        "review_queue_diagnostics": label.get("review_queue_diagnostics")
        if isinstance(label.get("review_queue_diagnostics"), Mapping)
        else {},
    }


def _prompt_text_artifacts(
    *,
    rule_summary: Mapping[str, Any],
    nl_summary: Mapping[str, Any],
    include_candidate_text: bool,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for label, summary in (("rule_only", rule_summary), ("nl_segmentation", nl_summary)):
        pipeline_artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
        requests = pipeline_artifacts.get("requests")
        if requests:
            artifacts.append(
                {
                    "variant": label,
                    "path": requests,
                    "reason": "source prompt request corpus includes raw prompt text",
                }
            )
        if include_candidate_text and pipeline_artifacts.get("semantic_candidates"):
            artifacts.append(
                {
                    "variant": label,
                    "path": pipeline_artifacts.get("semantic_candidates"),
                    "reason": "--include-candidate-text stores semantic candidate text for local judge/human review",
                }
            )
        if include_candidate_text and pipeline_artifacts.get("review_worklist_with_text"):
            artifacts.append(
                {
                    "variant": label,
                    "path": pipeline_artifacts.get("review_worklist_with_text"),
                    "reason": "--include-candidate-text stores review worklist text for local judge/human review",
                }
            )
    return artifacts


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


if __name__ == "__main__":
    raise SystemExit(main())
