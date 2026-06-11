from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .offline_semantic_suite import run_offline_semantic_suite


@dataclass(frozen=True)
class OfflineSemanticMatrixResult:
    output_dir: str
    summary_path: str
    report_path: str
    summary: dict[str, Any]


def run_offline_semantic_matrix(
    *,
    sources: Sequence[tuple[str, str | Path]],
    output_dir: str | Path,
    session_id: str = "offline-semantic-matrix",
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
) -> OfflineSemanticMatrixResult:
    out = Path(output_dir)
    source_out = out / "sources"
    reports = out / "reports"
    out.mkdir(parents=True, exist_ok=True)
    source_out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    prompt_text_artifacts: list[dict[str, Any]] = []
    safe_labels = _safe_label_map(sources)
    for index, (label, source_root) in enumerate(sources, start=1):
        safe_label = safe_labels[index - 1]
        source_path = Path(source_root)
        per_source_dir = source_out / safe_label
        if not source_path.exists():
            source_rows.append(
                {
                    "label": label,
                    "source_root": str(source_path),
                    "status": "fail",
                    "error": "source_root_missing",
                    "output_dir": str(per_source_dir),
                }
            )
            continue
        try:
            suite = run_offline_semantic_suite(
                source_root=source_path,
                output_dir=per_source_dir,
                session_id=f"{session_id}-{safe_label}",
                include_tests=include_tests,
                include_candidate_text=include_candidate_text,
                min_prompt_chars=min_prompt_chars,
                min_confidence=min_confidence,
                judge=judge,
                local_judge_base_url=local_judge_base_url,
                local_judge_model=local_judge_model,
                local_judge_api_key=local_judge_api_key,
                local_judge_timeout=local_judge_timeout,
                local_judge_scope=local_judge_scope,
                max_local_judge_calls=max_local_judge_calls,
            )
            row = _source_row(label=label, source_root=source_path, suite_summary=suite.summary)
            source_rows.append(row)
            for artifact in suite.summary.get("prompt_text_artifacts", []):
                if isinstance(artifact, Mapping):
                    prompt_text_artifacts.append({"label": label, **dict(artifact)})
        except Exception as exc:  # noqa: BLE001 - matrix should preserve partial evidence.
            source_rows.append(
                {
                    "label": label,
                    "source_root": str(source_path),
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                    "output_dir": str(per_source_dir),
                }
            )

    aggregate = _aggregate(source_rows)
    summary = {
        "schema_version": "prefix-offline-semantic-matrix-summary-v1",
        "prompt_safe_summary": True,
        "output_dir": str(out),
        "session_id": session_id,
        "judge_mode": judge,
        "local_judge_scope": local_judge_scope,
        "max_local_judge_calls": max_local_judge_calls,
        "include_tests": include_tests,
        "include_candidate_text": include_candidate_text,
        "source_count": len(sources),
        "sources": source_rows,
        "aggregate": aggregate,
        "prompt_text_artifacts": prompt_text_artifacts,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "Offline multi-source prompt coverage only. It can identify rule coverage gaps and justify real A/B "
            "smokes, but cannot prove provider cache, latency, cost, or task-success gains."
        ),
    }
    summary_path = out / "offline_semantic_matrix_summary.json"
    report_path = reports / "offline_semantic_matrix_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_markdown_matrix_report(summary), encoding="utf-8")
    return OfflineSemanticMatrixResult(
        output_dir=str(out),
        summary_path=str(summary_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_markdown_matrix_report(summary: Mapping[str, Any]) -> str:
    aggregate = summary.get("aggregate") if isinstance(summary.get("aggregate"), Mapping) else {}
    lines = [
        "# Offline Semantic Matrix",
        "",
        f"Sources: {_format_value(summary.get('source_count'))}",
        f"Completed: {_format_value(aggregate.get('completed_source_count'))}",
        f"Supported sources: {_format_value(aggregate.get('supported_source_count'))}",
        f"Recommendation: **{_format_value(aggregate.get('recommendation'))}**",
        "",
        "| Source | Status | Prompts | Supported | Rule applied | NL applied | Gain delta | Candidates | Review | Rule gap resolved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("sources", []) if isinstance(summary.get("sources"), list) else []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    _format_value(row.get("label")),
                    _format_value(row.get("status")),
                    _format_value(row.get("prompt_count")),
                    _format_value(row.get("supported_request_count")),
                    _format_value(row.get("rule_only_applied_count")),
                    _format_value(row.get("nl_segmentation_applied_count")),
                    _format_value(row.get("total_estimated_gain_chars_delta")),
                    _format_value(row.get("candidate_count")),
                    _format_value(row.get("review_candidate_count")),
                    _format_value(row.get("rule_gap_resolved")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key in (
        "source_count",
        "completed_source_count",
        "failed_source_count",
        "supported_source_count",
        "unsupported_source_count",
        "rule_gap_source_count",
        "rule_gap_resolved_source_count",
        "review_queue_source_count",
        "total_prompt_count",
        "total_supported_request_count",
        "total_rule_only_applied_count",
        "total_nl_segmentation_applied_count",
        "total_estimated_gain_chars_delta",
        "total_candidate_count",
        "total_accept_candidate_count",
        "total_review_candidate_count",
    ):
        lines.append(f"| {key} | {_format_value(aggregate.get(key))} |")
    lines.extend(
        [
            "",
            "## Prompt Extraction Diagnostics",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| extraction_counts | {_format_value(aggregate.get('extraction_counts'))} |",
        ]
    )
    review_queue = (
        aggregate.get("review_queue_diagnostics")
        if isinstance(aggregate.get("review_queue_diagnostics"), Mapping)
        else {}
    )
    lines.extend(
        [
            "",
            "## Review Queue",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| local_judge_priority_counts | {_format_value(review_queue.get('local_judge_priority_counts'))} |",
            f"| risk_tag_counts | {_format_value(review_queue.get('risk_tag_counts'))} |",
            f"| semantic_hint_counts | {_format_value(review_queue.get('semantic_hint_counts'))} |",
            f"| label_reason_counts | {_format_value(review_queue.get('label_reason_counts'))} |",
            f"| review_candidate_chars | {_format_value(review_queue.get('review_candidate_chars'))} |",
            f"| local_judge_action_counts | {_format_value(aggregate.get('local_judge_action_counts'))} |",
            f"| local_judge_skipped_by_scope_label_counts | {_format_value(aggregate.get('local_judge_skipped_by_scope_label_counts'))} |",
            f"| local_judge_skipped_by_budget_label_counts | {_format_value(aggregate.get('local_judge_skipped_by_budget_label_counts'))} |",
        ]
    )
    lines.extend(
        [
            "",
            "## Candidate Label Diagnostics",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| rule_label_counts | {_format_value(aggregate.get('rule_label_counts'))} |",
            f"| model_label_counts | {_format_value(aggregate.get('model_label_counts'))} |",
            f"| rule_to_model_label_counts | {_format_value(aggregate.get('rule_to_model_label_counts'))} |",
            f"| model_to_final_label_counts | {_format_value(aggregate.get('model_to_final_label_counts'))} |",
            f"| rule_to_final_label_counts | {_format_value(aggregate.get('rule_to_final_label_counts'))} |",
            f"| static_safety_clamp_count | {_format_value(aggregate.get('static_safety_clamp_count'))} |",
            f"| local_judge_effectiveness_diagnostics | {_format_value(aggregate.get('local_judge_effectiveness_diagnostics'))} |",
        ]
    )
    budget = (
        aggregate.get("local_judge_budget_diagnostics")
        if isinstance(aggregate.get("local_judge_budget_diagnostics"), Mapping)
        else {}
    )
    lines.extend(
        [
            "",
            "## Local Judge Budget",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| source_with_budget_diagnostics_count | {_format_value(budget.get('source_with_budget_diagnostics_count'))} |",
            f"| budget_limited_source_count | {_format_value(budget.get('budget_limited_source_count'))} |",
            f"| budget_exhausted_source_count | {_format_value(budget.get('budget_exhausted_source_count'))} |",
            f"| total_eligible_for_budget_count | {_format_value(budget.get('total_eligible_for_budget_count'))} |",
            f"| total_model_called_count | {_format_value(budget.get('total_model_called_count'))} |",
            f"| total_skipped_by_budget_count | {_format_value(budget.get('total_skipped_by_budget_count'))} |",
            f"| aggregate_budget_coverage_rate | {_format_value(budget.get('aggregate_budget_coverage_rate'))} |",
            f"| budget_strategy_counts | {_format_value(budget.get('budget_strategy_counts'))} |",
            f"| max_local_judge_calls_counts | {_format_value(budget.get('max_local_judge_calls_counts'))} |",
            f"| budget_exhausted_sources | {_format_value(budget.get('budget_exhausted_sources'))} |",
        ]
    )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This report is prompt-safe and does not include prompt or candidate text.",
            "- Listed prompt-text artifact paths point to local intermediate files under the output directory.",
            "- Real cached tokens, latency, cost, and task success still require provider/benchmark A/B data.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline semantic suites over multiple local framework source roots."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source specification as label=path. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-id", default="offline-semantic-matrix")
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
    result = run_offline_semantic_matrix(
        sources=[_parse_source_spec(value) for value in args.source],
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
    return 0 if result.summary["aggregate"]["completed_source_count"] > 0 else 1


def _parse_source_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.name or "source", value
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"missing source label in {value!r}")
    if not path.strip():
        raise ValueError(f"missing source path in {value!r}")
    return label, path.strip()


def _source_row(*, label: str, source_root: Path, suite_summary: Mapping[str, Any]) -> dict[str, Any]:
    rule = suite_summary.get("rule_only") if isinstance(suite_summary.get("rule_only"), Mapping) else {}
    nl = suite_summary.get("nl_segmentation") if isinstance(suite_summary.get("nl_segmentation"), Mapping) else {}
    comparison = suite_summary.get("comparison") if isinstance(suite_summary.get("comparison"), Mapping) else {}
    delta = comparison.get("delta") if isinstance(comparison.get("delta"), Mapping) else {}
    gate = comparison.get("experiment_gate") if isinstance(comparison.get("experiment_gate"), Mapping) else {}
    label_counts = nl.get("label_counts") if isinstance(nl.get("label_counts"), Mapping) else {}
    review_queue = (
        nl.get("review_queue_diagnostics")
        if isinstance(nl.get("review_queue_diagnostics"), Mapping)
        else {}
    )
    supported = _num(nl.get("supported_request_count") or rule.get("supported_request_count"))
    status = "pass" if supported > 0 else "no_supported_requests"
    return {
        "label": label,
        "source_root": str(source_root),
        "status": status,
        "summary_path": str(Path(str(suite_summary.get("output_dir"))) / "offline_semantic_suite_summary.json")
        if suite_summary.get("output_dir")
        else None,
        "output_dir": str(suite_summary.get("output_dir") or ""),
        "prompt_count": _num(nl.get("prompt_count") or rule.get("prompt_count")),
        "source_file_count": _num(nl.get("source_file_count") or rule.get("source_file_count")),
        "extraction_counts": nl.get("extraction_counts") if isinstance(nl.get("extraction_counts"), Mapping) else {},
        "supported_request_count": supported,
        "rule_only_applied_count": _num(rule.get("applied_count")),
        "nl_segmentation_applied_count": _num(nl.get("applied_count")),
        "applied_count_delta": _num(delta.get("applied_count_delta")),
        "reusable_prefix_request_count_delta": _num(delta.get("reusable_prefix_request_count_delta")),
        "total_estimated_gain_chars_delta": _num(delta.get("total_estimated_gain_chars_delta")),
        "rule_gap_observed": bool(rule.get("rule_gap_observed")),
        "nl_segmentation_rule_gap_observed": bool(nl.get("rule_gap_observed")),
        "rule_gap_resolved": bool(delta.get("rule_gap_resolved")),
        "candidate_count": _num(nl.get("candidate_count") or rule.get("candidate_count")),
        "accept_candidate_count": _num(label_counts.get("accept")),
        "review_candidate_count": _num(label_counts.get("review")),
        "reject_candidate_count": _num(label_counts.get("reject")),
        "rule_label_counts": nl.get("rule_label_counts") if isinstance(nl.get("rule_label_counts"), Mapping) else {},
        "model_label_counts": nl.get("model_label_counts") if isinstance(nl.get("model_label_counts"), Mapping) else {},
        "rule_to_model_label_counts": nl.get("rule_to_model_label_counts")
        if isinstance(nl.get("rule_to_model_label_counts"), Mapping)
        else {},
        "model_to_final_label_counts": nl.get("model_to_final_label_counts")
        if isinstance(nl.get("model_to_final_label_counts"), Mapping)
        else {},
        "rule_to_final_label_counts": nl.get("rule_to_final_label_counts")
        if isinstance(nl.get("rule_to_final_label_counts"), Mapping)
        else {},
        "static_safety_clamp_count": _num(nl.get("static_safety_clamp_count")),
        "local_judge_scope": nl.get("local_judge_scope") or suite_summary.get("local_judge_scope"),
        "max_local_judge_calls": nl.get("max_local_judge_calls"),
        "local_judge_budget_strategy": nl.get("local_judge_budget_strategy"),
        "local_judge_budget_diagnostics": nl.get("local_judge_budget_diagnostics")
        if isinstance(nl.get("local_judge_budget_diagnostics"), Mapping)
        else {},
        "local_judge_effectiveness_diagnostics": nl.get("local_judge_effectiveness_diagnostics")
        if isinstance(nl.get("local_judge_effectiveness_diagnostics"), Mapping)
        else {},
        "local_judge_action_counts": nl.get("local_judge_action_counts")
        if isinstance(nl.get("local_judge_action_counts"), Mapping)
        else {},
        "local_judge_skipped_by_scope_label_counts": nl.get("local_judge_skipped_by_scope_label_counts")
        if isinstance(nl.get("local_judge_skipped_by_scope_label_counts"), Mapping)
        else {},
        "local_judge_skipped_by_budget_label_counts": nl.get("local_judge_skipped_by_budget_label_counts")
        if isinstance(nl.get("local_judge_skipped_by_budget_label_counts"), Mapping)
        else {},
        "review_queue_observed": bool(nl.get("review_queue_observed") or rule.get("review_queue_observed")),
        "review_queue_diagnostics": review_queue,
        "review_local_judge_priority": review_queue.get("local_judge_priority") if review_queue else None,
        "recommendation": gate.get("recommendation") or suite_summary.get("recommendation"),
        "performance_conclusion_allowed": bool(gate.get("performance_conclusion_allowed")),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") in {"pass", "no_supported_requests"}]
    supported = [row for row in rows if row.get("status") == "pass"]
    source_count = len(rows)
    aggregate = {
        "source_count": source_count,
        "completed_source_count": len(completed),
        "failed_source_count": sum(1 for row in rows if row.get("status") == "fail"),
        "supported_source_count": len(supported),
        "unsupported_source_count": sum(1 for row in rows if row.get("status") == "no_supported_requests"),
        "rule_gap_source_count": sum(1 for row in supported if row.get("rule_gap_observed")),
        "rule_gap_resolved_source_count": sum(1 for row in supported if row.get("rule_gap_resolved")),
        "review_queue_source_count": sum(1 for row in supported if row.get("review_queue_observed")),
        "total_prompt_count": sum(_num(row.get("prompt_count")) for row in completed),
        "total_supported_request_count": sum(_num(row.get("supported_request_count")) for row in supported),
        "total_rule_only_applied_count": sum(_num(row.get("rule_only_applied_count")) for row in supported),
        "total_nl_segmentation_applied_count": sum(_num(row.get("nl_segmentation_applied_count")) for row in supported),
        "total_estimated_gain_chars_delta": sum(_num(row.get("total_estimated_gain_chars_delta")) for row in supported),
        "total_candidate_count": sum(_num(row.get("candidate_count")) for row in supported),
        "total_accept_candidate_count": sum(_num(row.get("accept_candidate_count")) for row in supported),
        "total_review_candidate_count": sum(_num(row.get("review_candidate_count")) for row in supported),
        "static_safety_clamp_count": sum(_num(row.get("static_safety_clamp_count")) for row in supported),
    }
    aggregate["rule_label_counts"] = _aggregate_row_counts(supported, "rule_label_counts")
    aggregate["extraction_counts"] = _aggregate_row_counts(supported, "extraction_counts")
    aggregate["model_label_counts"] = _aggregate_row_counts(supported, "model_label_counts")
    aggregate["rule_to_model_label_counts"] = _aggregate_row_counts(supported, "rule_to_model_label_counts")
    aggregate["model_to_final_label_counts"] = _aggregate_row_counts(supported, "model_to_final_label_counts")
    aggregate["rule_to_final_label_counts"] = _aggregate_row_counts(supported, "rule_to_final_label_counts")
    aggregate["local_judge_action_counts"] = _aggregate_row_counts(supported, "local_judge_action_counts")
    aggregate["local_judge_skipped_by_scope_label_counts"] = _aggregate_row_counts(
        supported,
        "local_judge_skipped_by_scope_label_counts",
    )
    aggregate["local_judge_skipped_by_budget_label_counts"] = _aggregate_row_counts(
        supported,
        "local_judge_skipped_by_budget_label_counts",
    )
    aggregate["local_judge_budget_diagnostics"] = _aggregate_local_judge_budget(supported)
    aggregate["local_judge_effectiveness_diagnostics"] = _aggregate_local_judge_effectiveness(supported)
    aggregate["review_queue_diagnostics"] = _aggregate_review_queue(supported)
    aggregate["supported_source_rate"] = _ratio(aggregate["supported_source_count"], source_count)
    aggregate["rule_gap_resolution_source_rate"] = _ratio(
        aggregate["rule_gap_resolved_source_count"],
        aggregate["rule_gap_source_count"],
    )
    aggregate["recommendation"] = _matrix_recommendation(aggregate)
    return aggregate


def _aggregate_local_judge_effectiveness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics_rows: list[Mapping[str, Any]] = [
        row.get("local_judge_effectiveness_diagnostics")
        for row in rows
        if isinstance(row.get("local_judge_effectiveness_diagnostics"), Mapping)
    ]
    rule_review_count = sum(_num(row.get("rule_review_candidate_count")) for row in diagnostics_rows)
    model_called_count = sum(_num(row.get("model_called_rule_review_count")) for row in diagnostics_rows)
    resolved_count = sum(_num(row.get("resolved_rule_review_count")) for row in diagnostics_rows)
    called_resolved_count = sum(_num(row.get("called_resolved_rule_review_count")) for row in diagnostics_rows)
    remaining_count = sum(_num(row.get("remaining_rule_review_count")) for row in diagnostics_rows)
    final_label_counts: dict[str, int] = {}
    model_label_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in diagnostics_rows:
        _merge_int_counts(final_label_counts, row.get("rule_review_final_label_counts"))
        _merge_int_counts(model_label_counts, row.get("rule_review_model_label_counts"))
        _merge_int_counts(action_counts, row.get("rule_review_local_judge_action_counts"))
    return {
        "schema_version": "prefix-matrix-local-judge-effectiveness-diagnostics-v1",
        "source_with_effectiveness_diagnostics_count": len(diagnostics_rows),
        "rule_review_candidate_count": rule_review_count,
        "model_called_rule_review_count": model_called_count,
        "resolved_rule_review_count": resolved_count,
        "called_resolved_rule_review_count": called_resolved_count,
        "remaining_rule_review_count": remaining_count,
        "resolution_rate": _ratio_or_none(resolved_count, rule_review_count),
        "called_resolution_rate": _ratio_or_none(called_resolved_count, model_called_count),
        "rule_review_final_label_counts": dict(sorted(final_label_counts.items())),
        "rule_review_model_label_counts": dict(sorted(model_label_counts.items())),
        "rule_review_local_judge_action_counts": dict(sorted(action_counts.items())),
    }


def _aggregate_local_judge_budget(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics_rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    strategy_counts: dict[str, int] = {}
    max_call_counts: dict[str, int] = {}
    source_coverage: list[dict[str, Any]] = []
    for row in rows:
        diagnostics = row.get("local_judge_budget_diagnostics")
        if not isinstance(diagnostics, Mapping) or not diagnostics.get("schema_version"):
            continue
        diagnostics_rows.append((row, diagnostics))
        strategy = diagnostics.get("budget_strategy") or row.get("local_judge_budget_strategy") or "none"
        strategy_counts[str(strategy)] = strategy_counts.get(str(strategy), 0) + 1
        max_calls = diagnostics.get("max_local_judge_calls")
        max_call_key = "unlimited" if max_calls is None else str(max_calls)
        max_call_counts[max_call_key] = max_call_counts.get(max_call_key, 0) + 1
        source_coverage.append(
            {
                "label": row.get("label"),
                "budget_limited": bool(diagnostics.get("budget_limited")),
                "max_local_judge_calls": diagnostics.get("max_local_judge_calls"),
                "eligible_for_budget_count": _num(diagnostics.get("eligible_for_budget_count")),
                "model_called_count": _num(diagnostics.get("model_called_count")),
                "skipped_by_budget_count": _num(diagnostics.get("skipped_by_budget_count")),
                "budget_exhausted": bool(diagnostics.get("budget_exhausted")),
                "budget_coverage_rate": diagnostics.get("budget_coverage_rate"),
            }
        )
    total_eligible = sum(
        _num(diagnostics.get("eligible_for_budget_count")) for _row, diagnostics in diagnostics_rows
    )
    total_called = sum(_num(diagnostics.get("model_called_count")) for _row, diagnostics in diagnostics_rows)
    total_skipped = sum(
        _num(diagnostics.get("skipped_by_budget_count")) for _row, diagnostics in diagnostics_rows
    )
    exhausted_sources = [
        {
            "label": coverage.get("label"),
            "eligible_for_budget_count": coverage.get("eligible_for_budget_count"),
            "model_called_count": coverage.get("model_called_count"),
            "skipped_by_budget_count": coverage.get("skipped_by_budget_count"),
            "budget_coverage_rate": coverage.get("budget_coverage_rate"),
        }
        for coverage in source_coverage
        if coverage.get("budget_exhausted")
    ]
    return {
        "schema_version": "prefix-matrix-local-judge-budget-diagnostics-v1",
        "source_with_budget_diagnostics_count": len(diagnostics_rows),
        "budget_limited_source_count": sum(
            1 for _row, diagnostics in diagnostics_rows if diagnostics.get("budget_limited")
        ),
        "budget_exhausted_source_count": len(exhausted_sources),
        "budget_limited": any(diagnostics.get("budget_limited") for _row, diagnostics in diagnostics_rows),
        "budget_exhausted": bool(exhausted_sources),
        "total_eligible_for_budget_count": total_eligible,
        "total_model_called_count": total_called,
        "total_skipped_by_budget_count": total_skipped,
        "aggregate_budget_coverage_rate": _ratio_or_none(total_called, total_eligible),
        "budget_strategy_counts": dict(sorted(strategy_counts.items())),
        "max_local_judge_calls_counts": dict(sorted(max_call_counts.items())),
        "budget_exhausted_sources": exhausted_sources,
        "source_coverage": source_coverage,
    }


def _aggregate_review_queue(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    risk_counts: dict[str, int] = {}
    hint_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    top_sources: list[dict[str, Any]] = []
    review_parent_block_count = 0
    review_source_file_count = 0
    review_candidate_chars = 0
    for row in rows:
        diagnostics = row.get("review_queue_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        _merge_int_counts(risk_counts, diagnostics.get("risk_tag_counts"))
        _merge_int_counts(hint_counts, diagnostics.get("semantic_hint_counts"))
        _merge_int_counts(reason_counts, diagnostics.get("label_reason_counts"))
        priority = diagnostics.get("local_judge_priority")
        if priority:
            priority_counts[str(priority)] = priority_counts.get(str(priority), 0) + 1
        review_parent_block_count += _num(diagnostics.get("review_parent_block_count"))
        review_source_file_count += _num(diagnostics.get("review_source_file_count"))
        review_candidate_chars += _num(diagnostics.get("review_candidate_chars"))
        for source_row in diagnostics.get("top_source_files") if isinstance(diagnostics.get("top_source_files"), list) else []:
            if not isinstance(source_row, Mapping):
                continue
            top_sources.append(
                {
                    "framework": row.get("label"),
                    "source_path": source_row.get("source_path"),
                    "review_candidate_count": _num(source_row.get("review_candidate_count")),
                }
            )
    top_sources.sort(key=lambda item: (-_num(item.get("review_candidate_count")), str(item.get("framework")), str(item.get("source_path"))))
    return {
        "schema_version": "prefix-matrix-review-queue-diagnostics-v1",
        "review_parent_block_count": review_parent_block_count,
        "review_source_file_count": review_source_file_count,
        "review_candidate_chars": review_candidate_chars,
        "risk_tag_counts": dict(sorted(risk_counts.items())),
        "semantic_hint_counts": dict(sorted(hint_counts.items())),
        "label_reason_counts": dict(sorted(reason_counts.items())),
        "local_judge_priority_counts": dict(sorted(priority_counts.items())),
        "top_source_files": top_sources[:10],
    }


def _aggregate_row_counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        _merge_int_counts(counts, row.get(key))
    return dict(sorted(counts.items()))


def _merge_int_counts(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    for key, count in value.items():
        target[str(key)] = target.get(str(key), 0) + _num(count)


def _matrix_recommendation(aggregate: Mapping[str, Any]) -> str:
    if _num(aggregate.get("supported_source_count")) == 0:
        return "improve_source_prompt_extraction"
    if _num(aggregate.get("rule_gap_resolved_source_count")) > 0 and _num(aggregate.get("review_queue_source_count")) > 0:
        return "run_small_real_ab_smoke_keep_review_candidates_disabled"
    if _num(aggregate.get("rule_gap_resolved_source_count")) > 0:
        return "run_small_real_ab_smoke"
    if _num(aggregate.get("total_accept_candidate_count")) > 0:
        return "improve_segmentation_or_local_judge_before_ab"
    return "collect_more_real_prompt_cases"


def _safe_label_map(sources: Sequence[tuple[str, str | Path]]) -> list[str]:
    used: dict[str, int] = {}
    labels: list[str] = []
    for label, _path in sources:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("._-") or "source"
        count = used.get(base, 0) + 1
        used[base] = count
        labels.append(base if count == 1 else f"{base}_{count}")
    return labels


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


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


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list | tuple):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
