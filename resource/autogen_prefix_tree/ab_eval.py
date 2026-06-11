from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset_eval import evaluate_dataset


@dataclass(frozen=True)
class ABEvaluationResult:
    baseline_path: str
    plugin_path: str
    summary_path: str | None
    report_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class TaskResultRecord:
    task_id: str | None
    success: bool | None
    score: float | None


@dataclass(frozen=True)
class GateThresholds:
    min_success_rate_delta: float = -0.02
    max_p95_latency_relative_change: float = 0.0
    max_cost_per_success_relative_change: float = 0.0


@dataclass(frozen=True)
class BootstrapConfig:
    iterations: int = 1000
    seed: int = 20260605
    confidence_level: float = 0.95


def evaluate_ab_traces(
    *,
    baseline_path: str | Path,
    plugin_path: str | Path,
    baseline_tasks_path: str | Path | None = None,
    plugin_tasks_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    gate_thresholds: GateThresholds | None = None,
    bootstrap_config: BootstrapConfig | None = None,
) -> ABEvaluationResult:
    thresholds = gate_thresholds or GateThresholds()
    bootstrap = bootstrap_config or BootstrapConfig()
    baseline_eval = evaluate_dataset(input_path=baseline_path)
    plugin_eval = evaluate_dataset(input_path=plugin_path)
    baseline_provider = baseline_eval.summary["provider_trace"]
    plugin_provider = plugin_eval.summary["provider_trace"]
    baseline_tasks = summarize_task_results(baseline_tasks_path) if baseline_tasks_path else None
    plugin_tasks = summarize_task_results(plugin_tasks_path) if plugin_tasks_path else None
    provider_delta = _delta_summary(baseline_provider, plugin_provider)
    task_delta = _task_delta_summary(baseline_tasks, plugin_tasks, bootstrap=bootstrap)
    combined = _combined_efficiency_summary(
        baseline_provider=baseline_provider,
        plugin_provider=plugin_provider,
        baseline_tasks=baseline_tasks,
        plugin_tasks=plugin_tasks,
    )
    gates = _gate_summary(
        provider_delta=provider_delta,
        task_delta=task_delta,
        combined=combined,
        thresholds=thresholds,
    )
    summary = {
        "schema_version": "prefix-reorder-ab-eval-summary-v1",
        "baseline_path": str(baseline_path),
        "plugin_path": str(plugin_path),
        "baseline_tasks_path": str(baseline_tasks_path) if baseline_tasks_path else None,
        "plugin_tasks_path": str(plugin_tasks_path) if plugin_tasks_path else None,
        "baseline": _run_summary(baseline_eval.summary, task_summary=baseline_tasks),
        "plugin": _run_summary(plugin_eval.summary, task_summary=plugin_tasks),
        "delta": provider_delta,
        "task_delta": task_delta,
        "combined_efficiency": combined,
        "gates": gates,
        "real_provider_metrics_available": bool(baseline_provider.get("supported") and plugin_provider.get("supported")),
        "task_metrics_available": _task_metrics_available(baseline_tasks, plugin_tasks),
        "notes": (
            "Provider metrics come from input traces only. Task metrics come from optional benchmark result files. "
            "This summary stores metric aggregates only."
        ),
    }
    target = _write_json(summary_path, summary)
    report_target = _write_text(report_path, render_markdown_report(summary))
    return ABEvaluationResult(
        baseline_path=str(baseline_path),
        plugin_path=str(plugin_path),
        summary_path=str(target) if target else None,
        report_path=str(report_target) if report_target else None,
        summary=summary,
    )


def summarize_task_results(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    loaded = _load_task_result_records(source_path)
    aggregate = _aggregate_task_summary(loaded)
    if aggregate is not None:
        return {
            "supported": True,
            "path": str(source_path),
            "source_format": _source_format(source_path),
            "aggregate_input": True,
            **aggregate,
            "task_id_count": 0,
            "paired_task_ids_available": False,
        }

    records = tuple(_parse_task_result_record(record, index=index) for index, record in enumerate(loaded, start=1))
    task_ids = {record.task_id for record in records if record.task_id}
    success_values = [record.success for record in records if record.success is not None]
    score_values = [record.score for record in records if record.score is not None]
    success_count = sum(1 for success in success_values if success)
    return {
        "supported": bool(records),
        "path": str(source_path),
        "source_format": _source_format(source_path),
        "aggregate_input": False,
        "task_count": len(records),
        "success_evaluated_count": len(success_values),
        "success_count": success_count if success_values else None,
        "success_rate": _ratio(success_count, len(success_values)) if success_values else None,
        "score_evaluated_count": len(score_values),
        "average_score": _ratio(sum(score_values), len(score_values)) if score_values else None,
        "task_id_count": len(task_ids),
        "paired_task_ids_available": bool(task_ids),
        "_records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline vs plugin provider telemetry JSONL traces.")
    parser.add_argument("--baseline", required=True, help="Baseline provider/proxy telemetry JSONL.")
    parser.add_argument("--plugin", required=True, help="Plugin provider/proxy telemetry JSONL.")
    parser.add_argument("--baseline-tasks", help="Optional baseline benchmark task result file: JSONL, JSON, or CSV.")
    parser.add_argument("--plugin-tasks", help="Optional plugin benchmark task result file: JSONL, JSON, or CSV.")
    parser.add_argument("--summary", help="Optional output summary JSON.")
    parser.add_argument("--report-md", help="Optional prompt-safe Markdown report path.")
    parser.add_argument(
        "--min-success-rate-delta",
        type=float,
        default=-0.02,
        help="Non-inferiority threshold for plugin success_rate - baseline success_rate. Default: -0.02.",
    )
    parser.add_argument(
        "--max-p95-latency-relative-change",
        type=float,
        default=0.0,
        help="Maximum allowed relative increase in p95 latency. Default: 0.0.",
    )
    parser.add_argument(
        "--max-cost-per-success-relative-change",
        type=float,
        default=0.0,
        help="Maximum allowed relative increase in cost per successful task. Default: 0.0.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260605)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_ab_traces(
        baseline_path=args.baseline,
        plugin_path=args.plugin,
        baseline_tasks_path=args.baseline_tasks,
        plugin_tasks_path=args.plugin_tasks,
        summary_path=args.summary,
        report_path=args.report_md,
        gate_thresholds=GateThresholds(
            min_success_rate_delta=args.min_success_rate_delta,
            max_p95_latency_relative_change=args.max_p95_latency_relative_change,
            max_cost_per_success_relative_change=args.max_cost_per_success_relative_change,
        ),
        bootstrap_config=BootstrapConfig(
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        ),
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), Mapping) else {}
    plugin = summary.get("plugin") if isinstance(summary.get("plugin"), Mapping) else {}
    baseline_provider = baseline.get("provider_trace") if isinstance(baseline.get("provider_trace"), Mapping) else {}
    plugin_provider = plugin.get("provider_trace") if isinstance(plugin.get("provider_trace"), Mapping) else {}
    delta = summary.get("delta") if isinstance(summary.get("delta"), Mapping) else {}
    baseline_tasks = baseline.get("task_results") if isinstance(baseline.get("task_results"), Mapping) else {}
    plugin_tasks = plugin.get("task_results") if isinstance(plugin.get("task_results"), Mapping) else {}
    task_delta = summary.get("task_delta") if isinstance(summary.get("task_delta"), Mapping) else {}
    combined = summary.get("combined_efficiency") if isinstance(summary.get("combined_efficiency"), Mapping) else {}
    gates = summary.get("gates") if isinstance(summary.get("gates"), Mapping) else {}

    lines = [
        "# Prefix Reorder A/B Summary",
        "",
        "## Provider Metrics",
        "",
        "| Metric | Baseline | Plugin | Delta |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        _markdown_metric_row(label, baseline_provider, plugin_provider, delta, key, delta_key)
        for label, key, delta_key in (
            ("Requests", "record_count", "record_count_delta"),
            ("Rewritten requests", "transformed_count", "transformed_count_delta"),
            ("Prompt tokens", "actual_prompt_tokens", "actual_prompt_tokens_delta"),
            ("Cached tokens", "actual_cached_tokens", "actual_cached_tokens_delta"),
            ("Cache hit ratio", "actual_cache_hit_ratio", "actual_cache_hit_ratio_delta"),
            ("Completion tokens", "actual_completion_tokens", "actual_completion_tokens_delta"),
            ("Total tokens", "actual_total_tokens", "actual_total_tokens_delta"),
            ("Cost USD", "actual_cost_usd", "actual_cost_usd_delta"),
            ("Average latency seconds", "average_latency_seconds", "average_latency_seconds_delta"),
            ("P50 latency seconds", "p50_latency_seconds", "p50_latency_seconds_delta"),
            ("P95 latency seconds", "p95_latency_seconds", "p95_latency_seconds_delta"),
            ("P99 latency seconds", "p99_latency_seconds", "p99_latency_seconds_delta"),
        )
    )

    lines.extend(["", "## Task Metrics", ""])
    if summary.get("task_metrics_available"):
        lines.extend(
            [
                "| Metric | Baseline | Plugin | Delta |",
                "|---|---:|---:|---:|",
                _markdown_metric_row("Task count", baseline_tasks, plugin_tasks, task_delta, "task_count", "task_count_delta"),
                _markdown_metric_row(
                    "Success count",
                    baseline_tasks,
                    plugin_tasks,
                    task_delta,
                    "success_count",
                    "success_count_delta",
                ),
                _markdown_metric_row(
                    "Success rate",
                    baseline_tasks,
                    plugin_tasks,
                    task_delta,
                    "success_rate",
                    "success_rate_delta",
                ),
                _markdown_metric_row(
                    "Average score",
                    baseline_tasks,
                    plugin_tasks,
                    task_delta,
                    "average_score",
                    "average_score_delta",
                ),
            ]
        )
        paired = task_delta.get("paired") if isinstance(task_delta.get("paired"), Mapping) else {}
        bootstrap = paired.get("bootstrap") if isinstance(paired.get("bootstrap"), Mapping) else {}
        success_ci = (
            bootstrap.get("success_rate_delta_ci") if isinstance(bootstrap.get("success_rate_delta_ci"), Mapping) else {}
        )
        score_ci = (
            bootstrap.get("average_score_delta_ci") if isinstance(bootstrap.get("average_score_delta_ci"), Mapping) else {}
        )
        lines.extend(
            [
                "",
                f"Paired common tasks: {_format_value(paired.get('common_task_count'))}",
                f"Paired success delta: {_format_value(paired.get('success_rate_delta'))}",
                f"Paired success delta 95% CI: {_format_ci(success_ci)}",
                f"Paired average score delta: {_format_value(paired.get('average_score_delta'))}",
                f"Paired average score delta 95% CI: {_format_ci(score_ci)}",
            ]
        )
    else:
        lines.append("Task result files were not provided or did not expose supported task metrics.")

    lines.extend(["", "## Combined Efficiency", ""])
    if combined.get("supported"):
        lines.extend(
            [
                "| Metric | Baseline | Plugin | Delta |",
                "|---|---:|---:|---:|",
                _markdown_combined_row(
                    "Cost per successful task USD",
                    combined,
                    "baseline_cost_per_successful_task_usd",
                    "plugin_cost_per_successful_task_usd",
                    "cost_per_successful_task_usd_delta",
                ),
                _markdown_combined_row(
                    "Latency seconds per successful task",
                    combined,
                    "baseline_latency_seconds_per_successful_task",
                    "plugin_latency_seconds_per_successful_task",
                    "latency_seconds_per_successful_task_delta",
                ),
            ]
        )
    else:
        lines.append("Combined efficiency requires both provider metrics and task result metrics.")

    lines.extend(["", "## Gates", ""])
    if gates:
        lines.extend(
            [
                f"Overall status: **{_format_value(gates.get('overall_status'))}**",
                "",
                "| Gate | Status | Evidence | Point estimate | Observed | Threshold | CI |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        gate_items = gates.get("items") if isinstance(gates.get("items"), list) else []
        for item in gate_items:
            if isinstance(item, Mapping):
                confidence_interval = (
                    item.get("confidence_interval") if isinstance(item.get("confidence_interval"), Mapping) else {}
                )
                lines.append(
                    "| "
                    + str(item.get("name"))
                    + " | "
                    + str(item.get("status"))
                    + " | "
                    + _format_value(item.get("evidence") or item.get("reason"))
                    + " | "
                    + _format_value(item.get("point_estimate"))
                    + " | "
                    + _format_value(item.get("observed"))
                    + " | "
                    + _format_value(item.get("threshold"))
                    + " | "
                    + _format_ci(confidence_interval)
                    + " |"
                )
    else:
        lines.append("No gates were evaluated.")

    lines.extend(
        [
            "",
            "## Limits",
            "",
            f"- Real provider metrics available: {_format_value(summary.get('real_provider_metrics_available'))}",
            f"- Task metrics available: {_format_value(summary.get('task_metrics_available'))}",
            "- This report is generated from aggregated telemetry only; it does not include prompt or task text.",
            "- Fake upstream or offline traces validate the reporting path only and cannot prove real cache, latency, cost, or success gains.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_summary(dataset_summary: Mapping[str, Any], *, task_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    provider = dataset_summary.get("provider_trace") or {}
    summary = {
        "input_record_count": dataset_summary.get("input_record_count"),
        "semantic_coverage_supported": dataset_summary.get("semantic_coverage_supported"),
        "supported_request_count": dataset_summary.get("supported_request_count"),
        "provider_trace": provider,
    }
    if task_summary is not None:
        summary["task_results"] = _public_task_summary(task_summary)
    return summary


def _delta_summary(baseline: Mapping[str, Any], plugin: Mapping[str, Any]) -> dict[str, Any]:
    prompt_delta = _num(plugin.get("actual_prompt_tokens")) - _num(baseline.get("actual_prompt_tokens"))
    cached_delta = _num(plugin.get("actual_cached_tokens")) - _num(baseline.get("actual_cached_tokens"))
    completion_delta = _num(plugin.get("actual_completion_tokens")) - _num(baseline.get("actual_completion_tokens"))
    total_delta = _num(plugin.get("actual_total_tokens")) - _num(baseline.get("actual_total_tokens"))
    latency_delta = _float(plugin.get("average_latency_seconds")) - _float(baseline.get("average_latency_seconds"))
    cache_hit_delta = _float(plugin.get("actual_cache_hit_ratio")) - _float(baseline.get("actual_cache_hit_ratio"))
    cost_delta = _float(plugin.get("actual_cost_usd")) - _float(baseline.get("actual_cost_usd"))
    return {
        "latency_supported": bool(baseline.get("latency_supported") and plugin.get("latency_supported")),
        "cost_supported": bool(baseline.get("cost_supported") and plugin.get("cost_supported")),
        "record_count_delta": _num(plugin.get("record_count")) - _num(baseline.get("record_count")),
        "transformed_count_delta": _num(plugin.get("transformed_count")) - _num(baseline.get("transformed_count")),
        "actual_prompt_tokens_delta": prompt_delta,
        "actual_cached_tokens_delta": cached_delta,
        "actual_completion_tokens_delta": completion_delta,
        "actual_total_tokens_delta": total_delta,
        "actual_cost_usd_delta": cost_delta,
        "average_cost_usd_delta": _float(plugin.get("average_cost_usd")) - _float(baseline.get("average_cost_usd")),
        "average_latency_seconds_delta": latency_delta,
        "p50_latency_seconds_delta": _float(plugin.get("p50_latency_seconds")) - _float(baseline.get("p50_latency_seconds")),
        "p95_latency_seconds_delta": _float(plugin.get("p95_latency_seconds")) - _float(baseline.get("p95_latency_seconds")),
        "p99_latency_seconds_delta": _float(plugin.get("p99_latency_seconds")) - _float(baseline.get("p99_latency_seconds")),
        "actual_cache_hit_ratio_delta": cache_hit_delta,
        "actual_cached_tokens_relative_change": _relative_change(
            _num(baseline.get("actual_cached_tokens")),
            _num(plugin.get("actual_cached_tokens")),
        ),
        "actual_cost_usd_relative_change": _relative_change(
            _float(baseline.get("actual_cost_usd")),
            _float(plugin.get("actual_cost_usd")),
        ),
        "average_latency_relative_change": _relative_change(
            _float(baseline.get("average_latency_seconds")),
            _float(plugin.get("average_latency_seconds")),
        ),
        "p50_latency_relative_change": _relative_change(
            _float(baseline.get("p50_latency_seconds")),
            _float(plugin.get("p50_latency_seconds")),
        ),
        "p95_latency_relative_change": _relative_change(
            _float(baseline.get("p95_latency_seconds")),
            _float(plugin.get("p95_latency_seconds")),
        ),
        "p99_latency_relative_change": _relative_change(
            _float(baseline.get("p99_latency_seconds")),
            _float(plugin.get("p99_latency_seconds")),
        ),
        "prompt_token_relative_change": _relative_change(
            _num(baseline.get("actual_prompt_tokens")),
            _num(plugin.get("actual_prompt_tokens")),
        ),
    }


def _combined_efficiency_summary(
    *,
    baseline_provider: Mapping[str, Any],
    plugin_provider: Mapping[str, Any],
    baseline_tasks: Mapping[str, Any] | None,
    plugin_tasks: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if baseline_tasks is None or plugin_tasks is None:
        return {"supported": False, "reason": "missing_task_result_paths"}

    baseline_success_count = _successful_task_count(baseline_tasks)
    plugin_success_count = _successful_task_count(plugin_tasks)
    baseline_cost_per_success = (
        _ratio_or_none(_float(baseline_provider.get("actual_cost_usd")), baseline_success_count)
        if baseline_provider.get("cost_supported")
        else None
    )
    plugin_cost_per_success = (
        _ratio_or_none(_float(plugin_provider.get("actual_cost_usd")), plugin_success_count)
        if plugin_provider.get("cost_supported")
        else None
    )
    baseline_latency_per_success = (
        _ratio_or_none(
            _float(baseline_provider.get("average_latency_seconds")) * _num(baseline_provider.get("record_count")),
            baseline_success_count,
        )
        if baseline_provider.get("latency_supported")
        else None
    )
    plugin_latency_per_success = (
        _ratio_or_none(
            _float(plugin_provider.get("average_latency_seconds")) * _num(plugin_provider.get("record_count")),
            plugin_success_count,
        )
        if plugin_provider.get("latency_supported")
        else None
    )
    return {
        "supported": any(
            value is not None
            for value in (
                baseline_cost_per_success,
                plugin_cost_per_success,
                baseline_latency_per_success,
                plugin_latency_per_success,
            )
        ),
        "baseline_success_count": baseline_success_count,
        "plugin_success_count": plugin_success_count,
        "baseline_cost_per_successful_task_usd": baseline_cost_per_success,
        "plugin_cost_per_successful_task_usd": plugin_cost_per_success,
        "cost_per_successful_task_usd_delta": _nullable_float_delta(
            plugin_cost_per_success,
            baseline_cost_per_success,
        ),
        "baseline_latency_seconds_per_successful_task": baseline_latency_per_success,
        "plugin_latency_seconds_per_successful_task": plugin_latency_per_success,
        "latency_seconds_per_successful_task_delta": _nullable_float_delta(
            plugin_latency_per_success,
            baseline_latency_per_success,
        ),
    }


def _successful_task_count(summary: Mapping[str, Any]) -> int:
    success_count = summary.get("success_count")
    if success_count is not None:
        return _num(success_count)
    success_rate = summary.get("success_rate")
    task_count = summary.get("success_evaluated_count") or summary.get("task_count")
    if success_rate is None or task_count is None:
        return 0
    return int(round(_float(success_rate) * _num(task_count)))


def _gate_summary(
    *,
    provider_delta: Mapping[str, Any],
    task_delta: Mapping[str, Any],
    combined: Mapping[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    items = [
        _success_rate_gate(task_delta, thresholds),
        _p95_latency_gate(provider_delta, thresholds),
        _cost_per_success_gate(combined, thresholds),
    ]
    known_statuses = [item["status"] for item in items if item["status"] != "unknown"]
    if any(status == "fail" for status in known_statuses):
        overall = "fail"
    elif len(known_statuses) == len(items) and all(status == "pass" for status in known_statuses):
        overall = "pass"
    else:
        overall = "unknown"
    return {
        "schema_version": "prefix-reorder-ab-gates-v1",
        "overall_status": overall,
        "thresholds": {
            "min_success_rate_delta": thresholds.min_success_rate_delta,
            "max_p95_latency_relative_change": thresholds.max_p95_latency_relative_change,
            "max_cost_per_success_relative_change": thresholds.max_cost_per_success_relative_change,
        },
        "items": items,
    }


def _success_rate_gate(task_delta: Mapping[str, Any], thresholds: GateThresholds) -> dict[str, Any]:
    aggregate_point_estimate = task_delta.get("success_rate_delta")
    paired = task_delta.get("paired") if isinstance(task_delta.get("paired"), Mapping) else {}
    paired_point_estimate = paired.get("success_rate_delta")
    point_estimate = paired_point_estimate if paired_point_estimate is not None else aggregate_point_estimate
    bootstrap = paired.get("bootstrap") if isinstance(paired.get("bootstrap"), Mapping) else {}
    success_ci = (
        bootstrap.get("success_rate_delta_ci") if isinstance(bootstrap.get("success_rate_delta_ci"), Mapping) else None
    )
    if success_ci and success_ci.get("supported") and success_ci.get("low") is not None:
        observed_float = _float(success_ci.get("low"))
        return _gate_item(
            name="task_success_non_inferiority",
            status="pass" if observed_float >= thresholds.min_success_rate_delta else "fail",
            observed=observed_float,
            threshold=thresholds.min_success_rate_delta,
            reason="paired_bootstrap_ci_low_success_rate_delta",
            evidence="paired_bootstrap_ci_low",
            point_estimate=_float(point_estimate) if point_estimate is not None else None,
            confidence_interval=success_ci,
        )

    if point_estimate is None:
        return _gate_item(
            name="task_success_non_inferiority",
            status="unknown",
            observed=None,
            threshold=thresholds.min_success_rate_delta,
            reason="missing_success_rate_delta",
            evidence="missing_success_rate_delta",
        )

    reason = "missing_paired_bootstrap_ci"
    if success_ci and success_ci.get("reason"):
        reason = str(success_ci.get("reason"))
    return _gate_item(
        name="task_success_non_inferiority",
        status="unknown",
        observed=None,
        threshold=thresholds.min_success_rate_delta,
        reason=reason,
        evidence="paired_bootstrap_ci_unavailable",
        point_estimate=_float(point_estimate),
        confidence_interval=success_ci,
    )


def _p95_latency_gate(provider_delta: Mapping[str, Any], thresholds: GateThresholds) -> dict[str, Any]:
    relative_change = provider_delta.get("p95_latency_relative_change")
    if not provider_delta.get("latency_supported") or relative_change is None:
        return _gate_item(
            name="p95_latency_no_regression",
            status="unknown",
            observed=None,
            threshold=thresholds.max_p95_latency_relative_change,
            reason="missing_p95_latency_relative_change",
        )
    observed_float = _float(relative_change)
    return _gate_item(
        name="p95_latency_no_regression",
        status="pass" if observed_float <= thresholds.max_p95_latency_relative_change else "fail",
        observed=observed_float,
        threshold=thresholds.max_p95_latency_relative_change,
        reason="plugin_p95_latency_relative_change",
    )


def _cost_per_success_gate(combined: Mapping[str, Any], thresholds: GateThresholds) -> dict[str, Any]:
    baseline = combined.get("baseline_cost_per_successful_task_usd")
    plugin = combined.get("plugin_cost_per_successful_task_usd")
    if baseline is None or plugin is None:
        return _gate_item(
            name="cost_per_success_no_regression",
            status="unknown",
            observed=None,
            threshold=thresholds.max_cost_per_success_relative_change,
            reason="missing_cost_per_success",
        )
    relative_change = _relative_change(_float(baseline), _float(plugin))
    return _gate_item(
        name="cost_per_success_no_regression",
        status="pass" if relative_change <= thresholds.max_cost_per_success_relative_change else "fail",
        observed=relative_change,
        threshold=thresholds.max_cost_per_success_relative_change,
        reason="plugin_cost_per_success_relative_change",
    )


def _gate_item(
    *,
    name: str,
    status: str,
    observed: float | None,
    threshold: float,
    reason: str,
    evidence: str | None = None,
    point_estimate: float | None = None,
    confidence_interval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "name": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "reason": reason,
    }
    if evidence is not None:
        item["evidence"] = evidence
    if point_estimate is not None:
        item["point_estimate"] = point_estimate
    if confidence_interval is not None:
        item["confidence_interval"] = dict(confidence_interval)
    return item


def _task_delta_summary(
    baseline: Mapping[str, Any] | None,
    plugin: Mapping[str, Any] | None,
    *,
    bootstrap: BootstrapConfig,
) -> dict[str, Any]:
    if baseline is None or plugin is None:
        return {"supported": False, "reason": "missing_task_result_paths"}

    paired = _paired_task_delta(baseline, plugin, bootstrap=bootstrap)
    return {
        "supported": _task_metrics_available(baseline, plugin),
        "task_count_delta": _nullable_num_delta(plugin.get("task_count"), baseline.get("task_count")),
        "success_count_delta": _nullable_num_delta(plugin.get("success_count"), baseline.get("success_count")),
        "success_rate_delta": _nullable_float_delta(plugin.get("success_rate"), baseline.get("success_rate")),
        "average_score_delta": _nullable_float_delta(plugin.get("average_score"), baseline.get("average_score")),
        "paired": paired,
    }


def _paired_task_delta(
    baseline: Mapping[str, Any],
    plugin: Mapping[str, Any],
    *,
    bootstrap: BootstrapConfig,
) -> dict[str, Any]:
    baseline_records = _records_by_task_id(baseline.get("_records"))
    plugin_records = _records_by_task_id(plugin.get("_records"))
    common_ids = set(baseline_records) & set(plugin_records)
    success_pairs = [
        (baseline_records[task_id].success, plugin_records[task_id].success)
        for task_id in common_ids
        if baseline_records[task_id].success is not None and plugin_records[task_id].success is not None
    ]
    score_pairs = [
        (baseline_records[task_id].score, plugin_records[task_id].score)
        for task_id in common_ids
        if baseline_records[task_id].score is not None and plugin_records[task_id].score is not None
    ]
    baseline_success_count = sum(1 for baseline_success, _ in success_pairs if baseline_success)
    plugin_success_count = sum(1 for _, plugin_success in success_pairs if plugin_success)
    baseline_score_values = [baseline_score for baseline_score, _ in score_pairs if baseline_score is not None]
    plugin_score_values = [plugin_score for _, plugin_score in score_pairs if plugin_score is not None]
    paired_score_deltas = [
        plugin_score - baseline_score
        for baseline_score, plugin_score in score_pairs
        if baseline_score is not None and plugin_score is not None
    ]
    success_delta_values = [
        (1.0 if plugin_success else 0.0) - (1.0 if baseline_success else 0.0)
        for baseline_success, plugin_success in success_pairs
    ]
    return {
        "common_task_count": len(common_ids),
        "baseline_only_task_count": len(set(baseline_records) - common_ids),
        "plugin_only_task_count": len(set(plugin_records) - common_ids),
        "paired_success_task_count": len(success_pairs),
        "baseline_success_rate": _ratio(baseline_success_count, len(success_pairs)) if success_pairs else None,
        "plugin_success_rate": _ratio(plugin_success_count, len(success_pairs)) if success_pairs else None,
        "success_rate_delta": (
            _ratio(plugin_success_count, len(success_pairs)) - _ratio(baseline_success_count, len(success_pairs))
            if success_pairs
            else None
        ),
        "success_transition_counts": {
            "both_success": sum(1 for baseline_success, plugin_success in success_pairs if baseline_success and plugin_success),
            "both_failure": sum(
                1 for baseline_success, plugin_success in success_pairs if not baseline_success and not plugin_success
            ),
            "baseline_only_success": sum(
                1 for baseline_success, plugin_success in success_pairs if baseline_success and not plugin_success
            ),
            "plugin_only_success": sum(
                1 for baseline_success, plugin_success in success_pairs if not baseline_success and plugin_success
            ),
        },
        "paired_score_task_count": len(score_pairs),
        "baseline_average_score": _ratio(sum(baseline_score_values), len(baseline_score_values)) if score_pairs else None,
        "plugin_average_score": _ratio(sum(plugin_score_values), len(plugin_score_values)) if score_pairs else None,
        "average_score_delta": _ratio(sum(paired_score_deltas), len(paired_score_deltas)) if paired_score_deltas else None,
        "bootstrap": {
            "iterations": bootstrap.iterations,
            "seed": bootstrap.seed,
            "confidence_level": bootstrap.confidence_level,
            "success_rate_delta_ci": _bootstrap_mean_ci(success_delta_values, config=bootstrap),
            "average_score_delta_ci": _bootstrap_mean_ci(paired_score_deltas, config=bootstrap),
        },
    }


def _records_by_task_id(records: Any) -> dict[str, TaskResultRecord]:
    if not isinstance(records, tuple):
        return {}
    return {record.task_id: record for record in records if isinstance(record, TaskResultRecord) and record.task_id}


def _bootstrap_mean_ci(values: Sequence[float], *, config: BootstrapConfig) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "supported": False,
            "reason": "insufficient_paired_samples",
            "sample_count": len(values),
            "low": None,
            "high": None,
        }
    if config.iterations <= 0:
        return {
            "supported": False,
            "reason": "bootstrap_disabled",
            "sample_count": len(values),
            "low": None,
            "high": None,
        }
    rng = random.Random(config.seed)
    sample_count = len(values)
    means = []
    for _ in range(config.iterations):
        sample = [values[rng.randrange(sample_count)] for _ in range(sample_count)]
        means.append(sum(sample) / sample_count)
    alpha = max(0.0, min(1.0, 1.0 - config.confidence_level))
    return {
        "supported": True,
        "sample_count": sample_count,
        "low": _quantile(means, alpha / 2.0),
        "high": _quantile(means, 1.0 - alpha / 2.0),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    clipped = max(0.0, min(1.0, probability))
    rank = (len(ordered) - 1) * clipped
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _task_metrics_available(baseline: Mapping[str, Any] | None, plugin: Mapping[str, Any] | None) -> bool:
    if baseline is None or plugin is None:
        return False
    baseline_supported = bool(baseline.get("success_evaluated_count") or baseline.get("score_evaluated_count"))
    plugin_supported = bool(plugin.get("success_evaluated_count") or plugin.get("score_evaluated_count"))
    return baseline_supported and plugin_supported


def _public_task_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in summary.items() if not str(key).startswith("_")}


def _nullable_num_delta(plugin: Any, baseline: Any) -> int | None:
    if plugin is None or baseline is None:
        return None
    return _num(plugin) - _num(baseline)


def _nullable_float_delta(plugin: Any, baseline: Any) -> float | None:
    if plugin is None or baseline is None:
        return None
    return _float(plugin) - _float(baseline)


def _ratio_or_none(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _load_task_result_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))

    text = path.read_text(encoding="utf-8-sig")
    stripped = text.strip()
    if not stripped:
        return ()
    if path.suffix.lower() == ".jsonl":
        return tuple(
            value
            for line in stripped.splitlines()
            if line.strip()
            for value in (json.loads(line),)
            if isinstance(value, Mapping)
        )

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return tuple(
            parsed
            for line in stripped.splitlines()
            if line.strip()
            for parsed in (json.loads(line),)
            if isinstance(parsed, Mapping)
        )
    return tuple(_json_value_to_records(value))


def _json_value_to_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                yield item
        return

    if not isinstance(value, Mapping):
        return

    for key in ("results", "tasks", "task_results", "records", "runs", "samples", "data", "items"):
        nested = value.get(key)
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, Mapping):
                    yield item
            return

    mapping_values = [item for item in value.values() if isinstance(item, Mapping)]
    if mapping_values and len(mapping_values) == len(value):
        for task_id, item in value.items():
            record = dict(item)
            record.setdefault("task_id", task_id)
            yield record
        return

    yield value


def _aggregate_task_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if len(records) != 1:
        return None
    record = records[0]
    if not any(key in record for key in ("task_count", "num_tasks", "success_count", "success_rate", "average_score")):
        return None

    task_count = _first_numeric(record, ("task_count", "num_tasks", "total_tasks", "n_tasks"))
    success_count = _first_numeric(record, ("success_count", "num_success", "passed_count", "pass_count"))
    success_rate = _first_float(record, ("success_rate", "pass_rate", "accuracy"))
    average_score = _first_float(record, ("average_score", "avg_score", "mean_score", "score"))
    if success_rate is None and task_count not in {None, 0} and success_count is not None:
        success_rate = success_count / task_count
    return {
        "task_count": task_count or 0,
        "success_evaluated_count": task_count or (success_count if success_count is not None else 0),
        "success_count": success_count,
        "success_rate": success_rate,
        "score_evaluated_count": task_count if average_score is not None and task_count is not None else 0,
        "average_score": average_score,
    }


def _parse_task_result_record(record: Mapping[str, Any], *, index: int) -> TaskResultRecord:
    task_id = _task_id(record)
    if task_id is None and "task_id" in record:
        task_id = str(record["task_id"])
    success = _success_value(record)
    score = _score_value(record)
    return TaskResultRecord(task_id=task_id or None, success=success, score=score)


def _candidate_task_mappings(record: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield record
    for key in ("result", "metrics", "evaluation", "task", "scores", "output"):
        value = record.get(key)
        if isinstance(value, Mapping):
            yield value


def _task_id(record: Mapping[str, Any]) -> str | None:
    for candidate in _candidate_task_mappings(record):
        for key in ("task_id", "id", "sample_id", "problem_id", "question_id", "name"):
            value = candidate.get(key)
            if value not in {None, ""}:
                return str(value)
    return None


def _success_value(record: Mapping[str, Any]) -> bool | None:
    for candidate in _candidate_task_mappings(record):
        for key in (
            "success",
            "is_success",
            "task_success",
            "passed",
            "pass",
            "correct",
            "completed",
            "solved",
            "status",
            "outcome",
            "result",
        ):
            if key in candidate:
                parsed = _parse_bool(candidate.get(key))
                if parsed is not None:
                    return parsed
    return None


def _score_value(record: Mapping[str, Any]) -> float | None:
    for candidate in _candidate_task_mappings(record):
        for key in ("task_score", "score", "final_score", "average_score", "accuracy", "reward", "grade"):
            if key in candidate:
                parsed = _parse_float(candidate.get(key))
                if parsed is not None:
                    return parsed
    return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "y", "1", "pass", "passed", "success", "succeeded", "correct", "solved"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", "fail", "failed", "failure", "incorrect", "unsolved"}:
        return False
    try:
        return bool(float(normalized))
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
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


def _first_numeric(record: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key in record:
            parsed = _parse_float(record.get(key))
            if parsed is not None:
                return int(parsed)
    return None


def _first_float(record: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in record:
            parsed = _parse_float(record.get(key))
            if parsed is not None:
                return parsed
    return None


def _source_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def _relative_change(baseline: int | float, plugin: int | float) -> float:
    if baseline == 0:
        return 0.0
    return (plugin - baseline) / baseline


def _num(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _markdown_metric_row(
    label: str,
    baseline: Mapping[str, Any],
    plugin: Mapping[str, Any],
    delta: Mapping[str, Any],
    key: str,
    delta_key: str,
) -> str:
    return (
        f"| {label} | {_format_value(baseline.get(key))} | "
        f"{_format_value(plugin.get(key))} | {_format_value(delta.get(delta_key))} |"
    )


def _markdown_combined_row(
    label: str,
    combined: Mapping[str, Any],
    baseline_key: str,
    plugin_key: str,
    delta_key: str,
) -> str:
    return (
        f"| {label} | {_format_value(combined.get(baseline_key))} | "
        f"{_format_value(combined.get(plugin_key))} | {_format_value(combined.get(delta_key))} |"
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_ci(value: Mapping[str, Any]) -> str:
    if not value.get("supported"):
        reason = value.get("reason")
        return f"n/a ({reason})" if reason else "n/a"
    return f"[{_format_value(value.get('low'))}, {_format_value(value.get('high'))}]"


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


if __name__ == "__main__":
    raise SystemExit(main())
