from __future__ import annotations

import json

import pytest

from autogen_prefix_tree.ab_eval import evaluate_ab_traces, main, render_markdown_report


def test_ab_eval_compares_provider_metrics_without_prompt_text(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    plugin_path = tmp_path / "plugin.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(
        baseline_path,
        [
            _provider_row("base-1", prompt=100, cached=10, completion=20, total=120, latency=2.0, transformed=False),
            _provider_row("base-2", prompt=100, cached=10, completion=20, total=120, latency=4.0, transformed=False),
        ],
    )
    _write_jsonl(
        plugin_path,
        [
            _provider_row("plugin-1", prompt=100, cached=40, completion=20, total=120, latency=1.0, transformed=False),
            _provider_row("plugin-2", prompt=100, cached=60, completion=20, total=120, latency=3.0, transformed=True),
        ],
    )

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        summary_path=summary_path,
    )

    summary = result.summary
    assert summary["real_provider_metrics_available"] is True
    assert summary["baseline"]["semantic_coverage_supported"] is False
    assert summary["plugin"]["provider_trace"]["transformed_count"] == 1
    assert summary["delta"]["actual_cached_tokens_delta"] == 80
    assert summary["delta"]["actual_cache_hit_ratio_delta"] == pytest.approx(0.4)
    assert summary["delta"]["actual_cached_tokens_relative_change"] == pytest.approx(4.0)
    assert summary["delta"]["average_latency_seconds_delta"] == pytest.approx(-1.0)
    assert summary["delta"]["average_latency_relative_change"] == pytest.approx(-1 / 3)
    assert summary["delta"]["p95_latency_seconds_delta"] == pytest.approx(-1.0)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "prefix-reorder-ab-eval-summary-v1"
    assert "prompt text" not in json.dumps(written, ensure_ascii=False)


def test_ab_eval_cli_writes_summary(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    plugin_path = tmp_path / "plugin.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(baseline_path, [_provider_row("base", prompt=50, cached=0, completion=5, total=55, latency=1.0)])
    _write_jsonl(plugin_path, [_provider_row("plugin", prompt=50, cached=25, completion=5, total=55, latency=1.0)])

    exit_code = main(["--baseline", str(baseline_path), "--plugin", str(plugin_path), "--summary", str(summary_path)])

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["delta"]["actual_cached_tokens_delta"] == 25


def test_ab_eval_merges_jsonl_task_success_and_scores(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.jsonl"
    plugin_tasks_path = tmp_path / "plugin_tasks.jsonl"
    _write_jsonl(baseline_path, [_provider_row("base", prompt=100, cached=10, completion=5, total=105, latency=2.0)])
    _write_jsonl(plugin_path, [_provider_row("plugin", prompt=100, cached=30, completion=5, total=105, latency=1.5)])
    _write_jsonl(
        baseline_tasks_path,
        [
            {"task_id": "task-1", "success": True, "task_score": 1.0, "prompt": "do not leak"},
            {"task_id": "task-2", "success": False, "task_score": 0.25},
            {"task_id": "task-3", "success": True, "task_score": 0.75},
        ],
    )
    _write_jsonl(
        plugin_tasks_path,
        [
            {"task_id": "task-1", "success": True, "task_score": 1.0},
            {"task_id": "task-2", "success": True, "task_score": 0.75},
            {"task_id": "task-4", "success": False, "task_score": 0.0},
        ],
    )

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        baseline_tasks_path=baseline_tasks_path,
        plugin_tasks_path=plugin_tasks_path,
    )

    summary = result.summary
    assert summary["task_metrics_available"] is True
    assert summary["baseline"]["task_results"]["task_count"] == 3
    assert summary["baseline"]["task_results"]["success_rate"] == pytest.approx(2 / 3)
    assert summary["plugin"]["task_results"]["average_score"] == pytest.approx(7 / 12)
    assert "_records" not in summary["baseline"]["task_results"]
    assert summary["task_delta"]["success_count_delta"] == 0
    assert summary["task_delta"]["success_rate_delta"] == pytest.approx(-0.0)
    assert summary["task_delta"]["average_score_delta"] == pytest.approx(-1 / 12)
    paired = summary["task_delta"]["paired"]
    assert paired["common_task_count"] == 2
    assert paired["baseline_only_task_count"] == 1
    assert paired["plugin_only_task_count"] == 1
    assert paired["success_rate_delta"] == pytest.approx(0.5)
    assert paired["success_transition_counts"]["plugin_only_success"] == 1
    assert paired["average_score_delta"] == pytest.approx(0.25)
    assert paired["bootstrap"]["success_rate_delta_ci"]["supported"] is True
    assert paired["bootstrap"]["success_rate_delta_ci"]["sample_count"] == 2
    assert paired["bootstrap"]["average_score_delta_ci"]["supported"] is True
    assert "do not leak" not in json.dumps(summary, ensure_ascii=False)


def test_ab_eval_reports_cost_and_efficiency_deltas(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.jsonl"
    plugin_tasks_path = tmp_path / "plugin_tasks.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _provider_row("base-1", prompt=100, cached=10, completion=10, total=110, latency=2.0, cost=0.20),
            _provider_row("base-2", prompt=100, cached=10, completion=10, total=110, latency=4.0, cost=0.30),
        ],
    )
    _write_jsonl(
        plugin_path,
        [
            _provider_row("plugin-1", prompt=100, cached=50, completion=10, total=110, latency=1.0, cost=0.10),
            _provider_row("plugin-2", prompt=100, cached=50, completion=10, total=110, latency=2.0, cost=0.15),
        ],
    )
    _write_jsonl(
        baseline_tasks_path,
        [{"task_id": "a", "success": True}, {"task_id": "b", "success": False}],
    )
    _write_jsonl(
        plugin_tasks_path,
        [{"task_id": "a", "success": True}, {"task_id": "b", "success": True}],
    )

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        baseline_tasks_path=baseline_tasks_path,
        plugin_tasks_path=plugin_tasks_path,
    )

    assert result.summary["baseline"]["provider_trace"]["actual_cost_usd"] == pytest.approx(0.50)
    assert result.summary["plugin"]["provider_trace"]["average_cost_usd"] == pytest.approx(0.125)
    assert result.summary["delta"]["actual_cost_usd_delta"] == pytest.approx(-0.25)
    assert result.summary["delta"]["actual_cost_usd_relative_change"] == pytest.approx(-0.5)
    assert result.summary["delta"]["p50_latency_seconds_delta"] == pytest.approx(-1.5)
    assert result.summary["delta"]["p95_latency_seconds_delta"] == pytest.approx(-1.95)
    assert result.summary["delta"]["p95_latency_relative_change"] == pytest.approx(-0.5)
    efficiency = result.summary["combined_efficiency"]
    assert efficiency["supported"] is True
    assert efficiency["baseline_cost_per_successful_task_usd"] == pytest.approx(0.50)
    assert efficiency["plugin_cost_per_successful_task_usd"] == pytest.approx(0.125)
    assert efficiency["cost_per_successful_task_usd_delta"] == pytest.approx(-0.375)
    assert efficiency["baseline_latency_seconds_per_successful_task"] == pytest.approx(6.0)
    assert efficiency["plugin_latency_seconds_per_successful_task"] == pytest.approx(1.5)
    gates = result.summary["gates"]
    assert gates["overall_status"] == "pass"
    assert {item["name"]: item["status"] for item in gates["items"]} == {
        "task_success_non_inferiority": "pass",
        "p95_latency_no_regression": "pass",
        "cost_per_success_no_regression": "pass",
    }


def test_ab_eval_gates_fail_on_quality_latency_and_cost_regressions(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.jsonl"
    plugin_tasks_path = tmp_path / "plugin_tasks.jsonl"
    _write_jsonl(
        baseline_path,
        [_provider_row("base", prompt=100, cached=10, completion=10, total=110, latency=1.0, cost=0.10)],
    )
    _write_jsonl(
        plugin_path,
        [_provider_row("plugin", prompt=100, cached=50, completion=10, total=110, latency=1.5, cost=0.50)],
    )
    _write_jsonl(
        baseline_tasks_path,
        [{"task_id": "a", "success": True}, {"task_id": "b", "success": True}],
    )
    _write_jsonl(
        plugin_tasks_path,
        [{"task_id": "a", "success": True}, {"task_id": "b", "success": False}],
    )

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        baseline_tasks_path=baseline_tasks_path,
        plugin_tasks_path=plugin_tasks_path,
    )

    gates = result.summary["gates"]
    assert gates["overall_status"] == "fail"
    statuses = {item["name"]: item for item in gates["items"]}
    assert statuses["task_success_non_inferiority"]["status"] == "fail"
    assert statuses["task_success_non_inferiority"]["observed"] <= -0.5
    assert statuses["p95_latency_no_regression"]["status"] == "fail"
    assert statuses["cost_per_success_no_regression"]["status"] == "fail"


def test_ab_eval_success_gate_uses_bootstrap_ci_lower_bound(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.jsonl"
    plugin_tasks_path = tmp_path / "plugin_tasks.jsonl"
    _write_jsonl(
        baseline_path,
        [_provider_row("base", prompt=100, cached=10, completion=10, total=110, latency=1.0, cost=0.10)],
    )
    _write_jsonl(
        plugin_path,
        [_provider_row("plugin", prompt=100, cached=50, completion=10, total=110, latency=1.0, cost=0.10)],
    )
    _write_jsonl(
        baseline_tasks_path,
        [
            {"task_id": "a", "success": True},
            {"task_id": "b", "success": False},
            {"task_id": "c", "success": False},
            {"task_id": "d", "success": False},
        ],
    )
    _write_jsonl(
        plugin_tasks_path,
        [
            {"task_id": "a", "success": True},
            {"task_id": "b", "success": False},
            {"task_id": "c", "success": False},
            {"task_id": "d", "success": True},
        ],
    )

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        baseline_tasks_path=baseline_tasks_path,
        plugin_tasks_path=plugin_tasks_path,
    )

    gate = {item["name"]: item for item in result.summary["gates"]["items"]}["task_success_non_inferiority"]
    assert gate["status"] == "pass"
    assert gate["evidence"] == "paired_bootstrap_ci_low"
    assert gate["reason"] == "paired_bootstrap_ci_low_success_rate_delta"
    assert gate["observed"] >= -0.02
    assert gate["point_estimate"] == pytest.approx(0.25)
    assert gate["confidence_interval"]["supported"] is True


def test_ab_eval_success_gate_fails_when_ci_lower_bound_regresses(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.jsonl"
    plugin_tasks_path = tmp_path / "plugin_tasks.jsonl"
    _write_jsonl(
        baseline_path,
        [_provider_row("base", prompt=100, cached=10, completion=10, total=110, latency=1.0, cost=0.10)],
    )
    _write_jsonl(
        plugin_path,
        [_provider_row("plugin", prompt=100, cached=50, completion=10, total=110, latency=1.0, cost=0.10)],
    )
    _write_jsonl(
        baseline_tasks_path,
        [
            {"task_id": "a", "success": True},
            {"task_id": "b", "success": True},
            {"task_id": "c", "success": False},
            {"task_id": "d", "success": False},
        ],
    )
    _write_jsonl(
        plugin_tasks_path,
        [
            {"task_id": "a", "success": False},
            {"task_id": "b", "success": True},
            {"task_id": "c", "success": False},
            {"task_id": "d", "success": True},
        ],
    )

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        baseline_tasks_path=baseline_tasks_path,
        plugin_tasks_path=plugin_tasks_path,
    )

    gate = {item["name"]: item for item in result.summary["gates"]["items"]}["task_success_non_inferiority"]
    assert gate["status"] == "fail"
    assert gate["point_estimate"] == pytest.approx(0.0)
    assert gate["observed"] < -0.02
    assert gate["observed"] == pytest.approx(gate["confidence_interval"]["low"])


def test_ab_eval_success_gate_unknown_without_paired_bootstrap_ci(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.jsonl"
    plugin_tasks_path = tmp_path / "plugin_tasks.jsonl"
    _write_jsonl(
        baseline_path,
        [_provider_row("base", prompt=100, cached=10, completion=10, total=110, latency=1.0, cost=0.10)],
    )
    _write_jsonl(
        plugin_path,
        [_provider_row("plugin", prompt=100, cached=50, completion=10, total=110, latency=1.0, cost=0.10)],
    )
    _write_jsonl(baseline_tasks_path, [{"task_id": "only", "success": False}])
    _write_jsonl(plugin_tasks_path, [{"task_id": "only", "success": True}])

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        baseline_tasks_path=baseline_tasks_path,
        plugin_tasks_path=plugin_tasks_path,
    )

    gate = {item["name"]: item for item in result.summary["gates"]["items"]}["task_success_non_inferiority"]
    assert gate["status"] == "unknown"
    assert gate["observed"] is None
    assert gate["point_estimate"] == pytest.approx(1.0)
    assert gate["evidence"] == "paired_bootstrap_ci_unavailable"
    assert gate["reason"] == "insufficient_paired_samples"


def test_ab_eval_gates_unknown_when_required_metrics_missing(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    _write_jsonl(baseline_path, [_provider_row("base", prompt=50, cached=0, completion=5, total=55, latency=1.0)])
    _write_jsonl(plugin_path, [_provider_row("plugin", prompt=50, cached=25, completion=5, total=55, latency=1.0)])

    result = evaluate_ab_traces(baseline_path=baseline_path, plugin_path=plugin_path)

    gates = result.summary["gates"]
    assert gates["overall_status"] == "unknown"
    statuses = {item["name"]: item["status"] for item in gates["items"]}
    assert statuses["task_success_non_inferiority"] == "unknown"
    assert statuses["p95_latency_no_regression"] == "pass"
    assert statuses["cost_per_success_no_regression"] == "unknown"


def test_ab_eval_merges_csv_task_results(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.csv"
    plugin_tasks_path = tmp_path / "plugin_tasks.csv"
    _write_jsonl(baseline_path, [_provider_row("base", prompt=80, cached=0, completion=5, total=85, latency=1.0)])
    _write_jsonl(plugin_path, [_provider_row("plugin", prompt=80, cached=20, completion=5, total=85, latency=1.0)])
    baseline_tasks_path.write_text("task_id,passed,score\nalpha,true,1.0\nbeta,false,0.0\n", encoding="utf-8")
    plugin_tasks_path.write_text("task_id,passed,score\nalpha,true,1.0\nbeta,true,0.5\n", encoding="utf-8")

    result = evaluate_ab_traces(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        baseline_tasks_path=baseline_tasks_path,
        plugin_tasks_path=plugin_tasks_path,
    )

    assert result.summary["baseline"]["task_results"]["source_format"] == "csv"
    assert result.summary["task_delta"]["success_rate_delta"] == pytest.approx(0.5)
    assert result.summary["task_delta"]["average_score_delta"] == pytest.approx(0.25)
    assert result.summary["task_delta"]["paired"]["common_task_count"] == 2


def test_ab_eval_cli_accepts_task_result_paths(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.json"
    plugin_tasks_path = tmp_path / "plugin_tasks.json"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(baseline_path, [_provider_row("base", prompt=50, cached=0, completion=5, total=55, latency=1.0)])
    _write_jsonl(plugin_path, [_provider_row("plugin", prompt=50, cached=25, completion=5, total=55, latency=1.0)])
    baseline_tasks_path.write_text(json.dumps({"results": [{"id": "a", "passed": True, "score": 1.0}]}), encoding="utf-8")
    plugin_tasks_path.write_text(json.dumps({"results": [{"id": "a", "passed": False, "score": 0.0}]}), encoding="utf-8")

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--plugin",
            str(plugin_path),
            "--baseline-tasks",
            str(baseline_tasks_path),
            "--plugin-tasks",
            str(plugin_tasks_path),
            "--min-success-rate-delta",
            "-0.5",
            "--bootstrap-iterations",
            "25",
            "--bootstrap-seed",
            "123",
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task_metrics_available"] is True
    assert summary["task_delta"]["success_rate_delta"] == pytest.approx(-1.0)
    assert summary["gates"]["thresholds"]["min_success_rate_delta"] == pytest.approx(-0.5)
    assert summary["task_delta"]["paired"]["bootstrap"]["iterations"] == 25
    assert summary["task_delta"]["paired"]["bootstrap"]["seed"] == 123


def test_ab_eval_cli_writes_markdown_report(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    baseline_tasks_path = tmp_path / "baseline_tasks.jsonl"
    plugin_tasks_path = tmp_path / "plugin_tasks.jsonl"
    report_path = tmp_path / "ab_report.md"
    _write_jsonl(baseline_path, [_provider_row("base", prompt=100, cached=10, completion=5, total=105, latency=2.0, cost=0.2)])
    _write_jsonl(plugin_path, [_provider_row("plugin", prompt=100, cached=40, completion=5, total=105, latency=1.0, cost=0.1)])
    _write_jsonl(baseline_tasks_path, [{"task_id": "a", "success": True, "score": 1.0, "prompt": "do not leak"}])
    _write_jsonl(plugin_tasks_path, [{"task_id": "a", "success": True, "score": 1.0}])

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--plugin",
            str(plugin_path),
            "--baseline-tasks",
            str(baseline_tasks_path),
            "--plugin-tasks",
            str(plugin_tasks_path),
            "--report-md",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report = report_path.read_text(encoding="utf-8")
    assert "# Prefix Reorder A/B Summary" in report
    assert "Provider Metrics" in report
    assert "Task Metrics" in report
    assert "Cost per successful task USD" in report
    assert "## Gates" in report
    assert "Overall status" in report
    assert "Paired success delta 95% CI" in report
    assert "do not leak" not in report


def test_render_markdown_report_handles_missing_task_metrics(tmp_path) -> None:
    baseline_path = tmp_path / "baseline_provider.jsonl"
    plugin_path = tmp_path / "plugin_provider.jsonl"
    _write_jsonl(baseline_path, [_provider_row("base", prompt=50, cached=0, completion=5, total=55, latency=1.0)])
    _write_jsonl(plugin_path, [_provider_row("plugin", prompt=50, cached=25, completion=5, total=55, latency=1.0)])
    summary = evaluate_ab_traces(baseline_path=baseline_path, plugin_path=plugin_path).summary

    report = render_markdown_report(summary)

    assert "Task result files were not provided" in report
    assert "Real provider metrics available: true" in report


def _provider_row(
    request_id: str,
    *,
    prompt: int,
    cached: int,
    completion: int,
    total: int,
    latency: float,
    cost: float | None = None,
    transformed: bool = False,
) -> dict:
    row = {
        "request_id": request_id,
        "schema_version": "prefix-forward-proxy-provider-telemetry-v1",
        "actual_prompt_tokens": prompt,
        "actual_cached_tokens": cached,
        "actual_completion_tokens": completion,
        "actual_total_tokens": total,
        "latency_seconds": latency,
        "rewrite_applied": transformed,
        "error": None,
    }
    if cost is not None:
        row["actual_cost_usd"] = cost
    return row


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
