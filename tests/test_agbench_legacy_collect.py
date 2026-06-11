from __future__ import annotations

import json

from autogen_prefix_tree.agbench_legacy_collect import (
    collect_legacy_agbench_results,
    main,
    parse_autogenbench_tabulate_csv,
)
from autogen_prefix_tree.agbench_legacy_suite import generate_legacy_agbench_suite


def test_parse_autogenbench_tabulate_csv_flattens_trials(tmp_path) -> None:
    csv_path = tmp_path / "tabulate.csv"
    csv_path.write_text(
        "Task Id,Trial 0 Success,Trial 0 Time,Trial 1 Success,Trial 1 Time\n"
        "\n"
        "HumanEval_0,True,1.0,False,2.5\n",
        encoding="utf-8",
    )

    rows = parse_autogenbench_tabulate_csv(csv_path, variant="baseline")

    assert rows == (
        {
            "task_id": "HumanEval_0::trial_0",
            "original_task_id": "HumanEval_0",
            "trial_index": 0,
            "success": True,
            "passed": True,
            "time_seconds": 1.0,
            "variant": "baseline",
            "source_format": "autogenbench_tabulate_csv",
        },
        {
            "task_id": "HumanEval_0::trial_1",
            "original_task_id": "HumanEval_0",
            "trial_index": 1,
            "success": False,
            "passed": False,
            "time_seconds": 2.5,
            "variant": "baseline",
            "source_format": "autogenbench_tabulate_csv",
        },
    )


def test_collect_legacy_agbench_results_writes_tasks_and_ab_reports(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-collect",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = result.manifest
    _write_provider(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"], cached=10, latency=2.0)
    _write_provider(manifest["oai_config_lists"]["plugin_rule_only"]["provider_telemetry_path"], cached=40, latency=1.0)
    _write_provider(manifest["oai_config_lists"]["plugin_nl_segmentation"]["provider_telemetry_path"], cached=60, latency=1.5)
    baseline_csv = tmp_path / "baseline.csv"
    rule_csv = tmp_path / "rule.csv"
    nl_csv = tmp_path / "nl.csv"
    _write_tabulate_csv(baseline_csv, [("HumanEval_0", "True", "1.0"), ("HumanEval_1", "False", "2.0")])
    _write_tabulate_csv(rule_csv, [("HumanEval_0", "True", "1.0"), ("HumanEval_1", "True", "2.0")])
    _write_tabulate_csv(nl_csv, [("HumanEval_0", "True", "1.0"), ("HumanEval_1", "True", "2.0")])

    collection = collect_legacy_agbench_results(
        manifest_path=result.manifest_path,
        baseline_tabulate_csv=baseline_csv,
        rule_tabulate_csv=rule_csv,
        nl_tabulate_csv=nl_csv,
        summary_path=tmp_path / "collection.json",
    )

    summary = collection.summary
    assert summary["ready_for_ab_report"] is True
    assert summary["real_provider_metrics_available"] is True
    assert summary["task_metrics_available"] is True
    assert summary["fake_upstream"] is False
    assert summary["artifact_provenance"]["schema_version"] == "prefix-legacy-artifact-provenance-v1"
    assert summary["artifact_provenance"]["real_provider_metrics_available"] is True
    assert summary["artifact_provenance"]["task_metrics_available"] is True
    assert summary["artifact_provenance"]["variants"]["baseline"]["provider_telemetry"][
        "cached_token_metric_available"
    ] is True
    assert summary["artifact_provenance"]["variants"]["baseline"]["provider_telemetry"][
        "real_provider_metrics_available"
    ] is True
    assert summary["artifact_provenance"]["variants"]["baseline"]["task_results"]["task_metrics_available"] is True
    assert summary["variants"]["baseline"]["task_results"]["success_rate"] == 0.5
    baseline_tasks = _read_jsonl(manifest["oai_config_lists"]["baseline"]["task_results_path"])
    assert baseline_tasks[0]["task_id"] == "HumanEval_0::trial_0"
    assert summary["comparisons"]["baseline_vs_rule_only"]["status"] == "ran"
    assert summary["comparisons"]["baseline_vs_nl_segmentation"]["status"] == "ran"
    assert summary["comparisons"]["baseline_vs_rule_only"]["input_provenance"]["baseline"][
        "real_provider_metrics_available"
    ] is True
    rule_summary_path = manifest["recommended_ab_eval"]["rule_only_summary"]
    rule_summary = json.loads(open(rule_summary_path, encoding="utf-8").read())
    assert rule_summary["delta"]["actual_cached_tokens_delta"] == 30
    assert "SHOULD_NOT_LEAK" not in json.dumps(summary, ensure_ascii=False)


def test_collect_legacy_agbench_results_marks_fake_provider_inputs_not_real(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-collect-fake",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )
    manifest = result.manifest
    _write_provider(manifest["oai_config_lists"]["baseline"]["provider_telemetry_path"], cached=10, latency=2.0, fake=True)
    _write_provider(
        manifest["oai_config_lists"]["plugin_rule_only"]["provider_telemetry_path"],
        cached=40,
        latency=1.0,
        fake=True,
    )
    _write_provider(
        manifest["oai_config_lists"]["plugin_nl_segmentation"]["provider_telemetry_path"],
        cached=60,
        latency=1.5,
        fake=True,
    )
    baseline_csv = tmp_path / "baseline.csv"
    rule_csv = tmp_path / "rule.csv"
    nl_csv = tmp_path / "nl.csv"
    _write_tabulate_csv(baseline_csv, [("HumanEval_0", "True", "1.0")])
    _write_tabulate_csv(rule_csv, [("HumanEval_0", "True", "1.0")])
    _write_tabulate_csv(nl_csv, [("HumanEval_0", "True", "1.0")])

    collection = collect_legacy_agbench_results(
        manifest_path=result.manifest_path,
        baseline_tabulate_csv=baseline_csv,
        rule_tabulate_csv=rule_csv,
        nl_tabulate_csv=nl_csv,
    )

    summary = collection.summary
    rule_summary = json.loads(open(manifest["recommended_ab_eval"]["rule_only_summary"], encoding="utf-8").read())
    assert summary["ready_for_ab_report"] is True
    assert summary["fake_upstream"] is True
    assert summary["real_provider_metrics_available"] is False
    assert summary["artifact_provenance"]["fake_upstream_detected"] is True
    assert summary["artifact_provenance"]["fake_upstream_variants"] == [
        "baseline",
        "plugin_rule_only",
        "plugin_nl_segmentation",
    ]
    assert summary["artifact_provenance"]["variants"]["baseline"]["provider_telemetry"][
        "fake_upstream_detected"
    ] is True
    assert summary["comparisons"]["baseline_vs_rule_only"]["ab_summary"]["fake_upstream"] is True
    assert summary["comparisons"]["baseline_vs_rule_only"]["ab_summary"]["real_provider_metrics_available"] is False
    assert rule_summary["fake_upstream"] is True
    assert rule_summary["real_provider_metrics_available"] is False


def test_agbench_legacy_collect_cli_requires_artifacts(tmp_path) -> None:
    result = generate_legacy_agbench_suite(
        output_dir=tmp_path / "legacy",
        suite_id="legacy-collect-cli",
        model="gpt-test",
        upstream_base_url="https://upstream.example/v1",
        task_jsonl_path=_task_jsonl(tmp_path),
    )

    exit_code = main(["--manifest", result.manifest_path, "--require-artifacts"])

    assert exit_code == 1


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


def _write_provider(path, *, cached: int, latency: float, fake: bool = False) -> None:
    row = {
        "schema_version": "prefix-forward-proxy-provider-telemetry-v1",
        "request_id": f"request-{cached}",
        "actual_prompt_tokens": 100,
        "actual_cached_tokens": cached,
        "actual_completion_tokens": 10,
        "actual_total_tokens": 110,
        "latency_seconds": latency,
        "rewrite_applied": cached > 10,
        "error": None,
    }
    if fake:
        row["fake_upstream"] = True
        row["real_provider_metrics_available"] = False
    target = __import__("pathlib").Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _write_tabulate_csv(path, rows) -> None:
    text = "Task Id,Trial 0 Success,Trial 0 Time\n\n"
    text += "\n\n".join(f"{task_id},{success},{time_seconds}" for task_id, success, time_seconds in rows)
    text += "\n"
    path.write_text(text, encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
