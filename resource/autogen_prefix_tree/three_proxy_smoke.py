from __future__ import annotations

import argparse
import json
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ab_eval import ABEvaluationResult, evaluate_ab_traces, render_markdown_report
from .agbench_legacy_collect import collect_legacy_agbench_results
from .agbench_legacy_runbook import build_legacy_agbench_runbook
from .dataset_eval import evaluate_dataset
from .openai_forward_proxy import build_forward_proxy_server


VARIANTS = ("baseline", "plugin_rule_only", "plugin_nl_segmentation")


@dataclass(frozen=True)
class ThreeProxySmokeResult:
    output_dir: str
    summary_path: str
    summary: dict[str, Any]


def run_three_proxy_smoke(
    *,
    output_dir: str | Path,
    session_id: str = "three-proxy-smoke",
    manifest_path: str | Path | None = None,
) -> ThreeProxySmokeResult:
    out = Path(output_dir)
    artifacts = out / "artifacts"
    reports = out / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    upstream_requests: list[dict[str, Any]] = []
    upstream = _start_fake_upstream(upstream_requests)
    upstream_base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    servers: dict[str, ThreadingHTTPServer] = {}
    manifest_file = Path(manifest_path) if manifest_path is not None else None
    manifest = _load_manifest(manifest_file)
    telemetry_paths = _telemetry_paths(artifacts=artifacts, manifest=manifest, manifest_file=manifest_file)
    shadow_trial_plan_path = _shadow_trial_plan_path(manifest=manifest, manifest_file=manifest_file)
    _reset_telemetry_paths(telemetry_paths.values())
    try:
        for variant in VARIANTS:
            servers[variant] = build_forward_proxy_server(
                host="127.0.0.1",
                port=0,
                upstream_base_url=upstream_base_url,
                session_id=f"{session_id}-{variant}",
                telemetry_log_path=telemetry_paths[variant],
                enabled=variant != "baseline",
                enable_natural_language_segmentation=variant == "plugin_nl_segmentation",
                shadow_trial_plan_path=shadow_trial_plan_path if variant != "baseline" else None,
            )
            threading.Thread(target=servers[variant].serve_forever, daemon=True).start()

        for variant in VARIANTS:
            bodies = _marked_request_bodies() if variant != "plugin_nl_segmentation" else _natural_language_bodies()
            for body in bodies:
                _post_json(servers[variant].server_port, body)
    finally:
        for server in servers.values():
            server.shutdown()
            server.server_close()
        upstream.shutdown()
        upstream.server_close()

    dataset_summaries = {variant: evaluate_dataset(input_path=path).summary for variant, path in telemetry_paths.items()}
    provider_summaries = {variant: summary["provider_trace"] for variant, summary in dataset_summaries.items()}
    shadow_trial_summaries = {variant: summary["shadow_trial_trace"] for variant, summary in dataset_summaries.items()}
    for path in telemetry_paths.values():
        _mark_fake_provider_telemetry(path)
    _write_fake_tabulate_csvs(manifest=manifest, manifest_file=manifest_file)
    collection = (
        collect_legacy_agbench_results(
            manifest_path=manifest_file,
            summary_path=_legacy_collection_summary_path(manifest=manifest, manifest_file=manifest_file),
        )
        if manifest_file is not None and manifest is not None
        else None
    )
    if collection is not None:
        _mark_fake_ab_artifacts(manifest=manifest, manifest_file=manifest_file)
        _mark_fake_collection_result(collection)
        runbook = build_legacy_agbench_runbook(
            manifest_path=manifest_file,
            summary_path=_legacy_runbook_summary_path(manifest=manifest, manifest_file=manifest_file),
            report_path=_legacy_runbook_report_path(manifest=manifest, manifest_file=manifest_file),
            env={"OPENAI_API_KEY": "fake-upstream-smoke-key"},
            require_api_config=True,
        )
        rule_ab_summary = _load_json(_manifest_path(manifest["recommended_ab_eval"]["rule_only_summary"], manifest_file))
        nl_ab_summary = _load_json(_manifest_path(manifest["recommended_ab_eval"]["nl_segmentation_summary"], manifest_file))
    else:
        rule_ab = evaluate_ab_traces(
            baseline_path=telemetry_paths["baseline"],
            plugin_path=telemetry_paths["plugin_rule_only"],
            summary_path=reports / "ab_baseline_vs_rule_only_summary.json",
            report_path=reports / "ab_baseline_vs_rule_only_report.md",
            bootstrap_config=None,
        )
        _mark_fake_ab_result(rule_ab)
        nl_ab = evaluate_ab_traces(
            baseline_path=telemetry_paths["baseline"],
            plugin_path=telemetry_paths["plugin_nl_segmentation"],
            summary_path=reports / "ab_baseline_vs_nl_segmentation_summary.json",
            report_path=reports / "ab_baseline_vs_nl_segmentation_report.md",
            bootstrap_config=None,
        )
        _mark_fake_ab_result(nl_ab)
        runbook = None
        rule_ab_summary = rule_ab.summary
        nl_ab_summary = nl_ab.summary
    summary = {
        "schema_version": "prefix-three-proxy-smoke-summary-v1",
        "session_id": session_id,
        "output_dir": str(out),
        "manifest_path": str(manifest_file) if manifest_file else None,
        "fake_upstream": True,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "This smoke uses a local fake upstream. It validates proxy wiring, telemetry parsing, "
            "and A/B aggregation only; it does not prove real provider cache, latency, cost, or task success."
        ),
        "upstream_request_count": len(upstream_requests),
        "upstream_paths": sorted({str(request.get("path")) for request in upstream_requests}),
        "telemetry_paths": {variant: str(path) for variant, path in telemetry_paths.items()},
        "provider_summaries": provider_summaries,
        "shadow_trial_plan_path": str(shadow_trial_plan_path) if shadow_trial_plan_path else None,
        "shadow_trial_summaries": shadow_trial_summaries,
        "ab_summaries": {
            "baseline_vs_rule_only": rule_ab_summary,
            "baseline_vs_nl_segmentation": nl_ab_summary,
        },
        "legacy_collection_summary_path": collection.summary_path if collection else None,
        "legacy_runbook_summary_path": runbook.summary_path if runbook else None,
        "legacy_runbook_next_action": runbook.summary.get("next_action") if runbook else None,
        "legacy_runbook_artifact_quality_warn_count": runbook.summary.get("artifact_quality_warn_count") if runbook else None,
    }
    summary_path = reports / "three_proxy_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return ThreeProxySmokeResult(output_dir=str(out), summary_path=str(summary_path), summary=summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-network baseline/rule-only/nl-segmentation proxy smoke against a local fake upstream."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-id", default="three-proxy-smoke")
    parser.add_argument(
        "--manifest",
        help=(
            "Optional legacy_suite_manifest.json. When provided, smoke artifacts are written into the "
            "manifest's expected provider/tabulate/task/A-B/runbook paths."
        ),
    )
    return parser


def _mark_fake_ab_result(result: ABEvaluationResult) -> None:
    result.summary["fake_upstream"] = True
    result.summary["real_provider_metrics_available"] = False
    result.summary["real_provider_metrics_note"] = (
        "This A/B summary was generated by three_proxy_smoke against a local fake upstream. "
        "It validates reporting only and must not be used as real provider evidence."
    )
    if result.summary_path:
        Path(result.summary_path).write_text(
            json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if result.report_path:
        Path(result.report_path).write_text(render_markdown_report(result.summary), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_three_proxy_smoke(output_dir=args.output_dir, session_id=args.session_id, manifest_path=args.manifest)
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _load_manifest(manifest_file: Path | None) -> dict[str, Any] | None:
    if manifest_file is None:
        return None
    value = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{manifest_file} must contain a JSON object")
    return value


def _telemetry_paths(
    *,
    artifacts: Path,
    manifest: Mapping[str, Any] | None,
    manifest_file: Path | None,
) -> dict[str, Path]:
    if manifest is None or manifest_file is None:
        return {variant: artifacts / f"{variant}_provider_telemetry.jsonl" for variant in VARIANTS}
    configs = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    return {
        variant: _manifest_path(configs[variant]["provider_telemetry_path"], manifest_file)
        for variant in VARIANTS
    }


def _reset_telemetry_paths(paths: Sequence[Path]) -> None:
    for path in sorted({Path(raw) for raw in paths}):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _shadow_trial_plan_path(*, manifest: Mapping[str, Any] | None, manifest_file: Path | None) -> Path | None:
    if manifest is None or manifest_file is None:
        return None
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    raw_path = local_judge.get("shadow_trial_plan_path") or local_judge.get("static_rule_calibration_eval_summary_path")
    if not raw_path:
        return None
    path = _manifest_path(raw_path, manifest_file)
    return path if path.exists() else None


def _mark_fake_provider_telemetry(path: Path) -> None:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict):
                row["fake_upstream"] = True
                row["real_provider_metrics_available"] = False
                row["real_provider_metrics_note"] = "local fake upstream smoke; not real provider evidence"
                rows.append(row)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_fake_tabulate_csvs(*, manifest: Mapping[str, Any] | None, manifest_file: Path | None) -> None:
    if manifest is None or manifest_file is None:
        return
    configs = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    rows_by_variant = {
        "baseline": (("FakeTask_0", "True", "1.0"), ("FakeTask_1", "False", "1.5")),
        "plugin_rule_only": (("FakeTask_0", "True", "1.0"), ("FakeTask_1", "True", "1.5")),
        "plugin_nl_segmentation": (("FakeTask_0", "True", "1.0"), ("FakeTask_1", "True", "1.5")),
    }
    for variant, rows in rows_by_variant.items():
        path = _manifest_path(configs[variant]["tabulate_csv_path"], manifest_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "Task Id,Trial 0 Success,Trial 0 Time\n"
        text += "\n".join(f"{task_id},{success},{time_seconds}" for task_id, success, time_seconds in rows)
        text += "\n"
        path.write_text(text, encoding="utf-8")


def _legacy_collection_summary_path(*, manifest: Mapping[str, Any] | None, manifest_file: Path | None) -> Path | None:
    if manifest is None or manifest_file is None:
        return None
    reports_dir = _manifest_path(manifest["reports_dir"], manifest_file)
    return reports_dir / "legacy_collection_summary.json"


def _legacy_runbook_summary_path(*, manifest: Mapping[str, Any] | None, manifest_file: Path | None) -> Path | None:
    if manifest is None or manifest_file is None:
        return None
    runbook = manifest.get("recommended_runbook") if isinstance(manifest.get("recommended_runbook"), Mapping) else {}
    return _manifest_path(runbook.get("summary") or Path(manifest["reports_dir"]) / "legacy_runbook_summary.json", manifest_file)


def _legacy_runbook_report_path(*, manifest: Mapping[str, Any] | None, manifest_file: Path | None) -> Path | None:
    if manifest is None or manifest_file is None:
        return None
    runbook = manifest.get("recommended_runbook") if isinstance(manifest.get("recommended_runbook"), Mapping) else {}
    return _manifest_path(runbook.get("report_md") or Path(manifest["reports_dir"]) / "legacy_runbook.md", manifest_file)


def _mark_fake_ab_artifacts(*, manifest: Mapping[str, Any], manifest_file: Path) -> None:
    recommended = manifest.get("recommended_ab_eval") if isinstance(manifest.get("recommended_ab_eval"), Mapping) else {}
    for summary_key, report_key in (
        ("rule_only_summary", "rule_only_report_md"),
        ("nl_segmentation_summary", "nl_segmentation_report_md"),
    ):
        summary_path = _manifest_path(recommended[summary_key], manifest_file)
        summary = _load_json(summary_path)
        summary["fake_upstream"] = True
        summary["real_provider_metrics_available"] = False
        summary["real_provider_metrics_note"] = (
            "This A/B summary was generated by three_proxy_smoke against a local fake upstream. "
            "It validates reporting only and must not be used as real provider evidence."
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        report_path = _manifest_path(recommended[report_key], manifest_file)
        report_path.write_text(render_markdown_report(summary), encoding="utf-8")


def _mark_fake_collection_result(collection: Any) -> None:
    collection.summary["fake_upstream"] = True
    collection.summary["real_provider_metrics_available"] = False
    collection.summary["real_provider_metrics_note"] = (
        "This collection summary includes artifacts produced by three_proxy_smoke against a local fake upstream. "
        "It validates collector wiring only and must not be used as real provider evidence."
    )
    comparisons = collection.summary.get("comparisons")
    if isinstance(comparisons, Mapping):
        for item in comparisons.values():
            if not isinstance(item, dict):
                continue
            ab_summary = item.get("ab_summary")
            if isinstance(ab_summary, dict):
                ab_summary["fake_upstream"] = True
                ab_summary["real_provider_metrics_available"] = False
    if collection.summary_path:
        Path(collection.summary_path).write_text(
            json.dumps(collection.summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _manifest_path(value: Any, manifest_file: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    manifest_relative = manifest_file.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return cwd_path


def _post_json(port: int, body: Mapping[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-smoke-token"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _start_fake_upstream(seen_requests: list[dict[str, Any]]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            applied_hint = _rewrite_applied_hint(body)
            seen_requests.append(
                {
                    "path": self.path,
                    "body": body,
                    "applied_hint": applied_hint,
                }
            )
            prompt_tokens = 140 if applied_hint else 150
            cached_tokens = 90 if applied_hint else 20
            response = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 8,
                    "total_tokens": prompt_tokens + 8,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    "cost_usd": 0.001,
                },
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _rewrite_applied_hint(body: Mapping[str, Any]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0] if isinstance(messages[0], Mapping) else {}
    content = first.get("content")
    return isinstance(content, str) and (
        content.startswith("USER_TASK_START") or content.startswith("Verify the answer carefully")
    )


def _marked_request_bodies() -> tuple[dict[str, Any], dict[str, Any]]:
    return (_marked_body("planner"), _marked_body("engineer"))


def _marked_body(agent: str) -> dict[str, Any]:
    return {
        "model": "local-smoke-model",
        "messages": [
            {
                "role": "system",
                "content": "\n\n".join(
                    [
                        "ROLE_SPECIFIC_INSTRUCTION_START\n"
                        f"AGENT_NAME: {agent}\n"
                        f"You are {agent}.\n"
                        "ROLE_SPECIFIC_INSTRUCTION_END",
                        "USER_TASK_START\nBuild the cache experiment.\nUSER_TASK_END",
                        "SHARED_GROUPCHAT_CONTEXT_START\nShared benchmark context.\nSHARED_GROUPCHAT_CONTEXT_END",
                        "TEAM_POLICY_START\nDo not move private memory or latest user instructions.\nTEAM_POLICY_END",
                        "TOOL_SCHEMA_START\nrecord_metric(name: string, value: number) -> string\nTOOL_SCHEMA_END",
                        "OUTPUT_FORMAT_START\nUse a concise status format with evidence and next action.\nOUTPUT_FORMAT_END",
                        "CURRENT_TURN_INSTRUCTION_START\nReport status.\nCURRENT_TURN_INSTRUCTION_END",
                    ]
                ),
            }
        ],
        "temperature": 0,
    }


def _natural_language_bodies() -> tuple[dict[str, Any], dict[str, Any]]:
    body = {
        "model": "local-smoke-model",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant.\n"
                    "When using code, indicate the script type in the code block.\n"
                    "Verify the answer carefully and include evidence.\n"
                    "Reply with concise final results."
                ),
            }
        ],
        "temperature": 0,
    }
    return (dict(body), dict(body))


if __name__ == "__main__":
    raise SystemExit(main())
