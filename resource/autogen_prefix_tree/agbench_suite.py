from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agbench_config import _read_structured, _write_structured, build_agbench_config


@dataclass(frozen=True)
class AgBenchSuiteResult:
    output_dir: str
    manifest_path: str
    manifest: dict[str, Any]


def generate_agbench_suite(
    *,
    base_config_path: str | Path,
    output_dir: str | Path,
    suite_id: str,
    include_message_content: bool = True,
    include_guard: bool = False,
    semantic_guard_config: Mapping[str, Any] | None = None,
    guard_enable_natural_language_segmentation: bool = True,
) -> AgBenchSuiteResult:
    base_path = Path(base_config_path)
    out = Path(output_dir)
    artifacts = out / "artifacts"
    reports = out / "reports"
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    if include_guard and semantic_guard_config is None:
        raise ValueError("semantic_guard_config is required when include_guard=True")

    base_config = _read_structured(base_path)
    variants: dict[str, dict[str, Any]] = {}

    _add_variant(
        variants,
        label="baseline",
        config_path=out / "baseline.yaml",
        config=build_agbench_config(base_config, mode="baseline", session_id=f"{suite_id}-baseline"),
        run_label=f"{suite_id}-baseline",
        mode="baseline",
        capture_log_path=None,
        telemetry_log_path=None,
        provider_telemetry_path=artifacts / "baseline_provider_telemetry.jsonl",
        task_results_path=artifacts / "baseline_task_results.jsonl",
        natural_language_segmentation_enabled=False,
        semantic_guard_enabled=False,
        include_message_content=include_message_content,
    )
    _add_variant(
        variants,
        label="capture",
        config_path=out / "capture.yaml",
        config=build_agbench_config(
            base_config,
            mode="capture",
            session_id=f"{suite_id}-capture",
            capture_log_path=str(artifacts / "capture_messages.jsonl"),
            include_message_content=include_message_content,
        ),
        run_label=f"{suite_id}-capture",
        mode="capture",
        capture_log_path=artifacts / "capture_messages.jsonl",
        telemetry_log_path=None,
        provider_telemetry_path=None,
        task_results_path=None,
        natural_language_segmentation_enabled=False,
        semantic_guard_enabled=False,
        include_message_content=include_message_content,
    )
    _add_variant(
        variants,
        label="plugin_rule_only",
        config_path=out / "plugin_rule_only.yaml",
        config=build_agbench_config(
            base_config,
            mode="plugin",
            session_id=f"{suite_id}-plugin-rule-only",
            capture_log_path=str(artifacts / "plugin_rule_only_messages.jsonl"),
            telemetry_log_path=str(artifacts / "plugin_rule_only_telemetry.jsonl"),
            include_message_content=include_message_content,
            enable_natural_language_segmentation=False,
        ),
        run_label=f"{suite_id}-plugin-rule-only",
        mode="plugin",
        capture_log_path=artifacts / "plugin_rule_only_messages.jsonl",
        telemetry_log_path=artifacts / "plugin_rule_only_telemetry.jsonl",
        provider_telemetry_path=artifacts / "plugin_rule_only_provider_telemetry.jsonl",
        task_results_path=artifacts / "plugin_rule_only_task_results.jsonl",
        natural_language_segmentation_enabled=False,
        semantic_guard_enabled=False,
        include_message_content=include_message_content,
    )
    _add_variant(
        variants,
        label="plugin_nl_segmentation",
        config_path=out / "plugin_nl_segmentation.yaml",
        config=build_agbench_config(
            base_config,
            mode="plugin",
            session_id=f"{suite_id}-plugin-nl-segmentation",
            capture_log_path=str(artifacts / "plugin_nl_segmentation_messages.jsonl"),
            telemetry_log_path=str(artifacts / "plugin_nl_segmentation_telemetry.jsonl"),
            include_message_content=include_message_content,
            enable_natural_language_segmentation=True,
        ),
        run_label=f"{suite_id}-plugin-nl-segmentation",
        mode="plugin",
        capture_log_path=artifacts / "plugin_nl_segmentation_messages.jsonl",
        telemetry_log_path=artifacts / "plugin_nl_segmentation_telemetry.jsonl",
        provider_telemetry_path=artifacts / "plugin_nl_segmentation_provider_telemetry.jsonl",
        task_results_path=artifacts / "plugin_nl_segmentation_task_results.jsonl",
        natural_language_segmentation_enabled=True,
        semantic_guard_enabled=False,
        include_message_content=include_message_content,
    )
    if include_guard:
        assert semantic_guard_config is not None
        _add_variant(
            variants,
            label="plugin_guard",
            config_path=out / "plugin_guard.yaml",
            config=build_agbench_config(
                base_config,
                mode="plugin",
                session_id=f"{suite_id}-plugin-guard",
                capture_log_path=str(artifacts / "plugin_guard_messages.jsonl"),
                telemetry_log_path=str(artifacts / "plugin_guard_telemetry.jsonl"),
                include_message_content=include_message_content,
                enable_natural_language_segmentation=guard_enable_natural_language_segmentation,
                semantic_guard_config=semantic_guard_config,
            ),
            run_label=f"{suite_id}-plugin-guard",
            mode="plugin",
            capture_log_path=artifacts / "plugin_guard_messages.jsonl",
            telemetry_log_path=artifacts / "plugin_guard_telemetry.jsonl",
            provider_telemetry_path=artifacts / "plugin_guard_provider_telemetry.jsonl",
            task_results_path=artifacts / "plugin_guard_task_results.jsonl",
            natural_language_segmentation_enabled=guard_enable_natural_language_segmentation,
            semantic_guard_enabled=True,
            include_message_content=include_message_content,
        )
    _add_variant(
        variants,
        label="static_capture",
        config_path=out / "static_capture.yaml",
        config=build_agbench_config(
            base_config,
            mode="static_capture",
            session_id=f"{suite_id}-static-capture",
            capture_log_path=str(artifacts / "static_capture_messages.jsonl"),
            include_message_content=include_message_content,
        ),
        run_label=f"{suite_id}-static-capture",
        mode="static_capture",
        capture_log_path=artifacts / "static_capture_messages.jsonl",
        telemetry_log_path=None,
        provider_telemetry_path=None,
        task_results_path=None,
        natural_language_segmentation_enabled=False,
        semantic_guard_enabled=False,
        include_message_content=include_message_content,
    )

    for variant in variants.values():
        _write_structured(Path(variant["config_path"]), variant.pop("_config"))

    manifest = _build_manifest(
        base_config_path=base_path,
        output_dir=out,
        artifacts_dir=artifacts,
        reports_dir=reports,
        suite_id=suite_id,
        variants=variants,
        include_message_content=include_message_content,
        include_guard=include_guard,
    )
    manifest_path = out / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return AgBenchSuiteResult(output_dir=str(out), manifest_path=str(manifest_path), manifest=manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a prompt-safe AutoGenBench A/B config suite for prefix reorder evaluation."
    )
    parser.add_argument("--base-config", required=True, help="Input YAML/JSON config containing model_config.")
    parser.add_argument("--output-dir", required=True, help="Directory that will receive configs and manifest.")
    parser.add_argument("--suite-id", required=True, help="Stable label used in sessions and output paths.")
    parser.add_argument("--redact-message-content", action="store_true")
    parser.add_argument("--include-guard", action="store_true", help="Also generate plugin_guard.yaml.")
    parser.add_argument(
        "--guard-rule-only",
        action="store_true",
        help="Generate plugin_guard.yaml without natural-language segmentation. Default keeps it enabled.",
    )
    parser.add_argument("--semantic-guard-base-url", help="Local OpenAI-compatible semantic guard base URL.")
    parser.add_argument("--semantic-guard-model", help="Model name for --semantic-guard-base-url.")
    parser.add_argument("--semantic-guard-timeout", type=float, default=30.0)
    parser.add_argument("--semantic-guard-min-confidence", type=float, default=0.75)
    parser.add_argument("--semantic-guard-max-message-chars", type=int, default=12000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_agbench_suite(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        suite_id=args.suite_id,
        include_message_content=not args.redact_message_content,
        include_guard=args.include_guard,
        semantic_guard_config=_semantic_guard_config_from_args(args) if args.include_guard else None,
        guard_enable_natural_language_segmentation=not args.guard_rule_only,
    )
    print(json.dumps(result.manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _add_variant(
    variants: dict[str, dict[str, Any]],
    *,
    label: str,
    config_path: Path,
    config: Mapping[str, Any],
    run_label: str,
    mode: str,
    capture_log_path: Path | None,
    telemetry_log_path: Path | None,
    provider_telemetry_path: Path | None,
    task_results_path: Path | None,
    natural_language_segmentation_enabled: bool,
    semantic_guard_enabled: bool,
    include_message_content: bool,
) -> None:
    variants[label] = {
        "label": label,
        "run_label": run_label,
        "mode": mode,
        "config_path": str(config_path),
        "capture_log_path": str(capture_log_path) if capture_log_path is not None else None,
        "telemetry_log_path": str(telemetry_log_path) if telemetry_log_path is not None else None,
        "provider_telemetry_path": str(provider_telemetry_path) if provider_telemetry_path is not None else None,
        "task_results_path": str(task_results_path) if task_results_path is not None else None,
        "natural_language_segmentation_enabled": natural_language_segmentation_enabled,
        "semantic_guard_enabled": semantic_guard_enabled,
        "captures_message_content": bool(capture_log_path is not None and include_message_content),
        "_config": dict(config),
    }


def _build_manifest(
    *,
    base_config_path: Path,
    output_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    suite_id: str,
    variants: Mapping[str, Mapping[str, Any]],
    include_message_content: bool,
    include_guard: bool,
) -> dict[str, Any]:
    baseline = variants["baseline"]
    rule = variants["plugin_rule_only"]
    nl = variants["plugin_nl_segmentation"]
    manifest = {
        "schema_version": "prefix-agbench-suite-manifest-v1",
        "suite_id": suite_id,
        "base_config_path": str(base_config_path),
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "reports_dir": str(reports_dir),
        "config_variants": dict(variants),
        "prompt_safe_manifest": True,
        "capture_files_may_contain_prompt_text": include_message_content,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "This suite only generates configs and expected artifact paths. "
            "Cached tokens, latency, cost, and task success require real provider/benchmark runs."
        ),
        "readiness_prerequisites": [
            "Run commands through the repository .venv interpreter.",
            "Install a matching autogen-ext[openai] package for real OpenAI-compatible clients.",
            "Set AUTOGEN_ALLOWED_PROVIDER_NAMESPACES=autogen_prefix_tree before loading wrapper configs.",
            "Provide OPENAI_API_KEY or OAI_CONFIG_LIST before real provider evaluation.",
            "Keep capture, telemetry, task result, and provider trace files out of git.",
        ],
        "suggested_commands": _suggested_commands(
            suite_id=suite_id,
            output_dir=output_dir,
            reports_dir=reports_dir,
            baseline=baseline,
            rule=rule,
            nl=nl,
            variants=variants,
        ),
        "recommended_comparisons": [
            {"baseline": "baseline", "candidate": "plugin_rule_only"},
            {"baseline": "baseline", "candidate": "plugin_nl_segmentation"},
        ],
    }
    if include_guard:
        manifest["recommended_comparisons"].append({"baseline": "baseline", "candidate": "plugin_guard"})
    return manifest


def _suggested_commands(
    *,
    suite_id: str,
    output_dir: Path,
    reports_dir: Path,
    baseline: Mapping[str, Any],
    rule: Mapping[str, Any],
    nl: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {
        "environment": [
            rf'$env:PYTHONPATH = "{Path.cwd()}"',
            r'$env:AUTOGEN_ALLOWED_PROVIDER_NAMESPACES = "autogen_prefix_tree"',
            r".venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --mode component --allow-missing-api",
        ],
        "autogenbench_note": [
            "Use the generated config_path for the scenario-specific AutoGenBench runner or task template.",
            "Run baseline, plugin_rule_only, and plugin_nl_segmentation with the same task subset, repeat count, and model.",
        ],
        "dataset_eval": [],
        "offline_semantic_suite": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_suite --source-root <autogen-source-root> --output-dir {output_dir}\offline_semantic_suite --session-id {suite_id}-offline-semantic-suite",
        ],
        "offline_compare": [
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.offline_compare --baseline {output_dir}\offline_rule_only\pipeline_summary.json --candidate {output_dir}\offline_nl_segmentation\pipeline_summary.json --baseline-label rule-only --candidate-label nl-segmentation --summary {reports_dir}\offline_rule_vs_nl_summary.json --report-md {reports_dir}\offline_rule_vs_nl_report.md",
        ],
        "ab_eval": [
            _ab_eval_command(
                baseline=baseline,
                candidate=rule,
                summary_path=reports_dir / "ab_baseline_vs_rule_only_summary.json",
                report_path=reports_dir / "ab_baseline_vs_rule_only_report.md",
            ),
            _ab_eval_command(
                baseline=baseline,
                candidate=nl,
                summary_path=reports_dir / "ab_baseline_vs_nl_segmentation_summary.json",
                report_path=reports_dir / "ab_baseline_vs_nl_segmentation_report.md",
            ),
        ],
    }
    for label in ("capture", "plugin_rule_only", "plugin_nl_segmentation", "plugin_guard", "static_capture"):
        variant = variants.get(label)
        if not variant or not variant.get("capture_log_path"):
            continue
        commands["dataset_eval"].append(
            rf".venv\Scripts\python.exe -m autogen_prefix_tree.dataset_eval --input {variant['capture_log_path']} --telemetry {reports_dir}\{label}_dataset_telemetry.jsonl --summary {reports_dir}\{label}_dataset_summary.json --require-messages"
        )
    guard = variants.get("plugin_guard")
    if guard is not None:
        commands["ab_eval"].append(
            _ab_eval_command(
                baseline=baseline,
                candidate=guard,
                summary_path=reports_dir / "ab_baseline_vs_guard_summary.json",
                report_path=reports_dir / "ab_baseline_vs_guard_report.md",
            )
        )
    commands["suite"] = [
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_suite --base-config <base.yaml> --output-dir {output_dir} --suite-id {suite_id}",
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_suite_verify --manifest {output_dir}\suite_manifest.json --summary {reports_dir}\suite_verification_summary.json",
    ]
    return commands


def _ab_eval_command(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    summary_path: Path,
    report_path: Path,
) -> str:
    return (
        rf".venv\Scripts\python.exe -m autogen_prefix_tree.ab_eval "
        rf"--baseline {baseline['provider_telemetry_path']} "
        rf"--plugin {candidate['provider_telemetry_path']} "
        rf"--baseline-tasks {baseline['task_results_path']} "
        rf"--plugin-tasks {candidate['task_results_path']} "
        rf"--summary {summary_path} "
        rf"--report-md {report_path} "
        r"--min-success-rate-delta -0.02 "
        r"--max-p95-latency-relative-change 0.0 "
        r"--max-cost-per-success-relative-change 0.0 "
        r"--bootstrap-iterations 1000 "
        r"--bootstrap-seed 20260605"
    )


def _semantic_guard_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.semantic_guard_base_url:
        raise ValueError("--semantic-guard-base-url is required with --include-guard")
    if not args.semantic_guard_model:
        raise ValueError("--semantic-guard-model is required with --include-guard")
    return {
        "provider": "openai-compatible",
        "base_url": args.semantic_guard_base_url,
        "model": args.semantic_guard_model,
        "timeout_seconds": args.semantic_guard_timeout,
        "min_confidence": args.semantic_guard_min_confidence,
        "max_message_chars": args.semantic_guard_max_message_chars,
    }


if __name__ == "__main__":
    raise SystemExit(main())
