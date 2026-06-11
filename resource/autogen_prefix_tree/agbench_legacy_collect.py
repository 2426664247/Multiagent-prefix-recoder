from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ab_eval import evaluate_ab_traces, render_markdown_report


VARIANTS = ("baseline", "plugin_rule_only", "plugin_nl_segmentation")


@dataclass(frozen=True)
class LegacyAgBenchCollectionResult:
    manifest_path: str
    summary_path: str | None
    summary: dict[str, Any]


def collect_legacy_agbench_results(
    *,
    manifest_path: str | Path,
    summary_path: str | Path | None = None,
    baseline_tabulate_csv: str | Path | None = None,
    rule_tabulate_csv: str | Path | None = None,
    nl_tabulate_csv: str | Path | None = None,
    baseline_provider_telemetry: str | Path | None = None,
    rule_provider_telemetry: str | Path | None = None,
    nl_provider_telemetry: str | Path | None = None,
    run_ab_eval: bool = True,
) -> LegacyAgBenchCollectionResult:
    manifest_file = Path(manifest_path)
    manifest = _load_json_mapping(manifest_file)
    configs = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    reports_dir = _manifest_path(manifest.get("reports_dir"), manifest_file=manifest_file)
    variant_status: dict[str, Any] = {}

    csv_overrides = {
        "baseline": baseline_tabulate_csv,
        "plugin_rule_only": rule_tabulate_csv,
        "plugin_nl_segmentation": nl_tabulate_csv,
    }
    provider_overrides = {
        "baseline": baseline_provider_telemetry,
        "plugin_rule_only": rule_provider_telemetry,
        "plugin_nl_segmentation": nl_provider_telemetry,
    }

    for label in VARIANTS:
        info = configs.get(label) if isinstance(configs.get(label), Mapping) else {}
        if not info:
            variant_status[label] = {"supported": False, "reason": "missing_manifest_variant"}
            continue
        task_status = _collect_task_results(
            label=label,
            info=info,
            csv_override=csv_overrides[label],
            manifest_file=manifest_file,
        )
        provider_status = _collect_provider_telemetry(
            label=label,
            info=info,
            source_override=provider_overrides[label],
            manifest_file=manifest_file,
        )
        variant_status[label] = {
            "supported": True,
            "task_results": task_status,
            "provider_telemetry": provider_status,
        }

    comparison_status = _run_comparisons(
        manifest=manifest,
        reports_dir=reports_dir,
        manifest_file=manifest_file,
        run_ab_eval=run_ab_eval,
    )
    artifact_provenance = _artifact_provenance(variants=variant_status, comparisons=comparison_status)
    summary = {
        "schema_version": "prefix-legacy-agbench-collection-v1",
        "manifest_path": str(manifest_file),
        "suite_id": manifest.get("suite_id"),
        "prompt_safe_summary": True,
        "variants": variant_status,
        "comparisons": comparison_status,
        "artifact_provenance": artifact_provenance,
        "fake_upstream": artifact_provenance["fake_upstream_detected"],
        "ready_for_ab_report": all(
            bool(item.get("status") == "ran") for item in comparison_status.values()
        )
        if comparison_status
        else False,
        "real_provider_metrics_available": (not artifact_provenance["fake_upstream_detected"])
        and all(
            bool((item.get("ab_summary") or {}).get("real_provider_metrics_available"))
            for item in comparison_status.values()
            if isinstance(item, Mapping) and item.get("status") == "ran"
        )
        if comparison_status
        else False,
        "task_metrics_available": all(
            bool((item.get("ab_summary") or {}).get("task_metrics_available"))
            for item in comparison_status.values()
            if isinstance(item, Mapping) and item.get("status") == "ran"
        )
        if comparison_status
        else False,
        "notes": (
            "This collector converts AutoGenBench tabulate CSV into task-result JSONL and optionally runs ab_eval. "
            "It stores aggregate metadata only and does not read prompt text."
        ),
    }
    target = _write_json(summary_path, summary)
    return LegacyAgBenchCollectionResult(
        manifest_path=str(manifest_file),
        summary_path=str(target) if target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect legacy AutoGenBench 0.0.3 tabulate/provider artifacts into manifest paths and run A/B summaries."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--baseline-tabulate-csv")
    parser.add_argument("--rule-tabulate-csv")
    parser.add_argument("--nl-tabulate-csv")
    parser.add_argument("--baseline-provider-telemetry")
    parser.add_argument("--rule-provider-telemetry")
    parser.add_argument("--nl-provider-telemetry")
    parser.add_argument("--skip-ab-eval", action="store_true")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Return non-zero if any task/provider artifact or A/B report is missing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = collect_legacy_agbench_results(
        manifest_path=args.manifest,
        summary_path=args.summary,
        baseline_tabulate_csv=args.baseline_tabulate_csv,
        rule_tabulate_csv=args.rule_tabulate_csv,
        nl_tabulate_csv=args.nl_tabulate_csv,
        baseline_provider_telemetry=args.baseline_provider_telemetry,
        rule_provider_telemetry=args.rule_provider_telemetry,
        nl_provider_telemetry=args.nl_provider_telemetry,
        run_ab_eval=not args.skip_ab_eval,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    if args.require_artifacts and not result.summary.get("ready_for_ab_report"):
        return 1
    return 0


def _collect_task_results(
    *,
    label: str,
    info: Mapping[str, Any],
    csv_override: str | Path | None,
    manifest_file: Path,
) -> dict[str, Any]:
    csv_path = _artifact_path(csv_override or info.get("tabulate_csv_path"), manifest_file=manifest_file)
    target_path = _artifact_path(info.get("task_results_path"), manifest_file=manifest_file)
    if csv_path is None:
        return {"status": "skipped", "reason": "missing_tabulate_csv_path"}
    if target_path is None:
        return {"status": "skipped", "reason": "missing_task_results_path", "tabulate_csv_path": str(csv_path)}
    if not csv_path.exists():
        return {
            "status": "missing",
            "reason": "tabulate_csv_not_found",
            "tabulate_csv_path": str(csv_path),
            "task_results_path": str(target_path),
        }
    rows = parse_autogenbench_tabulate_csv(csv_path, variant=label)
    _write_jsonl(target_path, rows)
    success_values = [row.get("success") for row in rows if row.get("success") is not None]
    success_count = sum(1 for value in success_values if value is True)
    return {
        "status": "written",
        "tabulate_csv_path": str(csv_path),
        "task_results_path": str(target_path),
        "source_override_used": csv_override is not None,
        "source_path": str(csv_path),
        "target_path": str(target_path),
        "source_format": "autogenbench_tabulate_csv",
        "record_count": len(rows),
        "success_evaluated_count": len(success_values),
        "success_count": success_count,
        "success_rate": success_count / len(success_values) if success_values else None,
    }


def _collect_provider_telemetry(
    *,
    label: str,
    info: Mapping[str, Any],
    source_override: str | Path | None,
    manifest_file: Path,
) -> dict[str, Any]:
    target_path = _artifact_path(info.get("provider_telemetry_path"), manifest_file=manifest_file)
    if target_path is None:
        return {"status": "skipped", "reason": "missing_provider_telemetry_path"}
    source_path = target_path
    copied_from_source = False
    if source_override is not None:
        source_path = _artifact_path(source_override, manifest_file=manifest_file)
        if source_path is None or not source_path.exists():
            return {
                "status": "missing",
                "reason": "provider_telemetry_source_not_found",
                "provider_telemetry_path": str(target_path),
                "source_path": str(source_path) if source_path else None,
            }
        if source_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            copied_from_source = True
    exists = target_path.exists()
    size_bytes = target_path.stat().st_size if exists else 0
    provenance = _provider_telemetry_provenance(target_path) if exists and size_bytes > 0 else {}
    return {
        "status": "present" if exists and size_bytes > 0 else "missing",
        "provider_telemetry_path": str(target_path),
        "source_override_used": source_override is not None,
        "source_path": str(source_path) if source_path else None,
        "target_path": str(target_path),
        "copied_from_source": copied_from_source,
        "size_bytes": size_bytes,
        "variant": label,
        **provenance,
    }


def _run_comparisons(
    *,
    manifest: Mapping[str, Any],
    reports_dir: Path | None,
    manifest_file: Path,
    run_ab_eval: bool,
) -> dict[str, Any]:
    recommended = manifest.get("recommended_ab_eval") if isinstance(manifest.get("recommended_ab_eval"), Mapping) else {}
    comparisons = {
        "baseline_vs_rule_only": {
            "baseline": recommended.get("baseline_provider_telemetry"),
            "plugin": recommended.get("plugin_rule_only_provider_telemetry"),
            "baseline_tasks": recommended.get("baseline_task_results"),
            "plugin_tasks": recommended.get("plugin_rule_only_task_results"),
            "summary": recommended.get("rule_only_summary"),
            "report_md": recommended.get("rule_only_report_md"),
        },
        "baseline_vs_nl_segmentation": {
            "baseline": recommended.get("baseline_provider_telemetry"),
            "plugin": recommended.get("plugin_nl_segmentation_provider_telemetry"),
            "baseline_tasks": recommended.get("baseline_task_results"),
            "plugin_tasks": recommended.get("plugin_nl_segmentation_task_results"),
            "summary": recommended.get("nl_segmentation_summary"),
            "report_md": recommended.get("nl_segmentation_report_md"),
        },
    }
    result: dict[str, Any] = {}
    for label, paths in comparisons.items():
        resolved = {key: _artifact_path(value, manifest_file=manifest_file) for key, value in paths.items()}
        missing = [
            key
            for key in ("baseline", "plugin", "baseline_tasks", "plugin_tasks")
            if resolved.get(key) is None or not resolved[key].exists() or resolved[key].stat().st_size == 0
        ]
        if not run_ab_eval:
            result[label] = {"status": "skipped", "reason": "ab_eval_disabled", "missing_inputs": missing}
            continue
        if missing:
            result[label] = {"status": "skipped", "reason": "missing_inputs", "missing_inputs": missing}
            continue
        summary_path = resolved.get("summary")
        if summary_path is None and reports_dir is not None:
            summary_path = reports_dir / f"{label}_summary.json"
        report_path = resolved.get("report_md")
        if report_path is None and reports_dir is not None:
            report_path = reports_dir / f"{label}_report.md"
        ab_result = evaluate_ab_traces(
            baseline_path=resolved["baseline"],
            plugin_path=resolved["plugin"],
            baseline_tasks_path=resolved["baseline_tasks"],
            plugin_tasks_path=resolved["plugin_tasks"],
            summary_path=summary_path,
            report_path=report_path,
        )
        input_provenance = {
            key: _comparison_input_provenance(resolved[key])
            for key in ("baseline", "plugin", "baseline_tasks", "plugin_tasks")
            if resolved.get(key) is not None
        }
        fake_upstream = any(
            bool(item.get("fake_upstream_detected"))
            for key, item in input_provenance.items()
            if key in {"baseline", "plugin"}
        )
        if fake_upstream:
            ab_result.summary["fake_upstream"] = True
            ab_result.summary["real_provider_metrics_available"] = False
            ab_result.summary["real_provider_metrics_note"] = (
                "At least one provider telemetry input was marked fake_upstream=true. "
                "This A/B summary validates aggregation only and must not be used as real provider evidence."
            )
            if ab_result.summary_path:
                _write_json(ab_result.summary_path, ab_result.summary)
            if ab_result.report_path:
                _write_text(ab_result.report_path, render_markdown_report(ab_result.summary))
        result[label] = {
            "status": "ran",
            "summary_path": ab_result.summary_path,
            "report_path": ab_result.report_path,
            "input_provenance": input_provenance,
            "ab_summary": {
                "real_provider_metrics_available": ab_result.summary.get("real_provider_metrics_available"),
                "task_metrics_available": ab_result.summary.get("task_metrics_available"),
                "overall_gate_status": (ab_result.summary.get("gates") or {}).get("overall_status")
                if isinstance(ab_result.summary.get("gates"), Mapping)
                else None,
                "fake_upstream": bool(ab_result.summary.get("fake_upstream")),
            },
        }
    return result


def _artifact_provenance(*, variants: Mapping[str, Any], comparisons: Mapping[str, Any]) -> dict[str, Any]:
    variant_provenance: dict[str, Any] = {}
    fake_variants: list[str] = []
    real_provider_variants: list[str] = []
    task_metric_variants: list[str] = []
    for label in VARIANTS:
        status = variants.get(label) if isinstance(variants.get(label), Mapping) else {}
        provider = status.get("provider_telemetry") if isinstance(status.get("provider_telemetry"), Mapping) else {}
        task = status.get("task_results") if isinstance(status.get("task_results"), Mapping) else {}
        provider_provenance = {
            "path": provider.get("provider_telemetry_path"),
            "source_path": provider.get("source_path"),
            "source_override_used": bool(provider.get("source_override_used")),
            "copied_from_source": bool(provider.get("copied_from_source")),
            "status": provider.get("status"),
            "size_bytes": provider.get("size_bytes", 0),
            "json_record_count": provider.get("json_record_count", 0),
            "provider_record_count": provider.get("provider_record_count", 0),
            "fake_upstream_detected": bool(provider.get("fake_upstream_detected")),
            "fake_upstream_record_count": provider.get("fake_upstream_record_count", 0),
            "provider_usage_available": bool(provider.get("provider_usage_available")),
            "cached_token_metric_available": bool(provider.get("cached_token_metric_available")),
            "latency_metric_available": bool(provider.get("latency_metric_available")),
            "cost_metric_available": bool(provider.get("cost_metric_available")),
            "real_provider_metrics_available": bool(provider.get("real_provider_metrics_available")),
        }
        task_provenance = {
            "path": task.get("task_results_path"),
            "source_path": task.get("source_path") or task.get("tabulate_csv_path"),
            "source_override_used": bool(task.get("source_override_used")),
            "status": task.get("status"),
            "source_format": task.get("source_format"),
            "record_count": task.get("record_count", 0),
            "success_evaluated_count": task.get("success_evaluated_count", 0),
            "task_metrics_available": _num(task.get("success_evaluated_count")) > 0,
        }
        if provider_provenance["fake_upstream_detected"]:
            fake_variants.append(label)
        if provider_provenance["real_provider_metrics_available"]:
            real_provider_variants.append(label)
        if task_provenance["task_metrics_available"]:
            task_metric_variants.append(label)
        variant_provenance[label] = {
            "provider_telemetry": provider_provenance,
            "task_results": task_provenance,
        }

    comparison_provenance = {
        str(label): {
            "status": item.get("status") if isinstance(item, Mapping) else None,
            "summary_path": item.get("summary_path") if isinstance(item, Mapping) else None,
            "report_path": item.get("report_path") if isinstance(item, Mapping) else None,
            "fake_upstream": bool(((item.get("ab_summary") or {}) if isinstance(item, Mapping) else {}).get("fake_upstream")),
            "real_provider_metrics_available": bool(
                ((item.get("ab_summary") or {}) if isinstance(item, Mapping) else {}).get(
                    "real_provider_metrics_available"
                )
            ),
            "task_metrics_available": bool(
                ((item.get("ab_summary") or {}) if isinstance(item, Mapping) else {}).get("task_metrics_available")
            ),
            "input_provenance": item.get("input_provenance") if isinstance(item, Mapping) else None,
        }
        for label, item in comparisons.items()
    }
    return {
        "schema_version": "prefix-legacy-artifact-provenance-v1",
        "prompt_safe": True,
        "variant_count": len(variant_provenance),
        "variants": variant_provenance,
        "comparisons": comparison_provenance,
        "fake_upstream_detected": bool(fake_variants),
        "fake_upstream_variants": fake_variants,
        "real_provider_metrics_available": len(real_provider_variants) == len(VARIANTS),
        "real_provider_metric_variants": real_provider_variants,
        "task_metrics_available": len(task_metric_variants) == len(VARIANTS),
        "task_metric_variants": task_metric_variants,
    }


def _provider_telemetry_provenance(path: Path) -> dict[str, Any]:
    rows = _load_jsonl_mappings(path)
    provider_rows = [row for row in rows if _looks_like_provider_telemetry(row)]
    fake_count = sum(1 for row in rows if row.get("fake_upstream") is True)
    provider_usage_count = sum(1 for row in provider_rows if _provider_usage_available(row))
    cached_metric_count = sum(1 for row in provider_rows if _cached_token_metric_available(row))
    latency_count = sum(1 for row in provider_rows if row.get("latency_seconds") is not None or row.get("latency_ms") is not None)
    cost_count = sum(1 for row in provider_rows if _cost_metric_available(row))
    return {
        "json_record_count": len(rows),
        "provider_record_count": len(provider_rows),
        "fake_upstream_detected": fake_count > 0,
        "fake_upstream_record_count": fake_count,
        "provider_usage_available": provider_usage_count > 0,
        "provider_usage_record_count": provider_usage_count,
        "cached_token_metric_available": cached_metric_count > 0,
        "cached_token_metric_record_count": cached_metric_count,
        "latency_metric_available": latency_count > 0,
        "latency_metric_record_count": latency_count,
        "cost_metric_available": cost_count > 0,
        "cost_metric_record_count": cost_count,
        "real_provider_metrics_available": bool(provider_rows) and fake_count == 0,
    }


def _comparison_input_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": 0}
    exists = path.exists()
    size_bytes = path.stat().st_size if exists else 0
    result = {"path": str(path), "exists": exists, "size_bytes": size_bytes}
    if exists and path.suffix.lower() == ".jsonl":
        result.update(_provider_telemetry_provenance(path))
    return result


def _load_jsonl_mappings(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return tuple(rows)


def _looks_like_provider_telemetry(record: Mapping[str, Any]) -> bool:
    return any(
        key in record
        for key in (
            "actual_prompt_tokens",
            "actual_cached_tokens",
            "actual_completion_tokens",
            "actual_total_tokens",
            "latency_seconds",
            "latency_ms",
            "actual_cost_usd",
            "usage",
        )
    )


def _provider_usage_available(record: Mapping[str, Any]) -> bool:
    if any(
        record.get(key) is not None
        for key in ("actual_prompt_tokens", "actual_completion_tokens", "actual_total_tokens")
    ):
        return True
    usage = record.get("usage")
    return isinstance(usage, Mapping) and any(
        usage.get(key) is not None for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
    )


def _cached_token_metric_available(record: Mapping[str, Any]) -> bool:
    if any(
        record.get(key) is not None
        for key in (
            "actual_cached_tokens",
            "cached_tokens",
            "cached_prompt_tokens",
            "prompt_cache_hit_tokens",
            "cache_hit_tokens",
            "input_cached_tokens",
        )
    ):
        return True
    usage = record.get("usage")
    if not isinstance(usage, Mapping):
        return False
    for detail_key in ("prompt_tokens_details", "input_token_details", "input_tokens_details"):
        detail = usage.get(detail_key)
        if isinstance(detail, Mapping) and any(detail.get(key) is not None for key in ("cached_tokens", "cache_read", "cached")):
            return True
    return False


def _cost_metric_available(record: Mapping[str, Any]) -> bool:
    return any(record.get(key) is not None for key in ("actual_cost_usd", "cost_usd", "cost", "total_cost_usd", "total_cost"))


def parse_autogenbench_tabulate_csv(path: str | Path, *, variant: str) -> tuple[dict[str, Any], ...]:
    source_path = Path(path)
    lines = [line for line in source_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        return ()
    reader = csv.DictReader(lines)
    fieldnames = tuple(reader.fieldnames or ())
    task_field = _task_id_field(fieldnames)
    trial_indexes = _trial_indexes(fieldnames)
    rows: list[dict[str, Any]] = []
    for row in reader:
        raw_task_id = str(row.get(task_field) or "").strip()
        if not raw_task_id:
            continue
        for trial_index in trial_indexes:
            success_key = f"Trial {trial_index} Success"
            time_key = f"Trial {trial_index} Time"
            success = _parse_bool(row.get(success_key))
            if success is None and not str(row.get(success_key) or "").strip():
                continue
            rows.append(
                {
                    "task_id": f"{raw_task_id}::trial_{trial_index}",
                    "original_task_id": raw_task_id,
                    "trial_index": trial_index,
                    "success": success,
                    "passed": success,
                    "time_seconds": _parse_float(row.get(time_key)),
                    "variant": variant,
                    "source_format": "autogenbench_tabulate_csv",
                }
            )
    return tuple(rows)


def _task_id_field(fieldnames: Sequence[str]) -> str:
    for field in fieldnames:
        if field.strip().lower().replace(" ", "_") in {"task_id", "taskid"}:
            return field
    raise ValueError("tabulate CSV must include a Task Id column")


def _trial_indexes(fieldnames: Sequence[str]) -> tuple[int, ...]:
    indexes: list[int] = []
    for field in fieldnames:
        match = re.fullmatch(r"\s*Trial\s+(\d+)\s+Success\s*", field)
        if match:
            indexes.append(int(match.group(1)))
    return tuple(sorted(indexes))


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1", "success", "passed", "pass"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", "failure", "failed", "fail"}:
        return False
    return None


def _parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _artifact_path(value: Any, *, manifest_file: Path) -> Path | None:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value:
        path = Path(value)
    else:
        return None
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    manifest_relative = manifest_file.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return cwd_path


def _manifest_path(value: Any, *, manifest_file: Path) -> Path | None:
    return _artifact_path(value, manifest_file=manifest_file)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
