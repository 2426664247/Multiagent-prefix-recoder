from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogen import config_list_from_json


@dataclass(frozen=True)
class LegacyAgBenchSuiteVerificationResult:
    manifest_path: str
    summary_path: str | None
    summary: dict[str, Any]


def verify_legacy_agbench_suite(
    *,
    manifest_path: str | Path,
    summary_path: str | Path | None = None,
    require_provider_artifacts: bool = False,
    require_task_artifacts: bool = False,
) -> LegacyAgBenchSuiteVerificationResult:
    manifest_file = Path(manifest_path)
    manifest = _load_json_mapping(manifest_file)
    checks: list[dict[str, Any]] = []
    artifact_status: list[dict[str, Any]] = []
    _check_manifest(manifest, checks)
    _check_proxies(manifest, checks)
    _check_scenario_files(manifest, checks)
    config_lists = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    for label in ("baseline", "plugin_rule_only", "plugin_nl_segmentation", "combined"):
        _add_check(
            checks,
            name=f"oai_config_list:{label}",
            ok=label in config_lists,
            detail="present" if label in config_lists else "missing",
            next_step="Regenerate the legacy suite." if label not in config_lists else None,
        )
    for label, raw_config in config_lists.items():
        if not isinstance(raw_config, Mapping):
            _add_check(checks, name=f"oai_config_list:{label}:mapping", ok=False, detail="not a mapping")
            continue
        _verify_oai_config_list(label=str(label), info=raw_config, checks=checks)
        _collect_artifacts(
            label=str(label),
            info=raw_config,
            checks=checks,
            artifact_status=artifact_status,
            require_provider_artifacts=require_provider_artifacts,
            require_task_artifacts=require_task_artifacts,
        )
    summary = {
        "schema_version": "prefix-legacy-agbench-suite-verification-v1",
        "manifest_path": str(manifest_file),
        "suite_id": manifest.get("suite_id"),
        "ready": all(bool(check.get("ok")) for check in checks),
        "checks": checks,
        "artifact_status": artifact_status,
        "requirements": {
            "provider_artifacts_required": require_provider_artifacts,
            "task_artifacts_required": require_task_artifacts,
        },
        "prompt_safe_summary": True,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": (
            "This verification checks legacy OAI_CONFIG_LIST structure and local artifact presence. "
            "It does not run AutoGenBench or prove provider cached tokens, latency, cost, or task success."
        ),
    }
    target = _write_json(summary_path, summary)
    return LegacyAgBenchSuiteVerificationResult(
        manifest_path=str(manifest_file),
        summary_path=str(target) if target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a generated legacy AutoGenBench OAI_CONFIG_LIST suite.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--require-provider-artifacts", action="store_true")
    parser.add_argument("--require-task-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_legacy_agbench_suite(
        manifest_path=args.manifest,
        summary_path=args.summary,
        require_provider_artifacts=args.require_provider_artifacts,
        require_task_artifacts=args.require_task_artifacts,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.summary["ready"] else 1


def _check_manifest(manifest: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    _add_check(
        checks,
        name="manifest:schema_version",
        ok=manifest.get("schema_version") == "prefix-legacy-agbench-suite-manifest-v1",
        detail=str(manifest.get("schema_version")),
        next_step="Regenerate the legacy suite.",
    )
    _add_check(
        checks,
        name="manifest:prompt_safe",
        ok=manifest.get("prompt_safe_manifest") is True,
        detail=str(manifest.get("prompt_safe_manifest")),
    )
    _add_check(
        checks,
        name="manifest:secret_policy",
        ok=bool((manifest.get("secret_policy") or {}).get("api_key_value_not_written"))
        if isinstance(manifest.get("secret_policy"), Mapping)
        else False,
        detail=str(manifest.get("secret_policy")),
        next_step="Do not write real API keys into generated config files or manifests.",
    )
    _add_check(
        checks,
        name="manifest:pre_run_provider_metrics",
        ok=manifest.get("real_provider_metrics_available") is False,
        detail=str(manifest.get("real_provider_metrics_available")),
    )


def _check_proxies(manifest: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    proxies = manifest.get("proxies")
    _add_check(
        checks,
        name="manifest:proxies",
        ok=isinstance(proxies, Mapping),
        detail="present" if isinstance(proxies, Mapping) else "missing",
        next_step="Regenerate the legacy suite with dual proxy support.",
    )
    if not isinstance(proxies, Mapping):
        return
    baseline = proxies.get("baseline") if isinstance(proxies.get("baseline"), Mapping) else {}
    rule = proxies.get("plugin_rule_only") if isinstance(proxies.get("plugin_rule_only"), Mapping) else {}
    nl = proxies.get("plugin_nl_segmentation") if isinstance(proxies.get("plugin_nl_segmentation"), Mapping) else {}
    _add_check(
        checks,
        name="manifest:baseline_proxy_disabled",
        ok=baseline.get("rewrite_enabled") is False,
        detail=str(baseline.get("rewrite_enabled")),
    )
    _add_check(
        checks,
        name="manifest:rule_only_proxy_enabled",
        ok=rule.get("rewrite_enabled") is True,
        detail=str(rule.get("rewrite_enabled")),
    )
    _add_check(
        checks,
        name="manifest:nl_segmentation_proxy_enabled",
        ok=nl.get("rewrite_enabled") is True,
        detail=str(nl.get("rewrite_enabled")),
    )
    _add_check(
        checks,
        name="manifest:rule_only_nl_segmentation_disabled",
        ok=rule.get("natural_language_segmentation_enabled") is False,
        detail=str(rule.get("natural_language_segmentation_enabled")),
    )
    _add_check(
        checks,
        name="manifest:nl_segmentation_enabled",
        ok=nl.get("natural_language_segmentation_enabled") is True,
        detail=str(nl.get("natural_language_segmentation_enabled")),
    )
    ports = [baseline.get("port"), rule.get("port"), nl.get("port")]
    _add_check(
        checks,
        name="manifest:proxy_ports_distinct",
        ok=all(ports) and len(set(ports)) == 3,
        detail=f"baseline={baseline.get('port')}; rule={rule.get('port')}; nl={nl.get('port')}",
    )


def _check_scenario_files(manifest: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    scenario_files = manifest.get("scenario_files")
    _add_check(
        checks,
        name="manifest:scenario_files",
        ok=isinstance(scenario_files, Mapping),
        detail="present" if isinstance(scenario_files, Mapping) else "missing",
        next_step="Regenerate the legacy suite with --scenario-jsonl so A/B groups use distinct scenario filenames.",
    )
    if not isinstance(scenario_files, Mapping):
        return
    result_dir_names: list[str] = []
    for label in ("baseline", "plugin_rule_only", "plugin_nl_segmentation"):
        info = scenario_files.get(label) if isinstance(scenario_files.get(label), Mapping) else {}
        path_value = info.get("path")
        path = Path(path_value) if isinstance(path_value, str) and path_value else None
        exists = bool(path and path.exists())
        _add_check(
            checks,
            name=f"scenario_file:{label}:exists",
            ok=exists,
            detail=str(path) if path else "missing path",
            next_step="Regenerate the legacy suite with --scenario-jsonl.",
        )
        result_dir_name = info.get("result_dir_name")
        if isinstance(result_dir_name, str) and result_dir_name:
            result_dir_names.append(result_dir_name)
        _add_check(
            checks,
            name=f"scenario_file:{label}:template_paths_rewritten_absolute",
            ok=info.get("template_paths_rewritten_absolute") is True,
            detail=str(info.get("template_paths_rewritten_absolute")),
            next_step="Regenerate the legacy suite with --scenario-jsonl so copied scenarios can still find templates.",
        )
    _add_check(
        checks,
        name="scenario_files:result_dirs_distinct",
        ok=len(result_dir_names) == 3 and len(set(result_dir_names)) == 3,
        detail=", ".join(result_dir_names),
        next_step="Use distinct scenario filenames per A/B group; autogenbench 0.0.3 keys Results directories by scenario filename.",
    )


def _verify_oai_config_list(*, label: str, info: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    path_value = info.get("path")
    if not isinstance(path_value, str) or not path_value:
        _add_check(checks, name=f"oai_config_list:{label}:path", ok=False, detail="missing path")
        return
    path = Path(path_value)
    if not path.exists():
        _add_check(
            checks,
            name=f"oai_config_list:{label}:exists",
            ok=False,
            detail=str(path),
            next_step="Regenerate the legacy suite.",
        )
        return
    _add_check(checks, name=f"oai_config_list:{label}:exists", ok=True, detail=str(path))
    try:
        entries = config_list_from_json(str(path))
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, name=f"oai_config_list:{label}:load", ok=False, detail=f"{type(exc).__name__}: {exc}")
        return
    _add_check(checks, name=f"oai_config_list:{label}:load", ok=True, detail=f"{len(entries)} entries")
    expected_min = 2 if label == "combined" else 1
    _add_check(
        checks,
        name=f"oai_config_list:{label}:entry_count",
        ok=len(entries) >= expected_min,
        detail=str(len(entries)),
    )
    serialized = json.dumps(entries, ensure_ascii=False)
    _add_check(
        checks,
        name=f"oai_config_list:{label}:no_obvious_real_key",
        ok="sk-" not in serialized and "Bearer " not in serialized,
        detail="no obvious real key marker",
        next_step="Replace real API key values with placeholders before committing configs.",
    )
    for index, entry in enumerate(entries):
        _check_entry(label=label, index=index, entry=entry, checks=checks)
    expected_base_url = info.get("base_url")
    if label != "combined" and expected_base_url:
        _add_check(
            checks,
            name=f"oai_config_list:{label}:base_url_matches_manifest",
            ok=all(entry.get("base_url") == expected_base_url for entry in entries),
            detail=str(expected_base_url),
            next_step="Regenerate the legacy suite so OAI config and manifest agree.",
        )


def _check_entry(*, label: str, index: int, entry: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    prefix = f"oai_config_list:{label}:entry:{index}"
    _add_check(checks, name=f"{prefix}:model", ok=bool(entry.get("model")), detail=str(entry.get("model")))
    _add_check(checks, name=f"{prefix}:base_url", ok=bool(entry.get("base_url")), detail=str(entry.get("base_url")))
    _add_check(checks, name=f"{prefix}:api_key", ok=bool(entry.get("api_key")), detail="present")
    tags = entry.get("tags")
    _add_check(checks, name=f"{prefix}:tags", ok=isinstance(tags, list) and bool(tags), detail=str(tags))


def _collect_artifacts(
    *,
    label: str,
    info: Mapping[str, Any],
    checks: list[dict[str, Any]],
    artifact_status: list[dict[str, Any]],
    require_provider_artifacts: bool,
    require_task_artifacts: bool,
) -> None:
    for field, required in (
        ("provider_telemetry_path", require_provider_artifacts),
        ("task_results_path", require_task_artifacts),
    ):
        path_value = info.get(field)
        if not isinstance(path_value, str) or not path_value:
            continue
        path = Path(path_value)
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        artifact_status.append(
            {
                "variant": label,
                "field": field,
                "path": str(path),
                "exists": exists,
                "size_bytes": size_bytes,
                "required": required,
            }
        )
        if required:
            _add_check(
                checks,
                name=f"artifact:{label}:{field}",
                ok=exists and size_bytes > 0,
                detail=f"{path} ({size_bytes} bytes)" if exists else str(path),
                next_step="Run the matching AutoGenBench/proxy step before requiring this artifact.",
            )


def _add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    ok: bool,
    detail: str,
    next_step: str | None = None,
) -> None:
    check: dict[str, Any] = {"name": name, "ok": bool(ok), "detail": detail}
    if next_step and not ok:
        check["next_step"] = next_step
    checks.append(check)


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


if __name__ == "__main__":
    raise SystemExit(main())
