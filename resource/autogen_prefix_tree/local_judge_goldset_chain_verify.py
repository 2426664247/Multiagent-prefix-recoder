from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CHAIN_STEPS = (
    "labels_from_csv",
    "labels_validation",
    "apply_labels",
    "import_goldset",
    "goldset_validation",
    "quality_eval",
)


@dataclass(frozen=True)
class LocalJudgeGoldsetChainVerificationResult:
    summary_path: str | None
    report_path: str | None
    summary: dict[str, Any]


def verify_local_judge_goldset_chain(
    *,
    manifest_path: str | Path | None = None,
    labels_from_csv_summary_path: str | Path | None = None,
    labels_validation_summary_path: str | Path | None = None,
    apply_labels_summary_path: str | Path | None = None,
    import_summary_path: str | Path | None = None,
    goldset_validation_summary_path: str | Path | None = None,
    quality_eval_summary_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> LocalJudgeGoldsetChainVerificationResult:
    manifest_file = Path(manifest_path) if manifest_path is not None else None
    manifest = _load_json_mapping(manifest_file) if manifest_file and manifest_file.exists() else {}
    paths = _resolve_paths(
        manifest=manifest,
        manifest_file=manifest_file,
        labels_from_csv_summary_path=labels_from_csv_summary_path,
        labels_validation_summary_path=labels_validation_summary_path,
        apply_labels_summary_path=apply_labels_summary_path,
        import_summary_path=import_summary_path,
        goldset_validation_summary_path=goldset_validation_summary_path,
        quality_eval_summary_path=quality_eval_summary_path,
    )
    steps = [_step_status(name, paths.get(name)) for name in CHAIN_STEPS]
    checks = _checks(steps)
    next_action = _next_action(steps=steps, checks=checks)
    summary = {
        "schema_version": "prefix-local-judge-goldset-chain-verification-v1",
        "prompt_safe_summary": True,
        "manifest_path": str(manifest_file) if manifest_file is not None else None,
        "steps": steps,
        "checks": checks,
        "ready": all(bool(check.get("ok")) for check in checks),
        "chain_ready": all(bool(step.get("ready")) for step in steps)
        and all(bool(check.get("ok")) for check in checks),
        "next_action": next_action,
        "blocking_reasons": [str(check["name"]) for check in checks if not check.get("ok")],
        "allowed_claims": _allowed_claims(steps=steps, checks=checks),
        "prohibited_claims": _prohibited_claims(steps=steps, checks=checks),
        "suggested_commands": _suggested_commands(next_action=next_action, manifest=manifest),
        "real_provider_metrics_available": False,
        "limits": (
            "This verifier only reads prompt-safe summaries for the local human-label gold-set chain. "
            "It does not read candidate text, call a local judge, call a provider API, or prove cache, "
            "latency, cost, task success, or broad semantic-model quality."
        ),
    }
    summary_target = _write_json(summary_path, summary)
    report_target = _write_text(report_path, render_markdown_chain_verification(summary))
    return LocalJudgeGoldsetChainVerificationResult(
        summary_path=str(summary_target) if summary_target else None,
        report_path=str(report_target) if report_target else None,
        summary=summary,
    )


def render_markdown_chain_verification(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Local Judge Gold-Set Chain Verification",
        "",
        f"Next action: **{_format_value(summary.get('next_action'))}**",
        f"Ready: `{_format_value(summary.get('ready'))}`",
        f"Chain ready: `{_format_value(summary.get('chain_ready'))}`",
        "",
        "## Steps",
        "",
        "| Step | Status | Ready | Evidence |",
        "|---|---|---:|---|",
    ]
    steps = summary.get("steps") if isinstance(summary.get("steps"), list) else []
    for step in steps:
        if isinstance(step, Mapping):
            lines.append(
                f"| {step.get('name')} | {_format_value(step.get('status'))} | "
                f"{_format_value(step.get('ready'))} | {_format_value(step.get('evidence'))} |"
            )
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---:|---|"])
    checks = summary.get("checks") if isinstance(summary.get("checks"), list) else []
    for check in checks:
        if isinstance(check, Mapping):
            lines.append(
                f"| {check.get('name')} | {_format_value(check.get('ok'))} | {_format_value(check.get('detail'))} |"
            )
    blocking = summary.get("blocking_reasons") if isinstance(summary.get("blocking_reasons"), list) else []
    lines.extend(["", "## Blocking Reasons", ""])
    if blocking:
        lines.extend(f"- `{reason}`" for reason in blocking)
    else:
        lines.append("- none")
    commands = summary.get("suggested_commands") if isinstance(summary.get("suggested_commands"), list) else []
    if commands:
        lines.extend(["", "## Suggested Commands", "", "```powershell"])
        lines.extend(str(command) for command in commands)
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Limits",
            "",
            "- This report is prompt-safe and does not include candidate text or API key values.",
            "- Passing this verifier allows only the local gold-set chain claim, not provider cache or task-success claims.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the prompt-safe local human-label gold-set chain before local judge quality eval."
    )
    parser.add_argument("--manifest", help="Optional legacy suite manifest to resolve summary paths.")
    parser.add_argument("--labels-from-csv-summary")
    parser.add_argument("--labels-validation-summary")
    parser.add_argument("--apply-labels-summary")
    parser.add_argument("--import-summary")
    parser.add_argument("--goldset-validation-summary")
    parser.add_argument("--quality-eval-summary")
    parser.add_argument("--summary", help="Optional prompt-safe verification summary JSON.")
    parser.add_argument("--report-md", help="Optional prompt-safe Markdown report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_local_judge_goldset_chain(
        manifest_path=args.manifest,
        labels_from_csv_summary_path=args.labels_from_csv_summary,
        labels_validation_summary_path=args.labels_validation_summary,
        apply_labels_summary_path=args.apply_labels_summary,
        import_summary_path=args.import_summary,
        goldset_validation_summary_path=args.goldset_validation_summary,
        quality_eval_summary_path=args.quality_eval_summary,
        summary_path=args.summary,
        report_path=args.report_md,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.summary.get("ready") is True else 1


def _resolve_paths(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path | None,
    labels_from_csv_summary_path: str | Path | None,
    labels_validation_summary_path: str | Path | None,
    apply_labels_summary_path: str | Path | None,
    import_summary_path: str | Path | None,
    goldset_validation_summary_path: str | Path | None,
    quality_eval_summary_path: str | Path | None,
) -> dict[str, Path | None]:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    values = {
        "labels_from_csv": labels_from_csv_summary_path
        or local_judge.get("goldset_annotation_labels_from_csv_summary_path"),
        "labels_validation": labels_validation_summary_path
        or local_judge.get("goldset_annotation_labels_validation_summary_path"),
        "apply_labels": apply_labels_summary_path or local_judge.get("goldset_annotation_apply_summary_path"),
        "import_goldset": import_summary_path or local_judge.get("goldset_annotation_import_summary_path"),
        "goldset_validation": goldset_validation_summary_path or local_judge.get("goldset_validation_summary_path"),
        "quality_eval": quality_eval_summary_path or local_judge.get("quality_eval_summary_path"),
    }
    return {
        name: _artifact_path(value, manifest_file=manifest_file)
        for name, value in values.items()
    }


def _step_status(name: str, path: Path | None) -> dict[str, Any]:
    base = {
        "name": name,
        "path": str(path) if path is not None else None,
        "exists": bool(path and path.exists()),
        "ready": False,
        "status": "missing",
        "evidence": "summary path missing" if path is None else "summary file missing",
    }
    if path is None or not path.exists():
        return base
    try:
        summary = _load_json_mapping(path)
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "exists": True,
            "status": "fail",
            "evidence": f"summary unreadable: {type(exc).__name__}",
        }
    if summary.get("prompt_safe_summary") is not True:
        return {**base, "exists": True, "status": "fail", "evidence": "summary is not marked prompt-safe"}
    status = _status_from_summary(name, summary)
    return {
        **base,
        **status,
        "exists": True,
        "schema_version": summary.get("schema_version"),
        "row_count": _first_num(summary, "row_count", "sample_count"),
        "expected_label_counts": summary.get("expected_label_counts")
        if isinstance(summary.get("expected_label_counts"), Mapping)
        else {},
        "recommendation": summary.get("recommendation"),
    }


def _status_from_summary(name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    if name == "labels_from_csv":
        schema_ok = summary.get("schema_version") == "prefix-local-judge-goldset-labels-from-csv-summary-v1"
        invariant_ok = (
            summary.get("labels_jsonl_text_written") is False
            and summary.get("suggested_labels_not_auto_applied") is True
        )
        ready = bool(schema_ok and invariant_ok and summary.get("ready_for_apply_labels") is True)
        return _ready_status(
            ready=ready,
            schema_ok=schema_ok,
            invariant_ok=invariant_ok,
            fail_evidence="labels-from-csv must be prompt-safe and must not auto-apply suggestions",
            warn_evidence="labels-from-csv has pending/invalid labels or is not ready for apply-labels",
            pass_evidence="manual expected_label values exported to prompt-safe labels JSONL",
        )
    if name == "labels_validation":
        schema_ok = summary.get("schema_version") == "prefix-local-judge-goldset-labels-validation-summary-v1"
        invariant_ok = (
            summary.get("ready_for_goldset_import_after_apply") is True
            and summary.get("gold_labels_independent_from_rules_ready") is True
            and summary.get("possible_rule_self_confirmation") is not True
        )
        ready = bool(schema_ok and invariant_ok and summary.get("ready_for_apply_labels") is True)
        return _ready_status(
            ready=ready,
            schema_ok=schema_ok,
            invariant_ok=invariant_ok,
            fail_evidence="labels validation must show complete manually reviewed labels independent from rule labels",
            warn_evidence="labels validation has pending, invalid, duplicate, imbalanced, or self-confirming labels",
            pass_evidence="prompt-safe labels JSONL validated before apply-labels",
        )
    if name == "apply_labels":
        schema_ok = summary.get("schema_version") == "prefix-local-judge-goldset-label-apply-summary-v1"
        invariant_ok = summary.get("require_complete") is True
        ready = bool(
            schema_ok
            and invariant_ok
            and summary.get("output_written") is True
            and summary.get("ready_for_labeled_jsonl_import") is True
        )
        return _ready_status(
            ready=ready,
            schema_ok=schema_ok,
            invariant_ok=invariant_ok,
            fail_evidence="apply-labels must run with --require-complete",
            warn_evidence="apply-labels has not written a complete labeled CSV",
            pass_evidence="prompt-safe labels applied to local annotation CSV",
        )
    if name == "import_goldset":
        schema_ok = summary.get("schema_version") == "prefix-local-judge-goldset-csv-import-summary-v1"
        invariant_ok = summary.get("allow_incomplete_output") is False
        ready = bool(
            schema_ok
            and invariant_ok
            and summary.get("output_written") is True
            and summary.get("ready_for_local_judge_goldset_validate") is True
        )
        return _ready_status(
            ready=ready,
            schema_ok=schema_ok,
            invariant_ok=invariant_ok,
            fail_evidence="gold-set import must not use allow-incomplete output for quality eval",
            warn_evidence="labeled CSV has not been imported into complete local gold JSONL",
            pass_evidence="complete local human-labeled gold JSONL imported",
        )
    if name == "goldset_validation":
        schema_ok = summary.get("schema_version") == "prefix-local-judge-goldset-validation-summary-v1"
        invariant_ok = summary.get("ready_for_local_judge_quality_eval") is True
        ready = bool(schema_ok and invariant_ok and summary.get("ready") is True)
        return _ready_status(
            ready=ready,
            schema_ok=schema_ok,
            invariant_ok=invariant_ok,
            fail_evidence="gold-set validation must explicitly allow local judge quality eval",
            warn_evidence="gold set is not structurally ready for local judge quality eval",
            pass_evidence="local gold JSONL validated before quality eval",
        )
    if name == "quality_eval":
        schema_ok = summary.get("schema_version") == "prefix-local-judge-quality-eval-summary-v1"
        invariant_ok = summary.get("semantic_quality_metrics_available") is True
        ready = bool(schema_ok and invariant_ok and summary.get("ready") is True)
        if schema_ok and invariant_ok and not ready:
            return {
                "ready": False,
                "status": "warn",
                "evidence": "quality eval ran but is below semantic-quality thresholds",
            }
        return _ready_status(
            ready=ready,
            schema_ok=schema_ok,
            invariant_ok=invariant_ok,
            fail_evidence="quality eval must expose semantic quality metrics",
            warn_evidence="local judge quality eval is below thresholds or incomplete",
            pass_evidence="local judge quality eval passed on validated gold set",
        )
    return {"ready": False, "status": "fail", "evidence": f"unknown chain step: {name}"}


def _ready_status(
    *,
    ready: bool,
    schema_ok: bool,
    invariant_ok: bool,
    fail_evidence: str,
    warn_evidence: str,
    pass_evidence: str,
) -> dict[str, Any]:
    if not schema_ok:
        return {"ready": False, "status": "fail", "evidence": "unexpected summary schema"}
    if not invariant_ok:
        return {"ready": False, "status": "fail", "evidence": fail_evidence}
    if not ready:
        return {"ready": False, "status": "warn", "evidence": warn_evidence}
    return {"ready": True, "status": "pass", "evidence": pass_evidence}


def _checks(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(step.get("name")): step for step in steps}
    checks: list[dict[str, Any]] = []
    for name in CHAIN_STEPS:
        step = by_name[name]
        checks.append(
            {
                "name": f"{name}:ready",
                "ok": step.get("ready") is True,
                "detail": step.get("evidence"),
            }
        )
    checks.extend(_order_checks(by_name))
    checks.extend(_count_consistency_checks(by_name))
    checks.append(
        {
            "name": "quality_eval_after_goldset_validation",
            "ok": not _is_ready(by_name, "quality_eval") or _is_ready(by_name, "goldset_validation"),
            "detail": "quality eval must be backed by validated local gold JSONL",
        }
    )
    return checks


def _order_checks(by_name: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = (
        ("labels_validation", "labels_from_csv"),
        ("apply_labels", "labels_validation"),
        ("import_goldset", "apply_labels"),
        ("goldset_validation", "import_goldset"),
        ("quality_eval", "goldset_validation"),
    )
    return [
        {
            "name": f"order:{later}_after_{earlier}",
            "ok": not _exists(by_name, later) or _is_ready(by_name, earlier),
            "detail": f"{later} summary should only be trusted after {earlier} is ready",
        }
        for later, earlier in pairs
    ]


def _count_consistency_checks(by_name: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for later, earlier in (
        ("labels_validation", "labels_from_csv"),
        ("apply_labels", "labels_validation"),
        ("import_goldset", "apply_labels"),
        ("goldset_validation", "import_goldset"),
    ):
        if not (_exists(by_name, later) and _exists(by_name, earlier)):
            continue
        later_count = _num(by_name[later].get("row_count"))
        earlier_count = _num(by_name[earlier].get("row_count"))
        checks.append(
            {
                "name": f"row_count:{later}_matches_{earlier}",
                "ok": later_count == earlier_count,
                "detail": f"{later}={later_count}; {earlier}={earlier_count}",
            }
        )
    if _exists(by_name, "quality_eval") and _exists(by_name, "goldset_validation"):
        sample_count = _num(by_name["quality_eval"].get("row_count"))
        gold_count = _num(by_name["goldset_validation"].get("row_count"))
        checks.append(
            {
                "name": "row_count:quality_eval_not_more_than_goldset_validation",
                "ok": sample_count <= gold_count,
                "detail": f"quality_eval={sample_count}; goldset_validation={gold_count}",
            }
        )
    return checks


def _next_action(*, steps: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]]) -> str:
    for step in steps:
        if step.get("ready") is not True:
            if step.get("name") == "quality_eval" and step.get("exists") is True:
                return "improve_or_recalibrate_local_judge_quality_eval"
            return {
                "labels_from_csv": "run_labels_from_csv_after_manual_review",
                "labels_validation": "run_validate_labels_require_complete",
                "apply_labels": "run_apply_labels_require_complete",
                "import_goldset": "import_labeled_local_judge_goldset_csv",
                "goldset_validation": "validate_local_judge_goldset",
                "quality_eval": "run_local_judge_quality_eval",
            }.get(str(step.get("name")), "fix_goldset_chain")
    if any(not check.get("ok") for check in checks):
        return "fix_goldset_chain_order_or_counts"
    return "goldset_chain_ready_for_semantic_quality_claim"


def _allowed_claims(*, steps: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ready_steps = [str(step.get("name")) for step in steps if step.get("ready") is True]
    claims = [
        {
            "id": "local_human_label_progress",
            "claim": "Prompt-safe summaries show which local human-label steps have completed.",
            "evidence": ready_steps,
        }
    ]
    if all(bool(step.get("ready")) for step in steps) and all(bool(check.get("ok")) for check in checks):
        claims.append(
            {
                "id": "local_goldset_chain_ready",
                "claim": "The local human-labeled gold-set chain is ready for scoped local judge quality claims.",
                "evidence": ["labels_from_csv", "validate_labels", "apply_labels", "import", "goldset_validate", "quality_eval"],
            }
        )
    return claims


def _prohibited_claims(*, steps: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    missing = [str(step.get("name")) for step in steps if step.get("ready") is not True]
    failed = [str(check.get("name")) for check in checks if not check.get("ok")]
    claims = [
        {
            "id": "real_provider_cache_or_task_success",
            "claim": "Provider cache, latency, cost, or task-success gains are proven.",
            "missing_evidence": ["real provider A/B telemetry and AutoGenBench task results"],
        }
    ]
    if missing or failed:
        claims.append(
            {
                "id": "local_goldset_chain_ready",
                "claim": "The local human-labeled gold-set chain is complete and quality-evaluated.",
                "missing_evidence": missing + failed,
            }
        )
    return claims


def _suggested_commands(*, next_action: str, manifest: Mapping[str, Any]) -> list[str]:
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    goldset_commands = (
        [str(command) for command in commands.get("local_judge_goldset_template", [])]
        if isinstance(commands.get("local_judge_goldset_template"), list)
        else []
    )
    if next_action == "run_labels_from_csv_after_manual_review":
        return [command for command in goldset_commands if "local_judge_goldset_csv labels-from-csv" in command]
    if next_action == "run_validate_labels_require_complete":
        return [command for command in goldset_commands if "local_judge_goldset_csv validate-labels" in command]
    if next_action == "run_apply_labels_require_complete":
        return [command for command in goldset_commands if "local_judge_goldset_csv apply-labels" in command]
    if next_action == "import_labeled_local_judge_goldset_csv":
        return [command for command in goldset_commands if "local_judge_goldset_csv import" in command]
    if next_action == "validate_local_judge_goldset":
        values = commands.get("local_judge_goldset_validate") if isinstance(commands.get("local_judge_goldset_validate"), list) else []
        return [str(command) for command in values]
    if next_action == "run_local_judge_quality_eval":
        values = commands.get("local_judge_quality_eval") if isinstance(commands.get("local_judge_quality_eval"), list) else []
        return [str(command) for command in values]
    return []


def _artifact_path(value: Any, *, manifest_file: Path | None) -> Path | None:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value:
        path = Path(value)
    else:
        return None
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if manifest_file is not None:
        manifest_relative = manifest_file.parent / path
        if manifest_relative.exists():
            return manifest_relative
    return path


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


def _exists(by_name: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    return bool(by_name.get(name, {}).get("exists"))


def _is_ready(by_name: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    return by_name.get(name, {}).get("ready") is True


def _first_num(value: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if name in value:
            return _num(value.get(name))
    return 0


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


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
