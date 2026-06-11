from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agbench_legacy_preflight import build_legacy_agbench_preflight
from .agbench_legacy_runbook import build_legacy_agbench_runbook
from .local_judge_healthcheck import run_local_judge_healthcheck
from .offline_semantic_matrix import run_offline_semantic_matrix
from .offline_local_judge_matrix_smoke import run_offline_local_judge_matrix_smoke
from .semantic_guard_proxy_smoke import run_semantic_guard_proxy_smoke
from .three_proxy_smoke import run_three_proxy_smoke


@dataclass(frozen=True)
class LegacyPreflightSmokesResult:
    manifest_path: str
    output_dir: str
    summary_path: str
    summary: dict[str, Any]


def run_legacy_preflight_smokes(
    *,
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
    cwd: str | Path = ".",
    env: Mapping[str, str] | None = None,
    session_id: str | None = None,
    semantic_guard_judge_mode: str | None = None,
    semantic_guard_min_confidence: float = 0.75,
    run_real_local_judge_healthcheck: bool = False,
    local_judge_base_url: str | None = None,
    local_judge_model: str | None = None,
    local_judge_api_key_env: str | None = None,
    local_judge_timeout: float = 30.0,
    local_judge_min_confidence: float = 0.66,
    local_judge_sources: Sequence[tuple[str, str | Path]] | None = None,
    run_real_local_judge_matrix: bool = False,
    local_judge_scope: str = "review",
    max_local_judge_calls: int | None = 50,
    fake_judge_mode: str = "accept",
    allow_missing_api: bool = False,
) -> LegacyPreflightSmokesResult:
    manifest_file = Path(manifest_path)
    manifest = _load_json_mapping(manifest_file)
    suite_id = str(manifest.get("suite_id") or manifest_file.parent.name or "legacy-preflight-smokes")
    out = Path(output_dir) if output_dir is not None else _suite_dir(manifest=manifest, manifest_file=manifest_file) / "preflight_smokes"
    out.mkdir(parents=True, exist_ok=True)

    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    three_command = _first_command(commands.get("preflight_fake_smoke"))
    semantic_command = _first_command(commands.get("semantic_guard_fake_smoke"))
    local_judge_healthcheck_command = _first_command(commands.get("local_judge_healthcheck"))
    local_judge_command = _first_command(commands.get("offline_local_judge_matrix_smoke"))
    real_local_judge_command = _first_command(commands.get("offline_local_judge_matrix"))
    three_output_dir = _option_path(
        command=three_command,
        option="--output-dir",
        manifest_file=manifest_file,
        default=_suite_dir(manifest=manifest, manifest_file=manifest_file) / "preflight_three_proxy_smoke",
    )
    semantic_output_dir = _option_path(
        command=semantic_command,
        option="--output-dir",
        manifest_file=manifest_file,
        default=_suite_dir(manifest=manifest, manifest_file=manifest_file) / "preflight_semantic_guard_proxy_smoke",
    )
    base_session_id = session_id or suite_id
    three_session_id = _option_value(three_command, "--session-id") or f"{base_session_id}-preflight-smoke"
    semantic_session_id = _option_value(semantic_command, "--session-id") or f"{base_session_id}-semantic-guard-smoke"
    judge_mode = semantic_guard_judge_mode or _option_value(semantic_command, "--judge-mode") or "reject"
    local_judge_output_dir = _option_path(
        command=local_judge_command,
        option="--output-dir",
        manifest_file=manifest_file,
        default=_suite_dir(manifest=manifest, manifest_file=manifest_file) / "offline_local_judge_matrix_smoke",
    )
    local_judge_session_id = (
        _option_value(local_judge_command, "--session-id") or f"{base_session_id}-offline-local-judge-matrix-smoke"
    )
    real_local_judge_output_dir = _option_path(
        command=real_local_judge_command,
        option="--output-dir",
        manifest_file=manifest_file,
        default=_suite_dir(manifest=manifest, manifest_file=manifest_file) / "offline_local_judge_matrix",
    )
    real_local_judge_session_id = (
        _option_value(real_local_judge_command, "--session-id") or f"{base_session_id}-offline-local-judge-matrix"
    )
    healthcheck_base_url = (
        local_judge_base_url
        or _manifest_local_judge_value(manifest, "base_url")
        or _option_value(local_judge_healthcheck_command, "--base-url")
    )
    healthcheck_model = (
        local_judge_model
        or _manifest_local_judge_value(manifest, "model")
        or _option_value(local_judge_healthcheck_command, "--model")
    )
    healthcheck_api_key_env = (
        local_judge_api_key_env
        or _manifest_local_judge_value(manifest, "api_key_env")
        or _option_value(local_judge_healthcheck_command, "--api-key-env")
    )
    healthcheck_summary_path = _local_judge_healthcheck_summary_path(
        manifest=manifest,
        manifest_file=manifest_file,
        command=local_judge_healthcheck_command,
    )

    three_smoke = run_three_proxy_smoke(
        output_dir=three_output_dir,
        session_id=three_session_id,
        manifest_path=manifest_file,
    )
    semantic_smoke = run_semantic_guard_proxy_smoke(
        output_dir=semantic_output_dir,
        session_id=semantic_session_id,
        judge_mode=judge_mode,
        min_confidence=semantic_guard_min_confidence,
    )
    local_judge_healthcheck = (
        run_local_judge_healthcheck(
            base_url=str(healthcheck_base_url),
            model=str(healthcheck_model),
            api_key=os.environ.get(str(healthcheck_api_key_env)) if healthcheck_api_key_env else None,
            timeout=local_judge_timeout,
            min_confidence=local_judge_min_confidence,
            summary_path=healthcheck_summary_path,
        )
        if (run_real_local_judge_healthcheck or run_real_local_judge_matrix)
        and _usable_local_judge_value(healthcheck_base_url)
        and _usable_local_judge_value(healthcheck_model)
        else None
    )
    real_local_judge_matrix = (
        run_offline_semantic_matrix(
            sources=local_judge_sources,
            output_dir=real_local_judge_output_dir,
            session_id=real_local_judge_session_id,
            include_candidate_text=True,
            min_confidence=local_judge_min_confidence,
            judge="openai-compatible",
            local_judge_base_url=str(healthcheck_base_url),
            local_judge_model=str(healthcheck_model),
            local_judge_api_key=os.environ.get(str(healthcheck_api_key_env)) if healthcheck_api_key_env else None,
            local_judge_timeout=local_judge_timeout,
            local_judge_scope=local_judge_scope,
            max_local_judge_calls=max_local_judge_calls,
        )
        if run_real_local_judge_matrix
        and local_judge_sources
        and _usable_local_judge_value(healthcheck_base_url)
        and _usable_local_judge_value(healthcheck_model)
        else None
    )
    local_judge_matrix_smoke = (
        run_offline_local_judge_matrix_smoke(
            sources=local_judge_sources,
            output_dir=local_judge_output_dir,
            session_id=local_judge_session_id,
            local_judge_scope=local_judge_scope,
            max_local_judge_calls=max_local_judge_calls,
            fake_judge_mode=fake_judge_mode,
        )
        if local_judge_sources
        else None
    )

    effective_env = os.environ if env is None else env
    runbook_summary_path, runbook_report_path = _runbook_paths(manifest=manifest, manifest_file=manifest_file)
    runbook = build_legacy_agbench_runbook(
        manifest_path=manifest_file,
        summary_path=runbook_summary_path,
        report_path=runbook_report_path,
        env=effective_env,
        require_api_config=not allow_missing_api,
    )
    preflight_summary_path, preflight_report_path = _preflight_paths(manifest=manifest, manifest_file=manifest_file)
    preflight = build_legacy_agbench_preflight(
        manifest_path=manifest_file,
        cwd=cwd,
        summary_path=preflight_summary_path,
        report_path=preflight_report_path,
        env=effective_env,
        allow_missing_api=allow_missing_api,
    )

    preflight_gates = preflight.summary.get("gates") if isinstance(preflight.summary.get("gates"), Mapping) else {}
    semantic_summary = semantic_smoke.summary
    real_local_judge_used = bool(local_judge_healthcheck or real_local_judge_matrix)
    metrics_note = (
        "This command ran fake upstream/fake judge smokes and an explicitly requested local-judge check. "
        "It refreshes prompt-safe runbook/preflight reports, but does not prove real provider cache, latency, cost, "
        "or task success."
    ) if real_local_judge_used else (
        "This command only runs no-network fake upstream/fake judge smokes and refreshes prompt-safe "
        "runbook/preflight reports. It does not prove real provider cache, latency, cost, or task success."
    )
    summary = {
        "schema_version": "prefix-legacy-preflight-smokes-summary-v1",
        "prompt_safe_summary": True,
        "manifest_path": str(manifest_file),
        "suite_id": manifest.get("suite_id"),
        "output_dir": str(out),
        "fake_upstream": True,
        "fake_local_judge": bool(semantic_summary.get("fake_local_judge")),
        "local_judge_healthcheck_run": local_judge_healthcheck is not None,
        "local_judge_healthcheck_summary_path": local_judge_healthcheck.summary_path
        if local_judge_healthcheck
        else None,
        "local_judge_healthcheck_ready": bool(
            local_judge_healthcheck and local_judge_healthcheck.summary.get("ready") is True
        ),
        "real_local_judge_matrix_run": real_local_judge_matrix is not None,
        "offline_local_judge_matrix_summary_path": real_local_judge_matrix.summary_path
        if real_local_judge_matrix
        else None,
        "fake_local_judge_matrix_smoke_run": local_judge_matrix_smoke is not None,
        "real_provider_metrics_available": False,
        "real_provider_metrics_note": metrics_note,
        "three_proxy_smoke_summary_path": three_smoke.summary_path,
        "semantic_guard_smoke_summary_path": semantic_smoke.summary_path,
        "offline_local_judge_matrix_smoke_summary_path": local_judge_matrix_smoke.summary_path
        if local_judge_matrix_smoke
        else None,
        "legacy_runbook_summary_path": runbook.summary_path,
        "legacy_runbook_report_path": runbook.report_path,
        "legacy_runbook_next_action": runbook.summary.get("next_action"),
        "legacy_real_ab_preflight_summary_path": preflight.summary_path,
        "legacy_real_ab_preflight_report_path": preflight.report_path,
        "legacy_real_ab_preflight_next_action": preflight.summary.get("next_action"),
        "next_action": preflight.summary.get("next_action") or runbook.summary.get("next_action"),
        "provider_or_ab_fake_artifacts_detected": bool(runbook.summary.get("provider_or_ab_fake_artifacts_detected")),
        "semantic_guard_fake_smoke_ready": bool(runbook.summary.get("semantic_guard_fake_smoke_ready")),
        "local_judge_healthcheck_runbook_ready": bool(runbook.summary.get("local_judge_healthcheck_ready")),
        "offline_local_judge_matrix_ready": bool(runbook.summary.get("offline_local_judge_matrix_ready")),
        "offline_local_judge_matrix_smoke_ready": bool(
            runbook.summary.get("offline_local_judge_matrix_smoke_ready")
        ),
        "ready_to_claim_real_results": bool(preflight_gates.get("ready_to_claim_real_results")),
    }
    target = Path(summary_path) if summary_path is not None else out / "legacy_preflight_smokes_summary.json"
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return LegacyPreflightSmokesResult(
        manifest_path=str(manifest_file),
        output_dir=str(out),
        summary_path=str(target),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run both no-network legacy preflight smokes from a suite manifest and refresh runbook/preflight reports."
        )
    )
    parser.add_argument("--manifest", required=True, help="Generated legacy_suite_manifest.json.")
    parser.add_argument("--output-dir", help="Directory for the orchestration summary.")
    parser.add_argument("--summary", help="Optional summary JSON path. Defaults to <output-dir>/legacy_preflight_smokes_summary.json.")
    parser.add_argument("--cwd", default=".", help="Project root to inspect for refreshed real A/B preflight readiness.")
    parser.add_argument("--session-id", help="Base session id used when manifest smoke commands do not include one.")
    parser.add_argument(
        "--judge-mode",
        choices=("accept", "reject", "low-confidence"),
        help="Override the semantic guard fake judge mode from the manifest command.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument(
        "--run-local-judge-healthcheck",
        action="store_true",
        help=(
            "Run the one-request real local judge healthcheck using the configured OpenAI-compatible endpoint. "
            "Without this flag, local judge config in the manifest is only reported, not contacted."
        ),
    )
    parser.add_argument("--local-judge-base-url", help="Optional real local judge base URL for a one-request healthcheck.")
    parser.add_argument("--local-judge-model", help="Optional real local judge model for a one-request healthcheck.")
    parser.add_argument("--local-judge-api-key-env", help="Optional environment variable containing the local judge API key.")
    parser.add_argument("--local-judge-timeout", type=float, default=30.0)
    parser.add_argument("--local-judge-min-confidence", type=float, default=0.66)
    parser.add_argument(
        "--local-judge-source",
        action="append",
        help="Optional real source root for offline_local_judge_matrix_smoke, as label=path. May be repeated.",
    )
    parser.add_argument(
        "--run-real-local-judge-matrix",
        action="store_true",
        help=(
            "Also run offline_semantic_matrix with --judge openai-compatible using the configured real local judge. "
            "This sends candidate text to the local judge endpoint."
        ),
    )
    parser.add_argument(
        "--local-judge-scope",
        choices=("all", "review"),
        default="review",
        help="Scope passed to offline_local_judge_matrix_smoke when --local-judge-source is provided.",
    )
    parser.add_argument(
        "--max-local-judge-calls",
        type=int,
        default=50,
        help="Per-source fake local judge call cap for offline_local_judge_matrix_smoke.",
    )
    parser.add_argument(
        "--fake-judge-mode",
        choices=("accept", "review", "reject", "low-confidence"),
        default="accept",
        help="Fake local judge mode for offline_local_judge_matrix_smoke.",
    )
    parser.add_argument(
        "--allow-missing-api",
        action="store_true",
        help="Refresh runbook/preflight without failing API marker checks. This still does not claim real results.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_legacy_preflight_smokes(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        summary_path=args.summary,
        cwd=args.cwd,
        session_id=args.session_id,
        semantic_guard_judge_mode=args.judge_mode,
        semantic_guard_min_confidence=args.min_confidence,
        run_real_local_judge_healthcheck=args.run_local_judge_healthcheck,
        local_judge_base_url=args.local_judge_base_url,
        local_judge_model=args.local_judge_model,
        local_judge_api_key_env=args.local_judge_api_key_env,
        local_judge_timeout=args.local_judge_timeout,
        local_judge_min_confidence=args.local_judge_min_confidence,
        local_judge_sources=[_parse_source_spec(value) for value in args.local_judge_source]
        if args.local_judge_source
        else None,
        run_real_local_judge_matrix=args.run_real_local_judge_matrix,
        local_judge_scope=args.local_judge_scope,
        max_local_judge_calls=args.max_local_judge_calls,
        fake_judge_mode=args.fake_judge_mode,
        allow_missing_api=args.allow_missing_api,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _suite_dir(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path:
    value = manifest.get("output_dir")
    if isinstance(value, str) and value:
        return _resolve_path(value, manifest_file=manifest_file)
    return manifest_file.parent


def _runbook_paths(*, manifest: Mapping[str, Any], manifest_file: Path) -> tuple[Path, Path]:
    recommended = manifest.get("recommended_runbook") if isinstance(manifest.get("recommended_runbook"), Mapping) else {}
    reports_dir = _reports_dir(manifest=manifest, manifest_file=manifest_file)
    summary = recommended.get("summary") or reports_dir / "legacy_runbook_summary.json"
    report = recommended.get("report_md") or reports_dir / "legacy_runbook.md"
    return _resolve_path(summary, manifest_file=manifest_file), _resolve_path(report, manifest_file=manifest_file)


def _preflight_paths(*, manifest: Mapping[str, Any], manifest_file: Path) -> tuple[Path, Path]:
    commands = manifest.get("suggested_commands") if isinstance(manifest.get("suggested_commands"), Mapping) else {}
    command = _first_command(commands.get("real_ab_preflight"))
    reports_dir = _reports_dir(manifest=manifest, manifest_file=manifest_file)
    summary = _option_value(command, "--summary") or str(reports_dir / "legacy_real_ab_preflight_summary.json")
    report = _option_value(command, "--report-md") or str(reports_dir / "legacy_real_ab_preflight.md")
    return _resolve_path(summary, manifest_file=manifest_file), _resolve_path(report, manifest_file=manifest_file)


def _local_judge_healthcheck_summary_path(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
    command: str | None,
) -> Path:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    configured = local_judge.get("healthcheck_summary_path") if isinstance(local_judge, Mapping) else None
    value = configured if isinstance(configured, str) and configured else _option_value(command, "--summary")
    if value:
        return _resolve_path(value, manifest_file=manifest_file)
    return _suite_dir(manifest=manifest, manifest_file=manifest_file) / "artifacts" / "local_judge_healthcheck_summary.json"


def _manifest_local_judge_value(manifest: Mapping[str, Any], key: str) -> str | None:
    local_judge = manifest.get("local_judge") if isinstance(manifest.get("local_judge"), Mapping) else {}
    value = local_judge.get(key) if isinstance(local_judge, Mapping) else None
    return str(value) if isinstance(value, str) and value else None


def _usable_local_judge_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return bool(stripped and not stripped.startswith("<") and not stripped.endswith(">"))


def _reports_dir(*, manifest: Mapping[str, Any], manifest_file: Path) -> Path:
    value = manifest.get("reports_dir")
    if isinstance(value, str) and value:
        return _resolve_path(value, manifest_file=manifest_file)
    return manifest_file.parent / "reports"


def _option_path(*, command: str | None, option: str, manifest_file: Path, default: Path) -> Path:
    value = _option_value(command, option)
    if not value:
        return default
    return _resolve_path(value, manifest_file=manifest_file)


def _resolve_path(value: Any, *, manifest_file: Path) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    manifest_relative = manifest_file.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return cwd_path


def _first_command(value: Any) -> str | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return str(value[0])
    return None


def _option_value(command: str | None, option: str) -> str | None:
    if not command:
        return None
    parts = command.split()
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        prefix = option + "="
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


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


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
