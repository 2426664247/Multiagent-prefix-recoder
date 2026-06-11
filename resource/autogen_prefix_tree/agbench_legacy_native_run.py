from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogen import config_list_from_json
from autogenbench.run_cmd import expand_scenario
from autogenbench.tabulate_cmd import SUCCESS_STRINGS, default_scorer

from .agbench_legacy_collect import collect_legacy_agbench_results
from .openai_forward_proxy import build_forward_proxy_server


VARIANTS = ("baseline", "plugin_rule_only", "plugin_nl_segmentation")


@dataclass(frozen=True)
class LegacyNativeRunResult:
    manifest_path: str
    summary_path: str | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class _StartedProxy:
    variant: str
    server: Any
    thread: threading.Thread
    base_url: str


def run_legacy_native_ab_sample(
    *,
    manifest_path: str | Path,
    variants: Sequence[str] = VARIANTS,
    task_ids: Sequence[str] = (),
    max_tasks: int | None = None,
    repeat: int = 1,
    results_dir: str | Path = "Results",
    sample_id: str | None = None,
    summary_path: str | Path | None = None,
    collection_summary_path: str | Path | None = None,
    start_proxies: bool = True,
    use_project_deepseek_config: bool = True,
    project_config_path: str | Path | None = None,
    dynamic_proxy_ports: bool = True,
    proxy_host: str = "127.0.0.1",
    timeout_seconds: float = 600.0,
    enable_groupchat_history_reordering: bool = False,
    clear_existing: bool = True,
    run_collection: bool = True,
    run_ab_eval: bool = True,
) -> LegacyNativeRunResult:
    manifest_file = Path(manifest_path)
    manifest = _load_json_mapping(manifest_file)
    selected_variants = tuple(_validate_variants(variants))
    selected_tasks = _select_task_records(
        manifest=manifest,
        manifest_file=manifest_file,
        task_ids=tuple(task_ids),
        max_tasks=max_tasks,
    )
    if not selected_tasks:
        raise ValueError("selected task set is empty")

    effective_sample_id = sample_id or f"{manifest.get('suite_id') or 'legacy-agbench'}.native_sample"
    results_root = Path(results_dir)
    sample_scenarios_dir = _artifact_path(manifest.get("output_dir"), manifest_file=manifest_file) / "scenarios" / "native_samples"
    reports_dir = _artifact_path(manifest.get("reports_dir"), manifest_file=manifest_file)
    sample_scenarios_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    started: dict[str, _StartedProxy] = {}
    if start_proxies:
        started = _start_proxies(
            manifest=manifest,
            manifest_file=manifest_file,
            variants=selected_variants,
            host=proxy_host,
            dynamic_ports=dynamic_proxy_ports,
            use_project_deepseek_config=use_project_deepseek_config,
            project_config_path=project_config_path,
            enable_groupchat_history_reordering=enable_groupchat_history_reordering,
        )

    variant_summaries: dict[str, Any] = {}
    tabulate_paths: dict[str, str] = {}
    try:
        for variant in selected_variants:
            info = _variant_info(manifest, variant)
            scenario_path = _artifact_path(info.get("scenario_path"), manifest_file=manifest_file)
            records = _filter_records_for_ids(_read_scenario_records(scenario_path), [task["id"] for task in selected_tasks])
            sample_scenario_path = sample_scenarios_dir / f"{effective_sample_id}.{variant}.jsonl"
            _write_jsonl(sample_scenario_path, records)
            config_list = _native_config_list(
                info=info,
                manifest=manifest,
                manifest_file=manifest_file,
                proxy=started.get(variant),
            )
            scenario_summary = _run_variant_records(
                variant=variant,
                scenario_path=sample_scenario_path,
                records=records,
                repeat=repeat,
                results_root=results_root,
                config_list=config_list,
                timeout_seconds=timeout_seconds,
                clear_existing=clear_existing,
            )
            tabulate_path = reports_dir / f"{variant}_native_tabulate.csv"
            _write_tabulate_csv(
                tabulate_path,
                results_scenario_dir=Path(scenario_summary["results_scenario_dir"]),
                task_ids=[str(record["id"]) for record in records],
            )
            tabulate_paths[variant] = str(tabulate_path)
            variant_summaries[variant] = {
                **scenario_summary,
                "sample_scenario_path": str(sample_scenario_path),
                "tabulate_csv_path": str(tabulate_path),
                "provider_telemetry_path": str(_artifact_path(info.get("provider_telemetry_path"), manifest_file=manifest_file)),
            }
    finally:
        for proxy in started.values():
            proxy.server.shutdown()
            proxy.server.server_close()
            proxy.thread.join(timeout=5)

    collection_summary: dict[str, Any] | None = None
    if run_collection:
        collection_target = collection_summary_path or reports_dir / "legacy_native_collection_summary.json"
        collection = collect_legacy_agbench_results(
            manifest_path=manifest_file,
            summary_path=collection_target,
            baseline_tabulate_csv=tabulate_paths.get("baseline"),
            rule_tabulate_csv=tabulate_paths.get("plugin_rule_only"),
            nl_tabulate_csv=tabulate_paths.get("plugin_nl_segmentation"),
            run_ab_eval=run_ab_eval,
        )
        collection_summary = collection.summary

    summary = {
        "schema_version": "prefix-legacy-agbench-native-run-v1",
        "prompt_safe_summary": True,
        "manifest_path": str(manifest_file),
        "suite_id": manifest.get("suite_id"),
        "sample_id": effective_sample_id,
        "execution_mode": "native_windows_proxy",
        "docker_used": False,
        "repeat": repeat,
        "variant_count": len(selected_variants),
        "variants": variant_summaries,
        "selected_task_count": len(selected_tasks),
        "selected_task_ids": tuple(str(task["id"]) for task in selected_tasks),
        "results_dir": str(results_root),
        "start_proxies": start_proxies,
        "dynamic_proxy_ports": dynamic_proxy_ports,
        "groupchat_history_reordering_enabled": enable_groupchat_history_reordering,
        "collection_summary_path": str(collection_summary_path or reports_dir / "legacy_native_collection_summary.json")
        if run_collection
        else None,
        "collection": _collection_digest(collection_summary),
        "notes": (
            "Runs official AutoGenBench scenario expansion and success scoring, but executes scenario.py with "
            "the current Python interpreter because Docker and autogenbench --native are unavailable on this Windows host."
        ),
    }
    target = _write_json(summary_path, summary)
    return LegacyNativeRunResult(
        manifest_path=str(manifest_file),
        summary_path=str(target) if target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed, prompt-safe legacy AutoGenBench sample on Windows by reusing official scenario expansion "
            "and tabulation while executing scenario.py with the current Python interpreter."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", action="append", choices=VARIANTS, help="Variant to run. Defaults to all three.")
    parser.add_argument("--task-id", action="append", help="Specific AutoGenBench task id to include.")
    parser.add_argument("--max-tasks", type=int, help="Use the first N tasks from the baseline scenario when --task-id is absent.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--sample-id")
    parser.add_argument("--summary")
    parser.add_argument("--collection-summary")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--enable-groupchat-history-reordering",
        action="store_true",
        help="Enable exact repeated groupchat dialogue prefix reordering in plugin proxies.",
    )
    parser.add_argument("--no-proxies", action="store_true", help="Do not start local forward proxies.")
    parser.add_argument("--no-project-deepseek-config", action="store_true")
    parser.add_argument("--project-config")
    parser.add_argument("--fixed-proxy-ports", action="store_true", help="Use manifest proxy ports instead of dynamic ports.")
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-ab-eval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_legacy_native_ab_sample(
        manifest_path=args.manifest,
        variants=tuple(args.variant or VARIANTS),
        task_ids=tuple(args.task_id or ()),
        max_tasks=args.max_tasks,
        repeat=args.repeat,
        results_dir=args.results_dir,
        sample_id=args.sample_id,
        summary_path=args.summary,
        collection_summary_path=args.collection_summary,
        start_proxies=not args.no_proxies,
        use_project_deepseek_config=not args.no_project_deepseek_config,
        project_config_path=args.project_config,
        dynamic_proxy_ports=not args.fixed_proxy_ports,
        proxy_host=args.proxy_host,
        timeout_seconds=args.timeout_seconds,
        enable_groupchat_history_reordering=args.enable_groupchat_history_reordering,
        clear_existing=not args.keep_existing,
        run_collection=not args.skip_collection,
        run_ab_eval=not args.skip_ab_eval,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _run_variant_records(
    *,
    variant: str,
    scenario_path: Path,
    records: Sequence[Mapping[str, Any]],
    repeat: int,
    results_root: Path,
    config_list: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
    clear_existing: bool,
) -> dict[str, Any]:
    scenario_name = scenario_path.stem
    scenario_dir = scenario_path.parent.resolve()
    results_scenario_dir = results_root / scenario_name
    runs: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record["id"])
        for trial_index in range(repeat):
            target = results_scenario_dir / task_id / str(trial_index)
            if target.exists() and clear_existing:
                _safe_rmtree(target, root=results_root)
            elif target.exists():
                runs.append(_run_record(target, task_id=task_id, trial_index=trial_index, skipped=True))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            expand_scenario(str(scenario_dir), dict(record), str(target), requirements=None)
            _ensure_testbed_utils_compat(target)
            runs.append(
                _execute_scenario(
                    target,
                    task_id=task_id,
                    trial_index=trial_index,
                    config_list=config_list,
                    timeout_seconds=timeout_seconds,
                )
            )
    success_values = [run["success"] for run in runs if run.get("success") is not None]
    success_count = sum(1 for success in success_values if success is True)
    return {
        "variant": variant,
        "scenario_path": str(scenario_path),
        "results_scenario_dir": str(results_scenario_dir),
        "run_count": len(runs),
        "success_evaluated_count": len(success_values),
        "success_count": success_count,
        "success_rate": success_count / len(success_values) if success_values else None,
        "timeout_count": sum(1 for run in runs if run.get("timeout")),
        "nonzero_exit_count": sum(1 for run in runs if (run.get("returncode") or 0) != 0),
        "runs": tuple(runs),
    }


def _execute_scenario(
    work_dir: Path,
    *,
    task_id: str,
    trial_index: int,
    config_list: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["OAI_CONFIG_LIST"] = json.dumps(tuple(config_list), ensure_ascii=False)
    env.setdefault("OPENAI_API_KEY", "prefix-proxy-placeholder")
    env["AUTOGEN_TESTBED_SETTING"] = "NativeWindowsProxy"
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    console_log = work_dir / "console_log.txt"
    timeout = False
    returncode: int | None
    try:
        completed = subprocess.run(
            [sys.executable, "scenario.py"],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        output = (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timeout = True
        returncode = None
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = stdout + stderr + f"\nSCENARIO.PY TIMED OUT AFTER {timeout_seconds:.1f}s !#!#\n"
    duration = time.perf_counter() - started
    console_log.write_text(output, encoding="utf-8")
    success = _score_instance(work_dir)
    return {
        "task_id": task_id,
        "trial_index": trial_index,
        "work_dir": str(work_dir),
        "console_log_path": str(console_log),
        "returncode": returncode,
        "timeout": timeout,
        "duration_seconds": duration,
        "success": success,
    }


def _ensure_testbed_utils_compat(work_dir: Path) -> None:
    # AutoGenBench 0.0.3 imports "from pkg_resources import packaging" in its
    # bundled testbed_utils.py. Modern slim virtualenvs can omit pkg_resources.
    shim = work_dir / "pkg_resources.py"
    if shim.exists():
        return
    shim.write_text("import packaging as packaging\n", encoding="utf-8")


def _run_record(target: Path, *, task_id: str, trial_index: int, skipped: bool) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "trial_index": trial_index,
        "work_dir": str(target),
        "console_log_path": str(target / "console_log.txt"),
        "returncode": None,
        "timeout": False,
        "duration_seconds": 0.0,
        "success": _score_instance(target),
        "skipped_existing": skipped,
    }


def _start_proxies(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
    variants: Sequence[str],
    host: str,
    dynamic_ports: bool,
    use_project_deepseek_config: bool,
    project_config_path: str | Path | None,
    enable_groupchat_history_reordering: bool,
) -> dict[str, _StartedProxy]:
    proxy_config = manifest.get("proxies") if isinstance(manifest.get("proxies"), Mapping) else {}
    started: dict[str, _StartedProxy] = {}
    for variant in variants:
        info = proxy_config.get(variant) if isinstance(proxy_config.get(variant), Mapping) else {}
        if not info:
            raise ValueError(f"manifest missing proxy config for {variant}")
        telemetry_path = _artifact_path(info.get("telemetry_path"), manifest_file=manifest_file)
        port = 0 if dynamic_ports else int(info.get("port") or 0)
        server = build_forward_proxy_server(
            host=host,
            port=port,
            upstream_base_url=str(info.get("upstream_base_url") or ""),
            session_id=str(info.get("session_id") or f"{manifest.get('suite_id')}-{variant}-proxy"),
            telemetry_log_path=telemetry_path,
            enabled=bool(info.get("rewrite_enabled")),
            enable_natural_language_segmentation=bool(info.get("natural_language_segmentation_enabled")),
            enable_groupchat_history_reordering=enable_groupchat_history_reordering and bool(info.get("rewrite_enabled")),
            use_project_deepseek_config=use_project_deepseek_config,
            project_config_path=project_config_path,
            reset_telemetry_log=True,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started[variant] = _StartedProxy(
            variant=variant,
            server=server,
            thread=thread,
            base_url=f"http://{host}:{server.server_port}/v1",
        )
    return started


def _native_config_list(
    *,
    info: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file: Path,
    proxy: _StartedProxy | None,
) -> tuple[dict[str, Any], ...]:
    path = _artifact_path(info.get("path"), manifest_file=manifest_file)
    configs = config_list_from_json(str(path))
    native: list[dict[str, Any]] = []
    fallback_base_url = str(info.get("base_url") or "")
    for item in configs:
        row = dict(item)
        row["base_url"] = proxy.base_url if proxy is not None else fallback_base_url
        if not row.get("api_key") or str(row.get("api_key")).startswith("${"):
            row["api_key"] = "prefix-proxy-placeholder"
        native.append(row)
    return tuple(native)


def _write_tabulate_csv(path: Path, *, results_scenario_dir: Path, task_ids: Sequence[str]) -> None:
    max_trials = 0
    for task_id in task_ids:
        task_dir = results_scenario_dir / task_id
        if not task_dir.exists():
            continue
        trial_indexes = [
            int(child.name)
            for child in task_dir.iterdir()
            if child.is_dir() and child.name.isdigit()
        ]
        if trial_indexes:
            max_trials = max(max_trials, max(trial_indexes) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Task Id", *(f"Trial {index} Success" for index in range(max_trials))])
        for task_id in task_ids:
            row: list[Any] = [task_id]
            for trial_index in range(max_trials):
                success = _score_instance(results_scenario_dir / task_id / str(trial_index))
                row.append("" if success is None else str(success))
            writer.writerow(row)


def _score_instance(instance_dir: Path) -> bool | None:
    try:
        return default_scorer(str(instance_dir))
    except UnicodeDecodeError:
        console_log = instance_dir / "console_log.txt"
        if not console_log.is_file():
            return None
        content = console_log.read_text(encoding="utf-8", errors="replace")
        return any(marker in content for marker in SUCCESS_STRINGS)


def _select_task_records(
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
    task_ids: Sequence[str],
    max_tasks: int | None,
) -> tuple[dict[str, Any], ...]:
    baseline = _variant_info(manifest, "baseline")
    records = _read_scenario_records(_artifact_path(baseline.get("scenario_path"), manifest_file=manifest_file))
    if task_ids:
        wanted = set(task_ids)
        selected = [record for record in records if str(record.get("id")) in wanted]
    else:
        selected = list(records[: max_tasks or len(records)])
    return tuple(selected)


def _filter_records_for_ids(records: Sequence[dict[str, Any]], task_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
    by_id = {str(record.get("id")): record for record in records}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"scenario missing selected task ids: {missing}")
    return tuple(by_id[task_id] for task_id in task_ids)


def _read_scenario_records(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain JSON objects")
            rows.append(value)
    return tuple(rows)


def _collection_digest(collection: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if collection is None:
        return None
    comparisons = collection.get("comparisons") if isinstance(collection.get("comparisons"), Mapping) else {}
    return {
        "ready_for_ab_report": collection.get("ready_for_ab_report"),
        "real_provider_metrics_available": collection.get("real_provider_metrics_available"),
        "task_metrics_available": collection.get("task_metrics_available"),
        "fake_upstream": collection.get("fake_upstream"),
        "comparison_status": {
            str(label): item.get("status") if isinstance(item, Mapping) else None
            for label, item in comparisons.items()
        },
    }


def _validate_variants(variants: Sequence[str]) -> tuple[str, ...]:
    if not variants:
        return VARIANTS
    invalid = [variant for variant in variants if variant not in VARIANTS]
    if invalid:
        raise ValueError(f"unsupported variants: {invalid}")
    return tuple(variants)


def _variant_info(manifest: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    configs = manifest.get("oai_config_lists") if isinstance(manifest.get("oai_config_lists"), Mapping) else {}
    info = configs.get(variant) if isinstance(configs.get(variant), Mapping) else None
    if info is None:
        raise ValueError(f"manifest missing OAI config for {variant}")
    return info


def _safe_rmtree(target: Path, *, root: Path) -> None:
    resolved_target = target.resolve()
    resolved_root = root.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"refusing to remove path outside results root: {resolved_target}")
    shutil.rmtree(resolved_target)


def _artifact_path(value: Any, *, manifest_file: Path) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value:
        path = Path(value)
    else:
        raise ValueError("manifest artifact path is missing")
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    manifest_relative = manifest_file.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return cwd_path


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


if __name__ == "__main__":
    raise SystemExit(main())
