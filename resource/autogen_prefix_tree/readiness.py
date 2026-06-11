from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

from .project_api_config import load_project_provider_config


READINESS_MODES = ("all", "component", "legacy-proxy")


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    detail: str
    next_step: str | None = None


@dataclass(frozen=True)
class EvaluationReadinessReport:
    ready: bool
    checks: tuple[ReadinessCheck, ...]
    suggested_commands: tuple[str, ...]
    mode: str = "all"

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "mode": self.mode,
            "checks": tuple(asdict(check) for check in self.checks),
            "suggested_commands": self.suggested_commands,
        }


def check_evaluation_readiness(
    *,
    cwd: str | Path = ".",
    env: Mapping[str, str] | None = None,
    require_api_config: bool = True,
    require_autogenbench: bool = True,
    mode: str = "all",
) -> EvaluationReadinessReport:
    root = Path(cwd)
    effective_env = os.environ if env is None else env
    normalized_mode = _normalize_mode(mode)
    checks = [
        _check_virtual_environment(root),
        _check_docker(),
        _check_api_config(root, effective_env, required=require_api_config),
        _check_autogenbench(required=require_autogenbench),
        _check_autogen_core(),
    ]
    if normalized_mode in {"all", "component"}:
        checks.extend(
            (
                _check_autogen_ext_openai(),
                _check_allowed_provider_namespace(effective_env),
            )
        )
    ready = all(check.ok for check in checks)
    return EvaluationReadinessReport(
        ready=ready,
        checks=tuple(checks),
        suggested_commands=_suggested_commands(root, normalized_mode),
        mode=normalized_mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check readiness for real AutoGenBench/API evaluation.")
    parser.add_argument("--cwd", default=".", help="Project root to inspect.")
    parser.add_argument(
        "--mode",
        choices=READINESS_MODES,
        default="all",
        help=(
            "Evaluation path to check. 'legacy-proxy' targets autogenbench 0.0.3 OAI_CONFIG_LIST "
            "plus openai_forward_proxy and does not require autogen_ext.models.openai or component namespace setup."
        ),
    )
    parser.add_argument("--allow-missing-api", action="store_true", help="Do not fail readiness when API config is absent.")
    parser.add_argument(
        "--allow-missing-autogenbench",
        action="store_true",
        help="Do not fail readiness when autogenbench is not installed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_evaluation_readiness(
        cwd=args.cwd,
        require_api_config=not args.allow_missing_api,
        require_autogenbench=not args.allow_missing_autogenbench,
        mode=args.mode,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ready else 1


def _check_virtual_environment(root: Path) -> ReadinessCheck:
    local_venv = root / ".venv"
    active_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if active_venv:
        return ReadinessCheck("python_environment", True, f"active virtual environment: {sys.prefix}")
    if local_venv.exists():
        return ReadinessCheck(
            "python_environment",
            False,
            f"local .venv exists but current interpreter is not inside it: {sys.executable}",
            r"Use .venv\Scripts\python.exe for evaluation commands.",
        )
    return ReadinessCheck(
        "python_environment",
        False,
        "no active virtual environment detected",
        "Create one with python -m venv .venv and run commands through .venv.",
    )


def _check_docker() -> ReadinessCheck:
    docker = shutil.which("docker")
    if docker is None:
        return ReadinessCheck(
            "docker",
            False,
            "docker executable not found",
            "Install Docker Desktop or Docker Engine before AutoGenBench runs.",
        )
    try:
        completed = subprocess.run(
            [docker, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return ReadinessCheck("docker", False, f"docker version check failed: {type(exc).__name__}", "Verify Docker.")
    if completed.returncode != 0:
        return ReadinessCheck("docker", False, "docker --version returned non-zero", "Verify Docker.")
    return ReadinessCheck("docker", True, completed.stdout.strip())


def _check_api_config(root: Path, env: Mapping[str, str], *, required: bool) -> ReadinessCheck:
    markers = []
    if env.get("OPENAI_API_KEY"):
        markers.append("OPENAI_API_KEY")
    if env.get("OAI_CONFIG_LIST"):
        markers.append("OAI_CONFIG_LIST env")
    if (root / "OAI_CONFIG_LIST").exists():
        markers.append("OAI_CONFIG_LIST file")
    project_config = load_project_provider_config(cwd=root, env=env)
    if project_config is not None:
        markers.append(project_config.redacted_marker())
    if markers:
        return ReadinessCheck("api_config", True, "present: " + ", ".join(markers))
    return ReadinessCheck(
        "api_config",
        not required,
        "missing OPENAI_API_KEY/OAI_CONFIG_LIST/project DeepSeek config",
        "Set OPENAI_API_KEY, provide OAI_CONFIG_LIST, or create config/config.txt with DeepSeek credentials before real API evaluation.",
    )


def _check_autogenbench(*, required: bool) -> ReadinessCheck:
    cli = _resolve_autogenbench_cli()
    package = importlib.util.find_spec("autogenbench")
    if cli is None:
        if package is not None:
            return ReadinessCheck(
                "autogenbench",
                not required,
                "package installed but console script not found",
                r'Reinstall in .venv with .venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35".',
            )
        return ReadinessCheck(
            "autogenbench",
            not required,
            "not installed",
            r'Install in .venv with .venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35".',
        )
    try:
        completed = subprocess.run(
            [str(cli), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001
        return ReadinessCheck(
            "autogenbench",
            not required,
            f"cli smoke failed: {type(exc).__name__}",
            "Verify the AutoGenBench console script inside .venv.",
        )
    if completed.returncode != 0:
        first_line = _first_output_line(completed.stderr) or _first_output_line(completed.stdout)
        detail = f"cli smoke failed with exit {completed.returncode}"
        if first_line:
            detail += f": {first_line}"
        return ReadinessCheck(
            "autogenbench",
            not required,
            detail,
            "Verify AutoGenBench/pyautogen compatibility inside .venv; autogenbench 0.0.3 works with pyautogen==0.2.35.",
        )
    version = _first_output_line(completed.stdout)
    detail = f"cli smoke ok: {cli}"
    if version:
        detail += f"; {version}"
    return ReadinessCheck("autogenbench", True, detail)


def _resolve_autogenbench_cli() -> Path | None:
    cli = shutil.which("autogenbench")
    if cli:
        return Path(cli)
    executable_dir = Path(sys.executable).parent
    names = ("autogenbench.exe", "autogenbench-script.py") if os.name == "nt" else ("autogenbench",)
    for name in names:
        candidate = executable_dir / name
        if candidate.exists():
            return candidate
    return None


def _first_output_line(value: str) -> str:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _check_autogen_core() -> ReadinessCheck:
    if importlib.util.find_spec("autogen_core") is None:
        return ReadinessCheck(
            "autogen_core",
            False,
            "autogen_core not installed",
            r"Install in .venv with .venv\Scripts\python.exe -m pip install autogen-core.",
        )
    return ReadinessCheck("autogen_core", True, "installed")


def _check_autogen_ext_openai() -> ReadinessCheck:
    try:
        spec = importlib.util.find_spec("autogen_ext.models.openai")
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return ReadinessCheck(
            "autogen_ext_openai",
            False,
            "autogen_ext.models.openai not installed",
            rf'Install the matching AutoGen extension package: .venv\Scripts\python.exe -m pip install "{_autogen_ext_openai_requirement()}".',
        )
    return ReadinessCheck("autogen_ext_openai", True, "installed")


def _check_allowed_provider_namespace(env: Mapping[str, str]) -> ReadinessCheck:
    value = env.get("AUTOGEN_ALLOWED_PROVIDER_NAMESPACES", "")
    namespaces = {item.strip() for item in value.split(",") if item.strip()}
    accepted = any(namespace == "autogen_prefix_tree" or namespace.startswith("autogen_prefix_tree.") for namespace in namespaces)
    if accepted:
        return ReadinessCheck("autogen_prefix_tree_provider_namespace", True, "allowed")
    return ReadinessCheck(
        "autogen_prefix_tree_provider_namespace",
        False,
        "AUTOGEN_ALLOWED_PROVIDER_NAMESPACES does not include autogen_prefix_tree",
        "Set AUTOGEN_ALLOWED_PROVIDER_NAMESPACES=autogen_prefix_tree before loading wrapper model_config.",
    )


def _suggested_commands(root: Path, mode: str) -> tuple[str, ...]:
    component_commands = (
        rf'.venv\Scripts\python.exe -m pip install "{_autogen_ext_openai_requirement()}"',
        r'$env:AUTOGEN_ALLOWED_PROVIDER_NAMESPACES = "autogen_prefix_tree"',
        r".venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --mode component --allow-missing-api",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.microbench --telemetry tmp\prefix_microbench\requests.jsonl --summary tmp\prefix_microbench\summary.json --repeats 1",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_suite --source-root <autogen-source-root> --output-dir runs\offline_semantic_suite --session-id autogen-source-offline-suite",
    )
    legacy_proxy_commands = (
        r".venv\Scripts\python.exe -m autogen_prefix_tree.readiness --cwd . --mode legacy-proxy --allow-missing-api",
        r'.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"',
        r".venv\Scripts\autogenbench.exe clone HumanEval",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_suite --output-dir runs\agbench_legacy_prefix_suite --suite-id agbench-legacy-humaneval --model <model> --upstream-base-url <upstream-base-url> --scenario-jsonl HumanEval\Tasks\human_eval_two_agents.jsonl",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_suite_verify --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json --summary runs\agbench_legacy_prefix_suite\reports\legacy_suite_verification_summary.json",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.offline_semantic_suite --source-root <autogen-source-root> --output-dir runs\offline_semantic_suite --session-id autogen-source-offline-suite",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.three_proxy_smoke --output-dir runs\agbench_legacy_prefix_suite\preflight_three_proxy_smoke --session-id agbench-legacy-humaneval-preflight-smoke --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.semantic_guard_proxy_smoke --output-dir runs\agbench_legacy_prefix_suite\preflight_semantic_guard_proxy_smoke --session-id agbench-legacy-humaneval-semantic-guard-smoke --judge-mode reject",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight_smokes --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json --output-dir runs\agbench_legacy_prefix_suite\preflight_smokes",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_preflight --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json --summary runs\agbench_legacy_prefix_suite\reports\legacy_real_ab_preflight_summary.json --report-md runs\agbench_legacy_prefix_suite\reports\legacy_real_ab_preflight.md",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy --host 127.0.0.1 --port 8787 --upstream-base-url <upstream-base-url> --telemetry runs\agbench_legacy_prefix_suite\artifacts\baseline_provider_telemetry.jsonl --session-id agbench-legacy-humaneval-baseline-proxy --disabled",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy --host 127.0.0.1 --port 8788 --upstream-base-url <upstream-base-url> --telemetry runs\agbench_legacy_prefix_suite\artifacts\plugin_rule_only_provider_telemetry.jsonl --session-id agbench-legacy-humaneval-plugin-rule-only-proxy",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.openai_forward_proxy --host 127.0.0.1 --port 8789 --upstream-base-url <upstream-base-url> --telemetry runs\agbench_legacy_prefix_suite\artifacts\plugin_nl_segmentation_provider_telemetry.jsonl --session-id agbench-legacy-humaneval-plugin-nl-segmentation-proxy --enable-natural-language-segmentation",
        r".venv\Scripts\autogenbench.exe run -c runs\agbench_legacy_prefix_suite\OAI_CONFIG_LIST.baseline.json --model baseline --subsample 0.1 --repeat 3 runs\agbench_legacy_prefix_suite\scenarios\agbench-legacy-humaneval.baseline.jsonl",
        r".venv\Scripts\autogenbench.exe run -c runs\agbench_legacy_prefix_suite\OAI_CONFIG_LIST.plugin_rule_only.json --model plugin_rule_only --subsample 0.1 --repeat 3 runs\agbench_legacy_prefix_suite\scenarios\agbench-legacy-humaneval.plugin_rule_only.jsonl",
        r".venv\Scripts\autogenbench.exe run -c runs\agbench_legacy_prefix_suite\OAI_CONFIG_LIST.plugin_nl_segmentation.json --model plugin_nl_segmentation --subsample 0.1 --repeat 3 runs\agbench_legacy_prefix_suite\scenarios\agbench-legacy-humaneval.plugin_nl_segmentation.jsonl",
        r".venv\Scripts\autogenbench.exe tabulate Results\agbench-legacy-humaneval.baseline -c > runs\agbench_legacy_prefix_suite\reports\baseline_tabulate.csv",
        r".venv\Scripts\autogenbench.exe tabulate Results\agbench-legacy-humaneval.plugin_rule_only -c > runs\agbench_legacy_prefix_suite\reports\plugin_rule_only_tabulate.csv",
        r".venv\Scripts\autogenbench.exe tabulate Results\agbench-legacy-humaneval.plugin_nl_segmentation -c > runs\agbench_legacy_prefix_suite\reports\plugin_nl_segmentation_tabulate.csv",
        r".venv\Scripts\python.exe -m autogen_prefix_tree.agbench_legacy_collect --manifest runs\agbench_legacy_prefix_suite\legacy_suite_manifest.json --baseline-tabulate-csv runs\agbench_legacy_prefix_suite\reports\baseline_tabulate.csv --rule-tabulate-csv runs\agbench_legacy_prefix_suite\reports\plugin_rule_only_tabulate.csv --nl-tabulate-csv runs\agbench_legacy_prefix_suite\reports\plugin_nl_segmentation_tabulate.csv --summary runs\agbench_legacy_prefix_suite\reports\legacy_collection_summary.json",
    )
    if mode == "component":
        return component_commands
    if mode == "legacy-proxy":
        return legacy_proxy_commands
    return component_commands + legacy_proxy_commands


def _normalize_mode(mode: str) -> str:
    if mode not in READINESS_MODES:
        raise ValueError(f"unsupported readiness mode: {mode}")
    return mode


def _autogen_ext_openai_requirement() -> str:
    try:
        version = metadata.version("autogen-core")
    except metadata.PackageNotFoundError:
        return "autogen-ext[openai]"
    return f"autogen-ext[openai]=={version}"


if __name__ == "__main__":
    raise SystemExit(main())
