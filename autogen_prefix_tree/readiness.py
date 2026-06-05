from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


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

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "checks": tuple(asdict(check) for check in self.checks),
            "suggested_commands": self.suggested_commands,
        }


def check_evaluation_readiness(
    *,
    cwd: str | Path = ".",
    env: Mapping[str, str] | None = None,
    require_api_config: bool = True,
    require_autogenbench: bool = True,
) -> EvaluationReadinessReport:
    root = Path(cwd)
    effective_env = env or os.environ
    checks = (
        _check_virtual_environment(root),
        _check_docker(),
        _check_api_config(root, effective_env, required=require_api_config),
        _check_autogenbench(required=require_autogenbench),
        _check_autogen_core(),
    )
    ready = all(check.ok for check in checks)
    return EvaluationReadinessReport(
        ready=ready,
        checks=checks,
        suggested_commands=_suggested_commands(root),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check readiness for real AutoGenBench/API evaluation.")
    parser.add_argument("--cwd", default=".", help="Project root to inspect.")
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
    if markers:
        return ReadinessCheck("api_config", True, "present: " + ", ".join(markers))
    return ReadinessCheck(
        "api_config",
        not required,
        "missing OPENAI_API_KEY/OAI_CONFIG_LIST",
        "Set OPENAI_API_KEY or provide OAI_CONFIG_LIST before real API evaluation.",
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


def _suggested_commands(root: Path) -> tuple[str, ...]:
    return (
        r".venv\Scripts\python.exe -m autogen_prefix_tree.microbench --telemetry tmp\prefix_microbench\requests.jsonl --summary tmp\prefix_microbench\summary.json --repeats 1",
        r'.venv\Scripts\python.exe -m pip install autogenbench "pyautogen==0.2.35"',
        r".venv\Scripts\autogenbench.exe clone HumanEval",
        "cd HumanEval",
        r"..\.venv\Scripts\autogenbench.exe run --subsample 0.1 --repeat 3 Tasks/human_eval_two_agents.jsonl",
        r"..\.venv\Scripts\autogenbench.exe tabulate Results/human_eval_two_agents",
    )


if __name__ == "__main__":
    raise SystemExit(main())
