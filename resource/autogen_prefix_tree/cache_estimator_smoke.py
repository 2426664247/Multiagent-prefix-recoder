from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from autogen_core.models import SystemMessage

from .cache_estimator import cache_estimator_config_name
from .compiler import LocalPromptCompiler
from .planner import HierarchicalPrefixPlanner, rewrite_messages
from .telemetry import dataclass_to_dict
from .validator import CacheUtilityValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-network smoke test for the unified CacheEstimator gate.")
    parser.add_argument(
        "--estimator",
        default="prefix-tree",
        choices=("prefix-tree", "cache-hit-proxy", "provider-telemetry"),
        help="Estimator to configure. cache-hit-proxy falls back to prefix-tree when unavailable.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for prompt-safe summary.json.")
    parser.add_argument("--session-id", default="cache-estimator-smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_smoke(
        estimator=args.estimator,
        output_dir=args.output_dir,
        session_id=args.session_id,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def run_smoke(*, estimator: str, output_dir: str | Path, session_id: str = "cache-estimator-smoke") -> dict[str, Any]:
    normalized_estimator = cache_estimator_config_name(estimator)
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    validator = CacheUtilityValidator(cache_estimator=estimator)

    cold = compiler.compile(_messages("planner"), session_id=session_id)
    planner.plan(cold, session_id=session_id)
    warm = compiler.compile(_messages("engineer"), session_id=session_id)
    plan = planner.plan(warm, session_id=session_id)
    rewritten = rewrite_messages(warm, plan)
    report = validator.validate(
        original_messages=warm.messages,
        rewritten_messages=rewritten,
        compile_result=warm,
        plan=plan,
    )
    cache_report = dataclass_to_dict(report.cache_estimate_report) or {}
    utility_report = dataclass_to_dict(report.utility_estimate) or {}
    summary = {
        "schema_version": "cache-estimator-smoke-summary-v1",
        "prompt_safe_summary": True,
        "network_access_required": False,
        "configured_estimator": normalized_estimator,
        "estimator_name": cache_report.get("estimator_name"),
        "estimator_available": cache_report.get("estimator_available"),
        "used_fallback": cache_report.get("used_fallback"),
        "validation_reason": report.reason,
        "applied": report.applied,
        "fallback": report.fallback,
        "cache_hit_increased": report.cache_hit_increased,
        "estimated_cache_gain": cache_report.get("estimated_cache_gain"),
        "cached_tokens_delta": cache_report.get("cached_tokens_delta"),
        "utility_estimated_gain_chars": utility_report.get("estimated_gain_chars"),
        "reason": cache_report.get("reason"),
        "warnings": cache_report.get("warnings"),
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _messages(agent: str) -> tuple[SystemMessage, ...]:
    return (
        SystemMessage(
            content="\n\n".join(
                [
                    "ROLE_SPECIFIC_INSTRUCTION_START\n"
                    f"AGENT_NAME: {agent}\n"
                    f"You are the {agent}.\n"
                    "ROLE_SPECIFIC_INSTRUCTION_END",
                    "USER_TASK_START\nBuild the cache estimator smoke.\nUSER_TASK_END",
                    "SHARED_GROUPCHAT_CONTEXT_START\nShared benchmark context.\nSHARED_GROUPCHAT_CONTEXT_END",
                    "TEAM_POLICY_START\nPreserve tool boundaries and verify before finalizing.\nTEAM_POLICY_END",
                    "TOOL_SCHEMA_START\nshared_tool(x: string) -> string\nTOOL_SCHEMA_END",
                    "OUTPUT_FORMAT_START\nReturn a concise status.\nOUTPUT_FORMAT_END",
                    "CURRENT_TURN_INSTRUCTION_START\nHandle this smoke request only.\nCURRENT_TURN_INSTRUCTION_END",
                ]
            )
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
