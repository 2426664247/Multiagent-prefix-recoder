from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogen_core.models import SystemMessage

from .compiler import LocalPromptCompiler
from .planner import HierarchicalPrefixPlanner, rewrite_messages
from .planner_ranker_baseline import PlannerRankerModel, load_planner_ranker_dataset
from .replay import build_replay_run_record
from .telemetry import dataclass_to_dict
from .utility_validator_baseline import UtilityValidatorModel, load_expanded_utility_validator_dataset
from .validator import CacheUtilityValidator


def run_framework_training_baseline_v0(
    *,
    repo_root: str | Path = ".",
    dataset_root: str | Path = "datasets/utility_validator",
    utility_model_path: str | Path = "artifacts/utility_validator/baseline_v0/model.pkl",
    planner_ranker_model_path: str | Path = "artifacts/planner_ranker/baseline_v0/model.pkl",
    output_dir: str | Path = "artifacts/framework_training/baseline_v0",
    utility_shadow_min_confidence: float = 0.55,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    utility_model = UtilityValidatorModel.load(Path(repo_root) / utility_model_path)
    planner_ranker = PlannerRankerModel.load(Path(repo_root) / planner_ranker_model_path)
    utility_dataset = load_expanded_utility_validator_dataset(repo_root=repo_root, dataset_root=dataset_root)
    planner_dataset = load_planner_ranker_dataset(repo_root=repo_root, dataset_root=dataset_root)

    test_split = utility_dataset.splits["expanded_test"]
    planner_test = planner_dataset.splits["expanded_test"]

    validator_shadow_report = _validator_shadow_report(
        rows=test_split.rows,
        labels=test_split.y,
        model=utility_model,
        min_confidence=utility_shadow_min_confidence,
    )
    planner_ranker_report = _planner_ranker_report(
        rows=planner_test.rows,
        labels=planner_test.y,
        utility_labels=planner_test.utility_labels,
        model=planner_ranker,
    )
    replay_smoke = _runtime_replay_smoke(
        utility_model=utility_model,
        planner_ranker=planner_ranker,
        min_confidence=utility_shadow_min_confidence,
    )
    framework_replay_report = _framework_replay_report(
        validator_shadow_report=validator_shadow_report,
        planner_ranker_report=planner_ranker_report,
        runtime_replay_smoke=replay_smoke,
    )

    _write_json(output / "validator_shadow_report.json", validator_shadow_report)
    _write_json(output / "planner_ranker_report.json", planner_ranker_report)
    _write_json(output / "framework_replay_report.json", framework_replay_report)
    (output / "framework_training_report.md").write_text(
        _framework_training_report(
            validator_shadow_report=validator_shadow_report,
            planner_ranker_report=planner_ranker_report,
            framework_replay_report=framework_replay_report,
            output_dir=output,
        ),
        encoding="utf-8",
    )

    return {
        "validator_shadow_report_path": str(output / "validator_shadow_report.json"),
        "planner_ranker_report_path": str(output / "planner_ranker_report.json"),
        "framework_replay_report_path": str(output / "framework_replay_report.json"),
        "framework_training_report_path": str(output / "framework_training_report.md"),
        "validator_shadow_report": validator_shadow_report,
        "planner_ranker_report": planner_ranker_report,
        "framework_replay_report": framework_replay_report,
    }


def _validator_shadow_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    model: UtilityValidatorModel,
    min_confidence: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    false_kill_risks = 0
    leak_risks = 0
    observe_count = 0
    for row, label in zip(rows, labels):
        prediction = model.predict(row)
        probability = model.predict_proba(row)
        predicted = bool(prediction["is_utility_preserved"])
        confidence = float(prediction["confidence"])
        shadow_decision = _shadow_decision(predicted=predicted, confidence=confidence, min_confidence=min_confidence)
        if shadow_decision == "observe":
            observe_count += 1
        model_gate_decision = _model_gate_decision(
            predicted=predicted,
            confidence=confidence,
            min_confidence=min_confidence,
        )
        actual = bool(label)
        if model_gate_decision == "reject" and actual:
            false_kill_risks += 1
        if model_gate_decision == "accept" and not actual:
            leak_risks += 1
        hard_gate_result = "passed" if row.get("hard_constraint_passed", True) is not False else "failed"
        cache_gate_result = "passed" if _cache_gate_value(row) > 0 else "failed"
        record = {
            "sample_id": row.get("sample_id"),
            "task_id": row.get("task_id"),
            "candidate_id": row.get("candidate_id"),
            "dataset_source": row.get("dataset_source"),
            "scenario_type": row.get("scenario_type"),
            "candidate_strategy": row.get("candidate_strategy"),
            "actual_is_utility_preserved": actual,
            "utility_model_prediction": predicted,
            "utility_model_probability": probability,
            "utility_model_confidence": confidence,
            "utility_model_threshold": model.threshold,
            "utility_model_shadow_decision": shadow_decision,
            "model_gate_if_enabled_decision": model_gate_decision,
            "hard_gate_result": hard_gate_result,
            "cache_gate_result": cache_gate_result,
            "estimated_cache_gain": _cache_gate_value(row),
        }
        records.append(record)
        counts[shadow_decision] += 1

    return {
        "schema_version": "framework-validator-shadow-report-v1",
        "model_name": model.model_name,
        "sample_count": len(records),
        "min_confidence": min_confidence,
        "shadow_decision_counts": dict(sorted(counts.items())),
        "model_gate_if_enabled": {
            "accepted_count": sum(1 for record in records if record["model_gate_if_enabled_decision"] == "accept"),
            "rejected_count": sum(1 for record in records if record["model_gate_if_enabled_decision"] == "reject"),
            "observe_count": sum(1 for record in records if record["model_gate_if_enabled_decision"] == "observe"),
            "false_kill_risk_count": false_kill_risks,
            "leak_risk_count": leak_risks,
        },
        "observe_count": observe_count,
        "records": records,
        "prompt_text_used": False,
        "ds_api_called": False,
    }


def _planner_ranker_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    utility_labels: Sequence[int],
    model: PlannerRankerModel,
) -> dict[str, Any]:
    ranked = list(model.rank(rows))
    label_by_id = {
        str(row.get("candidate_id") or row.get("sample_id") or row.get("label_id")): {
            "is_priority_candidate": bool(label),
            "is_utility_preserved": bool(utility_label),
            "task_id": row.get("task_id"),
        }
        for row, label, utility_label in zip(rows, labels, utility_labels)
    }
    enriched_ranked = []
    for rank, item in enumerate(ranked, start=1):
        labels_for_item = label_by_id.get(str(item.get("candidate_id")), {})
        enriched_ranked.append(
            {
                "rank": rank,
                **item,
                "actual_is_priority_candidate": labels_for_item.get("is_priority_candidate"),
                "actual_is_utility_preserved": labels_for_item.get("is_utility_preserved"),
                "task_id": labels_for_item.get("task_id"),
            }
        )

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_ranked:
        grouped[str(item.get("task_id") or "unknown")].append(item)
    per_task_rankings = {
        task_id: sorted(items, key=lambda value: int(value["rank"]))
        for task_id, items in sorted(grouped.items())
        if len(items) > 1
    }
    unsafe_in_top_5 = sum(1 for item in enriched_ranked[:5] if item.get("actual_is_utility_preserved") is False)
    priority_in_top_5 = sum(1 for item in enriched_ranked[:5] if item.get("actual_is_priority_candidate") is True)
    return {
        "schema_version": "framework-planner-ranker-report-v1",
        "model_name": model.model_name,
        "sample_count": len(enriched_ranked),
        "ranked_candidates": enriched_ranked,
        "per_task_rankings": per_task_rankings,
        "top_5": {
            "priority_count": priority_in_top_5,
            "unsafe_count": unsafe_in_top_5,
        },
        "prompt_text_used": False,
        "ds_api_called": False,
    }


def _runtime_replay_smoke(
    *,
    utility_model: UtilityValidatorModel,
    planner_ranker: PlannerRankerModel,
    min_confidence: float,
) -> dict[str, Any]:
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    planner.plan(compiler.compile(_messages("planner"), session_id="framework-replay"), session_id="framework-replay")
    compile_result = compiler.compile(_messages("engineer"), session_id="framework-replay")
    plan = planner.plan(compile_result, session_id="framework-replay")
    rewritten_messages = rewrite_messages(compile_result, plan)

    default_report = CacheUtilityValidator().validate(
        original_messages=compile_result.messages,
        rewritten_messages=rewritten_messages,
        compile_result=compile_result,
        plan=plan,
    )
    shadow_report = CacheUtilityValidator(
        utility_model_validator=utility_model,
        utility_model_mode="shadow",
        utility_model_min_confidence=min_confidence,
    ).validate(
        original_messages=compile_result.messages,
        rewritten_messages=rewritten_messages,
        compile_result=compile_result,
        plan=plan,
    )
    gate_report = CacheUtilityValidator(
        utility_model_validator=utility_model,
        utility_model_mode="gate",
        utility_model_min_confidence=min_confidence,
    ).validate(
        original_messages=compile_result.messages,
        rewritten_messages=rewritten_messages,
        compile_result=compile_result,
        plan=plan,
    )

    replay_record = build_replay_run_record(
        run_id="framework-replay-smoke",
        compile_result=compile_result,
        plan=plan,
        validation_report=shadow_report,
        candidates=(plan.prefix_tree_candidate,) if plan.prefix_tree_candidate is not None else (),
        final_decision="fallback" if shadow_report.fallback else "applied",
        include_text=False,
    )
    ranker_prediction = planner_ranker.predict(_runtime_ranker_row(compile_result=compile_result, plan=plan))
    return {
        "schema_version": "framework-runtime-replay-smoke-v1",
        "default_validation": _validation_summary(default_report),
        "shadow_validation": _validation_summary(shadow_report),
        "gate_if_enabled_validation": _validation_summary(gate_report),
        "shadow_does_not_change_default_decision": (
            default_report.applied == shadow_report.applied
            and default_report.fallback == shadow_report.fallback
            and default_report.reason == shadow_report.reason
        ),
        "replay_record_shadow_fields": {
            key: replay_record.get(key)
            for key in (
                "utility_model_prediction",
                "utility_model_confidence",
                "utility_model_threshold",
                "utility_model_shadow_decision",
                "hard_gate_result",
                "cache_gate_result",
            )
        },
        "planner_ranker_runtime_prediction": ranker_prediction,
        "prompt_text_included": replay_record.get("prompt_text_included"),
    }


def _framework_replay_report(
    *,
    validator_shadow_report: Mapping[str, Any],
    planner_ranker_report: Mapping[str, Any],
    runtime_replay_smoke: Mapping[str, Any],
) -> dict[str, Any]:
    gate_summary = validator_shadow_report.get("model_gate_if_enabled") or {}
    return {
        "schema_version": "framework-replay-report-v1",
        "hard_gate_checked": True,
        "utility_validator_shadow_checked": True,
        "cache_gain_gate_checked": True,
        "planner_ranker_sorting_checked": True,
        "model_gate_if_enabled_checked": True,
        "sample_level_model_gate_if_enabled": gate_summary,
        "runtime_replay_smoke": runtime_replay_smoke,
        "risk_summary": {
            "false_kill_risk_count": gate_summary.get("false_kill_risk_count", 0),
            "leak_risk_count": gate_summary.get("leak_risk_count", 0),
            "planner_top_5_unsafe_count": (planner_ranker_report.get("top_5") or {}).get("unsafe_count", 0),
        },
        "prompt_text_used": False,
        "ds_api_called": False,
    }


def _validation_summary(report: Any) -> dict[str, Any]:
    utility_model_gate = dataclass_to_dict(getattr(report, "utility_model_gate_report", None)) or {}
    return {
        "applied": report.applied,
        "fallback": report.fallback,
        "reason": report.reason,
        "hard_constraint_passed": report.hard_constraint_passed,
        "cache_hit_increased": report.cache_hit_increased,
        "utility_status": report.utility_status,
        "utility_model_gate_report": utility_model_gate,
        "utility_model_prediction": utility_model_gate.get("utility_model_prediction"),
        "utility_model_confidence": utility_model_gate.get("utility_model_confidence"),
        "utility_model_threshold": utility_model_gate.get("utility_model_threshold"),
        "utility_model_shadow_decision": utility_model_gate.get("utility_model_shadow_decision"),
        "hard_gate_result": utility_model_gate.get("hard_gate_result"),
        "cache_gate_result": utility_model_gate.get("cache_gate_result"),
    }


def _runtime_ranker_row(*, compile_result: Any, plan: Any) -> dict[str, Any]:
    from .utility_validator_baseline import runtime_feature_row_from_gate_inputs

    return runtime_feature_row_from_gate_inputs(
        original_messages=compile_result.messages,
        rewritten_messages=rewrite_messages(compile_result, plan),
        compile_result=compile_result,
        plan=plan,
        cache_utility_estimate=CacheUtilityValidator()._estimate_cache_utility(compile_result, plan, None),
    )


def _shadow_decision(*, predicted: bool, confidence: float, min_confidence: float) -> str:
    if confidence < min_confidence:
        return "observe"
    return "would_accept" if predicted else "would_reject"


def _model_gate_decision(*, predicted: bool, confidence: float, min_confidence: float) -> str:
    if confidence < min_confidence:
        return "observe"
    return "accept" if predicted else "reject"


def _cache_gate_value(row: Mapping[str, Any]) -> float:
    cache_estimate = row.get("cache_estimate_report") if isinstance(row.get("cache_estimate_report"), Mapping) else {}
    cache_gain_report = row.get("cache_gain_report") if isinstance(row.get("cache_gain_report"), Mapping) else {}
    value = row.get("estimated_cache_gain") or cache_estimate.get("estimated_cache_gain") or cache_gain_report.get("estimated_cache_gain")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _messages(agent: str) -> tuple[SystemMessage, ...]:
    return (
        SystemMessage(
            content="\n\n".join(
                [
                    "ROLE_SPECIFIC_INSTRUCTION_START\n"
                    f"AGENT_NAME: {agent}\n"
                    f"You are {agent}.\n"
                    "ROLE_SPECIFIC_INSTRUCTION_END",
                    "USER_TASK_START\nBuild the framework replay report.\nUSER_TASK_END",
                    "SHARED_GROUPCHAT_CONTEXT_START\nShared project context.\nSHARED_GROUPCHAT_CONTEXT_END",
                    "TOOL_SCHEMA_START\nshared_tool(x: string) -> string\nTOOL_SCHEMA_END",
                    "OUTPUT_FORMAT_START\nReturn a concise status.\nOUTPUT_FORMAT_END",
                ]
            )
        ),
    )


def _framework_training_report(
    *,
    validator_shadow_report: Mapping[str, Any],
    planner_ranker_report: Mapping[str, Any],
    framework_replay_report: Mapping[str, Any],
    output_dir: Path,
) -> str:
    gate = validator_shadow_report.get("model_gate_if_enabled") or {}
    risk = framework_replay_report.get("risk_summary") or {}
    return "\n".join(
        [
            "# Framework Training baseline_v0 Report",
            "",
            "baseline_v0 is a replay and shadow validation baseline, not a final enabled policy.",
            "",
            "## Validator Shadow",
            f"- samples: {validator_shadow_report.get('sample_count')}",
            f"- shadow decisions: {validator_shadow_report.get('shadow_decision_counts')}",
            f"- model gate if enabled: {gate}",
            "",
            "## Planner Ranker",
            f"- samples: {planner_ranker_report.get('sample_count')}",
            f"- top_5: {planner_ranker_report.get('top_5')}",
            "",
            "## Replay Checks",
            f"- shadow preserves default decision: {(framework_replay_report.get('runtime_replay_smoke') or {}).get('shadow_does_not_change_default_decision')}",
            f"- risk summary: {risk}",
            "",
            "## Artifacts",
            f"- {output_dir / 'validator_shadow_report.json'}",
            f"- {output_dir / 'planner_ranker_report.json'}",
            f"- {output_dir / 'framework_replay_report.json'}",
            f"- {output_dir / 'framework_training_report.md'}",
            "",
        ]
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run framework training baseline_v0 replay/shadow reports.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset-root", default="datasets/utility_validator")
    parser.add_argument("--utility-model", default="artifacts/utility_validator/baseline_v0/model.pkl")
    parser.add_argument("--planner-ranker-model", default="artifacts/planner_ranker/baseline_v0/model.pkl")
    parser.add_argument("--output-dir", default="artifacts/framework_training/baseline_v0")
    parser.add_argument("--utility-shadow-min-confidence", type=float, default=0.55)
    args = parser.parse_args(argv)
    result = run_framework_training_baseline_v0(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        utility_model_path=args.utility_model,
        planner_ranker_model_path=args.planner_ranker_model,
        output_dir=args.output_dir,
        utility_shadow_min_confidence=args.utility_shadow_min_confidence,
    )
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_report")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
