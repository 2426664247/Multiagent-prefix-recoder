from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from autogen_core.models import LLMMessage, SystemMessage

from .compiler import LocalPromptCompiler
from .planner import HierarchicalPrefixPlanner, rewrite_messages
from .telemetry import dataclass_to_dict, serialize_prefix_tree_candidate
from .validator import CacheUtilityValidator


@dataclass(frozen=True)
class UtilityGoldsetTemplateResult:
    input_paths: tuple[str, ...]
    output_path: str
    summary_path: str | None
    summary: dict[str, Any]


def build_humaneval_utility_goldset_template(
    *,
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    summary_path: str | Path | None = None,
    include_text: bool = False,
    max_rows: int | None = 8,
    seed: int = 20260611,
    scenario_ids: Sequence[str] = ("two_agents", "groupchat_three_agents"),
) -> UtilityGoldsetTemplateResult:
    if not input_paths:
        raise ValueError("at least one HumanEval task JSONL path is required")
    source_paths = tuple(Path(path) for path in input_paths)
    tasks = tuple(
        task
        for path in source_paths
        for task in _load_humaneval_tasks(path)
    )
    selected = _select_tasks(tasks, max_rows=max_rows, seed=seed)
    rows = tuple(
        _build_annotation_row(task=task, scenario_id=scenario_id, include_text=include_text, output_index=index)
        for index, (task, scenario_id) in enumerate(
            (item for task in selected for item in ((task, scenario_id) for scenario_id in scenario_ids)),
            start=1,
        )
    )
    output_target = Path(output_path)
    _write_jsonl(output_target, rows)
    summary = _summary(
        input_paths=tuple(str(path) for path in source_paths),
        output_path=str(output_target),
        rows=rows,
        task_count=len(tasks),
        selected_task_count=len(selected),
        include_text=include_text,
        max_rows=max_rows,
        seed=seed,
        scenario_ids=tuple(scenario_ids),
    )
    summary_target = _write_json(summary_path, summary)
    return UtilityGoldsetTemplateResult(
        input_paths=tuple(str(path) for path in source_paths),
        output_path=str(output_target),
        summary_path=str(summary_target) if summary_target else None,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a HumanEval utility-preservation annotation template for validated prefix reorders."
    )
    parser.add_argument("--input", action="append", required=True, help="HumanEval task JSONL. Repeatable.")
    parser.add_argument("--output", required=True, help="Output utility goldset template JSONL.")
    parser.add_argument("--summary", help="Optional prompt-safe summary JSON.")
    parser.add_argument("--max-rows", "--sample-size", dest="max_rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        help="Scenario id to include. Repeatable. Defaults to two_agents and groupchat_three_agents.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include original/reordered block text for local annotation. Keep the output local.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_humaneval_utility_goldset_template(
        input_paths=args.input,
        output_path=args.output,
        summary_path=args.summary,
        include_text=args.include_text,
        max_rows=args.max_rows,
        seed=args.seed,
        scenario_ids=tuple(args.scenario_ids or ("two_agents", "groupchat_three_agents")),
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _build_annotation_row(
    *,
    task: Mapping[str, Any],
    scenario_id: str,
    include_text: bool,
    output_index: int,
) -> dict[str, Any]:
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    validator = CacheUtilityValidator()
    session_id = f"utility-goldset:{scenario_id}:{task.get('id')}"
    cold = compiler.compile(_messages_for_task(task, agent="planner", scenario_id=scenario_id), session_id=session_id)
    planner.plan(cold, session_id=session_id)
    candidate = compiler.compile(_messages_for_task(task, agent="engineer", scenario_id=scenario_id), session_id=session_id)
    plan = planner.plan(candidate, session_id=session_id)
    rewritten = rewrite_messages(candidate, plan)
    report = validator.validate(
        original_messages=candidate.messages,
        rewritten_messages=rewritten,
        compile_result=candidate,
        plan=plan,
    )
    blocks_by_id = {block.block_id: block for block in candidate.blocks}
    original_blocks = tuple(_block_metadata(blocks_by_id[block_id], include_text=include_text) for block_id in plan.original_order)
    reordered_blocks = tuple(_block_metadata(blocks_by_id[block_id], include_text=include_text) for block_id in plan.new_order)
    moved = tuple(
        _moved_block_summary(blocks_by_id[block_id], plan.move_reason.get(block_id), include_text=include_text)
        for block_id in plan.moved_blocks
        if block_id in blocks_by_id
    )
    utility_estimate = report.utility_estimate
    cache_estimate_report = dataclass_to_dict(report.cache_estimate_report)
    candidate_record = serialize_prefix_tree_candidate(plan.prefix_tree_candidate, include_text=include_text)
    materialized_prompts = (
        dict(plan.prefix_tree_candidate.materialized_prompts or {})
        if include_text and plan.prefix_tree_candidate is not None
        else {
            agent_id: _safe_hash(value)
            for agent_id, value in (plan.prefix_tree_candidate.materialized_prompts or {}).items()
        }
        if plan.prefix_tree_candidate is not None
        else {}
    )
    placement_changes = tuple(
        {
            "placement_id": placement.placement_id,
            "block_id": placement.block_id,
            "block_hash": placement.block_hash,
            "original_scope": _scope_value(placement.original_scope),
            "target_scope": _scope_value(placement.target_scope),
            "target_agent_group": placement.target_agent_group,
            "moved": placement.moved,
            "risk_tags": placement.risk_tags,
            "dependency_notes": placement.dependency_notes,
            "cache_contribution": placement.cache_contribution,
            "placement_score": placement.placement_score,
            "placement_score_breakdown": placement.placement_score_breakdown,
        }
        for placement in plan.placements
    )
    item: dict[str, Any] = {
        "schema_version": "prefix-utility-goldset-template-item-v1",
        "id": _safe_id(f"utility-{output_index:04d}:{scenario_id}:{task.get('id')}"),
        "task_id": task.get("id"),
        "scenario_id": scenario_id,
        "entry_point": _entry_point(task),
        "source_template": task.get("template"),
        "original_prompt_blocks": original_blocks,
        "prefix_tree_candidate": candidate_record,
        "materialized_reordered_prompts": materialized_prompts,
        "moved_blocks_summary": moved,
        "placement_changes": placement_changes,
        "hard_constraint_passed": report.hard_constraint_passed,
        "hard_constraints_passed": report.hard_constraint_passed,
        "hard_constraint_report": report.hard_constraint_report,
        "validation_reason": report.reason,
        "estimated_cache_gain": plan.estimated_cache_gain,
        "expected_is_utility_preserved": None,
        "annotator_reason": "",
        "annotation_reason": None,
        "oracle_result": None,
        "label_status": "unlabeled",
        "label_options": [True, False],
        "annotation_focus": _annotation_focus(moved),
        "original_block_order": original_blocks,
        "reordered_block_order": reordered_blocks,
        "moved_block_summary": moved,
        "moved_blocks_summary_legacy": moved,
        "utility_preservation": dataclass_to_dict(report.utility_preservation_report),
        "cache_estimate_report": cache_estimate_report,
        "cache_hit_increased": bool(report.cache_hit_increased),
        "estimated_gain_chars": utility_estimate.estimated_gain_chars if utility_estimate else None,
        "cached_tokens_delta": (
            report.cache_estimate_report.cached_tokens_delta
            if report.cache_estimate_report is not None
            else None
        ),
        "cacheable_prefix_block_count": len(plan.cacheable_prefix_blocks),
        "moved_block_count": len(plan.moved_blocks),
        "prompt_text_included": include_text,
        "manual_labeling_required": True,
        "notes": (
            "Label only legal reorder utility preservation. Hard-constraint failures belong to validator tests, "
            "not this utility goldset."
        ),
    }
    return {
        key: value
        for key, value in item.items()
        if value is not None or key in {"expected_is_utility_preserved", "annotation_reason", "oracle_result"}
    }


def _messages_for_task(task: Mapping[str, Any], *, agent: str, scenario_id: str) -> tuple[LLMMessage, ...]:
    prompt = _prompt_text(task)
    system_text = "\n\n".join(
        [
            "ROLE_SPECIFIC_INSTRUCTION_START\n"
            f"AGENT_NAME: {agent}\n"
            f"You are the {agent} in the {scenario_id} HumanEval workflow.\n"
            "ROLE_SPECIFIC_INSTRUCTION_END",
            "USER_TASK_START\n"
            f"{prompt}\n"
            "USER_TASK_END",
            "SHARED_GROUPCHAT_CONTEXT_START\n"
            "The team should solve the programming task, run available tests, and preserve the requested function signature.\n"
            "SHARED_GROUPCHAT_CONTEXT_END",
            "TEAM_POLICY_START\n"
            "Keep reasoning scoped to the task, do not leak private memory, and verify code before finalizing.\n"
            "TEAM_POLICY_END",
            "TOOL_SCHEMA_START\n"
            "write_file(path: string, content: string) -> string\n"
            "run_tests(path: string) -> string\n"
            "TOOL_SCHEMA_END",
            "OUTPUT_FORMAT_START\n"
            "Return the final answer as a concise implementation summary and test status.\n"
            "OUTPUT_FORMAT_END",
            "CURRENT_TURN_INSTRUCTION_START\n"
            "Work on the current HumanEval function only.\n"
            "CURRENT_TURN_INSTRUCTION_END",
        ]
    )
    return (SystemMessage(content=system_text),)


def _block_metadata(block: Any, *, include_text: bool) -> dict[str, Any]:
    row = {
        "block_id": block.block_id,
        "source_role": block.source_role,
        "source_type": block.source_type,
        "agent_or_source": block.agent_or_source,
        "content_hash": block.content_hash,
        "semantic_hint": block.semantic_type.value,
        "movability": block.movability.value,
        "share_scope": block.share_scope.value,
        "risk_tags": block.risk_tags,
        "has_hard_risk": block.has_hard_risk,
        "dependency_refs": block.dependency_refs,
        "summary": block.summary,
        "original_position": {
            "message_index": block.original_position.message_index,
            "part_index": block.original_position.part_index,
        },
    }
    if include_text and block.rendered_text is not None:
        row["text"] = block.rendered_text
    return row


def _moved_block_summary(block: Any, move_reason: str | None, *, include_text: bool) -> dict[str, Any]:
    row = {
        "block_id": block.block_id,
        "semantic_hint": block.semantic_type.value,
        "movability": block.movability.value,
        "share_scope": block.share_scope.value,
        "risk_tags": block.risk_tags,
        "dependency_refs": block.dependency_refs,
        "summary": block.summary,
        "move_reason": move_reason,
    }
    if include_text and block.rendered_text is not None:
        row["text"] = block.rendered_text
    return row


def _annotation_focus(moved_blocks: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    focus: list[str] = []
    moved_types = {str(block.get("semantic_hint")) for block in moved_blocks}
    risk_tags = {
        str(tag)
        for block in moved_blocks
        for tag in (block.get("risk_tags") if isinstance(block.get("risk_tags"), (list, tuple)) else ())
    }
    if "output_format" in moved_types:
        focus.append("output_format_movement_may_affect_test_reporting")
    if "shared_tool_description" in moved_types:
        focus.append("tool_instruction_movement_may_affect_code_or_test_actions")
    if "global_task_background" in moved_types:
        focus.append("task_background_movement_should_preserve_function_semantics")
    if "team_policy" in moved_types:
        focus.append("team_policy_movement_should_preserve_collaboration_behavior")
    if "conditional_instruction" in risk_tags:
        focus.append("conditional_instruction_order_sensitivity")
    if not focus:
        focus.append("conservative_shared_prefix_reorder")
    return tuple(focus)


def _summary(
    *,
    input_paths: tuple[str, ...],
    output_path: str,
    rows: Sequence[Mapping[str, Any]],
    task_count: int,
    selected_task_count: int,
    include_text: bool,
    max_rows: int | None,
    seed: int,
    scenario_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "prefix-utility-goldset-template-summary-v1",
        "prompt_safe_summary": True,
        "input_paths": input_paths,
        "output_path": output_path,
        "task_count": task_count,
        "selected_task_count": selected_task_count,
        "row_count": len(rows),
        "max_rows": max_rows,
        "seed": seed,
        "scenario_ids": scenario_ids,
        "include_text": include_text,
        "gold_text_written": include_text,
        "expected_label_pending_count": sum(1 for row in rows if row.get("expected_is_utility_preserved") is None),
        "validation_reason_counts": dict(Counter(str(row.get("validation_reason")) for row in rows)),
        "annotation_focus_counts": _focus_counts(rows),
        "cache_hit_increased_count": sum(1 for row in rows if row.get("cache_hit_increased") is True),
        "manual_labeling_required": True,
        "ready_for_local_utility_validator_training": False,
        "recommendation": (
            "fill_expected_is_utility_preserved_then_train_or_evaluate_local_utility_validator"
            if rows
            else "add_humaneval_tasks_before_utility_goldset_labeling"
        ),
        "limits": (
            "This template contains validated reorder metadata for manual utility labels. It does not train a "
            "utility model, call a provider, or prove cached-token, latency, cost, or task-success gains."
        ),
        "text_artifact_note": (
            "output JSONL contains prompt text; keep it local and do not commit"
            if include_text
            else "output JSONL is prompt-safe metadata only"
        ),
    }


def _load_humaneval_tasks(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for source_row_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                value.setdefault("source_input_path", str(path))
                value.setdefault("source_row_index", source_row_index)
                yield value


def _select_tasks(tasks: Sequence[Mapping[str, Any]], *, max_rows: int | None, seed: int) -> tuple[Mapping[str, Any], ...]:
    ordered = tuple(sorted(tasks, key=lambda row: (str(row.get("id")), str(row.get("source_input_path")))))
    if max_rows is None:
        return ordered
    limit = max(0, int(max_rows))
    if limit == 0:
        return ()
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    return tuple(sorted(shuffled[:limit], key=lambda row: (str(row.get("id")), str(row.get("source_input_path")))))


def _prompt_text(task: Mapping[str, Any]) -> str:
    substitutions = task.get("substitutions")
    if isinstance(substitutions, Mapping):
        prompt_file = substitutions.get("prompt.txt")
        if isinstance(prompt_file, Mapping) and isinstance(prompt_file.get("__PROMPT__"), str):
            return prompt_file["__PROMPT__"].strip()
    return str(task.get("prompt") or task.get("content") or "")


def _entry_point(task: Mapping[str, Any]) -> str | None:
    substitutions = task.get("substitutions")
    if isinstance(substitutions, Mapping):
        scenario = substitutions.get("scenario.py")
        if isinstance(scenario, Mapping) and isinstance(scenario.get("__ENTRY_POINT__"), str):
            return scenario["__ENTRY_POINT__"]
    return None


def _is_hard_constraint_failure(reason: str) -> bool:
    hard_prefixes = {
        "block_id_multiset_changed",
        "forbidden_block_moved",
        "prefix_tree_missing",
        "prefix_tree_block_coverage_mismatch",
        "cacheable_prefix_not_at_front",
        "block_hash_multiset_changed",
        "tools_hash_changed",
        "model_args_hash_changed",
        "forbidden_block_relative_order_changed",
        "message_count_changed",
        "message_type_changed",
        "message_source_changed",
        "message_identity_changed",
        "rewritten_content_mismatch",
    }
    return any(reason == hard or reason.startswith(f"{hard}:") for hard in hard_prefixes)


def _focus_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(focus) for focus in row.get("annotation_focus") or ())
    return dict(sorted(counts.items()))


def _safe_id(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value).strip()).strip("_")
    return cleaned[:180] or "utility-row"


def _safe_hash(value: Any) -> str:
    from .ir import stable_hash

    return stable_hash(value)


def _scope_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    if raw == "agent":
        return "agent_local"
    return str(raw)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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
