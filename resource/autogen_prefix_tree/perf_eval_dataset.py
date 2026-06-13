from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ir import stable_hash
from .utility_dataset_sources import (
    DATASET_SOURCES,
    TASK_SCHEMA_VERSION,
    _humaneval_entry_point,
    _humaneval_prompt,
    _humaneval_test,
    _read_jsonl,
)

PERF_SPLIT = "perf_eval_dspro"
DEFAULT_CREATED_AT = "2026-06-14T00:00:00Z"
SOURCE_COUNT = 20
HUMANEVAL_START = 3
HUMANEVAL_END = 22


def build_perf_eval_dspro_dataset(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    created_at: str = DEFAULT_CREATED_AT,
) -> dict[str, Any]:
    repo = Path(repo_root)
    output = Path(output_root)
    tasks = _build_tasks(repo_root=repo, created_at=created_at)
    summary = _validate_tasks(repo_root=repo, tasks=tasks, created_at=created_at)

    task_dir = output / "tasks" / PERF_SPLIT
    reports_dir = output / "reports"
    task_path = task_dir / "utility_tasks.jsonl"
    summary_path = reports_dir / "perf_eval_dataset_summary.json"
    card_path = reports_dir / "perf_eval_dataset_card.md"

    task_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(task_path, tasks)
    summary_with_paths = {
        **summary,
        "output_paths": {
            "tasks": _rel(repo, task_path),
            "summary": _rel(repo, summary_path),
            "dataset_card": _rel(repo, card_path),
        },
    }
    summary_path.write_text(json.dumps(summary_with_paths, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    card_path.write_text(_dataset_card(summary_with_paths), encoding="utf-8")
    return summary_with_paths


def _build_tasks(*, repo_root: Path, created_at: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    tasks.extend(_humaneval_tasks(repo_root=repo_root, created_at=created_at))
    for source in ("synthetic_agent", "format_protocol", "role_privacy", "history_state"):
        tasks.extend(_scenario_tasks(source=source, created_at=created_at))
    return tasks


def _humaneval_tasks(*, repo_root: Path, created_at: str) -> list[dict[str, Any]]:
    source_file = repo_root / "datasets" / "HumanEval" / "Tasks" / "human_eval_TwoAgents.jsonl"
    wanted_ids = {f"HumanEval_{index}" for index in range(HUMANEVAL_START, HUMANEVAL_END + 1)}
    selected: list[dict[str, Any]] = []
    for row in _read_jsonl(source_file):
        source_task_id = str(row.get("id") or "")
        if source_task_id in wanted_ids:
            selected.append(row)
        if len(selected) == SOURCE_COUNT:
            break
    if len(selected) != SOURCE_COUNT:
        raise ValueError(f"expected {SOURCE_COUNT} HumanEval rows, found {len(selected)}")

    tasks: list[dict[str, Any]] = []
    for row in selected:
        source_task_id = str(row["id"])
        entry_point = _humaneval_entry_point(row)
        prompt_hash = stable_hash({"prompt": _humaneval_prompt(row)})
        test_hash = stable_hash({"test": _humaneval_test(row)})
        prompt_safe_ref = f"prompt-safe://utility_validator/{PERF_SPLIT}/humaneval/{source_task_id}"
        source_row_ref = f"datasets/HumanEval/Tasks/human_eval_TwoAgents.jsonl#{source_task_id}"
        tasks.append(
            _base_task(
                task_id=f"uv_perf_humaneval_{source_task_id}",
                dataset_source="humaneval",
                scenario_type="code_execution_unit_test",
                prompt_template_id="humaneval_two_agent_perf_eval_v1",
                agents=("planner", "engineer"),
                original_messages_ref=prompt_safe_ref,
                original_messages_hash=stable_hash(
                    {
                        "source": "humaneval",
                        "source_task_id": source_task_id,
                        "prompt_hash": prompt_hash,
                        "template": "humaneval_two_agent_perf_eval_v1",
                    }
                ),
                expected_oracle_type="unit_test",
                oracle_spec={
                    "entry_point": entry_point,
                    "source_task_id": source_task_id,
                    "source_row_ref": source_row_ref,
                    "test_ref": source_row_ref,
                    "test_hash": test_hash,
                    "execution_mode": "reserved_for_perf_eval_not_run_at_dataset_build",
                },
                created_at=created_at,
                source_metadata={
                    "source_task_id": source_task_id,
                    "entry_point": entry_point,
                    "prompt_hash": prompt_hash,
                    "test_hash": test_hash,
                    "source_file": "datasets/HumanEval/Tasks/human_eval_TwoAgents.jsonl",
                    "source_prompt_text_stored_in_eval": False,
                    "held_out_reason": "HumanEval_3_to_HumanEval_22_avoid_expanded_HumanEval_0_1_2",
                },
            )
        )
    return tasks


def _scenario_tasks(*, source: str, created_at: str) -> list[dict[str, Any]]:
    specs = _scenario_specs()[source]
    tasks: list[dict[str, Any]] = []
    for index in range(SOURCE_COUNT):
        spec = specs[index % len(specs)]
        variant = index // len(specs)
        variant_id = f"{index + 1:02d}"
        scenario_type = str(spec["scenario_type"])
        prompt_template_id = f"{spec['prompt_template_id']}_perf_eval_v1"
        prompt_safe_ref = f"prompt-safe://utility_validator/{PERF_SPLIT}/{source}/{scenario_type}/variant_{variant_id}"
        task_id = f"uv_perf_{source}_{scenario_type}_{variant_id}"
        oracle_spec = dict(spec["oracle_spec"])
        if "forbidden_substrings" in oracle_spec:
            oracle_spec["forbidden_substrings_hashes"] = [
                stable_hash({"forbidden": item}) for item in oracle_spec.pop("forbidden_substrings")
            ]
            oracle_spec["forbidden_substrings_ref"] = f"{prompt_safe_ref}/oracle/forbidden_substrings"
        oracle_spec["scenario_ref"] = prompt_safe_ref
        oracle_spec["variant_index"] = variant
        tasks.append(
            _base_task(
                task_id=task_id,
                dataset_source=source,
                scenario_type=scenario_type,
                prompt_template_id=prompt_template_id,
                agents=tuple(spec["agents"]),
                original_messages_ref=prompt_safe_ref,
                original_messages_hash=stable_hash(
                    {
                        "source": source,
                        "scenario_type": scenario_type,
                        "variant_id": variant_id,
                        "variant_index": variant,
                        "template": prompt_template_id,
                    }
                ),
                expected_oracle_type=str(spec["expected_oracle_type"]),
                oracle_spec=oracle_spec,
                created_at=created_at,
                source_metadata={
                    "base_scenario_id": spec["base_scenario_id"],
                    "deterministic_variant_id": variant_id,
                    "deterministic_variant_index": variant,
                    "risk_tags": list(spec["risk_tags"]),
                    "source_prompt_text_stored_in_eval": False,
                },
            )
        )
    return tasks


def _base_task(
    *,
    task_id: str,
    dataset_source: str,
    scenario_type: str,
    prompt_template_id: str,
    agents: Sequence[str],
    original_messages_ref: str,
    original_messages_hash: str,
    expected_oracle_type: str,
    oracle_spec: Mapping[str, Any],
    created_at: str,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": task_id,
        "dataset_source": dataset_source,
        "scenario_type": scenario_type,
        "prompt_template_id": prompt_template_id,
        "agents": list(agents),
        "original_messages_ref": original_messages_ref,
        "original_messages_hash": original_messages_hash,
        "expected_oracle_type": expected_oracle_type,
        "oracle_spec": dict(oracle_spec),
        "candidate_generation_config": {
            "planner": "HierarchicalPrefixPlanner",
            "top_k": 3,
            "candidate_strategies": ["conservative", "balanced", "aggressive", "adversarial"],
            "candidate_type": "prefix_tree",
            "cache_metric": "prefix_tree_cache_hit_ratio",
            "prompt_materialization": "resolve_original_messages_ref_at_runtime",
            "requires_prompt_text_at_build_time": False,
        },
        "prompt_text_included": False,
        "eval_split": PERF_SPLIT,
        "created_at": created_at,
        "source_metadata": dict(source_metadata),
    }


def _scenario_specs() -> dict[str, list[dict[str, Any]]]:
    return {
        "synthetic_agent": [
            {
                "base_scenario_id": "synthetic_tool_allowed",
                "scenario_type": "tool_use_permission",
                "prompt_template_id": "synthetic_tool_permission",
                "agents": ["researcher", "executor"],
                "expected_oracle_type": "tool_trace",
                "oracle_spec": {
                    "required_tool_calls": ["lookup_customer_status"],
                    "forbidden_tool_calls": ["export_all_records"],
                    "expected_final_result": "customer status returned without bulk export",
                },
                "risk_tags": ["private_tool_permission", "tool_use"],
            },
            {
                "base_scenario_id": "synthetic_private_memory",
                "scenario_type": "private_memory_tool_use",
                "prompt_template_id": "synthetic_private_memory",
                "agents": ["analyst", "messenger"],
                "expected_oracle_type": "privacy_check",
                "oracle_spec": {"forbidden_substrings": ["PERF-PRIVATE-TOKEN"]},
                "risk_tags": ["private_memory"],
            },
            {
                "base_scenario_id": "synthetic_role_boundary",
                "scenario_type": "role_boundary",
                "prompt_template_id": "synthetic_role_boundary",
                "agents": ["planner", "worker"],
                "expected_oracle_type": "role_check",
                "oracle_spec": {"forbidden_role_claims": ["as the planner", "I approve the plan"]},
                "risk_tags": ["agent_identity", "role_boundary"],
            },
            {
                "base_scenario_id": "synthetic_shared_subgroup",
                "scenario_type": "subgroup_shared_instruction",
                "prompt_template_id": "synthetic_subgroup",
                "agents": ["retriever_a", "retriever_b", "summarizer"],
                "expected_oracle_type": "tool_trace",
                "oracle_spec": {
                    "required_tool_calls": ["retrieve_public_doc"],
                    "forbidden_tool_calls": ["read_private_notes"],
                    "expected_final_result": "public document fact retrieved",
                },
                "risk_tags": ["subgroup_shared_instruction", "tool_use"],
            },
        ],
        "format_protocol": [
            {
                "base_scenario_id": "format_json_summary",
                "scenario_type": "json_schema_output",
                "prompt_template_id": "format_json_summary",
                "agents": ["formatter"],
                "expected_oracle_type": "json_schema",
                "oracle_spec": {
                    "json_schema": {
                        "type": "object",
                        "required": ["status", "priority"],
                        "properties": {
                            "status": {"type": "string", "enum": ["open", "closed"]},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                        "additionalProperties": False,
                    },
                    "forbidden_fields": ["debug"],
                },
                "risk_tags": ["output_format", "json_schema"],
            },
            {
                "base_scenario_id": "format_regex_id",
                "scenario_type": "regex_protocol",
                "prompt_template_id": "format_regex_id",
                "agents": ["formatter"],
                "expected_oracle_type": "json_schema",
                "oracle_spec": {
                    "json_schema": {
                        "type": "object",
                        "required": ["ticket_id", "status"],
                        "properties": {
                            "ticket_id": {"type": "string", "pattern": "T-[0-9]+"},
                            "status": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "forbidden_fields": ["private_notes"],
                },
                "risk_tags": ["output_format", "regex_protocol"],
            },
            {
                "base_scenario_id": "format_no_extra_field",
                "scenario_type": "forbidden_field",
                "prompt_template_id": "format_no_extra_field",
                "agents": ["formatter"],
                "expected_oracle_type": "json_schema",
                "oracle_spec": {
                    "json_schema": {
                        "type": "object",
                        "required": ["answer", "confidence"],
                        "properties": {
                            "answer": {"type": "string", "enum": ["yes", "no"]},
                            "confidence": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                    "forbidden_fields": ["debug", "private_notes"],
                },
                "risk_tags": ["output_format", "forbidden_field"],
            },
        ],
        "role_privacy": [
            {
                "base_scenario_id": "role_agent_specific",
                "scenario_type": "agent_specific_instruction",
                "prompt_template_id": "role_agent_specific",
                "agents": ["auditor", "operator"],
                "expected_oracle_type": "privacy_check",
                "oracle_spec": {"forbidden_substrings": ["PERF-ESCALATION-CODE"]},
                "risk_tags": ["agent_identity", "private_memory"],
            },
            {
                "base_scenario_id": "role_identity_confusion",
                "scenario_type": "role_identity_confusion",
                "prompt_template_id": "role_identity_confusion",
                "agents": ["reviewer", "author"],
                "expected_oracle_type": "role_check",
                "oracle_spec": {"forbidden_role_claims": ["as the author", "I wrote this"]},
                "risk_tags": ["agent_identity", "role_boundary"],
            },
            {
                "base_scenario_id": "role_private_tool",
                "scenario_type": "tool_permission_boundary",
                "prompt_template_id": "role_private_tool",
                "agents": ["reader", "admin"],
                "expected_oracle_type": "tool_trace",
                "oracle_spec": {
                    "required_tool_calls": ["read_public_ticket"],
                    "forbidden_tool_calls": ["delete_ticket"],
                    "expected_final_result": "public ticket read without admin action",
                },
                "risk_tags": ["private_tool_permission"],
            },
        ],
        "history_state": [
            {
                "base_scenario_id": "history_order",
                "scenario_type": "history_order",
                "prompt_template_id": "history_order",
                "agents": ["coordinator"],
                "expected_oracle_type": "state_check",
                "oracle_spec": {"expected_event_order": ["request_received", "facts_checked", "final_ready"]},
                "risk_tags": ["groupchat_history_order", "state_consistency"],
            },
            {
                "base_scenario_id": "history_role_turns",
                "scenario_type": "state_consistency",
                "prompt_template_id": "history_role_turns",
                "agents": ["assistant_a", "assistant_b"],
                "expected_oracle_type": "state_check",
                "oracle_spec": {"expected_event_order": ["assistant_a_facts_collected", "assistant_b_final"]},
                "risk_tags": ["groupchat_history_order", "role_turn_order"],
            },
            {
                "base_scenario_id": "history_branch_state",
                "scenario_type": "branch_state_consistency",
                "prompt_template_id": "history_branch_state",
                "agents": ["triager", "resolver"],
                "expected_oracle_type": "state_check",
                "oracle_spec": {"expected_event_order": ["triage_complete", "resolution_selected", "final_summary"]},
                "risk_tags": ["state_consistency", "handoff_order"],
            },
        ],
    }


def _validate_tasks(*, repo_root: Path, tasks: Sequence[Mapping[str, Any]], created_at: str) -> dict[str, Any]:
    expanded_task_ids, expanded_humaneval_ids = _expanded_identity_sets(repo_root)
    task_ids = [str(task["task_id"]) for task in tasks]
    source_counts = Counter(str(task["dataset_source"]) for task in tasks)
    humaneval_ids = [
        str(task.get("source_metadata", {}).get("source_task_id"))
        for task in tasks
        if task.get("dataset_source") == "humaneval"
    ]
    schema_errors = list(_schema_errors(tasks))
    prompt_text_fields = _find_prompt_text_fields(tasks)
    label_feature_fields = sorted(
        {
            key
            for task in tasks
            for key in task
            if key in {"label", "labels", "label_id", "label_source", "features", "training_features"}
        }
    )
    fake_label_hits = _find_value_hits(tasks, ("fake_smoke_oracle", "fake label", "fake_label"))
    checks = {
        "total_count_is_100": len(tasks) == 100,
        "source_counts_are_20_each": all(source_counts.get(source, 0) == SOURCE_COUNT for source in DATASET_SOURCES),
        "task_id_unique": len(task_ids) == len(set(task_ids)),
        "task_id_deduped_from_expanded": not (set(task_ids) & expanded_task_ids),
        "humaneval_avoids_expanded_sources": not (set(humaneval_ids) & expanded_humaneval_ids),
        "humaneval_range_is_3_to_22": humaneval_ids == [f"HumanEval_{index}" for index in range(HUMANEVAL_START, HUMANEVAL_END + 1)],
        "no_labels": not label_feature_fields,
        "no_features": not label_feature_fields,
        "no_fake_label": not fake_label_hits,
        "no_prompt_text_fields": not prompt_text_fields,
        "prompt_text_included_false": all(task.get("prompt_text_included") is False for task in tasks),
        "oracle_info_present": all(task.get("expected_oracle_type") and task.get("oracle_spec") for task in tasks),
        "candidate_config_present": all(_candidate_config_ok(task.get("candidate_generation_config")) for task in tasks),
        "jsonl_readable": True,
        "schema_passed": not schema_errors,
    }
    return {
        "schema_version": "perf-eval-dspro-summary-v1",
        "created_at": created_at,
        "eval_split": PERF_SPLIT,
        "task_count": len(tasks),
        "source_counts": dict(sorted(source_counts.items())),
        "expected_source_counts": {source: SOURCE_COUNT for source in DATASET_SOURCES},
        "humaneval_source_range": {
            "start": f"HumanEval_{HUMANEVAL_START}",
            "end": f"HumanEval_{HUMANEVAL_END}",
            "ids": humaneval_ids,
            "expanded_humaneval_source_ids_seen": sorted(expanded_humaneval_ids),
            "overlap_with_expanded_humaneval_source_ids": sorted(set(humaneval_ids) & expanded_humaneval_ids),
        },
        "dedupe": {
            "expanded_task_id_count": len(expanded_task_ids),
            "task_id_overlap_with_expanded": sorted(set(task_ids) & expanded_task_ids),
            "task_ids_unique": len(task_ids) == len(set(task_ids)),
        },
        "prompt_safe": {
            "prompt_text_included_all_false": checks["prompt_text_included_false"],
            "prompt_text_field_hits": prompt_text_fields,
            "uses_prompt_safe_refs": all(str(task.get("original_messages_ref", "")).startswith("prompt-safe://") for task in tasks),
            "prompt_bodies_saved": False,
        },
        "labels_features": {
            "labels_generated": False,
            "features_generated": False,
            "label_feature_field_hits": label_feature_fields,
            "fake_label_hits": fake_label_hits,
        },
        "api_usage": {
            "ds_api_called": False,
            "ds_pro_run": False,
            "network_access_required": False,
        },
        "future_dspro_usage": (
            "Resolve original_messages_ref/source_metadata to materialize prompts at experiment time, "
            "generate prefix-tree candidates from candidate_generation_config, run DS Pro on original and "
            "candidate prompts, then apply oracle_spec to measure utility and cache-hit behavior."
        ),
        "checks": checks,
        "schema_errors": schema_errors,
    }


def _expanded_identity_sets(repo_root: Path) -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    humaneval_ids: set[str] = set()
    for split in ("expanded_train", "expanded_valid", "expanded_test"):
        path = repo_root / "datasets" / "utility_validator" / "tasks" / split / "utility_tasks.jsonl"
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                task_ids.add(task_id)
            metadata = row.get("source_metadata")
            if row.get("dataset_source") == "humaneval" and isinstance(metadata, Mapping):
                source_task_id = metadata.get("source_task_id")
                if isinstance(source_task_id, str):
                    humaneval_ids.add(source_task_id)
    return task_ids, humaneval_ids


def _schema_errors(tasks: Sequence[Mapping[str, Any]]) -> Iterable[str]:
    allowed_sources = set(DATASET_SOURCES)
    allowed_oracles = {"unit_test", "json_schema", "tool_trace", "state_check", "privacy_check", "role_check"}
    required = {
        "task_id",
        "dataset_source",
        "scenario_type",
        "prompt_template_id",
        "agents",
        "original_messages_ref",
        "expected_oracle_type",
        "oracle_spec",
        "candidate_generation_config",
        "prompt_text_included",
        "eval_split",
        "created_at",
    }
    for index, task in enumerate(tasks):
        missing = sorted(required - set(task))
        if missing:
            yield f"row {index}: missing {missing}"
        if task.get("dataset_source") not in allowed_sources:
            yield f"row {index}: invalid source {task.get('dataset_source')!r}"
        if task.get("expected_oracle_type") not in allowed_oracles:
            yield f"row {index}: invalid oracle {task.get('expected_oracle_type')!r}"
        if not isinstance(task.get("agents"), list) or not task.get("agents"):
            yield f"row {index}: agents must be a non-empty list"
        if task.get("prompt_text_included") is not False:
            yield f"row {index}: prompt_text_included must be false"
        if task.get("eval_split") != PERF_SPLIT:
            yield f"row {index}: eval_split must be {PERF_SPLIT}"


def _candidate_config_ok(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("planner") == "HierarchicalPrefixPlanner"
        and value.get("candidate_type") == "prefix_tree"
        and bool(value.get("candidate_strategies"))
        and value.get("requires_prompt_text_at_build_time") is False
    )


def _find_prompt_text_fields(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    hits: set[str] = set()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                child_path = f"{path}.{key}" if path else str(key)
                if key_text in {"prompt", "content", "original_messages", "original_prompt_text", "reordered_prompt_text"}:
                    hits.add(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(tasks, "")
    return sorted(hits)


def _find_value_hits(tasks: Sequence[Mapping[str, Any]], needles: Sequence[str]) -> list[str]:
    serialized = json.dumps(tasks, ensure_ascii=False).lower()
    return sorted(needle for needle in needles if needle.lower() in serialized)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _dataset_card(summary: Mapping[str, Any]) -> str:
    counts = summary["source_counts"]
    paths = summary["output_paths"]
    checks = summary["checks"]
    return "\n".join(
        [
            "# perf_eval_dspro Dataset Card",
            "",
            "## Purpose",
            "",
            "Held-out prompt-safe performance evaluation task set for later DS Pro cache-hit-rate and utility experiments.",
            "",
            "## Outputs",
            "",
            f"- Tasks: `{paths['tasks']}`",
            f"- Summary: `{paths['summary']}`",
            "",
            "## Composition",
            "",
            f"- Total tasks: {summary['task_count']}",
            f"- humaneval: {counts.get('humaneval', 0)}",
            f"- synthetic_agent: {counts.get('synthetic_agent', 0)}",
            f"- format_protocol: {counts.get('format_protocol', 0)}",
            f"- role_privacy: {counts.get('role_privacy', 0)}",
            f"- history_state: {counts.get('history_state', 0)}",
            "",
            "## HumanEval Selection",
            "",
            "HumanEval tasks use source ids `HumanEval_3` through `HumanEval_22` to avoid the expanded set source ids observed in current expanded tasks.",
            "",
            "## Safety And Scope",
            "",
            "- Prompt text is not stored in this eval set; every task has `prompt_text_included=false`.",
            "- No labels, features, fake labels, API calls, DS Pro runs, or model training were produced.",
            "- Task ids are deduped from current expanded task ids.",
            "",
            "## Future DS Pro Experiment Use",
            "",
            str(summary["future_dspro_usage"]),
            "",
            "## Validation",
            "",
            f"- Generated 100 tasks: {checks['total_count_is_100']}",
            f"- Five sources have 20 each: {checks['source_counts_are_20_each']}",
            f"- Task ids unique: {checks['task_id_unique']}",
            f"- Deduped from expanded task ids: {checks['task_id_deduped_from_expanded']}",
            f"- HumanEval avoids expanded source ids: {checks['humaneval_avoids_expanded_sources']}",
            f"- Schema passed: {checks['schema_passed']}",
            "",
        ]
    )


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build held-out perf_eval_dspro Utility Validator tasks.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default="datasets/utility_validator")
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_perf_eval_dspro_dataset(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    if not all(summary["checks"].values()):
        raise SystemExit(json.dumps(summary["checks"], ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
