from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .ir import PrefixTree, PrefixTreeCandidate, PrefixTreeNode

TelemetrySink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class JsonlTelemetryLogger:
    path: str | Path

    def __call__(self, record: Mapping[str, Any]) -> None:
        target = Path(self.path)
        if target.parent != Path("."):
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True)
class TelemetrySummary:
    request_count: int
    applied_count: int
    fallback_count: int
    no_rewrite_count: int
    moved_block_count: int
    total_original_prefix_chars: int
    total_rewritten_prefix_chars: int
    total_estimated_gain_chars: int
    total_moved_block_chars: int
    reusable_prefix_request_count: int
    repeated_prefix_request_count: int
    unique_reusable_prefix_count: int
    validation_reason_counts: dict[str, int]

    @property
    def fallback_rate(self) -> float:
        return _safe_ratio(self.fallback_count, self.request_count)

    @property
    def applied_rate(self) -> float:
        return _safe_ratio(self.applied_count, self.request_count)

    @property
    def repeated_prefix_rate(self) -> float:
        return _safe_ratio(self.repeated_prefix_request_count, self.reusable_prefix_request_count)

    @property
    def average_estimated_gain_chars(self) -> float:
        return _safe_ratio(self.total_estimated_gain_chars, self.request_count)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fallback_rate"] = self.fallback_rate
        data["applied_rate"] = self.applied_rate
        data["repeated_prefix_rate"] = self.repeated_prefix_rate
        data["average_estimated_gain_chars"] = self.average_estimated_gain_chars
        return data


def load_jsonl_telemetry(path: str | Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return tuple(records)


def summarize_telemetry(records: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> TelemetrySummary:
    validation_reason_counts: dict[str, int] = {}
    seen_prefix_signatures: set[tuple[str, ...]] = set()
    unique_prefix_signatures: set[tuple[str, ...]] = set()
    applied_count = 0
    fallback_count = 0
    no_rewrite_count = 0
    moved_block_count = 0
    total_original_prefix_chars = 0
    total_rewritten_prefix_chars = 0
    total_estimated_gain_chars = 0
    total_moved_block_chars = 0
    reusable_prefix_request_count = 0
    repeated_prefix_request_count = 0

    for record in records:
        validation = record.get("validation") or {}
        reason = str(validation.get("reason") or "unknown")
        validation_reason_counts[reason] = validation_reason_counts.get(reason, 0) + 1
        if validation.get("applied"):
            applied_count += 1
        if validation.get("fallback"):
            fallback_count += 1
        if reason == "no_rewrite_needed":
            no_rewrite_count += 1

        moved_block_count += len(tuple(record.get("blocks_moved") or ()))

        utility = record.get("utility_estimate") or {}
        total_original_prefix_chars += int(utility.get("original_prefix_chars") or 0)
        total_rewritten_prefix_chars += int(utility.get("rewritten_prefix_chars") or 0)
        total_estimated_gain_chars += int(utility.get("estimated_gain_chars") or 0)
        total_moved_block_chars += int(utility.get("moved_block_chars") or 0)

        signature = tuple(str(block_id) for block_id in (record.get("cacheable_prefix_blocks") or ()))
        if signature:
            reusable_prefix_request_count += 1
            if signature in seen_prefix_signatures:
                repeated_prefix_request_count += 1
            seen_prefix_signatures.add(signature)
            unique_prefix_signatures.add(signature)

    return TelemetrySummary(
        request_count=len(records),
        applied_count=applied_count,
        fallback_count=fallback_count,
        no_rewrite_count=no_rewrite_count,
        moved_block_count=moved_block_count,
        total_original_prefix_chars=total_original_prefix_chars,
        total_rewritten_prefix_chars=total_rewritten_prefix_chars,
        total_estimated_gain_chars=total_estimated_gain_chars,
        total_moved_block_chars=total_moved_block_chars,
        reusable_prefix_request_count=reusable_prefix_request_count,
        repeated_prefix_request_count=repeated_prefix_request_count,
        unique_reusable_prefix_count=len(unique_prefix_signatures),
        validation_reason_counts=validation_reason_counts,
    )


def serialize_prefix_tree(tree: PrefixTree | None) -> dict[str, Any] | None:
    if tree is None:
        return None
    return {
        "session_id": tree.session_id,
        "leaf_path": tree.leaf_path,
        "root": _serialize_prefix_tree_node(tree.root),
    }


def serialize_prefix_tree_candidate(candidate: PrefixTreeCandidate | None, *, include_text: bool = False) -> dict[str, Any] | None:
    if candidate is None:
        return None
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "root_node_id": candidate.root_node_id,
        "nodes": {
            node_id: _serialize_prefix_tree_node(node)
            for node_id, node in candidate.nodes.items()
        },
        "agent_paths": dict(candidate.agent_paths),
        "placements": tuple(
            {
                "placement_id": placement.placement_id,
                "block_id": placement.block_id,
                "block_hash": placement.block_hash,
                "original_agent_id": placement.original_agent_id,
                "original_position": dataclass_to_dict(placement.original_position),
                "original_scope": _enum_value(placement.original_scope),
                "target_scope": _enum_value(placement.target_scope),
                "target_node_id": placement.target_node_id,
                "target_agent_group": placement.target_agent_group,
                "moved": placement.moved,
                "risk_tags": placement.risk_tags,
                "dependency_notes": placement.dependency_notes,
                "cache_contribution": placement.cache_contribution,
                "placement_score": placement.placement_score,
                "placement_score_breakdown": placement.placement_score_breakdown,
            }
            for placement in candidate.placements
        ),
        "estimated_cache_gain": candidate.estimated_cache_gain,
        "planner_score": candidate.planner_score,
        "planner_score_breakdown": candidate.planner_score_breakdown,
        "generation_reason": candidate.generation_reason,
        "agent_block_orders": dict(candidate.agent_block_orders or {}),
        "block_hash_by_id": dict(candidate.block_hash_by_id or {}),
        "cache_gain_report": candidate.cache_gain_report,
    }
    if include_text:
        row["materialized_prompts"] = dict(candidate.materialized_prompts or {})
        row["block_text_by_id"] = dict(candidate.block_text_by_id or {})
    return row


def dataclass_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return asdict(value)


def _serialize_prefix_tree_node(node: PrefixTreeNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "scope": node.scope.value,
        "scope_type": _enum_value(node.scope_type),
        "label": node.label,
        "agent_ids": node.agent_ids,
        "block_ids": node.block_ids,
        "parent_id": node.parent_id,
        "children_ids": node.children_ids,
        "token_len": node.token_len,
        "risk_tags": node.risk_tags,
        "explanation": node.explanation,
        "children": tuple(_serialize_prefix_tree_node(child) for child in node.children),
    }


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
