from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .ir import PrefixTree, PrefixTreeNode

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


def dataclass_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return asdict(value)


def _serialize_prefix_tree_node(node: PrefixTreeNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "scope": node.scope.value,
        "label": node.label,
        "block_ids": node.block_ids,
        "children": tuple(_serialize_prefix_tree_node(child) for child in node.children),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
