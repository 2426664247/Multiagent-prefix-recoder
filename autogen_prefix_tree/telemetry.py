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
