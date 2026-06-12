from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class SemanticType(str, Enum):
    GLOBAL_TASK_BACKGROUND = "global_task_background"
    SHARED_CONTEXT = "shared_context"
    TEAM_POLICY = "team_policy"
    SHARED_TOOL_DESCRIPTION = "shared_tool_description"
    ROLE_IDENTITY = "role_identity"
    PRIVATE_MEMORY = "private_memory"
    PRIVATE_TOOL_PERMISSION = "private_tool_permission"
    CONVERSATION_HISTORY = "conversation_history"
    TOOL_RESULT = "tool_result"
    CURRENT_USER_INSTRUCTION = "current_user_instruction"
    CURRENT_TURN_INSTRUCTION = "current_turn_instruction"
    OUTPUT_FORMAT = "output_format"
    UNKNOWN = "unknown"


class Movability(str, Enum):
    SAFE_PREFIX = "safe_prefix"
    CONDITIONAL_PREFIX = "conditional_prefix"
    LOCAL_ONLY = "local_only"
    ORDER_SENSITIVE = "order_sensitive"
    NEVER_MOVE = "never_move"


class ShareScope(str, Enum):
    GLOBAL = "global"
    SUBGROUP = "subgroup"
    AGENT = "agent"
    PRIVATE = "private"
    UNKNOWN = "unknown"


class PrefixScope(str, Enum):
    GLOBAL = "global"
    SUBGROUP = "subgroup"
    AGENT_LOCAL = "agent_local"


@dataclass(frozen=True)
class BlockPosition:
    message_index: int
    part_index: int


@dataclass(frozen=True)
class PromptBlock:
    block_id: str
    source_message_id: str
    source_role: str
    source_type: str
    agent_or_source: str | None
    text_or_payload: Any
    content_hash: str
    semantic_type: SemanticType
    movability: Movability
    share_scope: ShareScope
    risk_tags: tuple[str, ...]
    original_position: BlockPosition
    rendered_text: str | None = None
    is_system_text: bool = False
    has_hard_risk: bool = False
    dependency_refs: tuple[str, ...] = ()
    summary: str | None = None
    token_len: int = 0
    candidate_shared_agents: tuple[str, ...] = ()
    dependency_before: tuple[str, ...] = ()
    dependency_after: tuple[str, ...] = ()
    movable_hint: str | None = None
    contains_private_info: bool = False
    contains_role_identity: bool = False
    contains_tool_permission: bool = False
    contains_latest_user_instruction: bool = False
    contains_tool_result: bool = False
    contains_credential: bool = False

    @property
    def block_hash(self) -> str:
        return self.content_hash

    @property
    def raw_text(self) -> str | None:
        return self.rendered_text

    @property
    def secure_text_ref(self) -> str | None:
        if self.rendered_text is not None:
            return None
        return f"payload:{self.content_hash[:16]}"

    @property
    def original_agent_id(self) -> str | None:
        return self.agent_or_source

    @property
    def original_message_id(self) -> str:
        return self.source_message_id


@dataclass(frozen=True)
class PrefixTreeNode:
    node_id: str
    scope: ShareScope
    label: str
    block_ids: tuple[str, ...] = ()
    children: tuple["PrefixTreeNode", ...] = ()
    scope_type: PrefixScope | str | None = None
    agent_ids: tuple[str, ...] = ()
    parent_id: str | None = None
    children_ids: tuple[str, ...] = ()
    token_len: int = 0
    risk_tags: tuple[str, ...] = ()
    explanation: str | None = None


@dataclass(frozen=True)
class PrefixTree:
    session_id: str
    root: PrefixTreeNode
    leaf_path: tuple[str, ...]


@dataclass(frozen=True)
class BlockPlacement:
    placement_id: str
    block_id: str
    block_hash: str
    original_agent_id: str
    original_position: BlockPosition
    original_scope: ShareScope | str
    target_scope: PrefixScope | str
    target_node_id: str
    target_agent_group: tuple[str, ...]
    moved: bool
    risk_tags: tuple[str, ...] = ()
    dependency_notes: tuple[str, ...] = ()
    cache_contribution: float = 0.0
    placement_score: float = 0.0
    placement_score_breakdown: Mapping[str, float] | None = None


@dataclass(frozen=True)
class PrefixTreeCandidate:
    candidate_id: str
    root_node_id: str
    nodes: Mapping[str, PrefixTreeNode]
    agent_paths: Mapping[str, tuple[str, ...]]
    placements: tuple[BlockPlacement, ...]
    estimated_cache_gain: float = 0.0
    planner_score: float = 0.0
    planner_score_breakdown: Mapping[str, Any] | None = None
    generation_reason: str = ""
    agent_block_orders: Mapping[str, tuple[str, ...]] | None = None
    materialized_prompts: Mapping[str, str] | None = None
    block_hash_by_id: Mapping[str, str] | None = None
    block_text_by_id: Mapping[str, str] | None = None
    cache_gain_report: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CompileResult:
    messages: Sequence[Any]
    blocks: tuple[PromptBlock, ...]
    tools_hash: str
    model_args_hash: str
    session_id: str


def normalize_for_hash(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return normalize_for_hash(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_for_hash(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): normalize_for_hash(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalize_for_hash(item) for item in value]
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if callable(value):
        return {"callable": getattr(value, "__qualname__", repr(value))}
    return value


def stable_json(value: Any) -> str:
    return json.dumps(normalize_for_hash(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def hash_tools(tools: Sequence[Any]) -> str:
    normalized = []
    for tool in tools:
        if hasattr(tool, "schema"):
            normalized.append(getattr(tool, "schema"))
        else:
            normalized.append(tool)
    return stable_hash(normalized)


def hash_model_args(tool_choice: Any, json_output: Any, extra_create_args: Mapping[str, Any] | None) -> str:
    return stable_hash(
        {
            "tool_choice": tool_choice,
            "json_output": json_output,
            "extra_create_args": dict(extra_create_args or {}),
        }
    )
