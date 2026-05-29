from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .ir import CompileResult, Movability, PromptBlock, SemanticType, ShareScope


@dataclass(frozen=True)
class PrefixPlan:
    original_order: tuple[str, ...]
    new_order: tuple[str, ...]
    moved_blocks: tuple[str, ...]
    kept_blocks: tuple[str, ...]
    move_reason: dict[str, str]
    risk_notes: tuple[str, ...] = ()
    fallback_required: bool = False
    fallback_reason: str | None = None


@dataclass
class _SeenBlock:
    count: int = 0
    agents_or_sources: set[str] = field(default_factory=set)
    semantic_types: set[SemanticType] = field(default_factory=set)


class HierarchicalPrefixPlanner:
    """保守 planner：第一版只把 exact hash 已观察到的共享 system block 提到前缀。"""

    def __init__(self) -> None:
        self._seen_by_session: dict[str, dict[str, _SeenBlock]] = defaultdict(dict)

    def plan(self, compile_result: CompileResult, *, session_id: str | None = None) -> PrefixPlan:
        effective_session_id = session_id or compile_result.session_id
        blocks = compile_result.blocks
        original_order = tuple(block.block_id for block in blocks)
        if not blocks:
            return PrefixPlan(
                original_order=original_order,
                new_order=original_order,
                moved_blocks=(),
                kept_blocks=original_order,
                move_reason={},
            )

        candidates = [block for block in blocks if self._can_promote(block, compile_result, effective_session_id)]
        prefix_ids = tuple(block.block_id for block in candidates)
        new_order = prefix_ids + tuple(block.block_id for block in blocks if block.block_id not in set(prefix_ids))

        moved_blocks = tuple(
            block.block_id
            for block in candidates
            if original_order.index(block.block_id) != new_order.index(block.block_id)
        )
        move_reason = {
            block.block_id: "exact_hash_seen_in_session_and_marked_shared_prefix"
            for block in candidates
            if block.block_id in moved_blocks
        }
        risk_notes = tuple(
            f"{block.block_id}: {','.join(block.risk_tags)}"
            for block in blocks
            if block.movability in {Movability.LOCAL_ONLY, Movability.ORDER_SENSITIVE, Movability.NEVER_MOVE}
        )

        plan = PrefixPlan(
            original_order=original_order,
            new_order=new_order,
            moved_blocks=moved_blocks,
            kept_blocks=tuple(block_id for block_id in original_order if block_id not in moved_blocks),
            move_reason=move_reason,
            risk_notes=risk_notes,
        )
        self.observe(compile_result, session_id=effective_session_id)
        return plan

    def observe(self, compile_result: CompileResult, *, session_id: str | None = None) -> None:
        effective_session_id = session_id or compile_result.session_id
        session_seen = self._seen_by_session[effective_session_id]
        for block in compile_result.blocks:
            if block.share_scope not in {ShareScope.GLOBAL, ShareScope.SUBGROUP}:
                continue
            if block.movability not in {Movability.SAFE_PREFIX, Movability.CONDITIONAL_PREFIX}:
                continue
            seen = session_seen.setdefault(block.content_hash, _SeenBlock())
            seen.count += 1
            seen.semantic_types.add(block.semantic_type)
            seen.agents_or_sources.add(block.agent_or_source or block.source_message_id)

    def _can_promote(self, block: PromptBlock, compile_result: CompileResult, session_id: str) -> bool:
        if not block.is_system_text or block.source_role != "system":
            return False
        if block.original_position.message_index != self._first_system_message_index(compile_result):
            return False
        if block.movability not in {Movability.SAFE_PREFIX, Movability.CONDITIONAL_PREFIX}:
            return False
        if block.share_scope not in {ShareScope.GLOBAL, ShareScope.SUBGROUP}:
            return False
        if self._crosses_current_instruction(block, compile_result):
            return False

        seen = self._seen_by_session.get(session_id, {}).get(block.content_hash)
        return seen is not None and seen.count > 0

    def _first_system_message_index(self, compile_result: CompileResult) -> int | None:
        for block in compile_result.blocks:
            if block.source_role == "system" and block.is_system_text:
                return block.original_position.message_index
        return None

    def _crosses_current_instruction(self, block: PromptBlock, compile_result: CompileResult) -> bool:
        current_positions = [
            other.original_position
            for other in compile_result.blocks
            if other.semantic_type in {SemanticType.CURRENT_USER_INSTRUCTION, SemanticType.CURRENT_TURN_INSTRUCTION}
        ]
        if not current_positions:
            return False
        first_current = min((pos.message_index, pos.part_index) for pos in current_positions)
        return (block.original_position.message_index, block.original_position.part_index) > first_current


def rewrite_messages(compile_result: CompileResult, plan: PrefixPlan) -> tuple[Any, ...]:
    if not plan.moved_blocks:
        return tuple(compile_result.messages)

    blocks_by_id = {block.block_id: block for block in compile_result.blocks}
    moved_message_indexes = {
        blocks_by_id[block_id].original_position.message_index
        for block_id in plan.moved_blocks
        if block_id in blocks_by_id
    }
    if len(moved_message_indexes) != 1:
        raise ValueError("MVP only rewrites blocks inside one system message")

    target_message_index = next(iter(moved_message_indexes))
    target_message = compile_result.messages[target_message_index]
    target_block_ids = {
        block.block_id
        for block in compile_result.blocks
        if block.original_position.message_index == target_message_index
    }
    ordered_target_blocks = [blocks_by_id[block_id] for block_id in plan.new_order if block_id in target_block_ids]
    if any(block.rendered_text is None for block in ordered_target_blocks):
        raise ValueError("Cannot rewrite non-text system blocks")
    new_content = "".join(block.rendered_text or "" for block in ordered_target_blocks)

    rewritten = list(compile_result.messages)
    if hasattr(target_message, "model_copy"):
        rewritten[target_message_index] = target_message.model_copy(update={"content": new_content})
    else:
        raise TypeError("AutoGen message does not support model_copy")
    return tuple(rewritten)
