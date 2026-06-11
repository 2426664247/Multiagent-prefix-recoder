from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .ir import CompileResult, Movability, PrefixTree, PrefixTreeNode, PromptBlock, SemanticType, ShareScope


@dataclass(frozen=True)
class PrefixPlan:
    original_order: tuple[str, ...]
    new_order: tuple[str, ...]
    moved_blocks: tuple[str, ...]
    kept_blocks: tuple[str, ...]
    move_reason: dict[str, str]
    cacheable_prefix_blocks: tuple[str, ...] = ()
    prefix_tree: PrefixTree | None = None
    risk_notes: tuple[str, ...] = ()
    fallback_required: bool = False
    fallback_reason: str | None = None


@dataclass
class _SeenBlock:
    count: int = 0
    agents_or_sources: set[str] = field(default_factory=set)
    semantic_types: set[SemanticType] = field(default_factory=set)


class HierarchicalPrefixPlanner:
    """保守 planner：只把有 exact-hash 复用证据的共享 system block 放进 prefix tree。"""

    _semantic_priority: dict[SemanticType, int] = {
        SemanticType.GLOBAL_TASK_BACKGROUND: 10,
        SemanticType.SHARED_CONTEXT: 20,
        SemanticType.TEAM_POLICY: 30,
        SemanticType.SHARED_TOOL_DESCRIPTION: 40,
        SemanticType.OUTPUT_FORMAT: 50,
    }

    def __init__(self, *, enable_groupchat_history_reordering: bool = False) -> None:
        self.enable_groupchat_history_reordering = enable_groupchat_history_reordering
        self._seen_by_session: dict[str, dict[str, _SeenBlock]] = defaultdict(dict)

    def plan(self, compile_result: CompileResult, *, session_id: str | None = None) -> PrefixPlan:
        effective_session_id = session_id or compile_result.session_id
        blocks = compile_result.blocks
        original_order = tuple(block.block_id for block in blocks)
        if not blocks:
            prefix_tree = self._build_prefix_tree((), (), (), blocks, effective_session_id)
            return PrefixPlan(
                original_order=original_order,
                new_order=original_order,
                moved_blocks=(),
                kept_blocks=original_order,
                move_reason={},
                prefix_tree=prefix_tree,
            )

        system_candidates = self._rank_candidates(
            [block for block in blocks if self._can_promote(block, compile_result, effective_session_id)]
        )
        history_candidates = self._history_prefix_candidates(
            compile_result,
            session_id=effective_session_id,
        )
        candidates = (*system_candidates, *history_candidates)
        global_candidates = tuple(
            block
            for block in (*system_candidates, *history_candidates)
            if block.share_scope == ShareScope.GLOBAL
        )
        subgroup_candidates = tuple(
            block
            for block in (*system_candidates, *history_candidates)
            if block.share_scope == ShareScope.SUBGROUP
        )
        prefix_ids = tuple(block.block_id for block in (*global_candidates, *subgroup_candidates))
        new_order = prefix_ids + tuple(block.block_id for block in blocks if block.block_id not in set(prefix_ids))

        moved_blocks = tuple(
            block.block_id
            for block in candidates
            if original_order.index(block.block_id) != new_order.index(block.block_id)
        )
        move_reason = {
            block.block_id: self._move_reason(block)
            for block in candidates
            if block.block_id in moved_blocks
        }
        risk_notes = tuple(
            f"{block.block_id}: {','.join(block.risk_tags)}"
            for block in blocks
            if block.movability in {Movability.LOCAL_ONLY, Movability.ORDER_SENSITIVE, Movability.NEVER_MOVE}
        )

        prefix_tree = self._build_prefix_tree(global_candidates, subgroup_candidates, new_order, blocks, effective_session_id)
        plan = PrefixPlan(
            original_order=original_order,
            new_order=new_order,
            moved_blocks=moved_blocks,
            kept_blocks=tuple(block_id for block_id in original_order if block_id not in moved_blocks),
            move_reason=move_reason,
            cacheable_prefix_blocks=prefix_ids,
            prefix_tree=prefix_tree,
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

    def _history_prefix_candidates(
        self,
        compile_result: CompileResult,
        *,
        session_id: str,
    ) -> tuple[PromptBlock, ...]:
        if not self.enable_groupchat_history_reordering:
            return ()

        candidates: list[PromptBlock] = []
        history_started = False
        for block in compile_result.blocks:
            if block.semantic_type != SemanticType.CONVERSATION_HISTORY:
                if history_started:
                    break
                continue
            history_started = True
            if not self._can_promote_history(block, session_id=session_id):
                break
            candidates.append(block)
        if not candidates:
            return ()

        initial_task_blocks = [
            block
            for block in compile_result.blocks
            if block.semantic_type == SemanticType.GLOBAL_TASK_BACKGROUND
            and block.source_role == "user"
            and "initial_user_task" in block.risk_tags
            and self._can_promote_initial_user_task(block, session_id=session_id)
        ]
        return tuple((*initial_task_blocks, *candidates))

    def _can_promote_history(self, block: PromptBlock, *, session_id: str) -> bool:
        if block.semantic_type != SemanticType.CONVERSATION_HISTORY:
            return False
        if block.movability != Movability.CONDITIONAL_PREFIX:
            return False
        if block.share_scope != ShareScope.SUBGROUP:
            return False
        if "groupchat_history_order" not in block.risk_tags:
            return False
        seen = self._seen_by_session.get(session_id, {}).get(block.content_hash)
        return seen is not None and seen.count > 0

    def _can_promote_initial_user_task(self, block: PromptBlock, *, session_id: str) -> bool:
        if block.movability != Movability.CONDITIONAL_PREFIX:
            return False
        if block.share_scope != ShareScope.GLOBAL:
            return False
        if "groupchat_dialogue_order" not in block.risk_tags:
            return False
        seen = self._seen_by_session.get(session_id, {}).get(block.content_hash)
        return seen is not None and seen.count > 0

    def _rank_candidates(self, candidates: list[PromptBlock]) -> tuple[PromptBlock, ...]:
        return tuple(
            sorted(
                candidates,
                key=lambda block: (
                    0 if block.share_scope == ShareScope.GLOBAL else 1,
                    self._semantic_priority.get(block.semantic_type, 100),
                    block.original_position.message_index,
                    block.original_position.part_index,
                    block.block_id,
                ),
            )
        )

    def _move_reason(self, block: PromptBlock) -> str:
        return (
            "exact_hash_seen_in_session_and_prefix_tree_scope="
            f"{block.share_scope.value};semantic_type={block.semantic_type.value}"
        )

    def _build_prefix_tree(
        self,
        global_candidates: tuple[PromptBlock, ...],
        subgroup_candidates: tuple[PromptBlock, ...],
        new_order: tuple[str, ...],
        blocks: tuple[PromptBlock, ...],
        session_id: str,
    ) -> PrefixTree:
        global_ids = tuple(block.block_id for block in global_candidates)
        subgroup_ids = tuple(block.block_id for block in subgroup_candidates)
        shared_ids = set(global_ids) | set(subgroup_ids)
        leaf_ids = tuple(block_id for block_id in new_order if block_id not in shared_ids)

        leaf = PrefixTreeNode(
            node_id=f"{session_id}:leaf",
            scope=ShareScope.AGENT,
            label="agent_local_suffix",
            block_ids=leaf_ids,
        )
        if subgroup_ids:
            subgroup = PrefixTreeNode(
                node_id=f"{session_id}:subgroup",
                scope=ShareScope.SUBGROUP,
                label="subgroup_shared_prefix",
                block_ids=subgroup_ids,
                children=(leaf,),
            )
            children = (subgroup,)
            leaf_path = ("root", "subgroup", "leaf")
        else:
            children = (leaf,)
            leaf_path = ("root", "leaf")

        root = PrefixTreeNode(
            node_id=f"{session_id}:root",
            scope=ShareScope.GLOBAL,
            label="global_shared_prefix",
            block_ids=global_ids,
            children=children,
        )
        tree = PrefixTree(session_id=session_id, root=root, leaf_path=leaf_path)

        # 冷启动时没有可移动 block，leaf 仍然承载全部 block，确保整棵树覆盖当前请求。
        if not blocks and not global_ids and not subgroup_ids:
            return tree
        return tree

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
    if len(moved_message_indexes) == 1:
        target_message_index = next(iter(moved_message_indexes))
        target_message = compile_result.messages[target_message_index]
        target_block_ids = {
            block.block_id
            for block in compile_result.blocks
            if block.original_position.message_index == target_message_index
        }
        ordered_target_blocks = [blocks_by_id[block_id] for block_id in plan.new_order if block_id in target_block_ids]
        if all(block.is_system_text for block in ordered_target_blocks):
            if any(block.rendered_text is None for block in ordered_target_blocks):
                raise ValueError("Cannot rewrite non-text system blocks")
            new_content = "".join(block.rendered_text or "" for block in ordered_target_blocks)

            rewritten = list(compile_result.messages)
            if hasattr(target_message, "model_copy"):
                rewritten[target_message_index] = target_message.model_copy(update={"content": new_content})
            else:
                raise TypeError("AutoGen message does not support model_copy")
            return tuple(rewritten)

    return _rewrite_cross_message_order(compile_result, plan, blocks_by_id)


def _rewrite_cross_message_order(
    compile_result: CompileResult,
    plan: PrefixPlan,
    blocks_by_id: dict[str, PromptBlock],
) -> tuple[Any, ...]:
    rewritten_messages: list[Any] = []
    pending_system_message_index: int | None = None
    pending_system_blocks: list[PromptBlock] = []

    def flush_system() -> None:
        nonlocal pending_system_message_index, pending_system_blocks
        if pending_system_message_index is None:
            return
        original_message = compile_result.messages[pending_system_message_index]
        if any(block.rendered_text is None for block in pending_system_blocks):
            raise ValueError("Cannot rewrite non-text system blocks")
        new_content = "".join(block.rendered_text or "" for block in pending_system_blocks)
        if not hasattr(original_message, "model_copy"):
            raise TypeError("AutoGen message does not support model_copy")
        rewritten_messages.append(original_message.model_copy(update={"content": new_content}))
        pending_system_message_index = None
        pending_system_blocks = []

    for block_id in plan.new_order:
        block = blocks_by_id.get(block_id)
        if block is None:
            continue
        message_index = block.original_position.message_index
        if block.is_system_text:
            if pending_system_message_index in {None, message_index}:
                pending_system_message_index = message_index
                pending_system_blocks.append(block)
            else:
                flush_system()
                pending_system_message_index = message_index
                pending_system_blocks.append(block)
            continue
        flush_system()
        rewritten_messages.append(compile_result.messages[message_index])
    flush_system()

    original_message_indexes = {
        block.original_position.message_index
        for block_id in plan.new_order
        for block in [blocks_by_id.get(block_id)]
        if block is not None
    }
    if original_message_indexes != set(range(len(compile_result.messages))):
        raise ValueError("Cross-message rewrite dropped a message")
    return tuple(rewritten_messages)
