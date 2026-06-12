from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .ir import (
    BlockPlacement,
    CompileResult,
    Movability,
    PrefixScope,
    PrefixTree,
    PrefixTreeCandidate,
    PrefixTreeNode,
    PromptBlock,
    SemanticType,
    ShareScope,
)


class PlannerFeedbackPolicy(Protocol):
    def success_rate_for(self, block: PromptBlock) -> float | None:
        ...


@dataclass(frozen=True)
class PrefixPlan:
    original_order: tuple[str, ...]
    new_order: tuple[str, ...]
    moved_blocks: tuple[str, ...]
    kept_blocks: tuple[str, ...]
    move_reason: dict[str, str]
    cacheable_prefix_blocks: tuple[str, ...] = ()
    prefix_tree: PrefixTree | None = None
    prefix_tree_candidate: PrefixTreeCandidate | None = None
    placements: tuple[BlockPlacement, ...] = ()
    estimated_cache_gain: float = 0.0
    planner_score: float = 0.0
    planner_score_breakdown: Mapping[str, Any] | None = None
    cache_gain_report: Mapping[str, Any] | None = None
    risk_notes: tuple[str, ...] = ()
    fallback_required: bool = False
    fallback_reason: str | None = None


@dataclass
class _SeenBlock:
    count: int = 0
    agents_or_sources: set[str] = field(default_factory=set)
    semantic_types: set[SemanticType] = field(default_factory=set)


class HierarchicalPrefixPlanner:
    """Planner that emits replayable prefix-tree candidates while preserving the old PrefixPlan API."""

    _semantic_priority: dict[SemanticType, int] = {
        SemanticType.GLOBAL_TASK_BACKGROUND: 10,
        SemanticType.SHARED_CONTEXT: 20,
        SemanticType.TEAM_POLICY: 30,
        SemanticType.SHARED_TOOL_DESCRIPTION: 40,
        SemanticType.OUTPUT_FORMAT: 50,
    }

    def __init__(
        self,
        *,
        enable_groupchat_history_reordering: bool = False,
        feedback_policy: PlannerFeedbackPolicy | None = None,
    ) -> None:
        self.enable_groupchat_history_reordering = enable_groupchat_history_reordering
        self.feedback_policy = feedback_policy
        self._seen_by_session: dict[str, dict[str, _SeenBlock]] = defaultdict(dict)
        self._last_candidates: tuple[PrefixTreeCandidate, ...] = ()

    @property
    def last_candidates(self) -> tuple[PrefixTreeCandidate, ...]:
        return self._last_candidates

    def plan(self, compile_result: CompileResult, *, session_id: str | None = None) -> PrefixPlan:
        effective_session_id = session_id or compile_result.session_id
        candidates = self.generate_candidates(compile_result, session_id=effective_session_id, top_k=3)
        candidate = candidates[0] if candidates else self._empty_candidate(compile_result, effective_session_id)
        plan = self._plan_from_candidate(compile_result, candidate)
        self.observe(compile_result, session_id=effective_session_id)
        return plan

    def generate_candidates(
        self,
        compile_result: CompileResult,
        *,
        session_id: str | None = None,
        top_k: int = 3,
    ) -> tuple[PrefixTreeCandidate, ...]:
        effective_session_id = session_id or compile_result.session_id
        blocks = compile_result.blocks
        system_candidates = self._rank_candidates(
            [block for block in blocks if self._can_promote(block, compile_result, effective_session_id)]
        )
        history_candidates = self._history_prefix_candidates(
            compile_result,
            session_id=effective_session_id,
        )
        eligible = (*system_candidates, *history_candidates)

        policies: tuple[tuple[str, tuple[PromptBlock, ...], str], ...] = (
            (
                "conservative",
                tuple(
                    block
                    for block in eligible
                    if block.movability == Movability.SAFE_PREFIX
                    and block.share_scope == ShareScope.GLOBAL
                    and not block.has_hard_risk
                ),
                "only exact repeated low-risk global safe-prefix blocks are promoted",
            ),
            (
                "balanced",
                eligible,
                "exact repeated global and subgroup safe/conditional blocks are arranged as a prefix tree",
            ),
            (
                "aggressive",
                eligible,
                "all hard-gate-eligible repeated shared blocks are considered; validator still fail-closes",
            ),
        )
        candidates = tuple(
            self._build_candidate(
                compile_result=compile_result,
                session_id=effective_session_id,
                policy_name=policy_name,
                promoted_blocks=promoted,
                generation_reason=reason,
            )
            for policy_name, promoted, reason in policies
        )
        ordered = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.planner_score,
                    {"balanced": 0, "conservative": 1, "aggressive": 2}.get(item.generation_reason.split(":", 1)[0], 3),
                    item.candidate_id,
                ),
            )
        )
        limit = max(1, int(top_k))
        self._last_candidates = ordered[:limit]
        return self._last_candidates

    def materialize_prompt(self, candidate: PrefixTreeCandidate, agent_id: str) -> str:
        if agent_id not in candidate.agent_block_orders:
            raise KeyError(f"agent_id not present in candidate: {agent_id}")
        block_order = candidate.agent_block_orders[agent_id]
        if len(block_order) != len(set(block_order)):
            raise ValueError("candidate materialization would duplicate a block")
        if candidate.block_text_by_id is None:
            raise ValueError("candidate does not contain materializable block text")
        missing = [block_id for block_id in block_order if block_id not in candidate.block_text_by_id]
        if missing:
            raise ValueError(f"candidate materialization missing block text: {','.join(missing)}")
        return "".join(str(candidate.block_text_by_id[block_id]) for block_id in block_order)

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

    def _plan_from_candidate(self, compile_result: CompileResult, candidate: PrefixTreeCandidate) -> PrefixPlan:
        agent_id = self._agent_id_for(compile_result)
        original_order = tuple(block.block_id for block in compile_result.blocks)
        new_order = tuple(candidate.agent_block_orders.get(agent_id, original_order))
        moved_blocks = tuple(placement.block_id for placement in candidate.placements if placement.moved)
        move_reason = {
            placement.block_id: self._move_reason_for_placement(placement, compile_result)
            for placement in candidate.placements
            if placement.moved
        }
        risk_notes = tuple(
            f"{block.block_id}: {','.join(block.risk_tags)}"
            for block in compile_result.blocks
            if block.movability in {Movability.LOCAL_ONLY, Movability.ORDER_SENSITIVE, Movability.NEVER_MOVE}
        )
        cacheable_prefix_blocks = tuple(
            block_id
            for node_id in candidate.agent_paths.get(agent_id, ())
            for block_id in candidate.nodes[node_id].block_ids
            if node_id != candidate.agent_paths.get(agent_id, ())[-1]
        )
        return PrefixPlan(
            original_order=original_order,
            new_order=new_order,
            moved_blocks=moved_blocks,
            kept_blocks=tuple(block_id for block_id in original_order if block_id not in moved_blocks),
            move_reason=move_reason,
            cacheable_prefix_blocks=cacheable_prefix_blocks,
            prefix_tree=self._tree_from_candidate(candidate),
            prefix_tree_candidate=candidate,
            placements=candidate.placements,
            estimated_cache_gain=candidate.estimated_cache_gain,
            planner_score=candidate.planner_score,
            planner_score_breakdown=candidate.planner_score_breakdown,
            cache_gain_report=candidate.cache_gain_report,
            risk_notes=risk_notes,
        )

    def _build_candidate(
        self,
        *,
        compile_result: CompileResult,
        session_id: str,
        policy_name: str,
        promoted_blocks: tuple[PromptBlock, ...],
        generation_reason: str,
    ) -> PrefixTreeCandidate:
        blocks = compile_result.blocks
        agent_id = self._agent_id_for(compile_result)
        candidate_id = f"{session_id}:{agent_id}:{policy_name}"
        original_order = tuple(block.block_id for block in blocks)
        global_blocks = tuple(block for block in promoted_blocks if block.share_scope == ShareScope.GLOBAL)
        subgroup_blocks = tuple(block for block in promoted_blocks if block.share_scope == ShareScope.SUBGROUP)
        global_ids = tuple(block.block_id for block in global_blocks)
        subgroup_ids = tuple(block.block_id for block in subgroup_blocks)
        shared_ids = set(global_ids) | set(subgroup_ids)
        new_order = global_ids + subgroup_ids + tuple(block_id for block_id in original_order if block_id not in shared_ids)

        root_id = f"{candidate_id}:root"
        subgroup_id = f"{candidate_id}:subgroup"
        leaf_id = f"{candidate_id}:leaf:{agent_id}"
        global_agent_ids = self._target_group_for(global_blocks, compile_result, session_id, fallback_agent=agent_id)
        subgroup_agent_ids = self._target_group_for(subgroup_blocks, compile_result, session_id, fallback_agent=agent_id)
        leaf_agent_ids = (agent_id,)

        leaf = PrefixTreeNode(
            node_id=leaf_id,
            scope=ShareScope.AGENT,
            scope_type=PrefixScope.AGENT_LOCAL,
            label="agent_local_suffix",
            agent_ids=leaf_agent_ids,
            block_ids=tuple(block_id for block_id in new_order if block_id not in shared_ids),
            parent_id=subgroup_id if subgroup_ids else root_id,
            token_len=self._sum_tokens(blocks, tuple(block_id for block_id in new_order if block_id not in shared_ids)),
            risk_tags=self._risk_tags_for(blocks, tuple(block_id for block_id in new_order if block_id not in shared_ids)),
            explanation="agent-local suffix preserves role, private state, latest instructions, and non-shared blocks",
        )
        if subgroup_ids:
            subgroup = PrefixTreeNode(
                node_id=subgroup_id,
                scope=ShareScope.SUBGROUP,
                scope_type=PrefixScope.SUBGROUP,
                label="subgroup_shared_prefix",
                agent_ids=subgroup_agent_ids,
                block_ids=subgroup_ids,
                parent_id=root_id,
                children_ids=(leaf_id,),
                children=(leaf,),
                token_len=self._sum_tokens(blocks, subgroup_ids),
                risk_tags=self._risk_tags_for(blocks, subgroup_ids),
                explanation="subgroup prefix contains exact repeated blocks shared by a subset of agents",
            )
            root_children = (subgroup,)
            root_child_ids = (subgroup_id,)
            agent_path = (root_id, subgroup_id, leaf_id)
            nodes = {subgroup_id: subgroup, leaf_id: leaf}
        else:
            root_children = (leaf,)
            root_child_ids = (leaf_id,)
            agent_path = (root_id, leaf_id)
            nodes = {leaf_id: leaf}

        root = PrefixTreeNode(
            node_id=root_id,
            scope=ShareScope.GLOBAL,
            scope_type=PrefixScope.GLOBAL,
            label="global_shared_prefix",
            agent_ids=global_agent_ids,
            block_ids=global_ids,
            parent_id=None,
            children_ids=root_child_ids,
            children=root_children,
            token_len=self._sum_tokens(blocks, global_ids),
            risk_tags=self._risk_tags_for(blocks, global_ids),
            explanation="global prefix contains exact repeated blocks eligible for all-agent sharing",
        )
        nodes[root_id] = root
        ordered_nodes = {node_id: nodes[node_id] for node_id in (root_id, *agent_path[1:])}

        placements = self._placements_for(
            candidate_id=candidate_id,
            compile_result=compile_result,
            new_order=new_order,
            root_id=root_id,
            subgroup_id=subgroup_id if subgroup_ids else None,
            leaf_id=leaf_id,
            global_ids=set(global_ids),
            subgroup_ids=set(subgroup_ids),
            global_agent_ids=global_agent_ids,
            subgroup_agent_ids=subgroup_agent_ids,
            leaf_agent_ids=leaf_agent_ids,
        )
        cache_gain_report = self._cache_gain_report(
            candidate_id=candidate_id,
            nodes=ordered_nodes,
            agent_paths={agent_id: agent_path},
            placements=placements,
        )
        estimated_cache_gain = float(cache_gain_report["estimated_cache_gain"])
        planner_score, planner_score_breakdown = self._score_candidate(
            nodes=ordered_nodes,
            placements=placements,
            estimated_cache_gain=estimated_cache_gain,
        )
        block_text_by_id = {
            block.block_id: (block.rendered_text if block.rendered_text is not None else str(block.text_or_payload))
            for block in blocks
        }
        return PrefixTreeCandidate(
            candidate_id=candidate_id,
            root_node_id=root_id,
            nodes=ordered_nodes,
            agent_paths={agent_id: agent_path},
            placements=placements,
            estimated_cache_gain=estimated_cache_gain,
            planner_score=planner_score,
            planner_score_breakdown=planner_score_breakdown,
            generation_reason=f"{policy_name}:{generation_reason}",
            agent_block_orders={agent_id: new_order},
            materialized_prompts={agent_id: "".join(block_text_by_id[block_id] for block_id in new_order)},
            block_hash_by_id={block.block_id: block.content_hash for block in blocks},
            block_text_by_id=block_text_by_id,
            cache_gain_report=cache_gain_report,
        )

    def _placements_for(
        self,
        *,
        candidate_id: str,
        compile_result: CompileResult,
        new_order: tuple[str, ...],
        root_id: str,
        subgroup_id: str | None,
        leaf_id: str,
        global_ids: set[str],
        subgroup_ids: set[str],
        global_agent_ids: tuple[str, ...],
        subgroup_agent_ids: tuple[str, ...],
        leaf_agent_ids: tuple[str, ...],
    ) -> tuple[BlockPlacement, ...]:
        placements: list[BlockPlacement] = []
        original_order = tuple(block.block_id for block in compile_result.blocks)
        for index, block in enumerate(compile_result.blocks):
            if block.block_id in global_ids:
                target_scope: PrefixScope | str = PrefixScope.GLOBAL
                target_node_id = root_id
                target_group = global_agent_ids
            elif block.block_id in subgroup_ids and subgroup_id is not None:
                target_scope = PrefixScope.SUBGROUP
                target_node_id = subgroup_id
                target_group = subgroup_agent_ids
            else:
                target_scope = PrefixScope.AGENT_LOCAL
                target_node_id = leaf_id
                target_group = leaf_agent_ids
            moved = (
                block.block_id in global_ids | subgroup_ids
                and original_order.index(block.block_id) != new_order.index(block.block_id)
            )
            cache_contribution = self._placement_cache_contribution(block, target_scope, target_group)
            placement_score, placement_breakdown = self._score_placement(
                block=block,
                target_scope=target_scope,
                moved=moved,
                target_group=target_group,
                cache_contribution=cache_contribution,
            )
            placements.append(
                BlockPlacement(
                    placement_id=f"{candidate_id}:placement:{index}:{block.block_id}",
                    block_id=block.block_id,
                    block_hash=block.content_hash,
                    original_agent_id=block.agent_or_source or compile_result.session_id,
                    original_position=block.original_position,
                    original_scope=block.share_scope,
                    target_scope=target_scope,
                    target_node_id=target_node_id,
                    target_agent_group=target_group,
                    moved=moved,
                    risk_tags=block.risk_tags,
                    dependency_notes=block.dependency_refs,
                    cache_contribution=cache_contribution,
                    placement_score=placement_score,
                    placement_score_breakdown=placement_breakdown,
                )
            )
        return tuple(placements)

    def _score_placement(
        self,
        *,
        block: PromptBlock,
        target_scope: PrefixScope | str,
        moved: bool,
        target_group: tuple[str, ...],
        cache_contribution: float,
    ) -> tuple[float, Mapping[str, float]]:
        normalized_scope = _scope_value(target_scope)
        cache_gain_score = min(1.0, cache_contribution / 512.0)
        sharing_scope_score = {"global": 1.0, "subgroup": 0.7, "agent_local": 0.1}.get(normalized_scope, 0.0)
        historical_placement_score = self._historical_score_for_placement(block, target_scope, moved, target_group)
        risk_penalty = (0.75 if block.has_hard_risk and normalized_scope != "agent_local" else 0.0) + min(
            0.3,
            len(block.risk_tags) * 0.03,
        )
        dependency_penalty = min(0.35, len(block.dependency_refs) * (0.08 if moved else 0.02))
        movement_penalty = 0.05 if moved else 0.0
        privacy_penalty = (
            1.0
            if normalized_scope != "agent_local"
            and (block.contains_private_info or block.contains_credential or block.contains_tool_permission)
            else 0.0
        )
        score = (
            0.35 * cache_gain_score
            + 0.2 * sharing_scope_score
            + 0.25 * historical_placement_score
            - risk_penalty
            - dependency_penalty
            - movement_penalty
            - privacy_penalty
        )
        breakdown = {
            "cache_gain_score": cache_gain_score,
            "sharing_scope_score": sharing_scope_score,
            "historical_placement_score": historical_placement_score,
            "risk_penalty": risk_penalty,
            "dependency_penalty": dependency_penalty,
            "movement_penalty": movement_penalty,
            "privacy_penalty": privacy_penalty,
        }
        return score, breakdown

    def _score_candidate(
        self,
        *,
        nodes: Mapping[str, PrefixTreeNode],
        placements: tuple[BlockPlacement, ...],
        estimated_cache_gain: float,
    ) -> tuple[float, Mapping[str, Any]]:
        if placements:
            aggregate_placement_score = sum(placement.placement_score for placement in placements) / len(placements)
            historical_scores = [
                float((placement.placement_score_breakdown or {}).get("historical_placement_score", 0.5))
                for placement in placements
            ]
        else:
            aggregate_placement_score = 0.0
            historical_scores = [0.5]
        hard_risk_count = sum(
            1
            for placement in placements
            if _scope_value(placement.target_scope) != "agent_local"
            and bool(set(placement.risk_tags) & {"agent_identity", "private_memory", "private_tool_permission", "credential"})
        )
        tree_complexity_penalty = max(0, len(nodes) - 2) * 0.05
        historical_feedback_adjustment = (sum(historical_scores) / len(historical_scores)) - 0.5
        cache_gain_score = min(2.0, estimated_cache_gain / 256.0)
        score = (
            aggregate_placement_score
            + cache_gain_score
            - hard_risk_count * 2.0
            - tree_complexity_penalty
            + historical_feedback_adjustment
        )
        breakdown = {
            "placement_score_aggregate": aggregate_placement_score,
            "estimated_cache_gain": estimated_cache_gain,
            "estimated_cache_gain_score": cache_gain_score,
            "hard_risk_summary": {"shared_hard_risk_count": hard_risk_count},
            "tree_complexity_penalty": tree_complexity_penalty,
            "historical_feedback_adjustment": historical_feedback_adjustment,
        }
        return score, breakdown

    def _cache_gain_report(
        self,
        *,
        candidate_id: str,
        nodes: Mapping[str, PrefixTreeNode],
        agent_paths: Mapping[str, tuple[str, ...]],
        placements: tuple[BlockPlacement, ...],
    ) -> Mapping[str, Any]:
        node_contributions = {}
        global_prefix_tokens = 0
        subgroup_prefix_tokens = 0
        for node_id, node in nodes.items():
            contribution = float(node.token_len * max(0, len(node.agent_ids) - 1))
            if _scope_value(node.scope_type or node.scope) == "global":
                global_prefix_tokens += node.token_len
            if _scope_value(node.scope_type or node.scope) == "subgroup":
                subgroup_prefix_tokens += node.token_len
            node_contributions[node_id] = {
                "scope_type": _scope_value(node.scope_type or node.scope),
                "agent_count": len(node.agent_ids),
                "token_len": node.token_len,
                "cache_contribution": contribution,
            }
        estimated_cache_gain = sum(value["cache_contribution"] for value in node_contributions.values())
        lcp_tokens = self._longest_common_prefix_tokens(nodes=nodes, agent_paths=agent_paths)
        return {
            "schema_version": "prefix-tree-cache-gain-report-v1",
            "candidate_id": candidate_id,
            "longest_common_prefix_tokens": lcp_tokens,
            "global_prefix_tokens": global_prefix_tokens,
            "subgroup_prefix_tokens": subgroup_prefix_tokens,
            "node_cache_contributions": node_contributions,
            "placement_cache_contributions": {
                placement.placement_id: placement.cache_contribution for placement in placements
            },
            "estimated_cache_gain": estimated_cache_gain,
        }

    def _longest_common_prefix_tokens(
        self,
        *,
        nodes: Mapping[str, PrefixTreeNode],
        agent_paths: Mapping[str, tuple[str, ...]],
    ) -> int:
        paths = list(agent_paths.values())
        if len(paths) < 2:
            return sum(
                node.token_len
                for node in nodes.values()
                if _scope_value(node.scope_type or node.scope) in {"global", "subgroup"}
                and len(node.agent_ids) > 1
            )
        total = 0
        for group in zip(*paths):
            if len(set(group)) != 1:
                break
            total += nodes[group[0]].token_len
        return total

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
        if self._has_nonshareable_hard_risk(block):
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
                    self._feedback_rank(block),
                    self._semantic_priority.get(block.semantic_type, 100),
                    block.original_position.message_index,
                    block.original_position.part_index,
                    block.block_id,
                ),
            )
        )

    def _move_reason_for_placement(self, placement: BlockPlacement, compile_result: CompileResult) -> str:
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        block = blocks_by_id.get(placement.block_id)
        semantic_hint = block.semantic_type.value if block is not None else "unknown"
        return (
            f"candidate_placement={_scope_value(placement.target_scope)};"
            f"semantic_hint={semantic_hint};placement_score={placement.placement_score:.3f}"
        )

    def _feedback_score(self, block: PromptBlock) -> float | None:
        if self.feedback_policy is None:
            return None
        return self.feedback_policy.success_rate_for(block)

    def _feedback_rank(self, block: PromptBlock) -> float:
        score = self._feedback_score(block)
        if score is None:
            return 0.5
        return 1.0 - score

    def _historical_score_for_placement(
        self,
        block: PromptBlock,
        target_scope: PrefixScope | str,
        moved: bool,
        target_group: tuple[str, ...],
    ) -> float:
        if self.feedback_policy is None:
            return 0.5
        getter = getattr(self.feedback_policy, "get_historical_score_for_placement", None)
        if callable(getter):
            try:
                value = getter(
                    block_id=block.block_id,
                    block_hash=block.content_hash,
                    original_scope=block.share_scope,
                    target_scope=target_scope,
                    target_agent_group=target_group,
                    risk_tags=block.risk_tags,
                    moved=moved,
                )
                if value is not None:
                    return float(value)
            except TypeError:
                pass
        score = self._feedback_score(block)
        return 0.5 if score is None else float(score)

    def _tree_from_candidate(self, candidate: PrefixTreeCandidate) -> PrefixTree:
        root = candidate.nodes[candidate.root_node_id]
        first_path = next(iter(candidate.agent_paths.values()), (candidate.root_node_id,))
        return PrefixTree(
            session_id=candidate.candidate_id.rsplit(":", 2)[0],
            root=root,
            leaf_path=first_path,
        )

    def _empty_candidate(self, compile_result: CompileResult, session_id: str) -> PrefixTreeCandidate:
        return self._build_candidate(
            compile_result=compile_result,
            session_id=session_id,
            policy_name="balanced",
            promoted_blocks=(),
            generation_reason="no eligible repeated shared blocks",
        )

    def _agent_id_for(self, compile_result: CompileResult) -> str:
        for block in compile_result.blocks:
            if block.contains_role_identity and block.agent_or_source:
                return str(block.agent_or_source)
        for block in compile_result.blocks:
            if block.agent_or_source and str(block.agent_or_source).lower() not in {"user", "human", "system"}:
                return str(block.agent_or_source)
        return compile_result.session_id

    def _target_group_for(
        self,
        blocks: tuple[PromptBlock, ...],
        compile_result: CompileResult,
        session_id: str,
        *,
        fallback_agent: str,
    ) -> tuple[str, ...]:
        agents: set[str] = set()
        for block in blocks:
            seen = self._seen_by_session.get(session_id, {}).get(block.content_hash)
            if seen is not None:
                agents.update(seen.agents_or_sources)
            if block.agent_or_source:
                agents.add(str(block.agent_or_source))
        if not agents:
            agents.add(fallback_agent)
        if len(agents) == 1:
            seen_counts = [
                self._seen_by_session.get(session_id, {}).get(block.content_hash).count
                for block in blocks
                if self._seen_by_session.get(session_id, {}).get(block.content_hash) is not None
            ]
            if any(count and count > 0 for count in seen_counts):
                agents.add(f"{session_id}:previous")
        return tuple(sorted(agents))

    def _placement_cache_contribution(
        self,
        block: PromptBlock,
        target_scope: PrefixScope | str,
        target_group: tuple[str, ...],
    ) -> float:
        if _scope_value(target_scope) == "agent_local":
            return 0.0
        return float(block.token_len * max(0, len(target_group) - 1))

    def _sum_tokens(self, blocks: tuple[PromptBlock, ...], block_ids: tuple[str, ...]) -> int:
        by_id = {block.block_id: block for block in blocks}
        return sum(by_id[block_id].token_len for block_id in block_ids if block_id in by_id)

    def _risk_tags_for(self, blocks: tuple[PromptBlock, ...], block_ids: tuple[str, ...]) -> tuple[str, ...]:
        by_id = {block.block_id: block for block in blocks}
        tags = {
            tag
            for block_id in block_ids
            for tag in (by_id[block_id].risk_tags if block_id in by_id else ())
        }
        return tuple(sorted(tags))

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

    def _has_nonshareable_hard_risk(self, block: PromptBlock) -> bool:
        return (
            block.contains_private_info
            or block.contains_role_identity
            or block.contains_tool_permission
            or block.contains_latest_user_instruction
            or block.contains_tool_result
            or block.contains_credential
        )


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


def _scope_value(scope: PrefixScope | ShareScope | str | None) -> str:
    if scope is None:
        return "unknown"
    value = getattr(scope, "value", scope)
    if value == "agent":
        return "agent_local"
    return str(value)
