from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from autogen_core.models import LLMMessage

from .ir import CompileResult, Movability, PrefixTreeNode, hash_model_args, hash_tools, stable_hash
from .planner import PrefixPlan
from .semantic_guard import SemanticGuard, SemanticGuardReport


@dataclass(frozen=True)
class CacheUtilityEstimate:
    cacheable_prefix_blocks: tuple[str, ...]
    original_prefix_chars: int
    rewritten_prefix_chars: int
    moved_block_chars: int
    estimated_gain_chars: int
    prefix_fingerprint_before: str
    prefix_fingerprint_after: str


@dataclass(frozen=True)
class ValidationReport:
    applied: bool
    fallback: bool
    reason: str
    utility_estimate: CacheUtilityEstimate | None = None
    semantic_guard_report: SemanticGuardReport | None = None
    risk_notes: tuple[str, ...] = ()


class CacheUtilityValidator:
    """独立安全阀：无法证明只发生允许的 block 顺序变化时，一律回退。"""

    def __init__(self, *, min_estimated_gain_chars: int = 1, semantic_guard: SemanticGuard | None = None) -> None:
        self.min_estimated_gain_chars = min_estimated_gain_chars
        self.semantic_guard = semantic_guard

    def validate(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] | None = None,
    ) -> ValidationReport:
        utility_estimate = self._estimate_cache_utility(compile_result, plan)

        if plan.fallback_required:
            return ValidationReport(
                False,
                True,
                plan.fallback_reason or "planner_required_fallback",
                utility_estimate,
                None,
                plan.risk_notes,
            )

        original_ids = tuple(block.block_id for block in compile_result.blocks)
        if Counter(original_ids) != Counter(plan.new_order):
            return ValidationReport(False, True, "block_id_multiset_changed", utility_estimate, None, plan.risk_notes)

        if plan.prefix_tree is None:
            return ValidationReport(False, True, "prefix_tree_missing", utility_estimate, None, plan.risk_notes)
        tree_ids = tuple(self._iter_tree_block_ids(plan.prefix_tree.root))
        if Counter(tree_ids) != Counter(original_ids):
            return ValidationReport(
                False,
                True,
                "prefix_tree_block_coverage_mismatch",
                utility_estimate,
                None,
                plan.risk_notes,
            )
        if not self._cacheable_prefix_is_front_loaded(plan):
            return ValidationReport(False, True, "cacheable_prefix_not_at_front", utility_estimate, None, plan.risk_notes)

        original_hashes = Counter(block.content_hash for block in compile_result.blocks)
        planned_hashes = Counter(
            block.content_hash
            for block in compile_result.blocks
            if block.block_id in set(plan.new_order)
        )
        if original_hashes != planned_hashes:
            return ValidationReport(False, True, "block_hash_multiset_changed", utility_estimate, None, plan.risk_notes)

        if compile_result.tools_hash != hash_tools(tools):
            return ValidationReport(False, True, "tools_hash_changed", utility_estimate, None, plan.risk_notes)

        if compile_result.model_args_hash != hash_model_args(tool_choice, json_output, extra_create_args):
            return ValidationReport(False, True, "model_args_hash_changed", utility_estimate, None, plan.risk_notes)

        forbidden_original_order: list[str] = []
        forbidden_new_order = [
            block_id
            for block_id in plan.new_order
            if any(
                block.block_id == block_id
                and block.movability in {Movability.NEVER_MOVE, Movability.LOCAL_ONLY, Movability.ORDER_SENSITIVE}
                for block in compile_result.blocks
            )
        ]
        for block in compile_result.blocks:
            if block.movability in {Movability.NEVER_MOVE, Movability.LOCAL_ONLY, Movability.ORDER_SENSITIVE}:
                forbidden_original_order.append(block.block_id)
                if block.block_id in plan.moved_blocks:
                    return ValidationReport(
                        False,
                        True,
                        f"forbidden_block_moved:{block.block_id}",
                        utility_estimate,
                        None,
                        plan.risk_notes,
                    )
        if tuple(forbidden_original_order) != tuple(forbidden_new_order):
            return ValidationReport(
                False,
                True,
                "forbidden_block_relative_order_changed",
                utility_estimate,
                None,
                plan.risk_notes,
            )

        if len(original_messages) != len(rewritten_messages):
            return ValidationReport(False, True, "message_count_changed", utility_estimate, None, plan.risk_notes)

        for original, rewritten in zip(original_messages, rewritten_messages):
            if getattr(original, "type", type(original).__name__) != getattr(rewritten, "type", type(rewritten).__name__):
                return ValidationReport(False, True, "message_type_changed", utility_estimate, None, plan.risk_notes)
            if getattr(original, "source", None) != getattr(rewritten, "source", None):
                return ValidationReport(False, True, "message_source_changed", utility_estimate, None, plan.risk_notes)
            if not hasattr(rewritten, "model_dump"):
                return ValidationReport(
                    False,
                    True,
                    "rewritten_message_not_serializable",
                    utility_estimate,
                    None,
                    plan.risk_notes,
                )

        if plan.moved_blocks and not self._rewritten_messages_match_plan(
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
        ):
            return ValidationReport(False, True, "rewritten_content_mismatch", utility_estimate, None, plan.risk_notes)

        if plan.moved_blocks and utility_estimate.estimated_gain_chars < self.min_estimated_gain_chars:
            return ValidationReport(False, True, "insufficient_cache_utility", utility_estimate, None, plan.risk_notes)

        semantic_guard_report = self._run_semantic_guard(
            original_messages=original_messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
        )
        if semantic_guard_report is not None and not semantic_guard_report.passed:
            return ValidationReport(
                False,
                True,
                f"semantic_guard_failed:{semantic_guard_report.reason}",
                utility_estimate,
                semantic_guard_report,
                plan.risk_notes,
            )

        return ValidationReport(
            applied=bool(plan.moved_blocks),
            fallback=False,
            reason="validated" if plan.moved_blocks else "no_rewrite_needed",
            utility_estimate=utility_estimate,
            semantic_guard_report=semantic_guard_report,
            risk_notes=plan.risk_notes,
        )

    def _run_semantic_guard(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> SemanticGuardReport | None:
        if self.semantic_guard is None or not plan.moved_blocks:
            return None
        try:
            return self.semantic_guard.evaluate(
                original_messages=original_messages,
                rewritten_messages=rewritten_messages,
                compile_result=compile_result,
                plan=plan,
            )
        except Exception as exc:  # noqa: BLE001
            return SemanticGuardReport(
                passed=False,
                reason=f"exception:{type(exc).__name__}",
                checks=("semantic_guard_exception",),
            )

    def _iter_tree_block_ids(self, node: PrefixTreeNode) -> tuple[str, ...]:
        ids: list[str] = list(node.block_ids)
        for child in node.children:
            ids.extend(self._iter_tree_block_ids(child))
        return tuple(ids)

    def _cacheable_prefix_is_front_loaded(self, plan: PrefixPlan) -> bool:
        if not plan.cacheable_prefix_blocks:
            return True
        prefix_len = len(plan.cacheable_prefix_blocks)
        return plan.new_order[:prefix_len] == plan.cacheable_prefix_blocks

    def _rewritten_messages_match_plan(
        self,
        *,
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> bool:
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        moved_message_indexes = {
            blocks_by_id[block_id].original_position.message_index
            for block_id in plan.moved_blocks
            if block_id in blocks_by_id
        }
        for message_index in moved_message_indexes:
            target_ids = {
                block.block_id
                for block in compile_result.blocks
                if block.original_position.message_index == message_index
            }
            ordered_blocks = [blocks_by_id[block_id] for block_id in plan.new_order if block_id in target_ids]
            if any(block.rendered_text is None for block in ordered_blocks):
                return False
            expected_content = "".join(block.rendered_text or "" for block in ordered_blocks)
            if getattr(rewritten_messages[message_index], "content", None) != expected_content:
                return False
        return True

    def _estimate_cache_utility(self, compile_result: CompileResult, plan: PrefixPlan) -> CacheUtilityEstimate:
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        cacheable_ids = tuple(block_id for block_id in plan.cacheable_prefix_blocks if block_id in blocks_by_id)
        cacheable_set = set(cacheable_ids)
        original_prefix_chars = self._leading_cacheable_chars(plan.original_order, cacheable_set, blocks_by_id)
        rewritten_prefix_chars = self._leading_cacheable_chars(plan.new_order, cacheable_set, blocks_by_id)
        moved_block_chars = sum(
            len(blocks_by_id[block_id].rendered_text or "")
            for block_id in plan.moved_blocks
            if block_id in blocks_by_id
        )
        before_fingerprint = stable_hash(
            [(block_id, blocks_by_id[block_id].content_hash) for block_id in plan.original_order if block_id in blocks_by_id]
        )
        after_fingerprint = stable_hash(
            [(block_id, blocks_by_id[block_id].content_hash) for block_id in plan.new_order if block_id in blocks_by_id]
        )
        return CacheUtilityEstimate(
            cacheable_prefix_blocks=cacheable_ids,
            original_prefix_chars=original_prefix_chars,
            rewritten_prefix_chars=rewritten_prefix_chars,
            moved_block_chars=moved_block_chars,
            estimated_gain_chars=max(0, rewritten_prefix_chars - original_prefix_chars),
            prefix_fingerprint_before=before_fingerprint,
            prefix_fingerprint_after=after_fingerprint,
        )

    def _leading_cacheable_chars(
        self,
        order: tuple[str, ...],
        cacheable_set: set[str],
        blocks_by_id: dict[str, Any],
    ) -> int:
        chars = 0
        for block_id in order:
            if block_id not in cacheable_set:
                break
            chars += len(blocks_by_id[block_id].rendered_text or "")
        return chars
