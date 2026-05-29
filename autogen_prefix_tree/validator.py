from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from autogen_core.models import LLMMessage

from .ir import CompileResult, Movability, hash_model_args, hash_tools
from .planner import PrefixPlan


@dataclass(frozen=True)
class ValidationReport:
    applied: bool
    fallback: bool
    reason: str
    risk_notes: tuple[str, ...] = ()


class CacheUtilityValidator:
    """独立安全阀：无法证明只发生允许的 block 顺序变化时，一律回退。"""

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
        if plan.fallback_required:
            return ValidationReport(False, True, plan.fallback_reason or "planner_required_fallback", plan.risk_notes)

        original_ids = tuple(block.block_id for block in compile_result.blocks)
        if Counter(original_ids) != Counter(plan.new_order):
            return ValidationReport(False, True, "block_id_multiset_changed", plan.risk_notes)

        original_hashes = Counter(block.content_hash for block in compile_result.blocks)
        planned_hashes = Counter(
            block.content_hash
            for block in compile_result.blocks
            if block.block_id in set(plan.new_order)
        )
        if original_hashes != planned_hashes:
            return ValidationReport(False, True, "block_hash_multiset_changed", plan.risk_notes)

        if compile_result.tools_hash != hash_tools(tools):
            return ValidationReport(False, True, "tools_hash_changed", plan.risk_notes)

        if compile_result.model_args_hash != hash_model_args(tool_choice, json_output, extra_create_args):
            return ValidationReport(False, True, "model_args_hash_changed", plan.risk_notes)

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
                    return ValidationReport(False, True, f"forbidden_block_moved:{block.block_id}", plan.risk_notes)
        if tuple(forbidden_original_order) != tuple(forbidden_new_order):
            return ValidationReport(False, True, "forbidden_block_relative_order_changed", plan.risk_notes)

        if len(original_messages) != len(rewritten_messages):
            return ValidationReport(False, True, "message_count_changed", plan.risk_notes)

        for original, rewritten in zip(original_messages, rewritten_messages):
            if getattr(original, "type", type(original).__name__) != getattr(rewritten, "type", type(rewritten).__name__):
                return ValidationReport(False, True, "message_type_changed", plan.risk_notes)
            if getattr(original, "source", None) != getattr(rewritten, "source", None):
                return ValidationReport(False, True, "message_source_changed", plan.risk_notes)
            if not hasattr(rewritten, "model_dump"):
                return ValidationReport(False, True, "rewritten_message_not_serializable", plan.risk_notes)

        return ValidationReport(
            applied=bool(plan.moved_blocks),
            fallback=False,
            reason="validated" if plan.moved_blocks else "no_rewrite_needed",
            risk_notes=plan.risk_notes,
        )
