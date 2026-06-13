from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from autogen_core.models import LLMMessage

from .cache_estimator import CacheEstimateReport, CacheEstimator, cache_gate_value, resolve_cache_estimator
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
    longest_common_prefix_tokens: int = 0
    global_prefix_tokens: int = 0
    subgroup_prefix_tokens: int = 0
    node_cache_contributions: Mapping[str, Any] | None = None
    cache_gain_report: Mapping[str, Any] | None = None
    cache_estimate_report: CacheEstimateReport | None = None


@dataclass(frozen=True)
class UtilityPreservationReport:
    is_utility_preserved: bool | None
    confidence: float
    reason: str
    checks: tuple[str, ...] = ()
    model_name: str | None = None
    prompt_safe: bool = True
    utility_status: str = "unverified"


@dataclass(frozen=True)
class UtilityModelGateReport:
    utility_model_prediction: bool | None
    utility_model_confidence: float
    utility_model_threshold: float | None
    utility_model_shadow_decision: str
    hard_gate_result: str
    cache_gate_result: str
    mode: str = "disabled"
    reason: str | None = None
    model_name: str | None = None
    prompt_safe: bool = True


class UtilityPreservationValidator(Protocol):
    def evaluate(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
        cache_utility_estimate: CacheUtilityEstimate,
    ) -> UtilityPreservationReport:
        ...


@dataclass(frozen=True)
class ValidationReport:
    applied: bool
    fallback: bool
    reason: str
    utility_estimate: CacheUtilityEstimate | None = None
    semantic_guard_report: SemanticGuardReport | None = None
    risk_notes: tuple[str, ...] = ()
    utility_preservation_report: UtilityPreservationReport | None = None
    hard_constraint_passed: bool = False
    hard_constraint_report: Mapping[str, Any] | None = None
    utility_status: str = "unverified"
    cache_hit_increased: bool | None = None
    cache_estimate_report: CacheEstimateReport | None = None
    utility_model_gate_report: UtilityModelGateReport | None = None


class CacheUtilityValidator:
    """独立安全阀：无法证明只发生允许的 block 顺序变化时，一律回退。"""

    def __init__(
        self,
        *,
        min_estimated_gain_chars: int = 1,
        semantic_guard: SemanticGuard | None = None,
        utility_validator: UtilityPreservationValidator | None = None,
        utility_model_validator: UtilityPreservationValidator | None = None,
        utility_model_mode: str = "disabled",
        utility_model_min_confidence: float = 0.55,
        cache_estimator: str | CacheEstimator | None = None,
        rejected_move_risk_tags: Sequence[str] = ("natural_language_segment",),
    ) -> None:
        self.min_estimated_gain_chars = min_estimated_gain_chars
        self.semantic_guard = semantic_guard
        self.utility_validator = utility_validator
        self.utility_model_validator = utility_model_validator
        self.utility_model_mode = _normalize_utility_model_mode(utility_model_mode)
        self.utility_model_min_confidence = max(0.0, min(1.0, float(utility_model_min_confidence)))
        self.cache_estimator = resolve_cache_estimator(cache_estimator)
        self.rejected_move_risk_tags = tuple(str(tag) for tag in rejected_move_risk_tags)

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
        utility_estimate = self._estimate_cache_utility(compile_result, plan, cache_estimate_report=None)

        if plan.fallback_required:
            return self._fail(
                reason=plan.fallback_reason or "planner_required_fallback",
                utility_estimate=utility_estimate,
                risk_notes=plan.risk_notes,
            )

        original_ids = tuple(block.block_id for block in compile_result.blocks)
        if Counter(original_ids) != Counter(plan.new_order):
            return self._fail("block_id_multiset_changed", utility_estimate, plan.risk_notes)

        if plan.prefix_tree is None:
            return self._fail("prefix_tree_missing", utility_estimate, plan.risk_notes)
        tree_ids = tuple(self._iter_tree_block_ids(plan.prefix_tree.root))
        if Counter(tree_ids) != Counter(original_ids):
            return self._fail("prefix_tree_block_coverage_mismatch", utility_estimate, plan.risk_notes)
        if not self._cacheable_prefix_is_front_loaded(plan):
            return self._fail("cacheable_prefix_not_at_front", utility_estimate, plan.risk_notes)

        original_hashes = Counter(block.content_hash for block in compile_result.blocks)
        planned_hashes = Counter(
            block.content_hash
            for block in compile_result.blocks
            if block.block_id in set(plan.new_order)
        )
        if original_hashes != planned_hashes:
            return self._fail("block_hash_multiset_changed", utility_estimate, plan.risk_notes)

        if compile_result.tools_hash != hash_tools(tools):
            return self._fail("tools_hash_changed", utility_estimate, plan.risk_notes)

        if compile_result.model_args_hash != hash_model_args(tool_choice, json_output, extra_create_args):
            return self._fail("model_args_hash_changed", utility_estimate, plan.risk_notes)

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
                    return self._fail(f"forbidden_block_moved:{block.block_id}", utility_estimate, plan.risk_notes)
        if tuple(forbidden_original_order) != tuple(forbidden_new_order):
            return self._fail("forbidden_block_relative_order_changed", utility_estimate, plan.risk_notes)

        message_shape_reason = self._message_shape_violation_reason(
            original_messages=original_messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
        )
        if message_shape_reason is not None:
            return self._fail(message_shape_reason, utility_estimate, plan.risk_notes)

        if plan.moved_blocks and not self._rewritten_messages_match_plan(
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
        ):
            return self._fail("rewritten_content_mismatch", utility_estimate, plan.risk_notes)

        rejected_risk_tags = self._moved_rejected_risk_tags(compile_result, plan)
        if rejected_risk_tags:
            return self._fail(
                "calibrated_utility_risk:" + ",".join(rejected_risk_tags),
                utility_estimate,
                tuple((*plan.risk_notes, *(f"moved_risk_tag:{tag}" for tag in rejected_risk_tags))),
            )

        utility_model_gate_report = self._run_utility_model_gate(
            original_messages=original_messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
            cache_utility_estimate=utility_estimate,
            hard_gate_result="passed",
            cache_gate_result="not_run",
        )
        if utility_model_gate_report.utility_model_shadow_decision == "reject":
            return ValidationReport(
                False,
                True,
                f"utility_model_gate_failed:{utility_model_gate_report.reason}",
                utility_estimate,
                None,
                plan.risk_notes,
                None,
                hard_constraint_passed=True,
                hard_constraint_report=self._hard_constraint_report("passed"),
                utility_status="failed",
                cache_hit_increased=False if plan.moved_blocks else None,
                utility_model_gate_report=utility_model_gate_report,
            )

        utility_preservation_report = self._run_utility_preservation_gate(
            original_messages=original_messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
            cache_utility_estimate=utility_estimate,
        )
        if utility_preservation_report.is_utility_preserved is False:
            return ValidationReport(
                False,
                True,
                f"utility_preservation_failed:{utility_preservation_report.reason}",
                utility_estimate,
                None,
                plan.risk_notes,
                utility_preservation_report,
                hard_constraint_passed=True,
                hard_constraint_report=self._hard_constraint_report("passed"),
                utility_status=utility_preservation_report.utility_status,
                cache_hit_increased=False if plan.moved_blocks else None,
                utility_model_gate_report=utility_model_gate_report,
            )

        cache_estimate_report = self._estimate_cache(
            original_messages=original_messages,
            rewritten_messages=rewritten_messages,
            compile_result=compile_result,
            plan=plan,
        )
        utility_estimate = self._estimate_cache_utility(compile_result, plan, cache_estimate_report)
        utility_model_gate_report = self._update_utility_model_gate_cache_result(
            utility_model_gate_report,
            cache_gate_result=(
                "passed"
                if (not plan.moved_blocks or cache_gate_value(cache_estimate_report) >= self.min_estimated_gain_chars)
                else "failed"
            ),
        )

        if plan.moved_blocks and cache_gate_value(cache_estimate_report) < self.min_estimated_gain_chars:
            return ValidationReport(
                False,
                True,
                "insufficient_cache_utility",
                utility_estimate,
                None,
                plan.risk_notes,
                utility_preservation_report,
                hard_constraint_passed=True,
                hard_constraint_report=self._hard_constraint_report("passed"),
                utility_status=utility_preservation_report.utility_status,
                cache_hit_increased=False,
                cache_estimate_report=cache_estimate_report,
                utility_model_gate_report=utility_model_gate_report,
            )

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
                utility_preservation_report,
                hard_constraint_passed=True,
                hard_constraint_report=self._hard_constraint_report("passed"),
                utility_status=utility_preservation_report.utility_status,
                cache_hit_increased=self._cache_hit_increased(plan, cache_estimate_report),
                cache_estimate_report=cache_estimate_report,
                utility_model_gate_report=utility_model_gate_report,
            )

        cache_hit_increased = self._cache_hit_increased(plan, cache_estimate_report)
        return ValidationReport(
            applied=bool(plan.moved_blocks),
            fallback=False,
            reason="validated" if plan.moved_blocks else "no_rewrite_needed",
            utility_estimate=utility_estimate,
            semantic_guard_report=semantic_guard_report,
            risk_notes=plan.risk_notes,
            utility_preservation_report=utility_preservation_report,
            hard_constraint_passed=True,
            hard_constraint_report=self._hard_constraint_report("passed"),
            utility_status=utility_preservation_report.utility_status,
            cache_hit_increased=cache_hit_increased,
            cache_estimate_report=cache_estimate_report,
            utility_model_gate_report=utility_model_gate_report,
        )

    def _fail(
        self,
        reason: str,
        utility_estimate: CacheUtilityEstimate | None,
        risk_notes: tuple[str, ...],
    ) -> ValidationReport:
        return ValidationReport(
            applied=False,
            fallback=True,
            reason=reason,
            utility_estimate=utility_estimate,
            risk_notes=risk_notes,
            hard_constraint_passed=False,
            hard_constraint_report=self._hard_constraint_report("failed", reason=reason),
            utility_status="not_run",
            cache_hit_increased=False,
            cache_estimate_report=(
                utility_estimate.cache_estimate_report
                if utility_estimate is not None
                else None
            ),
            utility_model_gate_report=self._disabled_utility_model_gate_report(
                hard_gate_result="failed",
                cache_gate_result="not_run",
                reason=reason,
            ),
        )

    def _hard_constraint_report(self, status: str, *, reason: str | None = None) -> Mapping[str, Any]:
        return {
            "schema_version": "prefix-hard-constraint-report-v1",
            "status": status,
            "passed": status == "passed",
            "reason": reason,
            "checks": (
                "block_multiset_preserved",
                "block_hash_multiset_preserved",
                "tools_hash_unchanged",
                "model_args_hash_unchanged",
                "forbidden_blocks_not_moved",
                "conversation_shape_preserved",
                "rewritten_content_matches_plan",
                "prefix_tree_covers_blocks",
                "cacheable_prefix_front_loaded",
            ),
        }

    def _run_utility_preservation_gate(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
        cache_utility_estimate: CacheUtilityEstimate,
    ) -> UtilityPreservationReport:
        if not plan.moved_blocks:
            return UtilityPreservationReport(
                is_utility_preserved=None,
                confidence=1.0,
                reason="no_rewrite_needed",
                checks=("hard_constraints_passed",),
                utility_status="unverified",
            )
        if self.utility_validator is None:
            return UtilityPreservationReport(
                is_utility_preserved=None,
                confidence=0.0,
                reason="utility_validator_not_configured",
                checks=("hard_constraints_passed", "interface_reserved"),
                utility_status="not_configured",
            )
        try:
            report = self.utility_validator.evaluate(
                original_messages=original_messages,
                rewritten_messages=rewritten_messages,
                compile_result=compile_result,
                plan=plan,
                cache_utility_estimate=cache_utility_estimate,
            )
            if report.utility_status == "unverified":
                status = "passed" if report.is_utility_preserved is True else "failed" if report.is_utility_preserved is False else "unverified"
                return UtilityPreservationReport(
                    is_utility_preserved=report.is_utility_preserved,
                    confidence=report.confidence,
                    reason=report.reason,
                    checks=report.checks,
                    model_name=report.model_name,
                    prompt_safe=report.prompt_safe,
                    utility_status=status,
                )
            return report
        except Exception as exc:  # noqa: BLE001
            return UtilityPreservationReport(
                is_utility_preserved=False,
                confidence=0.0,
                reason=f"exception:{type(exc).__name__}",
                checks=("utility_validator_exception",),
                utility_status="failed",
            )

    def _run_utility_model_gate(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
        cache_utility_estimate: CacheUtilityEstimate,
        hard_gate_result: str,
        cache_gate_result: str,
    ) -> UtilityModelGateReport:
        if self.utility_model_validator is None or self.utility_model_mode == "disabled" or not plan.moved_blocks:
            reason = "no_rewrite_needed" if not plan.moved_blocks else "utility_model_not_configured"
            return self._disabled_utility_model_gate_report(
                hard_gate_result=hard_gate_result,
                cache_gate_result=cache_gate_result,
                reason=reason,
            )
        try:
            report = self.utility_model_validator.evaluate(
                original_messages=original_messages,
                rewritten_messages=rewritten_messages,
                compile_result=compile_result,
                plan=plan,
                cache_utility_estimate=cache_utility_estimate,
            )
            prediction = report.is_utility_preserved
            confidence = float(report.confidence)
            threshold = float(getattr(self.utility_model_validator, "threshold", self.utility_model_min_confidence))
            if prediction is None or confidence < self.utility_model_min_confidence:
                decision = "observe"
            elif prediction is False:
                decision = "reject" if self.utility_model_mode == "gate" else "would_reject"
            else:
                decision = "accept" if self.utility_model_mode == "gate" else "would_accept"
            return UtilityModelGateReport(
                utility_model_prediction=prediction,
                utility_model_confidence=confidence,
                utility_model_threshold=threshold,
                utility_model_shadow_decision=decision,
                hard_gate_result=hard_gate_result,
                cache_gate_result=cache_gate_result,
                mode=self.utility_model_mode,
                reason=report.reason,
                model_name=report.model_name,
                prompt_safe=report.prompt_safe,
            )
        except Exception as exc:  # noqa: BLE001
            return UtilityModelGateReport(
                utility_model_prediction=None,
                utility_model_confidence=0.0,
                utility_model_threshold=self.utility_model_min_confidence,
                utility_model_shadow_decision="observe",
                hard_gate_result=hard_gate_result,
                cache_gate_result=cache_gate_result,
                mode=self.utility_model_mode,
                reason=f"exception:{type(exc).__name__}",
                model_name=type(self.utility_model_validator).__name__,
                prompt_safe=True,
            )

    def _disabled_utility_model_gate_report(
        self,
        *,
        hard_gate_result: str,
        cache_gate_result: str,
        reason: str | None = None,
    ) -> UtilityModelGateReport:
        return UtilityModelGateReport(
            utility_model_prediction=None,
            utility_model_confidence=0.0,
            utility_model_threshold=(
                getattr(self.utility_model_validator, "threshold", None)
                if self.utility_model_validator is not None
                else None
            ),
            utility_model_shadow_decision="not_configured" if self.utility_model_validator is None else "disabled",
            hard_gate_result=hard_gate_result,
            cache_gate_result=cache_gate_result,
            mode=self.utility_model_mode,
            reason=reason,
            model_name=getattr(self.utility_model_validator, "model_name", None),
            prompt_safe=True,
        )

    def _update_utility_model_gate_cache_result(
        self,
        report: UtilityModelGateReport,
        *,
        cache_gate_result: str,
    ) -> UtilityModelGateReport:
        return UtilityModelGateReport(
            utility_model_prediction=report.utility_model_prediction,
            utility_model_confidence=report.utility_model_confidence,
            utility_model_threshold=report.utility_model_threshold,
            utility_model_shadow_decision=report.utility_model_shadow_decision,
            hard_gate_result=report.hard_gate_result,
            cache_gate_result=cache_gate_result,
            mode=report.mode,
            reason=report.reason,
            model_name=report.model_name,
            prompt_safe=report.prompt_safe,
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

    def _message_shape_violation_reason(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> str | None:
        if not all(hasattr(message, "model_dump") for message in rewritten_messages):
            return "rewritten_message_not_serializable"

        if len(original_messages) == len(rewritten_messages):
            for original, rewritten in zip(original_messages, rewritten_messages):
                if getattr(original, "type", type(original).__name__) != getattr(
                    rewritten, "type", type(rewritten).__name__
                ):
                    return "message_type_changed"
                if getattr(original, "source", None) != getattr(rewritten, "source", None):
                    return "message_source_changed"
            return None

        if not self._is_allowed_split_system_history_shape(compile_result, plan):
            return "message_count_changed"

        original_non_system = [
            message for message in original_messages if getattr(message, "type", type(message).__name__) != "SystemMessage"
        ]
        rewritten_non_system = [
            message for message in rewritten_messages if getattr(message, "type", type(message).__name__) != "SystemMessage"
        ]
        if len(original_non_system) != len(rewritten_non_system):
            return "message_count_changed"
        original_counter = Counter(self._message_identity(message) for message in original_non_system)
        rewritten_counter = Counter(self._message_identity(message) for message in rewritten_non_system)
        if original_counter != rewritten_counter:
            return "message_identity_changed"
        return None

    def _is_allowed_split_system_history_shape(self, compile_result: CompileResult, plan: PrefixPlan) -> bool:
        if not plan.moved_blocks:
            return False
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        moved_blocks = [blocks_by_id[block_id] for block_id in plan.moved_blocks if block_id in blocks_by_id]
        if not moved_blocks:
            return False
        allowed_risk_tags = {
            "assistant_history_order",
            "groupchat_dialogue_order",
            "groupchat_history_order",
            "initial_user_task",
            "output_format",
            "shared_context",
            "task_background",
            "team_policy",
            "tool_schema",
            "user_message_history_order",
        }
        if any(not set(block.risk_tags).issubset(allowed_risk_tags) for block in moved_blocks):
            return False
        return any(block.semantic_type == block.semantic_type.CONVERSATION_HISTORY for block in moved_blocks)

    def _message_identity(self, message: LLMMessage) -> tuple[str, Any, Any]:
        return (
            getattr(message, "type", type(message).__name__),
            getattr(message, "source", None),
            getattr(message, "content", None),
        )

    def _rewritten_messages_match_plan(
        self,
        *,
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> bool:
        blocks_by_id = {block.block_id: block for block in compile_result.blocks}
        if len(rewritten_messages) != len(compile_result.messages):
            return self._rewritten_split_messages_match_plan(
                rewritten_messages=rewritten_messages,
                compile_result=compile_result,
                plan=plan,
                blocks_by_id=blocks_by_id,
            )
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

    def _rewritten_split_messages_match_plan(
        self,
        *,
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
        blocks_by_id: dict[str, Any],
    ) -> bool:
        expected: list[tuple[str, Any, Any]] = []
        pending_system_index: int | None = None
        pending_system_text: list[str] = []

        def flush_system() -> None:
            nonlocal pending_system_index, pending_system_text
            if pending_system_index is None:
                return
            original = compile_result.messages[pending_system_index]
            expected.append(
                (
                    "SystemMessage",
                    getattr(original, "source", None),
                    "".join(pending_system_text),
                )
            )
            pending_system_index = None
            pending_system_text = []

        for block_id in plan.new_order:
            block = blocks_by_id.get(block_id)
            if block is None:
                continue
            message_index = block.original_position.message_index
            if block.is_system_text:
                if pending_system_index in {None, message_index}:
                    pending_system_index = message_index
                    pending_system_text.append(block.rendered_text or "")
                else:
                    flush_system()
                    pending_system_index = message_index
                    pending_system_text.append(block.rendered_text or "")
                continue
            flush_system()
            original = compile_result.messages[message_index]
            expected.append(self._message_identity(original))
        flush_system()

        actual = [self._message_identity(message) for message in rewritten_messages]
        return expected == actual

    def _moved_rejected_risk_tags(self, compile_result: CompileResult, plan: PrefixPlan) -> tuple[str, ...]:
        if not self.rejected_move_risk_tags or not plan.moved_blocks:
            return ()
        rejected = set(self.rejected_move_risk_tags)
        moved_ids = set(plan.moved_blocks)
        matched: set[str] = set()
        for block in compile_result.blocks:
            if block.block_id not in moved_ids:
                continue
            matched.update(tag for tag in block.risk_tags if tag in rejected)
        return tuple(sorted(matched))

    def _cache_hit_increased(self, plan: PrefixPlan, cache_estimate_report: CacheEstimateReport | None) -> bool | None:
        if not plan.moved_blocks:
            return None
        return cache_gate_value(cache_estimate_report) >= self.min_estimated_gain_chars

    def _estimate_cache(
        self,
        *,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        compile_result: CompileResult,
        plan: PrefixPlan,
    ) -> CacheEstimateReport:
        try:
            return self.cache_estimator.estimate(
                original_messages,
                rewritten_messages,
                candidate=plan.prefix_tree_candidate,
                context={
                    "compile_result": compile_result,
                    "plan": plan,
                    "session_id": compile_result.session_id,
                    "candidate": plan.prefix_tree_candidate,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return CacheEstimateReport(
                estimator_name=type(self.cache_estimator).__name__,
                estimator_available=False,
                reason=f"cache_estimator_exception:{type(exc).__name__}",
                warnings=("cache_estimator_exception",),
            )

    def _estimate_cache_utility(
        self,
        compile_result: CompileResult,
        plan: PrefixPlan,
        cache_estimate_report: CacheEstimateReport | None,
    ) -> CacheUtilityEstimate:
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
        cache_report = plan.cache_gain_report or {}
        if cache_estimate_report is not None:
            estimated_gain_chars = int(max(0, cache_gate_value(cache_estimate_report)))
        else:
            estimated_gain_chars = max(0, rewritten_prefix_chars - original_prefix_chars)
            if plan.moved_blocks and cache_report.get("estimated_cache_gain") is not None:
                estimated_gain_chars = max(estimated_gain_chars, int(cache_report.get("estimated_cache_gain") or 0))
        return CacheUtilityEstimate(
            cacheable_prefix_blocks=cacheable_ids,
            original_prefix_chars=original_prefix_chars,
            rewritten_prefix_chars=rewritten_prefix_chars,
            moved_block_chars=moved_block_chars,
            estimated_gain_chars=estimated_gain_chars,
            prefix_fingerprint_before=before_fingerprint,
            prefix_fingerprint_after=after_fingerprint,
            longest_common_prefix_tokens=int(cache_report.get("longest_common_prefix_tokens") or 0),
            global_prefix_tokens=int(cache_report.get("global_prefix_tokens") or 0),
            subgroup_prefix_tokens=int(cache_report.get("subgroup_prefix_tokens") or 0),
            node_cache_contributions=cache_report.get("node_cache_contributions")
            if isinstance(cache_report.get("node_cache_contributions"), Mapping)
            else None,
            cache_gain_report=cache_report if cache_report else None,
            cache_estimate_report=cache_estimate_report,
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


def _normalize_utility_model_mode(value: str) -> str:
    normalized = str(value or "disabled").strip().lower()
    if normalized not in {"disabled", "shadow", "gate"}:
        raise ValueError(f"Unsupported utility_model_mode: {value}")
    return normalized
