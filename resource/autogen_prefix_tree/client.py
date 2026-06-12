from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken, Component
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,  # type: ignore
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel
from typing_extensions import Self

from .cache_estimator import cache_estimator_config_name, observe_cache_estimator
from .compiler import LocalPromptCompiler
from .feedback import JsonlFeedbackLogger, PlannerFeedbackLearner
from .ir import CompileResult, PromptBlock
from .planner import HierarchicalPrefixPlanner, PrefixPlan, rewrite_messages
from .semantic_guard import OpenAICompatibleSemanticGuard
from .telemetry import JsonlTelemetryLogger, TelemetrySink, dataclass_to_dict, serialize_prefix_tree, serialize_prefix_tree_candidate
from .validator import CacheUtilityValidator, ValidationReport


class SemanticGuardConfig(BaseModel):
    provider: str = "openai-compatible"
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0
    min_confidence: float = 0.75
    max_message_chars: int = 12000


class PrefixReorderClientConfig(BaseModel):
    inner_client: dict[str, Any]
    session_id: str = "default"
    enabled: bool = True
    enable_natural_language_segmentation: bool = False
    enable_groupchat_history_reordering: bool = False
    telemetry_log_path: str | None = None
    feedback_log_path: str | None = None
    min_estimated_gain_chars: int = 1
    cache_estimator: str = "prefix_tree"
    semantic_guard: SemanticGuardConfig | None = None


class PrefixReorderClient(ChatCompletionClient, Component[PrefixReorderClientConfig]):
    """AutoGen model_client 前置 wrapper：只改请求前的 prompt block 顺序，不改响应。"""

    component_type = "model"
    component_config_schema = PrefixReorderClientConfig

    def __init__(
        self,
        inner_client: ChatCompletionClient,
        *,
        compiler: LocalPromptCompiler | None = None,
        planner: HierarchicalPrefixPlanner | None = None,
        validator: CacheUtilityValidator | None = None,
        session_id: str = "default",
        enabled: bool = True,
        enable_natural_language_segmentation: bool = False,
        enable_groupchat_history_reordering: bool = False,
        telemetry_sink: TelemetrySink | None = None,
        telemetry_log_path: str | Path | None = None,
        feedback_learner: PlannerFeedbackLearner | None = None,
        feedback_log_path: str | Path | None = None,
        cache_estimator: str | Any | None = None,
    ) -> None:
        self.inner_client = inner_client
        self.enable_natural_language_segmentation = enable_natural_language_segmentation
        self.enable_groupchat_history_reordering = enable_groupchat_history_reordering
        self.feedback_learner = feedback_learner or PlannerFeedbackLearner(
            feedback_sink=JsonlFeedbackLogger(feedback_log_path) if feedback_log_path is not None else None
        )
        self.compiler = compiler or LocalPromptCompiler(
            enable_natural_language_segmentation=enable_natural_language_segmentation,
            enable_groupchat_history_reordering=enable_groupchat_history_reordering,
        )
        self.planner = planner or HierarchicalPrefixPlanner(
            enable_groupchat_history_reordering=enable_groupchat_history_reordering,
            feedback_policy=self.feedback_learner,
        )
        if hasattr(self.planner, "feedback_policy") and getattr(self.planner, "feedback_policy") is None:
            self.planner.feedback_policy = self.feedback_learner
        if validator is None:
            self.validator = CacheUtilityValidator(cache_estimator=cache_estimator)
        else:
            self.validator = validator
        self.session_id = session_id
        self.enabled = enabled
        self.cache_estimator_config = cache_estimator_config_name(cache_estimator if validator is None else self.validator.cache_estimator)
        self.last_plan: PrefixPlan | None = None
        self.last_validation_report: ValidationReport | None = None
        self.last_feedback_record: dict[str, Any] | None = None
        self.last_telemetry_record: dict[str, Any] | None = None
        self.last_telemetry_error: str | None = None
        self._request_index = 0
        self._telemetry_sinks: list[TelemetrySink] = []
        if telemetry_sink is not None:
            self._telemetry_sinks.append(telemetry_sink)
        if telemetry_log_path is not None:
            self._telemetry_sinks.append(JsonlTelemetryLogger(telemetry_log_path))

    def _to_config(self) -> PrefixReorderClientConfig:
        return PrefixReorderClientConfig(
            inner_client=self.inner_client.dump_component().model_dump(exclude_none=True),
            session_id=self.session_id,
            enabled=self.enabled,
            enable_natural_language_segmentation=self.enable_natural_language_segmentation,
            enable_groupchat_history_reordering=self.enable_groupchat_history_reordering,
            telemetry_log_path=self._telemetry_log_path(),
            feedback_log_path=self._feedback_log_path(),
            min_estimated_gain_chars=self.validator.min_estimated_gain_chars,
            cache_estimator=self.cache_estimator_config,
        )

    @classmethod
    def _from_config(cls, config: PrefixReorderClientConfig) -> Self:
        inner_client = ChatCompletionClient.load_component(config.inner_client)
        semantic_guard = _semantic_guard_from_config(config.semantic_guard)
        return cls(
            inner_client,
            validator=CacheUtilityValidator(
                min_estimated_gain_chars=config.min_estimated_gain_chars,
                semantic_guard=semantic_guard,
                cache_estimator=config.cache_estimator,
            ),
            session_id=config.session_id,
            enabled=config.enabled,
            enable_natural_language_segmentation=config.enable_natural_language_segmentation,
            enable_groupchat_history_reordering=config.enable_groupchat_history_reordering,
            telemetry_log_path=config.telemetry_log_path,
            feedback_log_path=config.feedback_log_path,
        )

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        request_index = self._next_request_index()
        prepared_messages = self._prepare_messages(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            request_index=request_index,
            operation="create",
        )
        return await self.inner_client.create(
            prepared_messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        request_index = self._next_request_index()
        prepared_messages = self._prepare_messages(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            request_index=request_index,
            operation="create_stream",
        )
        return self.inner_client.create_stream(
            prepared_messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

    def _prepare_messages(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema],
        tool_choice: Any,
        json_output: Any,
        extra_create_args: Mapping[str, Any],
        request_index: int,
        operation: str,
    ) -> Sequence[LLMMessage]:
        if not self.enabled:
            self.last_validation_report = ValidationReport(False, True, "disabled")
            self._emit_telemetry(
                self._build_telemetry_record(
                    request_index=request_index,
                    operation=operation,
                    original_messages=messages,
                    prepared_messages=messages,
                    compile_result=None,
                    plan=None,
                    report=self.last_validation_report,
                )
            )
            return messages

        compile_result: CompileResult | None = None
        plan: PrefixPlan | None = None
        try:
            compile_result = self.compiler.compile(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                session_id=self.session_id,
            )
            plan = self.planner.plan(compile_result, session_id=self.session_id)
            self.last_plan = plan
            rewritten_messages = rewrite_messages(compile_result, plan)
            report = self.validator.validate(
                original_messages=messages,
                rewritten_messages=rewritten_messages,
                compile_result=compile_result,
                plan=plan,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
            )
            self.last_validation_report = report
            prepared_messages = messages if report.fallback else rewritten_messages
            self._observe_cache_estimator(
                compile_result=compile_result,
                plan=plan,
                original_messages=messages,
                rewritten_messages=rewritten_messages,
                accepted=not report.fallback,
            )
            self._record_feedback(compile_result=compile_result, plan=plan, report=report)
            self._emit_telemetry(
                self._build_telemetry_record(
                    request_index=request_index,
                    operation=operation,
                    original_messages=messages,
                    prepared_messages=prepared_messages,
                    compile_result=compile_result,
                    plan=plan,
                    report=report,
                )
            )
            if report.fallback:
                return messages
            return rewritten_messages
        except Exception as exc:  # noqa: BLE001
            # 任何模块异常都回退，保证 wrapper 不会阻断原始 model_client 调用。
            self.last_validation_report = ValidationReport(False, True, f"pipeline_exception:{type(exc).__name__}")
            self._emit_telemetry(
                self._build_telemetry_record(
                    request_index=request_index,
                    operation=operation,
                    original_messages=messages,
                    prepared_messages=messages,
                    compile_result=compile_result,
                    plan=plan,
                    report=self.last_validation_report,
                )
            )
            return messages

    def _next_request_index(self) -> int:
        self._request_index += 1
        return self._request_index

    def _record_feedback(
        self,
        *,
        compile_result: CompileResult | None,
        plan: PrefixPlan | None,
        report: ValidationReport,
    ) -> None:
        if compile_result is None or plan is None:
            self.last_feedback_record = None
            return
        try:
            record = self.feedback_learner.record(
                compile_result=compile_result,
                plan=plan,
                validation_report=report,
            )
            self.last_feedback_record = record.to_dict()
        except Exception as exc:  # noqa: BLE001
            self.last_feedback_record = {
                "schema_version": "prefix-planner-feedback-error-v1",
                "prompt_safe_summary": True,
                "reason": f"{type(exc).__name__}:{exc}",
            }

    def _emit_telemetry(self, record: dict[str, Any]) -> None:
        self.last_telemetry_record = record
        for sink in self._telemetry_sinks:
            try:
                sink(record)
            except Exception as exc:  # noqa: BLE001
                self.last_telemetry_error = f"{type(exc).__name__}:{exc}"

    def _build_telemetry_record(
        self,
        *,
        request_index: int,
        operation: str,
        original_messages: Sequence[LLMMessage],
        prepared_messages: Sequence[LLMMessage],
        compile_result: CompileResult | None,
        plan: PrefixPlan | None,
        report: ValidationReport,
    ) -> dict[str, Any]:
        blocks_by_id = {block.block_id: block for block in (compile_result.blocks if compile_result else ())}
        moved_details = tuple(
            self._block_telemetry(blocks_by_id[block_id])
            for block_id in (plan.moved_blocks if plan else ())
            if block_id in blocks_by_id
        )
        return {
            "schema_version": "prefix-reorder-telemetry-v1",
            "session_id": self.session_id,
            "request_index": request_index,
            "operation": operation,
            "enabled": self.enabled,
            "natural_language_segmentation_enabled": self.enable_natural_language_segmentation,
            "groupchat_history_reordering_enabled": self.enable_groupchat_history_reordering,
            "message_count_before": len(original_messages),
            "message_count_after": len(prepared_messages),
            "message_types_before": tuple(self._message_type(message) for message in original_messages),
            "message_types_after": tuple(self._message_type(message) for message in prepared_messages),
            "block_count": len(compile_result.blocks) if compile_result else 0,
            "blocks_moved": plan.moved_blocks if plan else (),
            "moved_block_details": moved_details,
            "cacheable_prefix_blocks": plan.cacheable_prefix_blocks if plan else (),
            "prefix_tree": serialize_prefix_tree(plan.prefix_tree if plan else None),
            "prefix_tree_candidate": serialize_prefix_tree_candidate(
                plan.prefix_tree_candidate if plan else None,
                include_text=False,
            ),
            "planner_score": plan.planner_score if plan else None,
            "planner_score_breakdown": plan.planner_score_breakdown if plan else None,
            "cache_gain_report": plan.cache_gain_report if plan else None,
            "validation": {
                "applied": report.applied,
                "fallback": report.fallback,
                "reason": report.reason,
            },
            "hard_constraint_passed": report.hard_constraint_passed,
            "hard_constraint_report": report.hard_constraint_report,
            "utility_status": report.utility_status,
            "cache_hit_increased": report.cache_hit_increased,
            "cache_estimate_report": dataclass_to_dict(report.cache_estimate_report),
            "utility_estimate": dataclass_to_dict(report.utility_estimate),
            "utility_preservation": dataclass_to_dict(report.utility_preservation_report),
            "semantic_guard": dataclass_to_dict(report.semantic_guard_report),
            "planner_feedback": self.last_feedback_record,
            "risk_notes": report.risk_notes,
            "hashes": {
                "tools": compile_result.tools_hash if compile_result else None,
                "model_args": compile_result.model_args_hash if compile_result else None,
            },
        }

    def _block_telemetry(self, block: PromptBlock) -> dict[str, Any]:
        return {
            "block_id": block.block_id,
            "semantic_type": block.semantic_type.value,
            "movability": block.movability.value,
            "share_scope": block.share_scope.value,
            "source_role": block.source_role,
            "source_type": block.source_type,
            "agent_or_source": block.agent_or_source,
            "original_position": {
                "message_index": block.original_position.message_index,
                "part_index": block.original_position.part_index,
            },
        }

    def _message_type(self, message: LLMMessage) -> str:
        return getattr(message, "type", type(message).__name__)

    def _observe_cache_estimator(
        self,
        *,
        compile_result: CompileResult,
        plan: PrefixPlan,
        original_messages: Sequence[LLMMessage],
        rewritten_messages: Sequence[LLMMessage],
        accepted: bool,
    ) -> None:
        observe_cache_estimator(
            self.validator.cache_estimator,
            original_messages,
            rewritten_messages,
            accepted=accepted,
            candidate=plan.prefix_tree_candidate,
            context={
                "compile_result": compile_result,
                "plan": plan,
                "session_id": compile_result.session_id,
                "candidate": plan.prefix_tree_candidate,
            },
        )

    def _telemetry_log_path(self) -> str | None:
        for sink in self._telemetry_sinks:
            if isinstance(sink, JsonlTelemetryLogger):
                return str(sink.path)
        return None

    def _feedback_log_path(self) -> str | None:
        for sink in self.feedback_learner._sinks:
            if isinstance(sink, JsonlFeedbackLogger):
                return str(sink.path)
        return None

    async def close(self) -> None:
        await self.inner_client.close()

    def actual_usage(self) -> RequestUsage:
        return self.inner_client.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self.inner_client.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self.inner_client.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self.inner_client.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        warnings.warn("capabilities is deprecated, use model_info instead", DeprecationWarning, stacklevel=2)
        return self.inner_client.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self.inner_client.model_info


def _semantic_guard_from_config(config: SemanticGuardConfig | None):
    if config is None:
        return None
    provider = config.provider.strip().lower()
    if provider not in {"openai-compatible", "openai_compatible"}:
        raise ValueError(f"Unsupported semantic guard provider: {config.provider}")
    return OpenAICompatibleSemanticGuard(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
        min_confidence=config.min_confidence,
        max_message_chars=config.max_message_chars,
    )
