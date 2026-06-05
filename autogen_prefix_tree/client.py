from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken
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

from .compiler import LocalPromptCompiler
from .ir import CompileResult, PromptBlock
from .planner import HierarchicalPrefixPlanner, PrefixPlan, rewrite_messages
from .telemetry import JsonlTelemetryLogger, TelemetrySink, dataclass_to_dict, serialize_prefix_tree
from .validator import CacheUtilityValidator, ValidationReport


class PrefixReorderClient(ChatCompletionClient):
    """AutoGen model_client 前置 wrapper：只改请求前的 prompt block 顺序，不改响应。"""

    component_type = "model"

    def __init__(
        self,
        inner_client: ChatCompletionClient,
        *,
        compiler: LocalPromptCompiler | None = None,
        planner: HierarchicalPrefixPlanner | None = None,
        validator: CacheUtilityValidator | None = None,
        session_id: str = "default",
        enabled: bool = True,
        telemetry_sink: TelemetrySink | None = None,
        telemetry_log_path: str | Path | None = None,
    ) -> None:
        self.inner_client = inner_client
        self.compiler = compiler or LocalPromptCompiler()
        self.planner = planner or HierarchicalPrefixPlanner()
        self.validator = validator or CacheUtilityValidator()
        self.session_id = session_id
        self.enabled = enabled
        self.last_plan: PrefixPlan | None = None
        self.last_validation_report: ValidationReport | None = None
        self.last_telemetry_record: dict[str, Any] | None = None
        self.last_telemetry_error: str | None = None
        self._request_index = 0
        self._telemetry_sinks: list[TelemetrySink] = []
        if telemetry_sink is not None:
            self._telemetry_sinks.append(telemetry_sink)
        if telemetry_log_path is not None:
            self._telemetry_sinks.append(JsonlTelemetryLogger(telemetry_log_path))

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
            "message_count_before": len(original_messages),
            "message_count_after": len(prepared_messages),
            "message_types_before": tuple(self._message_type(message) for message in original_messages),
            "message_types_after": tuple(self._message_type(message) for message in prepared_messages),
            "block_count": len(compile_result.blocks) if compile_result else 0,
            "blocks_moved": plan.moved_blocks if plan else (),
            "moved_block_details": moved_details,
            "cacheable_prefix_blocks": plan.cacheable_prefix_blocks if plan else (),
            "prefix_tree": serialize_prefix_tree(plan.prefix_tree if plan else None),
            "validation": {
                "applied": report.applied,
                "fallback": report.fallback,
                "reason": report.reason,
            },
            "utility_estimate": dataclass_to_dict(report.utility_estimate),
            "semantic_guard": dataclass_to_dict(report.semantic_guard_report),
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
