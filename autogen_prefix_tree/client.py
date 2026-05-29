from __future__ import annotations

import warnings
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
from .planner import HierarchicalPrefixPlanner, PrefixPlan, rewrite_messages
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
    ) -> None:
        self.inner_client = inner_client
        self.compiler = compiler or LocalPromptCompiler()
        self.planner = planner or HierarchicalPrefixPlanner()
        self.validator = validator or CacheUtilityValidator()
        self.session_id = session_id
        self.enabled = enabled
        self.last_plan: PrefixPlan | None = None
        self.last_validation_report: ValidationReport | None = None

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
        prepared_messages = self._prepare_messages(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
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
        prepared_messages = self._prepare_messages(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
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
    ) -> Sequence[LLMMessage]:
        if not self.enabled:
            self.last_validation_report = ValidationReport(False, True, "disabled")
            return messages

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
            if report.fallback:
                return messages
            return rewritten_messages
        except Exception as exc:  # noqa: BLE001
            # 任何模块异常都回退，保证 wrapper 不会阻断原始 model_client 调用。
            self.last_validation_report = ValidationReport(False, True, f"pipeline_exception:{type(exc).__name__}")
            return messages

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

