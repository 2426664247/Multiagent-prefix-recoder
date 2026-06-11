from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken, Component
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,  # type: ignore
    ModelFamily,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel
from typing_extensions import Self


class StaticResponseClientConfig(BaseModel):
    responses: list[str] = ["TERMINATE"]
    model: str = "static-response-client"
    prompt_tokens_per_message: int = 1
    completion_tokens: int = 1
    model_info: dict[str, Any] | None = None


@dataclass
class _UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StaticResponseClient(ChatCompletionClient, Component[StaticResponseClientConfig]):
    """No-network AutoGen model client for offline capture and component-loading smoke tests."""

    component_type = "model"
    component_config_schema = StaticResponseClientConfig

    def __init__(
        self,
        responses: Sequence[str] = ("TERMINATE",),
        *,
        model: str = "static-response-client",
        prompt_tokens_per_message: int = 1,
        completion_tokens: int = 1,
        model_info: Mapping[str, Any] | None = None,
    ) -> None:
        self.responses = tuple(responses) or ("TERMINATE",)
        self.model = model
        self.prompt_tokens_per_message = prompt_tokens_per_message
        self.completion_token_count = completion_tokens
        self._model_info = dict(model_info or {})
        self._index = 0
        self.create_calls: list[dict[str, Any]] = []
        self._actual_usage = _UsageTotals()
        self._total_usage = _UsageTotals()

    def _to_config(self) -> StaticResponseClientConfig:
        return StaticResponseClientConfig(
            responses=list(self.responses),
            model=self.model,
            prompt_tokens_per_message=self.prompt_tokens_per_message,
            completion_tokens=self.completion_token_count,
            model_info=self._model_info or None,
        )

    @classmethod
    def _from_config(cls, config: StaticResponseClientConfig) -> Self:
        return cls(
            responses=config.responses,
            model=config.model,
            prompt_tokens_per_message=config.prompt_tokens_per_message,
            completion_tokens=config.completion_tokens,
            model_info=config.model_info,
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
        self.create_calls.append(
            {
                "messages": tuple(messages),
                "tools": tuple(tools),
                "tool_choice": tool_choice,
                "json_output": json_output,
                "extra_create_args": dict(extra_create_args),
            }
        )
        content = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        usage = RequestUsage(
            prompt_tokens=max(1, len(messages) * self.prompt_tokens_per_message),
            completion_tokens=self.completion_token_count,
        )
        self._actual_usage = _UsageTotals(usage.prompt_tokens, usage.completion_tokens)
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        return CreateResult(
            finish_reason="stop",
            content=content,
            usage=usage,
            cached=False,
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
        async def _generator() -> AsyncGenerator[Union[str, CreateResult], None]:
            result = await self.create(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )
            yield result

        return _generator()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._actual_usage.prompt_tokens,
            completion_tokens=self._actual_usage.completion_tokens,
        )

    def total_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens,
        )

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return max(1, len(messages) * self.prompt_tokens_per_message)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 100000 - self.count_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        warnings.warn("capabilities is deprecated, use model_info instead", DeprecationWarning, stacklevel=2)
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        }

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": bool(self._model_info.get("vision", False)),
            "function_calling": bool(self._model_info.get("function_calling", False)),
            "json_output": bool(self._model_info.get("json_output", False)),
            "structured_output": bool(self._model_info.get("structured_output", False)),
            "family": self._model_info.get("family", ModelFamily.UNKNOWN),
        }
