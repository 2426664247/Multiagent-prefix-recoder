from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
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

from .ir import normalize_for_hash, stable_hash


@dataclass(frozen=True)
class JsonlRequestLogger:
    path: str | Path

    def __call__(self, record: Mapping[str, Any]) -> None:
        target = Path(self.path)
        if target.parent != Path("."):
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class RequestCaptureClientConfig(BaseModel):
    inner_client: dict[str, Any]
    capture_log_path: str | None = None
    session_id: str = "capture"
    include_message_content: bool = True


class RequestCaptureClient(ChatCompletionClient, Component[RequestCaptureClientConfig]):
    """AutoGen model_client wrapper that records typed requests before HTTP serialization.

    This capture is intentionally different from prompt-safe telemetry: it stores
    request messages so offline rule-coverage evaluation can inspect real prompts.
    Use it only for controlled benchmark traces.
    """

    component_type = "model"
    component_config_schema = RequestCaptureClientConfig

    def __init__(
        self,
        inner_client: ChatCompletionClient,
        *,
        capture_log_path: str | Path | None = None,
        capture_sink: Any | None = None,
        session_id: str = "capture",
        include_message_content: bool = True,
    ) -> None:
        self.inner_client = inner_client
        self.session_id = session_id
        self.include_message_content = include_message_content
        self.last_capture_record: dict[str, Any] | None = None
        self.last_capture_error: str | None = None
        self._request_index = 0
        self._capture_sinks: list[Any] = []
        if capture_sink is not None:
            self._capture_sinks.append(capture_sink)
        if capture_log_path is not None:
            self._capture_sinks.append(JsonlRequestLogger(capture_log_path))

    def _to_config(self) -> RequestCaptureClientConfig:
        return RequestCaptureClientConfig(
            inner_client=self.inner_client.dump_component().model_dump(exclude_none=True),
            capture_log_path=str(self._capture_sinks[-1].path)
            if self._capture_sinks and isinstance(self._capture_sinks[-1], JsonlRequestLogger)
            else None,
            session_id=self.session_id,
            include_message_content=self.include_message_content,
        )

    @classmethod
    def _from_config(cls, config: RequestCaptureClientConfig) -> Self:
        inner_client = ChatCompletionClient.load_component(config.inner_client)
        return cls(
            inner_client,
            capture_log_path=config.capture_log_path,
            session_id=config.session_id,
            include_message_content=config.include_message_content,
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
        self._capture_request(
            operation="create",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
        )
        return await self.inner_client.create(
            messages,
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
        self._capture_request(
            operation="create_stream",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
        )
        return self.inner_client.create_stream(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

    def _capture_request(
        self,
        *,
        operation: str,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        tool_choice: Any,
        json_output: Any,
        extra_create_args: Mapping[str, Any],
    ) -> None:
        self._request_index += 1
        record = {
            "schema_version": "autogen-request-capture-v1",
            "session_id": self.session_id,
            "request_index": self._request_index,
            "request_id": f"{self.session_id}:{self._request_index:06d}",
            "operation": operation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "body": {
                "messages": tuple(_serialize_message(message, include_content=self.include_message_content) for message in messages),
                "tools": normalize_for_hash(tuple(tools)),
                "tool_choice": normalize_for_hash(tool_choice),
                "json_output": normalize_for_hash(json_output),
                "extra_create_args": _filter_extra_create_args(extra_create_args),
            },
            "message_count": len(messages),
            "message_types": tuple(getattr(message, "type", type(message).__name__) for message in messages),
        }
        record["body_hash"] = stable_hash(record["body"])
        self.last_capture_record = record
        for sink in self._capture_sinks:
            try:
                sink(record)
            except Exception as exc:  # noqa: BLE001
                self.last_capture_error = f"{type(exc).__name__}:{exc}"

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


def _serialize_message(message: LLMMessage, *, include_content: bool) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        serialized = message.model_dump(mode="json")
    else:
        serialized = {"type": getattr(message, "type", type(message).__name__)}
        if hasattr(message, "source"):
            serialized["source"] = getattr(message, "source")
        if hasattr(message, "content"):
            serialized["content"] = getattr(message, "content")
    if not include_content and "content" in serialized:
        serialized = dict(serialized)
        serialized["content_hash"] = stable_hash(serialized["content"])
        serialized["content"] = None
    return serialized


def _filter_extra_create_args(extra_create_args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): normalize_for_hash(value)
        for key, value in extra_create_args.items()
        if not _looks_secret_name(str(key))
    }


def _looks_secret_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ("api_key", "authorization", "secret", "token", "password"))
