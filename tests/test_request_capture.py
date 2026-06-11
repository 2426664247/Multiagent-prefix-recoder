from __future__ import annotations

import asyncio
import json
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
    SystemMessage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel
from typing_extensions import Self

from autogen_prefix_tree import PrefixReorderClient, RequestCaptureClient
from autogen_prefix_tree.dataset_eval import evaluate_dataset


class FakeClientConfig(BaseModel):
    label: str = "fake"


class FakeClient(ChatCompletionClient, Component[FakeClientConfig]):
    component_type = "model"
    component_config_schema = FakeClientConfig

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []

    def _to_config(self) -> FakeClientConfig:
        return FakeClientConfig()

    @classmethod
    def _from_config(cls, config: FakeClientConfig) -> Self:
        return cls()

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
        self.create_calls.append({"messages": tuple(messages), "extra_create_args": dict(extra_create_args)})
        return CreateResult(
            finish_reason="stop",
            content="OK",
            usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
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
            yield CreateResult(
                finish_reason="stop",
                content="OK",
                usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
                cached=False,
            )

        return _generator()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=1, completion_tokens=1)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=1, completion_tokens=1)

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return len(messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 1000 - len(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        return {"vision": False, "function_calling": True, "json_output": True}

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        }


def test_request_capture_writes_autogen_typed_messages_and_filters_secrets(tmp_path) -> None:
    capture_path = tmp_path / "requests.jsonl"
    inner = FakeClient()
    client = RequestCaptureClient(inner, capture_log_path=capture_path, session_id="capture-test")

    asyncio.run(
        client.create(
            _messages("planner"),
            extra_create_args={
                "temperature": 0,
                "api_key": "should-not-be-written",
                "authorization": "Bearer secret",
            },
        )
    )

    rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "autogen-request-capture-v1"
    assert rows[0]["session_id"] == "capture-test"
    assert rows[0]["body"]["messages"][0]["type"] == "SystemMessage"
    assert "AGENT_NAME: planner" in rows[0]["body"]["messages"][0]["content"]
    assert rows[0]["body"]["extra_create_args"] == {"temperature": 0}
    serialized = json.dumps(rows[0], ensure_ascii=False)
    assert "should-not-be-written" not in serialized
    assert "Bearer secret" not in serialized
    assert inner.create_calls[0]["messages"][0].content.startswith("ROLE_SPECIFIC_INSTRUCTION_START")


def test_request_capture_log_feeds_dataset_eval(tmp_path) -> None:
    capture_path = tmp_path / "captured_requests.jsonl"
    client = RequestCaptureClient(FakeClient(), capture_log_path=capture_path, session_id="coverage")

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    result = evaluate_dataset(input_path=capture_path, session_id="coverage")
    summary = result.summary

    assert summary["input_record_count"] == 2
    assert summary["supported_request_count"] == 2
    assert summary["semantic_coverage_supported"] is True
    assert summary["validation_reason_counts"] == {"no_rewrite_needed": 1, "validated": 1}
    assert summary["applied_count"] == 1
    assert summary["fallback_count"] == 0
    assert summary["moved_semantic_type_counts"]["team_policy"] == 1
    assert summary["moved_semantic_type_counts"]["shared_context"] == 1


def test_request_capture_can_redact_message_content(tmp_path) -> None:
    capture_path = tmp_path / "redacted_requests.jsonl"
    client = RequestCaptureClient(
        FakeClient(),
        capture_log_path=capture_path,
        session_id="redacted",
        include_message_content=False,
    )

    asyncio.run(client.create(_messages("planner")))

    row = json.loads(capture_path.read_text(encoding="utf-8"))
    message = row["body"]["messages"][0]
    assert message["type"] == "SystemMessage"
    assert message["content"] is None
    assert "content_hash" in message
    assert "AGENT_NAME: planner" not in json.dumps(row, ensure_ascii=False)


def test_request_capture_loads_from_autogen_component_config(tmp_path) -> None:
    capture_path = tmp_path / "component_capture.jsonl"
    loaded = ChatCompletionClient.load_component(
        {
            "provider": "autogen_prefix_tree.request_capture.RequestCaptureClient",
            "component_type": "model",
            "config": {
                "inner_client": FakeClient().dump_component().model_dump(exclude_none=True),
                "capture_log_path": str(capture_path),
                "session_id": "component-capture",
            },
        }
    )

    assert isinstance(loaded, RequestCaptureClient)
    asyncio.run(loaded.create(_messages("planner")))

    row = json.loads(capture_path.read_text(encoding="utf-8"))
    assert row["session_id"] == "component-capture"
    assert row["body"]["messages"][0]["type"] == "SystemMessage"


def test_prefix_reorder_loads_from_autogen_component_config(tmp_path) -> None:
    telemetry_path = tmp_path / "prefix_telemetry.jsonl"
    loaded = ChatCompletionClient.load_component(
        {
            "provider": "autogen_prefix_tree.client.PrefixReorderClient",
            "component_type": "model",
            "config": {
                "inner_client": FakeClient().dump_component().model_dump(exclude_none=True),
                "session_id": "component-prefix",
                "telemetry_log_path": str(telemetry_path),
            },
        }
    )

    assert isinstance(loaded, PrefixReorderClient)
    asyncio.run(loaded.create(_messages("planner")))
    asyncio.run(loaded.create(_messages("engineer")))

    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["validation"]["reason"] == "no_rewrite_needed"
    assert rows[1]["validation"]["reason"] == "validated"


def _messages(agent: str) -> list[LLMMessage]:
    return [
        SystemMessage(
            content="\n\n".join(
                [
                    "ROLE_SPECIFIC_INSTRUCTION_START\n"
                    f"AGENT_NAME: {agent}\n"
                    f"You are {agent}.\n"
                    "ROLE_SPECIFIC_INSTRUCTION_END",
                    "USER_TASK_START\nBuild the cache experiment.\nUSER_TASK_END",
                    "SHARED_GROUPCHAT_CONTEXT_START\nShared benchmark context.\nSHARED_GROUPCHAT_CONTEXT_END",
                    "TEAM_POLICY_START\nDo not move private memory or latest user instructions.\nTEAM_POLICY_END",
                    "TOOL_SCHEMA_START\nrecord_metric(name: string, value: number) -> string\nTOOL_SCHEMA_END",
                    "CURRENT_TURN_INSTRUCTION_START\nReport status.\nCURRENT_TURN_INSTRUCTION_END",
                ]
            )
        )
    ]
