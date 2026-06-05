from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence, Union

import pytest
from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelCapabilities,  # type: ignore
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from autogen_prefix_tree import (
    CacheUtilityValidator,
    HierarchicalPrefixPlanner,
    LocalPromptCompiler,
    Movability,
    PrefixPlan,
    PrefixReorderClient,
    PrefixTree,
    PrefixTreeNode,
    SemanticGuardReport,
    SemanticType,
    ShareScope,
    load_jsonl_telemetry,
    rewrite_messages,
    summarize_telemetry,
)


class FakeClient(ChatCompletionClient):
    component_type = "model"

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.closed = False

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
                "cancellation_token": cancellation_token,
            }
        )
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
        self.stream_calls.append(
            {
                "messages": tuple(messages),
                "tools": tuple(tools),
                "tool_choice": tool_choice,
                "json_output": json_output,
                "extra_create_args": dict(extra_create_args),
                "cancellation_token": cancellation_token,
            }
        )

        async def _generator() -> AsyncGenerator[Union[str, CreateResult], None]:
            yield "chunk"
            yield CreateResult(
                finish_reason="stop",
                content="OK",
                usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
                cached=False,
            )

        return _generator()

    async def close(self) -> None:
        self.closed = True

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=1, completion_tokens=1)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=2, completion_tokens=2)

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


class FailingCompiler(LocalPromptCompiler):
    def compile(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        raise RuntimeError("compiler exploded")


class PassingSemanticGuard:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, **_: Any) -> SemanticGuardReport:
        self.calls += 1
        return SemanticGuardReport(
            passed=True,
            reason="semantic_invariants_hold",
            checks=("agent_identity", "latest_instruction"),
            model_name="fake-local-judge",
            confidence=0.91,
        )


class RejectingSemanticGuard:
    def evaluate(self, **_: Any) -> SemanticGuardReport:
        return SemanticGuardReport(
            passed=False,
            reason="agent_identity_uncertain",
            checks=("agent_identity",),
            model_name="fake-local-judge",
            confidence=0.42,
        )


class ThrowingSemanticGuard:
    def evaluate(self, **_: Any) -> SemanticGuardReport:
        raise RuntimeError("judge failed")


def _system_content(agent: str, shared: str = "Shared project context.") -> str:
    return "\n\n".join(
        [
            "ROLE_SPECIFIC_INSTRUCTION_START\n"
            f"AGENT_NAME: {agent}\n"
            f"You are {agent}.\n"
            "ROLE_SPECIFIC_INSTRUCTION_END",
            "USER_TASK_START\nBuild the cache experiment.\nUSER_TASK_END",
            f"SHARED_GROUPCHAT_CONTEXT_START\n{shared}\nSHARED_GROUPCHAT_CONTEXT_END",
            "TOOL_SCHEMA_START\nshared_tool(x: string) -> string\nTOOL_SCHEMA_END",
            "CURRENT_TURN_INSTRUCTION_START\nAnswer this turn only.\nCURRENT_TURN_INSTRUCTION_END",
        ]
    )


def _messages(agent: str) -> list[LLMMessage]:
    return [SystemMessage(content=_system_content(agent))]


def _semantic_blocks(result, semantic_type: SemanticType):
    return [block for block in result.blocks if block.semantic_type == semantic_type]


def _tree_block_ids(node: PrefixTreeNode) -> tuple[str, ...]:
    ids = list(node.block_ids)
    for child in node.children:
        ids.extend(_tree_block_ids(child))
    return tuple(ids)


def _single_leaf_tree(order: tuple[str, ...]) -> PrefixTree:
    return PrefixTree(
        session_id="unit",
        root=PrefixTreeNode(
            node_id="unit:root",
            scope=ShareScope.GLOBAL,
            label="global_shared_prefix",
            children=(
                PrefixTreeNode(
                    node_id="unit:leaf",
                    scope=ShareScope.AGENT,
                    label="agent_local_suffix",
                    block_ids=order,
                ),
            ),
        ),
        leaf_path=("root", "leaf"),
    )


def test_compiler_classifies_autogen_typed_messages() -> None:
    compiler = LocalPromptCompiler()
    messages: list[LLMMessage] = [
        SystemMessage(content=_system_content("planner")),
        UserMessage(content="Please answer the latest user request.", source="user"),
        AssistantMessage(content="Earlier answer.", source="planner"),
        FunctionExecutionResultMessage(
            content=[
                FunctionExecutionResult(
                    content="tool output",
                    name="lookup",
                    call_id="call-1",
                )
            ]
        ),
    ]

    result = compiler.compile(
        messages,
        tools=[{"name": "shared_tool", "description": "shared"}],
        extra_create_args={"temperature": 0},
    )

    assert _semantic_blocks(result, SemanticType.ROLE_IDENTITY)[0].movability == Movability.LOCAL_ONLY
    assert _semantic_blocks(result, SemanticType.GLOBAL_TASK_BACKGROUND)[0].movability == Movability.SAFE_PREFIX
    assert _semantic_blocks(result, SemanticType.SHARED_CONTEXT)[0].movability == Movability.SAFE_PREFIX
    assert _semantic_blocks(result, SemanticType.SHARED_TOOL_DESCRIPTION)[0].movability == Movability.CONDITIONAL_PREFIX
    assert _semantic_blocks(result, SemanticType.CURRENT_TURN_INSTRUCTION)[0].movability == Movability.ORDER_SENSITIVE
    assert _semantic_blocks(result, SemanticType.CURRENT_USER_INSTRUCTION)[0].source_role == "user"
    assert _semantic_blocks(result, SemanticType.CONVERSATION_HISTORY)[0].source_role == "assistant"
    assert _semantic_blocks(result, SemanticType.TOOL_RESULT)[0].movability == Movability.NEVER_MOVE


def test_planner_uses_exact_hash_history_before_reordering() -> None:
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()

    first = compiler.compile(_messages("planner"), session_id="unit")
    first_plan = planner.plan(first, session_id="unit")
    assert first_plan.moved_blocks == ()

    second = compiler.compile(_messages("engineer"), session_id="unit")
    second_plan = planner.plan(second, session_id="unit")

    assert second_plan.moved_blocks
    moved_types = {
        block.semantic_type
        for block in second.blocks
        if block.block_id in set(second_plan.moved_blocks)
    }
    assert SemanticType.ROLE_IDENTITY not in moved_types
    assert SemanticType.SHARED_CONTEXT in moved_types
    assert SemanticType.GLOBAL_TASK_BACKGROUND in moved_types
    assert second_plan.prefix_tree is not None
    assert Counter(_tree_block_ids(second_plan.prefix_tree.root)) == Counter(block.block_id for block in second.blocks)
    assert second_plan.cacheable_prefix_blocks == second_plan.new_order[: len(second_plan.cacheable_prefix_blocks)]
    assert second_plan.prefix_tree.root.block_ids == second_plan.cacheable_prefix_blocks


def test_prefix_reorder_client_delegates_rewritten_messages_after_shared_evidence() -> None:
    inner = FakeClient()
    client = PrefixReorderClient(inner, session_id="team")

    first_result = asyncio.run(client.create(_messages("planner")))
    second_result = asyncio.run(client.create(_messages("engineer")))

    assert first_result.content == "OK"
    assert second_result.content == "OK"
    assert len(inner.create_calls) == 2

    first_content = inner.create_calls[0]["messages"][0].content
    second_content = inner.create_calls[1]["messages"][0].content
    assert first_content.index("ROLE_SPECIFIC_INSTRUCTION_START") < first_content.index("USER_TASK_START")
    assert second_content.index("USER_TASK_START") < second_content.index("ROLE_SPECIFIC_INSTRUCTION_START")
    assert "AGENT_NAME: engineer" in second_content
    assert client.last_validation_report is not None
    assert client.last_validation_report.applied is True
    assert client.last_validation_report.fallback is False
    assert client.last_validation_report.utility_estimate is not None
    assert client.last_validation_report.utility_estimate.estimated_gain_chars > 0


def test_prefix_reorder_client_emits_prompt_safe_telemetry() -> None:
    inner = FakeClient()
    records: list[dict[str, Any]] = []
    client = PrefixReorderClient(inner, session_id="team", telemetry_sink=records.append)

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    assert len(records) == 2
    assert records[0]["request_index"] == 1
    assert records[1]["request_index"] == 2
    assert records[1]["validation"] == {
        "applied": True,
        "fallback": False,
        "reason": "validated",
    }
    assert records[1]["blocks_moved"]
    assert records[1]["cacheable_prefix_blocks"]
    assert records[1]["prefix_tree"]["root"]["label"] == "global_shared_prefix"
    assert records[1]["utility_estimate"]["estimated_gain_chars"] > 0
    serialized = json.dumps(records[1], ensure_ascii=False)
    assert "AGENT_NAME: engineer" not in serialized
    assert "Shared project context." not in serialized


def test_semantic_guard_allows_validated_rewrite_and_emits_report() -> None:
    inner = FakeClient()
    guard = PassingSemanticGuard()
    records: list[dict[str, Any]] = []
    validator = CacheUtilityValidator(semantic_guard=guard)
    client = PrefixReorderClient(inner, session_id="team", validator=validator, telemetry_sink=records.append)

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    second_content = inner.create_calls[1]["messages"][0].content
    assert second_content.index("USER_TASK_START") < second_content.index("ROLE_SPECIFIC_INSTRUCTION_START")
    assert guard.calls == 1
    assert client.last_validation_report is not None
    assert client.last_validation_report.fallback is False
    assert client.last_validation_report.semantic_guard_report is not None
    assert client.last_validation_report.semantic_guard_report.passed is True
    assert records[-1]["semantic_guard"]["checks"] == ("agent_identity", "latest_instruction")


def test_semantic_guard_rejection_forces_fallback() -> None:
    inner = FakeClient()
    validator = CacheUtilityValidator(semantic_guard=RejectingSemanticGuard())
    client = PrefixReorderClient(inner, session_id="team", validator=validator)

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    second_content = inner.create_calls[1]["messages"][0].content
    assert second_content.index("ROLE_SPECIFIC_INSTRUCTION_START") < second_content.index("USER_TASK_START")
    assert client.last_validation_report is not None
    assert client.last_validation_report.fallback is True
    assert client.last_validation_report.reason == "semantic_guard_failed:agent_identity_uncertain"
    assert client.last_telemetry_record is not None
    assert client.last_telemetry_record["semantic_guard"]["passed"] is False


def test_semantic_guard_exception_fails_closed() -> None:
    inner = FakeClient()
    validator = CacheUtilityValidator(semantic_guard=ThrowingSemanticGuard())
    client = PrefixReorderClient(inner, session_id="team", validator=validator)

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    assert client.last_validation_report is not None
    assert client.last_validation_report.fallback is True
    assert client.last_validation_report.reason == "semantic_guard_failed:exception:RuntimeError"


def test_prefix_reorder_client_writes_jsonl_telemetry(tmp_path) -> None:
    inner = FakeClient()
    telemetry_path = tmp_path / "telemetry" / "requests.jsonl"
    client = PrefixReorderClient(inner, session_id="team", telemetry_log_path=telemetry_path)

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))

    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert [row["request_index"] for row in rows] == [1, 2]
    assert rows[0]["validation"]["reason"] == "no_rewrite_needed"
    assert rows[1]["validation"]["reason"] == "validated"


def test_telemetry_summary_counts_reuse_and_validation_reasons(tmp_path) -> None:
    inner = FakeClient()
    telemetry_path = tmp_path / "requests.jsonl"
    client = PrefixReorderClient(inner, session_id="team", telemetry_log_path=telemetry_path)

    asyncio.run(client.create(_messages("planner")))
    asyncio.run(client.create(_messages("engineer")))
    asyncio.run(client.create(_messages("reviewer")))

    records = load_jsonl_telemetry(telemetry_path)
    summary = summarize_telemetry(records)

    assert summary.request_count == 3
    assert summary.no_rewrite_count == 1
    assert summary.applied_count == 2
    assert summary.fallback_count == 0
    assert summary.validation_reason_counts == {"no_rewrite_needed": 1, "validated": 2}
    assert summary.reusable_prefix_request_count == 2
    assert summary.repeated_prefix_request_count == 1
    assert summary.unique_reusable_prefix_count == 1
    assert summary.total_estimated_gain_chars > 0
    assert summary.to_dict()["applied_rate"] == pytest.approx(2 / 3)


def test_prefix_reorder_client_falls_back_when_pipeline_raises() -> None:
    inner = FakeClient()
    client = PrefixReorderClient(inner, compiler=FailingCompiler())
    original_messages = _messages("planner")

    result = asyncio.run(client.create(original_messages))

    assert result.content == "OK"
    assert inner.create_calls[0]["messages"] == tuple(original_messages)
    assert client.last_validation_report is not None
    assert client.last_validation_report.fallback is True
    assert client.last_validation_report.reason.startswith("pipeline_exception")
    assert client.last_telemetry_record is not None
    assert client.last_telemetry_record["validation"]["reason"].startswith("pipeline_exception")


def test_create_stream_delegates_chunks_without_modifying_response() -> None:
    inner = FakeClient()
    client = PrefixReorderClient(inner)

    async def collect() -> list[Union[str, CreateResult]]:
        return [chunk async for chunk in client.create_stream(_messages("planner"))]

    chunks = asyncio.run(collect())

    assert chunks[0] == "chunk"
    assert isinstance(chunks[-1], CreateResult)
    assert chunks[-1].content == "OK"
    assert len(inner.stream_calls) == 1


def test_validator_rejects_forbidden_block_movement() -> None:
    compiler = LocalPromptCompiler()
    validator = CacheUtilityValidator()
    result = compiler.compile(_messages("planner"))
    role_block = _semantic_blocks(result, SemanticType.ROLE_IDENTITY)[0]
    original_ids = tuple(block.block_id for block in result.blocks)
    new_order = tuple(block_id for block_id in original_ids if block_id != role_block.block_id) + (role_block.block_id,)
    plan = PrefixPlan(
        original_order=original_ids,
        new_order=new_order,
        moved_blocks=(role_block.block_id,),
        kept_blocks=tuple(block_id for block_id in original_ids if block_id != role_block.block_id),
        move_reason={role_block.block_id: "illegal test move"},
        prefix_tree=_single_leaf_tree(new_order),
    )

    report = validator.validate(
        original_messages=result.messages,
        rewritten_messages=result.messages,
        compile_result=result,
        plan=plan,
    )

    assert report.fallback is True
    assert report.reason.startswith("forbidden_block_moved")


def test_validator_rejects_tools_hash_changes() -> None:
    compiler = LocalPromptCompiler()
    validator = CacheUtilityValidator()
    result = compiler.compile(_messages("planner"), tools=[{"name": "a"}])
    plan = PrefixPlan(
        original_order=tuple(block.block_id for block in result.blocks),
        new_order=tuple(block.block_id for block in result.blocks),
        moved_blocks=(),
        kept_blocks=tuple(block.block_id for block in result.blocks),
        move_reason={},
        prefix_tree=_single_leaf_tree(tuple(block.block_id for block in result.blocks)),
    )

    report = validator.validate(
        original_messages=result.messages,
        rewritten_messages=result.messages,
        compile_result=result,
        plan=plan,
        tools=[{"name": "b"}],
    )

    assert report.fallback is True
    assert report.reason == "tools_hash_changed"


def test_validator_rejects_prefix_tree_coverage_mismatch() -> None:
    compiler = LocalPromptCompiler()
    validator = CacheUtilityValidator()
    result = compiler.compile(_messages("planner"))
    original_ids = tuple(block.block_id for block in result.blocks)
    bad_tree = _single_leaf_tree(original_ids[:-1])
    plan = PrefixPlan(
        original_order=original_ids,
        new_order=original_ids,
        moved_blocks=(),
        kept_blocks=original_ids,
        move_reason={},
        prefix_tree=bad_tree,
    )

    report = validator.validate(
        original_messages=result.messages,
        rewritten_messages=result.messages,
        compile_result=result,
        plan=plan,
    )

    assert report.fallback is True
    assert report.reason == "prefix_tree_block_coverage_mismatch"


def test_validator_rejects_rewritten_content_mismatch() -> None:
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    validator = CacheUtilityValidator()
    planner.plan(compiler.compile(_messages("planner"), session_id="unit"), session_id="unit")
    result = compiler.compile(_messages("engineer"), session_id="unit")
    plan = planner.plan(result, session_id="unit")
    rewritten = list(rewrite_messages(result, plan))
    rewritten[0] = rewritten[0].model_copy(update={"content": "tampered"})

    report = validator.validate(
        original_messages=result.messages,
        rewritten_messages=rewritten,
        compile_result=result,
        plan=plan,
    )

    assert report.fallback is True
    assert report.reason == "rewritten_content_mismatch"


def test_rewrite_preserves_block_hash_multiset_and_message_type() -> None:
    compiler = LocalPromptCompiler()
    planner = HierarchicalPrefixPlanner()
    planner.plan(compiler.compile(_messages("planner"), session_id="unit"), session_id="unit")
    result = compiler.compile(_messages("engineer"), session_id="unit")
    plan = planner.plan(result, session_id="unit")

    rewritten = rewrite_messages(result, plan)

    assert len(rewritten) == len(result.messages)
    assert isinstance(rewritten[0], SystemMessage)
    assert rewritten[0].content != result.messages[0].content
    assert "AGENT_NAME: engineer" in rewritten[0].content


def test_prefix_reorder_client_is_autogen_chat_completion_client() -> None:
    client = PrefixReorderClient(FakeClient())

    assert isinstance(client, ChatCompletionClient)
    assert client.model_info["family"] == ModelFamily.UNKNOWN
    assert client.count_tokens(_messages("planner")) == 1
