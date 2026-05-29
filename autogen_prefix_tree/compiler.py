from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from .ir import (
    BlockPosition,
    CompileResult,
    Movability,
    PromptBlock,
    SemanticType,
    ShareScope,
    hash_model_args,
    hash_tools,
    stable_hash,
)


class LocalPromptCompiler:
    """把 AutoGen typed messages 编译成可验证、可规划的 prompt block IR。"""

    def compile(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] | None = None,
        session_id: str = "default",
    ) -> CompileResult:
        blocks: list[PromptBlock] = []
        for message_index, message in enumerate(messages):
            blocks.extend(self._compile_message(message, message_index))

        return CompileResult(
            messages=messages,
            blocks=tuple(blocks),
            tools_hash=hash_tools(tools),
            model_args_hash=hash_model_args(tool_choice, json_output, extra_create_args),
            session_id=session_id,
        )

    def _compile_message(self, message: LLMMessage, message_index: int) -> list[PromptBlock]:
        source_type = getattr(message, "type", type(message).__name__)
        source_role = self._source_role(message)
        agent_or_source = getattr(message, "source", None)
        source_message_id = f"msg:{message_index}:{source_type}:{agent_or_source or ''}"
        content = getattr(message, "content", None)

        if isinstance(message, SystemMessage) and isinstance(content, str):
            segments = _split_text_preserving_separators(content)
            return [
                self._make_text_block(
                    segment,
                    message_index=message_index,
                    part_index=part_index,
                    source_message_id=source_message_id,
                    source_role=source_role,
                    source_type=source_type,
                    agent_or_source=agent_or_source,
                    is_system_text=True,
                )
                for part_index, segment in enumerate(segments)
                if segment
            ]

        if isinstance(content, str):
            return [
                self._make_text_block(
                    content,
                    message_index=message_index,
                    part_index=0,
                    source_message_id=source_message_id,
                    source_role=source_role,
                    source_type=source_type,
                    agent_or_source=agent_or_source,
                    is_system_text=False,
                )
            ]

        return [
            self._make_payload_block(
                content,
                message=message,
                message_index=message_index,
                source_message_id=source_message_id,
                source_role=source_role,
                source_type=source_type,
                agent_or_source=agent_or_source,
            )
        ]

    def _make_text_block(
        self,
        text: str,
        *,
        message_index: int,
        part_index: int,
        source_message_id: str,
        source_role: str,
        source_type: str,
        agent_or_source: str | None,
        is_system_text: bool,
    ) -> PromptBlock:
        semantic_type, movability, share_scope, risk_tags = self._classify_text(
            text,
            source_role=source_role,
            source_type=source_type,
            is_system_text=is_system_text,
        )
        content_hash = stable_hash({"text": text})
        block_id = f"{message_index}:{part_index}:{content_hash[:12]}"
        return PromptBlock(
            block_id=block_id,
            source_message_id=source_message_id,
            source_role=source_role,
            source_type=source_type,
            agent_or_source=agent_or_source,
            text_or_payload=text,
            content_hash=content_hash,
            semantic_type=semantic_type,
            movability=movability,
            share_scope=share_scope,
            risk_tags=risk_tags,
            original_position=BlockPosition(message_index=message_index, part_index=part_index),
            rendered_text=text,
            is_system_text=is_system_text,
        )

    def _make_payload_block(
        self,
        payload: Any,
        *,
        message: LLMMessage,
        message_index: int,
        source_message_id: str,
        source_role: str,
        source_type: str,
        agent_or_source: str | None,
    ) -> PromptBlock:
        if isinstance(message, FunctionExecutionResultMessage):
            semantic_type = SemanticType.TOOL_RESULT
            movability = Movability.NEVER_MOVE
            share_scope = ShareScope.PRIVATE
            risk_tags = ("tool_result_order",)
        elif isinstance(message, AssistantMessage):
            semantic_type = SemanticType.CONVERSATION_HISTORY
            movability = Movability.ORDER_SENSITIVE
            share_scope = ShareScope.AGENT
            risk_tags = ("assistant_history_order",)
        elif isinstance(message, UserMessage):
            semantic_type = SemanticType.CURRENT_USER_INSTRUCTION
            movability = Movability.ORDER_SENSITIVE
            share_scope = ShareScope.AGENT
            risk_tags = ("latest_user_instruction",)
        else:
            semantic_type = SemanticType.UNKNOWN
            movability = Movability.NEVER_MOVE
            share_scope = ShareScope.UNKNOWN
            risk_tags = ("unknown_payload",)

        content_hash = stable_hash({"payload": payload})
        block_id = f"{message_index}:0:{content_hash[:12]}"
        return PromptBlock(
            block_id=block_id,
            source_message_id=source_message_id,
            source_role=source_role,
            source_type=source_type,
            agent_or_source=agent_or_source,
            text_or_payload=payload,
            content_hash=content_hash,
            semantic_type=semantic_type,
            movability=movability,
            share_scope=share_scope,
            risk_tags=risk_tags,
            original_position=BlockPosition(message_index=message_index, part_index=0),
            rendered_text=None,
            is_system_text=False,
        )

    def _classify_text(
        self,
        text: str,
        *,
        source_role: str,
        source_type: str,
        is_system_text: bool,
    ) -> tuple[SemanticType, Movability, ShareScope, tuple[str, ...]]:
        upper = text.upper()

        if source_role == "assistant":
            return SemanticType.CONVERSATION_HISTORY, Movability.ORDER_SENSITIVE, ShareScope.AGENT, (
                "assistant_history_order",
            )
        if source_role == "tool":
            return SemanticType.TOOL_RESULT, Movability.NEVER_MOVE, ShareScope.PRIVATE, ("tool_result_order",)
        if source_role == "user":
            return SemanticType.CURRENT_USER_INSTRUCTION, Movability.ORDER_SENSITIVE, ShareScope.AGENT, (
                "latest_user_instruction",
            )

        if "PRIVATE_MEMORY" in upper or "MEMORY" in upper and "PRIVATE" in upper:
            return SemanticType.PRIVATE_MEMORY, Movability.LOCAL_ONLY, ShareScope.PRIVATE, ("private_memory",)
        if "PRIVATE_TOOL" in upper or "TOOL_PERMISSION" in upper:
            return SemanticType.PRIVATE_TOOL_PERMISSION, Movability.LOCAL_ONLY, ShareScope.PRIVATE, (
                "private_tool_permission",
            )
        if "ROLE_SPECIFIC" in upper or "AGENT_NAME" in upper or re.search(r"\bYOU ARE\b", upper):
            return SemanticType.ROLE_IDENTITY, Movability.LOCAL_ONLY, ShareScope.AGENT, ("agent_identity",)
        if "CURRENT_TURN" in upper:
            return SemanticType.CURRENT_TURN_INSTRUCTION, Movability.ORDER_SENSITIVE, ShareScope.AGENT, (
                "current_turn_instruction",
            )
        if "CURRENT_USER" in upper or "LATEST_USER" in upper:
            return SemanticType.CURRENT_USER_INSTRUCTION, Movability.ORDER_SENSITIVE, ShareScope.AGENT, (
                "latest_user_instruction",
            )
        if "TOOL_SCHEMA" in upper or ("TOOL" in upper and "SCHEMA" in upper):
            return SemanticType.SHARED_TOOL_DESCRIPTION, Movability.CONDITIONAL_PREFIX, ShareScope.GLOBAL, (
                "tool_schema",
            )
        if "SHARED_GROUPCHAT_CONTEXT" in upper or "SHARED_CONTEXT" in upper:
            return SemanticType.SHARED_CONTEXT, Movability.SAFE_PREFIX, ShareScope.GLOBAL, ("shared_context",)
        if "TEAM_POLICY" in upper or ("TEAM" in upper and "POLICY" in upper):
            return SemanticType.TEAM_POLICY, Movability.SAFE_PREFIX, ShareScope.GLOBAL, ("team_policy",)
        if "OUTPUT_FORMAT" in upper:
            return SemanticType.OUTPUT_FORMAT, Movability.CONDITIONAL_PREFIX, ShareScope.GLOBAL, ("output_format",)
        if "USER_TASK" in upper or "GLOBAL_TASK" in upper or "TASK_BACKGROUND" in upper:
            return SemanticType.GLOBAL_TASK_BACKGROUND, Movability.SAFE_PREFIX, ShareScope.GLOBAL, (
                "task_background",
            )

        # 无法确定来源与边界时，宁可不移动，避免把私有或顺序敏感内容提升到共享前缀。
        if is_system_text:
            return SemanticType.UNKNOWN, Movability.ORDER_SENSITIVE, ShareScope.UNKNOWN, ("unknown_system_text",)
        return SemanticType.UNKNOWN, Movability.NEVER_MOVE, ShareScope.UNKNOWN, ("unknown_text",)

    def _source_role(self, message: LLMMessage) -> str:
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, UserMessage):
            return "user"
        if isinstance(message, AssistantMessage):
            return "assistant"
        if isinstance(message, FunctionExecutionResultMessage):
            return "tool"
        return "unknown"


def _split_text_preserving_separators(text: str) -> list[str]:
    parts = re.split(r"(\n{2,})", text)
    segments: list[str] = []
    index = 0
    while index < len(parts):
        segment = parts[index]
        if index + 1 < len(parts):
            segment += parts[index + 1]
        if segment:
            segments.append(segment)
        index += 2
    return segments or [text]

