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

    def __init__(
        self,
        *,
        enable_natural_language_segmentation: bool = False,
        enable_groupchat_history_reordering: bool = False,
    ) -> None:
        self.enable_natural_language_segmentation = enable_natural_language_segmentation
        self.enable_groupchat_history_reordering = enable_groupchat_history_reordering

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
        first_non_system_message_index = next(
            (index for index, message in enumerate(messages) if not isinstance(message, SystemMessage)),
            None,
        )
        for message_index, message in enumerate(messages):
            blocks.extend(
                self._compile_message(
                    message,
                    message_index,
                    first_non_system_message_index=first_non_system_message_index,
                )
            )

        return CompileResult(
            messages=messages,
            blocks=tuple(blocks),
            tools_hash=hash_tools(tools),
            model_args_hash=hash_model_args(tool_choice, json_output, extra_create_args),
            session_id=session_id,
        )

    def _compile_message(
        self,
        message: LLMMessage,
        message_index: int,
        *,
        first_non_system_message_index: int | None,
    ) -> list[PromptBlock]:
        source_type = getattr(message, "type", type(message).__name__)
        source_role = self._source_role(message)
        agent_or_source = getattr(message, "source", None)
        source_message_id = f"msg:{message_index}:{source_type}:{agent_or_source or ''}"
        content = getattr(message, "content", None)

        if isinstance(message, SystemMessage) and isinstance(content, str):
            segments = (
                _split_natural_language_system_text(content)
                if self.enable_natural_language_segmentation and not _has_explicit_section_markers(content)
                else _split_text_preserving_separators(content)
            )
            return [
                self._make_text_block(
                    segment,
                    message_index=message_index,
                    part_index=part_index,
                    source_message_id=source_message_id,
                    source_role=source_role,
                    source_type=source_type,
                    agent_or_source=agent_or_source,
                    is_initial_user_task=message_index == first_non_system_message_index,
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
                    is_initial_user_task=message_index == first_non_system_message_index,
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
        is_initial_user_task: bool,
        is_system_text: bool,
    ) -> PromptBlock:
        semantic_type, movability, share_scope, risk_tags = self._classify_text(
            text,
            source_role=source_role,
            source_type=source_type,
            agent_or_source=agent_or_source,
            is_initial_user_task=is_initial_user_task,
            is_system_text=is_system_text,
        )
        content_hash = stable_hash({"text": text}) if is_system_text else stable_hash(
            {
                "agent_or_source": agent_or_source,
                "source_role": source_role,
                "source_type": source_type,
                "text": text,
            }
        )
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
            movability = (
                Movability.CONDITIONAL_PREFIX
                if self.enable_groupchat_history_reordering
                else Movability.ORDER_SENSITIVE
            )
            share_scope = ShareScope.SUBGROUP
            risk_tags = ("assistant_history_order", "groupchat_history_order")
        elif isinstance(message, UserMessage):
            if self.enable_groupchat_history_reordering and _is_agent_history_source(agent_or_source):
                semantic_type = SemanticType.CONVERSATION_HISTORY
                movability = (
                    Movability.CONDITIONAL_PREFIX
                )
                share_scope = ShareScope.SUBGROUP
                risk_tags = ("user_message_history_order", "groupchat_history_order")
            else:
                semantic_type = SemanticType.CURRENT_USER_INSTRUCTION
                movability = Movability.ORDER_SENSITIVE
                share_scope = ShareScope.AGENT
                risk_tags = ("latest_user_instruction",)
        else:
            semantic_type = SemanticType.UNKNOWN
            movability = Movability.NEVER_MOVE
            share_scope = ShareScope.UNKNOWN
            risk_tags = ("unknown_payload",)

        content_hash = stable_hash(
            {
                "agent_or_source": agent_or_source,
                "payload": payload,
                "source_role": source_role,
                "source_type": source_type,
            }
        )
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
        agent_or_source: str | None,
        is_initial_user_task: bool,
        is_system_text: bool,
    ) -> tuple[SemanticType, Movability, ShareScope, tuple[str, ...]]:
        upper = text.upper()

        if source_role == "assistant":
            return (
                SemanticType.CONVERSATION_HISTORY,
                Movability.CONDITIONAL_PREFIX
                if self.enable_groupchat_history_reordering
                else Movability.ORDER_SENSITIVE,
                ShareScope.SUBGROUP,
                ("assistant_history_order", "groupchat_history_order"),
            )
        if source_role == "tool":
            return SemanticType.TOOL_RESULT, Movability.NEVER_MOVE, ShareScope.PRIVATE, ("tool_result_order",)
        if source_role == "user":
            if self.enable_groupchat_history_reordering and _is_agent_history_source(agent_or_source):
                return (
                    SemanticType.CONVERSATION_HISTORY,
                    Movability.CONDITIONAL_PREFIX,
                    ShareScope.SUBGROUP,
                    ("user_message_history_order", "groupchat_history_order"),
                )
            if self.enable_groupchat_history_reordering and is_initial_user_task:
                return (
                    SemanticType.GLOBAL_TASK_BACKGROUND,
                    Movability.CONDITIONAL_PREFIX,
                    ShareScope.GLOBAL,
                    ("initial_user_task", "groupchat_dialogue_order"),
                )
            return SemanticType.CURRENT_USER_INSTRUCTION, Movability.ORDER_SENSITIVE, ShareScope.AGENT, (
                "latest_user_instruction",
            )

        first_marker = _first_section_marker(upper)
        if first_marker in {"PRIVATE_MEMORY", "PRIVATE_MEMORY_INSTRUCTION"}:
            return SemanticType.PRIVATE_MEMORY, Movability.LOCAL_ONLY, ShareScope.PRIVATE, ("private_memory",)
        if first_marker in {"PRIVATE_TOOL", "TOOL_PERMISSION", "PRIVATE_TOOL_PERMISSION"}:
            return SemanticType.PRIVATE_TOOL_PERMISSION, Movability.LOCAL_ONLY, ShareScope.PRIVATE, (
                "private_tool_permission",
            )
        if first_marker in {"ROLE_SPECIFIC", "ROLE_SPECIFIC_INSTRUCTION", "AGENT_IDENTITY"}:
            return SemanticType.ROLE_IDENTITY, Movability.LOCAL_ONLY, ShareScope.AGENT, ("agent_identity",)
        if first_marker in {"CURRENT_TURN", "CURRENT_TURN_INSTRUCTION"}:
            return SemanticType.CURRENT_TURN_INSTRUCTION, Movability.ORDER_SENSITIVE, ShareScope.AGENT, (
                "current_turn_instruction",
            )
        if first_marker in {"CURRENT_USER", "LATEST_USER", "CURRENT_USER_INSTRUCTION"}:
            return SemanticType.CURRENT_USER_INSTRUCTION, Movability.ORDER_SENSITIVE, ShareScope.AGENT, (
                "latest_user_instruction",
            )
        if first_marker in {"TOOL_SCHEMA", "SHARED_TOOL_SCHEMA"}:
            return SemanticType.SHARED_TOOL_DESCRIPTION, Movability.CONDITIONAL_PREFIX, ShareScope.GLOBAL, (
                "tool_schema",
            )
        if first_marker in {"SHARED_GROUPCHAT_CONTEXT", "SHARED_CONTEXT"}:
            return SemanticType.SHARED_CONTEXT, Movability.SAFE_PREFIX, ShareScope.GLOBAL, ("shared_context",)
        if first_marker in {"TEAM_POLICY", "SHARED_TEAM_POLICY"}:
            return SemanticType.TEAM_POLICY, Movability.SAFE_PREFIX, ShareScope.GLOBAL, ("team_policy",)
        if first_marker == "OUTPUT_FORMAT":
            return SemanticType.OUTPUT_FORMAT, Movability.CONDITIONAL_PREFIX, ShareScope.GLOBAL, ("output_format",)
        if first_marker in {"USER_TASK", "GLOBAL_TASK", "TASK_BACKGROUND"}:
            return SemanticType.GLOBAL_TASK_BACKGROUND, Movability.SAFE_PREFIX, ShareScope.GLOBAL, (
                "task_background",
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
            natural_language_classification = (
                _classify_natural_language_system_prefix(text)
                if self.enable_natural_language_segmentation and is_system_text
                else None
            )
            if natural_language_classification is not None:
                return natural_language_classification
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


def _first_section_marker(upper_text: str) -> str | None:
    match = re.search(r"\b([A-Z][A-Z0-9_]+)_START\b", upper_text)
    if match is None:
        return None
    return match.group(1)


def _has_explicit_section_markers(text: str) -> bool:
    return _first_section_marker(text.upper()) is not None


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


def _split_natural_language_system_text(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines or [text]


def _classify_natural_language_system_prefix(
    text: str,
) -> tuple[SemanticType, Movability, ShareScope, tuple[str, ...]] | None:
    if not text.endswith(("\n", "\r")):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    high_risk_patterns = (
        r"\byou are\b",
        r"\bagent\b",
        r"\bcurrent\b",
        r"\blatest\b",
        r"\bprivate\b",
        r"\bmemory\b",
    )
    if any(re.search(pattern, lowered) for pattern in high_risk_patterns):
        return None
    if "user" in lowered and any(marker in lowered for marker in ("cannot", "can't", "must", "do not", "don't")):
        return None

    risk_tags = ["natural_language_segment"]
    if "if " in lowered or "when " in lowered:
        risk_tags.append("conditional_instruction")

    if any(marker in lowered for marker in ("tool", "function", "code block", "script", "execute", "browser")):
        return (
            SemanticType.SHARED_TOOL_DESCRIPTION,
            Movability.CONDITIONAL_PREFIX,
            ShareScope.GLOBAL,
            tuple((*risk_tags, "natural_language_tool_or_code_policy")),
        )
    if any(marker in lowered for marker in ("verify", "evidence", "check the", "check if", "confirmed")):
        return (
            SemanticType.TEAM_POLICY,
            Movability.SAFE_PREFIX,
            ShareScope.GLOBAL,
            tuple((*risk_tags, "natural_language_verification_policy")),
        )
    if any(marker in lowered for marker in ("do not", "don't", "must", "cannot", "can't", "only return", "reply with")):
        return (
            SemanticType.TEAM_POLICY,
            Movability.SAFE_PREFIX,
            ShareScope.GLOBAL,
            tuple((*risk_tags, "natural_language_team_policy")),
        )
    if any(marker in lowered for marker in ("step by step", "next step", "progress", "plan", "select the next")):
        return (
            SemanticType.TEAM_POLICY,
            Movability.SAFE_PREFIX,
            ShareScope.GLOBAL,
            tuple((*risk_tags, "natural_language_procedure_policy")),
        )
    return None


def _is_agent_history_source(source: str | None) -> bool:
    if source is None:
        return False
    normalized = str(source).strip().lower()
    if not normalized:
        return False
    return normalized not in {"user", "human", "system"}
