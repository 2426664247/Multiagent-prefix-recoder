from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from .compiler import LocalPromptCompiler
from .dataset_eval import openai_messages_to_autogen
from .planner import HierarchicalPrefixPlanner, rewrite_messages
from .semantic_guard import OpenAICompatibleSemanticGuard
from .shadow_trial import ShadowTrialPlan, evaluate_shadow_trial, load_shadow_trial_plan
from .telemetry import JsonlTelemetryLogger, TelemetrySink, dataclass_to_dict, serialize_prefix_tree
from .validator import CacheUtilityValidator


@dataclass(frozen=True)
class OpenAIRequestRewriteResult:
    original_body: Mapping[str, Any]
    rewritten_body: dict[str, Any]
    telemetry: dict[str, Any]

    @property
    def applied(self) -> bool:
        validation = self.telemetry.get("validation") or {}
        return bool(validation.get("applied"))

    @property
    def fallback(self) -> bool:
        validation = self.telemetry.get("validation") or {}
        return bool(validation.get("fallback"))


class OpenAICompatibleRequestAdapter:
    """Prefix-reorder adapter for OpenAI-compatible chat completion request bodies."""

    def __init__(
        self,
        *,
        compiler: LocalPromptCompiler | None = None,
        planner: HierarchicalPrefixPlanner | None = None,
        validator: CacheUtilityValidator | None = None,
        session_id: str = "default",
        enabled: bool = True,
        enable_natural_language_segmentation: bool = False,
        enable_groupchat_history_reordering: bool = False,
        shadow_trial_plan: ShadowTrialPlan | None = None,
        shadow_trial_plan_path: str | Path | None = None,
        telemetry_sink: TelemetrySink | None = None,
        telemetry_log_path: str | Path | None = None,
    ) -> None:
        self.enable_natural_language_segmentation = enable_natural_language_segmentation
        self.enable_groupchat_history_reordering = enable_groupchat_history_reordering
        self.compiler = compiler or LocalPromptCompiler(
            enable_natural_language_segmentation=enable_natural_language_segmentation,
            enable_groupchat_history_reordering=enable_groupchat_history_reordering,
        )
        self.planner = planner or HierarchicalPrefixPlanner(
            enable_groupchat_history_reordering=enable_groupchat_history_reordering
        )
        self.validator = validator or CacheUtilityValidator()
        if shadow_trial_plan is not None and shadow_trial_plan_path is not None:
            raise ValueError("Pass either shadow_trial_plan or shadow_trial_plan_path, not both")
        self.shadow_trial_plan = (
            shadow_trial_plan
            if shadow_trial_plan is not None
            else load_shadow_trial_plan(shadow_trial_plan_path)
            if shadow_trial_plan_path is not None
            else None
        )
        self.session_id = session_id
        self.enabled = enabled
        self.last_result: OpenAIRequestRewriteResult | None = None
        self._request_index = 0
        self._telemetry_sinks: list[TelemetrySink] = []
        if telemetry_sink is not None:
            self._telemetry_sinks.append(telemetry_sink)
        if telemetry_log_path is not None:
            self._telemetry_sinks.append(JsonlTelemetryLogger(telemetry_log_path))

    def rewrite_request_body(
        self,
        body: Mapping[str, Any],
        *,
        session_id: str | None = None,
        operation: str = "chat.completions.create",
    ) -> OpenAIRequestRewriteResult:
        request_index = self._next_request_index()
        effective_session_id = str(session_id or body.get("session_id") or self.session_id)
        rewritten_body = dict(body)
        original_messages: tuple[LLMMessage, ...] | None = None
        prepared_messages: Sequence[LLMMessage] | None = None
        compile_result = None
        plan = None

        if not self.enabled:
            telemetry = self._build_telemetry_record(
                request_index=request_index,
                operation=operation,
                session_id=effective_session_id,
                original_body=body,
                original_messages=(),
                prepared_messages=(),
                compile_result=None,
                plan=None,
                report_applied=False,
                report_fallback=True,
                report_reason="disabled",
                utility_estimate=None,
                semantic_guard_report=None,
                risk_notes=(),
            )
            return self._finish(body, rewritten_body, telemetry)

        try:
            original_messages = openai_messages_to_autogen(body.get("messages"))
            prepared_messages = original_messages
            tools = tuple(body.get("tools") or ())
            tool_choice = body.get("tool_choice", "auto")
            json_output = _json_output_arg(body)
            extra_create_args = _extra_create_args(body)
            compile_result = self.compiler.compile(
                original_messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                session_id=effective_session_id,
            )
            plan = self.planner.plan(compile_result, session_id=effective_session_id)
            candidate_messages = rewrite_messages(compile_result, plan)
            report = self.validator.validate(
                original_messages=original_messages,
                rewritten_messages=candidate_messages,
                compile_result=compile_result,
                plan=plan,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
            )
            if not report.fallback:
                prepared_messages = candidate_messages
                rewritten_body["messages"] = autogen_messages_to_openai(prepared_messages, original_body_messages=body.get("messages"))
            telemetry = self._build_telemetry_record(
                request_index=request_index,
                operation=operation,
                session_id=effective_session_id,
                original_body=body,
                original_messages=original_messages,
                prepared_messages=prepared_messages,
                compile_result=compile_result,
                plan=plan,
                report_applied=report.applied,
                report_fallback=report.fallback,
                report_reason=report.reason,
                utility_estimate=dataclass_to_dict(report.utility_estimate),
                semantic_guard_report=dataclass_to_dict(report.semantic_guard_report),
                risk_notes=report.risk_notes,
            )
            return self._finish(body, rewritten_body, telemetry)
        except Exception as exc:  # noqa: BLE001
            telemetry = self._build_telemetry_record(
                request_index=request_index,
                operation=operation,
                session_id=effective_session_id,
                original_body=body,
                original_messages=original_messages or (),
                prepared_messages=prepared_messages or original_messages or (),
                compile_result=compile_result,
                plan=plan,
                report_applied=False,
                report_fallback=True,
                report_reason=f"pipeline_exception:{type(exc).__name__}",
                utility_estimate=None,
                semantic_guard_report=None,
                risk_notes=(),
            )
            return self._finish(body, rewritten_body, telemetry)

    def _finish(
        self,
        original_body: Mapping[str, Any],
        rewritten_body: dict[str, Any],
        telemetry: dict[str, Any],
    ) -> OpenAIRequestRewriteResult:
        result = OpenAIRequestRewriteResult(original_body=original_body, rewritten_body=rewritten_body, telemetry=telemetry)
        self.last_result = result
        for sink in self._telemetry_sinks:
            sink(telemetry)
        return result

    def _next_request_index(self) -> int:
        self._request_index += 1
        return self._request_index

    def _build_telemetry_record(
        self,
        *,
        request_index: int,
        operation: str,
        session_id: str,
        original_body: Mapping[str, Any],
        original_messages: Sequence[LLMMessage],
        prepared_messages: Sequence[LLMMessage],
        compile_result: Any,
        plan: Any,
        report_applied: bool,
        report_fallback: bool,
        report_reason: str,
        utility_estimate: Mapping[str, Any] | None,
        semantic_guard_report: Mapping[str, Any] | None,
        risk_notes: Sequence[str],
    ) -> dict[str, Any]:
        blocks = compile_result.blocks if compile_result is not None else ()
        return {
            "schema_version": "prefix-openai-request-adapter-telemetry-v1",
            "session_id": session_id,
            "request_index": request_index,
            "operation": operation,
            "enabled": self.enabled,
            "natural_language_segmentation_enabled": self.enable_natural_language_segmentation,
            "groupchat_history_reordering_enabled": self.enable_groupchat_history_reordering,
            "body_shape": _body_shape(original_body),
            "message_count_before": len(original_messages),
            "message_count_after": len(prepared_messages),
            "message_types_before": tuple(_message_type(message) for message in original_messages),
            "message_types_after": tuple(_message_type(message) for message in prepared_messages),
            "block_count": len(blocks),
            "semantic_type_counts": dict(Counter(block.semantic_type.value for block in blocks)),
            "movability_counts": dict(Counter(block.movability.value for block in blocks)),
            "share_scope_counts": dict(Counter(block.share_scope.value for block in blocks)),
            "blocks_moved": plan.moved_blocks if plan is not None else (),
            "cacheable_prefix_blocks": plan.cacheable_prefix_blocks if plan is not None else (),
            "prefix_tree": serialize_prefix_tree(plan.prefix_tree if plan is not None else None),
            "validation": {
                "applied": report_applied,
                "fallback": report_fallback,
                "reason": report_reason,
            },
            "utility_estimate": utility_estimate,
            "semantic_guard": semantic_guard_report,
            "shadow_trial": evaluate_shadow_trial(self.shadow_trial_plan, blocks),
            "risk_notes": tuple(risk_notes),
            "hashes": {
                "tools": compile_result.tools_hash if compile_result is not None else None,
                "model_args": compile_result.model_args_hash if compile_result is not None else None,
            },
        }


def rewrite_openai_request_body(
    body: Mapping[str, Any],
    *,
    session_id: str = "default",
) -> OpenAIRequestRewriteResult:
    adapter = OpenAICompatibleRequestAdapter(session_id=session_id)
    return adapter.rewrite_request_body(body, session_id=session_id)


def autogen_messages_to_openai(
    messages: Sequence[LLMMessage],
    *,
    original_body_messages: Any = None,
) -> list[dict[str, Any]]:
    original_list = original_body_messages if isinstance(original_body_messages, list) else []
    converted: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        original = original_list[index] if index < len(original_list) and isinstance(original_list[index], Mapping) else {}
        if isinstance(message, SystemMessage):
            row = {}
            row["role"] = "system"
            row["content"] = message.content
        elif isinstance(message, UserMessage):
            row = {}
            row["role"] = "user"
            row["content"] = message.content
            if getattr(message, "source", None):
                row["name"] = message.source
        elif isinstance(message, AssistantMessage):
            row = {}
            row["role"] = "assistant"
            row["content"] = message.content
            if getattr(message, "source", None):
                row["name"] = message.source
        elif isinstance(message, FunctionExecutionResultMessage):
            row = {}
            row["role"] = _original_role(original, default="tool")
            row["content"] = _tool_result_content(message)
        else:
            row = {}
            row["role"] = str(row.get("role") or "user")
            row["content"] = str(getattr(message, "content", ""))
        converted.append(row)
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite OpenAI-compatible chat request JSONL without sending network calls.")
    parser.add_argument("--input", required=True, help="Input JSONL with OpenAI-compatible request bodies.")
    parser.add_argument("--output", required=True, help="Output JSONL with rewritten request bodies.")
    parser.add_argument("--telemetry", help="Optional prompt-safe telemetry JSONL.")
    parser.add_argument("--session-id", default="openai-request-adapter")
    parser.add_argument(
        "--enable-natural-language-segmentation",
        action="store_true",
        help="Experimental opt-in: split unmarked natural-language system prompts by line.",
    )
    parser.add_argument(
        "--enable-groupchat-history-reordering",
        action="store_true",
        help="Experimental opt-in: move exact repeated groupchat dialogue prefix ahead of agent-local suffix.",
    )
    parser.add_argument("--semantic-guard-base-url", help="Optional local OpenAI-compatible semantic guard base URL.")
    parser.add_argument("--semantic-guard-model", help="Model name for --semantic-guard-base-url.")
    parser.add_argument("--semantic-guard-timeout", type=float, default=30.0)
    parser.add_argument("--semantic-guard-min-confidence", type=float, default=0.75)
    parser.add_argument("--semantic-guard-max-message-chars", type=int, default=12000)
    parser.add_argument(
        "--shadow-trial-plan",
        help=(
            "Optional prompt-safe static_rule_calibration_eval summary or shadow_trial_plan JSON. "
            "Records shadow-only rule matches in telemetry without changing requests."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter = OpenAICompatibleRequestAdapter(
        session_id=args.session_id,
        telemetry_log_path=args.telemetry,
        enable_natural_language_segmentation=args.enable_natural_language_segmentation,
        enable_groupchat_history_reordering=args.enable_groupchat_history_reordering,
        validator=_validator_from_args(args),
        shadow_trial_plan_path=args.shadow_trial_plan,
    )
    output_path = Path(args.output)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open("r", encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            body = json.loads(stripped)
            if not isinstance(body, Mapping):
                continue
            result = adapter.rewrite_request_body(body)
            target.write(json.dumps(result.rewritten_body, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def _validator_from_args(args: argparse.Namespace) -> CacheUtilityValidator:
    if not args.semantic_guard_base_url:
        return CacheUtilityValidator()
    if not args.semantic_guard_model:
        raise ValueError("--semantic-guard-model is required when --semantic-guard-base-url is set")
    return CacheUtilityValidator(
        semantic_guard=OpenAICompatibleSemanticGuard(
            base_url=args.semantic_guard_base_url,
            model=args.semantic_guard_model,
            timeout_seconds=args.semantic_guard_timeout,
            min_confidence=args.semantic_guard_min_confidence,
            max_message_chars=args.semantic_guard_max_message_chars,
        )
    )


def _json_output_arg(body: Mapping[str, Any]) -> Any:
    if "json_output" in body:
        return body["json_output"]
    response_format = body.get("response_format")
    if isinstance(response_format, Mapping) and response_format.get("type") not in {None, "text"}:
        return response_format
    return None


def _extra_create_args(body: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"messages", "tools", "tool_choice", "json_output", "response_format", "stream"}
    return {
        str(key): value
        for key, value in body.items()
        if key not in excluded and not _looks_secret_name(str(key))
    }


def _looks_secret_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ("api_key", "authorization", "secret", "token", "password"))


def _body_shape(body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "keys": tuple(sorted(str(key) for key in body if not _looks_secret_name(str(key)))),
        "has_tools": isinstance(body.get("tools"), list),
        "has_response_format": "response_format" in body,
        "has_stream": "stream" in body,
    }


def _message_type(message: LLMMessage) -> str:
    return getattr(message, "type", type(message).__name__)


def _original_role(original: Mapping[str, Any], *, default: str) -> str:
    role = original.get("role")
    if isinstance(role, str) and role:
        return role
    return default


def _tool_result_content(message: FunctionExecutionResultMessage) -> str:
    contents = []
    for result in message.content:
        contents.append(str(result.content))
    return "\n".join(contents)


if __name__ == "__main__":
    raise SystemExit(main())
